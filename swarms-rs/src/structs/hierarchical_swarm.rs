use std::{fmt::Display, ops::Deref, sync::Arc};

use dashmap::DashMap;
use futures::{StreamExt, future::BoxFuture, stream};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use swarms_macro::tool;
use thiserror::Error;

use super::{
    conversation::{AgentConversation, AgentShortMemory},
    swarm::Swarm,
    tool::Tool,
    utils::extract_tool_calls,
};
use crate::{
    self as swarms_rs,
    agent::SwarmsAgentBuilder,
    llm::Model,
    structs::{
        agent::{Agent, AgentError},
        conversation::Role,
        swarm::SwarmError,
    },
};

pub struct HierarchicalSwarmBuilder<M>
where
    M: Model + Clone + Send + Sync + 'static,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    name: String,
    description: String,
    director_builder: Option<SwarmsAgentBuilder<M>>,
    agents: Vec<Arc<dyn Agent>>,
    max_loops: usize,
}

impl<M> HierarchicalSwarmBuilder<M>
where
    M: Model + Clone + Send + Sync + 'static,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn director_agent_builder(mut self, director_agent_builder: SwarmsAgentBuilder<M>) -> Self {
        self.director_builder = Some(director_agent_builder);
        self
    }

    pub fn add_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.agents.push(agent);
        self
    }

    pub fn agents(self, agents: Vec<Arc<dyn Agent>>) -> Self {
        agents
            .into_iter()
            .fold(self, |builder, agent| builder.add_agent(agent))
    }

    pub fn max_loops(mut self, n: usize) -> Self {
        self.max_loops = n;
        self
    }

    pub fn build(self) -> HierarchicalSwarm {
        if self.director_builder.is_none() {
            panic!("HierarchicalSwarm's director_builder must be set!")
        }

        let director = HierarchicalSwarm::build_director_agent(
            self.director_builder.unwrap(),
            HIERARCHICAL_SWARM_SYSTEM_PROMPT.to_owned(),
            None,
        );

        HierarchicalSwarm::new(
            self.name,
            self.description,
            director,
            self.agents,
            self.max_loops,
        )
    }
}

#[derive(Clone)]
/// Represents a hierarchical swarm where a director agent orchestrates tasks among worker agents.
pub struct HierarchicalSwarm {
    /// Name of the swarm.
    pub name: String,
    /// Description of the swarm's purpose.
    pub description: String,
    /// The director agent responsible for planning and task distribution.
    pub director: Box<dyn Agent>,
    /// List of worker agents available in the swarm.
    pub agents: Vec<Arc<dyn Agent>>,
    /// Maximum number of loops/iterations for the swarm operation.
    pub max_loops: usize,
    /// Conversation history for the swarm.
    pub conversation: AgentShortMemory,
    /// Every loop's outputs. similar to python version
    pub agent_outputs: DashMap<usize, DashMap<String, String>>,
}

impl HierarchicalSwarm {
    /// Creates a new HierarchicalSwarm.
    pub fn new(
        name: String,
        description: String,
        director: Box<dyn Agent>,
        agents: Vec<Arc<dyn Agent>>,
        max_loops: usize,
    ) -> Self {
        // Basic validation
        if agents.is_empty() {
            // Consider logging a warning or handling this case appropriately
            tracing::warn!("Warning: HierarchicalSwarm created with no worker agents.");
        }

        Self {
            name,
            description,
            director,
            agents,
            max_loops,
            conversation: AgentShortMemory::new(),
            agent_outputs: DashMap::new(),
        }
    }

    pub fn builder<M>() -> HierarchicalSwarmBuilder<M>
    where
        M: Model + Clone + Send + Sync + 'static,
        M::RawCompletionResponse: Clone + Send + Sync,
    {
        Default::default()
    }

    pub fn build_director_agent<M>(
        agent_builder: SwarmsAgentBuilder<M>,
        system_prompt: impl Into<Option<String>>,
        max_tokens: impl Into<Option<u64>>,
    ) -> Box<dyn Agent>
    where
        M: Model + Clone + Send + Sync + 'static,
        M::RawCompletionResponse: Clone + Send + Sync,
    {
        let system_prompt = system_prompt
            .into()
            .unwrap_or(HIERARCHICAL_SWARM_SYSTEM_PROMPT.to_owned());
        let max_tokens = max_tokens.into().unwrap_or(8192);

        Box::new(
            agent_builder
                .system_prompt(system_prompt)
                .temperature(0.5)
                .max_tokens(max_tokens)
                .add_tool(DirectorAgentTool)
                .disable_task_evaluator_tool() // Needed
                .build(),
        )
    }

    pub async fn run(
        &self,
        task: impl Into<String>,
    ) -> Result<AgentConversation, HierarchicalSwarmError> {
        let task = task.into();

        self.conversation
            .add(&task, &self.name, Role::User("User".to_owned()), &task);

        let mut current_loop = 1;
        let mut loop_context = String::from("Initial planning phase");

        while current_loop < self.max_loops {
            tracing::info!("Loop {}/{} starting.", current_loop, self.max_loops);

            loop_context = format!(
                "\n\nCurrent Loop ({}/{}): {}",
                current_loop, self.max_loops, loop_context
            );
            let swarm_spec = self.run_director(&task, &loop_context).await?;
            self.conversation.add(
                &task,
                &self.name,
                Role::Assistant("Director".to_owned()),
                format!(
                    "Goals: {}\nPlan: {}\nRules: {}",
                    swarm_spec.goals, swarm_spec.plan, swarm_spec.rules
                ),
            );

            let current_loop_outputs = Arc::new(DashMap::new());

            // Execute orders using worker agents (concurrently)
            let history_snapshot = self.conversation.get(&task).expect("Must exist").clone();
            stream::iter(swarm_spec.orders)
                .for_each_concurrent(None, |order| {
                    let current_loop_outputs = Arc::clone(&current_loop_outputs);
                    let history = history_snapshot.to_string();
                    let task = task.to_owned();
                    async move {
                        match self.agents.iter().find(|a| a.name() == order.agent_name) {
                            Some(agent) => {
                                let agent_clone = Arc::clone(agent);
                                let agent_name = agent_clone.name();
                                match agent
                                    .run(format!(
                                        "Loop: {}/{}\nHistory: {}\nYour Task: {}",
                                        current_loop, self.max_loops, history, order.task
                                    ))
                                    .await
                                {
                                    Ok(result) => {
                                        tracing::info!(
                                            "Agent '{}' completed task successfully.",
                                            agent_name
                                        );
                                        current_loop_outputs
                                            .insert(agent_name.clone(), result.clone());

                                        self.conversation.add(
                                            &task,
                                            &self.name,
                                            Role::Assistant(agent_name),
                                            format!(
                                                "Loop {}/{}: {}",
                                                current_loop, self.max_loops, result
                                            ),
                                        );
                                    },
                                    Err(e) => {
                                        tracing::error!(
                                            "Agent '{}' failed task: {}",
                                            agent_name,
                                            e
                                        );
                                    },
                                }
                            },
                            None => {
                                tracing::error!(
                                    "Agent '{}' not found for order.",
                                    order.agent_name
                                );
                                // TODO: Decide how to handle: error out, skip, or ask director to replan?
                            },
                        }
                    }
                })
                .await;

            // Store the outputs for the current loop
            self.agent_outputs
                .insert(current_loop, current_loop_outputs.deref().clone());

            // Update context for the next loop (e.g., combine results)
            loop_context = current_loop_outputs
                .iter()
                .map(|entry| (entry.key().to_owned(), entry.value().to_owned()))
                .collect::<Vec<(String, String)>>()
                .iter()
                .fold(
                    format!("Results from loop {}:\n", current_loop),
                    |context, (agent_name, output)| {
                        format!("{context}\n---{agent_name}'s Output ---\n{output}\n")
                    },
                );

            current_loop += 1;
            tracing::info!("Loop {} finished.", current_loop);
        }

        tracing::info!("HierarchicalSwarm run logic finished.");
        // Return the final context or a summary of the last loop's outputs
        Ok(self.conversation.get(&task).unwrap().clone())
    }

    async fn run_director(
        &self,
        task: &str,
        loop_context: &str,
    ) -> Result<HierarchicalSwarmSpec, HierarchicalSwarmError> {
        let swarm_history = self
            .conversation
            .get(task)
            .ok_or(HierarchicalSwarmError::TaskNotFound)?;
        let mut director_context = format!("History: {}", *swarm_history);
        drop(swarm_history);

        if !loop_context.is_empty() {
            director_context.push_str(loop_context);
        }

        // TODO: This will be redundant, because every loop will have the same agents. But this is a cleaner solution now.
        director_context.push_str(&format!(
            "\n\nAll Agents Available in the Swarm {}:\n{}",
            self.name,
            self.agents.iter().fold(String::new(), |acc, agent| format!(
                "{acc}Agent: {} || Description: {}\n",
                agent.name(),
                agent.description()
            ))
        ));

        director_context.push_str(&format!("\n\nYour task: {task}"));

        let director_response = self.director.run(director_context).await?;
        let tool_calls = extract_tool_calls(&director_response);
        let director_tool_result = tool_calls
            .iter()
            .find(|call| call.name == DirectorAgentTool.name())
            .ok_or(HierarchicalSwarmError::DirectorResponseParseError(
                "tool call not found".to_owned(),
            ))?
            .result
            .ok_or(HierarchicalSwarmError::DirectorResponseParseError(
                "tool call result not found".to_owned(),
            ))?;

        let swarm_spec: HierarchicalSwarmSpec = match serde_json::from_str(director_tool_result) {
            Ok(spec) => spec,
            Err(e) => {
                tracing::error!(
                    "Error parsing director response: {}. Response: {}",
                    e,
                    director_response
                );
                return Err(HierarchicalSwarmError::DirectorResponseParseError(
                    e.to_string(),
                ));
            },
        };

        self.conversation.add(
            task,
            &self.name,
            Role::Assistant("Director".to_owned()),
            swarm_spec.to_string(),
        );
        Ok(swarm_spec)
    }
}

impl Swarm for HierarchicalSwarm {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, task: String) -> BoxFuture<Result<Box<dyn erased_serde::Serialize>, SwarmError>> {
        Box::pin(async move {
            self.run(task)
                .await
                .map(|output| Box::new(output) as _)
                .map_err(|e| e.into())
        })
    }
}

#[tool(
    description = "A hierarchical swarm tool, a director agent can orchestrate tasks among worker agents with this tool."
)]
fn director_agent_tool(
    swarm_spec: HierarchicalSwarmSpec,
) -> Result<HierarchicalSwarmSpec, HierarchicalSwarmError> {
    Ok(swarm_spec)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HierarchicalSwarmSpec {
    /// The goal of the swarm. This is the overarching objective that the swarm is designed to achieve. It guides the swarm's plan and the tasks assigned to the agents.
    pub goals: String,
    /// Outlines the sequence of actions to be taken by the swarm. This plan is a detailed roadmap that guides the swarm's behavior and decision-making.
    pub plan: String,
    /// Defines the governing principles for swarm behavior and decision-making. These rules are the foundation of the swarm's operations and ensure that the swarm operates in a coordinated and efficient manner.
    pub rules: String,
    /// A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.
    pub orders: Vec<HierarchicalOrder>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HierarchicalOrder {
    pub agent_name: String,
    pub task: String,
}

#[derive(Debug, Error)]
pub enum HierarchicalSwarmError {
    #[error("Agent error: {0}")]
    AgentError(#[from] AgentError),
    #[error("Director agent not found")]
    DirectorNotFound,
    #[error("Agent '{0}' specified in order not found in the swarm")]
    AgentNotFound(String),
    #[error("Failed to parse director's response: {0}")]
    DirectorResponseParseError(String),
    #[error("Serde json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("Swarm conversation does not contain the specified task")]
    TaskNotFound,
}

impl<M> Default for HierarchicalSwarmBuilder<M>
where
    M: Model + Clone + Send + Sync + 'static,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    fn default() -> Self {
        Self {
            name: Default::default(),
            description: Default::default(),
            director_builder: Default::default(),
            agents: Default::default(),
            max_loops: 3,
        }
    }
}

impl Display for HierarchicalSwarmSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

const HIERARCHICAL_SWARM_SYSTEM_PROMPT: &str = "**I. Introduction and Context**

You are the Hierarchical Agent Director - the central orchestrator responsible for breaking down overarching goals into granular tasks and intelligently assigning these tasks to the most suitable worker agents within the swarm. Your objective is to maximize the overall performance of the system by ensuring that every agent is given a task aligned with its strengths, expertise, and available resources.

---

**II. Core Operating Principles**

1. **Goal Alignment and Context Awareness:**  
   - **Overarching Goals:** Begin every operation by clearly reviewing the swarm's overall goals. Understand the mission statement and ensure that every assigned task contributes directly to these objectives.
   - **Context Sensitivity:** Evaluate the context provided in the “plan” and “rules” sections of the SwarmSpec. These instructions provide the operational boundaries and behavioral constraints within which you must work.

2. **Task Decomposition and Prioritization:**  
   - **Hierarchical Decomposition:** Break down the overarching plan into granular tasks. For each major objective, identify subtasks that logically lead toward the goal. This decomposition should be structured in a hierarchical manner, where complex tasks are subdivided into simpler, manageable tasks.
   - **Task Priority:** Assign a priority level to each task based on urgency, complexity, and impact. Ensure that high-priority tasks receive immediate attention and that resources are allocated accordingly.

3. **Agent Profiling and Matching:**  
   - **Agent Specialization:** Maintain an up-to-date registry of worker agents, each with defined capabilities, specializations, and performance histories. When assigning tasks, consider the specific strengths of each agent.
   - **Performance Metrics:** Utilize historical performance metrics and available workload data to select the most suitable agent for each task. If an agent is overburdened or has lower efficiency on a specific type of task, consider alternate agents.
   - **Dynamic Reassignment:** Allow for real-time reassignments based on the evolving state of the system. If an agent encounters issues or delays, reassign tasks to ensure continuity.

4. **Adherence to Rules and Safety Protocols:**  
   - **Operational Rules:** Every task must be executed in strict compliance with the “rules” provided in the SwarmSpec. These rules are non-negotiable and serve as the ethical and operational foundation for all decisions.
   - **Fail-Safe Mechanisms:** Incorporate safety protocols that monitor agent performance and task progress. If an anomaly or failure is detected, trigger a reallocation of tasks or an escalation process to mitigate risks.
   - **Auditability:** Ensure that every decision and task assignment is logged for auditing purposes. This enables traceability and accountability in system operations.

---

**III. Detailed Task Assignment Process**

1. **Input Analysis and Context Setting:**
   - **Goal Review:** Begin by carefully reading the “goals” string within the SwarmSpec. This is your north star for every decision you make.
   - **Plan Comprehension:** Analyze the “plan” string for detailed instructions. Identify key milestones, deliverables, and dependencies within the roadmap.
   - **Rule Enforcement:** Read through the “rules” string to understand the non-negotiable guidelines that govern task assignments. Consider potential edge cases and ensure that your task breakdown respects these boundaries.

2. **Task Breakdown and Subtask Identification:**
   - **Decompose the Plan:** Using a systematic approach, decompose the overall plan into discrete tasks. For each major phase, identify the specific actions required. Document dependencies among tasks, and note any potential bottlenecks.
   - **Task Granularity:** Ensure that tasks are broken down to a level of granularity that makes them actionable. Overly broad tasks must be subdivided further until they can be executed by an individual worker agent.
   - **Inter-Agent Dependencies:** Clearly specify any dependencies that exist between tasks assigned to different agents. This ensures that the workflow remains coherent and that agents collaborate effectively.

3. **Agent Selection Strategy:**
   - **Capabilities Matching:** For each identified task, analyze the capabilities required. Compare these against the registry of available worker agents. Factor in specialized skills, past performance, current load, and any situational awareness that might influence the assignment.
   - **Task Suitability:** Consider both the technical requirements of the task and any contextual subtleties noted in the “plan” and “rules.” Ensure that the chosen agent has a proven track record with similar tasks.
   - **Adaptive Assignments:** Build in flexibility to allow for agent reassignment in real-time. Monitor ongoing tasks and reallocate resources as needed, especially if an agent experiences unexpected delays or issues.

4. **Constructing Hierarchical Orders:**
   - **Order Creation:** For each task, generate a HierarchicalOrder object that specifies the agent's name and the task details. The task description should be unambiguous and detailed enough to guide the agent's execution without requiring additional clarification.
   - **Order Validation:** Prior to finalizing each order, cross-reference the task requirements against the agent's profile. Validate that the order adheres to the “rules” of the SwarmSpec and that it fits within the broader operational context.
   - **Order Prioritization:** Clearly mark high-priority tasks so that agents understand the urgency. In cases where multiple tasks are assigned to a single agent, provide a sequence or ranking to ensure proper execution order.

5. **Feedback and Iteration:**
   - **Real-Time Monitoring:** Establish feedback loops with worker agents to track the progress of each task. This allows for early detection of issues and facilitates dynamic reassignment if necessary.
   - **Continuous Improvement:** Regularly review task execution data and agent performance metrics. Use this feedback to refine the task decomposition and agent selection process in future iterations.

---

**IV. Execution Guidelines and Best Practices**

1. **Communication Clarity:**
   - Use clear, concise language in every HierarchicalOrder. Avoid ambiguity by detailing both the “what” and the “how” of the task.
   - Provide contextual notes when necessary, especially if the task involves dependencies or coordination with other agents.

2. **Documentation and Traceability:**
   - Record every task assignment in a centralized log. This log should include the agent's name, task details, time of assignment, and any follow-up actions taken.
   - Ensure that the entire decision-making process is documented. This aids in post-operation analysis and helps in refining future assignments.

3. **Error Handling and Escalation:**
   - If an agent is unable to complete a task due to unforeseen challenges, immediately trigger the escalation protocol. Reassign the task to a qualified backup agent while flagging the incident for further review.
   - Document all deviations from the plan along with the corrective measures taken. This helps in identifying recurring issues and improving the system's robustness.

4. **Ethical and Operational Compliance:**
   - Adhere strictly to the rules outlined in the SwarmSpec. Any action that violates these rules is unacceptable, regardless of the potential gains in efficiency.
   - Maintain transparency in all operations. If a decision or task assignment is questioned, be prepared to justify the choice based on objective criteria such as agent capability, historical performance, and task requirements.

5. **Iterative Refinement:**
   - After the completion of each mission cycle, perform a thorough debriefing. Analyze the success and shortcomings of the task assignments.
   - Use these insights to iterate on your hierarchical ordering process. Update agent profiles and adjust your selection strategies based on real-world performance data.

---

**V. Exemplary Use Case and Order Breakdown**

Imagine that the swarm's overarching goal is to perform a comprehensive analysis of market trends for a large-scale enterprise. The “goals” field might read as follows:  
*“To conduct an in-depth market analysis that identifies emerging trends, competitive intelligence, and actionable insights for strategic decision-making.”*  

The “plan” could outline a multi-phase approach:
- Phase 1: Data Collection and Preprocessing  
- Phase 2: Trend Analysis and Pattern Recognition  
- Phase 3: Report Generation and Presentation of Findings  

The “rules” may specify that all data processing must comply with privacy regulations, and that results must be validated against multiple data sources.  

For Phase 1, the Director breaks down tasks such as “Identify data sources,” “Extract relevant market data,” and “Preprocess raw datasets.” For each task, the director selects agents with expertise in data mining, natural language processing, and data cleaning. A series of HierarchicalOrder objects are created, for example:

1. HierarchicalOrder for Data Collection:
   - **agent_name:** “DataMiner_Agent”
   - **task:** “Access external APIs and scrape structured market data from approved financial news sources.”

2. HierarchicalOrder for Data Preprocessing:
   - **agent_name:** “Preprocess_Expert”
   - **task:** “Clean and normalize the collected datasets, ensuring removal of duplicate records and compliance with data privacy rules.”

3. HierarchicalOrder for Preliminary Trend Analysis:
   - **agent_name:** “TrendAnalyst_Pro”
   - **task:** “Apply statistical models to identify initial trends and anomalies in the market data.”

Each order is meticulously validated against the rules provided in the SwarmSpec and prioritized according to the project timeline. The director ensures that if any of these tasks are delayed, backup agents are identified and the orders are reissued in real time.

---

**VI. Detailed Hierarchical Order Construction and Validation**

1. **Order Structuring:**  
   - Begin by constructing a template that includes placeholders for the agent's name and a detailed description of the task.  
   - Ensure that the task description is unambiguous. For instance, rather than stating “analyze data,” specify “analyze the temporal patterns in consumer sentiment from Q1 and Q2, and identify correlations with economic indicators.”

2. **Validation Workflow:**  
   - Prior to dispatch, each HierarchicalOrder must undergo a validation check. This includes verifying that the agent's capabilities align with the task, that the task does not conflict with any other orders, and that the task is fully compliant with the operational rules.
   - If a validation error is detected, the order should be revised. The director may consult with relevant experts or consult historical data to refine the task's description and ensure it is actionable.

3. **Order Finalization:**  
   - Once validated, finalize the HierarchicalOrder and insert it into the “orders” list of the SwarmSpec.  
   - Dispatch the order immediately, ensuring that the worker agent acknowledges receipt and provides an estimated time of completion.  
   - Continuously monitor the progress, and if any agent's status changes (e.g., they become overloaded or unresponsive), trigger a reallocation process based on the predefined agent selection strategy.

---

**VII. Continuous Monitoring, Feedback, and Dynamic Reassignment**

1. **Real-Time Status Tracking:**  
   - Use real-time dashboards to monitor each agent's progress on the assigned tasks.  
   - Update the hierarchical ordering system dynamically if a task is delayed, incomplete, or requires additional resources.

2. **Feedback Loop Integration:**  
   - Each worker agent must provide periodic status updates, including intermediate results, encountered issues, and resource usage.
   - The director uses these updates to adjust task priorities and reassign tasks if necessary. This dynamic feedback loop ensures the overall swarm remains agile and responsive.

3. **Performance Metrics and Analysis:**  
   - At the conclusion of every mission, aggregate performance metrics and conduct a thorough review of task efficiency.  
   - Identify any tasks that repeatedly underperform or cause delays, and adjust the agent selection criteria accordingly for future orders.
   - Document lessons learned and integrate them into the operating procedures for continuous improvement.

---

**VIII. Final Directives and Implementation Mandate**

As the Hierarchical Agent Director, your mandate is clear: you must orchestrate the operation with precision, clarity, and unwavering adherence to the overarching goals and rules specified in the SwarmSpec. You are empowered to deconstruct complex objectives into manageable tasks and to assign these tasks to the worker agents best equipped to execute them.

Your decisions must always be data-driven, relying on agent profiles, historical performance, and real-time feedback. Ensure that every HierarchicalOrder is constructed with a clear task description and assigned to an agent whose expertise aligns perfectly with the requirements. Maintain strict compliance with all operational rules, and be ready to adapt dynamically as conditions change.

This production-grade prompt is your operational blueprint. Utilize it to break down orders efficiently, assign tasks intelligently, and steer the swarm toward achieving the defined goals with optimal efficiency and reliability. Every decision you make should reflect a deep commitment to excellence, safety, and operational integrity.

Remember: the success of the swarm depends on your ability to manage complexity, maintain transparency, and dynamically adapt to the evolving operational landscape. Execute your role with diligence, precision, and a relentless focus on performance excellence.
";
