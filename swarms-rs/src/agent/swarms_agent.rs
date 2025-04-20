use std::{
    ffi::OsStr,
    hash::{Hash, Hasher},
    ops::Deref,
    path::Path,
    sync::Arc,
};

use dashmap::DashMap;
use futures::{StreamExt, future::BoxFuture, stream};
use reqwest::IntoUrl;
use rmcp::{
    ServiceExt,
    model::{ClientCapabilities, ClientInfo, Implementation},
    transport::{SseTransport, TokioChildProcess},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use swarms_macro::tool;
use thiserror::Error;
use tokio::{
    process::Command,
    sync::{Mutex, mpsc},
};
use twox_hash::XxHash3_64;

use crate::{
    self as swarms_rs,
    llm::{
        self,
        request::{CompletionRequest, ToolDefinition},
    },
    structs::{
        conversation::{AgentShortMemory, Role},
        persistence,
        tool::{MCPTool, Tool, ToolDyn},
    },
};

use crate::structs::agent::{Agent, AgentConfig, AgentError};

pub struct SwarmsAgentBuilder<M>
where
    M: llm::Model + Send + Sync,
    M::RawCompletionResponse: Send + Sync,
{
    model: M,
    config: AgentConfig,
    system_prompt: Option<String>,
    tools: Vec<ToolDefinition>,
    tools_impl: DashMap<String, Arc<dyn ToolDyn>>,
}

impl<M> SwarmsAgentBuilder<M>
where
    M: llm::Model + Clone + Send + Sync,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    pub fn new_with_model(model: M) -> Self {
        Self {
            model,
            config: AgentConfig::default(),
            system_prompt: None,
            tools: vec![],
            tools_impl: DashMap::new(),
        }
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn add_tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.push(tool.definition());
        self.tools_impl
            .insert(tool.name().to_string(), Arc::new(tool) as Arc<dyn ToolDyn>);
        self
    }

    pub async fn add_sse_mcp_server(self, name: impl Into<String>, url: impl IntoUrl) -> Self {
        let name = name.into();

        let transport = SseTransport::start(url)
            .await
            .expect("Failed to start SSE transport");

        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: name.clone(),
                version: "".to_owned(),
            },
        };

        let client = Arc::new(
            client_info
                .into_dyn()
                .serve(transport)
                .await
                .expect("Failed to start MCP server"),
        );

        let mcp_tools = client.list_all_tools().await.expect("Failed to list tools");
        mcp_tools.into_iter().fold(self, |acc, tool| {
            acc.add_tool(MCPTool::from_server(tool, Arc::clone(&client)))
        })
    }

    pub async fn add_stdio_mcp_server<I, S>(self, command: S, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let service = Arc::new(
            ().into_dyn()
                .serve(TokioChildProcess::new(Command::new(command).args(args)).unwrap())
                .await
                .expect("Failed to start MCP server"),
        );

        let mcp_tools = service
            .list_all_tools()
            .await
            .expect("Failed to list tools");
        mcp_tools.into_iter().fold(self, |acc, tool| {
            acc.add_tool(MCPTool::from_server(tool, Arc::clone(&service)))
        })
    }

    pub fn build(mut self) -> SwarmsAgent<M> {
        if self.config.task_evaluator_tool_enabled {
            self.tools.insert(0, ToolDyn::definition(&TaskEvaluator));
            self.tools_impl.insert(
                ToolDyn::name(&TaskEvaluator),
                Arc::new(TaskEvaluator) as Arc<dyn ToolDyn>,
            );
        }

        SwarmsAgent {
            model: self.model,
            config: self.config,
            system_prompt: self.system_prompt,
            short_memory: AgentShortMemory::new(),
            tools: self.tools,
            tools_impl: self.tools_impl,
        }
    }

    // Configuration methods

    pub fn agent_name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    pub fn user_name(mut self, name: impl Into<String>) -> Self {
        self.config.user_name = name.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = temperature;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    pub fn max_loops(mut self, max_loops: u32) -> Self {
        self.config.max_loops = max_loops;
        self
    }

    pub fn enable_plan(mut self, planning_prompt: impl Into<Option<String>>) -> Self {
        self.config.plan_enabled = true;
        self.config.planning_prompt = planning_prompt.into();
        self
    }

    pub fn enable_autosave(mut self) -> Self {
        self.config.autosave = true;
        self
    }

    pub fn retry_attempts(mut self, retry_attempts: u32) -> Self {
        self.config.retry_attempts = retry_attempts;
        self
    }

    pub fn enable_rag_every_loop(mut self) -> Self {
        self.config.rag_every_loop = true;
        self
    }

    pub fn save_state_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.save_state_dir = Some(dir.into());
        self
    }

    pub fn add_stop_word(mut self, stop_word: impl Into<String>) -> Self {
        self.config.stop_words.insert(stop_word.into());
        self
    }

    pub fn stop_words(self, stop_words: Vec<String>) -> Self {
        stop_words
            .into_iter()
            .fold(self, |builder, stop_word| builder.add_stop_word(stop_word))
    }

    pub fn disable_task_evaluator_tool(mut self) -> Self {
        self.config.task_evaluator_tool_enabled = false;
        self
    }

    /// Some tools doesn't support concurrent call, so we need to disable it
    pub fn disable_concurrent_tool_call(mut self) -> Self {
        self.config.concurrent_tool_call_enabled = false;
        self
    }
}

#[derive(Clone, Serialize)]
pub struct SwarmsAgent<M>
where
    M: llm::Model + Clone + Send + Sync,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    model: M,
    config: AgentConfig,
    system_prompt: Option<String>,
    short_memory: AgentShortMemory,
    tools: Vec<ToolDefinition>,
    #[serde(skip)]
    tools_impl: DashMap<String, Arc<dyn ToolDyn>>,
}

impl<M> SwarmsAgent<M>
where
    M: llm::Model + Clone + Send + Sync + 'static,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    pub fn new(model: M, system_prompt: impl Into<Option<String>>) -> Self {
        Self {
            model,
            system_prompt: system_prompt.into(),
            config: AgentConfig::default(),
            short_memory: AgentShortMemory::new(),
            tools: vec![],
            tools_impl: DashMap::new(),
        }
    }

    pub async fn chat(
        &self,
        prompt: impl Into<String>,
        chat_history: impl Into<Vec<llm::completion::Message>>,
    ) -> Result<ChatResponse, AgentError> {
        let chat_history = chat_history.into();

        let request = CompletionRequest {
            prompt: llm::completion::Message::user(prompt),
            system_prompt: self.system_prompt.clone(),
            chat_history,
            tools: self.tools.clone(),
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
        };

        let response = self.model.completion(request).await?;

        let choice = response.choice.first().ok_or(AgentError::NoChoiceFound)?;
        match ToOwned::to_owned(choice) {
            llm::completion::AssistantContent::Text(text) => Ok(ChatResponse::Text(text.text)), // <--- return Text
            llm::completion::AssistantContent::ToolCall(tool_call) => {
                let mut all_tool_calls = vec![tool_call.function];
                all_tool_calls.extend(response.choice.iter().skip(1).filter_map(|choice| {
                    match ToOwned::to_owned(choice) {
                        llm::completion::AssistantContent::Text(_) => None,
                        llm::completion::AssistantContent::ToolCall(tool_call) => {
                            Some(tool_call.function)
                        },
                    }
                }));

                // Call tools concurrently
                let results = Arc::new(Mutex::new(Vec::new()));
                if self.config.concurrent_tool_call_enabled {
                    stream::iter(all_tool_calls)
                        .for_each_concurrent(None, |tool_call| {
                            let results = Arc::clone(&results);
                            async move {
                                let tool = Arc::clone(
                                    match self.tools_impl.get(&tool_call.name) {
                                        Some(tool) => tool,
                                        None => {
                                            tracing::error!("Tool not found: {}", tool_call.name);
                                            results.lock().await.push(ToolCallOutput {
                                                name: tool_call.name,
                                                args: tool_call.arguments.to_string(),
                                                result: "Tool not found".to_owned(),
                                            });
                                            return;
                                        },
                                    }
                                    .deref(),
                                );
                                let args = tool_call.arguments.to_string();
                                // execute tool
                                let result = match tool.call(args.clone()).await {
                                    Ok(result) => result,
                                    Err(e) => {
                                        tracing::error!(
                                            "Failed to call tool<{}>, args: {}, error: {}",
                                            tool.name(),
                                            args,
                                            e
                                        );
                                        results.lock().await.push(ToolCallOutput {
                                            name: tool_call.name,
                                            args,
                                            result: e.to_string(),
                                        });
                                        return;
                                    },
                                };
                                results.lock().await.push(ToolCallOutput {
                                    name: tool_call.name,
                                    args,
                                    result,
                                });
                            }
                        })
                        .await;
                } else {
                    for tool_call in all_tool_calls {
                        let tool = Arc::clone(
                            self.tools_impl
                                .get(&tool_call.name)
                                .ok_or(AgentError::ToolNotFound(tool_call.name.clone()))?
                                .deref(),
                        );
                        let args = tool_call.arguments.to_string();
                        // execute tool
                        let result_str = tool.call(args.clone()).await?;
                        // collect results
                        results.lock().await.push(ToolCallOutput {
                            name: tool_call.name.clone(),
                            args,
                            result: result_str,
                        });
                    }
                }

                Ok(ChatResponse::ToolCalls(
                    Arc::clone(&results).lock().await.clone(),
                ))
            },
        }
    }

    pub async fn prompt(&self, prompt: impl Into<String>) -> Result<String, AgentError> {
        let prompt = prompt.into();
        let request = CompletionRequest {
            prompt: llm::completion::Message::user(prompt),
            system_prompt: self.system_prompt.clone(),
            chat_history: vec![],
            tools: vec![],
            temperature: Some(self.config.temperature),
            max_tokens: Some(self.config.max_tokens),
        };
        let response = self.model.completion(request).await?;
        let choice = response.choice.first().ok_or(AgentError::NoChoiceFound)?;
        match ToOwned::to_owned(choice) {
            llm::completion::AssistantContent::Text(text) => Ok(text.text),
            llm::completion::AssistantContent::ToolCall(_) => {
                unreachable!("We don't provide tools")
            },
        }
    }

    pub fn tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        let toolname = tool.name();
        let definition = tool.definition();
        self.tools.push(definition);
        self.tools_impl.insert(toolname, Arc::new(tool));
        self
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Handle error in attempts
    async fn handle_error_in_attempts(&self, task: &str, error: AgentError, attempt: u32) {
        let err_msg = format!("Attempt {}, task: {}, failed: {}", attempt + 1, task, error);
        tracing::error!(err_msg);

        if self.config.autosave {
            let _ = self.save_task_state(task.to_owned()).await.map_err(|e| {
                tracing::error!(
                    "Failed to save agent<{}> task<{}>,  state: {}",
                    self.config.name,
                    task,
                    e
                )
            });
        }
    }
}

impl<M> Agent for SwarmsAgent<M>
where
    M: llm::Model + Clone + Send + Sync + 'static,
    M::RawCompletionResponse: Clone + Send + Sync,
{
    fn run(&self, task: String) -> BoxFuture<Result<String, AgentError>> {
        Box::pin(async move {
            self.short_memory.add(
                &task,
                &self.config.name,
                Role::User(self.config.user_name.clone()),
                &task,
            );

            // Plan
            if self.config.plan_enabled {
                self.plan(task.clone()).await?;
            }

            // Query long term memory
            // if self.long_term_memory.is_some() {
            //     self.query_long_term_memory(task.clone()).await?;
            // }

            // Save state
            if self.config.autosave {
                self.save_task_state(task.clone()).await?;
            }

            // Run agent loop
            let mut last_response_text = String::new();
            let mut task_complete = false;
            let mut was_prev_call_task_evaluator = false;
            for loop_count in 0..self.config.max_loops {
                if task_complete {
                    break;
                }

                let current_prompt: String;

                if was_prev_call_task_evaluator {
                    current_prompt = format!(
                        "You previously called task_evaluator and indicated the task was not complete. The required next step or context provided was: '{}'. \
                        Focus ONLY on addressing this context. DO NOT call task_evaluator again in this turn. Proceed with the task based on the context.",
                        last_response_text // last_response is the context provided by task_evaluator
                    );

                    was_prev_call_task_evaluator = false;
                } else if loop_count > 0 {
                    current_prompt = format!(
                        "Now, you are in loop {} of {}, The dialogue will terminate upon reaching maximum iteration count. You must:
                         - Complete the user's task before termination
                         - Optimize loop efficiency
                         - Minimize resource consumption through minimal iterations

                        You should consider to use tools if they can help, but only if they are relevant to the task and are necessary for the task.
                        origin task:\n{}",
                        loop_count + 1,
                        self.config.max_loops,
                        task
                    )
                } else {
                    // first loop
                    // task is already in short_memory, short_memory will be passed to llm
                    // empty prompt should be ignored by LLM provider
                    current_prompt = "".to_owned();
                }

                let mut success = false;
                // let task_prompt = self.short_memory.get(&task).unwrap().to_string(); // Safety: task is in short_memory
                for attempt in 0..self.config.retry_attempts {
                    if success {
                        break;
                    }

                    // if self.long_term_memory.is_some() && self.config.rag_every_loop {
                    //     // FIXME: if RAG success, but then LLM fails, then RAG is not removed and maybe causes issues
                    //     if let Err(e) = self.query_long_term_memory(task_prompt.clone()).await {
                    //         self.handle_error_in_attempts(&task, e, attempt).await;
                    //         continue;
                    //     };
                    // }

                    // Generate response using LLM
                    let history = self.short_memory.get(&task).unwrap(); // Safety: task is in short_memory
                    let current_chat_response =
                        match self.chat(&current_prompt, history.deref()).await {
                            Ok(response) => response,
                            Err(e) => {
                                self.handle_error_in_attempts(&task, e, attempt).await;
                                continue;
                            },
                        };
                    // needed to drop the lock
                    // if use:
                    // let history = (&(*self.short_memory.get(&task).unwrap())).into();
                    // we don't need to drop the lock, because the lock is owned by temporary variable
                    drop(history);

                    // handle ChatResponse
                    let mut assistant_memory_content = String::new();
                    let mut is_task_evaluator_called = false;
                    match current_chat_response {
                        ChatResponse::Text(text) => {
                            last_response_text = text.clone();
                            assistant_memory_content = text;
                        },
                        ChatResponse::ToolCalls(tool_calls) => {
                            let mut formatted_tool_results = String::new();
                            for tool_call in tool_calls {
                                let formatted = format!(
                                    "[Tool name]: {}\n[Tool args]: {}\n[Tool result]: {}\n\n",
                                    tool_call.name, tool_call.args, tool_call.result
                                );
                                formatted_tool_results.push_str(&formatted);
                                if tool_call.name == ToolDyn::name(&TaskEvaluator) {
                                    is_task_evaluator_called = true;
                                    match serde_json::from_str::<TaskStatus>(&tool_call.result) {
                                        Ok(task_status) => {
                                            tracing::info!(
                                                "Task evaluator tool called, task status: {:#?}",
                                                task_status,
                                            );

                                            match task_status {
                                                TaskStatus::Complete => {
                                                    task_complete = true;
                                                    // Task is complete
                                                    // This may be a bit redundant, but it's here for clarity
                                                    // last_response_text = format!(
                                                    //     "Task marked as complete by task_evaluator. Result: {}",
                                                    //     tool_call.result
                                                    // );
                                                    assistant_memory_content = formatted; // Store the final tool call in memory
                                                },
                                                TaskStatus::Incomplete { context } => {
                                                    task_complete = false;
                                                    // If not complete, store the context for the next loop's prompt
                                                    last_response_text = context;
                                                    // Keep the raw tool result for memory
                                                    assistant_memory_content = formatted;
                                                },
                                            }
                                        },
                                        Err(e) => {
                                            tracing::error!(
                                                "Failed to parse task status from task_evaluator: {}. Raw result: {}",
                                                e,
                                                tool_call.result
                                            );

                                            task_complete = false;

                                            last_response_text = format!(
                                                "Error parsing task_evaluator result. Raw output: {}",
                                                tool_call.result
                                            );
                                            assistant_memory_content = formatted; // Store the problematic call
                                        },
                                    }
                                } else {
                                    // Handle other tool calls if necessary, for now just format them
                                    // If this is the *only* response part, update last_response_text
                                    if formatted_tool_results.len() == formatted.len() {
                                        // Check if it's the first/only tool result string being built
                                        last_response_text = formatted_tool_results.clone();
                                    } else {
                                        // Append to existing text/tool results for the final response string
                                        last_response_text.push_str(&formatted);
                                    }
                                }
                            }
                            // If multiple tools were called, or if task_evaluator wasn't the only one,
                            // ensure assistant_memory_content reflects all calls.
                            if assistant_memory_content.is_empty() || !is_task_evaluator_called {
                                assistant_memory_content = formatted_tool_results.clone();
                                // Update last_response_text if it wasn't set by task_evaluator
                                if !is_task_evaluator_called {
                                    last_response_text = formatted_tool_results;
                                }
                            }
                        },
                    }

                    // Update the flag for the *next* iteration based on *this* iteration's call
                    was_prev_call_task_evaluator = is_task_evaluator_called && !task_complete;

                    self.short_memory.add(
                        &task,
                        &self.config.name,
                        Role::Assistant(self.config.name.to_owned()),
                        assistant_memory_content.clone(), // Add the text or formatted tool calls
                    );

                    // TODO: evaluate response
                    // TODO: Sentiment analysis

                    success = true;
                }

                if !success {
                    // Exit the loop if all retry failed
                    break;
                }

                // Save state in each loop
                if self.config.autosave {
                    self.save_task_state(task.clone()).await?;
                }

                if self.is_response_complete(last_response_text.clone()) {
                    break;
                }

                // TODO: Loop interval, maybe add a sleep here
            }

            // TODO: Apply the cleaning function to the responses
            // clean and add to short memory. role: Assistant(Output Cleaner)

            // Save state
            if self.config.autosave {
                self.save_task_state(task.clone()).await?;
            }

            // TODO: Handle artifacts

            // TODO: More flexible output types, e.g. JSON, CSV, etc.
            Ok(self
                .short_memory
                .get(&task)
                .expect("Task should exist in short memory")
                .to_string())
        })
    }

    fn run_multiple_tasks(
        &mut self,
        tasks: Vec<String>,
    ) -> BoxFuture<Result<Vec<String>, AgentError>> {
        let agent_name = self.name();
        let mut results = Vec::with_capacity(tasks.len());

        Box::pin(async move {
            let agent_arc = Arc::new(self);
            let (tx, mut rx) = mpsc::channel(1);
            stream::iter(tasks)
                .for_each_concurrent(None, |task| {
                    let tx = tx.clone();
                    let agent = Arc::clone(&agent_arc);
                    async move {
                        let result = agent.run(task.clone()).await;
                        tx.send((task, result)).await.unwrap(); // Safety: we know rx is not dropped
                    }
                })
                .await;
            drop(tx);

            while let Some((task, result)) = rx.recv().await {
                match result {
                    Ok(result) => {
                        results.push(result);
                    },
                    Err(e) => {
                        tracing::error!("| Agent: {} | Task: {} | Error: {}", agent_name, task, e);
                    },
                }
            }

            Ok(results)
        })
    }

    fn plan(&self, task: String) -> BoxFuture<Result<(), AgentError>> {
        Box::pin(async move {
            if let Some(planning_prompt) = &self.config.planning_prompt {
                let planning_prompt = format!("{} {}", planning_prompt, task);
                let plan = self.prompt(planning_prompt).await?;
                tracing::debug!("Plan: {}", plan);
                // Add plan to memory
                self.short_memory.add(
                    task,
                    self.config.name.clone(),
                    Role::Assistant(self.config.name.clone()),
                    plan,
                );
            };
            Ok(())
        })
    }

    fn query_long_term_memory(&self, _task: String) -> BoxFuture<Result<(), AgentError>> {
        unimplemented!("query_long_term_memory not implemented")
    }

    fn save_task_state(&self, task: String) -> BoxFuture<Result<(), AgentError>> {
        let mut hasher = XxHash3_64::default();
        task.hash(&mut hasher);
        let task_hash = hasher.finish();
        let task_hash = format!("{:x}", task_hash & 0xFFFFFFFF); // lower 32 bits of the hash

        Box::pin(async move {
            let save_state_dir = self.config.save_state_dir.clone();
            if let Some(save_state_dir) = save_state_dir {
                let save_state_dir = Path::new(&save_state_dir);
                if !save_state_dir.exists() {
                    tokio::fs::create_dir_all(save_state_dir).await?;
                }

                let path = save_state_dir
                    .join(format!("{}_{}", self.name(), task_hash))
                    .with_extension("json");

                let json = serde_json::to_string_pretty(&self.short_memory.get(&task).unwrap())?; // TODO: Safety?
                persistence::save_to_file(&json, path).await?;
            }
            Ok(())
        })
    }

    fn is_response_complete(&self, response: String) -> bool {
        self.config
            .stop_words
            .iter()
            .any(|word| response.contains(word))
    }

    fn id(&self) -> String {
        self.config.id.clone()
    }

    fn name(&self) -> String {
        self.config.name.clone()
    }

    fn description(&self) -> String {
        self.config
            .description
            .clone()
            .unwrap_or(self.system_prompt.clone().unwrap_or_default())
    }

    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }
}

/// Represents the output of the chat function, distinguishing text from tool calls.
#[derive(Debug, Clone)]
pub enum ChatResponse {
    /// Plain text response from the LLM.
    Text(String),
    /// Results from executing one or more tool calls requested by the LLM.
    ToolCalls(Vec<ToolCallOutput>),
}

/// Holds the details of a single executed tool call.
#[derive(Debug, Clone, Serialize, Deserialize)] // Added Serialize/Deserialize for potential future use
pub struct ToolCallOutput {
    /// The name of the tool that was called.
    pub name: String,
    /// The arguments (as a JSON string) passed to the tool.
    pub args: String,
    /// The result (as a string) returned by the tool's execution.
    pub result: String,
}

#[tool(description = r#"
    **Important**
    If previous message is a `task_evaluator` call, then you shouldn't call this tool.

    **Task Evaluator**
    Finalize or request refinement for the current task.
    
    Call this when:
    - All user requirements are fully satisfied (set `Complete`)
    - Avoids unnecessary iterations, redundancy, or waste.
    - Additional input/clarification is needed (set `Incomplete` with context)
    
    When `Complete`, the context is ignored, because the dialogue will terminate.
    When `Incomplete`, your context becomes the system's next prompt, enabling iterative task refinement.
    Provide clear, actionable contexts to guide the next steps, the context should be used to guide yourself to complete the task.
"#)]
fn task_evaluator(status: TaskStatus) -> Result<TaskStatus, TaskEvaluatorError> {
    Ok(status)
}

/// Tracks task status and provides feedback for incomplete tasks.
/// When `Incomplete`, the `context` becomes the system's next prompt.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub enum TaskStatus {
    /// Indicates that the task is complete.
    Complete,
    /// Indicates that the task is incomplete.
    Incomplete {
        /// Required guidance when task is incomplete:
        /// - Clear description of missing elements
        /// - Specific questions needing clarification  
        /// - Remaining steps to completion
        context: String,
    },
}

#[derive(Debug, Error)]
pub enum TaskEvaluatorError {}
