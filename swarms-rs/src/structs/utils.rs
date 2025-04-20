use chrono::Local;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::structs::{
    agent::{Agent, AgentError},
    swarm::AgentOutputSchema,
};

pub async fn run_agent_with_output_schema(
    agent: &dyn Agent,
    task: String,
) -> Result<AgentOutputSchema, AgentError> {
    let start = Local::now();
    let output = agent.run(task.clone()).await?;

    let end = Local::now();
    let duration = end.signed_duration_since(start).num_seconds();

    let agent_output = AgentOutputSchema {
        run_id: Uuid::new_v4(),
        agent_name: agent.name(),
        task,
        output,
        start,
        end,
        duration,
    };

    Ok(agent_output)
}

/// Extracts all tool calls (name, args, and optional result) from a multi-line response string.
///
/// Assumes the format:
/// [Tool name]: tool_name
/// [Tool args]: {"arg1": "value1"}
/// [Tool result]: {"output": "result"}
/// ... potentially other lines or more tool calls ...
///
/// Handles multiple calls, including multiple calls to the same tool.
pub fn extract_tool_calls(response: &str) -> Vec<ToolCall<'_>> {
    let mut calls = Vec::new();
    let lines = response.lines().map(str::trim); // Iterate over trimmed lines

    let mut current_name: Option<&str> = None;
    let mut current_args: Option<&str> = None;

    for line in lines {
        if line.starts_with("[Tool name]:") {
            // If we were already tracking a name and args, finalize the previous call.
            // This handles cases where a result wasn't immediately following args,
            // or consecutive tool calls.
            if let (Some(name), Some(args)) = (current_name.take(), current_args.take()) {
                // The result would have been captured in the previous iteration if present.
                // If we reach here, it means no result line followed the args directly.
                calls.push(ToolCall {
                    name,
                    args,
                    result: None, // Assume None if no result line was found before this new name
                });
            }
            // Start tracking the new tool name.
            current_name = Some(line.trim_start_matches("[Tool name]:").trim());
            // Reset args for the new tool call.
            current_args = None;
        } else if line.starts_with("[Tool args]:") {
            // Only associate args if we are currently tracking a tool name.
            if current_name.is_some() {
                current_args = Some(line.trim_start_matches("[Tool args]:").trim());
            }
        } else if line.starts_with("[Tool result]:") {
            // If we find a result and have both name and args pending,
            // record the complete tool call and reset.
            if let (Some(name), Some(args)) = (current_name.take(), current_args.take()) {
                let result_str = line.trim_start_matches("[Tool result]:").trim();
                calls.push(ToolCall {
                    name,
                    args,
                    result: Some(result_str),
                });
            }
            // Reset state after processing a result, regardless of whether it was associated.
            // This prevents associating a result with a later tool call if it was orphaned.
            current_name = None;
            current_args = None;
        }
        // Ignore any other lines that don't match the expected prefixes.
    }

    // After the loop, check if there's a pending tool call (name and args)
    // that didn't get a result line before the end of the input.
    if let (Some(name), Some(args)) = (current_name, current_args) {
        calls.push(ToolCall {
            name,
            args,
            result: None,
        });
    }

    calls
}

/// Represents information about a single tool call extracted from agent response text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall<'a> {
    /// The name of the tool called.
    pub name: &'a str,
    /// The arguments passed to the tool, as a string.
    pub args: &'a str,
    /// The result returned by the tool, as a string, if present.
    pub result: Option<&'a str>,
}
