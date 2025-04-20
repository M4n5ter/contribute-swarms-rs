use std::ops::Deref;

use chrono::{DateTime, Local};
use dashmap::DashMap;
use erased_serde::Serialize as ErasedSerialize;
use futures::future::BoxFuture;
use serde::Serialize;
use thiserror::Error;
use uuid::Uuid;

use crate::structs::{
    concurrent_workflow::ConcurrentWorkflowError, hierarchical_swarm::HierarchicalSwarmError,
};

pub trait Swarm {
    fn name(&self) -> &str;

    fn run(&self, task: String) -> BoxFuture<Result<Box<dyn ErasedSerialize>, SwarmError>>;
}

#[derive(Debug, Error)]
pub enum SwarmError {
    #[error("ConcurrentWorkflowError: {0}")]
    ConcurrentWorkflowError(#[from] ConcurrentWorkflowError),
    #[error("HierarchicalSwarmError: {0}")]
    HierarchicalSwarmError(#[from] HierarchicalSwarmError),
}

#[derive(Clone, Default, Serialize)]
pub struct MetadataSchemaMap(DashMap<String, MetadataSchema>);

impl MetadataSchemaMap {
    pub fn add(&self, task: impl Into<String>, metadata: MetadataSchema) {
        self.insert(task.into(), metadata);
    }
}

impl Deref for MetadataSchemaMap {
    type Target = DashMap<String, MetadataSchema>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Default, Serialize)]
pub struct MetadataSchema {
    pub swarm_id: Uuid,
    pub task: String,
    pub description: String,
    pub agents_output_schema: Vec<AgentOutputSchema>,
    pub timestamp: DateTime<Local>,
}

#[derive(Clone, Serialize)]
pub struct AgentOutputSchema {
    pub run_id: Uuid,
    pub agent_name: String,
    pub task: String,
    pub output: String,
    pub start: DateTime<Local>,
    pub end: DateTime<Local>,
    pub duration: i64,
}
