use crate::{CheckpointStore, StepError, Step, Tracer};
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct Executor<S, E> {
    _phantom: PhantomData<E>,
    state: Arc<RwLock<S>>,
    checkpoint_store: Arc<dyn CheckpointStore<S>>,
    tracer: Arc<dyn Tracer>,
    trace_id: Uuid,
}

impl<S, E> Executor<S, E> {
    pub fn new(
        state: S,
        checkpoint_store: Arc<dyn CheckpointStore<S>>,
        tracer: Arc<dyn Tracer>,
    ) -> Self {
        Self {
            state: Arc::new(RwLock::new(state)),
            checkpoint_store,
            tracer,
            trace_id: Uuid::now_v7(),
            _phantom: PhantomData,
        }
    }

    pub async fn run(&mut self, step: &Step) -> Result<(), StepError> {
        let _ = step;
        todo!("depth-first step tree interpreter")
    }
}
