use std::sync::Arc;
use tokio::sync::RwLock;
use fosra_agent::{StepEvent, Tracer};

pub struct NoOpTracer;

impl Tracer for NoOpTracer {
    fn emit(&self, _event: StepEvent) {
        // no-op
    }
}

pub struct StdoutTracer;

impl Tracer for StdoutTracer {
    fn emit(&self, event: StepEvent) {
        println!(
            "[trace:{}] {:?} step={} depth={}",
            event.trace_id, event.event, event.step_name, event.step_depth
        );
    }
}

pub struct InMemoryTracer {
    pub events: Arc<RwLock<Vec<StepEvent>>>,
}

impl InMemoryTracer {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl Tracer for InMemoryTracer {
    fn emit(&self, event: StepEvent) {
        let mut events = self.events.blocking_write();
        events.push(event);
    }
}
