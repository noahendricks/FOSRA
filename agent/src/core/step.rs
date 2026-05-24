use futures::future::BoxFuture;
use rand::{Rng, RngExt, rng};
use std::{pin::Pin, sync::Arc};

use tokio::{
    sync::RwLock,
    time::{Duration, sleep},
};

use crate::{StepError, StepEvent, core::types::SessionState};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

pub enum Step {
    Run {
        func: Arc<dyn Fn(&mut SessionState) -> Result<StepOutcome, StepError> + Send + Sync>,
        timeout: Option<Duration>,
    },
    If {
        pred: Arc<dyn Fn(&SessionState) -> bool + Send + Sync>,
        then: Box<Step>,
        else_: Option<Box<Step>>,
    },
    Parallel(Vec<Step>),
    Until {
        max_iter: i32,
        condition: Arc<dyn Fn(&SessionState) -> bool + Send + Sync>,
        body: Box<Step>,
    },
    Checkpoint {
        label: Arc<dyn Fn(&SessionState) -> String + Send + Sync>,
        should_halt: Arc<dyn Fn(&SessionState) -> bool + Send + Sync>,
    },
    Retry {
        max_attempts: i32,
        body: Box<Step>,
        initial_wait: Duration,
        max_wait: Duration,
        retry_on: Box<dyn Fn(&SessionState) -> bool + Send + Sync>,
    },

    FanOut {
        items_fn: Box<dyn Fn(&SessionState) -> Vec<serde_json::Value> + Send + Sync>,
        body: Box<dyn Fn(&SessionState) -> Vec<serde_json::Value> + Send + Sync>,
    },
    Guarded {
        step: Box<Step>,
        handled: Box<Step>,
    },
    Agents {
        scope: String,
        extract: Box<dyn Fn(&SessionState) -> serde_json::Value + Send + Sync>,
        steps: Box<Step>,
        merge: Box<dyn Fn(&mut SessionState) -> Result<(), StepError> + Send + Sync>,
    },
    Tool {
        //llm_call: Box<dyn Fn(&[Message]) -> Result<LLMResponse, CoreErorr>>,
        //tools: Map<String, ToolDef>,
        //execution: ToolExecution,
        //on_error: ToolErrorPolicy,
        max_tool_rounds: u8,
    },
    Complete {
        placehold: String,
    },
}

impl Step {
    pub async fn run_closure(
        &self,
        func: &Arc<dyn Fn(&mut SessionState) -> Result<StepOutcome, StepError> + Send + Sync>,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        if let Ok(mut guard) = state.try_write() {
            func(&mut *guard)
        } else {
            Err(StepError::Domain(String::from("closure failure")))
        }
    }

    pub async fn run_parallel_closures(
        &self,
        callbacks: Vec<
            &Arc<dyn Fn(&mut SessionState) -> Result<StepOutcome, StepError> + Send + Sync>,
        >,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        //parallel closures ran async
        // errors returned as errors
        // state + errors merged on completion
        todo!("")
    }

    // step execute functions

    pub fn execute_step<'a>(
        &'a self,
        state: &'a Arc<RwLock<SessionState>>,
    ) -> BoxFuture<'a, Result<StepOutcome, StepError>> {
        // self.emit_trace(StepEvent {
        //     StepStarted,
        //     step_name,
        //     step_depth,
        // });

        Box::pin(async move {
            match self {
                Self::Run { .. } => self.execute_run(state).await,
                Self::If { .. } => self.execute_if(state).await,
                Self::Until { .. } => self.execute_until(state).await,
                Self::Checkpoint { .. } => self.execute_checkpoint(state).await, // likely will
                Self::Retry { .. } => self.execute_retry(state).await,
                // Step::FanOut(fan_out) => self.execute_run(),
                // Step::Agent(agent_step) => self.execute_run(),
                // Step::Tool(tool_step) => self.execute_step(),
                // Step::Complete(tool_step) => self.execute_step(),
                _ => Err(StepError::Domain(String::new())),
            }
        })

        // self.emit_trace(StepEvent {
        //     StepCompleted,
        //     step_name,
        //     step_depth,
        // });
    }

    pub async fn execute_run(
        &self,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        let Self::Run { timeout, func } = self else {
            return Err(StepError::Domain("Run".to_string()));
        };

        let fut = self.run_closure(func, state);

        match timeout.or(Some(DEFAULT_TIMEOUT)) {
            Some(dur) => tokio::time::timeout(dur, fut)
                .await
                .map_err(|_| StepError::Domain("timeout".into()))?
                .map_err(|e| StepError::Domain(e.to_string())),
            None => fut.await.map_err(|e| StepError::Domain(e.to_string())),
        };

        Ok(StepOutcome::completed())
    }

    pub async fn execute_if(
        &self,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        let Self::If { pred, then, else_ } = self else {
            return Err(StepError::Domain("Run".to_string()));
        };

        if let Ok(mut guard) = state.try_write() {
            if pred(&mut guard) {
                then.execute_step(state).await
            } else {
                match else_ {
                    Some(else_step) => else_step.execute_step(state).await,
                    None => Ok(StepOutcome::completed()),
                }
            }
        } else {
            Err(StepError::Domain(String::from("closure failure")))
        }
    }

    pub async fn execute_until(
        &self,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        let Self::Until {
            max_iter,
            condition,
            body,
        } = self
        else {
            return Err(StepError::Domain("Until".to_string()));
        };

        let mut prev_body_result: Option<Result<StepOutcome, StepError>> = None;

        for _ in 0..*max_iter {
            let is_met = condition(
                &state
                    .try_read()
                    .map_err(|_| {
                        StepError::Domain("lock poisoned in until step execution".to_string())
                    })
                    .unwrap(),
            );
            if is_met {
                return prev_body_result.unwrap();
            } else {
                prev_body_result = Some(body.execute_step(state).await);
                continue;
            }
        }
        Ok(StepOutcome::completed())
    }

    pub async fn execute_checkpoint(
        &self,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        // serialize state -> checkpoint object
        // add label
        // check if should halt
        // return completed outcome + error
        todo!()
    }

    pub async fn execute_retry(
        &self,
        state: &Arc<RwLock<SessionState>>,
    ) -> Result<StepOutcome, StepError> {
        let Self::Retry {
            max_attempts,
            initial_wait,
            max_wait,
            retry_on,
            body,
        } = self
        else {
            return Err(StepError::Domain("Until".to_string()));
        };

        fn update_backoff(attempt: u32, base_ms: u64, cap_ms: u64) -> std::time::Duration {
            let mut generator = rng();
            let exp = base_ms.saturating_mul(2_u64.saturating_pow(attempt));
            let capped = exp.min(cap_ms);
            Duration::from_secs(generator.random_range(0..=capped))
        }

        let mut current_wait: Duration = *initial_wait; // 1 second
        let mut prev_body_outcome: Option<Result<StepOutcome, StepError>> = None;

        for i in 0..*max_attempts {
            // check if retry on
            if !retry_on(&state.try_read().unwrap()) {
                return prev_body_outcome.unwrap();
            }

            // wait interval
            sleep(current_wait).await;

            // execute body
            prev_body_outcome = Some(body.execute_step(state).await);

            // sleep - tokio::time::sleep(duration)
            current_wait = update_backoff(
                i.try_into().unwrap(),
                current_wait.as_secs(),
                max_wait.as_secs(),
            )
        }
        return prev_body_outcome.unwrap();
    }

    pub async fn execute_fanout() {
        //
        todo!()
    }
}
pub struct StepMetadata {
    pub event: StepEvent,
}



pub struct StepOutcome {
    pub halt: bool,
    pub force_checkpoint: bool,
}

impl StepOutcome {
    pub fn completed() -> StepOutcome {
        Self {
            halt: false,
            force_checkpoint: true,
        }
    }
}
