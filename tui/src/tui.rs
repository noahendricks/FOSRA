use std::{
    ops::{Deref, DerefMut},
    time::Duration,
};

use color_eyre::eyre::{Result, eyre};
use futures::{FutureExt, StreamExt};
use ratatui::{
    backend::CrosstermBackend as Backend,
    crossterm::{
        cursor,
        event::{
            DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
            Event as CrosstermEvent, KeyEvent, KeyEventKind, MouseEvent,
        },
        terminal::{EnterAlternateScreen, LeaveAlternateScreen},
    },
};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::mpsc::{self, Receiver, Sender},
    task,
};
use tokio_util::sync::CancellationToken;

const CHANNEL_CAPACITY: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Event {
    /// render tick
    Render,

    /// logic tick
    Tick,

    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),

    /// bracketed paste content
    Paste(String),

    /// terminal focus.
    FocusGained,
    FocusLost,

    /// crossterm backend stream error | fatal
    Error,
}

pub struct Tui {
    pub terminal: ratatui::Terminal<Backend<std::io::Stderr>>,
    task: task::JoinHandle<()>,
    cancellation_token: CancellationToken,
    event_rx: Receiver<Event>,
    pub event_tx: Sender<Event>,
    frame_rate: f64,
    tick_rate: f64,
    mouse: bool,
    paste: bool,
}

impl Tui {
    pub fn new() -> Result<Self> {
        let (event_tx, event_rx) = mpsc::channel(CHANNEL_CAPACITY);

        Ok(Self {
            terminal: ratatui::Terminal::new(Backend::new(std::io::stderr()))?,
            task: tokio::spawn(async {}),
            cancellation_token: CancellationToken::new(),
            event_rx,
            event_tx,
            frame_rate: 60.0,
            tick_rate: 4.0,
            mouse: false,
            paste: false,
        })
    }

    // builders — called prior to enter().
    pub fn tick_rate(mut self, tick_rate: f64) -> Self {
        self.tick_rate = tick_rate;
        self
    }

    pub fn frame_rate(mut self, frame_rate: f64) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    pub fn mouse(mut self, mouse: bool) -> Self {
        self.mouse = mouse;
        self
    }

    pub fn paste(mut self, paste: bool) -> Self {
        self.paste = paste;
        self
    }

    /// enable raw mode, enter alternate screen, and spawn event task.
    pub fn enter(&mut self) -> Result<()> {
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(std::io::stderr(), EnterAlternateScreen, cursor::Hide)?;
        if self.mouse {
            crossterm::execute!(std::io::stderr(), EnableMouseCapture)?;
        }
        if self.paste {
            crossterm::execute!(std::io::stderr(), EnableBracketedPaste)?;
        }
        self.start();
        Ok(())
    }

    /// disable raw mode, exit alternate screen, and stop event task.
    pub fn exit(&mut self) -> Result<()> {
        // signal task to stop, wait for a clean finish
        self.cancellation_token.cancel();

        task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                tokio::time::timeout(Duration::from_millis(200), &mut self.task).await
            })
        })
        .unwrap_or_else(|_| {
            // timed out — abort task rather leaking it.
            self.task.abort();
            Ok(())
        })?;

        if crossterm::terminal::is_raw_mode_enabled()? {
            self.flush()?;
            if self.paste {
                crossterm::execute!(std::io::stderr(), DisableBracketedPaste)?;
            }
            if self.mouse {
                crossterm::execute!(std::io::stderr(), DisableMouseCapture)?;
            }
            crossterm::execute!(std::io::stderr(), LeaveAlternateScreen, cursor::Show)?;
            crossterm::terminal::disable_raw_mode()?;
        }
        Ok(())
    }

    /// receive the next event. Returns `None` only when the sender is dropped
    pub async fn next(&mut self) -> Option<Event> {
        self.event_rx.recv().await
    }

    /// cloned sender for external async task injection
    pub fn sender(&self) -> Sender<Event> {
        self.event_tx.clone()
    }

    #[cfg(unix)]
    pub fn suspend(&mut self) -> Result<()> {
        self.exit()?;
        signal_hook::low_level::raise(signal_hook::consts::signal::SIGTSTP)?;
        Ok(())
    }

    pub fn resume(&mut self) -> Result<()> {
        self.enter()
    }

    /// spawn background event task
    fn start(&mut self) {
        let tick_delay = Duration::from_secs_f64(1.0 / self.tick_rate);
        let render_delay = Duration::from_secs_f64(1.0 / self.frame_rate);

        // cancel previously running task before replacing
        self.cancellation_token.cancel();
        self.cancellation_token = CancellationToken::new();

        let token = self.cancellation_token.clone();
        let tx = self.event_tx.clone();

        self.task = tokio::spawn(async move {
            let mut reader = crossterm::event::EventStream::new();
            let mut tick_interval = tokio::time::interval(tick_delay);
            let mut render_interval = tokio::time::interval(render_delay);

            // skip missed ticks rather than bursting to catch up.  prevents flood of stale main loop render / tick events
            tick_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            render_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                let crossterm_event = reader.next().fuse();

                tokio::select! {
                    biased; // check cancellation first every iteration

                    _ = token.cancelled() => break,

                    maybe_event = crossterm_event => {
                        match maybe_event {
                            Some(Ok(evt)) => {
                                let event = match evt {
                                    CrosstermEvent::Key(key) if key.kind == KeyEventKind::Press => {
                                        Some(Event::Key(key))
                                    }
                                    CrosstermEvent::Key(_) => None, // ignore release and repeats
                                    CrosstermEvent::Mouse(mouse)    => Some(Event::Mouse(mouse)),
                                    CrosstermEvent::Resize(x, y)    => Some(Event::Resize(x, y)),
                                    CrosstermEvent::FocusGained     => Some(Event::FocusGained),
                                    CrosstermEvent::FocusLost       => Some(Event::FocusLost),
                                    CrosstermEvent::Paste(s)        => Some(Event::Paste(s)),
                                };
                                if let Some(e) = event {
                                    //  send error when main loop dropped | peaceful break
                                    if tx.send(e).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            Some(Err(_)) => {
                                let _ = tx.send(Event::Error).await;
                                break; // crossterm stream is broken | fatal
                            }
                            None => break, // stream ended
                        }
                    }

                    _ = tick_interval.tick() => {
                        if tx.send(Event::Tick).await.is_err() { break; }
                    }

                    _ = render_interval.tick() => {
                        if tx.send(Event::Render).await.is_err() { break; }
                    }
                }
            }
        });
    }
}

// allows tui.draw(...)
impl Deref for Tui {
    type Target = ratatui::Terminal<Backend<std::io::Stderr>>;
    fn deref(&self) -> &Self::Target {
        &self.terminal
    }
}

impl DerefMut for Tui {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.terminal
    }
}

// drop for restore on panic
impl Drop for Tui {
    fn drop(&mut self) {
        if let Err(e) = self.exit() {
            // can't propagate
            eprintln!("[tui] exit error during drop: {e}");
        }
    }
}
