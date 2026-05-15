use color_eyre::eyre::Result;
use ratatui::Frame;
use tokio::sync::mpsc::{Receiver, Sender};

use fosra::{Focus, Message, Role};

use crate::{
    action::Action,
    components::conversation::conversation::Conversation,
    tui::{Event, Tui},
};

/// Application state for the TUI.
pub struct App {
    pub messages: Vec<Message>,
    pub focus: Focus,
    pub should_quit: bool,

    pub conversation: Conversation,
    action_rx: Receiver<Action>,

    pub action_tx: Sender<Action>,
}

impl App {
    pub fn new() -> Self {
        let (action_tx, action_rx) = tokio::sync::mpsc::channel(256);

        Self {
            messages: Vec::new(),
            focus: Focus::Conversation,
            should_quit: false,
            conversation: Conversation::new(),
            action_rx,
            action_tx,
        }
    }

    pub fn action_sender(&self) -> Sender<Action> {
        self.action_tx.clone()
    }

    pub async fn run(&mut self) -> Result<()> {
        let mut tui = Tui::new()?
            .tick_rate(4.0)
            .frame_rate(60.0)
            .mouse(true)
            .paste(true);

        tui.enter()?;

        loop {
            while let Ok(action) = self.action_rx.try_recv() {
                self.update(action);
            }

            if let Some(event) = tui.next().await {
                if let Some(action) = self.handle_event(&event) {
                    let mut next = Some(action);
                    while let Some(a) = next {
                        next = self.update(a);
                    }
                }

                if let Event::Render = event {
                    tui.draw(|f| self.render(f))?;
                }
            } else {
                break;
            }
            if self.should_quit {
                break;
            }
        }
        tui.exit();
        Ok(())
    }

    fn handle_event(&self, event: &Event) -> Option<Action> {
        use ratatui::crossterm::event::{KeyCode, KeyModifiers};

        match event {
            Event::Tick => Some(Action::Tick),
            Event::Render => Some(Action::Render),
            Event::Resize(w, h) => Some(Action::Resize(*w, *h)),
            Event::FocusGained => Some(Action::FocusGained),
            Event::FocusLost => Some(Action::FocusLost),

            Event::Key(key) => match (key.modifiers, key.code) {
                (KeyModifiers::CONTROL, KeyCode::Char('c')) => Some(Action::Quit),
                (KeyModifiers::NONE, KeyCode::Tab) => Some(Action::FocusNext),
                (KeyModifiers::NONE, KeyCode::Up) => Some(Action::ScrollUp),
                (KeyModifiers::NONE, KeyCode::Down) => Some(Action::ScrollDown),
                (KeyModifiers::NONE, KeyCode::Backspace) => Some(Action::InputBackspace),
                (KeyModifiers::NONE, KeyCode::Char('u')) => Some(Action::InputClear),
                (KeyModifiers::NONE, KeyCode::Char(c)) => Some(Action::InputChar(c)),
                _ => None,
            },

            Event::Paste(s) => {
                let _ = s;
                None
            }

            Event::Mouse(_) | Event::Error => None,
        }
    }

    pub fn update(&mut self, action: Action) -> Option<Action> {
        match action {
            Action::Resize(_, _) => None,

            Action::Quit => {
                self.should_quit = true;
                None
            }

            Action::Tick | Action::Render | Action::FocusGained | Action::FocusLost => None,

            Action::FocusNext => {
                self.focus = match self.focus {
                    Focus::Conversation => Focus::Conversation,
                };
                None
            }

            Action::ScrollUp => {
                match self.focus {
                    Focus::Conversation => self.conversation.scroll_up(),
                    _ => {}
                }
                None
            }

            Action::ScrollDown => {
                match self.focus {
                    Focus::Conversation => self.conversation.scroll_down(),
                    _ => {}
                }
                None
            }

            Action::InputChar(c) => {
                let _ = c;
                None
            }

            Action::InputBackspace => None,
            Action::InputClear => None,

            Action::Submit(text) => {
                let _ = text;
                None
            }

            Action::LLMToken(token) => {
                self.conversation.push_token(&token);
                None
            }

            Action::LLMDone => {
                let finished = self.conversation.finish_stream();
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: finished,
                });
                None
            }

            Action::LLMError(e) => {
                self.conversation.finish_stream();
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: format!("[error] {e}"),
                });
                None
            }

            Action::Chunks(chunks) => {
                let _ = chunks;
                None
            }

            Action::Error(msg) => {
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: format!("[error] {msg}"),
                });
                None
            }
        }
    }

    fn render(&mut self, f: &mut Frame) {
        use ratatui::layout::{Constraint, Direction, Layout};

        let area = f.area();

        let vertical = Layout::new(
            Direction::Vertical,
            [Constraint::Min(1), Constraint::Length(3)],
        )
        .split(area);

        let horizontal = Layout::new(
            Direction::Horizontal,
            [Constraint::Percentage(70), Constraint::Percentage(30)],
        )
        .split(vertical[0]);

        let conversation = horizontal[0];

        self.conversation.render(
            f,
            conversation,
            &self.messages,
            self.focus == Focus::Conversation,
        );
    }
}
