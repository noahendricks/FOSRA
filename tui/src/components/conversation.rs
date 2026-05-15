pub mod conversation {
use fosra::{Message, Role};

    use ratatui::{
        Frame,
        layout::Rect,
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    };

    pub struct Conversation {
        scroll_offset: u16,
        stream_buffer: String,
        streaming: bool,
    }

    impl Conversation {
        pub fn new() -> Self {
            Self {
                scroll_offset: 0,
                stream_buffer: String::new(),
                streaming: false,
            }
        }

        pub fn begin_stream(&mut self) {
            self.stream_buffer.clear();
            self.streaming = true;
            self.scroll_offset = 0;
        }

        pub fn push_token(&mut self, token: &str) {
            self.stream_buffer.push_str(token)
        }

        pub fn finish_stream(&mut self) -> String {
            self.streaming = false;
            std::mem::take(&mut self.stream_buffer)
        }

        pub fn scroll_up(&mut self) {
            self.scroll_offset = self.scroll_offset.saturating_add(1);
        }

        pub fn scroll_down(&mut self) {
            self.scroll_offset = self.scroll_offset.saturating_sub(1);
        }

        pub fn render(&self, f: &mut Frame, area: Rect, messages: &[Message], focused: bool) {
            let border_style = if focused {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            let block = Block::default()
                .title(" Chat ")
                .borders(Borders::ALL)
                .border_style(border_style);

            let inner = block.inner(area);

            f.render_widget(block, area);

            // building existing messagse
            let mut items: Vec<ListItem> = messages
                .iter()
                .map(|msg| {
                    let (label, color) = match msg.role {
                        Role::User => ("You", Color::Green),
                        Role::Assistant => ("AI", Color::Cyan),
                    };

                    let header = Line::from(vec![Span::styled(
                        format!("{label}: "),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    )]);

                    let body = Line::from(Span::raw(msg.content.clone()));

                    ListItem::new(vec![header, body, Line::from("")])
                })
                .collect();

            // stream llm message
            if self.streaming && !self.stream_buffer.is_empty() {
                let header = Line::from(vec![
                    Span::styled(
                        "AI: ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("||", Style::default().fg(Color::Yellow)),
                ]);

                // token streaming
                let body = Line::from(Span::raw(self.stream_buffer.clone()));

                items.push(ListItem::new(vec![header, body]));
            }

            let list = List::new(items);
            f.render_widget(list, inner);
        }
    }
}
