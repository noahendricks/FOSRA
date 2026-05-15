mod tui;

mod action;
mod app;
mod components;

use color_eyre::eyre::Result;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let mut app = app::App::new();
    app.run().await
}
