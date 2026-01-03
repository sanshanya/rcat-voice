use anyhow::{Context, Result};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
};
use futures::StreamExt;
use rcat_voice::generator;
use rcat_voice::streaming::{StreamControl, StreamSession};
use serde_json::json;
use serde_json::Value;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{error, info};

struct RunningChat {
    session: StreamSession,
    control: StreamControl,
    cancel_tx: watch::Sender<bool>,
    handle: JoinHandle<Result<()>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let base_url = Arc::new(
        std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string()),
    );
    let api_key = Arc::new(
        std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY is required for terminal_chat example")?,
    );
    let model = Arc::new(
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string()),
    );

    let tts_engine = generator::build_from_env()?;
    let mut current: Option<RunningChat> = None;

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    loop {
        print!("> ");
        io::stdout().flush().ok();

        let Some(line) = lines.next_line().await? else {
            break;
        };
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("/exit") || prompt.eq_ignore_ascii_case("/quit") {
            break;
        }

        if let Some(running) = current.take() {
            stop_running(running).await?;
        }

        let session = StreamSession::from_env(tts_engine.clone());
        let control = session.control();
        control.mark_llm_start();

        let (cancel_tx, cancel_rx) = watch::channel(false);
        let delta_tx = control.sender();

        let handle = tokio::spawn(sse_stream_chat(
            base_url.clone(),
            api_key.clone(),
            model.clone(),
            vec![json!({"role":"user","content": prompt})],
            delta_tx,
            cancel_rx,
        ));

        current = Some(RunningChat {
            session,
            control,
            cancel_tx,
            handle,
        });
    }

    if let Some(running) = current.take() {
        stop_running(running).await?;
    }

    Ok(())
}

async fn stop_running(running: RunningChat) -> Result<()> {
    let _ = running.cancel_tx.send(true);
    let _ = running.control.cancel().await;
    let _ = running.handle.await;
    running.session.shutdown().await?;
    Ok(())
}

async fn sse_stream_chat(
    base_url: Arc<String>,
    api_key: Arc<String>,
    model: Arc<String>,
    messages_json: Vec<Value>,
    delta_tx: mpsc::Sender<String>,
    mut cancel: watch::Receiver<bool>,
) -> Result<()> {
    let config = OpenAIConfig::new()
        .with_api_key((*api_key).clone())
        .with_api_base((*base_url).clone());

    let client = Client::with_config(config);

    let mut messages = Vec::new();
    for msg in messages_json {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let message = match role {
            "system" => ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(content)
                    .build()?,
            ),
            _ => ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(content)
                    .build()?,
            ),
        };
        messages.push(message);
    }

    let request = CreateChatCompletionRequestArgs::default()
        .model((*model).clone())
        .messages(messages)
        .stream(true)
        .build()?;

    let mut stream = client.chat().create_stream(request).await?;

    loop {
        tokio::select! {
            _ = cancel.changed() => {
                if *cancel.borrow() { break; }
            }
            maybe_chunk = stream.next() => {
                match maybe_chunk {
                    Some(Ok(response)) => {
                        for choice in response.choices {
                            if let Some(content) = choice.delta.content {
                                if delta_tx.send(content).await.is_err() {
                                    return Ok(());
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        error!("OpenAI Stream Error: {}", e);
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    info!("LLM stream finished.");
    Ok(())
}
