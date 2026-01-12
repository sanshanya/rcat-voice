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
use rcat_voice::metrics::{MetricsSink, TracingMetricsSink};
use rcat_voice::streaming::StreamSessionBuilder;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY is required for deepseek_stream example")?;
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());

    let tts_engine = generator::build_from_env()?;
    let metrics: Arc<dyn MetricsSink> = Arc::new(TracingMetricsSink::from_env());
    let turn_id = 1;
    let session = StreamSessionBuilder::from_env(tts_engine)
        .turn_id(turn_id)
        .metrics_sink(metrics)
        .build();
    let control = session.control();
    control.mark_llm_start();

    let messages = vec![
        serde_json::json!({"role":"user","content":"请用两三句话解释为什么首段短、后续段长更适合流式TTS。"}),
    ];
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let cancel_ctrl = control.clone();
    let cancel_handle = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = cancel_tx.send(true);
            let _ = cancel_ctrl.cancel().await;
        }
    });

    let delta_tx = control.sender();
    let sse_handle = tokio::spawn(sse_stream_chat(
        base_url, api_key, model, messages, delta_tx, cancel_rx,
    ));

    if let Err(e) = sse_handle.await? {
        info!("SSE task finished with error: {:?}", e);
    } else {
        info!("SSE task finished successfully.");
    }

    let _ = cancel_handle.await;
    session.shutdown().await?;
    Ok(())
}

async fn sse_stream_chat(
    base_url: String,
    api_key: String,
    model: String,
    messages_json: Vec<Value>,
    delta_tx: mpsc::Sender<String>,
    mut cancel: watch::Receiver<bool>,
) -> Result<()> {
    let config = OpenAIConfig::new()
        .with_api_key(api_key)
        .with_api_base(base_url);

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
        .model(model)
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

    Ok(())
}
