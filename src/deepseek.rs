use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use tracing::error;

pub async fn sse_stream_chat(
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

    // Convert JSON messages to typed messages
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
