use crate::generator::{SynthesizedAudio, TtsEngine, TtsMetrics};
use crate::tokenizer::Segment;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep_until};
use tracing::{info, warn};

/// Pipeline scheduling knobs.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub parallel_synth: bool,
    pub synth_inflight: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel_synth: true,
            synth_inflight: 1,
        }
    }
}

impl PipelineConfig {
    pub fn from_env() -> Self {
        let parallel_synth = std::env::var("TTS_PARALLEL_SYNTH")
            .map(|v| v != "0")
            .unwrap_or(true);
        let synth_inflight = std::env::var("TTS_SYNTH_INFLIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        Self {
            parallel_synth,
            synth_inflight,
        }
    }
}

pub struct Pipeline {
    chunk_rx: mpsc::Receiver<Segment>,
    cancel_rx: watch::Receiver<bool>,
    pause_rx: watch::Receiver<bool>,
    engine: Arc<dyn TtsEngine>,
    config: PipelineConfig,
}

struct SynthOutcome {
    seq: u64,
    epoch: u64,
    segment: Segment,
    result: std::result::Result<Option<SynthesizedAudio>, anyhow::Error>,
}

struct SynthJob {
    seq: u64,
    epoch: u64,
    segment: Segment,
}

impl Pipeline {
    pub fn new(
        chunk_rx: mpsc::Receiver<Segment>,
        cancel_rx: watch::Receiver<bool>,
        pause_rx: watch::Receiver<bool>,
        engine: Arc<dyn TtsEngine>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            chunk_rx,
            cancel_rx,
            pause_rx,
            engine,
            config,
        }
    }

    pub async fn run(mut self) {
        if self.config.parallel_synth && self.engine.supports_synthesis_queue() {
            self.run_parallel().await;
            return;
        }

        let mut first = true;
        let last_play_done_ts: Arc<StdMutex<Option<tokio::time::Instant>>> =
            Arc::new(StdMutex::new(None));
        let mut second_gen_logged = false;
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();

        loop {
            if *self.cancel_rx.borrow() {
                break;
            }
            let maybe_segment = tokio::select! {
                _ = self.pause_rx.changed() => {
                    if *self.pause_rx.borrow() {
                        let _ = self.engine.stop().await;
                        while self.chunk_rx.try_recv().is_ok() {}
                        play_done_tasks.abort_all();
                        while play_done_tasks.try_join_next().is_some() {}
                        first = true;
                        second_gen_logged = false;
                        if let Ok(mut guard) = last_play_done_ts.lock() {
                            *guard = None;
                        }
                    }
                    None
                }
                _ = self.cancel_rx.changed() => None,
                maybe = self.chunk_rx.recv() => maybe,
            };
            let Some(segment) = maybe_segment else {
                if self.chunk_rx.is_closed() {
                    break;
                }
                continue;
            };

            log_segment_intro(&segment, &mut first, &mut second_gen_logged);

            match self.engine.speak(&segment.text).await {
                Ok(metrics) => {
                    log_playback_metrics(&segment, metrics, &last_play_done_ts, &mut play_done_tasks);
                }
                Err(e) => {
                    warn!("TTS engine failed: {}", e);
                }
            }
        }

        while play_done_tasks.join_next().await.is_some() {}
        let done_ts = last_play_done_ts
            .lock()
            .map(|guard| *guard)
            .unwrap_or(None);
        if let Some(done_ts) = done_ts {
            let now = Instant::now();
            if done_ts > now {
                sleep_until(done_ts).await;
            }
        }
    }

    async fn run_parallel(&mut self) {
        let mut first = true;
        let last_play_done_ts: Arc<StdMutex<Option<tokio::time::Instant>>> =
            Arc::new(StdMutex::new(None));
        let mut second_gen_logged = false;
        let mut play_done_tasks: JoinSet<()> = JoinSet::new();

        let (synth_tx, mut synth_rx) = mpsc::channel::<SynthOutcome>(128);
        let max_inflight = self.config.synth_inflight;
        let mut seq: u64 = 0;
        let mut next_seq: u64 = 0;
        let mut inflight: usize = 0;
        let mut input_closed = false;
        let mut epoch: u64 = 0;
        let mut pending: BTreeMap<u64, SynthOutcome> = BTreeMap::new();
        let mut backlog: VecDeque<SynthJob> = VecDeque::new();

        loop {
            if *self.cancel_rx.borrow() {
                break;
            }

            tokio::select! {
                _ = self.pause_rx.changed() => {
                    if *self.pause_rx.borrow() {
                        let _ = self.engine.stop().await;
                        while self.chunk_rx.try_recv().is_ok() {}
                        play_done_tasks.abort_all();
                        while play_done_tasks.try_join_next().is_some() {}
                        pending.clear();
                        backlog.clear();
                        inflight = 0;
                        epoch = epoch.wrapping_add(1);
                        seq = 0;
                        next_seq = 0;
                        first = true;
                        second_gen_logged = false;
                        if let Ok(mut guard) = last_play_done_ts.lock() {
                            *guard = None;
                        }
                    }
                }
                _ = self.cancel_rx.changed() => {}
                maybe_segment = self.chunk_rx.recv() => {
                    match maybe_segment {
                        Some(segment) => {
                            log_segment_intro(&segment, &mut first, &mut second_gen_logged);
                            let seq_id = seq;
                            seq = seq.wrapping_add(1);
                            let job = SynthJob {
                                seq: seq_id,
                                epoch,
                                segment,
                            };
                            if inflight < max_inflight {
                                inflight += 1;
                                let engine = self.engine.clone();
                                let synth_tx = synth_tx.clone();
                                tokio::spawn(async move {
                                    let result = engine.synthesize(&job.segment.text).await;
                                    let _ = synth_tx
                                        .send(SynthOutcome {
                                            seq: job.seq,
                                            epoch: job.epoch,
                                            segment: job.segment,
                                            result,
                                        })
                                        .await;
                                });
                            } else {
                                backlog.push_back(job);
                            }
                        }
                        None => {
                            input_closed = true;
                        }
                    }
                }
                maybe_result = synth_rx.recv() => {
                    if let Some(outcome) = maybe_result {
                        if outcome.epoch == epoch {
                            if inflight > 0 {
                                inflight -= 1;
                            }
                            pending.insert(outcome.seq, outcome);
                            while inflight < max_inflight {
                                let Some(job) = backlog.pop_front() else {
                                    break;
                                };
                                inflight += 1;
                                let engine = self.engine.clone();
                                let synth_tx = synth_tx.clone();
                                tokio::spawn(async move {
                                    let result = engine.synthesize(&job.segment.text).await;
                                    let _ = synth_tx
                                        .send(SynthOutcome {
                                            seq: job.seq,
                                            epoch: job.epoch,
                                            segment: job.segment,
                                            result,
                                        })
                                        .await;
                                });
                            }
                        }
                    }
                }
            }

            while let Some(outcome) = pending.remove(&next_seq) {
                if outcome.epoch != epoch {
                    continue;
                }
                next_seq = next_seq.wrapping_add(1);
                match outcome.result {
                    Ok(Some(audio)) => match self.engine.play_samples(audio).await {
                        Ok(Some(metrics)) => {
                            log_playback_metrics(
                                &outcome.segment,
                                metrics,
                                &last_play_done_ts,
                                &mut play_done_tasks,
                            );
                        }
                        Ok(None) => {
                            warn!("TTS engine play_samples returned None");
                        }
                        Err(e) => {
                            warn!("TTS engine failed: {}", e);
                        }
                    },
                    Ok(None) => {
                        warn!("TTS engine synthesize returned None");
                    }
                    Err(e) => {
                        warn!("TTS engine failed: {}", e);
                    }
                }
            }

            if input_closed && inflight == 0 && pending.is_empty() {
                break;
            }
        }

        while play_done_tasks.join_next().await.is_some() {}
        let done_ts = last_play_done_ts
            .lock()
            .map(|guard| *guard)
            .unwrap_or(None);
        if let Some(done_ts) = done_ts {
            let now = Instant::now();
            if done_ts > now {
                sleep_until(done_ts).await;
            }
        }
    }
}

fn log_segment_intro(segment: &Segment, first: &mut bool, second_gen_logged: &mut bool) {
    let chunk_chars = segment.text.chars().count();
    if *first {
        info!("=== 指标时间线 ===");

        if let Some(t1) = segment.first_token_ts {
            // LLM首字时延: 从请求到第一个字
            let llm_delay = t1.duration_since(segment.task_start);
            let llm_abs = llm_delay;
            info!("LLM首字时延: {:?} @ {:?}", llm_delay, llm_abs);

            // 分段器延迟: 从第一个字到第一段被分出来
            let chunker_delay = segment.segment_sent_ts.duration_since(t1);
            let chunker_abs = segment.segment_sent_ts.duration_since(segment.task_start);
            info!("分段器延迟: {:?} @ {:?}", chunker_delay, chunker_abs);
        }

        if let Some(t_last) = segment.last_token_ts {
            // 分段器末字延迟: 从最后一个字到分段送出
            let tail_delay = segment.segment_sent_ts.duration_since(t_last);
            let tail_abs = segment.segment_sent_ts.duration_since(segment.task_start);
            info!("分段器末字延迟: {:?} @ {:?}", tail_delay, tail_abs);
        }

        info!("首段长度: {} 字符", chunk_chars);

        *first = false;
    }

    if segment.first_token_ts.is_none() && !*second_gen_logged {
        let now = Instant::now();
        let gen_abs = now.duration_since(segment.task_start);
        let gen_delay = now.duration_since(segment.segment_sent_ts);
        info!("二句生成开始: {:?} @ {:?}", gen_delay, gen_abs);
        *second_gen_logged = true;
    }
}

fn log_playback_metrics(
    segment: &Segment,
    metrics: TtsMetrics,
    last_play_done_ts: &Arc<StdMutex<Option<tokio::time::Instant>>>,
    play_done_tasks: &mut JoinSet<()>,
) {
    let TtsMetrics {
        start_ts,
        first_audio_ts,
        gen_done_ts,
        play_done_ts: _,
        play_done_rx,
    } = metrics;
    let chunk_chars = segment.text.chars().count();
    let first_audio_ts = first_audio_ts.unwrap_or(start_ts);
    let first_audio_abs = first_audio_ts.duration_since(segment.task_start);
    let gen_done_abs = gen_done_ts.duration_since(segment.task_start);
    let is_first_chunk = segment.first_token_ts.is_some();

    // OS TTS 是同步的：合成+播放一体，这里是播放完成的时间
    if is_first_chunk {
        let first_audio_delay = first_audio_ts.duration_since(segment.segment_sent_ts);
        let gen_delay = gen_done_ts.duration_since(segment.segment_sent_ts);
        info!("首播时延: {:?} @ {:?}", first_audio_delay, first_audio_abs);
        info!("首句生成时延: {:?} @ {:?}", gen_delay, gen_done_abs);
    }

    if let Some(play_done_rx) = play_done_rx {
        let segment_sent_ts = segment.segment_sent_ts;
        let task_start = segment.task_start;
        let last_done = last_play_done_ts.clone();
        play_done_tasks.spawn(async move {
            if let Ok(ts) = play_done_rx.await {
                let play_done_abs = ts.saturating_duration_since(task_start);
                let play_delay = ts.saturating_duration_since(segment_sent_ts);
                if is_first_chunk {
                    info!(
                        "首句播放完成(真实): {:?} @ {:?} | {} 字符",
                        play_delay,
                        play_done_abs,
                        chunk_chars
                    );
                } else {
                    info!(
                        "音频块播放完成(真实): {:?} @ {:?} | {} 字符",
                        play_delay,
                        play_done_abs,
                        chunk_chars
                    );
                }
                let mut guard = last_done.lock().expect("playback done lock poisoned");
                if guard.map_or(true, |prev| ts > prev) {
                    *guard = Some(ts);
                }
            }
        });
    }
}
