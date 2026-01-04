use anyhow::{Result, bail};

pub fn pcm_i16_le_bytes_to_vec(pcm: &[u8]) -> Result<Vec<i16>> {
    if pcm.len() % 2 != 0 {
        bail!("pcm bytes length must be even (i16 LE)");
    }
    let mut out = Vec::with_capacity(pcm.len() / 2);
    for chunk in pcm.chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

pub fn pcm_i16_to_mono_f32(pcm: &[i16], channels: u16) -> Result<Vec<f32>> {
    match channels {
        0 => bail!("channels must be >= 1"),
        1 => Ok(pcm.iter().map(|&s| s as f32 / 32768.0).collect()),
        2 => {
            let frames = pcm.len() / 2;
            let mut out = Vec::with_capacity(frames);
            for i in 0..frames {
                let l = pcm[i * 2] as f32 / 32768.0;
                let r = pcm[i * 2 + 1] as f32 / 32768.0;
                out.push((l + r) * 0.5);
            }
            Ok(out)
        }
        _ => bail!("unsupported channels: {channels}"),
    }
}

/// Streaming linear resampler (mono).
///
/// Maintains internal state so it can be fed with arbitrary chunk sizes.
#[derive(Debug, Clone)]
pub struct LinearResampler {
    src_rate: u32,
    dst_rate: u32,
    step: f64,
    pos: f64,
    start: usize,
    buf: Vec<f32>,
}

impl LinearResampler {
    pub fn new(src_rate: u32, dst_rate: u32) -> Result<Self> {
        if src_rate == 0 || dst_rate == 0 {
            bail!("sample_rate must be > 0");
        }
        Ok(Self {
            src_rate,
            dst_rate,
            step: src_rate as f64 / dst_rate as f64,
            pos: 0.0,
            start: 0,
            buf: Vec::new(),
        })
    }

    pub fn src_rate(&self) -> u32 {
        self.src_rate
    }

    pub fn dst_rate(&self) -> u32 {
        self.dst_rate
    }

    pub fn reset(&mut self) {
        self.pos = 0.0;
        self.start = 0;
        self.buf.clear();
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.src_rate == self.dst_rate {
            return samples.to_vec();
        }
        self.buf.extend_from_slice(samples);

        let mut out = Vec::new();
        loop {
            let available = self.buf.len().saturating_sub(self.start);
            if available < 2 {
                break;
            }
            if self.pos + 1.0 >= available as f64 {
                break;
            }
            let i = self.pos.floor() as usize;
            let frac = (self.pos - i as f64) as f32;
            let base = self.start + i;
            let a = self.buf[base];
            let b = self.buf[base + 1];
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }

        let consumed = self.pos.floor() as usize;
        if consumed > 0 {
            self.start = self.start.saturating_add(consumed);
            self.pos -= consumed as f64;
            if self.start >= 4096 {
                self.buf.drain(0..self.start);
                self.start = 0;
            }
        }
        out
    }
}

