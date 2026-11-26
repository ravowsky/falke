use std::{
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, Result};

use crate::{
    camera::CameraPose,
    math::{Quat, Vec3, lerp_arr},
};

/// 4: like 3, but entries flagged in their bone count byte (0x80) store their bones at half
/// precision.
/// 5: like 4, plus each frame's time (f32 seconds since the recording started) after the
/// camera. Versions 3 and 4 still load (their frames are 1/60 s apart).
/// 6: the same frames, compressed in blocks (see `blocks`): about a fifth of the size,
/// nothing lost.
const FLK_REPLAY_VERSION: u8 = 6;
/// The frame rate replays without recorded times are played at.
pub const FRAME_RATE: f32 = 60.0;
/// A frame identical to the one before is not written (the game did not move on, e.g. the
/// same scene drawn twice), unless the last written one is this old (a paused game).
const SAME_FRAME_KEEP_AFTER: f32 = 0.25;
const HALF_FLAG: u8 = 0x80;
const REPLAY_FOLDER: &str = "Replays";

/// Room for the largest skeleton any supported game has (currently THPS4 with 55).
pub const MAX_BONES: usize = 55;

pub struct Replay {
    file_path: PathBuf,
    writer: Option<BlockWriter>,
    frame_count: u64,
    /// When the first frame was written.
    started: Option<Instant>,
    /// The last frame written and its time.
    last: Option<(f32, ReplayState)>,
}

/// One skater's pose for a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SkaterState {
    pub world_pos: [f32; 3],
    pub world_matrix: [f32; 16],
    /// How many leading entries of the bone arrays this skater's skeleton has.
    pub bone_count: u8,
    pub bone_quats: [[f32; 4]; MAX_BONES],
    pub bone_trans: [[f32; 4]; MAX_BONES],
}

impl SkaterState {
    /// `bone_count` as a usable length, clamped to what the arrays can hold.
    pub fn bones(&self) -> usize {
        (self.bone_count as usize).min(MAX_BONES)
    }

    /// Extra entries a game backend adds besides the skater (traffic, people) mark
    /// themselves with a large number (1000 or more) in the otherwise zero fourth column of
    /// the world matrix. Their bones are stored at half precision, which is plenty for
    /// background objects and halves their size; the skater keeps full precision.
    pub fn compact(&self) -> bool {
        self.world_matrix[3] >= 1000.0
    }
}

/// One frame of a replay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayState {
    pub skaters: Vec<SkaterState>,
    pub camera: Option<CameraPose>,
    /// When the frame is played, in seconds from the start of the replay (see
    /// `LoadedReplay::load`). Not used when recording.
    pub time: f32,
}

impl Replay {
    pub fn new(name: &str) -> Self {
        Self {
            file_path: PathBuf::from(REPLAY_FOLDER).join(name.to_owned() + ".flk"),
            writer: None,
            frame_count: 0,
            started: None,
            last: None,
        }
    }

    /// Creates the file, writes the header, and opens the streaming writer.
    pub fn create_file(&mut self) -> Result<()> {
        log::debug!("opening replay file: {:?}", self.file_path);
        let prefix = self.file_path.parent().unwrap();
        std::fs::create_dir_all(prefix).context("failed to create Replays directory")?;

        let file =
            std::fs::File::create(&self.file_path).context("failed to create replay file")?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&[FLK_REPLAY_VERSION])?;
        // Placeholder: frame count will be written in finish().
        writer.write_all(&0u64.to_le_bytes())?;
        self.writer = Some(BlockWriter::start(writer));
        Ok(())
    }

    /// Writes one captured frame to disk, with the time it was captured, and advances the
    /// frame counter. No-ops if the frame holds no skaters (e.g. none loaded yet), or if it is
    /// the same as the frame before (see `SAME_FRAME_KEEP_AFTER`).
    pub fn write_frame(&mut self, state: &ReplayState) -> Result<()> {
        if state.skaters.is_empty() {
            return Ok(());
        }
        let time = self
            .started
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        if let Some((last_time, last)) = &self.last {
            if last.skaters == state.skaters
                && last.camera == state.camera
                && time - last_time < SAME_FRAME_KEEP_AFTER
            {
                return Ok(());
            }
        }
        let writer = self.writer.as_mut().context("replay writer not open")?;
        writer.push(state, time)?;
        self.frame_count += 1;
        self.last = Some((time, state.clone()));
        Ok(())
    }

    /// Flushes the writer and patches the frame count into the file header.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            let mut file = writer.finish()?;
            file.seek(SeekFrom::Start(1))?;
            file.write_all(&self.frame_count.to_le_bytes())?;
            log::info!(
                "replay saved: {} frames → {:?}",
                self.frame_count,
                self.file_path
            );
        }
        Ok(())
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// Packs and writes the blocks of a recording on its own thread, so the game does not wait
/// for the compression.
struct BlockWriter {
    encoder: blocks::Encoder,
    tx: Option<std::sync::mpsc::Sender<Vec<u8>>>,
    thread: Option<std::thread::JoinHandle<Result<std::fs::File>>>,
}

impl BlockWriter {
    fn start(mut out: BufWriter<std::fs::File>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let thread = std::thread::Builder::new()
            .name("falke replay writer".into())
            .spawn(move || -> Result<std::fs::File> {
                for raw in rx {
                    out.write_all(&blocks::pack(&raw)?)?;
                    // Each whole block reaches the disk, so a crash loses at most one.
                    out.flush()?;
                }
                out.into_inner()
                    .map_err(|e| anyhow::anyhow!("flush failed: {}", e.error()))
            })
            .ok();
        Self {
            encoder: blocks::Encoder::default(),
            tx: Some(tx),
            thread,
        }
    }

    fn push(&mut self, state: &ReplayState, time: f32) -> Result<()> {
        self.encoder.push(state, time);
        if self.encoder.frames >= blocks::BLOCK_FRAMES {
            self.send()?;
        }
        Ok(())
    }

    fn send(&mut self) -> Result<()> {
        if self.encoder.frames == 0 {
            return Ok(());
        }
        let raw = self.encoder.take();
        self.tx
            .as_ref()
            .context("replay writer closed")?
            .send(raw)
            .map_err(|_| anyhow::anyhow!("replay writer stopped"))
    }

    fn finish(mut self) -> Result<std::fs::File> {
        self.send()?;
        drop(self.tx.take());
        self.thread
            .take()
            .context("replay writer thread did not start")?
            .join()
            .map_err(|_| anyhow::anyhow!("replay writer thread panicked"))?
    }
}

pub struct LoadedReplay {
    pub name: String,
    pub frame_count: u64,
    pub frames: Vec<ReplayState>,
}

impl LoadedReplay {
    pub fn load(name: &str) -> Result<Self> {
        let file_path = PathBuf::from(REPLAY_FOLDER).join(name.to_owned() + ".flk");
        // Buffered: a frame is read in hundreds of small pieces.
        let mut file = BufReader::with_capacity(
            1 << 20,
            std::fs::File::open(&file_path)
                .with_context(|| format!("failed to open replay file: {:?}", file_path))?,
        );

        let mut version = [0u8; 1];
        file.read_exact(&mut version)
            .context("failed to read replay version")?;
        let version = version[0];
        if !(3..=FLK_REPLAY_VERSION).contains(&version) {
            anyhow::bail!("unsupported replay version: {version}");
        }

        let mut frame_count_bytes = [0u8; 8];
        file.read_exact(&mut frame_count_bytes)
            .context("failed to read frame count")?;
        let frame_count = u64::from_le_bytes(frame_count_bytes);

        let mut frames = Vec::with_capacity(frame_count as usize);
        if version >= 6 {
            frames = blocks::read_all(&mut file)?;
            if frames.len() as u64 != frame_count {
                log::warn!(
                    "replay header says {frame_count} frames, the file holds {} (recording cut short?)",
                    frames.len()
                );
            }
        }

        for _ in 0..if version >= 6 { 0 } else { frame_count } {
            let mut skater_count = [0u8; 1];
            file.read_exact(&mut skater_count)?;
            let skater_count = skater_count[0] as usize;

            let mut skaters = Vec::with_capacity(skater_count);
            for _ in 0..skater_count {
                let mut bone_count = [0u8; 1];
                file.read_exact(&mut bone_count)?;
                let half = version >= 4 && bone_count[0] & HALF_FLAG != 0;
                let bone_count = if version >= 4 {
                    bone_count[0] & !HALF_FLAG
                } else {
                    bone_count[0]
                };
                if bone_count as usize > MAX_BONES {
                    anyhow::bail!(
                        "replay wants {bone_count} bones per skater, more than the {MAX_BONES} a frame holds"
                    );
                }

                let world_pos = read_f32s::<3>(&mut file)?;
                let world_matrix = read_f32s::<16>(&mut file)?;

                let mut bone_quats = [[0f32; 4]; MAX_BONES];
                for quat in bone_quats[..bone_count as usize].iter_mut() {
                    if half {
                        // Back to unit length after the rounding.
                        let q = read_f16s::<4>(&mut file)?;
                        let len = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                        *quat = if len > 0.0 {
                            q.map(|x| x / len)
                        } else {
                            [0.0, 0.0, 0.0, 1.0]
                        };
                    } else {
                        *quat = read_f32s::<4>(&mut file)?;
                    }
                }
                let mut bone_trans = [[0f32; 4]; MAX_BONES];
                for trans in bone_trans[..bone_count as usize].iter_mut() {
                    *trans = if half {
                        read_f16s::<4>(&mut file)?
                    } else {
                        read_f32s::<4>(&mut file)?
                    };
                }

                skaters.push(SkaterState {
                    world_pos,
                    world_matrix,
                    bone_count,
                    bone_quats,
                    bone_trans,
                });
            }

            // Camera is written after all skaters, behind a presence byte.
            let mut has_camera = [0u8; 1];
            file.read_exact(&mut has_camera)?;
            let camera = if has_camera[0] != 0 {
                Some(CameraPose {
                    position: Vec3::from_array(read_f32s(&mut file)?),
                    orientation: Quat::from_array(read_f32s(&mut file)?),
                    fov: read_f32s::<1>(&mut file)?[0],
                })
            } else {
                None
            };
            let time = if version >= 5 {
                read_f32s::<1>(&mut file)?[0]
            } else {
                frames.len() as f32 / FRAME_RATE
            };

            frames.push(ReplayState {
                skaters,
                camera,
                time,
            });
        }
        if version >= 5 {
            let recorded: Vec<f32> = frames.iter().map(|f| f.time).collect();
            let times = playback_times(&recorded, &frames_positions(&frames));
            for (f, t) in frames.iter_mut().zip(times) {
                f.time = t;
            }
        }

        Ok(Self {
            name: name.to_owned(),
            frame_count: frames.len() as u64,
            frames,
        })
    }

    pub fn get_frame(&self, index: usize) -> Option<&ReplayState> {
        self.frames.get(index)
    }

    /// How long the replay plays, in seconds.
    pub fn duration(&self) -> f32 {
        self.frames.last().map(|f| f.time).unwrap_or(0.0)
    }

    /// The frame played at `t` seconds (the last one at or before it).
    pub fn frame_at(&self, t: f32) -> usize {
        self.locate(t).map(|(i0, _, _)| i0).unwrap_or(0)
    }

    /// Like `sample`, but lets the game backend build the skaters itself from the frames
    /// around `t` (`blend(frames, i0, frac)`; `None` = generic interpolation).
    pub fn sample_with(
        &self,
        t: f32,
        blend: impl Fn(&[ReplayState], usize, f32) -> Option<Vec<SkaterState>>,
    ) -> Option<ReplayState> {
        let (i0, i1, frac) = self.locate(t)?;
        if i0 != i1 {
            if let Some(skaters) = blend(&self.frames, i0, frac) {
                let camera = lerp_camera(&self.frames[i0], &self.frames[i1], frac);
                return Some(ReplayState {
                    skaters,
                    camera,
                    time: t,
                });
            }
        }
        self.sample(t)
    }

    /// The two frames around `t` (seconds) and how far between them it is.
    fn locate(&self, t: f32) -> Option<(usize, usize, f32)> {
        let max = self.frames.len().checked_sub(1)?;
        // The first frame played after `t`.
        let after = self.frames.partition_point(|f| f.time <= t);
        if after == 0 {
            return Some((0, 1.min(max), 0.0));
        }
        if after > max {
            return Some((max, max, 0.0));
        }
        let (i0, i1) = (after - 1, after);
        let span = self.frames[i1].time - self.frames[i0].time;
        let frac = if span > 0.0 {
            ((t - self.frames[i0].time) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some((i0, i1, frac))
    }

    /// Sample the replay at `t` seconds, interpolating between the two surrounding frames.
    pub fn sample(&self, t: f32) -> Option<ReplayState> {
        let (i0, i1, frac) = self.locate(t)?;
        if i0 == i1 || frac == 0.0 {
            return Some(self.frames[i0].clone());
        }

        let a = &self.frames[i0];
        let b = &self.frames[i1];

        let camera = lerp_camera(a, b, frac);

        // A jump between the two frames must not be blended: a game can flip the skater 180
        // degrees in one frame (THAW does when landing switch/fakie), and animations can cut
        // from one pose to another. Halfway between, there is no valid pose and the bones
        // swing through the body, so the nearer frame is shown instead. Decided for the whole
        // frame, because one skater can be stored as several entries that belong together.
        // (Backends with their own blending, see `sample_with`, handle this better.)
        let jump =
            a.skaters.iter().zip(b.skaters.iter()).any(|(sa, sb)| {
                rotation_jump(&sa.world_matrix, &sb.world_matrix) || pose_jump(sa, sb)
            });
        if jump {
            let nearest = if frac < 0.5 { a } else { b };
            return Some(ReplayState {
                skaters: nearest.skaters.clone(),
                camera,
                time: t,
            });
        }

        // Skaters pair up by position; a frame with fewer is the limit.
        let skaters = a
            .skaters
            .iter()
            .zip(b.skaters.iter())
            .map(|(sa, sb)| {
                let bones = sa.bones().min(sb.bones());
                let mut bone_quats = [[0f32; 4]; MAX_BONES];
                let mut bone_trans = [[0f32; 4]; MAX_BONES];
                for i in 0..bones {
                    bone_quats[i] = slerp_quat(sa.bone_quats[i], sb.bone_quats[i], frac);
                    bone_trans[i] = lerp_arr(sa.bone_trans[i], sb.bone_trans[i], frac);
                }
                SkaterState {
                    world_pos: lerp_arr(sa.world_pos, sb.world_pos, frac),
                    world_matrix: lerp_arr(sa.world_matrix, sb.world_matrix, frac),
                    bone_count: bones as u8,
                    bone_quats,
                    bone_trans,
                }
            })
            .collect();

        Some(ReplayState {
            skaters,
            camera,
            time: t,
        })
    }
}

/// Where the first skater is in each frame (`None` without one).
fn frames_positions(frames: &[ReplayState]) -> Vec<Option<[f32; 3]>> {
    frames
        .iter()
        .map(|f| f.skaters.first().map(|s| s.world_pos))
        .collect()
}

/// When each frame is played, from the times they were captured (`recorded`).
///
/// A frame is captured when the game draws it, but how far the game moved on in it depends on
/// the time it measured for that frame: usually the frame before's, sometimes its own. A frame
/// that took long (a hitch) moves the skater further, and played at an even pace that shows as
/// a lurch. So the frames are spaced by the recorded times, whichever way round fits the
/// skater's motion best (the steadiest speed), or evenly if neither helps.
fn playback_times(recorded: &[f32], positions: &[Option<[f32; 3]>]) -> Vec<f32> {
    let n = recorded.len();
    if n < 3 {
        return (0..n).map(|i| i as f32 / FRAME_RATE).collect();
    }
    // The gaps between frames as captured (never zero or backwards).
    let gaps: Vec<f32> = recorded
        .windows(2)
        .map(|w| (w[1] - w[0]).max(1e-4))
        .collect();
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    // Gap `i` (frame i to i + 1) under each way of reading the times.
    let even = |_: usize| mean;
    let own = |i: usize| gaps[i];
    let before = |i: usize| if i == 0 { gaps[0] } else { gaps[i - 1] };
    // How unsteady the skater's speed is when frames are spaced by `gap`.
    let roughness = |gap: &dyn Fn(usize) -> f32| -> f32 {
        let step = |i: usize| -> Option<f32> {
            let (a, b) = (positions.get(i)?.as_ref()?, positions.get(i + 1)?.as_ref()?);
            let d: f32 = (0..3).map(|c| (b[c] - a[c]).powi(2)).sum::<f32>().sqrt();
            Some(d)
        };
        let mut sum = 0.0;
        for i in 1..gaps.len() {
            let (Some(s0), Some(s1)) = (step(i - 1), step(i)) else {
                continue;
            };
            // Only steady motion says anything: not standing still, not a snap (the game
            // putting the skater onto a wall, a respawn).
            if s0 < 2.0 || s1 < 2.0 || (s1 / s0).ln().abs() > 3f32.ln() {
                continue;
            }
            sum += ((s1 / gap(i)) / (s0 / gap(i - 1))).ln().abs();
        }
        sum
    };
    let (r_even, r_own, r_before) = (roughness(&even), roughness(&own), roughness(&before));
    // Even spacing unless a reading of the times is clearly steadier.
    let gap: &dyn Fn(usize) -> f32 = if r_own.min(r_before) < 0.97 * r_even {
        if r_own <= r_before { &own } else { &before }
    } else {
        &even
    };
    log::debug!(
        "replay timing: unsteadiness {r_even:.1} evenly, {r_own:.1} by own times, {r_before:.1} by the previous frame's"
    );
    let mut times = Vec::with_capacity(n);
    let mut t = 0.0;
    times.push(t);
    for i in 0..gaps.len() {
        t += gap(i);
        times.push(t);
    }
    times
}

/// The camera between two frames (the earlier one's if either has none).
fn lerp_camera(a: &ReplayState, b: &ReplayState, frac: f32) -> Option<CameraPose> {
    match (&a.camera, &b.camera) {
        (Some(ca), Some(cb)) => Some(CameraPose::lerp(ca, cb, frac)),
        (camera, _) => *camera,
    }
}

/// Whether the rotation part of two world matrices differs by more than about 45 degrees.
fn rotation_jump(a: &[f32; 16], b: &[f32; 16]) -> bool {
    let mut trace = 0.0;
    for r in 0..3 {
        let ra = [a[r * 4], a[r * 4 + 1], a[r * 4 + 2]];
        let rb = [b[r * 4], b[r * 4 + 1], b[r * 4 + 2]];
        let la = (ra[0] * ra[0] + ra[1] * ra[1] + ra[2] * ra[2]).sqrt();
        let lb = (rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2]).sqrt();
        if la < 1e-4 || lb < 1e-4 {
            return false;
        }
        trace += (ra[0] * rb[0] + ra[1] * rb[1] + ra[2] * rb[2]) / (la * lb);
    }
    // cos(angle) = (trace - 1) / 2; cos 45 degrees is about 0.707.
    (trace - 1.0) / 2.0 < 0.707
}

/// Whether any bone turns more than 60 degrees between two frames. Normal motion stays
/// under about 30 degrees per frame; pose cuts in THAW reach 65 to 180.
fn pose_jump(a: &SkaterState, b: &SkaterState) -> bool {
    let bones = a.bones().min(b.bones());
    (0..bones).any(|i| {
        let (p, q) = (a.bone_quats[i], b.bone_quats[i]);
        let lp = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]).sqrt();
        let lq = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if lp < 1e-6 || lq < 1e-6 {
            return false;
        }
        let dot = ((p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]) / (lp * lq)).abs();
        // The angle between two rotations is 2 * acos(|dot|): 60 degrees is |dot| = cos 30.
        dot < 0.866
    })
}

fn slerp_quat(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    // quat_slerp already takes the shortest path.
    let r = nalgebra_glm::quat_slerp(&Quat::from_array(a).into(), &Quat::from_array(b).into(), t);
    Quat::from(r).to_array()
}

fn read_f16s<const N: usize>(r: &mut impl Read) -> Result<[f32; N]> {
    let mut out = [0f32; N];
    for v in out.iter_mut() {
        let mut buf = [0u8; 2];
        r.read_exact(&mut buf)?;
        *v = f16_to_f32(u16::from_le_bytes(buf));
    }
    Ok(out)
}

/// IEEE half precision, rounded to nearest; out-of-range values are clamped to the largest
/// half (65504), NaN stays NaN.
fn f32_to_f16(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let v = v.clamp(-65504.0, 65504.0);
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if exp <= 0 {
        // Subnormal (or zero).
        if exp < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = (rest > mid || (rest == mid && half & 1 == 1)) as u32;
        return sign | (half + round) as u16;
    }
    let half = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let round = (rest > 0x1000 || (rest == 0x1000 && half & 1 == 1)) as u32;
    // A carry into the exponent is still the right (next) value.
    sign | (half + round).min(0x7bff) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn read_f32s<const N: usize>(r: &mut impl Read) -> Result<[f32; N]> {
    let mut out = [0f32; N];
    for v in out.iter_mut() {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        *v = f32::from_le_bytes(buf);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_round_trip_exactly() {
        let mut frames = Vec::new();
        for i in 0..150 {
            let x = i as f32 * 0.37;
            let mut skater = SkaterState {
                world_pos: [x, 1.5, -x * 2.0],
                world_matrix: std::array::from_fn(|k| (k as f32 + x).sin()),
                bone_count: 3,
                bone_quats: [[0.0, 0.0, 0.0, 1.0]; MAX_BONES],
                bone_trans: [[0f32; 4]; MAX_BONES],
            };
            for b in 0..3 {
                skater.bone_quats[b] = [x.sin() * 0.5, 0.1, b as f32 * 0.2, 0.8];
                skater.bone_trans[b] = [x, b as f32, -x, 0.0];
            }
            // An entry with only identity rotations, and one more on some frames.
            let mut part = skater.clone();
            part.bone_quats = [[0.0, 0.0, 0.0, 1.0]; MAX_BONES];
            let mut skaters = vec![skater, part];
            if i % 7 < 3 {
                let mut car = skaters[0].clone();
                car.world_matrix[3] = 4250.0;
                skaters.push(car);
            }
            frames.push(ReplayState {
                skaters,
                camera: (i % 11 != 0).then(|| CameraPose {
                    position: Vec3::from_array([x, 2.0, 3.0]),
                    orientation: Quat::from_array([0.0, 0.0, 0.0, 1.0]),
                    fov: 72.0,
                }),
                time: x,
            });
        }
        let mut file = Vec::new();
        let mut enc = blocks::Encoder::default();
        for f in &frames {
            enc.push(f, f.time);
            if enc.frames >= blocks::BLOCK_FRAMES {
                file.extend(blocks::pack(&enc.take()).unwrap());
            }
        }
        file.extend(blocks::pack(&enc.take()).unwrap());
        let back = blocks::read_all(&mut &file[..]).unwrap();
        assert_eq!(back.len(), frames.len());
        for (a, b) in frames.iter().zip(&back) {
            assert_eq!(a.time.to_bits(), b.time.to_bits());
            assert_eq!(a.camera, b.camera);
            assert_eq!(a.skaters.len(), b.skaters.len());
            for (x, y) in a.skaters.iter().zip(&b.skaters) {
                assert_eq!(x.world_pos, y.world_pos);
                assert_eq!(x.world_matrix, y.world_matrix);
                assert_eq!(x.bone_count, y.bone_count);
                // Traffic is kept at half precision (as before); the rest exactly.
                if !x.compact() {
                    for k in 0..x.bones() {
                        assert_eq!(x.bone_trans[k], y.bone_trans[k]);
                        assert_eq!(x.bone_quats[k], y.bone_quats[k]);
                    }
                }
            }
        }
    }

    #[test]
    fn half_precision_round_trip() {
        for v in [
            0.0f32, 1.0, -1.0, 0.5, 0.70710677, 3.14159, 72.37, -118.6, 1e-5, 60000.0, 1e9, -1e9,
        ] {
            let back = f16_to_f32(f32_to_f16(v));
            let expected = v.clamp(-65504.0, 65504.0);
            let tolerance = expected.abs() / 1024.0 + 1e-7;
            assert!((back - expected).abs() <= tolerance, "{v} -> {back}");
        }
        // Exhaustive: every finite half survives a round trip unchanged.
        for h in 0u16..=0xffff {
            let v = f16_to_f32(h);
            if v.is_finite() {
                assert_eq!(f32_to_f16(v), h, "half {h:#06x}");
            }
        }
    }

    fn frame_at(time: f32) -> ReplayState {
        ReplayState {
            time,
            ..Default::default()
        }
    }

    #[test]
    fn locate_by_time() {
        let replay = LoadedReplay {
            name: "t".into(),
            frame_count: 3,
            frames: vec![frame_at(0.0), frame_at(0.02), frame_at(0.05)],
        };
        assert_eq!(replay.locate(-1.0), Some((0, 1, 0.0)));
        assert_eq!(replay.locate(0.01), Some((0, 1, 0.5)));
        assert_eq!(replay.locate(0.035), Some((1, 2, 0.5)));
        assert_eq!(replay.locate(9.0), Some((2, 2, 0.0)));
        assert_eq!(replay.frame_at(0.03), 1);
        assert!((replay.duration() - 0.05).abs() < 1e-6);
    }

    /// A skater moving at a steady speed, captured with one slow frame: the times should
    /// space the frames so that the speed stays steady.
    #[test]
    fn hitch_is_spaced_out() {
        let speed = 900.0; // units a second
        // Captured every 1/60 s except one frame that took 1/30 s. The game moves the skater by
        // the time the frame before took.
        let mut recorded = vec![0.0f32];
        let mut dt_used = vec![];
        for i in 1..40 {
            let dt = if i == 20 { 1.0 / 30.0 } else { 1.0 / 60.0 };
            recorded.push(recorded[i - 1] + dt);
            dt_used.push(dt);
        }
        let mut positions = vec![Some([0.0f32, 0.0, 0.0])];
        for i in 1..40 {
            // The step from frame i - 1 to i used the time frame i - 1 took.
            let dt = if i >= 2 { dt_used[i - 2] } else { 1.0 / 60.0 };
            let x = positions[i - 1].unwrap()[0] + speed * dt;
            positions.push(Some([x, 0.0, 0.0]));
        }
        let times = playback_times(&recorded, &positions);
        for i in 1..39 {
            let dx = positions[i + 1].unwrap()[0] - positions[i].unwrap()[0];
            let v = dx / (times[i + 1] - times[i]);
            assert!((v - speed).abs() < 1.0, "frame {i}: speed {v}");
        }
    }

    #[test]
    fn steady_frames_stay_even() {
        let recorded: Vec<f32> = (0..30)
            .map(|i| i as f32 / 60.0 + if i % 2 == 0 { 0.001 } else { 0.0 })
            .collect();
        let positions: Vec<Option<[f32; 3]>> =
            (0..30).map(|i| Some([i as f32 * 10.0, 0.0, 0.0])).collect();
        let times = playback_times(&recorded, &positions);
        let gaps: Vec<f32> = times.windows(2).map(|w| w[1] - w[0]).collect();
        let (lo, hi) = gaps
            .iter()
            .fold((f32::MAX, 0f32), |(a, b), g| (a.min(*g), b.max(*g)));
        assert!(hi - lo < 1e-5, "{gaps:?}");
    }

    #[test]
    fn new_replay() {
        let replay = Replay::new("alcatraz");
        assert_eq!(replay.frame_count(), 0);
    }

    #[test]
    fn write_frame_without_skaters_is_a_noop() {
        let mut replay = Replay::new("alcatraz");
        // No writer open: an empty frame must bail out before touching it.
        replay.write_frame(&ReplayState::default()).unwrap();
        assert_eq!(replay.frame_count(), 0);
    }
}

// Version 6: frames in compressed blocks.
//
// A frame is split into its layout (counts and flags: the `meta` bytes) and its numbers,
// which are kept as raw bit patterns (lossless). Every number is stored XORed with the same
// number of the frame before (the same entry, if that entry has the same layout), so that a
// value that barely moves leaves mostly zero bits; the numbers are then laid out byte plane
// by byte plane (every first byte, then every second byte...) so those zeros line up, and
// the block is deflated. A block holds up to BLOCK_FRAMES frames and starts afresh (no
// reference to the block before), so a file cut short (the game crashed while recording)
// still loads up to its last whole block. Bone rotations that are all identity (the
// skater's extra pose entries) are not stored at all (NO_QUATS_FLAG).
mod blocks {
    use super::*;
    use anyhow::Context as _;

    pub(super) const NO_QUATS_FLAG: u8 = 0x40;
    pub(super) const BONES_MASK: u8 = 0x3f;
    /// Frames in one compressed block (about a second).
    pub(super) const BLOCK_FRAMES: usize = 60;
    const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    /// The numbers of one entry, as bit patterns.
    #[derive(Default, Clone)]
    struct Words {
        w32: Vec<u32>,
        w16: Vec<u16>,
    }

    /// What the numbers of a frame are predicted from: the two frames before.
    #[derive(Default)]
    struct Context {
        entries: Vec<(u8, Words)>,
        older: Vec<(u8, Words)>,
        camera: Option<[u32; 8]>,
        older_camera: Option<[u32; 8]>,
        time: u32,
        older_time: Option<u32>,
    }

    /// The prediction for a number: carried on from the two frames before (`2 * a - b`, as
    /// floats; the same sum on both sides, so the decoder gets the very same bits), or the
    /// frame before if only that one is known.
    fn predict(a: u32, b: Option<u32>) -> u32 {
        match b {
            Some(b) => {
                let p = 2.0 * f32::from_bits(a) - f32::from_bits(b);
                if p.is_finite() { p.to_bits() } else { a }
            }
            None => a,
        }
    }

    /// The entry `i` of the frame before (and the one before that), if it has this layout.
    fn reference<'a>(
        ctx: &'a Context,
        i: usize,
        header: u8,
        len32: usize,
        len16: usize,
    ) -> Option<(&'a Words, Option<&'a Words>)> {
        let fits = |e: &&(u8, Words)| e.0 == header && e.1.w32.len() == len32 && e.1.w16.len() == len16;
        let prev = ctx.entries.get(i).filter(fits)?;
        Some((&prev.1, ctx.older.get(i).filter(fits).map(|e| &e.1)))
    }

    fn header_of(s: &SkaterState) -> u8 {
        let bones = s.bones() as u8;
        let mut h = bones & BONES_MASK;
        if s.compact() {
            h |= HALF_FLAG;
        }
        if s.bone_quats[..s.bones()].iter().all(|q| *q == IDENTITY) {
            h |= NO_QUATS_FLAG;
        }
        h
    }

    fn words_of(s: &SkaterState, header: u8) -> Words {
        let half = header & HALF_FLAG != 0;
        let mut w = Words::default();
        w.w32.extend(s.world_pos.iter().map(|v| v.to_bits()));
        w.w32.extend(s.world_matrix.iter().map(|v| v.to_bits()));
        let n = s.bones();
        let mut put = |v: &[f32]| {
            if half {
                w.w16.extend(v.iter().map(|x| f32_to_f16(*x)));
            } else {
                w.w32.extend(v.iter().map(|x| x.to_bits()));
            }
        };
        if header & NO_QUATS_FLAG == 0 {
            for q in &s.bone_quats[..n] {
                put(q);
            }
        }
        for t in &s.bone_trans[..n] {
            put(t);
        }
        w
    }

    fn camera_prediction(ctx: &Context) -> [u32; 8] {
        match ctx.camera {
            Some(c) => std::array::from_fn(|k| predict(c[k], ctx.older_camera.map(|o| o[k]))),
            None => [0; 8],
        }
    }

    /// Collects frames and turns them into blocks.
    #[derive(Default)]
    pub(super) struct Encoder {
        meta: Vec<u8>,
        w32: Vec<u32>,
        w16: Vec<u16>,
        pub(super) frames: usize,
        ctx: Context,
    }

    impl Encoder {
        pub(super) fn push(&mut self, state: &ReplayState, time: f32) {
            self.meta.push(state.skaters.len() as u8);
            let mut entries = Vec::with_capacity(state.skaters.len());
            for (i, s) in state.skaters.iter().enumerate() {
                let header = header_of(s);
                self.meta.push(header);
                let words = words_of(s, header);
                match reference(&self.ctx, i, header, words.w32.len(), words.w16.len()) {
                    Some((p, older)) => {
                        self.w32.extend(words.w32.iter().enumerate().map(|(k, a)| {
                            a ^ predict(p.w32[k], older.map(|o| o.w32[k]))
                        }));
                        self.w16.extend(words.w16.iter().zip(&p.w16).map(|(a, b)| a ^ b));
                    }
                    None => {
                        self.w32.extend(&words.w32);
                        self.w16.extend(&words.w16);
                    }
                }
                entries.push((header, words));
            }
            self.ctx.older = std::mem::replace(&mut self.ctx.entries, entries);
            match &state.camera {
                Some(cam) => {
                    self.meta.push(1);
                    let mut c = [0u32; 8];
                    for (k, v) in cam
                        .position
                        .to_array()
                        .iter()
                        .chain(cam.orientation.to_array().iter())
                        .chain(std::iter::once(&cam.fov))
                        .enumerate()
                    {
                        c[k] = v.to_bits();
                    }
                    let pred = camera_prediction(&self.ctx);
                    self.w32.extend(c.iter().zip(pred).map(|(a, b)| a ^ b));
                    self.ctx.older_camera = self.ctx.camera.replace(c);
                }
                None => {
                    self.meta.push(0);
                    self.ctx.older_camera = self.ctx.camera.take();
                }
            }
            let t = time.to_bits();
            self.w32.push(t ^ predict(self.ctx.time, self.ctx.older_time));
            self.ctx.older_time = Some(self.ctx.time);
            self.ctx.time = t;
            self.frames += 1;
        }

        /// The block so far, uncompressed, and a fresh start.
        pub(super) fn take(&mut self) -> Vec<u8> {
            let e = std::mem::take(self);
            let mut out = Vec::with_capacity(16 + e.meta.len() + e.w32.len() * 4 + e.w16.len() * 2);
            for n in [e.frames, e.meta.len(), e.w32.len(), e.w16.len()] {
                out.extend_from_slice(&(n as u32).to_le_bytes());
            }
            out.extend_from_slice(&e.meta);
            for plane in 0..4 {
                out.extend(e.w32.iter().map(|w| (w >> (plane * 8)) as u8));
            }
            for plane in 0..2 {
                out.extend(e.w16.iter().map(|w| (w >> (plane * 8)) as u8));
            }
            out
        }
    }

    /// Compresses a block: its raw length, compressed length, then the deflated bytes.
    pub(super) fn pack(raw: &[u8]) -> Result<Vec<u8>> {
        use flate2::{Compression, write::DeflateEncoder};
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(6));
        enc.write_all(raw)?;
        let data = enc.finish()?;
        let mut out = Vec::with_capacity(8 + data.len());
        out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        Ok(out)
    }

    /// Reads blocks until the end of the file (or a block cut short).
    pub(super) fn read_all(file: &mut impl Read) -> Result<Vec<ReplayState>> {
        let mut frames = Vec::new();
        loop {
            let mut head = [0u8; 8];
            match file.read_exact(&mut head) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let raw_len = u32::from_le_bytes(head[..4].try_into().unwrap()) as usize;
            let comp_len = u32::from_le_bytes(head[4..].try_into().unwrap()) as usize;
            let mut data = vec![0u8; comp_len];
            if file.read_exact(&mut data).is_err() {
                log::warn!("replay ends in the middle of a block (recording cut short?)");
                break;
            }
            let mut raw = Vec::with_capacity(raw_len);
            flate2::read::DeflateDecoder::new(&data[..])
                .read_to_end(&mut raw)
                .context("damaged replay block")?;
            decode(&raw, &mut frames).context("damaged replay block")?;
        }
        Ok(frames)
    }

    fn decode(raw: &[u8], frames: &mut Vec<ReplayState>) -> Result<()> {
        let u32_at = |i: usize| -> Result<usize> {
            Ok(u32::from_le_bytes(raw.get(i..i + 4).context("short block")?.try_into()?) as usize)
        };
        let (count, meta_len, n32, n16) = (u32_at(0)?, u32_at(4)?, u32_at(8)?, u32_at(12)?);
        let meta = raw.get(16..16 + meta_len).context("short block")?;
        let p32 = 16 + meta_len;
        let p16 = p32 + n32 * 4;
        if raw.len() < p16 + n16 * 2 {
            anyhow::bail!("short block");
        }
        let w32 = |k: usize| -> u32 {
            (0..4).fold(0u32, |acc, plane| acc | (raw[p32 + plane * n32 + k] as u32) << (plane * 8))
        };
        let w16 = |k: usize| -> u16 {
            (0..2).fold(0u16, |acc, plane| acc | (raw[p16 + plane * n16 + k] as u16) << (plane * 8))
        };
        let (mut m, mut i32_, mut i16_) = (0usize, 0usize, 0usize);
        let mut next_meta = || -> Result<u8> {
            let v = *meta.get(m).context("short block")?;
            m += 1;
            Ok(v)
        };
        let mut ctx = Context::default();
        for _ in 0..count {
            let skater_count = next_meta()? as usize;
            let mut skaters = Vec::with_capacity(skater_count);
            let mut entries = Vec::with_capacity(skater_count);
            for e in 0..skater_count {
                let header = next_meta()?;
                let bones = (header & BONES_MASK) as usize;
                if bones > MAX_BONES {
                    anyhow::bail!("replay wants {bones} bones per skater, more than {MAX_BONES}");
                }
                let half = header & HALF_FLAG != 0;
                let quats = header & NO_QUATS_FLAG == 0;
                let per_bone = if quats { 8 } else { 4 };
                let (len32, len16) = if half {
                    (19, bones * per_bone)
                } else {
                    (19 + bones * per_bone, 0)
                };
                if i32_ + len32 > n32 || i16_ + len16 > n16 {
                    anyhow::bail!("short block");
                }
                let mut words = Words {
                    w32: (0..len32).map(|k| w32(i32_ + k)).collect(),
                    w16: (0..len16).map(|k| w16(i16_ + k)).collect(),
                };
                i32_ += len32;
                i16_ += len16;
                if let Some((p, older)) = reference(&ctx, e, header, len32, len16) {
                    for (k, a) in words.w32.iter_mut().enumerate() {
                        *a ^= predict(p.w32[k], older.map(|o| o.w32[k]));
                    }
                    words.w16.iter_mut().zip(&p.w16).for_each(|(a, b)| *a ^= b);
                }
                let f = |k: usize| f32::from_bits(words.w32[k]);
                let mut s = SkaterState {
                    world_pos: std::array::from_fn(|k| f(k)),
                    world_matrix: std::array::from_fn(|k| f(3 + k)),
                    bone_count: bones as u8,
                    bone_quats: [IDENTITY; MAX_BONES],
                    bone_trans: [[0f32; 4]; MAX_BONES],
                };
                let value = |k: usize| -> f32 {
                    if half {
                        f16_to_f32(words.w16[k])
                    } else {
                        f(19 + k)
                    }
                };
                let mut k = 0;
                if quats {
                    for q in s.bone_quats[..bones].iter_mut() {
                        let v: [f32; 4] = std::array::from_fn(|c| value(k + c));
                        k += 4;
                        *q = if half {
                            let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if len > 0.0 { v.map(|x| x / len) } else { IDENTITY }
                        } else {
                            v
                        };
                    }
                }
                for t in s.bone_trans[..bones].iter_mut() {
                    *t = std::array::from_fn(|c| value(k + c));
                    k += 4;
                }
                skaters.push(s);
                entries.push((header, words));
            }
            ctx.older = std::mem::replace(&mut ctx.entries, entries);
            let camera = if next_meta()? != 0 {
                if i32_ + 8 > n32 {
                    anyhow::bail!("short block");
                }
                let pred = camera_prediction(&ctx);
                let c: [u32; 8] = std::array::from_fn(|k| w32(i32_ + k) ^ pred[k]);
                i32_ += 8;
                ctx.older_camera = ctx.camera.replace(c);
                let f = |k: usize| f32::from_bits(c[k]);
                Some(CameraPose {
                    position: Vec3::from_array([f(0), f(1), f(2)]),
                    orientation: Quat::from_array([f(3), f(4), f(5), f(6)]),
                    fov: f(7),
                })
            } else {
                ctx.older_camera = ctx.camera.take();
                None
            };
            if i32_ >= n32 {
                anyhow::bail!("short block");
            }
            let t = w32(i32_) ^ predict(ctx.time, ctx.older_time);
            i32_ += 1;
            ctx.older_time = Some(ctx.time);
            ctx.time = t;
            frames.push(ReplayState {
                skaters,
                camera,
                time: f32::from_bits(t),
            });
        }
        Ok(())
    }
}
