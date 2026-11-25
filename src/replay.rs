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
const FLK_REPLAY_VERSION: u8 = 5;
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
    writer: Option<BufWriter<std::fs::File>>,
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
        self.writer = Some(writer);
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

        writer.write_all(&[state.skaters.len() as u8])?;
        for skater in &state.skaters {
            // Only the bones this skeleton has: the arrays are sized for the
            // largest game, not for the one being recorded.
            let bones = skater.bones();
            let half = skater.compact();
            writer.write_all(&[bones as u8 | if half { HALF_FLAG } else { 0 }])?;
            write_f32s(writer, &skater.world_pos)?;
            write_f32s(writer, &skater.world_matrix)?;
            let write = |w: &mut BufWriter<std::fs::File>, v: &[f32]| {
                if half {
                    write_f16s(w, v)
                } else {
                    write_f32s(w, v)
                }
            };
            for quat in &skater.bone_quats[..bones] {
                write(writer, quat)?;
            }
            for trans in &skater.bone_trans[..bones] {
                write(writer, trans)?;
            }
        }
        // Camera: a presence byte, then the pose. A frame recorded before the
        // camera was found genuinely has none, so it is not worth a pose of
        // zeros that reads back as a valid (and wrong) orientation.
        match &state.camera {
            Some(cam) => {
                writer.write_all(&[1])?;
                write_f32s(writer, &cam.position.to_array())?;
                write_f32s(writer, &cam.orientation.to_array())?;
                write_f32s(writer, &[cam.fov])?;
            }
            None => writer.write_all(&[0])?,
        }
        write_f32s(writer, &[time])?;
        self.frame_count += 1;
        self.last = Some((time, state.clone()));
        Ok(())
    }

    /// Flushes the writer and patches the frame count into the file header.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            let mut file = writer
                .into_inner()
                .map_err(|e| anyhow::anyhow!("flush failed: {}", e.error()))?;
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

        for _ in 0..frame_count {
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
            frame_count,
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

fn write_f32s(w: &mut impl Write, values: &[f32]) -> Result<()> {
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_f16s(w: &mut impl Write, values: &[f32]) -> Result<()> {
    for v in values {
        w.write_all(&f32_to_f16(*v).to_le_bytes())?;
    }
    Ok(())
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
