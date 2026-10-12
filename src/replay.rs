use std::{
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};

use crate::{
    camera::CameraPose,
    math::{Quat, Vec3, lerp_arr},
};

const FLK_REPLAY_VERSION: u8 = 4;
const REPLAY_FOLDER: &str = "Replays";
const REPLAY_EXTENSION: &str = "flk";

/// Capture rate assumed for a replay that was too short to measure one.
pub const DEFAULT_FPS: f32 = 60.0;

/// Range a measured capture rate is believed in, outside which the default wins.
const FPS_RANGE: std::ops::RangeInclusive<f32> = 1.0..=1000.0;

const OFFSET_FRAME_COUNT: u64 = 1;
const OFFSET_FPS: u64 = OFFSET_FRAME_COUNT + 8;

/// Room for the largest skeleton any supported game has (currently THPS4 with 55).
pub const MAX_BONES: usize = 55;

pub struct Replay {
    file_path: PathBuf,
    writer: Option<BufWriter<std::fs::File>>,
    frame_count: u64,
    /// Bracket the recording, so the rate is measured rather than assumed: one
    /// frame goes out per EndScene, at whatever rate the game happens to run.
    first_frame_at: Option<Instant>,
    last_frame_at: Option<Instant>,
}

/// One skater's pose for a single frame.
#[derive(Debug, Clone)]
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
}

/// One frame of a replay.
#[derive(Debug, Clone, Default)]
pub struct ReplayState {
    pub skaters: Vec<SkaterState>,
    pub camera: Option<CameraPose>,
}

pub fn replay_path(name: &str) -> PathBuf {
    PathBuf::from(REPLAY_FOLDER)
        .join(name)
        .with_extension(REPLAY_EXTENSION)
}

impl Replay {
    pub fn new(name: &str) -> Self {
        Self {
            file_path: replay_path(name),
            writer: None,
            frame_count: 0,
            first_frame_at: None,
            last_frame_at: None,
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
        // Patched in by finish(), once both are known.
        writer.write_all(&0u64.to_le_bytes())?;
        writer.write_all(&0f32.to_le_bytes())?;
        self.writer = Some(writer);
        Ok(())
    }

    /// Frames per second over the whole recording, or the default where there
    /// was too little of it to measure.
    pub fn measured_fps(&self) -> f32 {
        let (Some(first), Some(last)) = (self.first_frame_at, self.last_frame_at) else {
            return DEFAULT_FPS;
        };
        let elapsed = (last - first).as_secs_f32();
        if self.frame_count < 2 || elapsed <= 0.0 {
            return DEFAULT_FPS;
        }
        // n frames span n-1 intervals.
        let fps = (self.frame_count - 1) as f32 / elapsed;
        if FPS_RANGE.contains(&fps) {
            fps
        } else {
            DEFAULT_FPS
        }
    }

    /// Writes one captured frame to disk and advances the frame counter.
    /// No-ops if the frame holds no skaters (e.g. none loaded yet).
    pub fn write_frame(&mut self, state: &ReplayState) -> Result<()> {
        if state.skaters.is_empty() {
            return Ok(());
        }
        let writer = self.writer.as_mut().context("replay writer not open")?;

        writer.write_all(&[state.skaters.len() as u8])?;
        for skater in &state.skaters {
            // Only the bones this skeleton has: the arrays are sized for the
            // largest game, not for the one being recorded.
            let bones = skater.bones();
            writer.write_all(&[bones as u8])?;
            write_f32s(writer, &skater.world_pos)?;
            write_f32s(writer, &skater.world_matrix)?;
            for quat in &skater.bone_quats[..bones] {
                write_f32s(writer, quat)?;
            }
            for trans in &skater.bone_trans[..bones] {
                write_f32s(writer, trans)?;
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
        self.frame_count += 1;
        let now = Instant::now();
        self.first_frame_at.get_or_insert(now);
        self.last_frame_at = Some(now);
        Ok(())
    }

    /// Flush the writer and patch the frame count and capture rate into the header.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            let fps = self.measured_fps();
            let mut file = writer
                .into_inner()
                .map_err(|e| anyhow::anyhow!("flush failed: {}", e.error()))?;
            file.seek(SeekFrom::Start(OFFSET_FRAME_COUNT))?;
            file.write_all(&self.frame_count.to_le_bytes())?;
            file.seek(SeekFrom::Start(OFFSET_FPS))?;
            file.write_all(&fps.to_le_bytes())?;
            log::info!(
                "replay saved: {} frames at {:.1} fps → {:?}",
                self.frame_count,
                fps,
                self.file_path
            );
        }
        Ok(())
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// What a replay's header says about it, without reading the frames.
#[derive(Clone, Debug)]
pub struct ReplayInfo {
    pub name: String,
    pub frame_count: u64,
    pub fps: f32,
}

impl ReplayInfo {
    pub fn duration(&self) -> f32 {
        duration_of(self.frame_count, self.fps)
    }
}

/// Every replay in the folder, in name order. An unreadable file is logged and
/// skipped rather than failing the listing, so one bad file cannot hide the rest.
pub fn list() -> Vec<ReplayInfo> {
    let dir = Path::new(REPLAY_FOLDER);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(REPLAY_EXTENSION) {
            continue;
        }
        let Some(name) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
            continue;
        };
        match read_header(&path) {
            Ok((frame_count, fps)) => out.push(ReplayInfo {
                name,
                frame_count,
                fps,
            }),
            Err(e) => log::warn!("skipping replay {name}: {e}"),
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn read_header(path: &Path) -> Result<(u64, f32)> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open replay file: {path:?}"))?;
    read_header_from(&mut file)
}

fn read_header_from(file: &mut std::fs::File) -> Result<(u64, f32)> {
    let mut version = [0u8; 1];
    file.read_exact(&mut version)
        .context("failed to read replay version")?;
    if version[0] != FLK_REPLAY_VERSION {
        anyhow::bail!(
            "unsupported replay version: {} (this build reads {FLK_REPLAY_VERSION})",
            version[0]
        );
    }

    let mut frame_count_bytes = [0u8; 8];
    file.read_exact(&mut frame_count_bytes)
        .context("failed to read frame count")?;
    let mut fps_bytes = [0u8; 4];
    file.read_exact(&mut fps_bytes)
        .context("failed to read capture rate")?;

    let fps = f32::from_le_bytes(fps_bytes);
    // A recording that was cut off never had its header patched, so the rate
    // reads back as zero. Playing it at the default beats dividing by it.
    let fps = if FPS_RANGE.contains(&fps) {
        fps
    } else {
        DEFAULT_FPS
    };
    Ok((u64::from_le_bytes(frame_count_bytes), fps))
}

fn duration_of(frame_count: u64, fps: f32) -> f32 {
    if fps > 0.0 {
        frame_count as f32 / fps
    } else {
        0.0
    }
}

pub struct LoadedReplay {
    pub name: String,
    pub frame_count: u64,
    /// The rate it was captured at, which is the rate playback has to advance
    /// through it at. A game running at 30 or 144 would otherwise play wrong.
    pub fps: f32,
    pub frames: Vec<ReplayState>,
}

impl LoadedReplay {
    pub fn load(name: &str) -> Result<Self> {
        let file_path = replay_path(name);
        let mut file = std::fs::File::open(&file_path)
            .with_context(|| format!("failed to open replay file: {:?}", file_path))?;

        let (frame_count, fps) = read_header_from(&mut file)?;

        let mut frames = Vec::with_capacity(frame_count as usize);

        for _ in 0..frame_count {
            let mut skater_count = [0u8; 1];
            file.read_exact(&mut skater_count)?;
            let skater_count = skater_count[0] as usize;

            let mut skaters = Vec::with_capacity(skater_count);
            for _ in 0..skater_count {
                let mut bone_count = [0u8; 1];
                file.read_exact(&mut bone_count)?;
                let bone_count = bone_count[0];
                if bone_count as usize > MAX_BONES {
                    anyhow::bail!(
                        "replay wants {bone_count} bones per skater, more than the {MAX_BONES} a frame holds"
                    );
                }

                let world_pos = read_f32s::<3>(&mut file)?;
                let world_matrix = read_f32s::<16>(&mut file)?;

                let mut bone_quats = [[0f32; 4]; MAX_BONES];
                for quat in bone_quats[..bone_count as usize].iter_mut() {
                    *quat = read_f32s::<4>(&mut file)?;
                }
                let mut bone_trans = [[0f32; 4]; MAX_BONES];
                for trans in bone_trans[..bone_count as usize].iter_mut() {
                    *trans = read_f32s::<4>(&mut file)?;
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

            frames.push(ReplayState { skaters, camera });
        }

        Ok(Self {
            name: name.to_owned(),
            frame_count,
            fps,
            frames,
        })
    }

    pub fn duration(&self) -> f32 {
        duration_of(self.frame_count, self.fps)
    }

    /// Clamped to the last frame.
    pub fn frame_index_at(&self, seconds: f32) -> usize {
        let last = self.frames.len().saturating_sub(1);
        ((seconds.max(0.0) * self.fps) as usize).min(last)
    }

    /// Sample the replay `seconds` into it, at its own capture rate.
    pub fn sample_at(&self, seconds: f32) -> Option<ReplayState> {
        self.sample(seconds * self.fps)
    }

    /// Sample the replay at a fractional frame index, interpolating between the
    /// two surrounding frames. `t` is in frame units, so `seconds * fps`.
    pub fn sample(&self, t: f32) -> Option<ReplayState> {
        if self.frames.is_empty() {
            return None;
        }
        let max = self.frames.len() - 1;
        let t = t.clamp(0.0, max as f32);
        let i0 = (t as usize).min(max);
        let i1 = (i0 + 1).min(max);
        let frac = t - i0 as f32;

        if i0 == i1 {
            return Some(self.frames[i0].clone());
        }

        let a = &self.frames[i0];
        let b = &self.frames[i1];

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

        let camera = match (&a.camera, &b.camera) {
            (Some(ca), Some(cb)) => Some(CameraPose::lerp(ca, cb, frac)),
            (camera, _) => *camera,
        };

        Some(ReplayState { skaters, camera })
    }
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
