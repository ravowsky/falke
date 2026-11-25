//! Frame-locked video capture of replay playback.
//!
//! Every frame the game draws while a capture runs is one step of the output: the replay,
//! the camera path and the game world are all moved on by exactly one output frame
//! (`speed / fps` seconds of replay) per drawn frame, however long the frame takes to draw
//! and save. So a 500 fps capture is as smooth as a 30 fps one, and nothing is dropped.
//!
//! Each output frame can be drawn more than once ("passes", the world held still between
//! them):
//!   * Image: the frame as the game draws it, and its depth (Depth pass) read from the depth
//!     buffer, which is swapped for a readable one (INTZ) while capturing.
//!   * World: drawn again with the skater hidden.
//!   * Skater: the image with an alpha channel that keeps only where Image and World differ,
//!     i.e. the skater (and what it changes: its shadow, reflections).
//!
//! Timing: a step's settings (replay time, world speed, skater hidden) are applied in
//! `EndScene` and show in the frame drawn after it, so the image of a step is read at the
//! `Present` after the next `EndScene`.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Mutex,
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::JoinHandle,
};

use anyhow::{Context, Result, bail};
use windows::Win32::Graphics::Direct3D9::*;
use windows::core::Interface;

use crate::game;

// Settings

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    ProRes4444,
    ProRes422Hq,
    H264,
    PngSequence,
    TgaSequence,
}

impl Format {
    pub const ALL: [Format; 5] = [
        Format::ProRes4444,
        Format::ProRes422Hq,
        Format::H264,
        Format::PngSequence,
        Format::TgaSequence,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Format::ProRes4444 => "ProRes 4444 (.mov)",
            Format::ProRes422Hq => "ProRes 422 HQ (.mov)",
            Format::H264 => "H.264 (.mp4)",
            Format::PngSequence => "PNG sequence",
            Format::TgaSequence => "TGA sequence",
        }
    }

    /// Whether it is made by ffmpeg (else falke writes the files itself).
    pub fn needs_ffmpeg(self) -> bool {
        matches!(
            self,
            Format::ProRes4444 | Format::ProRes422Hq | Format::H264
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Folder name under `Captures`.
    pub name: String,
    pub fps: u32,
    pub format: Format,
    /// Replay time (seconds) the capture starts and ends at.
    pub start: f32,
    pub end: f32,
    /// Replay seconds per real second of output (the playback speed: 0.1 = ten times slower).
    pub speed: f32,
    pub image: bool,
    pub depth: bool,
    pub world: bool,
    pub skater: bool,
    /// The distance (game units) the Depth pass reaches white at.
    pub depth_far: f32,
    /// How different (0-255, any colour channel) Image and World must be for a pixel to
    /// count as the skater.
    pub matte_threshold: u8,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            name: String::new(),
            fps: 60,
            format: Format::PngSequence,
            start: 0.0,
            end: 0.0,
            speed: 1.0,
            image: true,
            depth: false,
            world: false,
            skater: false,
            depth_far: 3000.0,
            matte_threshold: 6,
        }
    }
}

impl Settings {
    /// Output frames from `start` to `end`.
    pub fn frame_count(&self) -> u32 {
        let span = (self.end - self.start).max(0.0);
        (span / self.speed.max(1e-4) * self.fps as f32).round() as u32 + 1
    }

    /// The replay time of output frame `k`.
    pub fn time_of(&self, k: u32) -> f32 {
        (self.start + k as f32 * self.speed / self.fps as f32).min(self.end.max(self.start))
    }

    /// Whether every frame is drawn a second time, without the skater.
    fn second_pass(&self) -> bool {
        self.world || self.skater
    }
}

/// Where captures go: `Captures\<name>` next to the game.
pub fn folder_of(name: &str) -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    base.join("Captures").join(name)
}

/// ffmpeg.exe next to the game, or on the PATH.
pub fn find_ffmpeg() -> Option<PathBuf> {
    let local = std::env::current_exe().ok()?.parent()?.join("ffmpeg.exe");
    if local.is_file() {
        return Some(local);
    }
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_window(&mut cmd);
    cmd.status()
        .ok()
        .filter(|s| s.success())
        .map(|_| "ffmpeg".into())
}

fn hide_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

// The running capture

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// The game as drawn (Image and Depth).
    Main,
    /// Without the skater, the world held still (World and Skater).
    NoSkater,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Step {
    frame: u32,
    pass: Pass,
}

/// What the UI applies for the frame being set up.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// Replay time to show.
    pub time: f32,
    /// Game world speed for this frame (see `Game::set_time_scale`).
    pub world_speed: f32,
    pub hide_skater: bool,
}

/// What the UI shows about a capture.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub running: bool,
    pub done: u32,
    pub total: u32,
    pub folder: PathBuf,
    pub error: Option<String>,
}

struct Session {
    settings: Settings,
    folder: PathBuf,
    total: u32,
    /// The next step to set up.
    next: Option<Step>,
    /// Set up at the last `EndScene` before the frame being drawn now.
    shown: Option<Step>,
    /// Set up at this frame's `EndScene` (shown in the next frame).
    set_up: Option<Step>,
    /// Frames finished (all passes read).
    done: u32,
    gpu: Option<Gpu>,
    writer: Option<Writer>,
    /// The Image pass of the frame whose NoSkater pass is still to come.
    held: Option<Vec<u8>>,
    /// A single frame shown in the Capture window instead of saved (see `start_preview`).
    preview: Option<PreviewPass>,
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);
static LAST_STATUS: Mutex<Option<Status>> = Mutex::new(None);

fn session() -> std::sync::MutexGuard<'static, Option<Session>> {
    SESSION.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start capturing. Checks what can be checked up front (ffmpeg, the folder).
pub fn start(settings: Settings) -> Result<()> {
    if settings.name.trim().is_empty() {
        bail!("give the capture a name");
    }
    if !(settings.image || settings.depth || settings.world || settings.skater) {
        bail!("pick at least one pass");
    }
    if settings.end <= settings.start {
        bail!("the end must be after the start");
    }
    let ffmpeg =
        if settings.format.needs_ffmpeg() {
            Some(find_ffmpeg().context(
                "ffmpeg.exe not found: put it next to the game's exe, or pick PNG or TGA",
            )?)
        } else {
            None
        };
    let folder = folder_of(settings.name.trim());
    std::fs::create_dir_all(&folder)
        .with_context(|| format!("could not create {}", folder.display()))?;

    let total = settings.frame_count();
    log::info!(
        "[capture] {} frames at {} fps ({:.2}s to {:.2}s at {}x) into {}",
        total,
        settings.fps,
        settings.start,
        settings.end,
        settings.speed,
        folder.display()
    );
    *session() = Some(Session {
        writer: Some(Writer::spawn(folder.clone(), ffmpeg, settings.clone())),
        settings,
        folder,
        total,
        next: Some(Step {
            frame: 0,
            pass: Pass::Main,
        }),
        shown: None,
        set_up: None,
        done: 0,
        gpu: None,
        held: None,
        preview: None,
    });
    *LAST_STATUS.lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(())
}

// Previews

/// A pass to preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewPass {
    Image,
    Depth,
    World,
    Skater,
}

impl PreviewPass {
    pub const ALL: [PreviewPass; 4] = [
        PreviewPass::Image,
        PreviewPass::Depth,
        PreviewPass::World,
        PreviewPass::Skater,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PreviewPass::Image => "Image",
            PreviewPass::Depth => "Depth",
            PreviewPass::World => "World",
            PreviewPass::Skater => "Skater",
        }
    }
}

/// A previewed pass, RGBA, scaled down to at most `PREVIEW_WIDTH` wide.
pub struct Preview {
    pub pass: PreviewPass,
    pub size: (u32, u32),
    pub rgba: Vec<u8>,
    /// Counts up with every new preview.
    pub generation: u64,
}

const PREVIEW_WIDTH: u32 = 1280;

static PREVIEW: Mutex<Option<Preview>> = Mutex::new(None);
static PREVIEW_ERROR: Mutex<Option<String>> = Mutex::new(None);
static PREVIEW_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draw the frame at replay time `time` once, through the same passes as a capture, and
/// keep `pass` to show instead of saving anything. The world is held still.
pub fn start_preview(base: &Settings, time: f32, pass: PreviewPass) -> Result<()> {
    let mut guard = session();
    if guard.is_some() {
        bail!("a capture is running");
    }
    let settings = Settings {
        start: time,
        end: time,
        image: pass == PreviewPass::Image,
        depth: pass == PreviewPass::Depth,
        world: pass == PreviewPass::World,
        skater: pass == PreviewPass::Skater,
        ..base.clone()
    };
    *PREVIEW_ERROR.lock().unwrap_or_else(|e| e.into_inner()) = None;
    *guard = Some(Session {
        writer: None,
        total: 1,
        settings,
        folder: PathBuf::new(),
        next: Some(Step {
            frame: 0,
            pass: Pass::Main,
        }),
        shown: None,
        set_up: None,
        done: 0,
        gpu: None,
        held: None,
        preview: Some(pass),
    });
    Ok(())
}

/// The latest preview, if it is newer than `seen` (its generation).
pub fn take_preview(seen: u64) -> Option<Preview> {
    let mut guard = PREVIEW.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_some_and(|p| p.generation > seen) {
        guard.take()
    } else {
        None
    }
}

/// Why the last preview failed.
pub fn preview_error() -> Option<String> {
    PREVIEW_ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Whether a preview is being drawn.
pub fn preview_running() -> bool {
    session().as_ref().is_some_and(|s| s.preview.is_some())
}

/// Keep a pass's output as the preview (scaled down, alpha over a checkerboard, depth as
/// grey).
fn keep_preview(pass: PreviewPass, job: Job) {
    let (size, rgba) = match job {
        Job::Color {
            size, bgra, alpha, ..
        } => {
            let mut rgba = Vec::with_capacity(bgra.len());
            for (i, px) in bgra.chunks_exact(4).enumerate() {
                let (x, y) = (i as u32 % size.0, i as u32 / size.0);
                let a = if alpha { px[3] as u32 } else { 255 };
                let check = if ((x / 16) + (y / 16)) % 2 == 0 {
                    0x60
                } else {
                    0x90
                };
                for c in [2, 1, 0] {
                    rgba.push(((px[c] as u32 * a + check * (255 - a)) / 255) as u8);
                }
                rgba.push(255);
            }
            (size, rgba)
        }
        Job::Depth { size, depth, .. } => {
            let mut rgba = Vec::with_capacity(depth.len() * 4);
            for v in depth {
                let g = (v >> 8) as u8;
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
            (size, rgba)
        }
    };
    let (size, rgba) = shrink(size, rgba);
    let generation = PREVIEW_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    *PREVIEW.lock().unwrap_or_else(|e| e.into_inner()) = Some(Preview {
        pass,
        size,
        rgba,
        generation,
    });
}

/// Halve an RGBA image (box filter) until it is at most `PREVIEW_WIDTH` wide.
fn shrink(mut size: (u32, u32), mut rgba: Vec<u8>) -> ((u32, u32), Vec<u8>) {
    while size.0 > PREVIEW_WIDTH && size.0 >= 2 && size.1 >= 2 {
        let (w, h) = (size.0 / 2, size.1 / 2);
        let src_w = size.0 as usize;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                for c in 0..4 {
                    let at = |dx: usize, dy: usize| {
                        rgba[((2 * y + dy) * src_w + 2 * x + dx) * 4 + c] as u32
                    };
                    out.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4) as u8);
                }
            }
        }
        size = (w, h);
        rgba = out;
    }
    (size, rgba)
}

/// Stop capturing (finishing the files written so far).
pub fn stop() {
    let taken = session().take();
    if let Some(s) = taken {
        finish(s, None);
    }
}

fn finish(mut s: Session, error: Option<String>) {
    if s.preview.is_some() {
        if let Some(gpu) = s.gpu.take() {
            gpu.release();
        }
        game::set_skater_hidden(false);
        if let Some(e) = error {
            log::warn!("[capture] preview failed: {e}");
            *PREVIEW_ERROR.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
        }
        return;
    }
    let mut error = error;
    if let Some(w) = s.writer.take() {
        if let Err(e) = w.finish() {
            error.get_or_insert(format!("{e:#}"));
        }
    }
    if let Some(gpu) = s.gpu.take() {
        gpu.release();
    }
    game::set_skater_hidden(false);
    match &error {
        Some(e) => log::error!("[capture] stopped: {e}"),
        None => log::info!(
            "[capture] done: {} frames in {}",
            s.done,
            s.folder.display()
        ),
    }
    *LAST_STATUS.lock().unwrap_or_else(|e| e.into_inner()) = Some(Status {
        running: false,
        done: s.done,
        total: s.total,
        folder: s.folder,
        error,
    });
}

pub fn status() -> Option<Status> {
    if let Some(s) = session().as_ref().filter(|s| s.preview.is_none()) {
        return Some(Status {
            running: true,
            done: s.done,
            total: s.total,
            folder: s.folder.clone(),
            error: None,
        });
    }
    LAST_STATUS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Called from the UI's update (in `EndScene`): what to set up for the next frame drawn.
/// The same plan for every `EndScene` until the frame is presented.
pub fn plan() -> Option<Plan> {
    let mut guard = session();
    let s = guard.as_mut()?;
    if s.set_up.is_none() {
        s.set_up = s.next.take();
        s.next = s
            .set_up
            .and_then(|step| successor(&s.settings, s.total, step));
    }
    let step = s.set_up?;
    let world_speed = match step.pass {
        // A preview is a moment, not a stretch of the replay.
        _ if s.preview.is_some() => 0.0,
        // One output frame of game time a drawn frame (the game runs 60 frames a second).
        Pass::Main => s.settings.speed * 60.0 / s.settings.fps as f32,
        // Held (nearly) still: the same moment drawn again.
        Pass::NoSkater => 0.0,
    };
    Some(Plan {
        time: s.settings.time_of(step.frame),
        world_speed,
        hide_skater: step.pass == Pass::NoSkater,
    })
}

fn successor(settings: &Settings, total: u32, step: Step) -> Option<Step> {
    let next = match step.pass {
        Pass::Main if settings.second_pass() => Step {
            frame: step.frame,
            pass: Pass::NoSkater,
        },
        _ => Step {
            frame: step.frame + 1,
            pass: Pass::Main,
        },
    };
    (next.frame < total).then_some(next)
}

/// Called at `Present`, before the overlay is drawn: reads the frame just drawn.
pub fn on_present(dev: &IDirect3DDevice9) {
    let mut guard = session();
    let Some(s) = guard.as_mut() else { return };
    let result = unsafe { present_step(s, dev) };
    s.shown = s.set_up.take();
    let finished = s.shown.is_none() && s.next.is_none() && s.done >= s.total;
    match result {
        Err(e) => {
            let s = guard.take().unwrap();
            drop(guard);
            finish(s, Some(format!("{e:#}")));
        }
        Ok(()) if finished => {
            let s = guard.take().unwrap();
            drop(guard);
            finish(s, None);
        }
        Ok(()) => {}
    }
}

unsafe fn present_step(s: &mut Session, dev: &IDirect3DDevice9) -> Result<()> {
    if s.gpu.is_none() {
        s.gpu = Some(unsafe { Gpu::new(dev, s.settings.depth)? });
    }
    let Some(step) = s.shown else { return Ok(()) };
    let gpu = s.gpu.as_mut().unwrap();
    let (w, h) = (gpu.width, gpu.height);
    let mut jobs = Vec::new();
    match step.pass {
        Pass::Main => {
            let image = unsafe { gpu.read_color(dev)? };
            if s.settings.depth {
                let raw = unsafe { gpu.read_depth(dev)? };
                let (near, far) = game::depth_range().unwrap_or((2.0, 100_000.0));
                jobs.push(Job::Depth {
                    frame: step.frame,
                    size: (w, h),
                    depth: depth16(&raw, near, far, s.settings.depth_far),
                });
            }
            if s.settings.second_pass() {
                if s.settings.image {
                    jobs.push(Job::Color {
                        pass: "image",
                        frame: step.frame,
                        size: (w, h),
                        bgra: image.clone(),
                        alpha: false,
                    });
                }
                s.held = Some(image);
            } else {
                if s.settings.image {
                    jobs.push(Job::Color {
                        pass: "image",
                        frame: step.frame,
                        size: (w, h),
                        bgra: image,
                        alpha: false,
                    });
                }
                s.done = step.frame + 1;
            }
        }
        Pass::NoSkater => {
            let world = unsafe { gpu.read_color(dev)? };
            let image = s.held.take().context("the Image pass is missing")?;
            if s.settings.skater {
                jobs.push(Job::Color {
                    pass: "skater",
                    frame: step.frame,
                    size: (w, h),
                    bgra: matte(&image, &world, s.settings.matte_threshold),
                    alpha: true,
                });
            }
            if s.settings.world {
                jobs.push(Job::Color {
                    pass: "world",
                    frame: step.frame,
                    size: (w, h),
                    bgra: world,
                    alpha: false,
                });
            }
            s.done = step.frame + 1;
        }
    }
    for job in jobs {
        match (s.preview, s.writer.as_mut()) {
            (Some(pass), _) => keep_preview(pass, job),
            (None, Some(writer)) => writer.send(job)?,
            (None, None) => bail!("no writer"),
        }
    }
    Ok(())
}

/// The device is being reset: the capture's GPU resources are gone with it.
pub fn on_reset() {
    let taken = session().take();
    if let Some(s) = taken {
        finish(
            s,
            Some("the game reset its display (resolution change?)".into()),
        );
    }
}

/// While the depth pass runs, the game's depth buffer is swapped for one that can be read.
/// Called by the `SetDepthStencilSurface` hook with what the game binds.
pub fn depth_substitute(surface: *mut core::ffi::c_void) -> Option<*mut core::ffi::c_void> {
    let guard = DEPTH_SWAP.lock().unwrap_or_else(|e| e.into_inner());
    let swap = guard.as_ref()?;
    (surface as usize == swap.0).then_some(swap.1 as *mut core::ffi::c_void)
}

/// The game's depth surface and ours (raw COM pointers; ours is kept alive by `Gpu`).
static DEPTH_SWAP: Mutex<Option<(usize, usize)>> = Mutex::new(None);

// Image maths

/// The image with alpha 255 where it differs from `world` by more than `threshold` in any
/// channel (the skater), 0 elsewhere.
fn matte(image: &[u8], world: &[u8], threshold: u8) -> Vec<u8> {
    let mut out = image.to_vec();
    for (px, w) in out.chunks_exact_mut(4).zip(world.chunks_exact(4)) {
        let diff = (0..3).map(|c| px[c].abs_diff(w[c])).max().unwrap_or(0);
        px[3] = if diff > threshold { 255 } else { 0 };
    }
    out
}

/// Depth buffer values (0..1, perspective) to 16-bit distances: 0 at the camera, 65535 at
/// `max` units and beyond (and the sky).
fn depth16(raw: &[f32], near: f32, far: f32, max: f32) -> Vec<u16> {
    raw.iter()
        .map(|&z| {
            if !(z < 1.0) {
                return u16::MAX;
            }
            let d = near * far / (far - z * (far - near));
            ((d / max).clamp(0.0, 1.0) * 65535.0).round() as u16
        })
        .collect()
}

// GPU side

const INTZ: D3DFORMAT = D3DFORMAT(0x5A54_4E49); // MAKEFOURCC('I','N','T','Z')

/// ps_2_0: `texld r0, t0, s0; mov oC0, r0`. The INTZ depth texture returns the depth in its
/// colour channels.
const COPY_SHADER: [u32; 15] = [
    0xFFFF_0200, // ps_2_0
    0x0200_001F,
    0x8000_0000,
    0xB003_0000, // dcl t0.xy
    0x0200_001F,
    0x9000_0000,
    0xA00F_0800, // dcl_2d s0
    0x0300_0042,
    0x800F_0000,
    0xB0E4_0000,
    0xA0E4_0800, // texld r0, t0, s0
    0x0200_0001,
    0x800F_0800,
    0x80E4_0000, // mov oC0, r0
    0x0000_FFFF, // end
];

struct Gpu {
    dev: IDirect3DDevice9,
    width: u32,
    height: u32,
    /// A plain render target the back buffer is copied to (resolving multisampling), and
    /// its copy in system memory.
    target: IDirect3DSurface9,
    system: IDirect3DSurface9,
    depth: Option<DepthGpu>,
}

// Only used on the game's render thread (the hooks); kept in a static for that thread.
unsafe impl Send for Gpu {}

struct DepthGpu {
    texture: IDirect3DTexture9,
    surface: IDirect3DSurface9,
    /// The game's own depth surface, put back when done.
    game: IDirect3DSurface9,
    target: IDirect3DSurface9,
    system: IDirect3DSurface9,
    shader: IDirect3DPixelShader9,
}

impl Gpu {
    unsafe fn new(dev: &IDirect3DDevice9, depth: bool) -> Result<Self> {
        unsafe {
            let back = dev
                .GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO)
                .context("no back buffer")?;
            let mut desc = D3DSURFACE_DESC::default();
            back.GetDesc(&mut desc)?;
            if desc.Format != D3DFMT_X8R8G8B8 && desc.Format != D3DFMT_A8R8G8B8 {
                bail!(
                    "the game draws to a {:?} back buffer, only 8-bit RGB can be captured",
                    desc.Format
                );
            }
            let mut target = None;
            dev.CreateRenderTarget(
                desc.Width,
                desc.Height,
                desc.Format,
                D3DMULTISAMPLE_NONE,
                0,
                false,
                &mut target,
                std::ptr::null_mut(),
            )
            .context("could not create a capture target")?;
            let mut system = None;
            dev.CreateOffscreenPlainSurface(
                desc.Width,
                desc.Height,
                desc.Format,
                D3DPOOL_SYSTEMMEM,
                &mut system,
                std::ptr::null_mut(),
            )
            .context("could not create a capture surface")?;
            let depth = if depth {
                Some(DepthGpu::new(dev, desc.Width, desc.Height)?)
            } else {
                None
            };
            log::info!(
                "[capture] {}x{} {:?}{}",
                desc.Width,
                desc.Height,
                desc.Format,
                if depth.is_some() { ", depth on" } else { "" }
            );
            Ok(Self {
                dev: dev.clone(),
                width: desc.Width,
                height: desc.Height,
                target: target.context("no capture target")?,
                system: system.context("no capture surface")?,
                depth,
            })
        }
    }

    /// The frame just drawn, BGRA top to bottom.
    unsafe fn read_color(&mut self, dev: &IDirect3DDevice9) -> Result<Vec<u8>> {
        unsafe {
            let back = dev.GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO)?;
            dev.StretchRect(
                &back,
                std::ptr::null(),
                &self.target,
                std::ptr::null(),
                D3DTEXF_NONE,
            )
            .context("could not copy the back buffer")?;
            dev.GetRenderTargetData(&self.target, &self.system)
                .context("could not read the back buffer")?;
            let mut out = read_surface(&self.system, self.width, self.height, 4)?;
            // X8R8G8B8 leaves the alpha byte undefined.
            for px in out.chunks_exact_mut(4) {
                px[3] = 255;
            }
            Ok(out)
        }
    }

    /// The depth of the frame just drawn (0..1 as stored), top to bottom.
    unsafe fn read_depth(&mut self, dev: &IDirect3DDevice9) -> Result<Vec<f32>> {
        let Some(d) = &self.depth else {
            bail!("depth is not set up")
        };
        unsafe {
            let state = dev.CreateStateBlock(D3DSBT_ALL)?;
            let old_target = dev.GetRenderTarget(0).ok();
            let old_depth = dev.GetDepthStencilSurface().ok();

            // The depth texture cannot be read while it is bound as the depth buffer.
            dev.SetDepthStencilSurface(None)?;
            dev.SetRenderTarget(0, &d.target)?;
            let vp = D3DVIEWPORT9 {
                X: 0,
                Y: 0,
                Width: self.width,
                Height: self.height,
                MinZ: 0.0,
                MaxZ: 1.0,
            };
            dev.SetViewport(&vp)?;
            dev.SetVertexShader(None)?;
            dev.SetPixelShader(&d.shader)?;
            dev.SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1)?;
            dev.SetTexture(0, &d.texture)?;
            for (state, value) in [
                (D3DSAMP_MINFILTER, D3DTEXF_POINT.0 as u32),
                (D3DSAMP_MAGFILTER, D3DTEXF_POINT.0 as u32),
                (D3DSAMP_MIPFILTER, D3DTEXF_NONE.0 as u32),
                (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP.0 as u32),
                (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP.0 as u32),
                (D3DSAMP_SRGBTEXTURE, 0),
            ] {
                dev.SetSamplerState(0, state, value)?;
            }
            for (state, value) in [
                (D3DRS_ZENABLE, 0),
                (D3DRS_ZWRITEENABLE, 0),
                (D3DRS_CULLMODE, D3DCULL_NONE.0 as u32),
                (D3DRS_ALPHABLENDENABLE, 0),
                (D3DRS_ALPHATESTENABLE, 0),
                (D3DRS_STENCILENABLE, 0),
                (D3DRS_SCISSORTESTENABLE, 0),
                (D3DRS_FOGENABLE, 0),
                (D3DRS_COLORWRITEENABLE, 0xF),
                (D3DRS_SRGBWRITEENABLE, 0),
            ] {
                dev.SetRenderState(state, value)?;
            }
            let (w, h) = (self.width as f32, self.height as f32);
            #[rustfmt::skip]
            let quad: [f32; 24] = [
                -0.5,     -0.5,     0.0, 1.0, 0.0, 0.0,
                w - 0.5,  -0.5,     0.0, 1.0, 1.0, 0.0,
                -0.5,     h - 0.5,  0.0, 1.0, 0.0, 1.0,
                w - 0.5,  h - 0.5,  0.0, 1.0, 1.0, 1.0,
            ];
            dev.BeginScene()?;
            let drawn = dev.DrawPrimitiveUP(
                D3DPT_TRIANGLESTRIP,
                2,
                quad.as_ptr().cast(),
                (6 * std::mem::size_of::<f32>()) as u32,
            );
            crate::dx_hooks::end_scene_unhooked(dev);

            // Put everything back before checking how it went.
            dev.SetTexture(0, None)?;
            if let Some(t) = &old_target {
                dev.SetRenderTarget(0, t)?;
            }
            dev.SetDepthStencilSurface(old_depth.as_ref())?;
            state.Apply()?;
            drawn.context("could not draw the depth")?;

            dev.GetRenderTargetData(&d.target, &d.system)
                .context("could not read the depth")?;
            let bytes = read_surface(&d.system, self.width, self.height, 4)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
    }

    fn release(self) {
        if let Some(d) = self.depth {
            d.release(&self.dev);
        }
    }
}

impl DepthGpu {
    unsafe fn new(dev: &IDirect3DDevice9, width: u32, height: u32) -> Result<Self> {
        unsafe {
            let d3d = dev.GetDirect3D()?;
            let mut params = D3DDEVICE_CREATION_PARAMETERS::default();
            dev.GetCreationParameters(&mut params)?;
            let mut mode = D3DDISPLAYMODE::default();
            dev.GetDisplayMode(0, &mut mode)?;
            d3d.CheckDeviceFormat(
                params.AdapterOrdinal,
                D3DDEVTYPE_HAL,
                mode.Format,
                D3DUSAGE_DEPTHSTENCIL as u32,
                D3DRTYPE_TEXTURE,
                INTZ,
            )
            .context("the graphics driver has no readable depth buffers (INTZ): no Depth pass")?;
            d3d.CheckDeviceFormat(
                params.AdapterOrdinal,
                D3DDEVTYPE_HAL,
                mode.Format,
                D3DUSAGE_RENDERTARGET as u32,
                D3DRTYPE_SURFACE,
                D3DFMT_R32F,
            )
            .context("the graphics driver cannot draw to 32-bit float targets: no Depth pass")?;

            let game = dev
                .GetDepthStencilSurface()
                .context("the game has no depth buffer bound")?;
            let mut desc = D3DSURFACE_DESC::default();
            game.GetDesc(&mut desc)?;
            if desc.MultiSampleType != D3DMULTISAMPLE_NONE {
                bail!("the game's depth buffer is multisampled (anti-aliasing on): no Depth pass");
            }
            if desc.Width != width || desc.Height != height {
                log::warn!(
                    "[capture] depth buffer {}x{}, back buffer {width}x{height}",
                    desc.Width,
                    desc.Height
                );
            }
            let mut texture = None;
            dev.CreateTexture(
                desc.Width,
                desc.Height,
                1,
                D3DUSAGE_DEPTHSTENCIL as u32,
                INTZ,
                D3DPOOL_DEFAULT,
                &mut texture,
                std::ptr::null_mut(),
            )
            .context("could not create a readable depth buffer")?;
            let texture = texture.context("no depth texture")?;
            let surface = texture.GetSurfaceLevel(0)?;
            let mut target = None;
            dev.CreateRenderTarget(
                width,
                height,
                D3DFMT_R32F,
                D3DMULTISAMPLE_NONE,
                0,
                false,
                &mut target,
                std::ptr::null_mut(),
            )
            .context("could not create a depth target")?;
            let mut system = None;
            dev.CreateOffscreenPlainSurface(
                width,
                height,
                D3DFMT_R32F,
                D3DPOOL_SYSTEMMEM,
                &mut system,
                std::ptr::null_mut(),
            )
            .context("could not create a depth surface")?;
            let shader = dev
                .CreatePixelShader(COPY_SHADER.as_ptr())
                .context("could not create the depth shader")?;

            // From now on the game draws its depth into ours.
            *DEPTH_SWAP.lock().unwrap_or_else(|e| e.into_inner()) =
                Some((game.as_raw() as usize, surface.as_raw() as usize));
            dev.SetDepthStencilSurface(&surface)?;

            Ok(Self {
                texture,
                surface,
                game,
                target: target.context("no depth target")?,
                system: system.context("no depth surface")?,
                shader,
            })
        }
    }

    fn release(self, dev: &IDirect3DDevice9) {
        *DEPTH_SWAP.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // Give the game its own depth buffer back if ours is still bound.
        unsafe {
            if let Ok(bound) = dev.GetDepthStencilSurface() {
                if bound.as_raw() == self.surface.as_raw() {
                    let _ = dev.SetDepthStencilSurface(&self.game);
                }
            }
        }
    }
}

/// A locked surface's rows, `bpp` bytes a pixel, top to bottom without padding.
unsafe fn read_surface(surface: &IDirect3DSurface9, w: u32, h: u32, bpp: usize) -> Result<Vec<u8>> {
    unsafe {
        let mut locked = D3DLOCKED_RECT::default();
        surface
            .LockRect(&mut locked, std::ptr::null(), D3DLOCK_READONLY as u32)
            .context("could not lock the capture surface")?;
        let row = w as usize * bpp;
        let mut out = vec![0u8; row * h as usize];
        for y in 0..h as usize {
            let src = (locked.pBits as *const u8).add(y * locked.Pitch as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(y * row), row);
        }
        let _ = surface.UnlockRect();
        Ok(out)
    }
}

// Writing files

enum Job {
    Color {
        pass: &'static str,
        frame: u32,
        size: (u32, u32),
        bgra: Vec<u8>,
        alpha: bool,
    },
    Depth {
        frame: u32,
        size: (u32, u32),
        depth: Vec<u16>,
    },
}

/// Saves frames on a thread of its own (a few frames ahead at most, then the game waits).
struct Writer {
    jobs: Option<SyncSender<Job>>,
    thread: Option<JoinHandle<Result<()>>>,
}

impl Writer {
    fn spawn(folder: PathBuf, ffmpeg: Option<PathBuf>, settings: Settings) -> Self {
        let (tx, rx) = sync_channel::<Job>(4);
        let thread = std::thread::spawn(move || write_all(rx, &folder, ffmpeg, &settings));
        Self {
            jobs: Some(tx),
            thread: Some(thread),
        }
    }

    fn send(&mut self, job: Job) -> Result<()> {
        let jobs = self.jobs.as_ref().context("the writer has stopped")?;
        if jobs.send(job).is_err() {
            // The thread ended early: its error says why.
            return Err(self
                .join()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("the writer stopped")));
        }
        Ok(())
    }

    fn join(&mut self) -> Result<()> {
        self.jobs = None;
        match self.thread.take().map(|t| t.join()) {
            Some(Ok(r)) => r,
            Some(Err(_)) => bail!("the writer crashed"),
            None => Ok(()),
        }
    }

    fn finish(mut self) -> Result<()> {
        self.join()
    }
}

/// One pass's output: an ffmpeg process, or files.
enum Sink {
    Ffmpeg(Child, BufWriter<ChildStdin>),
    Png { prefix: PathBuf },
    Tga { prefix: PathBuf },
}

fn write_all(
    jobs: Receiver<Job>,
    folder: &Path,
    ffmpeg: Option<PathBuf>,
    settings: &Settings,
) -> Result<()> {
    let mut sinks: Vec<(&'static str, Sink)> = Vec::new();
    let result = (|| -> Result<()> {
        for job in jobs {
            match job {
                Job::Color {
                    pass,
                    frame,
                    size: (w, h),
                    bgra,
                    alpha,
                } => {
                    if !sinks.iter().any(|(p, _)| *p == pass) {
                        let sink =
                            open_sink(folder, pass, ffmpeg.as_deref(), settings, (w, h), alpha)?;
                        sinks.push((pass, sink));
                    }
                    let sink = &mut sinks.iter_mut().find(|(p, _)| *p == pass).unwrap().1;
                    write_color(sink, frame, &bgra, (w, h), alpha)?;
                }
                Job::Depth {
                    frame,
                    size: (w, h),
                    depth,
                } => {
                    let path = folder.join(format!("depth_{frame:06}.png"));
                    let mut be = Vec::with_capacity(depth.len() * 2);
                    for v in depth {
                        be.extend_from_slice(&v.to_be_bytes());
                    }
                    write_png(
                        &path,
                        (w, h),
                        png::ColorType::Grayscale,
                        png::BitDepth::Sixteen,
                        &be,
                    )?;
                }
            }
        }
        Ok(())
    })();
    for (pass, sink) in sinks {
        if let Sink::Ffmpeg(mut child, stdin) = sink {
            drop(stdin);
            match child.wait() {
                Ok(s) if s.success() => {}
                Ok(s) => {
                    log::error!("[capture] ffmpeg ({pass}) ended with {s}, see {pass}.ffmpeg.log")
                }
                Err(e) => log::error!("[capture] ffmpeg ({pass}): {e}"),
            }
        }
    }
    result
}

fn open_sink(
    folder: &Path,
    pass: &str,
    ffmpeg: Option<&Path>,
    settings: &Settings,
    (w, h): (u32, u32),
    alpha: bool,
) -> Result<Sink> {
    // Video without alpha support gets the skater pass as PNGs.
    let format = match settings.format {
        Format::H264 if alpha => Format::PngSequence,
        f => f,
    };
    match format {
        Format::PngSequence => Ok(Sink::Png {
            prefix: folder.join(pass),
        }),
        Format::TgaSequence => Ok(Sink::Tga {
            prefix: folder.join(pass),
        }),
        video => {
            let ffmpeg = ffmpeg.context("ffmpeg.exe not found")?;
            let (codec, ext): (&[&str], &str) = match (video, alpha) {
                (Format::ProRes4444, true) => (
                    &[
                        "-c:v",
                        "prores_ks",
                        "-profile:v",
                        "4444",
                        "-pix_fmt",
                        "yuva444p10le",
                        "-vendor",
                        "apl0",
                    ],
                    "mov",
                ),
                (Format::ProRes4444, false) => (
                    &[
                        "-c:v",
                        "prores_ks",
                        "-profile:v",
                        "4444",
                        "-pix_fmt",
                        "yuv444p10le",
                        "-vendor",
                        "apl0",
                    ],
                    "mov",
                ),
                (Format::ProRes422Hq, false) => (
                    &[
                        "-c:v",
                        "prores_ks",
                        "-profile:v",
                        "3",
                        "-pix_fmt",
                        "yuv422p10le",
                        "-vendor",
                        "apl0",
                    ],
                    "mov",
                ),
                // 422 has no alpha: the skater pass goes 4444.
                (Format::ProRes422Hq, true) => (
                    &[
                        "-c:v",
                        "prores_ks",
                        "-profile:v",
                        "4444",
                        "-pix_fmt",
                        "yuva444p10le",
                        "-vendor",
                        "apl0",
                    ],
                    "mov",
                ),
                _ => (
                    &[
                        "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
                    ],
                    "mp4",
                ),
            };
            let log = File::create(folder.join(format!("{pass}.ffmpeg.log")))?;
            let mut cmd = Command::new(ffmpeg);
            cmd.args(["-y", "-hide_banner", "-f", "rawvideo", "-pix_fmt", "bgra"])
                .args(["-s", &format!("{w}x{h}")])
                .args(["-framerate", &settings.fps.to_string()])
                .args(["-i", "-"])
                .args(codec)
                .arg(folder.join(format!("{pass}.{ext}")))
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::from(log));
            hide_window(&mut cmd);
            let mut child = cmd.spawn().context("could not start ffmpeg")?;
            let stdin = child.stdin.take().context("no pipe to ffmpeg")?;
            Ok(Sink::Ffmpeg(
                child,
                BufWriter::with_capacity(1 << 22, stdin),
            ))
        }
    }
}

fn write_color(
    sink: &mut Sink,
    frame: u32,
    bgra: &[u8],
    (w, h): (u32, u32),
    alpha: bool,
) -> Result<()> {
    match sink {
        Sink::Ffmpeg(_, pipe) => pipe
            .write_all(bgra)
            .context("ffmpeg stopped taking frames (see its .ffmpeg.log)"),
        Sink::Png { prefix } => {
            let path = PathBuf::from(format!("{}_{frame:06}.png", prefix.display()));
            let (color, bytes) = if alpha {
                (png::ColorType::Rgba, bgra_to(bgra, 4))
            } else {
                (png::ColorType::Rgb, bgra_to(bgra, 3))
            };
            write_png(&path, (w, h), color, png::BitDepth::Eight, &bytes)
        }
        Sink::Tga { prefix } => {
            let path = PathBuf::from(format!("{}_{frame:06}.tga", prefix.display()));
            let mut f = BufWriter::new(File::create(&path)?);
            let mut header = [0u8; 18];
            header[2] = 2; // uncompressed true colour
            header[12..14].copy_from_slice(&(w as u16).to_le_bytes());
            header[14..16].copy_from_slice(&(h as u16).to_le_bytes());
            header[16] = 32;
            header[17] = 0x28; // top-left origin, 8 alpha bits
            f.write_all(&header)?;
            f.write_all(bgra)?;
            f.flush()?;
            Ok(())
        }
    }
}

/// BGRA to RGB (`channels` 3) or RGBA (4).
fn bgra_to(bgra: &[u8], channels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bgra.len() / 4 * channels);
    for px in bgra.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0]]);
        if channels == 4 {
            out.push(px[3]);
        }
    }
    out
}

fn write_png(
    path: &Path,
    (w, h): (u32, u32),
    color: png::ColorType,
    depth: png::BitDepth,
    data: &[u8],
) -> Result<()> {
    let file = BufWriter::new(
        File::create(path).with_context(|| format!("could not write {}", path.display()))?,
    );
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(color);
    enc.set_depth(depth);
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header()?;
    writer.write_image_data(data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_cover_the_range() {
        let s = Settings {
            start: 1.0,
            end: 2.0,
            fps: 60,
            speed: 0.5,
            ..Default::default()
        };
        // One replay second at half speed is two seconds of output.
        assert_eq!(s.frame_count(), 121);
        assert!((s.time_of(0) - 1.0).abs() < 1e-6);
        assert!((s.time_of(120) - 2.0).abs() < 1e-5);
        assert!((s.time_of(60) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn steps_go_through_both_passes() {
        let s = Settings {
            world: true,
            ..Default::default()
        };
        let a = Step {
            frame: 0,
            pass: Pass::Main,
        };
        let b = successor(&s, 2, a).unwrap();
        assert_eq!(
            b,
            Step {
                frame: 0,
                pass: Pass::NoSkater
            }
        );
        let c = successor(&s, 2, b).unwrap();
        assert_eq!(
            c,
            Step {
                frame: 1,
                pass: Pass::Main
            }
        );
        let d = successor(&s, 2, c).unwrap();
        assert!(successor(&s, 2, d).is_none());
    }

    #[test]
    fn matte_keeps_what_differs() {
        let image = [10, 10, 10, 255, 200, 50, 50, 255];
        let world = [12, 10, 10, 255, 20, 50, 50, 255];
        let m = matte(&image, &world, 6);
        assert_eq!(m[3], 0);
        assert_eq!(m[7], 255);
        assert_eq!(&m[4..7], &[200, 50, 50]);
    }

    #[test]
    fn preview_is_halved_until_small_enough() {
        let (w, h) = (2600u32, 4u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        // The first 2x2 block: two white pixels, two black ones.
        for (x, y) in [(0usize, 0usize), (1, 1)] {
            let i = (y * w as usize + x) * 4;
            rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
        let (size, out) = shrink((w, h), rgba);
        assert_eq!(size, (1300 / 2, 1));
        assert_eq!(out.len(), 650 * 4);
        // Halved twice: the 4x4 block holds 2 white pixels of 16.
        assert_eq!(out[0], 127 / 4);
    }

    #[test]
    fn depth_is_linear_distance() {
        let (near, far) = (2.0, 10_000.0);
        // The stored value for a point 1500 units away.
        let d: f32 = 1500.0;
        let z = far / (far - near) * (1.0 - near / d);
        let v = depth16(&[z, 1.0], near, far, 3000.0);
        assert!((v[0] as i32 - 32768).abs() < 40, "{}", v[0]);
        assert_eq!(v[1], u16::MAX);
    }
}
