//! Tony Hawk's American Wasteland / reTHAWed backend.
//!
//! Found by reverse engineering reTHAWed.exe (image base 0x400000).
//!
//! Camera:
//!   * View mode: FUN_0058f380(this, mode) sets it: 0 = skate cam, 1 = freecam, 2 = fixed cam.
//!     It is hooked to know the mode.
//!   * Render camera (FUN_004edaa0's tables, re-implemented here without calling it):
//!       tbl    = *(0x0071cef0)             i32[] viewport -> camera index (-1 = none)
//!       slot   = *(0x007d0a10 + idx * 4)
//!       camera = *(slot + 0x18)
//!     Layout: position +0x70, 4x4 rotation matrix +0x80, vertical FOV +0xc0 (+0xc4 horizontal
//!     FOV and +0xc8 a tan-based scale are derived from it). The camera object is swapped when
//!     the freecam is entered, so it is looked up on every access.
//!   * The game copies the "viewer_cam" object's transform (CRC 0xeb17151b, found with
//!     FUN_004331e0) onto the render camera every frame, so poses are written there too. Its
//!     layout is checked against the render camera before the first write.
//!
//! Skater:
//!   * Objects are found through the object manager (*0x00742a10): two hash tables at +0x8 and
//!     +0xc, 12-byte buckets { name CRC, object, next }. Object layout: type +0x20 (skater
//!     0xbaf9b341), position +0x68, 4x4 matrix +0x88, component table +0x60 / count +0x64
//!     (entries { CRC, component }).
//!   * The pose the skater is drawn with: skeleton component (0x222756d5) +0x18 -> skeleton,
//!     skeleton +0x4 -> 52 bone matrices of 64 bytes in model space (scaled and skewed, so they
//!     are stored as rotation + triangular factors, see `pose_states`).
//!   * 0x509e10 (vtable slot 0x80 of the model parts), thiscall(this, world, src, count, a4, a5),
//!     `ret 0x14`: copies the 4x4 world matrix at `world` to *(this + 0x50) + 0xc and `count`
//!     matrices from `src` to *(*(this + 0x50) + 0x4c). The game calls it once per model part
//!     (12 for the skater) every frame. It is hooked to learn the skater's parts and what they
//!     are drawn with.
//!
//! Replay: the skater is recorded every frame (pose plus the world matrix it is drawn with) and
//! replayed in the freecam only. The game does not update the skater there, so the pose is
//! written into the pose buffer and the commit function is called for the skater's parts with
//! the replay's world matrix: the skater is redrawn, the game's real skater is left alone.
//! (Moving the real skater, as skate-cam playback did, crashed the game on far starts.)
//! Cars and the nearest pedestrians are recorded too and drawn as recorded through the same
//! hook (see "Traffic replay"); the game clock follows the playback speed (0x436210).

use std::{
    ffi::c_void,
    mem::transmute,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result};
use nalgebra_glm as glm;
use retour::static_detour;

use crate::{
    camera::CameraPose,
    game::Game,
    math::Vec3,
    replay::{MAX_BONES, ReplayState, SkaterState},
    thaw_blend::*,
};

// Code.
const ADDR_SET_VIEW_MODE_FIELD: usize = 0x0058f380;
const ADDR_COMMIT_PART: usize = 0x00509e10;
const ADDR_GET_OBJECT_BY_NAME: usize = 0x004331e0;
/// The game's time scale setter (what the SetSlomo script command calls), cdecl(scale).
/// Everything on the game clock follows it: traffic, pedestrians, particles, water.
/// The current scale is the f32 at 0x719ea0.
const ADDR_SET_TIME_SCALE: usize = 0x00436210;
const ADDR_TIME_SCALE: usize = 0x00719ea0;
type SetTimeScaleFn = unsafe extern "C" fn(scale: f32);
type GetObjectByNameFn = unsafe extern "thiscall" fn(manager: usize, name_crc: u32) -> usize;

// Data.
const ADDR_SCREEN_ANGLE: usize = 0x007d0a3c;
const ADDR_VIEWPORT_CAM_TABLE_PTR: usize = 0x0071cef0;
const ADDR_CAMERA_SLOTS: usize = 0x007d0a10;
const CAMERA_SLOT_COUNT: i32 = 10;
const SLOT_CAMERA_OFFSET: usize = 0x18;
/// The wrapper at 0x004331e0 loads the manager from here itself.
const ADDR_OBJECT_MANAGER_PTR: usize = 0x00742a10;
/// The game's code, for recognising return addresses in crash logs.
const GAME_CODE: std::ops::Range<usize> = 0x0040_1000..0x006d_b000;

// Render camera layout.
const CAM_POSITION: usize = 0x70;
const CAM_MATRIX: usize = 0x80;
const CAM_VFOV: usize = 0xc0;
const CAM_HFOV: usize = 0xc4;
const CAM_FOV_SCALE: usize = 0xc8;

// Object layout.
const OBJ_TYPE: usize = 0x20;
const OBJ_COMPONENT_TABLE: usize = 0x60;
const OBJ_COMPONENT_COUNT: usize = 0x64;
const OBJ_POSITION: usize = 0x68;
const OBJ_MATRIX: usize = 0x88;
const TYPE_SKATER: u32 = 0xbaf9b341;
const CRC_VIEWER_CAM: u32 = 0xeb17151b; // "viewer_cam"
const CRC_SKELETON_COMPONENT: u32 = 0x222756d5; // "skeleton"
const CRC_MODEL: u32 = 0x286a8d26; // "model"

// Object manager tables.
const MANAGER_TABLE_OFFSETS: [usize; 2] = [0x8, 0xc];
const TABLE_BUCKETS_OFFSET: usize = 0x4;
const BUCKET_SIZE: usize = 12;
const MAX_TABLE_SHIFT: u32 = 16;
const MAX_CHAIN: u32 = 64;

// Model parts (see 0x509e10).
const PART_SECTOR: usize = 0x50;
const SECTOR_POSE: usize = 0x4c;

const MODE_FREECAM: i32 = 1;
/// The skater's pose buffer needs at least this many readable bone matrices.
const MIN_BONES: usize = 40;

static_detour! {
    static ViewModeFieldHook: unsafe extern "thiscall" fn(*mut usize, i32);
    static CommitPartHook: unsafe extern "thiscall" fn(*mut c_void, u32, *mut u32, u32, u32, u32) -> u32;
}

/// The most recent view mode passed to FUN_0058f380. -1 until the first call.
static LAST_MODE: AtomicI32 = AtomicI32::new(-1);
/// The viewer_cam object whose layout was confirmed against the render camera.
static VALIDATED_VIEWER: AtomicUsize = AtomicUsize::new(0);
static WARNED_NO_VIEWER: AtomicBool = AtomicBool::new(false);
static WARNED_MISMATCH: AtomicBool = AtomicBool::new(false);
/// The freecam camera object last seen, only to log when it changes.
static LAST_CAMERA: AtomicUsize = AtomicUsize::new(0);

pub struct Thaw;

impl Game for Thaw {
    fn name(&self) -> &'static str {
        "THAW / reTHAWed"
    }

    unsafe fn install_hooks(&self) -> Result<()> {
        unsafe {
            ViewModeFieldHook
                .initialize(transmute(ADDR_SET_VIEW_MODE_FIELD), hk_view_mode_field)
                .context("failed to init FUN_0058f380 hook")?
                .enable()
                .context("failed to enable FUN_0058f380 hook")?;
        }
        install_crash_logging();

        // Only needed for skater replay: if it fails, the camera features still work.
        let commit_hook = unsafe {
            CommitPartHook
                .initialize(transmute(ADDR_COMMIT_PART), hk_commit_part)
                .and_then(|h| h.enable())
        };
        if let Err(e) = commit_hook {
            log::warn!("[THAW] hook on 0x{ADDR_COMMIT_PART:x} failed, no skater replay: {e}");
        }
        Ok(())
    }

    unsafe fn capture_frame(&self, frame: &mut ReplayState) {
        // The camera the game renders through (skate cam, freecam or fixed cam).
        if let Some(cam) = active_camera(0) {
            frame.camera = read_pose(cam);
        }
        let skater = capture_skater();
        if !skater.is_empty() {
            frame.skaters.extend(skater);
            frame.skaters.extend(capture_cars());
            if crate::game::sync_moving_objects() {
                frame.skaters.extend(capture_movers());
                check_mover_layout();
            }
            report_moving_objects();
        }
    }

    /// Redraws the skater from the replay. Freecam only: nothing is written elsewhere.
    unsafe fn apply_frame(&self, frame: &ReplayState) {
        if frame.skaters.is_empty() || LAST_MODE.load(Ordering::Relaxed) != MODE_FREECAM {
            return;
        }
        let Some(obj) = skater_object() else { return };
        let Some((pose_addr, bones)) = pose_buffer(obj) else {
            return;
        };
        let pose = rebuild_pose(frame);
        write_matrices(pose_addr, &pose, pose.len().min(bones));

        let first = frame
            .skaters
            .iter()
            .find(|s| s.bones() > 0)
            .unwrap_or(&frame.skaters[0]);
        let mut matrix = first.world_matrix;
        orthonormalize(&mut matrix);

        // Entries after the three pose entries without bones are the parts' own world
        // matrices.
        let part_worlds: Vec<[f32; 16]> = frame
            .skaters
            .iter()
            .skip(3)
            .filter(|s| s.bones() == 0)
            .map(|s| {
                let mut m = s.world_matrix;
                orthonormalize(&mut m);
                world_matrix_of(&s.world_pos, &m)
            })
            .collect();
        // The game does not commit the skater in the freecam; if it ever does, its own
        // commit is left alone.
        if P_COMMITS.swap(0, Ordering::Relaxed) == 0 {
            commit_skater(pose_addr, &first.world_pos, &matrix, &part_worlds);
        }
        play_cars(frame);
        if crate::game::sync_moving_objects() {
            play_movers(frame);
        }
    }

    fn in_freecam(&self) -> bool {
        LAST_MODE.load(Ordering::Relaxed) == MODE_FREECAM
    }

    fn is_replay_available(&self) -> bool {
        true
    }

    /// The freecam camera while flying it; otherwise the camera the game renders through
    /// (read only: `set_camera` still only touches the freecam). Returning something
    /// outside the freecam is what lets falke's UI run in skate cam.
    fn camera(&self) -> Option<CameraPose> {
        match freecam_camera() {
            Some(cam) => read_pose(cam),
            None => read_pose(active_camera(0)?),
        }
    }

    fn set_camera(&self, pose: &CameraPose) {
        if let Some(cam) = freecam_camera() {
            write_pose(cam, pose);
        }
    }

    fn fov(&self) -> f32 {
        match freecam_camera().and_then(|cam| read::<f32>(cam + CAM_VFOV)) {
            Some(v) => v,
            None => read::<f32>(ADDR_SCREEN_ANGLE).unwrap_or(0.0),
        }
    }

    fn set_fov(&self, fov: f32) {
        if let Some(cam) = freecam_camera() {
            write_fov(cam, fov);
        }
    }

    fn blend_skaters(
        &self,
        frames: &[ReplayState],
        i0: usize,
        frac: f32,
    ) -> Option<Vec<SkaterState>> {
        blend_in_world_space(frames, i0, frac)
    }

    /// Called every frame by the UI; only calls the game when the speed changes, so the
    /// game's own slow motion (e.g. focus) is left alone the rest of the time.
    fn set_time_scale(&self, scale: f32) {
        if !scale.is_finite() {
            return;
        }
        // Not 0: the game divides by it. 0.001 is still enough to stand still while paused
        // (a crane at 1% speed was seen creeping).
        let scale = scale.clamp(0.001, 10.0);
        let last = f32::from_bits(TIME_SCALE_SET.load(Ordering::Relaxed));
        if (scale - last).abs() < 1e-4 {
            return;
        }
        // Only touch it when the game is running a level (the value is a sane number).
        let Some(current) =
            read::<f32>(ADDR_TIME_SCALE).filter(|v| v.is_finite() && *v > 0.0 && *v < 100.0)
        else {
            return;
        };
        TIME_SCALE_SET.store(scale.to_bits(), Ordering::Relaxed);
        if (current - scale).abs() < 1e-4 {
            return;
        }
        let set: SetTimeScaleFn = unsafe { transmute(ADDR_SET_TIME_SCALE) };
        unsafe { set(scale) };
        log::debug!("[THAW] game speed {current} -> {scale}");
    }
}

/// The time scale falke last asked for (starts at 1.0, the game's normal speed).
static TIME_SCALE_SET: AtomicU32 = AtomicU32::new(0x3f80_0000);

fn hk_view_mode_field(this: *mut usize, mode: i32) {
    let prev_mode = LAST_MODE.swap(mode, Ordering::Relaxed);
    if prev_mode != mode {
        log::debug!("[THAW] view mode {prev_mode} -> {mode}");
    }
    unsafe { ViewModeFieldHook.call(this, mode) };
}

// The skater's model parts.

/// One call of the commit function for the skater, kept to repeat it in the freecam.
#[derive(Clone, Copy)]
struct PartCall {
    seen: Instant,
    this: usize,
    count: u32,
    a4: u32,
    a5: u32,
    /// Row lengths of the world matrix the game gave this part (parts are scaled
    /// differently, e.g. 0.99 / 1.06 / 1.03).
    scale: [f32; 3],
    /// That world matrix with the scale removed.
    drawn: [f32; 16],
}

/// The skater's pose buffer as last seen (refreshed from inside the hook).
static P_ADDR: AtomicUsize = AtomicUsize::new(0);
/// Commits of that buffer since `apply_frame` last looked.
static P_COMMITS: AtomicU32 = AtomicU32::new(0);
static PARTS: Mutex<Vec<PartCall>> = Mutex::new(Vec::new());
static HOOK_CALLS: AtomicU32 = AtomicU32::new(0);
/// Whether that world matrix is `world_matrix_of(object position, object matrix)`, which
/// is what `commit_skater` passes. Checked on normal frames.
static LAYOUT_OK: AtomicBool = AtomicBool::new(false);
static LAYOUT_MATCH: AtomicU32 = AtomicU32::new(0);
static LAYOUT_MISMATCH: AtomicU32 = AtomicU32::new(0);
static LAYOUT_NOTES: AtomicU32 = AtomicU32::new(0);

fn hk_commit_part(
    this: *mut c_void,
    world: u32,
    src: *mut u32,
    count: u32,
    a4: u32,
    a5: u32,
) -> u32 {
    if HOOK_CALLS.fetch_add(1, Ordering::Relaxed) % 120 == 0 {
        refresh_pose_address();
    }
    let src_addr = src as usize;
    if src_addr != 0 && src_addr == P_ADDR.load(Ordering::Relaxed) {
        let given = floats(&read_block(world as usize, 64));
        // The pose as it is drawn, taken at the first part of the frame (see `capture_skater`).
        if !POSE_TAKEN.swap(true, Ordering::Relaxed) {
            let bytes = read_block(src_addr, count.min(MAX_BONES as u32) as usize * 64);
            if let Ok(mut p) = DRAWN_POSE.try_lock() {
                *p = Some((Instant::now(), bytes));
            }
        }
        if given.len() == 16 {
            let drawn = unscaled(&given);
            note_part(this as usize, count, a4, a5, row_lengths(&given), drawn);
            check_layout(&drawn);
        }
    }
    if LAST_MODE.load(Ordering::Relaxed) == MODE_FREECAM {
        replace_car(world, src_addr, count);
        if crate::game::sync_moving_objects() {
            place_movers();
        }
    }
    unsafe { CommitPartHook.call(this, world, src, count, a4, a5) }
}

/// The skater's pose copied when the game draws it, and whether this frame's copy is taken.
static DRAWN_POSE: Mutex<Option<(Instant, Vec<u8>)>> = Mutex::new(None);
static POSE_TAKEN: AtomicBool = AtomicBool::new(false);

fn refresh_pose_address() {
    let Some(obj) = skater_object() else { return };
    let Some((addr, _)) = pose_buffer(obj) else {
        return;
    };
    if P_ADDR.swap(addr, Ordering::Relaxed) != addr {
        PARTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

fn note_part(this: usize, count: u32, a4: u32, a5: u32, scale: [f32; 3], drawn: [f32; 16]) {
    P_COMMITS.fetch_add(1, Ordering::Relaxed);
    let Ok(mut parts) = PARTS.try_lock() else {
        return;
    };
    let call = PartCall {
        seen: Instant::now(),
        this,
        count,
        a4,
        a5,
        scale,
        drawn,
    };
    if let Some(c) = parts.iter_mut().find(|c| c.this == this) {
        *c = call;
    } else if parts.len() < 24 {
        parts.push(call);
    }
}

/// Checks what the skater is drawn with against the object's transform.
fn check_layout(drawn: &[f32; 16]) {
    if LAYOUT_OK.load(Ordering::Relaxed) && LAYOUT_MATCH.load(Ordering::Relaxed) >= 1000 {
        return;
    }
    let Some(obj) = skater_object() else { return };
    let (Some(pos), Some(mat)) = (read3(obj + OBJ_POSITION), read_matrix(obj + OBJ_MATRIX)) else {
        return;
    };
    let built = world_matrix_of(&pos, &mat);
    let rot_err = (0..12)
        .map(|i| (drawn[i] - built[i]).abs())
        .fold(0f32, f32::max);
    let pos_err = (12..15)
        .map(|i| (drawn[i] - built[i]).abs())
        .fold(0f32, f32::max);
    // Mismatches are expected in grinds and manuals (the drawn skater leans).
    let (m, x) = if rot_err < 0.01 && pos_err < 1.0 {
        (
            LAYOUT_MATCH.fetch_add(1, Ordering::Relaxed) + 1,
            LAYOUT_MISMATCH.load(Ordering::Relaxed),
        )
    } else {
        (
            LAYOUT_MATCH.load(Ordering::Relaxed),
            LAYOUT_MISMATCH.fetch_add(1, Ordering::Relaxed) + 1,
        )
    };
    let ok = m >= 20 && x < m;
    if LAYOUT_OK.swap(ok, Ordering::Relaxed) != ok {
        log::info!(
            "[THAW] skater replay {}",
            if ok { "ready" } else { "not ready yet" }
        );
    }
}

/// Calls the commit function for the skater's parts with the replay's world matrix, for
/// the parts that were seen recently and still draw from the skater's pose buffer.
/// `part_worlds` holds one recorded world matrix per part (in part-address order), or is
/// empty for recordings made before parts were recorded; then all parts share `matrix`.
fn commit_skater(pose_addr: usize, pos: &[f32; 3], matrix: &[f32; 16], part_worlds: &[[f32; 16]]) {
    if !LAYOUT_OK.load(Ordering::Relaxed) {
        if LAYOUT_NOTES.fetch_add(1, Ordering::Relaxed) % 600 == 0 {
            log::info!(
                "[THAW] skater replay is not ready: skate for a few seconds in skate cam first"
            );
        }
        return;
    }
    let parts: Vec<PartCall> = PARTS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let Some(newest) = parts.iter().map(|c| c.seen).max() else {
        return;
    };
    let world = world_matrix_of(pos, matrix);
    let mut usable: Vec<PartCall> = parts
        .iter()
        .copied()
        .filter(|c| newest.duration_since(c.seen).as_millis() < 1000)
        .filter(|c| {
            read::<usize>(c.this + PART_SECTOR)
                .filter(|s| *s != 0)
                .and_then(|s| read::<usize>(s + SECTOR_POSE))
                == Some(pose_addr)
        })
        .collect();
    usable.sort_by_key(|c| c.this);
    usable.truncate(MAX_PARTS);
    let per_part = part_worlds.len() == usable.len();
    for (k, c) in usable.iter().enumerate() {
        let base = if per_part { &part_worlds[k] } else { &world };
        let mut part_world = scaled(base, c.scale);
        unsafe {
            CommitPartHook.call(
                c.this as *mut c_void,
                part_world.as_mut_ptr() as u32,
                pose_addr as *mut u32,
                c.count,
                c.a4,
                c.a5,
            )
        };
    }
}

// Traffic replay (cars and pedestrians).
//
// Cars (type 0x5c0312eb) and pedestrians (type 0xd2db7f97) have a skeleton like the skater
// (a handful of bones for a car's body and wheels, a full body for a person) and are drawn
// through the commit function, which gets their world matrix and bone matrices. Every frame
// of a recording stores each car and the MAX_PEDESTRIANS people nearest to the skater
// (object position and matrix, and the bones) as an extra replay entry. In the freecam the
// game keeps them moving, so during playback the commit hook replaces what each one is
// drawn with by its recorded self. Cars that were not recorded are hidden; people that were
// not recorded (too far away then) keep walking.
//
// These entries come after the skater's pose and part entries. They have bones, and their
// world matrix carries a marker and the object's key (its address, so only within the game
// session it was recorded in) in the otherwise unused fourth column. Throughout this section
// "car" means either kind.

const TYPE_CAR: u32 = 0x5c0312eb;
const TYPE_PEDESTRIAN: u32 = 0xd2db7f97;
const MAX_CAR_BONES: usize = 16;
/// Pedestrians recorded per frame (the nearest to the skater); each is about 1.8 KB a frame.
const MAX_PEDESTRIANS: usize = 24;
const CAR_MARK: f32 = 4242.0;
/// Cars not in the recording are moved this far down while a replay plays.
const HIDE_DEPTH: f32 = 100_000.0;

/// One recorded car, ready to be drawn.
#[derive(Clone)]
struct CarFrame {
    key: u32,
    pos: [f32; 3],
    rows: [[f32; 3]; 3],
    bones: Vec<[f32; 16]>,
}

/// The cars of the frame being played, and the live cars' pose buffers.
static PLAYED_CARS: Mutex<Option<(Instant, Vec<CarFrame>)>> = Mutex::new(None);
/// Live pose buffers: (buffer, key, is a pedestrian).
static CAR_BUFFERS: Mutex<Vec<(usize, u32, Kind)>> = Mutex::new(Vec::new());

/// What a recorded object with a skeleton is.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Car,
    Pedestrian,
    /// Anything else with a skeleton (animals, animated props, other characters).
    Other,
}

fn kind_of(obj: usize) -> Kind {
    match read::<u32>(obj + OBJ_TYPE) {
        Some(TYPE_CAR) => Kind::Car,
        Some(TYPE_PEDESTRIAN) => Kind::Pedestrian,
        _ => Kind::Other,
    }
}

/// Other objects with a skeleton recorded per frame (the nearest to the skater).
const MAX_OTHERS: usize = 24;
static CAR_LOGS: AtomicU32 = AtomicU32::new(0);

/// Every vehicle (any object with a small skeleton: cars, and whatever else drives around
/// the same way) and the pedestrians, but not the skater. With `all_pedestrians` false only
/// the pedestrians nearest to the skater are taken (for recording).
fn car_objects(all_pedestrians: bool) -> Vec<usize> {
    let mut cars: Vec<usize> = Vec::new();
    let mut peds: Vec<usize> = Vec::new();
    let mut others: Vec<usize> = Vec::new();
    for_each_object(|obj| {
        if cars.len() >= 64 || peds.len() + others.len() >= 512 {
            return;
        }
        if cars.contains(&obj) || peds.contains(&obj) || others.contains(&obj) {
            return;
        }
        let ty = read::<u32>(obj + OBJ_TYPE).unwrap_or(0);
        if ty == TYPE_SKATER {
            return;
        }
        let Some((_, bones)) = car_pose_buffer(obj) else {
            return;
        };
        match kind_of(obj) {
            Kind::Car => cars.push(obj),
            Kind::Pedestrian => peds.push(obj),
            Kind::Other => {
                others.push(obj);
                note_object_type(ty, bones);
            }
        }
    });
    if !all_pedestrians {
        let me = skater_object()
            .and_then(|o| read3(o + OBJ_POSITION))
            .unwrap_or([0.0; 3]);
        let dist = |o: usize| {
            read3(o + OBJ_POSITION).map_or(f32::INFINITY, |p| {
                let d: [f32; 3] = std::array::from_fn(|i| p[i] - me[i]);
                dot3(d, d)
            })
        };
        for (list, max) in [(&mut peds, MAX_PEDESTRIANS), (&mut others, MAX_OTHERS)] {
            if list.len() > max {
                list.sort_by(|a, b| dist(*a).total_cmp(&dist(*b)));
                list.truncate(max);
            }
        }
    }
    note_pedestrians(&peds);
    cars.extend(peds);
    cars.extend(others);
    cars.sort_unstable();
    cars
}

/// Logs once that pedestrians are recorded. (Their bone matrices are plain rotations with
/// a uniform scale, so the car encoding stores them exactly.)
fn note_pedestrians(peds: &[usize]) {
    static DONE: AtomicBool = AtomicBool::new(false);
    if !peds.is_empty() && !DONE.swap(true, Ordering::Relaxed) {
        log::info!(
            "[THAW] pedestrians are recorded ({} nearest to the skater)",
            peds.len()
        );
    }
}

/// Logs each type of other object with a skeleton once.
fn note_object_type(ty: u32, bones: usize) {
    let Ok(mut seen) = VEHICLE_TYPES.try_lock() else {
        return;
    };
    if !seen.contains(&ty) && seen.len() < 32 {
        seen.push(ty);
        log::info!("[THAW] objects of type 0x{ty:08x} ({bones} bones) are recorded");
    }
}

static VEHICLE_TYPES: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// A car's pose buffer and how many bone matrices it holds.
fn car_pose_buffer(obj: usize) -> Option<(usize, usize)> {
    let skeleton = read::<usize>(component_of(obj, CRC_SKELETON_COMPONENT)? + 0x18)?;
    let addr = read::<usize>(skeleton + 0x4)?;
    if !addr_ok::<u32>(addr) {
        return None;
    }
    let bones = bone_run(&read_block(addr, MAX_BONES * 64));
    let bones = if kind_of(obj) == Kind::Car {
        bones.min(MAX_CAR_BONES)
    } else {
        bones.min(MAX_BONES)
    };
    (bones >= 1).then_some((addr, bones))
}

/// The key a car is recognised by during playback (its object, same game session).
fn car_key(obj: usize) -> u32 {
    obj as u32
}

fn key_floats(key: u32) -> (f32, f32) {
    ((key >> 16) as f32, (key & 0xffff) as f32)
}

fn key_of(m: &[f32; 16]) -> Option<u32> {
    (m[3] == CAR_MARK).then(|| ((m[7] as u32) << 16) | (m[11] as u32 & 0xffff))
}

/// The replay entries for all cars in this frame.
fn capture_cars() -> Vec<SkaterState> {
    let mut out = Vec::new();
    for obj in car_objects(false) {
        let Some((addr, bones)) = car_pose_buffer(obj) else {
            continue;
        };
        let block = read_block(addr, bones * 64);
        if block.len() < bones * 64 {
            continue;
        }
        let (Some(pos), Some(mut matrix)) =
            (read3(obj + OBJ_POSITION), read_matrix(obj + OBJ_MATRIX))
        else {
            continue;
        };
        orthonormalize(&mut matrix);
        let (hi, lo) = key_floats(car_key(obj));
        matrix[3] = CAR_MARK;
        matrix[7] = hi;
        matrix[11] = lo;
        let mut state = SkaterState {
            world_pos: pos,
            world_matrix: matrix,
            bone_count: bones as u8,
            bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
            bone_trans: [[0f32; 4]; MAX_BONES],
        };
        for b in 0..bones {
            let m: [f32; 16] = std::array::from_fn(|i| f32_at(&block, b * 64 + i * 4));
            let len = row_lengths(&m);
            let mut unit = m;
            orthonormalize(&mut unit);
            let q = rows_to_quat(&[
                [unit[0], unit[1], unit[2]],
                [unit[4], unit[5], unit[6]],
                [unit[8], unit[9], unit[10]],
            ]);
            state.bone_quats[b] = [q.coords[0], q.coords[1], q.coords[2], q.coords[3]];
            state.bone_trans[b] = [m[12], m[13], m[14], (len[0] + len[1] + len[2]) / 3.0];
        }
        out.push(state);
    }
    out
}

/// The recorded cars of a frame (entries after the skater's).
fn car_entries(f: &ReplayState) -> impl Iterator<Item = (u32, &SkaterState)> {
    f.skaters
        .iter()
        .skip(3)
        .filter(|s| s.bones() > 0)
        .filter_map(|s| key_of(&s.world_matrix).map(|k| (k, s)))
}

fn car_bone_matrix(s: &SkaterState, b: usize) -> [f32; 16] {
    let q = s.bone_quats[b];
    let rows = quat_to_rows(&glm::Quat::new(q[3], q[0], q[1], q[2]));
    let t = s.bone_trans[b];
    let k = if t[3].is_finite() && t[3] > 0.2 && t[3] < 5.0 {
        t[3]
    } else {
        1.0
    };
    let mut m = [0f32; 16];
    for r in 0..3 {
        for c in 0..3 {
            m[r * 4 + c] = rows[r][c] * k;
        }
    }
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1.0;
    m
}

/// The cars between two recorded frames, matched by key (a car only in `a` stays as it was).
fn blend_cars(a: &ReplayState, b: &ReplayState, t: f32) -> Vec<SkaterState> {
    let later: Vec<(u32, &SkaterState)> = car_entries(b).collect();
    car_entries(a)
        .map(|(key, ca)| {
            let Some((_, cb)) = later.iter().find(|(k, _)| *k == key) else {
                return ca.clone();
            };
            // A car or person the game moves somewhere else (respawn) is shown as a cut.
            let d: [f32; 3] = std::array::from_fn(|i| ca.world_pos[i] - cb.world_pos[i]);
            if dot3(d, d).sqrt() > TELEPORT {
                return if t < 0.5 { ca.clone() } else { (*cb).clone() };
            }
            let bones = ca.bones().min(cb.bones());
            let (_, mut matrix) = blend_world(&ca.world_matrix, &cb.world_matrix, t);
            matrix[3] = ca.world_matrix[3];
            matrix[7] = ca.world_matrix[7];
            matrix[11] = ca.world_matrix[11];
            let mut s = SkaterState {
                world_pos: lerp3(ca.world_pos, cb.world_pos, t),
                world_matrix: matrix,
                bone_count: bones as u8,
                bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
                bone_trans: [[0f32; 4]; MAX_BONES],
            };
            for i in 0..bones {
                let qa = ca.bone_quats[i];
                let qb = cb.bone_quats[i];
                let q = slerp(
                    &glm::Quat::new(qa[3], qa[0], qa[1], qa[2]),
                    &glm::Quat::new(qb[3], qb[0], qb[1], qb[2]),
                    t,
                );
                s.bone_quats[i] = [q.coords[0], q.coords[1], q.coords[2], q.coords[3]];
                s.bone_trans[i] = std::array::from_fn(|k| {
                    ca.bone_trans[i][k] + (cb.bone_trans[i][k] - ca.bone_trans[i][k]) * t
                });
            }
            s
        })
        .collect()
}

/// Called from `apply_frame` in the freecam: hands the recorded cars to the commit hook.
fn play_cars(frame: &ReplayState) {
    let cars: Vec<CarFrame> = car_entries(frame)
        .map(|(key, s)| CarFrame {
            key,
            pos: s.world_pos,
            rows: rotation_rows(&{
                let mut m = s.world_matrix;
                m[3] = 0.0;
                m[7] = 0.0;
                m[11] = 0.0;
                m
            }),
            bones: (0..s.bones()).map(|b| car_bone_matrix(s, b)).collect(),
        })
        .collect();
    let live: Vec<(usize, u32, Kind)> = car_objects(true)
        .into_iter()
        .filter_map(|o| car_pose_buffer(o).map(|(a, _)| (a, car_key(o), kind_of(o))))
        .collect();
    if CAR_LOGS.fetch_add(1, Ordering::Relaxed) % 600 == 0 {
        let count = |kind: Kind| {
            let l: Vec<_> = live.iter().filter(|x| x.2 == kind).collect();
            (
                l.len(),
                l.iter()
                    .filter(|(_, k, _)| cars.iter().any(|c| c.key == *k))
                    .count(),
            )
        };
        let ((c, cm), (p, pm), (o, om)) = (
            count(Kind::Car),
            count(Kind::Pedestrian),
            count(Kind::Other),
        );
        log::info!(
            "[THAW] traffic replay: {} recorded in this frame; cars: {c} live, {cm} recorded; pedestrians: {p} live, {pm} recorded; other: {o} live, {om} recorded",
            cars.len()
        );
    }
    *CAR_BUFFERS.lock().unwrap_or_else(|e| e.into_inner()) = live;
    *PLAYED_CARS.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), cars));
}

/// In the commit hook: draws a live car as its recorded self (or hides it).
fn replace_car(world: u32, src: usize, count: u32) {
    let (key, kind) = {
        let Ok(bufs) = CAR_BUFFERS.try_lock() else {
            return;
        };
        let Some((_, key, kind)) = bufs.iter().find(|(a, _, _)| *a == src) else {
            return;
        };
        (*key, *kind)
    };
    let Ok(played) = PLAYED_CARS.try_lock() else {
        return;
    };
    let Some((stamp, cars)) = played.as_ref() else {
        return;
    };
    // Recordings made before cars were recorded leave the traffic alone.
    if stamp.elapsed().as_millis() > 150 || cars.is_empty() {
        return;
    }
    let given = floats(&read_block(world as usize, 64));
    if given.len() != 16 {
        return;
    }
    match cars.iter().find(|c| c.key == key) {
        Some(car) => {
            let n = (count as usize).min(car.bones.len());
            write_matrices(src, &car.bones, n);
            let mut m = [0f32; 16];
            for r in 0..3 {
                for c in 0..3 {
                    m[r * 4 + c] = car.rows[r][c];
                }
            }
            let world_m = scaled(&world_matrix_of(&car.pos, &m), row_lengths(&given));
            write_matrices(world as usize, &[world_m], 1);
        }
        // Only cars are hidden: pedestrians and other objects that were not recorded (too
        // far away then) keep moving as they are.
        None if kind == Kind::Car => {
            write::<f32>(world as usize + 13 * 4, given[13] - HIDE_DEPTH);
        }
        None => {}
    }
}

// Moving objects without a skeleton (experimental, off unless switched on in the Replay
// window: game::SYNC_MOVING_OBJECTS).
//
// Cranes, crates, the train: plain objects with a model and a motion component (the game
// moves them along a path) but no skeleton, so they are not drawn through the commit
// function. The model instance (*(model component + 0x14)) holds the transform they are
// drawn with in blocks 0x500 apart: position (w = 1) at +0x3e8, rows at +0x408.., and
// more copies the game decides with whether to draw them (culling) are spread over the
// instance and what it points to. While recording, the nearest MAX_MOVERS objects seen
// moving are recorded: their transform, and every value found to move with them (raw, as
// an extra entry). In the freecam all of it is written back every frame.

/// Moving objects recorded per frame (the nearest to the skater).
const MAX_MOVERS: usize = 16;
const MOVER_MARK: f32 = 4243.0;
const MODEL_INSTANCE: usize = 0x14;
/// A model object counts as moving above this many units per frame.
const MOVING_STEP: f32 = 2.0;

/// Model objects seen moving while recording (object, last position).
static MOVERS: Mutex<Vec<(usize, [f32; 3], bool)>> = Mutex::new(Vec::new());
static MOVER_LOGS: AtomicU32 = AtomicU32::new(0);

/// The model's own transform copy, if its layout matches the object right now.
fn model_transform(obj: usize) -> Option<usize> {
    let inst = read::<usize>(component_of(obj, CRC_MODEL)? + MODEL_INSTANCE)?;
    if !addr_ok::<u32>(inst) {
        return None;
    }
    Some(inst)
}

fn capture_movers() -> Vec<SkaterState> {
    let mut movers = MOVERS.lock().unwrap_or_else(|e| e.into_inner());
    // Keep track of every model object without a skeleton.
    let mut live: Vec<usize> = Vec::new();
    for_each_object(|obj| {
        if live.len() >= 1024 || live.contains(&obj) {
            return;
        }
        if read::<u32>(obj + OBJ_TYPE) == Some(TYPE_SKATER) {
            return;
        }
        // Moved by the game along a path (motion component), drawn without a skeleton.
        if component_of(obj, CRC_MODEL).is_some()
            && component_of(obj, CRC_MOTION).is_some()
            && component_of(obj, CRC_SKELETON_COMPONENT).is_none()
        {
            live.push(obj);
        }
    });
    movers.retain(|(o, _, _)| live.contains(o));
    let mut out = Vec::new();
    let me = skater_object()
        .and_then(|o| read3(o + OBJ_POSITION))
        .unwrap_or([0.0; 3]);
    let mut recorded = 0;
    let mut by_distance: Vec<(f32, usize)> = live
        .iter()
        .map(|o| {
            let d = read3(o + OBJ_POSITION).map_or(f32::INFINITY, |p| {
                let d: [f32; 3] = std::array::from_fn(|i| p[i] - me[i]);
                dot3(d, d)
            });
            (d, *o)
        })
        .collect();
    by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, obj) in by_distance {
        let pos: [f32; 3] =
            std::array::from_fn(|i| read::<f32>(obj + OBJ_POSITION + i * 4).unwrap_or(f32::NAN));
        if !pos.iter().all(|v| v.is_finite()) {
            continue;
        }
        let moving = match movers.iter_mut().find(|(o, _, _)| *o == obj) {
            Some(m) => {
                let d: [f32; 3] = std::array::from_fn(|i| pos[i] - m.1[i]);
                if dot3(d, d).sqrt() > MOVING_STEP && !m.2 {
                    m.2 = true;
                    log::info!(
                        "[THAW] moving object 0x{obj:x} (type 0x{:08x}) is recorded",
                        read::<u32>(obj + OBJ_TYPE).unwrap_or(0)
                    );
                }
                m.1 = pos;
                m.2
            }
            None => {
                movers.push((obj, pos, false));
                false
            }
        };
        if !moving || recorded >= MAX_MOVERS {
            continue;
        }
        recorded += 1;
        let mut matrix: [f32; 16] =
            std::array::from_fn(|i| read::<f32>(obj + OBJ_MATRIX + i * 4).unwrap_or(f32::NAN));
        if !matrix.iter().all(|v| v.is_finite()) {
            continue;
        }
        orthonormalize(&mut matrix);
        let (hi, lo) = key_floats(car_key(obj));
        matrix[3] = MOVER_MARK;
        matrix[7] = hi;
        matrix[11] = lo;
        out.push(SkaterState {
            world_pos: pos,
            world_matrix: matrix,
            // One identity bone, so the entry is never taken for a skater part.
            bone_count: 1,
            bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
            bone_trans: [[0f32; 4]; MAX_BONES],
        });
        if let Some(extra) = capture_slots(obj, car_key(obj)) {
            out.push(extra);
        }
    }
    out
}

fn mover_entries(f: &ReplayState) -> impl Iterator<Item = (u32, &SkaterState)> {
    f.skaters
        .iter()
        .skip(3)
        .filter(|s| s.world_matrix[3] == MOVER_MARK)
        .map(|s| {
            (
                ((s.world_matrix[7] as u32) << 16) | (s.world_matrix[11] as u32 & 0xffff),
                s,
            )
        })
}

fn blend_movers(a: &ReplayState, b: &ReplayState, t: f32) -> Vec<SkaterState> {
    let later: Vec<(u32, &SkaterState)> = mover_entries(b).collect();
    mover_entries(a)
        .map(|(key, ta)| {
            let Some((_, tb)) = later.iter().find(|(k, _)| *k == key) else {
                return ta.clone();
            };
            // The train loops by jumping back to the start: shown as a cut, not a flight.
            let d: [f32; 3] = std::array::from_fn(|i| ta.world_pos[i] - tb.world_pos[i]);
            if dot3(d, d).sqrt() > TELEPORT {
                return if t < 0.5 { ta.clone() } else { (*tb).clone() };
            }
            let (_, mut matrix) = blend_world(&ta.world_matrix, &tb.world_matrix, t);
            matrix[3] = MOVER_MARK;
            matrix[7] = ta.world_matrix[7];
            matrix[11] = ta.world_matrix[11];
            let mut s = ta.clone();
            s.world_pos = lerp3(ta.world_pos, tb.world_pos, t);
            s.world_matrix = matrix;
            s
        })
        .collect()
}

/// The model instance holds the drawn transform in blocks 0x500 apart (most carts use the
/// first, some the second): position (w = 1) at +0x0, rotation rows at +0x20/+0x30/+0x40,
/// and a second copy of the rows at +0x74/+0x84/+0x94.
/// How far behind its position a block's rows start: 0x20 for the train, 0x10 for cranes.
const BLOCK_ROW_OFFSETS: [usize; 2] = [0x20, 0x10];
/// A second copy of the rows, behind 0x20-blocks.
const BLOCK_ROWS2: usize = 0x74;
/// How much of the model instance is searched for blocks.
const INSTANCE_SIZE: usize = 0x1000;

/// One drawn transform block, stored relative to the object: block rows = local * object
/// rows, block position = object position + offset * object rows.
#[derive(Clone, Copy)]
struct Block {
    addr: usize,
    local: [[f32; 3]; 3],
    offset: [f32; 3],
    /// Where the rows start, from `addr`.
    rows_off: usize,
    second_copy: bool,
    /// Which parts this entry holds: extra copies have only a position or only rows.
    has_pos: bool,
    has_rows: bool,
}

/// row vector times rows: v0 * rows[0] + v1 * rows[1] + v2 * rows[2].
fn times_rows(v: [f32; 3], rows: &[[f32; 3]; 3]) -> [f32; 3] {
    std::array::from_fn(|c| v[0] * rows[0][c] + v[1] * rows[1][c] + v[2] * rows[2][c])
}

/// The object's drawn transform blocks, measured while the game still owns the train.
fn find_blocks(obj: usize) -> Vec<Block> {
    let Some(inst) = model_transform(obj) else {
        return Vec::new();
    };
    let Some(opos) = read3(obj + OBJ_POSITION) else {
        return Vec::new();
    };
    let mut m: [f32; 16] =
        std::array::from_fn(|i| read::<f32>(obj + OBJ_MATRIX + i * 4).unwrap_or(f32::NAN));
    if !m.iter().all(|v| v.is_finite()) {
        return Vec::new();
    }
    orthonormalize(&mut m);
    let orows = rotation_rows(&m);
    // Transposed object rows: v * object rows^T gives v in the object's frame.
    let ot: [[f32; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| orows[c][r]));
    let mut out: Vec<Block> = Vec::new();
    let f = floats(&read_block(inst, INSTANCE_SIZE));
    let near = |a: [f32; 3], b: [f32; 3], tol: f32| (0..3).all(|i| (a[i] - b[i]).abs() < tol);
    let mut k = 0;
    while k + 4 <= f.len() {
        // A position with w = 1 at the object's position...
        let pos = [f[k], f[k + 1], f[k + 2]];
        let d: [f32; 3] = std::array::from_fn(|i| pos[i] - opos[i]);
        if (f[k + 3] - 1.0).abs() > 1e-3
            || !pos.iter().all(|x| x.is_finite())
            || dot3(d, d).sqrt() > 60.0
        {
            k += 1;
            continue;
        }
        // ...followed by the object's rotation.
        let found = BLOCK_ROW_OFFSETS.iter().find_map(|off| {
            let r0 = k + off / 4;
            if r0 + 11 > f.len() {
                return None;
            }
            let rows: [[f32; 3]; 3] =
                std::array::from_fn(|r| [f[r0 + r * 4], f[r0 + r * 4 + 1], f[r0 + r * 4 + 2]]);
            (0..3)
                .all(|r| near(rows[r], orows[r], 0.1))
                .then_some((*off, rows))
        });
        let Some((rows_off, rows)) = found else {
            k += 1;
            continue;
        };
        let addr = inst + k * 4;
        let local: [[f32; 3]; 3] = std::array::from_fn(|r| times_rows(rows[r], &ot));
        let offset = times_rows(d, &ot);
        // The second copy only when it agrees with the first right now.
        let second_copy = rows_off == 0x20
            && (0..3).all(|r| {
                read3(addr + BLOCK_ROWS2 + r * 16).is_some_and(|v| near(v, rows[r], 0.05))
            });
        log::info!(
            "[THAW]   moving object 0x{obj:x}: block at instance+0x{:x}, rows at +0x{rows_off:x}, offset {:?}, second copy {second_copy}",
            addr - inst,
            offset.map(|x| (x * 10.0).round() / 10.0)
        );
        out.push(Block {
            addr,
            local,
            offset,
            rows_off,
            second_copy,
            has_pos: true,
            has_rows: true,
        });
        if out.len() >= 8 {
            break;
        }
        k += (rows_off + 0x30) / 4;
    }
    out
}

/// Live train objects with their recorded transform and drawn blocks.
type MoverTarget = (usize, Vec<Block>, [f32; 3], [[f32; 3]; 3]);
/// Raw values to put back: (address, value).
static MOVER_RAW: Mutex<Vec<(usize, [f32; 3])>> = Mutex::new(Vec::new());
static MOVER_TARGETS: Mutex<Option<(Instant, Vec<MoverTarget>)>> = Mutex::new(None);
/// Each live object's blocks, measured before the first write.
/// (object, model instance last seen, blocks relative to the instance).
static MOVER_BLOCKS: Mutex<Vec<(usize, usize, Vec<Block>)>> = Mutex::new(Vec::new());

const CRC_MOTION: u32 = 0x0a015e17;
fn play_movers(frame: &ReplayState) {
    let trains: Vec<(u32, [f32; 3], [[f32; 3]; 3])> = mover_entries(frame)
        .map(|(key, s)| {
            let mut m = s.world_matrix;
            m[3] = 0.0;
            m[7] = 0.0;
            m[11] = 0.0;
            (key, s.world_pos, rotation_rows(&m))
        })
        .collect();
    if trains.is_empty() {
        return;
    }
    // The recorded moving values, for trains whose slots are known in this session.
    let mut raw: Vec<(usize, [f32; 3])> = Vec::new();
    {
        let slots = MOVER_SLOTS.lock().unwrap_or_else(|e| e.into_inner());
        for (key, e) in extra_entries(frame) {
            let Some((obj, list)) = slots.iter().find(|(o, _)| car_key(*o) == key) else {
                continue;
            };
            if read::<u32>(obj + OBJ_TYPE).is_none() || list.len() != e.bones() {
                continue;
            }
            for (i, (base, off, kind)) in list.iter().enumerate() {
                // Values that jump on their own are still played back: without them the
                // game does not draw the train.
                let _ = slot_is_bad(*obj, i);
                let Some(addr) = resolve(*obj, *base).map(|a| a + off) else {
                    continue;
                };
                // Only where the memory still holds that kind of value.
                let Some(now) = read3(addr) else { continue };
                let fits = if *kind == 0 {
                    now.iter().all(|x| x.abs() < 1e6)
                } else {
                    (0.9..1.1).contains(&dot3(now, now).sqrt())
                };
                if fits {
                    let v = e.bone_trans[i];
                    raw.push((addr, [v[0], v[1], v[2]]));
                }
            }
        }
    }
    *MOVER_RAW.lock().unwrap_or_else(|e| e.into_inner()) = raw;
    let mut known = MOVER_BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
    let mut targets: Vec<MoverTarget> = Vec::new();
    for_each_object(|obj| {
        if targets.iter().any(|t| t.0 == obj) {
            return;
        }
        let Some((_, pos, rows)) = trains.iter().find(|(k, _, _)| *k == car_key(obj)) else {
            return;
        };
        // Blocks are kept relative to the model instance and placed on the current one every
        // frame: the game may give a train a new instance (when it respawns at the start of
        // its track), and the old one must not be written to any more.
        let Some(inst) = model_transform(obj) else {
            return;
        };
        let rel = match known.iter_mut().find(|(o, _, _)| *o == obj) {
            Some((_, last, b)) => {
                if *last != inst {
                    log::info!(
                        "[THAW] moving object 0x{obj:x}: new model instance 0x{inst:x} (was 0x{last:x})"
                    );
                    *last = inst;
                }
                b.clone()
            }
            None => {
                let b: Vec<Block> = find_blocks(obj)
                    .into_iter()
                    .map(|mut b| {
                        b.addr -= inst;
                        b
                    })
                    .collect();
                log::info!(
                    "[THAW] moving object 0x{obj:x}: {} drawn transform block(s){}",
                    b.len(),
                    if b.is_empty() {
                        ", only the object is set"
                    } else {
                        ""
                    }
                );
                known.push((obj, inst, b.clone()));
                b
            }
        };
        // Only where the layout still holds (a position with w = 1).
        let blocks: Vec<Block> = rel
            .into_iter()
            .map(|mut b| {
                b.addr += inst;
                b
            })
            .filter(|b| {
                !b.has_pos || read::<f32>(b.addr + 12).is_some_and(|w| (w - 1.0).abs() < 1e-3)
            })
            .collect();
        // The object itself is always set (the game builds what it draws from it on some
        // objects); the blocks, where found, are set directly as well.
        targets.push((obj, blocks, *pos, *rows));
    });
    if MOVER_LOGS.fetch_add(1, Ordering::Relaxed) % 600 == 0 {
        log::info!(
            "[THAW] moving objects replay: {} recorded, {} found, {} moving values",
            trains.len(),
            targets.len(),
            MOVER_RAW.lock().map(|r| r.len()).unwrap_or(0)
        );
    }
    let mut slot = MOVER_TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    *slot = Some((Instant::now(), targets));
    drop(slot);
    drop(known);
    place_movers();
}

/// Puts the recorded trains where they belong (object and drawn blocks).
fn place_movers() {
    let Ok(guard) = MOVER_TARGETS.try_lock() else {
        return;
    };
    let Some((stamp, targets)) = guard.as_ref() else {
        return;
    };
    if stamp.elapsed().as_millis() > 150 {
        return;
    }
    for (obj, blocks, pos, rows) in targets {
        for i in 0..3 {
            write::<f32>(obj + OBJ_POSITION + i * 4, pos[i]);
            for r in 0..3 {
                write::<f32>(obj + OBJ_MATRIX + r * 16 + i * 4, rows[r][i]);
            }
        }
        for b in blocks.iter() {
            let off = times_rows(b.offset, rows);
            let drawn: [[f32; 3]; 3] = std::array::from_fn(|r| times_rows(b.local[r], rows));
            for i in 0..3 {
                if b.has_pos {
                    write::<f32>(b.addr + i * 4, pos[i] + off[i]);
                }
                for r in 0..3 {
                    if !b.has_rows {
                        break;
                    }
                    write::<f32>(b.addr + b.rows_off + r * 16 + i * 4, drawn[r][i]);
                    if b.second_copy {
                        write::<f32>(b.addr + BLOCK_ROWS2 + r * 16 + i * 4, drawn[r][i]);
                    }
                }
            }
        }
    }
    if let Ok(raw) = MOVER_RAW.try_lock() {
        for (addr, v) in raw.iter() {
            for i in 0..3 {
                write::<f32>(addr + i * 4, v[i]);
            }
        }
    }
}

/// Where a memory region of a train cart is found: the model component, the model instance,
/// or a pointer stored in one of them (resolved again before use).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Base {
    Comp,
    Inst,
    CompPtr(usize),
    InstPtr(usize),
}

fn resolve(obj: usize, base: Base) -> Option<usize> {
    let comp = component_of(obj, CRC_MODEL)?;
    let addr = match base {
        Base::Comp => comp,
        Base::Inst => model_transform(obj)?,
        Base::CompPtr(w) => read::<usize>(comp + w)?,
        Base::InstPtr(w) => read::<usize>(model_transform(obj)? + w)?,
    };
    addr_ok::<u32>(addr).then_some(addr)
}

fn base_name(base: Base) -> String {
    match base {
        Base::Comp => "component".into(),
        Base::Inst => "instance".into(),
        Base::CompPtr(w) => format!("[component+0x{w:x}]"),
        Base::InstPtr(w) => format!("[instance+0x{w:x}]"),
    }
}

/// The regions searched for moving copies: component, instance and what they point to.
fn mover_regions(obj: usize) -> Vec<(Base, usize, Vec<u8>)> {
    let (Some(comp), Some(inst)) = (component_of(obj, CRC_MODEL), model_transform(obj)) else {
        return Vec::new();
    };
    let mut out = vec![
        (Base::Comp, comp, read_block(comp, 0x200)),
        (Base::Inst, inst, read_block(inst, 0x1000)),
    ];
    for (is_inst, top) in [
        (false, read_block(comp, 0x200)),
        (true, read_block(inst, 0x1000)),
    ] {
        for w in (0..top.len() / 4).map(|i| i * 4) {
            let v = u32::from_le_bytes([top[w], top[w + 1], top[w + 2], top[w + 3]]) as usize;
            // Only memory next to the instance (the game heap), never driver buffers.
            if v >> 28 != inst >> 28 || v == obj || out.len() >= 96 || out.iter().any(|r| r.1 == v)
            {
                continue;
            }
            let bytes = read_block(v, 0x400);
            if !bytes.is_empty() {
                out.push((
                    if is_inst {
                        Base::InstPtr(w)
                    } else {
                        Base::CompPtr(w)
                    },
                    v,
                    bytes,
                ));
            }
        }
    }
    out
}

/// Values that move with a train (the trailing carts, culling boxes, lights...): positions
/// (kind 0) and rotations (three rows 16 bytes apart, kinds 1..3), as (region, byte offset,
/// kind). They are recorded as they are and written back as they were.
type Slot = (Base, usize, u8);
/// Negative, so the replay file keeps these values at full precision (they are world
/// positions).
const MOVER_EXTRA_MARK: f32 = -4244.0;
static SNAPSHOTS: Mutex<Vec<(usize, [f32; 3], Vec<(Base, usize, Vec<u8>)>, bool)>> =
    Mutex::new(Vec::new());
static MOVER_SLOTS: Mutex<Vec<(usize, Vec<Slot>)>> = Mutex::new(Vec::new());

fn is_rotation(f: &[f32], k: usize) -> bool {
    if k + 11 > f.len() {
        return false;
    }
    let row = |r: usize| [f[k + r * 4], f[k + r * 4 + 1], f[k + r * 4 + 2]];
    let rows = [row(0), row(1), row(2)];
    let unit = rows
        .iter()
        .all(|r| r.iter().all(|x| x.is_finite()) && (dot3(*r, *r).sqrt() - 1.0).abs() < 0.02);
    let square = dot3(rows[0], rows[1]).abs() < 0.02
        && dot3(rows[0], rows[2]).abs() < 0.02
        && dot3(rows[1], rows[2]).abs() < 0.02;
    // Not an identity-like matrix that would match anything.
    let nontrivial = rows
        .iter()
        .flatten()
        .filter(|x| x.abs() > 0.05 && x.abs() < 0.95)
        .count()
        >= 2;
    unit && square && nontrivial
}

/// Compares a snapshot of the train's memory with now and lists what moved with it.
fn find_moving_slots(
    obj: usize,
    pos0: [f32; 3],
    before: &[(Base, usize, Vec<u8>)],
) -> Option<Vec<Slot>> {
    let pos = read3(obj + OBJ_POSITION)?;
    let dp: [f32; 3] = std::array::from_fn(|i| pos[i] - pos0[i]);
    if dot3(dp, dp).sqrt() < 100.0 {
        return None;
    }
    let mut m: [f32; 16] =
        std::array::from_fn(|i| read::<f32>(obj + OBJ_MATRIX + i * 4).unwrap_or(f32::NAN));
    orthonormalize(&mut m);
    let orows = rotation_rows(&m);
    let mut out: Vec<Slot> = Vec::new();
    let mut triples = 0;
    for (base, addr, bytes) in before {
        if resolve(obj, *base) != Some(*addr) {
            continue;
        }
        let (f1, f2) = (floats(bytes), floats(&read_block(*addr, bytes.len())));
        let n = f1.len().min(f2.len());
        let mut positions: Vec<usize> = Vec::new();
        let mut k = 0;
        while k + 3 <= n && triples < MAX_BONES {
            let v: [f32; 3] = [f2[k], f2[k + 1], f2[k + 2]];
            let d: [f32; 3] = std::array::from_fn(|i| v[i] - pos[i]);
            let m: [f32; 3] = std::array::from_fn(|i| f2[k + i] - f1[k + i]);
            let near = v.iter().all(|x| x.is_finite()) && dot3(d, d).sqrt() < 3000.0;
            let moved =
                m.iter().all(|x| x.is_finite()) && (50.0..20000.0).contains(&dot3(m, m).sqrt());
            if near && moved {
                out.push((*base, k * 4, 0));
                positions.push(k);
                triples += 1;
                k += 3;
            } else {
                k += 1;
            }
        }
        // Rotations next to a moving position, or rotations that turned.
        let mut k = 0;
        while k + 11 <= n && triples + 3 <= MAX_BONES {
            let taken = |j: usize| positions.iter().any(|p| j + 3 > *p && j < *p + 3);
            let overlaps = (0..3).any(|r| taken(k + r * 4))
                || out
                    .iter()
                    .any(|(b, o, kind)| b == base && *kind > 0 && (o / 4).abs_diff(k) < 11);
            let turned = (0..11).any(|i| (f2[k + i] - f1[k + i]).abs() > 1e-3);
            let beside = positions.iter().any(|p| p.abs_diff(k) <= 16);
            // The train's own rotation (the carts behind it on a straight piece of track).
            let same =
                (0..3).all(|r| (0..3).all(|c| (f2[k + r * 4 + c] - orows[r][c]).abs() < 0.02));
            if !overlaps && is_rotation(&f2, k) && (turned || beside || same) {
                for r in 0..3 {
                    out.push((*base, (k + r * 4) * 4, r as u8 + 1));
                }
                triples += 3;
                k += 11;
            } else {
                k += 1;
            }
        }
    }
    Some(out)
}

/// Per train: the slots' values in the last recorded frame and which slots jumped while the
/// train did not (track waypoints and other state of the game's train mover, never drawn:
/// writing those back makes the live train jump to its respawn point).
static SLOT_HISTORY: Mutex<Vec<(usize, [f32; 3], Vec<[f32; 3]>, Vec<bool>)>> =
    Mutex::new(Vec::new());

fn slot_is_bad(obj: usize, i: usize) -> bool {
    SLOT_HISTORY
        .try_lock()
        .ok()
        .and_then(|h| {
            h.iter()
                .find(|e| e.0 == obj)
                .map(|e| e.3.get(i).copied().unwrap_or(false))
        })
        .unwrap_or(false)
}

/// The train's extra entry: every slot's current value (kind in the fourth float).
fn capture_slots(obj: usize, key: u32) -> Option<SkaterState> {
    let all = MOVER_SLOTS.try_lock().ok()?;
    let (_, slots) = all.iter().find(|(o, _)| *o == obj)?;
    if slots.is_empty() {
        return None;
    }
    let mut trans = [[0f32; 4]; MAX_BONES];
    let mut values: Vec<[f32; 3]> = Vec::new();
    for (i, (base, off, kind)) in slots.iter().enumerate().take(MAX_BONES) {
        let v = read3(resolve(obj, *base)? + off)?;
        trans[i] = [v[0], v[1], v[2], *kind as f32];
        values.push(v);
    }
    if let (Some(pos), Ok(mut history)) = (read3(obj + OBJ_POSITION), SLOT_HISTORY.try_lock()) {
        match history.iter_mut().find(|e| e.0 == obj) {
            Some(e) if e.2.len() == values.len() => {
                let d: [f32; 3] = std::array::from_fn(|c| pos[c] - e.1[c]);
                if dot3(d, d).sqrt() < 100.0 {
                    for (i, (v, last)) in values.iter().zip(&e.2).enumerate() {
                        let m: [f32; 3] = std::array::from_fn(|c| v[c] - last[c]);
                        let limit = if slots[i].2 == 0 { 500.0 } else { 0.5 };
                        if dot3(m, m).sqrt() > limit && !e.3[i] {
                            // A rotation is one unit: all three rows go.
                            let group = if slots[i].2 == 0 {
                                i..i + 1
                            } else {
                                i + 1 - slots[i].2 as usize..i + 4 - slots[i].2 as usize
                            };
                            for j in group {
                                if let Some(b) = e.3.get_mut(j) {
                                    *b = true;
                                }
                            }
                            log::info!(
                                "[THAW] moving object 0x{obj:x}: value {i} jumps on its own"
                            );
                        }
                    }
                }
                e.1 = pos;
                e.2 = values;
            }
            Some(e) => {
                *e = (obj, pos, values.clone(), vec![false; values.len()]);
            }
            None => history.push((obj, pos, values.clone(), vec![false; values.len()])),
        }
    }
    let (hi, lo) = key_floats(key);
    let mut matrix = [0f32; 16];
    matrix[3] = MOVER_EXTRA_MARK;
    matrix[7] = hi;
    matrix[11] = lo;
    Some(SkaterState {
        world_pos: [0.0; 3],
        world_matrix: matrix,
        bone_count: slots.len().min(MAX_BONES) as u8,
        bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
        bone_trans: trans,
    })
}

fn extra_entries(f: &ReplayState) -> impl Iterator<Item = (u32, &SkaterState)> {
    f.skaters
        .iter()
        .skip(3)
        .filter(|s| s.world_matrix[3] == MOVER_EXTRA_MARK)
        .map(|s| {
            (
                ((s.world_matrix[7] as u32) << 16) | (s.world_matrix[11] as u32 & 0xffff),
                s,
            )
        })
}

/// Makes the rotations (kinds 1, 2, 3 in a row) of an extra entry square again.
fn square_rotations(s: &mut SkaterState) {
    let n = s.bones();
    let mut i = 0;
    while i + 3 <= n {
        if s.bone_trans[i][3] == 1.0
            && s.bone_trans[i + 1][3] == 2.0
            && s.bone_trans[i + 2][3] == 3.0
        {
            let mut m = [0f32; 16];
            for r in 0..3 {
                for c in 0..3 {
                    m[r * 4 + c] = s.bone_trans[i + r][c];
                }
            }
            m[15] = 1.0;
            orthonormalize(&mut m);
            for r in 0..3 {
                for c in 0..3 {
                    s.bone_trans[i + r][c] = m[r * 4 + c];
                }
            }
            i += 3;
        } else {
            i += 1;
        }
    }
}

/// The trains, averaged over five recorded frames around the played time. The game moves
/// them in steps that do not match the recorded frames now and then (a short step, then a
/// double one), which shows as a jerk, mostly in corners; the average evens that out.
fn smooth_movers(frames: &[ReplayState], i0: usize, t: f32) -> Vec<SkaterState> {
    const WEIGHTS: [(isize, f32); 5] = [(-2, 1.0), (-1, 2.0), (0, 3.0), (1, 2.0), (2, 1.0)];
    let sample = |k: isize| -> Option<Vec<SkaterState>> {
        let j = i0 as isize + k;
        if j < 0 || j as usize + 1 >= frames.len() {
            return None;
        }
        let (a, b) = (&frames[j as usize], &frames[j as usize + 1]);
        let mut v = blend_movers(a, b, t);
        v.extend(blend_extras(a, b, t));
        Some(v)
    };
    let Some(center) = sample(0) else {
        // The last frame: nothing to blend with.
        return frames
            .get(i0)
            .map(|f| {
                f.skaters
                    .iter()
                    .skip(3)
                    .filter(|s| {
                        s.world_matrix[3] == MOVER_MARK || s.world_matrix[3] == MOVER_EXTRA_MARK
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
    };
    let others: Vec<(f32, Vec<SkaterState>)> = WEIGHTS
        .iter()
        .filter(|(k, _)| *k != 0)
        .filter_map(|(k, w)| sample(*k).map(|v| (*w, v)))
        .collect();
    let same = |a: &SkaterState, b: &SkaterState| {
        a.world_matrix[3] == b.world_matrix[3]
            && a.world_matrix[7] == b.world_matrix[7]
            && a.world_matrix[11] == b.world_matrix[11]
    };
    center
        .into_iter()
        .map(|c| {
            let mut s = c.clone();
            let mut total = 3.0;
            if c.world_matrix[3] == MOVER_MARK {
                let mut pos = c.world_pos.map(|x| x * 3.0);
                let mut m: [f32; 16] = std::array::from_fn(|i| c.world_matrix[i] * 3.0);
                for (w, v) in &others {
                    let Some(o) = v.iter().find(|o| same(o, &c)) else {
                        continue;
                    };
                    let d: [f32; 3] = std::array::from_fn(|i| o.world_pos[i] - c.world_pos[i]);
                    // Never across a jump (the train looping), and only neighbours.
                    if dot3(d, d).sqrt() > 200.0 {
                        continue;
                    }
                    for i in 0..3 {
                        pos[i] += o.world_pos[i] * w;
                    }
                    for (i, x) in m.iter_mut().enumerate() {
                        *x += o.world_matrix[i] * w;
                    }
                    total += w;
                }
                s.world_pos = pos.map(|x| x / total);
                let mut rows: [f32; 16] = m.map(|x| x / total);
                orthonormalize(&mut rows);
                for r in 0..3 {
                    for col in 0..3 {
                        s.world_matrix[r * 4 + col] = rows[r * 4 + col];
                    }
                }
            } else {
                let n = c.bones();
                let mut acc: Vec<[f32; 3]> = (0..n)
                    .map(|i| {
                        [
                            c.bone_trans[i][0] * 3.0,
                            c.bone_trans[i][1] * 3.0,
                            c.bone_trans[i][2] * 3.0,
                        ]
                    })
                    .collect();
                let mut weight = vec![3.0f32; n];
                for (w, v) in &others {
                    let Some(o) = v.iter().find(|o| same(o, &c)).filter(|o| o.bones() == n) else {
                        continue;
                    };
                    for i in 0..n {
                        let d: [f32; 3] =
                            std::array::from_fn(|k| o.bone_trans[i][k] - c.bone_trans[i][k]);
                        // Values that jump on their own stay as they are.
                        let limit = if c.bone_trans[i][3] == 0.0 {
                            200.0
                        } else {
                            0.5
                        };
                        if dot3(d, d).sqrt() > limit {
                            continue;
                        }
                        for k in 0..3 {
                            acc[i][k] += o.bone_trans[i][k] * w;
                        }
                        weight[i] += w;
                    }
                }
                for i in 0..n {
                    for k in 0..3 {
                        s.bone_trans[i][k] = acc[i][k] / weight[i];
                    }
                }
                square_rotations(&mut s);
            }
            let _ = total;
            s
        })
        .collect()
}

/// Positions are blended straight, rotations row by row and then made square again; a jump
/// (the train looping) is cut.
fn blend_extras(a: &ReplayState, b: &ReplayState, t: f32) -> Vec<SkaterState> {
    let later: Vec<(u32, &SkaterState)> = extra_entries(b).collect();
    extra_entries(a)
        .map(|(key, ea)| {
            let Some((_, eb)) = later.iter().find(|(k, _)| *k == key) else {
                return ea.clone();
            };
            let n = ea.bones().min(eb.bones());
            if ea.bones() != eb.bones() {
                return if t < 0.5 { ea.clone() } else { (*eb).clone() };
            }
            let mut s = ea.clone();
            for i in 0..n {
                // A value that jumps (the train looping) is cut, on its own.
                let d: [f32; 3] =
                    std::array::from_fn(|c| ea.bone_trans[i][c] - eb.bone_trans[i][c]);
                if ea.bone_trans[i][3] == 0.0 && dot3(d, d).sqrt() > TELEPORT {
                    if t >= 0.5 {
                        s.bone_trans[i] = eb.bone_trans[i];
                    }
                    continue;
                }
                for c in 0..3 {
                    s.bone_trans[i][c] =
                        ea.bone_trans[i][c] + (eb.bone_trans[i][c] - ea.bone_trans[i][c]) * t;
                }
            }
            // Square up each rotation (kinds 1, 2, 3 in a row).
            let mut i = 0;
            while i + 3 <= n {
                if s.bone_trans[i][3] == 1.0
                    && s.bone_trans[i + 1][3] == 2.0
                    && s.bone_trans[i + 2][3] == 3.0
                {
                    let mut m = [0f32; 16];
                    for r in 0..3 {
                        for c in 0..3 {
                            m[r * 4 + c] = s.bone_trans[i + r][c];
                        }
                    }
                    m[15] = 1.0;
                    orthonormalize(&mut m);
                    for r in 0..3 {
                        for c in 0..3 {
                            s.bone_trans[i + r][c] = m[r * 4 + c];
                        }
                    }
                    i += 3;
                } else {
                    i += 1;
                }
            }
            s
        })
        .collect()
}

/// While recording: logs where each moving cart's drawn transform is and finds the copies
/// of its position that move with it.
fn check_mover_layout() {
    static CHECKED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let Ok(movers) = MOVERS.try_lock() else {
        return;
    };
    let Ok(mut checked) = CHECKED.try_lock() else {
        return;
    };
    let Ok(mut snaps) = SNAPSHOTS.try_lock() else {
        return;
    };
    for (obj, _, moving) in movers.iter() {
        if !moving {
            continue;
        }
        if !checked.contains(obj) && checked.len() < 64 {
            checked.push(*obj);
            let blocks = find_blocks(*obj);
            log::info!(
                "[THAW] moving object 0x{obj:x}: {} drawn transform block(s)",
                blocks.len()
            );
        }
        let room = snaps.len() < 64;
        match snaps.iter_mut().find(|s| s.0 == *obj) {
            None if room => {
                if let Some(p) = read3(obj + OBJ_POSITION) {
                    let regions = mover_regions(*obj);
                    snaps.push((*obj, p, regions, false));
                }
            }
            Some(s) if !s.3 => {
                if let Some(slots) = find_moving_slots(*obj, s.1, &s.2) {
                    s.3 = true;
                    s.2 = Vec::new();
                    log::info!(
                        "[THAW]   moving object 0x{obj:x}, values moving with it: {}",
                        slots
                            .iter()
                            .filter(|(_, _, k)| *k <= 1)
                            .map(|(b, o, k)| format!(
                                "{} {}+0x{o:x}",
                                if *k == 0 { "position" } else { "rotation" },
                                base_name(*b)
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    let mut all = MOVER_SLOTS.lock().unwrap_or_else(|e| e.into_inner());
                    all.retain(|(o, _)| o != obj);
                    all.push((*obj, slots));
                }
            }
            _ => {}
        }
    }
}

// Moving objects without a skeleton (survey).
//
// Objects that are drawn without a skeleton (platforms, doors, the train) cannot be replayed
// the way the others are. While recording, every few seconds for about a minute, falke.log
// lists those that moved, so it is known what a level has.

const SURVEY_EVERY_MS: u128 = 5000;
const SURVEY_REPORTS: u32 = 12;

struct Survey {
    last: Option<Instant>,
    reports: u32,
    prev: Vec<(usize, [f32; 3])>,
    seen: Vec<usize>,
}

static SURVEY: Mutex<Survey> = Mutex::new(Survey {
    last: None,
    reports: 0,
    prev: Vec::new(),
    seen: Vec::new(),
});

/// Names of the components seen on moving objects (CRC of the name).
fn component_name(crc: u32) -> String {
    let name = match crc {
        0x286a8d26 => "model",
        0x222756d5 => "skeleton",
        0x0a015e17 => "motion",
        0xce0ca665 => "suspend",
        0x07713c7b => "sound",
        0x0f1641e3 => "stream",
        0xa432dc41 => "collision",
        0x5cb5cb7e => "lockobj",
        0x1d5b1ae6 => "objectproximity",
        0xa6a3147e => "rail",
        0xe891ffcd => "pathfollow",
        0x41272956 => "velocity",
        0xffb39248 => "rigidbody",
        0x72ad7b23 => "animation",
        0xe37e78c5 => "script",
        0x8a897dd2 => "shadow",
        0xa9db601e => "particle",
        _ => return format!("{crc:08x}"),
    };
    name.to_string()
}

fn report_moving_objects() {
    let Ok(mut d) = SURVEY.try_lock() else {
        return;
    };
    if d.reports >= SURVEY_REPORTS
        || d.last
            .is_some_and(|t| t.elapsed().as_millis() < SURVEY_EVERY_MS)
    {
        return;
    }
    let first = d.last.is_none();
    d.last = Some(Instant::now());
    let mut now: Vec<(usize, [f32; 3])> = Vec::new();
    for_each_object(|obj| {
        if now.len() < 4000 && !now.iter().any(|(o, _)| *o == obj) {
            if let Some(p) = read3(obj + OBJ_POSITION) {
                now.push((obj, p));
            }
        }
    });
    if !first {
        d.reports += 1;
        let secs = SURVEY_EVERY_MS as f32 / 1000.0;
        let mut lines: Vec<String> = Vec::new();
        for (obj, p) in &now {
            let ty = read::<u32>(obj + OBJ_TYPE).unwrap_or(0);
            if ty == TYPE_SKATER || component_of(*obj, CRC_SKELETON_COMPONENT).is_some() {
                continue;
            }
            let Some((_, q)) = d.prev.iter().find(|(o, _)| o == obj) else {
                continue;
            };
            let dd: [f32; 3] = std::array::from_fn(|i| p[i] - q[i]);
            let moved = dot3(dd, dd).sqrt();
            if !(moved > 20.0 && moved < 1e5) || d.seen.contains(obj) || lines.len() >= 16 {
                continue;
            }
            d.seen.push(*obj);
            let comps: Vec<String> = match (
                read::<usize>(obj + OBJ_COMPONENT_TABLE),
                read::<u32>(obj + OBJ_COMPONENT_COUNT),
            ) {
                (Some(t), Some(n)) => (0..n.min(64) as usize)
                    .filter_map(|i| read::<u32>(t + i * 8))
                    .map(component_name)
                    .collect(),
                _ => Vec::new(),
            };
            lines.push(format!(
                "object 0x{obj:x} type 0x{ty:08x} at {:.0} {:.0} {:.0}, {:.0} units/s, components: {}",
                p[0],
                p[1],
                p[2],
                moved / secs,
                comps.join(" ")
            ));
        }
        if !lines.is_empty() {
            log::info!(
                "[THAW] moving objects without a skeleton (new ones, report {}):",
                d.reports
            );
            for l in lines {
                log::info!("[THAW]   {l}");
            }
        }
    }
    d.prev = now;
}

// Skater lookup.

static SKATER_OBJ: AtomicUsize = AtomicUsize::new(0);
static LAST_SCAN: Mutex<Option<Instant>> = Mutex::new(None);

/// Calls `f(object)` for every entry of the manager's two name tables.
fn for_each_object(mut f: impl FnMut(usize)) {
    let Some(manager) = read::<usize>(ADDR_OBJECT_MANAGER_PTR).filter(|m| *m != 0) else {
        return;
    };
    for table_offset in MANAGER_TABLE_OFFSETS {
        let Some(table) = read::<usize>(manager + table_offset).filter(|t| *t != 0) else {
            continue;
        };
        let Some(header) = read::<u32>(table) else {
            continue;
        };
        let shift = header & 0x1f;
        if shift > MAX_TABLE_SHIFT {
            continue;
        }
        let Some(buckets) = read::<usize>(table + TABLE_BUCKETS_OFFSET) else {
            continue;
        };
        for i in 0..(1usize << shift) {
            let mut node = buckets + i * BUCKET_SIZE;
            let mut hops = 0;
            while node != 0 && hops < MAX_CHAIN {
                let (Some(value), Some(next)) = (read::<usize>(node + 4), read::<usize>(node + 8))
                else {
                    break;
                };
                if value != 0 {
                    f(value);
                }
                node = next;
                hops += 1;
            }
        }
    }
}

/// The skater object: cached, checked on every call, and looked up again (at most twice a
/// second) when it is gone (level change, respawn).
fn skater_object() -> Option<usize> {
    let cached = SKATER_OBJ.load(Ordering::Relaxed);
    if cached != 0 && read::<u32>(cached + OBJ_TYPE) == Some(TYPE_SKATER) {
        return Some(cached);
    }
    {
        let mut last = LAST_SCAN.try_lock().ok()?;
        if last.is_some_and(|t| t.elapsed().as_millis() < 500) {
            return None;
        }
        *last = Some(Instant::now());
    }
    let mut found = None;
    for_each_object(|obj| {
        if found.is_none() && read::<u32>(obj + OBJ_TYPE) == Some(TYPE_SKATER) {
            found = Some(obj);
        }
    });
    let obj = found?;
    log::debug!("[THAW] skater object: 0x{obj:x}");
    SKATER_OBJ.store(obj, Ordering::Relaxed);
    Some(obj)
}

fn component_of(obj: usize, crc: u32) -> Option<usize> {
    let table = read::<usize>(obj + OBJ_COMPONENT_TABLE)?;
    let count = read::<u32>(obj + OBJ_COMPONENT_COUNT)?;
    for i in 0..count.min(64) as usize {
        if read::<u32>(table + i * 8)? == crc {
            return read::<usize>(table + i * 8 + 4);
        }
    }
    None
}

/// The buffer the skater is drawn from (skeleton component +0x18, then +0x4) and how many
/// of its matrices are usable.
fn pose_buffer(obj: usize) -> Option<(usize, usize)> {
    let skeleton = read::<usize>(component_of(obj, CRC_SKELETON_COMPONENT)? + 0x18)?;
    let addr = read::<usize>(skeleton + 0x4)?;
    if !addr_ok::<u32>(addr) {
        return None;
    }
    let bones = bone_run(&read_block(addr, MAX_BONES * 64));
    (bones >= MIN_BONES).then_some((addr, bones))
}

/// Number of plausible bone matrices at the start of `block`: finite, last column
/// 0 0 0 1, rows of sane length (they are scaled and skewed, so not more than that).
fn bone_run(block: &[u8]) -> usize {
    let mut n = 0;
    while n < MAX_BONES && (n + 1) * 64 <= block.len() {
        let f = |i: usize| f32_at(block, n * 64 + i * 4);
        let finite = (0..16).all(|i| f(i).is_finite());
        let last_column = f(3).abs() < 1e-3
            && f(7).abs() < 1e-3
            && f(11).abs() < 1e-3
            && (f(15) - 1.0).abs() < 1e-3;
        let len =
            |r: usize| (f(r * 4).powi(2) + f(r * 4 + 1).powi(2) + f(r * 4 + 2).powi(2)).sqrt();
        let sane = (0..3).all(|r| (0.05..20.0).contains(&len(r)));
        if !(finite && last_column && sane) {
            break;
        }
        n += 1;
    }
    n
}

// Recording.

/// The world matrix most of the skater's parts were drawn with this frame. Not just the
/// last one: in a samba flip on vert, one commit in some frames carried the upright
/// matrix while the rest of the skater was drawn pitched, which recorded single frames
/// where the skater flashed 90 degrees round.
fn drawn_world_matrix() -> Option<[f32; 16]> {
    let parts: Vec<PartCall> = PARTS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let recent: Vec<[f32; 16]> = parts
        .iter()
        .filter(|c| c.seen.elapsed().as_millis() < 50)
        .map(|c| c.drawn)
        .collect();
    let same = |a: &[f32; 16], b: &[f32; 16]| {
        (0..15).all(|i| (a[i] - b[i]).abs() < 0.02 * if i >= 12 { 50.0 } else { 1.0 })
    };
    recent
        .iter()
        .max_by_key(|m| recent.iter().filter(|o| same(m, o)).count())
        .copied()
}

/// One frame of the skater (three entries, see `pose_states`), or nothing when the pose
/// buffer is not available.
fn capture_skater() -> Vec<SkaterState> {
    let Some(obj) = skater_object() else {
        return Vec::new();
    };
    let Some((addr, bones)) = pose_buffer(obj) else {
        return Vec::new();
    };

    let (Some(mut world_pos), Some(mut world_matrix)) =
        (read3(obj + OBJ_POSITION), read_matrix(obj + OBJ_MATRIX))
    else {
        return Vec::new();
    };
    // Prefer what the skater was drawn with this frame (includes the grind/manual lean).
    if let Some(d) = drawn_world_matrix() {
        for r in 0..3 {
            for c in 0..3 {
                world_matrix[r * 4 + c] = d[r * 4 + c];
            }
        }
        world_pos = [d[12], d[13], d[14]];
    }

    // Record the pose the skater was drawn with this frame, not what the buffer holds at
    // the end of the frame: when an animation starts, the game can already have moved on.
    let drawn = DRAWN_POSE.lock().unwrap_or_else(|e| e.into_inner()).take();
    POSE_TAKEN.store(false, Ordering::Relaxed);
    let bytes = match drawn {
        Some((t, b)) if t.elapsed().as_millis() < 50 && b.len() >= bones * 64 => b,
        _ => read_block(addr, bones * 64),
    };
    let Some(states) = pose_states(&bytes, bones, world_pos, world_matrix) else {
        return Vec::new();
    };
    let mut out = Vec::from(states);
    // Every model part is drawn with its own world matrix, and they do not always agree
    // (vert airs, some flip tricks). They are recorded as extra entries without bones,
    // in a fixed order (by part address), and replayed per part.
    for drawn in part_world_matrices() {
        out.push(SkaterState {
            world_pos: [drawn[12], drawn[13], drawn[14]],
            world_matrix: drawn,
            bone_count: 0,
            bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
            bone_trans: [[0f32; 4]; MAX_BONES],
        });
    }
    out
}

/// The world matrix (scale removed) of each of the skater's parts drawn this frame,
/// ordered by part.
fn part_world_matrices() -> Vec<[f32; 16]> {
    let mut parts: Vec<PartCall> = PARTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|c| c.seen.elapsed().as_millis() < 50)
        .copied()
        .collect();
    parts.sort_by_key(|c| c.this);
    parts.truncate(MAX_PARTS);
    parts.iter().map(|c| c.drawn).collect()
}

const MAX_PARTS: usize = 16;

fn write_matrices(addr: usize, matrices: &[[f32; 16]], count: usize) {
    for (i, m) in matrices.iter().take(count).enumerate() {
        for (j, v) in m.iter().enumerate() {
            write::<f32>(addr + i * 64 + j * 4, *v);
        }
    }
}

fn blend_in_world_space(frames: &[ReplayState], i0: usize, t: f32) -> Option<Vec<SkaterState>> {
    let (a, b) = (frames.get(i0)?, frames.get(i0 + 1)?);
    let mut out = blend_skater(frames, i0, t)?;
    out.extend(blend_cars(a, b, t));
    out.extend(smooth_movers(frames, i0, t));
    Some(out)
}

// Memory access. Plausibility-checked only: it cannot catch every bad address.

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn ReadProcessMemory(
        process: isize,
        base: *const c_void,
        buffer: *mut c_void,
        size: usize,
        bytes_read: *mut usize,
    ) -> i32;
    fn AddVectoredExceptionHandler(
        first: u32,
        handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
    ) -> *mut c_void;
}

/// Reads up to `len` bytes of this process's memory and returns what was readable
/// (possibly less, possibly nothing). Unlike a plain read it cannot fault.
fn read_block(addr: usize, len: usize) -> Vec<u8> {
    if addr < 0x1_0000 || addr > 0xBFFE_0000 {
        return Vec::new();
    }
    let mut buf = vec![0u8; len];
    let mut got = 0usize;
    unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            addr as *const c_void,
            buf.as_mut_ptr() as *mut c_void,
            len,
            &mut got,
        );
    }
    buf.truncate(got);
    buf
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Three finite floats (a position).
fn read3(addr: usize) -> Option<[f32; 3]> {
    let v = [
        read::<f32>(addr)?,
        read::<f32>(addr + 4)?,
        read::<f32>(addr + 8)?,
    ];
    v.iter().all(|x| x.is_finite()).then_some(v)
}

/// Sixteen finite floats (a 4x4 matrix).
fn read_matrix(addr: usize) -> Option<[f32; 16]> {
    let f = floats(&read_block(addr, 64));
    (f.len() == 16 && f.iter().all(|x| x.is_finite())).then(|| std::array::from_fn(|i| f[i]))
}

fn addr_ok<T>(addr: usize) -> bool {
    addr >= 0x1_0000 && addr <= 0xBFFE_0000 && addr % std::mem::align_of::<T>() == 0
}

fn read<T: Copy>(addr: usize) -> Option<T> {
    if !addr_ok::<T>(addr) {
        return None;
    }
    Some(unsafe { std::ptr::read_volatile(addr as *const T) })
}

fn write<T: Copy>(addr: usize, value: T) -> bool {
    if !addr_ok::<T>(addr) {
        return false;
    }
    unsafe { std::ptr::write_volatile(addr as *mut T, value) };
    true
}

// Crash logging: a crash in the game, or a panic in falke (which ends the game without an
// exception), is written to falke.log with the game code that was running.

/// CONTEXT (32-bit x86 layout, 716 bytes).
#[repr(C)]
struct ThreadContext {
    context_flags: u32,
    debug_registers: [u32; 6],
    float_save: [u8; 112],
    segments: [u32; 4],
    edi: u32,
    esi: u32,
    ebx: u32,
    edx: u32,
    ecx: u32,
    eax: u32,
    ebp: u32,
    eip: u32,
    seg_cs: u32,
    eflags: u32,
    esp: u32,
    seg_ss: u32,
    extended_registers: [u8; 512],
}

const _: () = assert!(std::mem::size_of::<ThreadContext>() == 716);

#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record: *mut ExceptionRecord,
    exception_address: usize,
    number_parameters: u32,
    information: [usize; 15],
}

#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut ThreadContext,
}

static CRASH_LOGS: AtomicU32 = AtomicU32::new(0);

fn install_crash_logging() {
    unsafe { AddVectoredExceptionHandler(1, crash_handler) };
    std::panic::set_hook(Box::new(|info| {
        log::error!("[THAW] PANIC: {info}");
        let marker = 0u32;
        log::error!(
            "[THAW] PANIC: game code on the stack: {}",
            game_return_addresses(&marker as *const u32 as usize)
        );
        log::logger().flush();
    }));
}

unsafe extern "system" fn crash_handler(info: *mut ExceptionPointers) -> i32 {
    unsafe {
        if info.is_null() || (*info).exception_record.is_null() || (*info).context_record.is_null()
        {
            return 0;
        }
        let rec = &*(*info).exception_record;
        let fatal = matches!(
            rec.exception_code,
            0xC000_0005
                | 0xC000_001D
                | 0xC000_0094
                | 0xC000_0096
                | 0xC000_00FD
                | 0xC000_008C
                | 0xC000_0025
        );
        if !fatal || CRASH_LOGS.fetch_add(1, Ordering::Relaxed) >= 3 {
            return 0;
        }
        let ctx = &*(*info).context_record;
        log::error!(
            "[THAW] CRASH: exception 0x{:08x} at 0x{:x}, access type {} address 0x{:x}",
            rec.exception_code,
            rec.exception_address,
            rec.information[0],
            rec.information[1]
        );
        log::error!(
            "[THAW] CRASH: eax {:08x} ebx {:08x} ecx {:08x} edx {:08x} esi {:08x} edi {:08x} ebp {:08x} esp {:08x}",
            ctx.eax,
            ctx.ebx,
            ctx.ecx,
            ctx.edx,
            ctx.esi,
            ctx.edi,
            ctx.ebp,
            ctx.esp
        );
        log::error!(
            "[THAW] CRASH: game code on the stack: {}",
            game_return_addresses(ctx.esp as usize)
        );
        log::logger().flush();
        0
    }
}

/// Return addresses into the game's code (words right after a call instruction) on the
/// stack starting at `sp`.
fn game_return_addresses(sp: usize) -> String {
    let bytes = read_block(sp, 4096 * 4);
    let mut found = Vec::new();
    for c in bytes.chunks_exact(4) {
        let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize;
        if !GAME_CODE.contains(&w) || found.len() >= 24 {
            continue;
        }
        let b = read_block(w - 6, 6);
        let is_call = b.len() == 6
            && (b[1] == 0xE8
                || (b[0] == 0xFF && (b[1] & 0x38) == 0x10)
                || (b[3] == 0xFF && (b[4] & 0x38) == 0x10)
                || (b[4] == 0xFF && (b[5] & 0x38) == 0x10));
        if is_call {
            found.push(format!("0x{w:x}"));
        }
    }
    found.join(" ")
}

// Camera lookup (FUN_004edaa0 re-implemented without calling game code).

fn active_camera(viewport: usize) -> Option<usize> {
    let table = read::<usize>(ADDR_VIEWPORT_CAM_TABLE_PTR)?;
    let idx = read::<i32>(table + viewport * 4)?;
    if !(0..CAMERA_SLOT_COUNT).contains(&idx) {
        return None;
    }
    let slot = read::<usize>(ADDR_CAMERA_SLOTS + idx as usize * 4)?;
    read::<usize>(slot + SLOT_CAMERA_OFFSET)
}

/// The render camera, but only while the skater is in freecam mode, so the
/// skate cam and fixed cam are never written to.
fn freecam_camera() -> Option<usize> {
    if LAST_MODE.load(Ordering::Relaxed) != MODE_FREECAM {
        return None;
    }
    let cam = active_camera(0)?;
    if LAST_CAMERA.swap(cam, Ordering::Relaxed) != cam {
        log::debug!("[THAW] freecam camera: 0x{cam:x}");
    }
    Some(cam)
}

// The viewer_cam game object.

/// FUN_004331e0, the same call SetViewMode makes, with the same arguments.
fn viewer_object() -> Option<usize> {
    let manager = read::<usize>(ADDR_OBJECT_MANAGER_PTR)?;
    if manager == 0 {
        return None;
    }
    let func: GetObjectByNameFn = unsafe { transmute(ADDR_GET_OBJECT_BY_NAME) };
    let obj = unsafe { func(manager, CRC_VIEWER_CAM) };
    (obj != 0).then_some(obj)
}

/// True when the object's transform equals the render camera's.
fn object_in_sync(obj: usize, cam: usize) -> bool {
    for i in 0..3 {
        let (Some(a), Some(b)) = (
            read::<f32>(obj + OBJ_POSITION + i * 4),
            read::<f32>(cam + CAM_POSITION + i * 4),
        ) else {
            return false;
        };
        if !((a - b).abs() < 2.0) {
            return false;
        }
    }
    for row in 0..3 {
        for col in 0..3 {
            let idx = row * 4 + col;
            let (Some(a), Some(b)) = (
                read::<f32>(obj + OBJ_MATRIX + idx * 4),
                read::<f32>(cam + CAM_MATRIX + idx * 4),
            ) else {
                return false;
            };
            if !((a - b).abs() < 0.05) {
                return false;
            }
        }
    }
    true
}

/// The viewer_cam object, once its transform has been seen to match the render
/// camera. Must be called BEFORE we write the render camera, so both are in sync.
fn validated_viewer_object(cam: usize) -> Option<usize> {
    let Some(obj) = viewer_object() else {
        if !WARNED_NO_VIEWER.swap(true, Ordering::Relaxed) {
            log::warn!("[THAW] no object named viewer_cam found; camera paths will not work");
        }
        return None;
    };
    if VALIDATED_VIEWER.load(Ordering::Relaxed) == obj {
        return Some(obj);
    }
    if object_in_sync(obj, cam) {
        VALIDATED_VIEWER.store(obj, Ordering::Relaxed);
        log::debug!("[THAW] viewer_cam confirmed: 0x{obj:x}");
        Some(obj)
    } else {
        if !WARNED_MISMATCH.swap(true, Ordering::Relaxed) {
            log::warn!(
                "[THAW] viewer_cam 0x{obj:x} does not match the render camera 0x{cam:x}; will retry"
            );
        }
        None
    }
}

// Camera operations.

fn read_pose(cam: usize) -> Option<CameraPose> {
    let mut m = [0f32; 16];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = read::<f32>(cam + CAM_MATRIX + i * 4)?;
    }
    let position = [
        read::<f32>(cam + CAM_POSITION)?,
        read::<f32>(cam + CAM_POSITION + 4)?,
        read::<f32>(cam + CAM_POSITION + 8)?,
    ];

    let mat4: glm::Mat4 = glm::make_mat4(&m);
    let mat3: glm::Mat3 = glm::mat4_to_mat3(&mat4);
    let quat: glm::Quat = glm::quat_normalize(&glm::mat3_to_quat(&mat3));

    Some(CameraPose {
        position: Vec3::from_array(position),
        orientation: quat.into(),
        fov: read::<f32>(cam + CAM_VFOV)?,
    })
}

fn write_pose(cam: usize, pose: &CameraPose) {
    // Validate against the camera BEFORE it is overwritten.
    let viewer = validated_viewer_object(cam);

    let rot: glm::Quat = pose.orientation.into();
    let mat4 = glm::quat_to_mat4(&rot);
    let position = pose.position.to_array();

    for (i, v) in position.iter().enumerate() {
        write::<f32>(cam + CAM_POSITION + i * 4, *v);
    }
    for (i, v) in mat4.as_slice().iter().enumerate() {
        write::<f32>(cam + CAM_MATRIX + i * 4, *v);
    }
    if let Some(obj) = viewer {
        for (i, v) in position.iter().enumerate() {
            write::<f32>(obj + OBJ_POSITION + i * 4, *v);
        }
        for (i, v) in mat4.as_slice().iter().enumerate() {
            write::<f32>(obj + OBJ_MATRIX + i * 4, *v);
        }
    }
    write_fov(cam, pose.fov);
}

/// Set the vertical FOV and keep the two derived values consistent with it.
/// The ratios between them are taken from the camera's own current values,
/// so no aspect ratio has to be known.
fn write_fov(cam: usize, fov: f32) {
    if !fov.is_finite() {
        return;
    }
    let fov = fov.clamp(5.0, 150.0);

    let v = read::<f32>(cam + CAM_VFOV);
    let h = read::<f32>(cam + CAM_HFOV);
    let s = read::<f32>(cam + CAM_FOV_SCALE);

    if let (Some(v), Some(h), Some(s)) = (v, h, s) {
        let tv = (v.to_radians() / 2.0).tan();
        let th = (h.to_radians() / 2.0).tan();
        if tv.is_finite() && th.is_finite() && tv > 0.01 && th > 0.01 && s.is_finite() {
            let k = th / tv;
            let c = s / tv;
            let t = (fov.to_radians() / 2.0).tan();
            write::<f32>(cam + CAM_HFOV, (2.0 * (t * k).atan()).to_degrees());
            write::<f32>(cam + CAM_FOV_SCALE, t * c);
        }
    }
    write::<f32>(cam + CAM_VFOV, fov);
}
