use std::{
    mem::transmute,
    ptr,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use anyhow::{Context, Result};
use nalgebra_glm as glm;
use retour::static_detour;

use thps_rs::thps4::{
    ADDR_GET_ACTIVE_CAMERA, ADDR_SKELETON_UPDATE, CSkater, GfxCamera, MthMatrix, MthVector,
    SIG_B4_DO_FREECAM_LOGIC, gfx_camera_matrix, gfx_camera_pos, gfx_camera_set_matrix,
    gfx_camera_set_pos, model_render, screen_angle, set_screen_angle,
};

use crate::{
    camera::CameraPose,
    game::Game,
    math::Vec3,
    replay::{MAX_BONES, ReplayState, SkaterState},
};

/// Bones in a THPS4 skater skeleton.
const BONE_COUNT: usize = 55;

static_detour! {
    static DoFreecamLogicHook: unsafe extern "C" fn(*mut CSkater);
    static SkeletonUpdateHook: unsafe extern "thiscall" fn(*const usize, *mut [f32; 4], *mut [f32; 4]);
    static GetActiveCameraHook: unsafe extern fn(i32) -> usize;
}

/// Pointer ot the skater.
static SKATER: AtomicUsize = AtomicUsize::new(0);

/// The camera the game is currently rendering through, from
/// `hk_get_active_camera`. What a POV track wants.
static ACTIVE_CAMERA: AtomicUsize = AtomicUsize::new(0);

/// The shared bone buffers, from the last `CSkeleton::Update` seen. Stored as
/// addresses because they are the same for every skeleton anyway.
static BONE_QUATS: AtomicUsize = AtomicUsize::new(0);
static BONE_TRANS: AtomicUsize = AtomicUsize::new(0);

/// One skeleton's bone pose, copied out of the shared buffers.
#[derive(Copy, Clone)]
struct BonePose {
    quats: [[f32; 4]; BONE_COUNT],
    trans: [[f32; 4]; BONE_COUNT],
}

const EMPTY_POSE: BonePose = BonePose {
    quats: [[0f32; 4]; BONE_COUNT],
    trans: [[0f32; 4]; BONE_COUNT],
};

/// The skater's latest pose, copied inside the update hook.
static SKATER_POSE: Mutex<Option<BonePose>> = Mutex::new(None);

pub struct Thps4;

impl Game for Thps4 {
    fn name(&self) -> &'static str {
        "THPS4"
    }

    unsafe fn install_hooks(&self) -> Result<()> {
        unsafe { install_hooks() }
    }

    unsafe fn capture_frame(&self, frame: &mut ReplayState) {
        // Prefer the camera the game renders through; the skater's own camera
        // is the fallback until GetActiveCamera has fired.
        let cam = match ACTIVE_CAMERA.load(Ordering::Relaxed) {
            0 => camera_obj().unwrap_or(0),
            active => active,
        };
        if cam != 0 {
            frame.camera = Some(unsafe { read_camera(cam) });
        }

        if let Some(skater) = unsafe { capture_skater_state() } {
            frame.skaters.push(skater);
        }
    }

    fn in_freecam(&self) -> bool {
        let Some(skater) = skater() else {
            return false;
        };
        unsafe { (*skater).view_mode > 0 }
    }

    fn is_replay_available(&self) -> bool {
        true
    }

    fn camera(&self) -> Option<CameraPose> {
        Some(unsafe { read_camera(camera_obj()?) })
    }

    fn set_camera(&self, pose: &CameraPose) {
        let Some(camera) = camera_obj() else {
            return;
        };

        let rot: glm::Quat = pose.orientation.into();
        let matrix = mat_from_slice(glm::quat_to_mat4(&rot).as_slice());

        unsafe {
            gfx_camera_set_matrix(camera, &matrix);
            let mut pos = gfx_camera_pos(camera);
            pos.x = pose.position.x;
            pos.y = pose.position.y;
            pos.z = pose.position.z;
            gfx_camera_set_pos(camera, &pos);
        }

        write_fov(pose.fov);
    }

    fn fov(&self) -> f32 {
        unsafe { screen_angle() }
    }

    fn set_fov(&self, fov: f32) {
        write_fov(fov);
    }

    unsafe fn apply_frame(&self, frame: &ReplayState) {
        // Single-skater, as capture is.
        let Some(skater) = frame.skaters.first() else {
            return;
        };

        let (Some(model), Some(skeleton)) = (skater_model(), skater_skeleton()) else {
            return;
        };

        // The buffers `Update` reads a pose out of.
        let qp = BONE_QUATS.load(Ordering::Relaxed) as *mut [f32; 4];
        let tp = BONE_TRANS.load(Ordering::Relaxed) as *mut [f32; 4];
        if qp.is_null() || tp.is_null() {
            return;
        }

        static LOGGED_FOR: AtomicUsize = AtomicUsize::new(0);
        if LOGGED_FOR.swap(skeleton, Ordering::Relaxed) != skeleton {
            log::debug!(
                "[playback] posing skeleton 0x{skeleton:x} for model 0x{model:x} (quats=0x{:x}, trans=0x{:x})",
                qp as usize,
                tp as usize,
            );
            log::debug!("[playback] bone[0] quat={:?}", skater.bone_quats[0]);
        }

        // The draw takes position in the matrix's w row.
        let mut render_matrix = skater.world_matrix;
        render_matrix[12] = skater.world_pos[0];
        render_matrix[13] = skater.world_pos[1];
        render_matrix[14] = skater.world_pos[2];
        render_matrix[15] = 1.0;

        let bones = skater.bones().min(BONE_COUNT);
        unsafe {
            let mut saved_q = [[0f32; 4]; BONE_COUNT];
            let mut saved_t = [[0f32; 4]; BONE_COUNT];
            ptr::copy_nonoverlapping(qp, saved_q.as_mut_ptr(), BONE_COUNT);
            ptr::copy_nonoverlapping(tp, saved_t.as_mut_ptr(), BONE_COUNT);

            ptr::copy_nonoverlapping(skater.bone_quats.as_ptr(), qp, bones);
            ptr::copy_nonoverlapping(skater.bone_trans.as_ptr(), tp, bones);
            SkeletonUpdateHook.call(skeleton as *const usize, qp, tp);

            model_render(model, &render_matrix, true);

            ptr::copy_nonoverlapping(saved_q.as_ptr(), qp, BONE_COUNT);
            ptr::copy_nonoverlapping(saved_t.as_ptr(), tp, BONE_COUNT);
        }
    }
}

/// Write the fov, in degrees, unless the game already has it.
fn write_fov(fov: f32) {
    if unsafe { screen_angle() } != fov {
        unsafe { set_screen_angle(fov) };
    }
}

/// Read a pose out of a camera object.
unsafe fn read_camera(camera: GfxCamera) -> CameraPose {
    let (pos, matrix) = unsafe { (gfx_camera_pos(camera), gfx_camera_matrix(camera)) };

    let mat4: glm::Mat4 = glm::make_mat4(&mat_to_array(&matrix));
    let mat3: glm::Mat3 = glm::mat4_to_mat3(&mat4);
    let quat: glm::Quat = glm::quat_normalize(&glm::mat3_to_quat(&mat3));

    CameraPose {
        position: Vec3::new(pos.x, pos.y, pos.z),
        orientation: quat.into(),
        fov: unsafe { screen_angle() },
    }
}

// Frame capture
unsafe fn capture_skater_state() -> Option<SkaterState> {
    let object = unsafe { (*skater()?).object };
    if object.is_null() {
        log::warn!("[capture] skater has no CCompositeObject");
        return None;
    }
    let object = unsafe { *object };

    let pose = {
        let slot = SKATER_POSE.lock().ok().or_else(|| {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!("[capture] failed to lock SKATER_POSE");
            }
            None
        })?;
        (*slot).or_else(|| {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!("[capture] no pose yet - no skater skeleton update seen");
            }
            None
        })?
    };

    let mut bone_quats = [[0f32; 4]; MAX_BONES];
    let mut bone_trans = [[0f32; 4]; MAX_BONES];
    bone_quats[..BONE_COUNT].copy_from_slice(&pose.quats);
    bone_trans[..BONE_COUNT].copy_from_slice(&pose.trans);

    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::debug!("[capture] bone[0] quat={:?}", bone_quats[0]);
        log::debug!("[capture] bone[0] trans={:?}", bone_trans[0]);
        log::debug!("[capture] world_pos={:?}", object.position);
    }

    Some(SkaterState {
        // We use control_root position, maybe GetBoneWorldPosition isn't needed in THUG2 at all?
        world_pos: [object.position.x, object.position.y, object.position.z],
        world_matrix: mat_to_array(&object.matrix),
        bone_count: BONE_COUNT as u8,
        bone_quats,
        bone_trans,
    })
}

/// The skater's `Nx::CModel`, at `CSkater + 0x608`. Read fresh every time
/// rather than cached, so a level change or skater reload cannot leave a stale
/// pointer behind.
fn skater_model() -> Option<usize> {
    let model = unsafe { (*skater()?).model };
    (!model.is_null()).then_some(model as usize)
}

/// The skater's own skeleton, at `CModel + 0x8`.
fn skater_skeleton() -> Option<usize> {
    let model = unsafe { (*skater()?).model };
    if model.is_null() {
        return None;
    }
    let skeleton = unsafe { (*model).skeleton };
    (skeleton != 0).then_some(skeleton)
}

fn skater() -> Option<*mut CSkater> {
    let skater = SKATER.load(Ordering::Relaxed) as *mut CSkater;
    (!skater.is_null()).then_some(skater)
}

fn camera_obj() -> Option<GfxCamera> {
    let camera = unsafe { (*skater()?).camera } as GfxCamera;
    (camera != 0).then_some(camera)
}

fn mat_to_array(m: &MthMatrix) -> [f32; 16] {
    [
        m.x.x, m.x.y, m.x.z, m.x.w, m.y.x, m.y.y, m.y.z, m.y.w, m.z.x, m.z.y, m.z.z, m.z.w, m.w.x,
        m.w.y, m.w.z, m.w.w,
    ]
}

fn mat_from_slice(a: &[f32]) -> MthMatrix {
    let row = |i: usize| MthVector {
        x: a[i],
        y: a[i + 1],
        z: a[i + 2],
        w: a[i + 3],
    };
    MthMatrix {
        x: row(0),
        y: row(4),
        z: row(8),
        w: row(12),
    }
}

// Hook installation
unsafe fn install_hooks() -> Result<()> {
    unsafe {
        let better4 = toy_arms::internal::module::Module::from_name("better4.dll")
            .context("failed to find better4.dll — is Better4 loaded?")?;
        log::debug!("[addr] better4.dll base: 0x{:x}", better4.base_address);

        let addr_do_freecam_logic = better4
            .find_pattern(SIG_B4_DO_FREECAM_LOGIC)
            .context("failed to find do_freecam_logic")?
            + better4.base_address;
        log::debug!("[addr] do_freecam_logic: 0x{:x}", addr_do_freecam_logic);
        DoFreecamLogicHook
            .initialize(transmute(addr_do_freecam_logic), hk_do_freecam_logic)
            .context("failed to init do_freecam_logic hook")?
            .enable()
            .context("failed to enable do_freecam_logic hook")?;
        log::info!("[THPS4] do_freecam_logic hook installed");

        log::debug!("[addr] CSkeleton::Update: 0x{ADDR_SKELETON_UPDATE:x}");
        SkeletonUpdateHook
            .initialize(transmute(ADDR_SKELETON_UPDATE), hk_skeleton_update)
            .context("failed to init CSkeleton::Update hook")?
            .enable()
            .context("failed to enable CSkeleton::Update hook")?;

        log::debug!("[addr] GetActiveCamera: 0x{ADDR_GET_ACTIVE_CAMERA:x}");
        GetActiveCameraHook
            .initialize(transmute(ADDR_GET_ACTIVE_CAMERA), hk_get_active_camera)
            .context("failed to init GetActiveCamera hook")?
            .enable()
            .context("failed to enable GetActiveCamera hook")?;

        Ok(())
    }
}

// Detour handlers
fn hk_do_freecam_logic(skater: *mut CSkater) {
    let previous = SKATER.swap(skater as usize, Ordering::Relaxed);
    if previous != 0 && previous != skater as usize {
        // A reload. The model and skeleton are read through the new skater so
        // they need no clearing, but a pose copied from the old one is stale.
        log::debug!(
            "[THPS4] skater changed: 0x{previous:x} -> 0x{:x}, dropping its pose",
            skater as usize
        );
        if let Ok(mut slot) = SKATER_POSE.lock() {
            *slot = None;
        }
    }
    if previous == 0 {
        unsafe {
            log::debug!(
                "[THPS4] first freecam tick: CSkater=0x{:x} view_mode={} CSkaterCam=0x{:x}",
                skater as usize,
                (*skater).view_mode,
                (*skater).camera as usize,
            );
        }
    }

    unsafe { DoFreecamLogicHook.call(skater) };
}

fn hk_skeleton_update(this: *const usize, pquat: *mut [f32; 4], ptrans: *mut [f32; 4]) {
    unsafe { SkeletonUpdateHook.call(this, pquat, ptrans) };

    BONE_QUATS.store(pquat as usize, Ordering::Relaxed);
    BONE_TRANS.store(ptrans as usize, Ordering::Relaxed);

    if skater_skeleton() != Some(this as usize) {
        return;
    }

    let mut pose = EMPTY_POSE;
    unsafe {
        ptr::copy_nonoverlapping(pquat, pose.quats.as_mut_ptr(), BONE_COUNT);
        ptr::copy_nonoverlapping(ptrans, pose.trans.as_mut_ptr(), BONE_COUNT);
    }

    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::debug!(
            "[THPS4] skater skeleton 0x{:x}: bone[0] quat={:?}",
            this as usize,
            pose.quats[0]
        );
    }

    if let Ok(mut slot) = SKATER_POSE.lock() {
        *slot = Some(pose);
    }
}

fn hk_get_active_camera(index: i32) -> usize {
    let camera = unsafe { GetActiveCameraHook.call(index) };
    if camera != 0 && ACTIVE_CAMERA.swap(camera, Ordering::Relaxed) != camera {
        log::debug!("[addr] active camera: 0x{camera:x} (viewport {index})");
    }
    camera
}
