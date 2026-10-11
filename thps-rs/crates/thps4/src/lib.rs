use std::mem::transmute;

pub const SIG_B4_DO_FREECAM_LOGIC: &str =
    "55 8B EC 51 A1 ? ? ? ? 89 45 FC 83 7D FC 00 ? ? 83 7D FC 01 74 ? EB ?";

pub const ADDR_SET_SCREEN_ANGLE: usize = 0x00470360;

pub const ADDR_SCREEN_ANGLE: usize = 0x00ab4b3c;

pub const SIG_SKELETON_UPDATE: &str = "81 EC B0 00 00 00 53 55 8B E9 8B 9C 24 BC 00 00 00 56 57";
pub const ADDR_SKELETON_UPDATE: usize = 0x00478c50;

pub const SIG_GET_ACTIVE_CAMERA: &str = "8B 0D CC 17 5A 00 56 8B 74 24 08 33 D2 33 C0";
pub const ADDR_GET_ACTIVE_CAMERA: usize = 0x004700f0;

pub const SIG_MODEL_RENDER: &str =
    "81 EC 8C 00 00 00 53 55 56 8B F1 57 8B 4E 04 0F BF AE 08 01 00 00";
pub const ADDR_MODEL_RENDER: usize = 0x004687e0;

pub const SIG_MODEL_SET_ACTIVE: &str = "53 8B 5C 24 08 57 8B F9 8A 87 0B 01 00 00 84 C0";
pub const ADDR_MODEL_SET_ACTIVE: usize = 0x00468eb0;

pub const ADDR_GFX_CAMERA_GET_POS: usize = 0x0045df70;
pub const ADDR_GFX_CAMERA_SET_POS: usize = 0x0045df50;
pub const ADDR_GFX_CAMERA_GET_MATRIX: usize = 0x0045e000;
pub const ADDR_GFX_CAMERA_SET_MATRIX: usize = 0x0045df80;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct MthVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct MthMatrix {
    pub x: MthVector,
    pub y: MthVector,
    pub z: MthVector,
    pub w: MthVector,
}

// Object system

/// A moving thing in the world. Only the two fields a frame is built out of
/// are named; `old_position`, `velocity` and the lerping matrix below it are
/// bytes to step over.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CCompositeObject {
    pub position: MthVector, // 0x00
    _pad_010: [u8; 0x30],
    pub matrix: MthMatrix, // 0x40
}

/// The renderable model.
#[repr(C)]
pub struct CModel {
    _pad_000: [u8; 0x8],
    pub skeleton: usize, // 0x08
}

// Skater
//
// Padded down to the four fields we read. What the gaps swallow is input (the
// pad at 0x780), collision (the feeler at 0x3560) and trick state - none of it
// pose or camera data, so it is bytes to step over rather than types to
// transcribe.
#[repr(C)]
pub struct CSkater {
    _pad_000: [u8; 0x608],
    pub model: *mut CModel, // 0x0608
    _pad_60c: [u8; 0x28],
    pub object: *mut CCompositeObject, // 0x0634
    _pad_638: [u8; 0x318c],
    pub view_mode: u32, // 0x37c4
    _pad_37c8: [u8; 0xc],
    /// The skater's `CSkaterCam`, held as the handle the `Gfx_Camera`
    /// accessors take - which is all we ever do with it.
    pub camera: GfxCamera, // 0x37d4
}

pub type GfxCamera = usize;

pub type SetScreenAngleFn = unsafe extern "C" fn(f32);
pub type ModelRenderFn = unsafe extern "thiscall" fn(usize, *const [f32; 16], bool);
pub type ModelSetActiveFn = unsafe extern "thiscall" fn(usize, bool);
pub type GfxCameraGetPosFn = unsafe extern "thiscall" fn(GfxCamera) -> *mut MthVector;
pub type GfxCameraSetPosFn = unsafe extern "thiscall" fn(GfxCamera, *const MthVector);
pub type GfxCameraGetMatrixFn = unsafe extern "thiscall" fn(GfxCamera) -> *mut MthMatrix;
pub type GfxCameraSetMatrixFn = unsafe extern "thiscall" fn(GfxCamera, *const MthMatrix);

// Camera accessors

/// Set the screen angle.
pub unsafe fn set_screen_angle(angle: f32) {
    let func: SetScreenAngleFn = unsafe { transmute(ADDR_SET_SCREEN_ANGLE) };
    unsafe { func(angle) };
}

pub unsafe fn screen_angle() -> f32 {
    unsafe { *(ADDR_SCREEN_ANGLE as *const f32) }
}

pub unsafe fn model_render(model: usize, matrix: &[f32; 16], no_anim: bool) {
    let func: ModelRenderFn = unsafe { transmute(ADDR_MODEL_RENDER) };
    unsafe { func(model, matrix, no_anim) };
}

pub unsafe fn model_set_active(model: usize, active: bool) {
    let func: ModelSetActiveFn = unsafe { transmute(ADDR_MODEL_SET_ACTIVE) };
    unsafe { func(model, active) };
}

pub unsafe fn gfx_camera_pos(camera: GfxCamera) -> MthVector {
    let func: GfxCameraGetPosFn = unsafe { transmute(ADDR_GFX_CAMERA_GET_POS) };
    unsafe { *func(camera) }
}

pub unsafe fn gfx_camera_set_pos(camera: GfxCamera, pos: &MthVector) {
    let func: GfxCameraSetPosFn = unsafe { transmute(ADDR_GFX_CAMERA_SET_POS) };
    unsafe { func(camera, pos) };
}

pub unsafe fn gfx_camera_matrix(camera: GfxCamera) -> MthMatrix {
    let func: GfxCameraGetMatrixFn = unsafe { transmute(ADDR_GFX_CAMERA_GET_MATRIX) };
    unsafe { *func(camera) }
}

pub unsafe fn gfx_camera_set_matrix(camera: GfxCamera, matrix: &MthMatrix) {
    let func: GfxCameraSetMatrixFn = unsafe { transmute(ADDR_GFX_CAMERA_SET_MATRIX) };
    unsafe { func(camera, matrix) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Padding is the only thing holding these fields in place, so the offsets
    // the comments claim are asserted. 32-bit pointers assumed - the whole
    // workspace builds for i686.

    #[test]
    fn mth_sizes() {
        assert_eq!(size_of::<MthVector>(), 0x10);
        assert_eq!(size_of::<MthMatrix>(), 0x40);
    }

    #[test]
    fn composite_object_layout() {
        assert_eq!(offset_of!(CCompositeObject, position), 0x00);
        assert_eq!(offset_of!(CCompositeObject, matrix), 0x40);
        assert_eq!(size_of::<CCompositeObject>(), 0x80);
    }

    #[test]
    fn model_layout() {
        assert_eq!(offset_of!(CModel, skeleton), 0x08);
    }

    #[test]
    fn skater_layout() {
        assert_eq!(offset_of!(CSkater, model), 0x608);
        assert_eq!(offset_of!(CSkater, object), 0x634);
        assert_eq!(offset_of!(CSkater, view_mode), 0x37c4);
        assert_eq!(offset_of!(CSkater, camera), 0x37d4);
    }
}
