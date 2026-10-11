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

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct CCompositeObject {
    pub position: MthVector,     // 0x00
    pub old_position: MthVector, // 0x10
    _pad_020: [u8; 0x10],
    pub velocity: MthVector,       // 0x30
    pub matrix: MthMatrix,         // 0x40
    pub lerping_matrix: MthMatrix, // 0x80
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CFeeler {
    _pad_000: [u8; 0x8c],
    pub normal: MthVector, // 0x8c
}

// Pad input
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct CSkaterButton {
    pub pressed: u32,       // 0x00
    pub pressed_time: u32,  // 0x04
    pub released_time: u32, // 0x08
    pub pressure: u32,      // 0x0c
    pub triggered: u32,     // 0x10
    pub released: u32,      // 0x14
    pub checksum: u32,      // 0x18
    pub debounce: f32,      // 0x1c
    pub unk: u32,           // 0x20
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct CSkaterPad {
    pub up: CSkaterButton,       // 0x000
    pub down: CSkaterButton,     // 0x024
    pub left: CSkaterButton,     // 0x048
    pub right: CSkaterButton,    // 0x06c
    pub l1: CSkaterButton,       // 0x090
    pub l2: CSkaterButton,       // 0x0b4
    pub r1: CSkaterButton,       // 0x0d8
    pub r2: CSkaterButton,       // 0x0fc
    pub circle: CSkaterButton,   // 0x120
    pub square: CSkaterButton,   // 0x144
    pub triangle: CSkaterButton, // 0x168
    pub x: CSkaterButton,        // 0x18c
    pub start: CSkaterButton,    // 0x1b0
    pub select: CSkaterButton,   // 0x1d4
}

/// The renderable model.
#[repr(C)]
pub struct CModel {
    _pad_000: [u8; 0x8],
    pub skeleton: usize, // 0x08
}

// Skater
#[repr(C)]
pub struct CSkater {
    _pad_000: [u8; 0x608],
    pub model: *mut CModel, // 0x0608
    _pad_60c: [u8; 0x28],
    pub object: *mut CCompositeObject, // 0x0634
    _pad_638: [u8; 0x148],
    pub pad: CSkaterPad, // 0x0780
    _pad_978: [u8; 2],
    pub input_disabled: u8, // 0x097a
    _pad_97b: [u8; 0x1db9],
    pub doing_balance_trick: u8, // 0x2734
    _pad_2735: [u8; 0xe2b],
    pub feeler: CFeeler, // 0x3560
    _pad_35fc: [u8; 0x184],
    pub current_normal: MthVector, // 0x3780
    _pad_3790: [u8; 0x34],
    pub view_mode: u32, // 0x37c4
    _pad_37c8: [u8; 0xc],
    pub camera: *mut CSkaterCam, // 0x37d4
}

#[repr(C)]
pub struct CSkaterCam {
    _pad_000: [u8; 0xc4],
    pub skater: *mut CSkater, // 0xc4
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

