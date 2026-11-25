//! The final image: what goes to the screen after ReShade and its add-ons (DLSS, RenoDX, ...),
//! read from the D3D11 swap chain that dgVoodoo presents the game through.
//!
//! falke registers itself with ReShade as an add-on (any module in the process may) and reads
//! the back buffer at `reshade_present`: after ReShade's effects, its add-ons and its own
//! overlay, right before the frame is presented. That runs on whichever thread presents,
//! which with dgVoodoo is not the game's, so this side never waits on the game: the capture
//! says which image it wants (`want`), the present thread reads it when it comes by, and the
//! capture picks it up later (`take`).

use std::{
    ffi::c_void,
    mem::transmute,
    sync::{Mutex, MutexGuard},
};

use anyhow::{Context, Result, bail};
use windows::{
    Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D11::*,
            Dxgi::{Common::*, IDXGIOutput6, IDXGISwapChain},
        },
        System::{
            LibraryLoader::{
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                GetModuleHandleExW, GetProcAddress,
            },
            ProcessStatus::K32EnumProcessModules,
            Threading::GetCurrentProcess,
        },
    },
    core::{Interface, PCWSTR, s},
};

/// `reshade::addon_event::reshade_present`: after ReShade has drawn its effects and overlay.
const EVENT_RESHADE_PRESENT: u32 = 75;
/// The oldest add-on API with everything used here; any newer ReShade accepts it.
const API_VERSION: u32 = 1;
/// `reshade::api::device_api::d3d11`.
const API_D3D11: u32 = 0xb000;
/// How the swap chain's numbers are to be read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Encoding {
    Sdr,
    /// HDR10: BT.2020 primaries, PQ.
    Pq,
    /// Linear BT.709, 1.0 = 80 nits.
    Scrgb,
}
/// HDR is brought down to SDR with this many nits as white (ITU-R BT.2408 reference white).
const PAPER_WHITE_NITS: f32 = 203.0;

/// One image read from the swap chain.
pub struct Frame {
    /// What `want` was set to when it was read.
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
}

#[derive(Default)]
struct State {
    /// Registered with ReShade (or why not).
    registered: Option<Result<(), String>>,
    /// The image the capture is waiting for.
    want: Option<u64>,
    /// The last `want` read (or failed).
    read: Option<u64>,
    frames: Vec<Frame>,
    error: Option<String>,
    /// `reshade_present` has been called at all.
    seen_present: bool,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn state() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = state();
    f(guard.get_or_insert_with(State::default))
}

/// Registers with ReShade if it hasn't been done yet. `Err` says why the final image can't
/// be read (no ReShade, or one without add-on support).
pub fn connect() -> Result<(), String> {
    if let Some(r) = with(|s| s.registered.clone()) {
        return r;
    }
    let result = unsafe { register() }.map_err(|e| format!("{e:#}"));
    match &result {
        Ok(()) => log::info!("[final image] registered with ReShade"),
        Err(e) => log::warn!("[final image] {e}"),
    }
    with(|s| s.registered = Some(result.clone()));
    result
}

/// Whether ReShade has called back at all (a frame has gone through it since connecting).
pub fn seen_present() -> bool {
    with(|s| s.seen_present)
}

/// Ask for the next image presented to be read and tagged `id` (`None`: nothing wanted).
pub fn want(id: Option<u64>) {
    with(|s| s.want = id);
}

/// Images read since the last call, and an error if reading failed.
pub fn take() -> (Vec<Frame>, Option<String>) {
    with(|s| (std::mem::take(&mut s.frames), s.error.take()))
}

/// Forget what was asked for and read (a new capture starts).
pub fn reset() {
    with(|s| {
        s.want = None;
        s.read = None;
        s.frames.clear();
        s.error = None;
    });
    *staging() = None;
}

unsafe fn register() -> Result<()> {
    let reshade = unsafe { find_reshade() }.context(
        "ReShade isn't loaded (the final image needs ReShade 5 or newer with add-on support)",
    )?;
    let register_addon = unsafe { GetProcAddress(reshade, s!("ReShadeRegisterAddon")) }
        .context("this ReShade has no add-on support (use the build with full add-on support)")?;
    let register_event = unsafe { GetProcAddress(reshade, s!("ReShadeRegisterEvent")) }
        .context("this ReShade has no add-on support")?;
    let register_addon: unsafe extern "C" fn(*mut c_void, u32) -> bool =
        unsafe { transmute(register_addon) };
    let register_event: unsafe extern "C" fn(u32, *mut c_void) =
        unsafe { transmute(register_event) };

    let mut me = HMODULE::default();
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(on_reshade_present as *const () as *const u16),
            &mut me,
        )
    }
    .context("could not find falke's own module")?;

    if !unsafe { register_addon(me.0, API_VERSION) } {
        bail!("ReShade refused falke as an add-on (see ReShade.log)");
    }
    unsafe {
        register_event(
            EVENT_RESHADE_PRESENT,
            on_reshade_present as *const () as *mut c_void,
        )
    };
    Ok(())
}

/// The loaded module that exports ReShade's add-on API.
unsafe fn find_reshade() -> Option<HMODULE> {
    let mut modules = vec![HMODULE::default(); 1024];
    let mut needed = 0u32;
    unsafe {
        K32EnumProcessModules(
            GetCurrentProcess(),
            modules.as_mut_ptr(),
            (modules.len() * size_of::<HMODULE>()) as u32,
            &mut needed,
        )
        .ok()
        .ok()?;
    }
    let count = (needed as usize / size_of::<HMODULE>()).min(modules.len());
    modules[..count]
        .iter()
        .copied()
        .find(|&m| unsafe { GetProcAddress(m, s!("ReShadeRegisterAddon")) }.is_some())
}

// The present thread

extern "C" fn on_reshade_present(runtime: *mut c_void) {
    let id = {
        let mut guard = state();
        let Some(s) = guard.as_mut() else { return };
        s.seen_present = true;
        match s.want {
            Some(id) if s.read != Some(id) => id,
            _ => return,
        }
    };
    let result = unsafe { read_runtime(runtime) };
    with(|s| {
        s.read = Some(id);
        match result {
            Ok((width, height, bgra)) => s.frames.push(Frame {
                id,
                width,
                height,
                bgra,
            }),
            Err(e) => s.error = Some(format!("{e:#}")),
        }
    });
}

/// Calls the `index`th virtual function of a ReShade API object (MSVC `thiscall`, no arguments,
/// a scalar result).
unsafe fn virtual_call<R>(object: *mut c_void, index: usize) -> R {
    unsafe {
        let vtable = *(object as *const *const usize);
        let f: extern "thiscall" fn(*mut c_void) -> R = transmute(*vtable.add(index));
        f(object)
    }
}

/// The back buffer of a ReShade effect runtime, as BGRA.
unsafe fn read_runtime(runtime: *mut c_void) -> Result<(u32, u32, Vec<u8>)> {
    // effect_runtime : device_object : api_object
    //   0 get_native (the runtime's IDXGISwapChain), 1 get_private_data,
    //   2 set_private_data, 3 get_device, 4 get_hwnd, ...
    // device : api_object
    //   3 get_api
    // Only these, which take no arguments, are called.
    let device: *mut c_void = unsafe { virtual_call(runtime, 3) };
    if device.is_null() {
        bail!("ReShade gave no device");
    }
    let api: u32 = unsafe { virtual_call(device, 3) };
    if api != API_D3D11 {
        bail!(
            "ReShade isn't running on Direct3D 11 here (API {api:#x}); the final image needs \
             dgVoodoo's D3D11 output"
        );
    }
    let native: u64 = unsafe { virtual_call(runtime, 0) };
    let raw = native as usize as *mut c_void;
    let swapchain = unsafe { IDXGISwapChain::from_raw_borrowed(&raw) }
        .context("ReShade gave no swap chain")?;

    unsafe { read_back_buffer(swapchain) }
}

struct Staging {
    device: usize,
    desc: D3D11_TEXTURE2D_DESC,
    staging: ID3D11Texture2D,
    /// For a multisampled back buffer: where it is resolved to first.
    resolved: Option<ID3D11Texture2D>,
}

// Only touched from the present thread (and dropped by `reset`).
unsafe impl Send for Staging {}

static STAGING: Mutex<Option<Staging>> = Mutex::new(None);

fn staging() -> MutexGuard<'static, Option<Staging>> {
    STAGING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the swap chain's display is in HDR (Windows HDR on): a 10-bit or float swap chain
/// presented to it holds HDR10 or scRGB, and SDR otherwise.
unsafe fn display_is_hdr(swapchain: &IDXGISwapChain) -> bool {
    unsafe {
        let Ok(output) = swapchain.GetContainingOutput() else {
            return false;
        };
        let Ok(output6) = output.cast::<IDXGIOutput6>() else {
            return false;
        };
        output6
            .GetDesc1()
            .is_ok_and(|d| d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)
    }
}

unsafe fn read_back_buffer(swapchain: &IDXGISwapChain) -> Result<(u32, u32, Vec<u8>)> {
    unsafe {
        let back: ID3D11Texture2D = swapchain.GetBuffer(0).context("GetBuffer")?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        back.GetDesc(&mut desc);
        let device = back.GetDevice().context("GetDevice")?;
        let context = device.GetImmediateContext().context("GetImmediateContext")?;

        let mut guard = staging();
        let fits = guard.as_ref().is_some_and(|s| {
            s.device == device.as_raw() as usize
                && s.desc.Width == desc.Width
                && s.desc.Height == desc.Height
                && s.desc.Format == desc.Format
                && s.desc.SampleDesc.Count == desc.SampleDesc.Count
        });
        if !fits {
            *guard = None;
            let mut staging_desc = D3D11_TEXTURE2D_DESC {
                Width: desc.Width,
                Height: desc.Height,
                MipLevels: 1,
                ArraySize: 1,
                Format: desc.Format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut staging = None;
            device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
                .context("could not make a texture to read the final image into")?;
            let resolved = if desc.SampleDesc.Count > 1 {
                staging_desc.Usage = D3D11_USAGE_DEFAULT;
                staging_desc.CPUAccessFlags = 0;
                let mut t = None;
                device
                    .CreateTexture2D(&staging_desc, None, Some(&mut t))
                    .context("could not make a texture to resolve the final image into")?;
                t
            } else {
                None
            };
            *guard = Some(Staging {
                device: device.as_raw() as usize,
                desc,
                staging: staging.context("no staging texture")?,
                resolved,
            });
        }
        let s = guard.as_ref().unwrap();

        match &s.resolved {
            Some(resolved) => {
                context.ResolveSubresource(resolved, 0, &back, 0, desc.Format);
                context.CopyResource(&s.staging, resolved);
            }
            None => context.CopyResource(&s.staging, &back),
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context
            .Map(&s.staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .context("could not read the final image")?;
        let (w, h) = (desc.Width as usize, desc.Height as usize);
        let bpp = match desc.Format {
            DXGI_FORMAT_R16G16B16A16_FLOAT => 8,
            _ => 4,
        };
        let mut raw = vec![0u8; w * h * bpp];
        for y in 0..h {
            let src = (mapped.pData as *const u8).add(y * mapped.RowPitch as usize);
            std::ptr::copy_nonoverlapping(src, raw.as_mut_ptr().add(y * w * bpp), w * bpp);
        }
        context.Unmap(&s.staging, 0);
        drop(guard);

        let encoding = match desc.Format {
            DXGI_FORMAT_R10G10B10A2_UNORM if display_is_hdr(swapchain) => Encoding::Pq,
            DXGI_FORMAT_R16G16B16A16_FLOAT if display_is_hdr(swapchain) => Encoding::Scrgb,
            _ => Encoding::Sdr,
        };
        let bgra = to_bgra(&raw, desc.Format, encoding)?;
        Ok((desc.Width, desc.Height, bgra))
    }
}

// Formats

fn to_bgra(raw: &[u8], format: DXGI_FORMAT, encoding: Encoding) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len());
    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        | DXGI_FORMAT_B8G8R8X8_UNORM
        | DXGI_FORMAT_B8G8R8X8_UNORM_SRGB => {
            for p in raw.chunks_exact(4) {
                out.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => {
            for p in raw.chunks_exact(4) {
                out.extend_from_slice(&[p[2], p[1], p[0], 255]);
            }
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let pq = encoding == Encoding::Pq;
            for p in raw.chunks_exact(4) {
                let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                let c = [v & 1023, (v >> 10) & 1023, (v >> 20) & 1023];
                let rgb = if pq {
                    let n = c.map(|x| pq_to_nits(x as f32 / 1023.0) / PAPER_WHITE_NITS);
                    encode_sdr(bt2020_to_709(n), true)
                } else {
                    c.map(|x| (x >> 2) as u8)
                };
                out.extend_from_slice(&[rgb[2], rgb[1], rgb[0], 255]);
            }
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            // scRGB: linear BT.709, 1.0 = 80 nits. An SDR swap chain in this format holds
            // 1.0 as white.
            let hdr = encoding == Encoding::Scrgb;
            let scale = if hdr { 80.0 / PAPER_WHITE_NITS } else { 1.0 };
            for p in raw.chunks_exact(8) {
                let ch = |i: usize| half_to_f32(u16::from_le_bytes([p[i], p[i + 1]])) * scale;
                let rgb = encode_sdr([ch(0), ch(2), ch(4)], hdr);
                out.extend_from_slice(&[rgb[2], rgb[1], rgb[0], 255]);
            }
        }
        other => bail!(
            "the final image is in a format falke can't save yet (DXGI format {})",
            other.0
        ),
    }
    Ok(out)
}

/// SMPTE ST 2084 (PQ) code value to nits.
fn pq_to_nits(e: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;
    let p = e.clamp(0.0, 1.0).powf(1.0 / M2);
    10000.0 * ((p - C1).max(0.0) / (C2 - C3 * p)).powf(1.0 / M1)
}

fn bt2020_to_709(c: [f32; 3]) -> [f32; 3] {
    [
        1.6605 * c[0] - 0.5876 * c[1] - 0.0728 * c[2],
        -0.1246 * c[0] + 1.1329 * c[1] - 0.0083 * c[2],
        -0.0182 * c[0] - 0.1006 * c[1] + 1.1187 * c[2],
    ]
}

/// Linear light (1.0 = white) to 8-bit sRGB. HDR (`roll`) has its highlights rolled off
/// above 0.8 instead of clipped.
fn encode_sdr(c: [f32; 3], roll: bool) -> [u8; 3] {
    c.map(|v| {
        let v = v.max(0.0);
        let v = if roll && v > 0.8 {
            let over = v - 0.8;
            0.8 + 0.2 * over / (over + 0.2)
        } else {
            v
        };
        let s = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    })
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    let v = match exp {
        0 => frac / 1024.0 * 2f32.powi(-14),
        31 => {
            if frac == 0.0 {
                f32::INFINITY
            } else {
                0.0
            }
        }
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    };
    sign * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_decode() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
    }

    #[test]
    fn pq_reference_white() {
        // 203 nits is PQ code 0.58.
        let n = pq_to_nits(0.5807);
        assert!((n - 203.0).abs() < 2.0, "{n}");
    }

    #[test]
    fn sdr_white_stays_white() {
        assert_eq!(encode_sdr([0.0, 1.0, 50.0], false), [0, 255, 255]);
        let [_, white, over] = encode_sdr([0.0, 1.0, 50.0], true);
        assert!(white > 230 && white < over && over == 255, "{white} {over}");
    }

    #[test]
    fn rgba_swaps_to_bgra() {
        let out = to_bgra(&[1, 2, 3, 4], DXGI_FORMAT_R8G8B8A8_UNORM, Encoding::Sdr).unwrap();
        assert_eq!(out, [3, 2, 1, 255]);
    }
}
