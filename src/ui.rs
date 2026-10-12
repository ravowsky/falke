use egui::{
    Align2, Color32, Context as eguiContext, CursorIcon, DragValue, FontId, Id, LayerId, Order,
    Pos2, Rect, Sense, Shape, Stroke, TopBottomPanel, Vec2,
};
use std::{
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use crate::{
    camera::{CameraPath, CameraPose, Keyframe, OrbitCamera, fov_approx_eq},
    capture,
    falke::version_string,
    game,
    math::Vec3,
    replay::{self, LoadedReplay, ReplayInfo, ReplayState},
};

pub static UI_SYSTEM: OnceLock<Mutex<UiSystem>> = OnceLock::new();

/// Narrowest the zoom window may get, in seconds.
const MIN_VISIBLE_DURATION: f32 = 0.5;

/// Fov for a pose built before the game has told us what the camera uses.
const DEFAULT_FOV: f32 = 90.0;

/// A selected keyframe the camera has since moved off.
const STALE_KEY_COLOR: Color32 = Color32::from_rgb(255, 95, 95);

const STATUS_ERROR_COLOR: Color32 = Color32::from_rgb(255, 120, 110);

/// Dark enough that the default light text stays readable on them.
const APPLY_COLOR: Color32 = Color32::from_rgb(38, 105, 58);
const RESET_COLOR: Color32 = Color32::from_rgb(128, 102, 24);

/// Why every authoring control is off. They are all off for the same reason.
const POV_HINT: &str =
    "Replay POV flies the recorded camera. Pick Path to fly and author your own.";

/// How close two keyframe times have to be to count as the same time.
const KEYFRAME_EPSILON: f32 = 1e-3;

/// How far a new keyframe steps on when the playhead's time is taken.
const KEYFRAME_SPACING: f32 = 1.0;

/// How far ahead of the camera the orbit pivot lands when orbit mode starts.
const DEFAULT_ORBIT_DISTANCE: f32 = 300.0;

/// Radians of orbit per pixel dragged.
const ORBIT_ROTATE_SPEED: f32 = 0.005;

/// Target movement per pixel dragged, as a fraction of the orbit distance, so
/// panning feels the same close up and far away.
const ORBIT_PAN_SPEED: f32 = 0.002;

/// Zoom per point of wheel scroll, one notch is roughly 50 points.
const ORBIT_ZOOM_SPEED: f32 = 0.002;

const WINDOW_WIDTH: f32 = 220.0;
const WINDOW_MARGIN: f32 = 24.0;

/// Roughly how tall the floating windows get. Only used to keep them on screen,
/// so an overestimate is the safe direction.
const KEYFRAME_WINDOW_HEIGHT: f32 = 400.0;
const REPLAY_WINDOW_HEIGHT: f32 = 170.0;
const KEYS_WINDOW_HEIGHT: f32 = 300.0;

/// How long an informational status line stays up. Errors stay until replaced.
const STATUS_FADE: Duration = Duration::from_secs(6);

/// Against the right edge, `fraction` of the way down, pulled up if `height`
/// would hang off the bottom. `default_pos` applies once, so off screen is final.
fn window_pos(ctx: &eguiContext, height: f32, fraction: f32) -> Pos2 {
    let screen = ctx.screen_rect();
    let x = (screen.right() - WINDOW_WIDTH - WINDOW_MARGIN).max(screen.left() + WINDOW_MARGIN);
    let top = screen.top() + WINDOW_MARGIN;
    let bottom = (screen.bottom() - height - WINDOW_MARGIN).max(top);
    let y = (screen.top() + screen.height() * fraction).clamp(top, bottom);
    egui::pos2(x, y)
}

/// Owns the camera path and timeline viewport state.
pub struct Timeline {
    pub path: CameraPath,
    /// The keyframe the playhead is on.
    pub selected_index: Option<usize>,
    pub dragging_index: Option<usize>,
    pub total_duration: f32,
    pub visible_duration: f32,
    pub scroll_offset: f32,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            path: CameraPath::new(),
            selected_index: None,
            dragging_index: None,
            total_duration: 90.0,
            visible_duration: 30.0,
            scroll_offset: 0.0,
        }
    }
}

impl Timeline {
    /// Add a keyframe.
    pub fn add_keyframe(&mut self, kf: Keyframe) {
        self.path.push(kf);
    }

    /// The keyframe the playhead is on, if it is on one.
    pub fn key_at(&self, time: f32) -> Option<usize> {
        self.path
            .frames()
            .iter()
            .position(|f| (f.time - time).abs() < KEYFRAME_EPSILON)
    }

    /// The time of the nearest keyframe before `time`.
    pub fn time_before(&self, time: f32) -> Option<f32> {
        self.path
            .frames()
            .iter()
            .rev()
            .map(|f| f.time)
            .find(|t| *t < time - KEYFRAME_EPSILON)
    }

    /// The time of the nearest keyframe after `time`.
    pub fn time_after(&self, time: f32) -> Option<f32> {
        self.path
            .frames()
            .iter()
            .map(|f| f.time)
            .find(|t| *t > time + KEYFRAME_EPSILON)
    }

    /// Where a new keyframe goes when the playhead is at `time`.
    pub fn free_time_at(&self, time: f32) -> f32 {
        let mut t = time;
        while self
            .path
            .frames()
            .iter()
            .any(|f| (f.time - t).abs() < KEYFRAME_EPSILON)
        {
            t += KEYFRAME_SPACING;
        }
        t
    }

    /// Delete the keyframe the playhead is on.
    pub fn delete_selected(&mut self) {
        if let Some(i) = self.selected_index.take() {
            self.path.remove(i);
        }
    }

    /// Re-sort the path. Nothing to preserve across it: selection is derived
    /// from the playhead, not from an index into this list.
    pub fn sort(&mut self) {
        self.path.sort();
    }

    /// Back to a pristine timeline: no path, default length and zoom.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// How much of the timeline is on screen, as a percentage: 100% is all of
    /// it, 50% is half, and smaller means zoomed further in.
    pub fn zoom_percent(&self) -> f32 {
        if self.total_duration > 0.0 {
            (self.visible_duration / self.total_duration) * 100.0
        } else {
            100.0
        }
    }

    /// Tightest zoom available — the strip showing `MIN_VISIBLE_DURATION`.
    pub fn min_zoom_percent(&self) -> f32 {
        ((MIN_VISIBLE_DURATION / self.total_duration) * 100.0).min(100.0)
    }

    /// Set the zoom by percentage, holding `anchor` (in seconds) still on screen.
    pub fn set_zoom_percent(&mut self, percent: f32, anchor: f32) {
        let percent = percent.clamp(self.min_zoom_percent(), 100.0);
        self.set_visible_duration(self.total_duration * (percent / 100.0), anchor);
    }

    /// The time at the centre of the current viewport — the natural anchor when
    /// zooming from a control rather than from the strip.
    pub fn viewport_centre(&self) -> f32 {
        self.scroll_offset + self.visible_duration / 2.0
    }

    /// Discard the authored path and re-base the timeline on a newly loaded
    /// replay. Destructive by design: the replay is the content, so keyframes
    /// authored against a different one do not carry over.
    pub fn reset_for_replay(&mut self, duration: f32) {
        self.path.clear();
        self.selected_index = None;
        self.dragging_index = None;
        self.scroll_offset = 0.0;
        self.visible_duration = duration.max(MIN_VISIBLE_DURATION);
        self.set_total_duration(duration);
    }

    /// Set the timeline's length, keeping the zoom window and scroll inside it.
    pub fn set_total_duration(&mut self, duration: f32) {
        self.total_duration = duration.max(MIN_VISIBLE_DURATION);
        self.visible_duration = self
            .visible_duration
            .clamp(MIN_VISIBLE_DURATION, self.total_duration);
        self.clamp_scroll();
    }

    /// Resize the zoom window, holding `anchor` (in seconds) still on screen.
    /// `anchor` is normally the time under the cursor, so zooming keeps the bit
    /// you are pointing at where it is.
    pub fn set_visible_duration(&mut self, visible: f32, anchor: f32) {
        let old = self.visible_duration;
        let new = visible.clamp(MIN_VISIBLE_DURATION, self.total_duration);
        if (new - old).abs() < f32::EPSILON {
            return;
        }
        // Fraction of the way across the viewport the anchor sits at.
        let frac = if old > 0.0 {
            ((anchor - self.scroll_offset) / old).clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.visible_duration = new;
        self.scroll_offset = anchor - frac * new;
        self.clamp_scroll();
    }

    fn clamp_scroll(&mut self) {
        let max = (self.total_duration - self.visible_duration).max(0.0);
        self.scroll_offset = self.scroll_offset.clamp(0.0, max);
    }

    /// Scroll the viewport so `time` is visible (centred if outside).
    pub fn scroll_to(&mut self, time: f32) {
        let visible_start = self.scroll_offset;
        let visible_end = visible_start + self.visible_duration;
        if time < visible_start || time > visible_end {
            let max = (self.total_duration - self.visible_duration).max(0.0);
            self.scroll_offset = (time - self.visible_duration / 2.0).clamp(0.0, max);
        }
    }

    pub fn format_time_label(&self, seconds: f32, show_decimals: bool) -> String {
        if seconds < 0.0 {
            return "0s".to_string();
        }
        let minutes = (seconds / 60.0).floor() as u32;
        let remaining_seconds = seconds % 60.0;
        if minutes > 0 {
            if show_decimals {
                format!("{}m{:05.2}s", minutes, remaining_seconds)
            } else {
                format!("{}m{:02.0}s", minutes, remaining_seconds.floor())
            }
        } else if show_decimals {
            format!("{:.2}s", remaining_seconds)
        } else {
            format!("{}s", remaining_seconds.floor())
        }
    }
}

// PlaybackController

/// Owns playback time, play/pause state, and playback mode.
pub struct PlaybackController {
    pub time: f32,
    pub playing: bool,
    pub speed: f32,
    pub last_update: Option<Instant>,
}

impl Default for PlaybackController {
    fn default() -> Self {
        Self {
            time: 0.0,
            playing: false,
            speed: 1.0,
            last_update: None,
        }
    }
}

impl PlaybackController {
    /// Advance `time` by wall-clock delta scaled by `speed` if currently playing.
    pub fn advance(&mut self) {
        if !self.playing {
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.last_update {
            // TODO: when game timescale hook is implemented, drive this from the game's
            // own delta time instead of wall-clock so playback stays in sync automatically.
            self.time += (now - last).as_secs_f32() * self.speed;
        }
        self.last_update = Some(now);
    }

    /// `end_time` is the total content duration, and the playhead rewinds to 0
    /// when it is at or past it.
    pub fn toggle(&mut self, end_time: f32) {
        self.playing = !self.playing;
        if self.playing {
            self.last_update = Some(Instant::now());
            if self.time >= end_time {
                self.time = 0.0;
            }
        }
    }

    pub fn seek(&mut self, time: f32) {
        self.time = time;
        self.last_update = Some(Instant::now());
    }
}

/// Tracks recording state and drives the capture pipeline in `game`.
pub struct RecordingSession {
    pub active: bool,
    /// The name typed by the user (doubles as the active recording name).
    pub name: String,
}

impl Default for RecordingSession {
    fn default() -> Self {
        Self {
            active: false,
            name: String::new(),
        }
    }
}

impl RecordingSession {
    /// Create the replay file and hand it to the capture pipeline.
    pub fn start(&mut self) -> anyhow::Result<()> {
        use crate::replay::Replay;
        let mut replay = Replay::new(&self.name);
        replay.create_file()?;
        capture::start(replay);
        self.active = true;
        Ok(())
    }

    /// Returns the name it was saved under, for the caller to offer back.
    pub fn stop(&mut self) -> Result<String, String> {
        let name = std::mem::take(&mut self.name);
        self.active = false;
        match capture::stop() {
            Some(mut replay) => match replay.finish() {
                Ok(()) => Ok(name),
                Err(e) => Err(format!("failed to finish {name}: {e}")),
            },
            None => Err(format!("recording {name} was not active")),
        }
    }
}

/// What drives the camera while the playhead moves. There is one camera object,
/// so only one thing can write it, and this is what that thing is.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum PlaybackCamera {
    #[default]
    Path,
    /// Needs a loaded replay to read a camera out of, so it is not offered
    /// without one.
    Pov,
}

impl PlaybackCamera {
    const ALL: [PlaybackCamera; 2] = [PlaybackCamera::Path, PlaybackCamera::Pov];

    fn label(self) -> &'static str {
        match self {
            PlaybackCamera::Path => "Path",
            PlaybackCamera::Pov => "Replay POV",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            PlaybackCamera::Path => "Play the keyframe path you authored",
            PlaybackCamera::Pov => "Play the camera the replay was shot with",
        }
    }
}

/// What the mouse flies, and what a new keyframe records.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Rig {
    #[default]
    Freecam,
    /// Keys placed here store the orbit rather than a world pose, so a run of
    /// them sweeps round the pivot at a constant radius.
    Orbit,
}

impl Rig {
    const ALL: [Rig; 2] = [Rig::Freecam, Rig::Orbit];

    fn label(self) -> &'static str {
        match self {
            Rig::Freecam => "Freecam",
            Rig::Orbit => "Orbit",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Rig::Freecam => "Fly with the game's own camera controls",
            Rig::Orbit => "Drag to circle the skater, shift+drag to pan, wheel to zoom",
        }
    }
}

/// The Keyframe window as a staging area. One buffer, so one dirty test:
/// framing with the mouse and typing a fov are edits to the same pending key.
struct StagedKey {
    original: Keyframe,
    edited: Keyframe,
}

impl StagedKey {
    fn load(key: Keyframe) -> Self {
        Self {
            original: key.clone(),
            edited: key,
        }
    }

    /// Where the camera sits and looks, on screen-visible tolerances. An orbit
    /// key compares as its orbit, since `resolve` rebuilds its pose from that.
    fn framing_differs(&self) -> bool {
        let (a, b) = (&self.original, &self.edited);
        match (a.orbit, b.orbit) {
            (Some(x), Some(y)) => !x.approx_eq(&y),
            // Fov is its own part below, so hold it equal here.
            (None, None) => !a.pose.approx_eq(&CameraPose {
                fov: a.pose.fov,
                ..b.pose
            }),
            _ => true,
        }
    }

    fn fov_differs(&self) -> bool {
        !fov_approx_eq(self.original.pose.fov, self.edited.pose.fov)
    }

    /// Compared exactly: these are typed rather than measured.
    fn fields_differ(&self) -> bool {
        self.original.time != self.edited.time || self.original.tension != self.edited.tension
    }

    fn is_dirty(&self) -> bool {
        self.framing_differs() || self.fov_differs() || self.fields_differ()
    }

    /// What is waiting to be applied. Only built when it is drawn.
    fn pending(&self) -> Option<String> {
        let mut parts = Vec::new();
        if self.framing_differs() {
            parts.push("framing");
        }
        if self.fov_differs() {
            parts.push("fov");
        }
        if self.original.time != self.edited.time {
            parts.push("time");
        }
        if self.original.tension != self.edited.tension {
            parts.push("tension");
        }
        (!parts.is_empty()).then(|| format!("● Not applied yet: {}", parts.join(", ")))
    }
}

/// A line of feedback for the user. Nobody reads the log file mid-session.
struct Status {
    text: String,
    error: bool,
    at: Instant,
}

/// The skater in a replay frame, as recorded: the root bone.
fn skater_position(frame: Option<&ReplayState>) -> Option<Vec3> {
    let pos = frame?.skaters.first()?.world_pos;
    Some(Vec3::new(pos[0], pos[1], pos[2]))
}

pub struct UiSystem {
    pub is_showing: bool,
    /// Was the panel on screen last frame? Only used to spot the frame it
    /// opens on, so focus can be cleared then.
    pub was_showing: bool,
    /// Is the camera the rig's, or playback's? Flying the rig takes it, moving
    /// the playhead hands it back.
    rig_owns_camera: bool,
    /// The pose we last wrote, read back off the game. Anything else there is
    /// the player flying, which is the only tell the freecam rig gives.
    last_applied: Option<CameraPose>,
    /// `playback.time` as `update` last left it, so a playhead that moved from
    /// anywhere else can be recognised as a scrub.
    last_seen_time: f32,
    /// The fov the game had when freecam was entered, held for as long as we
    /// are in it. `Some` means we are in freecam and owe the game its fov back.
    pub entry_fov: Option<f32>,
    pub timeline: Timeline,
    pub playback: PlaybackController,
    pub recording: RecordingSession,
    pub loaded_replay: Option<LoadedReplay>,
    pub load_name: String,
    replays: Vec<ReplayInfo>,
    replays_read: bool,
    /// Does moving the playhead by hand show the shot at it? Playing always does.
    pub preview_on_scrub: bool,
    pub playback_camera: PlaybackCamera,
    pub rig: Rig,
    pub orbit: OrbitCamera,
    staged: Option<StagedKey>,
    status: Option<Status>,
    last_logged_frame: usize,
}

impl Default for UiSystem {
    fn default() -> Self {
        Self {
            is_showing: true,
            was_showing: false,
            rig_owns_camera: true,
            last_applied: None,
            last_seen_time: 0.0,
            entry_fov: None,
            timeline: Timeline::default(),
            playback: PlaybackController::default(),
            recording: RecordingSession::default(),
            loaded_replay: None,
            load_name: String::new(),
            replays: Vec::new(),
            replays_read: false,
            preview_on_scrub: true,
            playback_camera: PlaybackCamera::default(),
            rig: Rig::default(),
            orbit: OrbitCamera {
                target: Vec3::ZERO,
                yaw: 0.0,
                pitch: 0.3,
                distance: DEFAULT_ORBIT_DISTANCE,
                aim_height: OrbitCamera::DEFAULT_AIM_HEIGHT,
            },
            staged: None,
            status: None,
            last_logged_frame: usize::MAX,
        }
    }
}

impl UiSystem {
    pub fn is_showing(&self) -> bool {
        self.is_showing && game::in_freecam()
    }

    pub fn toggle_showing(&mut self) {
        if !game::in_freecam() {
            return;
        }
        self.is_showing = !self.is_showing;
    }

    fn say(&mut self, text: impl Into<String>) {
        let text = text.into();
        log::info!("{text}");
        self.status = Some(Status {
            text,
            error: false,
            at: Instant::now(),
        });
    }

    /// Stays on screen until replaced or dismissed: an error nobody saw is an
    /// error that happens twice.
    fn fail(&mut self, text: impl Into<String>) {
        let text = text.into();
        log::error!("{text}");
        self.status = Some(Status {
            text,
            error: true,
            at: Instant::now(),
        });
    }

    /// Called when the window first draws, when a recording finishes, and from
    /// the refresh button.
    fn refresh_replays(&mut self) {
        self.replays = replay::list();
        self.replays_read = true;
    }

    /// Per-frame update: advance time, apply the interpolated camera pose, and
    /// return the replay frame to draw this tick, if any.
    pub fn update(&mut self) -> Option<ReplayState> {
        // Take the player's fov before anything below can write one.
        if game::in_freecam() && self.entry_fov.is_none() {
            self.entry_fov = game::fov();
        }

        // POV reads its camera out of a replay frame, so it cannot outlive one.
        if self.playback_camera == PlaybackCamera::Pov && self.loaded_replay.is_none() {
            self.playback_camera = PlaybackCamera::Path;
        }

        self.playback.advance();

        // The orbit rig announces itself in `update_orbit_input`. The freecam rig
        // has no input of ours to watch, so a camera that moved is the only tell.
        if (self.playback.time - self.last_seen_time).abs() > f32::EPSILON {
            self.rig_owns_camera = false;
        }
        self.last_seen_time = self.playback.time;
        if !self.playback.playing && self.player_moved_camera() {
            self.rig_owns_camera = true;
        }

        let previously_selected = self.timeline.selected_index;
        self.timeline.selected_index = self.timeline.key_at(self.playback.time);
        if self.timeline.selected_index != previously_selected {
            self.load_selected_orbit();
        }

        let end_time = self.playback_end_time();

        // Drawn whenever a replay is loaded: the recorded skater is what a shot
        // is framed against and what an orbit pivots on.
        let mut logged = self.last_logged_frame;
        let frame = match self.loaded_replay.as_ref() {
            Some(r) => {
                let frame_idx = r.frame_index_at(self.playback.time);
                if frame_idx != logged {
                    let player_pos = r
                        .frames
                        .get(frame_idx)
                        .and_then(|f| f.skaters.first())
                        .map(|s| {
                            format!(
                                "({:.1},{:.1},{:.1})",
                                s.world_pos[0], s.world_pos[1], s.world_pos[2]
                            )
                        })
                        .unwrap_or_else(|| "?".into());
                    log::debug!(
                        "[playback] frame={}/{} t={:.2}s player={}",
                        frame_idx,
                        r.frames.len(),
                        self.playback.time,
                        player_pos
                    );
                    logged = frame_idx;
                }
                r.sample_at(self.playback.time)
            }
            None => {
                logged = usize::MAX;
                None
            }
        };
        self.last_logged_frame = logged;

        // Camera
        if !capture::is_recording() && game::in_freecam() {
            if let Some(pose) = self.camera_for_tick(frame.as_ref()) {
                game::set_camera(&pose);
                // What the game holds, not what we asked for, so the next
                // comparison measures movement and not a clamp.
                self.last_applied = game::camera().or(Some(pose));
            } else {
                self.last_applied = None;
            }
        }

        self.restage();

        // Stop playing when we reach the end of all content.
        if self.playback.playing && end_time > 0.0 && self.playback.time >= end_time {
            self.playback.playing = false;
            // Make sure we end on the actual end keyframe.
            self.playback.time = end_time;
            log::debug!("[playback] reached end ({:.2}s), stopping", end_time);
        }

        frame
    }

    /// Who writes the camera this tick, and what they want it to be. `None` is
    /// nobody: the freecam rig is the game's own camera, left alone.
    fn camera_for_tick(&mut self, frame: Option<&ReplayState>) -> Option<CameraPose> {
        let skater = skater_position(frame);

        if self.playback_drives_camera() {
            match self.playback_pose(frame, skater) {
                Some(pose) => {
                    self.carry_rig_to(&pose, skater);
                    return Some(pose);
                }
                // A path of one key, or a playhead past the end of it. Nobody
                // holds the camera, so the rig has it back.
                None => self.rig_owns_camera = true,
            }
        }

        match self.rig {
            Rig::Freecam => None,
            Rig::Orbit => Some(
                self.orbit
                    .pose_around(self.orbit.pivot(skater), self.rig_fov()),
            ),
        }
    }

    /// POV always drives, being a watching mode. Under the path, playing always
    /// does and scrubbing is the user's choice.
    fn playback_drives_camera(&self) -> bool {
        self.watching() || self.playback.playing || (self.preview_on_scrub && !self.rig_owns_camera)
    }

    /// Is the camera somebody else's? POV flies what was recorded, so the rig
    /// and everything that authors against it has nothing to do.
    fn watching(&self) -> bool {
        self.playback_camera == PlaybackCamera::Pov
    }

    fn playback_pose(
        &self,
        frame: Option<&ReplayState>,
        skater: Option<Vec3>,
    ) -> Option<CameraPose> {
        match self.playback_camera {
            PlaybackCamera::Pov => frame.and_then(|f| f.camera),
            PlaybackCamera::Path => self.timeline.path.sample(self.playback.time, skater),
        }
    }

    /// Keep the orbit rig on the pose playback is showing, so taking over with
    /// the mouse carries on from the screen instead of snapping back.
    fn carry_rig_to(&mut self, pose: &CameraPose, skater: Option<Vec3>) {
        if self.rig != Rig::Orbit {
            return;
        }
        // An orbit key says what the orbit is. Anything else only says where
        // the camera is, so derive the orbit from that and the pivot.
        match self.timeline.path.orbit_at(self.playback.time) {
            Some(orbit) => self.orbit = orbit,
            None => {
                let pivot = self.orbit.pivot(skater);
                self.orbit = OrbitCamera::from_eye(pose.position, pivot, self.orbit.aim_height)
                    .with_yaw_near(self.orbit.yaw);
            }
        }
    }

    /// Has the camera left the pose we put it in? That is the player flying
    /// the game's own freecam, which is the freecam rig taking over.
    fn player_moved_camera(&self) -> bool {
        let (Some(last), Some(live)) = (self.last_applied, game::camera()) else {
            return false;
        };
        !last.approx_eq(&live)
    }

    /// The time at which all playback content ends, the later of the replay's
    /// length and the last keyframe.
    fn playback_end_time(&self) -> f32 {
        let replay_end = self
            .loaded_replay
            .as_ref()
            .map(|r| r.duration())
            .unwrap_or(0.0);
        let path_end = self
            .timeline
            .path
            .frames()
            .last()
            .map(|f| f.time)
            .unwrap_or(0.0);
        replay_end.max(path_end)
    }

    /// The staged key's, so typing in the field previews instead of the rig
    /// writing the stored value back over it. Else the game's own.
    fn rig_fov(&self) -> f32 {
        self.staged
            .as_ref()
            .map(|s| s.edited.pose.fov)
            .or_else(|| game::camera().map(|c| c.fov))
            .unwrap_or(DEFAULT_FOV)
    }

    /// Pivot the orbit around whatever the camera is looking at, so entering
    /// orbit mode does not jump the view.
    fn recentre_orbit(&mut self) {
        if let Some(pose) = game::camera() {
            self.orbit = OrbitCamera::from_pose(&pose, DEFAULT_ORBIT_DISTANCE)
                .with_yaw_near(self.orbit.yaw)
                .with_aim_height(self.orbit.aim_height);
        }
    }

    /// Keep the staging buffer pointed at the selected key, and feed the live
    /// camera into it.
    fn restage(&mut self) {
        let selected = self
            .timeline
            .selected_index
            .and_then(|i| self.timeline.path.get(i))
            .cloned();
        let Some(key) = selected else {
            self.staged = None;
            return;
        };
        // `original` no longer matching the stored key is the whole test for
        // "something else changed this": a drag, a commit, a replay load.
        match &self.staged {
            Some(staged) if staged.original == key => {}
            _ => self.staged = Some(StagedKey::load(key)),
        }

        // Framing with the mouse is an edit like any other. Only while the rig
        // holds the camera, or POV would stage the replay's camera over the key.
        if self.playback_drives_camera() || !game::in_freecam() {
            return;
        }
        let Some(live) = game::camera() else {
            return;
        };
        let orbit = self.orbit_for_new_keyframe();
        if let Some(staged) = self.staged.as_mut() {
            staged.edited.pose = live;
            staged.edited.orbit = orbit;
        }
    }

    /// The marker turns red on this, and the two buttons are enabled by it.
    fn selected_is_dirty(&self) -> bool {
        self.staged.as_ref().is_some_and(StagedKey::is_dirty)
    }

    fn pending_label(&self) -> Option<String> {
        self.staged.as_ref().and_then(StagedKey::pending)
    }

    fn commit_staged(&mut self) {
        let Some(staged) = self.staged.as_ref() else {
            return;
        };
        let edited = staged.edited.clone();
        let retimed = edited.time != staged.original.time;
        let Some(i) = self.timeline.selected_index else {
            return;
        };
        if let Some(key) = self.timeline.path.get_mut(i) {
            *key = edited.clone();
        }
        if retimed {
            self.timeline.sort();
            // The playhead is what selects a key, so a retimed key takes it along.
            self.playback.seek(edited.time);
            self.timeline.scroll_to(edited.time);
        }
    }

    /// Works the levers rather than writing a pose and hoping: an orbit key's
    /// framing goes back into the rig, and playback is handed the camera.
    fn discard_staged(&mut self) {
        let Some(staged) = self.staged.as_mut() else {
            return;
        };
        let key = staged.original.clone();
        staged.edited = key.clone();
        if let Some(orbit) = key.orbit {
            self.orbit = orbit;
        }
        self.rig_owns_camera = false;
        game::set_camera(&key.resolve(None));
    }

    /// What a keyframe placed now should remember of the orbit, if anything.
    fn orbit_for_new_keyframe(&self) -> Option<OrbitCamera> {
        (self.rig == Rig::Orbit).then_some(self.orbit)
    }

    /// Picking an orbit key while orbiting puts the mouse back where that key
    /// was, so it can be adjusted and overwritten.
    fn load_selected_orbit(&mut self) {
        if self.rig != Rig::Orbit {
            return;
        }
        let selected = self
            .timeline
            .selected_index
            .and_then(|i| self.timeline.path.get(i));
        if let Some(orbit) = selected.and_then(|kf| kf.orbit) {
            self.orbit = orbit;
        }
    }

    /// Drag orbits, shift+drag pans the pivot, wheel zooms. Only over the game
    /// view: input aimed at the timeline or a window stays with egui.
    pub fn update_orbit_input(&mut self, ctx: &eguiContext) {
        // Playback owns the camera, so dragging would only bank up a jump for
        // the moment it stops.
        if self.rig != Rig::Orbit
            || self.watching()
            || self.playback.playing
            || ctx.is_pointer_over_area()
            || ctx.is_using_pointer()
        {
            return;
        }

        let (dragging, shift, delta, scroll) = ctx.input(|i| {
            (
                i.pointer.primary_down() || i.pointer.secondary_down(),
                i.modifiers.shift,
                i.pointer.delta(),
                i.smooth_scroll_delta.y,
            )
        });

        let moved = dragging && delta != Vec2::ZERO;
        if moved && shift {
            let speed = self.orbit.distance * ORBIT_PAN_SPEED;
            self.orbit.pan(-delta.x * speed, delta.y * speed);
        } else if moved {
            self.orbit
                .rotate(-delta.x * ORBIT_ROTATE_SPEED, delta.y * ORBIT_ROTATE_SPEED);
        }
        if scroll != 0.0 {
            self.orbit.zoom((-scroll * ORBIT_ZOOM_SPEED).exp());
        }

        // Flying the rig is the input that takes the camera back off the path.
        if moved || scroll != 0.0 {
            self.rig_owns_camera = true;
        }
    }

    // egui panels

    /// Shared by the Replay window's button and F2, so neither fails in silence.
    pub fn toggle_recording(&mut self) {
        if self.recording.active {
            match self.recording.stop() {
                Ok(name) => {
                    // You just typed this name. Being asked for it again is the
                    // tool forgetting what it was told.
                    self.say(format!("saved replay {name}"));
                    self.load_name = name;
                }
                Err(e) => self.fail(e),
            }
            self.refresh_replays();
            return;
        }
        if self.loaded_replay.is_some() {
            self.fail("unload the replay before recording a new one");
            return;
        }
        if self.recording.name.trim().is_empty() {
            self.fail("name the recording first");
            return;
        }
        // The file is named after this, so the overwrite check above has to have
        // looked at the same string.
        self.recording.name = self.recording.name.trim().to_owned();
        match self.recording.start() {
            Ok(()) => self.say(format!("recording {}", self.recording.name)),
            Err(e) => self.fail(format!("could not start recording: {e}")),
        }
    }

    fn load_replay(&mut self) {
        let name = self.load_name.clone();
        match LoadedReplay::load(&name) {
            Ok(r) => {
                // The replay is the content: it sets the timeline's length and
                // clears a path authored against a different one.
                let discarded = self.timeline.path.len();
                self.timeline.reset_for_replay(r.duration());
                self.playback.seek(0.0);
                self.playback.playing = false;
                self.last_logged_frame = usize::MAX;
                let summary = format!(
                    "loaded {}, {:.1}s at {:.0} fps",
                    r.name,
                    r.duration(),
                    r.fps
                );
                self.loaded_replay = Some(r);
                // The one thing a fresh replay can do with nothing authored yet.
                self.playback_camera = PlaybackCamera::Pov;
                self.rig_owns_camera = false;
                match discarded {
                    0 => self.say(summary),
                    n => self.say(format!("{summary}. {n} keyframe(s) discarded")),
                }
            }
            Err(e) => {
                self.fail(format!("could not load {name}: {e}"));
                // The listing offered that name, so it is out of date.
                self.refresh_replays();
            }
        }
    }

    fn unload_replay(&mut self) {
        self.timeline.reset();
        self.playback.seek(0.0);
        self.playback.playing = false;
        self.last_logged_frame = usize::MAX;
        self.loaded_replay = None;
        self.say("unloaded the replay, timeline cleared");
    }

    pub fn update_replay_detail(&mut self, ctx: &eguiContext) {
        if !self.replays_read {
            self.refresh_replays();
        }

        // Acted on after the closure, so each handler can take all of `self`.
        let mut toggle_record = false;
        let mut load = false;
        let mut unload = false;
        let mut refresh = false;

        egui::Window::new("Replay")
            .default_pos(window_pos(ctx, REPLAY_WINDOW_HEIGHT, 0.62))
            .default_width(WINDOW_WIDTH)
            .collapsible(true)
            .resizable(false)
            .show(ctx, |ui| {
                let recording = self.recording.active;
                let loaded = self.loaded_replay.is_some();

                ui.label("Record");
                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.add_enabled_ui(!recording, |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.recording.name)
                                .desired_width(120.0),
                        );
                    });
                });

                let named = !self.recording.name.trim().is_empty();
                // Asked of the listing, not the filesystem: this runs every frame.
                let typed = self.recording.name.trim();
                let overwriting = !recording && self.replays.iter().any(|r| r.name == typed);
                let label = if recording {
                    "⏹ Stop recording"
                } else if overwriting {
                    "🔴 Record over it"
                } else {
                    "🔴 Record"
                };
                let width = ui.available_width();
                let button = ui.add_enabled(
                    recording || (named && !loaded),
                    egui::Button::new(label).min_size(Vec2::new(width, 24.0)),
                );
                // A disabled button with no reason given reads as broken.
                let button = if loaded {
                    button.on_disabled_hover_text(
                        "Unload the replay first: recording needs the live skater",
                    )
                } else if !named {
                    button.on_disabled_hover_text("Type a name first")
                } else if overwriting {
                    button.on_hover_text(format!(
                        "{} already exists and will be replaced",
                        self.recording.name.trim()
                    ))
                } else {
                    button
                };
                toggle_record = button.clicked();

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Load");
                    refresh = ui
                        .small_button("⟳")
                        .on_hover_text("Re-read the Replays folder")
                        .clicked();
                });

                if self.replays.is_empty() {
                    ui.weak("Nothing in the Replays folder yet.");
                } else {
                    let width = ui.available_width();
                    ui.add_enabled_ui(!loaded, |ui| {
                        let selected = if self.load_name.is_empty() {
                            "Pick a replay".to_owned()
                        } else {
                            self.load_name.clone()
                        };
                        egui::ComboBox::from_id_salt("ReplayPick")
                            .selected_text(selected)
                            .width(width - 24.0)
                            .show_ui(ui, |ui| {
                                for info in &self.replays {
                                    let label = format!(
                                        "{}  ({:.1}s, {} frames)",
                                        info.name,
                                        info.duration(),
                                        info.frame_count
                                    );
                                    ui.selectable_value(
                                        &mut self.load_name,
                                        info.name.clone(),
                                        label,
                                    );
                                }
                            });
                    });
                }

                ui.horizontal(|ui| {
                    load = ui
                        .add_enabled(
                            !loaded && !self.load_name.is_empty(),
                            egui::Button::new("📂 Load"),
                        )
                        .clicked();
                    if loaded {
                        unload = ui
                            .button("⏏ Unload")
                            .on_hover_text("Clears the keyframe timeline too")
                            .clicked();
                    }
                });

                if let Some(r) = &self.loaded_replay {
                    ui.weak(format!(
                        "▶ {} · {:.1}s · {:.0} fps",
                        r.name,
                        r.duration(),
                        r.fps
                    ));
                }
            });

        if refresh {
            self.refresh_replays();
        }
        if toggle_record {
            self.toggle_recording();
        }
        if load {
            self.load_replay();
        }
        if unload {
            self.unload_replay();
        }
    }

    pub fn update_keyframe_detail(&mut self, ctx: &eguiContext) {
        // Acted on after the closure, so each handler can take all of `self`.
        let mut apply = false;
        let mut reset = false;

        egui::Window::new("Keyframe")
            .default_pos(window_pos(ctx, KEYFRAME_WINDOW_HEIGHT, 0.18))
            .default_width(WINDOW_WIDTH)
            .collapsible(true)
            .resizable(false)
            .show(ctx, |ui| {
                let authoring = !self.watching();
                let orbiting = self.rig == Rig::Orbit && authoring;

                // Everything in here edits the staging buffer, never the stored
                // key. Apply is what commits it.
                if let Some(before) = self.staged.as_ref().map(|s| s.edited.clone()) {
                    let mut kf = before.clone();

                    ui.add_enabled_ui(authoring, |ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label("Time");
                                ui.add(DragValue::new(&mut kf.time).speed(0.01).suffix("s"));
                            });
                            // An orbit key's position comes from its orbit.
                            ui.add_enabled_ui(kf.orbit.is_none(), |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Position");
                                    ui.add(
                                        DragValue::new(&mut kf.pose.position.x)
                                            .speed(0.4)
                                            .prefix("x: "),
                                    );
                                    ui.add(
                                        DragValue::new(&mut kf.pose.position.y)
                                            .speed(0.4)
                                            .prefix("y: "),
                                    );
                                    ui.add(
                                        DragValue::new(&mut kf.pose.position.z)
                                            .speed(0.4)
                                            .prefix("z: "),
                                    );
                                })
                            });
                            ui.horizontal(|ui| {
                                ui.label("Rotation");
                                ui.add_enabled_ui(false, |ui| {
                                    ui.add(
                                        DragValue::new(&mut kf.pose.orientation.0)
                                            .speed(0.01)
                                            .prefix("x: "),
                                    );
                                    ui.add(
                                        DragValue::new(&mut kf.pose.orientation.1)
                                            .speed(0.01)
                                            .prefix("y: "),
                                    );
                                    ui.add(
                                        DragValue::new(&mut kf.pose.orientation.2)
                                            .speed(0.01)
                                            .prefix("z: "),
                                    );
                                    ui.add(
                                        DragValue::new(&mut kf.pose.orientation.3)
                                            .speed(0.01)
                                            .prefix("w: "),
                                    );
                                });
                            });
                            ui.horizontal(|ui| {
                                ui.label("FOV");
                                ui.add(
                                    DragValue::new(&mut kf.pose.fov)
                                        .speed(1.0)
                                        .range(15.0..=140.0),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("Path tension");
                                ui.add(
                                    DragValue::new(&mut kf.tension).speed(0.01).range(0.0..=1.0),
                                );
                            });
                            // These are the values the orbit rig flies, so with
                            // any other they are not editable.
                            if let Some(orbit) = kf.orbit.as_mut().filter(|_| orbiting) {
                                ui.separator();
                                ui.label("Orbit");
                                ui.horizontal(|ui| {
                                    ui.label("Distance");
                                    ui.add(
                                        DragValue::new(&mut orbit.distance)
                                            .speed(1.0)
                                            .range(OrbitCamera::MIN_DISTANCE..=f32::MAX),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Yaw");
                                    ui.drag_angle(&mut orbit.yaw);
                                    ui.label("Pitch");
                                    ui.drag_angle(&mut orbit.pitch);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Aim height");
                                    ui.add(
                                        DragValue::new(&mut orbit.aim_height)
                                            .speed(0.5)
                                            .range(0.0..=f32::MAX),
                                    );
                                })
                                .response
                                .on_hover_text(
                                    "This key's pivot height above the skater. A run of keys \
                                     interpolates it, so a shot can rise.",
                                );
                                // Re-clamp whatever the drag fields let through.
                                orbit.rotate(0.0, 0.0);
                            }
                        });
                    });

                    if kf != before {
                        // The fields drive the levers and `restage` feeds the
                        // buffer back, or the edit is overwritten by the old camera.
                        if let Some(orbit) = kf.orbit {
                            self.orbit = orbit;
                        }
                        if !self.playback_drives_camera() {
                            if let Some(live) = game::camera() {
                                // Only the fov unless the freecam rig is flying a
                                // world key, since the rig builds the rest itself.
                                let pose = if kf.orbit.is_none() && self.rig == Rig::Freecam {
                                    kf.pose
                                } else {
                                    CameraPose {
                                        fov: kf.pose.fov,
                                        ..live
                                    }
                                };
                                game::set_camera(&pose);
                            }
                        }
                        if let Some(staged) = self.staged.as_mut() {
                            staged.edited = kf;
                        }
                    }
                } else {
                    ui.label("Playhead is not on a keyframe.");
                    ui.weak("Click a marker, or step with ⏪ ⏩, to edit one.");
                }

                // The pivot and the mouse bindings belong to the orbit rig
                // rather than to any one key, so they stay reachable with
                // nothing selected - you need them to frame the first key.
                if orbiting {
                    ui.separator();
                    ui.label("Orbit pivot");
                    ui.horizontal(|ui| {
                        ui.label("Aim height");
                        ui.add(
                            DragValue::new(&mut self.orbit.aim_height)
                                .speed(0.5)
                                .range(0.0..=f32::MAX),
                        );
                    })
                    .response
                    .on_hover_text(
                        "The live rig's pivot height above the skater - the recorded position \
                         is the root bone, between the feet. A new key takes this; an \
                         existing key keeps its own, editable above.",
                    );
                    if ui
                        .button("🎯 Re-centre on camera")
                        .on_hover_text(
                            "Pivot on whatever the camera is looking at now. Only bites \
                             with no replay skater to orbit, as does panning.",
                        )
                        .clicked()
                    {
                        self.recentre_orbit();
                    }
                    ui.weak("drag orbit · shift+drag pan · wheel zoom");
                }

                // Always shown rather than appearing with the edit, so the layout
                // does not jump and the colour is the thing that changes.
                ui.separator();
                if let Some(pending) = self.pending_label() {
                    ui.colored_label(STALE_KEY_COLOR, pending);
                }
                let dirty = self.selected_is_dirty();
                // Green only while there is something to commit.
                let apply_fill = if dirty {
                    APPLY_COLOR
                } else {
                    ui.visuals().widgets.inactive.bg_fill
                };
                let width = ui.available_width();
                ui.add_enabled_ui(dirty, |ui| {
                    apply = ui
                        .add(
                            egui::Button::new("⬇ Apply")
                                .fill(apply_fill)
                                .min_size(Vec2::new(width, 26.0)),
                        )
                        .on_hover_text(concat!(
                            "Write the staged keyframe over the stored one: the framing, ",
                            "fov, time and tension above.",
                        ))
                        .on_disabled_hover_text(if authoring {
                            "Nothing staged: the keyframe already matches what is on screen"
                        } else {
                            POV_HINT
                        })
                        .clicked();
                    reset = ui
                        .add(
                            egui::Button::new("↩ Reset")
                                .fill(RESET_COLOR)
                                .min_size(Vec2::new(width, 22.0)),
                        )
                        .on_hover_text(
                            "Throw the staged changes away and put the camera back on the \
                             stored keyframe.",
                        )
                        .clicked();
                });
            });

        if (apply || reset) && self.selected_is_dirty() {
            if apply {
                self.commit_staged();
            } else {
                self.discard_staged();
            }
        }
    }

    pub fn update_timeline(&mut self, ctx: &eguiContext) {
        // Nothing in here selects anything: the strip moves the playhead and
        // `update` derives the selection from it.
        self.draw_timeline(ctx);
    }

    fn draw_timeline(&mut self, ctx: &eguiContext) {
        TopBottomPanel::bottom(egui::Id::new("TimelinePanel")).show(ctx, |ui| {
            // Two bars: what you are editing, then how you are looking at it.
            ui.horizontal_wrapped(|ui| {
                ui.label("Keyframe");

                // Under POV a key placed here is one you cannot see, framed with
                // a rig that is not moving anything.
                let authoring = !self.watching();

                ui.add_enabled_ui(authoring, |ui| {
                    if ui
                        .button("➕ Add")
                        .on_hover_text("Stamp the live camera as a new keyframe at the playhead")
                        .on_disabled_hover_text(POV_HINT)
                        .clicked()
                    {
                        if let Some(pose) = game::camera() {
                            let orbit = self.orbit_for_new_keyframe();
                            let time = self.timeline.free_time_at(self.playback.time);
                            self.timeline.add_keyframe(Keyframe {
                                time,
                                pose,
                                tension: 0.25,
                                orbit,
                            });
                            self.playback.seek(time);
                        }
                    }
                });

                ui.add_enabled_ui(authoring && self.timeline.selected_index.is_some(), |ui| {
                    if ui
                        .button("📋 Duplicate")
                        .on_disabled_hover_text(POV_HINT)
                        .clicked()
                    {
                        if let Some(i) = self.timeline.selected_index {
                            if let Some(kf) = self.timeline.path.get(i).cloned() {
                                let mut dup = kf;
                                dup.time = self.timeline.free_time_at(dup.time + KEYFRAME_SPACING);
                                let time = dup.time;
                                self.timeline.add_keyframe(dup);
                                // Follow the copy, so it is the selected key.
                                self.playback.seek(time);
                            }
                        }
                    }
                });

                ui.add_enabled_ui(self.timeline.selected_index.is_some(), |ui| {
                    if ui.button("❌ Delete").clicked() {
                        self.timeline.delete_selected();
                    }
                });

                ui.separator();

                let prev_key = self.timeline.time_before(self.playback.time);
                if ui
                    .add_enabled(prev_key.is_some(), egui::Button::new("⏪ Previous"))
                    .clicked()
                {
                    if let Some(time) = prev_key {
                        self.playback.seek(time);
                        self.timeline.scroll_to(time);
                    }
                }

                let play_pause_icon = if self.playback.playing {
                    "⏸ Pause"
                } else {
                    "▶ Play"
                };
                if ui.button(play_pause_icon).clicked() {
                    self.playback.toggle(self.playback_end_time());
                }

                let next_key = self.timeline.time_after(self.playback.time);
                if ui
                    .add_enabled(next_key.is_some(), egui::Button::new("⏩ Next"))
                    .clicked()
                {
                    if let Some(time) = next_key {
                        self.playback.seek(time);
                        self.timeline.scroll_to(time);
                    }
                }

                ui.separator();
                ui.label("Speed");
                ui.add(
                    DragValue::new(&mut self.playback.speed)
                        .speed(0.01)
                        .range(0.01..=4.0)
                        .suffix("x"),
                );
            });

            ui.horizontal_wrapped(|ui| {
                // Two independent questions, so two controls. Bundled, picking a
                // playback camera silently changed what a new key recorded.
                ui.label("Plays")
                    .on_hover_text("What drives the camera while the playhead moves");
                let pov_offered = game::is_replay_available();
                let pov_ready = self.loaded_replay.is_some();
                egui::ComboBox::from_id_salt("PlaybackCamera")
                    .selected_text(self.playback_camera.label())
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        for source in PlaybackCamera::ALL {
                            if source == PlaybackCamera::Pov {
                                if !pov_offered {
                                    continue;
                                }
                                if !pov_ready {
                                    ui.add_enabled_ui(false, |ui| {
                                        ui.selectable_label(false, source.label())
                                            .on_disabled_hover_text("Load a replay first");
                                    });
                                    continue;
                                }
                            }
                            ui.selectable_value(&mut self.playback_camera, source, source.label())
                                .on_hover_text(source.hint());
                        }
                    });

                // Kept, not reset: going back to the path restores this rig.
                let flying = !self.watching();
                ui.add_enabled_ui(flying, |ui| {
                    ui.label("Fly")
                        .on_hover_text("What the mouse moves, and what a new keyframe records")
                        .on_disabled_hover_text(POV_HINT);
                });
                let prev_rig = self.rig;
                ui.add_enabled_ui(flying, |ui| {
                    egui::ComboBox::from_id_salt("Rig")
                        .selected_text(self.rig.label())
                        .width(80.0)
                        .show_ui(ui, |ui| {
                            for rig in Rig::ALL {
                                ui.selectable_value(&mut self.rig, rig, rig.label())
                                    .on_hover_text(rig.hint());
                            }
                        });
                });
                if self.rig == Rig::Orbit && prev_rig != Rig::Orbit {
                    // Pick the rig up where the camera already looks, so
                    // switching to it does not jump the view.
                    self.recentre_orbit();
                    self.load_selected_orbit();
                }

                ui.add_enabled_ui(flying, |ui| {
                    if ui
                        .checkbox(&mut self.preview_on_scrub, "Preview on scrub")
                        .on_hover_text(concat!(
                            "Dragging the playhead shows the shot at it. Off, the camera stays ",
                            "where you left it while the playhead moves. Playing always previews.",
                        ))
                        .on_disabled_hover_text(POV_HINT)
                        .changed()
                        && !self.preview_on_scrub
                    {
                        self.rig_owns_camera = true;
                    }
                });

                ui.separator();

                ui.label("Duration");
                let mut total = self.timeline.total_duration;
                if ui
                    .add(
                        DragValue::new(&mut total)
                            .speed(1.0)
                            .range(MIN_VISIBLE_DURATION..=f32::MAX)
                            .suffix("s"),
                    )
                    .changed()
                {
                    self.timeline.set_total_duration(total);
                }

                ui.separator();

                ui.label("Zoom");
                let mut zoom = self.timeline.zoom_percent();
                if ui
                    .add(
                        DragValue::new(&mut zoom)
                            .speed(0.5)
                            .range(self.timeline.min_zoom_percent()..=100.0)
                            .fixed_decimals(1)
                            .suffix("%"),
                    )
                    .changed()
                {
                    let centre = self.timeline.viewport_centre();
                    self.timeline.set_zoom_percent(zoom, centre);
                }
                if ui.button("Fit").clicked() {
                    self.timeline.set_zoom_percent(100.0, 0.0);
                }

                ui.separator();

                // Takes the rest of the bar: a scrollbar you can actually aim at.
                ui.label("Scroll");
                let max_scroll =
                    (self.timeline.total_duration - self.timeline.visible_duration).max(0.0);
                ui.add_enabled(
                    max_scroll > 0.0,
                    egui::Slider::new(&mut self.timeline.scroll_offset, 0.0..=max_scroll)
                        .show_value(false),
                );
            });

            // Space belongs in a replay name, and Delete to the caret.
            let (space, delete) = if ctx.wants_keyboard_input() {
                (false, false)
            } else {
                ui.input(|i| {
                    (
                        i.key_pressed(egui::Key::Space),
                        i.key_pressed(egui::Key::Delete),
                    )
                })
            };
            if space {
                self.playback.toggle(self.playback_end_time());
            }
            if delete && self.timeline.selected_index.is_some() {
                self.timeline.delete_selected();
            }

            let pixels_per_second = ui.available_width() / self.timeline.visible_duration;
            let timeline_height = 50.0;
            let (timeline_rect, timeline_response) = ui.allocate_exact_size(
                Vec2::new(ui.available_width(), timeline_height),
                Sense::click_and_drag(),
            );
            let painter = ui.painter_at(timeline_rect);
            let timeline_top = timeline_rect.top();
            let timeline_left = timeline_rect.left();
            painter.rect_filled(timeline_rect, 0.0, Color32::DARK_GRAY);

            let visible_start = self.timeline.scroll_offset;
            let visible_end = visible_start + self.timeline.visible_duration;
            let start_second = visible_start.floor() as i32;
            let end_second = visible_end.ceil() as i32;
            let time_to_x = |t: f32| timeline_left + (t - visible_start) * pixels_per_second;

            for second_i32 in start_second..=end_second {
                let second = second_i32 as f32;
                if second < 0.0 {
                    continue;
                }
                let x = time_to_x(second);
                if x >= timeline_left && x <= timeline_rect.right() {
                    painter.line_segment(
                        [
                            Pos2::new(x, timeline_top),
                            Pos2::new(x, timeline_top + 10.0),
                        ],
                        (1.0, Color32::WHITE),
                    );
                    let time_text = self.timeline.format_time_label(second, false);
                    painter.text(
                        Pos2::new(x + 2.0, timeline_top + 12.0),
                        egui::Align2::LEFT_TOP,
                        time_text,
                        egui::FontId::monospace(10.0),
                        Color32::WHITE,
                    );
                }
            }

            let dirty = self.selected_is_dirty();
            let mut needs_sort = false;
            for i in 0..self.timeline.path.len() {
                let kf_time = {
                    let frames = self.timeline.path.frames();
                    let t = frames[i].time;
                    if t < visible_start || t > visible_end {
                        continue;
                    }
                    t
                };

                let x = time_to_x(kf_time);
                let kf_top = timeline_top + 24.0;
                let kf_bottom = timeline_rect.bottom() - 3.0;
                let color = if self.timeline.selected_index == Some(i) {
                    // Only the selected key has a staging buffer to be dirty.
                    if dirty {
                        STALE_KEY_COLOR
                    } else {
                        Color32::YELLOW
                    }
                } else if self.timeline.path.frames()[i].orbit.is_some() {
                    Color32::from_rgb(255, 160, 60)
                } else {
                    Color32::LIGHT_BLUE
                };
                let line_rect =
                    Rect::from_min_max(Pos2::new(x - 3.0, kf_top), Pos2::new(x + 3.0, kf_bottom));
                let response = ui.allocate_rect(line_rect, Sense::click_and_drag());
                painter.rect_filled(line_rect, 4.0, color);

                // Going to a key is selecting it, so a click is a seek.
                if response.clicked() {
                    self.playback.seek(kf_time);
                }
                if response.drag_started() {
                    self.timeline.dragging_index = Some(i);
                }
                if response.dragged() && self.timeline.dragging_index == Some(i) {
                    if let Some(kf) = self.timeline.path.get_mut(i) {
                        kf.time += response.drag_delta().x / pixels_per_second;
                        kf.time = kf.time.clamp(0.0, self.timeline.total_duration);
                        // Carry the playhead along, or dragging a key would
                        // slide it out from under the selection.
                        self.playback.time = kf.time;
                    }
                    self.playback.last_update = Some(Instant::now());
                }
                if response.drag_stopped() {
                    needs_sort = true;
                }
            }
            if needs_sort {
                self.timeline.sort();
            }

            // Draw playhead.
            let pb_time = self.playback.time;
            if pb_time >= visible_start && pb_time <= visible_end {
                let playhead_x = time_to_x(pb_time);
                painter.line_segment(
                    [
                        Pos2::new(playhead_x, timeline_rect.top()),
                        Pos2::new(playhead_x, timeline_rect.bottom()),
                    ],
                    (2.0, Color32::GREEN),
                );
                let time_text = self.timeline.format_time_label(pb_time, true);
                painter.text(
                    Pos2::new(playhead_x + 4.0, timeline_rect.top()),
                    egui::Align2::LEFT_TOP,
                    time_text,
                    egui::FontId::monospace(12.0),
                    Color32::GREEN,
                );
            }

            // Wheel over the strip zooms around the time under the cursor.
            if timeline_response.hovered() {
                let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_y != 0.0 {
                    let anchor = ui
                        .input(|i| i.pointer.hover_pos())
                        .map(|p| ((p.x - timeline_left) / pixels_per_second) + visible_start)
                        .unwrap_or(visible_start + self.timeline.visible_duration / 2.0);
                    // Wheel up zooms in (less of the timeline on screen), down out.
                    let step = if scroll_y > 0.0 { 1.0 / 1.15 } else { 1.15 };
                    let zoom = self.timeline.zoom_percent() * step;
                    self.timeline.set_zoom_percent(zoom, anchor);
                }
            }

            if timeline_response.clicked() || timeline_response.dragged() {
                if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let relative_x = pointer_pos.x - timeline_rect.left();
                    let new_time = (relative_x / pixels_per_second) + visible_start;
                    self.playback
                        .seek(new_time.clamp(0.0, self.timeline.total_duration));
                }
            }
            if timeline_response.dragged() {
                self.playback.playing = false;
            }

            self.draw_status(ui);
        });
    }

    /// The one place on screen that says what went wrong.
    fn draw_status(&mut self, ui: &mut egui::Ui) {
        // Info lines expire, errors stay until replaced or dismissed.
        if self
            .status
            .as_ref()
            .is_some_and(|s| !s.error && s.at.elapsed() > STATUS_FADE)
        {
            self.status = None;
        }
        let Some(status) = self.status.as_ref() else {
            return;
        };

        let color = if status.error {
            STATUS_ERROR_COLOR
        } else {
            Color32::LIGHT_GREEN
        };
        let mut dismiss = false;
        ui.horizontal(|ui| {
            dismiss = ui.small_button("✕").clicked();
            ui.colored_label(color, &status.text);
        });
        if dismiss {
            self.status = None;
        }
    }
}

// Render loop (called by egui-d3d9 every Present)

pub fn ui_render_loop(ctx: &eguiContext, _i: &mut i32) {
    let mut ui = UI_SYSTEM
        .get_or_init(|| Mutex::new(UiSystem::default()))
        .lock()
        .unwrap();

    if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
        ui.toggle_showing();
    }

    ctx.request_repaint();

    // We have to restore games FOV.
    if !game::in_freecam() {
        if let Some(fov) = ui.entry_fov.take() {
            log::debug!("[camera] left freecam, restoring fov {:.1}", fov);
            game::set_fov(fov);
        }
    }

    // A backend with stubbed capture gets no recording UI at all: not the F2
    // shortcut, not the overlay, not the Replay window.
    let replay_available = game::is_replay_available();

    if replay_available {
        if ctx.input(|i| i.key_pressed(egui::Key::F2)) && !ctx.wants_keyboard_input() {
            ui.toggle_recording();
        }

        if ui.recording.active {
            ctx.debug_painter().text(
                Pos2::new(0., 40.),
                Align2::LEFT_TOP,
                format!("Recording: {}", ui.recording.name),
                FontId::default(),
                Color32::LIGHT_RED,
            );
            ctx.debug_painter().text(
                Pos2::new(0., 60.),
                Align2::LEFT_TOP,
                "Press F2 to stop recording.",
                FontId::default(),
                Color32::WHITE,
            );
        }
    }

    let has_camera = game::camera().is_some();
    if !ui.is_showing() {
        ui.was_showing = false;
        return;
    }
    let just_opened = !std::mem::replace(&mut ui.was_showing, true);

    // No camera yet means freecam was never entered, so there is nothing to
    // author against. Nothing above here may paint: see CLAUDE.md, UI layout.
    if has_camera {
        ctx.debug_painter().text(
            Pos2::new(0., 0.),
            Align2::LEFT_TOP,
            version_string(),
            FontId::default(),
            Color32::LIGHT_YELLOW,
        );
        ui.update_timeline(ctx);
        if replay_available {
            ui.update_replay_detail(ctx);
        }
        ui.update_keyframe_detail(ctx);
        ui.update_orbit_input(ctx);
        draw_keys_window(ctx, replay_available);
    }

    // We don't want to autofocus any text inputs.
    if just_opened {
        ctx.memory_mut(|m| m.stop_text_input());
    }

    draw_cursor(ctx);
}

/// Collapsed by default, so it costs one line once you know them.
fn draw_keys_window(ctx: &eguiContext, replay_available: bool) {
    egui::Window::new("Keys")
        .default_pos(window_pos(ctx, KEYS_WINDOW_HEIGHT, 0.02))
        .default_width(WINDOW_WIDTH)
        .default_open(false)
        .resizable(false)
        .show(ctx, |ui| {
            fn key(ui: &mut egui::Ui, key: &str, what: &str) {
                ui.horizontal(|ui| {
                    ui.monospace(key);
                    ui.weak(what);
                });
            }
            key(ui, "F1", "show or hide this panel");
            if replay_available {
                key(ui, "F2", "start or stop recording");
            }
            key(ui, "Space", "play or pause");
            key(ui, "Delete", "delete the selected keyframe");
            ui.separator();
            ui.weak("Orbit rig");
            key(ui, "drag", "circle the pivot");
            key(ui, "shift+drag", "pan the pivot");
            key(ui, "wheel", "zoom");
            ui.separator();
            ui.weak("Timeline");
            key(ui, "wheel", "zoom around the cursor");
            key(ui, "click", "move the playhead");
        });
}

/// Use egui cursor.
fn draw_cursor(ctx: &eguiContext) {
    let Some(tip) = ctx.pointer_latest_pos() else {
        return;
    };
    if ctx.output(|o| o.cursor_icon) == CursorIcon::None {
        return;
    }

    const OUTLINE: [[f32; 2]; 7] = [
        [0.0, 0.0],
        [0.0, 17.0],
        [4.2, 13.2],
        [6.8, 18.8],
        [9.6, 17.5],
        [7.0, 12.0],
        [12.0, 12.0],
    ];
    let p = |i: usize| tip + Vec2::new(OUTLINE[i][0], OUTLINE[i][1]);

    let painter = ctx.layer_painter(LayerId::new(Order::Debug, Id::new("falke_cursor")));

    for piece in [
        vec![p(0), p(1), p(2)],
        vec![p(0), p(2), p(5)],
        vec![p(0), p(5), p(6)],
        vec![p(2), p(3), p(4), p(5)],
    ] {
        painter.add(Shape::convex_polygon(piece, Color32::WHITE, Stroke::NONE));
    }
    painter.add(Shape::closed_line(
        (0..OUTLINE.len()).map(p).collect(),
        Stroke::new(1.0f32, Color32::BLACK),
    ));
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;

    fn kf(time: f32) -> Keyframe {
        Keyframe {
            time,
            pose: CameraPose {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                fov: 90.0,
            },
            tension: 0.25,
            orbit: None,
        }
    }

    /// Adding twice without moving the playhead - the normal way to frame two
    /// orbit angles in a row - must not stack both keys on one time, or the
    /// path has no span to interpolate across.
    #[test]
    fn adding_twice_at_one_playhead_time_leaves_a_span() {
        let mut timeline = Timeline::default();

        let first = timeline.free_time_at(0.0);
        timeline.add_keyframe(kf(first));
        let second = timeline.free_time_at(0.0);
        timeline.add_keyframe(kf(second));

        assert_eq!(first, 0.0);
        assert_eq!(second, KEYFRAME_SPACING);
        assert!(timeline.path.sample(KEYFRAME_SPACING / 2.0, None).is_some());
    }

    /// Selection is a question about the playhead, not a stored index, so a
    /// time between keys has nothing selected and the transport is measured
    /// from the playhead too.
    #[test]
    fn selection_is_whichever_key_the_playhead_is_on() {
        let mut timeline = Timeline::default();
        timeline.add_keyframe(kf(0.0));
        timeline.add_keyframe(kf(2.0));

        assert_eq!(timeline.key_at(0.0), Some(0));
        assert_eq!(timeline.key_at(2.0), Some(1));
        assert_eq!(timeline.key_at(1.0), None);

        assert_eq!(timeline.time_after(0.0), Some(2.0));
        assert_eq!(timeline.time_after(1.0), Some(2.0));
        assert_eq!(timeline.time_after(2.0), None);
        assert_eq!(timeline.time_before(2.0), Some(0.0));
        assert_eq!(timeline.time_before(1.0), Some(0.0));
        assert_eq!(timeline.time_before(0.0), None);
    }

    /// A key added before the existing ones is still the one the playhead ends
    /// up on - the old code selected `len - 1`, which `push`'s sort had made
    /// somebody else.
    #[test]
    fn adding_out_of_order_selects_the_new_key() {
        let mut timeline = Timeline::default();
        timeline.add_keyframe(kf(5.0));
        timeline.add_keyframe(kf(1.0));

        assert_eq!(timeline.key_at(1.0), Some(0));
        assert_eq!(timeline.path.len(), 2);
    }

    #[test]
    fn free_time_at_steps_past_every_taken_slot() {
        let mut timeline = Timeline::default();
        for t in [0.0, KEYFRAME_SPACING, 2.0 * KEYFRAME_SPACING] {
            timeline.add_keyframe(kf(t));
        }

        assert_eq!(timeline.free_time_at(0.0), 3.0 * KEYFRAME_SPACING);
        // An untaken time is left alone.
        assert_eq!(timeline.free_time_at(10.0), 10.0);
    }
}
