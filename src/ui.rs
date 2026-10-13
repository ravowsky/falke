use egui::{
    Align2, Color32, Context as eguiContext, CursorIcon, DragValue, FontId, Id, LayerId, Order,
    Pos2, Rect, Sense, Shape, Stroke, TopBottomPanel, Vec2,
};
use std::{
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU8, Ordering},
    },
    time::Instant,
};

use crate::{
    camera::{CameraPath, CameraPose, Keyframe, OrbitCamera, fov_approx_eq},
    capture,
    falke::version_string,
    game,
    math::Vec3,
    replay::{LoadedReplay, ReplayState},
};

pub static UI_SYSTEM: OnceLock<Mutex<UiSystem>> = OnceLock::new();

/// Narrowest the zoom window may get, in seconds.
const MIN_VISIBLE_DURATION: f32 = 0.5;

/// Fov for a pose built before the game has told us what the camera uses.
const DEFAULT_FOV: f32 = 90.0;

/// A selected keyframe the camera has since moved off.
const STALE_KEY_COLOR: Color32 = Color32::from_rgb(255, 95, 95);

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
    pub mode_active: bool,
    pub speed: f32,
    pub last_update: Option<Instant>,
}

impl Default for PlaybackController {
    fn default() -> Self {
        Self {
            time: 0.0,
            playing: false,
            mode_active: false,
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

    /// Toggle play/pause. `end_time` is the total content duration; rewinds to 0
    /// when the playhead is at or past it.
    pub fn toggle(&mut self, end_time: f32) {
        self.playing = !self.playing;
        if self.playing {
            self.last_update = Some(Instant::now());
            self.mode_active = true;
            if self.time >= end_time {
                self.time = 0.0;
            }
        }
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

    /// Finish the replay file and stop capturing.
    pub fn stop(&mut self) {
        if let Some(mut replay) = capture::stop() {
            if let Err(e) = replay.finish() {
                log::error!("failed to finish replay: {e}");
            }
        }
        self.active = false;
        self.name.clear();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum CameraMode {
    /// You fly the camera, and the authored keyframe path moves it during
    /// playback. The camera the replay was shot with is ignored.
    #[default]
    Freecam,
    /// Play back through the camera the replay was recorded with. The authored
    /// path is ignored.
    Pov,
    /// Drag to circle a pivot, which may be a fixed point or the replay's
    /// skater. Keys placed here store the orbit rather than a world pose, so a
    /// run of them sweeps round the pivot at a constant radius. Paused, the
    /// mouse owns the camera, playing, the path does.
    Orbit,
}

impl CameraMode {
    const ALL: [CameraMode; 3] = [CameraMode::Freecam, CameraMode::Pov, CameraMode::Orbit];

    fn label(self) -> &'static str {
        match self {
            CameraMode::Freecam => "Freecam",
            CameraMode::Pov => "POV",
            CameraMode::Orbit => "Orbit",
        }
    }
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
    /// Orbit mode only: is the camera the rig's or the path's?
    rig_owns_camera: bool,
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
    pub camera_mode: CameraMode,
    pub orbit: OrbitCamera,
    last_logged_frame: usize,
}

impl Default for UiSystem {
    fn default() -> Self {
        Self {
            is_showing: true,
            was_showing: false,
            rig_owns_camera: true,
            last_seen_time: 0.0,
            entry_fov: None,
            timeline: Timeline::default(),
            playback: PlaybackController::default(),
            recording: RecordingSession::default(),
            loaded_replay: None,
            load_name: String::new(),
            camera_mode: CameraMode::default(),
            orbit: OrbitCamera {
                target: Vec3::ZERO,
                yaw: 0.0,
                pitch: 0.3,
                distance: DEFAULT_ORBIT_DISTANCE,
                aim_height: OrbitCamera::DEFAULT_AIM_HEIGHT,
            },
            last_logged_frame: usize::MAX,
        }
    }
}

impl UiSystem {
    /// Whether the overlay (F1) is switched on. The Replay window follows this in
    /// any camera mode; the authoring panels additionally need the freecam.
    pub fn is_showing(&self) -> bool {
        self.is_showing
    }

    pub fn toggle_showing(&mut self) {
        self.is_showing = !self.is_showing;
    }

    /// Per-frame update: advance time, apply the interpolated camera pose, and
    /// return the replay frame to draw this tick, if any.
    pub fn update(&mut self) -> Option<ReplayState> {
        // Take the player's fov before anything below can write one.
        if game::in_freecam() && self.entry_fov.is_none() {
            self.entry_fov = game::fov();
        }

        // POV plays back the camera a replay was shot with, so it cannot
        // outlive the replay: unloading one drops the mode with it.
        if self.camera_mode == CameraMode::Pov && self.loaded_replay.is_none() {
            self.camera_mode = CameraMode::Freecam;
        }

        // Where playback mode is not a choice, it is not state either.
        if self.playback_mode_is_implied() {
            self.playback.mode_active = true;
        }

        self.playback.advance();

        if (self.playback.time - self.last_seen_time).abs() > f32::EPSILON {
            self.rig_owns_camera = false;
        }
        self.last_seen_time = self.playback.time;

        let previously_selected = self.timeline.selected_index;
        self.timeline.selected_index = self.timeline.key_at(self.playback.time);
        if self.timeline.selected_index != previously_selected {
            self.load_selected_orbit();
        }

        let end_time = self.playback_end_time();

        // Replay playback
        let frame = if self.playback.mode_active {
            self.loaded_replay.as_ref().and_then(|r| {
                let t = self.playback.time * 60.0;
                let frame_idx = (t as usize).min(r.frames.len().saturating_sub(1));
                if frame_idx != self.last_logged_frame {
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
                    self.last_logged_frame = frame_idx;
                }
                r.sample_with(t, game::blend_skaters)
            })
        } else {
            self.last_logged_frame = usize::MAX;
            None
        };

        // The game world (traffic, pedestrians, rain, cranes) runs at the playback speed
        // while a replay plays, stands (nearly) still while it is paused in the freecam, and
        // runs at normal speed otherwise.
        let world_speed = if self.playback.mode_active && self.playback.playing {
            self.playback.speed
        } else if self.playback.mode_active && self.loaded_replay.is_some() && game::in_freecam() {
            0.0
        } else {
            1.0
        };
        game::set_time_scale(world_speed);

        // Camera
        if !capture::is_recording() && game::in_freecam() {
            let skater = skater_position(frame.as_ref());
            let pose: Option<CameraPose> = match self.camera_mode {
                // `frame` is only Some in playback mode, which gates this.
                CameraMode::Pov => frame.as_ref().and_then(|f| f.camera),
                CameraMode::Freecam => {
                    if self.playback.mode_active {
                        self.timeline.path.sample(self.playback.time, skater)
                    } else {
                        None
                    }
                }
                // The path runs while it is being played or scrubbed, the rig
                // runs while it is being flown. See `rig_owns_camera`.
                CameraMode::Orbit => {
                    let path_driving = self.playback.mode_active
                        && (self.playback.playing || !self.rig_owns_camera);
                    let from_path = path_driving
                        .then(|| self.timeline.path.sample(self.playback.time, skater))
                        .flatten();

                    match from_path {
                        Some(pose) => {
                            if let Some(orbit) = self.timeline.path.orbit_at(self.playback.time) {
                                self.orbit = orbit;
                            }
                            Some(pose)
                        }
                        None => Some(
                            self.orbit
                                .pose_around(self.orbit.pivot(skater), self.rig_fov()),
                        ),
                    }
                }
            };
            if let Some(pose) = pose {
                game::set_camera(&pose);
            }
        }

        // Stop playing when we reach the end of all content.
        if self.playback.playing && end_time > 0.0 && self.playback.time >= end_time {
            self.playback.playing = false;
            // Make sure we end on the actual end keyframe.
            self.playback.time = end_time;
            log::debug!("[playback] reached end ({:.2}s), stopping", end_time);
        }

        frame
    }

    /// The time at which all playback content ends — max of replay duration and last keyframe.
    fn playback_end_time(&self) -> f32 {
        let replay_end = self
            .loaded_replay
            .as_ref()
            .map(|r| r.frame_count as f32 / 60.0)
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

    /// Is playback mode forced on rather than being the user's to pick?
    fn playback_mode_is_implied(&self) -> bool {
        self.loaded_replay.is_some()
            && matches!(self.camera_mode, CameraMode::Pov | CameraMode::Orbit)
    }

    /// The fov the orbit rig flies at.
    fn rig_fov(&self) -> f32 {
        self.timeline
            .selected_index
            .and_then(|i| self.timeline.path.get(i))
            .map(|k| k.pose.fov)
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

    fn apply_would_change_selected(&self) -> bool {
        if self.camera_mode == CameraMode::Pov {
            return false;
        }
        let Some(kf) = self
            .timeline
            .selected_index
            .and_then(|i| self.timeline.path.get(i))
        else {
            return false;
        };
        let Some(live) = game::camera() else {
            return false;
        };

        match (kf.orbit, self.orbit_for_new_keyframe()) {
            (Some(stored), Some(rig)) => {
                !stored.approx_eq(&rig) || !fov_approx_eq(kf.pose.fov, live.fov)
            }
            (None, None) => !kf.pose.approx_eq(&live),
            _ => true,
        }
    }

    /// Apply the camera to the selected keyframe.
    fn apply_camera_to_selected(&mut self) {
        let Some(i) = self.timeline.selected_index else {
            return;
        };
        let Some(pose) = game::camera() else {
            return;
        };
        let orbit = self.orbit_for_new_keyframe();
        if let Some(kf) = self.timeline.path.get_mut(i) {
            kf.pose = pose;
            kf.orbit = orbit;
        }
    }

    /// Resets the camera to the selected keyframe data.
    fn reset_camera_to_selected(&mut self) {
        let Some(kf) = self
            .timeline
            .selected_index
            .and_then(|i| self.timeline.path.get(i))
            .cloned()
        else {
            return;
        };

        if let Some(orbit) = kf.orbit {
            self.orbit = orbit;
        }
        self.rig_owns_camera = false;
        game::set_camera(&kf.resolve(None));
    }

    /// What a keyframe placed now should remember of the orbit, if anything.
    fn orbit_for_new_keyframe(&self) -> Option<OrbitCamera> {
        (self.camera_mode == CameraMode::Orbit).then_some(self.orbit)
    }

    /// Picking an orbit key while orbiting puts the mouse back where that key
    /// was, so it can be adjusted and overwritten.
    fn load_selected_orbit(&mut self) {
        if self.camera_mode != CameraMode::Orbit {
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
        // Playing, the path owns the camera; dragging then would only bank up
        // a jump for the moment it pauses.
        if self.camera_mode != CameraMode::Orbit
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

    pub fn update_replay_detail(&mut self, ctx: &eguiContext) {
        let top_padding = 880.0;
        let right_padding = 40.0;
        let window_width = 220.0;

        let screen_rect = ctx.screen_rect();
        let window_pos = egui::pos2(
            screen_rect.right() - window_width - right_padding,
            top_padding,
        );

        egui::Window::new("Replay")
            .default_pos(window_pos)
            .default_width(window_width)
            .collapsible(true)
            .resizable(false)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.heading("Replay");
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.recording.name)
                                .desired_width(120.0),
                        );

                        let button_txt = if self.recording.active {
                            "⏹ Stop Recording"
                        } else {
                            "🔴 Record"
                        };
                        let can_record = self.recording.active || self.loaded_replay.is_none();

                        if ui
                            .add_enabled(can_record, egui::Button::new(button_txt))
                            .clicked()
                        {
                            if self.recording.active {
                                self.recording.stop();
                            } else if !self.recording.name.is_empty() {
                                match self.recording.start() {
                                    Ok(()) => {
                                        log::debug!("started recording: {}", self.recording.name)
                                    }
                                    Err(e) => log::error!("failed creating replay: {}", e),
                                }
                            }
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Load:");
                        ui.add_enabled_ui(self.loaded_replay.is_none(), |ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.load_name)
                                    .desired_width(120.0),
                            );
                        });

                        let can_load = !self.load_name.is_empty() && self.loaded_replay.is_none();
                        if ui
                            .add_enabled(can_load, egui::Button::new("📂 Load"))
                            .clicked()
                        {
                            match LoadedReplay::load(&self.load_name) {
                                Ok(r) => {
                                    log::info!(
                                        "loaded replay: {} ({} frames)",
                                        r.name,
                                        r.frame_count
                                    );
                                    // The replay is the content: it defines the
                                    // timeline's length and clears any path authored
                                    // against a previous one.
                                    let replay_duration = r.frame_count as f32 / 60.0;
                                    let discarded = self.timeline.path.len();
                                    self.timeline.reset_for_replay(replay_duration);
                                    if discarded > 0 {
                                        log::info!(
                                            "discarded {discarded} keyframe(s) on replay load"
                                        );
                                    }
                                    self.playback.time = 0.0;
                                    self.playback.playing = false;
                                    self.last_logged_frame = usize::MAX;
                                    self.loaded_replay = Some(r);
                                    // Make POV the default camera mode after loading a replay.
                                    // Maybe Orbit is better? Idk.
                                    self.camera_mode = CameraMode::Pov;
                                }
                                Err(e) => log::error!("failed to load replay: {e}"),
                            }
                        }

                        if self.loaded_replay.is_some() && ui.button("⏏ Unload").clicked() {
                            // Unloading the content resets the timeline with it.
                            self.timeline.reset();
                            self.playback.time = 0.0;
                            self.playback.playing = false;
                            self.last_logged_frame = usize::MAX;
                            self.loaded_replay = None;
                            self.load_name.clear();
                        }
                    });

                    if let Some(r) = &self.loaded_replay {
                        ui.label(format!("▶ {} ({} frames)", r.name, r.frame_count));
                    }

                    ui.separator();
                    let mut sync = game::sync_moving_objects();
                    if ui
                        .checkbox(&mut sync, "Sync moving objects (experimental)")
                        .on_hover_text(
                            "Also record and replay cranes, crates and similar objects. \
                             Needs to be on while recording and while playing back.",
                        )
                        .changed()
                    {
                        game::set_sync_moving_objects(sync);
                    }
                });
            });
    }

    pub fn update_keyframe_detail(&mut self, ctx: &eguiContext) {
        let top_padding = 480.0;
        let right_padding = 40.0;
        let window_width = 220.0;

        let screen_rect = ctx.screen_rect();
        let window_pos = egui::pos2(
            screen_rect.right() - window_width - right_padding,
            top_padding,
        );

        egui::Window::new("Keyframe")
            .default_pos(window_pos)
            .default_width(window_width)
            .collapsible(true)
            .resizable(false)
            .show(ctx, |ui| {
                let orbiting = self.camera_mode == CameraMode::Orbit;
                if let Some(i) = self.timeline.selected_index {
                    let Some(pre_edit) = self.timeline.path.get(i).cloned() else {
                        return;
                    };
                    let mut kf = pre_edit.clone();

                    ui.vertical(|ui| {
                        ui.heading("Keyframe details");
                        ui.horizontal(|ui| {
                            ui.label("Time");
                            ui.add(DragValue::new(&mut kf.time).speed(0.01).suffix("s"));
                        });
                        // An orbit key's position comes from its orbit, so the
                        // snapshot is not worth editing.
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
                            ui.add(DragValue::new(&mut kf.tension).speed(0.01).range(0.0..=1.0));
                        });
                        // Only while orbiting: these are the values the mouse
                        // is flying, so out of orbit mode they are neither
                        // editable nor worth looking at.
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

                    if pre_edit != kf {
                        if let Some(frame) = self.timeline.path.get_mut(i) {
                            *frame = kf.clone();
                        }
                        if pre_edit.time != kf.time {
                            self.timeline.sort();
                            // The playhead is what marks this key as selected,
                            // so retiming the key takes the playhead with it.
                            self.playback.time = kf.time;
                            self.playback.last_update = Some(Instant::now());
                        }
                        if pre_edit.orbit != kf.orbit {
                            self.load_selected_orbit();
                        }

                        if self.playback.mode_active {
                            game::set_camera(&kf.resolve(None));
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

                let mut apply = false;
                let mut reset = false;
                if self.apply_would_change_selected() {
                    ui.separator();
                    ui.colored_label(STALE_KEY_COLOR, "● Camera has moved off this keyframe");
                    apply = ui
                        .add_sized(
                            [ui.available_width(), 26.0],
                            egui::Button::new("⬇ Apply camera to keyframe"),
                        )
                        .on_hover_text(concat!(
                            "Overwrite this keyframe with the live camera: position, ",
                            "rotation, fov, and the orbit the mouse is flying.",
                        ))
                        .clicked();
                    reset = ui
                        .add_sized(
                            [ui.available_width(), 22.0],
                            egui::Button::new("↩ Reset camera to keyframe"),
                        )
                        .on_hover_text(concat!(
                            "Put the camera back on this keyframe, dropping framing ",
                            "that has not been applied.",
                        ))
                        .clicked();
                }

                if apply {
                    self.apply_camera_to_selected();
                }
                if reset {
                    self.reset_camera_to_selected();
                }
            });
    }

    pub fn update_timeline(&mut self, ctx: &eguiContext) {
        // Nothing in here selects anything: the strip moves the playhead and
        // `update` derives the selection from it.
        self.draw_timeline(ctx);
    }

    fn draw_timeline(&mut self, ctx: &eguiContext) {
        TopBottomPanel::bottom(egui::Id::new("TimelinePanel")).show(ctx, |ui| {
            // Two bars: what you are editing, then how you are looking at it.
            ui.horizontal(|ui| {
                ui.label("Keyframe");

                // POV flies the camera the replay was shot with and ignores the
                // authored path.
                let authoring = self.camera_mode != CameraMode::Pov;
                const POV_HINT: &str = "POV plays the replay's own camera — keyframes are ignored";

                ui.add_enabled_ui(authoring, |ui| {
                    if ui
                        .button("➕ Add")
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
                            self.playback.time = time;
                            self.playback.last_update = Some(Instant::now());
                        }
                    }
                });

                ui.add_enabled_ui(authoring && self.timeline.selected_index.is_some(), |ui| {
                    if ui.button("📋 Duplicate").clicked() {
                        if let Some(i) = self.timeline.selected_index {
                            if let Some(kf) = self.timeline.path.get(i).cloned() {
                                let mut dup = kf;
                                dup.time = self.timeline.free_time_at(dup.time + KEYFRAME_SPACING);
                                let time = dup.time;
                                self.timeline.add_keyframe(dup);
                                // Follow the copy, so it is the selected key.
                                self.playback.time = time;
                                self.playback.last_update = Some(Instant::now());
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

                // Shown checked and locked in the modes that imply it, so the
                // one control that can stop a replay dead is never a mystery.
                let implied = self.playback_mode_is_implied();
                ui.add_enabled_ui(!implied, |ui| {
                    if ui
                        .checkbox(&mut self.playback.mode_active, "Playback mode")
                        .on_disabled_hover_text(
                            "Always on in this mode: it has no camera of its own                              without the replay running.",
                        )
                        .changed()
                        && !self.playback.mode_active
                    {
                        self.playback.playing = false;
                    }
                });

                let prev_key = self.timeline.time_before(self.playback.time);
                if ui
                    .add_enabled(prev_key.is_some(), egui::Button::new("⏪ Previous"))
                    .clicked()
                {
                    if let Some(time) = prev_key {
                        self.playback.time = time;
                        self.playback.last_update = Some(Instant::now());
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
                        self.playback.time = time;
                        self.playback.last_update = Some(Instant::now());
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

            ui.horizontal(|ui| {
                let prev_mode = self.camera_mode;
                let pov_offered = game::is_replay_available();
                let pov_ready = self.loaded_replay.is_some();
                egui::ComboBox::from_id_salt("CameraMode")
                    .selected_text(self.camera_mode.label())
                    .width(80.0)
                    .show_ui(ui, |ui| {
                        for mode in CameraMode::ALL {
                            if mode == CameraMode::Pov {
                                if !pov_offered {
                                    continue;
                                }
                                if !pov_ready {
                                    ui.add_enabled_ui(false, |ui| {
                                        ui.selectable_label(false, mode.label())
                                            .on_disabled_hover_text("Load a replay first");
                                    });
                                    continue;
                                }
                            }
                            ui.selectable_value(&mut self.camera_mode, mode, mode.label());
                        }
                    });
                if self.camera_mode == CameraMode::Orbit && prev_mode != CameraMode::Orbit {
                    // Pick the rig up around whatever the camera already looks
                    // at, or where the selected orbit key left it, so entering
                    // the mode does not jump the view.
                    self.recentre_orbit();
                    self.load_selected_orbit();
                }

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

            let (space_pressed, delete_pressed) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::Delete),
                )
            });
            if space_pressed {
                self.playback.toggle(self.playback_end_time());
            }
            if delete_pressed && self.timeline.selected_index.is_some() {
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

            let stale = self.apply_would_change_selected();
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
                    // Only the selected key can be stale: it is the only one
                    // being compared against the live camera.
                    if stale {
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
                    self.playback.time = kf_time;
                    self.playback.last_update = Some(Instant::now());
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
                    self.playback.time = new_time.clamp(0.0, self.timeline.total_duration);
                    self.playback.last_update = Some(Instant::now());
                }
            }
            if timeline_response.dragged() {
                self.playback.playing = false;
            }
        });
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
        if ctx.input(|i| i.key_pressed(egui::Key::F2)) {
            if ui.recording.active {
                ui.recording.stop();
            } else if !ui.recording.name.is_empty() {
                match ui.recording.start() {
                    Ok(()) => log::debug!("started recording via F2: {}", ui.recording.name),
                    Err(e) => log::error!("failed creating replay: {}", e),
                }
            }
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

    ctx.debug_painter().text(
        Pos2::new(0., 0.),
        Align2::LEFT_TOP,
        version_string(),
        FontId::default(),
        Color32::LIGHT_YELLOW,
    );

    // The path and camera panels only make sense in the freecam. No camera yet
    // means freecam has never been entered, so there is nothing to author against.
    let authoring = game::in_freecam() && has_camera;

    if authoring {
        ui.update_timeline(ctx);
    }
    // The Replay window works in any camera mode, so a recording can be named and
    // started while skating, not only from the freecam.
    if replay_available {
        ui.update_replay_detail(ctx);
    }
    if authoring {
        ui.update_keyframe_detail(ctx);
        ui.update_orbit_input(ctx);
    }

    // We don't want to autofocus any text inputs.
    if just_opened {
        ctx.memory_mut(|m| m.stop_text_input());
    }

    draw_cursor(ctx);
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

    /// POV has no camera without the replay running and Orbit has nothing to
    /// pivot on, so with one loaded neither mode leaves playback mode to the
    /// user. Freecam authors against the level, so there it stays a choice.
    #[test]
    fn playback_mode_is_only_a_choice_in_freecam() {
        let mut ui = UiSystem::default();

        for mode in [CameraMode::Freecam, CameraMode::Pov, CameraMode::Orbit] {
            ui.camera_mode = mode;
            assert!(!ui.playback_mode_is_implied(), "{mode:?} with no replay");
        }

        ui.loaded_replay = Some(LoadedReplay {
            name: "test".into(),
            frame_count: 0,
            frames: Vec::new(),
        });

        ui.camera_mode = CameraMode::Freecam;
        assert!(!ui.playback_mode_is_implied());
        ui.camera_mode = CameraMode::Pov;
        assert!(ui.playback_mode_is_implied());
        ui.camera_mode = CameraMode::Orbit;
        assert!(ui.playback_mode_is_implied());
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
