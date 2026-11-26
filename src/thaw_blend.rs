//! THAW skater playback: blending the recorded skater between two frames.
//!
//! Pure maths on recorded frames (no game memory), split off from `thaw.rs` so it can be
//! tested on its own.

use std::sync::Mutex;

use nalgebra_glm as glm;

use crate::replay::{MAX_BONES, ReplayState, SkaterState};

/// The skater's pose as three replay entries, because the matrices are scaled and skewed and
/// a rotation plus a translation cannot hold them:
///   entry 0: rotation (quaternion) and translation of every bone,
///   entry 1: translation slots hold l00 l10 l11 l20,
///   entry 2: translation slots hold l21 l22 (see `split_rows`).
/// The replay code slerps the quaternions and lerps the translations, which keeps all three
/// consistent between two recorded frames.
pub(crate) fn pose_states(
    bytes: &[u8],
    bones: usize,
    world_pos: [f32; 3],
    world_matrix: [f32; 16],
) -> Option<[SkaterState; 3]> {
    if bytes.len() < bones * 64 {
        return None;
    }
    let blank = || SkaterState {
        world_pos,
        world_matrix,
        bone_count: bones as u8,
        bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
        bone_trans: [[0f32; 4]; MAX_BONES],
    };
    let mut states = [blank(), blank(), blank()];
    for bone in 0..bones {
        let mut m = [0f32; 16];
        for (i, v) in m.iter_mut().enumerate() {
            *v = f32_at(bytes, bone * 64 + i * 4);
        }
        let (e, l) = split_rows(&m);
        // Same layout as the memory: rows e0, e1, e2 go in as the columns of a glm matrix.
        let mut unit = [0f32; 16];
        unit[15] = 1.0;
        for r in 0..3 {
            for c in 0..3 {
                unit[r * 4 + c] = e[r][c];
            }
        }
        let mat3: glm::Mat3 = glm::mat4_to_mat3(&glm::make_mat4(&unit));
        let q: glm::Quat = glm::quat_normalize(&glm::mat3_to_quat(&mat3));
        states[0].bone_quats[bone] = [q.coords[0], q.coords[1], q.coords[2], q.coords[3]];
        states[0].bone_trans[bone] = [m[12], m[13], m[14], 0.0];
        states[1].bone_trans[bone] = [l[0], l[1], l[2], l[3]];
        states[2].bone_trans[bone] = [l[4], l[5], 0.0, 0.0];
    }
    Some(states)
}

/// Rebuilds the bone matrices from the three entries written by `pose_states`.
pub(crate) fn rebuild_pose(frame: &ReplayState) -> Vec<[f32; 16]> {
    let Some(pose) = frame.skaters.first() else {
        return Vec::new();
    };
    let extras = match (frame.skaters.get(1), frame.skaters.get(2)) {
        (Some(a), Some(b)) if a.bones() >= pose.bones() && b.bones() >= pose.bones() => {
            Some((a, b))
        }
        _ => None,
    };
    let mut out = Vec::with_capacity(pose.bones());
    for bone in 0..pose.bones() {
        let q = pose.bone_quats[bone];
        let quat = glm::quat_normalize(&glm::Quat::new(q[3], q[0], q[1], q[2]));
        let mat4 = glm::quat_to_mat4(&quat);
        let s = mat4.as_slice();
        let e = [[s[0], s[1], s[2]], [s[4], s[5], s[6]], [s[8], s[9], s[10]]];
        let rows = match extras {
            Some((a, b)) => {
                let (la, lb) = (a.bone_trans[bone], b.bone_trans[bone]);
                join_rows(&e, &[la[0], la[1], la[2], la[3], lb[0], lb[1]])
            }
            None => e,
        };
        let mut m = [0f32; 16];
        for r in 0..3 {
            for c in 0..3 {
                m[r * 4 + c] = rows[r][c];
            }
        }
        m[12] = pose.bone_trans[bone][0];
        m[13] = pose.bone_trans[bone][1];
        m[14] = pose.bone_trans[bone][2];
        m[15] = 1.0;
        out.push(m);
    }
    out
}

// Blending two recorded frames.
//
// The generic interpolation blends the skater's world matrix and the bones (which are
// relative to it) separately. When the game turns the skater's world matrix round and
// turns the bones back the other way in the same frame (switch/fakie landings, vert airs,
// some trick starts), the two blends do not cancel and the body swings through itself for
// the in-between frames; at 0.1 speed that is a visible twitch. When the skater turns as a
// whole instead (about 17 degrees a frame on a wallride, spins), the bones are best
// blended relative to the skater, so they swing round with it instead of cutting across.
// So each pair of frames is blended in whichever of the two spaces the pose moves less in
// (see `displayed`), and the result is expressed relative to the shown world rotation.

/// Rotation rows (as in memory) to a quaternion; same convention as `pose_states`.
pub(crate) fn rows_to_quat(e: &[[f32; 3]; 3]) -> glm::Quat {
    let mut unit = [0f32; 16];
    unit[15] = 1.0;
    for r in 0..3 {
        for c in 0..3 {
            unit[r * 4 + c] = e[r][c];
        }
    }
    glm::quat_normalize(&glm::mat3_to_quat(&glm::mat4_to_mat3(&glm::make_mat4(
        &unit,
    ))))
}

/// Inverse of `rows_to_quat`; same convention as `rebuild_pose`.
pub(crate) fn quat_to_rows(q: &glm::Quat) -> [[f32; 3]; 3] {
    let m = glm::quat_to_mat4(&glm::quat_normalize(q));
    let s = m.as_slice();
    [[s[0], s[1], s[2]], [s[4], s[5], s[6]], [s[8], s[9], s[10]]]
}

/// Shortest-way spherical interpolation; never panics.
pub(crate) fn slerp(a: &glm::Quat, b: &glm::Quat, t: f32) -> glm::Quat {
    let (a, mut b) = (glm::quat_normalize(a), glm::quat_normalize(b));
    let mut d = a.coords.dot(&b.coords);
    if d < 0.0 {
        b = -b;
        d = -d;
    }
    if d > 0.9995 {
        return glm::quat_normalize(&glm::Quat::from(a.coords + (b.coords - a.coords) * t));
    }
    let th = d.min(1.0).acos();
    let (sa, sb) = (((1.0 - t) * th).sin() / th.sin(), (t * th).sin() / th.sin());
    glm::Quat::from(a.coords * sa + b.coords * sb)
}

/// Rows of `x * r` (row vectors times a rotation given by its rows).
pub(crate) fn mul_rows(x: &[[f32; 3]; 3], r: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|i| vec_mul(x[i], r))
}

/// Rows of `x * r^T` (undoes `mul_rows`).
pub(crate) fn mul_rows_t(x: &[[f32; 3]; 3], r: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|i| vec_mul_t(x[i], r))
}

pub(crate) fn vec_mul(v: [f32; 3], r: &[[f32; 3]; 3]) -> [f32; 3] {
    std::array::from_fn(|c| v[0] * r[0][c] + v[1] * r[1][c] + v[2] * r[2][c])
}

pub(crate) fn vec_mul_t(v: [f32; 3], r: &[[f32; 3]; 3]) -> [f32; 3] {
    std::array::from_fn(|c| dot3(v, r[c]))
}

pub(crate) fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// The rotation rows of a world matrix, made a proper rotation.
pub(crate) fn rotation_rows(m: &[f32; 16]) -> [[f32; 3]; 3] {
    let mut m = *m;
    orthonormalize(&mut m);
    [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]]
}

/// A world matrix halfway (`t`) between two, rotation along the shortest way.
pub(crate) fn blend_world(a: &[f32; 16], b: &[f32; 16], t: f32) -> ([[f32; 3]; 3], [f32; 16]) {
    let rows = quat_to_rows(&slerp(
        &rows_to_quat(&rotation_rows(a)),
        &rows_to_quat(&rotation_rows(b)),
        t,
    ));
    let mut m: [f32; 16] = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
    for r in 0..3 {
        for c in 0..3 {
            m[r * 4 + c] = rows[r][c];
        }
    }
    (rows, m)
}

/// One bone in world space (position includes the object's position).
#[derive(Clone, Copy)]
pub(crate) struct WorldBone {
    pub(crate) rot: [[f32; 3]; 3],
    pub(crate) pos: [f32; 3],
    l: [f32; 6],
}

/// A recorded frame: the skater's own frame (world rotation and position) and every bone in
/// world space. With a skeleton, the rotation of a bone that has a parent is kept relative to
/// the parent instead (see `to_local`).
pub(crate) struct WorldPose {
    pub(crate) origin: [f32; 3],
    /// The skater's world rotation (rows), what the bones are drawn relative to.
    pub(crate) frame: [[f32; 3]; 3],
    pub(crate) bones: Vec<WorldBone>,
}

pub(crate) fn world_pose(f: &ReplayState) -> Option<WorldPose> {
    if f.skaters.len() < 3 {
        return None;
    }
    let bones = f.skaters[0].bones();
    if bones == 0 || f.skaters[1].bones() < bones || f.skaters[2].bones() < bones {
        return None;
    }
    let m = rebuild_pose(f);
    let rot = rotation_rows(&f.skaters[0].world_matrix);
    let origin = f.skaters[0].world_pos;
    let bones = m
        .iter()
        .take(bones)
        .map(|mb| {
            let (e, l) = split_rows(mb);
            let p = vec_mul([mb[12], mb[13], mb[14]], &rot);
            WorldBone {
                rot: mul_rows(&e, &rot),
                pos: [p[0] + origin[0], p[1] + origin[1], p[2] + origin[2]],
                l,
            }
        })
        .collect();
    Some(WorldPose {
        origin,
        frame: rot,
        bones,
    })
}

/// Makes every child bone's rotation relative to its parent. Blending those keeps a hand
/// turning with its arm: blended in world space, two neighbouring bones that both turn
/// close to 180 degrees can go round opposite ways, and the mesh between them is torn.
pub(crate) fn to_local(skeleton: &Skeleton, bones: &mut [WorldBone]) {
    let world: Vec<[[f32; 3]; 3]> = bones.iter().map(|b| b.rot).collect();
    for (c, joint) in skeleton.joint.iter().enumerate() {
        if let (Some((p, _)), true) = (joint, c < bones.len()) {
            if *p < world.len() {
                bones[c].rot = mul_rows(&world[c], &transpose3(&world[*p]));
            }
        }
    }
}

/// Inverse of `to_local` (parents come first in `skeleton.order`).
pub(crate) fn to_world(skeleton: &Skeleton, bones: &mut [WorldBone]) {
    for &c in &skeleton.order {
        if let Some((p, _)) = skeleton.joint.get(c).copied().flatten() {
            if p < bones.len() && c < bones.len() {
                let parent = bones[p].rot;
                bones[c].rot = mul_rows(&bones[c].rot, &parent);
            }
        }
    }
}

pub(crate) fn rot_slerp(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3], t: f32) -> [[f32; 3]; 3] {
    quat_to_rows(&slerp(&rows_to_quat(a), &rows_to_quat(b), t))
}

pub(crate) fn transpose3(a: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

pub(crate) const IDENTITY3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// How many recorded frames an animation cut is spread over.
pub(crate) const CUT_FRAMES: usize = 4;
/// A cut moves the bones at least this far (mean, in game units) in one frame...
pub(crate) const CUT_MIN_STEP: f32 = 10.0;
/// ...and at least this many times further than the frames around it.
pub(crate) const CUT_RATIO: f32 = 2.5;
pub(crate) const MAX_CUT_DEPTH: u32 = 3;
/// A jump of the skater further than this is a teleport (respawn, restart), shown as is.
pub(crate) const TELEPORT: f32 = 200.0;

/// Recorded frames needed while building one sample, each converted once.
pub(crate) struct PoseCache<'a> {
    frames: &'a [ReplayState],
    skeleton: Option<std::sync::Arc<Skeleton>>,
    poses: Vec<(usize, Option<std::rc::Rc<WorldPose>>)>,
    raw: Vec<(usize, Option<std::rc::Rc<WorldPose>>)>,
}

impl<'a> PoseCache<'a> {
    pub(crate) fn new(frames: &'a [ReplayState]) -> Self {
        PoseCache {
            frames,
            skeleton: skeleton_for(frames),
            poses: Vec::new(),
            raw: Vec::new(),
        }
    }

    /// Whether bone `k` has no parent: its rotation is the skater's, not relative to a
    /// parent bone. Without a skeleton every bone is one.
    fn is_root(&self, k: usize) -> bool {
        match &self.skeleton {
            Some(s) => s.joint.get(k).is_none_or(|j| j.is_none()),
            None => true,
        }
    }

    /// A frame exactly as recorded.
    fn raw(&mut self, i: usize) -> Option<std::rc::Rc<WorldPose>> {
        if let Some((_, p)) = self.raw.iter().find(|(k, _)| *k == i) {
            return p.clone();
        }
        let p = self
            .frames
            .get(i)
            .and_then(world_pose)
            .map(|mut p| {
                if let Some(s) = &self.skeleton {
                    to_local(s, &mut p.bones);
                }
                p
            })
            .map(std::rc::Rc::new);
        self.raw.push((i, p.clone()));
        p
    }

    fn raw_step(&mut self, a: usize, b: usize) -> Option<f32> {
        let (pa, pb) = (self.raw(a)?, self.raw(b)?);
        Some(pose_step(&pa, &pb))
    }

    /// A single frame that fits neither the frame before nor the one after (the game
    /// passes through an odd pose for one frame when it cuts between two animations).
    fn is_lone(&mut self, i: usize) -> bool {
        if i < 2 {
            return false;
        }
        let (Some(s0), Some(s1), Some(s2), Some(s3)) = (
            self.raw_step(i - 2, i - 1),
            self.raw_step(i - 1, i),
            self.raw_step(i, i + 1),
            self.raw_step(i + 1, i + 2),
        ) else {
            return false;
        };
        s1 > CUT_MIN_STEP && s2 > CUT_MIN_STEP && s1.min(s2) > CUT_RATIO * s0.max(s3).max(2.0)
    }

    /// A frame whose skater's frame is turned half round against both of its neighbours,
    /// with the bones turned along with it on one side: landing fakie into a manual, the
    /// game shows the new animation for one frame with the skater's frame still the wrong
    /// way round, and turns it back the frame after. Played as recorded, the skater would
    /// spin all the way round. Switching sides on a grind does it the other way round: the
    /// new animation one frame early in the old frame, the frame catching up half a turn
    /// later. Returns that frame turned the right way (the bones kept as they are relative to
    /// it).
    fn flipped(&mut self, i: usize) -> Option<WorldPose> {
        if i == 0 {
            return None;
        }
        let (a, b, c) = (self.raw(i - 1)?, self.raw(i)?, self.raw(i + 1)?);
        let turned = |x: &WorldPose, y: &WorldPose| turn_angle(&x.frame, &y.frame);
        // The bones turned with the frame (not back against it, as on an ordinary
        // switch/fakie landing): they hardly move relative to it on one side, but a lot in
        // the world.
        let along = |x: &WorldPose, y: &WorldPose| skater_step(x, y) < 0.5 * relative_step(x, y);
        let frame = if turned(&a, &b) >= FLIP_MIN
            && turned(&b, &c) >= FLIP_MIN
            && turned(&a, &c) <= TURN_MAX
            && (along(&a, &b) || along(&b, &c))
        {
            // Turned away and back: the frame halfway between its neighbours'.
            rot_slerp(&a.frame, &c.frame, 0.5)
        } else if turned(&a, &b) < FLIP_MIN
            && turned(&b, &c) >= FLIP_MIN
            && along(&b, &c)
            && skater_step(&a, &b) > CUT_MIN_STEP
            && skater_step(&b, &c) < 0.5 * CUT_MIN_STEP
        {
            // The other way round (switching sides on a grind): the new animation shows a
            // frame early, still in the old frame, and the frame catches up with it half a
            // turn in the next frame. The new animation belongs to the next frame's.
            c.frame
        } else {
            return None;
        };
        let bones = (0..b.bones.len())
            .map(|k| {
                let root = self.is_root(k);
                let local = to_skater(&b.frame, b.origin, &b.bones[k], root);
                from_skater(&frame, b.origin, &local, root)
            })
            .collect();
        Some(WorldPose {
            origin: b.origin,
            frame,
            bones,
        })
    }

    /// A frame as it is played: a lone frame is replaced by the halfway point of its
    /// neighbours, so the cut runs straight from one animation to the other, and a frame
    /// turned the wrong way round is turned back (see `flipped`).
    fn get(&mut self, i: usize) -> Option<std::rc::Rc<WorldPose>> {
        if let Some((_, p)) = self.poses.iter().find(|(k, _)| *k == i) {
            return p.clone();
        }
        let p = if let Some(p) = self.flipped(i) {
            Some(std::rc::Rc::new(p))
        } else if self.is_lone(i) {
            match (self.raw(i - 1), self.raw(i + 1)) {
                (Some(a), Some(b)) => Some(std::rc::Rc::new(WorldPose {
                    origin: lerp3(a.origin, b.origin, 0.5),
                    frame: rot_slerp(&a.frame, &b.frame, 0.5),
                    bones: a
                        .bones
                        .iter()
                        .zip(&b.bones)
                        .map(|(x, y)| WorldBone {
                            rot: rot_slerp(&x.rot, &y.rot, 0.5),
                            pos: lerp3(x.pos, y.pos, 0.5),
                            l: std::array::from_fn(|k| (x.l[k] + y.l[k]) * 0.5),
                        })
                        .collect(),
                })),
                _ => self.raw(i),
            }
        } else {
            self.raw(i)
        };
        self.poses.push((i, p.clone()));
        p
    }

    /// How far the bones move (see `pose_step`), and how far the skater itself moves,
    /// between two recorded frames.
    fn step(&mut self, a: usize, b: usize) -> Option<(f32, f32)> {
        let (pa, pb) = (self.get(a)?, self.get(b)?);
        let o: [f32; 3] = std::array::from_fn(|i| pa.origin[i] - pb.origin[i]);
        Some((pose_step(&pa, &pb), dot3(o, o).sqrt()))
    }

    /// Whether the animation cuts between frames `c` and `c + 1`: the bones, or the
    /// skater itself (snapping onto or off a wall), jump much further than in the frames
    /// around it.
    pub(crate) fn is_cut(&mut self, c: usize) -> bool {
        if c == 0 {
            return false;
        }
        let (Some(s0), Some(s1), Some(s2)) = (
            self.step(c - 1, c),
            self.step(c, c + 1),
            self.step(c + 1, c + 2),
        ) else {
            return false;
        };
        let jump = |x: f32, before: f32, after: f32| {
            x > CUT_MIN_STEP && x > CUT_RATIO * before.max(after).max(2.0)
        };
        s1.1 < TELEPORT && (jump(s1.0, s0.0, s2.0) || jump(s1.1, s0.1, s2.1))
    }

    /// Bone `k` of a frame in the skater's own space (see `to_skater`).
    fn skater_bone(&self, p: &WorldPose, k: usize) -> WorldBone {
        to_skater(&p.frame, p.origin, &p.bones[k], self.is_root(k))
    }
}

/// A bone relative to the skater's own frame: its position, and a root bone's rotation
/// (a child bone's rotation is relative to its parent already and stays as it is).
pub(crate) fn to_skater(
    frame: &[[f32; 3]; 3],
    origin: [f32; 3],
    b: &WorldBone,
    root: bool,
) -> WorldBone {
    let d: [f32; 3] = std::array::from_fn(|i| b.pos[i] - origin[i]);
    WorldBone {
        rot: if root {
            mul_rows_t(&b.rot, frame)
        } else {
            b.rot
        },
        pos: vec_mul_t(d, frame),
        l: b.l,
    }
}

/// Inverse of `to_skater`.
pub(crate) fn from_skater(
    frame: &[[f32; 3]; 3],
    origin: [f32; 3],
    b: &WorldBone,
    root: bool,
) -> WorldBone {
    let p = vec_mul(b.pos, frame);
    WorldBone {
        rot: if root { mul_rows(&b.rot, frame) } else { b.rot },
        pos: std::array::from_fn(|i| p[i] + origin[i]),
        l: b.l,
    }
}

/// Mean distance the bones move relative to the skater's position between two frames,
/// measured in world space (a turn of the whole skater counts as motion).
pub(crate) fn relative_step(pa: &WorldPose, pb: &WorldPose) -> f32 {
    let n = pa.bones.len().min(pb.bones.len());
    if n == 0 {
        return 0.0;
    }
    let rel: f32 = pa
        .bones
        .iter()
        .zip(&pb.bones)
        .map(|(x, y)| {
            let d: [f32; 3] =
                std::array::from_fn(|i| (x.pos[i] - pa.origin[i]) - (y.pos[i] - pb.origin[i]));
            dot3(d, d).sqrt()
        })
        .sum();
    rel / n as f32
}

/// The same, measured in the skater's own frame (a turn of the whole skater, as on a
/// wallride or in a spin, is no motion at all).
pub(crate) fn skater_step(pa: &WorldPose, pb: &WorldPose) -> f32 {
    let n = pa.bones.len().min(pb.bones.len());
    if n == 0 {
        return 0.0;
    }
    let rel: f32 = pa
        .bones
        .iter()
        .zip(&pb.bones)
        .map(|(x, y)| {
            let da: [f32; 3] = std::array::from_fn(|i| x.pos[i] - pa.origin[i]);
            let db: [f32; 3] = std::array::from_fn(|i| y.pos[i] - pb.origin[i]);
            let (la, lb) = (vec_mul_t(da, &pa.frame), vec_mul_t(db, &pb.frame));
            let d: [f32; 3] = std::array::from_fn(|i| la[i] - lb[i]);
            dot3(d, d).sqrt()
        })
        .sum();
    rel / n as f32
}

/// How far the pose changes between two frames: the smaller of the two measures, so that
/// neither a turn of the whole skater (wallrides, spins) nor a turn the bones undo
/// (switch/fakie landings) looks like an animation cut.
pub(crate) fn pose_step(pa: &WorldPose, pb: &WorldPose) -> f32 {
    relative_step(pa, pb).min(skater_step(pa, pb))
}

/// A turn of the skater's frame this large (radians, 100 degrees) between two frames is the
/// game flipping it (switch/fakie), not the skater turning.
const FLIP_MIN: f32 = 1.75;

/// The angle (radians) between two rotations.
pub(crate) fn turn_angle(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> f32 {
    let v = rot_log(&mul_rows(&transpose3(a), b));
    dot3(v, v).sqrt()
}

/// A turn a frame as a speed to carry on with: a turn past `TURN_MAX` in one frame is a
/// snap (the game flipping or cutting), not a speed, and counts as none.
fn as_speed(r: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let v = rot_log(&r);
    if dot3(v, v).sqrt() > TURN_MAX { IDENTITY3 } else { r }
}

pub(crate) fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The new motion's first frame `ab`, run back to the cut (`tau` < 1 frames after it) at the
/// speed it moves and turns to the frame after (`nb`).
fn run_back(ab: &WorldBone, nb: &WorldBone, tau: f32) -> WorldBone {
    let step = as_speed(mul_rows(&transpose3(&ab.rot), &nb.rot));
    WorldBone {
        rot: mul_rows(&ab.rot, &power(&transpose3(&step), 1.0 - tau)),
        pos: std::array::from_fn(|i| ab.pos[i] + (ab.pos[i] - nb.pos[i]) * (1.0 - tau)),
        l: ab.l,
    }
}

/// How much of the difference at a cut is still shown `tau` frames after it, as an
/// inertialization: the offset between the old motion and the new one starts with the old
/// motion's position (`w` = 1) and speed (`g` grows like `tau`), and fades to nothing, with
/// no speed left, `CUT_FRAMES + 1` frames after the cut. So the skater does not stall at the
/// cut: it carries on the way it was moving and curves into the new motion.
pub(crate) struct Ease {
    w: f32,
    g: f32,
}

impl Ease {
    pub(crate) fn at(tau: f32) -> Self {
        let len = (CUT_FRAMES + 1) as f32;
        let s = (tau / len).clamp(0.0, 1.0);
        Ease {
            w: 1.0 - smoothstep(s),
            g: len * s * (1.0 - s) * (1.0 - s),
        }
    }

    /// The offset for a difference `x0` that grows by `v0` a frame at the cut. Speed that
    /// would carry the offset further away first (an overshoot) is left out.
    fn offset(&self, x0: [f32; 3], v0: [f32; 3]) -> [f32; 3] {
        let (dir, len) = norm3(x0);
        let along = dot3(v0, dir);
        let v0: [f32; 3] = if len > 1e-4 && along > 0.0 {
            std::array::from_fn(|i| v0[i] - along * dir[i])
        } else {
            v0
        };
        std::array::from_fn(|i| x0[i] * self.w + v0[i] * self.g)
    }
}

/// `r` to the power `t` (`t` of the way from no turn to `r`).
fn power(r: &[[f32; 3]; 3], t: f32) -> [[f32; 3]; 3] {
    rot_slerp(&IDENTITY3, r, t)
}

/// `cur` with the difference between the old motion (`before` at the cut, `before_prev` the
/// frame before) and the new one (`ab` its first frame, `nb` the next) added, eased out (see
/// `Ease`). Rotations turn as `rot * step`, `step` being one frame's turn.
fn fade(
    cur: &WorldBone,
    ab: &WorldBone,
    nb: &WorldBone,
    before: &WorldBone,
    before_prev: &WorldBone,
    ease: &Ease,
) -> WorldBone {
    // Where the new motion would have been at the cut, moving at its own speed.
    let at_cut: [f32; 3] = std::array::from_fn(|i| 2.0 * ab.pos[i] - nb.pos[i]);
    let offset = ease.offset(
        std::array::from_fn(|i| before.pos[i] - at_cut[i]),
        std::array::from_fn(|i| (before.pos[i] - before_prev.pos[i]) - (nb.pos[i] - ab.pos[i])),
    );
    // The rotation that takes the new motion (run back to the cut) to the old one, and each
    // motion's turn a frame.
    let d = mul_rows(&transpose3(&run_back(ab, nb, 0.0).rot), &before.rot);
    let step_old = as_speed(mul_rows(&transpose3(&before_prev.rot), &before.rot));
    let step_new = as_speed(mul_rows(&transpose3(&ab.rot), &nb.rot));
    let dw = mul_rows(
        &mul_rows(&power(&transpose3(&step_new), ease.g), &power(&d, ease.w)),
        &power(&step_old, ease.g),
    );
    WorldBone {
        rot: mul_rows(&cur.rot, &dw),
        pos: std::array::from_fn(|i| cur.pos[i] + offset[i]),
        l: std::array::from_fn(|i| cur.l[i] + (before.l[i] - ab.l[i]) * ease.w),
    }
}

/// A rotation as a vector (axis times angle), and back. Only differences of such vectors
/// between nearby rotations are used, so the convention just has to be the same both ways.
fn rot_log(r: &[[f32; 3]; 3]) -> [f32; 3] {
    let q = rows_to_quat(r);
    let (mut v, mut w) = ([q.coords[0], q.coords[1], q.coords[2]], q.coords[3]);
    if w < 0.0 {
        v = v.map(|x| -x);
        w = -w;
    }
    let s = dot3(v, v).sqrt();
    if s < 1e-7 {
        return v.map(|x| 2.0 * x);
    }
    let angle = 2.0 * s.atan2(w);
    v.map(|x| x / s * angle)
}

/// A turn `v` (as `rot_log` gives it, at most half a turn) to be faded out while the motion
/// turns on by `ahead`: the same turn the other way round (a whole turn less) when that
/// makes what is shown turn less in all (`ahead - v`).
fn wound(v: [f32; 3], ahead: [f32; 3]) -> [f32; 3] {
    let (axis, angle) = norm3(v);
    if angle < 1e-4 {
        return v;
    }
    let other: [f32; 3] = std::array::from_fn(|i| v[i] - axis[i] * std::f32::consts::TAU);
    let total = |x: [f32; 3]| {
        let d: [f32; 3] = std::array::from_fn(|i| ahead[i] - x[i]);
        dot3(d, d)
    };
    if total(other) < total(v) { other } else { v }
}

fn rot_exp(v: [f32; 3]) -> [[f32; 3]; 3] {
    let angle = dot3(v, v).sqrt();
    if angle < 1e-7 {
        return IDENTITY3;
    }
    let (s, c) = (0.5 * angle).sin_cos();
    let axis = v.map(|x| x / angle * s);
    quat_to_rows(&glm::Quat::new(c, axis[0], axis[1], axis[2]))
}

/// How much longer than twice its shorter neighbour a curve's direction at a recorded frame
/// may be (game units, radians; see `Curve::bend`).
const POINT_TINY: f32 = 8.0;
const TURN_TINY: f32 = 0.35;
/// Turns longer than this (radians, a frame) are blended straight.
const TURN_MAX: f32 = 1.05;

/// Blending between two recorded frames along a curve through the frames around them
/// (cubic Hermite, Catmull-Rom tangents). The recorded frames themselves are met exactly; a
/// missing neighbour (at a cut) makes that end plain linear.
pub(crate) struct Curve {
    t: f32,
    /// Weights of the start and end tangents' differences from the straight line.
    h10: f32,
    h11: f32,
}

impl Curve {
    pub(crate) fn at(t: f32) -> Self {
        Curve {
            t,
            h10: t * (1.0 - t) * (1.0 - t),
            h11: t * t * (t - 1.0),
        }
    }

    /// How far the curve is off the straight line, from the steps before (`d_in`), between
    /// (`d`) and after (`d_out`) the two frames.
    ///
    /// The curve's direction at a recorded frame is the mean of the steps on either side.
    /// Next to a single far longer step (a board flip that turns a bone half round in one
    /// frame) that would bend the short steps around it into a wobble, so it is kept to at
    /// most twice the shorter step (plus `tiny`); a motion that starts from standing still
    /// still eases in. Both curves meeting at a frame work this out the same way, so the
    /// motion stays smooth through it. Steps longer than `max` (rotations past about 60
    /// degrees a frame, where the in-between turn is anyone's guess) are blended straight.
    fn bend(
        &self,
        d_in: Option<[f32; 3]>,
        d: [f32; 3],
        d_out: Option<[f32; 3]>,
        tiny: f32,
        max: f32,
    ) -> [f32; 3] {
        let len = |x: [f32; 3]| dot3(x, x).sqrt();
        if len(d) > max {
            return [0.0; 3];
        }
        let tangent = |x: Option<[f32; 3]>| -> [f32; 3] {
            let Some(x) = x.filter(|x| len(*x) <= max) else {
                return d;
            };
            let m: [f32; 3] = std::array::from_fn(|i| 0.5 * (x[i] + d[i]));
            let cap = 2.0 * len(x).min(len(d)) + tiny;
            let lm = len(m);
            if lm > cap { m.map(|v| v * cap / lm) } else { m }
        };
        let (m0, m1) = (tangent(d_in), tangent(d_out));
        std::array::from_fn(|i| self.h10 * (m0[i] - d[i]) + self.h11 * (m1[i] - d[i]))
    }

    fn point(
        &self,
        p: Option<[f32; 3]>,
        a: [f32; 3],
        b: [f32; 3],
        q: Option<[f32; 3]>,
    ) -> [f32; 3] {
        let sub = |x: [f32; 3], y: [f32; 3]| -> [f32; 3] { std::array::from_fn(|i| x[i] - y[i]) };
        let bend = self.bend(
            p.map(|p| sub(a, p)),
            sub(b, a),
            q.map(|q| sub(q, b)),
            POINT_TINY,
            f32::INFINITY,
        );
        let line = lerp3(a, b, self.t);
        std::array::from_fn(|i| line[i] + bend[i])
    }

    /// A rotation that turns as `step * r` (the skater's frame).
    fn frame(
        &self,
        p: Option<&[[f32; 3]; 3]>,
        a: &[[f32; 3]; 3],
        b: &[[f32; 3]; 3],
        q: Option<&[[f32; 3]; 3]>,
    ) -> [[f32; 3]; 3] {
        let step = |x: &[[f32; 3]; 3], y: &[[f32; 3]; 3]| rot_log(&mul_rows(y, &transpose3(x)));
        let bend = self.bend(
            p.map(|p| step(p, a)),
            step(a, b),
            q.map(|q| step(b, q)),
            TURN_TINY,
            TURN_MAX,
        );
        mul_rows(&rot_exp(bend), &rot_slerp(a, b, self.t))
    }

    /// A bone, whose rotation turns as `r * step`.
    fn bone(
        &self,
        p: Option<&WorldBone>,
        a: &WorldBone,
        b: &WorldBone,
        q: Option<&WorldBone>,
    ) -> WorldBone {
        let step = |x: &WorldBone, y: &WorldBone| rot_log(&mul_rows(&transpose3(&x.rot), &y.rot));
        let bend = self.bend(
            p.map(|p| step(p, a)),
            step(a, b),
            q.map(|q| step(b, q)),
            TURN_TINY,
            TURN_MAX,
        );
        WorldBone {
            rot: mul_rows(&rot_slerp(&a.rot, &b.rot, self.t), &rot_exp(bend)),
            pos: self.point(p.map(|p| p.pos), a.pos, b.pos, q.map(|q| q.pos)),
            l: std::array::from_fn(|i| a.l[i] + (b.l[i] - a.l[i]) * self.t),
        }
    }
}

/// What is shown at one moment: the skater's frame and every bone in the skater's own
/// space (see `to_skater`).
#[derive(Clone)]
pub(crate) struct Shown {
    pub(crate) frame: [[f32; 3]; 3],
    pub(crate) origin: [f32; 3],
    pub(crate) bones: Vec<WorldBone>,
}

/// A recorded frame as shown.
pub(crate) fn shown_frame(cache: &PoseCache, p: &WorldPose) -> Shown {
    Shown {
        frame: p.frame,
        origin: p.origin,
        bones: (0..p.bones.len())
            .map(|k| cache.skater_bone(p, k))
            .collect(),
    }
}

/// The skater shown at time `t` (in recorded frames), considering animation cuts up to
/// frame `max_cut` only.
///
/// Between two frames the skater's frame (rotation and position) is blended, and so are
/// the bones: in the skater's own space where the pose moves less there (the skater turns
/// as a whole: wallrides, spins), otherwise in world space (the game turns the skater one
/// way and the bones back the other, as on switch/fakie landings: no motion at all).
///
/// Where the animation cuts (the game starts a new animation without blending into it, or
/// snaps the skater onto or off a wall), the jump is spread over `CUT_FRAMES` frames: the
/// difference between what was shown at the cut and the new motion is carried along and
/// faded out. It is carried in the skater's own space, so it turns with the skater (on a
/// wallride the skater keeps turning about 17 degrees a frame while the cut fades), and the
/// skater's frame is faded the same way, so a 90 degree snap onto a wall becomes a quick
/// turn instead of a twitch.
pub(crate) fn displayed(
    cache: &mut PoseCache,
    t: f32,
    depth: u32,
    max_cut: usize,
) -> Option<Shown> {
    let last = cache.frames.len().checked_sub(1)?;
    let t = t.clamp(0.0, last as f32);
    let i0 = (t as usize).min(last);
    let i1 = (i0 + 1).min(last);
    let frac = t - i0 as f32;
    let (a, b) = (cache.get(i0)?, cache.get(i1)?);
    let n = a.bones.len().min(b.bones.len());
    // A teleport is shown as a cut, not as a flight across the map. Up to it, frame `i0`
    // is held, still easing out a cut before it (below).
    let o: [f32; 3] = std::array::from_fn(|i| a.origin[i] - b.origin[i]);
    let teleport = dot3(o, o).sqrt() > TELEPORT;
    if teleport && frac >= 0.5 {
        return Some(shown_frame(cache, &b));
    }
    // The frames before and after these two, to keep the motion's speed and turn going
    // smoothly through each recorded frame (instead of changing them at every frame, which
    // shows in slow motion as a judder along curves). Not across a cut or a teleport.
    let near = |p: &WorldPose, q: &WorldPose| {
        let o: [f32; 3] = std::array::from_fn(|i| p.origin[i] - q.origin[i]);
        dot3(o, o).sqrt() <= TELEPORT
    };
    let cut_here = i1 > i0 && cache.is_cut(i0);
    let prev = if i0 >= 1 && !cut_here && !cache.is_cut(i0 - 1) {
        cache
            .get(i0 - 1)
            .filter(|p| near(p, &a) && p.bones.len() >= n)
    } else {
        None
    };
    let next = if i1 > i0 && i1 < last && !cut_here && !cache.is_cut(i1) {
        cache
            .get(i1 + 1)
            .filter(|q| near(&b, q) && q.bones.len() >= n)
    } else {
        None
    };
    let curve = Curve::at(if teleport { 0.0 } else { frac });

    let frame = curve.frame(
        prev.as_ref().map(|p| &p.frame),
        &a.frame,
        &b.frame,
        next.as_ref().map(|q| &q.frame),
    );
    let origin = curve.point(
        prev.as_ref().map(|p| p.origin),
        a.origin,
        b.origin,
        next.as_ref().map(|q| q.origin),
    );
    let in_skater_space = skater_step(&a, &b) < relative_step(&a, &b);
    let mut out = Shown {
        frame,
        origin,
        bones: (0..n)
            .map(|k| {
                let root = cache.is_root(k);
                if in_skater_space {
                    let bone = |p: &WorldPose| cache.skater_bone(p, k);
                    curve.bone(
                        prev.as_ref().map(|p| bone(p)).as_ref(),
                        &bone(&a),
                        &bone(&b),
                        next.as_ref().map(|q| bone(q)).as_ref(),
                    )
                } else {
                    let world = curve.bone(
                        prev.as_ref().map(|p| &p.bones[k]),
                        &a.bones[k],
                        &b.bones[k],
                        next.as_ref().map(|q| &q.bones[k]),
                    );
                    to_skater(&frame, origin, &world, root)
                }
            })
            .collect(),
    };

    // The most recent cut within reach (`c` is the frame before the jump).
    for c in (i0.saturating_sub(CUT_FRAMES)..=i0.min(max_cut)).rev() {
        if c + 1 > last || !cache.is_cut(c) {
            continue;
        }
        let tau = t - c as f32;
        if tau >= (CUT_FRAMES + 1) as f32 {
            break;
        }
        // Not across a teleport since the cut.
        let jumped = (c + 1..i0).any(|j| match (cache.get(j), cache.get(j + 1)) {
            (Some(p), Some(q)) => !near(&p, &q),
            _ => true,
        });
        if jumped {
            break;
        }
        let ease = Ease::at(tau);
        // What was shown at the cut and one frame before it (themselves possibly still
        // fading an earlier cut): the old motion, and its speed.
        let (before, before_prev) = if depth < MAX_CUT_DEPTH {
            (
                displayed(cache, c as f32, depth + 1, c - 1)?,
                displayed(cache, (c - 1) as f32, depth + 1, c - 1)?,
            )
        } else {
            let (p, q) = (cache.get(c)?, cache.get(c - 1)?);
            (shown_frame(cache, &p), shown_frame(cache, &q))
        };
        let after = cache.get(c + 1)?;
        // The frame after that gives the new motion's speed, to run it back to the cut.
        let next = cache
            .get((c + 2).min(last))
            .unwrap_or_else(|| after.clone());
        let m = n
            .min(before.bones.len())
            .min(after.bones.len())
            .min(next.bones.len());

        // The fade is carried in the skater's own space when the skater turned with the
        // jump (snapping onto or off a wall: the whole skater turns, and keeps turning, so
        // the fade has to turn with it), and in world space when the game turned the skater
        // and the bones back the other way (switch/fakie landings: fading the skater's turn
        // would swing the body round).
        let skater_fade = {
            let pc = cache.get(c)?;
            skater_step(&pc, &after) < relative_step(&pc, &after)
        };
        if skater_fade {
            // A flip of the skater's frame right before or after the cut is no speed to
            // carry on with (in the skater's own space everything swaps sides).
            let next = if turn_angle(&after.frame, &next.frame) > TURN_MAX {
                after.clone()
            } else {
                next
            };
            let before_prev = if turn_angle(&before_prev.frame, &before.frame) > TURN_MAX {
                before.clone()
            } else {
                before_prev
            };
            // The skater's frame. Until the new animation's first frame is reached, it is
            // run backwards from there.
            let turn_new = mul_rows(&next.frame, &transpose3(&after.frame));
            let (cur_frame, cur_origin) = if tau < 1.0 {
                (
                    mul_rows(&power(&transpose3(&turn_new), 1.0 - tau), &after.frame),
                    std::array::from_fn(|i| {
                        after.origin[i] + (after.origin[i] - next.origin[i]) * (1.0 - tau)
                    }),
                )
            } else {
                (out.frame, out.origin)
            };
            // The turn that takes the new frame (run back to the cut) to the old one, applied
            // in the skater's own space (`df * frame`), and how fast each frame was turning (`turn * frame` is
            // the frame one recorded frame later).
            let frame_at_cut = mul_rows(&transpose3(&turn_new), &after.frame);
            let df = mul_rows(&before.frame, &transpose3(&frame_at_cut));
            // Which way round the difference is faded: the way that keeps the whole turn,
            // from what was shown at the cut to where the new motion is when the fade is
            // over, under half a turn. A snap of about half a turn (switching sides on a
            // grind) followed by the game turning the skater on further would otherwise be
            // faded the long way round: a whole spin.
            let mut ahead = rot_log(&turn_new);
            let end = (c + 1 + CUT_FRAMES).min(last);
            let mut prev_frame = after.frame;
            for j in c + 2..=end {
                let Some(p) = cache.get(j) else { break };
                let step = rot_log(&mul_rows(&p.frame, &transpose3(&prev_frame)));
                ahead = std::array::from_fn(|i| ahead[i] + step[i]);
                prev_frame = p.frame;
            }
            let back = wound(rot_log(&df), ahead);
            let turn_old = mul_rows(&before.frame, &transpose3(&before_prev.frame));
            let d = mul_rows(
                &mul_rows(&power(&turn_old, ease.g), &rot_exp(back.map(|x| x * ease.w))),
                &power(&transpose3(&turn_new), ease.g),
            );
            out.frame = mul_rows(&d, &cur_frame);
            let origin_at_cut: [f32; 3] =
                std::array::from_fn(|i| 2.0 * after.origin[i] - next.origin[i]);
            let offset = ease.offset(
                std::array::from_fn(|i| before.origin[i] - origin_at_cut[i]),
                std::array::from_fn(|i| {
                    (before.origin[i] - before_prev.origin[i]) - (next.origin[i] - after.origin[i])
                }),
            );
            out.origin = std::array::from_fn(|i| cur_origin[i] + offset[i]);

            for k in 0..m {
                let (ab, nb) = (cache.skater_bone(&after, k), cache.skater_bone(&next, k));
                let cur = if tau < 1.0 {
                    run_back(&ab, &nb, tau)
                } else {
                    out.bones[k]
                };
                out.bones[k] = fade(
                    &cur,
                    &ab,
                    &nb,
                    &before.bones[k],
                    &before_prev.bones[k],
                    &ease,
                );
            }
        } else {
            // In world space; the skater's frame is not faded.
            let (f, o) = (out.frame, out.origin);
            for k in 0..m {
                let root = cache.is_root(k);
                let (ab, nb) = (&after.bones[k], &next.bones[k]);
                let bb = from_skater(&before.frame, before.origin, &before.bones[k], root);
                let bp = from_skater(
                    &before_prev.frame,
                    before_prev.origin,
                    &before_prev.bones[k],
                    root,
                );
                let cur = if tau < 1.0 {
                    run_back(ab, nb, tau)
                } else {
                    from_skater(&f, o, &out.bones[k], root)
                };
                out.bones[k] = to_skater(&f, o, &fade(&cur, ab, nb, &bb, &bp, &ease), root);
            }
        }
        break;
    }
    Some(out)
}

// Keeping the joints together.
//
// The bone matrices are skinning matrices, not joint transforms, so blending each bone on
// its own lets neighbouring bones drift apart at the joints whenever a bone turns far in
// one frame (animation cuts turn bones up to 180 degrees); the mesh between them is then
// drawn stretched. Every joint has a fixed point (in the model's rest space) that both
// bones of the joint move to the same place: x * M_child = x * M_parent, M being the
// bone's matrix with its translation. The joints are found once per replay from its own
// frames (least squares over many frames, then a tree of the best-fitting joints), and
// after blending, every child bone is moved so that its joint point sits exactly where the
// parent's does.

/// A joint fits when its point stays this close (mean, game units) on both bones.
pub(crate) const JOINT_MAX_ERROR: f32 = 1.0;
pub(crate) const SKELETON_SAMPLES: usize = 200;

pub(crate) struct Skeleton {
    /// Bones in an order where every parent comes before its children.
    order: Vec<usize>,
    /// Parent bone and joint point (rest space) of each bone; `None` for free bones.
    joint: Vec<Option<(usize, [f32; 3])>>,
}

static SKELETON: Mutex<Option<(usize, usize, std::sync::Arc<Skeleton>)>> = Mutex::new(None);

pub(crate) fn skeleton_for(frames: &[ReplayState]) -> Option<std::sync::Arc<Skeleton>> {
    let key = (frames.as_ptr() as usize, frames.len());
    let mut guard = SKELETON.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, n, s)) = guard.as_ref() {
        if (*p, *n) == key {
            return Some(s.clone());
        }
    }
    let s = std::sync::Arc::new(find_skeleton(frames)?);
    *guard = Some((key.0, key.1, s.clone()));
    Some(s)
}

/// Rows and translation of a bone matrix.
pub(crate) fn rows_t(m: &[f32; 16]) -> ([[f32; 3]; 3], [f32; 3]) {
    (
        [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]],
        [m[12], m[13], m[14]],
    )
}

/// Solves the 3x3 system `a x = b` (a symmetric, slightly regularised).
pub(crate) fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let mut m = *a;
    for (i, row) in m.iter_mut().enumerate() {
        row[i] += 1e-6;
    }
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let col = |c: usize| -> f64 {
        let mut mc = m;
        for r in 0..3 {
            mc[r][c] = b[r];
        }
        mc[0][0] * (mc[1][1] * mc[2][2] - mc[1][2] * mc[2][1])
            - mc[0][1] * (mc[1][0] * mc[2][2] - mc[1][2] * mc[2][0])
            + mc[0][2] * (mc[1][0] * mc[2][1] - mc[1][1] * mc[2][0])
    };
    Some([col(0) / det, col(1) / det, col(2) / det])
}

pub(crate) fn find_skeleton(frames: &[ReplayState]) -> Option<Skeleton> {
    let step = (frames.len() / SKELETON_SAMPLES).max(1);
    let poses: Vec<Vec<[f32; 16]>> = frames
        .iter()
        .step_by(step)
        .map(rebuild_pose)
        .filter(|p| !p.is_empty())
        .collect();
    let bones = poses.iter().map(|p| p.len()).min()?;
    if poses.len() < 20 || bones < 2 {
        return None;
    }
    // For every pair: the point both bones carry to the same place, and how well it fits.
    let mut fit = vec![vec![(f32::INFINITY, [0f32; 3]); bones]; bones];
    for j in 0..bones {
        for p in 0..j {
            let mut ata = [[0f64; 3]; 3];
            let mut atb = [0f64; 3];
            let mut btb = 0f64;
            for pose in &poses {
                let (rj, tj) = rows_t(&pose[j]);
                let (rp, tp) = rows_t(&pose[p]);
                // x * (Rj - Rp) = tp - tj, as D^T x = b with D = Rj - Rp.
                let d: [[f64; 3]; 3] =
                    std::array::from_fn(|r| std::array::from_fn(|c| (rj[r][c] - rp[r][c]) as f64));
                let b: [f64; 3] = std::array::from_fn(|c| (tp[c] - tj[c]) as f64);
                for r in 0..3 {
                    for s in 0..3 {
                        ata[r][s] += (0..3).map(|c| d[r][c] * d[s][c]).sum::<f64>();
                    }
                    atb[r] += (0..3).map(|c| d[r][c] * b[c]).sum::<f64>();
                }
                btb += b.iter().map(|v| v * v).sum::<f64>();
            }
            let Some(x) = solve3(&ata, &atb) else {
                continue;
            };
            let xax: f64 = (0..3)
                .map(|r| (0..3).map(|s| x[r] * ata[r][s] * x[s]).sum::<f64>())
                .sum();
            let xb: f64 = (0..3).map(|r| x[r] * atb[r]).sum();
            let sse = (btb - 2.0 * xb + xax).max(0.0);
            let rms = (sse / (poses.len() as f64 * 3.0)).sqrt() as f32;
            let x = [x[0] as f32, x[1] as f32, x[2] as f32];
            fit[j][p] = (rms, x);
            fit[p][j] = (rms, x);
        }
    }
    // A tree of the best joints (Prim), grown from bone 0; bones that do not fit any
    // joint start their own tree and are blended freely.
    let mut joint: Vec<Option<(usize, [f32; 3])>> = vec![None; bones];
    let mut done = vec![false; bones];
    let mut order = Vec::with_capacity(bones);
    while order.len() < bones {
        let root = (0..bones).find(|&b| !done[b])?;
        done[root] = true;
        order.push(root);
        loop {
            let mut best: Option<(f32, usize, usize)> = None;
            for &p in &order {
                for c in 0..bones {
                    if done[c] {
                        continue;
                    }
                    let e = fit[c][p].0;
                    if e <= JOINT_MAX_ERROR && best.is_none_or(|b| e < b.0) {
                        best = Some((e, c, p));
                    }
                }
            }
            let Some((_, c, p)) = best else { break };
            done[c] = true;
            joint[c] = Some((p, fit[c][p].1));
            order.push(c);
        }
    }
    let joints = joint.iter().filter(|j| j.is_some()).count();
    log::debug!("[THAW] skeleton for this replay: {joints} joints between {bones} bones");
    Some(Skeleton { order, joint })
}

/// Where a child bone has to be for its joint point to sit where the parent's does.
pub(crate) fn joint_position(parent: &WorldBone, child: &WorldBone, x: [f32; 3]) -> [f32; 3] {
    let on_parent = vec_mul(x, &join_rows(&parent.rot, &parent.l));
    let on_child = vec_mul(x, &join_rows(&child.rot, &child.l));
    std::array::from_fn(|i| parent.pos[i] + on_parent[i] - on_child[i])
}

/// How far each bone of a recorded frame is from where its joint puts it (a few units:
/// the joints are not perfect). Kept when blending, so recorded frames stay exact.
pub(crate) fn joint_offsets(skeleton: &Skeleton, bones: &[WorldBone]) -> Vec<[f32; 3]> {
    (0..bones.len())
        .map(|c| match skeleton.joint.get(c).copied().flatten() {
            Some((p, x)) if p < bones.len() => {
                let at = joint_position(&bones[p], &bones[c], x);
                std::array::from_fn(|i| bones[c].pos[i] - at[i])
            }
            _ => [0.0; 3],
        })
        .collect()
}

/// Moves every child bone so that its joint point sits where the parent's does (plus the
/// recorded frames' own small offsets).
pub(crate) fn keep_joints(skeleton: &Skeleton, bones: &mut [WorldBone], offsets: &[[f32; 3]]) {
    for &c in &skeleton.order {
        let Some((p, x)) = skeleton.joint.get(c).copied().flatten() else {
            continue;
        };
        if p >= bones.len() || c >= bones.len() {
            continue;
        }
        let at = joint_position(&bones[p], &bones[c], x);
        let off = offsets.get(c).copied().unwrap_or([0.0; 3]);
        bones[c].pos = std::array::from_fn(|i| at[i] + off[i]);
    }
}

// Matrix math (rows as stored in memory: row r is m[r * 4 .. r * 4 + 3]).

pub(crate) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm3(v: [f32; 3]) -> ([f32; 3], f32) {
    let l = dot3(v, v).sqrt();
    if l < 1e-6 {
        ([1.0, 0.0, 0.0], 0.0)
    } else {
        ([v[0] / l, v[1] / l, v[2] / l], l)
    }
}

pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Splits the 3x3 part of a bone matrix (its rows are scaled and slightly skewed) into a pure
/// rotation Q (rows e0, e1, e2) and a lower triangular L, so that row0 = l00 e0,
/// row1 = l10 e0 + l11 e1, row2 = l20 e0 + l21 e1 + l22 e2. Nothing is lost.
pub(crate) fn split_rows(m: &[f32; 16]) -> ([[f32; 3]; 3], [f32; 6]) {
    let r0 = [m[0], m[1], m[2]];
    let r1 = [m[4], m[5], m[6]];
    let r2 = [m[8], m[9], m[10]];
    let (e0, l00) = norm3(r0);
    let l10 = dot3(r1, e0);
    let v1 = [
        r1[0] - l10 * e0[0],
        r1[1] - l10 * e0[1],
        r1[2] - l10 * e0[2],
    ];
    let (mut e1, l11) = norm3(v1);
    if l11 == 0.0 {
        // r1 is parallel to r0: any perpendicular direction will do.
        let helper = if e0[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let d = dot3(helper, e0);
        e1 = norm3([
            helper[0] - d * e0[0],
            helper[1] - d * e0[1],
            helper[2] - d * e0[2],
        ])
        .0;
    }
    let e2 = cross3(e0, e1);
    (
        [e0, e1, e2],
        [l00, l10, l11, dot3(r2, e0), dot3(r2, e1), dot3(r2, e2)],
    )
}

/// Inverse of `split_rows` for the 3x3 part: rotation rows plus the six factors.
pub(crate) fn join_rows(e: &[[f32; 3]; 3], l: &[f32; 6]) -> [[f32; 3]; 3] {
    let mut rows = [[0f32; 3]; 3];
    for c in 0..3 {
        rows[0][c] = l[0] * e[0][c];
        rows[1][c] = l[1] * e[0][c] + l[2] * e[1][c];
        rows[2][c] = l[3] * e[0][c] + l[4] * e[1][c] + l[5] * e[2][c];
    }
    rows
}

/// Makes the 3x3 part of a matrix a proper rotation again (component-wise interpolation
/// of two rotations is not one, and its rows come out shorter than 1). The object
/// matrix is a pure rotation, so the rows are made unit length again.
pub(crate) fn orthonormalize(m: &mut [f32; 16]) {
    let r0 = [m[0], m[1], m[2]];
    let r1 = [m[4], m[5], m[6]];
    let r2 = [m[8], m[9], m[10]];
    let (len0, len1, len2) = (norm3(r0).1, norm3(r1).1, norm3(r2).1);
    if !(len0.is_finite() && len1.is_finite() && len2.is_finite())
        || len0 < 1e-4
        || len1 < 1e-4
        || len2 < 1e-4
    {
        return;
    }
    let (u0, _) = norm3(r0);
    let d = dot3(r1, u0);
    let (u1, l1) = norm3([r1[0] - d * u0[0], r1[1] - d * u0[1], r1[2] - d * u0[2]]);
    if l1 < 1e-4 {
        return;
    }
    let mut u2 = cross3(u0, u1);
    if dot3(u2, r2) < 0.0 {
        u2 = [-u2[0], -u2[1], -u2[2]];
    }
    for i in 0..3 {
        m[i] = u0[i];
        m[4 + i] = u1[i];
        m[8 + i] = u2[i];
    }
}

/// The 4x4 world matrix of a model part as the game builds it: the object's rotation rows
/// with the position as the fourth row.
pub(crate) fn world_matrix_of(pos: &[f32; 3], matrix: &[f32; 16]) -> [f32; 16] {
    let mut m = *matrix;
    m[3] = 0.0;
    m[7] = 0.0;
    m[11] = 0.0;
    m[12] = pos[0];
    m[13] = pos[1];
    m[14] = pos[2];
    m[15] = 1.0;
    m
}

pub(crate) fn row_lengths(m: &[f32]) -> [f32; 3] {
    std::array::from_fn(|r| norm3([m[r * 4], m[r * 4 + 1], m[r * 4 + 2]]).1)
}

/// Multiplies the three rotation rows by `scale`.
pub(crate) fn scaled(m: &[f32; 16], scale: [f32; 3]) -> [f32; 16] {
    let mut out = *m;
    for r in 0..3 {
        let k = if scale[r].is_finite() && scale[r] > 0.2 && scale[r] < 5.0 {
            scale[r]
        } else {
            1.0
        };
        for c in 0..3 {
            out[r * 4 + c] *= k;
        }
    }
    out
}

/// The rotation rows made unit length (removes a part's scale).
pub(crate) fn unscaled(m: &[f32]) -> [f32; 16] {
    let mut out: [f32; 16] = std::array::from_fn(|i| m[i]);
    let len = row_lengths(m);
    for r in 0..3 {
        if len[r] > 1e-4 {
            for c in 0..3 {
                out[r * 4 + c] /= len[r];
            }
        }
    }
    out
}

pub(crate) fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// The skater between recorded frames `i0` and `i0 + 1` (`t` of the way): its three pose
/// entries and its parts' world matrices, as `capture_skater` records them.
pub(crate) fn blend_skater(frames: &[ReplayState], i0: usize, t: f32) -> Option<Vec<SkaterState>> {
    let (a, b) = (frames.get(i0)?, frames.get(i0 + 1)?);
    if a.skaters.len() < 3 || b.skaters.len() < 3 {
        return None;
    }
    let mut cache = PoseCache::new(frames);
    let shown = displayed(&mut cache, i0 as f32 + t, 0, usize::MAX)?;
    if shown.bones.is_empty() {
        return None;
    }
    let (w, world_pos) = (shown.frame, shown.origin);
    let mut bones: Vec<WorldBone> = shown
        .bones
        .iter()
        .enumerate()
        .map(|(k, b)| from_skater(&w, world_pos, b, cache.is_root(k)))
        .collect();
    if let Some(skeleton) = cache.skeleton.clone() {
        to_world(&skeleton, &mut bones);
        // The recorded frames' own joint offsets (a lone frame takes its neighbours').
        let world_of = |p: &WorldPose| {
            let mut b = p.bones.clone();
            to_world(&skeleton, &mut b);
            b
        };
        let mut offsets_of = |i: usize| -> Option<Vec<[f32; 3]>> {
            if cache.is_lone(i) {
                let (a, b) = (cache.raw(i - 1)?, cache.raw(i + 1)?);
                let (oa, ob) = (
                    joint_offsets(&skeleton, &world_of(&a)),
                    joint_offsets(&skeleton, &world_of(&b)),
                );
                Some(
                    oa.iter()
                        .zip(&ob)
                        .map(|(x, y)| lerp3(*x, *y, 0.5))
                        .collect(),
                )
            } else {
                Some(joint_offsets(&skeleton, &world_of(&*cache.raw(i)?)))
            }
        };
        if let (Some(oa), Some(ob)) = (offsets_of(i0), offsets_of(i0 + 1)) {
            let offsets: Vec<[f32; 3]> =
                oa.iter().zip(&ob).map(|(x, y)| lerp3(*x, *y, t)).collect();
            keep_joints(&skeleton, &mut bones, &offsets);
        }
    }
    // The world matrix: the recorded ones blended, with the shown rotation.
    let mut world_matrix: [f32; 16] = std::array::from_fn(|i| {
        a.skaters[0].world_matrix[i]
            + (b.skaters[0].world_matrix[i] - a.skaters[0].world_matrix[i]) * t
    });
    for r in 0..3 {
        for c in 0..3 {
            world_matrix[r * 4 + c] = w[r][c];
        }
    }
    // How far a faded cut holds the skater from where the recorded frames put it.
    let plain_pos = lerp3(a.skaters[0].world_pos, b.skaters[0].world_pos, t);
    let held: [f32; 3] = std::array::from_fn(|i| world_pos[i] - plain_pos[i]);

    let blank = || SkaterState {
        world_pos,
        world_matrix,
        bone_count: bones.len().min(MAX_BONES) as u8,
        bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
        bone_trans: [[0f32; 4]; MAX_BONES],
    };
    let mut out = vec![blank(), blank(), blank()];
    for (bone, wb) in bones.iter().enumerate().take(MAX_BONES) {
        // Back to model space, relative to the shown world rotation.
        let e = mul_rows_t(&wb.rot, &w);
        let tm = vec_mul_t(
            [
                wb.pos[0] - world_pos[0],
                wb.pos[1] - world_pos[1],
                wb.pos[2] - world_pos[2],
            ],
            &w,
        );
        let q = rows_to_quat(&e);
        out[0].bone_quats[bone] = [q.coords[0], q.coords[1], q.coords[2], q.coords[3]];
        out[0].bone_trans[bone] = [tm[0], tm[1], tm[2], 0.0];
        out[1].bone_trans[bone] = [wb.l[0], wb.l[1], wb.l[2], wb.l[3]];
        out[2].bone_trans[bone] = [wb.l[4], wb.l[5], 0.0, 0.0];
    }
    // The parts' own world matrices.
    for (pa, pb) in a.skaters.iter().zip(b.skaters.iter()).skip(3) {
        if pa.bones() != 0 || pb.bones() != 0 {
            break;
        }
        // The parts are drawn with the skater's world rotation (checked on recordings), so
        // they get the shown rotation; their positions are blended on their own and moved
        // with the skater where a cut is being faded.
        let pos: [f32; 3] = std::array::from_fn(|i| {
            pa.world_pos[i] + (pb.world_pos[i] - pa.world_pos[i]) * t + held[i]
        });
        let mut m: [f32; 16] = std::array::from_fn(|i| {
            pa.world_matrix[i] + (pb.world_matrix[i] - pa.world_matrix[i]) * t
        });
        for r in 0..3 {
            for c in 0..3 {
                m[r * 4 + c] = w[r][c];
            }
            m[12 + r] = pos[r];
        }
        out.push(SkaterState {
            world_pos: pos,
            world_matrix: m,
            bone_count: 0,
            bone_quats: [[0f32, 0.0, 0.0, 1.0]; MAX_BONES],
            bone_trans: [[0f32; 4]; MAX_BONES],
        });
    }
    Some(out)
}
