//! The bone camera: a camera fixed to one of the skater's bones (board cam, head cam,
//! hand cam), with an offset, following the replayed skater.

use nalgebra_glm as glm;

use crate::{camera::CameraPose, game::Bone};

/// How a camera sits on a bone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneCamera {
    /// Which bone (index into the skater's skeleton).
    pub bone: usize,
    /// Where the camera sits relative to the bone, in the bone's own axes (or the world's,
    /// without `follow_rotation`), in game units.
    pub offset: [f32; 3],
    /// Turn of the camera relative to the bone, in degrees: round its up axis, its right
    /// axis, and the way it looks.
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Turn with the bone. Off, the camera only goes where the bone goes and keeps looking
    /// the way `yaw` and `pitch` say in the world.
    pub follow_rotation: bool,
    /// Keep the horizon level (no roll from the bone).
    pub level_horizon: bool,
    /// Smooth the bone's motion over this many seconds of replay (0 = rigid).
    pub stabilize: f32,
    pub fov: f32,
    /// Draw the bones' numbers over the game, to pick one.
    pub show_bones: bool,
}

impl Default for BoneCamera {
    fn default() -> Self {
        Self {
            bone: 0,
            offset: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            follow_rotation: true,
            level_horizon: false,
            stabilize: 0.0,
            fov: 90.0,
            show_bones: true,
        }
    }
}

/// Samples taken across the stabilize window (odd, centred on the playhead).
const STABILIZE_SAMPLES: i32 = 7;

impl BoneCamera {
    /// The camera at replay time `t`. `bones_at` gives the skater's bones at a replay time.
    /// `None` when the bone does not exist there.
    pub fn pose(&self, t: f32, bones_at: impl Fn(f32) -> Vec<Bone>) -> Option<CameraPose> {
        let bone = if self.stabilize > 0.0 {
            // A triangle-weighted mean over the window: steadier, still centred on `t`.
            let half = STABILIZE_SAMPLES / 2;
            let mut taken = Vec::new();
            for k in -half..=half {
                let dt = self.stabilize * 0.5 * k as f32 / half as f32;
                let w = (half + 1 - k.abs()) as f32;
                if let Some(b) = bones_at(t + dt).get(self.bone) {
                    taken.push((*b, w));
                }
            }
            mean_bone(&taken)?
        } else {
            *bones_at(t).get(self.bone)?
        };
        Some(self.pose_on(&bone))
    }

    /// The camera on `bone`.
    pub fn pose_on(&self, bone: &Bone) -> CameraPose {
        let rows = bone.rows.map(glm::Vec3::from);
        let world = [
            glm::vec3(1.0, 0.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec3(0.0, 0.0, 1.0),
        ];
        // The camera's own axes (right, up, back) before the turn: the bone's, or the world's.
        let axes = if self.follow_rotation { rows } else { world };
        let offset = axes[0] * self.offset[0] + axes[1] * self.offset[1] + axes[2] * self.offset[2];
        let position = glm::Vec3::from(bone.pos) + offset;

        let base = glm::mat3_to_quat(&glm::Mat3::from_columns(&axes));
        let turn = glm::quat_angle_axis(self.yaw.to_radians(), &glm::vec3(0.0, 1.0, 0.0))
            * glm::quat_angle_axis(self.pitch.to_radians(), &glm::vec3(1.0, 0.0, 0.0))
            * glm::quat_angle_axis(self.roll.to_radians(), &glm::vec3(0.0, 0.0, 1.0));
        let mut q = glm::quat_normalize(&(base * turn));
        if self.level_horizon {
            q = crate::camera::look_rotation(&crate::camera::forward(q.into())).into();
        }
        CameraPose {
            position: position.into(),
            orientation: q.into(),
            fov: self.fov,
        }
    }
}

/// The weighted mean of some bone transforms (rotation by summed, re-orthonormalised axes).
fn mean_bone(taken: &[(Bone, f32)]) -> Option<Bone> {
    let total: f32 = taken.iter().map(|(_, w)| w).sum();
    if taken.is_empty() || total <= 0.0 {
        return None;
    }
    let mut pos = glm::Vec3::zeros();
    let mut x = glm::Vec3::zeros();
    let mut y = glm::Vec3::zeros();
    for (b, w) in taken {
        pos += glm::Vec3::from(b.pos) * *w;
        x += glm::Vec3::from(b.rows[0]) * *w;
        y += glm::Vec3::from(b.rows[1]) * *w;
    }
    pos /= total;
    let x = x.try_normalize(1e-6)?;
    let z = x.cross(&y).try_normalize(1e-6)?;
    let y = z.cross(&x);
    Some(Bone {
        pos: pos.into(),
        rows: [x.into(), y.into(), z.into()],
    })
}

/// Where a world point shows on screen (pixels from the top left), for a camera with a
/// horizontal field of view `fov` degrees, or `None` behind the camera.
pub fn project(pose: &CameraPose, size: (f32, f32), point: [f32; 3]) -> Option<(f32, f32)> {
    let q: glm::Quat = pose.orientation.into();
    let rel = glm::Vec3::from(point) - glm::Vec3::from(pose.position);
    let v = glm::quat_rotate_vec3(&glm::quat_conjugate(&q), &rel);
    let ahead = -v.z;
    if ahead < 1.0 {
        return None;
    }
    let (w, h) = size;
    let tan = (pose.fov.to_radians() * 0.5).tan();
    let x = v.x / (ahead * tan);
    let y = v.y / (ahead * tan) * (w / h);
    Some(((x + 1.0) * 0.5 * w, (1.0 - y) * 0.5 * h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Quat, Vec3};

    fn bone(pos: [f32; 3], rows: [[f32; 3]; 3]) -> Bone {
        Bone { pos, rows }
    }

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn sits_on_the_bone_with_its_offset() {
        // A bone turned a quarter round the up axis: its x axis points along world -z.
        let b = bone(
            [10.0, 20.0, 30.0],
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        );
        let cam = BoneCamera {
            offset: [5.0, 0.0, 0.0],
            ..Default::default()
        };
        let pose = cam.pose_on(&b);
        assert!((glm::Vec3::from(pose.position) - glm::vec3(10.0, 20.0, 25.0)).norm() < 1e-4);
        // Looks along the bone's -z, which is world -x.
        let f = crate::camera::forward(pose.orientation);
        assert!((f - glm::vec3(-1.0, 0.0, 0.0)).norm() < 1e-4, "{f:?}");
    }

    #[test]
    fn without_rotation_follow_looks_the_world_way() {
        let b = bone(
            [0.0; 3],
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        );
        let cam = BoneCamera {
            follow_rotation: false,
            yaw: 90.0,
            ..Default::default()
        };
        let f = crate::camera::forward(cam.pose_on(&b).orientation);
        // Turned left a quarter from looking along -z: along -x.
        assert!((f - glm::vec3(-1.0, 0.0, 0.0)).norm() < 1e-4, "{f:?}");
    }

    #[test]
    fn stabilize_averages_round_the_playhead() {
        let cam = BoneCamera {
            stabilize: 1.0,
            ..Default::default()
        };
        // The bone moves along x at 10 units a second: the mean is where it is at t.
        let pose = cam
            .pose(2.0, |t| vec![bone([t * 10.0, 0.0, 0.0], IDENTITY)])
            .unwrap();
        assert!((pose.position.x - 20.0).abs() < 1e-3);
    }

    #[test]
    fn projects_the_point_ahead_to_the_centre() {
        let pose = CameraPose {
            position: Vec3::new(0.0, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            fov: 90.0,
        };
        let (x, y) = project(&pose, (1920.0, 1080.0), [0.0, 0.0, -100.0]).unwrap();
        assert!((x - 960.0).abs() < 1e-3 && (y - 540.0).abs() < 1e-3);
        // At the edge of a 90 degree view: the right border.
        let (x, _) = project(&pose, (1920.0, 1080.0), [100.0, 0.0, -100.0]).unwrap();
        assert!((x - 1920.0).abs() < 1e-2);
        assert!(project(&pose, (1920.0, 1080.0), [0.0, 0.0, 100.0]).is_none());
    }
}
