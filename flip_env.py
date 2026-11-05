"""Backflip env: from a crouch, jump, rotate backward 360 degrees, land on the feet, stay up.

Left to itself the policy settles for jumping and falling on its back, so it follows the whole
motion from flip_search.py (flip_reference.npz), which lands but has no balance feedback. After that
motion ends the policy has to keep the robot up by itself. Many episodes start part way along it
(reference state initialisation, as in DeepMimic), so it can practise every part from the start. The legs are mirrored: a flip is a motion
in the sagittal plane, and six actions are much easier to explore than twelve.
"""

import os

import gymnasium as gym
import mujoco
import numpy as np

from flip_model import Motors, load_flip_model

CTRL_DT = 0.01       # a push lasts about 0.1 s, so 50 Hz is too coarse
EPISODE_S = 4.0
JUMP_BY_S = 1.0      # end the episode if it hasn't left the ground by then
FLIGHT_STEPS = 3     # control steps without contact that count as having taken off...
TAKEOFF_COM = 0.47   # ...with the centre of mass this high (m; 0.43 standing), or pulling the feet up would count
FULL_FLIP = 2 * np.pi
ROT_CAP = FULL_FLIP + 0.6
TAKEOFF_VZ, TAKEOFF_SPIN = 1.5, 6.0   # m/s and kg m^2/s, around what flip_search.py's motion has
# Target range around the home pose, for HR, HAA, HFE, KFE, FFE, FAA (rad).
ACTION_SCALE = np.array([0.3, 0.3, 2.5, 2.5, 1.5, 0.3])
MIRROR = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
OBS_SIZE = 47
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flip_reference.npz")


class BackflipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CTRL_DT)}

    def __init__(self, air_start_prob=0.1, ref_start_prob=0.6, render_mode=None, render_size=(480, 640)):
        self.model = m = load_flip_model()
        self.data = mujoco.MjData(m)
        self.motors = Motors(m)
        self.air_start_prob = air_start_prob
        self.ref_start_prob = ref_start_prob
        self.ref = dict(np.load(REFERENCE))
        self.ref_frames = len(self.ref["qpos"])
        w, y = self.ref["qpos"][0][[3, 5]]
        self.ref_lean = -2.0 * np.arctan2(y, w)  # backward lean of the crouch counts as rotation (it leans slightly forward)
        self.render_mode = render_mode
        self._render_size = render_size
        self._renderer = None
        self.n_substeps = int(round(CTRL_DT / m.opt.timestep))
        self.home = m.keyframe("home").qpos.copy()
        self.floor = m.geom("floor").id
        self.foot_bodies = (m.geom_bodyid[m.geom("l_foot1").id], m.geom_bodyid[m.geom("r_foot1").id])
        self.collision_geoms = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
        self.sagittal = np.array([m.joint(f"{s}_{j}").qposadr[0] - 7 for s in ("LL", "LR")
                                  for j in ("HFE", "KFE", "FFE")])
        self.torso = m.body("torso").id
        self.imu = m.site("imu").id
        self.s = {}
        for name in ("gyro", "local_linvel"):
            i = m.sensor(name).id
            self.s[name] = slice(m.sensor_adr[i], m.sensor_adr[i] + m.sensor_dim[i])
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)

    # ---------- helpers ----------
    def _set_pose(self, hip, knee):
        q = self.home[7:].copy()
        for off in (0, 3):
            q[self.sagittal[off:off + 3]] = hip, knee, np.clip(0.17 - hip - knee, -0.52, 0.7)
        return q

    def _air_start(self):
        """Mid-flip, with speeds around what the motor search found (6-14 rad/s, up to 3 m/s falling)."""
        rng, d = self.np_random, self.data
        self.rot = self.torso_rot = self.foot_rot = rng.uniform(np.radians(120), np.radians(330))
        d.qpos[3:7] = np.cos(self.rot / 2), 0.0, -np.sin(self.rot / 2), 0.0
        d.qpos[7:] = self._set_pose(rng.uniform(-1.5, -0.4), rng.uniform(0.8, 2.0))
        d.qpos[2] = 1.5
        mujoco.mj_forward(self.model, d)
        lowest = min(d.geom_xpos[g][2] - self.model.geom_rbound[g] for g in self.collision_geoms)
        d.qpos[2] += rng.uniform(0.02, 0.2) - lowest
        d.qvel[0] = rng.uniform(-0.6, 0.2)
        d.qvel[2] = rng.uniform(-3.0, 0.5)
        d.qvel[4] = -rng.uniform(6.0, 14.0)
        self.flown = True
        self.t = rng.uniform(0.3, 0.8)

    def _ref_start(self, frame):
        rng, d, r = self.np_random, self.data, self.ref
        d.qpos[:] = r["qpos"][frame]
        d.qvel[:] = r["qvel"][frame]
        if frame:  # the crouch itself stays exact, it's where every real attempt begins
            d.qpos[7:] += rng.uniform(-0.03, 0.03, 12)
            d.qvel[:] += rng.uniform(-0.1, 0.1, d.qvel.shape)
        self.torso_rot = r["torso_rot"][frame] + self.ref_lean
        self.foot_rot = r["foot_rot"][frame] + self.ref_lean
        self.rot = min(self.torso_rot, self.foot_rot)
        self.t = frame * float(r["dt"])
        self.flown = bool(r["airborne"][frame])
        self.landed = bool(r["landed"][frame])

    def _ref_match(self):
        """How closely the robot follows the reference at this moment, 0 to 1."""
        d, r = self.data, self.ref
        k = int(self.t / float(r["dt"]))
        if k >= self.ref_frames:
            return 0.0
        err = (d.qpos[7:] - r["qpos"][k][7:])[self.sagittal]
        turn = self.torso_rot - (r["torso_rot"][k] + self.ref_lean)
        return float(np.exp(-(err @ err) / 2.0) * np.exp(-turn ** 2 / 0.5) * np.exp(-(d.qpos[2] - r["qpos"][k][2]) ** 2 / 0.02))

    def _contacts(self):
        d, m = self.data, self.model
        feet = [False, False]
        body_hit = False
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            b = m.geom_bodyid[g2 if g1 == self.floor else g1]
            if b == self.foot_bodies[0]:
                feet[0] = True
            elif b == self.foot_bodies[1]:
                feet[1] = True
            else:
                body_hit = True
        return feet, body_hit

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.home
        self.rot = self.torso_rot = self.foot_rot = 0.0
        self.flown, self.landed, self.t = False, False, 0.0
        pick = self.np_random.uniform()
        self.from_air = bool(pick < self.air_start_prob + self.ref_start_prob)  # anything but the crouch
        if pick < self.air_start_prob:
            self._air_start()
        elif pick < self.air_start_prob + self.ref_start_prob:
            self._ref_start(int(self.np_random.integers(1, self.ref_frames)))
        else:
            self._ref_start(0)
        mujoco.mj_forward(m, d)
        self.best_rot = self.rot
        self.air_steps = 0
        self.land_up = 0.0
        self.last_act = np.zeros(6)
        self.peak_z = float(d.subtree_com[0][2])
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        half = action * ACTION_SCALE
        target = self.home[7:] + np.concatenate([half, half * MIRROR])
        for _ in range(self.n_substeps):
            d.ctrl[:] = self.motors.torque(target, d.qpos[7:], d.qvel[6:])
            mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)
        self.t += CTRL_DT

        feet, body_hit = self._contacts()
        bad = np.isnan(d.qpos).any() or np.isnan(d.qvel).any()
        vz = d.subtree_linvel[0][2]
        spin_back = -d.subtree_angmom[0][1]
        up = d.site_xmat[self.imu][8]  # world z of the torso's z axis
        self.peak_z = max(self.peak_z, float(d.subtree_com[0][2]))

        # The flip is only as far along as the part that has rotated least: swinging the legs
        # turns the torso a long way without the whole body going over.
        best_rot = self.best_rot
        self.torso_rot -= d.cvel[self.torso][1] * CTRL_DT
        self.foot_rot -= d.cvel[self.foot_bodies[0]][1] * CTRL_DT
        self.rot = min(self.torso_rot, self.foot_rot)
        self.best_rot = max(self.best_rot, self.rot)
        change = action - self.last_act
        overspeed = np.maximum(np.abs(d.qvel[6:]) - self.motors.vmax, 0.0)
        reward = -0.02 * float(change @ change) - 0.001 * float(overspeed @ overspeed)
        touching = feet[0] or feet[1]
        following = self.t < self.ref_frames * float(self.ref["dt"])
        if following:
            reward += CTRL_DT * 30.0 * self._ref_match()

        if not self.flown:
            self.air_steps = 0 if (touching or body_hit) else self.air_steps + 1
            self.flown = self.air_steps >= FLIGHT_STEPS and d.subtree_com[0][2] > TAKEOFF_COM
            # Before take-off: reward the two things a take-off has to produce.
            # Height is capped at what the flip needs, or spin loses out to jumping higher.
            reward += CTRL_DT * (5.0 * np.clip(vz, 0.0, TAKEOFF_VZ) + 5.0 * np.clip(spin_back, 0.0, TAKEOFF_SPIN))
        elif not self.landed:
            reward += CTRL_DT * 2.0  # time in the air
            # Only new rotation counts, or swinging the legs back and forth would be paid every swing.
            reward += 3.0 * (min(self.best_rot, ROT_CAP) - min(best_rot, ROT_CAP))
            if touching and not body_hit:
                self.landed = True
                self.land_up = float(up)  # no bonus for touching: it pays to kick a foot at the floor
        if self.landed and not following:
            # Staying up only pays after a real rotation, or jumping in place would be the best plan.
            pose_err = d.qpos[7:] - self.home[7:]
            done_frac = min(self.rot / FULL_FLIP, 1.0) ** 2
            upright = np.exp(-(1.0 - up) ** 2 / 0.05)
            reward += CTRL_DT * 30.0 * done_frac * upright * np.exp(
                -float(pose_err[self.sagittal] @ pose_err[self.sagittal]) / 2.0) * np.exp(-d.qvel[4] ** 2 / 20.0)

        fell = bool(bad or body_hit or (self.landed and up < 0.3))
        late = not self.flown and self.t > JUMP_BY_S
        terminated = fell or late
        if fell:
            reward -= 2.0
        truncated = self.t >= EPISODE_S
        self.last_act = action

        info = {}
        if terminated or truncated:
            stood = bool(truncated and not fell and self.landed and abs(self.rot - FULL_FLIP) < 0.6)
            info["flip"] = dict(rot=self.rot, flown=self.flown, landed=self.landed, stood=stood,
                                from_air=self.from_air, land_up=self.land_up, peak=self.peak_z)
        if bad:
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        d = self.data
        gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        feet, _ = self._contacts()
        return np.concatenate([
            d.sensordata[self.s["local_linvel"]],
            d.sensordata[self.s["gyro"]],
            gravity,
            d.qpos[7:] - self.home[7:],
            d.qvel[6:],
            self.last_act,
            [np.sin(2 * np.pi * self.t / EPISODE_S), np.cos(2 * np.pi * self.t / EPISODE_S), self.t / EPISODE_S],
            [self.rot / FULL_FLIP, float(self.flown)],
            np.array(feet, dtype=np.float64),
            [d.qpos[2]],
        ]).astype(np.float32)

    def render(self):
        if self._renderer is None:
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = self.torso
            self._cam.distance, self._cam.elevation, self._cam.azimuth = 2.5, -10.0, 90.0
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
