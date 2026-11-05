"""Berkeley Humanoid joystick-walking env for CPU MuJoCo + Gymnasium.

A NumPy port of MuJoCo Playground's `BerkeleyHumanoidJoystickFlatTerrain`
(google-deepmind/mujoco_playground, locomotion/berkeley_humanoid/joystick.py),
so it trains with Stable-Baselines3 on a laptop instead of JAX/MJX on a GPU.
"""

import functools
import hashlib
import os

import gymnasium as gym
import mujoco
import numpy as np

# Robot model: $BH_XML, else a Playground clone next to this file or in ~/venvs.
_XML = os.path.join("mujoco_playground", "_src", "locomotion", "berkeley_humanoid", "xmls",
                    "scene_mjx_feetonly_flat_terrain.xml")
_XML_PATHS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_playground", _XML),
              os.path.join(os.path.expanduser("~"), "venvs", "mujoco_playground_src", _XML)]
DEFAULT_XML = os.environ.get("BH_XML") or next((p for p in _XML_PATHS if os.path.exists(p)), _XML_PATHS[0])

# Reward scales copied from Playground's default_config(); zero-scale terms are dropped.
REWARD_SCALES = dict(
    tracking_lin_vel=1.0,
    tracking_ang_vel=0.5,
    ang_vel_xy=-0.15,
    orientation=-1.0,
    torques=-2.5e-5,
    action_rate=-0.01,
    feet_air_time=2.0,
    feet_slip=-0.25,
    feet_phase=1.0,
    termination=-1.0,
    joint_deviation_knee=-0.1,
    joint_deviation_hip=-0.25,
    dof_pos_limits=-1.0,
    pose=-1.0,
)
TRACKING_SIGMA = 0.5
# Tighter on terrain: at 0.5, standing still in front of stairs still earns 60% of the
# tracking reward at 0.5 m/s, so stalling beats risking a fall.
TRACKING_SIGMA_TERRAIN = 0.25
MAX_FOOT_HEIGHT = 0.1
ACTION_SCALE = 0.5
CTRL_DT = 0.02
SOFT_LIMIT_FACTOR = 0.95
NOISE_SCALES = dict(hip_pos=0.03, kfe_pos=0.05, ffe_pos=0.08, faa_pos=0.03,
                    joint_vel=1.5, gravity=0.05, linvel=0.1, gyro=0.2)
# Torso height below which we call it a fall. Playground only terminates when the
# torso is upside down; this ends episodes earlier so fewer CPU steps are wasted.
MIN_TORSO_HEIGHT = 0.3
OBS_BLIND = 52
# Height scan: terrain sampled on a grid fixed to the robot's heading, 20 cm behind it
# to 80 cm ahead and 20 cm to each side (metres, x forward, y left).
SCAN_X, SCAN_Y = (g.ravel() for g in np.meshgrid(np.arange(-2, 9) * 0.1, np.arange(-2, 3) * 0.1))
SCAN_NOISE = 0.01


def get_rz(phi, swing_height):
    """Reference foot height for a gait phase in [-pi, pi) (Playground gait.get_rz)."""
    def bezier(y0, y1, x):
        return y0 + (y1 - y0) * (x**3 + 3 * (x**2 * (1 - x)))
    x = (phi + np.pi) / (2 * np.pi)
    return np.where(x <= 0.5, bezier(0, swing_height, 2 * x), bezier(swing_height, 0, 2 * x - 1))


@functools.lru_cache(maxsize=None)
def _load_model(xml_path, terrain):
    """One read-only model per process: each copy is ~100 MB, and only MjData changes per env."""
    course = None
    if terrain:
        from terrain import Terrain
        course = Terrain()
        xml_path = course.scene_xml(os.path.dirname(xml_path))
    # Compiling the XML leaves ~500 MB of heap behind in each process, loading a binary doesn't.
    with open(xml_path, "rb") as f:
        tag = hashlib.sha1(f.read() + mujoco.__version__.encode()).hexdigest()[:12]
    mjb = f"{xml_path}.{tag}.mjb"
    if not os.path.exists(mjb):
        tmp = f"{mjb}.{os.getpid()}.tmp"
        mujoco.mj_saveModel(mujoco.MjModel.from_xml_path(xml_path), tmp)
        os.replace(tmp, mjb)  # several worker processes may get here at once
    model = mujoco.MjModel.from_binary_path(mjb)
    if course is not None:
        course.apply(model)
    return model, course


class BerkeleyHumanoidEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CTRL_DT)}

    def __init__(
        self,
        xml_path=DEFAULT_XML,
        cmd_x=(0.0, 1.0),
        cmd_y=(-0.3, 0.3),
        cmd_yaw=(-0.5, 0.5),
        zero_cmd_prob=0.1,
        push=False,
        noise=1.0,
        episode_steps=1000,
        render_mode=None,
        render_size=(480, 640),
        terrain=False,
        spawn_tile=None,
        swing_height=None,
        random_init=True,
        height_scan=False,
    ):
        # terrain=True swaps the flat floor for terrain.py's course of rough ground,
        # slopes and stairs, with a per-env difficulty curriculum. spawn_tile=(type, level)
        # pins the spawn tile and turns the curriculum off (used by play.py).
        self.model, self.terrain = _load_model(xml_path, terrain)
        self.data = mujoco.MjData(self.model)
        self.spawn_tile = spawn_tile
        self.random_init = random_init  # False: start from the home pose (eval / viewing)
        self.height_scan = height_scan
        self.levels = np.zeros(4, dtype=int)  # curriculum level per terrain type
        self.type_idx = 0
        self.swing_height = swing_height or (0.12 if terrain else MAX_FOOT_HEIGHT)
        self.tracking_sigma = TRACKING_SIGMA_TERRAIN if terrain else TRACKING_SIGMA
        self.n_substeps = int(round(CTRL_DT / self.model.opt.timestep))
        self.dt = CTRL_DT

        self.cmd_x, self.cmd_y, self.cmd_yaw = cmd_x, cmd_y, cmd_yaw
        self.zero_cmd_prob = zero_cmd_prob
        self.push = push
        self.noise = noise
        self.episode_steps = episode_steps
        self.fixed_command = None  # set by play.py to drive the robot by hand

        m = self.model
        self.init_q = m.keyframe("home").qpos.copy()
        self.default_pose = self.init_q[7:].copy()
        lowers, uppers = m.jnt_range[1:].T  # first joint is the freejoint
        c, r = (lowers + uppers) / 2, uppers - lowers
        self.soft_lowers = c - 0.5 * r * SOFT_LIMIT_FACTOR
        self.soft_uppers = c + 0.5 * r * SOFT_LIMIT_FACTOR

        self.hip_idx = np.array([m.joint(f"{s}_{j}").qposadr[0] - 7
                                 for s in ("LL", "LR") for j in ("HR", "HAA")])
        self.knee_idx = np.array([m.joint(f"{s}_KFE").qposadr[0] - 7 for s in ("LL", "LR")])
        self.pose_weights = np.array([1.0, 1.0, 0.01, 0.01, 1.0, 1.0] * 2)

        self.imu_site = m.site("imu").id
        self.feet_sites = np.array([m.site(n).id for n in ("l_foot", "r_foot")])
        self.torso_id = m.body("torso").id

        def sensor_slice(name):
            s = m.sensor(name)
            return slice(m.sensor_adr[s.id], m.sensor_adr[s.id] + m.sensor_dim[s.id])
        self.s = {n: sensor_slice(n) for n in (
            "gyro", "local_linvel", "upvector", "global_linvel", "global_angvel",
            "l_foot1_floor_found", "r_foot1_floor_found")}

        qnoise = np.zeros(12)
        qnoise[[0, 1, 2, 6, 7, 8]] = NOISE_SCALES["hip_pos"]
        qnoise[[3, 9]] = NOISE_SCALES["kfe_pos"]
        qnoise[[4, 10]] = NOISE_SCALES["ffe_pos"]
        qnoise[[5, 11]] = NOISE_SCALES["faa_pos"]
        # Per-dimension noise for the whole obs vector (0 for command, last action, phase).
        self.obs_noise = np.concatenate([
            np.full(3, NOISE_SCALES["linvel"]), np.full(3, NOISE_SCALES["gyro"]),
            np.full(3, NOISE_SCALES["gravity"]), np.zeros(3), qnoise,
            np.full(12, NOISE_SCALES["joint_vel"]), np.zeros(12), np.zeros(4),
            np.full(SCAN_X.size if height_scan else 0, SCAN_NOISE)])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.obs_noise.shape, np.float32)

        self.render_mode = render_mode
        self._renderer = None
        self._render_size = render_size

    # ---------- helpers ----------
    def _sensor(self, name):
        return self.data.sensordata[self.s[name]]

    def _ground(self, x, y):
        return np.zeros_like(x) if self.terrain is None else self.terrain.height(x, y)

    def _scan(self):
        """Torso height above each scan point, minus the standing height (about 0 on flat ground)."""
        d = self.data
        fx, fy = d.site_xmat[self.imu_site][[0, 3]]  # the IMU's x axis points forward
        n = np.hypot(fx, fy) + 1e-6
        fx, fy = fx / n, fy / n
        x = d.qpos[0] + fx * SCAN_X - fy * SCAN_Y
        y = d.qpos[1] + fy * SCAN_X + fx * SCAN_Y
        return np.clip(d.qpos[2] - self._ground(x, y) - 0.5, -1.0, 1.0)

    def _update_curriculum(self):
        """Isaac Lab-style: harder tile after walking off the tile, easier after an early fall
        or after covering less than half the commanded distance."""
        from terrain import N_LEVELS, TILE
        if self.spawn_tile is not None or not hasattr(self, "spawn_xy"):
            return
        dist = np.linalg.norm(self.data.qpos[:2] - self.spawn_xy)
        level = self.levels[self.type_idx]
        if dist > TILE / 2:
            level = level + 1 if level < N_LEVELS - 1 else int(self.np_random.integers(N_LEVELS))
        elif (self.fell and dist < TILE / 4) or dist < 0.5 * self.cmd_path:
            level = max(level - 1, 0)
        self.levels[self.type_idx] = level

    def _contact(self):
        return np.array([self._sensor("l_foot1_floor_found")[0] > 0,
                         self._sensor("r_foot1_floor_found")[0] > 0])

    def sample_command(self):
        if self.fixed_command is not None:
            return np.asarray(self.fixed_command, dtype=np.float64)
        if self.np_random.uniform() < self.zero_cmd_prob:
            return np.zeros(3)
        return np.array([self.np_random.uniform(*self.cmd_x),
                         self.np_random.uniform(*self.cmd_y),
                         self.np_random.uniform(*self.cmd_yaw)])

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        qpos = self.init_q.copy()
        qvel = np.zeros(self.model.nv)
        qpos[0:2] += rng.uniform(-0.5, 0.5, 2)
        yaw = rng.uniform(-3.14, 3.14)
        yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        mujoco.mju_mulQuat(qpos[3:7], self.init_q[3:7].copy(), yaw_quat)
        if self.random_init:
            qpos[7:] *= rng.uniform(0.5, 1.5, 12)
            qvel[0:6] = rng.uniform(-0.5, 0.5, 6)

        if self.terrain is not None:
            from terrain import TYPES
            self._update_curriculum()
            if self.spawn_tile is not None:
                self.type_idx, level = TYPES.index(self.spawn_tile[0]), self.spawn_tile[1]
            else:
                self.type_idx = int(rng.integers(len(TYPES)))
                level = self.levels[self.type_idx]
            self.spawn_xy = self.terrain.tile_center(self.type_idx, level) + rng.uniform(-0.3, 0.3, 2)
            qpos[0:2] = self.spawn_xy
            qpos[2] += self._ground(*self.spawn_xy)
        self.fell = False

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = qpos[7:]
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0      # total steps this episode (for truncation)
        self.cmd_step = 0        # steps since last command resample
        self.cmd_path = 0.0      # distance the commands asked for so far
        self.command = self.sample_command()
        self.last_act = np.zeros(self.model.nu)
        self.feet_air_time = np.zeros(2)
        self.last_contact = np.zeros(2, dtype=bool)
        self.swing_peak = np.zeros(2)
        gait_freq = rng.uniform(1.25, 1.5)
        self.phase_dt = 2 * np.pi * self.dt * gait_freq
        self.phase = np.array([0.0, np.pi])
        self.push_interval = int(round(rng.uniform(5.0, 10.0) / self.dt))
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        d = self.data

        if self.push and (self.step_count + 1) % self.push_interval == 0:
            theta = self.np_random.uniform(0, 2 * np.pi)
            mag = self.np_random.uniform(0.1, 2.0)
            d.qvel[:2] += mag * np.array([np.cos(theta), np.sin(theta)])

        d.ctrl[:] = self.default_pose + action * ACTION_SCALE
        mujoco.mj_step(self.model, d, nstep=self.n_substeps)

        contact = self._contact()
        contact_filt = contact | self.last_contact
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt
        feet = d.site_xpos[self.feet_sites]
        foot_z = feet[:, 2] - self._ground(feet[:, 0], feet[:, 1])  # height above the ground below
        self.swing_peak = np.maximum(self.swing_peak, foot_z)

        bad = np.isnan(d.qpos).any() or np.isnan(d.qvel).any()
        torso_h = d.qpos[2] - self._ground(d.qpos[0], d.qpos[1])
        fell = self._sensor("upvector")[2] < 0.0 or torso_h < MIN_TORSO_HEIGHT
        terminated = bool(bad or fell)
        self.fell = terminated

        terms = self._reward_terms(action, contact, first_contact, foot_z, terminated)
        reward = float(np.clip(sum(REWARD_SCALES[k] * v for k, v in terms.items()) * self.dt, 0.0, 1e4))

        # Advance the clock / histories.
        self.step_count += 1
        self.cmd_step += 1
        self.cmd_path += np.linalg.norm(self.command[:2]) * self.dt
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        self.last_act = action
        if self.cmd_step > 500:
            self.command = self.sample_command()
            self.cmd_step = 0
        self.feet_air_time *= ~contact
        self.last_contact = contact
        self.swing_peak *= ~contact

        if bad:  # don't feed NaNs to the policy
            mujoco.mj_resetData(self.model, d)
            mujoco.mj_forward(self.model, d)
        truncated = self.step_count >= self.episode_steps
        if self.terrain is not None and not self.terrain.inside(d.qpos[0], d.qpos[1]):
            truncated = True  # walked off the edge of the map
        info = {"forward_vel": float(self._sensor("local_linvel")[0]),
                "cmd_vx": float(self.command[0])}
        if self.terrain is not None:
            info["terrain_levels"] = self.levels.copy()
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        d = self.data
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate([
            self._sensor("local_linvel"),
            self._sensor("gyro"),
            gravity,
            self.command,
            d.qpos[7:] - self.default_pose,
            d.qvel[6:],
            self.last_act,
            np.cos(self.phase),
            np.sin(self.phase),
            self._scan() if self.height_scan else [],
        ])
        if self.noise:
            obs += self.np_random.uniform(-1.0, 1.0, obs.shape) * self.noise * self.obs_noise
        return obs.astype(np.float32)

    def _reward_terms(self, action, contact, first_contact, foot_z, terminated):
        d, cmd = self.data, self.command
        qpos = d.qpos[7:]
        local_vel = self._sensor("local_linvel")
        gyro = self._sensor("gyro")
        cmd_norm = np.sqrt(cmd @ cmd)

        def sq(x):
            return float(x @ x)

        air = np.minimum((self.feet_air_time - 0.2) * first_contact, 0.5 - 0.2)
        out_of_limits = np.maximum(self.soft_lowers - qpos, 0.0) + np.maximum(qpos - self.soft_uppers, 0.0)
        rz = get_rz(self.phase, self.swing_height)
        body_xy_speed = np.sqrt(sq(self._sensor("global_linvel")[:2]))
        dq = qpos - self.default_pose

        return dict(
            tracking_lin_vel=np.exp(-sq(cmd[:2] - local_vel[:2]) / self.tracking_sigma),
            tracking_ang_vel=np.exp(-((cmd[2] - gyro[2]) ** 2) / self.tracking_sigma),
            ang_vel_xy=sq(self._sensor("global_angvel")[:2]),
            orientation=sq(self._sensor("upvector")[:2]),
            torques=np.abs(d.actuator_force).sum(),
            action_rate=sq(action - self.last_act),
            feet_air_time=air.sum() * (cmd_norm > 0.1),
            feet_slip=body_xy_speed * contact.sum(),
            feet_phase=np.exp(-sq(foot_z - rz) / 0.01),
            termination=float(terminated),
            joint_deviation_knee=np.abs(dq[self.knee_idx]).sum(),
            joint_deviation_hip=np.abs(dq[self.hip_idx]).sum() * (abs(cmd[1]) > 0.1),
            dof_pos_limits=out_of_limits.sum(),
            pose=float(dq**2 @ self.pose_weights),
        )

    def render(self):
        if self._renderer is None:
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = self.torso_id
            self._cam.distance, self._cam.elevation, self._cam.azimuth = 2.0, -15.0, 120.0
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="BerkeleyHumanoidWalk-v0", entry_point=BerkeleyHumanoidEnv)
