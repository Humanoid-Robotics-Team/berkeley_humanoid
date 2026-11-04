"""Procedural terrain course on a single MuJoCo heightfield.

A grid of TILE x TILE tiles: one column per terrain type, one row per difficulty
level. Every tile has height 0 along its border so neighbouring tiles join up,
and a flat platform in the middle where robots spawn.
"""

import os

import numpy as np

TYPES = ("rough", "slope", "stairs_up", "stairs_down")
N_LEVELS = 5
TILE = 5.0        # m, tile side
MARGIN = 3.0      # m of flat ground around the course
RES = 0.04        # m per heightfield cell
PLATFORM = 0.5    # m, half-size of the flat spawn platform
STEP_WIDTH = 0.30  # m, stair tread depth


def stair_height(d):
    return 0.02 + 0.08 * d          # 2 cm .. 10 cm


def slope_grade(d):
    return 0.05 + 0.25 * d          # 3 deg .. 17 deg


def rough_amplitude(d):
    return 0.01 + 0.04 * d          # +-1 cm .. +-5 cm


class Terrain:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.nx = int(round((len(TYPES) * TILE + 2 * MARGIN) / RES)) + 1
        self.ny = int(round((N_LEVELS * TILE + 2 * MARGIN) / RES)) + 1
        self.half_x = (self.nx - 1) * RES / 2
        self.half_y = (self.ny - 1) * RES / 2
        H = np.zeros((self.ny, self.nx))

        n = int(round(TILE / RES))
        local = (np.arange(n) + 0.5) * RES - TILE / 2
        dx, dy = np.meshgrid(local, local)
        r = np.maximum(np.abs(dx), np.abs(dy))  # square "radius" for pyramids
        for t, name in enumerate(TYPES):
            for lvl in range(N_LEVELS):
                d = lvl / (N_LEVELS - 1)
                c0 = int(round(MARGIN / RES)) + t * n
                r0 = int(round(MARGIN / RES)) + lvl * n
                H[r0:r0 + n, c0:c0 + n] = self._tile(name, d, lvl, r, rng)
        self.H = H

    @staticmethod
    def _tile(name, d, lvl, r, rng):
        rc = np.clip(r, PLATFORM, TILE / 2)
        if name == "rough":
            k = 3  # noise cells of 3 x RES = 12 cm
            m = r.shape[0] // k + 1
            coarse = rng.uniform(-1, 1, (m, m)) * rough_amplitude(d)
            h = np.kron(coarse, np.ones((k, k)))[: r.shape[0], : r.shape[1]]
            return np.where(r < PLATFORM, 0.0, h)
        if name == "slope":
            sign = 1.0 if lvl % 2 == 0 else -1.0  # alternate hill / valley
            return sign * slope_grade(d) * (TILE / 2 - rc)
        steps = np.floor((TILE / 2 - rc) / STEP_WIDTH) * stair_height(d)
        # stairs_down: centre is the top of a pyramid; stairs_up: centre is a pit.
        return steps if name == "stairs_down" else -steps

    # ---------- queries ----------
    def tile_center(self, type_idx, level):
        x = -self.half_x + MARGIN + (type_idx + 0.5) * TILE
        y = -self.half_y + MARGIN + (level + 0.5) * TILE
        return np.array([x, y])

    def height(self, x, y):
        col = np.clip(np.rint((np.asarray(x) + self.half_x) / RES).astype(int), 0, self.nx - 1)
        row = np.clip(np.rint((np.asarray(y) + self.half_y) / RES).astype(int), 0, self.ny - 1)
        return self.H[row, col]

    def inside(self, x, y, pad=0.5):
        return abs(x) < self.half_x - pad and abs(y) < self.half_y - pad

    # ---------- MuJoCo glue ----------
    def scene_xml(self, robot_xml_dir):
        """Writes the terrain scene next to the robot XML (so its relative mesh paths resolve)."""
        zmin, zmax = self.H.min(), self.H.max()
        xml = f"""<mujoco model="berkeley humanoid terrain scene">
  <include file="berkeley_humanoid_mjx_feetonly.xml"/>
  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".9 .93 1" rgb2=".55 .65 .8" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge" rgb1=".62 .62 .6" rgb2=".52 .52 .5"
      markrgb=".3 .3 .3" width="512" height="512"/>
    <material name="terrain" texture="grid" texrepeat="10 10" texuniform="true"/>
    <hfield name="terrain" nrow="{self.ny}" ncol="{self.nx}"
      size="{self.half_x:.4f} {self.half_y:.4f} {max(zmax - zmin, 1e-3):.4f} 0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 {zmin:.4f}" material="terrain"
      contype="1" conaffinity="0" priority="1" friction="0.8" condim="3"/>
  </worldbody>
  <include file="sensor.xml"/>
  <keyframe>
    <key name="home"
      qpos="0 0 0.515 1 0 0 0
            -0.071 0.103 -0.463 0.983 -0.350 0.126
            0.071 -0.103 -0.463 0.983 -0.350 -0.126"
      ctrl="-0.071 0.103 -0.463 0.983 -0.350 0.126
            0.071 -0.103 -0.463 0.983 -0.350 -0.126"/>
  </keyframe>
</mujoco>
"""
        path = os.path.join(robot_xml_dir, "bh_terrain_scene.xml")
        old = open(path).read() if os.path.exists(path) else None
        if old != xml:  # several worker processes may get here at once
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(xml)
            os.replace(tmp, path)
        return path

    def apply(self, model):
        """Copies the heights into the compiled model's heightfield."""
        zmin, zmax = self.H.min(), self.H.max()
        model.hfield_data[:] = ((self.H - zmin) / max(zmax - zmin, 1e-3)).ravel()
