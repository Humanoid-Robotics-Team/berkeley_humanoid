"""The robot model set up for jumping and flipping, with the real motors' limits.

The walking env only needs the feet to touch the ground and joints that don't move fast.
A flip depends on both, so here:
- every collision geom touches the floor (but not the other links), so a bad landing is a crash;
- torque follows the torque-speed limits from the Berkeley team's Isaac Lab config
  (HybridRobotics/isaac_berkeley_humanoid, assets/berkeley_humanoid.py), applied every physics step;
- damping is part of the PD controller instead of a passive joint force. It then counts against
  the motor's torque limit, as it does on the robot, rather than coming on top of it.
"""

import copy
import functools

import mujoco
import numpy as np

from bh_env import DEFAULT_XML, _load_model

# Joint name suffix: (effort limit Nm, velocity limit rad/s, saturation effort Nm)
MOTORS = dict(HR=(20.0, 23.0, 402.0), HAA=(20.0, 23.0, 402.0), HFE=(30.0, 20.0, 443.0),
              KFE=(30.0, 14.0, 560.0), FFE=(20.0, 23.0, 402.0), FAA=(5.0, 42.0, 112.0))
# PD gains are the controller's, not the hardware's: same stiffness as the walking env.
GAINS = dict(HR=(35.0, 1.5), HAA=(35.0, 1.5), HFE=(35.0, 1.5), KFE=(35.0, 1.5), FFE=(35.0, 1.5), FAA=(1.0, 0.1))
FLOOR_BIT = 2


@functools.lru_cache(maxsize=None)
def load_flip_model(xml_path=DEFAULT_XML):
    m = copy.copy(_load_model(xml_path, False)[0])  # the walking env's cached copy stays as it is
    # Playground's settings are tuned for speed on gentle walking contacts, not hard landings.
    m.opt.iterations, m.opt.ls_iterations = 10, 10

    floor = m.geom("floor").id
    m.geom_contype[floor], m.geom_conaffinity[floor] = FLOOR_BIT, 0
    for g in range(m.ngeom):
        if g != floor and m.geom_group[g] == 3:  # collision geoms
            m.geom_contype[g], m.geom_conaffinity[g] = 0, FLOOR_BIT
    # The compiler caches these per body, and the collision filter reads the cache.
    m.body_contype[:] = m.body_conaffinity[:] = 0
    for g in range(m.ngeom):
        m.body_contype[m.geom_bodyid[g]] |= m.geom_contype[g]
        m.body_conaffinity[m.geom_bodyid[g]] |= m.geom_conaffinity[g]

    # Actuators become plain torque motors; Motors.torque() does the PD and the limits.
    m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    m.actuator_gainprm[:] = 0.0
    m.actuator_biasprm[:] = 0.0
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_ctrllimited[:] = 0
    m.actuator_forcelimited[:] = 0
    m.jnt_actfrclimited[:] = 0
    m.dof_damping[6:] = 0.0
    return m


class Motors:
    def __init__(self, m):
        names = [m.joint(j).name.split("_")[1] for j in range(1, m.njnt)]
        self.effort, self.vmax, self.sat = np.array([MOTORS[n] for n in names]).T
        self.kp, self.kd = np.array([GAINS[n] for n in names]).T

    def torque(self, target, q, qd):
        """PD torque, clipped to the DC motor's torque-speed limits."""
        hi = np.clip(self.sat * (1.0 - qd / self.vmax), 0.0, self.effort)
        lo = np.clip(self.sat * (-1.0 - qd / self.vmax), -self.effort, 0.0)
        return np.clip(self.kp * (target - q) - self.kd * qd, lo, hi)
