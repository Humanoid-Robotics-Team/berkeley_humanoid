"""Finds a whole backflip the motors can make, for training to follow.

Cross-entropy search over open-loop joint targets, in two parts:
- Flip, from a crouch: a push on 30 ms knots, then a tuck once airborne. It scores the rotation of
  the part that turned least (torso or foot), so swinging the legs doesn't count, and prefers
  touching down feet first and within the speed limits.
- Landing, from mid-flip on that motion: 30 ms knots, scored on surviving the touchdown and ending
  upright and still.

The saved reference is one unbroken simulation of both parts. With no feedback it tips over slowly
after landing, so it stops once the robot is back up.

    python flip_search.py                                      # search, write flip_reference.npz
    python flip_search.py --replay flip.npy --replay-land land.npy
"""

import argparse
import multiprocessing as mp
import time

import mujoco
import numpy as np

from flip_model import Motors, load_flip_model

KNOTS, KNOT_S = 12, 0.03
DIM = KNOTS * 3 + 3 + 3   # push targets, tuck targets, crouch (hip, knee, lean)
LOG_EVERY = 5             # physics steps per reference frame: the env's 100 Hz
TOP_ROT = 2 * np.pi + 0.3
SWITCH_FRAME = 36         # where the landing part takes over (0.36 s, about 200 degrees round)
LAND_KNOTS, LAND_S, HOLD_S = 24, 0.72, 0.6
REFERENCE_S = 1.7         # length of the saved motion
_g = {}


def init():
    m = load_flip_model()
    _g.update(m=m, d=mujoco.MjData(m), motors=Motors(m), home=m.keyframe("home").qpos.copy(),
              idx=np.array([[m.joint(f"{s}_{j}").qposadr[0] - 7 for s in ("LL", "LR")] for j in ("HFE", "KFE", "FFE")]),
              feet=(m.geom("l_foot1").id, m.geom("r_foot1").id),
              foot_body=m.geom_bodyid[m.geom("l_foot1").id], torso=m.body("torso").id,
              foot_bodies={m.geom_bodyid[m.geom("l_foot1").id], m.geom_bodyid[m.geom("r_foot1").id]},
              floor=m.geom("floor").id, imu=m.site("imu").id)


def crouch(p):
    """Puts the robot in the crouch that parameters p start from; returns the joint positions."""
    m, d, home, idx = _g["m"], _g["d"], _g["home"], _g["idx"]
    mujoco.mj_resetData(m, d)
    d.qpos[:] = home
    hip = np.clip(-0.46 + p[-3], -1.5, 0.0)
    knee = np.clip(0.98 + p[-2], 0.3, 2.0)
    lean = np.clip(p[-1] * 0.3, -0.35, 0.1)
    d.qpos[7 + idx[0]], d.qpos[7 + idx[1]] = hip, knee
    d.qpos[7 + idx[2]] = np.clip(0.17 - hip - knee, -0.52, 0.70)
    d.qpos[3], d.qpos[5] = np.cos(lean / 2), np.sin(lean / 2)
    mujoco.mj_forward(m, d)
    d.qpos[2] -= min(d.geom_xpos[g][2] for g in _g["feet"]) - 0.015
    return d.qpos[7:].copy()


def flip_target(start, knots, tuck, t, airborne):
    target, idx = start.copy(), _g["idx"]
    if airborne:
        target[idx[0]], target[idx[1]], target[idx[2]] = -0.46 + tuck[0], 0.98 + tuck[1], -0.35 + tuck[2]
    else:
        k = min(int(t / KNOT_S), KNOTS - 1)
        for j in range(3):
            target[idx[j]] = start[idx[j]] + knots[k, j]
    return target


def land_target(land, t):
    home, idx = _g["home"], _g["idx"]
    target = home[7:].copy()
    if t < LAND_S:
        k = min(int(t / KNOT_S), LAND_KNOTS - 1)
        for j in range(3):
            target[idx[j]] = home[7 + idx[j][0]] + land[k, j]
    return target


def touching():
    """Whether a foot and whether anything else is on the floor."""
    m, d = _g["m"], _g["d"]
    foot, body = False, False
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if m.geom_bodyid[g2 if g1 == _g["floor"] else g1] in _g["foot_bodies"]:
            foot = True
        else:
            body = True
    return foot, body


def rollout(p):
    """Flip search: how far round does it get by the first touchdown?"""
    m, d, motors = _g["m"], _g["d"], _g["motors"]
    start = crouch(p)
    knots, tuck = p[:KNOTS * 3].reshape(KNOTS, 3), p[KNOTS * 3:KNOTS * 3 + 3]
    airborne, air_t, torso_rot, foot_rot, feet_first, over = False, 0.0, 0.0, 0.0, False, 0.0
    dt = m.opt.timestep
    for i in range(int(1.6 / dt)):
        target = flip_target(start, knots, tuck, i * dt, airborne)
        d.ctrl[:] = motors.torque(target, d.qpos[7:], d.qvel[6:])
        mujoco.mj_step(m, d)
        torso_rot -= d.cvel[_g["torso"]][1] * dt
        foot_rot -= d.cvel[_g["foot_body"]][1] * dt
        over = max(over, float(np.max(np.abs(d.qvel[6:]) / motors.vmax)))
        if d.ncon == 0:
            air_t += dt
            airborne = airborne or air_t >= 0.03
        elif airborne:
            feet_first = all(d.contact[c].geom1 in _g["feet"] or d.contact[c].geom2 in _g["feet"] for c in range(d.ncon))
            break
        else:
            air_t = 0.0
    rot = min(torso_rot, foot_rot)
    score = min(rot, TOP_ROT) + 0.5 * air_t + feet_first - 0.5 * max(over - 1.0, 0.0) if airborne else -5.0
    return dict(score=score, rot=rot)


def advance_to_switch(p):
    """Runs the flip part up to where the landing takes over, so both parts are one simulation."""
    m, d, motors = _g["m"], _g["d"], _g["motors"]
    start = crouch(p)
    knots, tuck = p[:KNOTS * 3].reshape(KNOTS, 3), p[KNOTS * 3:KNOTS * 3 + 3]
    airborne, air_t, torso_rot, foot_rot = False, 0.0, 0.0, 0.0
    dt = m.opt.timestep
    log = []
    for i in range(SWITCH_FRAME * LOG_EVERY):
        if i % LOG_EVERY == 0:
            log.append(np.concatenate([d.qpos, d.qvel, [torso_rot, foot_rot, airborne, False]]))
        d.ctrl[:] = motors.torque(flip_target(start, knots, tuck, i * dt, airborne), d.qpos[7:], d.qvel[6:])
        mujoco.mj_step(m, d)
        torso_rot -= d.cvel[_g["torso"]][1] * dt
        foot_rot -= d.cvel[_g["foot_body"]][1] * dt
        if d.ncon == 0:
            air_t += dt
            airborne = airborne or air_t >= 0.03
        elif not airborne:
            air_t = 0.0
    return log, torso_rot, foot_rot


def rollout_landing(args):
    """Landing search: from the switch point, survive the touchdown and end upright and still."""
    p, land = args
    m, d, motors = _g["m"], _g["d"], _g["motors"]
    advance_to_switch(p)
    land = land.reshape(LAND_KNOTS, 3)
    t_touch, hit_t, dt = None, None, m.opt.timestep
    for i in range(int((LAND_S + HOLD_S) / dt)):
        d.ctrl[:] = motors.torque(land_target(land, i * dt), d.qpos[7:], d.qvel[6:])
        mujoco.mj_step(m, d)
        foot, body = touching()
        if foot and t_touch is None:
            t_touch = i * dt
        if body:
            hit_t = i * dt
            break
    up = d.site_xmat[_g["imu"]][8]
    end = hit_t if hit_t is not None else LAND_S + HOLD_S
    survive = end - (t_touch if t_touch is not None else end)
    if hit_t is not None:
        return dict(score=4.0 * survive - 2.0, up=float(up))
    return dict(score=4.0 * survive + 3.0 * up - 0.1 * min(abs(d.qvel[4]), 20.0) + 2.0 * (up > 0.7), up=float(up))


def reference(p, land):
    """The whole motion as one simulation, at the env's 100 Hz."""
    m, d, motors = _g["m"], _g["d"], _g["motors"]
    log, torso_rot, foot_rot = advance_to_switch(p)
    land, dt = land.reshape(LAND_KNOTS, 3), m.opt.timestep
    landed, peak = False, np.zeros(12)
    first_touch, body_hit = None, None
    before = SWITCH_FRAME * LOG_EVERY
    for i in range(int(REFERENCE_S / dt) - before):
        if i % LOG_EVERY == 0:
            log.append(np.concatenate([d.qpos, d.qvel, [torso_rot, foot_rot, True, landed]]))
        d.ctrl[:] = motors.torque(land_target(land, i * dt), d.qpos[7:], d.qvel[6:])
        mujoco.mj_step(m, d)
        torso_rot -= d.cvel[_g["torso"]][1] * dt
        foot_rot -= d.cvel[_g["foot_body"]][1] * dt
        peak = np.maximum(peak, np.abs(d.qvel[6:]) / motors.vmax)
        foot, body = touching()
        if foot and not landed:
            landed, first_touch = True, (before + i) * dt
        if body and body_hit is None:
            body_hit = (before + i) * dt
    up = float(d.site_xmat[_g["imu"]][8])
    print(f"touchdown {first_touch:.2f} s, body contact {body_hit}, at {REFERENCE_S} s: upright {up:.2f}, "
          f"torso {np.degrees(torso_rot):.0f} deg, foot {np.degrees(foot_rot):.0f} deg, "
          f"peak speed {peak.max():.2f} x limit after the switch")
    return np.array(log)


def cem(fn, make_args, mu, sigma, gens, floor, name, seeds=3, pop=128, elite=16):
    best = None
    t0 = time.time()
    with mp.get_context("spawn").Pool(4, initializer=init) as pool:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            mean, std = mu.copy(), sigma.copy()
            for _ in range(gens):
                xs = mean + std * rng.standard_normal((pop, mu.size))
                results = pool.map(fn, [make_args(x) for x in xs], chunksize=8)
                order = np.argsort([r["score"] for r in results])[::-1]
                top = xs[order[:elite]]
                mean, std = top.mean(0), 0.9 * std + 0.1 * (top.std(0) + floor)
                if best is None or results[order[0]]["score"] > best[0]["score"]:
                    best = results[order[0]], xs[order[0]].copy()
            print(f"{name} seed {seed} {time.time() - t0:4.0f}s {best[0]}", flush=True)
    return best[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default=None, help="saved flip parameters (.npy) instead of searching")
    p.add_argument("--replay-land", default=None, help="saved landing parameters (.npy) instead of searching")
    p.add_argument("--out", default="flip_reference.npz")
    args = p.parse_args()

    init()
    if args.replay:
        flip = np.load(args.replay)
    else:
        mu = np.zeros(DIM)
        mu[:KNOTS * 3] = np.tile([0.8, -1.5, 0.8], KNOTS)
        flip = cem(rollout, lambda x: x, mu, np.full(DIM, 1.5), 90, 0.1, "flip")
    if args.replay_land:
        land = np.load(args.replay_land)
    else:
        mu = np.tile([-0.8, 0.8, 0.2], (LAND_KNOTS, 1)).ravel()
        land = cem(rollout_landing, lambda x: (flip, x), mu, np.full(mu.size, 0.8), 70, 0.05, "landing")

    log = reference(flip, land)
    m = _g["m"]
    np.savez(args.out, params=flip, land_params=land, qpos=log[:, :m.nq], qvel=log[:, m.nq:m.nq + m.nv],
             torso_rot=log[:, -4], foot_rot=log[:, -3], airborne=log[:, -2], landed=log[:, -1],
             dt=LOG_EVERY * m.opt.timestep)
    print(f"{args.out}: {len(log)} frames")


if __name__ == "__main__":
    main()
