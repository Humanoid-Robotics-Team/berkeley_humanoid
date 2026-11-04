"""Renders the images for the Medium write-up into article_images/.

    python make_article_images.py
"""

import argparse
import os

import imageio
import matplotlib
import mujoco
import numpy as np
from stable_baselines3 import PPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bh_env import BerkeleyHumanoidEnv  # noqa: E402
from play import load_obs_normalizer, resolve_model  # noqa: E402
from terrain import TYPES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "article_images")
W, H = 1600, 900
ACCENT, GREY, INK, QUIET = "#2b7bd8", "#c3c2b8", "#1f1f1f", "#6b6b6b"


def make_renderer(env):
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, W)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, H)
    return mujoco.Renderer(env.model, height=H, width=W)


def tracking_cam(env, distance, elevation, azimuth):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.torso_id
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    return cam


def run_and_pick(env, model_path, cmd, steps, score, seed=0, yaw=0.0):
    """Runs the policy, returns the rendered frame (after warm-up) with the highest score(env)."""
    model = PPO.load(model_path, device="cpu")
    norm = load_obs_normalizer(model_path)
    env.fixed_command = np.array(cmd, float)
    obs, _ = env.reset(seed=seed)
    if yaw is not None:  # face a known direction so the camera framing is predictable
        env.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(env.model, env.data)
    best, best_state = -np.inf, None
    for t in range(steps):
        obs, _, term, _, _ = env.step(model.predict(norm(obs), deterministic=True)[0])
        if term:
            raise RuntimeError(f"robot fell at step {t}; try another seed")
        if t > steps // 2:
            s = score(env)
            if s > best:
                best, best_state = s, (env.data.qpos.copy(), env.data.qvel.copy())
    env.data.qpos[:], env.data.qvel[:] = best_state
    mujoco.mj_forward(env.model, env.data)


def stride(env):
    feet = env.data.site_xpos[env.feet_sites]
    return np.linalg.norm(feet[0, :2] - feet[1, :2])


def hero(flat_model):
    env = BerkeleyHumanoidEnv(noise=0.0, random_init=False)
    run_and_pick(env, flat_model, (0.6, 0, 0), 200, stride)
    r = make_renderer(env)
    r.update_scene(env.data, tracking_cam(env, 1.7, -12, 135))
    imageio.imwrite(os.path.join(OUT, "01_hero_walking.png"), r.render())


def course_overview():
    env = BerkeleyHumanoidEnv(terrain=True, spawn_tile=("stairs_down", 2), random_init=False, noise=0.0)
    env.reset(seed=0)
    r = make_renderer(env)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance, cam.elevation, cam.azimuth = 26.0, -38.0, 60.0
    r.update_scene(env.data, cam)
    imageio.imwrite(os.path.join(OUT, "04_terrain_course.png"), r.render())


def stairs(terrain_model, tile, level, name, azimuth, seed=0):
    env = BerkeleyHumanoidEnv(terrain=True, spawn_tile=(tile, level), random_init=False, noise=0.0)
    # Score = how far out onto the stairs the robot is while mid-stride.
    run_and_pick(env, terrain_model, (0.5, 0, 0), 140,
                 lambda e: min(np.linalg.norm(e.data.qpos[:2] - e.spawn_xy), 1.6) + 0.5 * stride(e), seed=seed)
    r = make_renderer(env)
    r.update_scene(env.data, tracking_cam(env, 2.0, -18, azimuth))
    imageio.imwrite(os.path.join(OUT, name), r.render())


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(QUIET)
    ax.spines["bottom"].set_color(QUIET)
    ax.tick_params(colors=QUIET)


def tracking_chart():
    cmd = np.array([0.0, 0.3, 0.6, 1.0])
    got = np.array([0.02, 0.26, 0.57, 0.85])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.plot([0, 1.05], [0, 1.05], ls="--", color=GREY, lw=1.2)
    ax.text(0.80, 0.86, "perfect tracking", color=QUIET, fontsize=10, rotation=33)
    ax.plot(cmd, got, "o-", color=ACCENT, lw=2, ms=7)
    for c, g in zip(cmd, got):
        ax.annotate(f"{g:.2f}", (c, g), textcoords="offset points", xytext=(8, -14), color=INK, fontsize=10)
    ax.set_xlabel("Commanded forward speed (m/s)", color=INK)
    ax.set_ylabel("Measured forward speed (m/s)", color=INK)
    ax.set_xlim(-0.05, 1.1), ax.set_ylim(-0.05, 1.1)
    ax.set_title("The robot tracks speed commands closely up to 0.6 m/s", loc="left", color=INK,
                 fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.02, "Mean over five 10 s runs per command, sensor noise on, no falls in any run",
            transform=ax.transAxes, color=QUIET, fontsize=9.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_speed_tracking.png"))
    plt.close(fig)


def terrain_heatmap():
    # Runs without a fall, out of 20 (8 s at 0.5 m/s), measured with eval_terrain.py --episodes 20.
    before = np.array([[7, 0, 0, 0, 0], [18, 9, 2, 4, 0], [10, 0, 0, 0, 0], [12, 2, 0, 0, 0]])
    after = np.array([[20, 20, 17, 8, 4], [20, 20, 19, 20, 16], [20, 14, 0, 0, 0], [20, 19, 13, 4, 1]])
    rows = ["Rough\n±1 → ±5 cm", "Slope\n3° → 17°", "Stairs up\n2 → 10 cm", "Stairs down\n2 → 10 cm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, sharey=True)
    for ax, data, title in zip(axes, (before, after), ("Flat-ground policy", "After 20M terrain steps")):
        ax.imshow(data / 20, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{data[i, j]}/20", ha="center", va="center", fontsize=11,
                        color="white" if data[i, j] >= 13 else INK)
        ax.set_xticks(range(5), [f"L{j}" for j in range(5)])
        ax.set_yticks(range(4), rows)
        ax.set_title(title, color=INK, fontsize=12)
        ax.tick_params(length=0, colors=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_xlabel("Difficulty level", color=QUIET)
    axes[1].set_xlabel("Difficulty level", color=QUIET)
    fig.suptitle("Terrain training conquered slopes and bumps, but stairs up stop at 4 cm",
                 x=0.02, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.02, 0.885, "Runs out of 20 that finished 8 s at 0.5 m/s without falling, from a standing start",
             color=QUIET, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "06_terrain_results.png"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flat-model", default="runs/bh_ppo/final_model.zip")
    p.add_argument("--terrain-model", default="runs/bh_terrain/final_model.zip")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.chdir(HERE)
    hero(resolve_model(args.flat_model))
    tracking_chart()
    course_overview()
    tm = resolve_model(args.terrain_model)
    stairs(tm, "stairs_down", 2, "05a_stairs_down.png", azimuth=150)
    stairs(tm, "stairs_up", 1, "05b_stairs_up.png", azimuth=150)
    terrain_heatmap()
    print("\n".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
