"""Watch a trained Berkeley Humanoid policy in the MuJoCo viewer, or record a video.

    python play.py                                  # runs/bh_ppo/final_model.zip
    python play.py --model runs/bh_ppo/checkpoints/bh_5000000_steps.zip
    python play.py --record walk.mp4 --seconds 15

Viewer keys: Up/Down = forward speed, Left/Right = turn, , / . = sideways, Space = stop.
"""

import argparse
import glob
import os
import pickle
import time

import numpy as np
from stable_baselines3 import PPO

from bh_env import BerkeleyHumanoidEnv, OBS_BLIND
from train import vecnormalize_path

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE, KEY_COMMA, KEY_PERIOD = 265, 264, 263, 262, 32, 44, 46


def load_obs_normalizer(model_path):
    """Returns a function that applies the VecNormalize stats saved next to the model."""
    with open(vecnormalize_path(model_path), "rb") as f:
        vn = pickle.load(f)
    rms, eps, clip = vn.obs_rms, vn.epsilon, vn.clip_obs
    return lambda o: np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)


HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_model(path):
    """Finds the model from any working dir; falls back to the newest checkpoint mid-training."""
    for cand in (path, os.path.join(HERE, path)):
        if os.path.exists(cand):
            return cand
    ckpt_dir = os.path.join(os.path.dirname(os.path.join(HERE, path)), "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "bh_*_steps.zip"))
    if not ckpts:
        raise FileNotFoundError(f"no model at {path} and no checkpoints in {ckpt_dir}")
    newest = max(ckpts, key=os.path.getmtime)
    print(f"{path} not found (training still running?), using {os.path.basename(newest)}")
    return newest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/bh_ppo/final_model.zip")
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--record", default=None, help="write an .mp4 instead of opening the viewer")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--tile", default=None, choices=["rough", "slope", "stairs_up", "stairs_down"],
                   help="spawn on the terrain course on this tile type")
    p.add_argument("--level", type=int, default=4, help="terrain difficulty 0-4 (with --tile)")
    args = p.parse_args()

    args.model = resolve_model(args.model)
    model = PPO.load(args.model, device="cpu")
    norm = load_obs_normalizer(args.model)
    env = BerkeleyHumanoidEnv(noise=0.0, episode_steps=10**9, random_init=False,
                              render_mode="rgb_array" if args.record else None,
                              terrain=args.tile is not None,
                              spawn_tile=(args.tile, args.level) if args.tile else None,
                              height_scan=model.observation_space.shape[0] > OBS_BLIND)
    env.fixed_command = np.array([args.vx, args.vy, args.yaw])
    obs, _ = env.reset(seed=0)

    def act(o):
        return model.predict(norm(o), deterministic=True)[0]

    if args.record:
        import imageio
        frames = []
        for _ in range(int(args.seconds / env.dt)):
            obs, _, term, _, _ = env.step(act(obs))
            frames.append(env.render())
            if term:
                obs, _ = env.reset()
        imageio.mimsave(args.record, frames, fps=int(1 / env.dt))
        print(f"wrote {args.record}")
        return

    import mujoco
    import mujoco.viewer

    def on_key(key):
        c = env.fixed_command
        if key == KEY_UP: c[0] = min(c[0] + 0.1, 1.0)
        elif key == KEY_DOWN: c[0] = max(c[0] - 0.1, -1.0)
        elif key == KEY_LEFT: c[2] = min(c[2] + 0.1, 1.0)
        elif key == KEY_RIGHT: c[2] = max(c[2] - 0.1, -1.0)
        elif key == KEY_COMMA: c[1] = min(c[1] + 0.1, 1.0)
        elif key == KEY_PERIOD: c[1] = max(c[1] - 0.1, -1.0)
        elif key == KEY_SPACE: c[:] = 0.0
        env.command = c.copy()
        print(f"command vx={c[0]:+.1f} vy={c[1]:+.1f} yaw={c[2]:+.1f}")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.torso_id
        viewer.cam.distance = 2.0
        while viewer.is_running():
            t0 = time.time()
            obs, _, term, _, _ = env.step(act(obs))
            if term:
                obs, _ = env.reset()
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
