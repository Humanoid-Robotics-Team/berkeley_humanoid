"""Watch or score a backflip policy, always from the crouch that every real attempt starts in.

    python play_flip.py                                   # newest model, in the MuJoCo viewer
    python play_flip.py --record flip.mp4 --tries 3       # video, half speed
    python play_flip.py --score 20                        # how many of 20 flips landed
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from flip_env import BackflipEnv, CTRL_DT
from play import load_obs_normalizer, resolve_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/bh_flip_best/final_model.zip")
    p.add_argument("--record", default=None, help="write an .mp4 instead of opening the viewer")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--speed", type=float, default=0.5, help="video playback speed")
    p.add_argument("--score", type=int, default=0, help="run this many flips and print the results")
    p.add_argument("--perturb", type=float, default=0.0,
                   help="disturbance scale: 1 = joints off by up to 0.03 rad at the start, motor commands off by 5%%")
    args = p.parse_args()

    path = resolve_model(args.model)
    model = PPO.load(path, device="cpu")
    norm = load_obs_normalizer(path)
    env = BackflipEnv(air_start_prob=0.0, ref_start_prob=0.0, render_mode="rgb_array" if args.record else None)

    def run(seed, on_step):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        env.data.qpos[7:] += rng.uniform(-0.03, 0.03, 12) * args.perturb
        while True:
            action = model.predict(norm(obs), deterministic=True)[0]
            action = action + rng.normal(0.0, 0.05, 6) * args.perturb
            obs, _, term, trunc, info = env.step(action)
            on_step()
            if term or trunc:
                return info["flip"]

    def describe(f):
        return (f"rotation {np.degrees(f['rot']):4.0f} deg, peak height {f['peak']:.2f} m, "
                f"landed on feet {f['landed']}, stayed up {f['stood']}")

    if args.score:
        results = [run(1000 + i, lambda: None) for i in range(args.score)]
        rot = np.degrees([f["rot"] for f in results])
        print(f"{path}: {args.score} tries from the crouch, rotation {rot.mean():.0f} deg (best {rot.max():.0f})")
        print(f"took off {sum(f['flown'] for f in results)}, landed on feet {sum(f['landed'] for f in results)}, "
              f"full flip and stayed up {sum(f['stood'] for f in results)}")
        return

    if args.record:
        import imageio
        frames, count = [], [0]

        def grab():
            count[0] += 1
            if count[0] % 2 == 0:
                frames.append(env.render())
        for i in range(args.tries):
            print(f"try {i + 1}: {describe(run(i, grab))}")
        imageio.mimsave(args.record, frames, fps=max(int(args.speed / (2 * CTRL_DT)), 1))
        print(f"wrote {args.record}")
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.torso
        viewer.cam.distance = 2.5

        def show():
            viewer.sync()
            time.sleep(CTRL_DT / args.speed)
        seed = 0
        while viewer.is_running():
            print(describe(run(seed, show)))
            seed += 1
            time.sleep(1.0)


if __name__ == "__main__":
    main()
