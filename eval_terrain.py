"""Score a policy on every terrain tile type and difficulty level.

    python eval_terrain.py --model runs/bh_terrain/final_model.zip

For each (tile, level) it runs several 8 s episodes walking forward at --vx from the
tile centre (random heading) and reports how many finished without falling.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from bh_env import BerkeleyHumanoidEnv, OBS_BLIND
from play import load_obs_normalizer, resolve_model
from terrain import N_LEVELS, TYPES, stair_height


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/bh_terrain/final_model.zip")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--vx", type=float, default=0.5)
    args = p.parse_args()

    path = resolve_model(args.model)
    model = PPO.load(path, device="cpu")
    norm = load_obs_normalizer(path)

    print(f"success rate (no fall in 8 s at vx={args.vx}) — {path}")
    print(f"{'level':>14} " + " ".join(f"{l:>6}" for l in range(N_LEVELS)))
    print(f"{'stair height':>14} " + " ".join(f"{stair_height(l / (N_LEVELS - 1)) * 100:4.0f}cm" for l in range(N_LEVELS)))
    for name in TYPES:
        row = []
        for lvl in range(N_LEVELS):
            env = BerkeleyHumanoidEnv(terrain=True, spawn_tile=(name, lvl), random_init=False,
                                      height_scan=model.observation_space.shape[0] > OBS_BLIND)
            env.fixed_command = np.array([args.vx, 0.0, 0.0])
            ok = 0
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=1000 + ep)
                for _ in range(400):
                    obs, _, term, _, _ = env.step(model.predict(norm(obs), deterministic=True)[0])
                    if term:
                        break
                ok += not term
            row.append(f"{ok}/{args.episodes}")
        print(f"{name:>14} " + " ".join(f"{r:>6}" for r in row))


if __name__ == "__main__":
    main()
