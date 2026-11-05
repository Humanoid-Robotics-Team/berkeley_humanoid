"""Train the backflip with PPO on CPU.

    python train_flip.py --steps 15e6
    python train_flip.py --resume runs/bh_flip/checkpoints/bh_5000000_steps.zip --steps 10e6

The last checkpoint is often not the best: score them with play_flip.py --score 20 --perturb 1.
"""

import argparse
import os

import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from batched_vec_env import BatchedSubprocVecEnv
from flip_env import BackflipEnv, FULL_FLIP
from train import vecnormalize_path


class FlipStatsCallback(BaseCallback):
    """Logs results separately for episodes started from the crouch and from part way through."""

    def _on_step(self):
        for info in self.locals["infos"]:
            if "flip" in info:
                f = info["flip"]
                tag = "air" if f["from_air"] else "crouch"
                self.logger.record_mean(f"flip_{tag}/rotation_deg", f["rot"] * 360 / FULL_FLIP)
                self.logger.record_mean(f"flip_{tag}/landed", float(f["landed"]))
                self.logger.record_mean(f"flip_{tag}/stood", float(f["stood"]))
                if not f["from_air"]:
                    self.logger.record_mean("flip_crouch/flown", float(f["flown"]))
                    self.logger.record_mean("flip_crouch/peak_com", f["peak"])
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=float, default=15e6)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="bh_flip")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    run_dir = os.path.join("runs", args.name)
    torch.set_num_threads(4)
    venv = make_vec_env(BackflipEnv, n_envs=args.envs, vec_env_cls=BatchedSubprocVecEnv,
                        vec_env_kwargs=dict(n_workers=args.workers))
    if args.resume:
        venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=run_dir)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=2e-4,
            n_steps=512,
            batch_size=3072,
            n_epochs=5,
            gamma=0.99,            # 1 s horizon at 100 Hz: the take-off has to pay off at the landing
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            max_grad_norm=1.0,
            target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                               activation_fn=nn.ELU, log_std_init=-1.3),
            tensorboard_log=run_dir,
            device="cpu",
            verbose=1,
        )
    callbacks = [
        CheckpointCallback(save_freq=max(500_000 // args.envs, 1), save_path=os.path.join(run_dir, "checkpoints"),
                           name_prefix="bh", save_vecnormalize=True),
        FlipStatsCallback(),
    ]
    try:
        model.learn(total_timesteps=int(args.steps), callback=callbacks,
                    reset_num_timesteps=not args.resume, tb_log_name="ppo")
    finally:
        model.save(os.path.join(run_dir, "final_model"))
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
        print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
