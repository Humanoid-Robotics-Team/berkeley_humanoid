"""Train a Berkeley Humanoid walking policy with PPO on CPU.

    python train.py                      # fresh run, 20M steps
    python train.py --steps 5e6 --envs 6
    python train.py --resume runs/bh_ppo/checkpoints/bh_5000000_steps.zip
    python train.py --terrain --scan --init runs/bh_terrain/final_model.zip --name bh_scan
"""

import argparse
import os
import pickle

import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from batched_vec_env import BatchedSubprocVecEnv
from bh_env import BerkeleyHumanoidEnv, OBS_BLIND
from terrain import TYPES


def vecnormalize_path(model_path):
    """Stats file that CheckpointCallback (or a finished run) saved next to a model."""
    folder, name = os.path.split(model_path)
    ckpt = os.path.join(folder, name.replace(".zip", ".pkl").replace("bh_", "bh_vecnormalize_", 1))
    return ckpt if os.path.exists(ckpt) else os.path.join(folder, "vecnormalize.pkl")


class WalkStatsCallback(BaseCallback):
    """Logs how fast the robot walks vs. the commanded speed."""

    def _on_step(self):
        infos = self.locals["infos"]
        vel = sum(i["forward_vel"] for i in infos) / len(infos)
        cmd = sum(i["cmd_vx"] for i in infos) / len(infos)
        self.logger.record_mean("walk/forward_vel", vel)
        self.logger.record_mean("walk/cmd_vx", cmd)
        if "terrain_levels" in infos[0]:
            levels = sum(i["terrain_levels"] for i in infos) / len(infos)
            self.logger.record_mean("walk/terrain_level", levels.mean())
            for name, level in zip(TYPES, levels):
                self.logger.record_mean(f"walk/level_{name}", level)
        return True


def widen_from_blind(model, venv, blind_path):
    """Starts a height-scan model from a blind one: scan inputs get zero weight, so it acts the same at first."""
    new = model.policy.state_dict()
    for key, old in PPO.load(blind_path, device="cpu").policy.state_dict().items():
        if key == "log_std":  # keep the fresh exploration noise
            continue
        if old.shape == new[key].shape:
            new[key].copy_(old)
        else:
            new[key].zero_()
            new[key][:, :OBS_BLIND] = old
    model.policy.load_state_dict(new)
    with open(vecnormalize_path(blind_path), "rb") as f:
        blind = pickle.load(f)
    venv.obs_rms.mean[:OBS_BLIND] = blind.obs_rms.mean
    venv.obs_rms.var[:OBS_BLIND] = blind.obs_rms.var
    # Start the scan at a 10 cm scale. A high count keeps all the stats steady: with a low
    # one, early falls skew the joint stats, the policy gets worse, and it collapses.
    venv.obs_rms.mean[OBS_BLIND:] = 0.0
    venv.obs_rms.var[OBS_BLIND:] = 0.1**2
    venv.obs_rms.count = blind.obs_rms.count
    venv.ret_rms = blind.ret_rms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=float, default=20e6)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--workers", type=int, default=4, help="processes; envs are split across them")
    p.add_argument("--name", default="bh_ppo")
    p.add_argument("--push", action="store_true", help="random shoves for robustness")
    p.add_argument("--terrain", action="store_true", help="stairs / slopes / rough ground course")
    p.add_argument("--resume", default=None, help="checkpoint .zip to continue (same run, same step count)")
    p.add_argument("--init", default=None, help="model .zip to start a new run from (e.g. flat -> terrain)")
    p.add_argument("--scan", action="store_true", help="add a terrain height scan to the policy's inputs")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    torch.set_num_threads(4)

    venv = make_vec_env(BerkeleyHumanoidEnv, n_envs=args.envs, vec_env_cls=BatchedSubprocVecEnv,
                        vec_env_kwargs=dict(n_workers=args.workers),
                        env_kwargs=dict(push=args.push, terrain=args.terrain, height_scan=args.scan))

    start_from = args.resume or args.init
    widen = args.init and PPO.load(args.init, device="cpu").observation_space.shape != venv.observation_space.shape
    if start_from and not widen:
        venv = VecNormalize.load(vecnormalize_path(start_from), venv)
        model = PPO.load(start_from, env=venv, device="cpu", tensorboard_log=run_dir)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.97)
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=3e-4,
            n_steps=512,           # x 24 envs = 12288 samples per update
            batch_size=3072,
            n_epochs=5,
            gamma=0.97,            # same discount as Playground's config
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                               activation_fn=nn.ELU, log_std_init=-0.7),
            tensorboard_log=run_dir,
            device="cpu",
            verbose=1,
        )
        if widen:
            widen_from_blind(model, venv, args.init)

    callbacks = [
        CheckpointCallback(save_freq=max(500_000 // args.envs, 1), save_path=ckpt_dir,
                           name_prefix="bh", save_vecnormalize=True),
        WalkStatsCallback(),
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
