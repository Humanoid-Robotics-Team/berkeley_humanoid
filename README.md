# Berkeley Humanoid walking — MuJoCo + PPO on a laptop

Trains UC Berkeley's **Berkeley Humanoid** (12-DoF legged humanoid, MJCF from
DeepMind's MuJoCo Playground / Menagerie) to walk on a velocity command, then on slopes,
rough ground and stairs, using CPU MuJoCo and Stable-Baselines3 PPO. No GPU or WSL needed.

| File | What it is |
|---|---|
| `bh_env.py` | Gymnasium env: NumPy port of Playground's `BerkeleyHumanoidJoystickFlatTerrain` (same obs, rewards, gait clock, noise) |
| `terrain.py` | The heightfield course of rough ground, slopes and stairs |
| `batched_vec_env.py` | Parallel env runner that puts several robots in each worker process (Windows pipes are slow) |
| `train.py` | PPO training, checkpoints every 500k steps to `runs/<name>/checkpoints/` |
| `play.py` | Watch a policy in the MuJoCo viewer, or record an `.mp4` |
| `eval_terrain.py` | Score a policy on every terrain type and level |
| `make_article_images.py` | Render the write-up's images into `article_images/` |

`runs/bh_ppo` (flat ground) and `runs/bh_terrain` (blind terrain policy) hold trained models.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Fetch the robot model and meshes into this folder (or anywhere else, and point `BH_XML` at
`scene_mjx_feetonly_flat_terrain.xml`):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_playground.git
git -C mujoco_playground sparse-checkout set mujoco_playground/_src/locomotion/berkeley_humanoid
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git mujoco_playground/mujoco_menagerie
git -C mujoco_playground/mujoco_menagerie sparse-checkout set berkeley_humanoid
```

## Train

```bash
python train.py --steps 20e6
```

Options: `--envs 24 --workers 4`, `--push` (random shoves, for robustness), `--name myrun`,
`--resume runs/bh_ppo/checkpoints/bh_5000032_steps.zip`. With `--resume`, `--steps` counts the
additional steps.

Progress curves:

```bash
tensorboard --logdir runs
```

Watch `rollout/ep_len_mean` (1000 = never falls in a 20 s episode), `rollout/ep_rew_mean`,
and `walk/forward_vel` vs `walk/cmd_vx`.

## Watch it walk

```bash
python play.py --model runs/bh_ppo/final_model.zip
```

Viewer keys: **Up/Down** forward speed · **Left/Right** turn · **, / .** sideways · **Space** stop.

Record a video instead: `play.py --record walk.mp4 --seconds 15 --vx 0.6`.

## Stairs and uneven terrain

`terrain.py` builds a 26 × 31 m heightfield course: four tile types (`rough`, `slope`,
`stairs_up`, `stairs_down`) × five difficulty levels (stairs 2 → 10 cm, slopes 3° → 17°,
bumps ±1 → ±5 cm). Each env starts on level 0, moves up a level after walking off its tile
and down after an early fall (`walk/terrain_level` in TensorBoard is the average level).
Levels are tracked per terrain type, so failing on stairs doesn't hold back slopes
(`walk/level_stairs_up` etc. in TensorBoard). By default the policy is blind: it feels the
terrain through its joints and IMU. It stalls at 4 cm stairs going up.

`--scan` adds 55 terrain heights around the robot (20 cm behind to 80 cm ahead, ±20 cm sideways)
to its inputs. `--init` with a blind model starts from that model with the new inputs switched off.

```bash
python train.py --terrain --init runs/bh_ppo/final_model.zip --name bh_terrain
python train.py --terrain --scan --init runs/bh_terrain/final_model.zip --name bh_scan
python eval_terrain.py --model runs/bh_scan/final_model.zip
python play.py --model runs/bh_terrain/final_model.zip --tile stairs_up --level 1
```

## Results

Runs out of 20 that finished 8 s at 0.5 m/s without falling, from a standing start, for the
blind policy after 20M terrain steps:

| Terrain | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|---|
| Slope (3°, 6°, 10°, 13°, 17°) | 20 | 20 | 19 | 20 | 16 |
| Rough (±1 to ±5 cm) | 20 | 20 | 17 | 8 | 4 |
| Stairs down (2 to 10 cm) | 20 | 19 | 13 | 4 | 1 |
| Stairs up (2 to 10 cm) | 20 | 14 | 0 | 0 | 0 |

The height-scan run hasn't finished, so no scan model is included yet. An early checkpoint at 8.5M
steps scored 20/20 on 4 cm stairs up, 9/20 on 6 cm stairs up, 17/20 on 6 cm stairs down and
12/20 on 8 cm stairs down.

## Differences from the Playground task

- Commands default to vx ∈ [0, 1], vy ∈ [-0.3, 0.3], yaw ∈ [-0.5, 0.5] m/s, rad/s (Playground: ±1 on all), and pushes are off by default. A narrower task learns in ~20M steps instead of ~150M.
- Episodes also end when the torso drops below 0.3 m. Only the feet collide with the floor in this model, so a fallen robot sinks through the ground.
