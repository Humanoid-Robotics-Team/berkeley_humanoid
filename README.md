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
| `flip_model.py` | The robot set up for flipping: full-body collisions and the real motors' torque-speed limits |
| `flip_search.py` | Finds an open-loop backflip the motors can make, saved as `flip_reference.npz` |
| `flip_env.py`, `train_flip.py` | Backflip env that follows that motion, and its PPO training |
| `play_flip.py` | Watch, record or score a backflip policy |

`runs/bh_ppo` (flat ground), `runs/bh_terrain` (blind terrain policy), `runs/bh_scan2` (terrain policy
with a height scan, the best one on terrain) and `runs/bh_flip_best` (backflip) hold trained models.

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
python train.py --terrain --scan --init runs/bh_scan/checkpoints/bh_13999776_steps.zip --name bh_scan2 --lr 1e-4 --ent-coef 0.001 --target-kl 0.02 --std 0.4 --steps 15e6
python eval_terrain.py --model runs/bh_scan2/final_model.zip
python play.py --model runs/bh_scan2/final_model.zip --tile stairs_up --level 1
```

The last training step is a fine-tune. In the `bh_scan` run the exploration noise kept growing
(std 0.44 to 0.62) and the curriculum stopped improving after about 2M steps, so `bh_scan2` starts
from its 14M checkpoint with a lower learning rate, entropy bonus and noise. On terrain the speed
reward is now tighter (`TRACKING_SIGMA_TERRAIN`) and an env also drops a level if it covers less than
half the commanded distance, so stalling doesn't pay.

Training needs about 3.5 GB of free memory with the default 4 workers. On Windows, running out of
commit (RAM + pagefile) crashes it with `MemoryError`; use `--workers 2` or close other apps.

## Results

Runs out of 20 that walked: no fall in 8 s at a 0.5 m/s command from a standing start, covering
at least half the commanded distance (`eval_terrain.py --episodes 20`).

Height-scan policy, `runs/bh_scan2`:

| Terrain | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|---|
| Slope (3°, 6°, 10°, 13°, 17°) | 20 | 20 | 20 | 4 | 17 |
| Rough (±1 to ±5 cm) | 20 | 20 | 20 | 11 | 9 |
| Stairs down (2 to 10 cm) | 20 | 20 | 18 | 11 | 3 |
| Stairs up (2 to 10 cm) | 20 | 20 | 10 | 2 | 0 |

Blind policy, `runs/bh_terrain`, after 20M terrain steps:

| Terrain | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|---|
| Slope (3°, 6°, 10°, 13°, 17°) | 20 | 19 | 19 | 0 | 16 |
| Rough (±1 to ±5 cm) | 20 | 20 | 16 | 8 | 3 |
| Stairs down (2 to 10 cm) | 20 | 19 | 13 | 4 | 1 |
| Stairs up (2 to 10 cm) | 20 | 12 | 0 | 0 | 0 |

Slope tiles alternate between hill and valley. Odd levels are valleys, so level 3 means walking
uphill at 13°, where the robot stays upright but stalls (20/20 stayed up, 4 walked). Level 4 is a hill,
so its score only covers walking down 17°. Stairs up above 6 cm is still out of reach.

## Backflip

From a crouch on flat ground, the robot jumps, rotates backward, lands on its feet and stays up.

```bash
python play_flip.py                                   # watch it, half speed
python play_flip.py --record backflip.mp4 --tries 2
python play_flip.py --score 40 --perturb 1            # flips landed under small disturbances
```

To train it again:

```bash
python flip_search.py                # about 10 minutes; flip_reference.npz is already included
python train_flip.py --steps 15e6
```

`flip_search.py` looks for joint targets that make a whole flip, with no feedback. The result lands
and gets back up, then tips over slowly. PPO follows that motion, then has to balance by itself once
it ends. Episodes also start part way along the motion, so landing gets practised from the start.
Without this motion to follow, the policy settled for jumping and falling on its back.

More training didn't keep helping. Results under disturbance peaked between 5M and 9.5M steps and
then fell, and the model from the end of the 15M run failed even without disturbance. So scoring
every checkpoint picked `runs/bh_flip_best`, the 6.5M one, and the results below use different random
runs from the ones it was picked on.

| Disturbance | Landed on feet | Full flip, stayed up |
|---|---|---|
| None | yes | yes (349°, torso upright at the end) |
| ×0.5 | 40/40 | 35/40 |
| ×1 (joints up to 0.03 rad off at the start, 5% command noise) | 39/40 | 28/40 |
| ×2 | 32/40 | 11/40 |
| ×4 | 19/40 | 3/40 |

Without disturbance every run is the same, because the start and the policy are the same. That run
also recovers from a 0.5 m/s shove forward or backward after landing, and stays up for 10 s.

This is a simulation result, not ready for the real robot:

- **Joint speed:** the knee reaches 1.6x the real motor's speed limit before touchdown, and 14 of the
  401 control steps exceed a limit on some joint. Motor torque falls to zero at the limit, but the joint
  can still be pushed past it by the rest of the moving leg, and the env only penalises that.
- **Start:** it starts from a fixed crouch, placed there directly rather than reached from standing.
- **Sim only:** it has perfect sensors (no noise), mirrored legs, flat ground and one friction value.
- **Torque limits:** from the Berkeley team's Isaac Lab config. The torque-speed curve assumes Isaac
  Lab's DC motor formula.

## Differences from the Playground task

- Commands default to vx ∈ [0, 1], vy ∈ [-0.3, 0.3], yaw ∈ [-0.5, 0.5] m/s, rad/s (Playground: ±1 on all), and pushes are off by default. A narrower task learns in ~20M steps instead of ~150M.
- On terrain, the velocity tracking reward uses sigma 0.25 instead of 0.5.
- Episodes also end when the torso drops below 0.3 m. Only the feet collide with the floor in this model, so a fallen robot sinks through the ground.
