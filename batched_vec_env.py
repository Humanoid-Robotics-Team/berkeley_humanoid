"""SubprocVecEnv variant where each worker process runs several envs.

On Windows, SB3's SubprocVecEnv pays a pipe round trip per env per step, which
costs more than simulating the robot. Grouping envs per worker cuts that down.
"""

import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(remote, parent_remote, env_fns):
    parent_remote.close()
    venv = DummyVecEnv(env_fns.var)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(venv.step(data))
            elif cmd == "reset":
                seed, options = data
                if seed is not None:
                    venv.seed(seed)
                venv.set_options(options)
                remote.send((venv.reset(), venv.reset_infos))
            elif cmd == "call":
                method, args, kwargs, idx = data
                remote.send(getattr(venv, method)(*args, indices=idx, **kwargs))
            elif cmd == "close":
                venv.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, n_workers=4):
        self.closed = False
        n_workers = min(n_workers, len(env_fns))
        self.chunks = np.array_split(np.arange(len(env_fns)), n_workers)
        ctx = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for chunk in self.chunks:
            remote, work_remote = ctx.Pipe()
            fns = CloudpickleWrapper([env_fns[i] for i in chunk])
            proc = ctx.Process(target=_worker, args=(work_remote, remote, fns), daemon=True)
            proc.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(proc)

        self.remotes[0].send(("call", ("get_attr", ("observation_space",), {}, [0])))
        obs_space = self.remotes[0].recv()[0]
        self.remotes[0].send(("call", ("get_attr", ("action_space",), {}, [0])))
        act_space = self.remotes[0].recv()[0]
        super().__init__(len(env_fns), obs_space, act_space)

    def step_async(self, actions):
        for remote, chunk in zip(self.remotes, self.chunks):
            remote.send(("step", actions[chunk]))

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return (np.concatenate(obs), np.concatenate(rews), np.concatenate(dones),
                [i for group in infos for i in group])

    def reset(self):
        for k, (remote, chunk) in enumerate(zip(self.remotes, self.chunks)):
            seed = self._seeds[chunk[0]] if self._seeds[chunk[0]] is not None else None
            remote.send(("reset", (seed, self._options[chunk[0]])))
        results = [r.recv() for r in self.remotes]
        self.reset_infos = [i for _, infos in results for i in infos]
        self._reset_seeds()
        self._reset_options()
        return np.concatenate([o for o, _ in results])

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.processes:
            proc.join()
        self.closed = True

    # --- attribute/method plumbing, routed to the worker that owns each env ---
    def _call(self, method, args, kwargs, indices):
        indices = self._get_indices(indices)
        out = {}
        for remote, chunk in zip(self.remotes, self.chunks):
            local = [int(i - chunk[0]) for i in indices if chunk[0] <= i <= chunk[-1]]
            if local:
                remote.send(("call", (method, args, kwargs, local)))
                for i, v in zip(local, remote.recv()):
                    out[chunk[0] + i] = v
        return [out[i] for i in indices]

    def get_attr(self, attr_name, indices=None):
        return self._call("get_attr", (attr_name,), {}, indices)

    def set_attr(self, attr_name, value, indices=None):
        self._call("set_attr", (attr_name, value), {}, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._call("env_method", (method_name, *method_args), method_kwargs, indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._call("env_is_wrapped", (wrapper_class,), {}, indices)
