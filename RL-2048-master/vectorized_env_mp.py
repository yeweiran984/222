"""
Multiprocessing vectorized environment for Game2048.

This implements the same public interface as the in-process VectorizedEnv:
- __init__(num_envs:int=4, target:int=2048)
- reset() -> np.ndarray[num_envs, 4, 4]
- step(actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
- close() -> None

Each worker process owns a single gymnasium "Game2048-v0" environment and
auto-resets on done, mirroring the behavior of the original VectorizedEnv.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Tuple

import numpy as np
import gymnasium as gym
import game2048


def _worker(conn: Connection, target: int) -> None:
    """Environment worker process.

    Receives commands via a Pipe from the parent process.
    Supported commands:
      - ("reset", None) -> sends back obs (np.ndarray)
      - ("step", action:int) -> steps env; auto-resets on done; sends back
        (obs, reward:float, done:bool)
      - ("close", None) -> closes env and exits
    """
    env = None
    try:
        env = gym.make("Game2048-v0", target=target)
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, _ = env.reset()
                conn.send(obs)
            elif cmd == "step":
                action = int(data)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                if done:
                    obs, _ = env.reset()
                conn.send((obs, float(reward), done))
            elif cmd == "close":
                try:
                    env.close()
                finally:
                    conn.close()
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    except (EOFError, KeyboardInterrupt):
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class VectorizedEnvMP:
    """Multiprocessing vectorized env with the same interface as VectorizedEnv.

    Example:
        from vectorized_env_mp import VectorizedEnvMP as VectorizedEnv
        vec_env = VectorizedEnv(num_envs=8, target=2048)
        states = vec_env.reset()
        actions = np.random.randint(0, 4, size=8)
        next_states, rewards, dones = vec_env.step(actions)
        vec_env.close()
    """

    def __init__(self, num_envs: int = 4, target: int = 2048) -> None:
        # Best-effort spawn to be CUDA-safe; ignore if already set
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            # start method already set by parent process
            pass

        self.num_envs = int(max(1, num_envs))
        self._target = int(target)
        self._conns = []
        self._procs = []

        for _ in range(self.num_envs):
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(target=_worker, args=(child_conn, self._target), daemon=True)
            proc.start()
            child_conn.close()  # close child end in parent
            self._conns.append(parent_conn)
            self._procs.append(proc)

    def reset(self) -> np.ndarray:
        """Reset all envs and return stacked observations [num_envs, 4, 4]."""
        for conn in self._conns:
            conn.send(("reset", None))
        states = [conn.recv() for conn in self._conns]
        return np.asarray(states)

    def step(self, actions: np.ndarray | List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all envs once with the provided actions.

        Args:
            actions: Iterable of ints of length num_envs.
        Returns:
            Tuple of (states, rewards, dones) arrays, each of length num_envs.
            If an env reaches done, it is immediately reset and the returned
            state is the reset observation, matching the in-process VectorizedEnv.
        """
        if isinstance(actions, np.ndarray):
            if actions.ndim != 1 or actions.shape[0] != self.num_envs:
                actions = actions.reshape(self.num_envs)
            action_list = actions.tolist()
        else:
            action_list = list(actions)
            assert len(action_list) == self.num_envs, "actions length must equal num_envs"

        for conn, act in zip(self._conns, action_list):
            conn.send(("step", int(act)))

        states: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        for conn in self._conns:
            obs, rew, done = conn.recv()
            states.append(obs)
            rewards.append(rew)
            dones.append(done)

        return np.asarray(states), np.asarray(rewards, dtype=float), np.asarray(dones, dtype=bool)

    def reset_one(self, idx: int) -> np.ndarray:
        """Reset a single env and return its observation.

        This is useful for forcing an episode boundary for a specific worker
        without disturbing others.
        """
        i = int(idx)
        if i < 0 or i >= self.num_envs:
            raise IndexError("env index out of range")
        conn = self._conns[i]
        conn.send(("reset", None))
        obs = conn.recv()
        return np.asarray(obs)

    def close(self) -> None:
        """Close all envs and join worker processes."""
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._conns.clear()
        self._procs.clear()


# Optional drop-in alias to match the original class name if you prefer:
# from vectorized_env_mp import VectorizedEnv  # equals VectorizedEnvMP
VectorizedEnv = VectorizedEnvMP
__all__ = ["VectorizedEnvMP", "VectorizedEnv"]
