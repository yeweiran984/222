"""
Skylark-TRPO adaptation for Game2048 (discrete actions).

This file reimplements a TRPO agent inspired by ref/Skylark_TRPO.py
but adapted to a categorical policy for the 4-direction actions used by
Game2048 (0: up, 1: right, 2: down, 3: left).

Key differences from the reference:
- Categorical policy (logits -> Categorical) instead of Gaussian.
- KL and loss are computed with discrete distributions.
- Value network is trained with torch LBFGS/Adam fallback.
- Gymnasium API for reset/step and auto-reset on done.

Usage (minimal):
  python skylark_trpo_2048.py --total-steps 50000 --steps-per-update 2048 --num-envs 1

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import game2048  # noqa: F401  # ensure env registration

try:
    from vectorized_env_mp import VectorizedEnv as MPVectorizedEnv
except Exception:
    MPVectorizedEnv = None  # type: ignore


# ----------------------- Networks -----------------------
class PolicyNet(nn.Module):
    """Simple MLP policy on 4x4 board producing logits for 4 actions."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(16, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [...,4,4] or [...,16]
        if x.dim() == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] != 16:
            x = x.reshape(x.shape[0], -1)
        # log2 normalize to keep magnitudes stable
        x = torch.log2(x.float() + 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ValueNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(16, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] != 16:
            x = x.reshape(x.shape[0], -1)
        x = torch.log2(x.float() + 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.v(x).squeeze(-1)


# ----------------------- Utilities -----------------------

def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx: idx + n].view_as(p))
        idx += n


def flat_grad(grads: List[torch.Tensor], params: List[torch.Tensor]) -> torch.Tensor:
    pieces = []
    for g, p in zip(grads, params):
        if g is None:
            pieces.append(torch.zeros_like(p).view(-1))
        else:
            pieces.append(g.contiguous().view(-1))
    return torch.cat(pieces)


@dataclass
class TrajBatch:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]
    last_value: float


# ----------------------- TRPO Agent -----------------------
class SkylarkTRPO2048:
    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lam: float = 0.95,
        max_kl: float = 1e-2,
        damping: float = 1e-2,
        entropy_coef: float = 0.0,
        value_lr: float = 3e-4,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        num_envs: int = 1,
        vec_impl: str = "auto",
        env_target: int = 2048,
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.damping = damping
        self.entropy_coef = entropy_coef
        self.value_lr = value_lr
        self.num_envs = max(1, int(num_envs))
        self.vec_impl = vec_impl
        self.env_target = int(env_target)

        self.policy = PolicyNet(hidden_dim).to(self.device)
        self.value = ValueNet(hidden_dim).to(self.device)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        # 起始迭代（用于进度条从特定迭代数开始计数，保持与其他脚本一致）
        self.start_iter = 0

    # ---------- Save/Load ----------
    def save_model(self, path: str) -> None:
        """保存模型（含策略与价值网络）。

        采用与 trpo_game2048.py 相同的键名，便于一致性：
        - policy_state_dict
        - value_state_dict
        """
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value.state_dict(),
            }, path)
            print(f"模型已保存至: {path}")
        except Exception as e:
            print(f"保存模型失败: {e}")

    def load_model(self, path: str) -> None:
        """加载模型，兼容可能的旧键名，并尝试从文件名解析迭代次数。"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"加载模型失败: {e}")
            return

        # 兼容两种键名
        pol_key = 'policy_state_dict' if 'policy_state_dict' in checkpoint else 'policy'
        val_key = 'value_state_dict' if 'value_state_dict' in checkpoint else 'value'
        if pol_key not in checkpoint or val_key not in checkpoint:
            print("模型文件缺少必要键: 'policy_state_dict'/'value_state_dict' (或兼容键 'policy'/'value')")
            return

        try:
            self.policy.load_state_dict(checkpoint[pol_key])
            self.value.load_state_dict(checkpoint[val_key])
        except Exception as e:
            print(f"加载权重出错: {e}")
            return

        # 从文件名解析起始迭代 `_XXXit_YYbatch.pth`
        try:
            fname = os.path.basename(str(path))
            m = re.search(r"_(\d+)it(?:_|\.pth$)", fname)
            if m:
                self.start_iter = int(m.group(1))
                print(f"模型已从 {path} 加载 (起始迭代={self.start_iter})")
                return
        except Exception:
            pass
        print(f"模型已从 {path} 加载")

    def _format_save_path(self, save_path: str | None, iters: int, batch_size: int) -> str:
        """根据给定路径/文件名，拼接携带迭代与 batch 信息的保存路径。"""
        base_default = "skylark_trpo_2048"
        ext = ".pth"
        if not save_path:
            directory = "."
            base = base_default
        else:
            directory, name = os.path.split(save_path)
            if not directory:
                directory = "."
            root, ext_in = os.path.splitext(name)
            if ext_in:
                base = root
            else:
                base = name if name else base_default
        filename = f"{base}_{iters}it_{batch_size}batch{ext}"
        return os.path.join(directory, filename)

    # ---------- Sampling ----------
    def _select_actions(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        st = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        logits = self.policy(st)
        dist = Categorical(logits=logits)
        acts = dist.sample()
        logp = dist.log_prob(acts)
        return acts.detach().cpu().numpy(), logp.detach().cpu().numpy()

    def _values(self, states: np.ndarray) -> np.ndarray:
        st = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.value(st)
        return v.detach().cpu().numpy()

    def collect_batch(self, steps: int) -> Tuple[TrajBatch, List[float]]:
        """Collect a batch of transitions.

        Note: when num_envs > 1, "steps" means TOTAL transitions across all envs,
        matching the meaning in other scripts. We therefore iterate until we
        accumulate at least `steps` samples (not steps per-env).
        """
        if self.num_envs > 1:
            vec = self._make_vec_env(self.num_envs, self.env_target)
            try:
                current_states = vec.reset()
                ep_rewards = [0.0] * self.num_envs
                env_completed = [0] * self.num_envs

                all_states: List[np.ndarray] = []
                all_actions: List[int] = []
                all_rewards: List[float] = []
                all_dones: List[bool] = []
                all_logps: List[float] = []
                all_values: List[float] = []

                total = 0
                while True:
                    actions, logps = self._select_actions(current_states)
                    vals = self._values(current_states)
                    next_states, rewards, dones = vec.step(actions)

                    for i in range(self.num_envs):
                        all_states.append(current_states[i])
                        all_actions.append(int(actions[i]))
                        all_rewards.append(float(rewards[i]))
                        all_dones.append(bool(dones[i]))
                        all_logps.append(float(logps[i]))
                        all_values.append(float(vals[i]))
                        ep_rewards[i] += float(rewards[i])
                        if dones[i]:
                            env_completed[i] += 1
                        total += 1

                    current_states = next_states

                    # stop when total >= steps and every env has at least one episode
                    if total >= steps and all(c >= 1 for c in env_completed):
                        break

                # compute last value per env for GAE tail
                last_values: List[float] = []
                for i in range(self.num_envs):
                    if dones[i]:
                        last_values.append(0.0)
                    else:
                        v = self._values(current_states[i][None, ...])
                        last_values.append(float(np.asarray(v).reshape(-1)[0]))

                batch = TrajBatch(
                    states=all_states,
                    actions=all_actions,
                    rewards=all_rewards,
                    dones=all_dones,
                    log_probs=all_logps,
                    values=all_values,
                    last_value=0.0,  # unused by vectorized advantage
                )
                # Attach as attribute to reuse in vectorized advantage
                batch._last_values_per_env = last_values  # type: ignore
                return batch, ep_rewards
            finally:
                vec.close()
        else:
            state, _ = self.env.reset()
            ep_rewards = []
            all_states: List[np.ndarray] = []
            all_actions: List[int] = []
            all_rewards: List[float] = []
            all_dones: List[bool] = []
            all_logps: List[float] = []
            all_values: List[float] = []

            for _ in range(steps):
                action, logp = self._select_actions(np.asarray(state)[None, ...])
                value = self._values(np.asarray(state)[None, ...])
                next_state, reward, terminated, truncated, _ = self.env.step(int(action[0]))
                done = bool(terminated or truncated)

                all_states.append(state)
                all_actions.append(int(action[0]))
                all_rewards.append(float(reward))
                all_dones.append(done)
                all_logps.append(float(logp[0]))
                all_values.append(float(value[0]))

                state = next_state
                if done:
                    ep_rewards.append(sum(_r for _r, _d in zip(all_rewards[::-1], all_dones[::-1]) if _d) if all_dones[-1] else float(reward))
                    state, _ = self.env.reset()

            if len(all_dones) > 0 and all_dones[-1]:
                last_v = 0.0
            else:
                last_v = float(self._values(np.asarray(state)[None, ...])[0])

            batch = TrajBatch(
                states=all_states,
                actions=all_actions,
                rewards=all_rewards,
                dones=all_dones,
                log_probs=all_logps,
                values=all_values,
                last_value=last_v,
            )
            return batch, ep_rewards

    def _make_vec_env(self, n: int, target: int):
        impl = self.vec_impl
        if impl == "auto":
            impl = "mp"
        if impl == "mp" and MPVectorizedEnv is not None:
            return MPVectorizedEnv(num_envs=n, target=target)
        # simple in-proc fallback
        class InProc:
            def __init__(self, num_envs: int, target: int) -> None:
                self.num_envs = num_envs
                self.envs = [gym.make("Game2048-v0", target=target) for _ in range(num_envs)]
            def reset(self):
                return np.asarray([e.reset()[0] for e in self.envs])
            def step(self, actions):
                S, R, D = [], [], []
                for e, a in zip(self.envs, actions):
                    s, r, term, trunc, _ = e.step(int(a))
                    d = bool(term or trunc)
                    if d:
                        s, _ = e.reset()
                    S.append(s); R.append(r); D.append(d)
                return np.asarray(S), np.asarray(R), np.asarray(D)
            def close(self):
                for e in self.envs:
                    try: e.close()
                    except Exception: pass
        return InProc(n, target)

    # ---------- GAE ----------
    def compute_advantages(self, batch: TrajBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        values = np.asarray(batch.values, dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.float32)
        adv = np.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = batch.last_value
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            lastgaelam = delta + self.gamma * self.lam * (1.0 - dones[t]) * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return adv_t, returns_t

    def compute_advantages_vectorized(self, batch: TrajBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages for interleaved multi-env samples.

        This version works even when the total number of samples isn't a
        multiple of num_envs. It reconstructs each env's trajectory via slicing
        with stride and fills the output back in original interleaved order.
        """
        N = self.num_envs
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        values = np.asarray(batch.values, dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.float32)

        adv_out = np.zeros_like(rewards, dtype=np.float32)
        ret_out = np.zeros_like(rewards, dtype=np.float32)

        # Optional per-env last values computed during collection (if available)
        last_values = getattr(batch, '_last_values_per_env', None)

        for i in range(N):
            env_rewards = rewards[i::N]
            env_values = values[i::N]
            env_dones = dones[i::N]

            adv = np.zeros_like(env_rewards, dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(len(env_rewards))):
                if t == len(env_rewards) - 1:
                    next_value = 0.0 if last_values is None else float(last_values[i])
                else:
                    next_value = env_values[t + 1]
                delta = env_rewards[t] + self.gamma * next_value * (1.0 - env_dones[t]) - env_values[t]
                lastgaelam = delta + self.gamma * self.lam * (1.0 - env_dones[t]) * lastgaelam
                adv[t] = lastgaelam

            ret = adv + env_values

            # place back to interleaved positions
            adv_out[i::N][:len(adv)] = adv
            ret_out[i::N][:len(ret)] = ret

        adv_t = torch.as_tensor(adv_out, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret_out, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return adv_t, ret_t

    # ---------- TRPO core ----------
    def policy_loss(self, states_t: torch.Tensor, actions_t: torch.Tensor, adv_t: torch.Tensor, old_logp_t: torch.Tensor) -> torch.Tensor:
        logits = self.policy(states_t)
        dist = Categorical(logits=logits)
        new_logp = dist.log_prob(actions_t)
        ratio = torch.exp(new_logp - old_logp_t)
        ent = dist.entropy().mean()
        return -(ratio * adv_t).mean() - self.entropy_coef * ent

    def fisher_vector_product(self, states_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        logits = self.policy(states_t)
        with torch.no_grad():
            old_probs = F.softmax(logits, dim=-1)
        # KL(old || new) = sum old * (log old - log new)
        new_log_probs = F.log_softmax(self.policy(states_t), dim=-1)
        kl = (old_probs * (torch.log(old_probs + 1e-10) - new_log_probs)).sum(-1).mean()
        grads = torch.autograd.grad(kl, list(self.policy.parameters()), create_graph=True)
        flat_g = flat_grad(list(grads), list(self.policy.parameters()))
        kl_v = torch.sum(flat_g * v)
        hvp = torch.autograd.grad(kl_v, list(self.policy.parameters()))
        flat_hvp = flat_grad(list(hvp), list(self.policy.parameters()))
        return flat_hvp + self.damping * v

    def conjugate_gradients(self, Avp, b: torch.Tensor, nsteps: int = 10, tol: float = 1e-10) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            Avp_p = Avp(p)
            alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)
            x += alpha * p
            r -= alpha * Avp_p
            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def line_search(self, states_t: torch.Tensor, actions_t: torch.Tensor, adv_t: torch.Tensor,
                    old_logp_t: torch.Tensor, fullstep: torch.Tensor, expected_improve_rate: torch.Tensor,
                    max_backtracks: int = 10, accept_ratio: float = 0.1) -> Tuple[bool, torch.Tensor]:
        f = lambda: self.policy_loss(states_t, actions_t, adv_t, old_logp_t)
        fval = f().item()
        old_params = flat_params(self.policy)
        for stepfrac in 0.5 ** torch.arange(0, max_backtracks + 1, dtype=torch.float32):
            new_params = old_params + stepfrac.item() * fullstep
            set_flat_params(self.policy, new_params)
            newfval = f().item()
            actual_improve = fval - newfval
            expected_improve = (expected_improve_rate * stepfrac).item()
            if expected_improve <= 0:
                continue
            ratio = actual_improve / (expected_improve + 1e-8)
            if ratio > accept_ratio and actual_improve > 0:
                return True, new_params
        set_flat_params(self.policy, old_params)
        return False, old_params

    def update_policy(self, states: List[np.ndarray], actions: List[int], adv: torch.Tensor, old_logp: List[float]) -> bool:
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        old_logp_t = torch.as_tensor(np.asarray(old_logp), dtype=torch.float32, device=self.device)

        loss = self.policy_loss(states_t, actions_t, adv, old_logp_t)
        grads = torch.autograd.grad(loss, list(self.policy.parameters()))
        loss_grad = flat_grad(list(grads), list(self.policy.parameters()))

        def Avp_func(v: torch.Tensor) -> torch.Tensor:
            return self.fisher_vector_product(states_t, v)

        step_dir = self.conjugate_gradients(Avp_func, -loss_grad, nsteps=10)
        shs = 0.5 * torch.dot(step_dir, Avp_func(step_dir))
        if shs <= 0:
            return False
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = step_dir / (lm + 1e-8)
        expected_improve = -torch.dot(loss_grad, fullstep)
        success, new_params = self.line_search(states_t, actions_t, adv, old_logp_t, fullstep, expected_improve)
        set_flat_params(self.policy, new_params)
        return success

    def update_value(self, states: List[np.ndarray], returns: torch.Tensor) -> float:
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        returns_t = returns.detach()
        # Try LBFGS for a few steps; fallback to Adam epochs
        try:
            optim = torch.optim.LBFGS(self.value.parameters(), lr=1.0, max_iter=25)
            def closure():
                optim.zero_grad()
                v = self.value(states_t)
                loss = F.mse_loss(v, returns_t)
                loss.backward()
                return loss
            loss_val = optim.step(closure)
            return float(loss_val)
        except Exception:
            loss_val = 0.0
            for _ in range(10):
                v = self.value(states_t)
                loss = F.mse_loss(v, returns_t)
                self.value_optim.zero_grad()
                loss.backward()
                self.value_optim.step()
                loss_val = float(loss.item())
            return loss_val

    # ---------- Train/Eval ----------
    def train(self, total_steps: int = 100_000, steps_per_update: int = 2048, num_envs: int | None = None,
              vec_impl: str | None = None, target: int | None = None, save_path: str | None = None) -> None:
        import time
        from collections import deque
        import tqdm
        from itertools import count
        if num_envs is not None:
            self.num_envs = max(1, int(num_envs))
        if vec_impl is not None:
            self.vec_impl = vec_impl
        if target is not None:
            self.env_target = int(target)

        episode_rewards = deque(maxlen=100)
        step_accum = 0
        # 使用与 trpo_game2048.py 相同的进度条策略：按“更新次数”驱动迭代
        if total_steps == -1:
            pbar = tqdm.tqdm(count(self.start_iter), desc="训练进度")
            total_updates = None
        else:
            total_updates = max(1, total_steps // steps_per_update)
            pbar = tqdm.tqdm(range(self.start_iter, total_updates), desc="训练进度")
        update_count = 0
        try:
            for _ in pbar:
                if total_steps != -1 and step_accum >= total_steps:
                    break
                t0 = time.time()
                batch, ep_rews = self.collect_batch(steps_per_update)
                step_accum += len(batch.actions)
                episode_rewards.extend(ep_rews)

                # advantages
                if self.num_envs > 1:
                    adv, rets = self.compute_advantages_vectorized(batch)
                else:
                    adv, rets = self.compute_advantages(batch)

                # policy update
                success = self.update_policy(batch.states, batch.actions, adv, batch.log_probs)

                # value update
                vloss = self.update_value(batch.states, rets)

                # 统计信息，参考 trpo_game2048.py 的显示
                dt = time.time() - t0
                rewards_np = np.asarray(batch.rewards, dtype=float) if batch.rewards else np.array([0.0])
                avg_reward = float(np.mean(rewards_np))

                # 最大块与平均最大块（按回合末状态统计）
                states_arr = np.asarray(batch.states)
                max_tile = int(np.max(states_arr)) if states_arr.size else 0
                dones_arr = np.asarray(batch.dones, dtype=bool)
                if dones_arr.any():
                    end_states = states_arr[dones_arr]
                    avg_max_tile = float(np.mean(np.max(end_states, axis=(1, 2))))
                else:
                    # 若本批次没有 episode 结束，用当前最后状态作为近似
                    last_state = states_arr[-1] if states_arr.size else np.zeros((4, 4))
                    avg_max_tile = float(np.max(last_state))

                # 无效动作比例（奖励 == -10.0）
                invalid_ratio = float((rewards_np == -10.0).sum() / max(1, len(rewards_np)) * 100.0)

                pbar.set_postfix({
                    '平均奖励': f'{avg_reward:.1f}',
                    '最大块': max_tile,
                    '平均最大块': f'{avg_max_tile:.1f}',
                    '无效动作': f'{invalid_ratio:.1f}%',
                    '价值损失': f'{vloss:.1f}',
                    '策略': '✓' if success else '✗',
                    'FPS': f'{len(batch.actions)/(dt+1e-6):.1f}'
                })
                # 使用 for pbar 迭代，tqdm 自动更新迭代进度，这里不手动 update
                update_count += 1
        finally:
            if save_path:
                try:
                    curr_iter = self.start_iter + update_count
                    final_path = self._format_save_path(save_path, curr_iter, steps_per_update)
                    self.save_model(final_path)
                except Exception as e:
                    print(f"保存失败: {e}")
            pbar.close()

    def evaluate(self, episodes: int = 10) -> Tuple[List[float], List[int]]:
        rewards, max_tiles = [], []
        for _ in range(episodes):
            s, _ = self.env.reset()
            done = False
            ep_r = 0.0
            while not done:
                with torch.no_grad():
                    logits = self.policy(torch.as_tensor(s[None, ...], dtype=torch.float32, device=self.device))
                    a = torch.argmax(logits, dim=-1).item()
                s, r, t, tr, _ = self.env.step(int(a))
                done = bool(t or tr)
                ep_r += float(r)
            rewards.append(ep_r)
            max_tiles.append(int(np.max(s)))
        print(f"评估: 平均奖励={np.mean(rewards):.1f}, 平均最大块={np.mean(max_tiles):.1f}")
        return rewards, max_tiles


# ----------------------- CLI -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skylark TRPO for Game2048")
    p.add_argument('--total-steps', type=int, default=100_000)
    p.add_argument('--steps-per-update', type=int, default=2048)
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--vec-impl', choices=['auto', 'inproc', 'mp'], default='auto')
    p.add_argument('--target', type=int, default=2048)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--max-kl', type=float, default=1e-2)
    p.add_argument('--damping', type=float, default=1e-2)
    p.add_argument('--entropy-coef', type=float, default=0.0)
    p.add_argument('--value-lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--save-path', type=str, default='skylark_trpo_2048.pth')
    p.add_argument('--load-path', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    env = gym.make("Game2048-v0", target=args.target)

    agent = SkylarkTRPO2048(
        env=env,
        hidden_dim=args.hidden,
        gamma=args.gamma,
        lam=args.lam,
        max_kl=args.max_kl,
        damping=args.damping,
        entropy_coef=args.entropy_coef,
        value_lr=args.value_lr,
        device=device,
        num_envs=args.num_envs,
        vec_impl=args.vec_impl,
        env_target=args.target,
    )

    # 加载预训练模型（若提供）
    if args.load_path:
        agent.load_model(args.load_path)

    agent.train(total_steps=args.total_steps, steps_per_update=args.steps_per_update,
                num_envs=args.num_envs, vec_impl=args.vec_impl, target=args.target,
                save_path=args.save_path)

    env.close()


if __name__ == '__main__':
    main()
