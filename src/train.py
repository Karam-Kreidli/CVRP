"""
Stage 5 - Training Loop: PPO-based optimization of the Fleet Manager.

Trains the Fleet Manager to select optimal solver strategies using PPO:
  - Reward: change in competition score (1000*NV + TD), normalized online
  - Running Mean/Std reward normalization (Welford's online algorithm)
  - FP16 mixed precision via autocast + GradScaler for Colab T4
  - GAE-λ for advantage estimation with optional early KL stopping
  - Fixed evaluation set for tracking real learning progress
"""

from __future__ import annotations

import csv
import math
import pathlib
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.agent_manager import FleetManager
from src.solver_engine import CVRPEnv


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    vf_coeff: float = 0.5
    ent_coeff: float = 0.05
    manager_lr: float = 1e-4
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5
    use_fp16: bool = True
    target_kl: Optional[float] = 0.015


# ---------------------------------------------------------------------------
# Running Mean/Std for Reward Normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for tracking mean and variance.

    Prevents the Manager's ±1000 vehicle rewards from causing gradient
    explosions by normalizing to unit variance.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of values."""
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics."""
        return (x - self.mean) / (math.sqrt(self.var) + 1e-8)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions for the Fleet Manager (fixed 132-dim observations)."""

    def __init__(self):
        self.clear()

    def store(self, obs, action, log_prob, value, reward, done, action_mask=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.advantages = []
        self.returns = []

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """Compute GAE-λ advantages and discounted returns."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            gae = delta + gamma * lam * non_terminal * gae
            advantages[t] = gae
        values_arr = np.array(self.values, dtype=np.float32)
        self.advantages = advantages.tolist()
        self.returns = (advantages + values_arr).tolist()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MARLTrainer:
    """PPO trainer for the Fleet Manager.

    Collects rollouts from CVRPEnv, normalizes rewards, and runs PPO updates.
    Evaluates on a fixed instance set for tracking real learning progress.

    Args:
        env: CVRPEnv instance.
        config: PPO hyperparameters.
        device: Torch device.
        log_dir: Directory for CSV logs and best model.
        gdrive_path: Optional Google Drive backup directory.
        total_epochs: Total training epochs (for LR scheduling).
        eval_instances: List of .vrp file paths for fixed evaluation.
    """

    def __init__(
        self,
        env: CVRPEnv,
        config: PPOConfig = PPOConfig(),
        device: torch.device = torch.device("cpu"),
        log_dir: str = "logs",
        gdrive_path: Optional[str] = None,
        total_epochs: int = 100,
        eval_instances: list[str] | None = None,
    ):
        self.env = env
        self.config = config
        self.device = device
        self.log_dir = pathlib.Path(log_dir)
        self.gdrive_path = gdrive_path

        # Model
        self.manager = FleetManager().to(device)

        # Optimizer with linear LR decay to 50%
        self.optimizer = torch.optim.Adam(
            self.manager.parameters(), lr=config.manager_lr
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0, end_factor=0.5,
            total_iters=total_epochs,
        )

        # FP16 mixed precision
        self.amp_enabled = config.use_fp16 and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Running reward normalization
        self.reward_rms = RunningMeanStd()

        # Logging
        self.epoch_stats: list[dict] = []
        self.best_score = float("inf")

        # Fixed evaluation instances for tracking real progress
        self.eval_instances = eval_instances or []

        # Initialize CSV log
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_metrics.csv"
        self._csv_header_written = False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run greedy evaluation on fixed instances (no exploration, no training).

        Returns dict with eval_score, eval_nv, eval_td.
        """
        if not self.eval_instances:
            return {}

        self.manager.eval()

        scores, nvs, tds = [], [], []
        orig_paths = self.env.instance_paths
        for inst_path in self.eval_instances:
            self.env.instance_paths = [pathlib.Path(inst_path)]
            obs, info = self.env.reset()
            done = False
            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                graph_emb = obs_t[:, :128]
                solver_stats = obs_t[:, 128:]
                action_mask = info.get("action_mask")
                mask_t = None
                if action_mask is not None:
                    mask_t = torch.tensor(
                        action_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = self.manager(
                        graph_emb, solver_stats, action_mask=mask_t
                    )
                    action_int = logits.argmax(dim=-1).item()  # greedy
                obs, _, terminated, truncated, info = self.env.step(action_int)
                done = terminated or truncated
            scores.append(info["score"])
            nvs.append(info["nv"])
            tds.append(info["td"])

        self.env.instance_paths = orig_paths
        return {
            "eval_score": float(np.mean(scores)),
            "eval_nv": float(np.mean(nvs)),
            "eval_td": float(np.mean(tds)),
        }

    # ------------------------------------------------------------------
    # Experience Collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, num_episodes: int = 1) -> dict:
        """Run episodes and collect transitions.

        Reward = prev_score - new_score (positive when score improves).
        Normalized online via Welford's algorithm.

        Returns:
            Dictionary with episode statistics (averaged over episodes).
        """
        self.buffer.clear()
        self.manager.eval()

        total_reward = 0.0
        total_steps = 0
        episode_scores = []
        episode_nvs = []
        episode_tds = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            action_mask = info.get("action_mask")
            done = False

            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                graph_emb = obs_t[:, :128]
                solver_stats = obs_t[:, 128:]

                mask_t = None
                if action_mask is not None:
                    mask_t = torch.tensor(
                        action_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, value = self.manager.select_action(
                        graph_emb, solver_stats, action_mask=mask_t
                    )
                action_int = action.item()

                next_obs, reward, terminated, truncated, info = self.env.step(action_int)
                done = terminated or truncated

                self.buffer.store(
                    obs=obs.copy(),
                    action=action_int,
                    log_prob=log_prob.item(),
                    value=value.squeeze().item(),
                    reward=reward,
                    done=done,
                    action_mask=action_mask.copy() if action_mask is not None else None,
                )

                total_reward += reward
                total_steps += 1
                obs = next_obs
                action_mask = info.get("action_mask")

            episode_scores.append(info.get("score", 0.0))
            episode_nvs.append(info.get("nv", 0))
            episode_tds.append(info.get("td", 0.0))

        # Normalize rewards with running statistics
        norm_mag = 0.0
        if len(self.buffer) > 0:
            raw = np.array(self.buffer.rewards)
            self.reward_rms.update(raw)
            normed = self.reward_rms.normalize(raw)
            self.buffer.rewards = normed.tolist()
            norm_mag = float(np.mean(np.abs(normed)))

        # Compute GAE-λ advantages
        if len(self.buffer) > 0:
            self.buffer.compute_gae(0.0, self.config.gamma, self.config.lam)

        return {
            "total_steps": total_steps,
            "total_reward": total_reward,
            "norm_mag": norm_mag,
            "avg_nv": float(np.mean(episode_nvs)),
            "avg_td": float(np.mean(episode_tds)),
            "avg_score": float(np.mean(episode_scores)),
            "best_score": float(np.min(episode_scores)),
            "num_episodes": num_episodes,
        }

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def ppo_update(self) -> dict:
        """Run PPO clipped objective update epochs on the Fleet Manager."""
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.manager.train()
        cfg = self.config

        obs = torch.tensor(
            np.array(self.buffer.observations),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_lp = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        advantages = torch.tensor(
            self.buffer.advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(
            self.buffer.returns, dtype=torch.float32, device=self.device
        )

        # Build action masks tensor (True = allowed)
        masks_list = self.buffer.action_masks
        if masks_list and masks_list[0] is not None:
            action_masks = torch.tensor(
                np.array(masks_list), dtype=torch.bool, device=self.device
            )
        else:
            action_masks = None

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(obs)
        mb_size = min(cfg.mini_batch_size, T)
        total_pl, total_vl, total_ent, n = 0.0, 0.0, 0.0, 0

        for _ in range(cfg.ppo_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, mb_size):
                idx = indices[start : start + mb_size]
                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_old_lp = old_lp[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_mask = action_masks[idx] if action_masks is not None else None

                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    logits, values = self.manager(
                        mb_obs[:, :128], mb_obs[:, 128:],
                        action_mask=mb_mask,
                    )
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.epsilon_clip, 1.0 + cfg.epsilon_clip
                    ) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values.squeeze(-1), mb_ret)
                    loss = (
                        policy_loss
                        + cfg.vf_coeff * value_loss
                        - cfg.ent_coeff * entropy
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.manager.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n += 1

            # Early stopping on KL divergence
            if cfg.target_kl is not None:
                with torch.no_grad():
                    logits_all, _ = self.manager(
                        obs[:, :128], obs[:, 128:],
                        action_mask=action_masks,
                    )
                    kl = (
                        old_lp
                        - Categorical(logits=logits_all).log_prob(actions)
                    ).mean().item()
                    if kl > 1.5 * cfg.target_kl:
                        break

        n = max(n, 1)
        return {
            "policy_loss": total_pl / n,
            "value_loss": total_vl / n,
            "entropy": total_ent / n,
        }

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    def train_epoch(self, num_episodes: int = 1, run_eval: bool = True) -> dict:
        """One full training iteration: collect rollouts + PPO update + eval."""
        rollout_stats = self.collect_rollouts(num_episodes)
        ppo_stats = self.ppo_update()

        self.scheduler.step()

        eval_stats = self.evaluate() if run_eval else {}

        stats = {
            **rollout_stats,
            **eval_stats,
            "policy_loss": ppo_stats["policy_loss"],
            "value_loss": ppo_stats["value_loss"],
            "entropy": ppo_stats["entropy"],
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.epoch_stats.append(stats)
        self._write_csv(stats)

        # Best-model tracking: use eval score if available, else training best
        tracking_score = stats.get("eval_score", stats.get("best_score", float("inf")))
        if tracking_score < self.best_score:
            self.best_score = tracking_score
            best_path = self.log_dir / "best_model.pth"
            self.save_checkpoint(str(best_path))
            if self.gdrive_path:
                self._push_to_gdrive(best_path)

        return stats

    def _write_csv(self, stats: dict):
        """Append one row of metrics to the CSV log."""
        fieldnames = [
            "epoch", "avg_nv", "avg_td", "avg_score", "best_score",
            "eval_score", "eval_nv", "eval_td",
            "num_episodes", "total_reward", "norm_mag", "total_steps",
            "policy_loss", "value_loss", "entropy", "lr",
        ]
        row = {k: stats.get(k, "") for k in fieldnames}
        row["epoch"] = len(self.epoch_stats)

        write_header = not self._csv_header_written
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def _push_to_gdrive(self, src: pathlib.Path):
        """Copy a file to Google Drive."""
        dst_dir = pathlib.Path(self.gdrive_path)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save model, optimizer, and normalization state."""
        torch.save(
            {
                "manager_state_dict": self.manager.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "reward_rms": {
                    "mean": self.reward_rms.mean,
                    "var": self.reward_rms.var,
                    "count": self.reward_rms.count,
                },
                "epoch_stats": self.epoch_stats,
                "best_score": self.best_score,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint and restore all state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.manager.load_state_dict(ckpt["manager_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        rms = ckpt["reward_rms"]
        self.reward_rms.mean = rms["mean"]
        self.reward_rms.var = rms["var"]
        self.reward_rms.count = rms["count"]
        self.epoch_stats = ckpt.get("epoch_stats", [])
        self.best_score = ckpt.get("best_score", float("inf"))
