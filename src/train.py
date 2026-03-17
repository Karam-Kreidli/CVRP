"""
Stage 5 - Multi-Agent Training Loop: PPO-based joint optimization.

Co-trains the Fleet Manager and Route Driver using Proximal Policy Optimization:
  - MACA Credit Assignment: Manager gets 1000×ΔNV, Driver gets ΔTD
  - Running Mean/Std reward normalization (Welford's online algorithm)
  - Separate optimizers per agent with different learning rates
  - FP16 mixed precision via autocast + GradScaler for Colab T4
  - GAE-λ for advantage estimation with optional early KL stopping
"""

from __future__ import annotations

import csv
import math
import os
import pathlib
import shutil
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.agent_manager import FleetManager
from src.agent_driver import RouteDriver
from src.solver_engine import CVRPEnv


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO hyperparameters for both agents."""
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    vf_coeff: float = 0.5
    ent_coeff: float = 0.05
    manager_lr: float = 1e-4
    driver_lr: float = 5e-4
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
    explosions while preserving the Driver's smaller distance-based signal.
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
# Rollout Buffers
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


class DriverRolloutBuffer:
    """Stores transitions for the Route Driver (variable-size node embeddings)."""

    def __init__(self):
        self.clear()

    def store(self, node_embeddings, action, log_prob, value, reward, done):
        self.node_embeddings.append(node_embeddings)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.node_embeddings = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
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
# MARL Trainer
# ---------------------------------------------------------------------------

class MARLTrainer:
    """Multi-Agent PPO trainer for Fleet Manager + Route Driver.

    Collects rollouts from CVRPEnv, decomposes the competition objective into
    agent-specific MACA rewards, normalizes them, and runs PPO updates with
    separate optimizers.

    Args:
        env: CVRPEnv instance (route_driver should be None; trainer manages it).
        config: PPO hyperparameters.
        device: Torch device.
    """

    def __init__(
        self,
        env: CVRPEnv,
        config: PPOConfig = PPOConfig(),
        device: torch.device = torch.device("cpu"),
        log_dir: str = "logs",
        gdrive_path: Optional[str] = None,
        total_epochs: int = 100,
    ):
        self.env = env
        self.config = config
        self.device = device
        self.log_dir = pathlib.Path(log_dir)
        self.gdrive_path = gdrive_path

        # Models
        self.manager = FleetManager().to(device)
        self.driver = RouteDriver().to(device)

        # Separate optimizers (different learning rates)
        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(), lr=config.manager_lr
        )
        self.driver_optimizer = torch.optim.Adam(
            self.driver.parameters(), lr=config.driver_lr
        )

        # Linear LR decay to 50% over total_epochs
        end_factor = 0.5
        self.mgr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.manager_optimizer,
            start_factor=1.0, end_factor=end_factor,
            total_iters=total_epochs,
        )
        self.drv_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.driver_optimizer,
            start_factor=1.0, end_factor=end_factor,
            total_iters=total_epochs,
        )

        # FP16 mixed precision
        self.amp_enabled = config.use_fp16 and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # Rollout buffers (separate per agent)
        self.mgr_buffer = RolloutBuffer()
        self.drv_buffer = DriverRolloutBuffer()

        # Running reward normalization (MACA)
        self.mgr_reward_rms = RunningMeanStd()
        self.drv_reward_rms = RunningMeanStd()

        # Logging
        self.epoch_stats: list[dict] = []
        self.best_score = float("inf")

        # Initialize CSV log
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_metrics.csv"
        self._csv_header_written = False

    # ------------------------------------------------------------------
    # Experience Collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, num_episodes: int = 1) -> dict:
        """Run episodes and collect transitions for both agents.

        MACA Credit Assignment:
          Manager reward = 1000 × (NV_prev - NV_current)
          Driver  reward = TD_prev - TD_current

        Returns:
            Dictionary with episode statistics.
        """
        self.mgr_buffer.clear()
        self.drv_buffer.clear()
        self.manager.eval()
        self.driver.eval()

        total_mgr_reward = 0.0
        total_drv_reward = 0.0
        total_steps = 0
        last_info = {}

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            prev_nv = info["nv"]
            prev_td = info["td"]
            action_mask = info.get("action_mask")
            done = False

            while not done:
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                graph_emb = obs_t[:, :128]
                solver_stats = obs_t[:, 128:]

                # Build mask tensor
                mask_t = None
                if action_mask is not None:
                    mask_t = torch.tensor(
                        action_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)

                # Manager decision
                with torch.no_grad():
                    action, mgr_lp, mgr_val = self.manager.select_action(
                        graph_emb, solver_stats, action_mask=mask_t
                    )
                action_int = action.item()

                # Driver decision (only on INTENSIVE_POLISH = action 0)
                drv_action = None
                drv_lp_val = None
                drv_v_val = None
                drv_node_emb = None
                if action_int == 0:
                    node_emb = self.env._node_embeddings
                    with torch.no_grad():
                        drv_action, drv_lp, drv_val = self.driver.select_operator(
                            node_emb
                        )
                    drv_lp_val = drv_lp.item()
                    drv_v_val = drv_val.squeeze().item()
                    drv_node_emb = node_emb.detach().clone()

                # Step environment
                next_obs, _, terminated, truncated, info = self.env.step(action_int)
                done = terminated or truncated

                # MACA credit assignment
                new_nv = info["nv"]
                new_td = info["td"]
                mgr_reward = 1000.0 * (prev_nv - new_nv)
                drv_reward = prev_td - new_td

                # Store manager transition
                self.mgr_buffer.store(
                    obs=obs.copy(),
                    action=action_int,
                    log_prob=mgr_lp.item(),
                    value=mgr_val.squeeze().item(),
                    reward=mgr_reward,
                    done=done,
                    action_mask=action_mask.copy() if action_mask is not None else None,
                )

                # Store driver transition (only when INTENSIVE_POLISH)
                if action_int == 0 and drv_action is not None:
                    self.drv_buffer.store(
                        node_embeddings=drv_node_emb,
                        action=drv_action,
                        log_prob=drv_lp_val,
                        value=drv_v_val,
                        reward=drv_reward,
                        done=done,
                    )
                    total_drv_reward += drv_reward

                total_mgr_reward += mgr_reward
                total_steps += 1
                prev_nv = new_nv
                prev_td = new_td
                obs = next_obs
                action_mask = info.get("action_mask")
                last_info = info

        # Normalize rewards with running statistics
        mgr_norm_mag = 0.0
        drv_norm_mag = 0.0
        if len(self.mgr_buffer) > 0:
            raw = np.array(self.mgr_buffer.rewards)
            self.mgr_reward_rms.update(raw)
            normed = self.mgr_reward_rms.normalize(raw)
            self.mgr_buffer.rewards = normed.tolist()
            mgr_norm_mag = float(np.mean(np.abs(normed)))

        if len(self.drv_buffer) > 0:
            raw = np.array(self.drv_buffer.rewards)
            self.drv_reward_rms.update(raw)
            normed = self.drv_reward_rms.normalize(raw)
            self.drv_buffer.rewards = normed.tolist()
            drv_norm_mag = float(np.mean(np.abs(normed)))

        # Compute GAE-λ advantages
        if len(self.mgr_buffer) > 0:
            self.mgr_buffer.compute_gae(0.0, self.config.gamma, self.config.lam)
        if len(self.drv_buffer) > 0:
            self.drv_buffer.compute_gae(0.0, self.config.gamma, self.config.lam)

        return {
            "total_steps": total_steps,
            "mgr_reward": total_mgr_reward,
            "drv_reward": total_drv_reward,
            "mgr_norm_mag": mgr_norm_mag,
            "drv_norm_mag": drv_norm_mag,
            "final_nv": last_info.get("nv", 0),
            "final_td": last_info.get("td", 0.0),
            "final_score": last_info.get("score", 0.0),
        }

    # ------------------------------------------------------------------
    # PPO Updates
    # ------------------------------------------------------------------

    def ppo_update_manager(self) -> dict:
        """Run PPO clipped objective update epochs on the Fleet Manager."""
        if len(self.mgr_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.manager.train()
        cfg = self.config

        obs = torch.tensor(
            np.array(self.mgr_buffer.observations),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.tensor(
            self.mgr_buffer.actions, dtype=torch.long, device=self.device
        )
        old_lp = torch.tensor(
            self.mgr_buffer.log_probs, dtype=torch.float32, device=self.device
        )
        advantages = torch.tensor(
            self.mgr_buffer.advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(
            self.mgr_buffer.returns, dtype=torch.float32, device=self.device
        )

        # Build action masks tensor (True = allowed)
        masks_list = self.mgr_buffer.action_masks
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
        total_pl, total_vl, total_ent, n = 0.0, 0.0, 0.0, 0

        for _ in range(cfg.ppo_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, cfg.mini_batch_size):
                idx = indices[start : start + cfg.mini_batch_size]
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

                self.manager_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.manager_optimizer)
                nn.utils.clip_grad_norm_(
                    self.manager.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.manager_optimizer)
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

    def ppo_update_driver(self) -> dict:
        """Run PPO update epochs on the Route Driver.

        Processes transitions individually due to variable-size node embeddings.
        """
        if len(self.drv_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.driver.train()
        cfg = self.config

        old_lp = torch.tensor(
            self.drv_buffer.log_probs, dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            self.drv_buffer.actions, dtype=torch.long, device=self.device
        )
        advantages = torch.tensor(
            self.drv_buffer.advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(
            self.drv_buffer.returns, dtype=torch.float32, device=self.device
        )

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(old_lp)
        total_pl, total_vl, total_ent, n = 0.0, 0.0, 0.0, 0

        for _ in range(cfg.ppo_epochs):
            for i in torch.randperm(T).tolist():
                node_emb = self.drv_buffer.node_embeddings[i].to(self.device)
                mb_act = actions[i : i + 1]
                mb_old_lp = old_lp[i]
                mb_adv = advantages[i]
                mb_ret = returns[i]

                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    logits, value = self.driver(node_emb)
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(mb_act).squeeze()
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.epsilon_clip, 1.0 + cfg.epsilon_clip
                    ) * mb_adv
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.mse_loss(value.squeeze(), mb_ret)
                    loss = (
                        policy_loss
                        + cfg.vf_coeff * value_loss
                        - cfg.ent_coeff * entropy
                    )

                self.driver_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.driver_optimizer)
                nn.utils.clip_grad_norm_(
                    self.driver.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.driver_optimizer)
                self.scaler.update()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n += 1

        n = max(n, 1)
        return {
            "policy_loss": total_pl / n,
            "value_loss": total_vl / n,
            "entropy": total_ent / n,
        }

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    def train_epoch(self, num_episodes: int = 1) -> dict:
        """One full training iteration: collect rollouts + PPO updates."""
        rollout_stats = self.collect_rollouts(num_episodes)
        mgr_stats = self.ppo_update_manager()
        drv_stats = self.ppo_update_driver()

        # Step LR schedulers
        self.mgr_scheduler.step()
        self.drv_scheduler.step()

        stats = {
            **rollout_stats,
            "mgr_policy_loss": mgr_stats["policy_loss"],
            "mgr_value_loss": mgr_stats["value_loss"],
            "mgr_entropy": mgr_stats["entropy"],
            "drv_policy_loss": drv_stats["policy_loss"],
            "drv_value_loss": drv_stats["value_loss"],
            "drv_entropy": drv_stats["entropy"],
            "mgr_lr": self.manager_optimizer.param_groups[0]["lr"],
            "drv_lr": self.driver_optimizer.param_groups[0]["lr"],
        }
        self.epoch_stats.append(stats)
        self._write_csv(stats)

        # Best-model tracking: save on new record score
        current_score = stats.get("final_score", float("inf"))
        if current_score < self.best_score:
            self.best_score = current_score
            best_path = self.log_dir / "best_model.pth"
            self.save_checkpoint(str(best_path))
            if self.gdrive_path:
                self._push_to_gdrive(best_path)

        return stats

    def _write_csv(self, stats: dict):
        """Append one row of metrics to the CSV log."""
        fieldnames = [
            "epoch", "final_nv", "final_td", "final_score",
            "mgr_reward", "drv_reward", "mgr_norm_mag", "drv_norm_mag",
            "total_steps",
            "mgr_policy_loss", "mgr_value_loss", "mgr_entropy",
            "drv_policy_loss", "drv_value_loss", "drv_entropy",
            "mgr_lr", "drv_lr",
        ]
        row = {k: stats.get(k, 0) for k in fieldnames}
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
        """Save both agent models, optimizers, and normalization state."""
        torch.save(
            {
                "manager_state_dict": self.manager.state_dict(),
                "driver_state_dict": self.driver.state_dict(),
                "manager_optimizer": self.manager_optimizer.state_dict(),
                "driver_optimizer": self.driver_optimizer.state_dict(),
                "mgr_scheduler": self.mgr_scheduler.state_dict(),
                "drv_scheduler": self.drv_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "mgr_reward_rms": {
                    "mean": self.mgr_reward_rms.mean,
                    "var": self.mgr_reward_rms.var,
                    "count": self.mgr_reward_rms.count,
                },
                "drv_reward_rms": {
                    "mean": self.drv_reward_rms.mean,
                    "var": self.drv_reward_rms.var,
                    "count": self.drv_reward_rms.count,
                },
                "epoch_stats": self.epoch_stats,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint and restore all state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.manager.load_state_dict(ckpt["manager_state_dict"])
        self.driver.load_state_dict(ckpt["driver_state_dict"])
        self.manager_optimizer.load_state_dict(ckpt["manager_optimizer"])
        self.driver_optimizer.load_state_dict(ckpt["driver_optimizer"])
        if "mgr_scheduler" in ckpt:
            self.mgr_scheduler.load_state_dict(ckpt["mgr_scheduler"])
        if "drv_scheduler" in ckpt:
            self.drv_scheduler.load_state_dict(ckpt["drv_scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        for attr, key in [
            (self.mgr_reward_rms, "mgr_reward_rms"),
            (self.drv_reward_rms, "drv_reward_rms"),
        ]:
            rms = ckpt[key]
            attr.mean = rms["mean"]
            attr.var = rms["var"]
            attr.count = rms["count"]
        self.epoch_stats = ckpt.get("epoch_stats", [])
