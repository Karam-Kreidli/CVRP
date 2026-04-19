"""
Stage 5 - Training Loop: PPO-based optimization of the Fleet Manager.

This is the learning engine. It trains the Fleet Manager to pick better
solver strategies over time using Proximal Policy Optimization (PPO).

HOW PPO TRAINING WORKS (simplified):
  1. COLLECT EXPERIENCE: Run 8 episodes, each with 50 steps. The agent picks
     actions, the solver runs, and we record (state, action, reward) transitions.
  2. COMPUTE ADVANTAGES: For each transition, calculate "how much better was
     this action compared to the average?" using GAE-λ (Generalized Advantage
     Estimation). This tells us which actions were surprisingly good or bad.
  3. UPDATE THE POLICY: Run 3 mini-epochs of gradient descent on the collected
     data. PPO's "clipped objective" prevents the policy from changing too much
     in one update (stability).
  4. EVALUATE: Run the current policy greedily (no randomness) on 5 fixed
     instances to track real learning progress.

KEY FEATURES:
  - Reward clipping: Rewards are clipped to [-10, 10] to prevent extreme values
    from destabilizing training. The percentage-based reward design keeps values
    in a reasonable range naturally.
  - FP16 mixed precision: Halves memory usage on GPU, enabling larger batches.
  - Action masking: Prevents impossible actions from being selected.
  - Linear LR decay: Learning rate decreases from 1e-4 to 5e-5 over training,
    allowing large updates early and fine-tuning later.
  - Fixed evaluation set: AvgScore is noisy (random instances each episode).
    Eval score on 5 fixed instances is the real progress metric.
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Model.Agent_Manager import FleetManager
from Model.Solver_Engine import CVRPEnv, INSTANCE_FEATURES_DIM


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO hyperparameters — these control the learning dynamics.

    Tuning these values affects training stability and speed. The defaults
    are standard PPO values that work well for this problem.
    """

    # --- Reward discounting ---
    gamma: float = 0.95        # Discount factor for future rewards.
                                # With 50 steps/episode, 0.95^50 ≈ 0.077 so the agent
                                # focuses on local consequences of its actions.

    lam: float = 0.90          # GAE-λ: bias-variance tradeoff for advantage estimation.
                                # λ=1.0 → high variance, unbiased (Monte Carlo-like)
                                # λ=0.0 → low variance, biased (TD-like)
                                # λ=0.90 with 50-step episodes gives more bias reduction
                                # which helps with noisier per-step rewards.

    # --- PPO clipping ---
    epsilon_clip: float = 0.2  # PPO clip range: policy ratio is clamped to [1-ε, 1+ε].
                                # This prevents the policy from changing too drastically
                                # in a single update. Without clipping, a few lucky rewards
                                # could cause the policy to overfit to specific strategies.

    # --- Loss coefficients ---
    vf_coeff: float = 0.5     # Weight of the value (critic) loss in the total loss.
                                # Total loss = policy_loss + 0.5*value_loss - 0.05*entropy

    ent_coeff: float = 0.02   # Entropy bonus: encourages exploration by penalizing
                                # overly confident policies. Lower than before (0.05) because
                                # 50 steps/episode provides natural exploration diversity.
                                # If entropy drops to 0, the agent has "collapsed" to always
                                # picking the same action (bad — it stopped exploring).

    # --- Optimizer ---
    manager_lr: float = 1e-4  # Learning rate for Adam optimizer. Decays linearly to 50%
                                # over training (see LinearLR scheduler in MARLTrainer).

    # --- Mini-batch training ---
    ppo_epochs: int = 3        # Number of passes over the collected data per PPO update.
                                # Reduced from 4 to avoid overfitting with larger batches.

    mini_batch_size: int = 128 # Transitions per gradient step. With 8 episodes × 50 steps
                                # = 400 transitions per rollout, this gives ~3 mini-batches.

    max_grad_norm: float = 0.5 # Gradient clipping: prevents exploding gradients by scaling
                                # down the gradient if its norm exceeds this threshold.

    # --- Mixed precision ---
    use_fp16: bool = True      # FP16 (half precision) training via PyTorch AMP.
                                # Cuts GPU memory usage roughly in half with minimal accuracy loss.

    # --- Safety ---
    target_kl: Optional[float] = 0.015  # KL divergence threshold for early stopping.
                                         # If the policy changes too much in one PPO update
                                         # (KL > 1.5 × target_kl), we stop early to prevent
                                         # catastrophic policy updates.

    # --- Reward clipping ---
    reward_clip_min: float = -10.0      # Lower clipping bound for rollout rewards.
    reward_clip_max: float = 10.0       # Upper clipping bound for rollout rewards.


# ---------------------------------------------------------------------------
# Reward Clipping
# ---------------------------------------------------------------------------

REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions collected during experience rollouts.

    During each epoch, the agent interacts with the environment for multiple
    episodes, generating transitions of the form:
      (observation, action, log_prob, value, reward, done, action_mask)

    These are stored here and then used for the PPO update. After each PPO
    update, the buffer is cleared for the next epoch's rollouts.

    The buffer also computes GAE-λ advantages, which tell PPO "how much better
    (or worse) was each action compared to what the critic expected?"
    """

    def __init__(self):
        self.clear()

    def store(self, obs, action, log_prob, value, reward, done, action_mask=None):
        """Record one transition (one step of one episode)."""
        self.observations.append(obs)     # (OBS_DIM,) observation vector
        self.actions.append(action)       # int: which action was taken (0-9)
        self.log_probs.append(log_prob)   # float: log π(a|s) at the time of action selection
        self.values.append(value)         # float: V(s) from the critic
        self.rewards.append(reward)       # float: reward received (after normalization)
        self.dones.append(done)           # bool: was this the last step of the episode?
        self.action_masks.append(action_mask)  # (10,) bool: which actions were allowed

    def clear(self):
        """Reset the buffer for a new epoch."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.advantages = []  # Computed by compute_gae()
        self.returns = []     # Computed by compute_gae()

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """Compute GAE-λ (Generalized Advantage Estimation) advantages.

        WHAT IS AN ADVANTAGE?
          A(s, a) = "How much better was action 'a' in state 's' compared to
                     the average action in that state?"
          Positive advantage → action was better than expected
          Negative advantage → action was worse than expected

        WHY GAE-λ?
          There are two extremes for computing advantages:
            - Monte Carlo (λ=1): Use actual cumulative returns. High variance
              (noisy because each episode is different) but unbiased.
            - TD(0) (λ=0): Use one-step bootstrapped returns. Low variance
              but biased (depends on critic accuracy).
          GAE-λ interpolates between these using λ=0.95, getting most of the
          variance reduction of TD while staying close to unbiased.

        The formula (working backwards from the last step):
          δ_t = r_t + γ * V(s_{t+1}) - V(s_t)     [TD error at step t]
          A_t = δ_t + γ * λ * A_{t+1}               [GAE accumulation]

        Returns (stored in self.returns) = advantages + values, which is what
        the critic is trained to predict.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # Walk backwards through the episode (must accumulate from the end)
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            non_terminal = 1.0 - float(self.dones[t])  # 0 at episode boundaries

            # TD error: actual reward + discounted next value - predicted value
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]

            # Accumulate GAE (exponentially weighted sum of TD errors)
            gae = delta + gamma * lam * non_terminal * gae
            advantages[t] = gae

        # Returns = advantages + values (target for critic training)
        values_arr = np.array(self.values, dtype=np.float32)
        self.advantages = advantages.tolist()
        self.returns = (advantages + values_arr).tolist()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MARLTrainer:
    """PPO trainer for the Fleet Manager.

    This class orchestrates the full training loop:
      1. collect_rollouts() — run episodes, store transitions
      2. ppo_update() — compute advantages, run clipped gradient updates
      3. evaluate() — test current policy on fixed instances (no exploration)
      4. Logging, checkpointing, and best-model tracking

    The trainer manages all training state (model, optimizer, scheduler,
    reward normalization statistics) and can save/load checkpoints for
    resuming interrupted training runs.

    Args:
        env: CVRPEnv instance (the Gymnasium environment).
        config: PPO hyperparameters (see PPOConfig).
        device: Torch device (cuda or cpu).
        log_dir: Directory for CSV logs and best model checkpoint.
        gdrive_path: Optional Google Drive directory for checkpoint backup.
        total_epochs: Total training epochs (used for LR scheduling).
        eval_instances: List of .vrp file paths for fixed evaluation set.
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
        holdout_instances: list[str] | None = None,
        holdout_eval_interval: int = 5,
        best_model_metric: str = "eval",
        run_config: dict | None = None,
    ):
        self.env = env
        self.config = config
        self.device = device
        self.log_dir = pathlib.Path(log_dir)
        self.gdrive_path = gdrive_path
        self.run_config = run_config or {}

        # --- Model ---
        # The Fleet Manager is the only trainable component.
        self.manager = FleetManager().to(device)

        # --- Optimizer ---
        # Adam: adaptive learning rate optimizer (standard for deep RL).
        # Linear LR decay: starts at manager_lr (1e-4), linearly decreases to 50%
        # (5e-5) over total_epochs. This allows large, bold updates early in training
        # and finer adjustments later as the policy matures.
        self.optimizer = torch.optim.Adam(
            self.manager.parameters(), lr=config.manager_lr
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0, end_factor=0.5,
            total_iters=total_epochs,
        )

        # --- FP16 Mixed Precision ---
        # Uses PyTorch's Automatic Mixed Precision (AMP) to run forward/backward
        # passes in FP16 (half precision) on GPU. This roughly halves memory usage
        # and can speed up computation on modern GPUs. GradScaler prevents underflow
        # in FP16 gradients by dynamically scaling the loss.
        self.amp_enabled = config.use_fp16 and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # --- Experience Buffer ---
        # Stores all transitions from the current epoch's rollouts.
        # Cleared at the start of each epoch.
        self.buffer = RolloutBuffer()

        # --- Logging & Tracking ---
        self.epoch_stats: list[dict] = []     # Full history of epoch metrics
        self.best_score = float("inf")        # Best eval score seen (lower is better)

        # Fixed evaluation instances: 5 instances used for consistent progress tracking.
        # AvgScore varies wildly because training uses random instances each episode.
        # Eval score on fixed instances is the TRUE measure of learning.
        self.eval_instances = eval_instances or []
        self.holdout_instances = holdout_instances or []
        self.holdout_eval_interval = max(1, int(holdout_eval_interval))
        self.best_model_metric = best_model_metric
        if self.best_model_metric not in {"eval", "holdout", "composite"}:
            raise ValueError(
                f"Unsupported best_model_metric='{self.best_model_metric}'. "
                "Expected one of: eval, holdout, composite"
            )

        # CSV log for plotting training curves
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "Training_Metrics.csv"
        self._csv_header_written = False
        self._write_run_config()

    def _write_run_config(self):
        """Write one JSON metadata file per run for reproducibility."""
        payload = {
            "ppo_config": asdict(self.config),
            "eval_instances": list(self.eval_instances),
            "holdout_instances": list(self.holdout_instances),
            "holdout_eval_interval": self.holdout_eval_interval,
            "best_model_metric": self.best_model_metric,
            "run_config": self.run_config,
        }
        run_config_path = self.log_dir / "Run_Configuration.json"
        run_config_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_instance_set(self, instance_paths: list[str]) -> dict:
        """Run greedy evaluation on a provided instance set.

        Returns:
            dict with keys: score, nv, td (averaged over the provided set)
        """
        if not instance_paths:
            return {}

        self.manager.eval()  # Switch to eval mode (disables dropout, etc.)

        scores, nvs, tds = [], [], []
        orig_paths = self.env.instance_paths  # Save original paths to restore later

        for inst_path in instance_paths:
            # Temporarily point the env at just this one instance
            self.env.instance_paths = [pathlib.Path(inst_path)]
            obs, info = self.env.reset()
            done = False

            while not done:
                # Parse observation into instance features + solver stats
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                graph_emb = obs_t[:, :INSTANCE_FEATURES_DIM]
                solver_stats = obs_t[:, INSTANCE_FEATURES_DIM:]

                # Build action mask tensor
                action_mask = info.get("action_mask")
                mask_t = None
                if action_mask is not None:
                    mask_t = torch.tensor(
                        action_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)

                # GREEDY action selection: pick the highest-scoring action
                # (no sampling, no exploration - pure exploitation)
                with torch.no_grad():
                    logits, _ = self.manager(
                        graph_emb, solver_stats, action_mask=mask_t
                    )
                    action_int = logits.argmax(dim=-1).item()  # Greedy!

                obs, _, terminated, truncated, info = self.env.step(action_int)
                done = terminated or truncated

            # Record final score for this instance
            scores.append(info["score"])
            nvs.append(info["nv"])
            tds.append(info["td"])

        # Restore original instance paths
        self.env.instance_paths = orig_paths

        return {
            "score": float(np.mean(scores)),
            "nv": float(np.mean(nvs)),
            "td": float(np.mean(tds)),
        }

    def evaluate(self) -> dict:
        """Run greedy evaluation on fixed instances - the REAL progress metric.

        WHY A SEPARATE EVALUATION?
          During training, the agent runs on random instances with stochastic
          action selection (exploration). The resulting AvgScore is very noisy —
          it depends on which random instances were selected and which random
          actions were sampled. You can't tell if the agent is actually learning.

          Evaluation uses:
            - The SAME 5 instances every time (consistency)
            - GREEDY action selection (argmax, no randomness)
          So the eval score purely reflects how well the POLICY has improved.

        Returns:
            dict with eval_score, eval_nv, eval_td (averaged over 5 instances)
        """
        metrics = self._evaluate_instance_set(self.eval_instances)
        if not metrics:
            return {}

        return {
            "eval_score": metrics["score"],
            "eval_nv": metrics["nv"],
            "eval_td": metrics["td"],
        }

    def evaluate_holdout(self) -> dict:
        """Run greedy evaluation on a separate holdout split."""
        metrics = self._evaluate_instance_set(self.holdout_instances)
        if not metrics:
            return {}
        return {
            "holdout_eval_score": metrics["score"],
            "holdout_eval_nv": metrics["nv"],
            "holdout_eval_td": metrics["td"],
        }

    # ------------------------------------------------------------------
    # Experience Collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, num_episodes: int = 1) -> dict:
        """Run episodes to collect experience for PPO training.

        This is Phase 1 of the training loop. The agent interacts with the
        environment using its CURRENT policy, and we record every transition.

        For each of the 8 episodes:
          1. Reset the environment (random instance, initial solve)
          2. For 50 steps:
             - Agent observes state → picks action (STOCHASTIC — with exploration)
             - Environment executes action → returns reward and next state
             - Store the transition in the buffer
          3. Record final score

        After all episodes, we:
          - Clip all rewards to [-10, 10]
          - Compute GAE-λ advantages (how good was each action?)

        The buffer now contains ~400 transitions (8 episodes × 50 steps)
        ready for the PPO update.

        Returns:
            Dictionary with episode statistics (avg score, NV, TD, etc.)
        """
        self.buffer.clear()     # Start fresh each epoch
        self.manager.eval()     # No dropout during data collection

        total_reward = 0.0
        total_steps = 0
        episode_scores = []
        episode_nvs = []
        episode_tds = []

        for ep_idx in range(num_episodes):
            ep_t0 = time.time()
            obs, info = self.env.reset()      # Random instance, initial solve
            action_mask = info.get("action_mask")
            done = False

            while not done:
                # Parse the observation into its two components
                obs_t = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)                          # (1, OBS_DIM)
                graph_emb = obs_t[:, :INSTANCE_FEATURES_DIM]              # Instance features (12)
                solver_stats = obs_t[:, INSTANCE_FEATURES_DIM:]           # Solver stats (7)

                # Build action mask tensor for the Fleet Manager
                mask_t = None
                if action_mask is not None:
                    mask_t = torch.tensor(
                        action_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)

                # Agent selects an action (stochastic sampling for exploration)
                with torch.no_grad():
                    action, log_prob, value = self.manager.select_action(
                        graph_emb, solver_stats, action_mask=mask_t
                    )
                action_int = action.item()

                # Environment executes the action (runs HGS for 500 iterations)
                next_obs, reward, terminated, truncated, info = self.env.step(action_int)
                done = terminated or truncated

                # Store this transition in the buffer for later PPO update
                self.buffer.store(
                    obs=obs.copy(),
                    action=action_int,
                    log_prob=log_prob.item(),   # Needed for PPO's importance ratio
                    value=value.squeeze().item(), # Critic's estimate V(s)
                    reward=reward,               # Raw reward (normalized below)
                    done=done,
                    action_mask=action_mask.copy() if action_mask is not None else None,
                )

                total_reward += reward
                total_steps += 1
                obs = next_obs
                action_mask = info.get("action_mask")

            # Record end-of-episode metrics
            episode_scores.append(info.get("score", 0.0))
            episode_nvs.append(info.get("nv", 0))
            episode_tds.append(info.get("td", 0.0))

            ep_elapsed = time.time() - ep_t0
            ep_mins, ep_secs = divmod(int(ep_elapsed), 60)
            print(
                f"  Episode {ep_idx+1}/{num_episodes}: "
                f"score={info.get('score', 0.0):.0f}, "
                f"nv={info.get('nv', 0)}, "
                f"td={info.get('td', 0.0):.0f} "
                f"({ep_mins}m{ep_secs:02d}s)"
            )

        # --- Reward Clipping ---
        # Clip rewards to [-10, 10] to prevent extreme values from
        # destabilizing training. The percentage-based reward design
        # keeps most values in a reasonable range naturally.
        if len(self.buffer) > 0:
            raw = np.array(self.buffer.rewards)
            clipped = np.clip(raw, self.config.reward_clip_min, self.config.reward_clip_max)
            self.buffer.rewards = clipped.tolist()

        # --- Compute Advantages ---
        # GAE-λ tells PPO which actions were better/worse than expected.
        # last_value=0.0 because episodes always end at step 50 (no bootstrap needed).
        if len(self.buffer) > 0:
            self.buffer.compute_gae(0.0, self.config.gamma, self.config.lam)

        return {
            "total_steps": total_steps,
            "total_reward": total_reward,
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
        """Run PPO clipped objective update — Phase 2 of the training loop.

        This is where the actual LEARNING happens. We take the transitions
        collected in collect_rollouts() and use them to improve the policy.

        PPO (Proximal Policy Optimization) key idea:
          We want to update the policy to make good actions more likely and
          bad actions less likely. But we can't change the policy TOO MUCH
          in one update, or training becomes unstable. PPO solves this with
          a "clipped surrogate objective" that limits the policy change.

        The update runs for 3 mini-epochs over the data. If the policy changes
        too much (KL divergence exceeds threshold), we stop early.

        Returns:
            dict with average policy_loss, value_loss, entropy over the update.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.manager.train()  # Enable dropout, etc.
        cfg = self.config

        # Convert buffer data to tensors for GPU computation
        obs = torch.tensor(
            np.array(self.buffer.observations),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_lp = torch.tensor(           # Log-probs from when the data was collected
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        advantages = torch.tensor(        # GAE-λ advantages (how good was each action?)
            self.buffer.advantages, dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(           # Target values for the critic
            self.buffer.returns, dtype=torch.float32, device=self.device
        )

        # Build action masks tensor (True = action was allowed)
        masks_list = self.buffer.action_masks
        if masks_list and masks_list[0] is not None:
            action_masks = torch.tensor(
                np.array(masks_list), dtype=torch.bool, device=self.device
            )
        else:
            action_masks = None

        # Normalize advantages to zero mean, unit variance within this batch.
        # This is standard PPO practice — it makes the gradient scale consistent
        # regardless of the absolute magnitude of advantages.
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(obs)                          # Total transitions (~400)
        mb_size = min(cfg.mini_batch_size, T)  # Mini-batch size for each gradient step
        total_pl, total_vl, total_ent, n = 0.0, 0.0, 0.0, 0

        # --- PPO Mini-Epoch Loop ---
        # Run 3 passes over the data, each time shuffling and splitting into mini-batches.
        for _ in range(cfg.ppo_epochs):
            indices = torch.randperm(T, device=self.device)  # Shuffle transition order
            for start in range(0, T, mb_size):
                idx = indices[start : start + mb_size]

                # Extract mini-batch
                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_old_lp = old_lp[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_mask = action_masks[idx] if action_masks is not None else None

                # Forward pass with optional FP16 for GPU efficiency
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    # Get current policy's logits and value estimates
                    logits, values = self.manager(
                        mb_obs[:, :INSTANCE_FEATURES_DIM], mb_obs[:, INSTANCE_FEATURES_DIM:],  # instance_features, solver_stats
                        action_mask=mb_mask,
                    )
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(mb_act)   # Log-prob under CURRENT policy
                    entropy = dist.entropy().mean()   # Policy randomness

                    # --- PPO Clipped Surrogate Objective ---
                    # ratio = π_new(a|s) / π_old(a|s)
                    # If ratio > 1: current policy is MORE likely to take this action
                    # If ratio < 1: current policy is LESS likely
                    ratio = torch.exp(new_lp - mb_old_lp)

                    # Two surrogate objectives:
                    # surr1: raw ratio × advantage (unconstrained update)
                    # surr2: clipped ratio × advantage (constrained to [1-ε, 1+ε])
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.epsilon_clip, 1.0 + cfg.epsilon_clip
                    ) * mb_adv

                    # Take the MINIMUM of the two → pessimistic bound.
                    # This prevents the policy from changing too much in one step.
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss: MSE between predicted V(s) and actual returns
                    value_loss = F.mse_loss(values.squeeze(-1), mb_ret)

                    # Total loss = policy + value - entropy_bonus
                    # The entropy bonus (subtracted because we minimize loss)
                    # encourages exploration by penalizing overly deterministic policies.
                    loss = (
                        policy_loss
                        + cfg.vf_coeff * value_loss    # 0.5 × value loss
                        - cfg.ent_coeff * entropy       # 0.05 × entropy bonus
                    )

                # --- Gradient Step ---
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()       # Compute gradients (FP16-safe)
                self.scaler.unscale_(self.optimizer)      # Unscale for gradient clipping
                nn.utils.clip_grad_norm_(                 # Clip gradient norm to 0.5
                    self.manager.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)          # Apply the update
                self.scaler.update()                      # Adjust FP16 loss scale

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n += 1

            # --- KL Divergence Early Stopping ---
            # After each mini-epoch, check if the policy has drifted too far
            # from the old policy. If KL > 1.5 × target_kl, stop updating
            # to prevent catastrophic policy changes.
            if cfg.target_kl is not None:
                with torch.no_grad():
                    logits_all, _ = self.manager(
                        obs[:, :INSTANCE_FEATURES_DIM], obs[:, INSTANCE_FEATURES_DIM:],
                        action_mask=action_masks,
                    )
                    kl = (
                        old_lp
                        - Categorical(logits=logits_all).log_prob(actions)
                    ).mean().item()
                    if kl > 1.5 * cfg.target_kl:
                        break  # Policy changed too much — stop here

        n = max(n, 1)
        return {
            "policy_loss": total_pl / n,
            "value_loss": total_vl / n,
            "entropy": total_ent / n,
        }

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        num_episodes: int = 1,
        run_eval: bool = True,
        epoch_index: int | None = None,
    ) -> dict:
        """One full training epoch: collect → update → evaluate.

        This is the top-level training step called once per epoch:
          1. Collect rollouts (8 episodes of 50 steps each)
          2. PPO update (learn from collected experience)
          3. Step the LR scheduler (linear decay)
          4. Evaluate on fixed instances (track real progress)
          5. Save best model if eval score improved
          6. Log metrics to CSV

        Returns:
            dict with all metrics for this epoch (for printing/logging).
        """
        # Phase 1: Collect experience using current policy
        rollout_stats = self.collect_rollouts(num_episodes)

        # Phase 2: Learn from the collected experience
        ppo_stats = self.ppo_update()

        # Step the learning rate scheduler (linear decay)
        self.scheduler.step()

        # Phase 3: Evaluate on fixed instances (no exploration)
        eval_stats = self.evaluate() if run_eval else {}
        holdout_stats = {}
        if run_eval and self.holdout_instances:
            epoch_num = epoch_index if epoch_index is not None else (len(self.epoch_stats) + 1)
            if epoch_num % self.holdout_eval_interval == 0:
                holdout_stats = self.evaluate_holdout()

        # Combine all metrics into one dict
        stats = {
            **rollout_stats,
            **eval_stats,
            **holdout_stats,
            "policy_loss": ppo_stats["policy_loss"],
            "value_loss": ppo_stats["value_loss"],
            "entropy": ppo_stats["entropy"],
            "lr": self.optimizer.param_groups[0]["lr"],
            "gamma": self.config.gamma,
            "lam": self.config.lam,
            "epsilon_clip": self.config.epsilon_clip,
            "vf_coeff": self.config.vf_coeff,
            "ent_coeff": self.config.ent_coeff,
            "manager_lr_init": self.config.manager_lr,
            "ppo_epochs": self.config.ppo_epochs,
            "mini_batch_size": self.config.mini_batch_size,
            "max_grad_norm": self.config.max_grad_norm,
            "target_kl": "" if self.config.target_kl is None else self.config.target_kl,
            "reward_clip_min": self.config.reward_clip_min,
            "reward_clip_max": self.config.reward_clip_max,
            "failure_penalty": getattr(self.env, "failure_penalty", ""),
            "no_improvement_penalty": getattr(self.env, "no_improvement_penalty", ""),
            "eval_instances_count": len(self.eval_instances),
            "holdout_instances_count": len(self.holdout_instances),
            "holdout_eval_interval": self.holdout_eval_interval,
            "best_model_metric": self.best_model_metric,
            "run_tag": self.run_config.get("run_tag", ""),
        }

        # Choose which metric controls best-checkpoint tracking.
        # For holdout/composite modes we only update best checkpoints on epochs
        # where holdout is actually evaluated.
        selection_metric = "eval_score"
        tracking_score = stats.get("eval_score", stats.get("best_score", float("inf")))

        if self.best_model_metric == "holdout":
            selection_metric = "holdout_eval_score"
            if "holdout_eval_score" in stats:
                tracking_score = stats["holdout_eval_score"]
            else:
                tracking_score = float("inf")
        elif self.best_model_metric == "composite":
            selection_metric = "composite_eval_holdout"
            if "eval_score" in stats and "holdout_eval_score" in stats:
                tracking_score = 0.5 * (stats["eval_score"] + stats["holdout_eval_score"])
            else:
                tracking_score = float("inf")

        stats["selection_metric"] = selection_metric
        stats["tracking_score"] = "" if not np.isfinite(tracking_score) else float(tracking_score)

        self.epoch_stats.append(stats)
        self._write_csv(stats)

        # --- Best Model Tracking ---
        # Save the model whenever eval score improves (lower is better).
        # This ensures we always have the best-performing policy saved,
        # even if later training epochs regress.
        if tracking_score < self.best_score:
            self.best_score = tracking_score
            best_path = self.log_dir / "Best_Model.pth"
            self.save_checkpoint(str(best_path))
            if self.gdrive_path:
                self._push_to_gdrive(best_path)

        return stats

    def _write_csv(self, stats: dict):
        """Append one row of metrics to the CSV log."""
        fieldnames = [
            "epoch", "avg_nv", "avg_td", "avg_score", "best_score",
            "eval_score", "eval_nv", "eval_td",
            "num_episodes", "total_reward", "total_steps",
            "policy_loss", "value_loss", "entropy", "lr",
            "gamma", "lam", "epsilon_clip", "vf_coeff", "ent_coeff",
            "manager_lr_init", "ppo_epochs", "mini_batch_size", "max_grad_norm", "target_kl",
            "reward_clip_min", "reward_clip_max",
            "failure_penalty", "no_improvement_penalty",
            "eval_instances_count",
            "holdout_eval_score", "holdout_eval_nv", "holdout_eval_td",
            "holdout_instances_count", "holdout_eval_interval",
            "best_model_metric", "selection_metric", "tracking_score",
            "run_tag",
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
        """Save full training state to a checkpoint file.

        Saves EVERYTHING needed to resume training from exactly where we left off:
          - Model weights (the learned policy)
          - Optimizer state (Adam's momentum/variance estimates)
          - LR scheduler state (current position in the decay schedule)
          - FP16 scaler state (loss scaling factor)
          - Training history (all epoch stats)
          - Best eval score (for best-model tracking)

        Use with load_checkpoint() + --resume CLI flag to continue training
        after interruptions (GPU quota exhaustion, accidental Ctrl+C, etc.)
        """
        torch.save(
            {
                "manager_state_dict": self.manager.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch_stats": self.epoch_stats,
                "best_score": self.best_score,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load a checkpoint and restore all training state.

        Restores the model, optimizer, scheduler, scaler, training history,
        and best score. After calling this, training can continue seamlessly
        from the checkpoint epoch.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.manager.load_state_dict(ckpt["manager_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])

        self.epoch_stats = ckpt.get("epoch_stats", [])
        self.best_score = ckpt.get("best_score", float("inf"))
