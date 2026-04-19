import 'package:RouteIQ_UI/theme/app_colors.dart';
import 'package:flutter/material.dart';

// ─── PIPELINE STAGE DATA ─────────────────────────────────────────────────────

class Stage {
  final int num;
  final String name;
  final String modelTag;
  final Color accentColor;
  final String description;
  final String input;
  final String output;
  final List<String> bullets;

  const Stage({
    required this.num,
    required this.name,
    required this.modelTag,
    required this.accentColor,
    required this.description,
    required this.input,
    required this.output,
    required this.bullets,
  });
}

const stages = [
  Stage(
    num: 1,
    name: 'Feature Extractor',
    modelTag: 'HAND-CRAFTED',
    accentColor: AppColors.cyan,
    description:
        'Computes a fixed 7-dimensional feature vector from the raw .vrp instance. '
        'Captures scale, demand distribution, spatial geometry, depot position, and '
        'capacity tightness — all deterministically, with zero learned parameters. '
        'Runs once per episode and is reused across all 50 solver steps.',
    input: '.vrp file\n(coords, demands, capacity)',
    output: 'inst_features (7-dim float32)',
    bullets: [
      'size_norm = num_customers / 400 — normalised instance scale',
      'demand_fill_ratio = total_demand / (nv_min × capacity) — vehicle packing tightness; near 1.0 means very little fleet slack',
      'mean_dist_norm & std_dist_norm — average and spread of inter-customer distances, normalised by max distance',
      'depot_centrality = mean_depot_dist / max_distance — peripheral depots create natural customer clusters (easier to consolidate)',
      'demand_cv = std_demand / mean_demand — coefficient of variation of customer demands; high CV means a few "heavy" customers dominate',
      'capacity_tightness = max_demand / capacity — if any single customer nearly fills a vehicle, that vehicle is almost always needed',
      'Computed once at episode start (reset()), cached for all 50 step() calls — no GPU required',
    ],
  ),
  Stage(
    num: 2,
    name: 'Fleet Manager',
    modelTag: 'ACTOR-CRITIC PPO',
    accentColor: AppColors.amber,
    description:
        'The RL "brain". At each of the 50 episode steps it observes the 7 instance features '
        'plus 7 real-time solver statistics (14-dim total) and selects one of 7 discrete '
        'fleet-target strategies via a tiny Actor-Critic network (~5,700 parameters). '
        'The key decision: when is it worth pushing for one fewer vehicle vs locking the '
        'current fleet and optimising distance instead?',
    input: 'inst_features (7) + solver_stats (7)\n= obs (14-dim)',
    output: 'action 0–6\n(fleet-target strategy)',
    bullets: [
      'Observation (14-dim): [size_norm, demand_fill_ratio, mean_dist_norm, std_dist_norm, depot_centrality, demand_cv, capacity_tightness | time_ratio, nv_ratio, score_ratio, stagnation_ratio, nv_gap, last_reward, last_action_norm]',
      '7 discrete actions: FREE_SAME / FREE_NEW (unconstrained fleet, 500 iters), LOCK_SAME / LOCK_NEW (lock best NV, 500 iters), PUSH_SAME / PUSH_NEW (best_nv − 1, 1000 iters), FORCE_MIN (nv_min, 1500 iters)',
      'Action masking: PUSH and FORCE_MIN are blocked when best_nv ≤ nv_min — impossible fleet reductions cannot be attempted',
      'Network: Linear(14→64)+ReLU → Linear(64→64)+ReLU → [Actor: Linear(64→7)] + [Critic: Linear(64→1)] — ~5,700 parameters total',
      'Removing one vehicle saves 1000 pts in score (1000×NV+TD); the agent learns instance-specifically when this risk is worthwhile',
    ],
  ),
  Stage(
    num: 3,
    name: 'HGS Engine',
    modelTag: 'HYB. GENETIC SEARCH',
    accentColor: AppColors.purple,
    description:
        'The actual CVRP solver — a high-performance C++ Hybrid Genetic Search '
        '(hygese / Vidal et al. 2012) wrapped as a Gymnasium environment. The Fleet '
        'Manager controls three levers per step: fleet target (free / lock / push / force), '
        'random seed (same = reproducible, new = escape local optima), and iteration '
        'budget (500 / 1000 / 1500 depending on action). Each step runs a completely '
        'fresh HGS solve — no warm starting.',
    input: 'action → (fleet_target, seed, nb_iter)',
    output: 'CVRP solution\n(NV, TD, routes)',
    bullets: [
      'FREE actions: unconstrained fleet, 500 iterations — lets HGS find the natural fleet size quickly',
      'LOCK actions: fixes num_vehicles = best_nv, 500 iterations — optimises total distance at the current fleet size',
      'PUSH actions: sets num_vehicles = best_nv − 1, 1000 iterations — forces HGS to find a route plan with one fewer truck',
      'FORCE_MIN: sets num_vehicles = nv_min = ⌈total_demand / capacity⌉, 1500 iterations — attempts the theoretical minimum fleet',
      'Fleet explosion guard: if a solve returns NV > best_nv + 2 or fails entirely, the result is discarded and a −5.0 penalty is issued',
      'Episode total: 50 steps × 500–1500 iters ≈ 25,000–75,000 cumulative HGS iterations per instance',
    ],
  ),
  Stage(
    num: 4,
    name: 'PPO Trainer',
    modelTag: 'REWARD PROPAGATION',
    accentColor: AppColors.green,
    description:
        'Trains the Fleet Manager using Proximal Policy Optimization (PPO). After each '
        'epoch (8 episodes × 50 steps = 400 transitions), GAE-λ advantages are computed '
        'and 3 mini-epochs of gradient updates are applied. The agent is evaluated '
        'greedily on 5 fixed instances each epoch to track true learning progress.',
    input: 'episode transitions\n(obs, action, reward, done)',
    output: 'updated Fleet Manager\npolicy weights',
    bullets: [
      'Reward: +pct × 100 when a new episode best is found (percentage improvement), −0.5 for no improvement, −5.0 for fleet explosion',
      'GAE-λ (λ=0.90, γ=0.95): generalised advantage estimation — interpolates between TD(0) low-variance and Monte Carlo unbiased estimates',
      'PPO clipped objective (ε=0.2): limits policy ratio to [0.8, 1.2] per update step, preventing catastrophic policy changes',
      'KL early stopping (target_kl=0.015): aborts the mini-epoch loop if policy drift is too large',
      'Entropy bonus (coeff=0.02): keeps the policy from collapsing to always picking one action — healthy entropy stays above ~0.5',
      'Curriculum: epochs 1–20 train on small instances (N ≤ 100); epochs 21+ unlock all 59 X-dataset instances up to N=1001',
      'LR schedule: Adam at 1e-4, linear decay to 5e-5 over total training epochs; gradient norm clipped at 0.5',
    ],
  ),
];
