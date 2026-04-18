# RL vs Baseline Competition-Oriented Report

## Setup
- Generated (UTC): 2026-04-18T15:57:47.723578+00:00
- Model: FleetManager @ logs/best_model.pth
- Device: cpu
- Instances evaluated: 59
- Baseline config: iters=25000, policy=single, seeds=[42]

## Head-to-Head Outcomes
- Wins/Losses/Ties (tie at 3 decimals): 6/42/11
- Win/Loss/Tie rates: 10.2% / 71.2% / 18.6%
- Mean score: RL=70301.763, BL=70266.220
- Delta RL-BL: mean=+35.542, median=+18.000 (negative is better for RL)
- Relative delta: mean=+0.013%, median=+0.032%

## Pairwise Formula-1 Surrogate Points
- Per-instance points: winner=10, runner-up=8, tie=9 each
- Total points: RL=495.0, BL=567.0, margin=-72.0
- Average points per instance: RL=8.390, BL=9.610

## Best/Worst Delta Cases
- Best for RL: X-n247-k50 (delta=-991.000, rel=-1.108%)
- Worst for RL: X-n289-k60 (delta=+1097.000, rel=+0.702%)

## Bootstrap Subset Robustness
- Trials=2000, subset_size=20, seed=123
- P(RL points > BL points) = 0.1%
- Expected points margin (RL-BL) = -24.163
- Points margin p05/p95 = -34.000 / -14.000
- P(subset mean delta < 0) = 31.8%

## Top 5 RL Improvements

| Instance | Delta (RL-BL) | Delta % | Outcome | Points |
|---|---:|---:|---:|---:|
| X-n247-k50 | -991.000 | -1.108% | win | 10-8 |
| X-n256-k16 | -980.000 | -2.731% | win | 10-8 |
| X-n148-k46 | -949.000 | -1.049% | win | 10-8 |
| X-n313-k71 | -622.000 | -0.372% | win | 10-8 |
| X-n322-k28 | -20.000 | -0.035% | win | 10-8 |

## Top 5 RL Regressions

| Instance | Delta (RL-BL) | Delta % | Outcome | Points |
|---|---:|---:|---:|---:|
| X-n289-k60 | +1097.000 | +0.702% | loss | 8-10 |
| X-n336-k84 | +1033.000 | +0.458% | loss | 8-10 |
| X-n266-k58 | +477.000 | +0.357% | loss | 8-10 |
| X-n384-k52 | +427.000 | +0.359% | loss | 8-10 |
| X-n359-k29 | +409.000 | +0.507% | loss | 8-10 |
