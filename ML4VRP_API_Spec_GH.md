# ML4VRP Web App — Backend API Specification

> Hand this document to the backend developer.
> All endpoints should be served from a **FastAPI** (or Flask) server running on `localhost:8080`.
> All request/response bodies are **JSON**. All endpoints must include **CORS headers** to allow the Flutter web app to call them from the browser.

---

## Quick Setup Requirement

```python
# FastAPI CORS setup (required for Flutter web)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Endpoint 1 — Health Check

```
GET /health
```

The web app calls this on startup to show the green "SYSTEM READY" dot in the sidebar and the `API: localhost:8080` status in the top bar.

**Response `200`**
```json
{
  "status": "ready",
  "device": "cuda",
  "gpu_name": "NVIDIA T4",
  "model_loaded": true,
  "stage_health": {
    "feature_extractor": 1.00,
    "fleet_manager": 0.84,
    "hgs_engine": 1.00,
    "ppo_trainer": 0.84
  },
  "total_training_epochs": 84,
  "best_score_ever": 54702.7
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ready"` or `"loading"` or `"error"` |
| `device` | string | `"cuda"` or `"cpu"` |
| `gpu_name` | string | GPU name, or `"CPU only"` |
| `model_loaded` | bool | Whether the Fleet Manager weights are loaded |
| `stage_health.feature_extractor` | float 0–1 | Always `1.0` — deterministic computation, no training required |
| `stage_health.fleet_manager` | float 0–1 | Actor-Critic PPO convergence: `current_epoch / total_epochs` |
| `stage_health.hgs_engine` | float 0–1 | Always `1.0` — HGS is a classical solver, not trained |
| `stage_health.ppo_trainer` | float 0–1 | Same as `fleet_manager` — reflects PPO training progress |
| `total_training_epochs` | int | How many PPO training epochs have completed so far |
| `best_score_ever` | float | Lowest `1000×NV + TD` score achieved across all training runs |

> **Note:** `stage_health` now has **4 keys** matching the 4-stage pipeline (Feature Extractor → Fleet Manager → HGS Engine → PPO Trainer). The old keys `gnn_observer`, `route_driver`, and `maca_trainer` are **removed**.

---

## Endpoint 2 — Start a Solver Job

```
POST /solve
Content-Type: multipart/form-data
```

Called when the user uploads a `.vrp` file and clicks **Run Solver**. Starts the 4-stage pipeline as a background job and immediately returns a `job_id`.

**Request body** — multipart form with:

| Field | Type | Description |
|---|---|---|
| `file` | file | The `.vrp` instance file (TSPLIB95 / VRPLIB format) |
| `track` | string | `"cvrp"` or `"cvrptw"` — default `"cvrp"` |
| `mode` | string | `"fast"` (60s), `"competition"` (300s), `"research"` (600s) |
| `time_limit_seconds` | int | Override time limit (optional, uses mode default if omitted) |

**How to implement:** spin up a background thread/task that runs `CVRPEnv` + `MARLTrainer` and stores progress in a dict keyed by `job_id`.

**Response `202 Accepted`**
```json
{
  "job_id": "a3f9c2b1",
  "instance_name": "X-n101-k25",
  "num_nodes": 101,
  "vehicle_capacity": 206,
  "total_demand": 5147,
  "nv_min": 25
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string | Unique ID — used to poll `/status` and `/result` |
| `instance_name` | string | Parsed from the `.vrp` filename |
| `num_nodes` | int | Total nodes including depot (from `DIMENSION` in file) |
| `vehicle_capacity` | int | From `CAPACITY` in the `.vrp` file |
| `total_demand` | int | Sum of all customer demands |
| `nv_min` | int | `ceil(total_demand / vehicle_capacity)` — theoretical fleet floor |

---

## Endpoint 3 — Poll Job Status

```
GET /status/{job_id}
```

The web app calls this **every 500ms** while a job is running to update the live pipeline monitor and metrics panel.

**Response `200`**
```json
{
  "job_id": "a3f9c2b1",
  "status": "running",
  "current_stage": 3,
  "stage_name": "HGS Engine",
  "stage_statuses": {
    "1": "done",
    "2": "done",
    "3": "running",
    "4": "waiting"
  },
  "stage_times_seconds": {
    "1": 0.02,
    "2": 0.4,
    "3": null,
    "4": null
  },
  "current_nv": 29,
  "best_nv": 26,
  "current_td": 31241.8,
  "best_td": 28702.7,
  "current_score": 60241.8,
  "best_score": 54702.7,
  "iteration": 8500,
  "max_iterations": 25000,
  "elapsed_seconds": 36,
  "current_action": "PUSH_NEW",
  "episode_step": 17,
  "episode_step_max": 50,
  "nv_min": 25,
  "log_lines": [
    "[00:00.022] [FE] Feature extraction complete — 7-dim vector (100 customers)",
    "[00:00.412] [FM] Step 16: action=LOCK_SAME fleet_target=26 seed=same iters=500",
    "[00:00.838] [HGS] New best: NV=26 TD=28702.7 score=54702.7",
    "[00:01.108] [FM] Step 17: action=PUSH_NEW fleet_target=25 seed=new iters=1000",
    "[00:03.201] [HGS] Running step 17 / 50 — iteration 8500 / 25000..."
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"running"`, `"complete"`, `"error"`, `"not_found"` |
| `current_stage` | int | 1–4, which stage is actively running |
| `stage_name` | string | Human-readable name of current stage |
| `stage_statuses` | object | Per-stage: `"done"`, `"running"`, `"waiting"`, `"error"` — keys `"1"` through `"4"` |
| `stage_times_seconds` | object | How long each stage took in seconds (null if not finished) — keys `"1"` through `"4"` |
| `current_nv` | int | NV of the candidate solution in the current step |
| `best_nv` | int | Best NV found so far across all episode steps |
| `current_td` | float | TD of candidate in the current step |
| `best_td` | float | Best TD found so far |
| `current_score` | float | `1000 × current_nv + current_td` |
| `best_score` | float | Best `1000 × NV + TD` found so far |
| `iteration` | int | Cumulative HGS iteration count across all episode steps |
| `max_iterations` | int | Total iteration budget (approx. `episode_step_max × avg_iters_per_step`) |
| `elapsed_seconds` | int | Wall time since job started |
| `current_action` | string | **NEW** — Last Fleet Manager action: one of `FREE_SAME`, `FREE_NEW`, `LOCK_SAME`, `LOCK_NEW`, `PUSH_SAME`, `PUSH_NEW`, `FORCE_MIN`. Empty string before first step. |
| `episode_step` | int | **NEW** — Current RL episode step index (0 = initial solve, 1–50 = agent-driven steps) |
| `episode_step_max` | int | **NEW** — Fixed upper bound for episode steps (always `50`) |
| `nv_min` | int | Theoretical minimum fleet = `ceil(total_demand / capacity)` |
| `log_lines` | array | Last 10 log messages (newest last) |

**Stage names by number:**

| # | Name | Model | Notes |
|---|---|---|---|
| 1 | Feature Extractor | Hand-crafted (7-dim) | Deterministic; runs in milliseconds |
| 2 | Fleet Manager | Actor-Critic PPO | Selects action each episode step |
| 3 | HGS Engine | Hybrid Genetic Search | Runs 500–1500 iters per step × 50 steps |
| 4 | PPO Trainer | Reward Propagation | Updates Fleet Manager weights after episode |

> **Note:** The old 5-stage schema (`1`–`5`) is **replaced** by this 4-stage schema (`1`–`4`). Stage keys `"5"` no longer exist in the response.

**Error Response `404`**
```json
{ "error": "Job not found", "job_id": "a3f9c2b1" }
```

---

### ⚠️ Critical — How the `status` Field Controls the Entire App Flow

The `status` field in the `/status` response is not just informational — **it directly controls what the Flutter app does next.** The app polls this endpoint every 500ms and takes a specific action depending on the value:

| `status` value | What the Flutter app does |
|---|---|
| `"running"` | Keeps polling every 500ms, updates pipeline stages and live metrics on screen |
| `"complete"` | **Stops polling immediately. Navigates to the Solution Viewer. Calls `GET /result/{job_id}` to load the final routes.** |
| `"error"` | Stops polling. Shows an error state on the Solver Console. |
| `"stopped"` | Stops polling. Shows a stopped state (triggered by the user clicking Stop). |

**If you never return `"complete"`, the app polls forever and the Solution Viewer never loads.** This is the most common integration mistake — the backend solve loop finishes but the job status is never updated from `"running"` to `"complete"`.

### How to Implement the Status Transition in Python

The key principle: your background thread must update `jobs[job_id]["status"]` to `"complete"` **after** all 4 stages have finished and the result has been stored. Here is the exact pattern:

```python
import threading
from uuid import uuid4

jobs: dict[str, dict] = {}

def run_solver_background(job_id: str, vrp_file_path: str):
    """
    This runs in a background thread. The main thread returns 202 immediately.
    Updates jobs[job_id] as it progresses through the 4-stage pipeline.
    """
    try:
        # ── Stage 1: Feature Extraction ────────────────────────────────────
        jobs[job_id]["current_stage"] = 1
        jobs[job_id]["stage_statuses"]["1"] = "running"
        jobs[job_id]["log_lines"].append("[FE] Computing 7-dim instance features...")

        from src.solver_engine import _parse_vrp_file, _compute_instance_features
        hgs_data = _parse_vrp_file(vrp_file_path)
        instance_features = _compute_instance_features(hgs_data)

        jobs[job_id]["stage_statuses"]["1"] = "done"
        jobs[job_id]["stage_times_seconds"]["1"] = 0.02   # typically <50ms
        jobs[job_id]["log_lines"].append(
            f"[FE] Feature extraction complete — 7-dim vector "
            f"({hgs_data['num_customers']} customers)"
        )

        # ── Stage 2 + 3: Fleet Manager + HGS Engine (interleaved per step) ──
        # The Fleet Manager selects an action each step; HGS executes it.
        # Both stages run concurrently from the RL agent's perspective.
        jobs[job_id]["current_stage"] = 2
        jobs[job_id]["stage_statuses"]["2"] = "running"
        jobs[job_id]["stage_statuses"]["3"] = "running"

        from src.solver_engine import CVRPEnv, ACTION_NAMES
        from src.agent_manager import FleetManager
        import torch

        env = CVRPEnv(instance_paths=[vrp_file_path], max_steps=50)
        manager = FleetManager()
        # Load best trained weights
        ckpt = torch.load("logs/best_model.pth", map_location="cpu", weights_only=False)
        manager.load_state_dict(ckpt["manager_state_dict"])
        manager.eval()

        obs, info = env.reset()
        done = False
        total_iters = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = info.get("action_mask")
            mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0) if mask is not None else None

            with torch.no_grad():
                logits, _ = manager(obs_t[:, :7], obs_t[:, 7:], action_mask=mask_t)
                action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step = info["step"]
            action_name = ACTION_NAMES[action]
            total_iters += 500 if action < 4 else (1000 if action < 6 else 1500)

            # Write progress so the Flutter app sees it on next poll
            jobs[job_id]["current_action"] = action_name
            jobs[job_id]["episode_step"] = step
            jobs[job_id]["best_nv"] = info["nv"]
            jobs[job_id]["best_td"] = info["td"]
            jobs[job_id]["best_score"] = info["score"]
            jobs[job_id]["iteration"] = total_iters
            jobs[job_id]["log_lines"].append(
                f"[FM] Step {step}: action={action_name} → NV={info['nv']} score={info['score']:.0f}"
            )
            jobs[job_id]["log_lines"] = jobs[job_id]["log_lines"][-10:]

        jobs[job_id]["stage_statuses"]["2"] = "done"
        jobs[job_id]["stage_statuses"]["3"] = "done"
        jobs[job_id]["stage_times_seconds"]["2"] = 0.5
        jobs[job_id]["stage_times_seconds"]["3"] = elapsed_so_far  # your timer

        # ── Stage 4: PPO Trainer (marks training update complete) ─────────
        # During inference/competition mode this stage is a no-op — the
        # model was already trained offline. Mark it instantly as done.
        jobs[job_id]["current_stage"] = 4
        jobs[job_id]["stage_statuses"]["4"] = "running"
        jobs[job_id]["log_lines"].append("[PPO] Inference mode — policy weights frozen")
        # (In training mode: run ppo_update() here and save the new checkpoint)
        jobs[job_id]["stage_statuses"]["4"] = "done"
        jobs[job_id]["stage_times_seconds"]["4"] = 0.0

        # ── Store final result ─────────────────────────────────────────────
        jobs[job_id]["result"] = {
            "num_vehicles": info["nv"],
            "total_distance": info["td"],
            "score": info["score"],
            "routes": [],    # fill from env._best_routes if tracked
        }

        # ── CRITICAL: flip status to "complete" ───────────────────────────
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["log_lines"].append("[✓] Solve complete")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log_lines"].append(f"[ERROR] {e}")
```

### Progress reporting hook (minimal version)

```python
def on_step(job_id: str, info: dict, action_name: str, total_iters: int):
    """Call this after every CVRPEnv.step() to keep the Flutter app updated."""
    jobs[job_id]["current_action"] = action_name
    jobs[job_id]["episode_step"]   = info["step"]
    jobs[job_id]["best_nv"]        = info["nv"]
    jobs[job_id]["best_td"]        = info["td"]
    jobs[job_id]["best_score"]     = info["score"]
    jobs[job_id]["iteration"]      = total_iters
    jobs[job_id]["log_lines"].append(
        f"[FM] Step {info['step']}: {action_name} → NV={info['nv']} score={info['score']:.0f}"
    )
    jobs[job_id]["log_lines"] = jobs[job_id]["log_lines"][-10:]  # keep last 10
```

### Initial job state (when POST /solve is called)

```python
job_id = uuid4().hex[:8]
jobs[job_id] = {
    "status":              "running",
    "current_stage":       1,
    "stage_statuses":      {"1": "waiting", "2": "waiting", "3": "waiting", "4": "waiting"},
    "stage_times_seconds": {"1": None,      "2": None,      "3": None,      "4": None},
    "current_nv":          None,
    "best_nv":             None,
    "current_td":          None,
    "best_td":             None,
    "current_score":       None,
    "best_score":          None,
    "iteration":           0,
    "max_iterations":      25000,   # 50 steps × ~500 avg iters
    "current_action":      "",
    "episode_step":        0,
    "episode_step_max":    50,
    "nv_min":              nv_min,  # parsed from .vrp at upload time
    "log_lines":           [],
    "result":              None,
    "instance_path":       vrp_file_path,
    "instance_name":       instance_name,
    "num_nodes":           num_nodes,
}
```

---

## Endpoint 4 — Fetch Final Result

```
GET /result/{job_id}
```

Called once when `/status` returns `"status": "complete"`. Returns the full solution for the Solution Viewer screen.

**Response `200`**
```json
{
  "job_id": "a3f9c2b1",
  "instance_name": "X-n101-k25",
  "num_nodes": 101,
  "num_vehicles": 26,
  "total_distance": 28702.7,
  "score": 54702.7,
  "nv_min": 25,
  "solve_time_seconds": 106,
  "depot": { "id": 1, "x": 365.0, "y": 689.0 },
  "customers": [
    { "id": 2,   "x": 146.0, "y": 180.0, "demand": 38  },
    { "id": 3,   "x": 792.0, "y": 5.0,   "demand": 51  },
    { "id": 4,   "x": 658.0, "y": 510.0, "demand": 73  },
    { "id": 5,   "x": 461.0, "y": 270.0, "demand": 70  },
    { "id": 6,   "x": 299.0, "y": 531.0, "demand": 58  },
    { "id": 7,   "x": 812.0, "y": 228.0, "demand": 54  },
    { "id": 8,   "x": 643.0, "y": 90.0,  "demand": 1   },
    { "id": 9,   "x": 615.0, "y": 630.0, "demand": 98  },
    { "id": 10,  "x": 258.0, "y": 42.0,  "demand": 62  },
    { "id": 11,  "x": 616.0, "y": 299.0, "demand": 98  },
    { "id": 12,  "x": 475.0, "y": 957.0, "demand": 25  },
    { "id": 13,  "x": 425.0, "y": 473.0, "demand": 86  },
    { "id": 14,  "x": 406.0, "y": 64.0,  "demand": 46  },
    { "id": 15,  "x": 656.0, "y": 369.0, "demand": 27  },
    { "id": 16,  "x": 202.0, "y": 467.0, "demand": 17  },
    { "id": 17,  "x": 318.0, "y": 21.0,  "demand": 97  },
    { "id": 18,  "x": 579.0, "y": 587.0, "demand": 74  },
    { "id": 19,  "x": 458.0, "y": 354.0, "demand": 81  },
    { "id": 20,  "x": 575.0, "y": 871.0, "demand": 62  },
    { "id": 21,  "x": 47.0,  "y": 512.0, "demand": 59  },
    { "id": 22,  "x": 568.0, "y": 742.0, "demand": 23  },
    { "id": 23,  "x": 128.0, "y": 436.0, "demand": 62  },
    { "id": 24,  "x": 546.0, "y": 806.0, "demand": 66  },
    { "id": 25,  "x": 197.0, "y": 696.0, "demand": 35  },
    { "id": 26,  "x": 615.0, "y": 300.0, "demand": 53  },
    { "id": 27,  "x": 852.0, "y": 563.0, "demand": 18  },
    { "id": 28,  "x": 772.0, "y": 803.0, "demand": 87  },
    { "id": 29,  "x": 678.0, "y": 342.0, "demand": 32  },
    { "id": 30,  "x": 916.0, "y": 176.0, "demand": 4   },
    { "id": 31,  "x": 390.0, "y": 949.0, "demand": 61  },
    { "id": 32,  "x": 113.0, "y": 782.0, "demand": 95  },
    { "id": 33,  "x": 226.0, "y": 736.0, "demand": 23  },
    { "id": 34,  "x": 119.0, "y": 923.0, "demand": 15  },
    { "id": 35,  "x": 584.0, "y": 572.0, "demand": 5   },
    { "id": 36,  "x": 134.0, "y": 554.0, "demand": 53  },
    { "id": 37,  "x": 912.0, "y": 173.0, "demand": 97  },
    { "id": 38,  "x": 827.0, "y": 233.0, "demand": 70  },
    { "id": 39,  "x": 851.0, "y": 677.0, "demand": 32  },
    { "id": 40,  "x": 598.0, "y": 322.0, "demand": 27  },
    { "id": 41,  "x": 627.0, "y": 472.0, "demand": 42  },
    { "id": 42,  "x": 94.0,  "y": 442.0, "demand": 67  },
    { "id": 43,  "x": 688.0, "y": 274.0, "demand": 76  },
    { "id": 44,  "x": 977.0, "y": 176.0, "demand": 15  },
    { "id": 45,  "x": 597.0, "y": 461.0, "demand": 39  },
    { "id": 46,  "x": 931.0, "y": 23.0,  "demand": 14  },
    { "id": 47,  "x": 170.0, "y": 640.0, "demand": 43  },
    { "id": 48,  "x": 941.0, "y": 601.0, "demand": 11  },
    { "id": 49,  "x": 873.0, "y": 487.0, "demand": 93  },
    { "id": 50,  "x": 797.0, "y": 95.0,  "demand": 53  },
    { "id": 51,  "x": 451.0, "y": 816.0, "demand": 44  },
    { "id": 52,  "x": 866.0, "y": 970.0, "demand": 80  },
    { "id": 53,  "x": 833.0, "y": 912.0, "demand": 87  },
    { "id": 54,  "x": 106.0, "y": 913.0, "demand": 97  },
    { "id": 55,  "x": 260.0, "y": 107.0, "demand": 67  },
    { "id": 56,  "x": 332.0, "y": 45.0,  "demand": 72  },
    { "id": 57,  "x": 685.0, "y": 613.0, "demand": 50  },
    { "id": 58,  "x": 728.0, "y": 372.0, "demand": 8   },
    { "id": 59,  "x": 487.0, "y": 497.0, "demand": 58  },
    { "id": 60,  "x": 702.0, "y": 440.0, "demand": 55  },
    { "id": 61,  "x": 717.0, "y": 412.0, "demand": 67  },
    { "id": 62,  "x": 635.0, "y": 794.0, "demand": 89  },
    { "id": 63,  "x": 927.0, "y": 972.0, "demand": 38  },
    { "id": 64,  "x": 635.0, "y": 356.0, "demand": 65  },
    { "id": 65,  "x": 634.0, "y": 540.0, "demand": 3   },
    { "id": 66,  "x": 658.0, "y": 261.0, "demand": 5   },
    { "id": 67,  "x": 303.0, "y": 168.0, "demand": 46  },
    { "id": 68,  "x": 707.0, "y": 410.0, "demand": 100 },
    { "id": 69,  "x": 254.0, "y": 135.0, "demand": 52  },
    { "id": 70,  "x": 346.0, "y": 29.0,  "demand": 28  },
    { "id": 71,  "x": 75.0,  "y": 79.0,  "demand": 96  },
    { "id": 72,  "x": 893.0, "y": 987.0, "demand": 18  },
    { "id": 73,  "x": 729.0, "y": 372.0, "demand": 16  },
    { "id": 74,  "x": 29.0,  "y": 910.0, "demand": 7   },
    { "id": 75,  "x": 356.0, "y": 39.0,  "demand": 73  },
    { "id": 76,  "x": 274.0, "y": 943.0, "demand": 76  },
    { "id": 77,  "x": 322.0, "y": 96.0,  "demand": 6   },
    { "id": 78,  "x": 664.0, "y": 396.0, "demand": 64  },
    { "id": 79,  "x": 704.0, "y": 236.0, "demand": 39  },
    { "id": 80,  "x": 415.0, "y": 837.0, "demand": 86  },
    { "id": 81,  "x": 576.0, "y": 587.0, "demand": 70  },
    { "id": 82,  "x": 750.0, "y": 977.0, "demand": 14  },
    { "id": 83,  "x": 726.0, "y": 363.0, "demand": 83  },
    { "id": 84,  "x": 861.0, "y": 948.0, "demand": 96  },
    { "id": 85,  "x": 302.0, "y": 129.0, "demand": 43  },
    { "id": 86,  "x": 415.0, "y": 989.0, "demand": 12  },
    { "id": 87,  "x": 199.0, "y": 135.0, "demand": 73  },
    { "id": 88,  "x": 801.0, "y": 405.0, "demand": 2   },
    { "id": 89,  "x": 679.0, "y": 426.0, "demand": 21  },
    { "id": 90,  "x": 994.0, "y": 804.0, "demand": 18  },
    { "id": 91,  "x": 311.0, "y": 116.0, "demand": 55  },
    { "id": 92,  "x": 739.0, "y": 898.0, "demand": 75  },
    { "id": 93,  "x": 268.0, "y": 97.0,  "demand": 68  },
    { "id": 94,  "x": 176.0, "y": 991.0, "demand": 100 },
    { "id": 95,  "x": 688.0, "y": 588.0, "demand": 61  },
    { "id": 96,  "x": 107.0, "y": 836.0, "demand": 24  },
    { "id": 97,  "x": 708.0, "y": 522.0, "demand": 40  },
    { "id": 98,  "x": 679.0, "y": 864.0, "demand": 48  },
    { "id": 99,  "x": 985.0, "y": 877.0, "demand": 51  },
    { "id": 100, "x": 954.0, "y": 950.0, "demand": 78  },
    { "id": 101, "x": 615.0, "y": 750.0, "demand": 35  }
  ],
  "routes": [
    { "route_id": 1,  "customer_ids": [68, 94, 8, 88],          "num_stops": 4, "distance": 1420.3, "total_load": 203, "capacity": 206, "capacity_pct": 98.5  },
    { "route_id": 2,  "customer_ids": [9, 11, 74],               "num_stops": 3, "distance": 890.4,  "total_load": 203, "capacity": 206, "capacity_pct": 98.5  },
    { "route_id": 3,  "customer_ids": [17, 37, 77],              "num_stops": 3, "distance": 1180.6, "total_load": 200, "capacity": 206, "capacity_pct": 97.1  },
    { "route_id": 4,  "customer_ids": [54, 32, 82],              "num_stops": 3, "distance": 680.2,  "total_load": 206, "capacity": 206, "capacity_pct": 100.0 },
    { "route_id": 5,  "customer_ids": [84, 71, 86],              "num_stops": 3, "distance": 1320.8, "total_load": 204, "capacity": 206, "capacity_pct": 99.0  },
    { "route_id": 6,  "customer_ids": [49, 13, 34, 66],          "num_stops": 4, "distance": 1050.4, "total_load": 199, "capacity": 206, "capacity_pct": 96.6  },
    { "route_id": 7,  "customer_ids": [80, 4, 85, 65],           "num_stops": 4, "distance": 920.5,  "total_load": 205, "capacity": 206, "capacity_pct": 99.5  },
    { "route_id": 8,  "customer_ids": [28, 62, 70],              "num_stops": 3, "distance": 1280.7, "total_load": 204, "capacity": 206, "capacity_pct": 99.0  },
    { "route_id": 9,  "customer_ids": [53, 52, 63],              "num_stops": 3, "distance": 1380.4, "total_load": 205, "capacity": 206, "capacity_pct": 99.5  },
    { "route_id": 10, "customer_ids": [19, 5, 50],               "num_stops": 3, "distance": 780.3,  "total_load": 204, "capacity": 206, "capacity_pct": 99.0  },
    { "route_id": 11, "customer_ids": [18, 81, 20],              "num_stops": 3, "distance": 1050.6, "total_load": 206, "capacity": 206, "capacity_pct": 100.0 },
    { "route_id": 12, "customer_ids": [55, 93, 42],              "num_stops": 3, "distance": 940.5,  "total_load": 202, "capacity": 206, "capacity_pct": 98.1  },
    { "route_id": 13, "customer_ids": [61, 60, 91, 40],          "num_stops": 4, "distance": 1180.4, "total_load": 204, "capacity": 206, "capacity_pct": 99.0  },
    { "route_id": 14, "customer_ids": [64, 78, 29, 58, 15],      "num_stops": 5, "distance": 780.3,  "total_load": 196, "capacity": 206, "capacity_pct": 95.1  },
    { "route_id": 15, "customer_ids": [56, 76, 59],              "num_stops": 3, "distance": 1250.6, "total_load": 206, "capacity": 206, "capacity_pct": 100.0 },
    { "route_id": 16, "customer_ids": [100, 92, 26],             "num_stops": 3, "distance": 1420.7, "total_load": 206, "capacity": 206, "capacity_pct": 100.0 },
    { "route_id": 17, "customer_ids": [83, 43, 45, 30],          "num_stops": 4, "distance": 1080.4, "total_load": 202, "capacity": 206, "capacity_pct": 98.1  },
    { "route_id": 18, "customer_ids": [57, 97, 95, 101, 27],     "num_stops": 5, "distance": 980.6,  "total_load": 204, "capacity": 206, "capacity_pct": 99.0  },
    { "route_id": 19, "customer_ids": [47, 2, 69, 14, 22],       "num_stops": 5, "distance": 1150.3, "total_load": 202, "capacity": 206, "capacity_pct": 98.1  },
    { "route_id": 20, "customer_ids": [41, 67, 10, 33, 44, 48],  "num_stops": 6, "distance": 1280.7, "total_load": 199, "capacity": 206, "capacity_pct": 96.6  },
    { "route_id": 21, "customer_ids": [87, 75, 79, 89],          "num_stops": 4, "distance": 1040.5, "total_load": 206, "capacity": 206, "capacity_pct": 100.0 },
    { "route_id": 22, "customer_ids": [6, 23, 21, 16, 35],       "num_stops": 5, "distance": 1350.6, "total_load": 201, "capacity": 206, "capacity_pct": 97.6  },
    { "route_id": 23, "customer_ids": [38, 7, 39, 46, 25],       "num_stops": 5, "distance": 1380.2, "total_load": 205, "capacity": 206, "capacity_pct": 99.5  },
    { "route_id": 24, "customer_ids": [24, 31, 51, 90, 73],      "num_stops": 5, "distance": 1620.5, "total_load": 205, "capacity": 206, "capacity_pct": 99.5  },
    { "route_id": 25, "customer_ids": [3, 98, 12, 72, 96],       "num_stops": 5, "distance": 830.4,  "total_load": 166, "capacity": 206, "capacity_pct": 80.6  },
    { "route_id": 26, "customer_ids": [36, 99],                  "num_stops": 2, "distance": 460.8,  "total_load": 104, "capacity": 206, "capacity_pct": 50.5  }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `num_vehicles` | int | Final number of routes/vehicles used |
| `total_distance` | float | Sum of all route distances |
| `score` | float | `1000 × num_vehicles + total_distance` |
| `depot` | object | Depot node with id, x, y |
| `customers` | array | All customer nodes with id, x, y, demand — used to draw the canvas |
| `routes[].customer_ids` | array | Ordered list of customer node IDs in visit sequence |
| `routes[].distance` | float | Total length of this specific route |
| `routes[].capacity_pct` | float | `(total_load / capacity) × 100` |

> **Note on coordinates:** The canvas in the Solution Viewer draws nodes using the raw `x, y` values from this response. Send the original (un-normalized) coordinates from the `.vrp` file, not the `[0,1]` normalized ones used internally by the GNN.

**Error Response `404`** — job not found  
**Error Response `425`** — job still running (use `/status` instead)

---

## Endpoint 5 — Recent Runs

```
GET /runs?limit=N
```

Called by the **Dashboard** on load and every 30 seconds. Returns a list of completed and running jobs.

Populates the **Recent Runs** table on the Dashboard screen and the score trend chart.

**Query Parameters (optional)**

| Param | Default | Description |
|---|---|---|
| `limit` | 20 | Max number of runs to return |
| `offset` | 0 | Pagination offset |

**Response `200`**
```json
{
  "total": 4,
  "runs": [
    {
      "job_id": "a3f9c2b1",
      "instance_name": "X-n101-k25",
      "num_vehicles": 26,
      "total_distance": 28702.7,
      "score": 54702.7,
      "solve_time_seconds": 106,
      "completed_at": "2026-03-27T14:32:01Z",
      "status": "complete"
    },
    {
      "job_id": "b8d1e4f2",
      "instance_name": "X-n101-k25",
      "num_vehicles": 27,
      "total_distance": 29841.3,
      "score": 56841.3,
      "solve_time_seconds": 60,
      "completed_at": "2026-03-27T13:18:44Z",
      "status": "complete"
    },
    {
      "job_id": "c7e2a5d3",
      "instance_name": "X-n101-k25",
      "num_vehicles": 28,
      "total_distance": 31204.8,
      "score": 59204.8,
      "solve_time_seconds": 42,
      "completed_at": "2026-03-27T12:05:11Z",
      "status": "complete"
    },
    {
      "job_id": "d4f6b1c8",
      "instance_name": "X-n101-k25",
      "num_vehicles": 30,
      "total_distance": 33512.1,
      "score": 63512.1,
      "solve_time_seconds": 31,
      "completed_at": "2026-03-26T18:47:30Z",
      "status": "running"
    }
  ]
}
```

> The web app reads `runs` in order and uses `score` values to draw the trend line chart on the dashboard.

---

## Endpoint 6 — Stop a Running Job

```
POST /stop/{job_id}
```

Called when the user clicks the **Stop** button in the Solver Console. Should gracefully terminate the background job and return the best solution found so far.

**Response `200`**
```json
{
  "job_id": "a3f9c2b1",
  "status": "stopped",
  "best_nv": 26,
  "best_td": 28702.7,
  "best_score": 54702.7
}
```

---

## Endpoint 7 — Benchmark Comparison

```
GET /benchmark/{job_id}
```

Called by the **Benchmark page** after a solve job completes. Runs PyVRP Default and PyVRP Large Population on the **same instance file** that was already uploaded for the RL solve job (identified by `job_id`). Returns a 3-way comparison.

**Why this endpoint, not a new file upload?**
The `.vrp` file is already on the server from the original `POST /solve` call. Reusing `job_id` avoids uploading the file twice and guarantees both algorithms are tested on the identical input.

**When to call it:**
The Flutter app calls this once after `GET /status/{job_id}` returns `"status": "complete"`. The benchmark computation runs in the background and may take a few seconds. Return HTTP 425 while still computing.

**Response `200 OK`**
```json
{
  "job_id": "a3f9c2b1",
  "instance_name": "X-n101-k25",
  "num_nodes": 101,
  "rl": {
    "nv": 26,
    "td": 28702.7,
    "score": 54702.7,
    "solve_time_seconds": 106
  },
  "hgs_default": {
    "nv": 29,
    "td": 31840.5,
    "score": 60840.5,
    "solve_time_seconds": 60
  },
  "hgs_large_pop": {
    "nv": 28,
    "td": 30614.2,
    "score": 58614.2,
    "solve_time_seconds": 120
  }
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string | Same job_id from the original solve |
| `instance_name` | string | Instance filename |
| `num_nodes` | int | Total nodes including depot |
| `rl.nv` | int | Vehicles used by the RL agent — **taken directly from the completed solve job** (no re-run needed) |
| `rl.td` | float | Total distance from the RL solution |
| `rl.score` | float | `1000 × rl.nv + rl.td` |
| `hgs_default.nv` | int | Vehicles used by PyVRP with `AlgorithmParameters()` factory defaults |
| `hgs_default.td` | float | Total distance from PyVRP default |
| `hgs_default.score` | float | `1000 × hgs_default.nv + hgs_default.td` |
| `hgs_large_pop.nv` | int | Vehicles used by PyVRP with `mu=50, lambda=80` |
| `hgs_large_pop.td` | float | Total distance from large population run |
| `hgs_large_pop.score` | float | `1000 × hgs_large_pop.nv + hgs_large_pop.td` |

**Response `425 Too Early`** — baselines still computing
```json
{ "status": "computing", "job_id": "a3f9c2b1" }
```

**Response `404 Not Found`**
```json
{ "error": "Job not found", "job_id": "a3f9c2b1" }
```

---

## Log Line Format Reference

Log lines in `log_lines` should follow this format so the Flutter console colours them correctly:

| Prefix | Stage | Colour in UI |
|---|---|---|
| `[FE]` | Feature Extractor | Green |
| `[FM]` | Fleet Manager | Cyan |
| `[HGS]` | HGS Engine | Amber |
| `[PPO]` | PPO Trainer | Purple |

Example set of log lines for a typical solve step:
```
"[FE] Feature extraction complete — 7-dim vector (100 customers)"
"[FM] Step 1: action=FREE_SAME fleet_target=None iters=500"
"[HGS] Initial solve: NV=29 TD=31241.8 score=60241.8"
"[FM] Step 2: action=PUSH_NEW fleet_target=26 seed=new iters=1000"
"[HGS] New best: NV=26 TD=28702.7 score=54702.7"
"[FM] Step 17: action=LOCK_SAME fleet_target=26 seed=same iters=500"
"[PPO] Inference mode — policy weights frozen"
```

> **Removed prefixes:** `[GNN]` is no longer used — the GNN Observer stage does not exist in the current architecture.

---

## Summary Table

| # | Method | Endpoint | Used By | When |
|---|---|---|---|---|
| 1 | `GET` | `/health` | Sidebar dot, Topbar indicator | App startup + every 15s |
| 2 | `POST` | `/solve` | Solver Console — Run button | User clicks Run |
| 3 | `GET` | `/status/{job_id}` | Solver Console — live monitor | Every 500ms while running |
| 4 | `GET` | `/result/{job_id}` | Solution Viewer | Once job completes |
| 5 | `GET` | `/runs` | Dashboard table + chart | Dashboard load + every 30s |
| 6 | `POST` | `/stop/{job_id}` | Solver Console — Stop button | User clicks Stop |
| **7** | **`GET`** | **`/benchmark/{job_id}`** | **Benchmark page** | **After solve completes** |

---

## Implementation Notes for the Backend Dev

### Where each response field comes from in the Python code

| API Field | Source in Python Code |
|---|---|
| `nv_min` | `CVRPEnv._compute_nv_min()` in `solver_engine.py` |
| `current_nv` / `best_nv` | `info["nv"]` returned from `CVRPEnv.step()` |
| `current_td` / `best_td` | `info["td"]` returned from `CVRPEnv.step()` |
| `best_score` | `info["score"]` = `competition_score(nv, td)` = `1000 * nv + td` |
| `current_action` | `ACTION_NAMES[action]` from `solver_engine.py` |
| `episode_step` | `info["step"]` returned from `CVRPEnv.step()` |
| `episode_step_max` | `CVRPEnv.max_steps` — always 50 |
| `iteration` | Cumulative sum of `iters_used` per step (500 / 1000 / 1500) |
| `max_iterations` | `episode_step_max × avg_iters` ≈ 50 × 500 = 25,000 (approximate) |
| `stage_statuses` | Track manually as you call stages 1→4 sequentially |
| `stage_times_seconds` | Wall-clock time between stage start and end |
| `log_lines` | Append strings to a list as each stage/step runs |

### Action → iteration budget mapping

| Action | `num_vehicles` | `nb_iter` |
|---|---|---|
| `FREE_SAME` (0) | `None` (unconstrained) | 500 |
| `FREE_NEW` (1) | `None` | 500 |
| `LOCK_SAME` (2) | `best_nv` | 500 |
| `LOCK_NEW` (3) | `best_nv` | 500 |
| `PUSH_SAME` (4) | `best_nv − 1` | 1000 |
| `PUSH_NEW` (5) | `best_nv − 1` | 1000 |
| `FORCE_MIN` (6) | `nv_min` | 1500 |

---

*ML4VRP 2026 — GECCO Competition · API Spec v3.0 (4-stage architecture)*
