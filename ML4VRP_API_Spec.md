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
    "gnn_observer":   1.00,
    "fleet_manager":  1.00,
    "hgs_engine":     1.00,
    "route_driver":   0.88,
    "maca_trainer":   0.61
  },
  "total_training_epochs": 84,
  "best_score_ever": 15821.3
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ready"` or `"loading"` or `"error"` |
| `device` | string | `"cuda"` or `"cpu"` |
| `gpu_name` | string | GPU name, or `"CPU only"` |
| `model_loaded` | bool | Whether GNNEncoder + agents are loaded |
| `stage_health.gnn_observer` | float 0–1 | GNN encoder training convergence. Use `1.0` once training is complete. |
| `stage_health.fleet_manager` | float 0–1 | Fleet Manager PPO agent convergence ratio (e.g. current epoch / total epochs). |
| `stage_health.hgs_engine` | float 0–1 | Always `1.0` — HGS is a classical solver, not trained. |
| `stage_health.route_driver` | float 0–1 | Route Driver PPO agent convergence ratio. |
| `stage_health.maca_trainer` | float 0–1 | Reward propagation training progress. |
| `total_training_epochs` | int | How many PPO training epochs have completed so far. |
| `best_score_ever` | float | The lowest `1000×NV + TD` score achieved across all training runs. |


---

## Endpoint 2 — Start a Solver Job

```
POST /solve
Content-Type: multipart/form-data
```

Called when the user uploads a `.vrp` file and clicks **Run Solver**. Starts the 5-stage pipeline as a background job and immediately returns a `job_id`.

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
  "instance_name": "X-n303-k46",
  "num_nodes": 303,
  "vehicle_capacity": 1000,
  "total_demand": 28450,
  "nv_min": 29
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string | Unique ID — used to poll `/status` and `/result` |
| `instance_name` | string | Parsed from the `.vrp` filename |
| `num_nodes` | int | Total nodes including depot (from `DIMENSION` in file) |
| `vehicle_capacity` | int | From `CAPACITY` in the `.vrp` file |
| `total_demand` | int | Sum of all customer demands |
| `nv_min` | int | `ceil(total_demand / vehicle_capacity)` — theoretical floor |

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
    "4": "waiting",
    "5": "waiting"
  },
  "stage_times_seconds": {
    "1": 0.3,
    "2": 1.2,
    "3": null,
    "4": null,
    "5": null
  },
  "current_nv": 46,
  "best_nv": 44,
  "current_td": 7903.2,
  "best_td": 7821.5,
  "current_score": 53903.2,
  "best_score": 51821.5,
  "iteration": 3412,
  "max_iterations": 10000,
  "elapsed_seconds": 48,
  "log_lines": [
    "[00:00.312] [GNN] Graph embedding complete — 128-dim vector",
    "[00:01.521] [FM] Strategic mode: ROUTE_ELIMINATION",
    "[00:01.523] [HGS] Crossover SREX initiated — pop size 20",
    "[00:04.201] [HGS] New best: NV=44 TD=7821.5 score=51821.5",
    "[00:04.205] [HGS] Running iteration 3412 / 10000..."
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"running"`, `"complete"`, `"error"`, `"not_found"` |
| `current_stage` | int | 1–5, which stage is actively running |
| `stage_name` | string | Human-readable name of current stage |
| `stage_statuses` | object | Per-stage: `"done"`, `"running"`, `"waiting"`, `"error"` |
| `stage_times_seconds` | object | How long each stage took (null if not finished) |
| `current_nv` | int | NV of the current solution being evaluated |
| `best_nv` | int | Best NV found so far |
| `current_td` | float | TD of current solution |
| `best_td` | float | Best TD found so far |
| `current_score` | float | `1000 × current_nv + current_td` |
| `best_score` | float | Best `1000 × NV + TD` found so far |
| `iteration` | int | Current HGS iteration count |
| `max_iterations` | int | Total iteration budget |
| `elapsed_seconds` | int | Wall time since job started |
| `log_lines` | array | Last 10 log messages (newest last) |

**Stage names by number:**

| # | Name |
|---|---|
| 1 | GNN Observer |
| 2 | Fleet Manager |
| 3 | HGS Engine |
| 4 | Route Driver |
| 5 | MACA Trainer |

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

The key principle: your background thread must update `jobs[job_id]["status"]` to `"complete"` **after** all 5 stages have finished and the result has been stored. Here is the exact pattern:

```python
import threading
from uuid import uuid4

jobs: dict[str, dict] = {}

def run_solver_background(job_id: str, vrp_file_path: str):
    """
    This runs in a background thread. The main thread returns 202 immediately.
    This function updates jobs[job_id] as it progresses through all 5 stages.
    """
    try:
        # ── Stage 1: GNN Observer ─────────────────────────────────────────
        jobs[job_id]["current_stage"] = 1
        jobs[job_id]["stage_statuses"]["1"] = "running"
        jobs[job_id]["log_lines"].append("[GNN] Starting graph embedding...")

        # ... your actual GNN code here ...
        embedding = gnn_encoder.encode(vrp_file_path)

        jobs[job_id]["stage_statuses"]["1"] = "done"
        jobs[job_id]["stage_times_seconds"]["1"] = 0.3
        jobs[job_id]["log_lines"].append("[GNN] Graph embedding complete — 128-dim vector")

        # ── Stage 2: Fleet Manager ────────────────────────────────────────
        jobs[job_id]["current_stage"] = 2
        jobs[job_id]["stage_statuses"]["2"] = "running"

        # ... your Fleet Manager PPO code here ...

        jobs[job_id]["stage_statuses"]["2"] = "done"
        jobs[job_id]["stage_times_seconds"]["2"] = 1.2

        # ── Stage 3: HGS Engine ───────────────────────────────────────────
        jobs[job_id]["current_stage"] = 3
        jobs[job_id]["stage_statuses"]["3"] = "running"

        # ... your HGS solve loop here ...
        # Inside the loop, update metrics so the frontend sees live progress:
        jobs[job_id]["current_nv"] = info["nv"]
        jobs[job_id]["best_nv"]    = info["best_nv"]
        jobs[job_id]["current_td"] = info["td"]
        jobs[job_id]["best_td"]    = info["best_td"]
        jobs[job_id]["iteration"]  = info["total_iters"]
        jobs[job_id]["log_lines"].append(f"[HGS] Iteration {info['total_iters']} / 10000")
        jobs[job_id]["log_lines"] = jobs[job_id]["log_lines"][-10:]  # keep last 10

        jobs[job_id]["stage_statuses"]["3"] = "done"

        # ── Stages 4 & 5: Route Driver + MACA ────────────────────────────
        # ... repeat the same pattern for stages 4 and 5 ...

        jobs[job_id]["stage_statuses"]["4"] = "done"
        jobs[job_id]["stage_statuses"]["5"] = "done"

        # ── CRITICAL STEP: Store the final result BEFORE marking complete ─
        # The Flutter app will immediately call GET /result/{job_id}
        # as soon as it sees "complete". If result is not stored yet,
        # that call will fail. Always store result first, then set complete.
        jobs[job_id]["result"] = build_result_payload(job_id, solution, data)

        # ── CRITICAL STEP: Mark the job as complete ───────────────────────
        # This is the line that tells Flutter to stop polling and navigate
        # to the Solution Viewer. Without this line, the app polls forever.
        jobs[job_id]["status"] = "complete"                          # ← DO NOT FORGET THIS
        jobs[job_id]["log_lines"].append("[✓] Solve complete")

    except Exception as e:
        # If anything goes wrong, set "error" so the frontend stops polling
        # and shows an error state instead of spinning forever.
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log_lines"].append(f"[ERROR] {str(e)}")


@app.post("/solve")
async def solve(file: UploadFile, track: str = "cvrp", mode: str = "competition"):
    job_id = uuid4().hex[:8]

    # Initialise the job dict — status starts as "running"
    jobs[job_id] = {
        "status": "running",          # ← starts as running
        "current_stage": 1,
        "stage_statuses": {"1": "waiting", "2": "waiting", "3": "waiting", "4": "waiting", "5": "waiting"},
        "stage_times_seconds": {"1": None, "2": None, "3": None, "4": None, "5": None},
        "current_nv": 0, "best_nv": 0,
        "current_td": 0.0, "best_td": 0.0,
        "current_score": 0.0, "best_score": 0.0,
        "iteration": 0, "max_iterations": 10000,
        "elapsed_seconds": 0,
        "log_lines": [],
        "result": None,
    }

    # Save file to disk so the background thread can read it
    vrp_path = f"/tmp/{job_id}.vrp"
    with open(vrp_path, "wb") as f:
        f.write(await file.read())

    # Launch background thread — returns 202 immediately while solve runs
    thread = threading.Thread(
        target=run_solver_background,
        args=(job_id, vrp_path),
        daemon=True,
    )
    thread.start()

    # Return immediately with the job_id
    return JSONResponse(status_code=202, content={
        "job_id": job_id,
        "instance_name": file.filename.replace(".vrp", ""),
        # parse num_nodes, vehicle_capacity, total_demand, nv_min from the file
        ...
    })
```

### The Complete Status Lifecycle

```
POST /solve called
       │
       ▼
jobs[job_id]["status"] = "running"   ← set at job creation
       │
       ▼  (background thread works through stages 1→5)
       │
       ▼
jobs[job_id]["result"] = { ... }     ← STORE RESULT FIRST
       │
       ▼
jobs[job_id]["status"] = "complete"  ← THEN mark complete
       │
       ▼  (Flutter sees "complete" on next 500ms poll)
       │
       ▼
Flutter stops polling
Flutter calls GET /result/{job_id}
Flutter navigates to Solution Viewer ✓
```

**The order matters:** always store the result object before setting status to `"complete"`. If you set `"complete"` first, the Flutter app will immediately call `GET /result/{job_id}` and find `result: None`.

---

## Endpoint 4 — Get Final Result

```
GET /result/{job_id}
```

Called once when `/status` returns `"status": "complete"`. Returns the full solution for the Solution Viewer screen.

**Response `200`**
```json
{
  "job_id": "a3f9c2b1",
  "instance_name": "X-n303-k46",
  "num_nodes": 303,
  "num_vehicles": 11,
  "total_distance": 4821.3,
  "score": 15821.3,
  "nv_min": 9,
  "solve_time_seconds": 124,
  "depot": { "id": 1, "x": 50.0, "y": 50.0 },
  "customers": [
    { "id": 2, "x": 23.4, "y": 67.1, "demand": 120 },
    { "id": 3, "x": 81.2, "y": 44.8, "demand": 95 }
  ],
  "routes": [
    {
      "route_id": 1,
      "customer_ids": [3, 1, 2],
      "num_stops": 18,
      "distance": 421.2,
      "total_load": 940,
      "capacity": 1000,
      "capacity_pct": 94.0
    },
    {
      "route_id": 2,
      "customer_ids": [6, 5, 4],
      "num_stops": 22,
      "distance": 388.7,
      "total_load": 880,
      "capacity": 1000,
      "capacity_pct": 88.0
    }
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

## Endpoint 5 — List Past Runs

```
GET /runs
```

Populates the **Recent Runs** table on the Dashboard screen and the score trend chart.

**Query Parameters (optional)**

| Param | Default | Description |
|---|---|---|
| `limit` | 20 | Max number of runs to return |
| `offset` | 0 | Pagination offset |

**Response `200`**
```json
{
  "total": 5,
  "runs": [
    {
      "job_id": "a3f9c2b1",
      "instance_name": "X-n303-k46",
      "num_vehicles": 11,
      "total_distance": 4821.3,
      "score": 15821.3,
      "solve_time_seconds": 124,
      "completed_at": "2026-03-25T14:32:01Z",
      "status": "complete"
    },
    {
      "job_id": "b8d1e4f2",
      "instance_name": "X-n200-k36",
      "num_vehicles": 34,
      "total_distance": 5512.3,
      "score": 39512.3,
      "solve_time_seconds": 72,
      "completed_at": "2026-03-25T13:18:44Z",
      "status": "complete"
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
  "best_nv": 44,
  "best_td": 7821.5,
  "best_score": 51821.5
}
```

---

## Summary Table

| # | Method | Endpoint | Used By | When |
|---|---|---|---|---|
| 1 | `GET` | `/health` | Sidebar status dot, Topbar API indicator | App startup |
| 2 | `POST` | `/solve` | Solver Console — Run button | User clicks Run |
| 3 | `GET` | `/status/{job_id}` | Solver Console — live monitor | Every 500ms while running |
| 4 | `GET` | `/result/{job_id}` | Solution Viewer screen | Once job completes |
| 5 | `GET` | `/runs` | Dashboard — recent runs table + chart | Dashboard load |
| 6 | `POST` | `/stop/{job_id}` | Solver Console — Stop button | User clicks Stop |

---

## Implementation Notes for the Backend Dev

### Where each response comes from in the existing code

| API Field | Source in Python Code |
|---|---|
| `nv_min` | `CVRPEnv._compute_nv_min()` in `solver_engine.py` |
| `current_nv` / `best_nv` | `info["nv"]` returned from `CVRPEnv.step()` |
| `current_td` / `best_td` | `info["td"]` returned from `CVRPEnv.step()` |
| `score` | `competition_score(nv, td)` = `1000 * nv + td` in `solver_engine.py` |
| `stage_statuses` | Track manually as you call stages 1→5 sequentially |
| `routes[].customer_ids` | `solution.routes()` from PyVRP's `Solution` object |
| `depot` / `customers` coords | `data.location(i).x`, `.y` from PyVRP's `ProblemData` |
| `log_lines` | Append strings to a list as each stage runs |
| `iteration` | `res.num_iterations` from `pyvrp.solve()` result |

### Job state storage (simplest approach)

```python
# In-memory store is fine for a competition demo
jobs: dict[str, dict] = {}

# When POST /solve is called:
job_id = uuid4().hex[:8]
jobs[job_id] = {
    "status": "running",
    "current_stage": 1,
    "best_nv": None,
    "best_td": None,
    "log_lines": [],
    "result": None,
    ...
}
# Launch background thread that updates jobs[job_id] as it runs
```

### Progress reporting hook

Add a simple callback inside your solve loop so the background thread can write progress without blocking:

```python
def on_step(info: dict, stage: int, log_msg: str):
    jobs[job_id]["current_stage"] = stage
    jobs[job_id]["best_nv"] = info["nv"]
    jobs[job_id]["best_td"] = info["td"]
    jobs[job_id]["best_score"] = info["score"]
    jobs[job_id]["iteration"] = info["total_iters"]
    jobs[job_id]["log_lines"].append(log_msg)
    jobs[job_id]["log_lines"] = jobs[job_id]["log_lines"][-10:]  # keep last 10
```

---

*ML4VRP 2026 — GECCO Competition · API Spec v2.0*
