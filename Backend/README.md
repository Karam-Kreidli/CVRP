# Backend Folder

## Why This Folder Exists

`Backend/` contains the API service layer that turns the solver stack into an interactive application endpoint.

Without this folder, the project remains script-only. With it, external tools (web UI, notebooks, automation clients) can submit instances, poll progress, fetch final solutions, and request baseline benchmarks through HTTP.

## What This Folder Contains

```text
Backend/
  Backend.py
```

- `Backend.py`: FastAPI service that wires the trained policy (`Model/Agent_Manager.py`) and environment (`Model/Solver_Engine.py`) into asynchronous job workflows.

## Runtime Responsibilities

The backend performs all of the following:

- loads `Logs/Best_Model.pth` if available,
- accepts `.vrp` uploads,
- computes per-job metadata (`dim`, `capacity`, `total_demand`, `nv_min`),
- runs staged solve loop in a background thread,
- streams status/log updates,
- stores completed run summaries in in-memory history,
- optionally computes two benchmark baselines after completion.

## Solve Pipeline in Backend

For each submitted job:

1. Stage 1: feature extraction + initial environment reset.
2. Stage 2/3 loop: greedy policy action selection + repeated environment step.
3. Stage 4: inference-mode completion marker.
4. Finalization: format best routes and expose result payload.

The backend keeps best score tracking synchronized with environment state and only logs true best improvements.

## API Endpoints

Implemented in `Backend.py`:

- `GET /health`
  - service/device/model status and static stage-health metadata.
- `POST /solve`
  - accepts `.vrp` upload and starts async solve job.
- `GET /status/{job_id}`
  - current progress, stages, score, logs, and timing.
- `GET /result/{job_id}`
  - final formatted route result once complete.
- `GET /runs`
  - active + completed run history feed.
- `POST /stop/{job_id}`
  - cooperative stop request for running job.
- `GET /benchmark/{job_id}`
  - computes/returns baseline comparisons (default + large-pop HGS).

## How to Run

From repository root:

```bash
python Backend/Backend.py
```

Then open:

- `http://localhost:8080/docs`

for interactive Swagger UI.

## Request/Response Behavior Notes

- Jobs are in-memory (`jobs`, `runs_db` dictionaries/lists).
- If service restarts, active and completed job states are lost.
- `/solve` returns quickly with `202 Accepted` and a `job_id`.
- Poll `/status/{job_id}` until status becomes `complete`.
- `/result/{job_id}` returns `425` while still running.

## Folder-Level Dependencies

Primary imports:

- `FastAPI`, `uvicorn`
- `torch`
- `hygese`
- `Model.Agent_Manager`
- `Model.Solver_Engine`

## Why There Is Only One File Here

The backend is intentionally compact: one service file with clear endpoint and worker boundaries. This keeps deployment simple for hackathon/research workflows while still exposing all critical operations.

If this grows into production shape, common next splits are:

- routers,
- service layer,
- schemas,
- persistence layer.
