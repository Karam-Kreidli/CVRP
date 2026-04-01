# Dear Makhavi, this is tested with existing src files. Test and replace with actual implementations once ready.
# Install these: pip install fastapi uvicorn python-multipart torch torch-geometric numpy gymnasium hygese
# To run: "python backend.py" or "uvicorn backend:app --port 8080 --reload"
# On browser: http://localhost:8080/docs for interactive API docs

import time
import math
import threading
import uvicorn
import pathlib
import numpy as np
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import torch
import hygese as hgs
from src.model_vision import GNNEncoder
from src.agent_manager import FleetManager
from src.solver_engine import CVRPEnv, _parse_vrp_file 

# =============================================================================
# APP SETUP & STATE MANAGEMENT
# =============================================================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory databases for our current session. 
# (If the server restarts, these will clear out!)
jobs: dict[str, dict] = {}           # Tracks active /solve jobs
runs_db: list[dict] = []             # History of completed runs for the dashboard
benchmarks: dict[str, dict] = {}     # Tracks active /benchmark jobs

# =============================================================================
# AI MODEL INITIALIZATION (POST-TRAINING TODOS HERE!)
# =============================================================================
# Automatically use the GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize the raw model architectures
gnn_model = GNNEncoder().to(device)     
fleet_manager = FleetManager().to(device)       

# DEAR MAKHAVI, ACTION REQUIRED HERE POST-TRAINING 
# Once the models finish training, try UNCOMMENTING the 4 lines below or do it your own way.
# Note: We use `map_location=device` so the app doesn't crash if it tries to load a GPU-trained model on a machine that only has a CPU.
# -----------------------------------------------------------------------------
# gnn_model.load_state_dict(torch.load("checkpoints/gnn_latest.pth", map_location=device)) 
# fleet_manager.load_state_dict(torch.load("checkpoints/fm_latest.pth", map_location=device)) 
# gnn_model.eval()       # Locks the GNN in evaluation mode (important!)
# fleet_manager.eval()   # Locks the RL agent in evaluation mode (important!)
# -----------------------------------------------------------------------------


# =============================================================================
# HELPER UTILITIES
# =============================================================================
def parse_vrp_metadata(file_path: str):
    """
    Quickly scans the .vrp text file to grab basic info (Dimension, Capacity, Demand).
    We use this to immediately reply to the frontend with instance stats while 
    the heavy solver starts spinning up in the background.
    """
    dim, cap, total_dem = 0, 0, 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
    in_demand = False
    for line in lines:
        if line.startswith("DIMENSION"): dim = int(line.split()[-1])
        elif line.startswith("CAPACITY"): cap = int(line.split()[-1])
        elif line.startswith("DEMAND_SECTION"): in_demand = True
        elif line.startswith("DEPOT_SECTION") or line.startswith("EOF"): in_demand = False
        elif in_demand:
            parts = line.split()
            if len(parts) >= 2: total_dem += int(parts[1])
            
    # Calculate the mathematical absolute minimum number of vehicles needed
    nv_min = math.ceil(total_dem / cap) if cap > 0 else 0
    return dim, cap, total_dem, nv_min

def format_solution(job_id: str, instance_name: str, hgs_routes: list[list[int]], data: dict, elapsed_sec: int):
    """
    Translates the raw outputs from the HGS solver into the exact JSON structure 
    the Flutter frontend expects to draw the routes on the canvas.
    """
    formatted_routes = []
    x_coords, y_coords = data["x_coordinates"], data["y_coordinates"]
    demands, capacity = data["demands"], data["vehicle_capacity"]
    
    total_dist = 0.0
    for idx, route_nodes in enumerate(hgs_routes, start=1):
        route_dist = 0.0
        route_load = sum(demands[node] for node in route_nodes)
        
        # Calculate route distance step-by-step: Depot -> Node 1 -> Node 2 -> Depot
        prev_node = 0
        for node in route_nodes:
            route_dist += math.hypot(x_coords[node] - x_coords[prev_node], y_coords[node] - y_coords[prev_node])
            prev_node = node
        route_dist += math.hypot(x_coords[0] - x_coords[prev_node], y_coords[0] - y_coords[prev_node])
        
        total_dist += route_dist
        cap_pct = (route_load / capacity) * 100 if capacity else 0
        
        formatted_routes.append({
            "route_id": idx,
            "customer_ids": route_nodes,
            "num_stops": len(route_nodes),
            "distance": round(route_dist, 2),
            "total_load": int(route_load),
            "capacity": int(capacity),
            "capacity_pct": round(cap_pct, 1)
        })

    num_veh = len(formatted_routes)
    depot = {"id": 1, "x": float(x_coords[0]), "y": float(y_coords[0])}
    
    # Bundle all customers (skipping index 0, which is the depot)
    customers = [
        {"id": i + 1, "x": float(x_coords[i]), "y": float(y_coords[i]), "demand": int(demands[i])}
        for i in range(1, len(x_coords))
    ]

    return {
        "job_id": job_id, "instance_name": instance_name, "num_nodes": len(x_coords),
        "num_vehicles": num_veh, "total_distance": round(total_dist, 2),
        "score": round((1000 * num_veh) + total_dist, 2),
        "nv_min": math.ceil(sum(c["demand"] for c in customers) / capacity),
        "solve_time_seconds": elapsed_sec, "depot": depot,
        "customers": customers, "routes": formatted_routes
    }


# =============================================================================
# BACKGROUND WORKERS (THE HEAVY LIFTING)
# =============================================================================
def run_real_solver(job_id: str, vrp_path: str, instance_name: str):
    """
    This runs in a separate thread so the API doesn't freeze. 
    It steps through the 20-step RL environment loop, updating the `jobs` dict 
    so the frontend can poll live metrics.
    """
    try:
        start_time = time.time()
        
        # --- STAGES 1 & 2: GNN Encoding & Environment Init ---
        jobs[job_id]["current_stage"] = 1
        jobs[job_id]["stage_statuses"]["1"] = "running"
        jobs[job_id]["log_lines"].append(f"[SYS] Mode: {jobs[job_id]['mode'].upper()} | Limit: {jobs[job_id]['time_limit_seconds']}s")
        jobs[job_id]["log_lines"].append("[GNN] Encoding spatial graph...")

        # env.reset() automatically triggers the GNNEncoder to scan the instance
        env = CVRPEnv(instance_paths=[vrp_path], encoder=gnn_model, device=device)
        obs, info = env.reset()
        
        jobs[job_id]["stage_statuses"]["1"] = "done"
        jobs[job_id]["stage_times_seconds"]["1"] = round(time.time() - start_time, 2)
        jobs[job_id]["stage_statuses"]["2"] = "done"
        jobs[job_id]["stage_times_seconds"]["2"] = 0.1
        
        # --- STAGE 3: The RL / HGS Loop ---
        jobs[job_id]["current_stage"] = 3
        jobs[job_id]["stage_statuses"]["3"] = "running"
        
        stage_3_start = time.time()
        
        # The RL agent makes 20 strategic decisions per solve
        for step in range(20):
            # Abort if the user clicked "Stop" on the frontend
            if jobs[job_id]["status"] == "stopped": return
            
            # The observation comes back as a single 132-dim array.
            # We split it: first 128 are the GNN embedding, last 4 are the live stats.
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            graph_emb, stats = obs_tensor[:, :128], obs_tensor[:, 128:]
            
            # Fetch the action mask to prevent the agent from trying to reduce vehicles 
            # if we are already at the mathematical minimum.
            mask = info.get("action_mask")
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0) if mask is not None else None
            
            # POST-TRAINING TODO: Keep this wrapped in torch.no_grad()!
            # It tells PyTorch not to calculate gradients, saving a ton of memory.
            with torch.no_grad():
                action_logits, _ = fleet_manager(graph_emb, stats, action_mask=mask_tensor)
                
            action = torch.argmax(action_logits).item()
            
            # Execute the chosen strategy in the environment
            obs, reward, done, trunc, info = env.step(action)
            
            # Update live metrics for the frontend to poll
            jobs[job_id]["iteration"] = (step + 1) * 500
            jobs[job_id]["current_nv"] = info.get("nv", 0)
            jobs[job_id]["best_nv"] = info.get("nv", 0)
            jobs[job_id]["current_td"] = round(info.get("td", 0.0), 2)
            jobs[job_id]["best_td"] = round(info.get("td", 0.0), 2)
            jobs[job_id]["current_score"] = round(info.get("score", 0.0), 2)
            jobs[job_id]["best_score"] = round(info.get("score", 0.0), 2)
            jobs[job_id]["log_lines"].append(f"[FM] Step {step+1}/20: Strategy {action}")
            jobs[job_id]["log_lines"] = jobs[job_id]["log_lines"][-10:]

        jobs[job_id]["stage_statuses"]["3"] = "done"
        jobs[job_id]["stage_times_seconds"]["3"] = round(time.time() - stage_3_start, 2)

        # --- STAGES 4 & 5: Mock completion for the frontend UI flow ---
        for stage in [4, 5]:
            jobs[job_id]["current_stage"] = stage
            jobs[job_id]["stage_statuses"][str(stage)] = "done"
            jobs[job_id]["stage_times_seconds"][str(stage)] = 0.1

        # We run one final, ultra-fast pass just to extract the final discrete route 
        # objects from HGS so we can draw them on the map.
        final_params = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=100)
        final_result = env._solve_hgs(final_params)

        # Build the massive JSON object for the Solution Viewer
        elapsed_sec = int(time.time() - start_time)
        jobs[job_id]["result"] = format_solution(
            job_id, instance_name, final_result["routes"], env._hgs_data, elapsed_sec
        )

        # Save to history for the dashboard chart
        runs_db.insert(0, {
            "job_id": job_id, "instance_name": instance_name,
            "num_vehicles": final_result["nv"], "total_distance": round(final_result["td"], 2),
            "score": round(final_result["score"], 2), "solve_time_seconds": elapsed_sec,
            "status": "complete",
            "completed_at": datetime.utcnow().isoformat() + "Z"  # FIX 3: Added completed_at timestamp

        })

        # CRITICAL: Tell the frontend the job is done so it stops polling!
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["log_lines"].append("[✓] Solve complete")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log_lines"].append(f"[ERROR] {str(e)}")


def run_benchmark_background(benchmark_id: str, vrp_path: str, rl_nv: int, rl_td: float, rl_score: float, rl_time: int):
    """
    Runs the standard baseline solvers (HGS Default and HGS Large) to compare 
    against our RL agent's performance. Runs in the background.
    """
    try:
        hgs_data = _parse_vrp_file(pathlib.Path(vrp_path))

        # --- Baseline 1: Standard HGS ---
        benchmarks[benchmark_id]["message"] = "Running HGS Default (stage 1/2)..."
        t0 = time.time()
        params_default = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=10000)
        solver_default = hgs.Solver(parameters=params_default, verbose=False)
        res_default = solver_default.solve_cvrp(hgs_data, rounding=True)
        time_default = int(time.time() - t0)

        # --- Baseline 2: High-Pressure HGS ---
        benchmarks[benchmark_id]["message"] = "Running HGS Large Population (stage 2/2)..."
        t1 = time.time()
        params_large = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=10000, mu=100, lambda_=150)
        solver_large = hgs.Solver(parameters=params_large, verbose=False)
        res_large = solver_large.solve_cvrp(hgs_data, rounding=True)
        time_large = int(time.time() - t1)

        # Package the comparisons for the frontend UI
        benchmarks[benchmark_id]["comparisons"] = [
            {
                "name": "RouteIQ RL",
                "nv": rl_nv,
                "td": round(rl_td, 2),
                "score": round(rl_score, 2),
                "solve_time_seconds": rl_time,
            },
            {
                "name": "HGS Default",
                "nv": len(res_default.routes),
                "td": round(res_default.cost, 2),
                "score": round((1000 * len(res_default.routes)) + res_default.cost, 2),
                "solve_time_seconds": time_default,
            },
            {
                "name": "HGS Large Pop",
                "nv": len(res_large.routes),
                "td": round(res_large.cost, 2),
                "score": round((1000 * len(res_large.routes)) + res_large.cost, 2),
                "solve_time_seconds": time_large,
            },
        ]
        benchmarks[benchmark_id]["status"] = "complete"

    except Exception as e:
        benchmarks[benchmark_id]["status"] = "error"
        benchmarks[benchmark_id]["message"] = str(e)


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check(): #Dear Makhavi, the numbers below are placeholders. Replace with actual metrics once the model is running.
    """Checked by the frontend on boot to ensure the backend is alive."""
    return {
        "status": "ready", "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "model_loaded": True,
        "stage_health": {"gnn_observer": 1.0, "fleet_manager": 1.0, "hgs_engine": 1.0, "route_driver": 1.0, "maca_trainer": 1.0},
        "total_training_epochs": 84, "best_score_ever": 54702.7
    }

@app.post("/solve")
async def solve(
    file: UploadFile = File(...), 
    track: str = Form("cvrp"), 
    mode: str = Form("competition"), 
    time_limit_seconds: int | None = Form(None)
):
    """Entry point for the main RL solver. Returns 202 immediately, solves in background."""
    job_id = uuid4().hex[:8]
    instance_name = file.filename.replace(".vrp", "") if file.filename else "unknown"
    vrp_path = f"/tmp/{job_id}.vrp"

    # Save the uploaded file to disk so the background thread can read it
    with open(vrp_path, "wb") as f: f.write(await file.read())
    dim, cap, total_dem, nv_min = parse_vrp_metadata(vrp_path)

    # FIX 4: Map and store mode and time limits properly
    default_limits = {"fast": 60, "competition": 300, "research": 600}
    actual_time_limit = time_limit_seconds if time_limit_seconds is not None else default_limits.get(mode, 300)

    # Initialize the job's state
    jobs[job_id] = {
        "status": "running", "current_stage": 1,
        "stage_statuses": {"1": "waiting", "2": "waiting", "3": "waiting", "4": "waiting", "5": "waiting"},
        "stage_times_seconds": {"1": None, "2": None, "3": None, "4": None, "5": None},
        "current_nv": 0, "best_nv": 0, "current_td": 0.0, "best_td": 0.0,
        "current_score": 0.0, "best_score": 0.0,
        "iteration": 0, "max_iterations": 10000, "elapsed_seconds": 0,
        "start_time": time.time(), "log_lines": [], "result": None,
        "mode": mode, "time_limit_seconds": actual_time_limit

    }

    # Fire off the worker
    threading.Thread(target=run_real_solver, args=(job_id, vrp_path, instance_name), daemon=True).start()
    
    return JSONResponse(status_code=202, content={"job_id": job_id, "instance_name": instance_name, "num_nodes": dim, "vehicle_capacity": cap, "total_demand": total_dem, "nv_min": nv_min})

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Polled by the frontend every 500ms to update the UI progress bars."""
    if job_id not in jobs: 
        # FIX 3: Correct JSONResponse for 404
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})
    
    # Calculate live elapsed time
    if jobs[job_id]["status"] == "running":
        jobs[job_id]["elapsed_seconds"] = int(time.time() - jobs[job_id]["start_time"])
    
    # Strip start_time before sending to frontend (keep JSON clean)
    response = {"job_id": job_id, **jobs[job_id]}
    response.pop("start_time", None)
    
    # FIX 2: Human-readable stage mapping
    STAGE_NAMES = {
        1: "GNN Observer",
        2: "Fleet Manager",
        3: "HGS Engine",
        4: "Route Driver",
        5: "MACA Trainer",
    }
    response["stage_name"] = STAGE_NAMES.get(response.get("current_stage", 0), "Unknown")

    return response

@app.get("/result/{job_id}")
def get_result(job_id: str):
    """Called once by the frontend when status shifts to 'complete' to load the map."""
    if job_id not in jobs: 
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})
    if not jobs[job_id]["result"]: 
        return JSONResponse(status_code=425, content={"error": "Job still running", "job_id": job_id})
    return jobs[job_id]["result"]

@app.get("/runs")
def list_runs(limit: int = 20, offset: int = 0):
    """Feeds the Dashboard's recent runs table."""
    return {"total": len(runs_db), "runs": runs_db[offset : offset + limit]}

@app.post("/stop/{job_id}")
def stop_job(job_id: str):
    """Allows the user to manually abort a running solve."""
    if job_id not in jobs: 
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})

    jobs[job_id]["status"] = "stopped"
    jobs[job_id]["log_lines"].append("[!] Job stopped by user")
    return {"job_id": job_id, "status": "stopped", "best_nv": jobs[job_id].get("best_nv", 0), "best_td": jobs[job_id].get("best_td", 0.0), "best_score": jobs[job_id].get("best_score", 0.0)}

@app.post("/benchmark")
async def start_benchmark(
    file: UploadFile = File(...),
    rl_nv: int = Form(0),
    rl_td: float = Form(0.0),
    rl_score: float = Form(0.0),
    rl_job_id: str = Form("")
):
    """Kicks off the baseline comparison runner."""
    benchmark_id = "bm_" + uuid4().hex[:8]
    instance_name = file.filename.replace(".vrp", "") if file.filename else "unknown"
    vrp_path = f"/tmp/{benchmark_id}.vrp"

    with open(vrp_path, "wb") as f:
        f.write(await file.read())

    # Auto-fetch RL stats if missing but job_id is provided
    rl_time = 0
    if (rl_nv == 0 or rl_nv is None) and rl_job_id and rl_job_id in jobs and jobs[rl_job_id]["status"] == "complete":
        rl_nv = jobs[rl_job_id]["best_nv"]
        rl_td = jobs[rl_job_id]["best_td"]
        rl_score = jobs[rl_job_id]["best_score"]
        rl_time = jobs[rl_job_id].get("elapsed_seconds", 0)

    benchmarks[benchmark_id] = {
        "status": "running",
        "message": "Starting benchmark...",
        "comparisons": [],
        "instance_name": instance_name
    }

    threading.Thread(
        target=run_benchmark_background,
        args=(benchmark_id, vrp_path, rl_nv, rl_td, rl_score, rl_time),
        daemon=True,
    ).start()

    return JSONResponse(status_code=202, content={"benchmark_id": benchmark_id})

@app.get("/benchmark/{benchmark_id}")
def get_benchmark(benchmark_id: str):
    """Polled by the frontend to check benchmark status."""
    bm = benchmarks.get(benchmark_id)
    if bm is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
        
    return {"benchmark_id": benchmark_id, **bm}

# =============================================================================
# SERVER STARTUP
# =============================================================================
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8080, reload=True)
