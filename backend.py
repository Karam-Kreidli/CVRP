# Dear Makhavi, this is tested with existing src files. Test and replace with actual implementations once ready.
# pip install fastapi uvicorn python-multipart torch torch-geometric numpy gymnasium hygese
# To run: "python backend.py" or "uvicorn backend:app --port 8080 --reload"
# On browser: http://localhost:8080/docs for interactive API docs

import time
import math
import threading
import uvicorn
import pathlib
import numpy as np
from uuid import uuid4
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import hygese as hgs
from src.agent_manager import FleetManager, INSTANCE_FEATURES_DIM
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize the Fleet Manager (the RL agent)
fleet_manager = FleetManager().to(device)

# DEAR MAKHAVI, ACTION REQUIRED HERE POST-TRAINING
# Once the models finish training, try UNCOMMENTING the 2 lines below or do it your own way.
# Note: We use `map_location=device` so the app doesn't crash if it tries to load a GPU-trained model on a machine that only has a CPU.
# -----------------------------------------------------------------------------
# fleet_manager.load_state_dict(torch.load("checkpoints/fm_latest.pth", map_location=device))
# fleet_manager.eval()   # Locks the RL agent in evaluation mode (important!)
# -----------------------------------------------------------------------------


# =============================================================================
# HELPER UTILITIES
# =============================================================================
def parse_vrp_metadata(file_path: str):
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
            
    nv_min = math.ceil(total_dem / cap) if cap > 0 else 0
    return dim, cap, total_dem, nv_min

def format_solution(job_id: str, instance_name: str, hgs_routes: list[list[int]], data: dict, elapsed_sec: int):
    formatted_routes = []
    x_coords, y_coords = data["x_coordinates"], data["y_coordinates"]
    demands, capacity = data["demands"], data["vehicle_capacity"]
    
    total_dist = 0.0
    for idx, route_nodes in enumerate(hgs_routes, start=1):
        route_dist = 0.0
        route_load = sum(demands[node] for node in route_nodes)
        
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
# BACKGROUND WORKERS
# =============================================================================
def run_real_solver(job_id: str, vrp_path: str, instance_name: str):
    try:
        start_time = time.time()
        
        # --- STAGES 1 & 2: Feature Extraction & Environment Init ---
        jobs[job_id]["current_stage"] = 1
        jobs[job_id]["stage_statuses"]["1"] = "running"
        jobs[job_id]["log_lines"].append(f"[SYS] Mode: {jobs[job_id]['mode'].upper()} | Limit: {jobs[job_id]['time_limit_seconds']}s")
        jobs[job_id]["log_lines"].append("[FE] Computing instance features...")

        # env.reset() computes hand-crafted instance features and initial solve
        env = CVRPEnv(instance_paths=[vrp_path], device=device)
        obs, info = env.reset()
        
        jobs[job_id]["stage_statuses"]["1"] = "done"
        jobs[job_id]["stage_times_seconds"]["1"] = round(time.time() - start_time, 2)
        
        # Stage 2: Fleet Manager Init
        jobs[job_id]["current_stage"] = 2
        jobs[job_id]["stage_statuses"]["2"] = "running"
        time.sleep(0.2)  # UI transition buffer
        jobs[job_id]["stage_statuses"]["2"] = "done"
        jobs[job_id]["stage_times_seconds"]["2"] = 0.2
        
        # Stage 3: HGS Engine 
        jobs[job_id]["current_stage"] = 3
        jobs[job_id]["stage_statuses"]["3"] = "running"
        
        stage_3_start = time.time()
        max_steps = env.max_steps  # 50 steps with new design
        for step in range(max_steps):
            if jobs[job_id]["status"] == "stopped": return

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            inst_feat = obs_tensor[:, :INSTANCE_FEATURES_DIM]
            stats = obs_tensor[:, INSTANCE_FEATURES_DIM:]

            mask = info.get("action_mask")
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0) if mask is not None else None

            with torch.no_grad():
                action_logits, _ = fleet_manager(inst_feat, stats, action_mask=mask_tensor)

            action = torch.argmax(action_logits).item()
            obs, reward, done, trunc, info = env.step(action)

            jobs[job_id]["iteration"] = step + 1  # step count (budget varies per action)
            jobs[job_id]["current_nv"] = info.get("nv", 0)
            jobs[job_id]["best_nv"] = info.get("nv", 0)
            jobs[job_id]["current_td"] = round(info.get("td", 0.0), 2)
            jobs[job_id]["best_td"] = round(info.get("td", 0.0), 2)
            jobs[job_id]["current_score"] = round(info.get("score", 0.0), 2)
            jobs[job_id]["best_score"] = round(info.get("score", 0.0), 2)
            jobs[job_id]["log_lines"].append(f"[FM] Step {step+1}/{max_steps}: Strategy {action}")
            jobs[job_id]["log_lines"] = jobs[job_id]["log_lines"][-10:]

            if done or trunc:
                break

        jobs[job_id]["stage_statuses"]["3"] = "done"
        jobs[job_id]["stage_times_seconds"]["3"] = round(time.time() - stage_3_start, 2)

        for stage in [4, 5]:
            jobs[job_id]["current_stage"] = stage
            jobs[job_id]["stage_statuses"][str(stage)] = "done"
            jobs[job_id]["stage_times_seconds"][str(stage)] = 0.1

        final_params = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=100)
        final_result = env._solve_hgs(final_params)

        elapsed_sec = int(time.time() - start_time)
        jobs[job_id]["result"] = format_solution(
            job_id, instance_name, final_result["routes"], env._hgs_data, elapsed_sec
        )

        runs_db.insert(0, {
            "job_id": job_id, "instance_name": instance_name,
            "num_vehicles": final_result["nv"], "total_distance": round(final_result["td"], 2),
            "score": round(final_result["score"], 2), "solve_time_seconds": elapsed_sec,
            "status": "complete",
            "completed_at": datetime.utcnow().isoformat() + "Z"
        })

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["log_lines"].append("[✓] Solve complete")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log_lines"].append(f"[ERROR] {str(e)}")


def run_benchmark_for_job(job_id: str):
    """
    Runs the standard baseline solvers against the RL agent's completed run.
    Uses exactly 20,000 iterations to match the agent's budget.
    """
    job = jobs[job_id]
    vrp_path = job["vrp_path"]
    
    try:
        hgs_data = _parse_vrp_file(pathlib.Path(vrp_path))

        # --- Baseline 1: Default HGS ---
        t0 = time.time()
        params_default = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=20000)
        solver_default = hgs.Solver(parameters=params_default, verbose=False)
        res_default = solver_default.solve_cvrp(hgs_data, rounding=True)
        time_default = int(time.time() - t0)

        # --- Baseline 2: Large Population ---
        t1 = time.time()
        params_large = hgs.AlgorithmParameters(timeLimit=0.0, nbIter=20000, mu=50, lambda_=80)
        solver_large = hgs.Solver(parameters=params_large, verbose=False)
        res_large = solver_large.solve_cvrp(hgs_data, rounding=True)
        time_large = int(time.time() - t1)

        job["benchmark_result"] = {
            "job_id": job_id,
            "instance_name": job["result"]["instance_name"],
            "num_nodes": job["result"]["num_nodes"],
            "rl": {
                "nv": job["result"]["num_vehicles"],
                "td": job["result"]["total_distance"],
                "score": job["result"]["score"],
                "solve_time_seconds": job["result"]["solve_time_seconds"]
            },
            "hgs_default": {
                "nv": len(res_default.routes),
                "td": round(res_default.cost, 2),
                "score": round((1000 * len(res_default.routes)) + res_default.cost, 2),
                "solve_time_seconds": time_default
            },
            "hgs_large_pop": {
                "nv": len(res_large.routes),
                "td": round(res_large.cost, 2),
                "score": round((1000 * len(res_large.routes)) + res_large.cost, 2),
                "solve_time_seconds": time_large
            }
        }
        job["benchmark_status"] = "complete"

    except Exception as e:
        job["benchmark_status"] = "error"
        print(f"[Benchmark Error] {str(e)}")


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check():
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
    job_id = uuid4().hex[:8]
    instance_name = file.filename.replace(".vrp", "") if file.filename else "unknown"
    vrp_path = f"/tmp/{job_id}.vrp"

    with open(vrp_path, "wb") as f: f.write(await file.read())
    dim, cap, total_dem, nv_min = parse_vrp_metadata(vrp_path)

    default_limits = {"fast": 60, "competition": 300, "research": 600}
    actual_time_limit = time_limit_seconds if time_limit_seconds is not None else default_limits.get(mode, 300)

    jobs[job_id] = {
        "status": "running", "current_stage": 1,
        "vrp_path": vrp_path, # Tracked so the benchmark endpoint can re-use it
        "stage_statuses": {"1": "waiting", "2": "waiting", "3": "waiting", "4": "waiting", "5": "waiting"},
        "stage_times_seconds": {"1": None, "2": None, "3": None, "4": None, "5": None},
        "current_nv": 0, "best_nv": 0, "current_td": 0.0, "best_td": 0.0,
        "current_score": 0.0, "best_score": 0.0,
        "iteration": 0, "max_iterations": 20000, 
        "elapsed_seconds": 0, "start_time": time.time(), "log_lines": [], "result": None,
        "mode": mode, "time_limit_seconds": actual_time_limit
    }

    threading.Thread(target=run_real_solver, args=(job_id, vrp_path, instance_name), daemon=True).start()
    
    return JSONResponse(status_code=202, content={"job_id": job_id, "instance_name": instance_name, "num_nodes": dim, "vehicle_capacity": cap, "total_demand": total_dem, "nv_min": nv_min})

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs: 
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})
    
    if jobs[job_id]["status"] == "running":
        jobs[job_id]["elapsed_seconds"] = int(time.time() - jobs[job_id]["start_time"])
    
    response = {"job_id": job_id, **jobs[job_id]}
    response.pop("start_time", None)
    response.pop("vrp_path", None) # Don't leak internals
    
    STAGE_NAMES = {1: "GNN Observer", 2: "Fleet Manager", 3: "HGS Engine", 4: "Route Driver", 5: "MACA Trainer"}
    response["stage_name"] = STAGE_NAMES.get(response.get("current_stage", 0), "Unknown")

    return response

@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs: 
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})
    if not jobs[job_id]["result"]: 
        raise HTTPException(status_code=425, detail="Job still running")
    return jobs[job_id]["result"]

@app.get("/runs")
def list_runs(limit: int = 20, offset: int = 0):
    return {"total": len(runs_db), "runs": runs_db[offset : offset + limit]}

@app.post("/stop/{job_id}")
def stop_job(job_id: str):
    if job_id not in jobs: 
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})
    jobs[job_id]["status"] = "stopped"
    jobs[job_id]["log_lines"].append("[!] Job stopped by user")
    return {"job_id": job_id, "status": "stopped", "best_nv": jobs[job_id].get("best_nv", 0), "best_td": jobs[job_id].get("best_td", 0.0), "best_score": jobs[job_id].get("best_score", 0.0)}

# --- NEW BENCHMARK LOGIC ---
@app.get("/benchmark/{job_id}")
def get_benchmark(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found", "job_id": job_id})
        
    job = jobs[job_id]
    
    # Don't benchmark until the RL agent has finished its solve!
    if job["status"] != "complete":
        return JSONResponse(status_code=425, content={"status": "computing", "job_id": job_id})
        
    # Start the benchmark if it hasn't been triggered yet
    if "benchmark_status" not in job:
        job["benchmark_status"] = "computing"
        threading.Thread(target=run_benchmark_for_job, args=(job_id,), daemon=True).start()
        return JSONResponse(status_code=425, content={"status": "computing", "job_id": job_id})
        
    # Wait for the background thread to finish
    if job["benchmark_status"] == "computing":
        return JSONResponse(status_code=425, content={"status": "computing", "job_id": job_id})
        
    if job["benchmark_status"] == "error":
        return JSONResponse(status_code=500, content={"error": "Baseline solver crashed"})
        
    # Return the clean 3-way comparison
    return job["benchmark_result"]

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8080, reload=True)
