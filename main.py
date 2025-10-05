from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
# Local Imports
from model import TrafficInput
from cps import  calculate_traffic_score,calculate_safety_penalty,calculate_green_wave_bonus,calculate_cps
from green_time import total_clear_time_and_rows

# Initialize FastAPI app
app = FastAPI()
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


signal_data_store: Dict[str, Dict] = {}


@app.get("/")
def read_root():
    return {"message": "Server is up and running"}
# Per-Signal Endpoint
@app.post("/intersection1/traffic{signal_id}")
def process_signal(signal_id: int, data: TrafficInput):
    if signal_id not in [1, 2, 3, 4]:
        return {"error": "Signal ID must be 1, 2, 3, or 4"}

    # Compute CPS
    traffic_score = calculate_traffic_score(data.vehicle_counts.dict())
    safety_penalty = calculate_safety_penalty(data.hard_brakes, data.tailgating_events)
    priority_bonus = calculate_green_wave_bonus(data.platoon_weight, data.distance_m, data.avg_speed_m_s)
    cps_score = calculate_cps(traffic_score, safety_penalty, priority_bonus)

    # Compute Green Time
    t_clear, total_rows = total_clear_time_and_rows(data.queue_length, data.lanes)

    # Store result
    signal_data_store[f"traffic{signal_id}"] = {
        "signal_id": signal_id,
        "cps_score": round(cps_score, 2),
        "traffic_score": round(traffic_score, 2),
        "safety_penalty": round(safety_penalty, 2),
        "priority_bonus": round(priority_bonus, 2),
        "t_clear": t_clear,
        "total_rows": total_rows
    }

    return signal_data_store[f"traffic{signal_id}"]

@app.get("/intersection1/traffic{signal_id}")
def get_signal_data(signal_id: int):
    key = f"traffic{signal_id}"
    if key in signal_data_store:
        return signal_data_store[key]
    return {"error": "No data found for this signal ID"}

# Aggregated Endpoint
@app.get("/intersection1")
def get_intersection_summary():
    for i in signal_data_store.keys():
        return signal_data_store[i]
    return {"message": "No data available"}