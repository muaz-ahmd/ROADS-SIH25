from fastapi import FastAPI
from tracking1 import analyze_traffic_comprehensive as analyze1
from tracking2 import analyze_traffic_comprehensive as analyze2
from tracking3 import analyze_traffic_comprehensive as analyze3
from tracking4 import analyze_traffic_comprehensive as analyze4
from threading import Thread
from green_time import total_clear_time_and_rows
from cps import calculate_traffic_score, calculate_safety_penalty, calculate_green_wave_bonus, calculate_cps

app = FastAPI(
    title="Intersection 1 Traffic Analysis API",
    description="Processes traffic data from 4 signals of intersection 1",
    version="1.0"
)

# Root route
@app.get("/")
async def root():
    return {"message": "Demo Server is Running"}


def calculate_signal_metrics(summary, lanes=2, platoon_weight=1.0, distance_m=100.0, avg_speed_m_s=10.0):
    """
    Calculate green time and CPS for a single signal based on its traffic data.
    """
    # Get vehicle counts for this signal
    vehicle_counts = summary["vehicle_counts"]["unique_vehicles"]
    total_vehicles = sum(vehicle_counts.values())
    
    # Calculate green time for this signal
    total_time, total_rows = total_clear_time_and_rows(total_vehicles, lanes=lanes)
    
    # Get violations for this signal
    hard_brakes = summary["violations"]["hard_braking_count"]
    tailgating = summary["violations"]["tailgating_count"]
    
    # Get speed for this signal
    avg_platoon_speed_kmh = summary["speed_analysis"]["avg_platoon_speed_kmh"]
    
    # Calculate CPS for this signal
    traffic_score = calculate_traffic_score(vehicle_counts)
    safety_penalty = calculate_safety_penalty(hard_brakes, tailgating)
    priority_bonus = calculate_green_wave_bonus(platoon_weight, distance_m, avg_speed_m_s)
    cps_value = calculate_cps(traffic_score, safety_penalty, priority_bonus)
    
    return {
        "vehicle_counts": vehicle_counts,
        "total_vehicles": total_vehicles,
        "violations": {
            "hard_brakes": hard_brakes,
            "tailgating": tailgating
        },
        "green_time": {
            "total_rows": total_rows,
            "T_clear_seconds": total_time
        },
        "cps": cps_value,
        "details": {
            "traffic_score": traffic_score,
            "safety_penalty": safety_penalty,
            "priority_bonus": priority_bonus,
            "avg_platoon_speed_kmh": avg_platoon_speed_kmh
        }
    }


# --- Individual Signal Endpoints ---
@app.get("/analyze/intersection1/signal1")
async def analyze_signal1(
    lanes: int = 2,
    platoon_weight: float = 1.0,
    distance_m: float = 100.0,
    avg_speed_m_s: float = 10.0
):
    summary1 = analyze1()
    metrics = calculate_signal_metrics(summary1, lanes, platoon_weight, distance_m, avg_speed_m_s)
    return {
        "intersection": "1",
        "signal": "1",
        "summary": summary1,
        "metrics": metrics
    }


@app.get("/analyze/intersection1/signal2")
async def analyze_signal2(
    lanes: int = 2,
    platoon_weight: float = 1.0,
    distance_m: float = 100.0,
    avg_speed_m_s: float = 10.0
):
    summary2 = analyze2()
    metrics = calculate_signal_metrics(summary2, lanes, platoon_weight, distance_m, avg_speed_m_s)
    return {
        "intersection": "1",
        "signal": "2",
        "summary": summary2,
        "metrics": metrics
    }


@app.get("/analyze/intersection1/signal3")
async def analyze_signal3(
    lanes: int = 2,
    platoon_weight: float = 1.0,
    distance_m: float = 100.0,
    avg_speed_m_s: float = 10.0
):
    summary3 = analyze3()
    metrics = calculate_signal_metrics(summary3, lanes, platoon_weight, distance_m, avg_speed_m_s)
    return {
        "intersection": "1",
        "signal": "3",
        "summary": summary3,
        "metrics": metrics
    }


@app.get("/analyze/intersection1/signal4")
async def analyze_signal4(
    lanes: int = 2,
    platoon_weight: float = 1.0,
    distance_m: float = 100.0,
    avg_speed_m_s: float = 10.0
):
    summary4 = analyze4()
    metrics = calculate_signal_metrics(summary4, lanes, platoon_weight, distance_m, avg_speed_m_s)
    return {
        "intersection": "1",
        "signal": "4",
        "summary": summary4,
        "metrics": metrics
    }


# --- Full Intersection Endpoint ---
@app.get("/analyze/intersection1")
async def analyze_intersection1(
    lanes: int = 2,
    platoon_weight: float = 1.0,
    distance_m: float = 100.0,
    avg_speed_m_s: float = 10.0
):
    """
    Run analysis for all 4 signals at Intersection 1.
    Each signal gets its own green time and CPS score.
    """

    # Step 1: Run tracking for each signal (analyze first)
    summary1 = analyze1()
    summary2 = analyze2()
    summary3 = analyze3()
    summary4 = analyze4()

    # Step 2: Calculate metrics for each signal individually (use the analyzed data)
    signal1_metrics = calculate_signal_metrics(summary1, lanes, platoon_weight, distance_m, avg_speed_m_s)
    signal2_metrics = calculate_signal_metrics(summary2, lanes, platoon_weight, distance_m, avg_speed_m_s)
    signal3_metrics = calculate_signal_metrics(summary3, lanes, platoon_weight, distance_m, avg_speed_m_s)
    signal4_metrics = calculate_signal_metrics(summary4, lanes, platoon_weight, distance_m, avg_speed_m_s)

    # Aggregate intersection-wide statistics
    combined_vehicle_counts = {}
    total_brakes = 0
    total_tailgating = 0

    for summary in [summary1, summary2, summary3, summary4]:
        for cls, count in summary["vehicle_counts"]["unique_vehicles"].items():
            combined_vehicle_counts[cls] = combined_vehicle_counts.get(cls, 0) + count

        total_brakes += summary["violations"]["hard_braking_count"]
        total_tailgating += summary["violations"]["tailgating_count"]

    total_vehicles_intersection = sum(combined_vehicle_counts.values())

    # Build response with individual signal metrics
    response = {
        "intersection": "1",
        "signals": {
            "signal1": {
                "summary": summary1,
                "metrics": signal1_metrics
            },
            "signal2": {
                "summary": summary2,
                "metrics": signal2_metrics
            },
            "signal3": {
                "summary": summary3,
                "metrics": signal3_metrics
            },
            "signal4": {
                "summary": summary4,
                "metrics": signal4_metrics
            }
        },
        "intersection_summary": {
            "total_vehicles": total_vehicles_intersection,
            "vehicle_counts": combined_vehicle_counts,
            "violations": {
                "hard_brakes": total_brakes,
                "tailgating": total_tailgating
            }
        }
    }

    return response
