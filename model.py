from pydantic import BaseModel

class VehicleCounts(BaseModel):
    car: int = 0
    bus: int = 0
    truck: int = 0
    motorcycle: int = 0
    ambulance: int = 0
# Input Model
class TrafficInput(BaseModel):
    vehicle_counts: VehicleCounts
    hard_brakes: int
    tailgating_events: int
    platoon_weight: float
    distance_m: float
    avg_speed_m_s: float
    queue_length: int
    lanes: int