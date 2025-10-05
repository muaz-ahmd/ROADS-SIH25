# ===============================
# Comprehensive Priority Score (CPS) Calculator
# ===============================

# -------------------------
# Tuning knobs (adjustable)
# -------------------------
# Percentage contributions (calibrated based on discussion)
TRAFFIC_CONTRIB = 0.65  # Part 1: Traffic Score
SAFETY_CONTRIB = 0.12   # Part 2: Safety Penalty
GREEN_WAVE_CONTRIB = 0.23  # Part 3: Green Wave Bonus

# Base weights for events
HARD_BRAKING_POINTS = 2
TAILGATING_POINTS = 1

# Max scale for ETA
MAX_ETA_SCALE = 6

# -------------------------
# Traffic class weights
# -------------------------
CLASS_WEIGHTS = {
    "ambulance": 12,
    "bus": 3.5,
    "truck": 2.5,
    "car": 1,
    "bike": 0.3
}

# -------------------------
# Part 1: Traffic Score
# -------------------------
def calculate_traffic_score(vehicle_counts):
    traffic_score = 0
    
    print("\n---- Traffic Score per Class ----")
    for cls, count in vehicle_counts.items():
        try:
            w_class = CLASS_WEIGHTS[cls]
            class_score = w_class * count
            print(f"  {cls.capitalize()}: {count} vehicles * weight {w_class} = {class_score:.2f}")
            traffic_score += class_score

        except:
            print(cls)
       
    print(f"\nTotal Traffic Score: {traffic_score:.2f}\n")
    # Scale traffic score according to its contribution
    traffic_score_scaled = traffic_score * TRAFFIC_CONTRIB
    return traffic_score_scaled

# -------------------------
# Part 2: Safety Penalty
# -------------------------
def calculate_safety_penalty(hard_brakes,tailgating_events):
    print("==== Safety Penalty Calculator ====\n")
    
    
    C_rate = hard_brakes * HARD_BRAKING_POINTS + tailgating_events * TAILGATING_POINTS
    safety_penalty_scaled = C_rate * SAFETY_CONTRIB

    print(f"\nConflict Rate C_rate = {C_rate}")
    print(f"Scaled Safety Penalty contribution (SAFETY_CONTRIB={SAFETY_CONTRIB}) = {safety_penalty_scaled:.2f}\n")
    return safety_penalty_scaled

# -------------------------
# Part 3: Green Wave / Priority Bonus
# -------------------------
def calculate_green_wave_bonus(platoon_weight,distance_m,avg_speed_m_s):
    print("==== Green Wave Priority Bonus Calculator ====\n")
    
    
    ETA = distance_m / avg_speed_m_s
    max_eta = ETA * MAX_ETA_SCALE
    P_imminent = platoon_weight * max(0, 1 - (ETA / max_eta))
    priority_bonus_scaled = P_imminent * GREEN_WAVE_CONTRIB

    print(f"Tuning factor GREEN_WAVE_CONTRIB = {GREEN_WAVE_CONTRIB}")
    print(f"ETA: {ETA:.2f} s, Max ETA for scaling: {max_eta:.2f} s")
    print(f"P_imminent (scaled platoon weight) = {P_imminent:.2f}")
    print(f"Scaled Priority Bonus contribution = {priority_bonus_scaled:.2f}\n")
    
    return priority_bonus_scaled

# -------------------------
# Full CPS Calculation
# -------------------------
def calculate_cps(traffic_score,safety_penalty,priority_bonus):
    print("===== Comprehensive Priority Score (CPS) Calculator =====\n")

    
    CPS = traffic_score - safety_penalty + priority_bonus
    return CPS   
 
# -------------------------
# Run the full CPS calculator
# -------------------------
if __name__ == "__main__":
    calculate_cps()
