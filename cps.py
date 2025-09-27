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
def calculate_traffic_score():
    traffic_score = 0
    vehicle_counts = {}
    print("---- Traffic Score Calculator ----")
    
    for cls, w_class in CLASS_WEIGHTS.items():
        while True:
            try:
                count = int(input(f"Enter number of {cls}s: "))
                if count < 0:
                    print("Please enter a non-negative number.")
                    continue
                vehicle_counts[cls] = count
                break
            except ValueError:
                print("Invalid input! Enter integer only.")
    
    print("\n---- Traffic Score per Class ----")
    for cls, count in vehicle_counts.items():
        w_class = CLASS_WEIGHTS[cls]
        class_score = w_class * count
        print(f"  {cls.capitalize()}: {count} vehicles * weight {w_class} = {class_score:.2f}")
        traffic_score += class_score

    print(f"\nTotal Traffic Score: {traffic_score:.2f}\n")
    # Scale traffic score according to its contribution
    traffic_score_scaled = traffic_score * TRAFFIC_CONTRIB
    return traffic_score_scaled

# -------------------------
# Part 2: Safety Penalty
# -------------------------
def calculate_safety_penalty():
    print("==== Safety Penalty Calculator ====\n")
    
    while True:
        try:
            hard_brakes = int(input("Enter number of hard braking events: "))
            if hard_brakes < 0:
                continue
            break
        except ValueError:
            print("Enter integer.")
    
    while True:
        try:
            tailgating_events = int(input("Enter number of tailgating events: "))
            if tailgating_events < 0:
                continue
            break
        except ValueError:
            print("Enter integer.")
    
    C_rate = hard_brakes * HARD_BRAKING_POINTS + tailgating_events * TAILGATING_POINTS
    safety_penalty_scaled = C_rate * SAFETY_CONTRIB

    print(f"\nConflict Rate C_rate = {C_rate}")
    print(f"Scaled Safety Penalty contribution (SAFETY_CONTRIB={SAFETY_CONTRIB}) = {safety_penalty_scaled:.2f}\n")
    return safety_penalty_scaled

# -------------------------
# Part 3: Green Wave / Priority Bonus
# -------------------------
def calculate_green_wave_bonus():
    print("==== Green Wave Priority Bonus Calculator ====\n")
    
    while True:
        try:
            platoon_weight = float(input("Enter platoon weight from upstream: "))
            if platoon_weight < 0:
                continue
            break
        except ValueError:
            print("Enter number.")
    
    while True:
        try:
            distance_m = float(input("Enter distance from upstream signal (m): "))
            if distance_m <= 0:
                continue
            break
        except ValueError:
            print("Enter number > 0.")
    
    while True:
        try:
            avg_speed_m_s = float(input("Enter average speed of platoon (m/s): "))
            if avg_speed_m_s <= 0:
                continue
            break
        except ValueError:
            print("Enter number > 0.")
    
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
def calculate_cps():
    print("===== Comprehensive Priority Score (CPS) Calculator =====\n")
    
    traffic_score = calculate_traffic_score()     # Part 1
    safety_penalty = calculate_safety_penalty()   # Part 2
    priority_bonus = calculate_green_wave_bonus() # Part 3
    
    CPS = traffic_score - safety_penalty + priority_bonus
    
    print("===== Final CPS Calculation =====")
    print(f"Traffic Score contribution: {traffic_score:.2f}")
    print(f"Safety Penalty contribution: -{safety_penalty:.2f}")
    print(f"Green Wave Priority Bonus contribution: +{priority_bonus:.2f}")
    print(f"\n✅ Final Comprehensive Priority Score (CPS): {CPS:.2f}")

# -------------------------
# Run the full CPS calculator
# -------------------------
if __name__ == "__main__":
    calculate_cps()
