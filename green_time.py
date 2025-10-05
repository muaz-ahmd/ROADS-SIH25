import math

def total_clear_time_and_rows(queue_length: int, lanes: int) -> tuple[int, int]:
    """Return total clearance time (as integer) and total number of rows."""
    if queue_length <= 0 or lanes <= 0:
        return 0, 0

    rows = math.ceil(queue_length / lanes)
    base_times = [3.8, 3.1, 2.7, 2.2]

    if rows <= 4:
        total_time = sum(base_times[:rows])
    else:
        total_time = sum(base_times) + (rows - 4) * 2.1

    return int(round(total_time)), rows  # Convert total_time to integer


if __name__ == "__main__":
    try:
        vehicles = int(input("Enter total number of vehicles: "))
        lanes = int(input("Enter number of lanes: "))

        if vehicles <= 0 or lanes <= 0:
            print("Error: Vehicles and lanes must be positive integers!")
        else:
            total_time, total_rows = total_clear_time_and_rows(vehicles, lanes)
            print(f"\nTotal Rows: {total_rows}")
            print(f"Total Queue Clearance Time (T_clear): {total_time} seconds")

    except ValueError:
        print("Invalid input! Please enter integers only.")
