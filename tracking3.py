import cv2
import math
from ultralytics import YOLO
from collections import defaultdict

def analyze_traffic_comprehensive(pixel_to_meter=0.05, tailgating_thresh=10, braking_thresh=8, platoon_dist=15, line_y_red=350, line_y_blue=400, save_video=False, output_video_path="output.mp4"):
    """
    Comprehensive traffic analysis combining counting, speed, violations, and distance analysis.
    
    Parameters:
    -----------
    video_path : str
        Path to input video file
    pixel_to_meter : float
        Calibration factor for pixel to meter conversion
    tailgating_thresh : float
        Distance threshold in meters for tailgating detection
    braking_thresh : float
        Speed drop threshold in km/h for hard braking detection
    platoon_dist : float
        Distance threshold in meters for platoon formation
    line_y_red : int
        Y-coordinate of red counting line
    line_y_blue : int
        Y-coordinate of blue counting line (should be > line_y_red)
    save_video : bool
        Whether to save annotated output video
    output_video_path : str
        Path for output video if save_video is True
    
    Returns:
    --------
    dict : Comprehensive summary with all metrics
    list : Detailed frame-by-frame data
    """
    
    # Load model
    model = YOLO("yolo11n.pt")
    class_list = model.names
    
    # Open video
    cap = cv2.VideoCapture('video3.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer for saving output
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Tracking dictionaries
    prev_positions = {}
    prev_speeds = {}
    vehicles_seen = {}
    
    # Violation tracking
    braking_flagged = set()
    tailgating_flagged = set()
    platoon_speeds = []
    
    # Directional counting
    counted_ids_red_to_blue = set()
    counted_ids_blue_to_red = set()
    count_red_to_blue = defaultdict(int)
    count_blue_to_red = defaultdict(int)
    crossed_red_first = {}
    crossed_blue_first = {}
    
    # Distance analysis storage
    distance_records = []
    
    # Frame data storage
    frame_data = []
    frame_no = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        
        # Run YOLO tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        
        frame_objects = []
        vehicle_positions = []
        
        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            
            # Draw counting lines (only if saving video)
            if save_video:
                cv2.line(frame, (70, line_y_red), (width-70, line_y_red), (0, 0, 255), 3)
                cv2.putText(frame, "Red Line", (20, line_y_red - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.line(frame, (70, line_y_blue), (width-70, line_y_blue), (255, 0, 0), 3)
                cv2.putText(frame, "Blue Line", (20, line_y_blue - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process each detected vehicle
            for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                class_name = class_list[class_idx]
                
                # Register vehicle
                vehicles_seen[track_id] = class_name
                
                # --- Speed Calculation ---
                if track_id in prev_positions:
                    px, py = prev_positions[track_id]
                    dist_px = math.sqrt((cx - px)**2 + (cy - py)**2)
                    dist_m = dist_px * pixel_to_meter
                    speed_mps = dist_m * fps
                    speed_kmh = speed_mps * 3.6
                else:
                    speed_kmh = 0.0
                
                # --- Hard Braking Detection ---
                braking_flag = False
                if track_id in prev_speeds:
                    speed_drop = prev_speeds[track_id] - speed_kmh
                    if speed_drop > braking_thresh and track_id not in braking_flagged:
                        braking_flagged.add(track_id)
                        braking_flag = True
                
                prev_positions[track_id] = (cx, cy)
                prev_speeds[track_id] = speed_kmh
                
                vehicle_positions.append((track_id, cx, cy, speed_kmh, class_name))
                
                # --- Line Crossing Logic ---
                if line_y_red - 3 <= cy <= line_y_red + 3:
                    if track_id not in crossed_red_first:
                        crossed_red_first[track_id] = True
                
                if line_y_blue - 3 <= cy <= line_y_blue + 3:
                    if track_id not in crossed_blue_first:
                        crossed_blue_first[track_id] = True
                
                # Downward counting (red -> blue)
                if track_id in crossed_red_first and track_id not in counted_ids_red_to_blue:
                    if line_y_blue - 5 <= cy <= line_y_blue + 5:
                        count_red_to_blue[class_name] += 1
                        counted_ids_red_to_blue.add(track_id)
                
                # Upward counting (blue -> red)
                if track_id in crossed_blue_first and track_id not in counted_ids_blue_to_red:
                    if line_y_red - 5 <= cy <= line_y_red + 5:
                        count_blue_to_red[class_name] += 1
                        counted_ids_blue_to_red.add(track_id)
                
                # Draw on frame (only if saving video)
                if save_video:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    label = f"ID:{track_id} {class_name} {speed_kmh:.1f}km/h"
                    if braking_flag:
                        label += " BRAKE!"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Store frame object data
                frame_objects.append({
                    "id": track_id,
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "speed": round(speed_kmh, 2),
                    "braking": braking_flag
                })
            
            # --- Tailgating & Distance Analysis ---
            for i in range(len(vehicle_positions)):
                for j in range(i + 1, len(vehicle_positions)):
                    id1, x1, y1, v1, cname1 = vehicle_positions[i]
                    id2, x2, y2, v2, cname2 = vehicle_positions[j]
                    dist_px = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    dist_m = dist_px * pixel_to_meter
                    
                    # Store distance record
                    distance_records.append({
                        "frame": frame_no,
                        "id1": id1,
                        "id2": id2,
                        "speed1": round(v1, 2),
                        "speed2": round(v2, 2),
                        "distance": round(dist_m, 2)
                    })
                    
                    # Tailgating detection
                    pair = tuple(sorted([id1, id2]))
                    if dist_m < tailgating_thresh and pair not in tailgating_flagged:
                        tailgating_flagged.add(pair)
                        frame_objects.append({
                            "tailgating_pair": pair,
                            "distance": round(dist_m, 2)
                        })
                    
                    # Draw distance lines (only if saving video and close)
                    if save_video and dist_m < tailgating_thresh:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"{dist_m:.1f}m",
                                   ((x1 + x2)//2, (y1 + y2)//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # --- Platoon Detection ---
            if len(vehicle_positions) >= 3:
                group_speeds = [v[3] for v in vehicle_positions]
                platoon_speeds.append(sum(group_speeds) / len(group_speeds))
            
            # Display counts on frame (only if saving video)
            if save_video:
                y_offset = 30
                for class_name, count in count_red_to_blue.items():
                    cv2.putText(frame, f"{class_name} (Down): {count}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                y_offset += 20
                for class_name, count in count_blue_to_red.items():
                    cv2.putText(frame, f"{class_name} (Up): {count}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
        
        # Store frame data
        frame_data.append({
            "frame": frame_no,
            "objects": frame_objects
        })
        
        # Write frame to output video
        if save_video and writer is not None:
            writer.write(frame)
    
    # Release resources
    cap.release()
    if writer is not None:
        writer.release()
    
    # --- Build Summary ---
    vehicle_counts = defaultdict(int)
    for cls in vehicles_seen.values():
        vehicle_counts[cls] += 1
    vehicle_counts["total_unique_vehicles"] = len(vehicles_seen)
    
    summary = {
        "video_info": {
            "total_frames": frame_no,
            "fps": fps,
            "duration_seconds": round(frame_no / fps, 2)
        },
        "vehicle_counts": {
            "unique_vehicles": dict(vehicle_counts),
            "directional_counts": {
                "downward": dict(count_red_to_blue),
                "upward": dict(count_blue_to_red),
                "total_downward": sum(count_red_to_blue.values()),
                "total_upward": sum(count_blue_to_red.values())
            }
        },
        "violations": {
            "hard_braking_count": len(braking_flagged),
            "hard_braking_ids": list(braking_flagged),
            "tailgating_count": len(tailgating_flagged),
            "tailgating_pairs": [list(pair) for pair in tailgating_flagged]
        },
        "speed_analysis": {
            "avg_platoon_speed_kmh": round(sum(platoon_speeds)/len(platoon_speeds), 2) if platoon_speeds else 0,
            "platoon_frames": len(platoon_speeds)
        },
        "distance_analysis": {
            "total_distance_records": len(distance_records),
            "close_proximity_count": sum(1 for d in distance_records if d["distance"] < tailgating_thresh)
        }
    }
    
    return summary