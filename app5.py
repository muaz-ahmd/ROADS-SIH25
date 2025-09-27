import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="Smart Traffic Monitoring")

st.title("🚦 Smart Traffic Density, Flow & Counts with Lane Analysis (YOLOv11)")

with st.sidebar:
    st.header("Input Settings")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    uploaded_model = st.file_uploader("Upload YOLO model weights (.pt)", type=["pt"])
    
    lane_rois = st.text_input(
        

        "Lane ROIs (semicolon separated)",
        value="0,0,640,720; 640,0,1280,720"
    )
    line_y = st.slider("Counting line (y pixel)", 50, 1000, 300)
    conf_thres = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
    iou_thresh = st.slider("Tracker IoU threshold", 0.1, 1.0, 0.5, 0.05)
    interval_len = st.number_input("Interval length (seconds)", value=10, step=5, min_value=5)

# ---------------------------
# Simple Tracker (IoU-based)
# ---------------------------
class Track:
    def __init__(self, tid, bbox, cls):
        self.id = tid
        self.bbox = bbox
        self.cls = cls
        self.missed = 0
        self.crossed = set()  # lanes where this track has crossed

def iou(b1, b2):
    x1, y1, x2, y2 = b1
    a1, b1_, a2, b2_ = b2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1_)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2_)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2_ - b1_)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def update_tracks(tracks, detections, iou_thresh=0.5):
    assigned = set()
    for tid, track in list(tracks.items()):
        best_iou = 0
        best_det = None
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            iou_val = iou(track.bbox, det[:4])
            if iou_val > best_iou:
                best_iou = iou_val
                best_det = i
        if best_iou > iou_thresh and best_det is not None:
            x1, y1, x2, y2, cls, conf = detections[best_det]
            track.bbox = (x1, y1, x2, y2)
            track.cls = cls
            track.missed = 0
            assigned.add(best_det)
        else:
            track.missed += 1
            if track.missed > 10:
                del tracks[tid]
    for i, det in enumerate(detections):
        if i not in assigned:
            x1, y1, x2, y2, cls, conf = det
            tid = len(tracks) + 1 + i
            tracks[tid] = Track(tid, (x1, y1, x2, y2), cls)
    return tracks

# ---------------------------
# Density & Flow Calculation
# ---------------------------
def compute_density(tracks, roi, class_names):
    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    densities = {}
    for target_cls in ["car", "bus", "truck", "motorcycle"]:
        total_area = 0
        for t in tracks.values():
            if t.cls < len(class_names) and class_names[t.cls] == target_cls:
                x1, y1, x2, y2 = t.bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                    total_area += (x2 - x1) * (y2 - y1)
        densities[target_cls] = total_area / roi_area if roi_area > 0 else 0
    total_area = 0
    for t in tracks.values():
        x1, y1, x2, y2 = t.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
            total_area += (x2 - x1) * (y2 - y1)
    densities["total"] = total_area / roi_area if roi_area > 0 else 0
    return densities

def compute_flow(tracks, roi, line_y, class_names, lane_id):
    flow_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
    for t in tracks.values():
        x1, y1, x2, y2 = t.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if roi[0] <= cx <= roi[2] and cy > line_y and lane_id not in t.crossed:
            cname = class_names[t.cls] if t.cls < len(class_names) else None
            if cname in flow_counts:
                flow_counts[cname] += 1
            t.crossed.add(lane_id)
    return flow_counts

# ---------------------------
# Run YOLO + Streamlit
# ---------------------------
if uploaded_video and uploaded_model:
    # Save uploaded files
    video_path = f"C:/Users/MEHTAB ALAM/Desktop/SIH/code/4.mp4{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    model_path = f"C:/Users/MEHTAB ALAM/Desktop/SIH/code/yolo11n.pt{uploaded_model.name}"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    model = YOLO(model_path)
    class_names = model.names

    # Parse lane ROIs
    lanes = []
    for roi_str in lane_rois.split(";"):
        x1, y1, x2, y2 = map(int, roi_str.strip().split(","))
        lanes.append((x1, y1, x2, y2))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_counts = st.empty()
    chart_density = st.empty()
    chart_flow = st.empty()

    tracks = {}
    unique_counts = defaultdict(set)
    interval_data = []

    frame_idx = 0
    start_time = time.time()
    interval_start = start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=conf_thres, verbose=False)[0]

        # ✅ FIX: use .tolist() for safe float unpacking
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append((x1, y1, x2, y2, cls, conf))

        tracks = update_tracks(tracks, detections, iou_thresh=iou_thresh)

        lane_metrics = []
        for lane_id, roi in enumerate(lanes, start=1):
            frame_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
            for t in tracks.values():
                if t.cls < len(class_names):
                    cname = class_names[t.cls]
                    if cname in frame_counts:
                        x1, y1, x2, y2 = t.bbox
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                            frame_counts[cname] += 1
                            unique_counts[cname].add(t.id)

            densities = compute_density(tracks, roi, class_names)
            flow_counts = compute_flow(tracks, roi, line_y, class_names, lane_id)

            lane_metrics.append({
                "lane": lane_id,
                "counts": frame_counts,
                "densities": densities,
                "flow": flow_counts
            })

            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        for t in tracks.values():
            x1, y1, x2, y2 = map(int, t.bbox)
            cname = class_names[t.cls] if t.cls < len(class_names) else "unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cname} ID:{t.id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        metrics_placeholder.write(f"Frame {frame_idx} | Lane metrics: {lane_metrics}")

        now = time.time()
        if now - interval_start >= interval_len:
            for m in lane_metrics:
                interval_data.append({
                    "time": now - start_time,
                    "lane": m["lane"],
                    **{f"{k}_count": v for k, v in m["counts"].items()},
                    **{f"{k}_density": v for k, v in m["densities"].items()},
                    "total_density": m["densities"]["total"],
                    "flow": sum(m["flow"].values())
                })
            interval_start = now

        if interval_data:
            df_plot = pd.DataFrame(interval_data)

            # Counts chart
            fig1, ax1 = plt.subplots()
            for lane_id in df_plot["lane"].unique():
                df_lane = df_plot[df_plot["lane"] == lane_id].set_index("time")
                df_lane[["car_count","bus_count","truck_count","motorcycle_count"]].plot(ax=ax1)
            ax1.set_title("Vehicle Counts per Interval (per lane)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Count")
            chart_counts.pyplot(fig1)

            # Density chart
            fig2, ax2 = plt.subplots()
            for lane_id in df_plot["lane"].unique():
                df_lane = df_plot[df_plot["lane"] == lane_id].set_index("time")
                df_lane[["car_density","bus_density","truck_density","motorcycle_density","total_density"]].plot(ax=ax2)
            ax2.set_title("Density per Interval (per lane)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Density")
            chart_density.pyplot(fig2)

            # Flow chart
            fig3, ax3 = plt.subplots()
            for lane_id in df_plot["lane"].unique():
                df_lane = df_plot[df_plot["lane"] == lane_id].set_index("time")
                df_lane[["flow"]].plot(ax=ax3)
            ax3.set_title("Traffic Flow per Interval (per lane)")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Vehicles crossed")
            chart_flow.pyplot(fig3)

    cap.release()

    # Save CSV
    if interval_data:
        df_csv = pd.DataFrame(interval_data)
        csv_path = "traffic_metrics.csv"
        df_csv.to_csv(csv_path, index=False)
        st.success(f"✅ Metrics saved to {csv_path}")
        st.dataframe(df_csv.head())

    st.subheader("📊 Aggregated Stats")
    agg_stats = {cls: len(ids) for cls, ids in unique_counts.items()}
    st.write("Total unique vehicles:", agg_stats)

else:
    st.info("Upload both a video and YOLO model weights to start.")
