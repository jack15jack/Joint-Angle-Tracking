# Live Webcam Pose Tracking with MEDIAPIPE POSE and OPENCV

This project performs real-time biomechanical gait analysis using OpenCV and MediaPipe Pose.

Body landmarks are detected from a live webcam feed, joint angles are computed per frame, and full gait cycles are automatically segmented, normalized, averaged, and visualized.

The system produces:

- Live skeleton overlay video
- Raw joint angle CSV data
- Time-series joint plots
- Isolated gait cycle CSV (0–100%)
- Overlaid gait cycle visualization with mean trajectory

---

## System Overview

The pipeline performs:

1. Live webcam capture  
2. Real-time pose detection (MediaPipe Pose Landmarker)  
3. Joint angle computation (knee and ankle)  
4. CSV logging of raw time-series angles  
5. Automatic gait cycle segmentation  
6. Time normalization of each cycle (0–100%)  
7. Overlay plotting of all cycles  
8. Mean gait cycle computation and visualization  

---

## Processing Pipeline

### 1. Live Video Capture
- Webcam accessed via `cv2.VideoCapture(0)`
- Resolution configurable (default: 1280x720)
- FPS automatically detected (fallback: 30 FPS)
- Press **Q** to stop recording
- Annotated video is saved to: outputs/video_masked.mp4


### 2. Pose Detection

- Uses MediaPipe Pose Landmarker (`pose_landmarker_heavy.task`)
- Runs in `LIVE_STREAM` asynchronous mode
- Landmarks are drawn on each frame
- Frames displayed in real time

---

### 3. Joint Angle Computation

For every detected frame:

- **Left knee angle**: hip → knee → ankle  
- **Right knee angle**: hip → knee → ankle  
- **Left ankle angle**: knee → ankle → toe  
- **Right ankle angle**: knee → ankle → toe  

Angles are computed using 3D vector geometry.

---

### 4. Raw Data Logging

Per frame, the following are written to CSV: 
- time_ms
- left_knee
- right_knee
- left_ankle
- right_ankle
Saved as: outputs/joint_angles_<timestamp>.csv

---

## Gait Cycle Isolation and Normalization

After recording ends:

### Step 1 — Automatic Gait Cycle Detection

- Left knee angle signal is smoothed (Savitzky-Golay filter)
- Heel-strike events are approximated by detecting local minima
- Each pair of consecutive minima defines one gait cycle

Implemented in: iso_gait_cycles()


---

### Step 2 — Time Normalization (0–100%)

Each detected gait cycle is:

- Resampled to 100 evenly spaced points  
- Converted into percentage of gait cycle (0–100%)

Implemented in: normalize_gait_cycles()

Resulting array shape: (num_cycles, 100)

Each row representes one isolated gait cycle

---

### Step 3 — Percent Gait Cycle CSV Export

A second CSV file is generated: outputs/isolated_gait_cycles.csv

Columns:
- Percent_Gait
- Cycle_1
- Cycle_2
- ...
- Cycle_N
- Mean

This file enables:

- Cross-cycle comparison  
- Machine learning feature extraction  
- Statistical variability analysis  

---

## Gait Cycle Visualization

An overlay plot is generated: outputs/gait_cycles_<timestamp>.png

Visualization includes:
- All individual cycles (semi-transparent)
- Mean gait cycle (thick, dominant curve)

---

## Time-Series Joint Plots

Separate plots are generated for:
- Knee angles over time
- Ankle angles over time

Saved as:
outputs/knee_angles_<timestamp>.png
outputs/ankle_angles_<timestamp>.png

---

## Project Structure
pose_tracking_realtime.py (for real time usage)
pose_tracking_video.py (for videoo usage)
utils/
pose.py (computations and gait segmentation)
draw.py (landmark drawing)
io.py (csv creation + plotting utilities)
outputs/

---

## How to Run

1. Ensure webcam is connected  
2. Place `pose_landmarker_heavy.task` in project root  
3. Run the script  
4. Walk naturally in frame  
5. Press **Q** to stop recording  

Outputs will appear in: outputs/

## Notes

- At least two full gait cycles are required for normalization and overlay plotting.
- If insufficient cycles are detected, only raw joint plots will be generated.
- Peak detection parameters may need tuning for slow or atypical gait.

---

## Potential Extensions

- Symmetry Analysis
- Balance Analysis (forward/lateral lean)
- Micro-phase detection  
- Cycle-to-cycle variability metrics
- Cadence and step timing calculation  
- Fall risk related data derivation
- Feature extraction for ML models   