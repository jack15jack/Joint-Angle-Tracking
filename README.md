# Real-Time Human Pose Tracking and Gait Analysis

A real-time human motion capture and gait analysis system built with **Python**, **MediaPipe**, **OpenCV**, and **NumPy**. The application estimates human pose from a webcam feed, computes lower-body joint kinematics, streams motion data to **Unreal Engine** via UDP, and performs post-processing to generate biomechanical visualizations and gait analysis results.

The project combines computer vision, biomechanics, and real-time networking into a lightweight motion analysis pipeline requiring only a standard webcam.

---

## Features

* Real-time webcam pose estimation using MediaPipe Pose Landmarker
* Live skeletal overlay visualization
* Automatic computation of:
  * Left & Right Knee Flexion
  * Left & Right Ankle Angle
  * Left & Right Hip Flexion
* CSV logging of all joint angle data
* Automatic generation of joint-angle plots
* Real-time UDP streaming of joint angles to Unreal Engine
* Automatic recording of annotated video

---

## Example Pipeline

```
Webcam
   │
   ▼
OpenCV Video Capture
   │
   ▼
MediaPipe Pose Estimation
   │
   ├───────────────► UDP Stream
   │                     │
   │                     ▼
   │              Unreal Engine
   │
   ▼
Joint Angle Computation
   │
   ▼
CSV Logging
   │
   ▼
Biomechanical Analysis
   │
   ▼
Joint Angle Plots
```

---

## Project Structure

```
.
├── main.py
├── pose_landmarker_heavy.task
├── outputs/
└── utils/
    ├── data_transfer.py
    ├── draw.py
    ├── io.py
    └── pose.py
```

---

## Technologies

* Python 3.0+
* MediaPipe Tasks
* OpenCV
* NumPy
* SciPy
* Pandas
* Matplotlib
* UDP Sockets

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jack15jack/Joint-Angle-Tracking.git
cd Joint-Angle-Tracking
```

Install dependencies:

```bash
pip install mediapipe opencv-python numpy scipy pandas matplotlib
```

Download the MediaPipe pose model:

```
pose_landmarker_heavy.task
```

and place it in the project root.

---

## Running

Simply execute:

```bash
python main.py
```

A webcam window will open showing the live skeleton overlay.

> **Important:** Press **Q** to exit the program properly. Closing the window directly will prevent the video and plots from being saved.

---

## Output Files

After each recording session, the program automatically generates an `outputs/` directory containing:

```
outputs/
├── video_masked.mp4
├── joint_angles_YYYYMMDD_HHMMSS.csv
├── knee_angles_*.png
├── ankle_angles_*.png
└── hip_angles_*.png
```

### Video

Annotated recording with the detected pose skeleton.

### CSV

Timestamped joint angle measurements for:

* Left Knee
* Right Knee
* Left Ankle
* Right Ankle
* Left Hip
* Right Hip

### Joint Angle Plots

Automatically generated figures showing each joint angle over time.

---

## Unreal Engine Integration

Joint angles are streamed in real time using UDP packets.

Default configuration:

```
IP Address : 127.0.0.1
Port       : 5005
```

The transmitted JSON packet contains:

```json
{
  "lk": 0,
  "rk": 0,
  "la": 0,
  "ra": 0,
  "lhf": 0,
  "rhf": 0
}
```

along with placeholder values for additional joints.

This allows Unreal Engine to animate a skeletal model directly from the tracked human motion.

---

## Joint Angle Calculations

The system computes anatomical joint angles from MediaPipe landmarks using vector geometry.

### Knee Flexion

Computed using:

* Hip
* Knee
* Ankle

### Ankle Angle

Computed using:

* Knee
* Ankle
* Toe

### Hip Flexion

Computed using:

* Shoulder
* Hip
* Knee

All calculations are performed in real time for every captured frame.

---

## Current Limitations

* Single-person tracking
* Webcam-based 3D pose estimation
* Lower-body analysis only
* Requires good lighting conditions
* No calibration to anatomical reference frames
* Uses MediaPipe's estimated 3D landmarks rather than marker-based motion capture

---

## Future Improvements

* Gait Isolation
* Fall Risk Detection
* Range of motion (ROM) calculations
* Step length estimation
* Walking speed estimation
* Posture analysis
* Vertical center-of-mass displacement
* Joint angular velocity and acceleration
* Temporal gait parameters
* Network support for remote Unreal Engine visualization

---

## Applications

Potential applications include:

* Digital Twin
* Biomechanics research
* Physical therapy
* Rehabilitation monitoring
* Sports performance analysis
* Animation and virtual production
* Robotics
* Educational demonstrations

---

## Acknowledgments

This project utilizes:

* Google's MediaPipe Pose Landmarker for real-time pose estimation
* OpenCV for image capture and visualization
* NumPy and SciPy for numerical computation
* Matplotlib and Pandas for data visualization and analysis
