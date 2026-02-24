# Live Webcam Pose Tracking with MEDIAPIPE POSE and OPENCV

Body movement is tracked in real time using OpenCV and MediaPipe Pose. Landmarks are drawn on the live video stream, and joint angles are recorded for analysis.

Landmarks are overlaid on the webcam feed, displayed in an OpenCV window, and optionally saved to a video file. Joint angles (knees and ankles) are calculated for each frame, written to a CSV file, and plotted for visualization.

---

## Workflow

1. Capture live video from webcam.
2. Run MediaPipe Pose to detect body landmarks.
3. Draw skeleton overlays on the live video.
4. Compute joint angles (knee and ankle) per frame.
5. Log data to CSV.
6. Plot results using Pandas.
7. Save an optional masked video with overlays.

---

## Main Components

### MAIN (LIVE WEBCAM)

- Opens webcam via `cv2.VideoCapture(0)`.
- Output video is saved as: outputs/video_masked.mp4
- Pose model is loaded from: pose_landmarker_heavy.task
- Each frame is converted to RGB and passed to MediaPipe.
- Pose landmarks (if detected) are drawn on the frame.
- Joint angles are calculated and written to CSV.
- Annotated frames are written to the output video.
- Live display is shown using `cv2.imshow()`.
- Press **Q** to quit and cleanly close resources.
- After exit, joint angles are plotted.

---

### LIVE CALLBACK (POSE RESULT HANDLING)

- MediaPipe runs asynchronously via livestream mode.
- Results are received in a callback.
- Landmarks are drawn on the frame.
- CSV logging occurs inside the callback.
- The most recent annotated frame is stored for display.

---

### LANDMARK DRAWING

- Circles are drawn at each joint.
- Lines connect key joints (arms, legs, and torso).
- Modifies the frame in place for display and video saving.

---

### ANGLE CALCULATION

- Knee angle: hip → knee → ankle
- Ankle angle: knee → ankle → toe
- Uses 3D coordinate geometry.
- Results saved per frame in CSV.

---

### CSV OUTPUT

- Timestamp (ms)
- Left knee angle
- Right knee angle
- Left ankle angle
- Right ankle angle

---

### DATA VISUALIZATION

- CSV is loaded using Pandas.
- Joint angles are plotted over time.
- PNG graphs are saved in the outputs folder.

---

## Usage

1. Ensure webcam is connected.
2. Run the script.
3. Press **Q** to stop.
4. View results in:
 - `outputs/`
 - generated plots
 - CSV data
 - masked video

---

## Possible Future Improvements

1. Identify and isolate singular gait cycles
2. Break cycle into microphases
3. Further biometric calculations
 - posture
    - measure balance
 - range of motion
    - measure mobility and strength
 - left/right symmetry
    - measure gait consistency
 - velocity/acceleration
    - measure movement control
 - step timing, cadence, cycle variability
    - measure stability
 - vertical lift height
    - measure stair performance