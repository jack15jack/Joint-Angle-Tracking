# Digital Twin for Stair Ascent Gait Analysis

## Object Detection and Data Transfer

This component of the **Digital Twin for Stair Ascent Gait Analysis** captures live video, detects human pose landmarks using MediaPipe, calculates lower-body joint angles, transmits biomechanical data to Unreal Engine through a UDP socket, and records the results for later analysis.

### System Overview

```text
Webcam
   │
   ▼
OpenCV Video Capture
   │
   ▼
MediaPipe Pose Landmarker
   │
   ├──────────────► Pose Visualization
   │
   ├──────────────► Joint Angle Calculation
   │                         │
   │                         ▼
   │                    CSV Recording
   │                         │
   │                         ▼
   │                    Angle Plots
   │
   └──────────────► UDP Data Transfer
                            │
                            ▼
                      Unreal Engine
                            │
                            ▼
                     Digital Twin
```

## Main Program

The main program coordinates video capture, pose detection, visualization, data collection, UDP transmission, and post-processing.

### Initialization

* Creates the output directory used to store recorded data, videos, CSV files, and plots.
* Starts an OpenCV video capture device. The camera index can be changed to select a different webcam.
* Configures the capture resolution by setting the frame width and height.
* Retrieves the camera's FPS, with the capture operating at 30 FPS if the FPS cannot be found.
* Initializes the frame counter.
* Creates a timestamped CSV file using `create_csv_writer()`.
* Loads the MediaPipe Pose Landmarker model.
* Configures MediaPipe for live-stream processing and asynchronous pose detection.
* Registers `result_callback()` as the callback function used when pose detection completes.

### Main Processing Loop

While the camera is running:

1. Captures a frame from OpenCV.
2. Converts the frame from BGR to RGB for MediaPipe.
3. Creates a MediaPipe image from the RGB frame.
4. Sends the frame and timestamp to `detect_async()`.
5. MediaPipe processes the frame asynchronously and calls `result_callback()` when the pose result is available.
6. Displays the most recently processed frame.
7. Writes processed frames to the output video.
8. Increments the frame counter.

The capture should be terminated by pressing **Q** in the OpenCV capture window. This ensures the video writer, CSV file, and other resources are properly closed and that the joint-angle plots are generated.

### Shutdown

When capture ends:

* Releases the OpenCV video capture.
* Releases the video writer.
* Closes OpenCV windows.
* Closes the CSV file.
* Calls `plot_joint_angles()` to generate joint-angle plots from the recorded data.

---

## `result_callback()`

The callback function receives the pose detection result from MediaPipe and performs the processing required for each detected frame.

### Inputs

* `result` — MediaPipe pose detection result.
* `output_image` — Processed MediaPipe image.
* `timestamp_ms` — Frame timestamp in milliseconds.

### Processing

1. Converts the MediaPipe output image into a NumPy array.
2. Converts the image from MediaPipe RGB format back to OpenCV BGR format.
3. Draws the detected pose landmarks and skeleton using `draw_pose_landmarks()`.
4. Sends the required pose data to Unreal Engine using `data_transfer()`.
5. Saves the annotated frame for display and video recording.
6. Extracts the required pose landmarks using `lm()`.
7. Calculates knee, ankle, and hip angles using `compute_angle()`.
8. Writes the calculated joint angles and timestamp to the CSV file.
9. Appends the biomechanical data to the in-memory `bio_data` collection.

This function forms the primary bridge between **pose detection** and **biomechanical analysis/data transfer**.

---

# CSV Data Recording

## `create_csv_writer()`

Creates and initializes the CSV file used to record joint-angle measurements.

### Imports

* `os`
* `csv`
* `datetime`

### Input

* `output_directory` — Directory where the CSV file should be saved.

### Returns

* `writer` — CSV writer object.
* `filename` — Generated CSV filename.
* `file` — Open CSV file handle.

### Processing

1. Generates a timestamp using `datetime`.
2. Creates a filename using the format:

```text
joint_angles_{timestamp}.csv
```

3. Opens the file for writing.
4. Creates the CSV writer.
5. Writes the header row containing the recorded biomechanical measurements.

The timestamped filename allows each recording session to produce a separate dataset.

---

# Joint Angle Analysis

## `compute_angle()`

Calculates the angle formed by three three-dimensional pose landmarks.

### Imports

* `numpy`

### Inputs

Three `[x, y, z]` coordinate arrays representing:

```text
Point A
  \
   \
    Point B
   /
  /
Point C
```

The middle point, **B**, is the joint being measured.

### Processing

The three points are converted into NumPy arrays. Two vectors are then created:

```text
BA = A - B
BC = C - B
```

The angle between the vectors is calculated using the dot product:

```text
cos(θ) = (BA · BC) / (|BA| |BC|)
```

The inverse cosine is then applied and the result is converted from radians to degrees.

### Output

Returns the joint angle in degrees.

This function provides the base geometric calculation used for the knee, ankle, and hip measurements.

---

## `lm()`

Extracts the three-dimensional coordinates of a specific MediaPipe pose landmark.

### Input

* `pose_landmarks` — MediaPipe pose landmark collection.
* `index` — Index of the desired landmark.

### Output

Returns:

```text
[x, y, z]
```

for the requested landmark.

This helper simplifies landmark extraction before performing joint-angle calculations or transmitting data.

---

# Pose Visualization

## `draw_pose_landmarks()`

Draws the detected human pose onto an OpenCV video frame.

### Imports

* `cv2`

### Inputs

* `frame` — OpenCV image frame.
* `pose_lm` — Detected MediaPipe pose landmark.

### Processing

1. Determines the frame height and width.
2. Iterates through the detected pose landmarks.
3. Converts normalized landmark coordinates into pixel coordinates.
4. Draws a point at each detected landmark.
5. Defines the connections between landmarks that form the human skeleton.
6. Draws lines between connected landmarks.

The resulting annotated frame provides visual confirmation of the pose detection being used for biomechanical calculations.

---

# UDP Object Detection Data Transfer

## `data_transfer()`

Transmits detected pose and joint-angle information from Python to the Unreal Engine digital twin using a UDP socket.

### Imports

* `socket`
* `json`
* `numpy`
* `compute_angle`
* `lm`

### Input

* `pose_lm` — MediaPipe pose landmark detected in the current frame.

### Socket Configuration

The UDP socket and Unreal Engine server address are configured outside of the function. The function uses this socket to send each processed pose measurement.

### Landmark Extraction

Required landmarks are extracted using `lm()` and converted into NumPy arrays for angle calculations.

### Joint Angle Transformation

The raw anatomical angles are converted into angle deviations relative to reference joint positions before being transmitted.

#### Knee

The knee reference position is defined as **180°**.

```text
transmitted_knee = 180 - computed_knee_angle
```

This produces the deviation of the knee from the straight-leg reference position.

#### Ankle

The ankle reference position is defined as **90°**.

```text
transmitted_ankle = 90 - computed_ankle_angle
```

This converts the measured ankle angle into a deviation relative to the defined reference orientation.

#### Hip

The hip reference position is defined as **180°**, with the transmitted value inverted to match the coordinate convention used by the digital twin.

```text
transmitted_hip = -1 * (180 - computed_hip_angle)
```

### Value Clamping

A helper `clamp()` function restricts transmitted joint-angle values to:

```text
-180° ≤ angle ≤ 180°
```

This prevents invalid or extreme values from being transmitted to Unreal Engine.

### JSON Packet

The processed joint-angle measurements are organized into a JSON data packet.

The packet contains the biomechanical values required by Unreal Engine to update the digital human model.

### Transmission

The JSON packet is:

1. Serialized into a JSON string.
2. Encoded into bytes.
3. Sent through the UDP socket to the configured Unreal Engine server address.

UDP allows the system to continuously stream pose information with low communication overhead, which is appropriate for real-time digital-twin visualization.

---

# Joint Angle Plotting

## `plot_joint_angles()`

Generates post-processing plots showing the recorded joint angles over the duration of a gait trial.

### Imports

* `os`
* `pandas`
* `datetime`
* `matplotlib`

### Inputs

* `output_directory` — Directory where generated plots are saved.
* `csv_path` — Path to the recorded joint-angle CSV file.

### Processing

1. Loads the recorded CSV using Pandas.
2. Converts recorded timestamps into elapsed time in seconds.
3. Generates a timestamp for the output filenames.
4. Creates a knee-angle plot containing both left and right knee measurements.
5. Saves the knee plot to:

```text
knee_angles_{timestamp}.png
```

6. Creates an ankle-angle plot containing both left and right ankle measurements.
7. Saves the ankle plot to:

```text
ankle_angles_{timestamp}.png
```

8. Creates a hip-angle plot containing both left and right hip measurements.
9. Saves the hip plot to:

```text
hip_angles_{timestamp}.png
```

These plots provide a visual representation of the subject's joint motion throughout the stair-ascent trial.

---

# Data Flow

The complete processing pipeline is:

```text
Camera Frame
     │
     ▼
OpenCV
     │
     ▼
MediaPipe Pose Detection
     │
     ▼
result_callback()
     │
     ├──► draw_pose_landmarks()
     │          │
     │          ▼
     │     Annotated Frame
     │
     ├──► lm()
     │      │
     │      ▼
     │   Landmark Coordinates
     │      │
     │      ▼
     │   compute_angle()
     │      │
     │      ▼
     │   Joint Angles
     │      │
     │      ├──► CSV
     │      │
     │      └──► plot_joint_angles()
     │
     └──► data_transfer()
              │
              ▼
          JSON Packet
              │
              ▼
          UDP Socket
              │
              ▼
        Unreal Engine
              │
              ▼
        Digital Twin
```

## Output Files

Each gait-analysis session can produce:

| Output                         | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `joint_angles_{timestamp}.csv` | Recorded joint-angle measurements over time          |
| `knee_angles_{timestamp}.png`  | Left and right knee angles over time                 |
| `ankle_angles_{timestamp}.png` | Left and right ankle angles over time                |
| `hip_angles_{timestamp}.png`   | Left and right hip angles over time                  |
| Recorded video                 | Video of the analyzed gait trial with pose landmarks |

## Purpose

The object detection and data-transfer system provides the real-time sensing layer of the stair-ascent digital twin. MediaPipe converts camera observations into human pose landmarks, the landmark coordinates are transformed into biomechanical joint-angle measurements, and those measurements are simultaneously **recorded for analysis** and **streamed to Unreal Engine** for real-time digital-twin visualization.
