import cv2
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# returns [x,y,z] of given landmark
def lm(pose_landmarks, i):
    return[pose_landmarks[i].x,
           pose_landmarks[i].y,
           pose_landmarks[i].z]


# basic angle computation
def compute_angle (a, b, c):
    # convert to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a-b
    bc= c-b
    angle = np.degrees(
        np.arccos(
            np.dot(ba, bc) /
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        )
    )
    return angle


def write_angles_to_csv(writer, timestamp_ms,
                        left_knee, right_knee,
                        left_ankle, right_ankle):
    writer.writerow([
        timestamp_ms,
        left_knee, right_knee,
        left_ankle, right_ankle
    ])


def draw_pose_landmarks(frame, pose_landmarks):
    
    h, w, _ = frame.shape

    # Draw points
    for lm in pose_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Define connections (skeleton lines)
    connections = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
        (11, 12), (23, 24),  # shoulders & hips
        (12, 24), (11, 23),  # torso diagonals
        (27, 31), (29, 31),  # left feet
        (28, 32), (30, 32)   # right feet
    ]

    # Draw lines
    for start_idx, end_idx in connections:
        x1, y1 = int(pose_landmarks[start_idx].x * w), int(pose_landmarks[start_idx].y * h)
        x2, y2 = int(pose_landmarks[end_idx].x * w), int(pose_landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():

    # Open video file
    capture = cv2.VideoCapture("video.mp4")
    if not capture.isOpened():
        raise RuntimeError("Error opening video file")

    # Get FPS
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError("FPS could not be determined")

    frame_index = 0

    # writer for skeleton overlay
    vid_writer = cv2.VideoWriter(
        "video_masked.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    # csv setup
    csv_file = open("joint_angles.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "time_ms", 
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ])


    # use to change model 
    model_path = "pose_landmarker_lite.task"

    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            timestamp_ms = int((frame_index / fps) * 1000)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # no pose detected -> skip the frame
            if not result.pose_landmarks:
                frame_index += 1
                continue

            pose_landmarks = result.pose_landmarks[0]

            # draw landmarks
            draw_pose_landmarks(frame, pose_landmarks)
            vid_writer.write(frame)
            
            # compute angles
            left_knee = compute_angle(lm(pose_landmarks, 23),
                                      lm(pose_landmarks, 25),
                                      lm(pose_landmarks, 27))
            
            right_knee = compute_angle(lm(pose_landmarks, 24), 
                                       lm(pose_landmarks, 26), 
                                       lm(pose_landmarks, 28))
            
            left_ankle = compute_angle(lm(pose_landmarks, 25), 
                                       lm(pose_landmarks, 27), 
                                       lm(pose_landmarks, 31))
            
            right_ankle = compute_angle(lm(pose_landmarks, 26), 
                                        lm(pose_landmarks, 28), 
                                        lm(pose_landmarks, 32))
            
            # write csv row
            write_angles_to_csv(
                csv_writer,
                timestamp_ms,
                left_knee, right_knee,
                left_ankle, right_ankle
            )

            #TODO: for each (knee and ankle) plot a graph of both left and right angles over time 

            frame_index += 1

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()