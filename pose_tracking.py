import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.pose import lm, compute_angle
from utils.draw import draw_pose_landmarks
from utils.io import create_csv_writer, plot_joint_angles

def main():

    # set the outputs to go into its own folder
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open video file; USER: Change to video file name
    capture = cv2.VideoCapture("video.MOV")
    if not capture.isOpened():
        raise RuntimeError("Error opening video file")

    # Get FPS
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError("FPS could not be determined")

    frame_index = 0

    # writer for skeleton overlay
    masked_filename = os.path.join(OUTPUT_DIR, f"video_masked.mp4")
    vid_writer = cv2.VideoWriter(
        masked_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    # create a new csv file
    csv_writer, csv_filename, csv_file = create_csv_writer(OUTPUT_DIR)

    # use to change model 
    model_path = "pose_landmarker_heavy.task"

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

            pose_lm = result.pose_landmarks[0]

            # draw landmarks
            draw_pose_landmarks(frame, pose_lm)
            vid_writer.write(frame)
            
            # compute angles
            left_knee = compute_angle(lm(pose_lm, 23), lm(pose_lm, 25), lm(pose_lm, 27))            
            right_knee = compute_angle(lm(pose_lm, 24), lm(pose_lm, 26), lm(pose_lm, 28))            
            left_ankle = compute_angle(lm(pose_lm, 25), lm(pose_lm, 27), lm(pose_lm, 31))
            right_ankle = compute_angle(lm(pose_lm, 26), lm(pose_lm, 28), lm(pose_lm, 32))
            
            # write csv row
            csv_writer.writerow([
                timestamp_ms,
                left_knee, right_knee,
                left_ankle, right_ankle
            ])

            frame_index += 1

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()
    
    # plot ankle results
    plot_joint_angles(OUTPUT_DIR, csv_filename)

if __name__ == "__main__":
    main()
    