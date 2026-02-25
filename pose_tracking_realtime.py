import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.pose import lm, compute_angle, iso_gait_cycles, normalize_gait_cycles
from utils.draw import draw_pose_landmarks
from utils.io import create_csv_writer, plot_joint_angles, build_percent_cycle_csv, plot_iso_cycles

# Important Note: Press Q to end (otherwise, video will not be saved and data will not be plotted)

def main():

    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Error opening webcam")

    # change to set resolution
    width = 1280
    height = 720

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback if webcam doesn't report FPS

    frame_index = 0

    # video writer for skeleton overlay
    masked_filename = os.path.join(OUTPUT_DIR, "video_masked.mp4")
    vid_writer = cv2.VideoWriter(
        masked_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # CSV writer
    csv_writer, csv_filename, csv_file = create_csv_writer(OUTPUT_DIR)

    # pose model
    model_path = "pose_landmarker_heavy.task"

    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # callback for asyncronous livestream results
    latest_frame = None
    bio_data = []
    def result_callback(result, output_image, timestamp_ms):
        nonlocal latest_frame

        if not result.pose_landmarks:
            return

        image = output_image.numpy_view()
        # MediaPipe RGB -> OpenCV BGR (for landmark drawing)
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        pose_lm = result.pose_landmarks[0]
        draw_pose_landmarks(frame_bgr, pose_lm)

        # save for display + video writing
        latest_frame = frame_bgr.copy()

        # calculate angles
        left_knee = compute_angle(lm(pose_lm, 23), lm(pose_lm, 25), lm(pose_lm, 27))
        right_knee = compute_angle(lm(pose_lm, 24), lm(pose_lm, 26), lm(pose_lm, 28))
        left_ankle = compute_angle(lm(pose_lm, 25), lm(pose_lm, 27), lm(pose_lm, 31))
        right_ankle = compute_angle(lm(pose_lm, 26), lm(pose_lm, 28), lm(pose_lm, 32))

        # csv logging
        csv_writer.writerow([timestamp_ms, left_knee, right_knee, left_ankle, right_ankle])
        bio_data.append([timestamp_ms, left_knee, right_knee, left_ankle, right_ankle])

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        output_segmentation_masks=False
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # OpenCV BGR -> MediaPipe RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_index / fps) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms) # this will send the frame to the callback

            # use annotated frame if available
            if latest_frame is not None:
                display_frame = latest_frame
            else:
                display_frame = frame

            cv2.imshow("Live Pose", display_frame)
            vid_writer.write(display_frame)

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # release after done
    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()
    
    # isolate gait cycles
    bio_data_np = np.array(bio_data)
    left_knee = bio_data_np[:, 1]
    cycles = iso_gait_cycles(left_knee, fps)

    # plot normalized and averaged gait cycles
    if len(cycles) > 1:
        normalized_cycles = normalize_gait_cycles(cycles, fps)
        build_percent_cycle_csv(OUTPUT_DIR, normalized_cycles)
        plot_iso_cycles(OUTPUT_DIR, normalized_cycles)

    # plot joint angle results
    plot_joint_angles(OUTPUT_DIR, csv_filename)


if __name__ == "__main__":
    main()