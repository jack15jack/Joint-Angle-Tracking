import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.data_transfer import transfer_data
from utils.pose import lm, compute_angle
from utils.draw import draw_pose_landmarks
from utils.io import create_csv_writer, plot_data

# Important Note: Press Q to end (otherwise, video will not be saved and data will not be plotted)

def main():
    # sets output directory, creates if it doesn't exist
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # starts opencv video capture, change number for different webcam connections
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Error opening webcam")

    # change to set resolution
    width = 1920
    height = 1080

    # set the width and height of capture frame
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # get the fps
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback if webcam doesn't report FPS

    # used to count frames
    frame_index = 0

    # video writer for skeleton overlay, masked video written and saved in outputs folder
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

        # draw pose landmarks on the frame
        pose_lm = result.pose_landmarks[0]
        draw_pose_landmarks(frame_bgr, pose_lm)

        # send the needed data to unreal engine
        transfer_data(pose_lm)

        # save frame for display + video writing
        latest_frame = frame_bgr.copy()

        # landmarks
        l_sh = np.array(lm(pose_lm, 11))
        r_sh = np.array(lm(pose_lm, 12))
        l_hip = np.array(lm(pose_lm, 23))
        r_hip = np.array(lm(pose_lm, 24))
        l_knee = np.array(lm(pose_lm, 25))
        r_knee = np.array(lm(pose_lm, 26))
        l_ank = np.array(lm(pose_lm, 27))
        r_ank = np.array(lm(pose_lm, 28))
        l_toe = np.array(lm(pose_lm, 31))
        r_toe = np.array(lm(pose_lm, 32))

        # calculate angles
        left_knee = compute_angle(l_hip, l_knee, l_ank)
        right_knee = compute_angle(r_hip, r_knee, r_ank)
        left_ankle = compute_angle(l_knee, l_ank, l_toe)
        right_ankle = compute_angle(r_knee, r_ank, r_toe)
        left_hip_flex = compute_angle(l_sh, l_hip, l_knee)
        right_hip_flex = compute_angle(r_sh, r_hip, r_knee)
        
        velo, accel = get_velo_accel(right_hip_flex)

        # csv logging
        csv_writer.writerow([timestamp_ms, left_knee, right_knee, left_ankle, right_ankle, left_hip_flex, right_hip_flex])
        # save bio data to be processed after video is done
        bio_data.append([timestamp_ms, left_knee, right_knee, left_ankle, right_ankle, left_hip_flex, right_hip_flex])

    # mediapipe config
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
            
            # convert: OpenCV BGR -> MediaPipe RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # send image to asyncronous detection
            timestamp_ms = int((frame_index / fps) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms) # this will send the frame to the callback

            # display annotated frame if available
            if latest_frame is not None:
                display_frame = latest_frame
            else:
                display_frame = frame

            # start video display window
            cv2.imshow("Live Pose", display_frame)
            vid_writer.write(display_frame)

            # for timestamp
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # release after done
    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()
    
    # plot data
    plot_data(OUTPUT_DIR, csv_filename)


if __name__ == "__main__":
    main()