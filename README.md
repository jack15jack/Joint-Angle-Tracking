Input a video of someone moving.
Their body is tracked using OpenCV and MediaPipe Pose. 
The landmarks from MediaPipe Pose are masked onto the original video and saved. 
The angles of their knee and ankle joints at each point in time are calculated, saved in a CSV file, and plotted using Pandas.
- In the future this might be changed to be real-time with a webcam.

The following is a full description of each of the functions and what they do (nearly line by line):

MAIN:
- Create the output directory if it doesn't exist already.
- Receive the video based off of filename using OpenCV's VideoCapture function. *FILENAME MUST BE CHANGED BY USER, WOULD BE 0 FOR REAL-TIME.* If the capture isn't opened, an error will throw.
- The fps of the video is determined using OpenCV. If the fps isn't determined, an error will throw.
- A video writer, from OpenCV's VideoWriter, is initialized for the skeleton overlay of the original video. New video will be named *video_masked.mp4.* and stored in the output folder
- The CREATE_CSV_WRITER function is called to open the csv file to store joint angles.
- The MediaPipe tracking model is determined using the file name of our .task file. This would need to be changed if we wanted a different tracking model.
- The options of the tracking mode are set as base (with the model path being our .task file) and running mode of Video Vision Running Mode. *WOULD NEED TO CHANGE IF NOT USING STATIC VIDEO.*
- The Pose Landmarker is created with those options. Starting a loop.
- Reads the next frame of the video. The frame is saved as a BGR image. If the frame was not read successfully, break the loop.
- The timestamp of the frame in milliseconds is saved using the fps frame index and the fps.
- The OpenCV BGR image is converted to RGB for MediaPipe to use it. The RGB image is then wrapped by the MediaPipe Image object.
- Pose detection is ran on the frame using the MediaPipe Image and timestamp, returning a Pose Landmarker Result.
- If MediaPipe failed to detect a person, the frame counter is incremented and the rest of the processing on the frame is skipped.
- The first detected person is chosen. Each landmark (or person) has x, y, z, and visibility values connected to them.
- The DRAW_POSE_LANDMARKS function is called, passing the frame and landmark. This will draw circles and lines on the person, representing joints and bones. Modifies the frame in place.
- The frame is written into the output video file (the output video is built frame-by-frame).
- The COMPUTE_ANGLE function is called for the left and right knees and ankles. The knees use the hip, knee, and ankle coordinates. The ankles use the knee, ankle, and toe coordinates.
- The WRITE_ANGLES_TO_CSV function is called, which writes the data for each frame into the next row.
- The frame index is incremented.
- The video capture, video writer, OpenCV GUI windows, and the CSV file are all closed. The CSV filename is returned.
- THE PLOT_JOIN_ANGLES function is called, using the returned CSV filename.

LM:
- *Parmeters*: Pose Landmarks, number of the landmark.
- *Returns*: Returns [x, y, z] of the given landmark.
- *Brief*: Helper function to get a 1x3 list of the coordinates of the landmark.

COMPUTE_ANGLE:
- *Parameters*: 3 sets coordinates.
- *Returns*: Angle in degrees.
- *Brief*: Basic angle calculation using 3D coordinates.

CREATE_CSV_WRITER:
- *Parameters*: output directory.
- *Returns*: csv writer, csv filename, csv file.
- *Brief*: Helper to create and open the csv file, set the csv writer, and write the top row of the csv file.

DRAW_POSE_LANDMARKS:
- *Parameters*: Frame, Pose Landmark.
- *Returns*: none.
- *Brief*: Draws each point on pose landmark with a circle. Draws lines between each circle (currently only between major points to save time).

PLOT_JOINT_ANGLES:
- *Parameters*: CSV Path, output directory.
- *Returns*: none.
- *Brief*: Loads the CSV. The knee angles over time and ankle angles over time are each plotted and saved on two .png files with timestamps in the output directory.
