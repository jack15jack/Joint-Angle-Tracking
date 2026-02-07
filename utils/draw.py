import cv2

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
        (27, 29), (28, 30),  # ankle-heel line
        (28, 32), (30, 32)   # right feet
    ]

    # Draw lines
    for start_idx, end_idx in connections:
        x1, y1 = int(pose_landmarks[start_idx].x * w), int(pose_landmarks[start_idx].y * h)
        x2, y2 = int(pose_landmarks[end_idx].x * w), int(pose_landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)