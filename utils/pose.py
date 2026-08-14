import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

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
        

#TODO: cycle variability
#TODO: posture measurements
#TODO: ROM
#TODO: step timing, cadence
#TODO: vertical lift height


