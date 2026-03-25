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

# project vectors onto sagittal or frontal planes, returning a signed angle 
def angle_in_plane(a, b, plane="sagittal"):
    a = np.array(a) 
    b = np.array(b) 
    if plane == "sagittal": 
        a = a[[1,2]] 
        b = b[[1,2]] 
    elif plane == "frontal": 
        a = a[[0,1]] 
        b = b[[0,1]] 

    angle = np.degrees(np.arccos( np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) )) 
    
    #sign from 2D cross product 
    cross = np.cross(np.append(a, 0), np.append(b, 0)) 
    return angle * np.sign(cross[2])

# isolates gait cycles, returning a list of cycles
def iso_gait_cycles(knee_data, fps):
    knee_smooth = savgol_filter(knee_data, window_length=11, polyorder=3)
    minima, _ = find_peaks(-knee_smooth, distance=fps*0.6, prominence=5)

    cycles = []

    for i in range(len(minima)-1):
        start = minima[i]
        end = minima[i+1]
        cycle = knee_smooth[start:end]
        cycles.append(cycle)
    return cycles

# normalizes a list of cycles into a percentage scale
def normalize_gait_cycles(cycles, fps):
    normalized_cycles = []

    for cycle in cycles:
        # skip noisy segments
        if len(cycle) < fps * 0.3:
            continue
        # normalize to 100 samples (0-100% gait)
        x_original = np.linspace(0, 1, len(cycle))
        f = interp1d(x_original, cycle)
        x_new = np.linspace(0, 1, 100)
        normalized = f(x_new)
        normalized_cycles.append(normalized)

    normalized_cycles_np = np.array(normalized_cycles)
    return normalized_cycles_np

#TODO: cycle variability
#TODO: posture measurements
#TODO: ROM for knees and ankles
#TODO: foot velocity, acceleration
#TODO: step timing, cadence
#TODO: vertical lift height


