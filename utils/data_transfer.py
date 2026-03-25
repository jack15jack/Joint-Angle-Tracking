import socket
import json
import numpy as np
from utils.pose import compute_angle, lm, angle_in_plane

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# local usage, would need to be expanded for multi-system setup (one computer for motion tracking, one for Unreal)
server_address = ('127.0.0.1', 5005) 

# calculate needed angles, create a json packet, and send it to unreal via UDP socket
def transfer_data(pose_lm):

    # names of landmarks for readability
    l_sh  = lm(pose_lm, 11)
    r_sh = lm(pose_lm, 12)
    l_elb = lm(pose_lm, 13)
    r_elb = lm(pose_lm, 14)
    l_wrist = lm(pose_lm, 15)
    r_wrist = lm(pose_lm, 16)
    l_hip  = lm(pose_lm, 23)
    r_hip = lm(pose_lm, 24)
    l_knee  = lm(pose_lm, 25)
    r_knee = lm(pose_lm, 26)
    l_ank  = lm(pose_lm, 27)
    r_ank = lm(pose_lm, 28)
    l_toe  = lm(pose_lm, 31)
    r_toe = lm(pose_lm, 32)

    # midpoints of hip and shoulder
    hip_center = [(l_hip[i] + r_hip[i]) / 2 for i in range(3)]
    shoulder_center = [(l_sh[i] + r_sh[i]) / 2 for i in range(3)]
    
    # calculated vectors (for use in angle in plane calculations)
    torso = np.array([shoulder_center[i] - hip_center[i] for i in range(3)])
    l_upper_arm = np.array([l_elb[i] - l_sh[i] for i in range(3)])
    r_upper_arm = np.array([r_elb[i] - r_sh[i] for i in range(3)])
    l_thigh = np.array([l_knee[i] - l_hip[i] for i in range(3)])
    r_thigh = np.array([r_knee[i] - r_hip[i] for i in range(3)])
    
    # calculate needed angles
    left_elbow          = compute_angle(l_sh, l_elb, l_wrist)
    right_elbow         = compute_angle(r_sh, r_elb, r_wrist)
    left_knee           = compute_angle(l_hip, l_knee, l_ank)
    right_knee          = compute_angle(r_hip, r_knee, r_ank)
    left_ankle          = compute_angle(l_knee, l_ank, l_toe)
    right_ankle         = compute_angle(r_knee, r_ank, r_toe)
    left_shoulder_flex  = angle_in_plane(torso, l_upper_arm, "sagittal")
    right_shoulder_flex = angle_in_plane(torso, r_upper_arm, "sagittal")
    left_shoulder_abd   = angle_in_plane(torso, l_upper_arm, "frontal")
    right_shoulder_abd  = angle_in_plane(torso, r_upper_arm, "frontal")
    left_hip_flex       = angle_in_plane(torso, l_thigh, "sagittal")
    right_hip_flex      = angle_in_plane(torso, r_thigh, "sagittal")
    left_hip_abd        = angle_in_plane(torso, l_thigh, "frontal")
    right_hip_abd       = angle_in_plane(torso, r_thigh, "frontal")


    # create json data
    data = {
        "lk": left_knee,
        "rk": right_knee,
        "la": left_ankle,
        "ra": right_ankle,
        "le": left_elbow,
        "re": right_elbow,
        "lsf": left_shoulder_flex,
        "rsf": right_shoulder_flex,
        "lsa": left_shoulder_abd,
        "rsa": right_shoulder_abd,
        "lhf": left_hip_flex,
        "rhf": right_hip_flex,
        "lha": left_hip_abd,
        "rha": right_hip_abd
    }

    # setup socket and send json
    sock.sendto(json.dumps(data).encode(), server_address)


