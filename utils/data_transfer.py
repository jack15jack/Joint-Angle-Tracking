import socket
import json
import numpy as np
from utils.pose import compute_angle, lm

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# local usage, would need to be expanded for multi-system setup (one computer for motion tracking, one for Unreal)
server_address = ('127.0.0.1', 5005) 

# calculate needed angles, create a json packet, and send it to unreal via UDP socket
def transfer_data(pose_lm):

    # names of landmarks for readability
    l_sh  = np.array(lm(pose_lm, 11))
    r_sh = np.array(lm(pose_lm, 12))
    l_elb = np.array(lm(pose_lm, 13))
    r_elb = np.array(lm(pose_lm, 14))
    l_wrist = np.array(lm(pose_lm, 15))
    r_wrist = np.array(lm(pose_lm, 16))
    l_hip  = np.array(lm(pose_lm, 23))
    r_hip = np.array(lm(pose_lm, 24))
    l_knee  = np.array(lm(pose_lm, 25))
    r_knee = np.array(lm(pose_lm, 26))
    l_ank  = np.array(lm(pose_lm, 27))
    r_ank = np.array(lm(pose_lm, 28))
    l_toe  = np.array(lm(pose_lm, 31))
    r_toe = np.array(lm(pose_lm, 32))

    # knees (reference model knee is at ~180 degrees)
    left_knee           = 180 - compute_angle(l_hip, l_knee, l_ank)
    right_knee          = 180 - compute_angle(r_hip, r_knee, r_ank)
    # ankles ( # reference model ankle is at ~90 degrees)
    left_ankle          = compute_angle(l_knee, l_ank, l_toe) - 90
    right_ankle         = compute_angle(r_knee, r_ank, r_toe) - 90
    # hips
    left_hip_flex = -1 * (180 - compute_angle(l_sh, l_hip, l_knee))
    right_hip_flex = -1 * (180 - compute_angle(r_sh, r_hip, r_knee))

    # locks values between -180 and 180
    def clamp(x, min_val=-180, max_val=180):
        return max(min(x, max_val), min_val)
    
    # create json data
    data = {
        "lk": clamp(left_knee),
        "rk": clamp(right_knee),
        "la": clamp(left_ankle),
        "ra": clamp(right_ankle),
        "le": 0,
        "re": 0,
        "lsf": 0,
        "rsf": 0,
        "lsa": 0,
        "rsa": 0,
        "lhf": clamp(left_hip_flex),
        "rhf": clamp(right_hip_flex),
        "lha": 0,
        "rha": 0
    }

    # setup socket and send json
    sock.sendto(json.dumps(data).encode(), server_address)


