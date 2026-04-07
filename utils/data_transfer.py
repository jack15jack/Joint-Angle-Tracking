import socket
import json
import numpy as np
from utils.pose import compute_angle, lm, angle_in_plane

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# local usage, would need to be expanded for multi-system setup (one computer for motion tracking, one for Unreal)
server_address = ('127.0.0.1', 5005) 

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def signed_angle(a, b, normal):
    
    # Returns signed angle between vectors a and b using reference normal
    
    a_n = normalize(a)
    b_n = normalize(b)

    angle = np.degrees(np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0)))

    cross = np.cross(a_n, b_n)
    sign = np.sign(np.dot(cross, normal))

    return angle * sign

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

    # midpoints of hip and shoulder
    hip_center = np.array([(l_hip[i] + r_hip[i]) / 2 for i in range(3)])
    shoulder_center = np.array([(l_sh[i] + r_sh[i]) / 2 for i in range(3)])
    
    # body axes
    torso = normalize(shoulder_center - hip_center)
    hip_axis = normalize(np.array(r_hip) - np.array(l_hip))
    body_normal = normalize(np.cross(hip_axis, torso))

    # limb vectors
    l_upper_arm = l_elb - l_sh
    r_upper_arm = r_elb - r_sh

    l_thigh = l_knee - l_hip
    r_thigh = r_knee - r_hip
    
    # elbows
    left_elbow = 90 - signed_angle(
        l_sh - l_elb,
        l_wrist - l_elb,
        body_normal
    )
    right_elbow = 90 - signed_angle(
        r_sh - r_elb,
        r_wrist - r_elb,
        body_normal
    )
    # knees (reference model knee is at ~180 degrees)
    left_knee           = 180 - compute_angle(l_hip, l_knee, l_ank)
    right_knee          = 180 - compute_angle(r_hip, r_knee, r_ank)
    # ankles ( # reference model ankle is at ~90 degrees)
    left_ankle          = 90 - compute_angle(l_knee, l_ank, l_toe)
    right_ankle         = 90 - compute_angle(r_knee, r_ank, r_toe)
    # shoulders 
    left_shoulder_flex  = angle_in_plane(torso, l_upper_arm, "sagittal")
    right_shoulder_flex = angle_in_plane(torso, r_upper_arm, "sagittal")
    left_shoulder_abd   = angle_in_plane(torso, l_upper_arm, "frontal")
    right_shoulder_abd  = angle_in_plane(torso, r_upper_arm, "frontal")
    # hips
    left_hip_flex = -1 * (180 -angle_in_plane(torso, l_thigh, "sagittal"))
    right_hip_flex = -1 * (180 - angle_in_plane(torso, r_thigh, "sagittal"))
    
    left_hip_abd = signed_angle(torso, l_thigh, body_normal)
    right_hip_abd = signed_angle(torso, r_thigh, body_normal)

    # locks values between -180 and 180
    def clamp(x, min_val=-180, max_val=180):
        return max(min(x, max_val), min_val)
    
    # create json data
    data = {
        "lk": clamp(left_knee),
        "rk": clamp(right_knee),
        "la": clamp(left_ankle),
        "ra": clamp(right_ankle),
        "le": clamp(left_elbow),
        "re": clamp(right_elbow),
        "lsf": clamp(left_shoulder_flex),
        "rsf": clamp(right_shoulder_flex),
        "lsa": clamp(left_shoulder_abd),
        "rsa": clamp(right_shoulder_abd),
        "lhf": clamp(left_hip_flex),
        "rhf": clamp(right_hip_flex),
        "lha": clamp(left_hip_abd),
        "rha": clamp(right_hip_abd)
    }

    # setup socket and send json
    sock.sendto(json.dumps(data).encode(), server_address)


