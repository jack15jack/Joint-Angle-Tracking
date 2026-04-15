import os
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_csv_writer(OUTPUT_DIR):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"joint_angles_{timestamp}.csv")
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "time_ms", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_hip", "right_hip"
    ])
    return writer, filename, f

def build_percent_cycle_csv(OUTPUT_DIR, normalized_cycles):
    percent = np.linspace(0, 100, 100)

    mean_cycle = np.mean(normalized_cycles, axis=0)
    
    output_csv = os.path.join(OUTPUT_DIR, "isolated_gait_cycles.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["Percent_Gait"]
        for i in range(len(normalized_cycles)):
            header.append(f"Cycle_{i+1}")
        header.append("Mean")

        writer.writerow(header)

        for i in range(100):
            row = [percent[i]]
            row.extend(normalized_cycles[:, i])
            row.append(mean_cycle[i])
            writer.writerow(row)

def plot_joint_angles(OUTPUT_DIR, csv_path="joint_angles.csv"):
    # Load the CSV
    df = pd.read_csv(csv_path)
    time_s = df["time_ms"] / 1000  # convert to seconds

    # timestamp for filenames
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot knees
    plt.figure(figsize=(10,5))
    plt.plot(time_s, df["left_knee"], label="Left Knee", color="blue")
    plt.plot(time_s, df["right_knee"], label="Right Knee", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Knee Angles Over Time")
    plt.legend()
    plt.grid(True)

    knee_filename = os.path.join(OUTPUT_DIR, f"knee_angles_{timestamp_str}.png")
    plt.savefig(knee_filename)
    plt.close()  # close figure to free memory
    print(f"Saved knee angles plot as {knee_filename}")

    # Plot ankles
    plt.figure(figsize=(10,5))
    plt.plot(time_s, df["left_ankle"], label="Left Ankle", color="blue")
    plt.plot(time_s, df["right_ankle"], label="Right Ankle", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Ankle Angles Over Time")
    plt.legend()
    plt.grid(True)

    ankle_filename = os.path.join(OUTPUT_DIR, f"ankle_angles_{timestamp_str}.png")
    plt.savefig(ankle_filename)
    plt.close()
    print(f"Saved ankle angles plot as {ankle_filename}")

    # Plot hips
    plt.figure(figsize=(10,5))
    plt.plot(time_s, df["left_hip"], label="Left Hip", color="blue")
    plt.plot(time_s, df["right_hip"], label="Right Hip", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Hip Angles Over Time")
    plt.legend()
    plt.grid(True)

    hip_filename = os.path.join(OUTPUT_DIR, f"hip_angles_{timestamp_str}.png")
    plt.savefig(hip_filename)
    plt.close()  # close figure to free memory
    print(f"Saved hip angles plot as {hip_filename}")

def plot_iso_cycles(OUTPUT_DIR, normalized_cycles):
    percent = np.linspace(0, 100, 100)
    mean_cycle = np.mean(normalized_cycles, axis=0)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(8,6))

    # Plot individual cycles lightly
    for cycle in normalized_cycles:
        plt.plot(percent, cycle, alpha=0.25)

    # Plot mean prominently
    plt.plot(percent, mean_cycle, linewidth=4)

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Knee Flexion Angle (deg)")
    plt.title("Overlaid Gait Cycles with Mean")
    plt.grid(True)

    iso_filename = os.path.join(OUTPUT_DIR, f"gait_cycles_{timestamp_str}.png")
    plt.savefig(iso_filename)
    plt.close()
    print(f"Saved gait cycles plot as {iso_filename}")