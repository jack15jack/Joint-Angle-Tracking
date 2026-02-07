import os
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def create_csv_writer(OUTPUT_DIR):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"joint_angles_{timestamp}.csv")
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "time_ms", "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ])
    return writer, filename, f

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