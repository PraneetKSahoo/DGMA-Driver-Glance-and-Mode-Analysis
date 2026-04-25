"""
quick_calibrate.py
------------------
Rapid Auto-Calibration using L2CS-Net and Gaussian Mixture Models.
Samples 6 x 30-second windows spread proportionally across the video,
finds the 'On-Road' gaze cluster, saves the thresholds to a JSON file,
and displays a gaze scatter plot + per-window dominance bar chart.
"""

import os
os.add_dll_directory(r'C:/opencv_build/bin/Release')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin')

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import filedialog

from face_analysis_dl import DeepGazeTracker
from preprocessing import crop_to_first_quadrant
from config import DEFAULT_THRESHOLDS

# --- CALIBRATION SETTINGS ---
N_WINDOWS = 3               # Number of 30s windows to sample
WINDOW_DURATION_SEC = 60    # Duration of each window in seconds
FRAME_SKIP = 10             # Process 1 out of every 10 frames (~3fps at 30fps source)
STD_DEV_MULTIPLIER = 3.0    # 3.0 std devs captures ~99% of the on-road cluster
N_CLUSTERS = 2              # On-Road, Off-Road

# Rationale: normal driving should yield ~85%+ on-road frames per window.
# A window dropping below 65% suggests heavy distraction — bad calibration sample.
DOMINANCE_WARNING_THRESHOLD = 0.65


def select_video():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Video for Quick DL Calibration",
        filetypes=[("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def parse_time_input(text, total_duration_sec):
    """Parse seconds, MM:SS, or HH:MM:SS. Returns None if invalid."""
    if text is None:
        return None
    text = text.strip()
    try:
        parts = text.split(":")
        if len(parts) == 1:
            secs = float(parts[0])
        elif len(parts) == 2:
            secs = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            return None
        if secs < 0 or secs > total_duration_sec:
            return None
        return secs
    except (ValueError, TypeError):
        return None


def get_segment_selection(total_duration_sec):
    """Ask the user for start/end times to define the calibration segment."""
    from tkinter import simpledialog
    duration_str = f"{int(total_duration_sec // 60)}:{int(total_duration_sec % 60):02d}"
    hint = (f"Enter time as seconds (e.g. 90) or MM:SS (e.g. 1:30)\n"
            f"Video length: {duration_str}")
    root = tk.Tk()
    root.withdraw()
    start_input = simpledialog.askstring(
        "Calibration Segment - Start",
        f"Start time for calibration segment?\n{hint}\n(Leave blank for 0:00)",
        parent=root
    )
    end_input = simpledialog.askstring(
        "Calibration Segment - End",
        f"End time for calibration segment?\n{hint}\n(Leave blank for end of video)",
        parent=root
    )
    root.destroy()

    seg_start = parse_time_input(start_input, total_duration_sec)
    seg_end   = parse_time_input(end_input,   total_duration_sec)
    if seg_start is None: seg_start = 0.0
    if seg_end   is None: seg_end   = total_duration_sec

    min_required = N_WINDOWS * WINDOW_DURATION_SEC
    if (seg_end - seg_start) < min_required:
        print(f"WARNING: Segment too short ({seg_end - seg_start:.0f}s). "
              f"Need at least {min_required}s for {N_WINDOWS} x {WINDOW_DURATION_SEC}s windows. "
              f"Falling back to full video.")
        seg_start, seg_end = 0.0, total_duration_sec
    return seg_start, seg_end


def extract_window(cap, start_frame, end_frame, gaze_tracker, window_label=""):

    """Seek to start_frame and extract gaze readings up to end_frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    yaw_list, pitch_list = [], []
    frame_count  = start_frame
    total_frames = end_frame - start_frame

    while frame_count < end_frame:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        pct = ((frame_count - start_frame) / total_frames) * 100
        print(f"\r{window_label}  {pct:.1f}% complete...", end="", flush=True)

        frame = crop_to_first_quadrant(frame)
        if frame is None or frame.size == 0:
            continue

        try:
            result = gaze_tracker.process_frame(frame)
            if result:
                yaw_list.append(result['yaw'])
                pitch_list.append(result['pitch'])
        except Exception:
            pass

    return yaw_list, pitch_list

def get_calibration_mode():
    """Ask the user if they want to use the first 3 minutes or a custom segment."""
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    # Returns True for "Yes" (First 3 mins), False for "No" (Custom spread)
    choice = messagebox.askyesno(
        "Calibration Mode",
        "Use the first 3 minutes of the video for calibration?\n\n"
        "(Select 'No' to define a custom segment with spread windows)"
    )
    root.destroy()
    return choice


def main():
    video_path = select_video()
    if not video_path:
        print("No video selected.")
        return

    print(f"Selected: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_sec = total_video_frames / fps
    min_required = N_WINDOWS * WINDOW_DURATION_SEC

    # Single-window mode: video (or selected segment) is too short to split into
    # N_WINDOWS × WINDOW_DURATION_SEC windows. Process the whole segment as one
    # block instead of forcing an artificial split.
    single_window_mode = False

    # Ask user which calibration mode to use
    use_first_3_mins = get_calibration_mode()

    if use_first_3_mins:
        if total_duration_sec < min_required:
            print(f"WARNING: Video is too short ({total_duration_sec:.0f}s) for "
                  f"{N_WINDOWS} x {WINDOW_DURATION_SEC}s windows (need {min_required:.0f}s). "
                  f"Processing full video as a single window.")
            seg_start_sec, seg_end_sec = 0.0, total_duration_sec
            single_window_mode = True
        else:
            seg_start_sec = 0.0
            seg_end_sec = float(N_WINDOWS * WINDOW_DURATION_SEC)
            # Create consecutive windows: [0.0, 60.0, 120.0]
            window_start_times = [float(i * WINDOW_DURATION_SEC) for i in range(N_WINDOWS)]
    else:
        # Ask user to define a calibration segment within the video
        seg_start_sec, seg_end_sec = get_segment_selection(total_duration_sec)

        if (seg_end_sec - seg_start_sec) < min_required:
            print(f"WARNING: Selected segment ({seg_end_sec - seg_start_sec:.0f}s) is too short "
                  f"for {N_WINDOWS} x {WINDOW_DURATION_SEC}s windows (need {min_required:.0f}s). "
                  f"Processing full segment as a single window.")
            single_window_mode = True

    # In single-window mode the whole segment is one window; skip the split.
    if single_window_mode:
        window_start_times = [seg_start_sec]
        effective_window_duration = seg_end_sec - seg_start_sec
    else:
        effective_window_duration = WINDOW_DURATION_SEC
        if not use_first_3_mins:
            # Spread N_WINDOWS evenly within the custom segment
            window_start_times = list(np.linspace(
                seg_start_sec,
                seg_end_sec - WINDOW_DURATION_SEC,
                N_WINDOWS
            ))

    n_windows_actual = len(window_start_times)
    print(f"\n--- Starting Quick DL Calibration ---")
    print(f"Video duration : {total_duration_sec / 60:.1f} min")
    print(f"Segment        : {seg_start_sec:.1f}s - {seg_end_sec:.1f}s  "
          f"({(seg_end_sec - seg_start_sec) / 60:.1f} min)")
    if single_window_mode:
        print(f"Sampling       : 1 window (full segment, {effective_window_duration:.0f}s) — short-video mode")
    else:
        print(f"Sampling       : {n_windows_actual} windows x {WINDOW_DURATION_SEC}s "
              f"= {n_windows_actual * WINDOW_DURATION_SEC}s total")
    print(f"Window starts  : {[f'{t:.1f}s' for t in window_start_times]}")

    print("\nLoading L2CS-Net onto GPU...")
    gaze_tracker = DeepGazeTracker(device='cuda')

    # --- EXTRACT GAZE DATA PER WINDOW ---
    all_yaw, all_pitch = [], []
    window_data = []  # List of (yaw_array, pitch_array) per window
    window_duration_frames = int(effective_window_duration * fps)

    for i, start_sec in enumerate(window_start_times):
        start_frame = int(start_sec * fps)
        end_frame = min(start_frame + window_duration_frames, total_video_frames)
        end_sec = start_sec + effective_window_duration

        label = (f"Extracting window {i + 1}/{n_windows_actual}  "
                 f"({start_sec:.1f}s - {end_sec:.1f}s)")
        yaw_w, pitch_w = extract_window(cap, start_frame, end_frame, gaze_tracker, window_label=label)
        print()  # newline after each window's progress line
        window_data.append((np.array(yaw_w), np.array(pitch_w)))
        all_yaw.extend(yaw_w)
        all_pitch.extend(pitch_w)

    cap.release()
    print("\nExtraction complete.")

    if len(all_yaw) < 10:
        print("Error: Not enough face data found across windows to calibrate.")
        return

    # --- FIT ONE GMM ON ALL COLLECTED DATA ---
    # A single GMM ensures cluster definitions are consistent when we score each window.
    print("\n--- Running GMM Clustering ---")
    X = np.column_stack((all_yaw, all_pitch))
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X)

    counts = np.bincount(labels)
    on_road_cluster_idx = np.argmax(counts)

    # Use GMM's own learned mean and covariance — more principled than recomputing from raw points
    mean_yaw, mean_pitch = gmm.means_[on_road_cluster_idx]
    cov = gmm.covariances_[on_road_cluster_idx]   # Full 2x2 covariance matrix
    std_yaw   = np.sqrt(cov[0, 0])                # Top-left  = yaw variance
    std_pitch = np.sqrt(cov[1, 1])                # Bottom-right = pitch variance

    # --- PER-WINDOW DOMINANCE ---
    # Use gmm.predict() (not fit_predict) so all windows use the same cluster definitions
    print("\n--- Per-Window Dominance ---")
    print(f"{'Win':<5} {'Time Range':<22} {'Points':<8} {'Dominance':<12} Status")
    print("─" * 58)

    window_dominances = []
    n_warnings = 0

    for i, (yaw_w, pitch_w) in enumerate(window_data):
        start_sec = window_start_times[i]
        end_sec   = start_sec + effective_window_duration
        time_str  = f"{start_sec:.1f}s – {end_sec:.1f}s"

        if len(yaw_w) < 2:
            window_dominances.append(None)
            print(f"{i+1:<5} {time_str:<22} {'N/A':<8} {'N/A':<12} ⚠️  Insufficient data")
            n_warnings += 1
            continue

        X_w      = np.column_stack((yaw_w, pitch_w))
        w_labels = gmm.predict(X_w)
        w_counts = np.bincount(w_labels, minlength=N_CLUSTERS)
        dominance = w_counts[on_road_cluster_idx] / len(w_labels)
        window_dominances.append(dominance)

        if dominance >= DOMINANCE_WARNING_THRESHOLD:
            status = "✅ OK"
        else:
            status = "⚠️  LOW"
            n_warnings += 1

        print(f"{i+1:<5} {time_str:<22} {len(yaw_w):<8} {dominance:.1%}{'':6} {status}")

    print("─" * 58)
    if n_warnings > 0:
        print(f"⚠️  WARNING: {n_warnings}/{n_windows_actual} windows below dominance threshold "
              f"({DOMINANCE_WARNING_THRESHOLD:.0%}). Calibration may be unreliable.")
    else:
        print(f"✅ All {n_windows_actual} windows passed dominance check.")

    # --- COMPUTE THRESHOLDS ---
    yaw_low   = round(mean_yaw   - (std_yaw   * STD_DEV_MULTIPLIER), 1)
    yaw_high  = round(mean_yaw   + (std_yaw   * STD_DEV_MULTIPLIER), 1)
    pitch_low  = round(mean_pitch - (std_pitch * STD_DEV_MULTIPLIER), 1)
    pitch_high = round(mean_pitch + (std_pitch * STD_DEV_MULTIPLIER), 1)

    print("\n--- Calibration Results ---")
    print(f"Forward Center -> Yaw: {mean_yaw:.1f}°, Pitch: {mean_pitch:.1f}°")
    print(f"Yaw Bounds     -> [{yaw_low}°, {yaw_high}°]")
    print(f"Pitch Bounds   -> [{pitch_low}°, {pitch_high}°]")

    # --- SAVE JSON ---
    calibrated_thresholds = {
        "yaw_low":   yaw_low,   
        "yaw_high":   yaw_high,
        "pitch_low": pitch_low, 
        "pitch_high": pitch_high,
        "camera_yaw_offset":   round(float(mean_yaw), 1),
        "camera_pitch_offset": round(float(mean_pitch), 1)
    }

    video_name, _ = os.path.splitext(os.path.basename(video_path))
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"thresholds_{video_name}_{timestamp}.json"
    with open(out_file, 'w') as f:
        json.dump(calibrated_thresholds, f, indent=4)
    print(f"\n✅ Thresholds saved to '{out_file}'")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Quick Auto-Calibration: {video_name}", fontsize=14, fontweight='bold')

    # ── Plot 1: Gaze Scatter ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                alpha=0.5, s=15, label='L2CS-Net Gaze Vectors')
    ax1.plot(mean_yaw, mean_pitch, 'ko', markersize=8,
             label='Calibrated Forward Center')

    rect = Rectangle(
        (yaw_low, pitch_low),
        (yaw_high - yaw_low), (pitch_high - pitch_low),
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
        label='Rectangular Zone'
    )
    ax1.add_patch(rect)

    ellipse = Ellipse(
        (mean_yaw, mean_pitch),
        width=(yaw_high - yaw_low), height=(pitch_high - pitch_low),
        linewidth=3, edgecolor='green', facecolor='none',
        label='Elliptical Cone of Vision'
    )
    ax1.add_patch(ellipse)

    ax1.set_title("Gaze Cluster Map")
    ax1.set_xlabel("Gaze Yaw (Degrees) — Left / Right")
    ax1.set_ylabel("Gaze Pitch (Degrees) — Up / Down")
    ax1.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax1.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_aspect('equal', adjustable='datalim')

    # ── Plot 2: Per-Window Dominance Bar Chart ────────────────────────────────
    ax2 = axes[1]

    valid = [(i, d) for i, d in enumerate(window_dominances) if d is not None]
    win_nums       = [i + 1 for i, _ in valid]
    win_dominances = [d      for _, d in valid]
    win_labels     = [f"W{i+1}\n{window_start_times[i]:.0f}s" for i, _ in valid]
    bar_colors     = ['#2ecc71' if d >= DOMINANCE_WARNING_THRESHOLD
                      else '#e74c3c' for d in win_dominances]

    bars = ax2.bar(win_nums, win_dominances, color=bar_colors,
                   edgecolor='black', linewidth=0.8, width=0.6)

    # Threshold reference line
    ax2.axhline(DOMINANCE_WARNING_THRESHOLD, color='orange', linewidth=2,
                linestyle='--', label=f'Warning Threshold ({DOMINANCE_WARNING_THRESHOLD:.0%})')

    # Value labels above each bar
    for bar, dom in zip(bars, win_dominances):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + 0.015,
            f'{dom:.1%}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    ax2.set_xticks(win_nums)
    ax2.set_xticklabels(win_labels, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("Window (Start Time)")
    ax2.set_ylabel("On-Road Dominance")
    ax2.set_title("Per-Window On-Road Dominance")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Colour legend for bars
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor='#2ecc71', edgecolor='black', label='Passed'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Failed'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold ({DOMINANCE_WARNING_THRESHOLD:.0%})')
    ], fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()