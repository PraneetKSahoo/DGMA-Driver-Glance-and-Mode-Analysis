"""
batch.py
--------
Batch Processing GUI for running Quick Calibration or Main Processing
on multiple videos without interactive prompts.

Creates one DeepGazeTracker (expensive GPU load) and reuses it across
all videos. Fresh TrackingState + GlanceTracker per video.
"""

import os
os.add_dll_directory(r'C:/opencv_build/bin/Release')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin')

import matplotlib
matplotlib.use('Agg')

import cv2
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Patch
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from quick_calibrate import (
    extract_window, parse_time_input,
    N_WINDOWS, WINDOW_DURATION_SEC, FRAME_SKIP, STD_DEV_MULTIPLIER,
    N_CLUSTERS, DOMINANCE_WARNING_THRESHOLD,
)
from face_analysis_dl import DeepGazeTracker
from io_utils import (
    open_video, save_gaze_csv, save_glance_history_csv,
    save_mode_intervals_csv, save_mode_glance_correlation_csv,
    save_report_long_csv, save_report_json,
)
from config import (
    ThresholdConfig, DEFAULT_THRESHOLDS, CAR_TYPES, DEFAULT_CAR_TYPE,
    DEFAULT_DELTA_INTERVAL, POSE_SMOOTHING_FACTOR,
    GLANCE_MIN_DURATION, CONFIDENCE_HISTORY_LENGTH, FRAME_SKIP_FACTOR,
    MODE_CHECK_INTERVAL_SECONDS,
)
from state import TrackingState
from glance import GlanceTracker
from preprocessing import crop_to_first_quadrant, setup_upscaler, upscale_frame

from datetime import datetime


# =============================================================================
# TextRedirector — pipes print() to ScrolledText widget
# =============================================================================

class TextRedirector:
    """Redirect stdout writes to a tkinter ScrolledText widget (thread-safe)."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        if text:
            self.widget.after(0, self._append, text)

    def _append(self, text):
        self.widget.configure(state='normal')
        # Handle \r (carriage return) by replacing the current line in-place
        if '\r' in text:
            # Split on \r — last segment is the one to keep
            parts = text.split('\r')
            for part in parts[:-1]:
                # Delete current line content before writing replacement
                self.widget.delete("end-1c linestart", "end-1c")
            text = parts[-1]
        if text:
            self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass


# =============================================================================
# calibrate_single_video — replicates quick_calibrate.py logic
# =============================================================================

def calibrate_single_video(video_path, gaze_tracker, start_time_str, end_time_str,
                           cancel_flag, log_fn=print):
    """Run calibration on a single video and save thresholds JSON + plot PNG.

    Returns dict with 'success', and on success 'threshold_path' / 'plot_path'.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    log_fn(f"[Calibrate] Starting: {video_name}")

    try:
        cap, total_frames, fps = open_video(video_path)
    except RuntimeError as e:
        return {'success': False, 'error': str(e)}

    total_duration_sec = total_frames / fps

    # Parse segment
    seg_start = parse_time_input(start_time_str, total_duration_sec) if start_time_str else None
    seg_end = parse_time_input(end_time_str, total_duration_sec) if end_time_str else None
    if seg_start is None:
        seg_start = 0.0
    if seg_end is None:
        seg_end = total_duration_sec

    min_required = N_WINDOWS * WINDOW_DURATION_SEC
    single_window_mode = False

    if (seg_end - seg_start) < min_required:
        log_fn(f"  Segment too short ({seg_end - seg_start:.0f}s) for "
               f"{N_WINDOWS}x{WINDOW_DURATION_SEC}s windows. Using single-window mode.")
        single_window_mode = True

    if single_window_mode:
        window_start_times = [seg_start]
        effective_window_duration = seg_end - seg_start
    else:
        effective_window_duration = WINDOW_DURATION_SEC
        window_start_times = list(np.linspace(
            seg_start, seg_end - WINDOW_DURATION_SEC, N_WINDOWS
        ))

    n_windows_actual = len(window_start_times)
    log_fn(f"  Segment: {seg_start:.1f}s - {seg_end:.1f}s  "
           f"({(seg_end - seg_start) / 60:.1f} min)")
    log_fn(f"  Windows: {n_windows_actual} x {effective_window_duration:.0f}s")

    # Extract gaze data per window
    all_yaw, all_pitch = [], []
    window_data = []
    window_duration_frames = int(effective_window_duration * fps)

    for i, start_sec in enumerate(window_start_times):
        if cancel_flag.is_set():
            cap.release()
            return {'success': False, 'error': 'Cancelled'}

        start_frame = int(start_sec * fps)
        end_frame = min(start_frame + window_duration_frames, total_frames)
        end_sec = start_sec + effective_window_duration

        label = (f"  Window {i + 1}/{n_windows_actual} "
                 f"({start_sec:.1f}s - {end_sec:.1f}s)")
        yaw_w, pitch_w = extract_window(cap, start_frame, end_frame,
                                        gaze_tracker, window_label=label)
        log_fn("")  # newline after progress
        window_data.append((np.array(yaw_w), np.array(pitch_w)))
        all_yaw.extend(yaw_w)
        all_pitch.extend(pitch_w)

    cap.release()

    if len(all_yaw) < 10:
        return {'success': False, 'error': 'Not enough face data to calibrate'}

    # Fit GMM
    X = np.column_stack((all_yaw, all_pitch))
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full',
                          random_state=42)
    labels = gmm.fit_predict(X)

    counts = np.bincount(labels)
    on_road_idx = np.argmax(counts)

    mean_yaw, mean_pitch = gmm.means_[on_road_idx]
    cov = gmm.covariances_[on_road_idx]
    std_yaw = np.sqrt(cov[0, 0])
    std_pitch = np.sqrt(cov[1, 1])

    # Per-window dominance
    window_dominances = []
    n_warnings = 0
    for i, (yaw_w, pitch_w) in enumerate(window_data):
        if len(yaw_w) < 2:
            window_dominances.append(None)
            n_warnings += 1
            continue
        X_w = np.column_stack((yaw_w, pitch_w))
        w_labels = gmm.predict(X_w)
        w_counts = np.bincount(w_labels, minlength=N_CLUSTERS)
        dominance = w_counts[on_road_idx] / len(w_labels)
        window_dominances.append(dominance)
        if dominance < DOMINANCE_WARNING_THRESHOLD:
            n_warnings += 1
        log_fn(f"  Window {i+1} dominance: {dominance:.1%}"
               f"{'  WARNING: LOW' if dominance < DOMINANCE_WARNING_THRESHOLD else ''}")

    if n_warnings > 0:
        log_fn(f"  WARNING: {n_warnings}/{n_windows_actual} windows below threshold")

    # Compute thresholds
    yaw_low = round(mean_yaw - (std_yaw * STD_DEV_MULTIPLIER), 1)
    yaw_high = round(mean_yaw + (std_yaw * STD_DEV_MULTIPLIER), 1)
    pitch_low = round(mean_pitch - (std_pitch * STD_DEV_MULTIPLIER), 1)
    pitch_high = round(mean_pitch + (std_pitch * STD_DEV_MULTIPLIER), 1)

    log_fn(f"  Center -> Yaw: {mean_yaw:.1f}, Pitch: {mean_pitch:.1f}")
    log_fn(f"  Bounds -> Yaw: [{yaw_low}, {yaw_high}], Pitch: [{pitch_low}, {pitch_high}]")

    # Save JSON
    calibrated = {
        "yaw_low": yaw_low, "yaw_high": yaw_high,
        "pitch_low": pitch_low, "pitch_high": pitch_high,
        "camera_yaw_offset": round(float(mean_yaw), 1),
        "camera_pitch_offset": round(float(mean_pitch), 1),
    }
    thresh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Thresholds")
    os.makedirs(thresh_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(thresh_dir, f"thresholds_{video_name}_{timestamp}.json")
    with open(out_file, 'w') as f:
        json.dump(calibrated, f, indent=4)
    log_fn(f"  Thresholds saved: {out_file}")

    # Save plot
    os.makedirs("Plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Quick Auto-Calibration: {video_name}", fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, s=15)
    ax1.plot(mean_yaw, mean_pitch, 'ko', markersize=8, label='Forward Center')
    rect = Rectangle((yaw_low, pitch_low), yaw_high - yaw_low, pitch_high - pitch_low,
                      linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
                      label='Rectangular Zone')
    ax1.add_patch(rect)
    ellipse = Ellipse((mean_yaw, mean_pitch), width=yaw_high - yaw_low,
                      height=pitch_high - pitch_low, linewidth=3, edgecolor='green',
                      facecolor='none', label='Elliptical Cone')
    ax1.add_patch(ellipse)
    ax1.set_title("Gaze Cluster Map")
    ax1.set_xlabel("Yaw (deg)")
    ax1.set_ylabel("Pitch (deg)")
    ax1.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax1.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_aspect('equal', adjustable='datalim')

    ax2 = axes[1]
    valid = [(i, d) for i, d in enumerate(window_dominances) if d is not None]
    if valid:
        win_nums = [i + 1 for i, _ in valid]
        win_doms = [d for _, d in valid]
        win_labels_plot = [f"W{i+1}\n{window_start_times[i]:.0f}s" for i, _ in valid]
        bar_colors = ['#2ecc71' if d >= DOMINANCE_WARNING_THRESHOLD else '#e74c3c'
                      for d in win_doms]
        bars = ax2.bar(win_nums, win_doms, color=bar_colors, edgecolor='black',
                       linewidth=0.8, width=0.6)
        ax2.axhline(DOMINANCE_WARNING_THRESHOLD, color='orange', linewidth=2,
                    linestyle='--')
        for bar, dom in zip(bars, win_doms):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.015,
                     f'{dom:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_xticks(win_nums)
        ax2.set_xticklabels(win_labels_plot, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel("Window")
    ax2.set_ylabel("On-Road Dominance")
    ax2.set_title("Per-Window Dominance")
    ax2.legend(handles=[
        Patch(facecolor='#2ecc71', edgecolor='black', label='Passed'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Failed'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold ({DOMINANCE_WARNING_THRESHOLD:.0%})')
    ], fontsize=9)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"Plots/calibration_{video_name}_{timestamp}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log_fn(f"  Plot saved: {plot_path}")

    return {'success': True, 'threshold_path': out_file, 'plot_path': plot_path}


# =============================================================================
# process_single_video — replicates main.py main loop (headless)
# =============================================================================

def process_single_video(video_path, gaze_tracker, prefs, threshold_path,
                         start_time_str, end_time_str, cancel_flag, log_fn=print,
                         custom_rois=None):
    """Run main processing on a single video (no visualization).

    Returns dict with 'success', and on success 'gaze_csv' / 'glance_csv'.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_basename = os.path.basename(video_path)
    log_fn(f"[Process] Starting: {video_name}")

    # Load thresholds
    if threshold_path and os.path.isfile(threshold_path):
        try:
            with open(threshold_path, 'r') as f:
                thresh = json.load(f)
            config = ThresholdConfig(thresh)
            log_fn(f"  Loaded thresholds from {os.path.basename(threshold_path)}")
        except Exception as e:
            log_fn(f"  Failed to load thresholds: {e}. Using defaults.")
            config = ThresholdConfig(DEFAULT_THRESHOLDS)
    else:
        config = ThresholdConfig(DEFAULT_THRESHOLDS)
        log_fn(f"  Using default thresholds")

    try:
        cap, total_frames, video_fps = open_video(video_path)
    except RuntimeError as e:
        return {'success': False, 'error': str(e)}

    total_duration_sec = total_frames / video_fps

    # Parse segment
    segment_start_frame = 0
    segment_end_frame = total_frames

    if start_time_str and start_time_str.strip():
        parsed = parse_time_input(start_time_str, total_duration_sec)
        if parsed is not None:
            segment_start_frame = int(parsed * video_fps)

    if end_time_str and end_time_str.strip():
        parsed = parse_time_input(end_time_str, total_duration_sec)
        if parsed is not None:
            end_frame = int(parsed * video_fps)
            if end_frame > segment_start_frame:
                segment_end_frame = end_frame

    if segment_start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start_frame)

    # Setup upscaler
    sr = setup_upscaler(prefs['upscale_factor']) if prefs.get('enable_upscale') else None

    # Mode detection (do NOT touch mode_detection.py — import only if enabled)
    mode_detector = None
    mode_tracker = None
    mode_check_frame_interval = int(video_fps * MODE_CHECK_INTERVAL_SECONDS) if video_fps > 0 else 300
    if prefs.get('enable_mode_detection'):
        from mode_detection import ModeDetector, ModeTracker
        mode_detector = ModeDetector(car_type=prefs.get('car_type', DEFAULT_CAR_TYPE))
        if custom_rois is not None:
            mode_detector.update_rois(custom_rois)
        mode_tracker = ModeTracker()

    # Fresh state per video
    state = TrackingState(
        delta_interval=prefs.get('delta_interval', DEFAULT_DELTA_INTERVAL),
        pose_alpha=POSE_SMOOTHING_FACTOR,
        confidence_history_len=CONFIDENCE_HISTORY_LENGTH,
    )
    glance_tracker = GlanceTracker(min_duration=GLANCE_MIN_DURATION)

    resolved = config.get_active_thresholds_resolved()
    yaw_range = (resolved['yaw_low'], resolved['yaw_high'])
    pitch_range = (resolved['pitch_low'], resolved['pitch_high'])

    frame_count = segment_start_frame
    collected_data = []
    last_progress_pct = -1

    seg_start_sec = segment_start_frame / video_fps
    seg_end_sec = segment_end_frame / video_fps
    log_fn(f"  Segment: {seg_start_sec:.1f}s - {seg_end_sec:.1f}s")

    try:
        while True:
            if cancel_flag.is_set():
                cap.release()
                return {'success': False, 'error': 'Cancelled'}

            success, frame_original = cap.read()
            if not success or frame_count >= segment_end_frame:
                break
            frame_count += 1

            if frame_count % FRAME_SKIP_FACTOR != 0:
                continue

            # Check cancel every ~100 frames
            if frame_count % 100 == 0 and cancel_flag.is_set():
                cap.release()
                return {'success': False, 'error': 'Cancelled'}

            try:
                frame = crop_to_first_quadrant(frame_original)
                if frame is None or frame.size == 0:
                    continue
                if sr:
                    frame = upscale_frame(sr, frame, prefs['upscale_factor'])

                current_time = frame_count / video_fps

                # Deep learning inference
                gaze_result = gaze_tracker.process_frame(frame)

                raw_yaw, raw_pitch, raw_roll = 0.0, 0.0, 0.0
                confidence = 0.0

                if gaze_result and gaze_result['confidence'] >= 0.50:
                    raw_yaw = gaze_result['yaw']
                    raw_pitch = gaze_result['pitch']
                    confidence = gaze_result['confidence']

                    state.update_pose(raw_yaw, raw_pitch, raw_roll, video_fps)

                    raw_zone = glance_tracker.classify_zone(state, yaw_range, pitch_range)
                    glance_zone, grace_confidence = glance_tracker.apply_grace_period(raw_zone, current_time)
                    if grace_confidence is not None:
                        confidence = grace_confidence

                    glance_tracker.last_valid_confidence = confidence
                    state.update_confidence(confidence)

                    row = state.get_collected_row(current_time)
                    if row:
                        collected_data.append(row)
                else:
                    state.clear_on_no_detection()
                    glance_zone, grace_confidence = glance_tracker.apply_grace_period("Undetermined", current_time)
                    state.update_confidence(grace_confidence if grace_confidence else 0.0)

                # Mode detection on full frame at configured interval
                if mode_detector is not None and (mode_tracker.current_mode is None or frame_count % mode_check_frame_interval == 0):
                    detected_mode, _ = mode_detector.detect(frame_original)
                    mode_tracker.update(detected_mode, current_time, frame_count)

                glance_tracker.update(glance_zone, current_time, frame_count,
                                     state.smoothed_confidence, state, mode_tracker)
                state.commit_prev_values()

            except Exception as e:
                pass  # skip frame errors silently

            # In-place progress update using \r (TextRedirector handles overwrite)
            seg_total = segment_end_frame - segment_start_frame
            if seg_total > 0:
                pct = (frame_count - segment_start_frame) / seg_total * 100
                pct_int = int(pct)
                if pct_int > last_progress_pct:
                    last_progress_pct = pct_int
                    bar_w = 30
                    filled = int(bar_w * pct / 100)
                    bar = '\u2588' * filled + '\u2591' * (bar_w - filled)
                    print(f"\r  Processing: [{bar}] {pct:.1f}%", end="", flush=True)

    except KeyboardInterrupt:
        pass

    log_fn("")  # newline after progress bar
    final_time = frame_count / video_fps

    if mode_tracker is not None:
        mode_tracker.flush(final_time, frame_count)

    glance_tracker.flush(final_time, frame_count, state.smoothed_confidence, mode_tracker)

    # Build and print report
    segment_info = {
        'video_file': video_basename,
        'segment_start': segment_start_frame / video_fps,
        'segment_end': final_time,
        'duration': final_time - (segment_start_frame / video_fps),
    }
    report = glance_tracker.build_report({}, segment_info, mode_tracker)
    glance_tracker.print_report(report)

    # Save CSVs
    gaze_csv = save_gaze_csv(collected_data, video_name=video_basename)
    glance_csv = save_glance_history_csv(glance_tracker.glance_history, video_name=video_basename)

    # Mode-specific exports (only when mode detection is enabled and has data)
    if mode_tracker is not None:
        save_mode_intervals_csv(mode_tracker.mode_intervals, video_name=video_basename)
        correlation = glance_tracker.correlate_glances_with_modes()
        save_mode_glance_correlation_csv(correlation, video_name=video_basename)

    # Report exports (always)
    save_report_long_csv(report, video_name=video_basename)
    save_report_json(report, video_name=video_basename)

    cap.release()
    log_fn(f"  Done: {video_name}")

    return {'success': True, 'gaze_csv': gaze_csv, 'glance_csv': glance_csv}


# =============================================================================
# BatchRunner — background thread for processing
# =============================================================================

class BatchRunner(threading.Thread):
    """Runs calibration or main processing on a list of videos in a background thread."""

    def __init__(self, mode, video_entries, prefs, cancel_flag, root,
                 on_progress, on_batch_complete, log_fn=print):
        super().__init__(daemon=True)
        self.mode = mode  # 'calibrate' or 'process'
        self.video_entries = video_entries
        self.prefs = prefs
        self.cancel_flag = cancel_flag
        self.root = root
        self.on_progress = on_progress
        self.on_batch_complete = on_batch_complete
        self.log_fn = log_fn

    def run(self):
        results = []
        try:
            self.log_fn("Loading L2CS-Net onto GPU...")
            gaze_tracker = DeepGazeTracker(device='cuda')
            self.log_fn("Model loaded.\n")

            total = len(self.video_entries)
            for idx, entry in enumerate(self.video_entries):
                if self.cancel_flag.is_set():
                    self.log_fn("\nBatch cancelled by user.")
                    break

                self.log_fn(f"\n[{idx + 1}/{total}] {os.path.basename(entry['video'])}")
                self.root.after(0, self.on_progress, idx, total)

                if self.mode == 'calibrate':
                    result = calibrate_single_video(
                        video_path=entry['video'],
                        gaze_tracker=gaze_tracker,
                        start_time_str=entry.get('start', ''),
                        end_time_str=entry.get('end', ''),
                        cancel_flag=self.cancel_flag,
                        log_fn=self.log_fn,
                    )
                else:
                    result = process_single_video(
                        video_path=entry['video'],
                        gaze_tracker=gaze_tracker,
                        prefs=self.prefs,
                        threshold_path=entry.get('threshold', ''),
                        start_time_str=entry.get('start', ''),
                        end_time_str=entry.get('end', ''),
                        cancel_flag=self.cancel_flag,
                        log_fn=self.log_fn,
                        custom_rois=entry.get('custom_rois'),
                    )

                result['video'] = entry['video']
                results.append(result)

                if not result['success']:
                    self.log_fn(f"  FAILED: {result.get('error', 'Unknown error')}")

        except Exception as e:
            self.log_fn(f"\nFatal error: {e}")
            results.append({'success': False, 'error': str(e), 'video': 'batch_runner'})
        finally:
            self.root.after(0, self.on_batch_complete, results)


# =============================================================================
# BatchApp — main GUI
# =============================================================================

class BatchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Processor")
        self.geometry("850x700")
        self.minsize(750, 600)

        self.video_entries = []  # list of dicts: {video, threshold, start, end}
        self.cancel_flag = threading.Event()
        self.runner = None

        self._build_mode_selector()
        self._build_preferences_panel()
        self._build_video_table()
        self._build_controls()
        self._build_log_panel()

        # Initial mode
        self.mode_var.set('calibrate')
        self._on_mode_change()

    # ── Mode Selector ────────────────────────────────────────────────────────

    def _build_mode_selector(self):
        frame = ttk.LabelFrame(self, text="Mode")
        frame.pack(fill='x', padx=8, pady=(8, 4))

        self.mode_var = tk.StringVar(value='calibrate')
        ttk.Radiobutton(frame, text="Quick Calibrate", variable=self.mode_var,
                        value='calibrate', command=self._on_mode_change).pack(side='left', padx=10)
        ttk.Radiobutton(frame, text="Main Processing", variable=self.mode_var,
                        value='process', command=self._on_mode_change).pack(side='left', padx=10)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        # Show/hide prefs
        if mode == 'process':
            self.prefs_frame.pack(fill='x', padx=8, pady=4, after=self.mode_frame_ref)
        else:
            self.prefs_frame.pack_forget()

        # Toggle ROI button based on mode and mode detection state
        if mode == 'process' and self.mode_detect_var.get():
            self.roi_btn.configure(state='normal')
        else:
            self.roi_btn.configure(state='disabled')

        # Reconfigure treeview columns
        self._configure_tree_columns(mode)

        # Clear table when switching modes
        self._on_clear_all()

    # ── Preferences Panel (Main mode only) ───────────────────────────────────

    def _build_preferences_panel(self):
        self.prefs_frame = ttk.LabelFrame(self, text="Preferences (Main mode only)")
        # Store reference for pack ordering
        self.mode_frame_ref = self.winfo_children()[0]

        row1 = ttk.Frame(self.prefs_frame)
        row1.pack(fill='x', padx=5, pady=2)

        self.upscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Upscale", variable=self.upscale_var).pack(side='left')
        ttk.Label(row1, text=" factor:").pack(side='left')
        self.upscale_factor_var = tk.StringVar(value='2')
        ttk.Combobox(row1, textvariable=self.upscale_factor_var, values=['2', '4'],
                     width=3, state='readonly').pack(side='left', padx=(0, 15))

        ttk.Label(row1, text="Delta interval:").pack(side='left')
        self.delta_var = tk.StringVar(value=str(DEFAULT_DELTA_INTERVAL))
        ttk.Spinbox(row1, textvariable=self.delta_var, from_=1, to=100,
                     width=5).pack(side='left', padx=(0, 15))

        row2 = ttk.Frame(self.prefs_frame)
        row2.pack(fill='x', padx=5, pady=2)

        self.mode_detect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Mode Detection", variable=self.mode_detect_var,
                        command=self._on_mode_detect_toggle).pack(side='left')

        ttk.Label(row2, text="  Car:").pack(side='left')
        car_labels = {k: v['label'] for k, v in CAR_TYPES.items()}
        self.car_var = tk.StringVar(value=DEFAULT_CAR_TYPE)
        ttk.Combobox(row2, textvariable=self.car_var,
                     values=list(CAR_TYPES.keys()), width=8,
                     state='readonly').pack(side='left')

    def _get_batch_preferences(self):
        """Build a prefs dict matching main.py's get_user_preferences() shape."""
        try:
            delta = int(self.delta_var.get())
            if delta < 1:
                delta = DEFAULT_DELTA_INTERVAL
        except ValueError:
            delta = DEFAULT_DELTA_INTERVAL

        return {
            'enable_realtime_viz': False,
            'enable_upscale': self.upscale_var.get(),
            'upscale_factor': int(self.upscale_factor_var.get()),
            'load_json': False,
            'delta_interval': delta,
            'enable_mode_detection': self.mode_detect_var.get(),
            'car_type': self.car_var.get(),
            'enable_mode_viz': False,
        }

    def _on_mode_detect_toggle(self):
        """Enable/disable the Draw ROIs button based on mode detection checkbox."""
        if self.mode_detect_var.get() and self.mode_var.get() == 'process':
            self.roi_btn.configure(state='normal')
        else:
            self.roi_btn.configure(state='disabled')
        # Reconfigure tree columns to show/hide ROI column
        self._configure_tree_columns(self.mode_var.get())
        # Re-populate tree (columns changed, but keep entries)
        self._repopulate_tree()

    def _repopulate_tree(self):
        """Re-insert all video_entries into the treeview (after column change)."""
        self.tree.delete(*self.tree.get_children())
        mode = self.mode_var.get()
        for i, entry in enumerate(self.video_entries):
            vname = os.path.basename(entry['video'])
            start = entry.get('start', '')
            end = entry.get('end', '') or '(end)'
            if mode == 'calibrate':
                self.tree.insert('', 'end', iid=str(i),
                                 values=(i + 1, vname, start, end))
            else:
                tname = os.path.basename(entry.get('threshold', '')) or '(defaults)'
                if self.mode_detect_var.get():
                    roi_status = 'Done' if entry.get('custom_rois') else ''
                    self.tree.insert('', 'end', iid=str(i),
                                     values=(i + 1, vname, tname, roi_status, start, end))
                else:
                    self.tree.insert('', 'end', iid=str(i),
                                     values=(i + 1, vname, tname, start, end))

    def _on_draw_rois(self):
        """Open interactive ROI drawing for the selected video."""
        selected = self.tree.selection()
        if not selected or len(selected) != 1:
            messagebox.showinfo("Draw ROIs", "Select exactly one video.")
            return
        idx = int(selected[0])
        entry = self.video_entries[idx]
        video_path = entry['video']
        video_name = os.path.basename(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Draw ROIs", f"Cannot open video: {video_name}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        from mode_detection import draw_rois_interactive
        roi_result = draw_rois_interactive(cap, car_type=self.car_var.get(), video_fps=video_fps)
        cap.release()

        if roi_result is not None:
            self.video_entries[idx]['custom_rois'] = roi_result
            self._log_to_panel(f"ROIs drawn for {video_name}")
        else:
            self._log_to_panel(f"ROI drawing cancelled for {video_name}")

        self._refresh_tree()

    # ── Video Table ──────────────────────────────────────────────────────────

    def _build_video_table(self):
        table_frame = ttk.LabelFrame(self, text="Video Queue")
        table_frame.pack(fill='both', expand=True, padx=8, pady=4)

        # Treeview
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill='both', expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(tree_container, show='headings', selectmode='extended')
        vsb = ttk.Scrollbar(tree_container, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        # Buttons
        btn_frame = ttk.Frame(table_frame)
        btn_frame.pack(fill='x', padx=5, pady=(0, 5))

        ttk.Button(btn_frame, text="+ Add Videos", command=self._on_add_videos).pack(side='left', padx=2)
        self.pair_btn = ttk.Button(btn_frame, text="Pair Threshold", command=self._on_pair_threshold)
        self.pair_btn.pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Edit Times", command=self._on_edit_timestamps).pack(side='left', padx=2)
        self.roi_btn = ttk.Button(btn_frame, text="Draw ROIs", command=self._on_draw_rois, state='disabled')
        self.roi_btn.pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Remove", command=self._on_remove_selected).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Clear", command=self._on_clear_all).pack(side='left', padx=2)

    def _configure_tree_columns(self, mode):
        """Set up treeview columns based on mode."""
        # Clear existing
        self.tree.delete(*self.tree.get_children())
        self.video_entries.clear()

        if mode == 'calibrate':
            cols = ('#', 'Video', 'Start', 'End')
            self.tree['columns'] = cols
            self.tree.column('#', width=30, minwidth=30, stretch=False)
            self.tree.column('Video', width=350, minwidth=150)
            self.tree.column('Start', width=80, minwidth=60)
            self.tree.column('End', width=80, minwidth=60)
            for c in cols:
                self.tree.heading(c, text=c)
            self.pair_btn.configure(state='disabled')
        else:
            if self.mode_detect_var.get():
                cols = ('#', 'Video', 'Threshold JSON', 'ROI', 'Start', 'End')
                self.tree['columns'] = cols
                self.tree.column('#', width=30, minwidth=30, stretch=False)
                self.tree.column('Video', width=250, minwidth=150)
                self.tree.column('Threshold JSON', width=170, minwidth=100)
                self.tree.column('ROI', width=50, minwidth=40, stretch=False)
                self.tree.column('Start', width=80, minwidth=60)
                self.tree.column('End', width=80, minwidth=60)
            else:
                cols = ('#', 'Video', 'Threshold JSON', 'Start', 'End')
                self.tree['columns'] = cols
                self.tree.column('#', width=30, minwidth=30, stretch=False)
                self.tree.column('Video', width=280, minwidth=150)
                self.tree.column('Threshold JSON', width=180, minwidth=100)
                self.tree.column('Start', width=80, minwidth=60)
                self.tree.column('End', width=80, minwidth=60)
            for c in cols:
                self.tree.heading(c, text=c)
            self.pair_btn.configure(state='normal')

    def _refresh_tree(self):
        """Rebuild treeview from self.video_entries."""
        self.tree.delete(*self.tree.get_children())
        mode = self.mode_var.get()
        for i, entry in enumerate(self.video_entries):
            vname = os.path.basename(entry['video'])
            start = entry.get('start', '')
            end = entry.get('end', '') or '(end)'
            if mode == 'calibrate':
                self.tree.insert('', 'end', iid=str(i),
                                 values=(i + 1, vname, start, end))
            else:
                tname = os.path.basename(entry.get('threshold', '')) or '(defaults)'
                if self.mode_detect_var.get():
                    roi_status = 'Done' if entry.get('custom_rois') else ''
                    self.tree.insert('', 'end', iid=str(i),
                                     values=(i + 1, vname, tname, roi_status, start, end))
                else:
                    self.tree.insert('', 'end', iid=str(i),
                                     values=(i + 1, vname, tname, start, end))

    def _on_add_videos(self):
        paths = filedialog.askopenfilenames(
            title="Select Videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All files", "*.*")]
        )
        thresh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Thresholds")
        matched, unmatched = 0, 0
        for p in paths:
            threshold = ''
            # Auto-match threshold JSON by video name in Main mode
            if self.mode_var.get() == 'process':
                video_stem = os.path.splitext(os.path.basename(p))[0]
                import glob
                pattern = os.path.join(thresh_dir, f"thresholds_{video_stem}_*.json")
                candidates = sorted(glob.glob(pattern), reverse=True)
                if candidates:
                    threshold = candidates[0]  # most recent by timestamp
                    matched += 1
                else:
                    unmatched += 1
            self.video_entries.append({
                'video': p, 'threshold': threshold, 'start': '', 'end': '',
                'custom_rois': None,
            })
        self._refresh_tree()
        if self.mode_var.get() == 'process' and paths:
            parts = []
            if matched:
                parts.append(f"{matched} auto-matched")
            if unmatched:
                parts.append(f"{unmatched} using defaults")
            self._log_to_panel(f"Thresholds: {', '.join(parts)}")

    def _log_to_panel(self, text):
        """Write a line to the log panel without going through stdout."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def _on_pair_threshold(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Pair Threshold", "Select one or more videos first.")
            return
        path = filedialog.askopenfilename(
            title="Select Threshold JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            for iid in selected:
                idx = int(iid)
                self.video_entries[idx]['threshold'] = path
            self._refresh_tree()

    def _on_edit_timestamps(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Edit Times", "Select a video first.")
            return
        idx = int(selected[0])
        entry = self.video_entries[idx]

        dlg = tk.Toplevel(self)
        dlg.title(f"Timestamps: {os.path.basename(entry['video'])}")
        dlg.geometry("300x150")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Start time (e.g. 1:30 or 90):").pack(padx=10, pady=(10, 2))
        start_entry = ttk.Entry(dlg)
        start_entry.insert(0, entry.get('start', ''))
        start_entry.pack(padx=10)

        ttk.Label(dlg, text="End time (blank = end of video):").pack(padx=10, pady=(10, 2))
        end_entry = ttk.Entry(dlg)
        end_entry.insert(0, entry.get('end', ''))
        end_entry.pack(padx=10)

        def _save():
            for iid in selected:
                i = int(iid)
                self.video_entries[i]['start'] = start_entry.get().strip()
                self.video_entries[i]['end'] = end_entry.get().strip()
            self._refresh_tree()
            dlg.destroy()

        ttk.Button(dlg, text="OK", command=_save).pack(pady=10)

    def _on_remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        indices = sorted([int(iid) for iid in selected], reverse=True)
        for idx in indices:
            self.video_entries.pop(idx)
        self._refresh_tree()

    def _on_clear_all(self):
        self.video_entries.clear()
        self.tree.delete(*self.tree.get_children())

    # ── Controls ─────────────────────────────────────────────────────────────

    def _build_controls(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(fill='x', padx=8, pady=4)

        self.run_btn = ttk.Button(ctrl, text="Run Batch", command=self._on_run)
        self.run_btn.pack(side='left', padx=4)

        self.cancel_btn = ttk.Button(ctrl, text="Cancel", command=self._on_cancel,
                                     state='disabled')
        self.cancel_btn.pack(side='left', padx=4)

        self.progress_bar = ttk.Progressbar(ctrl, length=250, mode='determinate')
        self.progress_bar.pack(side='left', padx=10)

        self.status_label = ttk.Label(ctrl, text="")
        self.status_label.pack(side='left', padx=4)

    def _on_run(self):
        if not self.video_entries:
            messagebox.showinfo("Run Batch", "Add at least one video first.")
            return

        self.cancel_flag.clear()
        mode = self.mode_var.get()
        prefs = self._get_batch_preferences() if mode == 'process' else {}

        # Validate ROIs when mode detection is enabled
        if mode == 'process' and prefs.get('enable_mode_detection'):
            missing = [os.path.basename(e['video']) for e in self.video_entries
                       if not e.get('custom_rois')]
            if missing:
                messagebox.showerror(
                    "Missing ROIs",
                    "Mode detection requires ROIs for every video.\n"
                    f"Missing ROIs for:\n" + "\n".join(f"  - {v}" for v in missing)
                )
                return

        # Disable controls
        self.run_btn.configure(state='disabled')
        self.cancel_btn.configure(state='normal')
        self.progress_bar['maximum'] = len(self.video_entries)
        self.progress_bar['value'] = 0
        self.status_label.configure(text=f"0/{len(self.video_entries)}")

        # Clear log
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')

        # Redirect stdout
        import sys
        self._old_stdout = sys.stdout
        sys.stdout = self.redirector

        self.runner = BatchRunner(
            mode=mode,
            video_entries=list(self.video_entries),
            prefs=prefs,
            cancel_flag=self.cancel_flag,
            root=self,
            on_progress=self._on_progress,
            on_batch_complete=self._on_batch_complete,
            log_fn=print,
        )
        self.runner.start()

    def _on_cancel(self):
        self.cancel_flag.set()
        self.cancel_btn.configure(state='disabled')
        self.status_label.configure(text="Cancelling...")

    def _on_progress(self, idx, total):
        self.progress_bar['value'] = idx
        self.status_label.configure(text=f"{idx}/{total}")

    def _on_batch_complete(self, results):
        import sys
        sys.stdout = self._old_stdout

        self.run_btn.configure(state='normal')
        self.cancel_btn.configure(state='disabled')
        self.progress_bar['value'] = self.progress_bar['maximum']

        successes = sum(1 for r in results if r.get('success'))
        failures = len(results) - successes
        self.status_label.configure(text=f"Done: {successes} OK, {failures} failed")

        # Summary
        detail_lines = []
        for r in results:
            vname = os.path.basename(r.get('video', ''))
            if r.get('success'):
                detail_lines.append(f"  OK: {vname}")
            else:
                detail_lines.append(f"  FAIL: {vname} — {r.get('error', '?')}")

        messagebox.showinfo(
            "Batch Complete",
            f"Processed {len(results)} videos.\n"
            f"Success: {successes}  |  Failed: {failures}\n\n"
            + "\n".join(detail_lines)
        )

    # ── Log Panel ────────────────────────────────────────────────────────────

    def _build_log_panel(self):
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill='both', expand=True, padx=8, pady=(4, 8))

        self.log_text = ScrolledText(log_frame, height=8, state='disabled',
                                     wrap='word', font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

        self.redirector = TextRedirector(self.log_text)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    BatchApp().mainloop()
