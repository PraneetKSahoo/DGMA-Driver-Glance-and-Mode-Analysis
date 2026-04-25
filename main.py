"""
main.py
----------
Deep Learning Orchestration Module for Head Pose & Eye Gaze Estimation.
Uses L2CS-Net on PyTorch (CUDA) instead of MediaPipe.
"""

import os

# Ensure these paths are correct for your system
os.add_dll_directory(r'C:/opencv_build/bin/Release')
os.add_dll_directory(r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin')
 
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from typing import Optional

from config import (
    ThresholdConfig, DEFAULT_THRESHOLDS, DEFAULT_DELTA_INTERVAL,
    DEFAULT_SHOW_OVERLAY, POSE_SMOOTHING_FACTOR,
    GLANCE_MIN_DURATION, CONFIDENCE_HISTORY_LENGTH, FRAME_SKIP_FACTOR,
    CAR_TYPES, DEFAULT_CAR_TYPE, MODE_CHECK_INTERVAL_SECONDS
)

from state import TrackingState
from glance import GlanceTracker
from face_analysis_dl import DeepGazeTracker 
from preprocessing import crop_to_first_quadrant, enhance_frame, setup_upscaler, upscale_frame
from io_utils import select_video_file, open_video, save_gaze_csv, save_glance_history_csv
from visualization import SettingsWindow, OverlayRenderer, PlotManager
from mode_detection import ModeDetector, ModeTracker

# =============================================================================
# Timestamp Parsing
# =============================================================================

def parse_timestamp(
    input_str: str,
    video_fps: float,
    total_frames: int,
) -> Optional[int]:
    input_str = input_str.strip()
    if not input_str:
        return None

    try:
        if ':' in input_str:
            parts = input_str.split(':')
            if len(parts) != 2:
                return None
            minutes = int(parts[0])
            seconds = float(parts[1])
            total_seconds = minutes * 60 + seconds
        else:
            total_seconds = float(input_str)

        if total_seconds < 0:
            return None

        frame = int(total_seconds * video_fps)
        return max(0, min(frame, total_frames - 1))

    except (ValueError, TypeError):
        return None


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


# =============================================================================
# User Preferences
# =============================================================================

def get_user_preferences() -> dict:
    prefs = {
        'enable_realtime_viz': False,
        'enable_upscale': False,
        'upscale_factor': 2,
        'load_json': False,
        'delta_interval': DEFAULT_DELTA_INTERVAL,
        'enable_mode_detection': False,
        'car_type': DEFAULT_CAR_TYPE,
        'draw_custom_rois': False,
        'enable_mode_viz': False,
    }

    # --- Real-time visualizations ---
    viz_input = input("Enable real-time visualizations (Y/N)? ").strip().upper()
    if viz_input == 'Y':
        prefs['enable_realtime_viz'] = True
        print("Real-time visualizations enabled.")
    else:
        print("Real-time visualizations disabled.")

    # --- Upscaling ---
    upscale_input = input("Enable FSRCNN upscaling (Y/N)? ").strip().upper()
    if upscale_input == 'Y':
        prefs['enable_upscale'] = True
        scale_input = input("Upscale factor (2 or 4)? [default: 2]: ").strip()
        if scale_input == '4':
            prefs['upscale_factor'] = 4
        else:
            prefs['upscale_factor'] = 2
        print(f"Upscaling enabled: {prefs['upscale_factor']}x FSRCNN")
    else:
        print("Upscaling disabled.")

    # --- Threshold JSON import ---
    json_input = input("Import threshold JSON file (Y/N)? [default: N]: ").strip().upper()
    if json_input == 'Y':
        prefs['load_json'] = True

    # --- Delta interval ---
    interval_input = input(f"Delta interval (frames for derivative computation) [default: {DEFAULT_DELTA_INTERVAL}]: ").strip()

    if not interval_input:
        prefs['delta_interval'] = DEFAULT_DELTA_INTERVAL
        print(f"Using default delta interval: {DEFAULT_DELTA_INTERVAL}")
    else:
        try:
            val = int(interval_input)
            if val < 1:
                print("Interval must be at least 1. Using default.")
                prefs['delta_interval'] = DEFAULT_DELTA_INTERVAL
            else:
                prefs['delta_interval'] = val
                print(f"Delta interval set to: {val}")
        except ValueError:
            print("Invalid input. Using default interval.")
            prefs['delta_interval'] = DEFAULT_DELTA_INTERVAL

    # --- Mode Detection ---
    mode_input = input("Enable driving mode detection (Y/N)? [default: N]: ").strip().upper()
    if mode_input == 'Y':
        prefs['enable_mode_detection'] = True

        print("\nAvailable car types:")
        for key, data in CAR_TYPES.items():
            print(f"  '{key}' - {data['label']}")

        car_input = input(f"Select car type[default: '{DEFAULT_CAR_TYPE}' ({CAR_TYPES[DEFAULT_CAR_TYPE]['label']})]: ").strip().lower()

        if car_input in CAR_TYPES:
            prefs['car_type'] = car_input
            print(f"Car type: {CAR_TYPES[car_input]['label']}")
        elif car_input == '':
            prefs['car_type'] = DEFAULT_CAR_TYPE
            print(f"Using default car type: {CAR_TYPES[DEFAULT_CAR_TYPE]['label']}")
        else:
            print(f"Unknown car type '{car_input}'. Using default: {CAR_TYPES[DEFAULT_CAR_TYPE]['label']}")
            prefs['car_type'] = DEFAULT_CAR_TYPE

        roi_input = input("Draw custom ROIs for mode detection (Y/N)? [default: N]: ").strip().upper()
        if roi_input == 'Y':
            prefs['draw_custom_rois'] = True

        if prefs['enable_realtime_viz']:
            mode_viz_input = input("Visualize mode detection (Y/N)?[default: N]: ").strip().upper()
            if mode_viz_input == 'Y':
                prefs['enable_mode_viz'] = True
                print("Mode detection visualization enabled.")
            else:
                print("Mode detection visualization disabled.")
        else:
            print("Mode detection visualization requires real-time visualizations. Skipped.")

        print("Mode detection enabled.")
    else:
        print("Mode detection disabled.")

    return prefs


# =============================================================================
# Segment Selection
# =============================================================================

def get_segment_selection(
    video_fps: float,
    total_frames: int,
) -> tuple:
    total_duration_sec = total_frames / video_fps
    total_ts = format_timestamp(total_duration_sec)

    print(f"\nVideo duration: {total_ts} ({total_frames} frames)")
    print(f"Process entire video or a specific segment?")
    print(f"  Enter timestamps as MM:SS or seconds (e.g., '1:30' or '90')")

    segment_start_frame = 0
    segment_end_frame = total_frames

    start_input = input(f"  Start timestamp [default: 0:00]: ").strip()
    if start_input:
        parsed = parse_timestamp(start_input, video_fps, total_frames)
        if parsed is not None:
            segment_start_frame = parsed
            start_sec = segment_start_frame / video_fps
            print(f"    Start: {format_timestamp(start_sec)} (frame {segment_start_frame})")
        else:
            print(f"    Invalid input. Using start of video.")

    end_input = input(f"  End timestamp [default: end ({total_ts})]: ").strip()
    if end_input:
        parsed = parse_timestamp(end_input, video_fps, total_frames)
        if parsed is not None:
            if parsed > segment_start_frame:
                segment_end_frame = parsed
                end_sec = segment_end_frame / video_fps
                print(f"    End: {format_timestamp(end_sec)} (frame {segment_end_frame})")
            else:
                print(f"    End must be after start. Using end of video.")
        else:
            print(f"    Invalid input. Using end of video.")

    seg_start_sec = segment_start_frame / video_fps
    seg_end_sec = segment_end_frame / video_fps
    seg_duration = seg_end_sec - seg_start_sec

    print(f"\n  Segment: {format_timestamp(seg_start_sec)} - {format_timestamp(seg_end_sec)} ({seg_duration:.2f}s, frames {segment_start_frame}-{segment_end_frame})")

    return segment_start_frame, segment_end_frame


# =============================================================================
# Progress Bar
# =============================================================================

def print_progress_bar(
    current_frame: int,
    segment_start: int,
    segment_end: int,
    bar_width: int = 40,
):
    segment_total = segment_end - segment_start
    if segment_total <= 0:
        return

    segment_progress = current_frame - segment_start
    percent = (segment_progress / segment_total) * 100
    percent = min(percent, 100.0)

    filled = int(bar_width * segment_progress / segment_total)
    filled = min(filled, bar_width)

    bar = '█' * filled + '░' * (bar_width - filled)
    print(f'\rProcessing: [{bar}] {percent:.1f}%', end='', flush=True)

def jump_to_frame_dl(cap, total_frames, video_fps, state, glance_tracker, 
                     plot_manager, mode_tracker, segment_start, segment_end):
    """Modified jump_to_frame: PyTorch models don't need to be destroyed/recreated like MediaPipe."""
    root = tk.Tk()
    root.withdraw()
    frame_input = simpledialog.askstring("Jump to Frame", f"Enter frame number ({segment_start} to {segment_end - 1}):", parent=root)
    root.destroy()

    try:
        frame_number = int(frame_input)
        if not (segment_start <= frame_number < segment_end): return False, 0
    except (ValueError, TypeError):
        return False, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    state.reset()
    glance_tracker.reset()
    if plot_manager: plot_manager.reset()
    if mode_tracker: mode_tracker.reset()
    state.confidence_history.clear()

    print(f"Jumped to frame {frame_number}")
    return True, frame_number


def main():
    prefs = get_user_preferences()

    config = ThresholdConfig(DEFAULT_THRESHOLDS)
    if prefs['load_json']: config.load_from_json()

    video_path = select_video_file()
    if not video_path: return
    video_basename = os.path.basename(video_path)

    cap, total_frames, video_fps = open_video(video_path)
    segment_start_frame, segment_end_frame = get_segment_selection(video_fps, total_frames)
    if segment_start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start_frame)

    sr = setup_upscaler(prefs['upscale_factor']) if prefs['enable_upscale'] else None

    # ---- LOAD DEEP LEARNING MODEL ON RTX 5000 ----
    gaze_tracker = DeepGazeTracker(device='cuda')

    mode_detector = ModeDetector(car_type=prefs['car_type']) if prefs['enable_mode_detection'] else None
    mode_tracker = ModeTracker() if prefs['enable_mode_detection'] else None
    mode_check_frame_interval = int(video_fps * MODE_CHECK_INTERVAL_SECONDS) if video_fps > 0 else 300

    state = TrackingState(
        delta_interval=prefs['delta_interval'], 
        pose_alpha=POSE_SMOOTHING_FACTOR,
        confidence_history_len=CONFIDENCE_HISTORY_LENGTH,
    )

    glance_tracker = GlanceTracker(min_duration=GLANCE_MIN_DURATION)

    settings_window = None
    overlay_renderer = None
    plot_manager = None
    if prefs['enable_realtime_viz']:
        settings_window = SettingsWindow(config)
        settings_window.create()
        from visualization import OverlayRenderer, PlotManager
        overlay_renderer = OverlayRenderer()
        plot_manager = PlotManager(delta_interval=prefs['delta_interval'])
        plot_manager.setup()
        cv2.namedWindow('Deep Gaze Estimation (RTX 5000)')

    frame_count = segment_start_frame
    is_paused = show_help = False
    collected_data =[]

    print(f"\nProcessing segment: {format_timestamp(segment_start_frame/video_fps)} - {format_timestamp(segment_end_frame/video_fps)}")

    clean_frame = None
    last_raw_yaw = 0.0
    last_raw_pitch = 0.0

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord(' '): is_paused = not is_paused
            if key == ord('h'): show_help = not show_help
            if key == ord('j') and is_paused:
                success, new_frame_count = jump_to_frame_dl(cap, total_frames, video_fps, state, glance_tracker, plot_manager, mode_tracker, segment_start_frame, segment_end_frame)
                if success: frame_count = new_frame_count

            resolved = settings_window.read_resolved_thresholds() if settings_window else config.get_active_thresholds_resolved()
            show_overlay = settings_window.get_show_overlay() if settings_window else bool(DEFAULT_SHOW_OVERLAY)

            yaw_range = (resolved['yaw_low'], resolved['yaw_high'])
            pitch_range = (resolved['pitch_low'], resolved['pitch_high'])
            
            if key == ord('s'):
                import json
                from datetime import datetime
                snapshot = {
                    "yaw_low":   resolved['yaw_low'],
                    "yaw_high":  resolved['yaw_high'],
                    "pitch_low": resolved['pitch_low'],
                    "pitch_high": resolved['pitch_high'],
                    "camera_yaw_offset":   round((resolved['yaw_low']   + resolved['yaw_high'])   / 2.0, 1),
                    "camera_pitch_offset": round((resolved['pitch_low'] + resolved['pitch_high']) / 2.0, 1),
                }
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = f"thresholds_{video_basename}_manual_{ts}.json"
                with open(out_file, 'w') as f:
                    json.dump(snapshot, f, indent=4)
                print(f"✅ Manual thresholds saved to '{out_file}'  "
                      f"(yaw [{snapshot['yaw_low']}, {snapshot['yaw_high']}], "
                      f"pitch [{snapshot['pitch_low']}, {snapshot['pitch_high']}])")

            if is_paused:
                # If paused, dynamically redraw the UI using the LIVE slider positions!
                if prefs['enable_realtime_viz'] and clean_frame is not None:
                    display_frame = clean_frame.copy()
                    if show_overlay:
                        # Instantly check if the new slider position makes the dot "On-Road"
                        temp_zone = glance_tracker.classify_zone(
                            state, yaw_range, pitch_range
                        )
                        
                        overlay_renderer.draw_metrics(
                            display_frame, state, temp_zone, state.smoothed_confidence,
                            raw_yaw=last_raw_yaw, raw_pitch=last_raw_pitch
                        )
                        overlay_renderer.draw_gaze_radar(
                            display_frame, state, yaw_range, pitch_range
                        )
                    cv2.imshow('Deep Gaze Estimation (RTX 5000)', display_frame)
                continue # Skip the neural network inference while paused

            success, frame_original = cap.read()
            if not success or frame_count >= segment_end_frame: break
            frame_count += 1

            if frame_count % FRAME_SKIP_FACTOR != 0: continue

            try:
                frame = crop_to_first_quadrant(frame_original)
                if frame is None or frame.size == 0: continue
                if sr: frame = upscale_frame(sr, frame, prefs['upscale_factor'])
                
                current_time = frame_count / video_fps

                clean_frame = frame.copy()
                image_bgr = clean_frame.copy()

                # ---- DEEP LEARNING INFERENCE ----
                gaze_result = gaze_tracker.process_frame(frame)

                raw_yaw, raw_pitch, raw_roll = 0.0, 0.0, 0.0
                confidence = 0.0

                # --- UPGRADE: Require at least 50% Confidence ---
                if gaze_result and gaze_result['confidence'] >= 0.50:
                    raw_yaw = gaze_result['yaw']
                    raw_pitch = gaze_result['pitch']
                    confidence = gaze_result['confidence']

                    last_raw_yaw = raw_yaw
                    last_raw_pitch = raw_pitch
                                                            
                    # Update Pose State
                    state.update_pose(raw_yaw, raw_pitch, raw_roll, video_fps)
                    
                    # Determine Zone using new Mahalanobis Covariance logic!
                    raw_zone = glance_tracker.classify_zone(
                        state, yaw_range, pitch_range
                    )

                    glance_zone, grace_confidence = glance_tracker.apply_grace_period(raw_zone, current_time)
                    if grace_confidence is not None: confidence = grace_confidence

                    glance_tracker.last_valid_confidence = confidence
                    state.update_confidence(confidence)

                    row = state.get_collected_row(current_time)
                    if row: collected_data.append(row)

                    # Optional: Draw 3D Gaze Vectors
                    if prefs['enable_realtime_viz']:
                        image_bgr = gaze_tracker.draw_gaze(image_bgr, gaze_result)

                else:
                    # If face is lost OR confidence is < 50%, route here!
                    # This safely triggers the Grace Period without polluting your EMA math.
                    state.clear_on_no_detection()
                    glance_zone, grace_confidence = glance_tracker.apply_grace_period("Undetermined", current_time)
                    state.update_confidence(grace_confidence if grace_confidence else 0.0)

                # Update trackers
                glance_tracker.update(glance_zone, current_time, frame_count, state.smoothed_confidence, state, mode_tracker)
                state.commit_prev_values()

                if plot_manager:
                    plot_manager.append_data(frame_count, state, state.smoothed_confidence)
                    plot_manager.update(frame_count)

                if prefs['enable_realtime_viz']:
                    if show_overlay:
                        overlay_renderer.draw_metrics(
                            image_bgr, state, glance_zone, state.smoothed_confidence,
                            raw_yaw=raw_yaw, raw_pitch=raw_pitch
                        )
                        overlay_renderer.draw_gaze_radar(
                            image_bgr, state, yaw_range, pitch_range
                        )
                    cv2.imshow('Deep Gaze Estimation (RTX 5000)', image_bgr)
                else:
                    print_progress_bar(frame_count, segment_start_frame, segment_end_frame)

            except Exception as e:
                print(f"\nError processing frame {frame_count}: {e}")

    except KeyboardInterrupt:
        pass

    if not prefs['enable_realtime_viz']: print()
    final_time = frame_count / video_fps

    glance_tracker.flush(final_time, frame_count, state.smoothed_confidence, mode_tracker)
    
    # Generate Reports
    segment_info = {'video_file': video_basename, 'segment_start': segment_start_frame/video_fps, 'segment_end': final_time, 'duration': final_time - (segment_start_frame/video_fps)}
    
    report = glance_tracker.build_report({}, segment_info, mode_tracker)
    glance_tracker.print_report(report)

    save_gaze_csv(collected_data, video_name=video_basename)
    save_glance_history_csv(glance_tracker.glance_history, video_name=video_basename)

    cap.release()
    if prefs['enable_realtime_viz']: cv2.destroyAllWindows()
    print("\nDeep Learning Pipeline finished successfully.")

if __name__ == '__main__':
    main()