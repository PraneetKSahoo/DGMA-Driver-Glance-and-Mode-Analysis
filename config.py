"""
config.py
---------
Central configuration module for the Head Pose & Eye Gaze Estimation system.

Adapted for L2CS-Net: Removed all MediaPipe facial blendshapes, 
blink constants, and legacy gaze tracking thresholds.
"""

import os
import json
import tkinter as tk
from tkinter import filedialog

# =============================================================================
# Smoothing Factors (Exponential Moving Average alpha values)
# =============================================================================
POSE_SMOOTHING_FACTOR = 0.1

# =============================================================================
# Frame Processing
# =============================================================================
DEFAULT_DELTA_INTERVAL = 10
FRAME_SKIP_FACTOR = 3

# =============================================================================
# Preprocessing Parameters
# =============================================================================
CONTRAST_ALPHA = 1.1
BRIGHTNESS_BETA = 5
BLUR_KERNEL = (3, 3)

# =============================================================================
# Glance Classification
# =============================================================================
GLANCE_MIN_DURATION = 0.01
CONFIDENCE_HIGH_THRESHOLD = 0.7
CONFIDENCE_MED_THRESHOLD = 0.4
NO_DETECTION_GRACE_PERIOD = 2 # seconds

# =============================================================================
# Off-Road Glance Duration Bins (seconds)
# =============================================================================
OFFROAD_THRESHOLD = 2.0

# =============================================================================
# Visualization
# =============================================================================
DEFAULT_SHOW_OVERLAY = 1
PLOT_HISTORY_LENGTH = 300
PLOT_UPDATE_INTERVAL = 5
CONFIDENCE_HISTORY_LENGTH = 5

# =============================================================================
# Default Trackbar / Threshold Values
# =============================================================================
DEFAULT_YAW_LOW = 8
DEFAULT_YAW_HIGH = 26
DEFAULT_PITCH_LOW = -10
DEFAULT_PITCH_HIGH = 10

DEFAULT_THRESHOLDS = {
    "yaw_low": DEFAULT_YAW_LOW,
    "yaw_high": DEFAULT_YAW_HIGH,
    "pitch_low": DEFAULT_PITCH_LOW,
    "pitch_high": DEFAULT_PITCH_HIGH,
}

# =============================================================================
# Mode Detection Configuration 
# =============================================================================

MODE_CHECK_INTERVAL_SECONDS = 10
MODE_REF_WIDTH = 944
MODE_REF_HEIGHT = 480
MODE_MORPH_KERNEL_SIZE = 9

MODE_HSV_RANGES = {
    'n': { 'green_lower': (35, 40, 30), 'green_upper': (85, 255, 255), 'white_lower': (0, 0, 200), 'white_upper': (180, 30, 255) },
    'c': { 'green_lower': (35, 40, 30), 'green_upper': (85, 255, 255), 'white_lower': (0, 0, 200), 'white_upper': (180, 30, 255) },
    's': { 'green_lower': (0, 0, 200), 'green_upper': (115, 90, 255), 'white_lower': (0, 0, 200), 'white_upper': (180, 30, 255) },
    '3': { 'green_lower': (0, 0, 0),   'green_upper': (157, 78, 81),  'white_lower': (0, 0, 0),   'white_upper': (0, 0, 10) },
    'v': { 'green_lower': (35, 40, 30), 'green_upper': (85, 255, 255), 'white_lower': (0, 0, 200), 'white_upper': (180, 30, 255) },
}

CAR_TYPES = {
    'c': {
        'label': 'Cadillac',
        'rois': {
            'top_left': { 'x': 260, 'y': 260, 'width': 140, 'height': 140, 'min_area': 400, 'max_area': 1200, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.0 },
            'bottom_center': { 'x': 800, 'y': 400, 'width': 600, 'height': 480, 'min_area': 2800, 'max_area': 8000, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.0 },
        },
    },
    'n': {
        'label': 'Nissan',
        'rois': {
            'top_left': { 'x': 520, 'y': 120, 'width': 120, 'height': 100, 'min_area': 200, 'max_area': 6000, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.5 },
            'bottom_center': { 'x': 520, 'y': 120, 'width': 120, 'height': 100, 'min_area': 1000, 'max_area': 6000, 'circularity_min': 0.2, 'circularity_max': 1.4, 'aspect_ratio_min': 0.5, 'aspect_ratio_max': 2.5 },
        },
    },
    's': {
        'label': 'Tesla S',
        'rois': {
            'top_left': { 'x': 495, 'y': 40, 'width': 45, 'height': 45, 'min_area': 200, 'max_area': 6000, 'circularity_min': 0.25, 'circularity_max': 1.35, 'aspect_ratio_min': 0.55, 'aspect_ratio_max': 1.9 },
            'bottom_center': { 'x': 495, 'y': 40, 'width': 45, 'height': 45, 'min_area': 760, 'max_area': 6000, 'circularity_min': 0.25, 'circularity_max': 1.35, 'aspect_ratio_min': 0.55, 'aspect_ratio_max': 2.5 },
        },
    },
    '3': {
        'label': 'Tesla 3',
        'rois': {
            'top_left': { 'x': 944, 'y': 180, 'width': 80, 'height': 80, 'min_area': 200, 'max_area': 800, 'circularity_min': 0.35, 'circularity_max': 1.25, 'aspect_ratio_min': 0.65, 'aspect_ratio_max': 1.7 },
            'bottom_center': { 'x': 944, 'y': 280, 'width': 200, 'height': 160, 'min_area': 1200, 'max_area': 6000, 'circularity_min': 0.35, 'circularity_max': 1.25, 'aspect_ratio_min': 0.65, 'aspect_ratio_max': 1.7 },
        },
    },
    'v': {
        'label': 'Volvo',
        'rois': {
            'top_left': { 'x': 944, 'y': 180, 'width': 80, 'height': 80, 'min_area': 200, 'max_area': 800, 'circularity_min': 0.4, 'circularity_max': 1.2, 'aspect_ratio_min': 0.7, 'aspect_ratio_max': 1.6 },
            'bottom_center': { 'x': 630, 'y': 360, 'width': 360, 'height': 360, 'min_area': 7500, 'max_area': 25000, 'circularity_min': -1, 'circularity_max': 1.2, 'aspect_ratio_min': -1, 'aspect_ratio_max': 2.1 },
        },
    },
}

DEFAULT_CAR_TYPE = 'n'

# =============================================================================
# ThresholdConfig Class
# =============================================================================

class ThresholdConfig:
    """Manages threshold loading, saving, validation, and access."""

    def __init__(self, defaults: dict = None):
        self.defaults = defaults if defaults is not None else DEFAULT_THRESHOLDS.copy()
        self.active = self.defaults.copy()

    def get_yaw_range(self) -> tuple:
        return (self.active['yaw_low'], self.active['yaw_high'])

    def get_pitch_range(self) -> tuple:
        return (self.active['pitch_low'], self.active['pitch_high'])

    def get_active_thresholds_resolved(self) -> dict:
        return {
            "yaw_low": self.active['yaw_low'],
            "yaw_high": self.active['yaw_high'],
            "pitch_low": self.active['pitch_low'],
            "pitch_high": self.active['pitch_high'],
        }

    def load_from_json(self) -> bool:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Threshold JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()

        if not file_path:
            return False

        try:
            with open(file_path, 'r') as f:
                thresholds = json.load(f)

            missing_keys =[k for k in self.defaults if k not in thresholds]
            if missing_keys:
                print(f"Warning: JSON missing keys: {missing_keys}. Using defaults for those.")
                for k in missing_keys:
                    thresholds[k] = self.defaults[k]

            self.active = thresholds
            self._print_loaded_thresholds(file_path)
            return True

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file: {e}")
            return False
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False

    def save_to_json(self, thresholds_dict: dict = None) -> bool:
        if thresholds_dict is None:
            thresholds_dict = self.active

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title="Save Thresholds as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()

        if not file_path:
            print("Save cancelled.")
            return False

        try:
            with open(file_path, 'w') as f:
                json.dump(thresholds_dict, f, indent=4)
            print(f"Thresholds saved to: {file_path}")
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False

    def _print_loaded_thresholds(self, file_path: str):
        print(f"Loaded thresholds from: {file_path}")
        print(f"  Yaw: [{self.active['yaw_low']}, {self.active['yaw_high']}]")
        print(f"  Pitch: [{self.active['pitch_low']}, {self.active['pitch_high']}]")