"""
visualization.py
----------------
All visualization components: OpenCV trackbar settings, frame overlays,
and matplotlib real-time plots with blitting.

Adapted for L2CS-Net: 
- Removed all MediaPipe blendshape metrics (eye openness, jaw, smile, etc.)
- Removed 478-point facial mesh drawing
- Simplified plots down to Head Pose, Confidence, and Velocities.

Usage:
    settings = SettingsWindow(config)
    settings.create()

    overlay = OverlayRenderer()
    overlay.draw_metrics(frame, state, zone, confidence)

    plots = PlotManager(delta_interval=10)
    plots.setup()
    plots.append_data(frame_count, state, confidence)
    plots.update(frame_count)
"""

import collections
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from config import (
    ThresholdConfig,
    DEFAULT_SHOW_OVERLAY,
    PLOT_HISTORY_LENGTH,
    PLOT_UPDATE_INTERVAL,
)
from state import TrackingState


# =============================================================================
# Settings Window (OpenCV Trackbars)
# =============================================================================

class SettingsWindow:
    """
    Manages the OpenCV trackbar settings window for adjusting thresholds.
    """

    WINDOW_NAME = 'Settings'

    def __init__(self, config: ThresholdConfig):
        self.config = config

    @staticmethod
    def _nothing(x):
        """No-op callback for trackbar events."""
        pass

    def create(self):
        """
        Create the OpenCV settings window with all trackbars.
        Trackbar ranges:
        - Yaw/Pitch: 0-360, offset by +180 (so 180 = 0 degrees)
        - Show Overlay: 0-1 toggle
        """
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.resizeWindow(self.WINDOW_NAME, 400, 250)

        active = self.config.active

        cv2.createTrackbar(
            'Yaw Low', self.WINDOW_NAME,
            int(active['yaw_low']) + 180, 360, self._nothing
        )
        cv2.createTrackbar(
            'Yaw High', self.WINDOW_NAME,
            int(active['yaw_high']) + 180, 360, self._nothing
        )
        cv2.createTrackbar(
            'Pitch Low', self.WINDOW_NAME,
            int(active['pitch_low']) + 180, 360, self._nothing
        )
        cv2.createTrackbar(
            'Pitch High', self.WINDOW_NAME,
            int(active['pitch_high']) + 180, 360, self._nothing
        )
        cv2.createTrackbar(
            'Show Overlay', self.WINDOW_NAME,
            DEFAULT_SHOW_OVERLAY, 1, self._nothing
        )

    def read_raw_thresholds(self) -> dict:
        """Read current trackbar positions and return as raw thresholds dict."""
        return {
            "yaw_low": cv2.getTrackbarPos('Yaw Low', self.WINDOW_NAME) - 180,
            "yaw_high": cv2.getTrackbarPos('Yaw High', self.WINDOW_NAME) - 180,
            "pitch_low": cv2.getTrackbarPos('Pitch Low', self.WINDOW_NAME) - 180,
            "pitch_high": cv2.getTrackbarPos('Pitch High', self.WINDOW_NAME) - 180,
        }

    def read_resolved_thresholds(self) -> dict:
        """Return thresholds in resolved (usable) form."""
        raw = self.read_raw_thresholds()
        return {
            "yaw_low": raw['yaw_low'],
            "yaw_high": raw['yaw_high'],
            "pitch_low": raw['pitch_low'],
            "pitch_high": raw['pitch_high'],
        }

    def validate_and_fix(self):
        """Ensure all low thresholds are strictly less than high thresholds."""
        resolved = self.read_resolved_thresholds()

        if resolved['yaw_low'] >= resolved['yaw_high']:
            cv2.setTrackbarPos(
                'Yaw High', self.WINDOW_NAME,
                resolved['yaw_low'] + 180 + 1
            )

        if resolved['pitch_low'] >= resolved['pitch_high']:
            cv2.setTrackbarPos(
                'Pitch High', self.WINDOW_NAME,
                resolved['pitch_low'] + 180 + 1
            )

    def get_show_overlay(self) -> bool:
        """Read the Show Overlay toggle state."""
        return bool(cv2.getTrackbarPos('Show Overlay', self.WINDOW_NAME))


# =============================================================================
# Overlay Renderer
# =============================================================================

class OverlayRenderer:
    """
    Draws text overlays, parameter tables, and gaze radar on video frames.
    """

    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE_SMALL = 0.5
    _FONT_SCALE_LARGE = 0.6
    _THICKNESS = 1
    _THICKNESS_BOLD = 2

    _COLOR_GREEN = (0, 255, 0)
    _COLOR_RED = (0, 0, 255)
    _COLOR_WHITE = (255, 255, 255)

    @staticmethod
    def draw_metrics(
        image: np.ndarray,
        state: TrackingState,
        glance_zone: str,
        smoothed_confidence: float,
        raw_yaw: float = 0.0,
        raw_pitch: float = 0.0,
    ):
        """Draw primary metrics text overlay on the frame."""
        font = OverlayRenderer._FONT
        scale = OverlayRenderer._FONT_SCALE_SMALL
        t = OverlayRenderer._THICKNESS
        green = OverlayRenderer._COLOR_GREEN
        red = OverlayRenderer._COLOR_RED

        zone_color = green if glance_zone == "On-Road" else red

        # Pose / Gaze text
        cv2.putText(image, f"Yaw: {raw_yaw:.1f}", (5, 20), font, scale, green, t)
        cv2.putText(image, f"Pitch: {raw_pitch:.1f}", (5, 40), font, scale, green, t)
        
        if state.smoothed_roll is not None:
            cv2.putText(image, f"Roll: {state.smoothed_roll:.1f}", (5, 60), font, scale, green, t)

        # Zone status
        cv2.putText(
            image, f"Status: {glance_zone}", (5, 95),
            font, OverlayRenderer._FONT_SCALE_LARGE, zone_color,
            OverlayRenderer._THICKNESS_BOLD
        )
        cv2.putText(
            image, f"Conf: {smoothed_confidence * 100:.0f}%", (5, 115),
            font, scale, zone_color, t
        )

    @staticmethod
    def draw_safe_box_table(
        image: np.ndarray,
        yaw_range: tuple,
        pitch_range: tuple,
        state: TrackingState,
    ):
        """Draw a parameter comparison table in the bottom-left corner."""
        h, w = image.shape[:2]
        table_w, table_h = 350, 90
        x_start = 10
        y_start = h - table_h - 10

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + table_w, y_start + table_h),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        color_hdr = (200, 200, 200)

        # Column positions
        col_x =[x_start + 10, x_start + 100, x_start + 240]
        row_h = 25

        # Headers
        for i, header in enumerate(["Param", "Safe Range", "Current"]):
            cv2.putText(
                image, header,
                (col_x[i], y_start + 20),
                font, scale, color_hdr, thickness
            )
        cv2.line(
            image,
            (x_start, y_start + 25),
            (x_start + table_w, y_start + 25),
            (255, 255, 255), 1
        )

        # Row definitions
        rows =[
            ("Yaw", yaw_range, state.smoothed_yaw),
            ("Pitch", pitch_range, state.smoothed_pitch),
        ]

        for i, (name, (low, high), curr) in enumerate(rows):
            y = y_start + 50 + (i * row_h)
            range_str = f"{low} : {high}"
            
            if curr is None:
                color = (0, 0, 255)
                curr_str = "N/A"
            else:
                is_safe = low <= curr <= high
                color = (0, 255, 0) if is_safe else (0, 0, 255)
                curr_str = f"{curr:.1f}"

            cv2.putText(image, name, (col_x[0], y), font, scale, color_hdr, thickness)
            cv2.putText(image, range_str, (col_x[1], y), font, scale, color_hdr, thickness)
            cv2.putText(image, curr_str, (col_x[2], y), font, scale, color, thickness)

    @staticmethod
    def draw_help_overlay(image: np.ndarray):
        """Draw keyboard shortcut help text on the frame."""
        help_lines =[
            "Keys: Space=Pause, q=Quit",
            "j=Jump to frame (when paused)",
            "s=Save thresholds (when paused)",
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(
                image, line,
                (5, 160 + i * 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1,
            )

    @staticmethod
    def draw_gaze_radar(
        image: np.ndarray,
        state: TrackingState,
        yaw_range: tuple,
        pitch_range: tuple
    ):
        """Draws a real-time 'Radar' showing the driver's gaze relative to the On-Road Ellipse."""
        if state.smoothed_yaw is None or state.smoothed_pitch is None:
            return

        h, w = image.shape[:2]
        radar_size = 200
        margin = 10
        
        # Center of the radar box
        x_center = w - (radar_size // 2) - margin
        y_center = h - (radar_size // 2) - margin

        # 1. Draw Radar Background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (w - radar_size - margin, h - radar_size - margin),
            (w - margin, h - margin),
            (20, 20, 20), -1
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # 2. Math for the Cone of Vision (Ellipse)
        y_low, y_high = yaw_range
        p_low, p_high = pitch_range
        
        logical_center_yaw = (y_low + y_high) / 2.0
        logical_center_pitch = (p_low + p_high) / 2.0
        
        radius_yaw = max(1.0, (y_high - y_low) / 2.0)
        radius_pitch = max(1.0, (p_high - p_low) / 2.0)

        scale_x = (radar_size / 2.0) / (radius_yaw * 3)
        scale_y = (radar_size / 2.0) / (radius_pitch * 3)

        # 3. Draw Crosshairs
        cv2.line(image, (w - radar_size - margin, y_center), (w - margin, y_center), (100, 100, 100), 1)
        cv2.line(image, (x_center, h - radar_size - margin), (x_center, h - margin), (100, 100, 100), 1)

        # 4. Draw the "On-Road" Ellipse Threshold
        cv2.ellipse(
            image,
            (x_center, y_center),
            (int(radius_yaw * scale_x), int(radius_pitch * scale_y)),
            angle=0, startAngle=0, endAngle=360,
            color=(0, 255, 0), thickness=2
        )

        # 5. Plot the Current Gaze Vector Dot
        dx = state.smoothed_yaw - logical_center_yaw
        dy = state.smoothed_pitch - logical_center_pitch
        
        px = int(x_center + (dx * scale_x))
        py = int(y_center + (dy * scale_y)) 

        # Clamp the dot to the radar box
        px = max(w - radar_size - margin, min(w - margin, px))
        py = max(h - radar_size - margin, min(h - margin, py))

        # Color the dot based on inside/outside ellipse
        elliptical_dist = (dx**2 / radius_yaw**2) + (dy**2 / radius_pitch**2)
        color = (0, 255, 0) if elliptical_dist <= 1.0 else (0, 0, 255)

        cv2.circle(image, (px, py), 6, color, -1)
        cv2.circle(image, (px, py), 2, (255, 255, 255), -1) 
        
        # 6. Floating Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"({state.smoothed_yaw:.1f}, {state.smoothed_pitch:.1f})", 
                    (px + 10, py - 5), font, 0.4, (255, 255, 255), 1)

        cv2.putText(image, f"{y_low:.0f}", (int(x_center - (radius_yaw * scale_x)) - 25, y_center + 4), font, 0.35, (0, 255, 0), 1)
        cv2.putText(image, f"{y_high:.0f}", (int(x_center + (radius_yaw * scale_x)) + 5, y_center + 4), font, 0.35, (0, 255, 0), 1)
        cv2.putText(image, f"{p_high:.0f}", (x_center - 10, int(y_center - (radius_pitch * scale_y)) - 5), font, 0.35, (0, 255, 0), 1)
        cv2.putText(image, f"{p_low:.0f}", (x_center - 10, int(y_center + (radius_pitch * scale_y)) + 12), font, 0.35, (0, 255, 0), 1)

        cv2.putText(image, "Gaze Radar", (w - radar_size - margin + 5, h - radar_size - margin + 20), 
                    font, 0.4, (200, 200, 200), 1)      


# =============================================================================
# Plot Manager (Matplotlib with robust blitting)
# =============================================================================

class PlotManager:
    """
    Manages real-time matplotlib plots with robust blitting.
    Simplified specifically for L2CS-Net outputs (Yaw, Pitch, Roll).
    """

    _FORCED_FULL_REDRAW_INTERVAL = 25

    def __init__(
        self,
        delta_interval: int,
        history_len: int = PLOT_HISTORY_LENGTH,
        update_interval: int = PLOT_UPDATE_INTERVAL,
    ):
        self.delta_interval = delta_interval
        self.history_len = history_len
        self.update_interval = update_interval
        self._update_counter = 0

        # --- History deques ---
        self.frame_indices = collections.deque(maxlen=history_len)
        self.yaw_history = collections.deque(maxlen=history_len)
        self.pitch_history = collections.deque(maxlen=history_len)
        self.roll_history = collections.deque(maxlen=history_len)
        
        self.delta_yaw_history = collections.deque(maxlen=history_len)
        self.delta_pitch_history = collections.deque(maxlen=history_len)
        self.delta_roll_history = collections.deque(maxlen=history_len)
        
        self.confidence_history = collections.deque(maxlen=history_len)
        
        self.head_ayaw_history = collections.deque(maxlen=history_len)
        self.head_apitch_history = collections.deque(maxlen=history_len)
        self.head_aroll_history = collections.deque(maxlen=history_len)

        # --- Matplotlib objects ---
        self.fig1 = None
        self.fig2 = None
        self.axes1 = None
        self.axes2 = None
        self.lines = {}

        # --- Background cache for blitting ---
        self._bg1 = None
        self._bg2 = None
        self._bg1_valid = False
        self._bg2_valid = False

        self._last_xlim_fig1 = None
        self._last_xlim_fig2 = None

    def _on_draw_fig1(self, event):
        if self.fig1 is not None:
            self._bg1 = self.fig1.canvas.copy_from_bbox(self.fig1.bbox)
            self._bg1_valid = True

    def _on_draw_fig2(self, event):
        if self.fig2 is not None:
            self._bg2 = self.fig2.canvas.copy_from_bbox(self.fig2.bbox)
            self._bg2_valid = True

    def _on_resize_fig1(self, event):
        self._bg1_valid = False

    def _on_resize_fig2(self, event):
        self._bg2_valid = False

    def setup(self):
        """Create figures, axes, and line objects."""
        plt.ion()

        # ---- Figure 1: Gaze / Head Pose ----
        self.fig1, self.axes1 = plt.subplots(
            nrows=3, ncols=1, figsize=(6, 8), sharex=True
        )
        self.fig1.suptitle("L2CS-Net Head Pose Output")
        self.fig1.canvas.manager.set_window_title("Window 1: Head Pose")

        # Yaw
        self.axes1[0].set_ylabel("Yaw", rotation=0, ha='right')
        self.axes1[0].grid(True)
        self.lines['yaw'], = self.axes1[0].plot([], [], label='Yaw', color='blue')
        self.axes1[0].legend(loc='upper right')

        # Pitch
        self.axes1[1].set_ylabel("Pitch", rotation=0, ha='right')
        self.axes1[1].grid(True)
        self.lines['pitch'], = self.axes1[1].plot([],[], label='Pitch', color='green')
        self.axes1[1].legend(loc='upper right')

        # Roll
        self.axes1[2].set_ylabel("Roll", rotation=0, ha='right')
        self.axes1[2].grid(True)
        self.lines['roll'], = self.axes1[2].plot([], [], label='Roll', color='orange')
        self.axes1[2].legend(loc='upper right')
        self.axes1[2].set_xlabel("Frame Number")

        plt.figure(self.fig1.number)
        plt.tight_layout()

        # ---- Figure 2: Derivatives & Confidence ----
        self.fig2, self.axes2 = plt.subplots(
            nrows=3, ncols=1, figsize=(6, 8), sharex=True
        )
        self.fig2.suptitle("Deltas, Confidence, & Acceleration")
        self.fig2.canvas.manager.set_window_title("Window 2: Derivatives")

        # Deltas
        self.axes2[0].set_ylabel(f"Delta ({self.delta_interval}f)", rotation=0, ha='right')
        self.axes2[0].grid(True)
        self.lines['d_yaw'], = self.axes2[0].plot([], [], label='dYaw')
        self.lines['d_pitch'], = self.axes2[0].plot([],[], label='dPitch')
        self.lines['d_roll'], = self.axes2[0].plot([],[], label='dRoll')
        self.axes2[0].legend(loc='upper right')

        # Confidence
        self.axes2[1].set_ylabel("Confidence", rotation=0, ha='right')
        self.axes2[1].grid(True)
        self.axes2[1].set_ylim(0, 1.1)
        self.lines['conf'], = self.axes2[1].plot([],[], label='Conf', color='green')
        self.axes2[1].legend(loc='upper right')

        # Head acceleration
        self.axes2[2].set_ylabel("Head Acc", rotation=0, ha='right')
        self.axes2[2].grid(True)
        self.lines['h_ayaw'], = self.axes2[2].plot([], [], label='Ay')
        self.lines['h_apitch'], = self.axes2[2].plot([], [], label='Ap')
        self.lines['h_aroll'], = self.axes2[2].plot([],[], label='Ar')
        self.axes2[2].legend(loc='upper right')
        self.axes2[2].set_xlabel("Frame Number")

        plt.figure(self.fig2.number)
        plt.tight_layout()

        # Connect event handlers
        self.fig1.canvas.mpl_connect('draw_event', self._on_draw_fig1)
        self.fig1.canvas.mpl_connect('resize_event', self._on_resize_fig1)
        self.fig2.canvas.mpl_connect('draw_event', self._on_draw_fig2)
        self.fig2.canvas.mpl_connect('resize_event', self._on_resize_fig2)

        self.fig1.canvas.draw()
        self.fig2.canvas.draw()

    def reset(self):
        """Clear all history deques."""
        for deque_attr in[
            self.frame_indices, self.yaw_history, self.pitch_history,
            self.roll_history, self.delta_yaw_history, self.delta_pitch_history,
            self.delta_roll_history, self.confidence_history,
            self.head_ayaw_history, self.head_apitch_history, self.head_aroll_history,
        ]:
            deque_attr.clear()
        self._update_counter = 0
        self._last_xlim_fig1 = None
        self._last_xlim_fig2 = None
        self._bg1_valid = False
        self._bg2_valid = False

    def append_data(self, frame_count: int, state: TrackingState, smoothed_confidence: float):
        """Append one frame's data to history deques."""
        def _val(v):
            return v if v is not None else np.nan

        self.frame_indices.append(frame_count)
        self.yaw_history.append(_val(state.smoothed_yaw))
        self.pitch_history.append(_val(state.smoothed_pitch))
        self.roll_history.append(_val(state.smoothed_roll))
        self.delta_yaw_history.append(state.delta_yaw)
        self.delta_pitch_history.append(state.delta_pitch)
        self.delta_roll_history.append(state.delta_roll)
        self.confidence_history.append(smoothed_confidence)
        self.head_ayaw_history.append(_val(state.smoothed_head_ayaw))
        self.head_apitch_history.append(_val(state.smoothed_head_apitch))
        self.head_aroll_history.append(_val(state.smoothed_head_aroll))

    def update(self, frame_count: int):
        """Update plots if the interval has elapsed."""
        if frame_count % self.update_interval != 0:
            return

        self._update_counter += 1

        try:
            frames_arr = np.array(self.frame_indices)
            arrays = self._convert_all_to_arrays()

            self._set_all_line_data(frames_arr, arrays)

            for ax in self.axes1:
                ax.relim()
                ax.autoscale_view(True, True, True)
            for ax in self.axes2:
                ax.relim()
                ax.autoscale_view(True, True, True)

            needs_full_redraw = self._check_limits_changed()

            if (not self._bg1_valid or not self._bg2_valid 
                or needs_full_redraw 
                or self._update_counter % self._FORCED_FULL_REDRAW_INTERVAL == 0):
                self._full_redraw()
            else:
                self._blit_update()

        except Exception as e:
            try:
                self._full_redraw()
            except Exception as e2:
                print(f"Error updating plot: {e}, fallback also failed: {e2}")

    def _check_limits_changed(self) -> bool:
        changed = False

        current_xlim1 = self.axes1[-1].get_xlim()
        if self._last_xlim_fig1 != current_xlim1:
            self._last_xlim_fig1 = current_xlim1
            changed = True

        current_xlim2 = self.axes2[-1].get_xlim()
        if self._last_xlim_fig2 != current_xlim2:
            self._last_xlim_fig2 = current_xlim2
            changed = True

        return changed

    def _convert_all_to_arrays(self) -> dict:
        return {
            'yaw': np.array(self.yaw_history),
            'pitch': np.array(self.pitch_history),
            'roll': np.array(self.roll_history),
            'd_yaw': np.array(self.delta_yaw_history),
            'd_pitch': np.array(self.delta_pitch_history),
            'd_roll': np.array(self.delta_roll_history),
            'conf': np.array(self.confidence_history),
            'h_ayaw': np.array(self.head_ayaw_history),
            'h_apitch': np.array(self.head_apitch_history),
            'h_aroll': np.array(self.head_aroll_history),
        }

    def _set_all_line_data(self, frames_arr: np.ndarray, arrays: dict):
        for key, line in self.lines.items():
            if key in arrays:
                line.set_data(frames_arr, arrays[key])

    def _full_redraw(self):
        self.fig1.canvas.draw_idle()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw_idle()
        self.fig2.canvas.flush_events()

    def _blit_update(self):
        if self._bg1 is None or self._bg2 is None:
            self._full_redraw()
            return

        # ---- Fig 1 ----
        self.fig1.canvas.restore_region(self._bg1)
        for key in ['yaw', 'pitch', 'roll']:
            if key in self.lines:
                self.lines[key].axes.draw_artist(self.lines[key])
        self.fig1.canvas.blit(self.fig1.bbox)
        self.fig1.canvas.flush_events()

        # ---- Fig 2 ----
        self.fig2.canvas.restore_region(self._bg2)
        for key in['d_yaw', 'd_pitch', 'd_roll', 'conf', 'h_ayaw', 'h_apitch', 'h_aroll']:
            if key in self.lines:
                self.lines[key].axes.draw_artist(self.lines[key])
        self.fig2.canvas.blit(self.fig2.bbox)
        self.fig2.canvas.flush_events()

    def cleanup(self):
        plt.ioff()
        plt.close('all')