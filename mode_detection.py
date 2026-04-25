"""
mode_detection.py
-----------------
CPU-based driving mode detection (Automated vs Manual) by detecting a green
steering wheel icon on the vehicle's instrument cluster.

Ported from the GPU-accelerated (cv2.cuda) implementation in the reference
project. Uses CPU-only OpenCV operations since mode detection only runs
every MODE_CHECK_INTERVAL_SECONDS (~10s), making GPU acceleration unnecessary.

Features:
    - ModeDetector: Detects mode from a single frame
    - ModeTracker: Tracks mode transitions over time
    - draw_rois_interactive(): Interactive ROI drawing on video frames
    - Optional debug visualization with mask/contour overlays

Usage:
    detector = ModeDetector(car_type='n')
    custom_rois = draw_rois_interactive(cap, 'n', fps)
    if custom_rois:
        detector.update_rois(custom_rois)
    mode, debug_image = detector.detect(full_frame, return_debug=True)
    tracker = ModeTracker()
    tracker.update(mode, current_time, frame_count)
"""

import cv2
import numpy as np
import collections
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import simpledialog

from config import (
    CAR_TYPES,
    DEFAULT_CAR_TYPE,
    MODE_HSV_RANGES,
    MODE_REF_WIDTH,
    MODE_REF_HEIGHT,
    MODE_MORPH_KERNEL_SIZE,
    MODE_CHECK_INTERVAL_SECONDS,
)


# =============================================================================
# Interactive ROI Drawing
# =============================================================================

def draw_rois_interactive(
    cap: cv2.VideoCapture,
    car_type: str,
    video_fps: float,
) -> Optional[dict]:
    """
    Interactive ROI drawing on a video frame's third quadrant.

    Flow:
    1. Ask user for a frame number via tkinter dialog
    2. Read that frame and crop to third quadrant
    3. Show the cropped frame with grid overlay
    4. User draws exactly 2 rectangles (first = top_left, second = bottom_center)
    5. Drawn coordinates are scaled to reference resolution (944x480)
    6. Returns ROI dict merged with default shape params for the car type

    Controls:
        - Click and drag to draw a rectangle
        - 'c' to confirm (requires exactly 2 ROIs)
        - 'r' to reset all drawn ROIs
        - 'q' to cancel and use default ROIs

    Args:
        cap: Opened VideoCapture (position will be restored after).
        car_type: Car type code for default shape parameters.
        video_fps: Video FPS (for frame-to-time display).

    Returns:
        Dictionary with 'top_left' and 'bottom_center' ROI definitions
        scaled to reference resolution, or None if cancelled.
    """
    # Save current position to restore later
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ask for frame number
    root = tk.Tk()
    root.withdraw()
    frame_input = simpledialog.askstring(
        "Select Frame for ROI Drawing",
        f"Enter frame number (0 to {total_frames - 1}):\n"
        f"(Choose a frame where the instrument cluster is clearly visible)",
        parent=root,
    )
    root.destroy()

    if frame_input is None:
        print("ROI drawing cancelled. Using default ROIs.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        return None

    try:
        frame_number = int(frame_input)
        if not (0 <= frame_number < total_frames):
            print(f"Invalid frame number {frame_number}. Using default ROIs.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
            return None
    except (ValueError, TypeError):
        print("Invalid input. Using default ROIs.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        return None

    # Read the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    # Restore original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if not ret or frame is None:
        print(f"Could not read frame {frame_number}. Using default ROIs.")
        return None

    # Crop to third quadrant (bottom-left)
    height, width = frame.shape[:2]
    cropped = frame[height // 2:height, 0:width // 2]

    if cropped.size == 0:
        print("Cropped frame is empty. Using default ROIs.")
        return None

    crop_height, crop_width = cropped.shape[:2]

    # Drawing state (using mutable container for closure access)
    draw_state = {
        'drawing': False,
        'start_point': (-1, -1),
        'end_point': (-1, -1),
        'rois_drawn': [],  # List of (x, y, w, h) tuples
        'base_frame': cropped.copy(),
    }

    window_name = (
        "Draw ROIs: Click+Drag | "
        "'c'=Confirm (need 2) | 'r'=Reset | 'q'=Cancel"
    )

    def _draw_grid(image, spacing=50):
        """Draw a reference grid on the image."""
        h, w = image.shape[:2]
        for x in range(0, w, spacing):
            cv2.line(image, (x, 0), (x, h), (128, 128, 128), 1)
        for y in range(0, h, spacing):
            cv2.line(image, (0, y), (w, y), (128, 128, 128), 1)

    def _get_display_frame():
        """Create a fresh display frame with grid and existing ROIs."""
        display = draw_state['base_frame'].copy()
        _draw_grid(display)

        roi_labels = ["top_left", "bottom_center"]
        for i, roi in enumerate(draw_state['rois_drawn']):
            x, y, w, h = roi
            label = roi_labels[i] if i < len(roi_labels) else f"roi_{i}"
            color = (255, 0, 0) if i == 0 else (0, 165, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                display, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        # Info text
        drawn_count = len(draw_state['rois_drawn'])
        time_str = f"{frame_number / video_fps:.1f}s" if video_fps > 0 else "?"
        cv2.putText(
            display,
            f"Frame: {frame_number} ({time_str}) | "
            f"ROIs: {drawn_count}/2",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        if drawn_count < 2:
            next_label = roi_labels[drawn_count]
            cv2.putText(
                display,
                f"Draw: {next_label}",
                (10, crop_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )
        else:
            cv2.putText(
                display,
                "Press 'c' to confirm or 'r' to reset",
                (10, crop_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

        return display

    def mouse_callback(event, x, y, flags, param):
        """Handle mouse events for rectangle drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(draw_state['rois_drawn']) >= 2:
                return  # Already have 2 ROIs
            draw_state['drawing'] = True
            draw_state['start_point'] = (x, y)
            draw_state['end_point'] = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and draw_state['drawing']:
            draw_state['end_point'] = (x, y)
            # Show live rectangle preview
            display = _get_display_frame()
            cv2.rectangle(
                display,
                draw_state['start_point'],
                draw_state['end_point'],
                (0, 255, 0), 2,
            )
            cv2.imshow(window_name, display)

        elif event == cv2.EVENT_LBUTTONUP and draw_state['drawing']:
            draw_state['drawing'] = False
            draw_state['end_point'] = (x, y)

            # Calculate rectangle
            sx, sy = draw_state['start_point']
            ex, ey = draw_state['end_point']
            rx = min(sx, ex)
            ry = min(sy, ey)
            rw = abs(ex - sx)
            rh = abs(ey - sy)

            # Minimum size check
            if rw > 10 and rh > 10:
                draw_state['rois_drawn'].append((rx, ry, rw, rh))
                idx = len(draw_state['rois_drawn']) - 1
                label = (
                    "top_left" if idx == 0
                    else "bottom_center" if idx == 1
                    else f"roi_{idx}"
                )
                print(f"  ROI '{label}' drawn: x={rx}, y={ry}, "
                      f"w={rw}, h={rh}")
            else:
                print("  Rectangle too small (min 10x10). Try again.")

            cv2.imshow(window_name, _get_display_frame())

    # Setup window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, _get_display_frame())

    print(f"\nROI Drawing Mode:")
    print(f"  Frame {frame_number} | Third quadrant: {crop_width}x{crop_height}")
    print(f"  Draw 2 rectangles: first = top_left, second = bottom_center")
    print(f"  'c' = confirm | 'r' = reset | 'q' = cancel (use defaults)")

    # Event loop
    result_rois = None
    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord('c'):
            if len(draw_state['rois_drawn']) == 2:
                result_rois = _build_roi_dict(
                    draw_state['rois_drawn'],
                    crop_width, crop_height,
                    car_type,
                )
                print("ROIs confirmed.")
                break
            else:
                print(f"  Need exactly 2 ROIs "
                      f"(have {len(draw_state['rois_drawn'])}). "
                      f"Draw more or press 'q' to cancel.")

        elif key == ord('r'):
            draw_state['rois_drawn'].clear()
            print("  ROIs reset. Draw again.")
            cv2.imshow(window_name, _get_display_frame())

        elif key == ord('q'):
            print("ROI drawing cancelled. Using default ROIs.")
            result_rois = None
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  # Process window destruction

    return result_rois


def _build_roi_dict(
    drawn_rois: list,
    crop_width: int,
    crop_height: int,
    car_type: str,
) -> dict:
    """
    Build a ROI dictionary from drawn rectangles.

    Scales pixel coordinates to reference resolution and merges with
    default shape parameters for the car type.

    Args:
        drawn_rois: List of 2 tuples [(x, y, w, h), (x, y, w, h)].
        crop_width: Width of the cropped frame where ROIs were drawn.
        crop_height: Height of the cropped frame.
        car_type: Car type code for default shape parameters.

    Returns:
        ROI dictionary with 'top_left' and 'bottom_center' entries.
    """
    x_scale = MODE_REF_WIDTH / crop_width
    y_scale = MODE_REF_HEIGHT / crop_height

    default_rois = CAR_TYPES[car_type]['rois']

    def _scale_roi(drawn, default_key):
        x, y, w, h = drawn
        defaults = default_rois[default_key]
        return {
            'x': int(x * x_scale),
            'y': int(y * y_scale),
            'width': int(w * x_scale),
            'height': int(h * y_scale),
            'min_area': defaults['min_area'],
            'max_area': defaults['max_area'],
            'circularity_min': defaults['circularity_min'],
            'circularity_max': defaults['circularity_max'],
            'aspect_ratio_min': defaults['aspect_ratio_min'],
            'aspect_ratio_max': defaults['aspect_ratio_max'],
        }

    return {
        'top_left': _scale_roi(drawn_rois[0], 'top_left'),
        'bottom_center': _scale_roi(drawn_rois[1], 'bottom_center'),
    }


# =============================================================================
# Mode Detector
# =============================================================================

class ModeDetector:
    """
    Detects driving mode (Automated/Manual) from instrument cluster imagery.

    Looks for a green steering wheel icon in predefined ROIs within the
    third quadrant (bottom-left) of the video frame.

    Supports optional debug visualization showing ROI rectangles, color masks,
    and detected contours.

    Attributes:
        car_type (str): Car type code.
        rois (dict): ROI definitions (may be updated with custom ROIs).
        hsv_ranges (dict): HSV color thresholds.
        kernel (np.ndarray): Morphology kernel.
    """

    AUTOMATED = "Automated Mode"
    MANUAL = "Manual Mode"

    # Visualization colors
    _VIZ_ROI_COLOR = (0, 255, 0)
    _VIZ_CONTOUR_COLOR = (255, 0, 0)
    _VIZ_MASK_ALPHA = 0.3
    _VIZ_GRID_COLOR = (128, 128, 128)
    _VIZ_GRID_SPACING = 50

    def __init__(self, car_type: str = DEFAULT_CAR_TYPE):
        """
        Initialize the mode detector for a specific car type.

        Args:
            car_type: Car type code. Must be a key in CAR_TYPES.

        Raises:
            ValueError: If car_type is not recognized.
        """
        if car_type not in CAR_TYPES:
            available = ', '.join(
                f"'{k}' ({v['label']})" for k, v in CAR_TYPES.items()
            )
            raise ValueError(
                f"Unknown car type '{car_type}'. Available: {available}"
            )

        self.car_type = car_type
        self.rois = CAR_TYPES[car_type]['rois'].copy()
        self.hsv_ranges = MODE_HSV_RANGES[car_type]
        self.kernel = np.ones(
            (MODE_MORPH_KERNEL_SIZE, MODE_MORPH_KERNEL_SIZE), np.uint8
        )

        car_label = CAR_TYPES[car_type]['label']
        print(f"Mode detector initialized for: {car_label} (type='{car_type}')")

    def update_rois(self, new_rois: dict):
        """
        Replace current ROIs with custom user-drawn ROIs.

        Args:
            new_rois: Dictionary with 'top_left' and 'bottom_center' ROI defs.
        """
        if 'top_left' in new_rois and 'bottom_center' in new_rois:
            self.rois = new_rois.copy()
            print("Mode detector ROIs updated with custom definitions.")
            print(f"  top_left: x={new_rois['top_left']['x']}, "
                  f"y={new_rois['top_left']['y']}, "
                  f"w={new_rois['top_left']['width']}, "
                  f"h={new_rois['top_left']['height']}")
            print(f"  bottom_center: x={new_rois['bottom_center']['x']}, "
                  f"y={new_rois['bottom_center']['y']}, "
                  f"w={new_rois['bottom_center']['width']}, "
                  f"h={new_rois['bottom_center']['height']}")
        else:
            print("Warning: Invalid ROI dict (missing keys). ROIs not updated.")

    def detect(
        self,
        full_frame: np.ndarray,
        return_debug: bool = False,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Detect driving mode from a full video frame.

        Args:
            full_frame: Full BGR video frame (not cropped).
            return_debug: If True, returns a debug visualization image
                          showing ROIs, masks, and contours.

        Returns:
            Tuple of (mode_string, debug_image_or_None).
            mode_string: "Automated Mode" or "Manual Mode".
            debug_image: BGR image of the third quadrant with overlays,
                         or None if return_debug is False.
        """
        # Crop to third quadrant
        cropped = self._crop_to_third_quadrant(full_frame)
        if cropped is None or cropped.size == 0:
            return self.MANUAL, None

        # Convert to HSV
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Scale ROIs
        height, width = cropped.shape[:2]
        scaled_rois = self._scale_rois(width, height)

        # Prepare debug image if requested
        debug_image = None
        if return_debug:
            debug_image = cropped.copy()
            self._draw_grid(debug_image)

        # Check both ROIs
        top_left_found = self._check_roi(
            hsv, scaled_rois['top_left'],
            debug_image=debug_image,
            roi_label='top_left',
        )
        bottom_center_found = self._check_roi(
            hsv, scaled_rois['bottom_center'],
            debug_image=debug_image,
            roi_label='bottom_center',
        )

        mode = (
            self.AUTOMATED
            if top_left_found or bottom_center_found
            else self.MANUAL
        )

        # Add mode label to debug image
        if debug_image is not None:
            mode_color = (
                (0, 255, 0) if mode == self.AUTOMATED else (0, 0, 255)
            )
            cv2.putText(
                debug_image, f"Mode: {mode}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2,
            )
            cv2.putText(
                debug_image,
                f"TL: {'YES' if top_left_found else 'no'} | "
                f"BC: {'YES' if bottom_center_found else 'no'}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        return mode, debug_image

    def _crop_to_third_quadrant(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Crop frame to the bottom-left quadrant."""
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if height <= 1 or width <= 1:
            return None
        return frame[height // 2:height, 0:width // 2]

    def _scale_rois(self, width: int, height: int) -> dict:
        """Scale ROI coordinates from reference resolution to current size."""
        area_scale = (width * height) / (MODE_REF_WIDTH * MODE_REF_HEIGHT)
        x_scale = width / MODE_REF_WIDTH
        y_scale = height / MODE_REF_HEIGHT

        scaled = {}
        for roi_name, roi in self.rois.items():
            scaled[roi_name] = {
                'x': int(roi['x'] * x_scale),
                'y': int(roi['y'] * y_scale),
                'width': int(roi['width'] * x_scale),
                'height': int(roi['height'] * y_scale),
                'min_area': int(roi['min_area'] * area_scale),
                'max_area': int(roi['max_area'] * area_scale),
                'circularity_min': roi['circularity_min'],
                'circularity_max': roi['circularity_max'],
                'aspect_ratio_min': roi['aspect_ratio_min'],
                'aspect_ratio_max': roi['aspect_ratio_max'],
            }
        return scaled

    def _draw_grid(self, image: np.ndarray):
        """Draw reference grid on debug image."""
        h, w = image.shape[:2]
        for x in range(0, w, self._VIZ_GRID_SPACING):
            cv2.line(image, (x, 0), (x, h), self._VIZ_GRID_COLOR, 1)
        for y in range(0, h, self._VIZ_GRID_SPACING):
            cv2.line(image, (0, y), (w, y), self._VIZ_GRID_COLOR, 1)

    def _check_roi(
        self,
        hsv_image: np.ndarray,
        roi: dict,
        debug_image: Optional[np.ndarray] = None,
        roi_label: str = "",
    ) -> bool:
        """
        Check a single ROI for the steering wheel icon shape.

        If debug_image is provided, draws ROI rectangle, mask overlay,
        and detected contours onto it.

        Args:
            hsv_image: Full HSV image of the cropped third quadrant.
            roi: Scaled ROI dictionary.
            debug_image: Optional image to draw debug overlays on.
            roi_label: Label string for debug annotation.

        Returns:
            True if a steering-wheel-shaped contour was found.
        """
        h, w = hsv_image.shape[:2]

        # Clip ROI to image bounds
        x1 = max(0, roi['x'])
        y1 = max(0, roi['y'])
        x2 = min(w, roi['x'] + roi['width'])
        y2 = min(h, roi['y'] + roi['height'])

        if x2 <= x1 or y2 <= y1:
            return False

        roi_region = hsv_image[y1:y2, x1:x2]

        # Color thresholding
        green_lower = np.array(self.hsv_ranges['green_lower'])
        green_upper = np.array(self.hsv_ranges['green_upper'])
        white_lower = np.array(self.hsv_ranges['white_lower'])
        white_upper = np.array(self.hsv_ranges['white_upper'])

        mask_green = cv2.inRange(roi_region, green_lower, green_upper)
        mask_white = cv2.inRange(roi_region, white_lower, white_upper)
        mask = cv2.bitwise_or(mask_green, mask_white)

        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        # Contour analysis
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        has_shape = False
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if area < roi['min_area'] or area > roi['max_area']:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            _, _, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = float(w_c) / h_c if h_c != 0 else 0

            if (roi['circularity_min'] < circularity < roi['circularity_max']
                    and roi['aspect_ratio_min'] < aspect_ratio < roi['aspect_ratio_max']):
                has_shape = True
                break

        # Debug visualization
        if debug_image is not None:
            # Draw ROI rectangle
            roi_color = (0, 255, 0) if has_shape else (0, 0, 255)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), roi_color, 2)

            # Label
            if roi_label:
                status = "FOUND" if has_shape else "none"
                cv2.putText(
                    debug_image,
                    f"{roi_label}: {status}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1,
                )

            # Mask overlay
            roi_view = debug_image[y1:y2, x1:x2]
            if roi_view.shape[:2] == mask.shape[:2]:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(
                    roi_view, 1.0 - self._VIZ_MASK_ALPHA,
                    mask_rgb, self._VIZ_MASK_ALPHA,
                    0,
                )
                debug_image[y1:y2, x1:x2] = overlay

            # Contour outlines (shifted to absolute coordinates)
            for contour in contours:
                shifted = contour + np.array([[x1, y1]])
                cv2.drawContours(
                    debug_image, [shifted], -1,
                    self._VIZ_CONTOUR_COLOR, 1,
                )

        return has_shape

    def __repr__(self) -> str:
        label = CAR_TYPES[self.car_type]['label']
        return f"ModeDetector(car_type='{self.car_type}', label='{label}')"


# =============================================================================
# Mode Tracker
# =============================================================================

class ModeTracker:
    """
    Tracks driving mode transitions over time.

    Records mode intervals (start/end times) and provides lookup
    functionality for determining which mode was active at any given time.
    """

    def __init__(self):
        """Initialize the mode tracker."""
        self.mode_intervals: List[dict] = []
        self.current_mode: Optional[str] = None
        self.current_mode_start: float = 0.0
        self.current_mode_start_frame: int = 0

    def reset(self):
        """Reset mode tracking state."""
        self.mode_intervals.clear()
        self.current_mode = None
        self.current_mode_start = 0.0
        self.current_mode_start_frame = 0

    def update(self, mode: str, current_time: float, frame_count: int):
        """
        Update mode tracking with a new detection result.

        Args:
            mode: Detected mode string.
            current_time: Current video time in seconds.
            frame_count: Current frame number.
        """
        if self.current_mode is None:
            self.current_mode = mode
            self.current_mode_start = current_time
            self.current_mode_start_frame = frame_count
            print(f"\r\033[K"
                  f"Initial mode detected: {mode} at {current_time:.2f}s")
            return

        if mode != self.current_mode:
            print(f"\r\033[K"
                  f"Mode change: {self.current_mode} -> {mode} "
                  f"at {current_time:.2f}s")
            self._finalize_current_interval(current_time, frame_count)
            self.current_mode = mode
            self.current_mode_start = current_time
            self.current_mode_start_frame = frame_count

    def _finalize_current_interval(self, end_time: float, end_frame: int):
        """Record the current mode interval."""
        if self.current_mode is not None:
            duration = end_time - self.current_mode_start
            if duration > 0:
                self.mode_intervals.append({
                    "mode": self.current_mode,
                    "start": self.current_mode_start,
                    "end": end_time,
                    "duration": duration,
                    "start_frame": self.current_mode_start_frame,
                    "end_frame": end_frame,
                })

    def flush(self, final_time: float, final_frame: int):
        """Flush the last active mode interval."""
        self._finalize_current_interval(final_time, final_frame)

    def get_mode_at_time(self, query_time: float) -> Optional[str]:
        """Look up which mode was active at a specific time."""
        for interval in self.mode_intervals:
            if interval['start'] <= query_time <= interval['end']:
                return interval['mode']

        if (self.current_mode is not None
                and query_time >= self.current_mode_start):
            return self.current_mode

        return None

    def get_mode_durations(self) -> dict:
        """Compute total duration spent in each mode."""
        durations = collections.defaultdict(float)
        for interval in self.mode_intervals:
            durations[interval['mode']] += interval['duration']
        return dict(durations)

    def print_summary(self):
        """Print a summary of mode detection results."""
        durations = self.get_mode_durations()
        total = sum(durations.values())

        print(f"\n--- Mode Detection Summary ---")
        for mode, dur in sorted(durations.items()):
            pct = (dur / total * 100) if total > 0 else 0
            print(f"  {mode}: {dur:.2f}s ({pct:.1f}%)")
        print(f"  Total tracked: {total:.2f}s")
        print(f"  Mode transitions: {len(self.mode_intervals)}")

    def __repr__(self) -> str:
        return (
            f"ModeTracker(current={self.current_mode}, "
            f"intervals={len(self.mode_intervals)})"
        )