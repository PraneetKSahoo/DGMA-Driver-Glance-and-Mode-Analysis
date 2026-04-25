"""
preprocessing.py
----------------
Video frame preprocessing: cropping, upscaling, and image enhancement.

Provides functions for preparing raw video frames for face detection,
including region-of-interest cropping, optional super-resolution upscaling,
and contrast/brightness/denoising adjustments.

Fixes addressed:
    #17 - Preprocessing parameters are named constants imported from config

Usage:
    frame = crop_to_first_quadrant(raw_frame)
    if sr:
        frame = upscale_frame(sr, frame, factor=2)
    frame = enhance_frame(frame)
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple

from config import CONTRAST_ALPHA, BRIGHTNESS_BETA, BLUR_KERNEL


# =============================================================================
# Frame Cropping
# =============================================================================

def crop_to_first_quadrant(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop frame to the upper-right region (driver face area in typical dashcam).

    Extracts the rightmost quarter (width) and top third (height) of the frame,
    which is where the driver's face typically appears in front-facing dashcam
    video with a standard mounting position.

    Crop region:
        x: [3/4 * width, width]
        y: [0, 1/3 * height]

    Args:
        frame: Input BGR frame (numpy array).

    Returns:
        Cropped frame, or the original frame if dimensions are too small
        or the crop region is invalid. Returns None if frame is None.
    """
    if frame is None:
        return None

    height, width = frame.shape[:2]

    if height <= 1 or width <= 1:
        return frame

    x_start = 3 * width // 4
    x_end = width
    y_start = 0
    y_end = height // 3

    if x_start >= x_end or y_start >= y_end:
        return frame

    return frame[y_start:y_end, x_start:x_end]


# =============================================================================
# Image Enhancement (#17: uses named constants from config)
# =============================================================================

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply mild contrast, brightness, and denoising adjustments.

    Preprocessing pipeline:
    1. Contrast and brightness adjustment via cv2.convertScaleAbs():
       - CONTRAST_ALPHA (1.1): 10% contrast boost to improve edge definition
         in low-quality dashcam crops, helping MediaPipe detect facial features.
       - BRIGHTNESS_BETA (5): Small brightness offset to lift shadow regions
         slightly, aiding detection in poorly lit cabin environments.

    2. Gaussian blur via cv2.GaussianBlur():
       - BLUR_KERNEL (3,3): Minimal smoothing to reduce JPEG/H.264 compression
         artifacts without significantly blurring facial landmarks. Larger
         kernels would degrade landmark precision.

    Args:
        frame: Input BGR frame (numpy array).

    Returns:
        Enhanced frame (same dimensions and dtype as input).
    """
    frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)
    frame = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
    return frame


# =============================================================================
# Super-Resolution Upscaling
# =============================================================================

def setup_upscaler(factor: int) -> Optional[object]:
    """
    Initialize an FSRCNN super-resolution upscaler.

    Downloads the model file from GitHub if it's missing or appears corrupt
    (file size < 1000 bytes). Uses OpenCV's DNN backend on CPU.

    Args:
        factor: Upscale factor (2 or 4).

    Returns:
        Initialized cv2.dnn_superres.DnnSuperResImpl object,
        or None if setup fails.
    """
    import urllib.request

    model_filename = f'FSRCNN_x{factor}.pb'

    # Download model if missing or corrupt
    if not os.path.exists(model_filename) or os.path.getsize(model_filename) < 1000:
        print(f"Model {model_filename} missing or invalid. Downloading...")
        url = (
            f"https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/"
            f"models/FSRCNN_x{factor}.pb"
        )
        try:
            urllib.request.urlretrieve(url, model_filename)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading upscaler model: {e}")
            return None

    # Initialize super-resolution
    print(f"Initializing FSRCNN x{factor} Upscaler...")
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_filename)
        sr.setModel("fsrcnn", factor)
        print("Using CPU mode for upscaling")
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return sr
    except Exception as e:
        print(f"Error loading upscaler model: {e}")
        print("Upscaling disabled.")
        return None


def upscale_frame(
    sr: object,
    frame: np.ndarray,
    factor: int,
) -> np.ndarray:
    """
    Upscale a frame using the FSRCNN super-resolution model.

    Falls back to cubic interpolation if FSRCNN fails.

    Args:
        sr: Initialized DnnSuperResImpl object.
        frame: Input BGR frame.
        factor: Upscale factor (must match the sr model's factor).

    Returns:
        Upscaled frame.
    """
    try:
        return sr.upsample(frame)
    except Exception as e:
        print(f"FSRCNN upscaling failed: {e}, falling back to INTER_CUBIC")
        h, w = frame.shape[:2]
        return cv2.resize(
            frame,
            (w * factor, h * factor),
            interpolation=cv2.INTER_CUBIC,
        )