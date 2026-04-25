# DGMA — Driver Glance and Mode Analysis

A deep learning pipeline for analyzing driver head pose and eye gaze in dashcam footage. Uses **L2CS-Net** (ResNet50 backbone, PyTorch/CUDA) to classify On-Road vs Off-Road glances, detect driving mode transitions, and export structured reports.

 <video src="demo.mp4" controls width="720"></video>

---

## Features

- **Deep learning gaze estimation** via L2CS-Net on CUDA GPU
- **Elliptical on-road zone** classification with Mahalanobis distance
- **Glance tracking** with debouncing, saccade rewind, and grace-period logic
- **Quick Auto-Calibration** — samples windows from the video, fits a Gaussian Mixture Model, and outputs a threshold JSON
- **Driving mode detection** — detects Automated vs Manual mode from instrument cluster imagery (Cadillac, Nissan, Tesla S/3, Volvo)
- **Batch processing GUI** — queue multiple videos for calibration or main processing
- **Optional FSRCNN upscaling** (2x / 4x) for low-resolution footage
- **CSV and JSON exports** — per-frame gaze data, glance history, mode intervals, correlation tables, and summary reports

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (developed on RTX 5000)
- CUDA 11.8 toolkit installed at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/`

### Software
- Python 3.8+
- OpenCV built with CUDA at `C:/opencv_build/bin/Release`

### Python Packages

```
torch
torchvision
opencv-python
numpy
l2cs
scikit-learn
matplotlib
```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy l2cs scikit-learn matplotlib
```
>FSRCNN model weights are downloaded automatically on first use (if upscaling is enabled) from github.com/Saafke/FSRCNN_Tensorflow. The .pb file is saved to the working directory and reused on subsequent runs. No manual download needed.

> **Note:** The `l2cs` package requires the `L2CSNet_gaze360.pkl` model weights file to be present in the working directory (or specify the path in `face_analysis_dl.py`).
Get the model from here: [https://github.com/ahmednull/l2cs-net](https://github.com/ahmednull/l2cs-net#demo)

---

## Project Structure

| File | Description |
|---|---|
| `main.py` | Main single-video processing pipeline with optional real-time visualization |
| `batch.py` | Tkinter GUI for batch calibration and processing of multiple videos |
| `quick_calibrate.py` | Standalone auto-calibration tool using GMM clustering |
| `face_analysis_dl.py` | L2CS-Net wrapper (`DeepGazeTracker`) for per-frame gaze inference |
| `glance.py` | Glance zone classification, debouncing, grace period, and report generation |
| `mode_detection.py` | Instrument cluster mode detector (`ModeDetector`, `ModeTracker`) and interactive ROI drawing |
| `state.py` | `TrackingState` — EMA smoothing, pose history, and confidence tracking |
| `preprocessing.py` | Frame cropping, contrast enhancement, and FSRCNN upscaling setup |
| `visualization.py` | OpenCV overlay renderer, matplotlib plot manager, and tkinter settings window |
| `io_utils.py` | Video I/O and CSV/JSON export helpers |
| `config.py` | All constants, thresholds, car type ROI definitions, and `ThresholdConfig` class |
| `rename.py` | Utility for batch renaming output files |

---

## Usage

### 1. Single Video — Main Pipeline

```bash
python main.py
```

On launch you will be prompted to:
- Enable real-time visualizations
- Enable FSRCNN upscaling (2x or 4x)
- Import a threshold JSON (from a prior calibration)
- Set the delta interval for derivative computation
- Enable driving mode detection and select a car type
- Select the video file and optionally define a start/end segment

**Keyboard controls during playback:**
| Key | Action |
|---|---|
| `q` | Quit |
| `Space` | Pause / Resume |
| `h` | Toggle help overlay |
| `j` | Jump to frame (while paused) |
| `s` | Save current thresholds to JSON |

---

### 2. Auto-Calibration

```bash
python quick_calibrate.py
```

Samples 3 × 60-second windows from the video, fits a 2-component GMM, and saves:
- `Thresholds/thresholds_<video>_<timestamp>.json`
- `Plots/calibration_<video>_<timestamp>.png`

---

### 3. Batch Processing GUI

```bash
python batch.py
```

Queue multiple videos for either **Quick Calibrate** or **Main Processing**. Threshold JSONs are auto-matched from the `Thresholds/` folder by video name. Supports per-video timestamps, threshold pairing, and optional ROI drawing for mode detection.

---

## Outputs

All outputs are saved to the working directory:

| File | Contents |
|---|---|
| `gaze_<video>_<timestamp>.csv` | Per-frame yaw, pitch, confidence, zone |
| `glance_history_<video>_<timestamp>.csv` | Each glance event with start/end times and duration |
| `mode_intervals_<video>_<timestamp>.csv` | Automated / Manual mode segments |
| `mode_glance_correlation_<video>_<timestamp>.csv` | Glance breakdown per driving mode |
| `report_<video>_<timestamp>.csv` | Long-form summary report |
| `report_<video>_<timestamp>.json` | JSON summary report |
| `Thresholds/thresholds_<video>_<timestamp>.json` | Calibrated yaw/pitch thresholds |
| `Plots/calibration_<video>_<timestamp>.png` | GMM cluster map + per-window dominance chart |

---

## Supported Car Types (Mode Detection)

| Code | Vehicle |
|---|---|
| `n` | Nissan (default) |
| `c` | Cadillac |
| `s` | Tesla Model S |
| `3` | Tesla Model 3 |
| `v` | Volvo |

Custom ROIs can be drawn interactively for any car type via the batch GUI or when prompted during `main.py`.

---

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `POSE_SMOOTHING_FACTOR` | `0.1` | EMA alpha for head pose smoothing |
| `FRAME_SKIP_FACTOR` | `3` | Process every Nth frame |
| `GLANCE_MIN_DURATION` | `0.01s` | Minimum duration to record a glance |
| `NO_DETECTION_GRACE_PERIOD` | `2s` | Hold last zone when face is lost |
| `OFFROAD_THRESHOLD` | `2.0s` | Boundary between short and prolonged off-road glances |
| `DEFAULT_DELTA_INTERVAL` | `10` | Frames used for head angular velocity computation |
