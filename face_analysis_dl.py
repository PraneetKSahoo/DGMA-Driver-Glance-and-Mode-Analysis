"""
face_analysis_dl.py
-------------------
Deep Learning Gaze Estimation using L2CS-Net (ResNet50 Backbone).
Replaces MediaPipe heuristics with a true PyTorch neural network.
"""

import cv2
import torch
import numpy as np
from l2cs import Pipeline, render

class DeepGazeTracker:
    def __init__(self, model_path='L2CSNet_gaze360.pkl', device='cuda'):
        """
        Initializes the L2CS-Net pipeline on the GPU.
        """
        print(f"Loading L2CS-Net onto {device.upper()}...")
        self.pipeline = Pipeline(
            weights=model_path,
            arch='ResNet50',
            device=torch.device(device)
        )
        print("✅ Deep Learning Model Ready!")

    def process_frame(self, frame_bgr):
        """
        Processes a single BGR frame.
        Finds the face and calculates the absolute Gaze Yaw and Pitch.
        """
        try:
            # The pipeline handles face detection and Gaze inference automatically
            results = self.pipeline.step(frame_bgr)
            
            # If no face is detected, return None
            if results is None or getattr(results, 'bboxes', None) is None or results.bboxes.shape[0] == 0:
                return None

            # We assume the driver is the primary face (index 0)
            yaw_deg = float(np.degrees(results.yaw[0]))
            pitch_deg = float(np.degrees(results.pitch[0]))
            
            # Invert pitch to match your old MediaPipe coordinate system (down = negative)
            pitch_deg = -pitch_deg 

            bbox = results.bboxes[0] #[x_min, y_min, x_max, y_max]

            # Extract the Face Detector's confidence score (fallback to 0.0 if not found)
            confidence = float(results.scores[0]) if hasattr(results, 'scores') and results.scores is not None else 0.0

            return {
                'yaw': yaw_deg,
                'pitch': pitch_deg,
                'bbox': bbox,
                'confidence': confidence
            }
            
        except ValueError as e:
            # Silently catch the L2CS-Net bug where it tries to np.stack() an empty face list
            if "stack" in str(e).lower() or "empty" in str(e).lower():
                return None
            # If it's a different ValueError, we still want to ignore it so it doesn't spam the console
            return None
        except Exception as e:
            # Catch any other random face detector glitches silently
            return None

    def draw_gaze(self, frame_bgr, results_dict):
        """
        Helper to draw the true 3D gaze vector on the frame.
        """
        if results_dict is None:
            return frame_bgr
            
        # Create a mock results object to use L2CS's built-in 3D renderer
        class MockResults:
            pass
        res = MockResults()
        res.bboxes = np.array([results_dict['bbox']])
        res.yaw = np.array([np.radians(results_dict['yaw'])])
        res.pitch = np.array([np.radians(-results_dict['pitch'])]) # Re-invert for renderer
        
        # Draws red and blue 3D gaze axes projecting out of the face
        return render(frame_bgr, res)