"""
state.py
--------
Encapsulates all per-frame tracking state for head pose and confidence.

Adapted for L2CS-Net: 
- Removed all facial blendshapes, eye openness, and MediaPipe gaze tracking variables.
- Kept Dynamic EMA for accurate saccade/glance bounding.
"""

import collections
from typing import Optional


class TrackingState:
    def __init__(self, delta_interval: int, pose_alpha: float,
                 confidence_history_len: int = 5):
        self.pose_alpha = pose_alpha
        self.delta_interval = delta_interval
        self._confidence_history_len = confidence_history_len

        # Smoothed values
        self.smoothed_yaw: Optional[float] = None
        self.smoothed_pitch: Optional[float] = None
        self.smoothed_roll: Optional[float] = None

        # Prev frame values
        self.prev_yaw: Optional[float] = None
        self.prev_pitch: Optional[float] = None
        self.prev_roll: Optional[float] = None

        # Smoothed accelerations
        self.smoothed_head_ayaw: float = 0.0
        self.smoothed_head_apitch: float = 0.0
        self.smoothed_head_aroll: float = 0.0

        # Raw (unsmoothed) derivatives
        self.delta_yaw: float = 0.0
        self.delta_pitch: float = 0.0
        self.delta_roll: float = 0.0
        self.head_ayaw: float = 0.0
        self.head_apitch: float = 0.0
        self.head_aroll: float = 0.0

        # Delta history deques
        self.yaw_history: collections.deque = collections.deque(maxlen=delta_interval)
        self.pitch_history: collections.deque = collections.deque(maxlen=delta_interval)
        self.roll_history: collections.deque = collections.deque(maxlen=delta_interval)

        # Confidence
        self.confidence_history: collections.deque = collections.deque(maxlen=confidence_history_len)
        self.current_confidence: float = 0.0
        self.smoothed_confidence: float = 0.0

    def reset(self):
        self.__init__(
            delta_interval=self.delta_interval,
            pose_alpha=self.pose_alpha,
            confidence_history_len=self._confidence_history_len,
        )

    @staticmethod
    def _ema(current_smoothed: Optional[float], new_value: float, alpha: float) -> float:
        if current_smoothed is None:
            return new_value
        return alpha * new_value + (1.0 - alpha) * current_smoothed

    @staticmethod
    def _dynamic_ema(current_smoothed: Optional[float], new_value: float,
                     base_alpha: float, sensitivity: float = 20.0, max_alpha: float = 0.85) -> float:
        """
        Dynamic Exponential Moving Average.
        Increases the alpha (responsiveness) linearly based on how fast the value is changing.
        This drops lag to zero during fast shoulder checks, fixing duration boundary errors.
        """
        if current_smoothed is None:
            return new_value
        
        diff = abs(new_value - current_smoothed)
        # Boost alpha during rapid head movements
        boost = min(diff / sensitivity, 1.0) * (max_alpha - base_alpha)
        dynamic_alpha = base_alpha + boost
        
        return dynamic_alpha * new_value + (1.0 - dynamic_alpha) * current_smoothed

    def update_pose(self, yaw: float, pitch: float, roll: float, fps: float):
        # Use Dynamic EMA for head pose to kill phase lag
        self.smoothed_yaw = self._dynamic_ema(self.smoothed_yaw, yaw, self.pose_alpha)
        self.smoothed_pitch = self._dynamic_ema(self.smoothed_pitch, pitch, self.pose_alpha)
        self.smoothed_roll = self._dynamic_ema(self.smoothed_roll, roll, self.pose_alpha)

        self.delta_yaw, self.delta_pitch, self.delta_roll = 0.0, 0.0, 0.0

        if len(self.yaw_history) >= self.delta_interval:
            self.delta_yaw = yaw - self.yaw_history[-self.delta_interval]
            self.delta_pitch = pitch - self.pitch_history[-self.delta_interval]
            self.delta_roll = roll - self.roll_history[-self.delta_interval]

        self.head_ayaw, self.head_apitch, self.head_aroll = 0.0, 0.0, 0.0

        if self.prev_yaw is not None and self.smoothed_yaw is not None:
            if self.prev_yaw is not None and abs(self.smoothed_yaw - self.prev_yaw) > 30.0:
                self.head_ayaw = 0.0
                self.head_apitch = 0.0
                self.head_aroll = 0.0
            else:
                scale = fps / self.delta_interval
                self.head_ayaw = (self.smoothed_yaw - self.prev_yaw) * scale
                self.head_apitch = (self.smoothed_pitch - self.prev_pitch) * scale
                self.head_aroll = (self.smoothed_roll - self.prev_roll) * scale

                self.smoothed_head_ayaw = self._ema(self.smoothed_head_ayaw, self.head_ayaw, self.pose_alpha)
                self.smoothed_head_apitch = self._ema(self.smoothed_head_apitch, self.head_apitch, self.pose_alpha)
                self.smoothed_head_aroll = self._ema(self.smoothed_head_aroll, self.head_aroll, self.pose_alpha)

        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        self.roll_history.append(roll)

    def update_confidence(self, confidence: float):
        self.current_confidence = confidence
        self.confidence_history.append(confidence)
        if self.confidence_history:
            self.smoothed_confidence = sum(self.confidence_history) / len(self.confidence_history)
        else:
            self.smoothed_confidence = confidence

    def commit_prev_values(self):
        self.prev_yaw = self.smoothed_yaw
        self.prev_pitch = self.smoothed_pitch
        self.prev_roll = self.smoothed_roll

    def clear_on_no_detection(self):
        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_roll = None
        self.current_confidence = 0.0
        self.smoothed_confidence = 0.0
        self.head_ayaw = 0.0
        self.head_apitch = 0.0
        self.head_aroll = 0.0

    def get_collected_row(self, timestamp: float) -> Optional[list]:
        # Only require Yaw and Pitch to save a row
        if (self.smoothed_yaw is not None and self.smoothed_pitch is not None):
            return[
                round(timestamp, 3),
                round(self.smoothed_yaw, 2),
                round(self.smoothed_pitch, 2),
                round(self.smoothed_confidence, 3)
            ]
        return None