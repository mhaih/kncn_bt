import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- HELPER FUNCTIONS (Logic remains the same) ---

def distance(point_1, point_2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """Calculate Eye Aspect Ratio for one eye using new landmark format"""
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            # Convert normalized to pixel coordinates
            coord = (int(lm.x * frame_width), int(lm.y * frame_height))
            coords_points.append(coord)

        # EAR Formula: (vertical distances) / (2 * horizontal distance)
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
        return ear, coords_points
    except Exception:
        return 0.0, None

class VideoFrameHandler:
    def __init__(self, model_path="face_landmarker.task"):
        # Eye indices (standard MediaPipe FaceMesh indices)
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Setup New MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE, # Streamlit-webrtc processes frame-by-frame
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # State tracking
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": (0, 255, 0), # Green
            "play_alarm": False,
        }

    def process(self, frame: np.array, thresholds: dict):
        frame_h, frame_w, _ = frame.shape
        
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Detection
        detection_result = self.detector.detect(mp_image)

        if detection_result.face_landmarks:
            # Get landmarks for the first face detected
            landmarks = detection_result.face_landmarks[0]
            
            # Calculate EAR
            left_ear, left_coords = get_ear(landmarks, self.eye_idxs["left"], frame_w, frame_h)
            right_ear, right_coords = get_ear(landmarks, self.eye_idxs["right"], frame_w, frame_h)
            EAR = (left_ear + right_ear) / 2.0

            # Drowsiness Logic
            if EAR < thresholds["EAR_THRESH"]:
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = (0, 0, 255) # Red

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    cv2.putText(frame, "WAKE UP!", (10, frame_h - 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = (0, 255, 0) # Green
                self.state_tracker["play_alarm"] = False

            # Draw landmarks
            for eye in [left_coords, right_coords]:
                if eye:
                    for pt in eye:
                        cv2.circle(frame, pt, 2, self.state_tracker["COLOR"], -1)

            # Overlay Text
            cv2.putText(frame, f"EAR: {round(EAR, 2)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.state_tracker["COLOR"], 2)
            cv2.putText(frame, f"Drowsy: {round(self.state_tracker['DROWSY_TIME'], 2)}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.state_tracker["COLOR"], 2)

        else:
            # Reset state if no face is seen
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["play_alarm"] = False

        return frame, self.state_tracker["play_alarm"]