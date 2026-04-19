import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
# CONFIGURABLE THRESHOLDS
# All tuneable values live here — never buried in logic below.
# =============================================================================
THRESHOLDS = {
    # ── Drowsiness (EAR) ──────────────────────────────────────────────────────
    "EAR_THRESH":    0.25,   # EAR below this → eyes considered closed
    "EAR_WAIT_TIME": 2.0,    # Seconds eyes must stay closed before alarm

    # ── Yawning (MAR) ─────────────────────────────────────────────────────────
    "MAR_THRESH":    0.60,   # MAR above this → mouth wide open
    "MAR_WAIT_TIME": 1.5,    # Seconds MAR must stay high to count as yawn

    # ── Head turn (face asymmetry via nose-to-face-edge distances) ────────────
    # asymmetry = |left_dist - right_dist| / (left_dist + right_dist)
    # 0 = symmetric (frontal); grows as head rotates
    "HEAD_TURN_THRESH":    0.25,  # Asymmetry above this → head turned
    "HEAD_TURN_WAIT_TIME": 7.0,   # Seconds before triggering not-focused alert

    # ── Face missing ──────────────────────────────────────────────────────────
    "FACE_MISSING_WAIT_TIME": 5.0,  # Seconds without a face → user absent

    # ── Escalating alarm levels (seconds of CONTINUOUS alert active) ──────────
    "ALARM_LVL1": 5.0,   # After 5s  → 1 beep
    "ALARM_LVL2": 10.0,  # After 10s → 3 beeps
    "ALARM_LVL3": 15.0,  # After 15s → 6 rapid beeps

    # Per-level cooldown (seconds between consecutive bursts of that level)
    "ALARM_COOLDOWN_LVL1": 6.0,
    "ALARM_COOLDOWN_LVL2": 4.0,
    "ALARM_COOLDOWN_LVL3": 1.5,
}

# =============================================================================
# LANDMARK INDEX CONSTANTS  (MediaPipe FaceMesh 478-point model)
# =============================================================================

LEFT_EYE_IDX   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
MOUTH_IDX = {
    "upper": [82,  13, 312],
    "lower": [87,  14, 317],
    "left":  78,
    "right": 308,
}

# Head-geometry landmarks
NOSE_TIP_IDX        = 1
LEFT_FACE_EDGE_IDX  = 234
RIGHT_FACE_EDGE_IDX = 454
CHIN_IDX            = 152
FOREHEAD_IDX        = 10

# =============================================================================
# LOW-LEVEL HELPERS
# =============================================================================

def _lm_to_px(lm, w, h):
    """Normalised MediaPipe landmark → integer pixel (x, y)."""
    return (int(lm.x * w), int(lm.y * h))


def _lm_to_np(lm, w, h):
    """Normalised MediaPipe landmark → float numpy [x, y]."""
    return np.array([lm.x * w, lm.y * h], dtype=float)


def euclidean(p1, p2):
    return float(np.linalg.norm(
        np.array(p1, dtype=float) - np.array(p2, dtype=float)))


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

# ── EAR ───────────────────────────────────────────────────────────────────────

def compute_ear(landmarks, eye_idxs, w, h):
    """
    Eye Aspect Ratio for one eye.
    EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    Returns (ear_value, pixel_coords_list).
    """
    try:
        pts = [_lm_to_px(landmarks[i], w, h) for i in eye_idxs]
        ear = (euclidean(pts[1], pts[5]) + euclidean(pts[2], pts[4])) \
              / (2.0 * euclidean(pts[0], pts[3]))
        return ear, pts
    except Exception:
        return 0.0, None


# ── MAR ───────────────────────────────────────────────────────────────────────

def compute_mar(landmarks, mouth_idx, w, h):
    """
    Mouth Aspect Ratio.
    MAR = mean_vertical_lip_distance / horizontal_mouth_width
    High MAR + sustained duration = yawn.
    """
    try:
        upper = [_lm_to_px(landmarks[i], w, h) for i in mouth_idx["upper"]]
        lower = [_lm_to_px(landmarks[i], w, h) for i in mouth_idx["lower"]]
        left  = _lm_to_px(landmarks[mouth_idx["left"]],  w, h)
        right = _lm_to_px(landmarks[mouth_idx["right"]], w, h)
        vertical   = float(np.mean([euclidean(u, l)
                                    for u, l in zip(upper, lower)]))
        horizontal = euclidean(left, right)
        return vertical / horizontal if horizontal > 0 else 0.0
    except Exception:
        return 0.0


# ── HEAD TURN ─────────────────────────────────────────────────────────────────

def detect_head_turn(landmarks, w, h, threshold):
    """
    Detect horizontal head rotation (yaw) via face-edge asymmetry.

    Logic
    -----
    Measure the horizontal distance (x-axis) from the nose tip to the
    left and right cheek/jaw edges of the face mesh:

        left_dist  = |nose_x - left_edge_x|
        right_dist = |nose_x - right_edge_x|
        asymmetry  = |left_dist - right_dist| / (left_dist + right_dist)

    Frontal face  → asymmetry ≈ 0
    Head rotated  → asymmetry grows toward 1

    Returns:
        is_turned (bool)
        asymmetry (float)  -- displayed on screen for tuning
    """
    try:
        nose       = _lm_to_np(landmarks[NOSE_TIP_IDX],        w, h)
        left_edge  = _lm_to_np(landmarks[LEFT_FACE_EDGE_IDX],  w, h)
        right_edge = _lm_to_np(landmarks[RIGHT_FACE_EDGE_IDX], w, h)

        left_dist  = abs(nose[0] - left_edge[0])
        right_dist = abs(nose[0] - right_edge[0])
        total      = left_dist + right_dist

        if total < 1:
            return False, 0.0

        asymmetry = abs(left_dist - right_dist) / total
        return asymmetry > threshold, round(asymmetry, 3)
    except Exception:
        return False, 0.0


# =============================================================================
# AUDIO ALERT
# =============================================================================

def _beep_once(freq=1000, duration=300):
    """Single beep. Uses winsound on Windows; terminal bell elsewhere."""
    try:
        import winsound
        winsound.Beep(freq, duration)
    except Exception:
        print("\a", end="", flush=True)


def play_alert(level: int):
    """
    Level 1 → 1 beep  900 Hz  400 ms  (mild warning)
    Level 2 → 3 beeps 1000 Hz 300 ms  (moderate)
    Level 3 → 6 beeps 1000 Hz 100 ms  (urgent)
    """
    if level == 1:
        _beep_once(900, 400)
    elif level == 2:
        for _ in range(3):
            _beep_once(1000, 300)
            time.sleep(0.08)
    else:
        for _ in range(6):
            _beep_once(1000, 100)
            time.sleep(0.001)


# =============================================================================
# MAIN HANDLER
# =============================================================================

class VideoFrameHandler:
    """
    Processes webcam frames and returns an annotated frame + alarm flag.

    Detection pipeline per frame
    ----------------------------
    1. EAR          – eye closure duration
    2. MAR          – yawn detection
    3. Head turn    – face-edge asymmetry
    4. Face missing – no landmarks for N seconds
    """

    def __init__(self, model_path="face_landmarker.task"):
        # MediaPipe FaceLandmarker (Tasks API, IMAGE mode)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # ── State trackers ───────────────────────────────────────────────────
        now = time.perf_counter()

        self.drowsy_state = {
            "start_time":  now,
            "DROWSY_TIME": 0.0,
            "color":       (0, 255, 0),
            "play_alarm":  False,
        }
        self.yawn_state = {
            "start_time": now, "yawn_time": 0.0, "detected": False,
        }
        self.head_turn_state = {
            "start": None, "time": 0.0, "alert": False,
        }
        self.face_missing_state = {
            "start": None, "time": 0.0, "alert": False,
        }

        # Escalating alarm state
        self.alert_active_since = None
        self.last_alert_time    = {1: 0.0, 2: 0.0, 3: 0.0}

    # =========================================================================
    # STATE-UPDATE METHODS
    # =========================================================================

    def _update_drowsy(self, ear, t):
        if ear < t["EAR_THRESH"]:
            end = time.perf_counter()
            self.drowsy_state["DROWSY_TIME"] += end - self.drowsy_state["start_time"]
            self.drowsy_state["start_time"]   = end
            self.drowsy_state["color"]        = (0, 0, 255)
            if self.drowsy_state["DROWSY_TIME"] >= t["EAR_WAIT_TIME"]:
                self.drowsy_state["play_alarm"] = True
        else:
            self.drowsy_state["start_time"]  = time.perf_counter()
            self.drowsy_state["DROWSY_TIME"] = 0.0
            self.drowsy_state["color"]       = (0, 255, 0)
            self.drowsy_state["play_alarm"]  = False

    def _update_yawn(self, mar, t):
        if mar > t["MAR_THRESH"]:
            end = time.perf_counter()
            self.yawn_state["yawn_time"] += end - self.yawn_state["start_time"]
            self.yawn_state["start_time"] = end
            if self.yawn_state["yawn_time"] >= t["MAR_WAIT_TIME"]:
                self.yawn_state["detected"] = True
        else:
            self.yawn_state["start_time"] = time.perf_counter()
            self.yawn_state["yawn_time"]  = 0.0
            self.yawn_state["detected"]   = False

    def _update_head_turn(self, is_turned, t):
        """
        Start timer on first turned-frame; reset when face is frontal again.
        Alert fires after HEAD_TURN_WAIT_TIME seconds of continuous turning.
        """
        now = time.time()
        if is_turned:
            if self.head_turn_state["start"] is None:
                self.head_turn_state["start"] = now
            self.head_turn_state["time"]  = now - self.head_turn_state["start"]
            self.head_turn_state["alert"] = (
                self.head_turn_state["time"] >= t["HEAD_TURN_WAIT_TIME"])
        else:
            self.head_turn_state["start"] = None
            self.head_turn_state["time"]  = 0.0
            self.head_turn_state["alert"] = False

    def _update_face_missing(self, face_present, t):
        """
        When face disappears start a timer.
        Timer resets the instant a face is detected again.
        Alert fires after FACE_MISSING_WAIT_TIME seconds.
        """
        now = time.time()
        if not face_present:
            if self.face_missing_state["start"] is None:
                self.face_missing_state["start"] = now
            self.face_missing_state["time"]  = (
                now - self.face_missing_state["start"])
            self.face_missing_state["alert"] = (
                self.face_missing_state["time"] >= t["FACE_MISSING_WAIT_TIME"])
        else:
            # Face found → immediately clear missing-state
            self.face_missing_state["start"] = None
            self.face_missing_state["time"]  = 0.0
            self.face_missing_state["alert"] = False

    # =========================================================================
    # ESCALATING AUDIO ALERT
    # =========================================================================

    def _maybe_alert(self, any_alert: bool, t: dict):
        """
        Fire escalating beeps on a daemon thread when conditions are met.

        Continuous-alert timer starts at first trigger and resets when
        all conditions clear.  Level (and frequency of beeps) escalates
        with elapsed time; each level has its own cooldown.
        """
        now = time.time()

        if not any_alert:
            self.alert_active_since = None
            return

        if self.alert_active_since is None:
            self.alert_active_since = now

        elapsed = now - self.alert_active_since

        if elapsed >= t["ALARM_LVL3"]:
            level, cooldown = 3, t["ALARM_COOLDOWN_LVL3"]
        elif elapsed >= t["ALARM_LVL2"]:
            level, cooldown = 2, t["ALARM_COOLDOWN_LVL2"]
        elif elapsed >= t["ALARM_LVL1"]:
            level, cooldown = 1, t["ALARM_COOLDOWN_LVL1"]
        else:
            return  # still in the silent grace window

        if now - self.last_alert_time[level] >= cooldown:
            self.last_alert_time[level] = now
            threading.Thread(
                target=play_alert, args=(level,), daemon=True).start()

    # =========================================================================
    # DRAWING
    # =========================================================================

    @staticmethod
    def _put(frame, text, pos, color=(255, 255, 255), scale=0.60, thick=2):
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _draw_overlay(self, frame, ear, mar, head_asym, t):
        fh, fw = frame.shape[:2]
        ec = self.drowsy_state["color"]

        def alert_col(flag):
            return (0, 0, 255) if flag else (0, 200, 255)

        # ── Raw metrics (top-left) ────────────────────────────────────────────
        self._put(frame, f"EAR: {ear:.2f}",  (10, 28),  ec)
        self._put(frame, f"MAR: {mar:.2f}",  (10, 52),  (200, 200, 0))
        self._put(frame, f"HeadTurn asym:  {head_asym:.3f}",
                  (10, 74), (180, 180, 180), scale=0.50)

        # ── Timers ────────────────────────────────────────────────────────────
        self._put(frame,
                  f"Drowsy:        {self.drowsy_state['DROWSY_TIME']:.1f}s",
                  (10, 100), ec)
        self._put(frame,
                  f"Head Turn:     {self.head_turn_state['time']:.1f}s",
                  (10, 124), alert_col(self.head_turn_state["alert"]))
        self._put(frame,
                  f"Face Missing:  {self.face_missing_state['time']:.1f}s",
                  (10, 148), alert_col(self.face_missing_state["alert"]))

        # ── Active condition labels (bottom, stacked upward) ──────────────────
        y = fh - 120
        lh = 26  # line height

        if self.drowsy_state["play_alarm"]:
            self._put(frame, "WAKE UP!", (10, y), (0, 0, 255), scale=0.80, thick=2)
            y += lh
        if self.yawn_state["detected"]:
            self._put(frame, "YAWNING DETECTED", (10, y), (0, 100, 255), scale=0.70, thick=2)
            y += lh
        if self.head_turn_state["alert"]:
            self._put(frame, "HEAD TURNED - NOT FOCUSED", (10, y), (0, 70, 255), scale=0.70, thick=2)
            y += lh
        if self.face_missing_state["alert"]:
            self._put(frame, "USER ABSENT", (10, y), (0, 0, 200), scale=0.70, thick=2)
            y += lh

        # ── Central alert banner with escalation level ────────────────────────
        if self._any_alert() and self.alert_active_since is not None:
            elapsed = time.time() - self.alert_active_since
            if elapsed >= t["ALARM_LVL3"]:
                txt, col = "!! ALERT - LEVEL 3 !!", (0, 0, 255)
            elif elapsed >= t["ALARM_LVL2"]:
                txt, col = "!  ALERT - LEVEL 2",    (0, 60, 255)
            elif elapsed >= t["ALARM_LVL1"]:
                txt, col = "   ALERT - LEVEL 1",    (0, 140, 255)
            else:
                txt, col = "   WARNING ...",         (0, 210, 210)
            self._put(frame, txt, (fw // 2 - 140, fh - 20),
                      col, scale=0.95, thick=3)

    # =========================================================================
    # CONVENIENCE
    # =========================================================================

    def _any_alert(self):
        return (self.drowsy_state["play_alarm"]  or
                self.yawn_state["detected"]      or
                self.head_turn_state["alert"]    or
                self.face_missing_state["alert"])

    def _reset_on_no_face(self, t):
        """
        Face is absent: reset all face-geometry detectors but let the
        face-missing timer keep running (handled by _update_face_missing).
        """
        self.drowsy_state["DROWSY_TIME"] = 0.0
        self.drowsy_state["play_alarm"]  = False
        self.drowsy_state["color"]       = (0, 255, 0)

        self.yawn_state.update(yawn_time=0.0, detected=False)
        self.head_turn_state.update(start=None, time=0.0, alert=False)

        self._update_face_missing(False, t)  # tick the missing timer

    # =========================================================================
    # MAIN PROCESS  (called once per frame)
    # =========================================================================

    def process(self, frame: np.ndarray, thresholds: dict):
        """
        Analyse one BGR webcam frame.

        Returns
        -------
        annotated_frame : np.ndarray  – frame with all overlays drawn
        alarm_active    : bool        – True if any condition is firing
        """
        fh, fw = frame.shape[:2]
        t = thresholds  # short alias

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        # Defaults for overlay (shown even when no face present)
        ear = mar = head_asym = 0.0

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            # Face is present → clear missing timer
            self._update_face_missing(True, t)

            # ── 1. Eye closure (EAR) ──────────────────────────────────────────
            l_ear, l_pts = compute_ear(lms, LEFT_EYE_IDX,  fw, fh)
            r_ear, r_pts = compute_ear(lms, RIGHT_EYE_IDX, fw, fh)
            ear = (l_ear + r_ear) / 2.0
            self._update_drowsy(ear, t)

            # ── 2. Yawning (MAR) ──────────────────────────────────────────────
            mar = compute_mar(lms, MOUTH_IDX, fw, fh)
            self._update_yawn(mar, t)

            # ── 3. Head turn ──────────────────────────────────────────────────
            is_turned, head_asym = detect_head_turn(
                lms, fw, fh, t["HEAD_TURN_THRESH"])
            self._update_head_turn(is_turned, t)

        else:
            # ── 6. Face missing ───────────────────────────────────────────────
            self._reset_on_no_face(t)

        # ── Escalating audio alert ────────────────────────────────────────────
        self._maybe_alert(self._any_alert(), t)

        # ── Visual overlay ────────────────────────────────────────────────────
        self._draw_overlay(frame, ear, mar, head_asym, t)

        return frame, self._any_alert()