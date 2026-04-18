import cv2
from drowsy_detection2 import VideoFrameHandler, THRESHOLDS

def main():
    # ── Initialise detector ────────────────────────────────────────────────
    # Make sure 'face_landmarker.task' is in the same folder!
    handler = VideoFrameHandler(model_path="face_landmarker.task")

    # All thresholds live in drowsy_detection2.THRESHOLDS.
    # Override any value here before the loop if you like, e.g.:
    #   THRESHOLDS["GAZE_OFF_LIMIT"] = 5.0
    thresholds = THRESHOLDS

    # ── Open webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    print("Drowsiness / Gaze / Yawn Detection started.")
    print("Press  q  to quit.\n")
    print("Detects:")
    print("  • Eye closure  (EAR < {EAR_THRESH} for {EAR_WAIT_TIME}s)".format(**thresholds))
    print("  • Gaze away    (> {GAZE_OFF_LIMIT}s off-screen)".format(**thresholds))
    print("  • Yawning      (MAR > {MAR_THRESH} for {MAR_WAIT_TIME}s)".format(**thresholds))
    print("  • Audio alert  (Lv1 cooldown {ALARM_COOLDOWN_LVL1}s, "
          "Lv2 {ALARM_COOLDOWN_LVL2}s, Lv3 {ALARM_COOLDOWN_LVL3}s)\n".format(**thresholds))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip for selfie-view
        frame = cv2.flip(frame, 1)

        # Process frame
        processed_frame, alarm_active = handler.process(frame, thresholds)

        # Show result
        cv2.imshow("Drowsiness Detector", processed_frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()