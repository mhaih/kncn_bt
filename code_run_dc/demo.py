import cv2
from drowsy_detection2 import VideoFrameHandler

def main():
    # Initialize the detector
    # Ensure 'face_landmarker.task' is in the same folder!
    handler = VideoFrameHandler(model_path="face_landmarker.task")

    # Configuration thresholds
    thresholds = {
        "EAR_THRESH": 0.25, # Adjust based on your eye shape
        "WAIT_TIME": 2.0    # Seconds eyes must be closed to trigger alarm
    }

    # Start webcam
    cap = cv2.VideoCapture(0)

    print("Drowsiness Detection Started. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Process the frame using your logic from drowsiness.py
        processed_frame, alarm_active = handler.process(frame, thresholds)

        # Logic for a sound alarm could go here
        if alarm_active:
            # You could use winsound or pygame here to play a beep
            pass

        # Display the output
        cv2.imshow('Drowsiness Detector', processed_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()