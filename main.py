import cv2
from crowd_detector import CrowdDensityDetector

def main():
    cap = cv2.VideoCapture(0)  # Change to video file if needed

    if not cap.isOpened():
        print("Camera not working. Congrats.")
        return

    detector = CrowdDensityDetector(model_size='n')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame dead.")
            break

        if detector.boundary_line is None:
            detector.set_boundary_line(frame.shape[0])

        detections = detector.detect_people(frame)
        output_frame = detector.draw_visualization(frame, detections)

        cv2.imshow("AI Crowd Monitoring", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
