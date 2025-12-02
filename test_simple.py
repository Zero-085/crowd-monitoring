"""
Simple test script to verify YOLO is working
Run this FIRST before anything else!
"""
import cv2
from ultralytics import YOLO

print("=" * 60)
print("TESTING YOLO SETUP")
print("=" * 60)

print("\n1️⃣ Loading YOLO model...")
try:
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("\n2️⃣ Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    print("💡 Try using a video file instead")
    print("   Edit this file and change: cap = cv2.VideoCapture('your_video.mp4')")
    exit(1)

print("✅ Webcam opened!")
print("\n3️⃣ Starting detection (Press 'q' to quit)...\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Run detection
    results = model(frame, classes=[0], verbose=False)  # 0 = person
    
    # Get count
    person_count = len(results[0].boxes)
    
    # Draw results
    annotated = results[0].plot()
    
    # Add count
    cv2.putText(annotated, f'People: {person_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.putText(annotated, f'Frame: {frame_count}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Show
    cv2.imshow('YOLO Test - Press Q to quit', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("✅ TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nIf you saw the video with people detected, you're ready!")
print("Next step: Run crowd_detector.py or streamlit run app.py")