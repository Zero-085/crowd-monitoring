"""
AI-Powered Crowd Density Monitoring System
Core Detection Module - crowd_detector.py

This handles all the AI detection logic
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 165)      # Speed (lower = slower)
engine.setProperty('volume', 1.0)    # Volume (0 to 1)
last_alert_time = 0
cooldown = 5   # seconds

class CrowdDetector:
    def __init__(self, model_size='n'):
        """
        Initialize the detector
        model_size: 'n' = nano (fastest), 's' = small, 'm' = medium
        """
        print("🔄 Loading YOLO AI model (this may take a minute first time)...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("✅ Model loaded successfully!")
        
        # Thresholds for zone classification
        self.green_threshold = 10    # Safe zone: 0-10 people
        self.yellow_threshold = 20   # Moderate: 11-20 people
        # Red zone: 21+ people (danger)
        
        # Virtual boundary line (for railway track detection)
        self.boundary_line = None
        
        # Statistics tracking
        self.stats = {
            'total_count': 0,
            'max_count': 0,
            'alerts_triggered': 0,
            'violations': 0
        }
    
    def set_boundary_line(self, frame_height, position=0.7):
        """
        Set virtual boundary line position
        position: 0.0 (top) to 1.0 (bottom)
        0.7 means 70% down from top
        """
        self.boundary_line = int(frame_height * position)
    
    def detect_people(self, frame):
        """
        Detect all people in the frame
        Returns list of detections with bounding boxes
        """
        # Run YOLO detection (class 0 = person)
        results = self.model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes
        
        detections = []
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Calculate center point (for boundary check)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            detections.append({
                'box': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': confidence
            })
        
        return detections
    
    def classify_zone(self, people_count):
        """
        Classify crowd density into zones
        Returns: zone name and color
        """
        if people_count <= self.green_threshold:
            return 'SAFE', (0, 255, 0)  # Green
        elif people_count <= self.yellow_threshold:
            return 'MODERATE', (0, 255, 255)  # Yellow
        else:
            return 'DANGER', (0, 0, 255)  # Red
    
    def check_boundary_violations(self, detections):
        """
        Check if anyone crossed the restricted boundary
        Returns: has_violation, list of violators
        """
        if self.boundary_line is None:
            return False, []
        
        violators = []
        for det in detections:
            cx, cy = det['center']
            # If person's center is below the line = violation
            if cy > self.boundary_line:
                violators.append(det)
        
        return len(violators) > 0, violators
    
    def draw_visualization(self, frame, detections, show_boundary=True):
        """
        Draw everything on the frame:
        - Bounding boxes around people
        - Zone status
        - Boundary line
        - Statistics
        - Alerts
        """
        vis_frame = frame.copy()
        people_count = len(detections)
        
        # Update statistics
        self.stats['total_count'] = people_count
        self.stats['max_count'] = max(self.stats['max_count'], people_count)
        
        # Get zone classification
        zone_status, zone_color = self.classify_zone(people_count)
        
        # Check for boundary violations
        has_violation, violators = self.check_boundary_violations(detections)
        if has_violation:
            self.stats['violations'] += 1
        
        # Draw boundary line if enabled
        if show_boundary and self.boundary_line:
            # Red dashed line
            line_y = self.boundary_line
            cv2.line(vis_frame, (0, line_y), (frame.shape[1], line_y), 
                    (0, 0, 255), 3)
            cv2.putText(vis_frame, 'RESTRICTED ZONE - DO NOT CROSS', 
                       (10, line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw each detected person
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = det['center']
            
            # Check if this person is violating boundary
            is_violator = det in violators
            
            # Color: Red for violators, Green for safe
            box_color = (0, 0, 255) if is_violator else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw center point
            cv2.circle(vis_frame, (cx, cy), 5, box_color, -1)
            
            # Label violators
            if is_violator:
                cv2.putText(vis_frame, 'VIOLATION!', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw information panel (top-left)
        panel_h = 180
        panel_w = 450
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)
        
        # Text information
        y_pos = 35
        cv2.putText(vis_frame, f'People Detected: {people_count}', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        y_pos += 40
        cv2.putText(vis_frame, f'Zone Status: {zone_status}', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, zone_color, 2)
        
        y_pos += 40
        if has_violation:
            cv2.putText(vis_frame, f'ALERT: {len(violators)} VIOLATION(S)!', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Trigger alert
            if people_count > 0:  # Only count real alerts
                self.stats['alerts_triggered'] += 1
        else:
            cv2.putText(vis_frame, 'No Violations', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        y_pos += 40
        cv2.putText(vis_frame, f'Max Count Today: {self.stats["max_count"]}', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return vis_frame, zone_status, has_violation
    
    def process_video(self, video_source=0, save_output=False):
        """
        Main processing loop
        video_source: 0 for webcam, or path to video file
        save_output: True to save output video
        """
        print(f"\n🎥 Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open video source!")
            print("💡 Try: Different camera index (1, 2) or check video file path")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"📐 Video size: {frame_width}x{frame_height} @ {fps} FPS")
        
        # Set up boundary line
        self.set_boundary_line(frame_height, position=0.70)
        
        # Setup video writer if saving
        writer = None
        if save_output:
            output_file = f'output_{int(time.time())}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_file, fourcc, fps, 
                                    (frame_width, frame_height))
            print(f"💾 Saving output to: {output_file}")
        
        print("\nProcessing started!")
        print("📌 Press 'q' to quit")
        print("📌 Press 's' to take screenshot\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame")
                break
            
            frame_count += 1
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Draw visualization
            vis_frame, zone, violation = self.draw_visualization(frame, detections)
            
            # Calculate and display FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f'FPS: {current_fps:.1f}', 
                       (frame_width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('🚨 Crowd Monitoring System - Press Q to quit', vis_frame)
            
            # Save if enabled
            if writer:
                writer.write(vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  Stopped by user")
                break
            elif key == ord('s'):
                screenshot_name = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(screenshot_name, vis_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Maximum people detected: {self.stats['max_count']}")
        print(f"Total alerts triggered: {self.stats['alerts_triggered']}")
        print(f"Average FPS: {current_fps:.1f}")
        print("="*50 + "\n")


# Standalone test mode
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚨 CROWD MONITORING SYSTEM - TEST MODE")
    print("="*60 + "\n")
    
    # Create detector
    detector = CrowdDetector(model_size='n')
    
    # Choose source
    print("Select video source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        detector.process_video(video_source=0, save_output=False)
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        detector.process_video(video_source=video_path, save_output=True)
    else:
        print("Invalid choice. Using webcam...")
        detector.process_video(video_source=0, save_output=False)