"""
AI-Powered Crowd Density Monitoring System
Advanced Crowd Density Detection Module - crowd_detector.py

Features:
- Density heatmap visualization
- Crowding intensity analysis
- Multi-zone density tracking
- Real-time alarm system
- Horizontal / Vertical boundary + auto-detect (Hough)
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import math
import platform

# cross-platform alarm support
try:
    if platform.system().lower().startswith("win"):
        import winsound
    else:
        winsound = None
except Exception:
    winsound = None

class CrowdDensityDetector:
    def __init__(self, model_size='n'):
        """
        Initialize advanced crowd density detector
        """
        print("🔄 Loading YOLO AI model...")
        # NOTE: user must have yolov8{size}.pt available or change to a valid model path.
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
        except Exception as e:
            print("⚠️ Warning: YOLO model load failed. Make sure the model file exists.")
            raise e
        print("✅ Model loaded successfully!")

        # Density thresholds (people per square meter equivalent)
        self.density_low = 0.3      # Low density
        self.density_medium = 0.6   # Medium density
        self.density_high = 0.9     # High density (danger)
        
        # Zone grid for density analysis (divide frame into grid)
        self.grid_rows = 6
        self.grid_cols = 8
        self.density_map = None
        
        # Virtual boundary
        # boundary_line: None or {'orientation':'horizontal'|'vertical', 'pos': int}
        self.boundary_line = None
        
        # Alarm system
        self.alarm_active = False
        self.last_alarm_time = 0
        self.alarm_cooldown = 3  # seconds between alarms
        
        # Statistics
        self.stats = {
            'current_density': 0.0,
            'max_density': 0.0,
            'avg_density': 0.0,
            'density_history': [],
            'alerts_triggered': 0,
            'critical_zones': 0,
            'violations': 0
        }
        
        # Alarm sound thread
        self.alarm_thread = None

    def set_boundary_line(self, frame_width, frame_height, position=0.7, orientation='horizontal'):
        """
        Set virtual boundary line.
        orientation: 'horizontal' -> line across width at y = frame_height * position
                     'vertical'   -> line across height at x = frame_width * position
        Stores as a dict: {'orientation':..., 'pos': int}
        """
        orientation = orientation.lower()
        if orientation == 'vertical':
            x = int(frame_width * position)
            self.boundary_line = {'orientation': 'vertical', 'pos': x}
        else:
            y = int(frame_height * position)
            self.boundary_line = {'orientation': 'horizontal', 'pos': y}

    def auto_detect_boundary(self, frame, orientation='horizontal', debug=False):
        """
        Auto-detect a dominant long line using Canny + Hough.
        orientation preference: 'horizontal' or 'vertical'
        Sets self.boundary_line if a suitable line is found; otherwise returns False.
        """
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        min_len = min(frame.shape[:2]) // 3  # require reasonably long lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                                minLineLength=min_len, maxLineGap=40)
        if lines is None:
            return False

        best_line = None
        best_len = 0
        target = orientation.lower()
        angle_tolerance = 25  # degrees tolerance

        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            angle = abs(math.degrees(math.atan2(dy, dx)))  # 0 deg horizontal, 90 deg vertical

            if target == 'horizontal':
                ang_diff = min(abs(angle - 0), abs(angle - 180))
            else:
                ang_diff = abs(angle - 90)

            if ang_diff <= angle_tolerance and length > best_len:
                best_len = length
                best_line = (x1, y1, x2, y2, angle)

        if best_line is None:
            return False

        x1, y1, x2, y2, _ = best_line
        if target == 'horizontal':
            y_mean = int((y1 + y2) / 2)
            self.boundary_line = {'orientation': 'horizontal', 'pos': y_mean}
        else:
            x_mean = int((x1 + x2) / 2)
            self.boundary_line = {'orientation': 'vertical', 'pos': x_mean}

        if debug:
            print(f"[auto_detect_boundary] set {self.boundary_line}")
        return True

    def play_alarm_sound(self, alarm_type='overcrowding'):
        """
        Play alarm sound (runs in separate thread)
        alarm_type: 'overcrowding', 'violation', 'critical'
        """
        try:
            if winsound:
                # Windows winsound
                if alarm_type == 'overcrowding':
                    for _ in range(3):
                        winsound.Beep(1000, 200)
                        time.sleep(0.1)
                elif alarm_type == 'violation':
                    for _ in range(2):
                        winsound.Beep(1500, 400)
                        time.sleep(0.15)
                elif alarm_type == 'critical':
                    for _ in range(5):
                        winsound.Beep(2000, 150)
                        time.sleep(0.05)
            else:
                # fallback: print (non-blocking) — sound on unix could be added if desired
                print(f"\nALARM: {alarm_type.upper()}")
        except Exception as e:
            print(f"\nALARM ERROR: {e}")

    def trigger_alarm(self, alarm_type='overcrowding'):
        """Trigger alarm with cooldown"""
        current_time = time.time()
        if current_time - self.last_alarm_time > self.alarm_cooldown:
            self.last_alarm_time = current_time
            # Play sound in separate thread (non-blocking)
            if self.alarm_thread is None or not self.alarm_thread.is_alive():
                self.alarm_thread = threading.Thread(target=self.play_alarm_sound, args=(alarm_type,))
                self.alarm_thread.daemon = True
                self.alarm_thread.start()
            return True
        return False

    def detect_people(self, frame):
        """Detect all people in frame and return list of detections dicts"""
        results = self.model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0]) if box.conf is not None else 0.0

            # Calculate center and size
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            area = width * height

            detections.append({
                'box': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'area': area,
                'confidence': confidence
            })

        return detections

    def calculate_density_map(self, frame_shape, detections):
        """
        Calculate crowd density across frame using grid-based analysis
        Returns: density map (heatmap values for each grid cell)
        """
        height, width = frame_shape[:2]
        
        # Initialize density grid
        density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Calculate cell dimensions
        cell_height = max(1, height // self.grid_rows)
        cell_width = max(1, width // self.grid_cols)
        
        # Map each detection to grid cells
        for det in detections:
            cx, cy = det['center']
            
            # Find grid cell
            grid_row = min(cy // cell_height, self.grid_rows - 1)
            grid_col = min(cx // cell_width, self.grid_cols - 1)
            
            # Increment density (weighted by person size)
            size_weight = min(det['area'] / (cell_height * cell_width), 1.0)
            density_grid[grid_row, grid_col] += size_weight
        
        # Normalize density (0 to 1 scale)
        if density_grid.max() > 0:
            density_grid = density_grid / density_grid.max()
        
        return density_grid

    def create_heatmap_overlay(self, frame, density_grid):
        """
        Create visual heatmap overlay on frame
        """
        height, width = frame.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        cell_height = max(1, height // self.grid_rows)
        cell_width = max(1, width // self.grid_cols)
        
        # Fill heatmap based on density grid
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1 = i * cell_height
                y2 = min((i + 1) * cell_height, height)
                x1 = j * cell_width
                x2 = min((j + 1) * cell_width, width)
                
                heatmap[y1:y2, x1:x2] = density_grid[i, j]
        
        # Apply Gaussian blur for smooth heatmap
        k = 51 if min(height, width) >= 51 else 11
        heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
        
        # Convert to color heatmap (JET is fine)
        heatmap_colored = cv2.applyColorMap((np.clip(heatmap,0,1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        return overlay, heatmap

    def analyze_crowd_density(self, density_grid, people_count, frame_area):
        """
        Analyze overall crowd density and identify critical zones
        Returns: density_level, critical_zone_count, density_score
        """
        # Calculate average density
        avg_density = np.mean(density_grid) if density_grid.size > 0 else 0.0
        
        # Count critical zones (high density cells)
        critical_zones = int(np.sum(density_grid > self.density_high)) if density_grid.size > 0 else 0
        
        # Calculate density score (0-100)
        density_score = (
            avg_density * 50 +  # Base density
            (critical_zones / (self.grid_rows * self.grid_cols)) * 30 +  # Critical zone ratio
            min(people_count / 20, 1.0) * 20  # People count factor
        )
        
        # Classify density level
        if density_score < 30:
            level = 'LOW'
            color = (0, 255, 0)  # Green
        elif density_score < 60:
            level = 'MODERATE'
            color = (0, 255, 255)  # Yellow
        elif density_score < 80:
            level = 'HIGH'
            color = (0, 165, 255)  # Orange
        else:
            level = 'CRITICAL'
            color = (0, 0, 255)  # Red
        
        return level, color, critical_zones, density_score

    def check_boundary_violations(self, detections):
        """Check boundary violations for both horizontal and vertical boundaries."""
        if not self.boundary_line:
            return False, []
        orient = self.boundary_line.get('orientation', 'horizontal')
        pos = self.boundary_line.get('pos', None)
        if pos is None:
            return False, []
        violators = []
        for det in detections:
            cx, cy = det['center']
            if orient == 'vertical':
                # default: crossing to right side (cx > pos) is violation
                if cx > pos:
                    violators.append(det)
            else:
                # default: crossing below line (cy > pos) is violation
                if cy > pos:
                    violators.append(det)
        return len(violators) > 0, violators

    def draw_visualization(self, frame, detections, show_heatmap=True, show_boundary=True):
        """
        Draw comprehensive visualization with density heatmap
        """
        people_count = len(detections)
        
        # Calculate density map
        density_grid = self.calculate_density_map(frame.shape, detections)
        
        # Create heatmap overlay if enabled
        if show_heatmap:
            vis_frame, heatmap = self.create_heatmap_overlay(frame, density_grid)
        else:
            vis_frame = frame.copy()
            heatmap = None
        
        # Analyze crowd density
        frame_area = frame.shape[0] * frame.shape[1]
        density_level, density_color, critical_zones, density_score = \
            self.analyze_crowd_density(density_grid, people_count, frame_area)
        
        # Update statistics
        self.stats['current_density'] = density_score
        self.stats['max_density'] = max(self.stats['max_density'], density_score)
        self.stats['critical_zones'] = critical_zones
        self.stats['density_history'].append(density_score)
        
        # Keep only last 100 records
        if len(self.stats['density_history']) > 100:
            self.stats['density_history'].pop(0)
        
        # Calculate average
        if self.stats['density_history']:
            self.stats['avg_density'] = float(np.mean(self.stats['density_history']))
        
        # Check violations
        has_violation, violators = self.check_boundary_violations(detections)
        if has_violation:
            self.stats['violations'] += 1
        
        # Draw boundary line (horizontal or vertical)
        if show_boundary and self.boundary_line:
            orient = self.boundary_line.get('orientation', 'horizontal')
            pos = self.boundary_line.get('pos')
            if orient == 'vertical':
                cv2.line(vis_frame, (pos, 0), (pos, vis_frame.shape[0]), (0, 0, 255), 4)
                cv2.putText(vis_frame, 'RESTRICTED ZONE', (max(10, pos+10), 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.line(vis_frame, (0, pos), (vis_frame.shape[1], pos), (0, 0, 255), 4)
                cv2.putText(vis_frame, 'RESTRICTED ZONE', (10, max(20, pos-15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Draw detections with special marking for violators
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = det['center']
            
            is_violator = det in violators
            box_color = (0, 0, 255) if is_violator else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw center point
            cv2.circle(vis_frame, (cx, cy), 4, box_color, -1)
            
            if is_violator:
                cv2.putText(vis_frame, 'VIOLATION', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw info panel with density information
        panel_h = 250
        panel_w = 500
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.5, overlay, 0.5, 0)
        
        # Display information
        y_pos = 35
        
        # People count
        cv2.putText(vis_frame, f'People Count: {people_count}', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_pos += 40
        # Density score with bar
        cv2.putText(vis_frame, f'Density Score: {density_score:.1f}/100', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Density bar
        bar_x = 15
        bar_y = y_pos + 10
        bar_width = 300
        bar_height = 20
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        filled_width = int((density_score / 100) * bar_width)
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                     density_color, -1)
        
        y_pos += 50
        # Density level
        cv2.putText(vis_frame, f'Density: {density_level}', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, density_color, 3)
        
        y_pos += 45
        # Critical zones
        critical_color = (0, 0, 255) if critical_zones > 0 else (0, 255, 0)
        cv2.putText(vis_frame, f'Critical Zones: {critical_zones}', 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, critical_color, 2)
        
        y_pos += 40
        # Violations
        if has_violation:
            cv2.putText(vis_frame, f'VIOLATIONS: {len(violators)}', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # Flash effect for violations
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], vis_frame.shape[0]),
                             (0, 0, 255), 15)
        else:
            cv2.putText(vis_frame, 'No Violations', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Trigger alarms based on conditions
        alarm_triggered = False
        alarm_type = None
        
        if has_violation:
            if self.trigger_alarm('violation'):
                alarm_triggered = True
                alarm_type = 'BOUNDARY VIOLATION'
                self.stats['alerts_triggered'] += 1
        
        elif density_level == 'CRITICAL':
            if self.trigger_alarm('critical'):
                alarm_triggered = True
                alarm_type = 'CRITICAL DENSITY'
                self.stats['alerts_triggered'] += 1
        
        elif density_level == 'HIGH' and critical_zones > 3:
            if self.trigger_alarm('overcrowding'):
                alarm_triggered = True
                alarm_type = 'HIGH DENSITY'
                self.stats['alerts_triggered'] += 1
        
        # Show alarm indicator
        if alarm_triggered or (time.time() - self.last_alarm_time < 1.0):
            alarm_panel_y = vis_frame.shape[0] - 100
            cv2.rectangle(vis_frame, (0, alarm_panel_y), 
                         (vis_frame.shape[1], vis_frame.shape[0]), 
                         (0, 0, 255), -1)
            cv2.putText(vis_frame, f'ALARM: {alarm_type if alarm_type else "ACTIVE"}', 
                       (max(10, vis_frame.shape[1]//2 - 250), alarm_panel_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return vis_frame, density_level, density_score, has_violation, alarm_triggered

    def process_video(self, video_source=0, save_output=False, show_heatmap=True):
        """
        Main processing loop with density analysis
        """
        print(f"\n🎥 Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open video source!")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Video size: {frame_width}x{frame_height} @ {fps} FPS")
        
        # Set boundary default, allow override later
        self.set_boundary_line(frame_width, frame_height, position=0.70, orientation='horizontal')
        
        # Setup video writer
        writer = None
        if save_output:
            output_file = f'density_output_{int(time.time())}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_file, fourcc, fps, 
                                    (frame_width, frame_height))
            print(f"💾 Saving output to: {output_file}")
        
        print("\n▶️  Processing started!")
        print("📌 Press 'q' to quit")
        print("📌 Press 's' to take screenshot")
        print("📌 Press 'h' to toggle heatmap\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect people
            detections = self.detect_people(frame)
            
            # Draw visualization with density analysis
            vis_frame, density_level, density_score, violation, alarm = \
                self.draw_visualization(frame, detections, 
                                       show_heatmap=show_heatmap,
                                       show_boundary=True)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f'FPS: {current_fps:.1f}', 
                       (frame_width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Crowd Density Monitoring - Press Q to quit', vis_frame)
            
            # Save if enabled
            if writer:
                writer.write(vis_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f'density_screenshot_{int(time.time())}.jpg'
                cv2.imwrite(screenshot_name, vis_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                print(f"🗺️  Heatmap: {'ON' if show_heatmap else 'OFF'}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Maximum density score: {self.stats['max_density']:.1f}/100")
        print(f"Average density score: {self.stats['avg_density']:.1f}/100")
        print(f"Total alarms triggered: {self.stats['alerts_triggered']}")
        print(f"Total violations: {self.stats['violations']}")
        print(f"Average FPS: {current_fps:.1f}")
        print("="*60 + "\n")

# Test mode
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED CROWD DENSITY MONITORING SYSTEM")
    print("="*60 + "\n")

    detector = CrowdDensityDetector(model_size='n')

    print("Select video source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        detector.process_video(video_source=0, save_output=False, show_heatmap=True)
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        detector.process_video(video_source=video_path, save_output=True, show_heatmap=True)
    else:
        print("Invalid choice. Using webcam...")
        detector.process_video(video_source=0, save_output=False, show_heatmap=True)
