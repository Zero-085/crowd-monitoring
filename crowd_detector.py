"""
AI-Powered Crowd Density Monitoring System
Advanced Crowd Density Detection Module - crowd_detector.py

Features:
- Density heatmap visualization
- Crowding intensity analysis
- Multi-zone density tracking
- Real-time alarm system
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import threading
import winsound  # For Windows alarm (use 'os' for Mac/Linux)

class CrowdDensityDetector:
    def __init__(self, model_size='n', demo_mode=True):
        """
        Initialize advanced crowd density detector
        demo_mode: True for hackathon/small room demos, False for real deployment
        """
        print("🔄 Loading YOLO AI model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("✅ Model loaded successfully!")
        
        # Demo mode uses lower thresholds for small spaces
        self.demo_mode = demo_mode
        
        if demo_mode:
            print("📌 Demo Mode: Optimized for small spaces")
            # Density thresholds for demo (small room)
            self.density_low = 0.15      # Low density
            self.density_medium = 0.35   # Medium density
            self.density_high = 0.55     # High density (danger)
            
            # People count thresholds
            self.people_threshold_low = 3
            self.people_threshold_medium = 6
            self.people_threshold_high = 10
        else:
            print("📌 Production Mode: Optimized for large venues")
            # Density thresholds for real deployment (large venues)
            self.density_low = 0.3       # Low density
            self.density_medium = 0.6    # Medium density
            self.density_high = 0.9      # High density (danger)
            
            # People count thresholds
            self.people_threshold_low = 10
            self.people_threshold_medium = 20
            self.people_threshold_high = 30
        
        # Zone grid for density analysis (divide frame into grid)
        self.grid_rows = 6
        self.grid_cols = 8
        self.density_map = None
        
        # Virtual boundary
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
        
        # NEW: Predictive Risk Analysis
        self.prediction_window = 10  # Analyze last 10 frames
        self.density_trend = []
        self.risk_prediction = {
            'accident_probability': 0.0,
            'time_to_critical': None,
            'risk_level': 'SAFE',
            'trend': 'STABLE',
            'recommendation': 'Normal monitoring'
        }
        
        # Alarm sound thread
        self.alarm_thread = None
    
    def set_boundary_line(self, frame_height, position=0.7):
        """Set virtual boundary line"""
        self.boundary_line = int(frame_height * position)
    
    def play_alarm_sound(self, alarm_type='overcrowding'):
        """
        Play alarm sound (runs in separate thread)
        alarm_type: 'overcrowding', 'violation', 'critical'
        """
        try:
            if alarm_type == 'overcrowding':
                # Three short beeps
                for _ in range(3):
                    winsound.Beep(1000, 200)  # 1000Hz, 200ms
                    time.sleep(0.1)
            elif alarm_type == 'violation':
                # Two long beeps
                for _ in range(2):
                    winsound.Beep(1500, 400)  # 1500Hz, 400ms
                    time.sleep(0.15)
            elif alarm_type == 'critical':
                # Continuous alarm (5 rapid beeps)
                for _ in range(5):
                    winsound.Beep(2000, 150)  # 2000Hz, 150ms
                    time.sleep(0.05)
        except Exception as e:
            # For Mac/Linux or if winsound fails
            print(f"\nALARM: {alarm_type.upper()}")
    
    def trigger_alarm(self, alarm_type='overcrowding'):
        """Trigger alarm with cooldown"""
        current_time = time.time()
        if current_time - self.last_alarm_time > self.alarm_cooldown:
            self.alarm_active = True
            self.last_alarm_time = current_time
            
            # Play sound in separate thread (non-blocking)
            if self.alarm_thread is None or not self.alarm_thread.is_alive():
                self.alarm_thread = threading.Thread(
                    target=self.play_alarm_sound, 
                    args=(alarm_type,)
                )
                self.alarm_thread.daemon = True
                self.alarm_thread.start()
            
            return True
        return False
    
    def detect_people(self, frame):
        """Detect all people in frame"""
        results = self.model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes
        
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
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
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
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
        
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Fill heatmap based on density grid
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y1 = i * cell_height
                y2 = (i + 1) * cell_height
                x1 = j * cell_width
                x2 = (j + 1) * cell_width
                
                heatmap[y1:y2, x1:x2] = density_grid[i, j]
        
        # Apply Gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        # Convert to color heatmap (blue=low, green=medium, red=high)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
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
        
        # Calculate density score (0-100) - adjusted for demo mode
        if self.demo_mode:
            # Demo mode: More sensitive to small numbers
            density_score = (
                avg_density * 100 +  # Base density (higher weight)
                (critical_zones / (self.grid_rows * self.grid_cols)) * 50 +  # Critical zone ratio
                min(people_count / self.people_threshold_high, 1.0) * 30  # People count factor
            )
        else:
            # Production mode: Calibrated for large venues
            density_score = (
                avg_density * 50 +  # Base density
                (critical_zones / (self.grid_rows * self.grid_cols)) * 30 +  # Critical zone ratio
                min(people_count / self.people_threshold_high, 1.0) * 20  # People count factor
            )
        
        # Classify density level with mode-specific thresholds
        if self.demo_mode:
            # Demo thresholds (lower for small rooms)
            if density_score < 25:
                level = 'LOW'
                color = (0, 255, 0)  # Green
            elif density_score < 50:
                level = 'MODERATE'
                color = (0, 255, 255)  # Yellow
            elif density_score < 70:
                level = 'HIGH'
                color = (0, 165, 255)  # Orange
            else:
                level = 'CRITICAL'
                color = (0, 0, 255)  # Red
        else:
            # Production thresholds (higher for large venues)
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
    
    def predict_accident_risk(self, current_density, density_history):
        """
        PREDICTIVE FEATURE: Analyze density trends to predict accident probability
        
        Returns:
        - accident_probability (0-100): Likelihood of accident in next 2-5 minutes
        - time_to_critical: Estimated seconds until critical density
        - risk_level: SAFE, LOW, MODERATE, HIGH, IMMINENT
        - trend: DECREASING, STABLE, INCREASING, RAPIDLY_INCREASING
        - recommendation: Action to take
        """
        
        if len(density_history) < 3:
            return {
                'accident_probability': 0.0,
                'time_to_critical': None,
                'risk_level': 'SAFE',
                'trend': 'STABLE',
                'recommendation': 'Insufficient data for prediction'
            }
        
        # Use last N frames for prediction
        recent_density = density_history[-self.prediction_window:] if len(density_history) >= self.prediction_window else density_history
        
        # Calculate trend (rate of change)
        if len(recent_density) >= 2:
            # Linear regression to find trend
            x = np.arange(len(recent_density))
            y = np.array(recent_density)
            
            # Calculate slope (rate of density increase)
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0
            
            # Calculate acceleration (is it speeding up?)
            if len(recent_density) >= 4:
                first_half_avg = np.mean(recent_density[:len(recent_density)//2])
                second_half_avg = np.mean(recent_density[len(recent_density)//2:])
                acceleration = second_half_avg - first_half_avg
            else:
                acceleration = 0
        else:
            slope = 0
            acceleration = 0
        
        # Determine trend
        if slope < -2:
            trend = 'DECREASING'
        elif slope < 2:
            trend = 'STABLE'
        elif slope < 5:
            trend = 'INCREASING'
        else:
            trend = 'RAPIDLY_INCREASING'
        
        # Calculate time to critical (extrapolation)
        time_to_critical = None
        critical_threshold = 85 if self.demo_mode else 80
        
        if slope > 0.5 and current_density < critical_threshold:
            # Estimate time to reach critical density
            density_gap = critical_threshold - current_density
            # Assuming 30 FPS, convert frames to seconds
            frames_to_critical = density_gap / slope
            time_to_critical = int(frames_to_critical / 30)  # Convert to seconds
            
            # Cap at reasonable values
            time_to_critical = max(10, min(time_to_critical, 600))  # Between 10s and 10min
        
        # Calculate accident probability (0-100)
        # Factors:
        # 1. Current density level (40% weight)
        # 2. Rate of increase (30% weight)
        # 3. Acceleration (20% weight)
        # 4. Critical zones (10% weight)
        
        density_factor = min(current_density / 100, 1.0) * 40
        
        # Slope factor (rapid increase = higher risk)
        slope_factor = min(abs(slope) / 10, 1.0) * 30 if slope > 0 else 0
        
        # Acceleration factor (accelerating increase = danger)
        accel_factor = min(abs(acceleration) / 20, 1.0) * 20 if acceleration > 0 else 0
        
        # Critical zones factor
        critical_zones_ratio = self.stats['critical_zones'] / (self.grid_rows * self.grid_cols)
        zones_factor = critical_zones_ratio * 10
        
        accident_probability = density_factor + slope_factor + accel_factor + zones_factor
        accident_probability = min(accident_probability, 100)
        
        # Determine risk level
        if accident_probability < 20:
            risk_level = 'SAFE'
            recommendation = 'Normal monitoring. No action needed.'
        elif accident_probability < 40:
            risk_level = 'LOW'
            recommendation = 'Monitor closely. Prepare crowd control measures.'
        elif accident_probability < 60:
            risk_level = 'MODERATE'
            recommendation = 'Alert security staff. Consider restricting entry.'
        elif accident_probability < 80:
            risk_level = 'HIGH'
            recommendation = '⚠️ URGENT: Stop entry immediately. Deploy crowd control.'
        else:
            risk_level = 'IMMINENT'
            recommendation = '🚨 CRITICAL: EVACUATION NEEDED! Accident likely within 2 minutes!'
        
        return {
            'accident_probability': round(accident_probability, 1),
            'time_to_critical': time_to_critical,
            'risk_level': risk_level,
            'trend': trend,
            'recommendation': recommendation,
            'slope': round(slope, 2),
            'acceleration': round(acceleration, 2)
        }
    
    def check_boundary_violations(self, detections, boundary_enabled=True):
        """
        Check boundary violations
        boundary_enabled: If False, always return no violations
        """
        if not boundary_enabled or self.boundary_line is None:
            return False, []
        
        violators = []
        for det in detections:
            cx, cy = det['center']
            if cy > self.boundary_line:
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
            self.stats['avg_density'] = np.mean(self.stats['density_history'])
        
        # NEW: Predict accident risk based on density trends
        self.risk_prediction = self.predict_accident_risk(density_score, self.stats['density_history'])
        
        # Check violations - pass boundary_enabled flag
        has_violation, violators = self.check_boundary_violations(detections, boundary_enabled=show_boundary)
        if has_violation:
            self.stats['violations'] += 1
        
        # Draw boundary line
        if show_boundary and self.boundary_line:
            cv2.line(vis_frame, (0, self.boundary_line), 
                    (frame.shape[1], self.boundary_line), 
                    (0, 0, 255), 4)
            cv2.putText(vis_frame, 'RESTRICTED ZONE', 
                       (10, self.boundary_line - 15),
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
                cv2.putText(vis_frame, '⚠️ VIOLATION', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw info panel with density information
        panel_h = 350  # Increased height for prediction info
        panel_w = 550
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
        # Violations - ONLY show if boundary is enabled
        if show_boundary and self.boundary_line is not None:
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
        else:
            cv2.putText(vis_frame, 'Boundary: Disabled', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Trigger alarms based on conditions
        alarm_triggered = False
        alarm_type = None
        
        # PRIORITY 1: Imminent risk prediction (NEW!)
        if self.risk_prediction['risk_level'] == 'IMMINENT':
            if self.trigger_alarm('critical'):
                alarm_triggered = True
                alarm_type = 'IMMINENT ACCIDENT RISK!'
                self.stats['alerts_triggered'] += 1
        
        # PRIORITY 2: Boundary violations (ONLY if boundary is enabled)
        elif has_violation and show_boundary and self.boundary_line is not None:
            if self.trigger_alarm('violation'):
                alarm_triggered = True
                alarm_type = 'BOUNDARY VIOLATION'
                self.stats['alerts_triggered'] += 1
        
        # PRIORITY 3: High accident probability (NEW!)
        elif self.risk_prediction['accident_probability'] > 70:
            if self.trigger_alarm('critical'):
                alarm_triggered = True
                alarm_type = f'HIGH ACCIDENT RISK ({self.risk_prediction["accident_probability"]:.0f}%)'
                self.stats['alerts_triggered'] += 1
        
        # PRIORITY 4: Critical density
        elif density_level == 'CRITICAL':
            if self.trigger_alarm('critical'):
                alarm_triggered = True
                alarm_type = 'CRITICAL DENSITY'
                self.stats['alerts_triggered'] += 1
        
        # PRIORITY 5: High density with critical zones
        elif density_level == 'HIGH' and critical_zones > 3:
            if self.trigger_alarm('overcrowding'):
                alarm_triggered = True
                alarm_type = 'HIGH DENSITY'
                self.stats['alerts_triggered'] += 1
        
        # Show alarm indicator
        if alarm_triggered or (time.time() - self.last_alarm_time < 1.0):
            # Flash alarm indicator
            alarm_panel_y = vis_frame.shape[0] - 100
            cv2.rectangle(vis_frame, (0, alarm_panel_y), 
                         (vis_frame.shape[1], vis_frame.shape[0]), 
                         (0, 0, 255), -1)
            cv2.putText(vis_frame, f'ALARM: {alarm_type if alarm_type else "ACTIVE"}', 
                       (vis_frame.shape[1]//2 - 250, alarm_panel_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return vis_frame, density_level, density_score, has_violation, alarm_triggered
    
    def process_video(self, video_source=0, save_output=False, show_heatmap=True, boundary_mode='percentage'):
        """
        Main processing loop with density analysis
        
        boundary_mode options:
        - 'auto': Try automatic detection
        - 'manual': User clicks to set
        - 'percentage': Fixed percentage (70%)
        - 'none': No boundary detection
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
        
        print(f"📐 Video size: {frame_width}x{frame_height} @ {fps} FPS")
        
        # Get first frame for boundary setup
        ret, first_frame = cap.read()
        if not ret:
            print("❌ Cannot read first frame!")
            return
        
        # Set up boundary based on mode
        if boundary_mode != 'none':
            print(f"\n🚧 Boundary Detection Mode: {boundary_mode.upper()}")
            
            if boundary_mode == 'auto':
                success = self.set_boundary_line(frame_height, frame=first_frame, mode='auto')
                if not success:
                    print("⚠️  Auto-detection failed. Try manual mode? (y/n)")
                    choice = input().strip().lower()
                    if choice == 'y':
                        self.set_boundary_line(frame_height, frame=first_frame, mode='manual')
                    else:
                        self.set_boundary_line(frame_height, position=0.70, mode='percentage')
            
            elif boundary_mode == 'manual':
                self.set_boundary_line(frame_height, frame=first_frame, mode='manual')
            
            else:  # percentage
                self.set_boundary_line(frame_height, position=0.70, mode='percentage')
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
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
            cv2.imshow('🚨 Crowd Density Monitoring - Press Q to quit', vis_frame)
            
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
    print("🚨 ADVANCED CROWD DENSITY MONITORING SYSTEM")
    print("="*60 + "\n")
    
    # Choose mode
    print("Select mode:")
    print("1. Demo Mode (Small room / Hackathon)")
    print("2. Production Mode (Large venue)")
    mode_choice = input("\nEnter choice (1 or 2): ").strip()
    
    demo_mode = True if mode_choice == '1' else False
    detector = CrowdDensityDetector(model_size='n', demo_mode=demo_mode)
    
    print("\nSelect video source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        video_src = 0
    elif choice == '2':
        video_src = input("Enter video file path: ").strip()
    else:
        print("Invalid choice. Using webcam...")
        video_src = 0
    
    # Boundary detection mode
    print("\nBoundary detection mode:")
    print("1. Auto-detect (finds platform edges automatically)")
    print("2. Manual (click to set)")
    print("3. Default 70% (simple horizontal line)")
    print("4. No boundary detection")
    boundary_choice = input("\nEnter choice (1-4): ").strip()
    
    boundary_modes = {
        '1': 'auto',
        '2': 'manual',
        '3': 'percentage',
        '4': 'none'
    }
    boundary_mode = boundary_modes.get(boundary_choice, 'percentage')
    
    detector.process_video(video_source=video_src, save_output=(choice=='2'), 
                          show_heatmap=True, boundary_mode=boundary_mode)