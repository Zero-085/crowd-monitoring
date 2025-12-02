"""
AI-Powered Crowd Density Monitoring System
Streamlit Dashboard - app.py

Run this with: streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
from crowd_detector import CrowdDetector
import time
from datetime import datetime
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="🚨 Crowd Monitoring System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = CrowdDetector(model_size='n')
    st.session_state.running = False
    st.session_state.history = []
    st.session_state.alert_log = []
    st.session_state.frame_count = 0

def main():
    # Header
    st.markdown('<p class="main-header">🚨 AI-Powered Crowd Monitoring System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time People Detection & Safety Alerts for Public Spaces</p>', 
                unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Video source selection
        st.subheader("📹 Video Source")
        source_type = st.radio("Select Source", ["📷 Webcam", "📁 Upload Video"], label_visibility="collapsed")
        
        video_source = 0
        temp_video_path = "temp_uploaded_video.mp4"
        
        if source_type == "📁 Upload Video":
            uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file:
                # Save uploaded file
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                video_source = temp_video_path
                st.success("✅ Video uploaded successfully!")
        
        st.divider()
        
        # Zone threshold settings
        st.subheader("🎯 Zone Thresholds")
        st.caption("Set crowd density limits for each zone")
        
        green_threshold = st.slider(
            "🟢 Safe → Moderate", 
            min_value=5, 
            max_value=30, 
            value=10,
            help="Maximum people for SAFE zone"
        )
        
        yellow_threshold = st.slider(
            "🟡 Moderate → Danger", 
            min_value=green_threshold + 1, 
            max_value=50, 
            value=20,
            help="Maximum people for MODERATE zone"
        )
        
        # Update detector thresholds
        st.session_state.detector.green_threshold = green_threshold
        st.session_state.detector.yellow_threshold = yellow_threshold
        
        st.divider()
        
        # Boundary detection
        st.subheader("🚧 Boundary Detection")
        boundary_enabled = st.checkbox("Enable Virtual Boundary", value=True)
        
        if boundary_enabled:
            boundary_pos = st.slider(
                "Boundary Position (%)", 
                min_value=30, 
                max_value=90, 
                value=70,
                help="Position of restricted zone line (% from top)"
            ) / 100
        else:
            boundary_pos = 0.70
        
        st.divider()
        
        # Alert settings
        st.subheader("🔔 Alert Settings")
        show_alerts = st.checkbox("Show Toast Alerts", value=True)
        
        st.divider()
        
        # System info
        st.subheader("ℹ️ System Info")
        st.info(f"""
        **Model**: YOLOv8 Nano  
        **Status**: {'🟢 Running' if st.session_state.running else '🔴 Stopped'}  
        **Frames**: {st.session_state.frame_count}
        """)
    
    # Main content area
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        people_metric = st.empty()
    with col2:
        zone_metric = st.empty()
    with col3:
        violation_metric = st.empty()
    with col4:
        max_metric = st.empty()
    
    # Initialize metrics
    people_metric.metric("👥 People Count", 0)
    zone_metric.metric("🎯 Zone Status", "🟢 SAFE")
    violation_metric.metric("⚠️ Violations", "None")
    max_metric.metric("📊 Max Count", 0)
    
    st.divider()
    
    # Main video and stats layout
    video_col, stats_col = st.columns([2.5, 1])
    
    with video_col:
        st.subheader("📹 Live Video Feed")
        frame_placeholder = st.empty()
        
        # Control buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            start_btn = st.button("▶️ Start Monitoring", use_container_width=True, type="primary")
        with btn_col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        with btn_col3:
            snapshot_btn = st.button("📸 Snapshot", use_container_width=True)
    
    with stats_col:
        st.subheader("📊 Live Statistics")
        stats_placeholder = st.empty()
        
        st.subheader("🚨 Recent Alerts")
        alert_placeholder = st.empty()
    
    # Chart section
    st.divider()
    st.subheader("📈 Crowd Density Timeline")
    chart_placeholder = st.empty()
    
    # Control logic
    if start_btn:
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.rerun()
    
    if stop_btn:
        st.session_state.running = False
        st.rerun()
    
    # Main processing loop
    if st.session_state.running:
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video source! Please check your camera or video file.")
            st.session_state.running = False
            st.stop()
        
        # Set boundary based on frame height
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if boundary_enabled:
            st.session_state.detector.set_boundary_line(frame_height, boundary_pos)
        else:
            st.session_state.detector.boundary_line = None
        
        # Process frames
        while st.session_state.running:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("⚠️ End of video or cannot read frame")
                st.session_state.running = False
                break
            
            st.session_state.frame_count += 1
            
            # Process every frame for smooth video
            # Detect people
            detections = st.session_state.detector.detect_people(frame)
            count = len(detections)
            
            # Draw visualization
            vis_frame, zone, violation = st.session_state.detector.draw_visualization(
                frame, detections, show_boundary=boundary_enabled
            )
            
            # Convert BGR to RGB for display
            vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            
            # Update metrics
            people_metric.metric("👥 People Count", count)
            
            # Zone status with colors
            zone_emoji = {'SAFE': '🟢', 'MODERATE': '🟡', 'DANGER': '🔴'}
            zone_metric.metric("🎯 Zone Status", f"{zone_emoji.get(zone, '⚪')} {zone}")
            
            # Violation status
            if violation:
                violation_metric.metric("⚠️ Violations", "🚨 ACTIVE", delta="Alert!", delta_color="inverse")
            else:
                violation_metric.metric("⚠️ Violations", "✅ None", delta=None)
            
            max_metric.metric("📊 Max Count", st.session_state.detector.stats['max_count'])
            
            # Record history
            timestamp = datetime.now()
            st.session_state.history.append({
                'time': timestamp,
                'count': count,
                'zone': zone,
                'violation': violation
            })
            
            # Keep only last 100 records
            if len(st.session_state.history) > 100:
                st.session_state.history.pop(0)
            
            # Log alerts
            if violation and show_alerts:
                alert_entry = {
                    'Time': timestamp.strftime("%H:%M:%S"),
                    'Type': '🚧 Boundary Violation',
                    'Count': count
                }
                if not st.session_state.alert_log or st.session_state.alert_log[0] != alert_entry:
                    st.session_state.alert_log.insert(0, alert_entry)
                    st.toast("🚨 BOUNDARY VIOLATION DETECTED!", icon="⚠️")
            
            if zone == 'DANGER' and show_alerts:
                alert_entry = {
                    'Time': timestamp.strftime("%H:%M:%S"),
                    'Type': '🔴 Overcrowding Alert',
                    'Count': count
                }
                if not st.session_state.alert_log or st.session_state.alert_log[0] != alert_entry:
                    st.session_state.alert_log.insert(0, alert_entry)
                    st.toast(f"🔴 DANGER ZONE: {count} people detected!", icon="🚨")
            
            # Keep only last 15 alerts
            st.session_state.alert_log = st.session_state.alert_log[:15]
            
            # Display video frame
            frame_placeholder.image(vis_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update statistics panel
            stats_data = {
                '👥 Current Count': count,
                '📊 Maximum Count': st.session_state.detector.stats['max_count'],
                '🚨 Total Alerts': st.session_state.detector.stats['alerts_triggered'],
                '🎯 Current Zone': zone,
                '🎬 Frames Processed': st.session_state.frame_count
            }
            stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
            stats_placeholder.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Update alert log
            if st.session_state.alert_log:
                alert_df = pd.DataFrame(st.session_state.alert_log)
                alert_placeholder.dataframe(alert_df, use_container_width=True, hide_index=True)
            else:
                alert_placeholder.info("No alerts yet")
            
            # Update chart
            if len(st.session_state.history) > 1:
                df = pd.DataFrame(st.session_state.history)
                df['time_str'] = df['time'].dt.strftime('%H:%M:%S')
                chart_placeholder.line_chart(df.set_index('time')['count'], use_container_width=True)
            
            # Handle snapshot
            if snapshot_btn:
                snapshot_path = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_path, vis_frame)
                st.success(f"📸 Snapshot saved: {snapshot_path}")
            
            # Small delay for smoother video
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        
        # Clean up temp video file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
    
    else:
        # Show placeholder when not running
        st.info("👆 Click 'Start Monitoring' to begin real-time detection")
        
        # Show sample instructions
        with st.expander("📖 How to Use This System"):
            st.markdown("""
            ### Quick Start Guide
            
            1. **Select Video Source**
               - Choose Webcam for live demo
               - Upload a video file for testing
            
            2. **Configure Settings**
               - Adjust zone thresholds (left sidebar)
               - Enable boundary detection if needed
            
            3. **Start Monitoring**
               - Click "▶️ Start Monitoring"
               - Watch real-time detection
               - Monitor alerts and statistics
            
            4. **Key Features**
               - 🟢 **Green Zone**: Safe (low density)
               - 🟡 **Yellow Zone**: Moderate crowd
               - 🔴 **Red Zone**: Danger (overcrowding)
               - 🚧 **Boundary Detection**: Restricted area alerts
            
            """)

if __name__ == "__main__":
    main()