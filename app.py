"""
AI-Powered Crowd Density Monitoring System
Advanced Streamlit Dashboard with Density Analysis & Alarms

Run with: streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
from crowd_detector import CrowdDensityDetector
import time
from datetime import datetime
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🚨 Crowd Density Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 46px;
        font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px;
        transition: all 0.3s;
    }
    .alarm-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #FF0000, #FF4B4B);
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .density-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 18px;
    }
    .density-low { background: #00FF00; color: black; }
    .density-moderate { background: #FFFF00; color: black; }
    .density-high { background: #FFA500; color: white; }
    .density-critical { background: #FF0000; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    # Start in DEMO MODE by default for hackathon
    st.session_state.detector = CrowdDensityDetector(model_size='n', demo_mode=True)
    st.session_state.running = False
    st.session_state.history = []
    st.session_state.alert_log = []
    st.session_state.frame_count = 0
    st.session_state.show_heatmap = True
    st.session_state.alarm_enabled = True
    st.session_state.alarm_count = 0
    st.session_state.boundary_pos = 0.70
    st.session_state.boundary_setup_mode = 'percentage'

def create_density_gauge(density_score):
    """Create a gauge chart for density score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = density_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Density Score", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFFF99'},
                {'range': [60, 80], 'color': '#FFB366'},
                {'range': [80, 100], 'color': '#FF6666'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        height=300
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🚨 AI-Powered Crowd Density Monitoring</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Density Analysis • Heatmap Visualization • Smart Alarm System</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Mode selection at the top
        st.subheader("🎯 Operation Mode")
        mode = st.radio(
            "Select Mode:",
            ["🎪 Demo Mode (Small Room)", "🏢 Production Mode (Large Venue)"],
            index=0,
            help="Demo mode is optimized for hackathon demos with 2-10 people"
        )
        
        # Switch mode if changed
        new_demo_mode = (mode == "🎪 Demo Mode (Small Room)")
        if new_demo_mode != st.session_state.detector.demo_mode:
            st.session_state.detector = CrowdDensityDetector(model_size='n', demo_mode=new_demo_mode)
            st.success(f"✅ Switched to {mode}")
        
        # Show current thresholds
        if st.session_state.detector.demo_mode:
            st.info(f"""
            **Demo Thresholds:**
            - Low → Moderate: {st.session_state.detector.people_threshold_low} people
            - Moderate → High: {st.session_state.detector.people_threshold_medium} people  
            - High → Critical: {st.session_state.detector.people_threshold_high} people
            """)
        else:
            st.info(f"""
            **Production Thresholds:**
            - Low → Moderate: {st.session_state.detector.people_threshold_low} people
            - Moderate → High: {st.session_state.detector.people_threshold_medium} people
            - High → Critical: {st.session_state.detector.people_threshold_high} people
            """)
        
        st.divider()
        
        # Video source
        st.subheader("📹 Video Source")
        source_type = st.radio("", ["📷 Webcam", "📁 Upload Video"], label_visibility="collapsed")
        
        video_source = 0
        temp_video_path = "temp_uploaded_video.mp4"
        
        if source_type == "📁 Upload Video":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file:
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                video_source = temp_video_path
                st.success("✅ Video uploaded!")
        
        st.divider()
        
        # Density thresholds
        st.subheader("🎯 Density Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            density_low = st.slider("Low Threshold", 0.1, 0.5, 0.3, 0.05)
        with col2:
            density_medium = st.slider("Med Threshold", 0.3, 0.8, 0.6, 0.05)
        
        density_high = st.slider("High Threshold", 0.5, 1.0, 0.9, 0.05)
        
        st.session_state.detector.density_low = density_low
        st.session_state.detector.density_medium = density_medium
        st.session_state.detector.density_high = density_high
        
        st.divider()
        
        # Grid configuration
        st.subheader("🗺️ Analysis Grid")
        grid_size = st.select_slider(
            "Grid Resolution",
            options=["Coarse (4x6)", "Medium (6x8)", "Fine (8x10)"],
            value="Medium (6x8)"
        )
        
        if grid_size == "Coarse (4x6)":
            st.session_state.detector.grid_rows, st.session_state.detector.grid_cols = 4, 6
        elif grid_size == "Fine (8x10)":
            st.session_state.detector.grid_rows, st.session_state.detector.grid_cols = 8, 10
        else:
            st.session_state.detector.grid_rows, st.session_state.detector.grid_cols = 6, 8
        
        st.divider()
        
        # Visualization options
        st.subheader("🎨 Visualization")
        st.session_state.show_heatmap = st.checkbox("Show Density Heatmap", value=True)
        show_boundary = st.checkbox("Show Boundary Line", value=True)
        
        if show_boundary:
            boundary_pos = st.slider("Boundary Position (%)", 30, 90, 70) / 100
        else:
            boundary_pos = 0.70
        
        st.divider()
        
        # Alarm settings
        st.subheader("🔔 Alarm System")
        st.session_state.alarm_enabled = st.checkbox("Enable Audio Alarms", value=True)
        alarm_cooldown = st.slider("Alarm Cooldown (sec)", 1, 10, 3)
        st.session_state.detector.alarm_cooldown = alarm_cooldown
        
        show_toast = st.checkbox("Show Toast Alerts", value=True)
        
        st.divider()
        
        # System status
        st.subheader("ℹ️ System Status")
        status_color = "🟢" if st.session_state.running else "🔴"
        st.info(f"""
        **Status**: {status_color} {'Running' if st.session_state.running else 'Stopped'}  
        **Model**: YOLOv8 Nano  
        **Frames**: {st.session_state.frame_count}  
        **Alarms**: {st.session_state.alarm_count}
        """)
    
    # Main metrics
    metric_cols = st.columns(6)  # Added one more column
    
    with metric_cols[0]:
        people_metric = st.empty()
    with metric_cols[1]:
        density_metric = st.empty()
    with metric_cols[2]:
        level_metric = st.empty()
    with metric_cols[3]:
        zones_metric = st.empty()
    with metric_cols[4]:
        violation_metric = st.empty()
    with metric_cols[5]:
        risk_metric = st.empty()  # NEW: Risk prediction
    
    # Initialize metrics
    people_metric.metric("👥 People", 0)
    density_metric.metric("📊 Density", "0.0")
    level_metric.metric("🎯 Level", "LOW")
    zones_metric.metric("⚠️ Critical Zones", 0)
    violation_metric.metric("🚧 Violations", 0)
    risk_metric.metric("🔮 Accident Risk", "0%")  # NEW
    
    st.divider()
    
    # Alarm banner (when active)
    alarm_placeholder = st.empty()
    
    # Main layout
    left_col, right_col = st.columns([2.5, 1.5])
    
    with left_col:
        st.subheader("📹 Live Density Analysis")
        frame_placeholder = st.empty()
        
        # Controls
        btn1, btn2, btn3, btn4 = st.columns(4)
        with btn1:
            start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
        with btn2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        with btn3:
            snapshot_btn = st.button("📸 Snapshot", use_container_width=True)
        with btn4:
            reset_btn = st.button("🔄 Reset Stats", use_container_width=True)
    
    with right_col:
        st.subheader("📊 Density Gauge")
        gauge_placeholder = st.empty()
        
        st.subheader("🔮 Risk Prediction")  # NEW
        risk_panel = st.empty()
        
        st.subheader("📈 Statistics")
        stats_placeholder = st.empty()
        
        st.subheader("🚨 Alert Log")
        alert_placeholder = st.empty()
    
    # Charts section
    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Density Timeline")
        timeline_chart = st.empty()
    
    with chart_col2:
        st.subheader("🔥 Density Distribution")
        distribution_chart = st.empty()
    
    # Control handlers
    if start_btn:
        st.session_state.running = True
        st.session_state.frame_count = 0
        st.rerun()
    
    if stop_btn:
        st.session_state.running = False
        st.rerun()
    
    if reset_btn:
        st.session_state.detector.stats = {
            'current_density': 0.0,
            'max_density': 0.0,
            'avg_density': 0.0,
            'density_history': [],
            'alerts_triggered': 0,
            'critical_zones': 0,
            'violations': 0
        }
        st.session_state.history = []
        st.session_state.alert_log = []
        st.session_state.alarm_count = 0
        st.success("📊 Statistics reset!")
    
    # Main processing
    if st.session_state.running:
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video source!")
            st.session_state.running = False
            st.stop()
        
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up boundary based on mode
        if st.session_state.boundary_setup_mode == 'auto':
            # Get first frame for auto-detection
            ret, first_frame = cap.read()
            if ret:
                with st.spinner("🔍 Auto-detecting boundary..."):
                    # Try to use new auto-detection if available
                    if hasattr(st.session_state.detector, 'detect_boundary_automatically'):
                        detected_y = st.session_state.detector.detect_boundary_automatically(first_frame)
                        if detected_y:
                            st.session_state.detector.boundary_line = detected_y
                            st.success(f"✅ Boundary auto-detected at {int(detected_y/frame_height*100)}%")
                        else:
                            st.warning("⚠️ Auto-detection failed. Using default 70%")
                            st.session_state.detector.boundary_line = int(frame_height * 0.70)
                    else:
                        st.warning("⚠️ Auto-detection not available. Using default 70%")
                        st.session_state.detector.boundary_line = int(frame_height * 0.70)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        elif st.session_state.boundary_setup_mode == 'percentage':
            st.session_state.detector.boundary_line = int(frame_height * st.session_state.boundary_pos)
        elif st.session_state.boundary_setup_mode == 'none':
            st.session_state.detector.boundary_line = None
        
        # Disable alarm if not enabled
        if not st.session_state.alarm_enabled:
            st.session_state.detector.alarm_cooldown = 999999
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ End of video")
                st.session_state.running = False
                break
            
            st.session_state.frame_count += 1
            
            # Process frame
            detections = st.session_state.detector.detect_people(frame)
            people_count = len(detections)
            
            # Get visualization
            vis_frame, density_level, density_score, violation, alarm = \
                st.session_state.detector.draw_visualization(
                    frame, detections,
                    show_heatmap=st.session_state.show_heatmap,
                    show_boundary=show_boundary
                )
            
            # Convert to RGB
            vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            
            # Update metrics
            people_metric.metric("👥 People", people_count)
            density_metric.metric("📊 Density", f"{density_score:.1f}/100")
            
            level_colors = {
                'LOW': '🟢',
                'MODERATE': '🟡',
                'HIGH': '🟠',
                'CRITICAL': '🔴'
            }
            level_metric.metric("🎯 Level", f"{level_colors.get(density_level, '⚪')} {density_level}")
            
            zones_metric.metric("⚠️ Critical Zones", 
                              st.session_state.detector.stats['critical_zones'])
            violation_metric.metric("🚧 Violations", 
                                   st.session_state.detector.stats['violations'])
            
            # NEW: Risk prediction metric
            risk_prob = st.session_state.detector.risk_prediction['accident_probability']
            risk_level = st.session_state.detector.risk_prediction['risk_level']
            risk_delta = "⚠️ Action needed!" if risk_prob > 60 else None
            risk_metric.metric("🔮 Accident Risk", 
                             f"{risk_prob:.0f}%",
                             delta=risk_delta,
                             delta_color="inverse" if risk_prob > 60 else "normal")
            
            # Show alarm banner
            if alarm:
                st.session_state.alarm_count += 1
                alarm_placeholder.markdown(
                    '<div class="alarm-box">🚨 ALARM ACTIVE 🚨</div>',
                    unsafe_allow_html=True
                )
                if show_toast:
                    if violation:
                        st.toast("🚧 BOUNDARY VIOLATION!", icon="🚨")
                    else:
                        st.toast(f"⚠️ {density_level} DENSITY ALERT!", icon="🔴")
            else:
                alarm_placeholder.empty()
            
            # Record history
            timestamp = datetime.now()
            st.session_state.history.append({
                'time': timestamp,
                'density': density_score,
                'people': people_count,
                'level': density_level,
                'critical_zones': st.session_state.detector.stats['critical_zones']
            })
            
            if len(st.session_state.history) > 200:
                st.session_state.history.pop(0)
            
            # Log alerts
            if alarm:
                alert_type = "🚧 Boundary Violation" if violation else f"🔴 {density_level} Density"
                alert_entry = {
                    'Time': timestamp.strftime("%H:%M:%S"),
                    'Type': alert_type,
                    'Density': f"{density_score:.1f}",
                    'People': people_count
                }
                st.session_state.alert_log.insert(0, alert_entry)
                st.session_state.alert_log = st.session_state.alert_log[:20]
            
            # Display frame
            frame_placeholder.image(vis_frame_rgb, use_container_width=True)
            
            # Update gauge
            gauge_fig = create_density_gauge(density_score)
            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{st.session_state.frame_count}")
            
            # NEW: Update risk prediction panel
            risk_pred = st.session_state.detector.risk_prediction
            risk_color = "🟢" if risk_pred['risk_level'] == 'SAFE' else "🟡" if risk_pred['risk_level'] == 'LOW' else "🟠" if risk_pred['risk_level'] == 'MODERATE' else "🔴"
            
            trend_arrow = "📉" if risk_pred['trend'] == 'DECREASING' else "➡️" if risk_pred['trend'] == 'STABLE' else "📈" if risk_pred['trend'] == 'INCREASING' else "⚠️"
            
            risk_info = f"""
            **{risk_color} Risk Level:** {risk_pred['risk_level']}  
            **{trend_arrow} Trend:** {risk_pred['trend']}  
            **🎯 Probability:** {risk_pred['accident_probability']:.1f}%  
            """
            
            if risk_pred['time_to_critical']:
                ttc = risk_pred['time_to_critical']
                if ttc < 60:
                    risk_info += f"**⏰ Time to Critical:** {ttc}s  \n"
                else:
                    risk_info += f"**⏰ Time to Critical:** {ttc//60}m {ttc%60}s  \n"
            
            risk_info += f"\n**💡 Action:** {risk_pred['recommendation']}"
            
            if risk_pred['accident_probability'] > 60:
                risk_panel.error(risk_info)
            elif risk_pred['accident_probability'] > 40:
                risk_panel.warning(risk_info)
            else:
                risk_panel.info(risk_info)
            
            # Update statistics
            stats_data = {
                'Metric': [
                    '📊 Current Density',
                    '🔝 Max Density',
                    '📉 Avg Density',
                    '🚨 Total Alarms',
                    '⚠️ Critical Zones'
                ],
                'Value': [
                    f"{density_score:.1f}",
                    f"{st.session_state.detector.stats['max_density']:.1f}",
                    f"{st.session_state.detector.stats['avg_density']:.1f}",
                    st.session_state.alarm_count,
                    st.session_state.detector.stats['critical_zones']
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_placeholder.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Update alert log
            if st.session_state.alert_log:
                alert_df = pd.DataFrame(st.session_state.alert_log)
                alert_placeholder.dataframe(alert_df, use_container_width=True, hide_index=True)
            else:
                alert_placeholder.info("No alerts yet")
            
            # Update charts
            if len(st.session_state.history) > 2:
                df = pd.DataFrame(st.session_state.history)
                
                # Timeline chart
                timeline_chart.line_chart(df.set_index('time')['density'], use_container_width=True)
                
                # Distribution chart
                density_ranges = ['LOW (0-30)', 'MODERATE (30-60)', 'HIGH (60-80)', 'CRITICAL (80-100)']
                counts = [
                    len([d for d in df['density'] if d < 30]),
                    len([d for d in df['density'] if 30 <= d < 60]),
                    len([d for d in df['density'] if 60 <= d < 80]),
                    len([d for d in df['density'] if d >= 80])
                ]
                
                dist_fig = px.bar(
                    x=density_ranges,
                    y=counts,
                    color=counts,
                    color_continuous_scale=['green', 'yellow', 'orange', 'red']
                )
                dist_fig.update_layout(
                    showlegend=False,
                    xaxis_title="Density Level",
                    yaxis_title="Frame Count",
                    height=300
                )
                distribution_chart.plotly_chart(dist_fig, use_container_width=True, key=f"dist_{st.session_state.frame_count}")
            
            # Snapshot
            if snapshot_btn:
                snap_path = f"density_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snap_path, vis_frame)
                st.success(f"📸 Saved: {snap_path}")
            
            time.sleep(0.01)
        
        cap.release()
        
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
    
    else:
        st.info("👆 Click **'Start'** to begin density monitoring")
        
        with st.expander("📖 System Features & Usage"):
            st.markdown("""
            ### 🌟 Advanced Features
            
            **1. Density Heatmap Visualization**
            - Color-coded density map overlay
            - Real-time hotspot identification
            - Grid-based analysis
            
            **2. Multi-Level Density Classification**
            - 🟢 LOW: Safe crowd levels
            - 🟡 MODERATE: Monitored areas
            - 🟠 HIGH: Warning threshold
            - 🔴 CRITICAL: Immediate action needed
            
            **3. Smart Alarm System**
            - Audio alerts (beep patterns)
            - Visual flash indicators
            - Configurable cooldown period
            - Alert logging with timestamps
            
            **4. Critical Zone Detection**
            - Identifies overcrowded grid cells
            - Tracks multiple hotspots
            - Boundary violation alerts
    
            
            ### 💡 What Makes This Unique
            
            ✨ **Beyond Simple Counting**: Analyzes spatial distribution  
            ✨ **Predictive Alerts**: Warns before critical levels  
            ✨ **Multi-Modal Alarms**: Audio + Visual + Logging  
            ✨ **Scalable Architecture**: Works on edge devices  
            ✨ **Real-World Ready**: Tested for public safety scenarios
            """)

if __name__ == "__main__":
    main()