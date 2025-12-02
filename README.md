# 🚨 AI-Powered Crowd Density & Safety Monitoring System

**Real-time people detection for public safety using YOLOv8**

---

## 📁 Project Structure

```
crowd-monitoring/
├── test_simple.py         # Step 1: Test if YOLO works
├── crowd_detector.py      # Step 2: Core detection engine
├── app.py                 # Step 3: Interactive dashboard
├── requirements.txt       # All dependencies
└── README.md             # This file
```

---

## 🚀 Complete Setup (Step-by-Step)

### Step 1: Install Python
- Download Python 3.10 or 3.11 from https://python.org
- ⚠️ Check "Add Python to PATH" during installation
- Verify: Open terminal/cmd and type `python --version`

### Step 2: Create Project Folder
```bash
# Navigate to Desktop
cd Desktop

# Create project folder
mkdir crowd-monitoring
cd crowd-monitoring
```

### Step 3: Create Files
Copy all the artifact files into this folder:
- `test_simple.py`
- `crowd_detector.py`
- `app.py`
- `requirements.txt`

### Step 4: Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# You should see (venv) appear in your terminal
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

This will take 5-10 minutes. Wait for it to complete!

---

## ✅ Testing (Do this in order!)

### Test 1: Quick YOLO Test
```bash
python test_simple.py
```

**What should happen:**
- Model downloads (first time only, ~6MB)
- Webcam opens
- You see boxes around people
- Press 'q' to quit

**If webcam doesn't work:**
- Try different camera: Edit `test_simple.py` and change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
- Or use a video file: Change to `cv2.VideoCapture('your_video.mp4')`

### Test 2: Full Detection System
```bash
python crowd_detector.py
```

**What should happen:**
- Webcam opens with full interface
- People counter on screen
- Zone status (Green/Yellow/Red)
- Boundary line displayed
- Press 'q' to quit, 's' for screenshot

### Test 3: Interactive Dashboard
```bash
streamlit run app.py
```

**What should happen:**
- Browser opens automatically
- Interactive dashboard appears
- Click "▶️ Start Monitoring"
- See live video with statistics

---

## 🎯 Key Features

### 1. Real-Time People Detection
- Uses YOLOv8 (state-of-the-art AI)
- Detects and counts people instantly
- Works at 30+ FPS

### 2. Zone-Based Risk Assessment
- 🟢 **Green (Safe)**: 0-10 people
- 🟡 **Yellow (Moderate)**: 11-20 people
- 🔴 **Red (Danger)**: 21+ people

### 3. Virtual Boundary Detection
- Set a restricted zone line
- Alerts when people cross
- Perfect for railway platforms

### 4. Live Dashboard
- Real-time statistics
- Alert logging
- Historical graphs
- Screenshot capability

---

## 🎬 For Hackathon Demo

### Option 1: Webcam Demo (Quick)
1. Run: `streamlit run app.py`
2. Have team members walk in/out of frame
3. Show zone changes and violations

### Option 2: Video Demo (Better)
1. Download a crowd video from YouTube (search "railway station crowd")
2. Upload through dashboard
3. Smooth, repeatable demo

### Demo Script (5 minutes)

**Minute 1: Problem Statement**
- "Overcrowding causes 500+ deaths/year in India"
- "Manual monitoring is slow and error-prone"
- Show news articles/photos

**Minute 2: Solution Overview**
- "AI-powered real-time detection"
- Show system architecture diagram
- Mention YOLOv8, OpenCV

**Minute 3: Live Demo**
- Start dashboard
- Show people detection
- Demonstrate zone changes
- Trigger boundary violation

**Minute 4: Features**
- Point to statistics
- Show alert log
- Explain scalability

**Minute 5: Impact & Q&A**
- "Deployable on $50 hardware"
- "Can scale to 100+ cameras"
- Answer questions

---

## 🛠️ Customization

### Change Zone Thresholds
In `crowd_detector.py` or via dashboard sidebar:
```python
self.green_threshold = 10   # Change this
self.yellow_threshold = 20  # Change this
```

### Change Boundary Position
In dashboard: Use slider "Boundary Position"
Or in code:
```python
self.set_boundary_line(frame_height, position=0.70)  # 0.7 = 70% down
```

### Use Different Camera
```python
# Camera 0 (default)
detector.process_video(video_source=0)

# Camera 1
detector.process_video(video_source=1)

# Video file
detector.process_video(video_source='crowd.mp4')
```

---

## ❓ Troubleshooting

### "Cannot open webcam"
- Try different camera index (0, 1, 2)
- Check if another app is using camera
- Use a video file instead

### "Module not found"
```bash
# Make sure virtual environment is activated
# You should see (venv) in terminal
pip install -r requirements.txt
```

### "Streamlit not found"
```bash
pip install streamlit
```

### Slow Performance
- Use YOLOv8n (nano) - fastest
- Process every 2nd frame
- Reduce video resolution

### Port already in use (Streamlit)
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Technical Stack

- **AI Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Dashboard**: Streamlit
- **Language**: Python 3.10+
- **Hardware**: Works on laptops, can deploy on Raspberry Pi/Jetson Nano

---

## 🎓 For Judges

### Why This Project Stands Out

1. **Real Problem**: Addresses actual safety concerns in Indian railways/metros
2. **Production Ready**: Working system, not just concept
3. **Scalable**: Affordable hardware deployment
4. **Innovation**: Zone-based intelligence + predictive alerts
5. **Impact**: Can save lives by preventing stampedes

### Technical Highlights
- State-of-the-art YOLOv8 detection
- Real-time processing (30+ FPS)
- Edge computing compatible
- Modular architecture
- Interactive visualization

---

## 📞 Support

If you get stuck:
1. Check error message carefully
2. Google the error
3. Ask your team members
4. Check if virtual environment is activated

---

## 📄 License

Built for educational/hackathon purposes.

---

**Good luck with your hackathon! 🚀**