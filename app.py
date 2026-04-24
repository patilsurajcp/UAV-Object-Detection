# app.py — Fixed Version (No raw HTML display)
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io
import time
from detect import UAVDetector
from utils import (
    get_detection_stats,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_object_map,
    get_detection_table
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title            = "UAV Detection System",
    page_icon             = "🚁",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# =====================================================
# CSS ONLY — No inline HTML blocks
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --cyan  : #00f5ff;
    --green : #00ff88;
    --red   : #ff003c;
    --bg    : #010409;
    --panel : rgba(0,20,40,0.85);
    --glow  : rgba(0,245,255,0.3);
}

/* Animated grid background */
.stApp {
    background-color  : var(--bg) !important;
    background-image  :
        linear-gradient(rgba(0,245,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.04) 1px, transparent 1px);
    background-size   : 50px 50px;
    animation         : gridScroll 20s linear infinite;
}
@keyframes gridScroll {
    0%   { background-position: 0 0; }
    100% { background-position: 50px 50px; }
}

/* Scan line */
.stApp::before {
    content   : '';
    position  : fixed;
    top       : -5px;
    left      : 0;
    width     : 100%;
    height    : 2px;
    background: linear-gradient(90deg,
        transparent 0%, var(--cyan) 50%, transparent 100%);
    animation : scanLine 5s linear infinite;
    z-index   : 9999;
    pointer-events: none;
    opacity   : 0.6;
}
@keyframes scanLine {
    0%   { top: -5px;  }
    100% { top: 100vh; }
}

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(
        180deg,
        rgba(0,8,16,0.98) 0%,
        rgba(0,20,40,0.98) 100%
    ) !important;
    border-right: 1px solid rgba(0,245,255,0.15) !important;
}

/* Main block */
.main .block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family   : 'Orbitron', monospace !important;
    color         : var(--cyan) !important;
    letter-spacing: 3px !important;
    text-shadow   : 0 0 30px rgba(0,245,255,0.4) !important;
}
h4, h5, h6 {
    font-family   : 'Orbitron', monospace !important;
    color         : var(--cyan) !important;
    letter-spacing: 2px !important;
}
p, label, li {
    font-family: 'Rajdhani', sans-serif !important;
    color      : #c0d8e8 !important;
    font-size  : 1rem !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background    : rgba(0,8,20,0.9) !important;
    border        : 1px solid rgba(0,245,255,0.2) !important;
    border-top    : 2px solid var(--cyan) !important;
    border-radius : 3px !important;
    padding       : 18px 16px !important;
}
div[data-testid="stMetricValue"] {
    font-family   : 'Orbitron', monospace !important;
    color         : var(--cyan) !important;
    font-size     : 1.8rem !important;
    font-weight   : 900 !important;
    text-shadow   : 0 0 20px var(--cyan) !important;
}
div[data-testid="stMetricLabel"] {
    font-family   : 'Share Tech Mono', monospace !important;
    color         : rgba(0,245,255,0.4) !important;
    font-size     : 0.65rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background   : rgba(0,8,20,0.8) !important;
    border-bottom: 1px solid rgba(0,245,255,0.2) !important;
    gap          : 0 !important;
}
button[data-baseweb="tab"] {
    font-family   : 'Orbitron', monospace !important;
    font-size     : 0.7rem !important;
    font-weight   : 700 !important;
    letter-spacing: 2px !important;
    color         : rgba(0,245,255,0.4) !important;
    padding       : 14px 20px !important;
    border-bottom : 2px solid transparent !important;
    background    : transparent !important;
    transition    : all 0.3s ease !important;
}
button[data-baseweb="tab"]:hover {
    color      : var(--cyan) !important;
    background : rgba(0,245,255,0.05) !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    color       : var(--cyan) !important;
    border-color: var(--cyan) !important;
    background  : rgba(0,245,255,0.08) !important;
    text-shadow : 0 0 15px var(--cyan) !important;
}

/* ── Buttons ── */
div.stButton > button {
    font-family   : 'Orbitron', monospace !important;
    font-size     : 0.75rem !important;
    font-weight   : 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    background    : transparent !important;
    border        : 1px solid var(--green) !important;
    color         : var(--green) !important;
    border-radius : 2px !important;
    padding       : 12px 28px !important;
    transition    : all 0.3s ease !important;
}
div.stButton > button:hover {
    background : rgba(0,255,136,0.1) !important;
    box-shadow : 0 0 25px rgba(0,255,136,0.4) !important;
    transform  : translateY(-2px) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background   : rgba(0,15,30,0.8) !important;
    border       : 1px dashed rgba(0,245,255,0.3) !important;
    border-radius: 3px !important;
    transition   : all 0.3s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    box-shadow  : 0 0 20px rgba(0,245,255,0.08) !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--cyan) !important;
}

/* ── Alert boxes ── */
div[data-testid="stInfo"] {
    background   : rgba(0,245,255,0.04) !important;
    border       : 1px solid rgba(0,245,255,0.2) !important;
    border-left  : 3px solid var(--cyan) !important;
    border-radius: 2px !important;
    font-family  : 'Rajdhani', sans-serif !important;
}
div[data-testid="stSuccess"] {
    background   : rgba(0,255,136,0.04) !important;
    border       : 1px solid rgba(0,255,136,0.2) !important;
    border-left  : 3px solid var(--green) !important;
    border-radius: 2px !important;
    font-family  : 'Rajdhani', sans-serif !important;
}
div[data-testid="stWarning"] {
    background   : rgba(255,160,0,0.04) !important;
    border       : 1px solid rgba(255,160,0,0.2) !important;
    border-left  : 3px solid #ffa000 !important;
    border-radius: 2px !important;
    font-family  : 'Rajdhani', sans-serif !important;
}
div[data-testid="stError"] {
    background   : rgba(255,0,60,0.04) !important;
    border       : 1px solid rgba(255,0,60,0.2) !important;
    border-left  : 3px solid var(--red) !important;
    border-radius: 2px !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,245,255,0.15) !important;
}

/* ── Checkbox ── */
label[data-testid="stCheckbox"] span {
    font-family: 'Share Tech Mono', monospace !important;
    color      : rgba(0,245,255,0.7) !important;
    font-size  : 0.8rem !important;
    letter-spacing: 1px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
    background   : var(--cyan);
    border-radius: 2px;
    opacity      : 0.5;
}

/* ── Radar widget ── */
#radar-widget {
    position      : fixed;
    bottom        : 25px;
    right         : 25px;
    width         : 110px;
    height        : 110px;
    z-index       : 9998;
    pointer-events: none;
    opacity       : 0.5;
}
@keyframes sweep {
    from { transform: rotate(0deg);   }
    to   { transform: rotate(360deg); }
}
@keyframes blip {
    0%,75%,100% { opacity: 0; transform: scale(0.5); }
    80%         { opacity: 1; transform: scale(2); }
    90%         { opacity: 0.4; }
}

/* ── Corner frames ── */
.corner-tl, .corner-tr, .corner-bl, .corner-br {
    position      : fixed;
    width         : 35px;
    height        : 35px;
    z-index       : 9997;
    pointer-events: none;
    opacity       : 0.4;
}
.corner-tl {
    top: 8px; left: 8px;
    border-top: 2px solid var(--cyan);
    border-left: 2px solid var(--cyan);
}
.corner-tr {
    top: 8px; right: 8px;
    border-top: 2px solid var(--cyan);
    border-right: 2px solid var(--cyan);
}
.corner-bl {
    bottom: 8px; left: 8px;
    border-bottom: 2px solid var(--cyan);
    border-left: 2px solid var(--cyan);
}
.corner-br {
    bottom: 8px; right: 8px;
    border-bottom: 2px solid var(--cyan);
    border-right: 2px solid var(--cyan);
}
</style>

<!-- Radar + Corner decorations -->
<div class="corner-tl"></div>
<div class="corner-tr"></div>
<div class="corner-bl"></div>
<div class="corner-br"></div>

<div id="radar-widget">
<svg viewBox="0 0 110 110" width="110" height="110">
  <circle cx="55" cy="55" r="50" fill="none" stroke="rgba(0,245,255,0.2)" stroke-width="0.5"/>
  <circle cx="55" cy="55" r="35" fill="none" stroke="rgba(0,245,255,0.2)" stroke-width="0.5"/>
  <circle cx="55" cy="55" r="20" fill="none" stroke="rgba(0,245,255,0.3)" stroke-width="0.5"/>
  <circle cx="55" cy="55" r="5"  fill="none" stroke="rgba(0,245,255,0.4)" stroke-width="0.8"/>
  <line x1="55" y1="5"  x2="55" y2="105" stroke="rgba(0,245,255,0.12)" stroke-width="0.5"/>
  <line x1="5"  y1="55" x2="105" y2="55" stroke="rgba(0,245,255,0.12)" stroke-width="0.5"/>
  <g style="transform-origin:55px 55px; animation:sweep 3s linear infinite;">
    <path d="M55,55 L55,5 A50,50 0 0,1 105,55 Z"
      fill="rgba(0,245,255,0.12)"/>
    <line x1="55" y1="55" x2="55" y2="5"
      stroke="rgba(0,245,255,0.9)" stroke-width="1.2"/>
  </g>
  <circle cx="72" cy="30" r="2"
    fill="#00ff88"
    style="animation:blip 3s 0.9s linear infinite;"/>
  <circle cx="38" cy="70" r="1.5"
    fill="#00ff88"
    style="animation:blip 3s 1.8s linear infinite;"/>
  <defs>
    <radialGradient id="g1" cx="55" cy="55" r="50" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="rgba(0,245,255,0.3)"/>
      <stop offset="100%" stop-color="rgba(0,245,255,0)"/>
    </radialGradient>
  </defs>
</svg>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    for path in ["best.pt", "models/best.pt"]:
        if os.path.exists(path):
            try:
                os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
                return UAVDetector(path)
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
    st.error("❌ Model not found! Upload best.pt")
    st.stop()

# =====================================================
# HEADER — Pure Streamlit (No HTML blocks)
# =====================================================
st.markdown("---")
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🚁")
with col_title:
    st.markdown("# UAV OBJECT DETECTION SYSTEM")
    st.caption(
        "[ SYS-ID: UAV-DET-7X  |  STATUS: ONLINE  |  "
        "MODEL: YOLOv8m  |  DATASET: VisDrone-2019  |  "
        "CLASSES: 10  |  CLEARANCE: ALPHA ]"
    )
st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ CONTROL PANEL")
    st.markdown("---")

    st.markdown("**▸ DETECTION PARAMS**")
    conf_thresh = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.10, max_value=0.90,
        value=0.35, step=0.05
    )
    iou_thresh = st.slider(
        "IoU THRESHOLD",
        min_value=0.10, max_value=0.90,
        value=0.45, step=0.05
    )

    st.markdown("---")
    st.markdown("**▸ DISPLAY OPTIONS**")
    show_stats = st.checkbox("SHOW STATISTICS",    value=True)
    show_map   = st.checkbox("SHOW LOCATION MAP",  value=True)
    show_table = st.checkbox("SHOW RESULTS TABLE", value=True)

    st.markdown("---")
    st.info("""
    **◈ MODEL INFO**

    Model: YOLOv8m
    Dataset: VisDrone 2019
    Classes: 10
    Epochs: 100
    """)

    st.markdown("---")
    st.markdown("""
    **◈ TARGET CLASSES**

    🚶 Pedestrian · 👥 People
    🚲 Bicycle · 🚗 Car
    🚐 Van · 🚚 Truck
    🛺 Tricycle · ⛺ Awning-Tricycle
    🚌 Bus · 🏍️ Motor
    """)

    st.markdown("---")
    st.success("🟢 SYSTEM ONLINE")

# =====================================================
# LOAD DETECTOR
# =====================================================
detector = load_model()

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "📸  IMAGE DETECTION",
    "🎥  VIDEO DETECTION",
    "📹  REAL-TIME DETECTION"
])

# =====================================================
# TAB 1 — IMAGE DETECTION
# =====================================================
with tab1:
    st.markdown("### 📡 IMAGE ANALYSIS")
    st.caption("UPLOAD AERIAL IMAGE FOR OBJECT DETECTION")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "DROP TARGET IMAGE HERE",
        type=['jpg', 'jpeg', 'png'],
        help="Upload drone/UAV captured image"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        image      = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_arr    = np.array(image)
        h, w       = img_arr.shape[:2]

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.caption("◈ INPUT FEED")
            st.image(image, use_column_width=True)
            st.caption(f"RES: {w}×{h}px | {uploaded_file.name}")

        if st.button("⚡ INITIATE DETECTION SCAN", type="primary"):
            with st.spinner("▸ SCANNING TARGET... RUNNING YOLOv8..."):
                start                 = time.time()
                annotated, detections = detector.detect_image(
                    img_arr, conf_thresh, iou_thresh
                )
                elapsed = time.time() - start

            with col2:
                st.caption("◈ DETECTION OUTPUT")
                st.image(annotated, use_column_width=True)
                st.caption(f"SCAN COMPLETE | TIME: {elapsed:.3f}s")

            # Metrics
            stats = get_detection_stats(detections)
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4, gap="small")
            with m1:
                st.metric("🎯 TOTAL TARGETS",  stats.get('total', 0))
            with m2:
                st.metric("📦 CLASSES FOUND",  len(stats.get('classes', [])))
            with m3:
                st.metric("💯 AVG CONFIDENCE", f"{stats.get('avg_conf', 0):.2f}")
            with m4:
                st.metric("⚡ SCAN TIME",      f"{elapsed:.2f}s")

            # Charts
            if show_stats and detections:
                st.markdown("---")
                st.markdown("### 📊 DETECTION ANALYTICS")
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    fig1 = plot_class_distribution(detections)
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                with c2:
                    fig2 = plot_confidence_distribution(detections)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)

            # Map
            if show_map and detections:
                st.markdown("---")
                st.markdown("### 🗺️ TACTICAL LOCATION MAP")
                fig3 = plot_object_map(detections, w, h)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)

            # Table
            if show_table and detections:
                st.markdown("---")
                st.markdown("### 📋 DETECTION LOG")
                df = get_detection_table(detections)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 EXPORT DETECTION LOG",
                    df.to_csv(index=False),
                    "uav_detection_log.csv", "text/csv"
                )

            if not detections:
                st.warning(
                    "⚠️ NO TARGETS DETECTED — "
                    "Lower confidence threshold to 0.20"
                )

# =====================================================
# TAB 2 — VIDEO DETECTION
# =====================================================
with tab2:
    st.markdown("### 🎬 VIDEO ANALYSIS")
    st.caption("FRAME-BY-FRAME AERIAL SURVEILLANCE")
    st.markdown("---")

    video_file = st.file_uploader(
        "DROP TARGET VIDEO HERE",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload drone/UAV video file"
    )

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4'
        )
        tfile.write(video_file.read())
        tfile.flush()

        st.caption("◈ SOURCE FEED")
        st.video(tfile.name)

        max_frames = st.slider(
            "MAX FRAMES TO ANALYZE",
            min_value=10, max_value=200,
            value=50, step=10
        )

        if st.button("🎯 LAUNCH VIDEO ANALYSIS", type="primary"):
            cap   = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS)

            st.info(
                f"TOTAL FRAMES: {total} | "
                f"SOURCE FPS: {fps:.1f} | "
                f"ANALYZING: {max_frames} FRAMES"
            )

            progress          = st.progress(0)
            status            = st.empty()
            frame_placeholder = st.empty()

            frame_count  = 0
            total_counts = []

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated, count = detector.detect_video_frame(
                    frame_rgb, conf_thresh, iou_thresh
                )
                total_counts.append(count)
                frame_count     += 1
                frame_placeholder.image(
                    annotated,
                    caption=f"FRAME {frame_count}/{max_frames} | TARGETS: {count}",
                    use_column_width=True
                )
                progress.progress(frame_count / max_frames)
                status.caption(
                    f"▸ PROCESSING FRAME {frame_count}/{max_frames}..."
                )

            cap.release()
            try:
                os.unlink(tfile.name)
            except:
                pass

            # Summary
            st.markdown("---")
            st.markdown("### 📊 MISSION SUMMARY")
            v1, v2, v3 = st.columns(3, gap="small")
            with v1:
                st.metric("🎬 FRAMES ANALYZED",  frame_count)
            with v2:
                avg = sum(total_counts)/len(total_counts) if total_counts else 0
                st.metric("📦 AVG TARGETS/FRAME", f"{avg:.1f}")
            with v3:
                st.metric("🔝 PEAK TARGETS",
                    max(total_counts) if total_counts else 0)

            if total_counts:
                import plotly.graph_objects as go
                import pandas as pd

                fig_v = go.Figure(go.Scatter(
                    x         = list(range(1, len(total_counts)+1)),
                    y         = total_counts,
                    mode      = 'lines',
                    line      = dict(color='#00f5ff', width=2),
                    fill      = 'tozeroy',
                    fillcolor = 'rgba(0,245,255,0.06)',
                    name      = 'Targets'
                ))
                fig_v.update_layout(
                    title         = 'TARGET COUNT PER FRAME',
                    xaxis_title   = 'FRAME',
                    yaxis_title   = 'TARGETS',
                    plot_bgcolor  = 'rgba(0,0,0,0)',
                    paper_bgcolor = 'rgba(0,8,20,0.8)',
                    font          = dict(
                        color  = 'rgba(0,245,255,0.7)',
                        family = 'Share Tech Mono'
                    ),
                    height = 350,
                    yaxis  = dict(gridcolor='rgba(0,245,255,0.07)'),
                    xaxis  = dict(gridcolor='rgba(0,245,255,0.07)'),
                )
                st.plotly_chart(fig_v, use_container_width=True)

            status.success("✓ MISSION COMPLETE — ANALYSIS FINISHED")

# =====================================================
# TAB 3 — REAL-TIME DETECTION
# =====================================================
with tab3:
    st.markdown("### 📡 LIVE SURVEILLANCE")
    st.caption("REAL-TIME AERIAL OBJECT DETECTION")
    st.markdown("---")

    st.info(
        "💻 **DESKTOP (Chrome)** → Use WebRTC Live Stream  |  "
        "📱 **MOBILE** → Use Snapshot Mode below  |  "
        "🔒 Camera permission required"
    )

    rt_col1, rt_col2, rt_col3 = st.columns(3, gap="small")
    with rt_col1:
        rt_status = st.empty()
    with rt_col2:
        rt_stream = st.empty()
    with rt_col3:
        rt_conf   = st.empty()

    rt_status.metric("🎯 STATUS",     "⭕ OFFLINE")
    rt_stream.metric("📡 STREAM",     "INACTIVE")
    rt_conf.metric("⚙️ CONFIDENCE",   f"{conf_thresh:.2f}")

    st.markdown("---")
    st.markdown("#### 🖥️ WEBRTC LIVE STREAM")

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        class UAVVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.conf        = 0.35
                self.iou         = 0.45
                self.frame_count = 0
                self.start_time  = time.time()

            def recv(self, frame):
                img     = frame.to_ndarray(format="bgr24")
                results = detector.model(
                    img, conf=self.conf,
                    iou=self.iou, imgsz=640, verbose=False
                )[0]
                annotated = results.plot(
                    font_size=8, line_width=1,
                    labels=True, conf=True
                )
                self.frame_count += 1
                elapsed  = time.time() - self.start_time
                fps      = self.frame_count / elapsed if elapsed > 0 else 0
                count    = len(results.boxes)
                cv2.putText(
                    annotated, f"FPS:{fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2
                )
                cv2.putText(
                    annotated, f"TARGETS:{count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,200,255), 2
                )
                return av.VideoFrame.from_ndarray(
                    annotated, format="bgr24"
                )

        ctx = webrtc_streamer(
            key                     = "uav-live",
            video_processor_factory = UAVVideoProcessor,
            media_stream_constraints= {
                "video": {
                    "width"    : {"ideal": 1280},
                    "height"   : {"ideal": 720},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing = True,
            rtc_configuration= {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                    {"urls": ["stun:stun.cloudflare.com:3478"]},
                    {"urls": ["stun:stun.relay.metered.ca:80"]},
                    {
                        "urls"      : ["turn:global.relay.metered.ca:80"],
                        "username"  : "uav",
                        "credential": "uavdetect"
                    },
                    {
                        "urls"      : ["turn:global.relay.metered.ca:443"],
                        "username"  : "uav",
                        "credential": "uavdetect"
                    },
                    {
                        "urls"      : ["turn:global.relay.metered.ca:443?transport=tcp"],
                        "username"  : "uav",
                        "credential": "uavdetect"
                    },
                ],
                "iceTransportPolicy": "all",
            }
        )

        if ctx.video_processor:
            ctx.video_processor.conf = conf_thresh
            ctx.video_processor.iou  = iou_thresh

        if ctx.state.playing:
            rt_status.metric("🎯 STATUS",   "🟢 LIVE")
            rt_stream.metric("📡 STREAM",   "ACTIVE ✅")
            rt_conf.metric("⚙️ CONF",       f"{conf_thresh:.2f}")
            st.success("✅ REAL-TIME DETECTION ACTIVE!")
        else:
            st.warning(
                "⚠️ Connection issue? Use Desktop Chrome | "
                "Or scroll down → use Snapshot Mode 👇"
            )

    except ImportError:
        st.error("❌ streamlit-webrtc not installed!")

    # ── Snapshot Mode ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📸 SNAPSHOT SCANNER")
    st.success("✅ WORKS ON ALL DEVICES — MOBILE, TABLET & DESKTOP")

    camera_image = st.camera_input(
        "◈ CAPTURE TARGET IMAGE",
        help="Works on all devices!"
    )

    if camera_image is not None:
        file_bytes = camera_image.read()
        snap_image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")
        snap_arr   = np.array(snap_image)
        snap_h, snap_w = snap_arr.shape[:2]

        with st.spinner("▸ SCANNING TARGET..."):
            start           = time.time()
            annotated, dets = detector.detect_image(
                snap_arr, conf_thresh, iou_thresh
            )
            elapsed = time.time() - start

        s1, s2 = st.columns(2, gap="medium")
        with s1:
            st.caption("◈ CAPTURED IMAGE")
            st.image(snap_image, use_column_width=True)
            st.caption(f"{snap_w}×{snap_h}px")
        with s2:
            st.caption("◈ SCAN RESULT")
            st.image(annotated, use_column_width=True)
            st.caption(f"TARGETS: {len(dets)} | TIME: {elapsed:.3f}s")

        if dets:
            snap_stats = get_detection_stats(dets)
            st.markdown("---")
            ss1, ss2, ss3 = st.columns(3, gap="small")
            with ss1:
                st.metric("🎯 TARGETS",    snap_stats.get('total', 0))
            with ss2:
                st.metric("📦 CLASSES",    len(snap_stats.get('classes', [])))
            with ss3:
                st.metric("💯 AVG CONF",   f"{snap_stats.get('avg_conf', 0):.2f}")

            if show_stats:
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    fig_s = plot_class_distribution(dets)
                    if fig_s:
                        st.plotly_chart(fig_s, use_container_width=True)
                with c2:
                    fig_c = plot_confidence_distribution(dets)
                    if fig_c:
                        st.plotly_chart(fig_c, use_container_width=True)

            if show_map:
                fig_m = plot_object_map(dets, snap_w, snap_h)
                if fig_m:
                    st.plotly_chart(fig_m, use_container_width=True)

            if show_table:
                df_snap = get_detection_table(dets)
                st.dataframe(
                    df_snap,
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "📥 EXPORT SCAN LOG",
                    df_snap.to_csv(index=False),
                    "scan_log.csv", "text/csv"
                )
        else:
            st.warning(
                "⚠️ NO TARGETS ACQUIRED — "
                "Lower confidence threshold to 0.20"
            )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "🚁 UAV OBJECT DETECTION SYSTEM  |  "
    "YOLOv8 ENGINE  |  VisDrone-2019  |  "
    "10 TARGET CLASSES  |  HUGGING FACE SPACES"
)