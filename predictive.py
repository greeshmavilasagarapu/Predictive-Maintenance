import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.models import load_model

# --------------------------------------------------
# 1. UI CONFIGURATION & ADVANCED CSS
# --------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance – ML System",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS for Professional Industrial UI
st.markdown("""
    <style>
    /* Main Background and Font */
    .stApp {
        background-color: #f8f9fc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    
    /* Force white color for Sidebar Titles and Subheaders */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p {
        color: #ffffff;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Styling */
    h1 { color: #0f172a; font-weight: 800; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }
    h2, h3 { color: #1e40af; font-weight: 600; }

    /* Custom Alert Boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 2. RESOURCE LOADING
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_rul_model.h5", compile=False)
    # Support original and renamed scaler file
    scaler_path = "scaler (1).pkl" if os.path.exists("scaler (1).pkl") else "scaler.pkl"
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"❌ System Error: Assets not found. {e}")
    st.stop()

# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.title("Navigation") 
    st.markdown("Navigate through the system modules below:")
    page = st.radio(
        "",
        ["Model Overview", "Dataset Visualization", "Prediction Results", "Analytics & Insights", "Project Information"]
    )
    st.markdown("---")
    st.subheader("Dataset Uplaod") 
    uploaded_file = st.file_uploader("Upload Engine Log (.txt)", type=["txt"])

# --------------------------------------------------
# 4. DATA ENGINE
# --------------------------------------------------
if not uploaded_file:
    st.title("🔧 Predictive Maintenance of Industrial Equipment")
    st.info("### Getting Started\nPlease upload the **.txt** file in the sidebar to initialize system.")
    
    st.stop()

@st.cache_data
def process_data(file):
    df = pd.read_csv(file, sep=r"\s+", header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = ["engine_id", "cycle", "os_1", "os_2", "os_3"] + [f"sensor_{i}" for i in range(1, 22)]
    return df

data = process_data(uploaded_file)
sensor_cols = [c for c in data.columns if "sensor" in c]
feature_cols = ["cycle"] + sensor_cols
X_scaled = scaler.transform(data[feature_cols].values)

# LSTM Sequence Setup
SEQ_LEN = 30
def create_sequences(X, seq_len):
    if len(X) < seq_len: return np.array([])
    return np.array([X[i:i+seq_len] for i in range(len(X)-seq_len+1)])

X_seq = create_sequences(X_scaled, SEQ_LEN)

if X_seq.size > 0:
    predicted_rul = model.predict(X_seq).flatten()
else:
    st.error(f"Insufficient Data: Need {SEQ_LEN} cycles for analysis.")
    st.stop()

# --------------------------------------------------
# 5. UI MODULES
# --------------------------------------------------

# --- MODULE 1: MODEL OVERVIEW ---
if page == "Model Overview":
    st.title("🧠 Model Overview")
    st.markdown("""
    **Problem:** Aircraft engines generate massive telemetry data. Predicting failure before it happens is the "Holy Grail" of maintenance.
    **Solution:** This LSTM model captures "time-memory" patterns in sensor noise to predict the **Remaining Useful Life (RUL)**.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Average RUL", f"{predicted_rul.mean():.1f} Cycles")
    col2.metric("Max RUL Detected", f"{predicted_rul.max():.1f} Cycles")
    col3.metric("Fleet Units", data["engine_id"].nunique())
    
    st.markdown("---")
    st.info("💡 **Developer Note:** The LSTM architecture uses a 30-cycle sliding window to maintain temporal context.")

# --- MODULE 2: DATASET VISUALIZATION ---
elif page == "Dataset Visualization":
    st.title("📈 Dataset Visualization")
    st.subheader("Raw Telemetry Stream (Sample)")
    st.dataframe(data.head(15), use_container_width=True)
    
    st.subheader("Sensor Trend Analysis")
    selected_sensor = st.selectbox("Select Sensor Stream", sensor_cols)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["cycle"], data[selected_sensor], color='#3b82f6', lw=2)
    ax.set_title(f"Performance History: {selected_sensor}")
    ax.set_xlabel("Operational Cycles")
    ax.set_ylabel("Reading")
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)

# --- MODULE 3: PREDICTION RESULTS ---
elif page == "Prediction Results":
    st.title("📉 Prediction Results")
    
    # RUL Decay Chart
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(predicted_rul, color='#1e293b', linewidth=2.5, label="Predicted RUL Path")
    ax.fill_between(range(len(predicted_rul)), 0, 30, color='#ef4444', alpha=0.1, label="Critical Zone")
    ax.fill_between(range(len(predicted_rul)), 30, 80, color='#f59e0b', alpha=0.1, label="Caution Zone")
    ax.set_ylabel("RUL (Cycles)")
    ax.set_xlabel("Observation Time")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("### Decision Logic")
    current_rul = predicted_rul[-1]
    
    res1, res2 = st.columns([1, 2])
    res1.metric("Current Engine RUL", f"{current_rul:.1f} Cycles")
    
    if current_rul > 80:
        res2.success("**Status: Healthy.** The engine shows minimal wear patterns.")
    elif current_rul > 30:
        res2.warning("**Status: Degrading.** Wear detected. Schedule maintenance within 20 cycles.")
    else:
        res2.error("**Status: Critical.** Failure is imminent. Ground unit immediately.")

# --- MODULE 4: ANALYTICS & INSIGHTS ---
elif page == "Analytics & Insights":
    st.title("📊 Analytics & Maintenance Insights")
    
    # FIX: Cast health to standard Python float to prevent the float32 Streamlit error
    health_raw = (predicted_rul[-1] / predicted_rul.max())
    health_float = float(np.clip(health_raw, 0.0, 1.0)) 
    
    st.write(f"### System Health Score: **{health_float * 100:.1f}%**")
    st.progress(health_float) 
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Maintenance Directive")
        if health_float < 0.3:
            st.error("🛑 **URGENT:** Immediate Maintenance Required")
        else:
            st.success("📅 **PLANNED:** Routine Maintenance Schedule Active")
            
    with col_b:
        st.subheader("Sensor Cross-Correlation")
        st.dataframe(data[sensor_cols].corr().iloc[:6, :6].style.background_gradient(cmap="Blues"))

# --- MODULE 5: PROJECT INFORMATION ---
elif page == "Project Information":
    st.title("ℹ️ Project Information")
    
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        **System Goal:** To transform reactive repairs into a proactive strategy using Deep Learning.
        
        **Technology Stack:**
        - **Core:** Python & TensorFlow/Keras
        - **Data:** Pandas, NumPy, Scikit-learn
        - **UI:** Streamlit with Custom CSS injection
        
        **Real-world Applications:**
        - ✈️ **Aviation:** Turbofan engine safety.
        - 🏭 **Manufacturing:** CNC and assembly line uptime.
        - ⚡ **Energy:** Wind turbine and generator monitoring.
        """)
    
    with p2:
        st.markdown("**Dataset Source:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)")

st.markdown("---")
st.caption("Predictive Maintenance System | Final ML Dashboard Implementation")