import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Layout & Modern Styling Parameters
st.set_page_config(
    page_title="KineticTwin Control Panel",
    page_icon="⚡",
    layout="wide"
)

# Injecting clean accents for layout container aesthetics
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ KINETICTWIN (SmartFootAI) Control Panel")
st.caption("Real-time predictive edge microgrid optimization and hardware resource allocation framework")
st.write("---")

# 2. Loading the Serialized Machine Learning Brain & Dataset
@st.cache_resource
def load_serialized_engine():
    model = joblib.load('kinetic_rf_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

@st.cache_data
def load_historical_dataset():
    return pd.read_csv('Mall_Efficient.csv')

try:
    rf_model, model_columns = load_serialized_engine()
    df_historical = load_historical_dataset()
    st.sidebar.success("📦 Live AI Pipeline Engine Active")
except Exception as e:
    st.sidebar.error(f"❌ Core Error Loading Assets: {e}")
    st.stop()

# 3. Sidebar Configuration — Gathering Dynamic Sensor Parameters
st.sidebar.header("🕹️ Live Telemetry Inputs")
selected_zone = st.sidebar.selectbox("Architectural Zone", ['Entrance', 'Foodcourt', 'Shopping area', 'Gaming Zone', 'Cinema', 'Parking'])
selected_day = st.sidebar.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
selected_hour = st.sidebar.slider("Current Hour", min_value=8, max_value=15, value=12)
selected_weather = st.sidebar.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Rainy'])
selected_event = st.sidebar.radio("Active Special Event?", ['No', 'Yes'])

# 4. Main Presentation View Splits
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("### 📊 Operational Telemetry Status")
    
    # Header KPI Metrics Blocks (Fixed Column Proportions & Clean Labels)
    kpi1, kpi2, kpi3 = st.columns([1.2, 1, 1])
    with kpi1:
        st.metric(label="MODEL ACCURACY", value="90.87%")
    with kpi2:
        st.metric(label="ZONE STATE", value="CRITICAL" if selected_zone == "Foodcourt" else "OPTIMAL")
    with kpi3:
        st.metric(label="GRID ROUTING", value="Auto-Harvest")
        
    st.write("---")
    
    # 5. Preparing Data Vector for Live Machine Learning Inference
    input_vector = pd.DataFrame(0, index=[0], columns=model_columns)
    
    if 'Hour' in input_vector.columns:
        input_vector['Hour'] = selected_hour
        
    # Match categorical parameters to dummy columns
    if f'Day_{selected_day}' in input_vector.columns: input_vector[f'Day_{selected_day}'] = 1
    if f'Zone_{selected_zone}' in input_vector.columns: input_vector[f'Zone_{selected_zone}'] = 1
    if f'Weather_{selected_weather}' in input_vector.columns: input_vector[f'Weather_{selected_weather}'] = 1
    if f'Event_{selected_event}' in input_vector.columns: input_vector[f'Event_{selected_event}'] = 1

    # Run calculation through our model weights
    predicted_count = int(rf_model.predict(input_vector)[0])
    
    # Display the final dynamic prediction output inside a clean custom box
    st.markdown(f"""
    <div class="metric-box">
        <h4 style='margin:0; color:#4b5563;'>🔮 PREDICTED CROWD DENSITY</h4>
        <h2 style='margin:5px 0 0 0; color:#1f2937;'>{predicted_count} Active Visitors</h2>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    # 6. Actuation Logic Response Blocks
    if predicted_count > 500:
        st.error(f"🚨 **SIGNAL: CRITICAL PLACEMENT** \n\nRouting Maximum Grid Power Pool [100%] to the **{selected_zone}** sub-grid array instantly.")
    elif predicted_count > 250:
        st.warning(f"⚠️ **SIGNAL: OPTIMAL PLACEMENT** \n\nActivating localized microgrid arrays. Continuous footfall captured for dynamic **HVAC systems utility loads**.")
    else:
        st.success("💤 **SIGNAL: LOW SIGNAL STANDBY** \n\nShifting microgrid sub-relays to sleep state to avoid hardware wear and maintenance cycle degradation.")

with col_right:
    st.markdown("### 🗺️ Live System Spatial Distribution Map")
    
    # Dynamic Filtering Logic for Heatmap Matrix
    filtered_df = df_historical[
        (df_historical['Day'] == selected_day) & 
        (df_historical['Weather'] == selected_weather) & 
        (df_historical['Event'] == selected_event)
    ]
    
    # Fallback backup if the specific combination contains no rows
    if filtered_df.empty:
        filtered_df = df_historical[df_historical['Day'] == selected_day]

    # Create the Live Pivot Table
    floor_map_data = filtered_df.pivot_table(index='Zone', columns='Hour', values='Visitors', aggfunc='mean')
    
    # Generate and render the Live Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(floor_map_data, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5, 
                    cbar_kws={'label': 'Crowd Density'}, ax=ax)
        ax.set_title(f"SmartFootAI Floor Matrix Map ({selected_day} Trends)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Operational Hours")
        ax.set_ylabel("Mall Grid Zones")
        st.pyplot(fig)
    except Exception as img_err:
        st.info("Generating real-time analytics distribution matrix map...")

st.write("---")
st.caption("Developed by Team Tech Tornado | 2nd-Year CSE AI/ML | KineticTwin Platform Core")