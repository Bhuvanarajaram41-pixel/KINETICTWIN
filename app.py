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
except Exception as e:
    st.error(f"❌ Core File Initialization Fault: {e}")
    st.info("💡 Ensure 'kinetic_rf_model.pkl', 'model_columns.pkl', and 'Mall_Efficient.csv' exist in the execution environment directory.")
    st.stop()

# 3. Sidebar UI Parameters & Feature Engineering Extraction
st.sidebar.header("🔍 Real-time Environmental Variables")

selected_hour = st.sidebar.slider("Operational Timeline (Hour)", 0, 23, 14)
selected_zone = st.sidebar.selectbox("Mall Grid Zone", df_historical['Zone'].unique())
selected_day = st.sidebar.selectbox("Day of Week Matrix", df_historical['Day'].unique())
selected_weather = st.sidebar.selectbox("External Climate Index", df_historical['Weather'].unique())
selected_event = st.sidebar.selectbox("Scheduled Structural Event", df_historical['Event'].unique())

# Preprocessing: Building the raw query dictionary
input_data = {
    'Hour': selected_hour,
    'Zone': selected_zone,
    'Day': selected_day,
    'Weather': selected_weather,
    'Event': selected_event
}
df_query = pd.DataFrame([input_data])

# Transform data to match the One-Hot Matrix Preprocessing pipeline columns
df_encoded = pd.get_dummies(df_query)
df_model_input = pd.DataFrame(0, index=[0], columns=model_columns)

for col in df_encoded.columns:
    if col in df_model_input.columns:
        df_model_input[col] = df_encoded[col].values

df_model_input['Hour'] = selected_hour

# 4. Live Model Inference Processing
predicted_density = int(rf_model.predict(df_model_input)[0])

# =========================================================
# NEW INTEGRATION: REAL-TIME KINETIC ENERGY CALCULATOR
# =========================================================
# Physics & Mathematical Formulation Engine:
# 600 N Force * 0.005 m compression * 16% mechanical-to-electrical efficiency = ~0.5 Joules/step
ENERGY_PER_STEP_JOULES = 0.5  
total_joules = predicted_density * ENERGY_PER_STEP_JOULES
total_watt_hours = total_joules / 3600.0
co2_saved_kg = total_watt_hours * 0.0004  # Standard grid mitigation coefficient

st.subheader("📊 Live Edge Grid Analytics & Clean Energy Yield")

# Render metrics beautifully inside a 4-Column structural UI layout
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(
        label="Predicted Step Density", 
        value=f"{predicted_density} steps/min", 
        delta="Peak Zone Surge" if predicted_density > 500 else "Stable Traffic"
    )

with m_col2:
    st.metric(
        label="Instantaneous Energy Yield", 
        value=f"{total_joules:.1f} J", 
        delta=f"+{ENERGY_PER_STEP_JOULES} J / Step Coefficient"
    )

with m_col3:
    st.metric(
        label="Power Conversion Output", 
        value=f"{total_watt_hours:.5f} Wh", 
        delta="Dynamic Microgrid Storage"
    )

with m_col4:
    st.metric(
        label="Net Carbon Offset (CO₂ Saved)", 
        value=f"{co2_saved_kg:.7f} kg",
        delta="Clean Power Offset",
        delta_color="inverse"
    )

st.write("---")

# 5. Dual Dashboard Control Layout Panels
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### ⚙️ Edge Device Optimization Logic")
    
    # Custom Dynamic Threshold Matrix Framework Logic
    if predicted_density > 500:
        st.error("🚀 **CRITICAL POWER INJECTION ACTIVE (>500 visitors)** \n\nRouting 100% available sub-grid microgrid storage loops into local infrastructure lines to manage sudden building load strains.")
    elif 250 <= predicted_density <= 500:
        st.warning("🔄 **OPTIMAL LOAD BALANCING ENABLED (250-500 visitors)** \n\nCapturing stable kinetic generation arrays to offset local smart-lighting and building automation parameters seamlessly.")
    else:
        st.success("💤 **LOW-SIGNAL STANDBY CIRCUIT ACTIVE (<250 visitors)** \n\nShifting microgrid sub-relays to sleep state to avoid hardware wear and maintenance cycle degradation.")

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
    # Generate and render the Live Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # FIXED: Removed the invalid escape slashes and aligned the string quotes cleanly
        sns.heatmap(
            floor_map_data, 
            annot=True, 
            fmt=".0f", 
            cmap="YlOrRd", 
            linewidths=0.5, 
            cbar_kws={'label': 'Crowd Density'}, 
            ax=ax
        )
        
        ax.set_title(f"SmartFootAI Floor Matrix Map ({selected_day} Trends)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Operational Hours")
        ax.set_ylabel("Mall Grid Zones")
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Spatial layout matrix display temporarily paused: {e}")

st.write("---")
st.caption("KineticTwin Dashboard Core Pipeline Framework | Built on Python, Joblib, & Streamlit Assets")
