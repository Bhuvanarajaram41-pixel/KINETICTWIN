# -*- coding: utf-8 -*-
"""
KineticTwin: The SmartfootAI
Production Pipeline Script (Day 1)
"""

import os
import pandas as pd
import numpy as np
import joblib  # Used to save and load model binaries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("⚡ Starting KineticTwin Data Pipeline...")

# ==========================================
# 1. SYNTHETIC REALISTIC DATA GENERATION
# ==========================================
np.random.seed(42)
hours = [8, 9, 10, 11, 12, 13, 14, 15]
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
zones = ['Entrance', 'Foodcourt', 'Shopping area', 'Gaming Zone', 'Cinema', 'Parking']
events = ['No', 'Yes']
weathers = ['Sunny', 'Cloudy', 'Rainy']

data_rows = []
for _ in range(300):
    hour = np.random.choice(hours)
    day = np.random.choice(days)
    zone = np.random.choice(zones)
    event = np.random.choice(events, p=[0.8, 0.2])
    weather = np.random.choice(weathers, p=[0.6, 0.3, 0.1])

    base_visitors = 100
    if day in ['Saturday', 'Sunday']: base_visitors += 250
    elif day == 'Friday': base_visitors += 100

    if zone == 'Entrance': base_visitors += 150
    elif zone == 'Foodcourt' and hour in [12, 13]: base_visitors += 300
    elif zone == 'Shopping area': base_visitors += 120
    elif zone == 'Cinema' and hour in [14, 15]: base_visitors += 180
    elif zone == 'Parking': base_visitors += 80

    if event == 'Yes': base_visitors += 400

    if weather == 'Rainy' and zone in ['Cinema', 'Foodcourt', 'Gaming Zone']: base_visitors += 150

    noise = np.random.normal(0, 25)
    final_visitors = int(max(10, base_visitors + noise))
    data_rows.append([hour, day, zone, event, weather, final_visitors])

df = pd.DataFrame(data_rows, columns=['Hour', 'Day', 'Zone', 'Event', 'Weather', 'Visitors'])
df.to_csv('Mall_Efficient.csv', index=False)
print("✅ Dataset generated successfully and saved as 'Mall_Efficient.csv'.")

# ==========================================
# 2. MODEL OPTIMIZATION & TRAINING
# ==========================================
# Applying One-Hot Encoding correctly to eliminate the Ordinal Fallacy
df_encoded = pd.get_dummies(df, columns=['Day', 'Zone', 'Weather', 'Event'], drop_first=True)

X = df_encoded.drop(columns=['Visitors'])
y = df_encoded['Visitors']

# Saving the exact feature names/order for the website backend processing later
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("✅ Saved operational column structures as 'model_columns.pkl'.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training our optimized framework
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Validating scores
y_pred = rf_model.predict(X_test)
accuracy = r2_score(y_test, y_pred) * 100
mae = mean_absolute_error(y_test, y_pred)

print(f"🎯 Random Forest R² Score Accuracy: {accuracy:.2f}%")
print(f"📉 Mean Absolute Error: {mae:.2f} visitors")

# ==========================================
# 3. SERIALIZE & EXPORT TRAINED WEIGHTS
# ==========================================
# This saves the trained brain as a file so the website can load it in 0.01 seconds
joblib.dump(rf_model, 'kinetic_rf_model.pkl')
print("📦 Successfully exported optimized model as 'kinetic_rf_model.pkl'.")

# ==========================================
# 4. EXPORT ANALYSIS VISUALIZATIONS
# ==========================================
plt.style.use('dark_background')  # Gives it a modern, clean dashboard feel
floor_map_data = df.pivot_table(index='Zone', columns='Hour', values='Visitors', aggfunc='mean')
plt.figure(figsize=(12, 6))

sns.heatmap(floor_map_data, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=1, cbar_kws={'label': 'Crowd Density'})
plt.title("SmartFootAI: Mall Spatial Footfall Floor Map (Average Density)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Operational Hours", fontsize=11)
plt.ylabel("Mall Grid Zones", fontsize=11)

plt.savefig('mall_floor_map.png', dpi=300, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("🖼️ Matrix heatmap visual pre-rendered and saved as 'mall_floor_map.png'.")
