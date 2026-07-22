import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests # <--- NEW: To talk to the FastAPI server

# --- 1. Page Setup ---
st.set_page_config(page_title="ABC Retail ML Dashboard", layout="wide")
st.title("📊 ABC Retail Group - Intelligent ML Dashboard")
st.markdown("Automated insights, anomaly detection, and forecasting powered by Machine Learning.")

# --- 2. Load ML-Enriched Data (Batch Processing) ---
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df_anomalies = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ml_anomalies.csv'))
    df_forecast = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ml_forecast.csv'))
    df_clusters = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ml_clusters.csv'))
    return df_anomalies, df_forecast, df_clusters

df_anomalies, df_forecast, df_clusters = load_data()

# --- 3. Create Dashboard Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([  # <--- NEW: Added a 4th tab
    "🚨 Operational Anomalies", 
    "📈 Revenue Forecast", 
    "📦 Inventory Segmentation",
    "⚡ Live API Inference"      # <--- NEW: Real-time serving tab
])

# ==========================================
# TAB 1: ANOMALY DETECTION (Batch)
# ==========================================
with tab1:
    st.header("Operational Failure Detection (Isolation Forest)")
    total_days = len(df_anomalies)
    failed_days = df_anomalies['is_anomaly'].sum()
    st.metric("Total Days Analyzed", total_days)
    st.metric("Critical Failures Detected", failed_days, delta_color="inverse")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_anomalies, x='delayed_shipments', y='tickets_opened', 
        hue='is_anomaly', palette={False: 'gray', True: 'red'}, s=100, ax=ax
    )
    st.pyplot(fig)

# ==========================================
# TAB 2: REVENUE FORECASTING (Batch)
# ==========================================
with tab2:
    st.header("Revenue Forecasting (Random Forest)")
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    df_forecast_sorted = df_forecast.sort_values('date').head(45)
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_forecast_sorted, x='date', y='revenue', label='Actual Revenue', marker='o', ax=ax2)
    sns.lineplot(data=df_forecast_sorted, x='date', y='predicted_revenue', label='AI Predicted Revenue', linestyle='--', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ==========================================
# TAB 3: INVENTORY SEGMENTATION (Batch)
# ==========================================
with tab3:
    st.header("Inventory Movement Categories (K-Means)")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_clusters, x='stock_level', y='quantity', 
        hue='business_category', palette='viridis', s=80, ax=ax3
    )
    st.pyplot(fig3)

# ==========================================
# TAB 4: LIVE API INFERENCE (Real-Time)
# ==========================================
with tab4:
    st.header("⚡ Real-Time Feature Serving (FastAPI Integration)")
    st.markdown("This tab sends a live JSON payload to the FastAPI server and retrieves an instant prediction.")
    
    st.subheader("Predict Daily Revenue")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        footfall = st.number_input("Footfall", min_value=0, value=500)
    with col2:
        marketing = st.number_input("Marketing Spend ($)", min_value=0, value=1200)
    with col3:
        operating = st.number_input("Operating Cost ($)", min_value=0, value=3000)
        
    if st.button("Generate Live Prediction"):
        # The exact payload FastAPI is expecting
        payload = {
            "footfall": footfall,
            "marketing_spend": marketing,
            "operating_cost": operating
        }
        
        try:
            # Send the data to your local FastAPI server
            response = requests.post("http://127.0.0.1:8000/predict/revenue", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"📈 The AI predicts a revenue of: **${result['predicted_revenue']}**")
            else:
                st.error(f"API Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Failed. Make sure the FastAPI server is running on port 8000!")