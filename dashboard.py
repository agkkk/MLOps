import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. Page Setup ---
st.set_page_config(page_title="ABC Retail ML Dashboard", layout="wide")
st.title("📊 ABC Retail Group - Intelligent ML Dashboard")
st.markdown("Automated insights, anomaly detection, and forecasting powered by Machine Learning.")

# --- 2. Load ML-Enriched Data ---
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df_anomalies = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ml_anomalies.csv'))
    df_forecast = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ml_forecast.csv'))
    df_clusters = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ml_clusters.csv'))
    return df_anomalies, df_forecast, df_clusters

df_anomalies, df_forecast, df_clusters = load_data()

# --- 3. Create Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🚨 Operational Anomalies (Isolation Forest)", 
    "📈 Revenue Forecast (Random Forest)", 
    "📦 Inventory Segmentation (K-Means)"
])

# ==========================================
# TAB 1: ANOMALY DETECTION
# ==========================================
with tab1:
    st.header("Operational Failure Detection")
    st.markdown("Highlights days where massive logistics delays caused critical spikes in customer support tickets.")
    
    # KPIs
    total_days = len(df_anomalies)
    failed_days = df_anomalies['is_anomaly'].sum()
    st.metric("Total Days Analyzed", total_days)
    st.metric("Critical Failures Detected", failed_days, delta_color="inverse")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_anomalies, x='delayed_shipments', y='tickets_opened', 
        hue='is_anomaly', palette={False: 'gray', True: 'red'}, s=100, ax=ax
    )
    ax.set_title("Delayed Shipments vs Customer Support Tickets")
    st.pyplot(fig)

    # Actionable Data Table
    st.subheader("⚠️ Action Required: Flagged Dates")
    flagged_data = df_anomalies[df_anomalies['is_anomaly'] == True]
    st.dataframe(flagged_data[['date', 'delayed_shipments', 'tickets_opened']].sort_values('delayed_shipments', ascending=False))

# ==========================================
# TAB 2: REVENUE FORECASTING
# ==========================================
with tab2:
    st.header("Revenue & Financial Forecasting")
    st.markdown("Comparing actual historical revenue against the Random Forest model's predictions.")
    
    # Plot
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    df_forecast_sorted = df_forecast.sort_values('date').head(45) # Show first 45 days for clarity
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_forecast_sorted, x='date', y='revenue', label='Actual Revenue', marker='o', ax=ax2)
    sns.lineplot(data=df_forecast_sorted, x='date', y='predicted_revenue', label='AI Predicted Revenue', linestyle='--', ax=ax2)
    ax2.set_title("Actual vs Predicted Revenue (45-Day View)")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.subheader("Data Overview")
    st.dataframe(df_forecast[['date', 'footfall', 'marketing_spend', 'revenue', 'predicted_revenue']].head(10))

# ==========================================
# TAB 3: INVENTORY SEGMENTATION
# ==========================================
with tab3:
    st.header("Inventory Movement Categories")
    st.markdown("Unsupervised clustering grouping business days by stock movement and discounts applied.")
    
    # Plot
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_clusters, x='stock_level', y='quantity', 
        hue='business_category', palette='viridis', s=80, ax=ax3
    )
    ax3.set_title("Stock Levels vs Quantity Sold by Category")
    st.pyplot(fig3)
    
    st.subheader("Segment Breakdown")
    st.dataframe(df_clusters[['date', 'category', 'stock_level', 'quantity', 'discount', 'business_category']].head(15))