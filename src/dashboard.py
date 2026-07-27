import streamlit as st
import pandas as pd
import subprocess
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sqlite3
import requests

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Anomalies', 'Revenue Forecast', 'Inventory Clusters', 'LR Trendline', 'Live API Inference'])

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
# TAB 4: LR TRENDLINE
# ==========================================
with tab4:
    st.header("📈 Footfall vs. Revenue (Linear Regression)")
    st.markdown("This model calculates a linear trendline to prove exactly how an increase in footfall mathematically impacts total daily revenue.")
    
    try:
        df_lr = pd.read_csv('src/data/ml_forecast.csv')
        # We sort by footfall so the trendline draws smoothly from left to right
        df_plot = df_lr[['footfall', 'revenue', 'lr_predicted_revenue']].sort_values(by='footfall').set_index('footfall')
        st.line_chart(df_plot)
    except FileNotFoundError:
        st.warning("Data not found. Please run the scheduler to generate the Linear Regression data.")
# ==========================================
# TAB 5: LIVE API INFERENCE & CRUD
# ==========================================
with tab5:
    st.header("⚡ Live Control Center")
    
    # ==========================================
    # SECTION A: LIVE API PREDICTION
    # ==========================================
    st.subheader("1. Live API Inference (Revenue Forecast)")
    st.markdown("Test the active Revenue Pickle model in real-time without altering the database.")
    
    with st.form("api_prediction_form"):
        st.markdown("Enter simulated daily metrics to get a live revenue prediction via FastAPI.")
        col1, col2 = st.columns(2)
        with col1:
            api_footfall = st.number_input("Footfall (Simulated)", value=350)
            api_marketing = st.number_input("Marketing Spend", value=500.0)
        with col2:
            api_operating = st.number_input("Operating Cost", value=2000.0)
            
        submit_api = st.form_submit_button("Predict Revenue via FastAPI")
        
    if submit_api:
        # Perfectly matching the schema shown in your Swagger UI image
        payload = {
            "footfall": api_footfall,
            "marketing_spend": api_marketing,
            "operating_cost": api_operating
        }
        try:
            # Pointing exactly to the /predict/revenue endpoint
            response = requests.post("http://127.0.0.1:8000/predict/revenue", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"**FastAPI Response:** Server successfully processed the data using the Pickle model.")
                st.json(result)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error("Connection Failed. Make sure your FastAPI server is running via `./run.sh`!")

    st.divider()
    
    # ==========================================
    # SECTION B: CRUD & RETRAINING
    # ==========================================
    st.subheader("2. Master Data Management (CRUD)")
    st.markdown("Inject outliers directly into the SQLite database and force the pipeline to retrain.")
    
    db_path = 'src/data/retail.db'
    operation = st.radio("Select Database Operation:", ["Create (Insert Sales Record)", "Delete (Remove Sales Record)"])
    
    with st.form("crud_form"):
        if "Create" in operation:
            st.markdown("💡 *Use an existing date (e.g., 2026-06-29) so it merges perfectly with the Finance data.*")
            date_input = st.text_input("Date (YYYY-MM-DD)", value="2026-06-29")
            cat_input = st.selectbox("Category", ["Electronics", "Clothing", "Grocery"], key="crud_cat")
            
            col1, col2 = st.columns(2)
            with col1:
                footfall_input = st.number_input("Footfall", value=15000)
            with col2:
                revenue_input = st.number_input("Revenue", value=300000.0)
                
            submit_crud = st.form_submit_button("Insert into Database & Retrain")
            
        elif "Delete" in operation:
            del_date = st.text_input("Date to Delete (YYYY-MM-DD)", value="2026-06-29")
            submit_crud = st.form_submit_button("Delete from Database & Retrain")

    if submit_crud:
        with st.spinner(f"Executing {operation.split()[0]} in SQLite, syncing CSVs, and retraining AI..."):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if "Create" in operation:
                cursor.execute('''
                    INSERT INTO sales (date, category, footfall, quantity, revenue, discount)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (date_input, cat_input, footfall_input, int(footfall_input * 0.2), revenue_input, 0))
            elif "Delete" in operation:
                cursor.execute('DELETE FROM sales WHERE date = ?', (del_date,))
                
            conn.commit()
            
            df_updated_sales = pd.read_sql_query("SELECT * FROM sales", conn)
            df_updated_sales.to_csv('src/data/sales_data.csv', index=False)
            conn.close()
            
            subprocess.run(["python", "src/ml/train_models.py"])
            
            st.success("Database operation complete! Pickle models updated. Refreshing UI...")
            st.rerun()