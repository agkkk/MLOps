import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import sqlite3
import subprocess
import requests
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit_authenticator as stauth

# ============================================================
# PAGE CONFIG & CSS
# ============================================================
st.set_page_config(
    page_title="ABC Retail - Intelligent ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #151b2b;
        border: 1px solid #2a344a;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* Divider */
    hr {
        border-color: #2a344a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Plotly charts wrapper */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2a344a;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTHENTICATION
# ============================================================
credentials = {
    "usernames": {
        "admin": {
            "name": "Admin User",
            "password": stauth.Hasher.hash("admin123"),
        },
        "hod": {
            "name": "HOD Faculty",
            "password": stauth.Hasher.hash("hod2026"),
        },
        "abc_retail": {
            "name": "ABC Retail Manager",
            "password": stauth.Hasher.hash("retail@2026"),
        }
    }
}

authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="abc_retail_auth",
    cookie_key="abc_retail_secret_key_2026",
    cookie_expiry_days=7,
)

authenticator.login()

if st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")
    st.info("**Demo Credentials:** Username: `admin` | Password: `admin123`")
    st.stop()
elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
    st.stop()

# ---- Authenticated content starts here ----
# Sidebar
with st.sidebar:
    st.markdown(f"### 👋 Welcome, *{st.session_state.get('name', 'User')}*")
    authenticator.logout("Logout", "sidebar")
    st.divider()
    st.markdown("### 📊 ABC Retail MLOps")
    st.markdown("Intelligent Dashboard Generator")
    st.markdown("---")
    st.markdown("**Data Source:** Global Superstore")
    st.markdown("**Models:** 15+ trained & compared")
    st.markdown("**Pipeline:** Fully automated")

# ============================================================
# HEADER
# ============================================================
st.title("📊 ABC Retail Group — Intelligent ML Dashboard")
st.caption("Automated insights, anomaly detection, and forecasting powered by Machine Learning • Real-world data from Global Superstore Dataset")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=60)
def load_ml_data():
    data = {}
    for fname in ['ml_anomalies', 'ml_forecast', 'ml_clusters']:
        path = os.path.join(DATA_DIR, f'{fname}.csv')
        if os.path.exists(path):
            data[fname] = pd.read_csv(path)
        else:
            data[fname] = pd.DataFrame()
    return data

@st.cache_data(ttl=60)
def load_preprocessing_report():
    path = os.path.join(DATA_DIR, 'preprocessing_report.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data(ttl=60)
def load_model_comparison():
    path = os.path.join(DATA_DIR, 'model_comparison.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data(ttl=60)
def load_raw_sample():
    path = os.path.join(DATA_DIR, 'raw_sample.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

ml_data = load_ml_data()
df_anomalies = ml_data['ml_anomalies']
df_forecast = ml_data['ml_forecast']
df_clusters = ml_data['ml_clusters']
preprocess_report = load_preprocessing_report()
comparison_report = load_model_comparison()

# ============================================================
# TABS
# ============================================================
tab_pipeline, tab_preprocess, tab_comparison, tab_anomaly, tab_forecast, tab_cluster, tab_lr, tab_live = st.tabs([
    '🔄 Pipeline Overview',
    '🧹 Data Preprocessing',
    '📈 Model Comparison',
    '🚨 Anomaly Detection',
    '💰 Revenue Forecast',
    '📦 Inventory Clusters',
    '📉 LR Trendline',
    '⚡ Live API & CRUD',
])

# ============================================================
# TAB 1: PIPELINE OVERVIEW (Phase-wise)
# ============================================================
with tab_pipeline:
    st.header("🔄 MLOps Pipeline — Phase-wise Architecture")
    st.markdown("This tab shows the complete end-to-end pipeline, from raw data acquisition to model deployment and monitoring.")

    # Phase diagram using columns
    phases = [
        {"icon": "📥", "title": "Phase 1: Data Acquisition", "color": "#4A90D9",
         "desc": "Real-world retail data sourced from the **Global Superstore Dataset** (1000+ transactions across 147 countries). Data includes sales, shipping, customer segments, and product categories.",
         "tech": "Source: Kaggle / Public Dataset • Format: CSV"},
        {"icon": "🧹", "title": "Phase 2: Data Preprocessing", "color": "#7B68EE",
         "desc": "8-step cleaning pipeline: duplicate removal, null imputation (median/mode), date parsing, outlier detection (IQR), feature engineering (profit margin, shipping delays, revenue/unit).",
         "tech": "Tools: Pandas, NumPy, Custom Pipeline"},
        {"icon": "🏗️", "title": "Phase 3: Data Transformation", "color": "#E67E22",
         "desc": "Cleaned data transformed into **5 departmental tables**: Sales, Inventory, Finance, Support, Logistics. Stored in SQLite + CSV for dual-access.",
         "tech": "Tools: SQLite3, Pandas"},
        {"icon": "🧠", "title": "Phase 4: Multi-Model Training", "color": "#27AE60",
         "desc": "**15+ ML models** trained across 3 tasks: Revenue Forecasting (6 models), Anomaly Detection (3 models), Inventory Clustering (6 models). Each model compared on standard metrics.",
         "tech": "Tools: Scikit-learn, MLflow"},
        {"icon": "📊", "title": "Phase 5: Model Comparison & Selection", "color": "#E74C3C",
         "desc": "Models ranked by performance metrics (R², MAE, RMSE, Silhouette Score). **Best model auto-selected** for deployment. Full comparison saved for audit trail.",
         "tech": "Metrics: R², MAE, RMSE, Silhouette"},
        {"icon": "🚀", "title": "Phase 6: Deployment & Serving", "color": "#1ABC9C",
         "desc": "Best models deployed via **FastAPI** REST endpoints for real-time inference. Streamlit dashboard for interactive visualization. Docker & K8s configs for containerized deployment.",
         "tech": "Tools: FastAPI, Streamlit, Docker, K8s"},
        {"icon": "🔄", "title": "Phase 7: Automation & Monitoring", "color": "#9B59B6",
         "desc": "Automated scheduler triggers data generation + retraining cycles. MLflow tracks experiment history, parameters, and model artifacts for version control.",
         "tech": "Tools: Python Schedule, MLflow, Git"},
    ]

    for i, phase in enumerate(phases):
        with st.container():
            col_icon, col_content = st.columns([1, 11])
            with col_icon:
                st.markdown(f"### {phase['icon']}")
            with col_content:
                st.markdown(f"**{phase['title']}**")
                st.markdown(phase['desc'])
                st.caption(phase['tech'])
            if i < len(phases) - 1:
                st.markdown("<div style='text-align:center; font-size:24px; color:#888;'>⬇️</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("🏛️ Architecture Components")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frontend", "Streamlit")
        st.caption("Interactive dashboard with auth")
    with col2:
        st.metric("Backend", "FastAPI")
        st.caption("REST API for model serving")
    with col3:
        st.metric("Tracking", "MLflow")
        st.caption("Experiment & model versioning")
    with col4:
        st.metric("Database", "SQLite")
        st.caption("Persistent data storage")


# ============================================================
# TAB 2: DATA PREPROCESSING
# ============================================================
with tab_preprocess:
    st.header("🧹 Data Preprocessing Pipeline")
    st.markdown("Showing the transformation from **raw real-world data** to **clean, analysis-ready departmental tables**.")

    if preprocess_report:
        before = preprocess_report['before']
        after = preprocess_report['after']

        # Before vs After metrics
        st.subheader("📊 Before vs After — Data Quality Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{after['total_rows']:,}",
                      delta=f"{after['total_rows'] - before['total_rows']:,} removed" if after['total_rows'] != before['total_rows'] else "No change")
        with col2:
            st.metric("Columns", f"{after['total_columns']}",
                      delta=f"+{after['total_columns'] - before['total_columns']} engineered")
        with col3:
            st.metric("Missing Values (Before)", f"{before['total_missing_values']:,}")
        with col4:
            st.metric("Missing Values (After)", f"{after['total_missing_values']:,}",
                      delta=f"-{before['total_missing_values'] - after['total_missing_values']:,}", delta_color="inverse")

        # Show preprocessing steps
        st.subheader("🔧 Preprocessing Steps Executed")
        steps = preprocess_report.get('preprocessing_steps', [])
        for step in steps:
            st.markdown(f"**Step {step['step']}: {step['name']}** — {step['description']}")

        # Outlier summary
        st.subheader("📊 Outlier Detection Summary (IQR Method)")
        outliers = preprocess_report.get('outlier_summary', {})
        if outliers:
            outlier_df = pd.DataFrame([
                {
                    'Column': col,
                    'Outliers Found': info['count'],
                    'Percentage': f"{info['percentage']}%",
                    'Lower Bound': info['lower_bound'],
                    'Upper Bound': info['upper_bound'],
                }
                for col, info in outliers.items()
            ])
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)

            # Outlier bar chart
            fig_outlier = px.bar(
                outlier_df, x='Column', y='Outliers Found',
                title='Outliers Detected per Column',
                color='Outliers Found',
                color_continuous_scale='Reds',
            )
            fig_outlier.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_outlier, use_container_width=True)

        # Show raw vs cleaned data samples
        st.subheader("👀 Raw Data Sample (Before Cleaning)")
        raw_sample = load_raw_sample()
        if raw_sample is not None:
            st.dataframe(raw_sample.head(10), use_container_width=True, hide_index=True)
        
        st.subheader("✅ Cleaned Departmental Data (After Transformation)")
        tab_s, tab_i, tab_f, tab_su, tab_l = st.tabs(['Sales', 'Inventory', 'Finance', 'Support', 'Logistics'])
        for tab_inner, name in zip([tab_s, tab_i, tab_f, tab_su, tab_l],
                                   ['sales', 'inventory', 'finance', 'support', 'logistics']):
            with tab_inner:
                path = os.path.join(DATA_DIR, f'{name}_data.csv')
                if os.path.exists(path):
                    st.dataframe(pd.read_csv(path).head(15), use_container_width=True, hide_index=True)

        # Columns with nulls before
        if before.get('columns_with_nulls'):
            st.subheader("🕳️ Columns with Missing Values (Before Cleaning)")
            null_data = pd.DataFrame([
                {'Column': col, 'Missing Count': info['count'], 'Missing %': f"{info['percentage']}%"}
                for col, info in before['columns_with_nulls'].items()
            ])
            st.dataframe(null_data, use_container_width=True, hide_index=True)

    else:
        st.warning("⚠️ Preprocessing report not found. Run the preprocessing pipeline first.")
        if st.button("▶️ Run Preprocessing Pipeline"):
            with st.spinner("Running data preprocessing..."):
                subprocess.run([sys.executable, os.path.join(BASE_DIR, "data_preprocessing.py")], check=True)
                st.success("Done! Refreshing...")
                st.rerun()


# ============================================================
# TAB 3: MODEL COMPARISON
# ============================================================
with tab_comparison:
    st.header("📈 Model Comparison — Why We Chose These Models")
    st.markdown("For each ML task, **multiple models were trained and evaluated** on standard metrics. The best model was automatically selected for deployment.")

    if comparison_report:
        results = comparison_report.get('results', {})
        best_models = comparison_report.get('best_models', {})

        # --- Revenue Forecasting Comparison ---
        st.subheader("💰 Task 1: Revenue Forecasting")
        if 'revenue_forecasting' in results:
            rev_df = pd.DataFrame(results['revenue_forecasting'])
            best_rev = best_models.get('revenue_forecasting', '')
            
            st.markdown(f"**🏆 Best Model Selected:** `{best_rev}`")

            # Metrics table
            st.dataframe(
                rev_df.style.highlight_max(subset=['R2_Score', 'CV_R2_Mean'], color='#2ecc71')
                           .highlight_min(subset=['MAE', 'RMSE'], color='#2ecc71'),
                use_container_width=True, hide_index=True
            )

            # R² comparison bar chart
            col1, col2 = st.columns(2)
            with col1:
                fig_r2 = px.bar(
                    rev_df, x='model', y='R2_Score',
                    title='R² Score Comparison (Higher is Better)',
                    color='R2_Score', color_continuous_scale='Greens',
                    text='R2_Score'
                )
                fig_r2.update_layout(template='plotly_dark', height=400)
                fig_r2.update_traces(textposition='outside')
                st.plotly_chart(fig_r2, use_container_width=True)

            with col2:
                fig_mae = px.bar(
                    rev_df, x='model', y='MAE',
                    title='MAE Comparison (Lower is Better)',
                    color='MAE', color_continuous_scale='Reds_r',
                    text='MAE'
                )
                fig_mae.update_layout(template='plotly_dark', height=400)
                fig_mae.update_traces(textposition='outside')
                st.plotly_chart(fig_mae, use_container_width=True)

        st.divider()

        # --- Anomaly Detection Comparison ---
        st.subheader("🚨 Task 2: Anomaly Detection")
        if 'anomaly_detection' in results:
            anom_df = pd.DataFrame(results['anomaly_detection'])
            best_anom = best_models.get('anomaly_detection', '')
            
            st.markdown(f"**🏆 Selected Model:** `{best_anom}` (industry standard for unsupervised anomaly detection)")
            st.dataframe(anom_df, use_container_width=True, hide_index=True)

            fig_anom = px.bar(
                anom_df, x='model', y='Anomalies_Detected',
                title='Anomalies Detected by Each Model',
                color='Anomaly_Rate_%', color_continuous_scale='YlOrRd',
                text='Anomalies_Detected'
            )
            fig_anom.update_layout(template='plotly_dark', height=400)
            fig_anom.update_traces(textposition='outside')
            st.plotly_chart(fig_anom, use_container_width=True)

        st.divider()

        # --- Clustering Comparison ---
        st.subheader("📦 Task 3: Inventory Clustering")
        if 'inventory_clustering' in results:
            clust_df = pd.DataFrame(results['inventory_clustering'])
            best_clust = best_models.get('inventory_clustering', '')
            
            st.markdown(f"**🏆 Best Model Selected:** `{best_clust}`")
            st.dataframe(
                clust_df.style.highlight_max(subset=['Silhouette_Score'], color='#2ecc71'),
                use_container_width=True, hide_index=True
            )

            fig_sil = px.bar(
                clust_df, x='model', y='Silhouette_Score',
                title='Silhouette Score Comparison (Higher is Better)',
                color='Silhouette_Score', color_continuous_scale='Viridis',
                text='Silhouette_Score'
            )
            fig_sil.update_layout(template='plotly_dark', height=400)
            fig_sil.update_traces(textposition='outside')
            st.plotly_chart(fig_sil, use_container_width=True)

    else:
        st.warning("⚠️ Model comparison data not found. Run the ML training pipeline first.")
        if st.button("▶️ Run ML Training Pipeline"):
            with st.spinner("Training all models..."):
                subprocess.run([sys.executable, os.path.join(BASE_DIR, "ml", "train_models.py")], check=True)
                st.success("Done! Refreshing...")
                st.rerun()


# ============================================================
# TAB 4: ANOMALY DETECTION
# ============================================================
with tab_anomaly:
    st.header("🚨 Operational Anomaly Detection (Isolation Forest)")

    if not df_anomalies.empty:
        total_days = len(df_anomalies)
        failed_days = df_anomalies['is_anomaly'].sum() if 'is_anomaly' in df_anomalies.columns else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days Analyzed", total_days)
        with col2:
            st.metric("Critical Failures Detected", int(failed_days), delta_color="inverse")
        with col3:
            st.metric("Anomaly Rate", f"{(failed_days/total_days*100):.1f}%")

        fig_anom = px.scatter(
            df_anomalies, x='delayed_shipments', y='tickets_opened',
            color='is_anomaly',
            color_discrete_map={True: '#e74c3c', False: '#95a5a6'},
            title='Delayed Shipments vs Support Tickets — Anomaly Map',
            labels={'is_anomaly': 'Is Anomaly'},
            hover_data=['date'] if 'date' in df_anomalies.columns else None,
        )
        fig_anom.update_layout(template='plotly_dark', height=500)
        fig_anom.update_traces(marker=dict(size=10, opacity=0.7))
        st.plotly_chart(fig_anom, use_container_width=True)

        # Show anomaly records
        with st.expander("🔍 View Detected Anomaly Records"):
            anomaly_rows = df_anomalies[df_anomalies['is_anomaly'] == True]
            if not anomaly_rows.empty:
                st.dataframe(anomaly_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No anomalies detected in current data.")
    else:
        st.warning("No anomaly data found. Run the pipeline first.")


# ============================================================
# TAB 5: REVENUE FORECAST
# ============================================================
with tab_forecast:
    st.header("💰 Revenue Forecasting (Best Model)")

    if not df_forecast.empty and 'date' in df_forecast.columns:
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])
        df_sorted = df_forecast.sort_values('date')

        # Metrics
        if 'predicted_revenue' in df_sorted.columns:
            from sklearn.metrics import mean_absolute_error as calc_mae, r2_score as calc_r2
            actual = df_sorted['revenue']
            predicted = df_sorted['predicted_revenue']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"${calc_mae(actual, predicted):,.2f}")
            with col2:
                st.metric("R² Score", f"{calc_r2(actual, predicted):.4f}")
            with col3:
                st.metric("Data Points", len(df_sorted))

        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=df_sorted['date'], y=df_sorted['revenue'],
            mode='lines+markers', name='Actual Revenue',
            line=dict(color='#3498db', width=2), marker=dict(size=4)
        ))
        if 'predicted_revenue' in df_sorted.columns:
            fig_rev.add_trace(go.Scatter(
                x=df_sorted['date'], y=df_sorted['predicted_revenue'],
                mode='lines', name='Predicted Revenue',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
        fig_rev.update_layout(
            title='Actual vs Predicted Daily Revenue',
            template='plotly_dark', height=500,
            xaxis_title='Date', yaxis_title='Revenue ($)'
        )
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.warning("No forecast data found. Run the pipeline first.")


# ============================================================
# TAB 6: INVENTORY CLUSTERS
# ============================================================
with tab_cluster:
    st.header("📦 Inventory Segmentation (Best Clustering Model)")

    if not df_clusters.empty:
        if 'business_category' in df_clusters.columns:
            # Distribution
            cat_counts = df_clusters['business_category'].value_counts()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Cluster Distribution")
                fig_pie = px.pie(
                    values=cat_counts.values, names=cat_counts.index,
                    title='Business Category Breakdown',
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_pie.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_clust = px.scatter(
                    df_clusters, x='stock_level', y='quantity',
                    color='business_category',
                    title='Inventory Clusters — Stock Level vs Quantity',
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    hover_data=['category'] if 'category' in df_clusters.columns else None,
                )
                fig_clust.update_layout(template='plotly_dark', height=400)
                fig_clust.update_traces(marker=dict(size=6, opacity=0.7))
                st.plotly_chart(fig_clust, use_container_width=True)
        else:
            st.warning("Cluster labels not found in data.")
    else:
        st.warning("No cluster data found. Run the pipeline first.")


# ============================================================
# TAB 7: LR TRENDLINE
# ============================================================
with tab_lr:
    st.header("📉 Footfall vs Revenue — Linear Regression Trendline")
    st.markdown("This model calculates a linear trendline to show how footfall mathematically impacts total daily revenue.")

    if not df_forecast.empty and 'lr_predicted_revenue' in df_forecast.columns:
        df_lr = df_forecast[['footfall', 'revenue', 'lr_predicted_revenue']].sort_values('footfall')

        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(
            x=df_lr['footfall'], y=df_lr['revenue'],
            mode='markers', name='Actual Data Points',
            marker=dict(color='#3498db', size=8, opacity=0.6)
        ))
        fig_lr.add_trace(go.Scatter(
            x=df_lr['footfall'], y=df_lr['lr_predicted_revenue'],
            mode='lines', name='Linear Regression Trendline',
            line=dict(color='#e74c3c', width=3)
        ))
        fig_lr.update_layout(
            title='Footfall Impact on Revenue (Linear Regression)',
            template='plotly_dark', height=500,
            xaxis_title='Daily Footfall', yaxis_title='Revenue ($)'
        )
        st.plotly_chart(fig_lr, use_container_width=True)
    else:
        st.warning("Linear regression data not found. Run the pipeline first.")


# ============================================================
# TAB 8: LIVE API & CRUD
# ============================================================
with tab_live:
    st.header("⚡ Live Control Center")

    # --- LIVE API ---
    st.subheader("1. Live API Inference (Revenue Forecast)")
    st.markdown("Test the deployed model in real-time via the FastAPI backend.")

    with st.form("api_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            api_footfall = st.number_input("Footfall (Simulated)", value=350)
            api_marketing = st.number_input("Marketing Spend", value=500.0)
        with col2:
            api_operating = st.number_input("Operating Cost", value=2000.0)

        submit_api = st.form_submit_button("🚀 Predict Revenue via FastAPI")

    if submit_api:
        payload = {
            "footfall": api_footfall,
            "marketing_spend": api_marketing,
            "operating_cost": api_operating
        }
        try:
            response = requests.post("http://127.0.0.1:8000/predict/revenue", json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                st.success(f"**FastAPI Response:** Model processed the data successfully.")
                st.json(result)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception:
            st.error("⚠️ Connection Failed. Make sure your FastAPI server is running via `./run.sh`!")

    st.divider()

    # --- CRUD ---
    st.subheader("2. Master Data Management (CRUD)")
    st.markdown("Inject records into the database and retrain all models.")

    db_path = os.path.join(DATA_DIR, 'retail.db')
    operation = st.radio("Select Operation:", ["Create (Insert Sales Record)", "Delete (Remove Sales Record)"])

    with st.form("crud_form"):
        if "Create" in operation:
            st.markdown("💡 *Enter a date that exists in the dataset for proper merging.*")
            date_input = st.text_input("Date (YYYY-MM-DD)", value="2014-11-11")
            cat_input = st.selectbox("Category", ["Technology", "Furniture", "Office Supplies"])

            col1, col2 = st.columns(2)
            with col1:
                footfall_input = st.number_input("Footfall (Orders)", value=50)
            with col2:
                revenue_input = st.number_input("Revenue", value=5000.0)

            submit_crud = st.form_submit_button("💾 Insert & Retrain")
        elif "Delete" in operation:
            del_date = st.text_input("Date to Delete (YYYY-MM-DD)", value="2014-11-11")
            submit_crud = st.form_submit_button("🗑️ Delete & Retrain")

    if submit_crud:
        with st.spinner(f"Executing {operation.split()[0]} + retraining all models..."):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                if "Create" in operation:
                    cursor.execute('''
                        INSERT INTO sales (date, category, footfall, quantity, revenue, discount)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (date_input, cat_input, footfall_input, int(footfall_input * 0.3), revenue_input, 0))
                elif "Delete" in operation:
                    cursor.execute('DELETE FROM sales WHERE date = ?', (del_date,))

                conn.commit()

                # Sync CSV
                df_updated = pd.read_sql_query("SELECT * FROM sales", conn)
                df_updated.to_csv(os.path.join(DATA_DIR, 'sales_data.csv'), index=False)
                conn.close()

                # Retrain
                subprocess.run([sys.executable, os.path.join(BASE_DIR, "ml", "train_models.py")], check=True)
                st.success("✅ Database updated and all models retrained!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")