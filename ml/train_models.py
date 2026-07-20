import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans

def run_full_ml_pipeline():
    print("🚀 Initializing Phase 3: Multi-Model ML Pipeline...\n")
    
    # Setup paths relative to the script location
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load the datasets generated in Phase 1
    df_sales = pd.read_csv(os.path.join(DATA_DIR, 'sales_data.csv'))
    df_inventory = pd.read_csv(os.path.join(DATA_DIR, 'inventory_data.csv'))
    df_finance = pd.read_csv(os.path.join(DATA_DIR, 'finance_data.csv'))
    df_support = pd.read_csv(os.path.join(DATA_DIR, 'support_data.csv'))
    df_logistics = pd.read_csv(os.path.join(DATA_DIR, 'logistics_data.csv'))

    # ==========================================
    # MODEL 1: ISOLATION FOREST (Anomaly Detection)
    # Goal: Detect operational failures (Delays causing Support Tickets)
    # ==========================================
    print("🧠 Training Model 1: Isolation Forest (Anomaly Detection)...")
    df_ops = pd.merge(df_logistics, df_support, on='date')
    X_iso = df_ops[['delayed_shipments', 'tickets_opened']]
    
    iso_model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    df_ops['anomaly_flag'] = iso_model.fit_predict(X_iso)
    df_ops['is_anomaly'] = df_ops['anomaly_flag'].apply(lambda x: True if x == -1 else False)
    
    joblib.dump(iso_model, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    df_ops.to_csv(os.path.join(DATA_DIR, 'ml_anomalies.csv'), index=False)
    print(f"   ✅ Detected {df_ops['is_anomaly'].sum()} severe operational failures.")

    # ==========================================
    # MODEL 2: RANDOM FOREST (Forecasting)
    # Goal: Predict daily revenue based on footfall and spending
    # ==========================================
    print("🧠 Training Model 2: Random Forest Regressor (Revenue Forecasting)...")
    # Aggregate sales by date since there are multiple categories per day
    daily_sales = df_sales.groupby('date').agg({'revenue': 'sum', 'footfall': 'sum'}).reset_index()
    df_fin_sales = pd.merge(daily_sales, df_finance, on='date')
    
    X_rf = df_fin_sales[['footfall', 'marketing_spend', 'operating_cost']]
    y_rf = df_fin_sales['revenue']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_rf, y_rf)
    
    # Predict on the dataset to show accuracy on dashboard later
    df_fin_sales['predicted_revenue'] = rf_model.predict(X_rf)
    
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    df_fin_sales.to_csv(os.path.join(DATA_DIR, 'ml_forecast.csv'), index=False)
    print("   ✅ Forecasting model trained and predictions saved.")

    # ==========================================
    # MODEL 3: K-MEANS CLUSTERING (Data Segmentation)
    # Goal: Categorize operational days based on stock movement
    # ==========================================
    print("🧠 Training Model 3: K-Means (Inventory/Sales Segmentation)...")
    df_inv_sales = pd.merge(df_sales, df_inventory, on=['date', 'category'])
    
    X_kmeans = df_inv_sales[['stock_level', 'quantity', 'discount']]
    
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_inv_sales['cluster'] = kmeans_model.fit_predict(X_kmeans)
    
    # Map cluster numbers to readable business categories
    cluster_mapping = {0: 'High Efficiency', 1: 'Average Day', 2: 'Problematic Day'}
    df_inv_sales['business_category'] = df_inv_sales['cluster'].map(cluster_mapping)
    
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'kmeans.pkl'))
    df_inv_sales.to_csv(os.path.join(DATA_DIR, 'ml_clusters.csv'), index=False)
    print("   ✅ Segmented data into 3 distinct business categories.")

    print("\n🎯 Phase 3 Complete! Models and enriched datasets are ready for the Dashboard.")

if __name__ == "__main__":
    run_full_ml_pipeline()