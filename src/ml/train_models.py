import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# ==========================================
# NEW: IMPORT LINEAR REGRESSION
# ==========================================
from sklearn.linear_model import LinearRegression

def validate_data_quality(df, expected_columns, dataset_name):
    print(f"🔍 Checking Data Quality for: {dataset_name}...")
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"🚨 SCHEMA DRIFT DETECTED in {dataset_name}! Missing columns: {missing_cols}. Training aborted.")
    
    for col in expected_columns:
        null_percentage = df[col].isnull().sum() / len(df)
        if null_percentage > 0.05:
            raise ValueError(f"🚨 DATA DRIFT DETECTED in {dataset_name}! Column '{col}' has {null_percentage*100:.1f}% missing values. Training aborted.")
            
    print(f"✅ {dataset_name} passed all quality and schema checks.\n")

def run_full_ml_pipeline():
    print("🚀 Initializing Phase 3: Multi-Model ML Pipeline...\n")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    df_sales = pd.read_csv(os.path.join(DATA_DIR, 'sales_data.csv'))
    df_inventory = pd.read_csv(os.path.join(DATA_DIR, 'inventory_data.csv'))
    df_finance = pd.read_csv(os.path.join(DATA_DIR, 'finance_data.csv'))
    df_support = pd.read_csv(os.path.join(DATA_DIR, 'support_data.csv'))
    df_logistics = pd.read_csv(os.path.join(DATA_DIR, 'logistics_data.csv'))

    validate_data_quality(df_sales, ['date', 'revenue', 'footfall'], "Sales Data")
    validate_data_quality(df_finance, ['date', 'marketing_spend', 'operating_cost'], "Finance Data")
    validate_data_quality(df_logistics, ['date', 'delayed_shipments'], "Logistics Data")
    validate_data_quality(df_support, ['date', 'tickets_opened'], "Support Data")

    print("🔄 Performing Data Transformation and Encoding...")
    encoder = LabelEncoder()
    df_inventory['category_encoded'] = encoder.fit_transform(df_inventory['category'])
    
    # --- MODEL 1: ISOLATION FOREST ---
    print("🧠 Training Model 1: Isolation Forest...")
    df_ops = pd.merge(df_logistics, df_support, on='date')
    X_iso = df_ops[['delayed_shipments', 'tickets_opened']]
    
    iso_model = make_pipeline(StandardScaler(), IsolationForest(n_estimators=100, contamination=0.03, random_state=42))
    df_ops['anomaly_flag'] = iso_model.fit_predict(X_iso)
    df_ops['is_anomaly'] = df_ops['anomaly_flag'].apply(lambda x: True if x == -1 else False)
    
    joblib.dump(iso_model, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    df_ops.to_csv(os.path.join(DATA_DIR, 'ml_anomalies.csv'), index=False)

    # --- MODEL 2: RANDOM FOREST ---
    print("🧠 Training Model 2: Random Forest...")
    daily_sales = df_sales.groupby('date').agg({'revenue': 'sum', 'footfall': 'sum'}).reset_index()
    df_fin_sales = pd.merge(daily_sales, df_finance, on='date')
    
    X_rf = df_fin_sales[['footfall', 'marketing_spend', 'operating_cost']]
    y_rf = df_fin_sales['revenue']
    
    mlflow.set_experiment("ABC_Retail_Revenue_Forecasting")
    with mlflow.start_run():
        n_estimators = 100
        rf_model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=n_estimators, random_state=42))
        rf_model.fit(X_rf, y_rf)
        predictions = rf_model.predict(X_rf)
        mae = mean_absolute_error(y_rf, predictions)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        
        df_fin_sales['predicted_revenue'] = predictions
        joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest.pkl'))

    # ==========================================
    # NEW: MODEL 4 (LINEAR REGRESSION)
    # ==========================================
    print("🧠 Training Model 4: Linear Regression (Trendline)...")
    X_lr = df_fin_sales[['footfall']]
    
    lr_model = make_pipeline(StandardScaler(), LinearRegression())
    lr_model.fit(X_lr, y_rf)
    
    df_fin_sales['lr_predicted_revenue'] = lr_model.predict(X_lr)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    
    # Save the file with both RF and LR predictions
    df_fin_sales.to_csv(os.path.join(DATA_DIR, 'ml_forecast.csv'), index=False)

    # --- MODEL 3: K-MEANS ---
    print("🧠 Training Model 3: K-Means...")
    df_inv_sales = pd.merge(df_sales, df_inventory, on=['date', 'category'])
    X_kmeans = df_inv_sales[['stock_level', 'quantity', 'discount']]
    
    kmeans_model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42, n_init=10))
    df_inv_sales['cluster'] = kmeans_model.fit_predict(X_kmeans)
    
    cluster_mapping = {0: 'High Efficiency', 1: 'Average Day', 2: 'Problematic Day'}
    df_inv_sales['business_category'] = df_inv_sales['cluster'].map(cluster_mapping)
    
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'kmeans.pkl'))
    df_inv_sales.to_csv(os.path.join(DATA_DIR, 'ml_clusters.csv'), index=False)

    print("\n🎯 Phase 3 Complete! All models (including Linear Regression) are trained.")

if __name__ == "__main__":
    run_full_ml_pipeline()