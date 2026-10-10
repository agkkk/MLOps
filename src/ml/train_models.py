"""
Multi-Model ML Training Pipeline with Comparison
=================================================
For each ML task (anomaly detection, revenue forecasting, inventory clustering),
this module trains MULTIPLE models, compares their performance metrics,
selects the best one, and saves comparison results for the dashboard.

Models Trained:
- Revenue Forecasting: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, SVR
- Anomaly Detection: Isolation Forest, Local Outlier Factor, One-Class SVM
- Inventory Clustering: K-Means (k=2,3,4,5), DBSCAN, Agglomerative Clustering
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import (
    IsolationForest, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, OneClassSVM
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from datetime import datetime


def validate_data_quality(df, expected_columns, dataset_name):
    """Validate data schema and quality before training."""
    print(f"🔍 Checking Data Quality for: {dataset_name}...")

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"🚨 SCHEMA DRIFT DETECTED in {dataset_name}! "
            f"Missing columns: {missing_cols}. Training aborted."
        )

    for col in expected_columns:
        null_percentage = df[col].isnull().sum() / len(df)
        if null_percentage > 0.05:
            raise ValueError(
                f"🚨 DATA DRIFT DETECTED in {dataset_name}! "
                f"Column '{col}' has {null_percentage*100:.1f}% missing values. Training aborted."
            )

    print(f"✅ {dataset_name} passed all quality and schema checks.\n")


def train_revenue_models(X, y):
    """Train multiple regression models and compare performance."""
    print("\n" + "=" * 50)
    print("  TASK 1: REVENUE FORECASTING — MODEL COMPARISON")
    print("=" * 50)

    models = {
        'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
        'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Decision Tree': make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=42, max_depth=10)),
        'Random Forest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
        'Gradient Boosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42)),
        'SVR (RBF Kernel)': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1000)),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n🧠 Training: {name}...")
        model.fit(X, y)
        predictions = model.predict(X)

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        # Cross-validation score (3-fold due to limited data)
        n_splits = min(3, len(X))
        if n_splits >= 2:
            cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
            cv_mean = cv_scores.mean()
        else:
            cv_mean = r2

        results.append({
            'model': name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2_Score': round(r2, 4),
            'CV_R2_Mean': round(cv_mean, 4),
        })
        trained_models[name] = (model, predictions)

        print(f"   MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} | CV R²: {cv_mean:.4f}")

    # Find best model by R² score
    results_df = pd.DataFrame(results)
    best_idx = results_df['R2_Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    print(f"\n🏆 Best Revenue Model: {best_model_name} (R² = {results_df.loc[best_idx, 'R2_Score']})")

    return results_df, trained_models, best_model_name


def train_anomaly_models(X):
    """Train multiple anomaly detection models and compare."""
    print("\n" + "=" * 50)
    print("  TASK 2: ANOMALY DETECTION — MODEL COMPARISON")
    print("=" * 50)

    models = {
        'Isolation Forest': make_pipeline(
            StandardScaler(),
            IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        ),
        'Local Outlier Factor': None,  # LOF has different API
        'One-Class SVM': make_pipeline(
            StandardScaler(),
            OneClassSVM(kernel='rbf', nu=0.05)
        ),
    }

    results = []
    predictions_dict = {}

    for name, model in models.items():
        print(f"\n🧠 Training: {name}...")

        if name == 'Local Outlier Factor':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            preds = lof.fit_predict(X_scaled)
            model = make_pipeline(scaler, lof)
        else:
            preds = model.fit_predict(X)

        anomalies_detected = (preds == -1).sum()
        anomaly_rate = anomalies_detected / len(X) * 100
        normal_count = (preds == 1).sum()

        results.append({
            'model': name,
            'Anomalies_Detected': int(anomalies_detected),
            'Normal_Points': int(normal_count),
            'Anomaly_Rate_%': round(anomaly_rate, 2),
            'Total_Points': len(X),
        })
        predictions_dict[name] = preds

        print(f"   Anomalies: {anomalies_detected}/{len(X)} ({anomaly_rate:.2f}%)")

    results_df = pd.DataFrame(results)
    return results_df, predictions_dict, models


def train_clustering_models(X):
    """Train multiple clustering models and compare using silhouette score."""
    print("\n" + "=" * 50)
    print("  TASK 3: INVENTORY CLUSTERING — MODEL COMPARISON")
    print("=" * 50)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'K-Means (k=2)': KMeans(n_clusters=2, random_state=42, n_init=10),
        'K-Means (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
        'K-Means (k=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
        'K-Means (k=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
        'Agglomerative (k=3)': AgglomerativeClustering(n_clusters=3),
    }

    results = []
    predictions_dict = {}
    pipelines = {}

    for name, model in models.items():
        print(f"\n🧠 Training: {name}...")
        labels = model.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters >= 2:
            sil_score = silhouette_score(X_scaled, labels)
        else:
            sil_score = -1

        results.append({
            'model': name,
            'N_Clusters': n_clusters,
            'Silhouette_Score': round(sil_score, 4),
        })
        predictions_dict[name] = labels

        # Create proper pipeline for saving
        if 'K-Means' in name:
            pipelines[name] = make_pipeline(scaler, model)

        print(f"   Clusters: {n_clusters} | Silhouette: {sil_score:.4f}")

    # Try DBSCAN separately (auto-determines clusters)
    try:
        print(f"\n🧠 Training: DBSCAN...")
        from sklearn.cluster import DBSCAN as DBSCAN_
        dbscan = DBSCAN_(eps=0.8, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        if n_clusters_db >= 2:
            # Only compute silhouette on non-noise points
            mask = dbscan_labels != -1
            if mask.sum() > n_clusters_db:
                sil_db = silhouette_score(X_scaled[mask], dbscan_labels[mask])
            else:
                sil_db = -1
        else:
            sil_db = -1
        results.append({
            'model': 'DBSCAN',
            'N_Clusters': n_clusters_db,
            'Silhouette_Score': round(sil_db, 4),
        })
        predictions_dict['DBSCAN'] = dbscan_labels
        print(f"   Clusters: {n_clusters_db} | Silhouette: {sil_db:.4f}")
    except Exception as e:
        print(f"   DBSCAN failed: {e}")

    results_df = pd.DataFrame(results)
    best_idx = results_df['Silhouette_Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    print(f"\n🏆 Best Clustering Model: {best_model_name} (Silhouette = {results_df.loc[best_idx, 'Silhouette_Score']})")

    return results_df, predictions_dict, pipelines, best_model_name


def run_full_ml_pipeline():
    """Run the complete multi-model ML pipeline with comparisons."""
    print("\n" + "=" * 60)
    print("  ABC RETAIL — MULTI-MODEL ML TRAINING PIPELINE")
    print("=" * 60 + "\n")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Load departmental data ----
    df_sales = pd.read_csv(os.path.join(DATA_DIR, 'sales_data.csv'))
    df_inventory = pd.read_csv(os.path.join(DATA_DIR, 'inventory_data.csv'))
    df_finance = pd.read_csv(os.path.join(DATA_DIR, 'finance_data.csv'))
    df_support = pd.read_csv(os.path.join(DATA_DIR, 'support_data.csv'))
    df_logistics = pd.read_csv(os.path.join(DATA_DIR, 'logistics_data.csv'))

    # ---- Data quality validation ----
    validate_data_quality(df_sales, ['date', 'revenue', 'footfall'], "Sales Data")
    validate_data_quality(df_finance, ['date', 'marketing_spend', 'operating_cost'], "Finance Data")
    validate_data_quality(df_logistics, ['date', 'delayed_shipments'], "Logistics Data")
    validate_data_quality(df_support, ['date', 'tickets_opened'], "Support Data")

    all_comparison_results = {}

    # ============================================================
    # TASK 1: REVENUE FORECASTING (Multiple Models)
    # ============================================================
    print("\n🔄 Preparing revenue forecasting data...")
    daily_sales = df_sales.groupby('date').agg({'revenue': 'sum', 'footfall': 'sum'}).reset_index()
    df_fin_sales = pd.merge(daily_sales, df_finance, on='date')

    X_revenue = df_fin_sales[['footfall', 'marketing_spend', 'operating_cost']]
    y_revenue = df_fin_sales['revenue']

    revenue_results, revenue_models, best_revenue = train_revenue_models(X_revenue, y_revenue)
    all_comparison_results['revenue_forecasting'] = revenue_results.to_dict('records')

    # Save best revenue model
    best_rev_model, best_rev_preds = revenue_models[best_revenue]
    joblib.dump(best_rev_model, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    df_fin_sales['predicted_revenue'] = best_rev_preds
    
    # Also save Linear Regression predictions for trendline tab
    lr_model, lr_preds = revenue_models['Linear Regression']
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    df_fin_sales['lr_predicted_revenue'] = lr_preds
    
    df_fin_sales.to_csv(os.path.join(DATA_DIR, 'ml_forecast.csv'), index=False)

    # Log best model to MLflow
    mlflow.set_experiment("ABC_Retail_Revenue_Forecasting")
    with mlflow.start_run(run_name=f"best_{best_revenue.lower().replace(' ', '_')}"):
        best_metrics = revenue_results[revenue_results['model'] == best_revenue].iloc[0]
        mlflow.log_param("model_type", best_revenue)
        mlflow.log_metric("MAE", best_metrics['MAE'])
        mlflow.log_metric("RMSE", best_metrics['RMSE'])
        mlflow.log_metric("R2", best_metrics['R2_Score'])
        mlflow.sklearn.log_model(best_rev_model, "best_revenue_model")

    # ============================================================
    # TASK 2: ANOMALY DETECTION (Multiple Models)
    # ============================================================
    print("\n🔄 Preparing anomaly detection data...")
    df_ops = pd.merge(df_logistics, df_support, on='date')
    X_anomaly = df_ops[['delayed_shipments', 'tickets_opened']]

    anomaly_results, anomaly_preds, anomaly_models = train_anomaly_models(X_anomaly)
    all_comparison_results['anomaly_detection'] = anomaly_results.to_dict('records')

    # Use Isolation Forest as the primary (most widely used)
    iso_model = anomaly_models['Isolation Forest']
    joblib.dump(iso_model, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    df_ops['anomaly_flag'] = anomaly_preds['Isolation Forest']
    df_ops['is_anomaly'] = df_ops['anomaly_flag'].apply(lambda x: True if x == -1 else False)

    # Save all model predictions for comparison display
    for model_name, preds in anomaly_preds.items():
        col_name = f"pred_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        df_ops[col_name] = preds
    df_ops.to_csv(os.path.join(DATA_DIR, 'ml_anomalies.csv'), index=False)

    # ============================================================
    # TASK 3: INVENTORY CLUSTERING (Multiple Models)
    # ============================================================
    print("\n🔄 Preparing inventory clustering data...")
    # Encode categories
    encoder = LabelEncoder()
    df_inventory['category_encoded'] = encoder.fit_transform(df_inventory['category'])

    df_inv_sales = pd.merge(df_sales, df_inventory, on=['date', 'category'])
    X_cluster = df_inv_sales[['stock_level', 'quantity', 'discount']]

    cluster_results, cluster_preds, cluster_pipelines, best_cluster = train_clustering_models(X_cluster)
    all_comparison_results['inventory_clustering'] = cluster_results.to_dict('records')

    # Save best clustering model
    if best_cluster in cluster_pipelines:
        joblib.dump(cluster_pipelines[best_cluster], os.path.join(MODELS_DIR, 'kmeans.pkl'))
    else:
        # Fallback to K-Means k=3
        if 'K-Means (k=3)' in cluster_pipelines:
            joblib.dump(cluster_pipelines['K-Means (k=3)'], os.path.join(MODELS_DIR, 'kmeans.pkl'))
            best_cluster = 'K-Means (k=3)'

    df_inv_sales['cluster'] = cluster_preds.get(best_cluster, cluster_preds.get('K-Means (k=3)'))

    # Dynamic cluster mapping based on actual cluster means
    cluster_means = df_inv_sales.groupby('cluster')['revenue'].mean().sort_values(ascending=False)
    labels = ['High Efficiency', 'Average Day', 'Problematic Day', 'Low Activity', 'Marginal']
    cluster_mapping = {}
    for i, cluster_id in enumerate(cluster_means.index):
        cluster_mapping[cluster_id] = labels[i] if i < len(labels) else f'Cluster {cluster_id}'

    df_inv_sales['business_category'] = df_inv_sales['cluster'].map(cluster_mapping)
    joblib.dump(cluster_mapping, os.path.join(MODELS_DIR, 'cluster_mapping.pkl'))

    df_inv_sales.to_csv(os.path.join(DATA_DIR, 'ml_clusters.csv'), index=False)

    # ============================================================
    # SAVE COMPARISON REPORT
    # ============================================================
    comparison_report = {
        'results': all_comparison_results,
        'best_models': {
            'revenue_forecasting': best_revenue,
            'anomaly_detection': 'Isolation Forest',  # industry standard choice
            'inventory_clustering': best_cluster,
        },
        'timestamp': datetime.now().isoformat(),
    }

    report_path = os.path.join(DATA_DIR, 'model_comparison.json')
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    print(f"\n📋 Model comparison report saved: {report_path}")

    print("\n" + "=" * 60)
    print("  ✅ ALL MODELS TRAINED AND COMPARED SUCCESSFULLY")
    print("=" * 60)

    return comparison_report


if __name__ == "__main__":
    run_full_ml_pipeline()