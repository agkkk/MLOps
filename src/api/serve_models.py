from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize the FastAPI App
app = FastAPI(
    title="ABC Retail ML API",
    description="Real-time feature serving for the MLOps pipeline. Supports anomaly detection, revenue forecasting, and inventory clustering."
)

# 2. Define the exact data structures expected by the API
class OperationsData(BaseModel):
    delayed_shipments: float
    tickets_opened: float

class FinanceData(BaseModel):
    footfall: float
    marketing_spend: float
    operating_cost: float

class InventoryData(BaseModel):
    stock_level: float
    quantity: float
    discount: float

# Global variables to hold our models in memory
models = {}
cluster_mapping = {}

# 3. Load Models on Server Startup
@app.on_event("startup")
def load_models():
    print("⏳ Loading ML Models into memory...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    try:
        models['isolation_forest'] = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
        models['random_forest'] = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
        models['kmeans'] = joblib.load(os.path.join(MODELS_DIR, 'kmeans.pkl'))
        models['linear_regression'] = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.pkl'))

        # Load dynamic cluster mapping
        mapping_path = os.path.join(MODELS_DIR, 'cluster_mapping.pkl')
        if os.path.exists(mapping_path):
            global cluster_mapping
            cluster_mapping = joblib.load(mapping_path)
        else:
            cluster_mapping = {0: 'High Efficiency', 1: 'Average Day', 2: 'Problematic Day'}

        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# 4. Define the Endpoints

# Health check
@app.get("/health")
def health_check():
    loaded = list(models.keys())
    return {"status": "API is live", "models_loaded": loaded, "total_models": len(loaded)}

# Endpoint 1: Anomaly Detection
@app.post("/predict/anomaly")
def predict_anomaly(data: OperationsData):
    if 'isolation_forest' not in models:
        raise HTTPException(status_code=503, detail="Isolation Forest model not loaded")
    df = pd.DataFrame([data.dict()])
    prediction = models['isolation_forest'].predict(df)[0]
    is_anomaly = bool(prediction == -1)
    return {"is_anomaly": is_anomaly, "model_used": "Isolation Forest"}

# Endpoint 2: Revenue Forecasting
@app.post("/predict/revenue")
def predict_revenue(data: FinanceData):
    if 'random_forest' not in models:
        raise HTTPException(status_code=503, detail="Revenue model not loaded")
    df = pd.DataFrame([data.dict()])
    prediction = models['random_forest'].predict(df)[0]
    return {"predicted_revenue": round(float(prediction), 2), "model_used": "Best Revenue Model"}

# Endpoint 3: Inventory Clustering
@app.post("/predict/cluster")
def predict_cluster(data: InventoryData):
    if 'kmeans' not in models:
        raise HTTPException(status_code=503, detail="KMeans model not loaded")
    df = pd.DataFrame([data.dict()])
    cluster = models['kmeans'].predict(df)[0]

    # Use dynamic mapping instead of hardcoded
    business_category = cluster_mapping.get(int(cluster), f"Cluster {cluster}")
    return {"cluster_id": int(cluster), "business_category": business_category}

# Endpoint 4: Linear Regression Trendline
@app.post("/predict/trendline")
def predict_trendline(data: FinanceData):
    if 'linear_regression' not in models:
        raise HTTPException(status_code=503, detail="Linear Regression model not loaded")
    df = pd.DataFrame([{"footfall": data.footfall}])
    prediction = models['linear_regression'].predict(df)[0]
    return {"predicted_revenue_trendline": round(float(prediction), 2), "model_used": "Linear Regression"}