from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize the FastAPI App
app = FastAPI(title="ABC Retail ML API", description="Real-time feature serving for MLOps pipeline")

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
        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# 4. Define the Endpoints

# A simple health check to ensure the server is running
@app.get("/health")
def health_check():
    return {"status": "API is live and models are loaded."}

# Endpoint 1: Anomaly Detection
@app.post("/predict/anomaly")
def predict_anomaly(data: OperationsData):
    # Convert incoming JSON data into a format the model understands (Pandas DataFrame)
    df = pd.DataFrame([data.dict()])
    
    # Predict (-1 is anomaly, 1 is normal)
    prediction = models['isolation_forest'].predict(df)[0]
    is_anomaly = bool(prediction == -1)
    
    return {"is_anomaly": is_anomaly}

# Endpoint 2: Revenue Forecasting
@app.post("/predict/revenue")
def predict_revenue(data: FinanceData):
    df = pd.DataFrame([data.dict()])
    
    # Predict revenue
    prediction = models['random_forest'].predict(df)[0]
    
    return {"predicted_revenue": round(float(prediction), 2)}

# Endpoint 3: Inventory Clustering
@app.post("/predict/cluster")
def predict_cluster(data: InventoryData):
    df = pd.DataFrame([data.dict()])
    
    # Predict cluster (0, 1, or 2)
    cluster = models['kmeans'].predict(df)[0]
    
    # Map back to business logic
    cluster_mapping = {0: 'High Efficiency', 1: 'Average Day', 2: 'Problematic Day'}
    business_category = cluster_mapping.get(int(cluster), "Unknown")
    
    return {"cluster_id": int(cluster), "business_category": business_category}