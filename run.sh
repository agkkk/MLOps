#!/bin/bash
echo "🚀 Booting up ABC Retail MLOps Architecture..."

# 1. Activate the virtual environment (Silenced for Docker compatibility)
source venv/bin/activate 2>/dev/null || true

# 2. Run data preprocessing pipeline (loads real data, cleans, transforms)
echo "🧹 Running Data Preprocessing Pipeline..."
python src/data_preprocessing.py

# 3. Run ML training pipeline (trains & compares all models)
echo "🧠 Running Multi-Model ML Training Pipeline..."
python src/ml/train_models.py

# 4. Start the MLflow tracking server in the background
echo "📊 Starting MLflow..."
mlflow ui --host 0.0.0.0 &

# 5. Start the FastAPI Live Inference server in the background
echo "⚡ Starting FastAPI Engine..."
uvicorn src.api.serve_models:app --host 0.0.0.0 --port 8000 --reload &

# 6. Start the Streamlit Dashboard (foreground so you can see logs)
echo "🖥️ Starting Streamlit Dashboard..."
streamlit run src/dashboard.py --server.port 8501