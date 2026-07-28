#!/bin/bash
echo "🚀 Booting up ABC Retail MLOps Architecture..."

# 1. Activate the virtual environment (Silenced for Docker compatibility)
source venv/bin/activate 2>/dev/null || true

# 2. Start the MLflow tracking server in the background
echo "📊 Starting MLflow..."
mlflow ui --host 0.0.0.0 &

# 3. Start the FastAPI Live Inference server in the background
echo "⚡ Starting FastAPI Engine..."
uvicorn src.api.serve_models:app --host 0.0.0.0 --port 8000 --reload &

# 4. Start the Streamlit Dashboard in the background
echo "🖥️ Starting Streamlit Dashboard..."
streamlit run src/dashboard.py &

# 5. Start the Automated Scheduler (Runs in the foreground so you can see it)
echo "🕰️ Starting Automation Scheduler..."
python src/scheduler.py