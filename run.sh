#!/bin/bash
echo "🚀 Booting up ABC Retail MLOps Architecture..."

# 1. Activate the virtual environment
source venv/bin/activate

# 2. Start the MLflow tracking server in the background
echo "📊 Starting MLflow..."
mlflow ui &

# 3. Start the FastAPI Live Inference server in the background
echo "⚡ Starting FastAPI Engine..."
uvicorn src.api.serve_models:app --reload &

# 4. Start the Streamlit Dashboard in the background
echo "🖥️ Starting Streamlit Dashboard..."
streamlit run src/dashboard.py &

# 5. Start the Automated Scheduler (Runs in the foreground so you can see it)
echo "🕰️ Starting Automation Scheduler..."
python src/scheduler.py