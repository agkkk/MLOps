import schedule
import time
import subprocess
import datetime
import sys

def run_pipeline():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[🕰️ {current_time}] CRON JOB TRIGGERED: Starting Automated ML Pipeline...")
    
    try:
        # 1. RUN THE GENERATOR FIRST to append the new day
        subprocess.run([sys.executable, "src/data_generator.py"], check=True)
        
        # 2. THEN TRAIN THE MODELS on the fresh data
        subprocess.run([sys.executable, "src/ml/train_models.py"], check=True)
        
        print(f"[✅ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Automated training cycle completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[❌] Pipeline execution failed: {e}")
# ==========================================
# SCHEDULE CONFIGURATION
# ==========================================
# For testing and demonstration (Runs every 1 minute)
schedule.every(1).minutes.do(run_pipeline)

# For production batch processing, you would uncomment this:
# schedule.every().day.at("00:00").do(run_pipeline)

print("⚙️  ABC Retail MLOps Scheduler Initialized.")
print("⏳ Waiting for the first scheduled cycle (1 minute)... Press Ctrl+C to exit.")

# The infinite loop that keeps the scheduler running in the background
while True:
    schedule.run_pending()
    time.sleep(1)