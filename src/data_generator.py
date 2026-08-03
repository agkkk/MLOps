import sqlite3
import random
from datetime import datetime, timedelta
import os
import pandas as pd

# FIXED: Added 'src' to the path so it finds the correct folder
DB_PATH = os.path.join('src', 'data', 'retail.db')

def generate_full_dataset():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Find the last date in the database to know where to append
    try:
        cursor.execute("SELECT MAX(date) FROM sales")
        last_date_str = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        # Failsafe just in case the table is completely uninitialized
        last_date_str = None

    # 2. Figure out what day comes next
    if last_date_str:
        # If data exists, start generating from the day AFTER the last entry
        start_date = datetime.strptime(last_date_str, '%Y-%m-%d') + timedelta(days=1)
        print(f"Appending new data for 1 day starting from {start_date.strftime('%Y-%m-%d')}...")
    else:
        # If the database is completely empty, start from the beginning
        start_date = datetime(2024, 1, 1)
        print("Database empty. Generating initial baseline data...")

    categories = ['Electronics', 'Clothing', 'Grocery']
    
    # 3. Only generate 1 NEW DAY of data every time this runs
    days_to_generate = 1

    for i in range(days_to_generate):
        current_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        
        # Base daily metrics
        daily_revenue = 0
        daily_footfall = 0
        daily_shipments = random.randint(50, 200)
        delayed = random.randint(0, 5) # Normal delay
        tickets = random.randint(10, 30) # Normal tickets
        
        # INJECT ANOMALY (3% chance of a Logistics Failure leading to Support crisis)
        if random.random() < 0.03:
            delayed = daily_shipments // 2  # Massive delay
            tickets = tickets * 5           # Support gets hammered
            
        for cat in categories:
            # Sales & Inventory
            footfall = random.randint(100, 500)
            quantity = int(footfall * random.uniform(0.1, 0.5))
            revenue = quantity * random.uniform(50, 500)
            stock = random.randint(50, 1000)
            
            # INJECT ANOMALY (Stockout leading to 0 revenue)
            if stock < 50 and random.random() < 0.1:
                quantity, revenue = 0, 0
                
            daily_revenue += revenue
            daily_footfall += footfall
            
            cursor.execute('INSERT INTO sales (date, category, footfall, quantity, revenue, discount) VALUES (?, ?, ?, ?, ?, ?)', 
                           (current_date, cat, footfall, quantity, round(revenue,2), random.choice([0, 10])))
            cursor.execute('INSERT INTO inventory (date, category, stock_level, restock_threshold) VALUES (?, ?, ?, ?)', 
                           (current_date, cat, stock, 100))
        
        # Support, Logistics, Finance
        op_cost = daily_revenue * 0.4
        marketing = 500
        profit = daily_revenue - (op_cost + marketing)
        if delayed > (daily_shipments * 0.3): # If massive delays, profit drops due to refunds
            profit -= (profit * 0.5)

        cursor.execute('INSERT INTO finance (date, operating_cost, marketing_spend, net_profit) VALUES (?, ?, ?, ?)', 
                       (current_date, round(op_cost,2), marketing, round(profit,2)))
        cursor.execute('INSERT INTO support (date, tickets_opened, tickets_resolved, avg_resolution_time) VALUES (?, ?, ?, ?)', 
                       (current_date, tickets, int(tickets*0.9), round(random.uniform(2, 12),1)))
        cursor.execute('INSERT INTO logistics (date, shipments_dispatched, delayed_shipments, transport_cost) VALUES (?, ?, ?, ?)', 
                       (current_date, daily_shipments, delayed, round(daily_shipments*15,2)))

    conn.commit()

    tables = ['sales', 'inventory', 'finance', 'support', 'logistics']
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(os.path.join('src', 'data', f'{table}_data.csv'), index=False)
        
    conn.close()
    print("✅ New daily data successfully appended to the database.")

if __name__ == "__main__":
    generate_full_dataset()