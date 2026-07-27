import sqlite3
import random
from datetime import datetime, timedelta
import os

# FIXED: Added 'src' to the path so it finds the correct folder
DB_PATH = os.path.join('src', 'data', 'retail.db')

def generate_full_dataset():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tables = ['sales', 'inventory', 'finance', 'support', 'logistics']
    for table in tables:
        cursor.execute(f'DELETE FROM {table}')

    # TWEAKED: Started earlier so we can generate more historical data
    start_date = datetime(2024, 1, 1) 
    categories = ['Electronics', 'Clothing', 'Grocery']
    
    # TWEAKED: Increased from 180 days to 730 days (2 years of data)
    print("Generating 730 days (2 years) of cross-department data...")

    for i in range(730):
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
    conn.close()
    print("✅ All tables populated with 730 days of realistic business data.")

if __name__ == "__main__":
    generate_full_dataset()