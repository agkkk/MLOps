import sqlite3
import os

DB_PATH = os.path.join('data', 'retail.db')

def setup_database():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Sales
    cursor.execute('''CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, category TEXT, footfall INTEGER, quantity INTEGER, revenue REAL, discount REAL
    )''')
    
    # 2. Inventory
    cursor.execute('''CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, category TEXT, stock_level INTEGER, restock_threshold INTEGER
    )''')
    
    # 3. Finance
    cursor.execute('''CREATE TABLE IF NOT EXISTS finance (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, operating_cost REAL, marketing_spend REAL, net_profit REAL
    )''')
    
    # 4. Customer Support
    cursor.execute('''CREATE TABLE IF NOT EXISTS support (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, tickets_opened INTEGER, tickets_resolved INTEGER, avg_resolution_time REAL
    )''')
    
    # 5. Logistics
    cursor.execute('''CREATE TABLE IF NOT EXISTS logistics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, shipments_dispatched INTEGER, delayed_shipments INTEGER, transport_cost REAL
    )''')

    conn.commit()
    conn.close()
    print("✅ Database and 5 core tables initialized successfully.")

if __name__ == "__main__":
    setup_database()