import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join('data', 'retail.db')

def execute_query(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def fetch_query(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

def create_record(table, columns, values):
    placeholders = ', '.join(['?'] * len(values))
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    execute_query(query, values)
    print(f"✅ Record added to {table}.")

def read_records(table):
    rows = fetch_query(f"SELECT * FROM {table} LIMIT 10")
    print(f"\n--- Top 10 Records in {table.upper()} ---")
    for row in rows:
        print(row)

def delete_record(table, record_id):
    execute_query(f"DELETE FROM {table} WHERE id = ?", (record_id,))
    print(f"✅ Record {record_id} deleted from {table}.")

def export_to_csv():
    tables = ['sales', 'inventory', 'finance', 'support', 'logistics']
    conn = sqlite3.connect(DB_PATH)
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(f"data/{table}_data.csv", index=False)
    conn.close()
    print("✅ All tables exported to CSV in the 'data/' folder.")