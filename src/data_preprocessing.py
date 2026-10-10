"""
Data Preprocessing Module
=========================
Loads the real-world Global Superstore dataset, performs comprehensive
data cleaning and quality checks, and transforms it into the 5 departmental
tables used by the ML pipeline.

Preprocessing Steps:
1. Load raw CSV with encoding handling
2. Remove duplicates
3. Handle missing values (imputation / removal)
4. Fix data types (dates, numerics)
5. Detect and handle outliers (IQR method)
6. Feature engineering (derived columns)
7. Transform into departmental tables (sales, inventory, finance, support, logistics)
8. Export cleaned data to CSV + SQLite
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime


def load_raw_data(filepath):
    """Step 1: Load raw data with proper encoding."""
    print("📂 Step 1: Loading raw dataset...")
    df = pd.read_csv(filepath, encoding='latin1')
    print(f"   → Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def generate_quality_report(df, stage_name="Raw"):
    """Generate a data quality summary for before/after comparison."""
    report = {
        "stage": stage_name,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isnull().sum().sum()),
        "columns_with_nulls": {},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            report["columns_with_nulls"][col] = {
                "count": null_count,
                "percentage": round(null_count / len(df) * 100, 2)
            }
    return report


def remove_duplicates(df):
    """Step 2: Remove duplicate rows."""
    print("🔄 Step 2: Removing duplicate rows...")
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"   → Removed {removed} duplicate rows ({len(df)} remaining)")
    return df


def handle_missing_values(df):
    """Step 3: Handle missing values with appropriate strategies."""
    print("🩹 Step 3: Handling missing values...")

    # Postal Code: fill with 'Unknown' (many international orders lack postal codes)
    if 'Postal Code' in df.columns:
        null_count = df['Postal Code'].isnull().sum()
        df['Postal Code'] = df['Postal Code'].fillna('Unknown')
        print(f"   → Postal Code: filled {null_count} nulls with 'Unknown'")

    # For numeric columns, fill remaining nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   → {col}: filled {null_count} nulls with median ({median_val:.2f})")

    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   → {col}: filled {null_count} nulls with mode ('{mode_val}')")

    remaining = df.isnull().sum().sum()
    print(f"   → Total remaining nulls: {remaining}")
    return df


def fix_data_types(df):
    """Step 4: Fix data types — parse dates, ensure numeric types."""
    print("🔧 Step 4: Fixing data types...")

    # Parse date columns
    for col in ['Order Date', 'Ship Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=False)
            print(f"   → {col}: converted to datetime")

    # Ensure numeric columns are proper types
    for col in ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("   → Numeric columns validated")
    return df


def detect_and_handle_outliers(df):
    """Step 5: Detect outliers using IQR method, flag but don't remove."""
    print("📊 Step 5: Detecting outliers (IQR method)...")

    outlier_cols = ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost']
    df['is_outlier'] = False
    outlier_summary = {}

    for col in outlier_cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        col_outliers = (df[col] < lower) | (df[col] > upper)
        count = col_outliers.sum()
        df.loc[col_outliers, 'is_outlier'] = True
        outlier_summary[col] = {
            "count": int(count),
            "percentage": round(count / len(df) * 100, 2),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2)
        }
        print(f"   → {col}: {count} outliers ({outlier_summary[col]['percentage']}%)")

    total_outlier_rows = df['is_outlier'].sum()
    print(f"   → Total rows flagged as outliers: {total_outlier_rows}")
    return df, outlier_summary


def engineer_features(df):
    """Step 6: Feature engineering — derive new useful columns."""
    print("⚙️  Step 6: Engineering features...")

    # Shipping delay in days
    if 'Ship Date' in df.columns and 'Order Date' in df.columns:
        df['shipping_delay_days'] = (df['Ship Date'] - df['Order Date']).dt.days
        print("   → Created: shipping_delay_days")

    # Profit margin
    if 'Profit' in df.columns and 'Sales' in df.columns:
        df['profit_margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'], 0)
        print("   → Created: profit_margin")

    # Revenue per unit
    if 'Sales' in df.columns and 'Quantity' in df.columns:
        df['revenue_per_unit'] = np.where(df['Quantity'] != 0, df['Sales'] / df['Quantity'], 0)
        print("   → Created: revenue_per_unit")

    # Extract date parts
    if 'Order Date' in df.columns:
        df['order_month'] = df['Order Date'].dt.month
        df['order_year'] = df['Order Date'].dt.year
        df['order_day_of_week'] = df['Order Date'].dt.dayofweek
        print("   → Created: order_month, order_year, order_day_of_week")

    return df


def transform_to_departmental_tables(df):
    """
    Step 7: Transform the cleaned Superstore data into 5 departmental tables
    that mirror a real retail organization's data structure.
    """
    print("🏗️  Step 7: Transforming into departmental tables...")

    df['date'] = df['Order Date'].dt.strftime('%Y-%m-%d')

    # --- SALES TABLE ---
    sales = df.groupby(['date', 'Category']).agg(
        footfall=('Order ID', 'nunique'),       # unique orders as proxy for footfall
        quantity=('Quantity', 'sum'),
        revenue=('Sales', 'sum'),
        discount=('Discount', 'mean'),
    ).reset_index()
    sales.rename(columns={'Category': 'category'}, inplace=True)
    sales['revenue'] = sales['revenue'].round(2)
    sales['discount'] = (sales['discount'] * 100).round(1)  # convert to percentage
    print(f"   → Sales: {len(sales)} rows")

    # --- INVENTORY TABLE ---
    inventory = df.groupby(['date', 'Category']).agg(
        stock_level=('Quantity', 'sum'),
        restock_threshold=('Quantity', lambda x: max(1, int(x.mean() * 0.3))),
    ).reset_index()
    inventory.rename(columns={'Category': 'category'}, inplace=True)
    # Simulate realistic stock levels (higher than sold quantity)
    inventory['stock_level'] = (inventory['stock_level'] * np.random.uniform(2, 5, len(inventory))).astype(int)
    print(f"   → Inventory: {len(inventory)} rows")

    # --- FINANCE TABLE ---
    daily_finance = df.groupby('date').agg(
        revenue=('Sales', 'sum'),
        operating_cost=('Shipping Cost', 'sum'),
        marketing_spend=('Discount', lambda x: (x * 500).sum()),  # discount-driven spend
        net_profit=('Profit', 'sum'),
    ).reset_index()
    daily_finance['operating_cost'] = daily_finance['operating_cost'].round(2)
    daily_finance['marketing_spend'] = daily_finance['marketing_spend'].clip(lower=100).round(2)
    daily_finance['net_profit'] = daily_finance['net_profit'].round(2)
    daily_finance.drop(columns=['revenue'], inplace=True)
    print(f"   → Finance: {len(daily_finance)} rows")

    # --- SUPPORT TABLE ---
    # Derive support tickets from order volume and profit margin issues
    daily_orders = df.groupby('date').agg(
        order_count=('Order ID', 'nunique'),
        avg_profit_margin=('profit_margin', 'mean'),
        delayed_orders=('shipping_delay_days', lambda x: (x > 5).sum()),
    ).reset_index()

    support = daily_orders.copy()
    np.random.seed(42)
    # More tickets on days with low profit margins or delays
    base_tickets = (support['order_count'] * 0.3).astype(int)
    extra_tickets = (support['delayed_orders'] * 2).astype(int)
    support['tickets_opened'] = base_tickets + extra_tickets + np.random.randint(1, 5, len(support))
    support['tickets_resolved'] = (support['tickets_opened'] * np.random.uniform(0.7, 0.95, len(support))).astype(int)
    support['avg_resolution_time'] = np.round(np.random.uniform(1.5, 12.0, len(support)), 1)
    support = support[['date', 'tickets_opened', 'tickets_resolved', 'avg_resolution_time']]
    print(f"   → Support: {len(support)} rows")

    # --- LOGISTICS TABLE ---
    logistics = df.groupby('date').agg(
        shipments_dispatched=('Order ID', 'nunique'),
        delayed_shipments=('shipping_delay_days', lambda x: (x > 4).sum()),
        transport_cost=('Shipping Cost', 'sum'),
    ).reset_index()
    logistics['transport_cost'] = logistics['transport_cost'].round(2)
    print(f"   → Logistics: {len(logistics)} rows")

    return sales, inventory, daily_finance, support, logistics


def save_to_csv_and_db(sales, inventory, finance, support, logistics, data_dir, db_path):
    """Step 8: Save cleaned departmental data to CSV files and SQLite database."""
    print("💾 Step 8: Saving to CSV and SQLite...")

    tables = {
        'sales': sales,
        'inventory': inventory,
        'finance': finance,
        'support': support,
        'logistics': logistics,
    }

    # Save CSVs
    for name, df in tables.items():
        csv_path = os.path.join(data_dir, f'{name}_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"   → Saved {csv_path}")

    # Save to SQLite
    conn = sqlite3.connect(db_path)
    for name, df in tables.items():
        df.to_sql(name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"   → Saved to SQLite: {db_path}")


def run_preprocessing_pipeline():
    """Execute the full preprocessing pipeline end-to-end."""
    print("=" * 60)
    print("  ABC RETAIL - DATA PREPROCESSING PIPELINE")
    print("  Source: Global Superstore Dataset (Real-World Data)")
    print("=" * 60 + "\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_FILE = os.path.join(DATA_DIR, 'raw_superstore.csv')
    DB_PATH = os.path.join(DATA_DIR, 'retail.db')
    REPORT_PATH = os.path.join(DATA_DIR, 'preprocessing_report.json')

    # Step 1: Load
    df = load_raw_data(RAW_FILE)

    # Generate BEFORE report
    before_report = generate_quality_report(df, "Before Preprocessing")
    # Save a sample of raw data for dashboard display
    df.head(50).to_csv(os.path.join(DATA_DIR, 'raw_sample.csv'), index=False)

    # Step 2: Duplicates
    df = remove_duplicates(df)

    # Step 3: Missing values
    df = handle_missing_values(df)

    # Step 4: Data types
    df = fix_data_types(df)

    # Step 5: Outliers
    df, outlier_summary = detect_and_handle_outliers(df)

    # Step 6: Feature engineering
    df = engineer_features(df)

    # Generate AFTER report
    after_report = generate_quality_report(df, "After Preprocessing")

    # Save the fully cleaned dataset
    df.to_csv(os.path.join(DATA_DIR, 'cleaned_superstore.csv'), index=False)
    print(f"\n✅ Cleaned dataset saved: {len(df)} rows × {len(df.columns)} columns")

    # Step 7: Transform to departmental tables
    sales, inventory, finance, support, logistics = transform_to_departmental_tables(df)

    # Step 8: Save
    save_to_csv_and_db(sales, inventory, finance, support, logistics, DATA_DIR, DB_PATH)

    # Save preprocessing report for dashboard
    full_report = {
        "before": before_report,
        "after": after_report,
        "outlier_summary": outlier_summary,
        "preprocessing_steps": [
            {"step": 1, "name": "Load Raw Data", "description": "Loaded Global Superstore CSV with latin1 encoding"},
            {"step": 2, "name": "Remove Duplicates", "description": f"Removed {before_report['duplicate_rows']} duplicate rows"},
            {"step": 3, "name": "Handle Missing Values", "description": "Imputed nulls using median (numeric) and mode (categorical)"},
            {"step": 4, "name": "Fix Data Types", "description": "Parsed dates, validated numeric columns"},
            {"step": 5, "name": "Detect Outliers", "description": "Flagged outliers using IQR method across 5 columns"},
            {"step": 6, "name": "Feature Engineering", "description": "Created shipping_delay_days, profit_margin, revenue_per_unit, date features"},
            {"step": 7, "name": "Departmental Transformation", "description": "Split into 5 business tables: Sales, Inventory, Finance, Support, Logistics"},
            {"step": 8, "name": "Export", "description": "Saved to CSV files and SQLite database"},
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"\n📋 Preprocessing report saved: {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("  ✅ PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)

    return full_report


if __name__ == "__main__":
    run_preprocessing_pipeline()
