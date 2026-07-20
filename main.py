from database import setup_database
from crud.operations import create_record, read_records, delete_record, export_to_csv

def main():
    setup_database()
    tables = {1: 'sales', 2: 'inventory', 3: 'finance', 4: 'support', 5: 'logistics'}
    
    while True:
        print("\n=== ABC RETAIL GROUP - MLOPS DASHBOARD ===")
        print("1. View Data")
        print("2. Delete Data (by ID)")
        print("3. Export All Data to CSV (For Phase 2 Analysis)")
        print("4. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == '1':
            dept = int(input("Select Dept (1-Sales, 2-Inventory, 3-Finance, 4-Support, 5-Logistics): "))
            if dept in tables:
                read_records(tables[dept])
        elif choice == '2':
            dept = int(input("Select Dept (1-Sales, 2-Inventory, 3-Finance, 4-Support, 5-Logistics): "))
            if dept in tables:
                rec_id = input("Enter ID to delete: ")
                delete_record(tables[dept], rec_id)
        elif choice == '3':
            export_to_csv()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid Choice.")

if __name__ == "__main__":
    main()