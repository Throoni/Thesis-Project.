import pandas as pd
import os

# --- SETTINGS ---
GHZ_FILE = "processed_data/ghz_predictors_87.parquet"       # Output of Script 26
FINAL_FILE = "processed_data/thesis_dataset_macro.parquet"   # Output of Script 20

def audit_file(filepath, name):
    print(f"\n🔎 AUDITING: {name}")
    print(f"   Path: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"   [ERROR] File not found.")
        return

    df = pd.read_parquet(filepath)
    cols = df.columns.tolist()
    
    # Check for specific "New" variables we expect
    key_vars = ['oper_prof', 'rd_sale', 'debt_assets', 'cash_ratio', 'mom12m', 'bm']
    found_vars = [v for v in key_vars if v in cols]
    
    print(f"   Rows: {len(df):,}")
    print(f"   Total Columns: {len(cols)}")
    print(f"   Key Variables Found: {len(found_vars)}/{len(key_vars)} -> {found_vars}")
    
    if len(cols) < 60:
        print("   ⚠️  WARNING: Column count is low. We are missing predictors.")
        # Print missing numeric columns to see what we actually have
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        print(f"   All Numeric Columns: {numeric}")
    else:
        print("   ✅ STATUS: Looks healthy (High column count).")

if __name__ == "__main__":
    print("--- DIAGNOSTIC MODE ---")
    
    # Check 1: Did Script 26 build the features correctly?
    audit_file(GHZ_FILE, "GHZ Predictors (Step 26)")
    
    # Check 2: Did Script 20 merge them correctly?
    audit_file(FINAL_FILE, "Final Dataset (Step 20)")
    
    print("\n-----------------------")