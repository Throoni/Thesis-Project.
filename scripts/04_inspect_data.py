import pandas as pd
import os

# Path to the file we just downloaded
file_path = "raw_data/crsp_stocks_full.parquet"

if os.path.exists(file_path):
    print(f"Opening {file_path}...")
    
    # Load the file
    df = pd.read_parquet(file_path)
    
    # Print summary
    print("\n--- DATA SHAPE ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())
    
    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())
    
    print("\n--- DATE RANGE ---")
    print(f"Start: {df['date'].min()}")
    print(f"End:   {df['date'].max()}")
    
else:
    print("File not found. Did the download finish?")
