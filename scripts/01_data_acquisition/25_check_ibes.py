import wrds
from config import WRDS_USERNAME

def check_ibes_access():
    print(f"--- Connecting to WRDS as {WRDS_USERNAME} ---")
    try:
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    print("\n[TEST] Checking I/B/E/S Access...")
    
    # We try to read 1 row from the Summary Statistics file (Consensus Estimates)
    # This is the most common table used for "Earnings Surprise"
    query = "SELECT * FROM ibes.statsum_epsus LIMIT 1"
    
    try:
        df = db.raw_sql(query)
        print("[SUCCESS] You have I/B/E/S access! ✅")
        print("We can build the full 94 predictors (including Analyst Surprises).")
    except Exception as e:
        print("[FAILURE] No I/B/E/S access. ❌")
        print(f"Error Details: {e}")
        print("\nResult: We will build ~87 predictors (CRSP + Compustat only).")

    db.close()

if __name__ == "__main__":
    check_ibes_access()