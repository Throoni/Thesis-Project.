import wrds
from config import WRDS_USERNAME

print("Connecting...")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

def check_table(schema, table):
    try:
        # Just try to read 1 row to see if we have access
        db.raw_sql(f"SELECT * FROM {schema}.{table} LIMIT 1")
        print(f"[SUCCESS] Found: {schema}.{table}")
        return True
    except Exception as e:
        print(f"[FAILED]  {schema}.{table} -> {str(e).splitlines()[0]}")
        return False

print("\n--- CHECKING LINK TABLES ---")
check_table("wrdsapps", "op_crsp_link")       # Modern Link
check_table("wrdsapps", "link_crsp_optionm")  # Old Link
check_table("home", "link_crsp_optionm")      # Backup Link

print("\n--- CHECKING OPTION TABLES ---")
check_table("optionm", "secnmd")              # Security Names (for manual linking)
check_table("optionm", "security")            # Alternative Name
check_table("optionm", "opprcd2020")          # Price Data (checking 2020 access)

print("\n--- DIAGNOSTIC COMPLETE ---")
db.close()