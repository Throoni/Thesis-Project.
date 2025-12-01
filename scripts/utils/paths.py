"""
Path utilities for the thesis project.
Dynamically resolves project root and provides standardized paths.
"""
from pathlib import Path

# Get project root (2 levels up from this file: scripts/utils/paths.py -> scripts -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard data directories
RAW_DATA = PROJECT_ROOT / "raw_data"
PROCESSED_DATA = PROJECT_ROOT / "processed_data"
RESULTS = PROJECT_ROOT / "results"

# Common file paths
def get_raw_data_path(filename: str) -> Path:
    """Get path to a file in raw_data directory."""
    return RAW_DATA / filename

def get_processed_data_path(filename: str) -> Path:
    """Get path to a file in processed_data directory."""
    return PROCESSED_DATA / filename

def get_results_path(filename: str) -> Path:
    """Get path to a file in results directory."""
    return RESULTS / filename

