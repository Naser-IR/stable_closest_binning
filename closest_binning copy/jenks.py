import pandas as pd
import jenkspy

def run_jenks_binning(csv_path: str, attr_col: str, k_values: list):
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Extract the column and remove any NaN values
    time_series = pd.to_numeric(df[attr_col], errors="coerce").dropna()
    
    print(f"Total valid rows analyzed: {len(time_series)}")
    print("=" * 50)
    
    for k in k_values:
        # Calculate Jenks Natural Breaks
        # n_classes=k creates exactly k bins, returning k+1 boundary points
        breaks = jenkspy.jenks_breaks(time_series, n_classes=k)
        
        # Strip the first (min) and last (max) values to get the internal cuts
        internal_cuts = [float(b) for b in breaks[1:-1]]
        
        print(f"Results for k = {k} bins:")
        print(f"  Full Boundaries (including min/max): {breaks}")
        print(f"  Internal Cuts (Your BASE_CUT):       {internal_cuts}")
        print("-" * 50)

if __name__ == "__main__":
    # --- Configuration ---
    CSV_PATH = ""
    ATTR_COL = "Time"
    K_VALUES = [4]  # Add or change the K values you want to test
    
    run_jenks_binning(CSV_PATH, ATTR_COL, K_VALUES)


