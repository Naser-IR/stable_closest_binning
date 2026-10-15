import time
import pandas as pd

# Ensure all algorithms are imported from your module
from closest import (
    algorithm5_independent_graph, 
    algorithm4_closest_independent_dp,
    algorithm6_closest_naive_bfs
)

# ----------------- Tunables -----------------
# CSV_PATH = "Crimes_rounded.csv"
CSV_PATH = ""
ATTR_COL = "Time"
Y_COL = None  # Set this to a column name if you use SUM, AVG, MIN, or MAX

# --- Output file for execution times ---
OUTPUT_CSV_FILE = "algorithm_comparison_runtimes.csv"

L, U, STEP = 0.0, 24.0, 1.0
USE_PREFILTER = True

# --- Set your specific k and epsilon ranges here ---
K_VALUES = [6]                           # List of k values to test
EPS_VALUES = [1.0]                       # List of epsilon values to test

# Base C value. For each iteration, C will dynamically be calculated as (current_eps * C_BASE)
C_BASE = 20000 

# --- Set your Constraints here ---
MIN_WIDTH = 0                # Bins must be at least this wide
MAX_WIDTH = 24               # Bins cannot be wider than this
EXCLUDED_BOUNDARIES = [11]   # Boundaries that are strictly forbidden
REQUIRED_BOUNDARIES = None   # Boundaries that MUST be selected

# --- NEW: Base Cut to map closest distances to ---
BASE_CUT = [3.0,6.0, 12.0,15.0, 18.0] 
# -------------------------------------------


def _timed_call(fn, *args, **kwargs):
    """Run a function and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def main():
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    x_series = pd.to_numeric(df[ATTR_COL], errors="coerce").dropna()
    
    print(f"\n{'='*50}")
    print(f"Global Constraints Active:")
    print(f"  - Min Width: {MIN_WIDTH}")
    print(f"  - Max Width: {MAX_WIDTH}")
    print(f"  - Forbidden: {EXCLUDED_BOUNDARIES}")
    print(f"  - Required:  {REQUIRED_BOUNDARIES}")
    print(f"  - Base Cut:  {BASE_CUT}")
    print(f"{'='*50}")

    results_list = []

    for k in K_VALUES:
        for eps in EPS_VALUES:
            current_c = eps * C_BASE
            
            print(f"\n\n{'#'*60}")
            print(f"Evaluating EXACTLY k = {k} | EPSILON = {eps} | C = {current_c}")
            print(f"{'#'*60}\n")
            
            time_alg4 = None
            dist_alg4 = None
            time_alg5 = None
            time_alg6 = None
            dist_alg6 = None

            # ---------- Algorithm 4 (Dynamic Programming) ----------
            print("--- Algorithm 4 (DP / Closest Cut) ---")
            try:
                res4, t4 = _timed_call(
                    algorithm4_closest_independent_dp,
                    x_series,
                    L, U, STEP, k, eps, current_c,
                    BASE_CUT,
                    min_width=MIN_WIDTH,
                    max_width=MAX_WIDTH,
                    excluded_boundaries=EXCLUDED_BOUNDARIES
                )
                cuts4, dist_alg4 = res4
                print(f"Time Taken: {t4:.6f} seconds")
                print(f"Stable Cuts Found: {cuts4} | Displacement Distance: {dist_alg4}")
                time_alg4 = round(t4, 6)
            except Exception as e:
                print(f"Error: {repr(e)}")
                time_alg4 = "ERROR"
                dist_alg4 = "ERROR"

            print("\n")

            # ---------- Algorithm 5 (Graph/DFS) ----------
            print("--- Algorithm 5 (Independent Graph / Closest Path) ---")
            try:
                res5, t5 = _timed_call(
                    algorithm5_independent_graph,
                    x_series,
                    L, U, STEP, k, eps, current_c,
                    use_boundary_prefilter=USE_PREFILTER,
                    min_width=MIN_WIDTH,
                    max_width=MAX_WIDTH,
                    excluded_boundaries=EXCLUDED_BOUNDARIES,
                    required_boundaries=REQUIRED_BOUNDARIES,
                    base_cut=BASE_CUT
                )
                
                cuts5, _ = res5 if isinstance(res5, tuple) else (res5, None)
                print(f"Time Taken: {t5:.6f} seconds")
                print(f"Stable Cuts Found: {cuts5}")
                time_alg5 = round(t5, 6)
            except Exception as e:
                print(f"Error: {repr(e)}")
                time_alg5 = "ERROR"
                
            print("\n")

            # ---------- Algorithm 6 (Naive BFS) ----------
            print("--- Algorithm 6 (Naive BFS / Closest Cut) ---")
            try:
                res6, t6 = _timed_call(
                    algorithm6_closest_naive_bfs,
                    x_series,
                    L, U, STEP, k, eps, current_c,
                    BASE_CUT,
                    min_width=MIN_WIDTH,
                    max_width=MAX_WIDTH,
                    excluded_boundaries=EXCLUDED_BOUNDARIES
                )
                cuts6, dist_alg6 = res6
                print(f"Time Taken: {t6:.6f} seconds")
                print(f"Stable Cuts Found: {cuts6} | Displacement Distance: {dist_alg6}")
                time_alg6 = round(t6, 6)
            except Exception as e:
                print(f"Error: {repr(e)}")
                time_alg6 = "ERROR"
                dist_alg6 = "ERROR"
            
            # Store results
            results_list.append({
                "k": k,
                "Epsilon": eps,
                "C_Threshold": current_c,
                "Alg4_Time_sec": time_alg4,
                "Alg4_Distance": dist_alg4,
                "Alg5_Time_sec": time_alg5,
                "Alg6_Time_sec": time_alg6,
                "Alg6_Distance": dist_alg6
            })
                
    print(f"\n{'='*50}")
    print(f"Finished evaluating all combinations.")
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Execution times saved successfully to '{OUTPUT_CSV_FILE}'")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()