import time
import signal
import numpy as np
import pandas as pd
from contextlib import contextmanager

from closest_gl import (
    algorithm5_independent_graph, 
    algorithm4_closest_independent_dp,
    algorithm6_closest_naive_bfs
)

# ----------------- Tunables -----------------
# CSV_PATH = ""
# ATTR_COL = "age"
# Y_COL = "income" 
CSV_PATH = ""
ATTR_COL = "song_popularity"
Y_COL = "song_duration_ms"  # Optional: Set to None if not needed

OUTPUT_CSV_FILE = "algorithm_SUM_C_comparison_runtimes_full_experiment.csv"

L, U, STEP = 0.0, 100.0, 1.0
USE_PREFILTER = True

# --- Experiment Ranges ---
AGG_FUNCS = ['SUM']  # List of aggregation functions to test
K_VALUES = [5.0]                        
EPS_VALUES = [3.0,]                  
C_VALUES = [41100152, 61100152, 81100152, 101100152, 121100152]         

MIN_WIDTH = 0                
MAX_WIDTH = 100               
EXCLUDED_BOUNDARIES = None   
REQUIRED_BOUNDARIES = None   
TIMEOUT_SECONDS = 120      # 2 minutes timeout limit
# -------------------------------------------

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to enforce a strict time limit on execution."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds.")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def _timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    with time_limit(TIMEOUT_SECONDS):
        out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def generate_base_cuts(L: float, U: float, step: float, k: int) -> dict:
    """Generates k-1 internal base cuts for Equi-width only and snaps them to the valid grid."""
    def snap_to_valid_grid(cuts, L, U, step, k):
        edges = np.arange(L, U + step, step)
        indices = [(np.abs(edges - c)).argmin() for c in cuts]
        
        unique_indices = sorted(list(set(indices)))
        unique_indices = [i for i in unique_indices if 0 < i < len(edges) - 1]
        
        if len(unique_indices) < k - 1:
            available = [i for i in range(1, len(edges) - 1) if i not in unique_indices]
            needed = (k - 1) - len(unique_indices)
            unique_indices.extend(available[:needed])
            unique_indices.sort()
            
        return [float(edges[i]) for i in unique_indices]

    cuts = {}
    
    # 1. Equi-width ONLY
    # raw_width = list(np.linspace(L, U, k + 1)[1:-1])
    raw_width = list(np.linspace(L, U, int(k) + 1)[1:-1])
    cuts['Width'] = snap_to_valid_grid(raw_width, L, U, step, k)
    
    return cuts


def main():
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    x_series = pd.to_numeric(df[ATTR_COL], errors="coerce").dropna()
    
    y_series = None
    if Y_COL and Y_COL in df.columns:
        y_series = pd.to_numeric(df[Y_COL], errors="coerce").dropna()
        # Align indices if there were NaNs
        common_idx = x_series.index.intersection(y_series.index)
        x_series = x_series.loc[common_idx]
        y_series = y_series.loc[common_idx]
    
    print(f"\n{'='*50}")
    print(f"Global Constraints Active:")
    print(f"  - Min Width: {MIN_WIDTH}")
    print(f"  - Max Width: {MAX_WIDTH}")
    print(f"  - Forbidden: {EXCLUDED_BOUNDARIES}")
    print(f"  - Timeout limit: {TIMEOUT_SECONDS}s")
    print(f"{'='*50}")

    results_list = []

    for agg_func in AGG_FUNCS:
        
        if agg_func != 'COUNT' and y_series is None:
            print(f"\n[!] Skipping {agg_func}: Y_COL '{Y_COL}' not found or invalid.")
            continue

        for k in K_VALUES:
            k = int(k)  # <--- ADD THIS LINE to force k to be an integer
            print(f"\nGenerating Base Cuts for k = {k}...")
            base_cuts_dict = generate_base_cuts(L, U, STEP, k)

            for eps in EPS_VALUES:
                for current_c in C_VALUES:
                    current_c = current_c * eps  # Ensure current_c is a float or int as needed
                    print(f"\n\n{'#'*80}")
                    print(f"Eval: AGG = {agg_func} | k = {k} | EPS = {eps} | C = {current_c}")
                    print(f"{'#'*80}\n")
                    
                    row_data = {
                        "Agg_Func": agg_func,
                        "k": k,
                        "Epsilon": eps,
                        "C_Threshold": current_c
                    }

                    for cut_type, base_cut in base_cuts_dict.items():
                        print(f"\n{'*'*40}")
                        print(f" BASE CUT TYPE: {cut_type} -> {base_cut}")
                        print(f"{'*'*40}")

                        # ---------- Algorithm 4 (DP) ----------
                        print("--- Algorithm 4 (DP / Closest Cut) ---")
                        try:
                            res4, t4 = _timed_call(
                                algorithm4_closest_independent_dp,
                                x_series, y_series, L, U, STEP, k, eps, current_c,
                                base_cut, agg_func=agg_func,
                                min_width=MIN_WIDTH, max_width=MAX_WIDTH, excluded_boundaries=EXCLUDED_BOUNDARIES
                            )
                            cuts4, dist_alg4 = res4
                            ans_found4 = cuts4 is not None
                            print(f"  Time: {t4:.6f}s | Found: {ans_found4} | Dist: {dist_alg4}")
                            
                            row_data[f"DP_Time_{cut_type}"] = round(t4, 6)
                            row_data[f"DP_Ans_{cut_type}"] = ans_found4
                            row_data[f"DP_Dist_{cut_type}"] = dist_alg4
                        except TimeoutException:
                            print(f"  [TIMEOUT] Exceeded {TIMEOUT_SECONDS} seconds.")
                            row_data[f"DP_Time_{cut_type}"] = "TIMEOUT"
                            row_data[f"DP_Ans_{cut_type}"] = "TIMEOUT"
                            row_data[f"DP_Dist_{cut_type}"] = "TIMEOUT"
                        except Exception as e:
                            print(f"  Error: {repr(e)}")
                            row_data[f"DP_Time_{cut_type}"] = "ERROR"
                            row_data[f"DP_Ans_{cut_type}"] = "ERROR"
                            row_data[f"DP_Dist_{cut_type}"] = "ERROR"

                        # ---------- Algorithm 5 (Graph) ----------
                        print("--- Algorithm 5 (Graph / Closest Path) ---")
                        try:
                            res5, t5 = _timed_call(
                                algorithm5_independent_graph,
                                x_series, y_series, L, U, STEP, k, eps, current_c, agg_func,
                                use_boundary_prefilter=USE_PREFILTER,
                                min_width=MIN_WIDTH, max_width=MAX_WIDTH,
                                excluded_boundaries=EXCLUDED_BOUNDARIES, required_boundaries=REQUIRED_BOUNDARIES,
                                base_cut=base_cut
                            )
                            cuts5, _ = res5 if isinstance(res5, tuple) else (res5, None)
                            ans_found5 = cuts5 is not None
                            print(f"  Time: {t5:.6f}s | Found: {ans_found5}")
                            
                            row_data[f"Graph_Time_{cut_type}"] = round(t5, 6)
                            row_data[f"Graph_Ans_{cut_type}"] = ans_found5
                        except TimeoutException:
                            print(f"  [TIMEOUT] Exceeded {TIMEOUT_SECONDS} seconds.")
                            row_data[f"Graph_Time_{cut_type}"] = "TIMEOUT"
                            row_data[f"Graph_Ans_{cut_type}"] = "TIMEOUT"
                        except Exception as e:
                            print(f"  Error: {repr(e)}")
                            row_data[f"Graph_Time_{cut_type}"] = "ERROR"
                            row_data[f"Graph_Ans_{cut_type}"] = "ERROR"

                        # ---------- Algorithm 6 (Naive) ----------
                        print("--- Algorithm 6 (Naive BFS / Closest Cut) ---")
                        try:
                            res6, t6 = _timed_call(
                                algorithm6_closest_naive_bfs,
                                x_series, y_series, L, U, STEP, k, eps, current_c,
                                base_cut, agg_func=agg_func,
                                min_width=MIN_WIDTH, max_width=MAX_WIDTH, excluded_boundaries=EXCLUDED_BOUNDARIES
                            )
                            cuts6, dist_alg6 = res6
                            ans_found6 = cuts6 is not None
                            print(f"  Time: {t6:.6f}s | Found: {ans_found6} | Dist: {dist_alg6}")
                            
                            row_data[f"Naive_Time_{cut_type}"] = round(t6, 6)
                            row_data[f"Naive_Ans_{cut_type}"] = ans_found6
                            row_data[f"Naive_Dist_{cut_type}"] = dist_alg6
                        except TimeoutException:
                            print(f"  [TIMEOUT] Exceeded {TIMEOUT_SECONDS} seconds.")
                            row_data[f"Naive_Time_{cut_type}"] = "TIMEOUT"
                            row_data[f"Naive_Ans_{cut_type}"] = "TIMEOUT"
                            row_data[f"Naive_Dist_{cut_type}"] = "TIMEOUT"
                        except Exception as e:
                            print(f"  Error: {repr(e)}")
                            row_data[f"Naive_Time_{cut_type}"] = "ERROR"
                            row_data[f"Naive_Ans_{cut_type}"] = "ERROR"
                            row_data[f"Naive_Dist_{cut_type}"] = "ERROR"

                    results_list.append(row_data)
                
    print(f"\n{'='*50}")
    print(f"Finished evaluating all combinations.")
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Execution times and results saved successfully to '{OUTPUT_CSV_FILE}'")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()