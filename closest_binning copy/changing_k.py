import time
import numpy as np
import pandas as pd

# Ensure all algorithms are imported from your module
from closest import (
    algorithm5_independent_graph, 
    algorithm4_closest_independent_dp,
    algorithm6_closest_naive_bfs
)

# ----------------- Tunables -----------------
# CSV_PATH = ""
CSV_PATH = ""
ATTR_COL = "Time"

# --- Output file for execution times ---
OUTPUT_CSV_FILE = "algorithm_comparison_runtimes_k_experiment.csv"

L, U, STEP = 0.0, 24.0, 1.0
USE_PREFILTER = True

# --- Set your specific k and epsilon ranges here ---
K_VALUES = list(range(3, 11))            # k from 3 up to 10
EPS_VALUES = [1.0]                       # List of epsilon values to test

# Base C value. For each iteration, C will dynamically be calculated as (current_eps * C_BASE)
C_BASE = 90000 

# --- Set your Constraints here ---
MIN_WIDTH = 0                # Bins must be at least this wide
MAX_WIDTH = 24               # Bins cannot be wider than this
EXCLUDED_BOUNDARIES = None   # Boundaries that are strictly forbidden
REQUIRED_BOUNDARIES = None   # Boundaries that MUST be selected
# -------------------------------------------


def _timed_call(fn, *args, **kwargs):
    """Run a function and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt

def generate_base_cuts(series: pd.Series, k: int, L: float, U: float, step: float) -> dict:
    """Generates k-1 internal base cuts and snaps them to the valid grid."""
    
    def snap_to_valid_grid(cuts, L, U, step, k):
        """Snaps float cuts to the nearest grid step and resolves duplicates."""
        edges = np.arange(L, U + step, step)
        indices = [(np.abs(edges - c)).argmin() for c in cuts]
        
        # Remove duplicates
        unique_indices = sorted(list(set(indices)))
        
        # Ensure they are internal boundaries (not 0 or max)
        unique_indices = [i for i in unique_indices if 0 < i < len(edges) - 1]
        
        # If we lost boundaries due to collisions, fill in from available slots
        if len(unique_indices) < k - 1:
            available = [i for i in range(1, len(edges) - 1) if i not in unique_indices]
            needed = (k - 1) - len(unique_indices)
            unique_indices.extend(available[:needed])
            unique_indices.sort()
            
        return [float(edges[i]) for i in unique_indices]

    cuts = {}
    
    # 1. Equi-width
    raw_width = list(np.linspace(L, U, k + 1)[1:-1])
    cuts['Width'] = snap_to_valid_grid(raw_width, L, U, step, k)
    
    # 2. Equi-depth (Quantiles)
    sorted_x = np.sort(series.dropna().values)
    n = len(sorted_x)
    depth_indices = [int(i * n / k) for i in range(1, k)]
    raw_depth = [float(sorted_x[idx]) for idx in depth_indices]
    cuts['Depth'] = snap_to_valid_grid(raw_depth, L, U, step, k)
    
    # 3. Jenks
    if n > 10000:
        jenks_sample = np.random.choice(sorted_x, size=10000, replace=False)
        jenks_sample.sort()
    else:
        jenks_sample = sorted_x

    try:
        import jenkspy
        j_breaks = jenkspy.jenks_breaks(jenks_sample, n_classes=k)
        raw_jenks = [float(x) for x in j_breaks[1:-1]]
        cuts['Jenks'] = snap_to_valid_grid(raw_jenks, L, U, step, k)
    except ImportError:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(jenks_sample.reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())
        raw_jenks = [(centers[i] + centers[i+1])/2 for i in range(k-1)]
        cuts['Jenks'] = snap_to_valid_grid(raw_jenks, L, U, step, k)
    except Exception:
        cuts['Jenks'] = cuts['Width']

    return cuts


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
    print(f"{'='*50}")

    results_list = []

    for k in K_VALUES:
        # Generate the three types of base cuts for the current k
        print(f"\nGenerating Base Cuts for k = {k}...")
        base_cuts_dict = generate_base_cuts(x_series, k, L, U, STEP)
        print(f"  Width Cuts: {base_cuts_dict['Width']}")
        print(f"  Depth Cuts: {base_cuts_dict['Depth']}")
        print(f"  Jenks Cuts: {base_cuts_dict['Jenks']}")

        for eps in EPS_VALUES:
            current_c = eps * C_BASE
            
            print(f"\n\n{'#'*70}")
            print(f"Evaluating EXACTLY k = {k} | EPSILON = {eps} | C = {current_c}")
            print(f"{'#'*70}\n")
            
            # This dictionary will store a single row for the CSV representing this k and eps
            row_data = {
                "k": k,
                "Epsilon": eps,
                "C_Threshold": current_c
            }

            for cut_type, base_cut in base_cuts_dict.items():
                print(f"\n{'*'*40}")
                print(f" BASE CUT TYPE: {cut_type} -> {base_cut}")
                print(f"{'*'*40}")

                # ---------- Algorithm 4 (Dynamic Programming) ----------
                print("--- Algorithm 4 (DP / Closest Cut) ---")
                try:
                    res4, t4 = _timed_call(
                        algorithm4_closest_independent_dp,
                        x_series, L, U, STEP, k, eps, current_c,
                        base_cut,
                        min_width=MIN_WIDTH, max_width=MAX_WIDTH, excluded_boundaries=EXCLUDED_BOUNDARIES
                    )
                    cuts4, dist_alg4 = res4
                    ans_found4 = cuts4 is not None
                    print(f"  Time: {t4:.6f}s | Found: {ans_found4} | Dist: {dist_alg4} | Cuts: {cuts4}")
                    
                    row_data[f"DP_Time_{cut_type}"] = round(t4, 6)
                    row_data[f"DP_Ans_{cut_type}"] = ans_found4
                    row_data[f"DP_Dist_{cut_type}"] = dist_alg4
                except Exception as e:
                    print(f"  Error: {repr(e)}")
                    row_data[f"DP_Time_{cut_type}"] = "ERROR"
                    row_data[f"DP_Ans_{cut_type}"] = "ERROR"
                    row_data[f"DP_Dist_{cut_type}"] = "ERROR"

                # ---------- Algorithm 5 (Graph/DFS) ----------
                print("--- Algorithm 5 (Graph / Closest Path) ---")
                try:
                    res5, t5 = _timed_call(
                        algorithm5_independent_graph,
                        x_series, L, U, STEP, k, eps, current_c,
                        use_boundary_prefilter=USE_PREFILTER,
                        min_width=MIN_WIDTH, max_width=MAX_WIDTH,
                        excluded_boundaries=EXCLUDED_BOUNDARIES, required_boundaries=REQUIRED_BOUNDARIES,
                        base_cut=base_cut
                    )
                    cuts5, _ = res5 if isinstance(res5, tuple) else (res5, None)
                    ans_found5 = cuts5 is not None
                    print(f"  Time: {t5:.6f}s | Found: {ans_found5} | Cuts: {cuts5}")
                    
                    row_data[f"Graph_Time_{cut_type}"] = round(t5, 6)
                    row_data[f"Graph_Ans_{cut_type}"] = ans_found5
                except Exception as e:
                    print(f"  Error: {repr(e)}")
                    row_data[f"Graph_Time_{cut_type}"] = "ERROR"
                    row_data[f"Graph_Ans_{cut_type}"] = "ERROR"

                # ---------- Algorithm 6 (Naive BFS) ----------
                print("--- Algorithm 6 (Naive BFS / Closest Cut) ---")
                try:
                    res6, t6 = _timed_call(
                        algorithm6_closest_naive_bfs,
                        x_series, L, U, STEP, k, eps, current_c,
                        base_cut,
                        min_width=MIN_WIDTH, max_width=MAX_WIDTH, excluded_boundaries=EXCLUDED_BOUNDARIES
                    )
                    cuts6, dist_alg6 = res6
                    ans_found6 = cuts6 is not None
                    print(f"  Time: {t6:.6f}s | Found: {ans_found6} | Dist: {dist_alg6} | Cuts: {cuts6}")
                    
                    row_data[f"Naive_Time_{cut_type}"] = round(t6, 6)
                    row_data[f"Naive_Ans_{cut_type}"] = ans_found6
                    row_data[f"Naive_Dist_{cut_type}"] = dist_alg6
                except Exception as e:
                    print(f"  Error: {repr(e)}")
                    row_data[f"Naive_Time_{cut_type}"] = "ERROR"
                    row_data[f"Naive_Ans_{cut_type}"] = "ERROR"
                    row_data[f"Naive_Dist_{cut_type}"] = "ERROR"

            # Append the aggregated row to the final list
            results_list.append(row_data)
                
    print(f"\n{'='*50}")
    print(f"Finished evaluating all combinations.")
    
    # Save directly to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Execution times and results saved successfully to '{OUTPUT_CSV_FILE}'")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
