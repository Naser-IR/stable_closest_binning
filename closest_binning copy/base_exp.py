import pandas as pd
import os
import time
import jenkspy

# Import your updated Algorithm 5
from closest import algorithm5_independent_graph

# ==========================================
# Helper Functions
# ==========================================
def _timed_call(fn, *args, **kwargs):
    """Run a function and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt

def get_base_cuts_and_c(df, attr_col, num_bins=10, step=1.0):
    """
    Calculates the total records for 'C', and extracts the inner 
    boundaries for Equi-Width, Equi-Depth, and Jenks.
    All boundaries are rounded to the grid step so they align.
    """
    total_records = len(df)
    c_value = total_records / 100.0
    
    # 1. Equi-Width
    _, width_edges = pd.cut(df[attr_col], bins=num_bins, retbins=True)
    width_cut = [round(float(x), 1) for x in width_edges[1:-1]]
    
    # 2. Equi-Depth
    _, depth_edges = pd.qcut(df[attr_col], q=num_bins, duplicates='drop', retbins=True)
    depth_cut = [round(float(x), 1) for x in depth_edges[1:-1]]
    
    # 3. Jenks Natural Breaks
    series = pd.to_numeric(df[attr_col], errors="coerce").dropna()
    
    # Take a sample if the dataset is too large so Jenks doesn't freeze
    if len(series) > 50000:
        sample_series = series.sample(n=50000, random_state=42)
    else:
        sample_series = series
        
    breaks = jenkspy.jenks_breaks(sample_series, n_classes=num_bins)
    jenks_cut = [round(float(b), 1) for b in breaks[1:-1]]
    
    return c_value, width_cut, depth_cut, jenks_cut

def is_exact_match(returned_cut, base_cut):
    """Helper to check if the algorithm's returned cut perfectly matches the base cut."""
    if returned_cut is None or base_cut is None:
        return False
    if len(returned_cut) != len(base_cut):
        return False
    # Using small tolerance for float comparison
    return all(abs(a - b) < 1e-5 for a, b in zip(returned_cut, base_cut))

# ==========================================
# Main Processing Loop
# ==========================================
def main_closest_comparison():
    input_dir = ""
    summary_output_file = "closest_stable_percentages_summary.csv"
    
    # --- Algorithm configuration (Static parameters) ---
    ATTR_COL = "AGE"     
    L, U, STEP = 0.0, 100.0, 1.0  
    USE_PREFILTER = True
    K = 10               
    MIN_WIDTH = 0             
    MAX_WIDTH = 100             
    EXCLUDED_BOUNDARIES = None  

    # --- Generate Dynamic Parameters ---
    eps_values = [3.0]
    multipliers = [round(2.5 + i * 0.1, 1) for i in range(11)]
    
    # --- Tracking Dictionary for Success Rates ---
    stats = {
        (eps, m): {
            "width_stable": 0, "width_total": 0, 
            "depth_stable": 0, "depth_total": 0,
            "jenks_stable": 0, "jenks_total": 0,
            "unconstrained_success": 0, "unconstrained_total": 0
        } 
        for eps in eps_values for m in multipliers
    }

    # Loop through all 56 state files
    for i in range(1, 57):
        file_num = f"{i:02d}"
        CSV_PATH = f"{input_dir}/state_{file_num}.csv"
        
        if not os.path.exists(CSV_PATH):
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing File {file_num}: {os.path.basename(CSV_PATH)}")
        print(f"{'='*60}")
        
        print("Loading data...")
        df = pd.read_csv(CSV_PATH, low_memory=False)
        series = pd.to_numeric(df[ATTR_COL], errors="coerce").dropna()
        
        current_c, base_width, base_depth, base_jenks = get_base_cuts_and_c(df, ATTR_COL, num_bins=K, step=STEP)
        
        expected_cut_len = K - 1
        
        # Test every combination of EPS and MULTIPLIER
        for eps in eps_values:
            for m in multipliers:
                C = current_c * eps * m
                
                # 1. --- EQUI-WIDTH ---
                if len(base_width) == expected_cut_len:
                    stats[(eps, m)]["width_total"] += 1
                    try:
                        # Notice we pass `series` as x_series, `None` as y_series, and `COUNT` as agg_func
                        (cuts_w, _) = _timed_call(
                            algorithm5_independent_graph, 
                            series, L, U, STEP, K, eps, C,
                            use_boundary_prefilter=USE_PREFILTER,
                            min_width=MIN_WIDTH,
                            max_width=MAX_WIDTH,
                            excluded_boundaries=EXCLUDED_BOUNDARIES,
                            base_cut=base_width
                        )
                        if is_exact_match(cuts_w, base_width): 
                            stats[(eps, m)]["width_stable"] += 1
                    except Exception as e:
                        pass 

                # 2. --- EQUI-DEPTH ---
                if len(base_depth) == expected_cut_len:
                    stats[(eps, m)]["depth_total"] += 1
                    try:
                        (cuts_d, _) = _timed_call(
                            algorithm5_independent_graph, 
                            series,L, U, STEP, K, eps, C,
                            use_boundary_prefilter=USE_PREFILTER,
                            min_width=MIN_WIDTH,
                            max_width=MAX_WIDTH,
                            excluded_boundaries=EXCLUDED_BOUNDARIES,
                            base_cut=base_depth
                        )
                        if is_exact_match(cuts_d, base_depth):
                            stats[(eps, m)]["depth_stable"] += 1
                    except Exception as e:
                        pass

                # 3. --- JENKS NATURAL BREAKS ---
                if len(base_jenks) == expected_cut_len:
                    stats[(eps, m)]["jenks_total"] += 1
                    try:
                        (cuts_j, _) = _timed_call(
                            algorithm5_independent_graph, 
                            series,L, U, STEP, K, eps, C,
                            use_boundary_prefilter=USE_PREFILTER,
                            min_width=MIN_WIDTH,
                            max_width=MAX_WIDTH,
                            excluded_boundaries=EXCLUDED_BOUNDARIES,
                            base_cut=base_jenks
                        )
                        if is_exact_match(cuts_j, base_jenks):
                            stats[(eps, m)]["jenks_stable"] += 1
                    except Exception as e:
                        pass
                    
                # 4. --- UNCONSTRAINED (Is ANY stable cut possible?) ---
                stats[(eps, m)]["unconstrained_total"] += 1
                try:
                    (cuts_u, _) = _timed_call(
                        algorithm5_independent_graph, 
                        series,L, U, STEP, K, eps, C,
                        use_boundary_prefilter=USE_PREFILTER,
                        min_width=MIN_WIDTH,
                        max_width=MAX_WIDTH,
                        excluded_boundaries=EXCLUDED_BOUNDARIES,
                        base_cut=None  # No base cut, just checking if ANY path exists
                    )
                    if cuts_u is not None:
                        stats[(eps, m)]["unconstrained_success"] += 1
                except Exception as e:
                    pass

    # ---------------------------------------------------------
    # Final Summary Calculation & File Save
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("CALCULATING FINAL PERCENTAGES AND SAVING TO CSV")
    print(f"{'='*60}")
    
    results_list = []
    
    for eps in eps_values:
        print(f"\n--- Results for EPS = {eps} ---")
        for m in multipliers:
            w_total = stats[(eps, m)]["width_total"]
            w_stable = stats[(eps, m)]["width_stable"]
            
            d_total = stats[(eps, m)]["depth_total"]
            d_stable = stats[(eps, m)]["depth_stable"]
            
            j_total = stats[(eps, m)]["jenks_total"]
            j_stable = stats[(eps, m)]["jenks_stable"]
            
            u_total = stats[(eps, m)]["unconstrained_total"]
            u_success = stats[(eps, m)]["unconstrained_success"]
            
            w_pct = (w_stable / w_total * 100) if w_total > 0 else 0.0
            d_pct = (d_stable / d_total * 100) if d_total > 0 else 0.0
            j_pct = (j_stable / j_total * 100) if j_total > 0 else 0.0
            u_pct = (u_success / u_total * 100) if u_total > 0 else 0.0
            
            results_list.append({
                "Epsilon": eps,
                "C_Multiplier": m,
                "Equi_Width_Total": w_total,
                "Equi_Width_Perfect_Stable": w_stable,
                "Equi_Width_Stable_Pct": round(w_pct, 2),
                
                "Equi_Depth_Total": d_total,
                "Equi_Depth_Perfect_Stable": d_stable,
                "Equi_Depth_Stable_Pct": round(d_pct, 2),
                
                "Jenks_Total": j_total,
                "Jenks_Perfect_Stable": j_stable,
                "Jenks_Stable_Pct": round(j_pct, 2),
                
                "Any_Stable_Cut_Total": u_total,
                "Any_Stable_Cut_Found": u_success,
                "Any_Stable_Cut_Pct": round(u_pct, 2)
            })
            
            print(f"  Multiplier {m}x -> Width: {w_pct:.1f}% | Depth: {d_pct:.1f}% | Jenks: {j_pct:.1f}% | Any: {u_pct:.1f}%")
        
    # Save to CSV
    summary_df = pd.DataFrame(results_list)
    summary_df = summary_df.sort_values(by=["Epsilon", "C_Multiplier"]) 
    summary_df.to_csv(summary_output_file, index=False)
    print(f"\nSuccess! Results saved to '{summary_output_file}'.")


if __name__ == "__main__":
    main_closest_comparison()
