from functools import lru_cache
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import time
import gc

# ============================================================
# Helpers
# ============================================================

def build_grid(series: pd.Series, L: float, U: float, step: float) -> Tuple[np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if U <= L:
        raise ValueError("U must be > L")

    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()

    m = int(np.ceil((U - L) / step))
    edges = (L + step * np.arange(m + 1)).astype(float)

    counts, _ = np.histogram(vals, bins=edges)

    pref = np.zeros(m + 1, dtype=int)
    pref[1:] = np.cumsum(counts, dtype=int)
    return edges, pref


def _boundary_values_to_indices(edges: np.ndarray, values: Optional[List[float]], m: int) -> Optional[set]:
    if values is None:
        return None

    edge_to_idx = {round(float(edges[i]), 12): i for i in range(len(edges))}
    out = set()

    for x in values:
        xf = round(float(x), 12)
        if xf not in edge_to_idx:
            raise ValueError(f"Boundary {x} is not on the grid. Grid values are {list(edges)}")

        idx = edge_to_idx[xf]
        if idx == 0 or idx == m:
            continue
        out.add(idx)

    return out


def _min_width_to_cells(min_width: Optional[float], step: float) -> Optional[int]:
    if min_width is None:
        return None
    return max(1, int(np.ceil(min_width / step)))


def _max_width_to_cells(max_width: Optional[float], step: float) -> Optional[int]:
    if max_width is None:
        return None
    return max(1, int(np.floor(max_width / step)))


def build_DEV_2d(pref: np.ndarray, s: int) -> np.ndarray:
    """Original matrix builder required for the unoptimized Algorithm 5"""
    m = len(pref) - 1
    INF = int(pref[m]) + 10**9
    DEV = np.full((m + 1, m + 1), INF, dtype=int)

    def cnt(a: int, b: int) -> int:
        return int(pref[b] - pref[a])

    for l in range(0, m):
        for r in range(l + 1, m + 1):
            base = cnt(l, r)
            max_err = -1
            valid_interval = False
            
            for dL in range(-s, s + 1):
                for dR in range(-s, s + 1):
                    Ls = l + dL
                    Rs = r + dR
                    
                    valid = True
                    if l == 0 and dL != 0: valid = False
                    if r == m and dR != 0: valid = False
                    if l != 0 and r != m:
                        if not (0 < Ls < m) or not (0 < Rs < m): valid = False
                    if not (0 <= Ls < m and 0 < Rs <= m): valid = False
                    if not (Ls < Rs): valid = False
                    
                    if valid:
                        shifted = cnt(Ls, Rs)
                        err = abs(base - shifted)
                        if err > max_err:
                            max_err = err
                        valid_interval = True
            
            if valid_interval:
                DEV[l, r] = max_err
                
    return DEV


# ============================================================
# Algorithm 5 — Original Graph DFS (Uses Dense DEV Matrix)
# ============================================================

def algorithm5_independent_graph(
    series: pd.Series,
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    *,
    use_boundary_prefilter: bool = True,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
    required_boundaries: Optional[List[float]] = None,
    base_cut: Optional[List[float]] = None,
) -> Optional[List[float]]:

    if base_cut is not None and len(base_cut) != k - 1:
        raise ValueError(f"For k={k} bins, base_cut must contain exactly {k-1} boundaries.")

    edges, pref = build_grid(series, L, U, step)
    m = len(edges) - 1
    s = int(np.floor(epsilon / step))

    if s < 0 or k < 1 or k > m: return None

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
    if k * (min_cells or 1) > m: return None
    if max_cells is not None and (min_cells or 1) > max_cells: return None

    excluded_idx = _boundary_values_to_indices(edges, excluded_boundaries, m)
    if excluded_idx is None: excluded_idx = set()
        
    required_idx = _boundary_values_to_indices(edges, required_boundaries, m)
    if required_idx is None: required_idx = set()
    
    if len(required_idx) > k - 1: return None
    if excluded_idx and required_idx.intersection(excluded_idx): return None

    required_list = sorted(required_idx)

    # DENSE MATRIX BUILD
    DEV = build_DEV_2d(pref, s)
    INF = int(pref[m]) + 10**9

    def boundary_allowed(j: int) -> bool:
        return j not in excluded_idx

    if use_boundary_prefilter:
        cand_interval = (DEV < C) & (DEV < INF)
        P = [0, m]
        for j in range(1, m):
            if not boundary_allowed(j): continue
            if cand_interval[:j, j].any() and cand_interval[j, j + 1:].any():
                P.append(j)
        P = np.array(sorted(set(P)), dtype=int)

    def get_next_required(i: int) -> int:
        for p in required_list:
            if p > i: return p
        return m + 1

    @lru_cache(maxsize=None)
    def dfs(i: int, bins_left: int) -> Optional[Tuple[float, Tuple[int, ...]]]:
        if bins_left == 1:
            if get_next_required(i) <= m: 
                return None
            
            last_w = m - i
            if min_cells is not None and last_w < min_cells: return None
            if max_cells is not None and last_w > max_cells: return None

            if DEV[i, m] < C:
                return (0.0, (i, m))
            return None

        lo = i + (min_cells or 1)
        hi = m - (bins_left - 1) * (min_cells or 1)
        if max_cells is not None: 
            hi = min(hi, i + max_cells)
            
        next_req = get_next_required(i)
        hi = min(hi, next_req)

        if use_boundary_prefilter:
            valid_js = P[(P >= lo) & (P <= hi)]
        else:
            valid_js = range(lo, hi + 1)
            
        best_cost = float('inf')
        best_path = None

        for j in valid_js:
            if not use_boundary_prefilter and not boundary_allowed(j): 
                continue

            if DEV[i, j] >= C:
                continue
            
            tail = dfs(j, bins_left - 1)
            if tail is not None:
                tail_cost, tail_path = tail
                
                if base_cut is not None:
                    cut_idx = k - bins_left
                    step_cost = abs(edges[j] - base_cut[cut_idx])
                else:
                    step_cost = 0.0
                
                total_cost = step_cost + tail_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = (i,) + tail_path

        if best_path is None:
            return None
            
        return (best_cost, best_path)

    result = dfs(0, k)
    if result is None: return None
    
    best_cost, path = result
    bnds = list(path)
    cuts_idx = bnds[1:-1]
    
    return [float(edges[c]) for c in cuts_idx]


# ============================================================
# Algorithm 5 — Optimized Graph DFS (On-the-fly, No Dense Matrix)
# ============================================================

def algorithm5_graph_edge_consistent_optimized(
    series: pd.Series,
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    *,
    use_boundary_prefilter: bool = True,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
    required_boundaries: Optional[List[float]] = None,
    base_cut: Optional[List[float]] = None,
) -> Optional[List[float]]:

    if base_cut is not None and len(base_cut) != k - 1:
        raise ValueError(f"For k={k} bins, base_cut must contain exactly {k-1} boundaries.")

    edges, pref = build_grid(series, L, U, step)
    m = len(edges) - 1
    s = int(np.floor(epsilon / step))

    if s < 0 or k < 1 or k > m: return None

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
    if k * (min_cells or 1) > m: return None
    if max_cells is not None and (min_cells or 1) > max_cells: return None

    excluded_idx = _boundary_values_to_indices(edges, excluded_boundaries, m)
    if excluded_idx is None: excluded_idx = set()
        
    required_idx = _boundary_values_to_indices(edges, required_boundaries, m)
    if required_idx is None: required_idx = set()
    
    if len(required_idx) > k - 1: return None
    if excluded_idx and required_idx.intersection(excluded_idx): return None

    required_list = sorted(required_idx)

    def boundary_allowed(j: int) -> bool:
        return j not in excluded_idx

    # --- On-the-fly Stability Check (Memory/Time Optimized) ---
    @lru_cache(maxsize=None)
    def is_stable(l: int, r: int) -> bool:
        base = int(pref[r] - pref[l])
        valid_interval = False
        
        for dL in range(-s, s + 1):
            for dR in range(-s, s + 1):
                Ls = l + dL
                Rs = r + dR
                
                valid = True
                if l == 0 and dL != 0: valid = False
                if r == m and dR != 0: valid = False
                if l != 0 and r != m:
                    if not (0 < Ls < m) or not (0 < Rs < m): valid = False
                if not (0 <= Ls < m and 0 < Rs <= m): valid = False
                if not (Ls < Rs): valid = False
                
                if valid:
                    shifted = int(pref[Rs] - pref[Ls])
                    if abs(base - shifted) > C:
                        return False 
                    valid_interval = True
        return valid_interval

    # On-the-fly Cand Interval logic (Replaces dense matrix check)
    if use_boundary_prefilter:
        cand_interval = np.zeros((m + 1, m + 1), dtype=bool)
        for l in range(m):
            for r in range(l + 1, m + 1):
                if is_stable(l, r):
                    cand_interval[l, r] = True
                    
        P = [0, m]
        for j in range(1, m):
            if not boundary_allowed(j): continue
            if cand_interval[:j, j].any() and cand_interval[j, j + 1:].any():
                P.append(j)
        P = np.array(sorted(set(P)), dtype=int)

    def get_next_required(i: int) -> int:
        for p in required_list:
            if p > i: return p
        return m + 1

    @lru_cache(maxsize=None)
    def dfs(i: int, bins_left: int) -> Optional[Tuple[float, Tuple[int, ...]]]:
        if bins_left == 1:
            if get_next_required(i) <= m: 
                return None
            
            last_w = m - i
            if min_cells is not None and last_w < min_cells: return None
            if max_cells is not None and last_w > max_cells: return None

            if is_stable(i, m):
                return (0.0, (i, m))
            return None

        lo = i + (min_cells or 1)
        hi = m - (bins_left - 1) * (min_cells or 1)
        if max_cells is not None: 
            hi = min(hi, i + max_cells)
            
        next_req = get_next_required(i)
        hi = min(hi, next_req)

        if use_boundary_prefilter:
            valid_js = P[(P >= lo) & (P <= hi)]
        else:
            valid_js = range(lo, hi + 1)
            
        best_cost = float('inf')
        best_path = None

        for j in valid_js:
            if not use_boundary_prefilter and not boundary_allowed(j): 
                continue

            # Directly use on-the-fly stability check
            if not is_stable(i, j):
                continue
            
            tail = dfs(j, bins_left - 1)
            if tail is not None:
                tail_cost, tail_path = tail
                
                if base_cut is not None:
                    cut_idx = k - bins_left
                    step_cost = abs(edges[j] - base_cut[cut_idx])
                else:
                    step_cost = 0.0
                
                total_cost = step_cost + tail_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = (i,) + tail_path

        if best_path is None:
            return None
            
        return (best_cost, best_path)

    result = dfs(0, k)
    if result is None: return None
    
    best_cost, path = result
    bnds = list(path)
    cuts_idx = bnds[1:-1]
    
    return [float(edges[c]) for c in cuts_idx]


# ============================================================
# Benchmark Runner
# ============================================================

def run_scaling_benchmark_with_matrix_time():
    results = []
    
    # Loop m from 1000 up to 6000 in steps of 500
    m_values = range(1000, 6000, 500)
    rng = np.random.default_rng()

    for m in m_values:
        print(f"\n{'='*60}")
        print(f"Testing Grid Size: m = {m}")
        print(f"{'='*60}")

        num_points = int(4.5 * m)
        vals = rng.integers(low=0, high=m, size=num_points)
        series = pd.Series(vals)

        # Parameters mapped to m bins
        L, U, step = 0.0, float(m), 1.0
        k = 10
        epsilon = 2.0
        C = 10
        
        # Generate a dummy base_cut mathematically (equi-width equivalent)
        # We need this because base_cut is now required for both versions.
        edges, _ = build_grid(series, L, U, step)
        base_cut_dummy = [float(edges[int(i * m / k)]) for i in range(1, k)]

        print(f"Data Points generated: {len(series)}")
        print(f"Base Cut: {base_cut_dummy[:3]}... (truncated)")

        try:
            # --------------------------------------------------------
            # Original Algorithm 5 (with Matrix E/DEV Build Time)
            # --------------------------------------------------------
            print("Measuring dense matrix DEV build time...")
            s = int(np.floor(epsilon / step))
            
            start_E = time.perf_counter()
            _ = build_DEV_2d(np.zeros(m + 1, dtype=int), s) # Dummy call just for timing matrix alloc
            end_E = time.perf_counter()
            time_E = end_E - start_E

            print("Running Algorithm 5 (Original DFS with Dense Matrix)...")
            start_5 = time.perf_counter()
            res_5 = algorithm5_independent_graph(
                series, L, U, step, k, epsilon, C, base_cut=base_cut_dummy
            )
            end_5 = time.perf_counter()
            
            time_5 = end_5 - start_5
            found_5 = "Found" if res_5 is not None else "Not Found"

            # --------------------------------------------------------
            # Optimized Algorithm 5
            # --------------------------------------------------------
            print("Running Algorithm 5 (Optimized DFS On-the-fly)...")
            start_5_opt = time.perf_counter()
            res_5_opt = algorithm5_graph_edge_consistent_optimized(
                series, L, U, step, k, epsilon, C, base_cut=base_cut_dummy
            )
            end_5_opt = time.perf_counter()
            
            time_5_opt = end_5_opt - start_5_opt
            found_5_opt = "Found" if res_5_opt is not None else "Not Found"

            # --------------------------------------------------------
            # Save and Print Results
            # --------------------------------------------------------
            results.append({
                "m": m,
                "num_points": len(series),
                "matrix_DEV_time": time_E,
                "alg5_original_time": time_5,
                "alg5_original_status": found_5,
                "alg5_optimized_time": time_5_opt,
                "alg5_optimized_status": found_5_opt
            })

            print("\n=== Results ===")
            print(f"Matrix DEV Build Time: {time_E:.5f} seconds")
            print("-" * 60)
            print(f"Algorithm 5 Original | Status: {found_5:<9} | Time: {time_5:.5f} s")
            print(f"Algorithm 5 Optimized| Status: {found_5_opt:<9} | Time: {time_5_opt:.5f} s")
            print("-" * 60)

            if time_5_opt < time_5:
                multiplier = time_5 / time_5_opt
                print(f"-> Alg 5 Optimized is {multiplier:.2f}x faster than Original.")

        except MemoryError:
            print(f"\n[!] MemoryError for m={m}")
            print("Attempting Algorithm 5 Optimized only (bypassing dense matrix)...")

            try:
                start_5_opt = time.perf_counter()
                res_5_opt = algorithm5_graph_edge_consistent_optimized(
                    series, L, U, step, k, epsilon, C, base_cut=base_cut_dummy
                )
                end_5_opt = time.perf_counter()
                
                time_5_opt = end_5_opt - start_5_opt
                found_5_opt = "Found" if res_5_opt is not None else "Not Found"

                results.append({
                    "m": m,
                    "num_points": len(series),
                    "matrix_DEV_time": None,
                    "alg5_original_time": None,
                    "alg5_original_status": "MemoryError",
                    "alg5_optimized_time": time_5_opt,
                    "alg5_optimized_status": found_5_opt
                })

                print(f"Algorithm 5 Optimized | Status: {found_5_opt:<9} | Time: {time_5_opt:.5f} s")

            except Exception as e:
                print(f"Alg 5 Optimized failed: {e}")
            break

        # Aggressively release unreferenced matrices from memory
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_csv("scaling_benchmark_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Benchmark finished.")
    print("Results saved to: scaling_benchmark_results.csv")
    print("=" * 60)
    print("\nFinal Results:")
    print(results_df.to_string())

if __name__ == "__main__":
    run_scaling_benchmark_with_matrix_time()
