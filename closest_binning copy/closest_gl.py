from functools import lru_cache
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import heapq


# ============================================================
# Helpers
# ============================================================

def build_universal_DEV_2d(
    x_series: pd.Series,
    y_series: Optional[pd.Series], 
    L: float, U: float, step: float, s: int, 
    agg_func: str = 'COUNT'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Universal builder for the DEV matrix supporting COUNT, SUM, AVG, MIN, MAX.
    """
    edges = np.arange(L, U + step, step)
    m = len(edges) - 1
    
    # Map each data point to its base cell (0 to m-1)
    cell_idx = np.digitize(x_series, edges) - 1
    valid_mask = (cell_idx >= 0) & (cell_idx < m)
    
    cells_valid = cell_idx[valid_mask]
    y_valid = y_series[valid_mask] if y_series is not None else np.ones(len(cells_valid))
    
    df_cells = pd.DataFrame({'cell': cells_valid, 'y': y_valid})
    
    # 1. Precompute the exact aggregate for every possible interval [l, r)
    VAL = np.zeros((m + 1, m + 1), dtype=float)
    agg = agg_func.upper()
    
    if agg in ['COUNT', 'SUM']:
        base_vals = df_cells.groupby('cell')['y'].sum().reindex(range(m), fill_value=0).values
        pref = np.zeros(m + 1)
        pref[1:] = np.cumsum(base_vals)
        for l in range(m):
            for r in range(l + 1, m + 1):
                VAL[l, r] = pref[r] - pref[l]
                
    elif agg == 'AVG':
        base_sums = df_cells.groupby('cell')['y'].sum().reindex(range(m), fill_value=0).values
        base_counts = df_cells.groupby('cell')['y'].count().reindex(range(m), fill_value=0).values
        pref_sum = np.zeros(m + 1)
        pref_sum[1:] = np.cumsum(base_sums)
        pref_count = np.zeros(m + 1)
        pref_count[1:] = np.cumsum(base_counts)
        for l in range(m):
            for r in range(l + 1, m + 1):
                c = pref_count[r] - pref_count[l]
                VAL[l, r] = (pref_sum[r] - pref_sum[l]) / c if c > 0 else 0.0
                
    elif agg == 'MAX':
        base_max = df_cells.groupby('cell')['y'].max().reindex(range(m), fill_value=-np.inf).values
        for l in range(m):
            curr_max = -np.inf
            for r in range(l + 1, m + 1):
                curr_max = max(curr_max, base_max[r - 1])
                VAL[l, r] = curr_max
                
    elif agg == 'MIN':
        base_min = df_cells.groupby('cell')['y'].min().reindex(range(m), fill_value=np.inf).values
        for l in range(m):
            curr_min = np.inf
            for r in range(l + 1, m + 1):
                curr_min = min(curr_min, base_min[r - 1])
                VAL[l, r] = curr_min
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    # 2. Build the DEV matrix by checking epsilon shifts
    INF = 10**12
    DEV = np.full((m + 1, m + 1), INF, dtype=float)
    
    for l in range(m):
        for r in range(l + 1, m + 1):
            base_val = VAL[l, r]
            max_err = -1.0
            valid_interval = False
            
            for dL in range(-s, s + 1):
                for dR in range(-s, s + 1):
                    Ls, Rs = l + dL, r + dR
                    
                    # Prevent out-of-bounds or crossing boundaries
                    if l == 0 and dL != 0: continue
                    if r == m and dR != 0: continue
                    if l != 0 and not (0 < Ls < m): continue
                    if r != m and not (0 < Rs < m): continue
                    if Ls >= Rs: continue
                    
                    shifted_val = VAL[Ls, Rs]
                    
                    # Handle empty bins for Min/Max
                    if np.isinf(base_val) or np.isinf(shifted_val):
                        err = 0.0 if (np.isinf(base_val) and np.isinf(shifted_val)) else INF
                    else:
                        err = abs(base_val - shifted_val)
                        
                    if err > max_err:
                        max_err = err
                    valid_interval = True
            
            if valid_interval:
                DEV[l, r] = max_err
                
    return DEV, edges


def _boundary_values_to_indices(edges: np.ndarray, values, m: int) -> Optional[set]:
    """
    Convert real boundary values like [6.0, 12.0] to grid indices.
    Ignore 0 and m since they are outer boundaries, not internal cuts.
    """
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




# ============================================================
# Naive Algorithm 2 - Exhaustive Search
# ============================================================

def algorithm6_closest_naive_bfs(
    x_series: pd.Series,
    y_series: Optional[pd.Series],
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    base_cut: List[float],
    agg_func: str = 'COUNT',
    *,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
) -> Tuple[Optional[List[float]], Optional[float]]:
    
    s = int(np.floor(epsilon / step))
    DEV, edges = build_universal_DEV_2d(x_series, y_series, L, U, step, s, agg_func)
    m = len(edges) - 1

    if s < 0 or k < 1 or k > m: return (None, None)
    if base_cut is None or len(base_cut) != k - 1:
        raise ValueError(f"base_cut must contain exactly {k-1} internal boundaries.")

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
    # Map excluded boundaries to indices
    excluded_idx = _boundary_values_to_indices(edges, excluded_boundaries, m)
    def boundary_allowed(idx: int) -> bool:
        return excluded_idx is None or idx not in excluded_idx

    # Map the base_cut to exact grid indices
    edge_to_idx = {round(float(edges[i]), 12): i for i in range(len(edges))}
    base_indices = []
    for x in base_cut:
        xf = round(float(x), 12)
        if xf not in edge_to_idx:
            raise ValueError(f"Base cut {x} is not on the grid.")
        base_indices.append(edge_to_idx[xf])
    
    base_state = tuple(base_indices)
    
    # Priority Queue stores: (total_distance, state_tuple)
    # Using a Min-Heap ensures we always evaluate the closest modifications first
    pq = [(0.0, base_state)]
    visited = set([base_state])
    
    while pq:
        dist, current_state = heapq.heappop(pq)
        
        # 1. Validate widths and excluded boundaries
        bounds = (0,) + current_state + (m,)
        valid_state = True
        
        for j in range(k):
            w = bounds[j+1] - bounds[j]
            if w <= 0: valid_state = False; break
            if min_cells is not None and w < min_cells: valid_state = False; break
            if max_cells is not None and w > max_cells: valid_state = False; break
            if j < k - 1 and not boundary_allowed(current_state[j]): valid_state = False; break
            
        if not valid_state:
            # Generate neighbors anyway to allow pathfinding around invalid states
            pass 
        else:
            # 2. Check Stability
            stable = True
            for j in range(k):
                if DEV[bounds[j], bounds[j+1]] > C:
                    stable = False
                    break
            
            # Since we use a priority queue, the first stable state we hit is guaranteed to be the closest
            if stable:
                cuts = [float(edges[i]) for i in current_state]
                return cuts, float(dist)
                
        # 3. Generate Neighbors (Shift each boundary by +1 and -1)
        for i in range(k - 1):
            for delta in (-1, 1):
                new_idx = current_state[i] + delta
                
                # Boundary bounds check
                if new_idx <= 0 or new_idx >= m:
                    continue
                    
                new_list = list(current_state)
                new_list[i] = new_idx
                new_state = tuple(new_list)
                
                # Prevent overlapping boundaries (x1 must strictly be < x2)
                is_ordered = all(new_state[j] < new_state[j+1] for j in range(k - 2))
                if not is_ordered:
                    continue
                    
                if new_state not in visited:
                    visited.add(new_state)
                    
                    # Calculate new L1 distance from the original base_cut
                    new_dist = sum(abs(edges[new_state[j]] - base_cut[j]) for j in range(k - 1))
                    heapq.heappush(pq, (new_dist, new_state))
                    
    return (None, None)






# ============================================================
# Algorithm 4 - dynamic programming with 1-to-1 mapping to base cut
# ============================================================

def algorithm4_closest_independent_dp(
    x_series: pd.Series,
    y_series: Optional[pd.Series],
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    base_cut: List[float], 
    agg_func: str = 'COUNT',
    *,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
) -> Tuple[Optional[List[float]], Optional[float]]:
    """
    Dynamic programming approach finding the closest stable cut based on the theoretical minimum total boundary displacement formula[span_0](start_span)[span_0](end_span).
    Supports COUNT, SUM, AVG, MAX, MIN aggregations via universal DEV matrix builder.
    """
    s = int(np.floor(epsilon / step))
    DEV, edges = build_universal_DEV_2d(x_series, y_series, L, U, step, s, agg_func)
    m = len(edges) - 1

    if s < 0 or k < 1 or k > m: return (None, None)
    
    if base_cut is None or len(base_cut) != k - 1:
        raise ValueError(f"base_cut must contain exactly {k-1} internal boundaries. Got {len(base_cut) if base_cut else 0}.")

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
    if min_cells is not None and k * min_cells > m: return (None, None)
    if min_cells is not None and max_cells is not None and min_cells > max_cells: return (None, None)
    if max_cells is not None and k * max_cells < m: return (None, None)

    excluded_idx = _boundary_values_to_indices(edges, excluded_boundaries, m)
    def boundary_allowed(j: int) -> bool:
        return excluded_idx is None or j not in excluded_idx

    INF = float('inf')

    # OPT table stores the minimum distance displacement
    OPT = np.full((m + 1, k + 1), INF, dtype=float)
    parent = np.zeros((m + 1, k + 1), dtype=int)

    # Base case: j = 1[span_1](start_span)[span_1](end_span)
    for i in range(1, m + 1):
        if min_cells is not None and i < min_cells: continue
        if max_cells is not None and i > max_cells: continue
        
        if DEV[0, i] <= C:
            OPT[i, 1] = 0.0

    # DP Formula for j > 1[span_2](start_span)[span_2](end_span)
    for j in range(2, k + 1):
        target_boundary = base_cut[j - 2] 
        
        for i in range(j * (min_cells or 1), m + 1):
            
            lo = (j - 1) * (min_cells or 1)
            if max_cells is not None:
                lo = max(lo, i - max_cells)
                
            hi = i - (min_cells or 1)
            if max_cells is not None:
                hi = min(hi, (j - 1) * max_cells)

            best_val = INF
            best_l = -1

            for l in range(lo, hi + 1):
                if not boundary_allowed(l): continue
                if OPT[l, j - 1] == INF: continue
                
                # Hard stability constraint[span_3](start_span)[span_3](end_span)
                if DEV[l, i] > C: continue 
                
                # Accumulated boundary displacement[span_4](start_span)[span_4](end_span)
                dist = abs(edges[l] - target_boundary)
                val = OPT[l, j - 1] + dist
                
                if val < best_val:
                    best_val = val
                    best_l = l
            
            OPT[i, j] = best_val
            parent[i, j] = best_l

    best_dist = OPT[m, k]
    if best_dist == INF:
        return (None, None)

    cuts_idx = []
    curr = m
    for j in range(k, 1, -1):
        curr = parent[curr, j]
        cuts_idx.append(curr)

    cuts_idx.reverse()
    return ([float(edges[c]) for c in cuts_idx], float(best_dist))






# ============================================================
# Algorithm 5 — Graph-Based Construction with 1-to-1 Mapping
# ============================================================

def algorithm5_independent_graph(
    x_series: pd.Series,
    y_series: Optional[pd.Series],
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    agg_func: str = 'COUNT', 
    *,
    use_boundary_prefilter: bool = True,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
    required_boundaries: Optional[List[float]] = None,
    base_cut: Optional[List[float]] = None,
) -> Optional[List[float]]:
    
    # Validation for the 1-to-1 mapping
    if base_cut is not None and len(base_cut) != k - 1:
        raise ValueError(f"For k={k} bins, base_cut must contain exactly {k-1} boundaries.")
        
    s = int(np.floor(epsilon / step))
    DEV, edges = build_universal_DEV_2d(x_series, y_series, L, U, step, s, agg_func)
    m = len(edges) - 1

    if s < 0 or k < 1 or k > m: return None

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
    excluded_idx = _boundary_values_to_indices(edges, excluded_boundaries, m)
    required_idx = _boundary_values_to_indices(edges, required_boundaries, m)
    if required_idx is None: required_idx = set()
    if len(required_idx) > k - 1 or (excluded_idx and required_idx.intersection(excluded_idx)):
        return None

    required_list = sorted(required_idx)
    req_to_bit = {r: i for i, r in enumerate(required_list)}
    full_required_mask = (1 << len(required_list)) - 1

    INF = int(DEV[m, m]) + 10**9 if not np.isinf(DEV[m, m]) else 10**12

    def boundary_allowed(j: int) -> bool:
        return excluded_idx is None or j not in excluded_idx

    # Pre-filtering based strictly on the 2D DEV matrix
    if use_boundary_prefilter:
        cand_interval = (DEV < C) & (DEV < INF)
        P = [0, m]
        for j in range(1, m):
            if not boundary_allowed(j): continue
            if cand_interval[:j, j].any() and cand_interval[j, j + 1:].any():
                P.append(j)
        P = np.array(sorted(set(P)), dtype=int)

        def iter_boundaries(i: int, bins_left: int):
            lo = i + (min_cells or 1)
            hi = m - (bins_left - 1) * (min_cells or 1)
            if max_cells is not None: hi = min(hi, i + max_cells)
            for j in P[(P >= lo) & (P <= hi)]:
                yield int(j)
    else:
        def iter_boundaries(i: int, bins_left: int):
            lo = i + (min_cells or 1)
            hi = m - (bins_left - 1) * (min_cells or 1)
            if max_cells is not None: hi = min(hi, i + max_cells)
            for j in range(lo, hi + 1):
                if boundary_allowed(j): yield j

    # --- Shortest Path DFS with 1-to-1 Mapping ---
    @lru_cache(maxsize=None)
    def dfs(i: int, bins_left: int, req_mask: int) -> Optional[Tuple[float, Tuple[int, ...]]]:
        missing_required = len(required_list) - bin(req_mask).count("1")
        if missing_required > bins_left - 1: return None

        if bins_left == 1:
            if req_mask != full_required_mask: return None
            last_w = m - i
            if min_cells is not None and last_w < min_cells: return None
            if max_cells is not None and last_w > max_cells: return None

            if DEV[i, m] < C:
                # Cost is 0 because 'm' is the fixed outer boundary, not an internal cut
                return (0.0, (i, m))
            return None

        best_cost = float('inf')
        best_path = None

        for j in iter_boundaries(i, bins_left):
            if DEV[i, j] >= C:
                continue

            new_mask = req_mask | (1 << req_to_bit[j]) if j in req_to_bit else req_mask
            
            tail = dfs(j, bins_left - 1, new_mask)
            if tail is not None:
                tail_cost, tail_path = tail
                
                # Dynamic 1-to-1 Mapping: Match the new boundary to the corresponding base_cut boundary
                if base_cut is not None:
                    cut_idx = k - bins_left
                    step_cost = abs(edges[j] - base_cut[cut_idx])
                else:
                    step_cost = 0.0
                
                total_cost = step_cost + tail_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = (i,) + tail_path

        if best_path is None: return None
        return (best_cost, best_path)

    # Initialize graph search
    result = dfs(0, k, 0)
    
    if result is None: return None
    
    best_cost, path = result
    bnds = list(path)
    if bnds[0] != 0: bnds = [0] + bnds
    cuts_idx = bnds[1:-1]
    
    if required_idx and not required_idx.issubset(set(cuts_idx)): return None
    return [float(edges[c]) for c in cuts_idx]