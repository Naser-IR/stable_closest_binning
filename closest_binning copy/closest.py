
#closest
from functools import lru_cache
import heapq
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

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
# Naive Algorithm 2 - Exhaustive Search
# ============================================================

def algorithm6_closest_naive_bfs(
    series: pd.Series,
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    base_cut: List[float],
    *,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
) -> Tuple[Optional[List[float]], Optional[float]]:
    
    # Use the original COUNT-only grid and DEV builders
    edges, pref = build_grid(series, L, U, step)
    m = len(edges) - 1
    s = int(np.floor(epsilon / step))
    DEV = build_DEV_2d(pref, s)

    if s < 0 or k < 1 or k > m: return (None, None)
    if base_cut is None or len(base_cut) != k - 1:
        raise ValueError(f"base_cut must contain exactly {k-1} internal boundaries.")

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
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
            # Still generate neighbors to pathfind around invalid states
            pass 
        else:
            # 2. Check Stability using the 2D DEV matrix
            stable = True
            for j in range(k):
                if DEV[bounds[j], bounds[j+1]] > C:
                    stable = False
                    break
            
            # The first stable and valid state popped from the min-heap is guaranteed to be the closest
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
                    
                    # Calculate new absolute distance from the original base_cut
                    new_dist = sum(abs(edges[new_state[j]] - base_cut[j]) for j in range(k - 1))
                    heapq.heappush(pq, (new_dist, new_state))
                    
    return (None, None)






# ============================================================
# Algorithm 4 - dynamic programming with 1-to-1 mapping to base cut
# ============================================================

def algorithm4_closest_independent_dp(
    series: pd.Series,
    L: float, U: float, step: float,
    k: int, epsilon: float, C: float,
    base_cut: List[float],  # New parameter for the target boundaries
    *,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    excluded_boundaries: Optional[List[float]] = None,
) -> Tuple[Optional[List[float]], Optional[float]]:

    edges, pref = build_grid(series, L, U, step)
    m = len(edges) - 1
    s = int(np.floor(epsilon / step))

    if s < 0 or k < 1 or k > m: return (None, None)
    
    # Ensure we have exactly k-1 target internal boundaries
    if base_cut is None or len(base_cut) != k - 1:
        raise ValueError(f"base_cut must contain exactly k-1 internal boundaries. Got {len(base_cut) if base_cut else 0}, expected {k-1}.")

    min_cells = _min_width_to_cells(min_width, step)
    max_cells = _max_width_to_cells(max_width, step)
    
    # Global impossibility checks
    if k * min_cells > m: return (None, None)
    if max_cells is not None and min_cells > max_cells: return (None, None)
    if max_cells is not None and k * max_cells < m: return (None, None)

    excluded_idx = _boundary_values_to_indices(edges, excluded_boundaries, m)
    def boundary_allowed(j: int) -> bool:
        return excluded_idx is None or j not in excluded_idx

    DEV = build_DEV_2d(pref, s)
    INF = float('inf')  # Changed to float for distance

    # OPT table now stores the minimum distance displacement (float)
    OPT = np.full((m + 1, k + 1), INF, dtype=float)
    parent = np.zeros((m + 1, k + 1), dtype=int)

    # Base case: j = 1 (A single bin from 0 to i)
    for i in range(1, m + 1):
        if i < min_cells: continue
        if max_cells is not None and i > max_cells: continue
        
        # As per the formula: 0 if DEV <= C, else +infinity
        if DEV[0, i] <= C:
            OPT[i, 1] = 0.0

    # DP Formula for j > 1
    for j in range(2, k + 1):
        # target_boundary is tau^0_{j-1}
        target_boundary = base_cut[j - 2] 
        
        # 'i' must be at least large enough to hold 'j' bins of min_cells
        for i in range(j * min_cells, m + 1):
            
            # MATH: Define the valid search window for the previous boundary 'l'
            lo = (j - 1) * min_cells
            if max_cells is not None:
                lo = max(lo, i - max_cells)
                
            hi = i - min_cells
            if max_cells is not None:
                hi = min(hi, (j - 1) * max_cells)

            best_val = INF
            best_l = -1

            # Only search within the mathematically valid window
            for l in range(lo, hi + 1):
                if not boundary_allowed(l): continue
                if OPT[l, j - 1] == INF: continue
                
                # Hard stability constraint: DEV[l, i] MUST be <= C
                if DEV[l, i] > C: continue 
                
                # Formula: OPT[l, j-1] + |x_l - tau^0_{j-1}|
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

    # Backtrack to find the optimal cut points
    cuts_idx = []
    curr = m
    for j in range(k, 1, -1):
        curr = parent[curr, j]
        cuts_idx.append(curr)

    cuts_idx.reverse()
    return ([float(edges[c]) for c in cuts_idx], float(best_dist))






# ============================================================
# Algorithm 5 — Graph-Based Construction (1-to-1 Mapping)
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

    # 1. Validation for the 1-to-1 mapping
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

    # --- Shortest Path DFS with 1-to-1 mapping ---
    @lru_cache(maxsize=None)
    def dfs(i: int, bins_left: int) -> Optional[Tuple[float, Tuple[int, ...]]]:
        if bins_left == 1:
            if get_next_required(i) <= m: 
                return None
            
            last_w = m - i
            if min_cells is not None and last_w < min_cells: return None
            if max_cells is not None and last_w > max_cells: return None

            if DEV[i, m] < C:
                # Cost is 0 because 'm' is the fixed outer boundary, not an internal cut
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

        if best_path is None:
            return None
            
        return (best_cost, best_path)

    result = dfs(0, k)
    
    if result is None: return None
    
    best_cost, path = result
    bnds = list(path)
    cuts_idx = bnds[1:-1]
    
    return [float(edges[c]) for c in cuts_idx]