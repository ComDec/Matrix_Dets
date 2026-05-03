"""
Multi-seed search: Try many different starting 11-column configurations and test
each for extendability by checking the maximum clique size in the compatibility graph.

The key insight from our analysis: the stored partial_11_cols_new.npy leads to
a compatibility graph with max clique ~6, but we need 18. Different 11-column
configurations may lead to much larger cliques.

Strategy:
1. Generate many random valid 11-column configurations
2. For each, compute the candidate set and compatibility graph for remaining 18 columns
3. Check if the max clique >= 18
4. If yes, use backtracking to find exact solution
"""
import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm

N = 29

def build_new_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    G[0, 3] = 9; G[3, 0] = 9
    G[1, 4] = 9; G[4, 1] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G

def verify_decomposition(R, G):
    if R is None:
        return False
    RR = R.astype(np.int64)
    if not np.all(np.abs(RR) == 1):
        return False
    return np.array_equal(RR.T @ RR, G)


def solve_pm1_system(constraint_vecs, constraint_vals, n, rng, max_sol=500, deadline=None):
    """Find +/-1 vectors x of length n satisfying constraints."""
    k = len(constraint_vecs)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]

    A_rows = [[Fraction(int(constraint_vecs[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(constraint_vals[i])) for i in range(k)]
    aug = [A_rows[i] + [b_vec[i]] for i in range(k)]

    pivot_cols = []; row_idx = 0
    for col in range(n):
        if row_idx >= k: break
        piv = -1
        for r in range(row_idx, k):
            if aug[r][col] != 0: piv = r; break
        if piv == -1: continue
        if piv != row_idx: aug[row_idx], aug[piv] = aug[piv], aug[row_idx]
        pv = aug[row_idx][col]
        for j in range(n + 1): aug[row_idx][j] /= pv
        for r in range(row_idx + 1, k):
            if aug[r][col] != 0:
                f = aug[r][col]
                for j in range(n + 1): aug[r][j] -= f * aug[row_idx][j]
        pivot_cols.append(col); row_idx += 1

    for r in range(len(pivot_cols), k):
        if aug[r][n] != 0: return []

    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                f = aug[r2][pc]
                for j in range(n + 1): aug[r2][j] -= f * aug[i][j]

    free_cols = [c for c in range(n) if c not in pivot_cols]
    nf = len(free_cols)
    np_ = len(pivot_cols)

    ic = np.zeros((np_, nf), dtype=np.int64)
    icons = np.zeros(np_, dtype=np.int64)
    iden = np.zeros(np_, dtype=np.int64)

    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free_cols]
        lcd = 1
        for d in ds: lcd = math_lcm(lcd, d)
        iden[i] = lcd; icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free_cols): ic[i, fi] = int(aug[i][fc] * lcd)

    solutions = []
    if nf == 0:
        x = np.zeros(n, dtype=np.int64)
        valid = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]: valid = False; break
            x[pivot_cols[i]] = icons[i] // iden[i]
        if valid: solutions.append(x)
    elif nf <= 20:
        total = 1 << nf
        bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]
        rng.shuffle(si)
        for idx in si[:max_sol]:
            if deadline and time.time() > deadline: break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
            for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        batch_size = min(1 << 20, max_sol * 200)
        for _ in range(max(1, (max_sol * 500) // batch_size)):
            if deadline and time.time() > deadline: break
            if len(solutions) >= max_sol: break
            fm = rng.choice([-1, 1], size=(batch_size, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(batch_size, dtype=bool)
            for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(solutions) >= max_sol: break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
                for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
    return solutions


def build_11_columns_greedy(G, rng, deadline):
    """Build a random valid set of 11 columns via greedy construction."""
    # Column order: 0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5]

    placed_cols = []
    placed_indices = []

    for step, target_idx in enumerate(col_order):
        if time.time() > deadline:
            return None

        constraint_vals = [int(G[placed_indices[i], target_idx]) for i in range(len(placed_cols))]
        solutions = solve_pm1_system(
            placed_cols, constraint_vals, N, rng,
            max_sol=1000, deadline=deadline
        )

        if not solutions:
            return None

        # Pick a random solution
        placed_cols.append(solutions[rng.randint(len(solutions))])
        placed_indices.append(target_idx)

    # Reorder to column indices 0..10
    result = {}
    for i, ci in enumerate(placed_indices):
        result[ci] = placed_cols[i]

    # Verify
    for a in range(11):
        for b in range(a + 1, 11):
            ip = int(np.dot(result[a], result[b]))
            if ip != int(G[a, b]):
                return None

    return result


def evaluate_11col_config(fixed_cols, G, rng):
    """
    Given 11 fixed columns, compute the candidate set for remaining columns
    and evaluate the maximum clique size in the compatibility graph.

    Returns (n_candidates, max_clique_estimate, candidates_array).
    """
    # Get candidates for col 11 (all remaining cols have same candidates)
    cands = []
    constraint_vecs = [fixed_cols[b] for b in sorted(fixed_cols.keys())]
    constraint_vals = [int(G[11, b]) for b in sorted(fixed_cols.keys())]

    all_sols = solve_pm1_system(constraint_vecs, constraint_vals, N, rng, max_sol=10000)

    if len(all_sols) == 0:
        return 0, 0, None

    cands = np.array(all_sols, dtype=np.int64)
    n_cands = len(cands)

    if n_cands < 18:
        return n_cands, n_cands, cands

    # Compute IP matrix
    IP = cands @ cands.T
    ip1_graph = (IP == 1)
    np.fill_diagonal(ip1_graph, False)

    # Quick greedy clique estimate
    best_clique_size = 0
    for trial in range(min(500, n_cands * 10)):
        start = rng.randint(n_cands)
        clique = [start]
        candidates = set(np.where(ip1_graph[start])[0])

        while candidates:
            # Pick random candidate
            v = rng.choice(list(candidates))
            clique.append(v)
            candidates = candidates & set(np.where(ip1_graph[v])[0])

        if len(clique) > best_clique_size:
            best_clique_size = len(clique)

    return n_cands, best_clique_size, cands


def try_complete_from_clique(fixed_cols, candidates, G, rng, deadline):
    """Try to complete the 29-column matrix using backtracking over candidates."""
    n_cands = len(candidates)

    # Compute IP matrix
    IP = candidates @ candidates.T
    ip1_graph = (IP == 1)
    np.fill_diagonal(ip1_graph, False)

    # Backtracking to find 18-clique
    best_depth = 0

    def backtrack(chosen, compatible):
        nonlocal best_depth
        if time.time() > deadline:
            return None
        if len(chosen) == 18:
            return list(chosen)
        if len(chosen) > best_depth:
            best_depth = len(chosen)

        if len(compatible) < 18 - len(chosen):
            return None  # Not enough candidates left

        # Try each compatible candidate
        for v in sorted(compatible):
            if time.time() > deadline:
                return None
            new_compat = compatible & set(np.where(ip1_graph[v])[0])
            if len(new_compat) < 18 - len(chosen) - 1:
                continue
            chosen.append(v)
            result = backtrack(chosen, new_compat)
            if result is not None:
                return result
            chosen.pop()

        return None

    # Start from the node with highest degree for better chances
    degrees = np.sum(ip1_graph, axis=1)

    for start_node in np.argsort(-degrees)[:min(50, n_cands)]:
        if time.time() > deadline:
            break
        compat = set(np.where(ip1_graph[start_node])[0])
        result = backtrack([int(start_node)], compat)
        if result is not None:
            # Build R matrix
            R = np.zeros((N, N), dtype=np.int64)
            for ci, col in fixed_cols.items():
                R[:, ci] = col
            for i, cand_idx in enumerate(result):
                R[:, 11 + i] = candidates[cand_idx]
            if verify_decomposition(R, G):
                return R
    return None


def main():
    G = build_new_gram()
    print("=" * 70)
    print("MULTI-SEED SEARCH: Generate many 11-col configs, test extendability")
    print("=" * 70)
    sys.stdout.flush()

    deadline = time.time() + 3600  # 1 hour
    rng = np.random.RandomState(42)

    best_clique = 0
    best_config = None
    best_candidates = None
    n_configs = 0
    n_good = 0

    while time.time() < deadline - 60:
        seed = rng.randint(10**9)
        rng_local = np.random.RandomState(seed)

        config = build_11_columns_greedy(G, rng_local, deadline - 60)
        if config is None:
            continue

        n_configs += 1

        # Normalize: make col 0 all +1
        col0 = config[0]
        negate_mask = (col0 == -1)
        for ci in config:
            config[ci] = config[ci].copy()
            config[ci][negate_mask] *= -1

        # Evaluate
        n_cands, clique_est, cands = evaluate_11col_config(config, G, rng_local)

        if clique_est > best_clique:
            best_clique = clique_est
            best_config = config
            best_candidates = cands
            print(f"  Config #{n_configs} (seed={seed}): {n_cands} candidates, clique >= {clique_est} *** NEW BEST ***")
            sys.stdout.flush()

            if clique_est >= 18:
                n_good += 1
                print(f"  PROMISING! Attempting completion...")
                sys.stdout.flush()
                R = try_complete_from_clique(config, cands, G, rng_local, deadline)
                if R is not None:
                    print(f"\n*** BREAKTHROUGH! ***")
                    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                    det_val = abs(np.linalg.det(R.astype(float)))
                    target_max = 1270698346568170340352
                    print(f"|det(R)| = {det_val:.6e}")
                    print(f"Score = {det_val / target_max:.6f}")
                    return R
        elif n_configs % 100 == 0:
            elapsed = time.time() - (deadline - 3600)
            print(f"  Config #{n_configs}: best clique so far = {best_clique}, {elapsed:.0f}s")
            sys.stdout.flush()

    elapsed = time.time() - (deadline - 3600)
    print(f"\nSearched {n_configs} configs in {elapsed:.0f}s")
    print(f"Best clique: {best_clique} (need 18)")
    print(f"Promising configs (clique >= 18): {n_good}")

    if best_clique >= 10 and best_config is not None:
        # Try harder with the best config
        print(f"\nTrying harder completion from best config (clique={best_clique})...")
        R = try_complete_from_clique(best_config, best_candidates, G, rng, time.time() + 60)
        if R is not None:
            print(f"\n*** BREAKTHROUGH! ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            return R

    return None


if __name__ == "__main__":
    main()
