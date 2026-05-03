"""
Find columns 0-10 to pair with Solomon's columns 11-28.

Key insight: With Solomon's cols 11-28 fixed, col 0 has 62 candidates,
col 3 has 62 candidates, and IP=9 IS achievable (144 valid pairs).

Strategy:
1. Enumerate candidates for each of columns 0-10 (given fixed 11-28)
2. Build compatibility matrices for all pairs
3. Backtrack to find a valid combination of 11 columns
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
    RR = R.astype(np.int64)
    if not np.all(np.abs(RR) == 1):
        return False
    return np.array_equal(RR.T @ RR, G)


def enumerate_all_pm1(constraint_vecs, constraint_vals, n):
    k = len(constraint_vecs)
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

    if nf > 20:
        # Too many free vars, sample
        rng = np.random.RandomState(42)
        batch = min(1 << 20, 1000000)
        fm = rng.choice([-1, 1], size=(batch, nf)).astype(np.int64)
        pv_arr = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(batch, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv_arr[:, i]) == iden[i])
        si = np.where(vm)[0]
    else:
        total = 1 << nf
        bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv_arr = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv_arr[:, i]) == iden[i])
        si = np.where(vm)[0]

    solutions = []
    for idx in si:
        x = np.zeros(n, dtype=np.int64)
        for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
        for i in range(np_): x[pivot_cols[i]] = pv_arr[idx, i] // iden[i]
        solutions.append(x)
    return solutions


def main():
    G = build_new_gram()
    print("=" * 70)
    print("SOLOMON EXTEND: Find cols 0-10 to pair with Solomon's 11-28")
    print("=" * 70)
    sys.stdout.flush()

    deadline = time.time() + 3600
    start_time = time.time()

    R_sol = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    # Fixed columns 11-28
    fixed_indices = list(range(11, 29))
    fixed_cols = [R_sol[:, k] for k in fixed_indices]

    # Columns to find: 0-10
    target_cols = list(range(11))

    # Step 1: Enumerate candidates for each target column
    print("\nStep 1: Enumerating candidates...")
    sys.stdout.flush()

    candidates = {}
    for ci in target_cols:
        constraint_vals = [int(G[ci, k]) for k in fixed_indices]
        cands = enumerate_all_pm1(fixed_cols, constraint_vals, N)
        candidates[ci] = np.array(cands, dtype=np.int64)
        print(f"  Col {ci}: {len(cands)} candidates")
        sys.stdout.flush()

    # Step 2: Precompute compatibility
    print("\nStep 2: Precomputing pairwise compatibility...")
    sys.stdout.flush()

    compat = {}
    for i in range(11):
        for j in range(i + 1, 11):
            target_ip = int(G[i, j])
            if len(candidates[i]) == 0 or len(candidates[j]) == 0:
                compat[(i, j)] = np.zeros((0, 0), dtype=bool)
                continue
            ip_mat = candidates[i] @ candidates[j].T
            compat[(i, j)] = (ip_mat == target_ip)
            n_compat = np.sum(compat[(i, j)])
            total = compat[(i, j)].size
            pct = 100 * n_compat / total if total > 0 else 0
            if target_ip != 1 or n_compat == 0:  # Only print special pairs
                print(f"    ({i},{j}): IP={target_ip}, {n_compat}/{total} compatible ({pct:.1f}%)")
    sys.stdout.flush()

    # Step 3: Backtracking with forward checking
    print("\nStep 3: Backtracking search...")
    sys.stdout.flush()

    # Order: place most constrained columns first
    # Cols 0 and 3 are special (IP=9 constraint between them)
    # Cols 1 and 4 are special (IP=9 constraint between them)
    # Col 6 is special (IP=-3 with cols 7-10)
    # Col order: 0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5]

    best_depth = 0
    n_nodes = 0
    n_backtracks = 0

    def get_compat_set(col_a, cand_a_idx, col_b):
        """Get set of candidate indices for col_b compatible with cand_a_idx of col_a."""
        a, b = min(col_a, col_b), max(col_a, col_b)
        cm = compat[(a, b)]
        if col_a == a:
            return set(np.where(cm[cand_a_idx, :])[0])
        else:
            return set(np.where(cm[:, cand_a_idx])[0])

    def backtrack(step, domains, assignments):
        nonlocal best_depth, n_nodes, n_backtracks

        if time.time() > deadline:
            return None

        n_nodes += 1

        if step == 11:
            return dict(assignments)

        if step > best_depth:
            best_depth = step
            elapsed = time.time() - start_time
            dom_sizes = sorted([len(domains[c]) for c in domains])
            print(f"  Depth {step}/11: domains = {dom_sizes} | {elapsed:.1f}s | nodes={n_nodes}")
            sys.stdout.flush()

        # MRV: choose unassigned column with smallest domain
        chosen_col = min(domains.keys(), key=lambda c: len(domains[c]))
        dom = list(domains[chosen_col])

        if not dom:
            n_backtracks += 1
            return None

        for cand_idx in dom:
            if time.time() > deadline:
                return None

            # Forward check: compute new domains
            new_domains = {}
            feasible = True
            for other_col, other_dom in domains.items():
                if other_col == chosen_col:
                    continue
                compatible = get_compat_set(chosen_col, cand_idx, other_col)
                new_dom = [v for v in other_dom if v in compatible]
                if not new_dom:
                    feasible = False
                    break
                new_domains[other_col] = new_dom

            if not feasible:
                n_backtracks += 1
                continue

            assignments[chosen_col] = cand_idx
            result = backtrack(step + 1, new_domains, assignments)
            if result is not None:
                return result
            del assignments[chosen_col]
            n_backtracks += 1

        return None

    # Initialize domains
    initial_domains = {ci: list(range(len(candidates[ci]))) for ci in target_cols}

    print(f"  Initial domains: {[len(initial_domains[c]) for c in col_order]}")
    print(f"  Search space: {np.prod([len(initial_domains[c]) for c in col_order]):.2e}")
    sys.stdout.flush()

    result = backtrack(0, initial_domains, {})

    if result is not None:
        # Build R matrix
        R = np.zeros((N, N), dtype=np.int64)
        for k in fixed_indices:
            R[:, k] = R_sol[:, k]
        for ci, cand_idx in result.items():
            R[:, ci] = candidates[ci][cand_idx]

        if verify_decomposition(R, G):
            elapsed = time.time() - start_time
            print(f"\n*** BREAKTHROUGH! Valid decomposition found in {elapsed:.1f}s ***")
            print(f"  Nodes explored: {n_nodes}, backtracks: {n_backtracks}")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)

            det_val = abs(np.linalg.det(R.astype(float)))
            target_max = 1270698346568170340352
            print(f"|det(R)| = {det_val:.6e}")
            print(f"Score = {det_val / target_max:.6f}")
            return R
        else:
            err = int(np.sum((R.T @ R - G) ** 2))
            print(f"  Solution does not verify! Error = {err}")
    else:
        elapsed = time.time() - start_time
        print(f"\nNo solution found in {elapsed:.1f}s")
        print(f"  Best depth: {best_depth}/11, nodes: {n_nodes}, backtracks: {n_backtracks}")

    # If that didn't work with Solomon's 11-28, try with different fixed sets
    # Try fixing only columns 12-28 (17 cols), freeing up col 11 too
    return try_fewer_fixed(G, R_sol, deadline, start_time)


def try_fewer_fixed(G, R_sol, deadline, start_time):
    """Try with fewer fixed columns if the first attempt fails."""
    print("\n" + "=" * 70)
    print("Trying with fewer fixed columns (only 13-28 fixed)...")
    sys.stdout.flush()

    fixed_indices = list(range(13, 29))
    fixed_cols = [R_sol[:, k] for k in fixed_indices]
    target_cols = list(range(13))

    candidates = {}
    for ci in target_cols:
        constraint_vals = [int(G[ci, k]) for k in fixed_indices]
        cands = enumerate_all_pm1(fixed_cols, constraint_vals, N)
        candidates[ci] = np.array(cands, dtype=np.int64) if cands else np.zeros((0, N), dtype=np.int64)
        print(f"  Col {ci}: {len(cands)} candidates")
        sys.stdout.flush()

    # Check IP=9 feasibility for (0,3) and (1,4)
    for a, b, target_ip in [(0, 3, 9), (1, 4, 9)]:
        if len(candidates[a]) > 0 and len(candidates[b]) > 0:
            ip_mat = candidates[a] @ candidates[b].T
            n_compat = np.sum(ip_mat == target_ip)
            print(f"  IP=9 pairs ({a},{b}): {n_compat}")
        else:
            print(f"  No candidates for ({a},{b})")

    # Backtrack with col order: 0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5, 11, 12
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5, 11, 12]

    # Precompute compatibility
    compat = {}
    for i in range(len(target_cols)):
        ci = target_cols[i]
        for j in range(i + 1, len(target_cols)):
            cj = target_cols[j]
            target_ip = int(G[ci, cj])
            if len(candidates[ci]) > 0 and len(candidates[cj]) > 0:
                ip_mat = candidates[ci] @ candidates[cj].T
                compat[(ci, cj)] = (ip_mat == target_ip)
            else:
                compat[(ci, cj)] = np.zeros((max(1, len(candidates[ci])), max(1, len(candidates[cj]))), dtype=bool)

    def get_compat_set(col_a, cand_a_idx, col_b):
        a, b = min(col_a, col_b), max(col_a, col_b)
        cm = compat[(a, b)]
        if col_a == a:
            return set(np.where(cm[cand_a_idx, :])[0])
        else:
            return set(np.where(cm[:, cand_a_idx])[0])

    best_depth = 0
    n_nodes = 0

    def backtrack(step, domains, assignments):
        nonlocal best_depth, n_nodes
        if time.time() > deadline:
            return None
        n_nodes += 1

        if step == len(target_cols):
            return dict(assignments)

        if step > best_depth:
            best_depth = step
            elapsed = time.time() - start_time
            dom_sizes = sorted([len(domains[c]) for c in domains])[:8]
            print(f"  Depth {step}/{len(target_cols)}: domains = {dom_sizes}... | {elapsed:.1f}s")
            sys.stdout.flush()

        chosen_col = min(domains.keys(), key=lambda c: len(domains[c]))
        dom = list(domains[chosen_col])

        if not dom:
            return None

        for cand_idx in dom:
            if time.time() > deadline:
                return None

            new_domains = {}
            feasible = True
            for other_col, other_dom in domains.items():
                if other_col == chosen_col:
                    continue
                compatible = get_compat_set(chosen_col, cand_idx, other_col)
                new_dom = [v for v in other_dom if v in compatible]
                if not new_dom:
                    feasible = False
                    break
                new_domains[other_col] = new_dom

            if not feasible:
                continue

            assignments[chosen_col] = cand_idx
            result = backtrack(step + 1, new_domains, assignments)
            if result is not None:
                return result
            del assignments[chosen_col]

        return None

    initial_domains = {ci: list(range(len(candidates[ci]))) for ci in target_cols}
    result = backtrack(0, initial_domains, {})

    if result is not None:
        R = np.zeros((N, N), dtype=np.int64)
        for k in fixed_indices:
            R[:, k] = R_sol[:, k]
        for ci, cand_idx in result.items():
            R[:, ci] = candidates[ci][cand_idx]

        if verify_decomposition(R, G):
            elapsed = time.time() - start_time
            print(f"\n*** BREAKTHROUGH! ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_val = abs(np.linalg.det(R.astype(float)))
            target_max = 1270698346568170340352
            print(f"|det(R)| = {det_val:.6e}")
            print(f"Score = {det_val / target_max:.6f}")
            return R

    return None


if __name__ == "__main__":
    main()
