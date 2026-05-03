"""
Smart backtracker for 29x29 Gram decomposition.

Key insight: With 11 columns fixed, each remaining column has only 160 valid
+/-1 candidates. We need to find 18 columns from these candidate sets such that
all mutual inner products equal 1.

This is a constraint satisfaction problem over 18 variables, each with ~160 values,
with 153 pairwise constraints (IP = 1 between all remaining pairs).

Strategy:
1. Enumerate all 160 candidates for each of the 18 remaining columns
2. Precompute all pairwise inner products between candidates
3. Use constraint propagation + backtracking (arc consistency)
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


def enumerate_pm1_solutions(col_idx, fixed_cols_dict, G):
    """Enumerate all +/-1 vectors satisfying inner product constraints with fixed columns."""
    constraint_vecs = []
    constraint_vals = []
    for b in sorted(fixed_cols_dict.keys()):
        constraint_vecs.append(fixed_cols_dict[b])
        constraint_vals.append(int(G[col_idx, b]))

    k = len(constraint_vecs)
    n = N

    A_rows = [[Fraction(int(constraint_vecs[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(constraint_vals[i])) for i in range(k)]
    aug = [A_rows[i] + [b_vec[i]] for i in range(k)]

    pivot_cols_list = []; row_idx = 0
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
        pivot_cols_list.append(col); row_idx += 1

    for r in range(len(pivot_cols_list), k):
        if aug[r][n] != 0: return []

    for i in range(len(pivot_cols_list) - 1, -1, -1):
        pc = pivot_cols_list[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                f = aug[r2][pc]
                for j in range(n + 1): aug[r2][j] -= f * aug[i][j]

    free_cols = [c for c in range(n) if c not in pivot_cols_list]
    nf = len(free_cols)
    np_ = len(pivot_cols_list)

    ic = np.zeros((np_, nf), dtype=np.int64)
    icons = np.zeros(np_, dtype=np.int64)
    iden = np.zeros(np_, dtype=np.int64)

    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free_cols]
        lcd = 1
        for d in ds: lcd = math_lcm(lcd, d)
        iden[i] = lcd; icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free_cols): ic[i, fi] = int(aug[i][fc] * lcd)

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
        for i in range(np_): x[pivot_cols_list[i]] = pv_arr[idx, i] // iden[i]
        solutions.append(x)

    return solutions


def solve_with_backtrack(G, time_limit=3600, verbose=True):
    """Main solver: enumerate candidates, precompute compatibility, backtrack."""
    if verbose:
        print("Smart backtracker with candidate enumeration")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    start_time = time.time()

    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)

    col0 = partial[:, 0]
    negate_rows = (col0 == -1)
    fixed_cols = {}
    for col_idx in range(11):
        col = partial[:, col_idx].copy()
        col[negate_rows] *= -1
        fixed_cols[col_idx] = col

    remaining = list(range(11, N))  # 18 columns
    n_rem = len(remaining)

    # Step 1: Enumerate all candidates for each remaining column
    if verbose:
        print(f"  Enumerating candidates for {n_rem} remaining columns...")
        sys.stdout.flush()

    candidates = {}
    for col_idx in remaining:
        cands = enumerate_pm1_solutions(col_idx, fixed_cols, G)
        candidates[col_idx] = np.array(cands, dtype=np.int64)  # shape (n_cands, 29)
        if verbose:
            print(f"    Col {col_idx}: {len(cands)} candidates")
    sys.stdout.flush()

    # Step 2: Precompute inner products between ALL candidate pairs
    # For columns a and b (both remaining), target IP = 1
    # IP(cand_i, cand_j) = cand_i . cand_j
    # We need #disagree = (29-1)/2 = 14, so IP = 1

    if verbose:
        print(f"  Precomputing compatibility matrices...")
        sys.stdout.flush()

    # compatibility[a][b] is a boolean matrix: compat[i,j] = True if
    # candidates[a][i] . candidates[b][j] == target_ip
    compat = {}
    for idx_a in range(n_rem):
        a = remaining[idx_a]
        for idx_b in range(idx_a + 1, n_rem):
            b = remaining[idx_b]
            target_ip = int(G[a, b])  # = 1
            # Compute IP matrix: candidates[a] @ candidates[b].T
            ip_matrix = candidates[a] @ candidates[b].T
            compat[(a, b)] = (ip_matrix == target_ip)

            n_compat = np.sum(compat[(a, b)])
            total = compat[(a, b)].size
            if verbose and (idx_a < 3 or (idx_a + idx_b) % 20 == 0):
                print(f"    ({a},{b}): {n_compat}/{total} compatible pairs ({100*n_compat/total:.1f}%)")

    sys.stdout.flush()

    # Step 3: Backtracking with arc consistency (AC-3)
    if verbose:
        print(f"\n  Starting constraint propagation + backtracking...")
        sys.stdout.flush()

    # Order columns by number of candidates (fewest first for better pruning)
    col_order = sorted(remaining, key=lambda c: len(candidates[c]))

    # Domain: for each column, which candidate indices are still viable
    # We'll use forward checking (not full AC-3 to start)

    best_depth = 0
    n_backtracks = 0
    n_solutions_checked = 0

    def forward_check(domains, placed_col, placed_cand_idx):
        """
        Given that placed_col is assigned placed_cand_idx,
        remove incompatible candidates from all unassigned columns.
        Returns new domains (or None if any domain becomes empty).
        """
        new_domains = {}
        for other_col, dom in domains.items():
            if other_col == placed_col:
                continue
            # Find compatible candidates
            a, b = min(placed_col, other_col), max(placed_col, other_col)
            cm = compat[(a, b)]

            if placed_col == a:
                # cm[placed_cand_idx, j] for j in dom
                mask = cm[placed_cand_idx, :]
                new_dom = [j for j in dom if mask[j]]
            else:
                # cm[j, placed_cand_idx] for j in dom
                mask = cm[:, placed_cand_idx]
                new_dom = [j for j in dom if mask[j]]

            if not new_dom:
                return None
            new_domains[other_col] = new_dom

        return new_domains

    def backtrack(step, domains, assignments):
        nonlocal best_depth, n_backtracks, n_solutions_checked

        if time.time() > deadline:
            return None

        if step == n_rem:
            # All columns assigned
            return dict(assignments)

        if step > best_depth:
            best_depth = step
            elapsed = time.time() - start_time
            if verbose:
                remaining_dom_sizes = [len(domains[c]) for c in sorted(domains.keys())]
                print(f"  Depth {step}/{n_rem}: domains = {remaining_dom_sizes[:8]}... | {elapsed:.1f}s | bt={n_backtracks}")
                sys.stdout.flush()

        # Choose the column with smallest domain (MRV heuristic)
        unassigned = [c for c in col_order if c in domains]
        if not unassigned:
            return None

        chosen_col = min(unassigned, key=lambda c: len(domains[c]))
        dom = domains[chosen_col]

        # Try each candidate
        for cand_idx in dom:
            if time.time() > deadline:
                return None

            n_solutions_checked += 1

            # Forward check
            remaining_domains = {c: d for c, d in domains.items() if c != chosen_col}
            new_domains = forward_check(remaining_domains, chosen_col, cand_idx)

            if new_domains is None:
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
    initial_domains = {col_idx: list(range(len(candidates[col_idx]))) for col_idx in remaining}

    if verbose:
        print(f"  Initial domain sizes: {[len(initial_domains[c]) for c in remaining]}")
        print(f"  Search space: {np.prod([len(initial_domains[c]) for c in remaining]):.2e}")
        sys.stdout.flush()

    result = backtrack(0, initial_domains, {})

    if result is not None:
        # Build R matrix
        R = np.zeros((N, N), dtype=np.int64)
        for ci, col in fixed_cols.items():
            R[:, ci] = col
        for col_idx, cand_idx in result.items():
            R[:, col_idx] = candidates[col_idx][cand_idx]

        if verify_decomposition(R, G):
            elapsed = time.time() - start_time
            if verbose:
                print(f"\n  SUCCESS! Found valid decomposition in {elapsed:.1f}s")
                print(f"  Backtracks: {n_backtracks}, solutions checked: {n_solutions_checked}")
            return R
        else:
            if verbose:
                err = int(np.sum((R.T @ R - G) ** 2))
                print(f"  ERROR: Solution does not verify! err={err}")

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  No solution found in {elapsed:.1f}s")
        print(f"  Best depth: {best_depth}/{n_rem}")
        print(f"  Backtracks: {n_backtracks}, solutions checked: {n_solutions_checked}")

    return None


def solve_with_multiple_seeds(G, time_limit=3600, verbose=True):
    """
    Try multiple random orderings of the partial 11 columns.

    The partial 11 columns might not be the best starting point.
    We can also try different orderings for the remaining columns
    and different random restarts with randomized value ordering.
    """
    if verbose:
        print("Multi-seed smart backtracker")
        sys.stdout.flush()

    deadline = time.time() + time_limit

    # First try the main approach
    R = solve_with_backtrack(G, time_limit=min(time_limit, 3600), verbose=verbose)
    if R is not None:
        return R

    return None


def solve_with_arc_consistency(G, time_limit=3600, verbose=True):
    """
    Enhanced solver with MAC (Maintaining Arc Consistency) instead of simple forward checking.
    """
    if verbose:
        print("MAC (Maintaining Arc Consistency) backtracker")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    start_time = time.time()

    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)

    col0 = partial[:, 0]
    negate_rows = (col0 == -1)
    fixed_cols = {}
    for col_idx in range(11):
        col = partial[:, col_idx].copy()
        col[negate_rows] *= -1
        fixed_cols[col_idx] = col

    remaining = list(range(11, N))
    n_rem = len(remaining)

    if verbose:
        print(f"  Enumerating candidates...")
        sys.stdout.flush()

    candidates = {}
    for col_idx in remaining:
        cands = enumerate_pm1_solutions(col_idx, fixed_cols, G)
        candidates[col_idx] = np.array(cands, dtype=np.int64)
        if verbose:
            print(f"    Col {col_idx}: {len(cands)} candidates")

    if verbose:
        print(f"  Precomputing compatibility...")
        sys.stdout.flush()

    # Precompute compatibility as before
    compat = {}
    for idx_a in range(n_rem):
        a = remaining[idx_a]
        for idx_b in range(idx_a + 1, n_rem):
            b = remaining[idx_b]
            target_ip = int(G[a, b])
            ip_matrix = candidates[a] @ candidates[b].T
            compat[(a, b)] = (ip_matrix == target_ip)

    # Precompute for each pair, the compatible sets for efficient lookup
    # compat_sets[(a,b)][i] = set of j's compatible with candidate i of column a
    compat_sets = {}
    for (a, b), cm in compat.items():
        compat_sets[(a, b)] = {}
        for i in range(cm.shape[0]):
            compat_sets[(a, b)][i] = set(np.where(cm[i])[0])
        compat_sets[(b, a)] = {}
        for j in range(cm.shape[1]):
            compat_sets[(b, a)][j] = set(np.where(cm[:, j])[0])

    def get_compat(col1, cand1_idx, col2):
        """Get set of candidates for col2 compatible with cand1_idx of col1."""
        a, b = min(col1, col2), max(col1, col2)
        if col1 == a:
            return compat_sets[(a, b)].get(cand1_idx, set())
        else:
            return compat_sets[(b, a)].get(cand1_idx, set())

    def ac3(domains, arcs=None):
        """
        AC-3 algorithm. Returns pruned domains or None if inconsistent.
        arcs: initial queue of arcs to process, or None for all arcs.
        """
        if arcs is None:
            queue = []
            cols = list(domains.keys())
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    queue.append((cols[i], cols[j]))
                    queue.append((cols[j], cols[i]))
        else:
            queue = list(arcs)

        while queue:
            if time.time() > deadline:
                return domains  # Return what we have if time runs out
            xi, xj = queue.pop(0)
            if xi not in domains or xj not in domains:
                continue

            # Revise: remove values from domains[xi] that have no support in domains[xj]
            dom_i = domains[xi]
            dom_j_set = set(domains[xj])
            new_dom_i = []

            for vi in dom_i:
                # Check if any value in domains[xj] is compatible with vi
                compatible = get_compat(xi, vi, xj)
                if compatible & dom_j_set:
                    new_dom_i.append(vi)

            if len(new_dom_i) < len(dom_i):
                if not new_dom_i:
                    return None  # Domain wipeout
                domains[xi] = new_dom_i
                # Add arcs from other neighbors of xi
                for xk in domains:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return domains

    # Initial arc consistency
    if verbose:
        print(f"\n  Running initial AC-3...")
        sys.stdout.flush()

    initial_domains = {col_idx: list(range(len(candidates[col_idx]))) for col_idx in remaining}
    initial_domains = ac3(initial_domains)

    if initial_domains is None:
        if verbose:
            print("  INFEASIBLE: AC-3 found inconsistency!")
        return None

    if verbose:
        dom_sizes = [len(initial_domains[c]) for c in remaining]
        print(f"  After AC-3: domain sizes = {dom_sizes}")
        print(f"  Search space: {np.prod([float(s) for s in dom_sizes]):.2e}")
        sys.stdout.flush()

    # MAC backtracking
    best_depth = 0
    n_backtracks = 0

    def mac_backtrack(step, domains, assignments):
        nonlocal best_depth, n_backtracks

        if time.time() > deadline:
            return None

        if step == n_rem:
            return dict(assignments)

        if step > best_depth:
            best_depth = step
            elapsed = time.time() - start_time
            if verbose:
                dom_sizes = sorted([len(domains[c]) for c in domains])
                print(f"  Depth {step}/{n_rem}: domains = {dom_sizes} | {elapsed:.1f}s | bt={n_backtracks}")
                sys.stdout.flush()

        # MRV: choose unassigned variable with smallest domain
        unassigned = [c for c in remaining if c in domains]
        if not unassigned:
            return None

        chosen_col = min(unassigned, key=lambda c: len(domains[c]))
        dom = list(domains[chosen_col])

        # Value ordering: try values that leave the most options for neighbors
        # (For speed, just try in order - can add LCV later if needed)

        for cand_idx in dom:
            if time.time() > deadline:
                return None

            # Make assignment
            new_domains = {}
            for c, d in domains.items():
                if c == chosen_col:
                    continue
                # Filter domain of c based on compatibility with chosen assignment
                compatible = get_compat(chosen_col, cand_idx, c)
                new_d = [v for v in d if v in compatible]
                if not new_d:
                    n_backtracks += 1
                    new_domains = None
                    break
                new_domains[c] = new_d

            if new_domains is None:
                continue

            # Run AC-3 on the reduced domains
            # Only need to check arcs involving neighbors of chosen_col
            arcs = []
            for c1 in new_domains:
                for c2 in new_domains:
                    if c1 != c2:
                        arcs.append((c1, c2))

            new_domains = ac3(dict(new_domains), arcs)

            if new_domains is None:
                n_backtracks += 1
                continue

            assignments[chosen_col] = cand_idx
            result = mac_backtrack(step + 1, new_domains, assignments)
            if result is not None:
                return result
            del assignments[chosen_col]
            n_backtracks += 1

        return None

    result = mac_backtrack(0, initial_domains, {})

    if result is not None:
        R = np.zeros((N, N), dtype=np.int64)
        for ci, col in fixed_cols.items():
            R[:, ci] = col
        for col_idx, cand_idx in result.items():
            R[:, col_idx] = candidates[col_idx][cand_idx]

        if verify_decomposition(R, G):
            elapsed = time.time() - start_time
            if verbose:
                print(f"\n  SUCCESS! Found valid decomposition in {elapsed:.1f}s")
                print(f"  Backtracks: {n_backtracks}")
            return R

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  No solution found in {elapsed:.1f}s")
        print(f"  Best depth: {best_depth}/{n_rem}, backtracks: {n_backtracks}")

    return None


def main():
    G = build_new_gram()
    print("=" * 70)
    print("SMART BACKTRACKER: 29x29 Gram decomposition")
    print("=" * 70)
    sys.stdout.flush()

    # Try forward-checking backtracker first (faster per node)
    print("\n--- Forward Checking Backtracker ---")
    R = solve_with_backtrack(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)

        # Compute determinant
        det_val = abs(np.linalg.det(R.astype(float)))
        target_max = 1270698346568170340352
        print(f"|det(R)| = {det_val:.6e}")
        print(f"Score = {det_val / target_max:.6f}")
        return R

    # Try MAC backtracker (stronger pruning)
    print("\n--- MAC Backtracker ---")
    R = solve_with_arc_consistency(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)

        det_val = abs(np.linalg.det(R.astype(float)))
        target_max = 1270698346568170340352
        print(f"|det(R)| = {det_val:.6e}")
        print(f"Score = {det_val / target_max:.6f}")
        return R

    print("\nAll approaches exhausted.")
    return None


if __name__ == "__main__":
    main()
