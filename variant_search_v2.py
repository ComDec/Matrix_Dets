"""
Systematic decomposition search across all 18 Gram variants.
V2: Direct column-by-column backtracking on ALL 29 columns.

Key improvements:
1. Smart column ordering (most constrained columns first)
2. Many random restarts with different seeds
3. Track max depth per variant to identify the easiest one
4. For promising variants, launch extended search
"""
import numpy as np
import time
import sys
import itertools
from fractions import Fraction
from math import lcm as math_lcm

N = 29
THEORETICAL_MAX = 1270698346568170340352


def enumerate_variants():
    """Enumerate all 18 variants."""
    rows = [0, 1, 2]
    cols = [3, 4, 5]
    variants = []
    for r_pair in itertools.combinations(rows, 2):
        for c_pair in itertools.combinations(cols, 2):
            variants.append(((r_pair[0], c_pair[0]), (r_pair[1], c_pair[1])))
            variants.append(((r_pair[0], c_pair[1]), (r_pair[1], c_pair[0])))
    return variants


def build_gram(nine_pairs):
    """Build the Gram matrix for a given variant."""
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for (i, j) in nine_pairs:
        G[i, j] = 9
        G[j, i] = 9
    for j in range(7, 11):
        G[6, j] = -3
        G[j, 6] = -3
    return G


def get_column_order(G):
    """Order columns by constraint difficulty (most constrained first).

    Columns with more special off-diagonal entries should be placed first.
    Also consider the ordering of dependencies.
    """
    # Count special (non-1, non-diagonal) entries per column
    special_count = np.zeros(N, dtype=int)
    for c in range(N):
        for r in range(N):
            if r != c and G[r, c] != 1:
                special_count[c] += 1

    # Column 6 has 4 special entries -> place first
    # Columns 0,1 or 3,4 etc have 1 special entry each
    # Columns 7-10 have 1 special entry each (the -3 with col 6)
    # Generic columns have 0 special entries

    # Strategy: Place col 6 first, then cols 7-10 (they constrain col 6),
    # then the nine-entry columns, then generic ones
    order = sorted(range(N), key=lambda c: -special_count[c])
    return order


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng,
                             max_sol=500, deadline=None):
    """Corrected RREF-based column constraint solver."""
    n = N
    k = len(placed_cols)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]
    A_rows = [[Fraction(int(placed_cols[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(G[placed_indices[i], target_idx])) for i in range(k)]
    aug = [A_rows[i] + [b_vec[i]] for i in range(k)]
    pivot_cols = []
    row_idx = 0
    for col in range(n):
        if row_idx >= k:
            break
        piv = -1
        for r in range(row_idx, k):
            if aug[r][col] != 0:
                piv = r
                break
        if piv == -1:
            continue
        if piv != row_idx:
            aug[row_idx], aug[piv] = aug[piv], aug[row_idx]
        pv = aug[row_idx][col]
        for j in range(n + 1):
            aug[row_idx][j] /= pv
        for r in range(row_idx + 1, k):
            if aug[r][col] != 0:
                f = aug[r][col]
                for j in range(n + 1):
                    aug[r][j] -= f * aug[row_idx][j]
        pivot_cols.append(col)
        row_idx += 1
    for r in range(len(pivot_cols), k):
        if aug[r][n] != 0:
            return []
    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                f = aug[r2][pc]
                for j in range(n + 1):
                    aug[r2][j] -= f * aug[i][j]
    free_cols = [c for c in range(n) if c not in pivot_cols]
    nf = len(free_cols)
    np_ = len(pivot_cols)
    ic = np.zeros((np_, nf), dtype=np.int64)
    icons = np.zeros(np_, dtype=np.int64)
    iden = np.zeros(np_, dtype=np.int64)
    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free_cols]
        lcd = 1
        for d in ds:
            lcd = math_lcm(lcd, d)
        iden[i] = lcd
        icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free_cols):
            ic[i, fi] = int(aug[i][fc] * lcd)
    solutions = []
    if nf == 0:
        x = np.zeros(n, dtype=np.int64)
        valid = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]:
                valid = False
                break
            x[pivot_cols[i]] = icons[i] // iden[i]
        if valid:
            solutions.append(x)
    elif nf <= 20:
        total = 1 << nf
        bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf):
            fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_):
            vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]
        rng.shuffle(si)
        for idx in si[:max_sol]:
            if deadline and time.time() > deadline:
                break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols):
                x[fc] = fm[idx, fi]
            for i in range(np_):
                x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        for _ in range(max(1, max_sol * 500 // (1 << 20))):
            if deadline and time.time() > deadline:
                break
            if len(solutions) >= max_sol:
                break
            bs = min(1 << 20, max_sol * 200)
            fm = rng.choice([-1, 1], size=(bs, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(bs, dtype=bool)
            for i in range(np_):
                vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(solutions) >= max_sol:
                    break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols):
                    x[fc] = fm[idx, fi]
                for i in range(np_):
                    x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
    return solutions


def backtrack_all_columns(G, col_order, rng, max_time=60.0, verbose=False):
    """Backtrack to find ALL 29 columns in the given order."""
    deadline = time.time() + max_time
    max_depth = 0

    def dfs(pc, pi, step):
        nonlocal max_depth
        if time.time() > deadline:
            return False
        if step == N:
            return True
        ti = col_order[step]

        if step > max_depth:
            max_depth = step
            if verbose:
                print(f"    depth {step}/{N}", flush=True)

        # Adaptive branching: more solutions for shallow depth, fewer for deep
        ms = min(50, max(3, 300 // (step + 1)))
        time_per_col = min(3.0, (deadline - time.time()) / max(1, N - step))
        sols = solve_column_constraints(pc, pi, ti, G, rng, max_sol=ms,
                                        deadline=min(time.time() + time_per_col, deadline))
        if not sols:
            return False

        rng.shuffle(sols)
        n_try = max(3, ms // 2)
        for s in sols[:n_try]:
            if time.time() > deadline:
                return False
            pc.append(s)
            pi.append(ti)
            if dfs(pc, pi, step + 1):
                return True
            pc.pop()
            pi.pop()
        return False

    cols = []
    indices = []

    if dfs(cols, indices, 0):
        R = np.zeros((N, N), dtype=np.int64)
        for i, ci in enumerate(indices):
            R[:, ci] = cols[i]
        return R, max_depth

    return None, max_depth


def search_variant_direct(variant_id, nine_pairs, rng, time_limit=60.0, verbose=True):
    """Search for a decomposition by direct column backtracking."""
    G = build_gram(nine_pairs)
    col_order = get_column_order(G)

    if verbose:
        p1, p2 = nine_pairs
        print(f"V{variant_id:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9  order={col_order[:6]}...", flush=True)

    deadline = time.time() + time_limit
    best_depth = 0
    attempts = 0

    while time.time() < deadline:
        attempts += 1
        remaining = deadline - time.time()
        if remaining < 2:
            break

        trial_time = min(30.0, remaining)
        R, depth = backtrack_all_columns(G, col_order, rng, max_time=trial_time, verbose=False)

        if depth > best_depth:
            best_depth = depth
            if verbose:
                print(f"  V{variant_id}: attempt {attempts}, new best depth {depth}/{N}", flush=True)

        if R is not None:
            if np.array_equal(R.T @ R, G) and np.all(np.abs(R) == 1):
                if verbose:
                    print(f"  *** V{variant_id}: DECOMPOSITION FOUND! ***", flush=True)
                return R, N
            else:
                print(f"  V{variant_id}: verification FAILED", flush=True)

    if verbose:
        print(f"  V{variant_id}: {attempts} attempts, best depth = {best_depth}/{N}", flush=True)
    return None, best_depth


def search_variant_with_scaffold(variant_id, nine_pairs, rng, time_limit=60.0, verbose=True):
    """Search using scaffold approach: first build 11 columns, then backtrack remaining 18.

    Use the column solver itself to build the first 11 columns step by step,
    then backtrack the remaining 18.
    """
    G = build_gram(nine_pairs)

    if verbose:
        p1, p2 = nine_pairs
        print(f"V{variant_id:2d} scaffold: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9", flush=True)

    deadline = time.time() + time_limit
    best_depth = 0  # out of 18 (remaining columns)
    best_total = 0  # out of 29 (all columns)
    attempts = 0

    # Column order for scaffold: 6 first (most constrained), then 7-10, then 0-5
    scaffold_order = [6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5]
    remaining_order = list(range(11, 29))

    while time.time() < deadline:
        attempts += 1
        remaining_time = deadline - time.time()
        if remaining_time < 3:
            break

        # Phase 1: Build scaffold (11 columns)
        scaffold_deadline = time.time() + min(5.0, remaining_time * 0.3)

        def build_scaffold():
            cols = []
            indices = []
            for step, ci in enumerate(scaffold_order):
                if time.time() > scaffold_deadline:
                    return None, None
                ms = min(100, max(10, 500 // (step + 1)))
                sols = solve_column_constraints(cols, indices, ci, G, rng, max_sol=ms,
                                                deadline=scaffold_deadline)
                if not sols:
                    return None, None
                # Pick random solution
                cols.append(sols[rng.randint(len(sols))])
                indices.append(ci)
            return cols, indices

        cols, indices = build_scaffold()
        if cols is None:
            continue

        # Verify partial Gram
        C = np.column_stack(cols)
        pg = C.T @ C
        G11 = G[np.ix_(indices, indices)]
        if not np.array_equal(pg, G11):
            continue

        # Phase 2: Backtrack remaining 18 columns
        bt_time = min(30.0, remaining_time - 2)
        max_bt_depth = 0

        def dfs(pc, pi, step):
            nonlocal max_bt_depth
            if time.time() > time.time() + bt_time:  # will be replaced
                return False
            if step == 18:
                return True
            ti = remaining_order[step]

            if step > max_bt_depth:
                max_bt_depth = step

            ms = min(30, max(3, 300 // (step + 1)))
            bt_deadline = time.time() + bt_time
            sols = solve_column_constraints(pc, pi, ti, G, rng, max_sol=ms,
                                            deadline=min(time.time() + 3, bt_deadline))
            if not sols:
                return False
            rng.shuffle(sols)
            for s in sols[:max(3, ms // 2)]:
                if time.time() > bt_deadline:
                    return False
                pc.append(s)
                pi.append(ti)
                if dfs(pc, pi, step + 1):
                    return True
                pc.pop()
                pi.pop()
            return False

        bt_deadline_actual = time.time() + bt_time

        def dfs2(pc, pi, step):
            nonlocal max_bt_depth
            if time.time() > bt_deadline_actual:
                return False
            if step == 18:
                return True
            ti = remaining_order[step]

            if step > max_bt_depth:
                max_bt_depth = step

            ms = min(30, max(3, 300 // (step + 1)))
            sols = solve_column_constraints(pc, pi, ti, G, rng, max_sol=ms,
                                            deadline=min(time.time() + 3, bt_deadline_actual))
            if not sols:
                return False
            rng.shuffle(sols)
            for s in sols[:max(3, ms // 2)]:
                if time.time() > bt_deadline_actual:
                    return False
                pc.append(s)
                pi.append(ti)
                if dfs2(pc, pi, step + 1):
                    return True
                pc.pop()
                pi.pop()
            return False

        if dfs2(cols, indices, 0):
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(indices):
                R[:, ci] = cols[i]
            if np.array_equal(R.T @ R, G) and np.all(np.abs(R) == 1):
                print(f"  *** V{variant_id}: DECOMPOSITION FOUND (scaffold)! ***", flush=True)
                return R, 29

        total_depth = 11 + max_bt_depth
        if total_depth > best_total:
            best_total = total_depth
            if verbose and max_bt_depth > best_depth:
                best_depth = max_bt_depth
                print(f"  V{variant_id}: attempt {attempts}, scaffold depth {max_bt_depth}/{18} (total {total_depth}/{29})", flush=True)

    return None, best_total


def main():
    print("="*70)
    print("SYSTEMATIC VARIANT DECOMPOSITION SEARCH v2")
    print("="*70)
    sys.stdout.flush()

    variants = enumerate_variants()
    print(f"\n{len(variants)} variants enumerated")

    # Phase 1: Quick scan with scaffold approach (60s per variant)
    print(f"\n{'='*70}")
    print("PHASE 1: Scaffold scan (60s per variant)")
    print(f"{'='*70}")
    sys.stdout.flush()

    results = {}
    for i, (p1, p2) in enumerate(variants):
        rng = np.random.RandomState(42 + i * 1000)
        R, depth = search_variant_with_scaffold(i, [p1, p2], rng, time_limit=60.0, verbose=True)
        results[i] = (R, depth)

        if R is not None:
            print(f"\n*** BREAKTHROUGH: Variant {i} decomposed! ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_val = abs(np.linalg.det(R.astype(float)))
            score = det_val / THEORETICAL_MAX
            print(f"Score = {score:.6f}")
            sys.stdout.flush()

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*70}")
    solved = []
    for i in sorted(results.keys()):
        R, depth = results[i]
        p1, p2 = variants[i]
        status = "SOLVED!" if R is not None else f"depth {depth}/{29}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")
        if R is not None:
            solved.append(i)
    sys.stdout.flush()

    if solved:
        print(f"\nSolved variants: {solved}")
        return

    # Phase 2: Focus on top 3 most promising variants
    top3 = sorted(results.keys(), key=lambda k: -results[k][1])[:3]
    print(f"\nTop 3 most promising: {top3}")
    print(f"{'='*70}")
    print("PHASE 2: Extended search (300s per variant)")
    print(f"{'='*70}")
    sys.stdout.flush()

    for vi in top3:
        p1, p2 = variants[vi]
        rng = np.random.RandomState(99999 + vi)
        R, depth = search_variant_with_scaffold(vi, [p1, p2], rng, time_limit=300.0, verbose=True)
        results[vi] = (R, depth) if R is not None else (None, max(results[vi][1], depth))

        if R is not None:
            print(f"\n*** BREAKTHROUGH: Variant {vi} decomposed! ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            break

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for i in sorted(results.keys()):
        R, depth = results[i]
        p1, p2 = variants[i]
        status = "SOLVED!" if R is not None else f"depth {depth}/{29}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")


if __name__ == "__main__":
    main()
