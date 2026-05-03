"""
Fast variant search: enumerate all 18 variants, backtrack with many restarts.
Track max depth per variant. Focus on the most promising.
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
    rows = [0, 1, 2]
    cols = [3, 4, 5]
    variants = []
    for r_pair in itertools.combinations(rows, 2):
        for c_pair in itertools.combinations(cols, 2):
            variants.append(((r_pair[0], c_pair[0]), (r_pair[1], c_pair[1])))
            variants.append(((r_pair[0], c_pair[1]), (r_pair[1], c_pair[0])))
    return variants


def build_gram(nine_pairs):
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for (i, j) in nine_pairs:
        G[i, j] = 9; G[j, i] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng,
                             max_sol=500, deadline=None):
    n = N; k = len(placed_cols)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]
    A_rows = [[Fraction(int(placed_cols[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(G[placed_indices[i], target_idx])) for i in range(k)]
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
    nf = len(free_cols); np_ = len(pivot_cols)
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
        x = np.zeros(n, dtype=np.int64); valid = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]: valid = False; break
            x[pivot_cols[i]] = icons[i] // iden[i]
        if valid: solutions.append(x)
    elif nf <= 20:
        total = 1 << nf; bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]; rng.shuffle(si)
        for idx in si[:max_sol]:
            if deadline and time.time() > deadline: break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
            for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        for _ in range(max(1, max_sol * 500 // (1 << 20))):
            if deadline and time.time() > deadline: break
            if len(solutions) >= max_sol: break
            bs = min(1 << 20, max_sol * 200)
            fm = rng.choice([-1, 1], size=(bs, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(bs, dtype=bool)
            for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(solutions) >= max_sol: break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
                for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
    return solutions


def backtrack_with_restart(G, col_order, rng, max_time=30.0):
    """Single backtracking attempt with DFS."""
    deadline = time.time() + max_time
    max_depth = [0]
    n_cols = len(col_order)

    def dfs(cols, indices, step):
        if time.time() > deadline:
            return None
        if step == n_cols:
            return True
        ti = col_order[step]
        if step > max_depth[0]:
            max_depth[0] = step

        ms = min(50, max(5, 200 // (step + 1)))
        sols = solve_column_constraints(cols, indices, ti, G, rng, max_sol=ms,
                                        deadline=min(time.time() + 2.0, deadline))
        if not sols:
            return False
        rng.shuffle(sols)
        n_try = min(len(sols), max(3, ms // 3))
        for s in sols[:n_try]:
            if time.time() > deadline:
                return None
            cols.append(s); indices.append(ti)
            result = dfs(cols, indices, step + 1)
            if result is True:
                return True
            cols.pop(); indices.pop()
            if result is None:  # timeout
                return None
        return False

    result = dfs([], [], 0)
    if result is True:
        return max_depth[0]
    return max_depth[0]


def quick_scan_variant(variant_id, nine_pairs, n_restarts=5, time_per_restart=10.0):
    """Quick scan: a few restarts to gauge difficulty."""
    G = build_gram(nine_pairs)

    # Try multiple column orderings
    orderings = [
        [6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5] + list(range(11, 29)),
        [6, 7, 8, 9, 10, 0, 3, 1, 4, 2, 5] + list(range(11, 29)),
        [6, 0, 3, 7, 1, 4, 8, 2, 5, 9, 10] + list(range(11, 29)),
    ]

    # Also include the nine_pairs columns early
    r1, c1 = nine_pairs[0]
    r2, c2 = nine_pairs[1]
    orderings.append([6, r1, c1, r2, c2, 7, 8, 9, 10] +
                     [x for x in range(6) if x not in [r1, r2]] +
                     [x for x in range(3, 6) if x not in [c1, c2]] +
                     list(range(11, 29)))

    best_depth = 0
    best_order = None
    found_R = None

    for restart in range(n_restarts):
        rng = np.random.RandomState(42 + variant_id * 1000 + restart * 100)
        order = orderings[restart % len(orderings)]

        depth = backtrack_with_restart(G, order, rng, max_time=time_per_restart)

        if depth > best_depth:
            best_depth = depth
            best_order = order

        if depth == 29:
            # Reconstruct
            rng2 = np.random.RandomState(42 + variant_id * 1000 + restart * 100)
            cols = []
            indices = []
            for step, ci in enumerate(order):
                ms = min(50, max(5, 200 // (step + 1)))
                sols = solve_column_constraints(cols, indices, ci, G, rng2, max_sol=ms,
                                                deadline=time.time() + 60)
                if sols:
                    cols.append(sols[0])
                    indices.append(ci)
            if len(cols) == 29:
                R = np.zeros((N, N), dtype=np.int64)
                for i, ci in enumerate(indices):
                    R[:, ci] = cols[i]
                if np.array_equal(R.T @ R, G):
                    found_R = R
            break

    return best_depth, best_order, found_R


def extended_search_variant(variant_id, nine_pairs, col_order, time_limit=600.0):
    """Extended search on a single variant with many restarts."""
    G = build_gram(nine_pairs)
    deadline = time.time() + time_limit
    best_depth = 0
    attempts = 0

    while time.time() < deadline:
        attempts += 1
        remaining = deadline - time.time()
        if remaining < 3:
            break

        rng = np.random.RandomState(attempts * 7919 + variant_id * 31337)

        # Vary column ordering slightly
        order = col_order.copy()
        if attempts > 1:
            # Shuffle the "generic" columns (11-28) randomly
            generic = order[11:]
            rng.shuffle(generic)
            order = order[:11] + list(generic)

            # Sometimes also shuffle within the first 11
            if rng.random() < 0.3:
                scaffold = order[:11]
                # Keep col 6 first, shuffle the rest
                rest = scaffold[1:]
                rng.shuffle(rest)
                order = [scaffold[0]] + list(rest) + list(generic)

        trial_time = min(20.0, remaining)
        depth = backtrack_with_restart(G, order, rng, max_time=trial_time)

        if depth > best_depth:
            best_depth = depth
            print(f"  V{variant_id}: attempt {attempts}, depth {depth}/{29}", flush=True)

        if depth == 29:
            print(f"  V{variant_id}: SOLVED at attempt {attempts}!", flush=True)
            # Reconstruct solution
            rng2 = np.random.RandomState(attempts * 7919 + variant_id * 31337)
            order2 = col_order.copy()
            if attempts > 1:
                generic = order2[11:]
                rng2_dummy = np.random.RandomState(attempts * 7919 + variant_id * 31337)
                rng2_dummy.shuffle(generic)
                order2 = order2[:11] + list(generic)
                if rng2_dummy.random() < 0.3:
                    scaffold = order2[:11]
                    rest = scaffold[1:]
                    rng2_dummy.shuffle(rest)
                    order2 = [scaffold[0]] + list(rest) + list(generic)

            # Actually, just rerun with same seed
            rng3 = np.random.RandomState(attempts * 7919 + variant_id * 31337)
            # ... this is tricky. Let me just collect cols during the DFS.
            return best_depth, None  # Will handle reconstruction separately

        if attempts % 50 == 0:
            print(f"  V{variant_id}: {attempts} attempts, best depth {best_depth}/{29}", flush=True)

    return best_depth, None


def full_backtrack_collect(G, col_order, rng, max_time=60.0):
    """Backtrack and collect the solution if found."""
    deadline = time.time() + max_time
    max_depth = [0]
    n_cols = len(col_order)

    def dfs(cols, indices, step):
        if time.time() > deadline:
            return None
        if step == n_cols:
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(indices):
                R[:, ci] = cols[i]
            return R
        ti = col_order[step]
        if step > max_depth[0]:
            max_depth[0] = step

        ms = min(50, max(5, 200 // (step + 1)))
        sols = solve_column_constraints(cols, indices, ti, G, rng, max_sol=ms,
                                        deadline=min(time.time() + 2.0, deadline))
        if not sols:
            return False
        rng.shuffle(sols)
        n_try = min(len(sols), max(3, ms // 3))
        for s in sols[:n_try]:
            if time.time() > deadline:
                return None
            cols.append(s); indices.append(ti)
            result = dfs(cols, indices, step + 1)
            if isinstance(result, np.ndarray):
                return result
            cols.pop(); indices.pop()
            if result is None:
                return None
        return False

    result = dfs([], [], 0)
    if isinstance(result, np.ndarray):
        return result, max_depth[0]
    return None, max_depth[0]


def main():
    t_start = time.time()
    print("="*70)
    print("FAST VARIANT SEARCH")
    print("="*70)
    sys.stdout.flush()

    variants = enumerate_variants()

    # Phase 1: Quick scan (5 restarts x 10s each = ~50s per variant, ~15 min total)
    print(f"\nPHASE 1: Quick scan ({len(variants)} variants, ~50s each)")
    print("-"*70)
    sys.stdout.flush()

    results = {}
    for i, (p1, p2) in enumerate(variants):
        t0 = time.time()
        depth, best_order, R = quick_scan_variant(i, [p1, p2], n_restarts=5, time_per_restart=10.0)
        elapsed = time.time() - t0
        results[i] = {
            'depth': depth,
            'order': best_order,
            'R': R,
            'pairs': (p1, p2),
        }
        status = f"SOLVED!" if R is not None else f"depth {depth}/{29}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status} ({elapsed:.1f}s)", flush=True)

        if R is not None:
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_val = abs(np.linalg.det(R.astype(float)))
            score = det_val / THEORETICAL_MAX
            print(f"    Score = {score:.6f}")

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print("-"*70)
    solved = [i for i in results if results[i]['R'] is not None]
    for i in sorted(results.keys()):
        r = results[i]
        p1, p2 = r['pairs']
        status = "SOLVED!" if r['R'] is not None else f"depth {r['depth']}/{29}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")
    sys.stdout.flush()

    if solved:
        print(f"\nSolved: {solved}")
        elapsed_total = time.time() - t_start
        print(f"Total time: {elapsed_total:.1f}s")
        return

    # Phase 2: Extended search on top variants
    ranked = sorted(results.keys(), key=lambda k: -results[k]['depth'])
    top_variants = ranked[:5]
    print(f"\nPHASE 2: Extended search on top {len(top_variants)} variants")
    print(f"Variants: {top_variants} (depths: {[results[v]['depth'] for v in top_variants]})")
    print("-"*70)
    sys.stdout.flush()

    for vi in top_variants:
        r = results[vi]
        p1, p2 = r['pairs']
        G = build_gram([p1, p2])

        print(f"\nExtended search on V{vi}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9")
        print(f"  Starting depth: {r['depth']}/{29}")
        sys.stdout.flush()

        # Try many orderings with extended time
        best_depth = r['depth']
        deadline = time.time() + 300.0  # 5 minutes per variant

        attempt = 0
        while time.time() < deadline:
            attempt += 1
            remaining = deadline - time.time()
            if remaining < 3:
                break

            rng = np.random.RandomState(attempt * 7919 + vi * 31337)

            # Generate column ordering
            base_order = [6, 7, 8, 9, 10]
            # Add the nine-pair columns
            nine_cols = sorted(set([p1[0], p1[1], p2[0], p2[1]]))
            other_first = [x for x in range(6) if x not in nine_cols]
            scaffold = base_order + list(rng.permutation(nine_cols)) + list(rng.permutation(other_first))
            generic = list(rng.permutation(range(11, 29)))
            order = scaffold + generic

            trial_time = min(15.0, remaining)
            R, depth = full_backtrack_collect(G, order, rng, max_time=trial_time)

            if depth > best_depth:
                best_depth = depth
                print(f"  V{vi}: attempt {attempt}, depth {depth}/{29}", flush=True)

            if R is not None:
                if np.array_equal(R.T @ R, G) and np.all(np.abs(R) == 1):
                    print(f"\n*** BREAKTHROUGH: V{vi} SOLVED! ***", flush=True)
                    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                    det_val = abs(np.linalg.det(R.astype(float)))
                    score = det_val / THEORETICAL_MAX
                    print(f"Score = {score:.6f}")
                    elapsed_total = time.time() - t_start
                    print(f"Total time: {elapsed_total:.1f}s")
                    return

            if attempt % 20 == 0:
                print(f"  V{vi}: {attempt} attempts, best {best_depth}/{29}, {remaining:.0f}s left", flush=True)

        results[vi]['depth'] = max(results[vi]['depth'], best_depth)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("-"*70)
    for i in sorted(results.keys()):
        r = results[i]
        p1, p2 = r['pairs']
        status = "SOLVED!" if r['R'] is not None else f"depth {r['depth']}/{29}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
