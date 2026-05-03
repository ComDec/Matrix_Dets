"""
Systematic search: for each variant, try many different (top_block, bottom_vectors)
pairs and measure backtracking depth for remaining 18 columns.

Key insight: the choice of top block and bottom vectors dramatically affects
how deep the backtracker can go. We need to find the "right" starting point.
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


def solve_bottom_constraints(placed, placed_idx, target_idx, BT, rng, max_sol=500, deadline=None):
    """Solve for 18-entry +-1 bottom vector given placed bottom vectors."""
    n = 18; k = len(placed)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]
    A_rows = [[Fraction(int(placed[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(BT[placed_idx[i], target_idx])) for i in range(k)]
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
    elif nf <= 18:
        total = 1 << nf
        if total <= 1 << 20:
            bits = np.arange(total, dtype=np.int32)
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
            for _ in range(max(1, max_sol * 10)):
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
    else:
        for _ in range(max(1, max_sol * 10)):
            if deadline and time.time() > deadline: break
            if len(solutions) >= max_sol: break
            bs = min(1 << 18, max_sol * 100)
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


def find_feasible_top(G11, rng, max_time=1.0):
    """Find a random 11x11 +-1 matrix where G11 - T^T T is PSD with diag 18."""
    deadline = time.time() + max_time
    while time.time() < deadline:
        T = rng.choice([-1, 1], size=(11, 11)).astype(np.int64)
        top_gram = T.T @ T
        BT = G11 - top_gram
        if not np.all(np.diag(BT) == 18):
            continue
        # Check off-diagonal even
        off = BT[np.triu_indices(11, 1)]
        if np.any(off % 2 != 0):
            continue
        # Check PSD
        eigvals = np.linalg.eigvalsh(BT.astype(float))
        if np.min(eigvals) >= -0.01:
            return T, BT
    return None, None


def construct_bottom_vectors(BT, rng, max_time=2.0):
    """Construct 11 bottom vectors (length 18, +-1) satisfying BT as Gram."""
    deadline = time.time() + max_time

    while time.time() < deadline:
        placed = []
        placed_idx = []
        success = True
        for c in range(11):
            if time.time() > deadline:
                success = False
                break
            sols = solve_bottom_constraints(placed, placed_idx, c, BT, rng,
                                            max_sol=50, deadline=min(time.time()+1, deadline))
            if not sols:
                success = False
                break
            placed.append(sols[rng.randint(len(sols))])
            placed_idx.append(c)

        if success:
            B = np.column_stack(placed)
            bg = B.T @ B
            if np.array_equal(bg, BT):
                return placed
    return None


def backtrack_remaining(placed_cols, placed_indices, G, rng, max_time=30.0, verbose=False):
    """Backtrack remaining 18 columns with DFS."""
    remaining = list(range(11, 29))
    deadline = time.time() + max_time
    max_depth = [0]

    def dfs(cols, indices, step):
        if time.time() > deadline:
            return None
        if step == 18:
            return True
        ci = remaining[step]
        if step > max_depth[0]:
            max_depth[0] = step
            if verbose:
                print(f'    depth {step}/{18}', flush=True)

        ms = min(40, max(5, 150 // (step + 1)))
        sols = solve_column_constraints(cols, indices, ci, G, rng, max_sol=ms,
                                        deadline=min(time.time() + 2, deadline))
        if not sols:
            return False
        rng.shuffle(sols)
        n_try = min(len(sols), max(3, ms // 3))
        for s in sols[:n_try]:
            if time.time() > deadline:
                return None
            cols.append(s); indices.append(ci)
            result = dfs(cols, indices, step + 1)
            if result is True:
                return True
            cols.pop(); indices.pop()
            if result is None:
                return None
        return False

    cols = [c.copy() for c in placed_cols]
    indices = list(placed_indices)
    result = dfs(cols, indices, 0)

    if result is True:
        R = np.zeros((N, N), dtype=np.int64)
        for i, ci in enumerate(indices):
            R[:, ci] = cols[i]
        return R, max_depth[0]
    return None, max_depth[0]


def search_variant(variant_id, nine_pairs, time_limit=300.0, verbose=True):
    """Full search pipeline for one variant."""
    G = build_gram(nine_pairs)
    G11 = G[:11, :11]
    deadline = time.time() + time_limit

    if verbose:
        p1, p2 = nine_pairs
        print(f"\nV{variant_id}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9", flush=True)

    best_depth = 0
    scaffolds_built = 0
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        remaining_time = deadline - time.time()
        if remaining_time < 3:
            break

        rng = np.random.RandomState(attempt * 997 + variant_id * 31337)

        # Step 1: Find feasible top block
        T, BT = find_feasible_top(G11, rng, max_time=min(1.0, remaining_time * 0.1))
        if T is None:
            continue

        # Step 2: Construct bottom vectors
        bottom = construct_bottom_vectors(BT, rng, max_time=min(2.0, remaining_time * 0.15))
        if bottom is None:
            continue

        scaffolds_built += 1

        # Assemble 11 columns
        cols_11 = []
        for j in range(11):
            c = np.zeros(N, dtype=np.int64)
            c[:11] = T[:, j]
            c[11:] = bottom[j]
            cols_11.append(c)

        # Verify
        pg = np.column_stack(cols_11).T @ np.column_stack(cols_11)
        if not np.array_equal(pg, G11):
            continue

        # Step 3: Backtrack remaining 18
        bt_time = min(20.0, remaining_time - 1)
        R, depth = backtrack_remaining(cols_11, list(range(11)), G, rng,
                                       max_time=bt_time, verbose=False)

        if depth > best_depth:
            best_depth = depth
            if verbose:
                print(f"  V{variant_id}: scaffold #{scaffolds_built}, depth {depth}/{18}", flush=True)

        if R is not None:
            if np.array_equal(R.T @ R, G) and np.all(np.abs(R) == 1):
                if verbose:
                    print(f"  *** V{variant_id}: DECOMPOSITION FOUND! ***", flush=True)
                return R, 18

        if attempt % 50 == 0 and verbose:
            print(f"  V{variant_id}: {attempt} attempts, {scaffolds_built} scaffolds, "
                  f"best depth {best_depth}/{18}, {remaining_time:.0f}s left", flush=True)

    if verbose:
        print(f"  V{variant_id}: DONE. {attempt} attempts, {scaffolds_built} scaffolds, "
              f"best depth {best_depth}/{18}", flush=True)
    return None, best_depth


def main():
    t_start = time.time()
    print("="*70)
    print("SYSTEMATIC SCAFFOLD SEARCH")
    print("="*70)
    sys.stdout.flush()

    variants = enumerate_variants()

    # Phase 1: Quick scan (60s per variant)
    print(f"\nPHASE 1: Quick scan (60s per variant, {len(variants)} variants)")
    print("-"*70)
    sys.stdout.flush()

    results = {}
    for i, (p1, p2) in enumerate(variants):
        R, depth = search_variant(i, [p1, p2], time_limit=60.0, verbose=True)
        results[i] = {'R': R, 'depth': depth, 'pairs': (p1, p2)}

        if R is not None:
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_val = abs(np.linalg.det(R.astype(float)))
            score = det_val / THEORETICAL_MAX
            print(f"    Score = {score:.6f}")

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print("-"*70)
    for i in sorted(results.keys()):
        r = results[i]
        p1, p2 = r['pairs']
        status = "SOLVED!" if r['R'] is not None else f"depth {r['depth']}/{18}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")
    sys.stdout.flush()

    solved = [i for i in results if results[i]['R'] is not None]
    if solved:
        print(f"\nSolved: {solved}")
        elapsed = time.time() - t_start
        print(f"Total time: {elapsed:.1f}s")
        return

    # Phase 2: Extended search on best variants
    ranked = sorted(results.keys(), key=lambda k: -results[k]['depth'])
    top3 = ranked[:3]
    print(f"\nPHASE 2: Extended search on top variants {top3}")
    print(f"Depths: {[results[v]['depth'] for v in top3]}")
    print("-"*70)
    sys.stdout.flush()

    for vi in top3:
        p1, p2 = results[vi]['pairs']
        print(f"\nExtended search V{vi} (600s)...")
        R, depth = search_variant(vi, [p1, p2], time_limit=600.0, verbose=True)
        if R is not None:
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_val = abs(np.linalg.det(R.astype(float)))
            score = det_val / THEORETICAL_MAX
            print(f"Score = {score:.6f}")
            break
        results[vi]['depth'] = max(results[vi]['depth'], depth)

    # Final
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    for i in sorted(results.keys()):
        r = results[i]
        p1, p2 = r['pairs']
        status = "SOLVED!" if r['R'] is not None else f"depth {r['depth']}/{18}"
        print(f"  V{i:2d}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")
    elapsed = time.time() - t_start
    print(f"Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
