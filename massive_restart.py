"""
Massive restart backtracker: try thousands of random seeds
with different column orderings, tracking exact depth distribution.
This helps determine if depth 20 is a hard barrier or just rare to pass.
"""
import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm
from collections import Counter

N = 29
THEORETICAL_MAX = 1270698346568170340352


def build_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    G[0, 3] = G[3, 0] = 9
    G[1, 4] = G[4, 1] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng,
                             max_sol=50, deadline=None):
    n = N; k = len(placed_cols)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 50))]
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
    return solutions


def single_forward_pass(G, rng, order=None):
    """Single greedy forward pass. Return max depth reached."""
    if order is None:
        order = list(range(N))
        rng.shuffle(order)

    cols = []
    indices = []
    for ci in order:
        sols = solve_column_constraints(cols, indices, ci, G, rng, max_sol=20, deadline=time.time() + 2)
        if not sols:
            return len(cols)
        cols.append(sols[rng.randint(len(sols))])
        indices.append(ci)

    return len(cols)


def single_dfs_pass(G, rng, order=None, max_time=10.0):
    """DFS with limited branching. Return max depth."""
    if order is None:
        order = list(range(N))
        rng.shuffle(order)

    deadline = time.time() + max_time
    max_depth = [0]

    def dfs(cols, indices, step):
        if time.time() > deadline:
            return None
        if step == N:
            return True
        ci = order[step]
        if step > max_depth[0]:
            max_depth[0] = step

        ms = min(20, max(3, 100 // (step + 1)))
        sols = solve_column_constraints(cols, indices, ci, G, rng, max_sol=ms,
                                        deadline=min(time.time() + 1, deadline))
        if not sols:
            return False
        rng.shuffle(sols)
        n_try = min(len(sols), max(2, ms // 3))
        for s in sols[:n_try]:
            if time.time() > deadline:
                return None
            cols.append(s); indices.append(ci)
            r = dfs(cols, indices, step + 1)
            if r is True:
                return True
            cols.pop(); indices.pop()
            if r is None:
                return None
        return False

    result = dfs([], [], 0)
    if result is True:
        return N
    return max_depth[0]


def main():
    t_start = time.time()
    print("="*70)
    print("MASSIVE RESTART SEARCH")
    print("Variant 0: G[0,3]=9, G[1,4]=9")
    print("="*70)
    sys.stdout.flush()

    G = build_gram()

    # Phase 1: Many forward passes to get depth distribution
    print("\nPhase 1: Forward pass depth distribution (2000 trials)")
    print("-"*70)
    sys.stdout.flush()

    depth_counter = Counter()
    for trial in range(2000):
        rng = np.random.RandomState(trial)
        d = single_forward_pass(G, rng)
        depth_counter[d] += 1
        if d >= 25:
            print(f"  Trial {trial}: depth {d} !!!", flush=True)
        if trial % 200 == 199:
            elapsed = time.time() - t_start
            print(f"  {trial+1} trials done ({elapsed:.0f}s). Distribution: {dict(sorted(depth_counter.items()))}", flush=True)

    print(f"\nForward pass distribution:")
    for d in sorted(depth_counter.keys()):
        print(f"  Depth {d:2d}: {depth_counter[d]:5d} ({100*depth_counter[d]/2000:.1f}%)")

    # Phase 2: DFS passes with more branching
    print(f"\n{'='*70}")
    print("Phase 2: DFS with branching (10s per attempt)")
    print("-"*70)
    sys.stdout.flush()

    best_depth = 0
    dfs_counter = Counter()
    for trial in range(200):
        elapsed = time.time() - t_start
        if elapsed > 7200:
            break

        rng = np.random.RandomState(trial * 31337)
        d = single_dfs_pass(G, rng, max_time=10.0)
        dfs_counter[d] += 1

        if d > best_depth:
            best_depth = d
            print(f"  Trial {trial}: NEW BEST depth {d}/{N}", flush=True)

        if d == N:
            print(f"\n*** SOLVED at trial {trial}! ***")
            break

        if trial % 20 == 19:
            print(f"  {trial+1} DFS trials, best {best_depth}/{N}. Distribution: {dict(sorted(dfs_counter.items()))}", flush=True)

    print(f"\nDFS distribution:")
    for d in sorted(dfs_counter.keys()):
        print(f"  Depth {d:2d}: {dfs_counter[d]:5d}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
