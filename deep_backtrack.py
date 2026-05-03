"""Deep backtracking search for Gram decomposition."""
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

def enumerate_pm1_solutions(constraint_vecs, constraint_vals, rng, max_sol=10000):
    k = len(constraint_vecs)
    n = N
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
    elif nf <= 22 and (1 << nf) <= 4194304:
        total = 1 << nf; bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]; rng.shuffle(si)
        for idx in si[:max_sol]:
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
            for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        batch_size = min(1 << 20, max_sol * 500)
        for _ in range(10):
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

def main():
    G = build_new_gram()
    print("=" * 70)
    print("DEEP BACKTRACKING SEARCH")
    print("=" * 70)
    sys.stdout.flush()

    deadline = time.time() + 3600
    start_time = time.time()

    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))

    global_best_depth = 0
    n_restarts = 0
    total_nodes = 0

    while time.time() < deadline:
        n_restarts += 1
        rng = np.random.RandomState(n_restarts)

        placed_cols = [np.ones(N, dtype=np.int64)]
        placed_indices = [0]
        best_depth_this_restart = 0

        def dfs(step):
            nonlocal global_best_depth, best_depth_this_restart, total_nodes
            if time.time() > deadline: return False
            if step == N: return True
            total_nodes += 1
            target_idx = col_order[step]
            constraint_vecs = placed_cols
            constraint_vals = [int(G[placed_indices[i], target_idx]) for i in range(len(placed_cols))]
            solutions = enumerate_pm1_solutions(constraint_vecs, constraint_vals, rng, max_sol=100000)
            if not solutions: return False
            if step > best_depth_this_restart: best_depth_this_restart = step
            if step > global_best_depth:
                global_best_depth = step
                elapsed = time.time() - start_time
                print(f"  Restart {n_restarts}: depth {step}/{N} (col {target_idx}), {len(solutions)} sols, nodes={total_nodes}, {elapsed:.1f}s")
                sys.stdout.flush()
            rng.shuffle(solutions)
            if step <= 5: n_try = min(len(solutions), 5)
            elif step <= 10: n_try = min(len(solutions), 10)
            elif step <= 15: n_try = min(len(solutions), 50)
            else: n_try = len(solutions)
            for sol in solutions[:n_try]:
                if time.time() > deadline: return False
                placed_cols.append(sol); placed_indices.append(target_idx)
                if dfs(step + 1): return True
                placed_cols.pop(); placed_indices.pop()
            return False

        if dfs(1):
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(placed_indices):
                R[:, ci] = placed_cols[i]
            if verify_decomposition(R, G):
                elapsed = time.time() - start_time
                print(f"\n*** BREAKTHROUGH! Restart {n_restarts}, {elapsed:.1f}s ***")
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                det_val = abs(np.linalg.det(R.astype(float)))
                target_max = 1270698346568170340352
                print(f"|det(R)| = {det_val:.6e}")
                print(f"Score = {det_val / target_max:.6f}")
                return R

        if n_restarts % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Restart {n_restarts}: best_this={best_depth_this_restart}, global={global_best_depth}, nodes={total_nodes}, {elapsed:.0f}s")
            sys.stdout.flush()

    print(f"\nNo solution in {time.time()-start_time:.0f}s, {n_restarts} restarts, best={global_best_depth}")
    return None

if __name__ == "__main__":
    main()
