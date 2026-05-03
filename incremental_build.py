"""
Incremental column construction with randomized restarts.

Build columns one at a time. Each column is constrained by inner products
with all previously placed columns. Use exact enumeration of valid candidates
when the free dimension is small enough, random sampling otherwise.

Key: Order columns to minimize search tree size.
Place the "special" columns first (those with non-standard inner products),
then the generic columns.

Column ordering strategy:
1. Col 0 (fixed to all +1)
2. Col 3 (IP=9 with col 0) - very constrained
3. Col 1 (IP=1 with col 0, IP=1 with col 3)
4. Col 4 (IP=9 with col 1)
5. Col 6 (IP=1 with 0,3,1,4)
6-9. Cols 7,8,9,10 (IP=-3 with col 6)
10. Col 2 (IP=1 with all placed)
11. Col 5 (IP=1 with all placed)
12-29. Cols 11-28 (IP=1 with all placed)
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


def enumerate_pm1_solutions(constraint_vecs, constraint_vals, rng, max_sol=10000):
    """
    Find +/-1 vectors satisfying linear constraints.
    Returns list of solutions as numpy arrays.
    """
    k = len(constraint_vecs)
    n = N

    if k == 0:
        # Return a few random vectors
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
    elif nf <= 22:
        total = 1 << nf
        if total <= 4194304:  # 2^22
            bits = np.arange(total, dtype=np.int32)
            fm = np.empty((total, nf), dtype=np.int64)
            for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(total, dtype=bool)
            for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
            si = np.where(vm)[0]
            rng.shuffle(si)
            for idx in si[:max_sol]:
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
                for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
        else:
            # Sample
            batch_size = min(1 << 20, max_sol * 500)
            for _ in range(5):
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


def incremental_search(G, time_limit=3600, verbose=True):
    """
    Build all 29 columns incrementally with restarts.
    """
    if verbose:
        print("Incremental column construction with restarts")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    start_time = time.time()

    # Column order: special columns first
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))

    best_depth = 0
    n_attempts = 0

    while time.time() < deadline:
        n_attempts += 1
        seed = n_attempts * 137 + 42
        rng = np.random.RandomState(seed)

        placed_cols = []
        placed_indices = []

        # Step 0: Fix column 0 = all +1
        col0 = np.ones(N, dtype=np.int64)
        placed_cols.append(col0)
        placed_indices.append(0)

        success = True
        for step in range(1, N):
            if time.time() > deadline:
                success = False
                break

            target_idx = col_order[step]

            constraint_vecs = placed_cols
            constraint_vals = [int(G[placed_indices[i], target_idx]) for i in range(len(placed_cols))]

            solutions = enumerate_pm1_solutions(
                constraint_vecs, constraint_vals, rng,
                max_sol=min(10000, max(100, 5000 // (step + 1)))
            )

            if not solutions:
                if step > best_depth:
                    best_depth = step
                    elapsed = time.time() - start_time
                    if verbose:
                        print(f"  Attempt {n_attempts}: depth {step}/{N} (col {target_idx}), {len(solutions)} sols, {elapsed:.1f}s")
                        sys.stdout.flush()
                success = False
                break

            # Pick a random solution
            chosen = solutions[rng.randint(len(solutions))]
            placed_cols.append(chosen)
            placed_indices.append(target_idx)

        if success and len(placed_cols) == N:
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(placed_indices):
                R[:, ci] = placed_cols[i]

            if verify_decomposition(R, G):
                elapsed = time.time() - start_time
                if verbose:
                    print(f"\n*** BREAKTHROUGH! Attempt {n_attempts}, {elapsed:.1f}s ***")
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                det_val = abs(np.linalg.det(R.astype(float)))
                target_max = 1270698346568170340352
                print(f"|det(R)| = {det_val:.6e}")
                print(f"Score = {det_val / target_max:.6f}")
                return R

        depth = len(placed_cols)
        if depth > best_depth:
            best_depth = depth
            elapsed = time.time() - start_time
            if verbose:
                print(f"  Attempt {n_attempts}: depth {depth}/{N}, {elapsed:.1f}s")
                sys.stdout.flush()

        if n_attempts % 1000 == 0 and verbose:
            elapsed = time.time() - start_time
            print(f"  {n_attempts} attempts, best_depth={best_depth}, {elapsed:.0f}s")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  No solution in {elapsed:.0f}s, {n_attempts} attempts, best_depth={best_depth}")
    return None


def incremental_with_backtracking(G, time_limit=3600, verbose=True):
    """
    Incremental construction with limited backtracking.
    Place columns one at a time, backtrack when stuck.
    """
    if verbose:
        print("Incremental construction with backtracking")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    start_time = time.time()

    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))

    best_depth = 0
    n_restarts = 0

    while time.time() < deadline:
        n_restarts += 1
        rng = np.random.RandomState(n_restarts * 31 + 7)

        placed_cols = [np.ones(N, dtype=np.int64)]  # col 0 = all +1
        placed_indices = [0]

        def dfs(step):
            nonlocal best_depth
            if time.time() > deadline:
                return False
            if step == N:
                return True

            target_idx = col_order[step]

            constraint_vecs = placed_cols
            constraint_vals = [int(G[placed_indices[i], target_idx]) for i in range(len(placed_cols))]

            # How many solutions to enumerate depends on depth
            max_sol = 500 if step < 5 else (200 if step < 11 else 50)
            solutions = enumerate_pm1_solutions(
                constraint_vecs, constraint_vals, rng, max_sol=max_sol
            )

            if not solutions:
                return False

            rng.shuffle(solutions)

            if step > best_depth:
                best_depth = step
                elapsed = time.time() - start_time
                if verbose:
                    print(f"  Restart {n_restarts}: depth {step}/{N} (col {target_idx}), {len(solutions)} sols, {elapsed:.1f}s")
                    sys.stdout.flush()

            # Try solutions (limit branching)
            n_try = min(len(solutions), 20 if step < 5 else (10 if step < 10 else 3))

            for sol in solutions[:n_try]:
                if time.time() > deadline:
                    return False
                placed_cols.append(sol)
                placed_indices.append(target_idx)
                if dfs(step + 1):
                    return True
                placed_cols.pop()
                placed_indices.pop()

            return False

        if dfs(1):  # Start from step 1 (col 0 already placed)
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(placed_indices):
                R[:, ci] = placed_cols[i]

            if verify_decomposition(R, G):
                elapsed = time.time() - start_time
                if verbose:
                    print(f"\n*** BREAKTHROUGH! Restart {n_restarts}, {elapsed:.1f}s ***")
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                det_val = abs(np.linalg.det(R.astype(float)))
                target_max = 1270698346568170340352
                print(f"|det(R)| = {det_val:.6e}")
                print(f"Score = {det_val / target_max:.6f}")
                return R

        if n_restarts % 10 == 0 and verbose:
            elapsed = time.time() - start_time
            print(f"  {n_restarts} restarts, best_depth={best_depth}, {elapsed:.0f}s")
            sys.stdout.flush()

    if verbose:
        elapsed = time.time() - start_time
        print(f"\n  No solution in {elapsed:.0f}s, {n_restarts} restarts, best_depth={best_depth}")
    return None


def main():
    G = build_new_gram()
    print("=" * 70)
    print("INCREMENTAL COLUMN CONSTRUCTION")
    print("=" * 70)
    sys.stdout.flush()

    # Try pure greedy first (very fast per attempt)
    print("\n--- Greedy with restarts ---")
    R = incremental_search(G, time_limit=300, verbose=True)
    if R is not None:
        return R

    # Try with backtracking
    print("\n--- With backtracking ---")
    R = incremental_with_backtracking(G, time_limit=3300, verbose=True)
    if R is not None:
        return R

    print("\nNo solution found.")
    return None


if __name__ == "__main__":
    main()
