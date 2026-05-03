"""
Decompose the new Gram by modifying the Solomon seed.

The new Gram differs from Solomon in the {0,1,2}x{3,4,5} block:
  Solomon: all 9 entries = 5
  New:     G[0,3]=G[1,4]=9, rest = 1

Strategy: Fix the Solomon seed columns 6-28 (the unconstrained ones),
then use backtracking to find new columns 0-5 that satisfy the new Gram.
"""
import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm

N = 29

SEED = np.array([
    [-1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,1,1,1,-1,1,1,1,-1],
    [-1,1,-1,1,1,1,-1,1,1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,1],
    [1,-1,-1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,1],
    [1,1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1],
    [1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,1,1],
    [1,1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1],
    [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1],
    [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,1,1,1,1,1,-1],
    [1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,1],
    [1,-1,1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,-1,1,1],
    [1,-1,1,1,1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,1,1,-1,-1,1],
    [1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,1],
    [1,-1,1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,-1],
    [1,1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,1],
    [1,1,-1,1,1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,1,-1],
    [1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1],
    [1,1,-1,-1,1,1,1,-1,1,1,-1,1,-1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,1,-1,1],
    [-1,1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
    [-1,1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,1,-1,1,1,1,1,-1,-1,-1],
    [-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1,1],
    [-1,1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1],
    [1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,1,1,1,-1,1,-1,1],
    [1,-1,1,1,-1,1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1],
    [-1,1,1,1,-1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1],
    [-1,1,1,1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,1,1],
    [1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,-1],
    [1,-1,1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,-1,1],
], dtype=np.int64)

def build_new_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    G[0, 3] = 9; G[3, 0] = 9
    G[1, 4] = 9; G[4, 1] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G

def verify_decomposition(R, G):
    return np.array_equal(R.astype(np.int64).T @ R.astype(np.int64), G)

def solve_column_constraints(placed_cols, placed_indices, target_idx, G,
                             rng, max_sol=5000, deadline=None):
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
        pivot = -1
        for r in range(row_idx, k):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row_idx:
            aug[row_idx], aug[pivot] = aug[pivot], aug[row_idx]
        piv_val = aug[row_idx][col]
        for j in range(n + 1):
            aug[row_idx][j] /= piv_val
        for r in range(row_idx + 1, k):
            if aug[r][col] != 0:
                factor = aug[r][col]
                for j in range(n + 1):
                    aug[r][j] -= factor * aug[row_idx][j]
        pivot_cols.append(col)
        row_idx += 1

    for r in range(len(pivot_cols), k):
        if aug[r][n] != 0:
            return []

    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                factor = aug[r2][pc]
                for j in range(n + 1):
                    aug[r2][j] -= factor * aug[i][j]

    free_cols = [c for c in range(n) if c not in pivot_cols]
    num_free = len(free_cols)
    num_pivots = len(pivot_cols)

    int_coeff = np.zeros((num_pivots, num_free), dtype=np.int64)
    int_const = np.zeros(num_pivots, dtype=np.int64)
    int_denom = np.zeros(num_pivots, dtype=np.int64)

    for i in range(num_pivots):
        denoms = [aug[i][n].denominator]
        for fi, fc in enumerate(free_cols):
            denoms.append(aug[i][fc].denominator)
        row_lcd = 1
        for d in denoms:
            row_lcd = math_lcm(row_lcd, d)
        int_denom[i] = row_lcd
        int_const[i] = int(aug[i][n] * row_lcd)
        for fi, fc in enumerate(free_cols):
            int_coeff[i, fi] = int(aug[i][fc] * row_lcd)

    solutions = []

    if num_free == 0:
        x = np.zeros(n, dtype=np.int64)
        valid = True
        for i in range(num_pivots):
            if abs(int_const[i]) != int_denom[i]:
                valid = False
                break
            x[pivot_cols[i]] = int_const[i] // int_denom[i]
        if valid:
            solutions.append(x)
        return solutions

    if num_free <= 20:
        total = 1 << num_free
        bits = np.arange(total, dtype=np.int32)
        free_matrix = np.empty((total, num_free), dtype=np.int64)
        for fi in range(num_free):
            free_matrix[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pivot_vals = int_const[np.newaxis, :] - free_matrix @ int_coeff.T
        valid_mask = np.ones(total, dtype=bool)
        for i in range(num_pivots):
            d = int_denom[i]
            valid_mask &= (np.abs(pivot_vals[:, i]) == d)
        sol_indices = np.where(valid_mask)[0]
        rng.shuffle(sol_indices)
        for idx in sol_indices[:max_sol]:
            if deadline and time.time() > deadline:
                break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols):
                x[fc] = free_matrix[idx, fi]
            for i in range(num_pivots):
                x[pivot_cols[i]] = pivot_vals[idx, i] // int_denom[i]
            solutions.append(x)
    else:
        batch_size = min(1 << 20, max_sol * 200)
        n_batches = max(1, (max_sol * 500) // batch_size)
        for _ in range(n_batches):
            if deadline and time.time() > deadline:
                break
            if len(solutions) >= max_sol:
                break
            free_matrix = rng.choice([-1, 1], size=(batch_size, num_free)).astype(np.int64)
            pivot_vals = int_const[np.newaxis, :] - free_matrix @ int_coeff.T
            valid_mask = np.ones(batch_size, dtype=bool)
            for i in range(num_pivots):
                d = int_denom[i]
                valid_mask &= (np.abs(pivot_vals[:, i]) == d)
            sol_indices = np.where(valid_mask)[0]
            for idx in sol_indices:
                if len(solutions) >= max_sol:
                    break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols):
                    x[fc] = free_matrix[idx, fi]
                for i in range(num_pivots):
                    x[pivot_cols[i]] = pivot_vals[idx, i] // int_denom[i]
                solutions.append(x)

    return solutions


def main():
    G = build_new_gram()

    print("Decomposing new Gram from Solomon seed")
    print("Fix columns 6-28 from seed, backtrack on 0-5")
    print()

    rng = np.random.RandomState(42)
    t_start = time.time()

    # Fix columns 6-28 from seed (23 columns)
    fixed_cols = list(range(6, 29))
    free_cols_list = [0, 1, 2, 3, 4, 5]

    # Try many random orderings of the free columns
    for attempt in range(10000):
        if time.time() - t_start > 250:
            break

        placed_cols = [SEED[:, i].copy() for i in fixed_cols]
        placed_indices = list(fixed_cols)

        rng.shuffle(free_cols_list)
        per_attempt_deadline = min(t_start + 250, time.time() + 5.0)

        success = True
        for step, target_idx in enumerate(free_cols_list):
            solutions = solve_column_constraints(
                placed_cols, placed_indices, target_idx, G, rng,
                max_sol=1000, deadline=per_attempt_deadline
            )

            if not solutions:
                success = False
                break

            # Try each solution
            found = False
            for sol in solutions:
                if time.time() > per_attempt_deadline:
                    success = False
                    found = False
                    break
                placed_cols.append(sol)
                placed_indices.append(target_idx)

                # Check if remaining columns can be found
                remaining_ok = True
                for next_step in range(step + 1, len(free_cols_list)):
                    next_target = free_cols_list[next_step]
                    next_sols = solve_column_constraints(
                        placed_cols, placed_indices, next_target, G, rng,
                        max_sol=1, deadline=per_attempt_deadline
                    )
                    if not next_sols:
                        remaining_ok = False
                        break
                    placed_cols.append(next_sols[0])
                    placed_indices.append(next_target)

                if remaining_ok and len(placed_cols) == N:
                    # Build and verify
                    R = np.zeros((N, N), dtype=np.int64)
                    for ci in range(N):
                        R[:, placed_indices[ci]] = placed_cols[ci]
                    if verify_decomposition(R, G):
                        elapsed = time.time() - t_start
                        print(f"SUCCESS! Found decomposition in {elapsed:.1f}s, attempt {attempt+1}")
                        save_result(R, G)
                        return R

                # Undo the remaining columns we tried
                while len(placed_cols) > len(fixed_cols) + step + 1:
                    placed_cols.pop()
                    placed_indices.pop()

                if remaining_ok:
                    # This solution worked but verification failed (shouldn't happen)
                    pass
                else:
                    # This solution didn't lead to a full solution, try next
                    placed_cols.pop()
                    placed_indices.pop()

            if not found and not success:
                break

        if attempt % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Attempt {attempt+1}: {elapsed:.1f}s")
            sys.stdout.flush()

    # Also try fixing fewer columns (e.g., just 7-28)
    print("\nTrying with fewer fixed columns...")
    for num_fixed_from_seed in [20, 18, 15, 12]:
        if time.time() - t_start > 250:
            break

        print(f"\n--- Fix {num_fixed_from_seed} seed columns ---")
        sys.stdout.flush()

        for attempt2 in range(500):
            if time.time() - t_start > 250:
                break

            # Always fix cols 6-10 (the -3 structure) and some others from seed
            must_fix = [6, 7, 8, 9, 10]
            remaining_cols = [c for c in range(29) if c not in must_fix]
            rng.shuffle(remaining_cols)
            extra_fix = remaining_cols[:num_fixed_from_seed - len(must_fix)]
            fix_set = sorted(must_fix + extra_fix)
            free_set = sorted([c for c in range(29) if c not in fix_set])

            placed_cols = [SEED[:, i].copy() for i in fix_set]
            placed_indices = list(fix_set)

            per_attempt_deadline = min(t_start + 250, time.time() + 2.0)
            rng.shuffle(free_set)

            all_found = True
            for target_idx in free_set:
                solutions = solve_column_constraints(
                    placed_cols, placed_indices, target_idx, G, rng,
                    max_sol=500, deadline=per_attempt_deadline
                )
                if not solutions:
                    all_found = False
                    break
                placed_cols.append(solutions[0])
                placed_indices.append(target_idx)

            if all_found and len(placed_cols) == N:
                R = np.zeros((N, N), dtype=np.int64)
                for ci in range(N):
                    R[:, placed_indices[ci]] = placed_cols[ci]
                if verify_decomposition(R, G):
                    elapsed = time.time() - t_start
                    print(f"SUCCESS! fix={num_fixed_from_seed}, attempt {attempt2+1}, {elapsed:.1f}s")
                    save_result(R, G)
                    return R

            if attempt2 % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  fix={num_fixed_from_seed}, attempt {attempt2+1}: {elapsed:.1f}s")
                sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f"\nNo decomposition found in {elapsed:.1f}s")
    return None


def save_result(R, G):
    import math

    def det_bareiss(A):
        n = len(A); M = [list(row) for row in A]
        sign = 1
        for k in range(n - 1):
            if M[k][k] == 0:
                for i in range(k + 1, n):
                    if M[i][k] != 0: M[k], M[i] = M[i], M[k]; sign *= -1; break
                else: return 0
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                    den = M[k - 1][k - 1] if k > 0 else 1
                    M[i][j] = num // den
        return sign * M[-1][-1]

    det_H = abs(det_bareiss(R.astype(int).tolist()))
    target = 1270698346568170340352
    threshold = int(0.94 * target)

    print(f"|det(H)| = {det_H}")
    print(f"Threshold = {threshold}")
    print(f"Above threshold: {det_H > threshold}")
    print(f"Ratio to target: {det_H / target:.6f}")

    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/best_H_new_gram.npy", R)

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"
    with open(results_file, "a") as f:
        f.write(f"\n\nDECOMPOSITION FOUND (from seed)!\n")
        f.write(f"|det(H)| = {det_H}\n")
        f.write(f"Ratio to target: {det_H / target:.6f}\n")
        f.write(f"H matrix:\n")
        for row in R.astype(int):
            f.write("  [" + ",".join(str(x) for x in row) + "]\n")

    print("\nH matrix (first 3 rows):")
    for i in range(3):
        print("  [" + " ".join(f"{R[i,j]:+d}" for j in range(N)) + "]")


if __name__ == "__main__":
    main()
