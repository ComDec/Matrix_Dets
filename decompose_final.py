"""
Decompose new Gram by fixing cols 6-28 from Solomon seed,
then doing FULL backtracking on cols 0-5.

The new Gram is identical to Solomon except:
  G[0,3] = 9 (was 5)
  G[1,4] = 9 (was 5)
  All other five-entries -> 1
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

    if num_free <= 22:
        total = 1 << num_free
        if total <= 2**22:
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
            return solutions

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
    rng = np.random.RandomState(42)
    t_start = time.time()

    print("Decomposing new Gram: full backtracking on cols 0-5")
    print("(cols 6-28 fixed from Solomon seed)")
    print()

    # Fix columns 6-28
    fixed_cols = list(range(6, 29))
    free_col_indices = [0, 1, 2, 3, 4, 5]

    # Try different orderings of the free columns
    orderings = [
        [0, 3, 1, 4, 2, 5],  # pair up the 9-entries first
        [3, 0, 4, 1, 5, 2],  # reverse pairing
        [0, 1, 2, 3, 4, 5],  # natural order
        [5, 4, 3, 2, 1, 0],  # reverse
        [2, 5, 0, 3, 1, 4],  # mixed
        [3, 4, 5, 0, 1, 2],  # second group first
    ]

    deadline = t_start + 300

    for ord_idx, col_order in enumerate(orderings):
        if time.time() > deadline:
            break

        print(f"Ordering {ord_idx}: {col_order}")
        sys.stdout.flush()

        placed_cols = [SEED[:, i].copy() for i in fixed_cols]
        placed_indices = list(fixed_cols)

        per_order_deadline = min(deadline, time.time() + 50)

        def backtrack(step):
            if time.time() > per_order_deadline:
                return False
            if step == 6:
                return True

            target_idx = col_order[step]
            ms = 10000 if step <= 2 else 5000

            solutions = solve_column_constraints(
                placed_cols, placed_indices, target_idx, G, rng,
                max_sol=ms, deadline=per_order_deadline
            )

            if not solutions:
                return False

            print(f"  Step {step}: col {target_idx}, {len(solutions)} candidates")
            sys.stdout.flush()

            for sol in solutions:
                if time.time() > per_order_deadline:
                    return False
                placed_cols.append(sol)
                placed_indices.append(target_idx)
                if backtrack(step + 1):
                    return True
                placed_cols.pop()
                placed_indices.pop()

            return False

        if backtrack(0):
            # Build result
            R = np.zeros((N, N), dtype=np.int64)
            for ci in range(N):
                R[:, placed_indices[ci]] = placed_cols[ci]

            if verify_decomposition(R, G):
                elapsed = time.time() - t_start
                print(f"\nSUCCESS! Decomposition found in {elapsed:.1f}s")

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
                    f.write(f"\n\nDECOMPOSITION FOUND!\n")
                    f.write(f"|det(H)| = {det_H}\n")
                    f.write(f"Ratio to target: {det_H / target:.6f}\n")
                    f.write(f"Gram: G[0,3]=G[3,0]=9, G[1,4]=G[4,1]=9, G[6,7..10]=-3, rest=1\n")
                    f.write(f"H matrix:\n")
                    for row in R.astype(int):
                        f.write("  [" + ",".join(str(x) for x in row) + "]\n")

                print("\nH matrix:")
                for i in range(N):
                    print("  [" + " ".join(f"{R[i,j]:+d}" for j in range(N)) + "]")

                return R
            else:
                print("Verification failed!")

        elapsed = time.time() - t_start
        print(f"  Ordering {ord_idx} done, {elapsed:.1f}s total\n")

    elapsed = time.time() - t_start
    print(f"\nNo decomposition found in {elapsed:.1f}s")
    return None


if __name__ == "__main__":
    main()
