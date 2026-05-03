"""
Hill climbing from the known Orrick-Solomon seed matrix.
Uses single-flip greedy ascent, then multi-flip perturbations.

Target: |det(H)| > 0.94 * 1270698346568170340352 = 1194456445774079983616
Solomon |det(H)| = 1188957517256767569920 (99.5% of threshold)

Key insight: The Solomon matrix is a LOCAL MAXIMUM for single-entry flips.
So we need multi-flip perturbations (simulated annealing, tabu search, etc.)
"""
import numpy as np
import math
import time
import sys

def det_bareiss(A):
    n = len(A)
    M = [list(row) for row in A]
    sign = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            found = False
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    sign *= -1
                    found = True
                    break
            if not found:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return sign * M[-1][-1]

# Known Orrick-Solomon seed matrix
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
], dtype=np.float64)


def greedy_climb(H):
    """Single-flip greedy ascent. Returns (H, |det|, iterations)."""
    n = H.shape[0]
    iteration = 0
    det_val = np.linalg.det(H)

    while True:
        iteration += 1
        Hinv = np.linalg.inv(H)
        # Ratio for flipping H[i,j]: 1 - 2*H[i,j]*Hinv[j,i]
        ratios = 1 - 2 * H * Hinv.T
        abs_ratios = np.abs(ratios)
        best_idx = np.unravel_index(np.argmax(abs_ratios), (n, n))
        best_ratio = abs_ratios[best_idx]

        if best_ratio <= 1.0 + 1e-10:
            break

        i, j = best_idx
        H[i, j] *= -1
        det_val *= ratios[i, j]

    return H, abs(det_val), iteration


def simulated_annealing(H_start, temp_init=0.5, temp_min=0.001, cooling=0.9999,
                        max_iters=500000, time_limit=60):
    """Simulated annealing on {-1,+1} matrix to maximize |det|."""
    n = H_start.shape[0]
    H = H_start.copy()
    det_val = np.linalg.det(H)
    abs_det = abs(det_val)
    best_abs_det = abs_det
    best_H = H.copy()

    temp = temp_init
    start_time = time.time()
    accepted = 0
    rejected = 0

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        temp *= cooling

        if temp < temp_min:
            temp = temp_min

        # Pick random entry to flip
        i = np.random.randint(n)
        j = np.random.randint(n)

        # Compute ratio
        Hinv = np.linalg.inv(H)
        ratio = 1 - 2 * H[i, j] * Hinv[j, i]
        new_abs_det = abs_det * abs(ratio)

        # Accept?
        if new_abs_det > abs_det:
            accept = True
        else:
            # SA acceptance
            if abs_det > 0 and new_abs_det > 0:
                delta = np.log(new_abs_det) - np.log(abs_det)
                accept = np.random.random() < np.exp(delta / temp)
            else:
                accept = False

        if accept:
            H[i, j] *= -1
            det_val *= ratio
            abs_det = abs(det_val)
            accepted += 1

            if abs_det > best_abs_det:
                best_abs_det = abs_det
                best_H = H.copy()
        else:
            rejected += 1

        if it % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"  SA iter {it}: |det|={abs_det:.6e}, best={best_abs_det:.6e}, "
                  f"temp={temp:.6f}, accept_rate={accepted/(accepted+rejected+1):.3f}, "
                  f"{elapsed:.0f}s")
            sys.stdout.flush()

    return best_H, best_abs_det


def multi_flip_search(H_start, time_limit=120):
    """
    Multi-flip perturbation + greedy climb.
    Strategy: perturb k entries, then climb back to local max.
    """
    n = H_start.shape[0]
    H_best = H_start.copy()
    det_best = abs(np.linalg.det(H_best))

    start_time = time.time()
    trial = 0

    for k in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
        while time.time() - start_time < time_limit:
            trial += 1
            H = H_best.copy()

            # Flip k random entries
            for _ in range(k):
                i = np.random.randint(n)
                j = np.random.randint(n)
                H[i, j] *= -1

            H, det_val, iters = greedy_climb(H)

            if det_val > det_best:
                det_best = det_val
                H_best = H.copy()
                elapsed = time.time() - start_time
                print(f"  Multi-flip k={k}, trial {trial}: NEW BEST |det|={det_val:.6e}, "
                      f"iters={iters}, {elapsed:.0f}s")
                sys.stdout.flush()

            if trial % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  k={k}, trial {trial}: best={det_best:.6e}, {elapsed:.0f}s")
                sys.stdout.flush()

            # Move to next k after some trials
            if trial % 1000 == 0:
                break

    return H_best, det_best


def row_swap_search(H_start, time_limit=120):
    """
    Try replacing entire rows with random {-1,+1} rows, then greedy climb.
    """
    n = H_start.shape[0]
    H_best = H_start.copy()
    det_best = abs(np.linalg.det(H_best))

    start_time = time.time()
    trial = 0

    while time.time() - start_time < time_limit:
        trial += 1
        H = H_best.copy()

        # Replace 1-3 random rows
        num_rows = np.random.randint(1, 4)
        rows = np.random.choice(n, size=num_rows, replace=False)
        for r in rows:
            H[r] = np.random.choice([-1, 1], size=n).astype(np.float64)

        H, det_val, iters = greedy_climb(H)

        if det_val > det_best:
            det_best = det_val
            H_best = H.copy()
            elapsed = time.time() - start_time
            print(f"  Row swap trial {trial}: NEW BEST |det|={det_val:.6e}, "
                  f"replaced rows {rows}, iters={iters}, {elapsed:.0f}s")
            sys.stdout.flush()

        if trial % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Row swap trial {trial}: best={det_best:.6e}, {elapsed:.0f}s")
            sys.stdout.flush()

    return H_best, det_best


def main():
    n = 29
    target_det = 1270698346568170340352
    threshold = 0.94 * target_det
    solomon_det = 1188957517256767569920

    print(f"Target |det(H)| > {threshold:.0f}")
    print(f"Solomon |det(H)| = {solomon_det}")
    print(f"Gap: {(threshold - solomon_det) / solomon_det * 100:.2f}%")
    print()

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"

    # Start from seed
    H = SEED.copy()
    seed_det = abs(np.linalg.det(H))
    print(f"Seed |det| ~ {seed_det:.6e}")

    # Verify seed is local maximum
    H_climbed, det_climbed, iters = greedy_climb(H.copy())
    print(f"After greedy climb: |det| ~ {det_climbed:.6e}, iters={iters}")
    if iters == 1:
        print("Seed is already a local maximum for single flips!")
    print()

    # Strategy 1: Multi-flip perturbation
    print("=" * 60)
    print("Strategy 1: Multi-flip perturbation + greedy climb")
    print("=" * 60)
    H1, det1 = multi_flip_search(H_climbed.copy(), time_limit=120)
    print(f"Multi-flip best: |det| ~ {det1:.6e}")
    print()

    # Strategy 2: Simulated Annealing
    print("=" * 60)
    print("Strategy 2: Simulated Annealing")
    print("=" * 60)
    H2, det2 = simulated_annealing(H_climbed.copy(), temp_init=1.0,
                                    cooling=0.99995, max_iters=2000000,
                                    time_limit=120)
    print(f"SA best: |det| ~ {det2:.6e}")
    print()

    # Strategy 3: Row replacement
    print("=" * 60)
    print("Strategy 3: Row replacement + greedy climb")
    print("=" * 60)
    H3, det3 = row_swap_search(H_climbed.copy(), time_limit=120)
    print(f"Row swap best: |det| ~ {det3:.6e}")
    print()

    # Best overall
    candidates = [(H1, det1), (H2, det2), (H3, det3)]
    best_H, best_det = max(candidates, key=lambda x: x[1])

    print("=" * 60)
    print(f"BEST OVERALL: |det| ~ {best_det:.6e}")
    print(f"Threshold: {threshold:.6e}")
    print(f"Ratio to threshold: {best_det / threshold:.6f}")
    print()

    # Exact verification
    H_int = np.round(best_H).astype(int)
    exact_det = abs(det_bareiss(H_int.tolist()))
    print(f"Exact |det| = {exact_det}")
    print(f"Exact ratio to threshold: {exact_det / threshold:.6f}")

    if exact_det > threshold:
        print("*** SOLUTION FOUND! ***")
        # Verify all entries are {-1,+1}
        assert np.all(np.isin(H_int, [-1, 1])), "Not all entries are +/-1!"
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/best_H.npy", H_int)
        with open(results_file, "a") as f:
            f.write(f"\n\nSOLUTION FOUND!\n")
            f.write(f"|det(H)| = {exact_det}\n")
            f.write(f"Ratio to target: {exact_det / target_det:.6f}\n")
            f.write(f"H = {H_int.tolist()}\n")
    else:
        print("Solution not found. Best result saved.")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/best_H.npy", H_int)
        with open(results_file, "a") as f:
            f.write(f"\n\nSeed climb best result\n")
            f.write(f"|det(H)| = {exact_det}\n")
            f.write(f"Ratio to threshold: {exact_det / threshold:.6f}\n")

if __name__ == "__main__":
    main()
