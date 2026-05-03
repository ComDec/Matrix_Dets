"""
Direct search for 29x29 {-1,+1} matrices with high determinant.
Uses steepest-ascent hill climbing with random restarts.

Target: |det(H)| > 0.94 * 1270698346568170340352 = 1194456445774079983616

Strategy:
1. Start from Solomon matrix (decomposed from Gram if possible) or random {-1,+1} matrix
2. Greedy flip: try all n^2 single-entry flips, pick the one that increases |det| the most
3. Repeat until no flip improves
4. Random restart and repeat

Uses rank-1 update formula for fast det computation:
  If H' = H + 2*delta*e_i*e_j^T (flipping H[i,j] from +1 to -1 or vice versa)
  then det(H') = det(H) * (1 + 2*delta * (H^{-1})[j,i])
  where delta = -H[i,j] (the flip changes H[i,j] to -H[i,j], so addition is -2*H[i,j]*e_i*e_j^T)

  Actually: H' = H - 2*H[i,j]*e_i*e_j^T
  So det(H') = det(H) * (1 - 2*H[i,j] * (H^{-1})[j,i])

  Ratio = 1 - 2*H[i,j] * Hinv[j,i]
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

def build_solomon_H():
    """Build a {-1,+1} matrix whose Gram matches the Solomon Gram.

    The Solomon Gram is:
    G = ones(29,29) with diag=29, G[i,j]=5 for i in {0,1,2}, j in {3,4,5},
    G[6,j]=-3 for j in {7,8,9,10}, rest off-diag = 1.

    Actually, we don't know the exact {-1,+1} matrix. Let's try to construct one.
    G = H @ H^T. Since G has specific inner product structure, H has specific row relationships.

    For now, let's use a random {-1,+1} matrix as starting point instead.
    """
    return None

def greedy_climb(H, verbose=True):
    """Greedy hill climbing: flip the entry that increases |det| the most."""
    n = H.shape[0]
    H = H.astype(np.float64)

    iteration = 0
    det_val = np.linalg.det(H)
    abs_det = abs(det_val)

    while True:
        iteration += 1
        Hinv = np.linalg.inv(H)

        # Compute ratio for each possible flip
        # ratio[i,j] = 1 - 2*H[i,j]*Hinv[j,i]
        # abs_det_new = abs(det_val * ratio[i,j])
        ratios = 1 - 2 * H * Hinv.T  # element-wise
        abs_ratios = np.abs(ratios)

        best_idx = np.unravel_index(np.argmax(abs_ratios), (n, n))
        best_ratio = abs_ratios[best_idx]

        if best_ratio <= 1.0 + 1e-10:
            break  # No improving flip

        i, j = best_idx
        H[i, j] *= -1
        det_val *= ratios[i, j]
        abs_det = abs(det_val)

        if verbose and iteration % 50 == 0:
            print(f"  iter {iteration}: |det| ~ {abs_det:.6e}, flipped ({i},{j})")

    return H, abs_det, iteration

def main():
    n = 29
    target_det = 1270698346568170340352
    threshold = 0.94 * target_det
    solomon_det_H = 1188957517256767569920

    print(f"Target |det(H)| > {threshold:.0f}")
    print(f"Solomon |det(H)| = {solomon_det_H}")
    print()

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"

    best_det = 0
    best_H = None
    num_trials = 0
    start_time = time.time()

    np.random.seed(42)

    while True:
        num_trials += 1

        # Random starting matrix
        H = np.random.choice([-1, 1], size=(n, n)).astype(np.float64)

        H_opt, abs_det, iters = greedy_climb(H, verbose=False)

        if abs_det > best_det:
            best_det = abs_det
            best_H = H_opt.copy()

            # Verify with exact computation
            H_int = np.round(H_opt).astype(int)
            exact_det = abs(det_bareiss(H_int.tolist()))

            elapsed = time.time() - start_time
            print(f"Trial {num_trials}: NEW BEST |det| ~ {abs_det:.6e}, exact={exact_det}, "
                  f"iters={iters}, time={elapsed:.0f}s")

            if exact_det > threshold:
                print(f"*** FOUND SOLUTION! |det| = {exact_det} > {threshold:.0f} ***")

                # Save
                with open(results_file, "a") as f:
                    f.write(f"\n\nDirect search solution found!\n")
                    f.write(f"|det(H)| = {exact_det}\n")
                    f.write(f"Ratio to target: {exact_det/target_det:.6f}\n")
                    f.write(f"Trial: {num_trials}, Time: {elapsed:.1f}s\n")
                    f.write(f"H = {H_int.tolist()}\n\n")

                # Also save matrix to separate file
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/best_H.npy", H_int)
                print(f"Matrix saved to best_H.npy")
                return H_int, exact_det

        if num_trials % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Trial {num_trials}: best_det ~ {best_det:.6e}, "
                  f"rate={num_trials/elapsed:.1f} trials/s, elapsed={elapsed:.0f}s")
            sys.stdout.flush()

        # Check time limit (10 minutes)
        if time.time() - start_time > 540:
            print(f"Time limit approaching. Best: {best_det:.6e}")
            break

    if best_H is not None:
        H_int = np.round(best_H).astype(int)
        exact_det = abs(det_bareiss(H_int.tolist()))
        print(f"\nFinal best: |det| = {exact_det}")
        print(f"Ratio to threshold: {exact_det / threshold:.6f}")

        with open(results_file, "a") as f:
            f.write(f"\n\nDirect search best result\n")
            f.write(f"|det(H)| = {exact_det}\n")
            f.write(f"Ratio to threshold: {exact_det / threshold:.6f}\n")
            f.write(f"Trials: {num_trials}\n")

        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/best_H.npy", H_int)

if __name__ == "__main__":
    main()
