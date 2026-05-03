"""Search for alternative Gram matrix structures for 29x29 maxdet.

The Solomon Gram G has diagonal=29, off-diagonal entries in {-3, 1, 5}.
det(G) = (29^29/2 * c/THEORETICAL_MAX)^2 where c=320.

Other Gram structures might yield higher c values. For a {-1,+1} matrix R,
G = R^T R has:
- Diagonal entries = 29 (sum of squares)
- Off-diagonal entries = sum of products, must be odd (sum of 29 odd products)
- Off-diagonal entries in {-29, -27, ..., 27, 29} (congruent to 29 mod 2, so odd)
- But more constrained: column sums constrain inner products

Key insight: det(G) = det(R)^2, so we want to maximize det(G).
The Hadamard bound: det(G) <= 29^29.
For G with all off-diag = 1: det(G) = (29-1)^(29-1) * (29 + 28) = 28^28 * 57
  = 28^28 * 57 which is much less than 29^29.

The question: what Gram structure gives the highest det(G)?

Method 1: Enumerate possible Gram structures (det(G) values for different patterns)
Method 2: Direct search over the Gram space
Method 3: Try to find decompositions R for higher-det Gram matrices
"""

import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm
from itertools import product

N = 29
THEORETICAL_MAX = 1270698346568170340352


def det_bareiss(A):
    n = len(A)
    if n == 0: return 1
    M = [row.copy() for row in A]
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0: M[k], M[i] = M[i], M[k]; break
            else: return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return M[-1][-1]


def hill_climb_sm(A, deadline):
    n = A.shape[0]; A = A.astype(float).copy()
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10: return A, ld
    Ai = np.linalg.inv(A); steps = 0
    while time.time() < deadline:
        r = 1.0 - 2.0*A*Ai.T; ar = np.abs(r)
        ix = np.unravel_index(np.argmax(ar), (n, n))
        if ar[ix] <= 1.0+1e-12: break
        i, j = ix; d = -2.0*A[i, j]; dn = 1.0+d*Ai[j, i]
        if abs(dn) < 1e-15: break
        Ai -= (d/dn)*np.outer(Ai[:, j], Ai[i, :]); A[i, j] *= -1.0
        steps += 1
        if steps % 30 == 0:
            s2, _ = np.linalg.slogdet(A)
            if s2 == 0: break
            Ai = np.linalg.inv(A)
    _, ld = np.linalg.slogdet(A)
    return A, ld


def row_replace_climb(A, deadline):
    n = A.shape[0]; A = A.astype(float).copy(); s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10: return A, ld
    Ai = np.linalg.inv(A)
    while time.time() < deadline:
        imp = False
        for k in range(n):
            rn = (np.sign(Ai[:, k])*s).astype(float); rn[rn == 0] = 1.0
            d = rn - A[k, :]
            if np.all(d == 0): continue
            dn = 1.0 + d @ Ai[:, k]
            if abs(dn) <= 1.0+1e-10: continue
            Ai -= np.outer(Ai[:, k], d @ Ai)/dn
            A[k, :] = rn; s2, ld = np.linalg.slogdet(A); s = s2; imp = True
        Ai = np.linalg.inv(A); s, ld = np.linalg.slogdet(A)
        if s == 0 or not imp: break
    return A, ld


def main():
    print("=" * 70)
    print("ALTERNATIVE GRAM STRUCTURE SEARCH for 29x29 maxdet")
    print("=" * 70)
    sys.stdout.flush()

    # First, let's understand the OS Gram
    G_os = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G_os, N)
    for i in range(3):
        for j in range(3, 6): G_os[i, j] = 5; G_os[j, i] = 5
    for j in range(7, 11): G_os[6, j] = -3; G_os[j, 6] = -3

    det_os = det_bareiss(G_os.tolist())
    print(f"OS Gram det = {det_os}")
    print(f"OS |det(R)| = sqrt(det(G)) = {np.sqrt(float(det_os)):.4e}")
    print(f"OS score = {np.sqrt(float(det_os))/THEORETICAL_MAX:.8f}")
    sys.stdout.flush()

    # Method 1: Search for higher-det Gram matrices with specific structures
    # A valid Gram matrix G for a {-1,+1}^{29x29} matrix must:
    # - Be symmetric positive semi-definite
    # - Have diagonal = 29
    # - Off-diagonal entries are odd integers in [-29, 29]
    # - G = R^T R for some R in {-1,+1}^{29x29} (realizability)

    # The Ehlich-Wojtas theory tells us which Gram structures are optimal.
    # For n=29 (n = 1 mod 4), the EW conjecture says the max det Gram has
    # a specific block structure.

    # Let's compute det(G) for various candidate Gram structures
    print(f"\n{'='*70}")
    print("Testing various Gram structures")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Structure 1: All off-diag = 1 (identity-like)
    G1 = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G1, N)
    det_g1 = det_bareiss(G1.tolist())
    print(f"\nG1 (all off-diag=1): det = {det_g1}")
    print(f"  score = {np.sqrt(abs(float(det_g1)))/THEORETICAL_MAX:.8f}")

    # Structure 2: OS Gram with -3 entries varied
    # The OS Gram has exactly 8 entries of -3 (4 pairs) and 12 entries of 5 (6 pairs)
    # What if we try different numbers?

    # Structure 3: Block diagonal-like Grams
    # Try: two blocks where off-diagonal between blocks = -1 or 3
    # and within each block, different patterns

    best_gram_det = abs(det_os)
    best_gram = G_os.copy()

    # Enumerate: for each pair of columns, the inner product must be 1 mod 2
    # (since sum of 29 products of +-1 is always odd)
    # Common values: -3, -1, 1, 3, 5

    # Strategy: Random Gram matrix search
    # Generate random symmetric matrices with diagonal=29, off-diagonal odd,
    # check if PSD, compute det
    rng = np.random.RandomState(42)
    n_tested = 0
    t0 = time.time()

    print(f"\nRandom Gram structure search...")
    sys.stdout.flush()

    for trial in range(100000):
        if time.time() - t0 > 300:  # 5 min budget
            break

        # Generate a random Gram-like matrix
        # Off-diagonal: randomly choose from {-3, -1, 1, 3, 5}
        # But weight toward values that give high det
        G = np.full((N, N), 29, dtype=np.int64)

        # Strategy: perturb OS Gram slightly
        G[:] = G_os
        n_perturb = rng.randint(2, 20)
        for _ in range(n_perturb):
            i = rng.randint(N)
            j = rng.randint(N)
            if i == j:
                continue
            new_val = rng.choice([-5, -3, -1, 1, 3, 5])
            G[i, j] = new_val
            G[j, i] = new_val

        # Check PSD (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(G.astype(float))
        if np.min(eigvals) < -0.01:
            continue

        det_g = det_bareiss(G.tolist())
        n_tested += 1

        if abs(det_g) > best_gram_det:
            best_gram_det = abs(det_g)
            best_gram = G.copy()
            score = np.sqrt(float(best_gram_det)) / THEORETICAL_MAX
            print(f"  Trial {trial}: NEW BEST Gram det = {det_g}, score = {score:.8f}")
            print(f"  Off-diag values: {sorted(set(G[np.triu_indices(N, 1)]))}")
            sys.stdout.flush()

        if trial % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {trial} trials ({n_tested} valid), best det = {best_gram_det}, "
                  f"{elapsed:.1f}s")
            sys.stdout.flush()

    print(f"\nGram search: {n_tested} valid out of {trial+1} trials")
    print(f"Best Gram det = {best_gram_det}")
    print(f"Best score bound = {np.sqrt(float(best_gram_det))/THEORETICAL_MAX:.8f}")

    # Method 2: Direct large-scale random matrix search with diverse starting points
    print(f"\n{'='*70}")
    print("Direct matrix search with diverse seeds")
    print(f"{'='*70}")
    sys.stdout.flush()

    best_score = 0.0
    best_matrix = None

    # Strategy: generate many random matrices, optimize each, track best
    deadline = time.time() + 3600  # 1 hour

    # Diverse starting strategies
    strategies = [
        "random",
        "random_balanced",  # rows with equal +1 and -1
        "circulant",
        "block_random",
        "perturbed_identity",
    ]

    count = 0
    while time.time() < deadline:
        count += 1
        strategy = strategies[count % len(strategies)]
        seed = count * 17 + 31
        rng_local = np.random.RandomState(seed)

        if strategy == "random":
            A = rng_local.choice([-1.0, 1.0], size=(N, N))

        elif strategy == "random_balanced":
            # Each row has ~14 or 15 ones
            A = np.ones((N, N))
            for i in range(N):
                n_neg = rng_local.choice([13, 14, 15])
                cols = rng_local.choice(N, size=n_neg, replace=False)
                A[i, cols] = -1.0

        elif strategy == "circulant":
            # Random first row, circulant structure
            row = rng_local.choice([-1.0, 1.0], size=N)
            A = np.zeros((N, N))
            for i in range(N):
                A[i] = np.roll(row, i)

        elif strategy == "block_random":
            # Two random blocks
            A = np.ones((N, N))
            k = rng_local.randint(8, 22)
            A[:k, :k] = rng_local.choice([-1.0, 1.0], size=(k, k))
            A[k:, k:] = rng_local.choice([-1.0, 1.0], size=(N-k, N-k))
            # Cross blocks
            A[:k, k:] = rng_local.choice([-1.0, 1.0], size=(k, N-k))
            A[k:, :k] = rng_local.choice([-1.0, 1.0], size=(N-k, k))

        elif strategy == "perturbed_identity":
            # Start near identity (high det but not +-1)
            A = rng_local.choice([-1.0, 1.0], size=(N, N))
            # Make diagonal dominant
            for i in range(N):
                A[i, i] = 1.0
                # Ensure most off-diag are -1 in same row
                n_pos = rng_local.randint(5, 15)
                cols = rng_local.choice([c for c in range(N) if c != i], size=n_pos, replace=False)
                A[i, :] = -1.0
                A[i, i] = 1.0
                A[i, cols] = 1.0

        # Optimize
        A, _ = row_replace_climb(A, time.time() + 0.15)
        A, _ = hill_climb_sm(A, time.time() + 0.1)

        s, ld = np.linalg.slogdet(A)
        if s != 0:
            score = np.exp(ld) / THEORETICAL_MAX
            if score > best_score + 1e-6:
                best_score = score
                best_matrix = np.sign(A).astype(int)
                best_matrix[best_matrix == 0] = 1

                if score > 0.93:
                    # Exact check
                    exact_det = abs(det_bareiss(best_matrix.tolist()))
                    exact_score = exact_det / THEORETICAL_MAX
                    print(f"  [{count}] strategy={strategy}: exact score = {exact_score:.8f}")

                    if exact_score > 0.9357 + 1e-6:
                        print(f"\n!!! BREAKTHROUGH: score = {exact_score:.8f} !!!")
                        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy",
                                best_matrix)
                        print("Saved!")
                    sys.stdout.flush()

        if count % 5000 == 0:
            elapsed = time.time() - (deadline - 3600)
            print(f"  {count} matrices tested, best score = {best_score:.8f}, {elapsed:.0f}s")
            sys.stdout.flush()

    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"Matrices tested: {count}")
    print(f"Best score: {best_score:.8f}")
    print(f"{'='*70}")

    if best_matrix is not None:
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/alt_gram_best.npy",
                best_matrix)

    sys.stdout.flush()


if __name__ == "__main__":
    main()
