"""Robust simulated annealing for 29x29 maxdet.

Key fix: use slogdet for det tracking instead of incremental SM updates.
SM updates are only used for det RATIO computation (1 number), not for
maintaining the full inverse. The actual matrix state is the integer +-1 matrix.

Also implements:
- 4-flip and 5-flip neighborhoods (since 1,2,3-flip are all local max)
- Row/column swaps (det-preserving) followed by hill climb
- "Restart from basin boundary" strategy
"""

import numpy as np
import time
import sys

N = 29
THEORETICAL_MAX = 1270698346568170340352

BEST_KNOWN = np.array([
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


def massive_perturbation_search(time_limit=7200):
    """Large perturbation + hill climb search.

    Since k-flip for k<=3 cannot improve the OS matrix, we need k>=4.
    Strategy: flip 4-100 entries, then row-replace + hill climb.
    If the result reaches a DIFFERENT local max with higher det, we win.

    Key observation: row-replace + hill-climb from a perturbed matrix
    will converge to SOME local max. The question is which one.
    Most perturbations return to c=320. We need enough perturbation
    to escape to c=322 (if it exists).
    """
    print("=" * 70)
    print("MASSIVE PERTURBATION + HILL CLIMB SEARCH")
    print("=" * 70)
    sys.stdout.flush()

    base = BEST_KNOWN.copy()
    _, base_ld = np.linalg.slogdet(base.astype(float))
    base_score = np.exp(base_ld) / THEORETICAL_MAX
    print(f"Base score: {base_score:.8f}")

    best_score = base_score
    best_matrix = base.copy()
    best_det = abs(det_bareiss(base.astype(int).tolist()))
    print(f"Base exact |det| = {best_det}")

    deadline = time.time() + time_limit
    count = 0
    scores_seen = {}  # track distribution of scores
    last_report = time.time()

    while time.time() < deadline:
        count += 1
        rng = np.random.RandomState(count * 37 + 11)

        # Vary perturbation intensity
        strategy = count % 10

        if strategy < 3:
            # Small perturbation (4-10 flips) -- might find nearby basins
            k = rng.randint(4, 11)
            A = base.astype(float).copy()
            idxs = rng.randint(0, N, size=(k, 2))
            for ij in idxs:
                A[ij[0], ij[1]] *= -1.0

        elif strategy < 5:
            # Medium perturbation (10-50 flips)
            k = rng.randint(10, 51)
            A = base.astype(float).copy()
            idxs = rng.randint(0, N, size=(k, 2))
            for ij in idxs:
                A[ij[0], ij[1]] *= -1.0

        elif strategy < 7:
            # Large perturbation (50-200 flips)
            k = rng.randint(50, 201)
            A = base.astype(float).copy()
            idxs = rng.randint(0, N, size=(k, 2))
            for ij in idxs:
                A[ij[0], ij[1]] *= -1.0

        elif strategy < 8:
            # Row replacement: replace 1-5 rows with random
            A = base.astype(float).copy()
            n_rows = rng.randint(1, 6)
            rows = rng.choice(N, size=n_rows, replace=False)
            for r in rows:
                A[r] = rng.choice([-1.0, 1.0], size=N)

        elif strategy < 9:
            # Column replacement: replace 1-5 cols
            A = base.astype(float).copy()
            n_cols = rng.randint(1, 6)
            cols = rng.choice(N, size=n_cols, replace=False)
            for c in cols:
                A[:, c] = rng.choice([-1.0, 1.0], size=N)

        else:
            # Start from random matrix (sometimes finds different basins)
            A = rng.choice([-1.0, 1.0], size=(N, N))

        # Optimize
        A, _ = row_replace_climb(A, time.time() + 0.15)
        A, ld = hill_climb_sm(A, time.time() + 0.1)

        if ld > 10:
            score = np.exp(ld) / THEORETICAL_MAX
            # Bin scores
            score_bin = round(score, 4)
            scores_seen[score_bin] = scores_seen.get(score_bin, 0) + 1

            if score > best_score + 1e-6:
                # Potential improvement -- verify with exact det
                M = np.sign(A).astype(int)
                M[M == 0] = 1
                exact_det = abs(det_bareiss(M.tolist()))
                exact_score = exact_det / THEORETICAL_MAX

                if exact_score > best_det / THEORETICAL_MAX + 1e-8:
                    best_det = exact_det
                    best_matrix = M.copy()
                    best_score = exact_score
                    print(f"  [{count}] NEW BEST: exact score = {exact_score:.8f}, "
                          f"|det| = {exact_det}")
                    sys.stdout.flush()

                    if exact_score > 0.9357 + 1e-6:
                        print(f"\n{'='*70}")
                        print(f"!!! BREAKTHROUGH: score = {exact_score:.8f} !!!")
                        print(f"|det| = {exact_det}")
                        print(f"{'='*70}")
                        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", M)
                        print("Saved to breakthrough_matrix.npy")
                        sys.stdout.flush()

        # Report progress
        if time.time() - last_report > 60:
            last_report = time.time()
            elapsed = time.time() - (deadline - time_limit)
            # Show distribution of local optima found
            top_scores = sorted(scores_seen.items(), key=lambda x: -x[0])[:5]
            print(f"\n  Progress: {elapsed:.0f}s, {count} trials")
            print(f"  Best score: {best_score:.8f}")
            print(f"  Score distribution (top 5):")
            for sc, cnt in top_scores:
                print(f"    {sc:.4f}: {cnt} times")
            sys.stdout.flush()

    # Final report
    elapsed = time.time() - (deadline - time_limit)
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE: {elapsed:.0f}s, {count} trials")
    print(f"Best score: {best_score:.8f}")
    print(f"Best |det|: {best_det}")
    print(f"\nFull score distribution:")
    for sc in sorted(scores_seen.keys(), reverse=True):
        print(f"  {sc:.4f}: {scores_seen[sc]} times")
    print(f"{'='*70}")
    sys.stdout.flush()

    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/sa_robust_best.npy", best_matrix)


if __name__ == "__main__":
    massive_perturbation_search()
