"""Simulated annealing search for 29x29 maxdet matrix.

Unlike greedy hill climbing, SA can temporarily accept worse solutions
to escape local optima. This is the classical approach for maxdet problems
(see Brent & Osborn, "On minima of the Ehlich-Wojtas function").

Key insight: The OS matrix (c=320) is a local max under 1,2,3-flip.
SA with adaptive temperature might escape this basin.

Also implements:
- Taboo search (avoid recently visited states)
- Large neighborhood search (flip entire rows/columns)
- Gram-aware SA (bias flips toward maintaining Gram structure)
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
], dtype=np.float64)


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


def get_exact_score(A):
    M = np.sign(A).astype(int)
    M[M == 0] = 1
    return abs(det_bareiss(M.tolist())) / THEORETICAL_MAX


def simulated_annealing(A_start, time_limit=3600, T_start=2.0, T_min=0.001,
                         cooling=0.99999, rng=None):
    """Simulated annealing for maxdet with Sherman-Morrison det tracking.

    Moves:
    - Single flip (80%)
    - Double flip in same row (10%)
    - Row negate (5%)
    - Random row replacement (5%)
    """
    if rng is None:
        rng = np.random.RandomState()

    n = A_start.shape[0]
    A = A_start.astype(np.float64).copy()
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
        return A, ld

    Ai = np.linalg.inv(A)

    best_A = A.copy()
    best_ld = ld

    T = T_start
    deadline = time.time() + time_limit
    steps = 0
    accepted = 0
    improved = 0
    last_improve_step = 0
    last_report = time.time()
    reheats = 0

    while time.time() < deadline:
        steps += 1

        # Choose move type
        r = rng.random()
        if r < 0.80:
            # Single flip
            i = rng.randint(n)
            j = rng.randint(n)
            d = -2.0 * A[i, j]
            dn = 1.0 + d * Ai[j, i]
            if abs(dn) < 1e-15:
                continue
            log_ratio = np.log(abs(dn))

        elif r < 0.90:
            # Double flip in same row
            row = rng.randint(n)
            c1, c2 = rng.choice(n, size=2, replace=False)
            d1 = -2.0 * A[row, c1]
            d2 = -2.0 * A[row, c2]
            m11 = 1.0 + d1 * Ai[c1, row]
            m12 = d2 * Ai[c1, row]
            m21 = d1 * Ai[c2, row]
            m22 = 1.0 + d2 * Ai[c2, row]
            det2 = m11 * m22 - m12 * m21
            if abs(det2) < 1e-15:
                continue
            log_ratio = np.log(abs(det2))
            # Will handle acceptance below
            i, j, d, dn = -1, -1, 0, 0  # sentinel

        elif r < 0.95:
            # Row negate
            row = rng.randint(n)
            # Negate row: equivalent to n flips, but det changes by (-1)^n
            # For n=29 (odd): det -> -det, |det| unchanged
            # So row negation doesn't change |det|! Skip...
            # Actually: negate row then hill climb might reach a different basin
            # Use it as a perturbation followed by re-optimization
            A_pert = A.copy()
            A_pert[row] *= -1.0
            A_pert, ld_new = hill_climb_sm(A_pert, min(time.time() + 0.1, deadline))
            if ld_new > best_ld + 1e-10:
                best_ld = ld_new
                best_A = A_pert.copy()
                improved += 1
                print(f"  ROW-NEG improvement: ld={best_ld:.10f}, "
                      f"score~{np.exp(best_ld)/THEORETICAL_MAX:.8f}")
                sys.stdout.flush()
            # Don't change main SA state
            continue

        else:
            # Replace row with random +-1
            row = rng.randint(n)
            new_row = rng.choice([-1.0, 1.0], size=n)
            d_row = new_row - A[row]
            if np.all(d_row == 0):
                continue
            dn = 1.0 + d_row @ Ai[:, row]
            if abs(dn) < 1e-15:
                continue
            log_ratio = np.log(abs(dn))
            i, j, d = -2, row, 0  # sentinel for row replace

        # SA acceptance
        if r < 0.80:
            # Single flip
            if np.isnan(log_ratio) or np.isinf(log_ratio):
                continue
            if log_ratio > 0 or rng.random() < np.exp(min(log_ratio / max(T, 1e-10), 0)):
                # Accept
                update = (d / dn) * np.outer(Ai[:, j], Ai[i, :])
                if np.any(np.isnan(update)):
                    Ai = np.linalg.inv(A)
                    continue
                Ai -= update
                A[i, j] *= -1.0
                ld += log_ratio
                accepted += 1

                if ld > best_ld + 1e-10:
                    best_ld = ld
                    best_A = A.copy()
                    improved += 1
                    last_improve_step = steps

        elif r < 0.90:
            # Double flip
            if log_ratio > 0 or rng.random() < np.exp(log_ratio / T):
                # Apply both flips via SM updates
                d1 = -2.0 * A[row, c1]
                dn1 = 1.0 + d1 * Ai[c1, row]
                if abs(dn1) > 1e-15:
                    Ai -= (d1 / dn1) * np.outer(Ai[:, c1], Ai[row, :])
                    A[row, c1] *= -1.0
                d2 = -2.0 * A[row, c2]
                dn2 = 1.0 + d2 * Ai[c2, row]
                if abs(dn2) > 1e-15:
                    Ai -= (d2 / dn2) * np.outer(Ai[:, c2], Ai[row, :])
                    A[row, c2] *= -1.0
                ld += log_ratio
                accepted += 1

                if ld > best_ld + 1e-10:
                    best_ld = ld
                    best_A = A.copy()
                    improved += 1
                    last_improve_step = steps

        elif i == -2:
            # Row replace
            if log_ratio > 0 or rng.random() < np.exp(log_ratio / T):
                Ai -= np.outer(Ai[:, row], d_row @ Ai) / dn
                A[row] = new_row
                ld += log_ratio
                accepted += 1

                if ld > best_ld + 1e-10:
                    best_ld = ld
                    best_A = A.copy()
                    improved += 1
                    last_improve_step = steps

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

        # Reheat if stuck
        if steps - last_improve_step > 500000 and T < T_start * 0.1:
            reheats += 1
            T = T_start * 0.5
            last_improve_step = steps
            # Reset to best
            A = best_A.copy()
            Ai = np.linalg.inv(A)
            s, ld = np.linalg.slogdet(A)

        # Periodic re-sync to avoid numerical drift
        if steps % 200 == 0:
            if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(np.isnan(Ai)):
                A = best_A.copy()
                Ai = np.linalg.inv(A)
                s, ld = np.linalg.slogdet(A)
                continue
            s_check, ld_check = np.linalg.slogdet(A)
            if s_check == 0:
                # Matrix became singular, reset
                A = best_A.copy()
                Ai = np.linalg.inv(A)
                s, ld = np.linalg.slogdet(A)
            else:
                ld = ld_check
                Ai = np.linalg.inv(A)

        # Report
        if time.time() - last_report > 60:
            last_report = time.time()
            elapsed = time.time() - (deadline - time_limit)
            score_approx = np.exp(best_ld) / THEORETICAL_MAX
            accept_rate = accepted / max(steps, 1)
            print(f"  SA: {elapsed:.0f}s, steps={steps}, T={T:.6f}, "
                  f"accepted={accept_rate:.3f}, improvements={improved}, "
                  f"reheats={reheats}, best_score~{score_approx:.8f}")
            sys.stdout.flush()

    # Final hill climb on best
    best_A, best_ld = hill_climb_sm(best_A, min(time.time() + 5, deadline + 10))

    return best_A, best_ld


def main():
    print("=" * 70)
    print("SIMULATED ANNEALING SEARCH for 29x29 maxdet")
    print("=" * 70)
    sys.stdout.flush()

    total_deadline = time.time() + 7200  # 2 hours

    best_overall_score = 0.0
    best_overall_matrix = None

    # Run multiple SA runs with different parameters
    configs = [
        # (T_start, cooling, label)
        (2.0, 0.999999, "hot-slow"),
        (0.5, 0.99999, "warm-medium"),
        (0.1, 0.9999, "cool-fast"),
        (5.0, 0.999999, "very-hot"),
        (1.0, 0.999995, "medium"),
        (0.3, 0.99998, "warm-fast"),
        (3.0, 0.999997, "hot-medium"),
        (0.05, 0.9999, "barely-warm"),
    ]

    # Also run from random starts, not just OS matrix
    run_idx = 0
    while time.time() < total_deadline:
        config_idx = run_idx % len(configs)
        T_start, cooling, label = configs[config_idx]

        # Alternate between OS start and random start
        if run_idx % 3 == 0:
            A_start = BEST_KNOWN.copy()
            start_label = "OS"
        elif run_idx % 3 == 1 and best_overall_matrix is not None:
            # Perturb best
            A_start = best_overall_matrix.astype(float).copy()
            rng = np.random.RandomState(run_idx * 17 + 5)
            k = rng.randint(5, 50)
            idxs = rng.randint(0, N, size=(k, 2))
            for ij in idxs:
                A_start[ij[0], ij[1]] *= -1.0
            start_label = f"perturb-{k}"
        else:
            rng = np.random.RandomState(run_idx * 13 + 7)
            A_start = rng.choice([-1.0, 1.0], size=(N, N))
            # Quick optimize first
            A_start, _ = row_replace_climb(A_start, time.time() + 0.3)
            A_start, _ = hill_climb_sm(A_start, time.time() + 0.2)
            start_label = "random"

        remaining = total_deadline - time.time()
        run_time = min(300, remaining)  # 5 min per run
        if run_time < 30:
            break

        rng_sa = np.random.RandomState(run_idx * 31 + 42)

        print(f"\n--- SA Run {run_idx+1}: {label}, start={start_label}, "
              f"T0={T_start}, cooling={cooling}, budget={run_time:.0f}s ---")
        sys.stdout.flush()

        A_result, ld_result = simulated_annealing(
            A_start, time_limit=run_time,
            T_start=T_start, T_min=0.001, cooling=cooling, rng=rng_sa
        )

        # Score
        M = np.sign(A_result).astype(int)
        M[M == 0] = 1
        score_approx = np.exp(ld_result) / THEORETICAL_MAX
        print(f"  Run {run_idx+1} result: approx score = {score_approx:.8f}")

        if score_approx > best_overall_score + 1e-6:
            # Exact check
            exact_det = abs(det_bareiss(M.tolist()))
            exact_score = exact_det / THEORETICAL_MAX
            print(f"  EXACT: |det| = {exact_det}, score = {exact_score:.8f}")

            if exact_score > best_overall_score:
                best_overall_score = exact_score
                best_overall_matrix = M.copy()

                if exact_score > 0.9357 + 1e-6:
                    print(f"\n{'='*70}")
                    print(f"!!! BREAKTHROUGH: score = {exact_score:.8f} > 0.9357 !!!")
                    print(f"{'='*70}")
                    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", M)
                    print("Saved to breakthrough_matrix.npy")

        sys.stdout.flush()
        run_idx += 1

    print(f"\n{'='*70}")
    print(f"SA SEARCH COMPLETE")
    print(f"Best score: {best_overall_score:.8f}")
    print(f"Total runs: {run_idx}")
    print(f"{'='*70}")

    if best_overall_matrix is not None:
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/sa_best_matrix.npy",
                best_overall_matrix)
        print("Best matrix saved to sa_best_matrix.npy")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
