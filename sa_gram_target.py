"""
Simulated annealing targeting the specific Gram matrix.

Strategy: Start from a random +/-1 matrix, flip entries to minimize ||R^T R - G||^2.
Use very efficient incremental updates.

Key: When we flip R[i,j], the Gram changes as:
G'[a,b] = G[a,b] + delta
where delta is nonzero only for entries involving column j.

Specifically, flipping R[i,j] (from +1 to -1 or vice versa):
- Changes R[i,j] by delta_ij = -2*R[i,j] (from +1 to -1: -2; from -1 to +1: +2)
- For G[j,b] = sum_k R[k,j]*R[k,b]:
  New G[j,b] = old G[j,b] + delta_ij * R[i,b]
- For G[a,j] = G[j,a] similarly
- For G[j,j] = sum_k R[k,j]^2 = N (always, so no change)

So the error change when flipping R[i,j]:
err_change = sum over b != j of: (new_G[j,b] - target[j,b])^2 - (old_G[j,b] - target[j,b])^2
           + sum over a != j of: (new_G[a,j] - target[a,j])^2 - (old_G[a,j] - target[a,j])^2
Since G is symmetric: err_change = 2 * sum_{b != j} of: (new_G[j,b] - target[j,b])^2 - (old_G[j,b] - target[j,b])^2

Where new_G[j,b] = old_G[j,b] + delta_ij * R[i,b]
     = old_G[j,b] - 2*R[i,j]*R[i,b]

Let e[j,b] = old_G[j,b] - target[j,b] (current error)
new_error[j,b] = e[j,b] - 2*R[i,j]*R[i,b]

err_change = 2 * sum_{b != j} ((e[j,b] - 2*R[i,j]*R[i,b])^2 - e[j,b]^2)
           = 2 * sum_{b != j} (-4*R[i,j]*R[i,b]*e[j,b] + 4*R[i,j]^2*R[i,b]^2)
           = 2 * sum_{b != j} (-4*R[i,j]*R[i,b]*e[j,b] + 4)
           = 2 * (-4*R[i,j] * sum_{b != j} R[i,b]*e[j,b] + 4*(N-1))
           = -8*R[i,j] * (sum_b R[i,b]*e[j,b] - R[i,j]*e[j,j]) + 8*(N-1)

But e[j,j] = old_G[j,j] - target[j,j] = N - N = 0 (diagonal is always correct)
So:
err_change = -8*R[i,j] * sum_b R[i,b]*e[j,b] + 8*(N-1)
           = -8*R[i,j] * (R[i,:] . e[j,:]) + 8*(N-1)

Where R[i,:] is row i of R and e[j,:] is the error row for column j.
"""
import numpy as np
import time
import sys
import math

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
    if R is None:
        return False
    RR = R.astype(np.int64)
    if not np.all(np.abs(RR) == 1):
        return False
    return np.array_equal(RR.T @ RR, G)


def sa_search(G, time_limit=3600, verbose=True):
    """Simulated annealing on R to minimize ||R^T R - G||^2."""
    if verbose:
        print("Simulated Annealing Gram Targeting")
        sys.stdout.flush()

    target = G.astype(np.int64)
    deadline = time.time() + time_limit
    start_time = time.time()

    best_R = None
    best_error = float('inf')
    n_restarts = 0
    n_found_zero = 0

    # Try starting from Solomon R (it has R^T R close to G)
    solomon_R = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    while time.time() < deadline:
        n_restarts += 1
        rng = np.random.RandomState(n_restarts)

        # Initialize R
        if n_restarts <= 5:
            R = solomon_R.copy()
            # Perturb a few entries
            n_flip = rng.randint(10, 100)
            for _ in range(n_flip):
                i, j = rng.randint(0, N), rng.randint(0, N)
                R[i, j] *= -1
        elif best_R is not None and rng.random() < 0.5:
            R = best_R.copy()
            n_flip = rng.randint(5, 50)
            for _ in range(n_flip):
                i, j = rng.randint(0, N), rng.randint(0, N)
                R[i, j] *= -1
        else:
            R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)

        # Compute Gram and error
        gram = (R.T @ R).astype(np.int64)
        error_matrix = gram - target
        total_error = int(np.sum(error_matrix ** 2))

        if total_error == 0:
            if verify_decomposition(R, target):
                if verbose:
                    print(f"  EXACT at restart {n_restarts}!")
                return R

        # SA parameters
        T = 50.0
        T_min = 0.01
        alpha = 0.9999
        steps_per_temp = N * N

        step = 0
        while time.time() < deadline and T > T_min:
            improved_this_round = False

            for _ in range(steps_per_temp):
                step += 1

                # Random flip
                i = rng.randint(0, N)
                j = rng.randint(0, N)

                # Compute error change efficiently
                # When flipping R[i,j]:
                # new_gram[j,b] = gram[j,b] - 2*R[i,j]*R[i,b] for all b
                # new_gram[a,j] = gram[a,j] - 2*R[i,j]*R[i,a] for all a
                # Diagonal: no change

                delta_rij = -2 * R[i, j]
                row_i = R[i, :]  # row i of R

                # Error change = 2 * sum_{b != j} [(e[j,b] + delta_rij * R[i,b])^2 - e[j,b]^2]
                # = 2 * sum_{b != j} [2*delta_rij*R[i,b]*e[j,b] + delta_rij^2*R[i,b]^2]
                # = 2 * [2*delta_rij * (sum_b R[i,b]*e[j,b] - R[i,j]*e[j,j]) + 4*(N-1)]
                # e[j,j] = 0 always

                dot_re = int(np.dot(row_i, error_matrix[j]))
                # Note: error_matrix is symmetric, so error_matrix[j] == error_matrix[:,j]

                err_change = 2 * (2 * delta_rij * dot_re + 4 * (N - 1))

                accept = False
                if err_change < 0:
                    accept = True
                elif T > 0 and err_change > 0:
                    if rng.random() < math.exp(-err_change / T):
                        accept = True

                if accept:
                    # Apply flip
                    R[i, j] *= -1

                    # Update gram and error_matrix
                    for b in range(N):
                        if b == j:
                            continue
                        gram[j, b] += delta_rij * row_i[b]
                        gram[b, j] = gram[j, b]
                        error_matrix[j, b] = gram[j, b] - target[j, b]
                        error_matrix[b, j] = error_matrix[j, b]

                    total_error += err_change

                    if err_change < 0:
                        improved_this_round = True

                    if total_error == 0:
                        if verify_decomposition(R, target):
                            elapsed = time.time() - start_time
                            if verbose:
                                print(f"  EXACT! restart={n_restarts}, step={step}, T={T:.2f}, {elapsed:.1f}s")
                            return R

            T *= alpha

            if total_error < best_error:
                best_error = total_error
                best_R = R.copy()
                if verbose and (best_error < 100 or n_restarts <= 5):
                    elapsed = time.time() - start_time
                    print(f"  Restart {n_restarts}: error={total_error}, T={T:.2f}, step={step}, {elapsed:.1f}s")
                    sys.stdout.flush()

            if not improved_this_round and T < 1.0:
                break  # Stuck, restart

        if n_restarts % 20 == 0 and verbose:
            elapsed = time.time() - start_time
            print(f"  Restart {n_restarts}: best_error={best_error}, {elapsed:.0f}s")
            sys.stdout.flush()

    if verbose:
        print(f"\nBest error: {best_error} after {n_restarts} restarts")
    return None


def greedy_descent(G, time_limit=3600, verbose=True):
    """
    Pure greedy descent: flip the entry that reduces error the most.
    Restart when stuck.
    """
    if verbose:
        print("Greedy descent with restarts")
        sys.stdout.flush()

    target = G.astype(np.int64)
    deadline = time.time() + time_limit
    start_time = time.time()

    best_R = None
    best_error = float('inf')
    n_restarts = 0

    solomon_R = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    while time.time() < deadline:
        n_restarts += 1
        rng = np.random.RandomState(n_restarts * 137 + 42)

        if n_restarts == 1:
            R = solomon_R.copy()
        elif best_R is not None and rng.random() < 0.6:
            R = best_R.copy()
            n_flip = rng.randint(1, 30)
            for _ in range(n_flip):
                R[rng.randint(N), rng.randint(N)] *= -1
        else:
            R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)

        gram = (R.T @ R).astype(np.int64)
        err = gram - target
        total_error = int(np.sum(err ** 2))

        if total_error == 0:
            if verify_decomposition(R, target):
                return R

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Precompute: for each (i,j), the error change if we flip R[i,j]
            # This is O(N^3) which is fine for N=29

            best_change = 0
            best_ij = None

            for j in range(N):
                for i in range(N):
                    delta_rij = -2 * R[i, j]
                    row_i = R[i, :]
                    dot_re = int(np.dot(row_i, err[j]))
                    err_change = 2 * (2 * delta_rij * dot_re + 4 * (N - 1))

                    if err_change < best_change:
                        best_change = err_change
                        best_ij = (i, j)

            if best_ij is not None and best_change < 0:
                i, j = best_ij
                delta_rij = -2 * R[i, j]
                R[i, j] *= -1

                for b in range(N):
                    if b == j:
                        continue
                    gram[j, b] += delta_rij * R[i, b]  # R[i,b] already has the old value (only R[i,j] changed)
                    # Wait, R[i,j] was the one flipped. R[i,b] for b != j is unchanged.
                    # But the row_i we used was BEFORE the flip. Since only R[i,j] changed,
                    # R[i,b] for b != j is correct.
                    gram[b, j] = gram[j, b]
                    err[j, b] = gram[j, b] - target[j, b]
                    err[b, j] = err[j, b]

                total_error += best_change
                improved = True

                if total_error == 0:
                    if verify_decomposition(R, target):
                        elapsed = time.time() - start_time
                        if verbose:
                            print(f"  EXACT! restart={n_restarts}, {elapsed:.1f}s")
                        return R

        if total_error < best_error:
            best_error = total_error
            best_R = R.copy()
            if verbose:
                elapsed = time.time() - start_time
                # Count non-zero errors
                n_wrong = np.sum(err != 0) // 2  # symmetric, so divide by 2
                print(f"  Restart {n_restarts}: error={total_error}, wrong_pairs={n_wrong}, {elapsed:.1f}s")
                sys.stdout.flush()

    if verbose:
        print(f"\nBest error: {best_error} after {n_restarts} restarts")
    return None


def row_replacement_targeting(G, time_limit=3600, verbose=True):
    """
    Row replacement: for each row of R, find the optimal +/-1 row
    to minimize the Gram error. Iterate until convergence.

    For row k, the optimal row r minimizes:
    ||R^T R - G||^2 where R has row k replaced by r.

    R^T R = sum_m r_m r_m^T
    Replacing row k: new Gram = old Gram - r_k r_k^T + r r^T
    = old Gram + (r - r_k)(r + r_k)^T + ... actually let me just compute directly.

    Error = ||sum_m r_m r_m^T - G||^2
    The term involving r (row k) is: r r^T
    So error = ||A + r r^T - G||^2 where A = sum_{m != k} r_m r_m^T

    To minimize over +/-1 vector r:
    Let T = G - A = G - gram + r_k r_k^T
    error = ||r r^T - T||^2 = sum_{a,b} (r[a]*r[b] - T[a,b])^2

    For given r: r[a]*r[b] = 1 if same sign, -1 if different.
    This is a graph cut problem!

    Expand: error = sum_{a,b} (r[a]^2*r[b]^2 - 2*r[a]*r[b]*T[a,b] + T[a,b]^2)
    = N^2 - 2*r^T T r + ||T||^2
    = const - 2*r^T T r

    So maximizing r^T T r over +/-1 vectors r minimizes the error.
    This is a MAXCUT-like problem (NP-hard in general), but for N=29 we can
    use the greedy + local search approach.
    """
    if verbose:
        print("Row replacement targeting (maximizing r^T T r)")
        sys.stdout.flush()

    target = G.astype(np.int64)
    deadline = time.time() + time_limit
    start_time = time.time()

    best_R = None
    best_error = float('inf')
    n_restarts = 0

    solomon_R = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    while time.time() < deadline:
        n_restarts += 1
        rng = np.random.RandomState(n_restarts * 7 + 13)

        if n_restarts == 1:
            R = solomon_R.copy()
        elif best_R is not None and rng.random() < 0.7:
            R = best_R.copy()
            n_perturb = rng.randint(1, 8)
            for _ in range(n_perturb):
                row = rng.randint(N)
                R[row] = rng.choice([-1, 1], size=N).astype(np.int64)
        else:
            R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)

        gram = (R.T @ R).astype(np.int64)
        total_error = int(np.sum((gram - target) ** 2))

        if total_error == 0:
            if verify_decomposition(R, target):
                return R

        # Row replacement iterations
        for iteration in range(200):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(N))
            rng.shuffle(order)

            for k in order:
                if time.time() > deadline:
                    break

                # T = target - gram + r_k r_k^T
                old_row = R[k].copy()
                T = target - gram + np.outer(old_row, old_row)

                # Maximize r^T T r over +/-1 r
                # Greedy: r[a] = sign(sum_b T[a,b] * r[b]) but this is circular
                # Use: r[a] = sign(T[a,:] . r) iteratively

                # Start from T's dominant eigenvector direction
                # Use power iteration on T
                r = rng.choice([-1, 1], size=N).astype(np.float64)
                for _ in range(20):
                    r = T.astype(np.float64) @ r
                    norm = np.linalg.norm(r)
                    if norm > 0:
                        r /= norm

                r_best = np.sign(r).astype(np.int64)
                r_best[r_best == 0] = 1
                score_best = int(r_best @ T @ r_best)

                # Local search
                for _ in range(5):
                    imp = False
                    Tr = T @ r_best
                    for a in range(N):
                        # Flip r_best[a]: new score = old score - 4*r_best[a]*Tr[a] + 4*T[a,a]
                        delta = -4 * int(r_best[a]) * int(Tr[a]) + 4 * int(T[a, a])
                        if delta > 0:
                            r_best[a] *= -1
                            Tr += T[:, a] * (-2 * r_best[a])  # Update Tr
                            # Wait, we just flipped r_best[a], so the new value is -old.
                            # Tr[b] += T[b,a] * (new_r[a] - old_r[a]) = T[b,a] * (-2*old_r[a])
                            # But we already flipped, so r_best[a] = -old_r[a], old = -r_best[a]
                            # Tr += T[:,a] * (r_best[a] - (-r_best[a])) = T[:,a] * 2*r_best[a]
                            Tr = T @ r_best  # Recompute for safety
                            score_best += delta
                            imp = True
                    if not imp:
                        break

                # Also try a few random starts
                for trial in range(10):
                    r2 = rng.choice([-1, 1], size=N).astype(np.int64)
                    Tr2 = T @ r2
                    for _ in range(5):
                        imp2 = False
                        for a in range(N):
                            delta2 = -4 * int(r2[a]) * int(Tr2[a]) + 4 * int(T[a, a])
                            if delta2 > 0:
                                old_val = r2[a]
                                r2[a] *= -1
                                Tr2 += T[:, a] * (r2[a] - old_val)
                                imp2 = True
                        if not imp2:
                            break
                    score2 = int(r2 @ T @ r2)
                    if score2 > score_best:
                        r_best = r2
                        score_best = score2

                # Apply if better
                new_gram = gram - np.outer(old_row, old_row) + np.outer(r_best, r_best)
                new_error = int(np.sum((new_gram - target) ** 2))

                if new_error < total_error:
                    R[k] = r_best
                    gram = new_gram
                    total_error = new_error
                    improved = True

                    if total_error == 0:
                        if verify_decomposition(R, target):
                            elapsed = time.time() - start_time
                            if verbose:
                                print(f"  EXACT! restart={n_restarts}, iter={iteration}, {elapsed:.1f}s")
                            return R

            if not improved:
                break

        if total_error < best_error:
            best_error = total_error
            best_R = R.copy()
            if verbose:
                elapsed = time.time() - start_time
                n_wrong = np.sum((gram - target) != 0) // 2
                print(f"  Restart {n_restarts}: error={total_error}, wrong_pairs={n_wrong}, {elapsed:.1f}s")
                sys.stdout.flush()

    if verbose:
        print(f"\nBest error: {best_error} after {n_restarts} restarts")
    return None


def main():
    G = build_new_gram()
    print("=" * 70)
    print("SA + GREEDY GRAM TARGETING")
    print("=" * 70)
    sys.stdout.flush()

    # Row replacement (most effective)
    print("\n--- Row Replacement ---")
    R = row_replacement_targeting(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
        det_val = abs(np.linalg.det(R.astype(float)))
        target_max = 1270698346568170340352
        print(f"|det(R)| = {det_val:.6e}")
        print(f"Score = {det_val / target_max:.6f}")
        return R

    # SA
    print("\n--- Simulated Annealing ---")
    R = sa_search(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
        det_val = abs(np.linalg.det(R.astype(float)))
        target_max = 1270698346568170340352
        print(f"|det(R)| = {det_val:.6e}")
        print(f"Score = {det_val / target_max:.6f}")
        return R

    print("\nNo exact solution found.")
    return None


if __name__ == "__main__":
    main()
