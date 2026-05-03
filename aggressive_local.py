"""
Aggressive local search for Gram decomposition.

The row-replacement approach can get error down to ~288 (9 wrong pairs).
This script uses multiple strategies to push past local minima:

1. Row replacement with random restarts
2. Two-row simultaneous optimization
3. Taboo search (remember recent flips)
4. Large neighborhood search (perturb k rows, then optimize)
"""
import numpy as np
import time
import sys

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


def compute_error(R, G):
    gram = R.T @ R
    return int(np.sum((gram - G) ** 2))


def optimize_row(R, G, k, rng, n_trials=30):
    """Find optimal +/-1 row k to minimize Gram error."""
    gram = R.T @ R
    old_row = R[k].copy()
    T = G - gram + np.outer(old_row, old_row)
    # Minimize error = const - 2*r^T T r => maximize r^T T r

    best_row = old_row.copy()
    best_score = int(old_row @ T @ old_row)

    for trial in range(n_trials):
        if trial == 0:
            r = old_row.copy()
        else:
            r = rng.choice([-1, 1], size=N).astype(np.int64)

        # Local search on r
        for _ in range(10):
            Tr = T @ r
            imp = False
            for a in range(N):
                delta = -4 * int(r[a]) * int(Tr[a]) + 4 * int(T[a, a])
                if delta > 0:
                    old_v = r[a]
                    r[a] = -old_v
                    Tr += T[:, a] * (r[a] - old_v)
                    imp = True
            if not imp:
                break

        score = int(r @ T @ r)
        if score > best_score:
            best_score = score
            best_row = r.copy()

    return best_row


def optimize_two_rows(R, G, k1, k2, rng, n_trials=50):
    """Simultaneously optimize rows k1 and k2."""
    gram = R.T @ R
    old_r1 = R[k1].copy()
    old_r2 = R[k2].copy()

    # Remove contributions of rows k1, k2
    A_base = gram - np.outer(old_r1, old_r1) - np.outer(old_r2, old_r2)
    # New gram = A_base + outer(r1, r1) + outer(r2, r2)
    # Error = ||A_base + outer(r1,r1) + outer(r2,r2) - G||^2
    # = ||outer(r1,r1) + outer(r2,r2) - (G - A_base)||^2
    T = G - A_base

    best_r1 = old_r1.copy()
    best_r2 = old_r2.copy()
    best_error = int(np.sum((A_base + np.outer(old_r1, old_r1) + np.outer(old_r2, old_r2) - G) ** 2))

    for trial in range(n_trials):
        if trial == 0:
            r1 = old_r1.copy()
            r2 = old_r2.copy()
        else:
            r1 = rng.choice([-1, 1], size=N).astype(np.int64)
            r2 = rng.choice([-1, 1], size=N).astype(np.int64)

        # Alternating optimization
        for _ in range(20):
            # Fix r2, optimize r1
            T1 = T - np.outer(r2, r2)
            Tr1 = T1 @ r1
            for __ in range(5):
                imp = False
                for a in range(N):
                    delta = -4 * int(r1[a]) * int(Tr1[a]) + 4 * int(T1[a, a])
                    if delta > 0:
                        old_v = r1[a]
                        r1[a] = -old_v
                        Tr1 += T1[:, a] * (r1[a] - old_v)
                        imp = True
                if not imp:
                    break

            # Fix r1, optimize r2
            T2 = T - np.outer(r1, r1)
            Tr2 = T2 @ r2
            for __ in range(5):
                imp = False
                for a in range(N):
                    delta = -4 * int(r2[a]) * int(Tr2[a]) + 4 * int(T2[a, a])
                    if delta > 0:
                        old_v = r2[a]
                        r2[a] = -old_v
                        Tr2 += T2[:, a] * (r2[a] - old_v)
                        imp = True
                if not imp:
                    break

        err = int(np.sum((A_base + np.outer(r1, r1) + np.outer(r2, r2) - G) ** 2))
        if err < best_error:
            best_error = err
            best_r1 = r1.copy()
            best_r2 = r2.copy()

    return best_r1, best_r2, best_error


def main():
    G = build_new_gram()
    print("=" * 70)
    print("AGGRESSIVE LOCAL SEARCH for Gram decomposition")
    print("=" * 70)
    sys.stdout.flush()

    target = G.astype(np.int64)
    deadline = time.time() + 3600
    start_time = time.time()

    solomon_R = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    best_R = None
    best_error = float('inf')
    n_restarts = 0

    while time.time() < deadline:
        n_restarts += 1
        rng = np.random.RandomState(n_restarts)

        # Initialize
        if n_restarts <= 3:
            R = solomon_R.copy()
            n_perturb = rng.randint(1, 5)
            for _ in range(n_perturb):
                R[rng.randint(N)] = rng.choice([-1, 1], size=N).astype(np.int64)
        elif best_R is not None and rng.random() < 0.7:
            R = best_R.copy()
            # Large neighborhood perturbation
            n_perturb = rng.randint(2, 8)
            for _ in range(n_perturb):
                R[rng.randint(N)] = rng.choice([-1, 1], size=N).astype(np.int64)
        else:
            R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)

        # Phase 1: Row replacement
        for iteration in range(100):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(N))
            rng.shuffle(order)

            for k in order:
                if time.time() > deadline:
                    break
                new_row = optimize_row(R, G, k, rng, n_trials=15)
                gram_new = R.T @ R - np.outer(R[k], R[k]) + np.outer(new_row, new_row)
                err_new = int(np.sum((gram_new - target) ** 2))
                err_old = compute_error(R, target)

                if err_new < err_old:
                    R[k] = new_row
                    improved = True

                    if err_new == 0:
                        if verify_decomposition(R, target):
                            elapsed = time.time() - start_time
                            print(f"  EXACT! restart={n_restarts}, phase1, iter={iteration}, {elapsed:.1f}s")
                            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                            return R

            if not improved:
                break

        current_error = compute_error(R, target)

        # Phase 2: Two-row optimization
        if current_error > 0 and current_error <= 500:
            for iteration in range(50):
                if time.time() > deadline:
                    break
                improved = False
                # Try all pairs (or a random subset)
                pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
                rng.shuffle(pairs)

                for k1, k2 in pairs[:min(100, len(pairs))]:
                    if time.time() > deadline:
                        break
                    new_r1, new_r2, err_two = optimize_two_rows(R, G, k1, k2, rng, n_trials=20)

                    if err_two < current_error:
                        R[k1] = new_r1
                        R[k2] = new_r2
                        current_error = err_two
                        improved = True

                        if current_error == 0:
                            if verify_decomposition(R, target):
                                elapsed = time.time() - start_time
                                print(f"  EXACT! restart={n_restarts}, phase2, {elapsed:.1f}s")
                                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                                return R

                if not improved:
                    break

        # Phase 3: Entry-level hill climbing
        if current_error > 0 and current_error <= 200:
            gram = (R.T @ R).astype(np.int64)
            err_matrix = gram - target

            for flip_iter in range(N * N * 10):
                if time.time() > deadline:
                    break

                # Find the best single entry flip
                best_change = 0
                best_ij = None

                for j in range(N):
                    for i in range(N):
                        delta_rij = -2 * R[i, j]
                        dot_re = int(np.dot(R[i, :], err_matrix[j, :]))
                        err_change = 2 * (2 * delta_rij * dot_re + 4 * (N - 1))

                        if err_change < best_change:
                            best_change = err_change
                            best_ij = (i, j)

                if best_ij is None or best_change >= 0:
                    break

                i, j = best_ij
                delta_rij = -2 * R[i, j]
                R[i, j] *= -1

                for b in range(N):
                    if b == j:
                        continue
                    gram[j, b] += delta_rij * R[i, b]
                    gram[b, j] = gram[j, b]
                    err_matrix[j, b] = gram[j, b] - target[j, b]
                    err_matrix[b, j] = err_matrix[j, b]

                current_error += best_change

                if current_error == 0:
                    if verify_decomposition(R, target):
                        elapsed = time.time() - start_time
                        print(f"  EXACT! restart={n_restarts}, phase3, {elapsed:.1f}s")
                        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                        return R

        if current_error < best_error:
            best_error = current_error
            best_R = R.copy()
            elapsed = time.time() - start_time
            n_wrong = np.sum((R.T @ R - target) != 0) // 2
            print(f"  Restart {n_restarts}: error={current_error}, wrong_pairs={n_wrong}, {elapsed:.1f}s")
            sys.stdout.flush()

        if n_restarts % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Checkpoint: restart={n_restarts}, best_error={best_error}, {elapsed:.0f}s")
            sys.stdout.flush()

    print(f"\nBest error: {best_error} after {n_restarts} restarts")
    return None


if __name__ == "__main__":
    main()
