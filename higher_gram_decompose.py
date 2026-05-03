"""Search for {-1,+1} decompositions of Gram matrices with det > OS det.

Strategy:
1. Find Gram matrices G with det(G) > det(G_OS) that are PSD
2. For each, attempt column-by-column backtracking to find R with R^T R = G
3. If successful, we have a matrix with higher |det| than the world record

Key insight: The OS Gram has specific structure. We search nearby Gram matrices
that have slightly higher det, hoping one is realizable.
"""

import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm

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


def build_os_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for i in range(3):
        for j in range(3, 6): G[i, j] = 5; G[j, i] = 5
    for j in range(7, 11): G[6, j] = -3; G[j, 6] = -3
    return G


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng, max_sol=500, deadline=None):
    n = N; k = len(placed_cols)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]
    A_rows = [[Fraction(int(placed_cols[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(G[placed_indices[i], target_idx])) for i in range(k)]
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
    nf = len(free_cols); np_ = len(pivot_cols)
    ic = np.zeros((np_, nf), dtype=np.int64); icons = np.zeros(np_, dtype=np.int64); iden = np.zeros(np_, dtype=np.int64)
    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free_cols]
        lcd = 1
        for d in ds: lcd = math_lcm(lcd, d)
        iden[i] = lcd; icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free_cols): ic[i, fi] = int(aug[i][fc] * lcd)
    solutions = []
    if nf == 0:
        x = np.zeros(n, dtype=np.int64); valid = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]: valid = False; break
            x[pivot_cols[i]] = icons[i] // iden[i]
        if valid: solutions.append(x)
    elif nf <= 20:
        total = 1 << nf; bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]; rng.shuffle(si)
        for idx in si[:max_sol]:
            if deadline and time.time() > deadline: break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
            for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        for _ in range(max(1, max_sol * 500 // (1 << 20))):
            if deadline and time.time() > deadline: break
            if len(solutions) >= max_sol: break
            bs = min(1 << 20, max_sol * 200)
            fm = rng.choice([-1, 1], size=(bs, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(bs, dtype=bool)
            for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(solutions) >= max_sol: break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
                for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
    return solutions


def try_decompose_gram(G, time_limit=30, rng=None):
    """Try to find R in {-1,+1}^{29x29} with R^T R = G."""
    if rng is None:
        rng = np.random.RandomState()

    deadline = time.time() + time_limit

    # Check basic feasibility
    if not np.all(np.diag(G) == N):
        return None
    eigvals = np.linalg.eigvalsh(G.astype(float))
    if np.min(eigvals) < -0.01:
        return None

    # Column ordering: start with most constrained columns
    # (those with most non-1 off-diagonal entries)
    non_standard = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j and G[i, j] != 1:
                non_standard[i] += 1
    col_order = list(np.argsort(-non_standard))

    for attempt in range(20):
        if time.time() > deadline:
            break

        rng.shuffle(col_order)  # After first attempt, randomize order

        placed_cols = []
        placed_indices = []

        def dfs(step):
            if time.time() > deadline:
                return False
            if step == N:
                return True

            target_idx = col_order[step]
            ms = min(50, max(3, 500 // (step + 1)))

            sols = solve_column_constraints(
                placed_cols, placed_indices, target_idx, G, rng,
                max_sol=ms, deadline=min(time.time() + 5, deadline)
            )
            if not sols:
                return False

            rng.shuffle(sols)
            for s in sols[:max(3, ms // 2)]:
                if time.time() > deadline:
                    return False
                placed_cols.append(s)
                placed_indices.append(target_idx)
                if dfs(step + 1):
                    return True
                placed_cols.pop()
                placed_indices.pop()
            return False

        if dfs(0):
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(placed_indices):
                R[:, ci] = placed_cols[i]
            if np.array_equal(R.T @ R, G):
                return R

    return None


def main():
    print("=" * 70)
    print("HIGHER GRAM DECOMPOSITION SEARCH")
    print("=" * 70)
    sys.stdout.flush()

    G_os = build_os_gram()
    det_os = det_bareiss(G_os.tolist())
    score_os = np.sqrt(float(abs(det_os))) / THEORETICAL_MAX
    print(f"OS Gram det = {det_os}")
    print(f"OS score = {score_os:.8f}")
    sys.stdout.flush()

    # Strategy: Perturb the OS Gram to find nearby PSD Gram matrices with higher det
    # The OS Gram has specific structure. We modify a few off-diagonal entries
    # and check if:
    # (a) The new Gram is PSD
    # (b) det(G_new) > det(G_OS)
    # (c) G_new is realizable (can find R with R^T R = G_new)

    rng = np.random.RandomState(42)
    best_gram_det = abs(det_os)
    best_decomp_score = score_os
    best_matrix = None

    deadline = time.time() + 7200  # 2 hours
    last_report = time.time()
    grams_tried = 0
    grams_higher = 0
    decompositions_attempted = 0
    decompositions_found = 0

    # Allowed off-diagonal values for a {-1,+1} matrix inner product
    # With n=29 columns, inner product = (number of agreements) - (number of disagreements)
    # = 2*(agreements) - 29, so inner product must be odd, in {-29, -27, ..., 27, 29}
    # But empirically, common values are in {-5, -3, -1, 1, 3, 5}
    allowed_values = [-5, -3, -1, 1, 3, 5]

    while time.time() < deadline:
        grams_tried += 1

        # Perturb OS Gram
        G = G_os.copy()
        n_perturb = rng.randint(1, 8)  # change 1-7 pairs of entries
        for _ in range(n_perturb):
            i = rng.randint(N)
            j = rng.randint(i + 1, N) if i < N - 1 else rng.randint(0, N - 1)
            if i == j:
                continue
            new_val = rng.choice(allowed_values)
            G[i, j] = new_val
            G[j, i] = new_val

        # Check PSD
        eigvals = np.linalg.eigvalsh(G.astype(float))
        if np.min(eigvals) < -0.01:
            continue

        # Compute det
        det_g = det_bareiss(G.tolist())
        if abs(det_g) <= abs(det_os):
            continue

        grams_higher += 1
        gram_score = np.sqrt(float(abs(det_g))) / THEORETICAL_MAX

        # Found a Gram with higher det! Try to decompose it.
        if gram_score > score_os + 0.001:
            print(f"\n  [{grams_tried}] Higher Gram found! det = {det_g}")
            print(f"    Gram score = {gram_score:.8f} (OS = {score_os:.8f})")
            # Show which entries differ from OS
            diffs = []
            for i in range(N):
                for j in range(i+1, N):
                    if G[i,j] != G_os[i,j]:
                        diffs.append((i, j, int(G_os[i,j]), int(G[i,j])))
            print(f"    {len(diffs)} pairs differ from OS Gram")
            for i, j, old, new in diffs:
                print(f"      G[{i},{j}]: {old} -> {new}")
            sys.stdout.flush()

            # Try decomposition (give more time for more promising Grams)
            time_for_decomp = min(120, max(20, 60 * (gram_score - score_os) / 0.01))
            decompositions_attempted += 1

            R = try_decompose_gram(G, time_limit=time_for_decomp, rng=rng)

            if R is not None:
                decompositions_found += 1
                actual_det = abs(det_bareiss(R.astype(int).tolist()))
                actual_score = actual_det / THEORETICAL_MAX
                print(f"    DECOMPOSITION FOUND!")
                print(f"    |det(R)| = {actual_det}")
                print(f"    score = {actual_score:.8f}")
                sys.stdout.flush()

                if actual_score > best_decomp_score:
                    best_decomp_score = actual_score
                    best_matrix = R.copy()

                    if actual_score > 0.9357 + 1e-6:
                        print(f"\n{'='*70}")
                        print(f"!!! BREAKTHROUGH: score = {actual_score:.8f} !!!")
                        print(f"{'='*70}")
                        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy",
                                R.astype(int))
                        print("Saved!")
                        sys.stdout.flush()
            else:
                print(f"    Could not decompose (time limit {time_for_decomp:.0f}s)")
                sys.stdout.flush()

        # Progress report
        if time.time() - last_report > 60:
            last_report = time.time()
            elapsed = time.time() - (deadline - 7200)
            print(f"\n  Progress: {elapsed:.0f}s, {grams_tried} Grams tried, "
                  f"{grams_higher} higher, {decompositions_attempted} decomp attempted, "
                  f"{decompositions_found} decomp found")
            print(f"  Best decomp score: {best_decomp_score:.8f}")
            sys.stdout.flush()

    # Final report
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"Grams tried: {grams_tried}")
    print(f"Grams with higher det: {grams_higher}")
    print(f"Decompositions attempted: {decompositions_attempted}")
    print(f"Decompositions found: {decompositions_found}")
    print(f"Best decomp score: {best_decomp_score:.8f}")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
