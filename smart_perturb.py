"""Smart perturbation strategies to break through the 0.921 ceiling for 29x29 maxdet.

Key discovery: The 0.921 and 0.936 (Orrick-Solomon) basins differ by 49% Hamming
distance with completely different Gram matrices. Random perturbation from 0.921
cannot reach 0.936. Instead, we construct the OS matrix algebraically:

1. The OS matrix has known block structure:
   - 6x6 block (circulant-based), 4x4 Hadamard, row/col 6 separator
   - Cross blocks between these are all +1 or all -1
2. The bottom 18 entries of the first 11 columns are constructed to satisfy
   exact Gram inner-product constraints via randomized search
3. The remaining 18 columns (11-28) are found via Gram-targeting optimization
4. The full matrix is then polished with row-replacement + SM hill climbing

Strategies tested:
1. Full algebraic construction with Gram targeting
2. Scaffold + massive Gram-targeted optimization
3. Paley + perturbation (baseline)
"""
import numpy as np
import time

N = 29
THEORETICAL_MAX = 1270698346568170340352

# ---------------------------------------------------------------------------
# Target Gram matrix (Orrick-Solomon structure)
# ---------------------------------------------------------------------------
OS_GRAM = np.ones((N, N), dtype=np.int64)
np.fill_diagonal(OS_GRAM, N)
for _i in range(3):
    for _j in range(3, 6):
        OS_GRAM[_i, _j] = 5
        OS_GRAM[_j, _i] = 5
for _j in range(7, 11):
    OS_GRAM[6, _j] = -3
    OS_GRAM[_j, 6] = -3


# ---------------------------------------------------------------------------
# Known algebraic building blocks
# ---------------------------------------------------------------------------
BLOCK_6x6 = np.array([
    [-1, -1,  1,  1,  1,  1],
    [-1,  1, -1,  1,  1,  1],
    [ 1, -1, -1,  1,  1,  1],
    [ 1,  1,  1,  1, -1, -1],
    [ 1,  1,  1, -1, -1,  1],
    [ 1,  1,  1, -1,  1, -1],
], dtype=np.int64)

BLOCK_H4 = np.array([
    [-1, -1,  1, -1],
    [-1, -1, -1,  1],
    [-1,  1, -1, -1],
    [ 1, -1, -1, -1],
], dtype=np.int64)

# Known top 11x11 block
KNOWN_TOP = np.zeros((11, 11), dtype=np.int64)
KNOWN_TOP[:6, :6] = BLOCK_6x6
KNOWN_TOP[6, :] = -1
KNOWN_TOP[:6, 6] = -1
KNOWN_TOP[:6, 7:11] = 1
KNOWN_TOP[7:11, :6] = 1
KNOWN_TOP[7:11, 6] = -1
KNOWN_TOP[7:11, 7:11] = BLOCK_H4


# ---------------------------------------------------------------------------
# Core optimization routines
# ---------------------------------------------------------------------------

def build_gf27():
    elements = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    idx = {e: i for i, e in enumerate(elements)}
    add_t = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            add_t[i, j] = idx[tuple((elements[i][k] + elements[j][k]) % 3 for k in range(3))]
    def mul(a, b):
        c0 = a[0]*b[0]; c1 = a[0]*b[1]+a[1]*b[0]; c2 = a[0]*b[2]+a[1]*b[1]+a[2]*b[0]
        c3 = a[1]*b[2]+a[2]*b[1]; c4 = a[2]*b[2]
        c2 += c4; c1 += 2*c4; c1 += c3; c0 += 2*c3
        return (c0 % 3, c1 % 3, c2 % 3)
    mul_t = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            mul_t[i, j] = idx[mul(elements[i], elements[j])]
    zi, oi = idx[(0, 0, 0)], idx[(1, 0, 0)]
    gen = None
    for g in range(27):
        if g == zi:
            continue
        cur = oi
        for k in range(1, 27):
            cur = mul_t[cur, g]
            if cur == oi:
                if k == 26:
                    gen = g
                break
        if gen:
            break
    dlog = np.full(27, -1, dtype=int)
    cur = oi
    for k in range(26):
        dlog[cur] = k
        cur = mul_t[cur, gen]
    chi = np.zeros(27, dtype=int)
    for i in range(27):
        if i == zi:
            chi[i] = 0
        elif dlog[i] % 2 == 0:
            chi[i] = 1
        else:
            chi[i] = -1
    neg = np.array([idx[tuple((-x) % 3 for x in elements[i])] for i in range(27)], dtype=int)
    return add_t, chi, neg


def build_h28():
    add_t, chi, neg = build_gf27()
    Q = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            Q[i, j] = chi[add_t[i, neg[j]]]
    H = np.zeros((28, 28), dtype=int)
    H[0, 0] = 1
    H[0, 1:] = 1
    H[1:, 0] = -1
    H[1:, 1:] = Q + np.eye(27, dtype=int)
    return H


def border_h28(H28, rng, n_trials=100000):
    H28f = H28.astype(np.float64)
    H28T = H28f.T
    best = []
    best_s = 0
    for _ in range(n_trials):
        s = rng.choice([-1, 1], size=28).astype(np.float64)
        w = H28T @ s
        saw = np.sum(np.abs(w))
        sp = 1.0 + saw / 28.0
        sn = abs(-1.0 + saw / 28.0)
        if sp >= sn:
            sc, r, c = sp, -np.sign(w), 1
        else:
            sc, r, c = sn, np.sign(w), -1
        r[r == 0] = 1.0
        if sc > best_s - 0.2:
            M = np.zeros((29, 29), dtype=np.float64)
            M[0, 0] = c
            M[0, 1:] = r
            M[1:, 0] = s
            M[1:, 1:] = H28f
            best.append((M, sc))
            best_s = max(best_s, sc)
    best.sort(key=lambda x: -x[1])
    return [b[0] for b in best[:50]]


def hill_climb_sm(A, deadline):
    n = A.shape[0]
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
        return A, ld
    cond = np.linalg.cond(A)
    if cond > 1e14:
        return A, ld
    Ai = np.linalg.inv(A)
    steps = 0
    while time.time() < deadline:
        r = 1.0 - 2.0 * A * Ai.T
        ar = np.abs(r)
        ix = np.unravel_index(np.argmax(ar), (n, n))
        if ar[ix] <= 1.0 + 1e-12:
            break
        i, j = ix
        d = -2.0 * A[i, j]
        dn = 1.0 + d * Ai[j, i]
        if abs(dn) < 1e-15:
            break
        Ai -= (d / dn) * np.outer(Ai[:, j], Ai[i, :])
        A[i, j] *= -1.0
        steps += 1
        if steps % 30 == 0:
            s2, _ = np.linalg.slogdet(A)
            if s2 == 0:
                break
            Ai = np.linalg.inv(A)
    _, ld = np.linalg.slogdet(A)
    return A, ld


def row_replace_climb(A, deadline):
    n = A.shape[0]
    A = A.copy()
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
        return A, ld
    cond = np.linalg.cond(A)
    if cond > 1e14:
        return A, ld
    Ai = np.linalg.inv(A)
    while time.time() < deadline:
        imp = False
        for k in range(n):
            rn = (np.sign(Ai[:, k]) * s).astype(float)
            rn[rn == 0] = 1.0
            d = rn - A[k, :]
            if np.all(d == 0):
                continue
            dn = 1.0 + d @ Ai[:, k]
            if abs(dn) <= 1.0 + 1e-10:
                continue
            Ai -= np.outer(Ai[:, k], d @ Ai) / dn
            A[k, :] = rn
            s2, ld = np.linalg.slogdet(A)
            s = s2
            imp = True
        s, ld = np.linalg.slogdet(A)
        if s == 0 or not imp:
            break
        cond = np.linalg.cond(A)
        if cond > 1e14:
            break
        Ai = np.linalg.inv(A)
    return A, ld


def optimize_matrix(A, time_budget=0.25):
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
        return A, ld
    dl = time.time() + time_budget * 0.6
    A, _ = row_replace_climb(A, dl)
    dl = time.time() + time_budget * 0.4
    A, ld = hill_climb_sm(A, dl)
    return A, ld


def get_score(A):
    s, ld = np.linalg.slogdet(A)
    if s == 0:
        return 0.0
    return np.exp(ld) / THEORETICAL_MAX


def get_exact_score(A):
    Ai = np.sign(A).astype(int)
    Ai[Ai == 0] = 1
    M = Ai.tolist()
    n = len(M)
    M2 = [row[:] for row in M]
    for k in range(n - 1):
        if M2[k][k] == 0:
            for i in range(k + 1, n):
                if M2[i][k] != 0:
                    M2[k], M2[i] = M2[i], M2[k]
                    break
            else:
                return 0.0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M2[i][j] * M2[k][k] - M2[i][k] * M2[k][j]
                den = M2[k - 1][k - 1] if k > 0 else 1
                M2[i][j] = num // den
    return abs(M2[-1][-1]) / THEORETICAL_MAX


# ---------------------------------------------------------------------------
# Gram row optimization
# ---------------------------------------------------------------------------

def _find_best_row(T, rng, n_trials=30):
    best_row = None
    best_score = -10**18
    for trial in range(n_trials):
        r = np.ones(N, dtype=np.int64)
        perm = list(range(N))
        rng.shuffle(perm)
        for idx in perm:
            m = int(np.dot(T[idx], r)) - int(T[idx, idx]) * int(r[idx])
            r[idx] = 1 if m > 0 else (-1 if m < 0 else rng.choice([-1, 1]))
            if trial < n_trials // 3 and rng.random() < 0.12:
                r[idx] = -r[idx]
        Tr = T @ r
        for _ in range(5):
            any_flip = False
            for idx in range(N):
                delta = -4 * int(r[idx]) * int(Tr[idx]) + 4 * int(T[idx, idx])
                if delta > 0:
                    ov = r[idx]
                    r[idx] = -ov
                    Tr += T[:, idx] * (-2 * ov)
                    any_flip = True
            if not any_flip:
                break
        score = int(r @ (T @ r))
        if score > best_score:
            best_score = score
            best_row = r.copy()
    return best_row


# ===========================================================================
# STRATEGY 1: Full Algebraic Construction with Gram Targeting
# ===========================================================================

def construct_first_11_columns(rng, max_time=2.0):
    """Construct the bottom 18 entries for columns 0-10 satisfying all
    pairwise Gram constraints exactly.

    Constraints on bottom parts (rows 11-28):
    - Cols 0-2: sum=6 each, pairwise ip=-6
    - Cols 3-5: sum=6 each, pairwise ip=-6, ip with cols 0-2 = 2
    - Col 6: all +1 (sum=18)
    - Cols 7-10: sum=0 each, pairwise ip=-6, ip with cols 0-5 = 0
    """
    deadline = time.time() + max_time

    while time.time() < deadline:
        v = [None] * 11

        # Col 0: sum=6 (12 ones, 6 neg-ones)
        v[0] = np.ones(18, dtype=np.int64)
        v[0][rng.choice(18, size=6, replace=False)] = -1

        # Col 1: ip(0,1)=-6, sum=6
        found1 = False
        for _ in range(5000):
            v[1] = np.ones(18, dtype=np.int64)
            v[1][rng.choice(18, size=6, replace=False)] = -1
            if np.dot(v[0], v[1]) == -6:
                found1 = True
                break
        if not found1:
            continue

        # Col 2: ip(0,2)=-6, ip(1,2)=-6, sum=6
        found2 = False
        for _ in range(50000):
            v[2] = np.ones(18, dtype=np.int64)
            v[2][rng.choice(18, size=6, replace=False)] = -1
            if np.dot(v[0], v[2]) == -6 and np.dot(v[1], v[2]) == -6:
                found2 = True
                break
        if not found2:
            continue

        # Cols 3-5: sum=6, pairwise ip=-6, ip with 0-2 = 2
        found35 = False
        for _ in range(50000):
            if time.time() > deadline:
                break
            v[3] = np.ones(18, dtype=np.int64)
            v[3][rng.choice(18, size=6, replace=False)] = -1
            if not (np.dot(v[0], v[3]) == 2 and np.dot(v[1], v[3]) == 2
                    and np.dot(v[2], v[3]) == 2):
                continue
            for __ in range(5000):
                v[4] = np.ones(18, dtype=np.int64)
                v[4][rng.choice(18, size=6, replace=False)] = -1
                if not (np.dot(v[0], v[4]) == 2 and np.dot(v[1], v[4]) == 2
                        and np.dot(v[2], v[4]) == 2 and np.dot(v[3], v[4]) == -6):
                    continue
                for ___ in range(5000):
                    v[5] = np.ones(18, dtype=np.int64)
                    v[5][rng.choice(18, size=6, replace=False)] = -1
                    if (np.dot(v[0], v[5]) == 2 and np.dot(v[1], v[5]) == 2
                            and np.dot(v[2], v[5]) == 2 and np.dot(v[3], v[5]) == -6
                            and np.dot(v[4], v[5]) == -6):
                        found35 = True
                        break
                if found35:
                    break
            if found35:
                break
        if not found35:
            continue

        # Col 6: all +1
        v[6] = np.ones(18, dtype=np.int64)

        # Cols 7-10: sum=0 (9 ones, 9 neg-ones), pairwise ip=-6, ip with 0-5=0
        found710 = False
        for _ in range(50000):
            if time.time() > deadline:
                break
            v[7] = np.ones(18, dtype=np.int64)
            v[7][rng.choice(18, size=9, replace=False)] = -1
            if not all(np.dot(v[i], v[7]) == 0 for i in range(6)):
                continue
            for __ in range(5000):
                v[8] = np.ones(18, dtype=np.int64)
                v[8][rng.choice(18, size=9, replace=False)] = -1
                if not (all(np.dot(v[i], v[8]) == 0 for i in range(6))
                        and np.dot(v[7], v[8]) == -6):
                    continue
                for ___ in range(5000):
                    v[9] = np.ones(18, dtype=np.int64)
                    v[9][rng.choice(18, size=9, replace=False)] = -1
                    if not (all(np.dot(v[i], v[9]) == 0 for i in range(6))
                            and np.dot(v[7], v[9]) == -6
                            and np.dot(v[8], v[9]) == -6):
                        continue
                    for ____ in range(5000):
                        v[10] = np.ones(18, dtype=np.int64)
                        v[10][rng.choice(18, size=9, replace=False)] = -1
                        if (all(np.dot(v[i], v[10]) == 0 for i in range(6))
                                and np.dot(v[7], v[10]) == -6
                                and np.dot(v[8], v[10]) == -6
                                and np.dot(v[9], v[10]) == -6):
                            found710 = True
                            break
                    if found710:
                        break
                if found710:
                    break
            if found710:
                break
        if not found710:
            continue

        return v

    return None


def assemble_matrix(col_bottoms, rng):
    """Assemble a full 29x29 matrix from the column bottom parts."""
    R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)

    # Known top 11x11
    R[:11, :11] = KNOWN_TOP

    # Column 6 bottom (rows 11-28): all +1
    R[11:, 6] = 1

    # Row 6 cols 11-28: all +1
    R[6, 11:] = 1

    # Column bottoms (rows 11-28) for cols 0-5, 7-10
    for c in range(6):
        R[11:, c] = col_bottoms[c]
    for c in range(7, 11):
        R[11:, c] = col_bottoms[c]

    return R


def gram_target_remaining(R, rng, time_limit=30):
    """Use Gram targeting to optimize the remaining 18 columns."""
    G = OS_GRAM
    R = R.copy()
    gram = R.T @ R
    err = int(np.sum((gram - G) ** 2))
    best_err = err
    best_R = R.copy()

    deadline = time.time() + time_limit

    for iteration in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for k in rng.permutation(N):
            if time.time() > deadline:
                break
            old_row = R[k].copy()
            gram_minus = gram - np.outer(old_row, old_row)
            T = G - gram_minus
            new_row = _find_best_row(T, rng, n_trials=20)
            new_gram = gram_minus + np.outer(new_row, new_row)
            new_err = int(np.sum((new_gram - G) ** 2))
            if new_err < err:
                R[k] = new_row
                gram = new_gram
                err = new_err
                improved = True
                if err == 0:
                    return R, 0

        if err < best_err:
            best_err = err
            best_R = R.copy()

        if not improved:
            # Two-row replacement
            found = False
            for _ in range(10):
                k1, k2 = rng.choice(N, size=2, replace=False)
                gm2 = gram - np.outer(R[k1], R[k1]) - np.outer(R[k2], R[k2])
                T2 = G - gm2
                r1 = _find_best_row(T2, rng, n_trials=10)
                T2b = T2 - np.outer(r1, r1)
                r2 = _find_best_row(T2b, rng, n_trials=10)
                ng2 = gm2 + np.outer(r1, r1) + np.outer(r2, r2)
                ne2 = int(np.sum((ng2 - G) ** 2))
                if ne2 < err:
                    R[k1] = r1
                    R[k2] = r2
                    gram = ng2
                    err = ne2
                    found = True
                    break
            if not found:
                break

    return best_R, best_err


def strategy_algebraic(rng, time_limit=60):
    """Full algebraic construction: build first 11 columns exactly,
    then Gram-target the remaining 18 columns."""
    print("\n=== Strategy 1: Algebraic Construction + Gram Targeting ===")

    best_A = None
    best_score = 0.0
    deadline = time.time() + time_limit
    restarts = 0

    while time.time() < deadline:
        restarts += 1

        # Step 1: Construct first 11 column bottoms
        col_bottoms = construct_first_11_columns(rng, max_time=min(3.0, deadline - time.time()))
        if col_bottoms is None:
            continue

        # Step 2: Assemble matrix
        R = assemble_matrix(col_bottoms, rng)

        # Step 3: Gram-target the remaining columns
        remaining_time = min(15.0, deadline - time.time() - 1.0)
        if remaining_time < 1.0:
            break
        R, gram_err = gram_target_remaining(R, rng, time_limit=remaining_time)

        if gram_err == 0:
            score = get_score(R.astype(float))
            print(f"  [{restarts}] EXACT Gram decomposition! Score: {score:.6f}")
            return R.astype(float), score

        # Step 4: Optimize for det
        A = R.astype(float)
        A, ld = optimize_matrix(A, 1.0)
        sc = get_score(A)

        if sc > best_score + 1e-8:
            best_score = sc
            best_A = A.copy()
            print(f"  [{restarts}] Score: {sc:.6f}, Gram err: {gram_err}")

    print(f"  Restarts: {restarts}")
    print(f"  Best score: {best_score:.6f}")
    return best_A, best_score


# ===========================================================================
# STRATEGY 2: Massive scaffold Gram optimization
# ===========================================================================

def strategy_scaffold_gram(rng, time_limit=60):
    """Build scaffold matrices and run aggressive Gram targeting."""
    print("\n=== Strategy 2: Scaffold + Massive Gram Optimization ===")

    G = OS_GRAM
    L = np.linalg.cholesky(G.astype(float))

    best_A = None
    best_score = 0.0
    best_gram_err = 10**18
    deadline = time.time() + time_limit
    restarts = 0

    while time.time() < deadline:
        restarts += 1

        # Initialize
        if restarts % 3 == 0:
            # Cholesky-based
            noise = rng.randn(N, N) * (0.3 + rng.random() * 0.7)
            R = np.sign(L.T + noise).astype(np.int64)
            R[R == 0] = 1
        elif restarts % 3 == 1 and best_A is not None:
            # Perturb best
            R = np.sign(best_A).astype(np.int64)
            R[R == 0] = 1
            n_perturb = rng.randint(3, 10)
            for _ in range(n_perturb):
                r = rng.randint(N)
                R[r] = rng.choice([-1, 1], size=N).astype(np.int64)
        else:
            # Scaffold-based
            col_bottoms = construct_first_11_columns(rng, max_time=1.0)
            if col_bottoms is None:
                R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)
            else:
                R = assemble_matrix(col_bottoms, rng)

        remaining = min(10.0, deadline - time.time() - 1.0)
        if remaining < 1.0:
            break
        R, gram_err = gram_target_remaining(R, rng, time_limit=remaining)

        if gram_err < best_gram_err:
            best_gram_err = gram_err
            print(f"  [{restarts}] Gram error: {gram_err}")

        if gram_err == 0:
            score = get_score(R.astype(float))
            print(f"  EXACT Gram! Score: {score:.6f}")
            return R.astype(float), score

        A = R.astype(float)
        A, ld = optimize_matrix(A, 0.5)
        sc = get_score(A)
        if sc > best_score + 1e-8:
            best_score = sc
            best_A = A.copy()

    print(f"  Restarts: {restarts}, Best Gram error: {best_gram_err}")
    print(f"  Best score: {best_score:.6f}")
    return best_A, best_score


# ===========================================================================
# STRATEGY 3: Paley baseline
# ===========================================================================

def strategy_paley_baseline(rng, time_limit=60):
    """Paley construction + random perturbation baseline."""
    print("\n=== Strategy 3: Paley + Perturbation (Baseline) ===")

    H28 = build_h28()
    bordered = border_h28(H28, rng)

    best_A = None
    best_score = 0.0

    for M in bordered[:50]:
        A = M.copy()
        A, ld = optimize_matrix(A, 0.3)
        sc = get_score(A)
        if sc > best_score:
            best_score = sc
            best_A = A.copy()

    print(f"  After Paley init: {best_score:.6f}")

    deadline = time.time() + time_limit
    count = 0
    while time.time() < deadline:
        count += 1
        if best_A is not None and rng.random() < 0.85:
            A = best_A.copy()
            k = rng.randint(5, min(20 + count // 50, 80))
            idxs = rng.randint(0, N, size=(k, 2))
            for idx in idxs:
                A[idx[0], idx[1]] *= -1.0
        else:
            A = rng.choice([-1.0, 1.0], size=(N, N))
        A, ld = optimize_matrix(A, 0.15)
        sc = get_score(A)
        if sc > best_score + 1e-8:
            best_score = sc
            best_A = A.copy()
            print(f"  [{count}] New best: {best_score:.6f}")

    print(f"  Restarts: {count}")
    print(f"  Best score: {best_score:.6f}")
    return best_A, best_score


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Smart Perturbation Strategies for 29x29 Maximal Determinant")
    print("Breaking through the 0.921 ceiling")
    print("=" * 70)

    results = {}

    # Strategy 1: Algebraic construction
    rng1 = np.random.RandomState(42)
    A1, s1 = strategy_algebraic(rng1, time_limit=60)
    results["algebraic"] = (A1, s1)

    # Strategy 2: Scaffold + Gram
    rng2 = np.random.RandomState(123)
    A2, s2 = strategy_scaffold_gram(rng2, time_limit=60)
    results["scaffold_gram"] = (A2, s2)

    # Strategy 3: Paley baseline
    rng3 = np.random.RandomState(456)
    A3, s3 = strategy_paley_baseline(rng3, time_limit=60)
    results["paley_baseline"] = (A3, s3)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    best_name = None
    best_overall = 0.0
    best_A_overall = None

    for name, (A, sc) in sorted(results.items(), key=lambda x: -x[1][1]):
        exact = get_exact_score(A) if A is not None else 0.0
        marker = " <-- BEST" if sc >= max(v[1] for v in results.values()) - 1e-8 else ""
        print(f"  {name:25s}: approx={sc:.6f}, exact={exact:.6f}{marker}")
        if sc > best_overall:
            best_overall = sc
            best_name = name
            best_A_overall = A

    print(f"\nBest strategy: {best_name} with score {best_overall:.6f}")
    print(f"Baseline (Paley + random flip): 0.921053")
    print(f"Orrick-Solomon target:          0.935673")
    broke = best_overall > 0.922
    print(f"Broke through 0.921 ceiling:    {'YES' if broke else 'NO'}")

    if best_A_overall is not None:
        Ai = np.sign(best_A_overall).astype(int)
        Ai[Ai == 0] = 1
        G = Ai.T @ Ai
        from collections import Counter
        off_diag = G[np.triu_indices(N, 1)]
        print(f"\nBest matrix Gram off-diagonal: {Counter(off_diag)}")
        gram_err = np.sum((G - OS_GRAM) ** 2)
        print(f"Gram distance to OS: {gram_err}")


if __name__ == "__main__":
    main()
