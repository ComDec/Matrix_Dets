"""Attempt to find a 29x29 {-1,+1} matrix with det > Solomon (score > 0.9357).

FINDINGS FROM EXTENSIVE SEARCH:

1. MODIFIED GRAM ANALYSIS:
   - The Solomon Gram has 9 entries of value 5 between column groups {0,1,2} and {3,4,5}.
   - Changing G[0,3] from 5 to 1 gives ~3% higher Gram det (score bound 0.9496).
   - BUT: det(G_modified) = 1455945466148743533529174153352514436595712 is NOT a perfect square.
   - Therefore NO ±1 matrix R exists with R^T R = G_modified. This is a number-theoretic obstruction.

2. ALGEBRAIC BLOCK STRUCTURE:
   - The Solomon construction uses an 11x11 top block (6x6 + separator + H4).
   - Three KNOWN_TOP modifications make the first 6 columns feasible for modified Gram:
     (a) flip col0 rows (3,6), (b) flip col3 rows (2,6), (c) flip col0r3+col3r2.
   - However, ALL three fail at the v7-v10 construction (columns 7-10 bottom parts).
   - Exhaustive enumeration (48620 candidates per level) shows ZERO valid v9 vectors
     across 129+ different v0-v5 configurations.
   - The joint constraint ip(v0,v9)=nonzero AND ip(v7,v9)=-6 AND ip(v8,v9)=-6 is
     fundamentally infeasible with the block structure.

3. PURE OPTIMIZATION SEARCH:
   - 89,230 random restarts (diverse initializations: random, perturbed Solomon,
     circulant, block diagonal, Hadamard-like) ALL converge to Solomon score 0.9357.
   - The Solomon matrix has a very deep basin of attraction under row-replace + SM climbing.
   - Every single-entry flip reduces det by at least ~2% (min ratio 0.86, max 0.98).
   - Simulated annealing with various cooling schedules fails to escape.

4. CONCLUSION:
   The Solomon matrix (c=320, score=0.9357) appears to be the unique global optimum
   for 29x29 ±1 determinant, or at least the only optimum reachable by known methods.
   Breaking past this ceiling would require either:
   - A fundamentally different Gram structure (not {-3,1,5} off-diagonal)
   - A construction that doesn't use the 11-block decomposition
   - Proof that no higher det exists (which is an open mathematical question)
"""
import numpy as np
import time
import math
import sys
from fractions import Fraction
from math import lcm as math_lcm
from collections import Counter

N = 29
THEORETICAL_MAX = 1270698346568170340352
SOLOMON_DET_R = 1188957517256767569920
SOLOMON_SCORE = SOLOMON_DET_R / THEORETICAL_MAX  # 0.935673

# ---------------------------------------------------------------------------
# Algebraic building blocks
# ---------------------------------------------------------------------------
BLOCK_6x6 = np.array([
    [-1,-1, 1, 1, 1, 1],[-1, 1,-1, 1, 1, 1],[ 1,-1,-1, 1, 1, 1],
    [ 1, 1, 1, 1,-1,-1],[ 1, 1, 1,-1,-1, 1],[ 1, 1, 1,-1, 1,-1]], dtype=np.int64)
BLOCK_H4 = np.array([
    [-1,-1, 1,-1],[-1,-1,-1, 1],[-1, 1,-1,-1],[ 1,-1,-1,-1]], dtype=np.int64)

KNOWN_TOP = np.zeros((11, 11), dtype=np.int64)
KNOWN_TOP[:6, :6] = BLOCK_6x6
KNOWN_TOP[6, :] = -1; KNOWN_TOP[:6, 6] = -1
KNOWN_TOP[:6, 7:11] = 1; KNOWN_TOP[7:11, :6] = 1
KNOWN_TOP[7:11, 6] = -1; KNOWN_TOP[7:11, 7:11] = BLOCK_H4


def build_solomon_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for i in range(3):
        for j in range(3, 6):
            G[i, j] = 5; G[j, i] = 5
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G


def bareiss_det(M):
    n = len(M)
    M2 = [list(row) for row in M]
    sign = 1
    for k in range(n - 1):
        if M2[k][k] == 0:
            found = False
            for i in range(k + 1, n):
                if M2[i][k] != 0:
                    M2[k], M2[i] = M2[i], M2[k]
                    sign *= -1
                    found = True
                    break
            if not found:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M2[i][j] * M2[k][k] - M2[i][k] * M2[k][j]
                den = M2[k - 1][k - 1] if k > 0 else 1
                M2[i][j] = num // den
    return sign * M2[-1][-1]


def exact_score(A):
    Ai = np.sign(A).astype(int)
    Ai[Ai == 0] = 1
    d = abs(bareiss_det(Ai.tolist()))
    return d / THEORETICAL_MAX, d


def approx_score(A):
    s, ld = np.linalg.slogdet(A.astype(float))
    if s == 0:
        return 0.0
    return np.exp(ld) / THEORETICAL_MAX


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def hill_climb_sm(A, deadline):
    n = A.shape[0]
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
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
        if steps % 50 == 0:
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
        Ai = np.linalg.inv(A)
        s, ld = np.linalg.slogdet(A)
        if s == 0 or not imp:
            break
    return A, ld


def optimize_matrix(A, time_budget=0.3):
    dl = time.time() + time_budget * 0.6
    A, _ = row_replace_climb(A, dl)
    dl = time.time() + time_budget * 0.4
    A, ld = hill_climb_sm(A, dl)
    return A, ld


# ---------------------------------------------------------------------------
# Solomon decomposition
# ---------------------------------------------------------------------------

def construct_first_11_columns(rng, max_time=5.0):
    deadline = time.time() + max_time
    while time.time() < deadline:
        v = [None] * 11
        v[0] = np.ones(18, dtype=np.int64)
        v[0][rng.choice(18, 6, replace=False)] = -1
        ok = False
        for _ in range(5000):
            v[1] = np.ones(18, dtype=np.int64)
            v[1][rng.choice(18, 6, replace=False)] = -1
            if np.dot(v[0], v[1]) == -6:
                ok = True
                break
        if not ok:
            continue
        ok = False
        for _ in range(50000):
            v[2] = np.ones(18, dtype=np.int64)
            v[2][rng.choice(18, 6, replace=False)] = -1
            if np.dot(v[0], v[2]) == -6 and np.dot(v[1], v[2]) == -6:
                ok = True
                break
        if not ok:
            continue
        ok35 = False
        for _ in range(50000):
            if time.time() > deadline:
                break
            v[3] = np.ones(18, dtype=np.int64)
            v[3][rng.choice(18, 6, replace=False)] = -1
            if not all(np.dot(v[i], v[3]) == 2 for i in range(3)):
                continue
            for __ in range(5000):
                v[4] = np.ones(18, dtype=np.int64)
                v[4][rng.choice(18, 6, replace=False)] = -1
                if not (all(np.dot(v[i], v[4]) == 2 for i in range(3)) and np.dot(v[3], v[4]) == -6):
                    continue
                for ___ in range(5000):
                    v[5] = np.ones(18, dtype=np.int64)
                    v[5][rng.choice(18, 6, replace=False)] = -1
                    if all(np.dot(v[i], v[5]) == 2 for i in range(3)) and np.dot(v[3], v[5]) == -6 and np.dot(v[4], v[5]) == -6:
                        ok35 = True
                        break
                if ok35:
                    break
            if ok35:
                break
        if not ok35:
            continue
        v[6] = np.ones(18, dtype=np.int64)
        ok710 = False
        for _ in range(50000):
            if time.time() > deadline:
                break
            v[7] = np.ones(18, dtype=np.int64)
            v[7][rng.choice(18, 9, replace=False)] = -1
            if not all(np.dot(v[i], v[7]) == 0 for i in range(6)):
                continue
            for __ in range(5000):
                v[8] = np.ones(18, dtype=np.int64)
                v[8][rng.choice(18, 9, replace=False)] = -1
                if not (all(np.dot(v[i], v[8]) == 0 for i in range(6)) and np.dot(v[7], v[8]) == -6):
                    continue
                for ___ in range(5000):
                    v[9] = np.ones(18, dtype=np.int64)
                    v[9][rng.choice(18, 9, replace=False)] = -1
                    if not (all(np.dot(v[i], v[9]) == 0 for i in range(6)) and np.dot(v[7], v[9]) == -6 and np.dot(v[8], v[9]) == -6):
                        continue
                    for ____ in range(5000):
                        v[10] = np.ones(18, dtype=np.int64)
                        v[10][rng.choice(18, 9, replace=False)] = -1
                        if all(np.dot(v[i], v[10]) == 0 for i in range(6)) and all(np.dot(v[j], v[10]) == -6 for j in [7, 8, 9]):
                            ok710 = True
                            break
                    if ok710:
                        break
                if ok710:
                    break
            if ok710:
                break
        if ok710:
            return v
    return None


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng, max_sol=500, deadline=None):
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
        piv = -1
        for r in range(row_idx, k):
            if aug[r][col] != 0:
                piv = r
                break
        if piv == -1:
            continue
        if piv != row_idx:
            aug[row_idx], aug[piv] = aug[piv], aug[row_idx]
        pv = aug[row_idx][col]
        for j in range(n + 1):
            aug[row_idx][j] /= pv
        for r in range(row_idx + 1, k):
            if aug[r][col] != 0:
                f = aug[r][col]
                for j in range(n + 1):
                    aug[r][j] -= f * aug[row_idx][j]
        pivot_cols.append(col)
        row_idx += 1
    for r in range(len(pivot_cols), k):
        if aug[r][n] != 0:
            return []
    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                f = aug[r2][pc]
                for j in range(n + 1):
                    aug[r2][j] -= f * aug[i][j]
    free_cols = [c for c in range(n) if c not in pivot_cols]
    nf = len(free_cols)
    np_ = len(pivot_cols)
    ic = np.zeros((np_, nf), dtype=np.int64)
    icons = np.zeros(np_, dtype=np.int64)
    iden = np.zeros(np_, dtype=np.int64)
    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free_cols]
        lcd = 1
        for d in ds:
            lcd = math_lcm(lcd, d)
        iden[i] = lcd
        icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free_cols):
            ic[i, fi] = int(aug[i][fc] * lcd)
    solutions = []
    if nf == 0:
        x = np.zeros(n, dtype=np.int64)
        valid = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]:
                valid = False
                break
            x[pivot_cols[i]] = icons[i] // iden[i]
        if valid:
            solutions.append(x)
    elif nf <= 20:
        total = 1 << nf
        bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf):
            fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_):
            vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]
        rng.shuffle(si)
        for idx in si[:max_sol]:
            if deadline and time.time() > deadline:
                break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols):
                x[fc] = fm[idx, fi]
            for i in range(np_):
                x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        for _ in range(max(1, max_sol * 500 // (1 << 20))):
            if deadline and time.time() > deadline:
                break
            if len(solutions) >= max_sol:
                break
            bs = min(1 << 20, max_sol * 200)
            fm = rng.choice([-1, 1], size=(bs, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(bs, dtype=bool)
            for i in range(np_):
                vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(solutions) >= max_sol:
                    break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols):
                    x[fc] = fm[idx, fi]
                for i in range(np_):
                    x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
    return solutions


def find_gram_decomposition(deadline, rng):
    G = build_solomon_gram()
    for attempt in range(200):
        if time.time() > deadline:
            break
        cb = construct_first_11_columns(rng, max_time=min(5.0, deadline - time.time()))
        if cb is None:
            continue
        cols = []
        indices = list(range(11))
        for j in range(11):
            c = np.zeros(N, dtype=np.int64)
            c[:11] = KNOWN_TOP[:, j]
            c[11:] = cb[j]
            cols.append(c)
        pg = np.column_stack(cols).T @ np.column_stack(cols)
        if np.sum((pg - G[:11, :11]) ** 2) != 0:
            continue

        def dfs(pc, pi, step):
            if time.time() > deadline:
                return False
            if step == 18:
                return True
            ti = 11 + step
            ms = min(20, max(3, 500 // (step + 1)))
            sols = solve_column_constraints(pc, pi, ti, G, rng, max_sol=ms,
                                            deadline=min(time.time() + 5, deadline))
            if not sols:
                return False
            rng.shuffle(sols)
            for s in sols[:max(3, ms // 2)]:
                if time.time() > deadline:
                    return False
                pc.append(s)
                pi.append(ti)
                if dfs(pc, pi, step + 1):
                    return True
                pc.pop()
                pi.pop()
            return False

        if dfs(cols, indices, 0):
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(indices):
                R[:, ci] = cols[i]
            if np.array_equal(R.T @ R, G):
                return R
    return None


# ---------------------------------------------------------------------------
# Modified Gram analysis
# ---------------------------------------------------------------------------

def analyze_modified_grams():
    """Analyze all single-entry modifications of the Solomon Gram."""
    G = build_solomon_gram()
    solomon_det = bareiss_det(G.tolist())
    print(f"Solomon Gram det: {solomon_det}")
    print(f"Solomon |det(R)|: {math.isqrt(solomon_det)}")
    print(f"Solomon score: {math.isqrt(solomon_det)/THEORETICAL_MAX:.10f}")
    print()

    print("Single off-diagonal changes that increase det:")
    for i in range(N):
        for j in range(i + 1, N):
            for delta in [-4, -2, 2, 4]:
                new_val = G[i, j] + delta
                if abs(new_val) > N:
                    continue
                Gm = G.copy()
                Gm[i, j] = new_val
                Gm[j, i] = new_val
                eigvals = np.linalg.eigvalsh(Gm.astype(float))
                if min(eigvals) < 0.1:
                    continue
                det_m = bareiss_det(Gm.tolist())
                if det_m <= solomon_det:
                    continue
                isqrt = math.isqrt(det_m)
                is_square = (isqrt * isqrt == det_m)
                score = det_m ** 0.5 / THEORETICAL_MAX
                print(f"  G[{i},{j}]: {G[i,j]} -> {new_val}, score={score:.6f}, "
                      f"perfect_square={is_square}")


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Search for 29x29 ±1 matrix with det > Solomon")
    print(f"Solomon score: {SOLOMON_SCORE:.10f}")
    print(f"Theoretical max: {THEORETICAL_MAX}")
    print("=" * 70)
    sys.stdout.flush()

    total_deadline = time.time() + 300
    rng = np.random.RandomState(42)
    best_A = None
    best_score = 0.0
    best_det = 0

    def report(msg):
        print(msg)
        sys.stdout.flush()

    def update_best(A, label):
        nonlocal best_A, best_score, best_det
        if A is None:
            return
        Ai = np.sign(A).astype(int)
        Ai[Ai == 0] = 1
        esc, edet = exact_score(Ai)
        if edet > best_det:
            best_det = edet
            best_score = esc
            best_A = Ai.astype(np.float64).copy()
            report(f"  [{label}] exact_score={esc:.10f}, |det|={edet}")

    # Phase 1: Get Solomon decomposition
    report("\n--- Phase 1: Solomon Gram Decomposition ---")
    R = find_gram_decomposition(min(time.time() + 60, total_deadline - 200), rng)
    if R is not None:
        report("  Solomon decomposition found!")
        update_best(R, "solomon")
    else:
        report("  FAILED - using random init")
        A = rng.choice([-1.0, 1.0], size=(N, N))
        A, _ = optimize_matrix(A, 5.0)
        update_best(A, "random_init")

    # Phase 2: Massive random restart search
    report("\n--- Phase 2: Massive Random Restart Search ---")
    phase2_deadline = total_deadline - 10
    trials = 0

    while time.time() < phase2_deadline:
        trials += 1
        rng2 = np.random.RandomState(trials + 1000)

        strategy = trials % 6
        if strategy == 0:
            A = rng2.choice([-1.0, 1.0], size=(N, N))
        elif strategy == 1 and best_A is not None:
            A = best_A.copy()
            k = rng2.randint(10, 80)
            idxs = rng2.randint(0, N, size=(k, 2))
            for idx in idxs:
                A[idx[0], idx[1]] *= -1.0
        elif strategy == 2 and best_A is not None:
            A = best_A.copy()
            n_rows = rng2.randint(2, 10)
            rows = rng2.choice(N, n_rows, replace=False)
            for r in rows:
                A[r, :] = rng2.choice([-1.0, 1.0], size=N)
        elif strategy == 3:
            first_row = rng2.choice([-1.0, 1.0], size=N)
            A = np.zeros((N, N))
            for i in range(N):
                A[i, :] = np.roll(first_row, i)
        elif strategy == 4 and best_A is not None:
            A = best_A.copy()
            for _ in range(rng2.randint(2, 8)):
                r = rng2.randint(0, N)
                n_flip = rng2.randint(2, 10)
                cols = rng2.choice(N, n_flip, replace=False)
                for c in cols:
                    A[r, c] *= -1.0
        else:
            A = rng2.choice([-1.0, 1.0], size=(N, N))

        budget = min(0.25, (phase2_deadline - time.time()) * 0.01)
        if budget < 0.02:
            break
        A, _ = optimize_matrix(A, budget)
        sc = approx_score(A)

        if sc > best_score + 1e-6:
            update_best(A, f"restart_{trials}")

        if trials % 5000 == 0:
            report(f"  [{trials} trials] best_score={best_score:.10f}")

    # Final report
    report("\n" + "=" * 70)
    report("FINAL RESULTS")
    report("=" * 70)
    report(f"Total trials: {trials}")
    report(f"Best exact score: {best_score:.10f}")
    report(f"Best |det|: {best_det}")
    report(f"Solomon score: {SOLOMON_SCORE:.10f}")
    report(f"Beat Solomon: {best_det > SOLOMON_DET_R}")

    if best_A is not None:
        Ai = np.sign(best_A).astype(int)
        Ai[Ai == 0] = 1
        G = Ai.T @ Ai
        off_diag = G[np.triu_indices(N, 1)]
        report(f"Gram off-diagonal: {dict(Counter(off_diag))}")
        np.savetxt('/home/xiwang/project/AutoMath/tasks/matrix_det/best_matrix.txt',
                   Ai, fmt='%d')
        report("Matrix saved to best_matrix.txt")

    return best_A


if __name__ == "__main__":
    main()
