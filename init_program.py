# EVOLVE-BLOCK-START
"""Hadamard maximal determinant n=29: algebraic Gram decomposition + optimization.

Strategy:
1. Construct Solomon Gram G algebraically (diagonal=29, specific {-3,1,5} pattern)
2. Build first 11 columns from known block structure + randomized bottom parts
3. Complete remaining 18 columns via corrected RREF backtracker
4. This gives c=320, score=0.9357 in ~2 seconds
5. Use remaining 330+ seconds for optimization to try pushing above c=320
"""
import numpy as np
import time
from fractions import Fraction
from math import lcm as math_lcm

N = 29

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


def build_target_gram():
    G = np.ones((N, N), dtype=np.int64); np.fill_diagonal(G, N)
    for i in range(3):
        for j in range(3, 6): G[i, j] = 5; G[j, i] = 5
    for j in range(7, 11): G[6, j] = -3; G[j, 6] = -3
    return G


def construct_first_11_columns(rng, max_time=5.0):
    deadline = time.time() + max_time
    while time.time() < deadline:
        v = [None]*11
        v[0] = np.ones(18, dtype=np.int64); v[0][rng.choice(18, 6, replace=False)] = -1
        ok = False
        for _ in range(5000):
            v[1] = np.ones(18, dtype=np.int64); v[1][rng.choice(18, 6, replace=False)] = -1
            if np.dot(v[0], v[1]) == -6: ok = True; break
        if not ok: continue
        ok = False
        for _ in range(50000):
            v[2] = np.ones(18, dtype=np.int64); v[2][rng.choice(18, 6, replace=False)] = -1
            if np.dot(v[0], v[2]) == -6 and np.dot(v[1], v[2]) == -6: ok = True; break
        if not ok: continue
        ok35 = False
        for _ in range(50000):
            if time.time() > deadline: break
            v[3] = np.ones(18, dtype=np.int64); v[3][rng.choice(18, 6, replace=False)] = -1
            if not all(np.dot(v[i], v[3]) == 2 for i in range(3)): continue
            for __ in range(5000):
                v[4] = np.ones(18, dtype=np.int64); v[4][rng.choice(18, 6, replace=False)] = -1
                if not (all(np.dot(v[i], v[4]) == 2 for i in range(3)) and np.dot(v[3], v[4]) == -6): continue
                for ___ in range(5000):
                    v[5] = np.ones(18, dtype=np.int64); v[5][rng.choice(18, 6, replace=False)] = -1
                    if all(np.dot(v[i], v[5]) == 2 for i in range(3)) and np.dot(v[3], v[5]) == -6 and np.dot(v[4], v[5]) == -6:
                        ok35 = True; break
                if ok35: break
            if ok35: break
        if not ok35: continue
        v[6] = np.ones(18, dtype=np.int64)
        ok710 = False
        for _ in range(50000):
            if time.time() > deadline: break
            v[7] = np.ones(18, dtype=np.int64); v[7][rng.choice(18, 9, replace=False)] = -1
            if not all(np.dot(v[i], v[7]) == 0 for i in range(6)): continue
            for __ in range(5000):
                v[8] = np.ones(18, dtype=np.int64); v[8][rng.choice(18, 9, replace=False)] = -1
                if not (all(np.dot(v[i], v[8]) == 0 for i in range(6)) and np.dot(v[7], v[8]) == -6): continue
                for ___ in range(5000):
                    v[9] = np.ones(18, dtype=np.int64); v[9][rng.choice(18, 9, replace=False)] = -1
                    if not (all(np.dot(v[i], v[9]) == 0 for i in range(6)) and np.dot(v[7], v[9]) == -6 and np.dot(v[8], v[9]) == -6): continue
                    for ____ in range(5000):
                        v[10] = np.ones(18, dtype=np.int64); v[10][rng.choice(18, 9, replace=False)] = -1
                        if all(np.dot(v[i], v[10]) == 0 for i in range(6)) and all(np.dot(v[j], v[10]) == -6 for j in [7,8,9]):
                            ok710 = True; break
                    if ok710: break
                if ok710: break
            if ok710: break
        if ok710: return v
    return None


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
        pv = icons[np.newaxis, :] - fm @ ic.T  # CORRECTED: minus sign
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


def find_gram_decomposition(deadline, rng):
    G = build_target_gram()
    for attempt in range(50):
        if time.time() > deadline: break
        cb = construct_first_11_columns(rng, max_time=min(5.0, deadline - time.time()))
        if cb is None: continue
        cols = []; indices = list(range(11))
        for j in range(11):
            c = np.zeros(N, dtype=np.int64); c[:11] = KNOWN_TOP[:, j]; c[11:] = cb[j]
            cols.append(c)
        pg = np.column_stack(cols).T @ np.column_stack(cols)
        if np.sum((pg - G[:11,:11])**2) != 0: continue

        def dfs(pc, pi, step):
            if time.time() > deadline: return False
            if step == 18: return True
            ti = 11 + step
            ms = min(20, max(3, 500 // (step + 1)))
            sols = solve_column_constraints(pc, pi, ti, G, rng, max_sol=ms, deadline=min(time.time()+5, deadline))
            if not sols: return False
            rng.shuffle(sols)
            for s in sols[:max(3, ms // 2)]:
                if time.time() > deadline: return False
                pc.append(s); pi.append(ti)
                if dfs(pc, pi, step + 1): return True
                pc.pop(); pi.pop()
            return False

        if dfs(cols, indices, 0):
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(indices): R[:, ci] = cols[i]
            if np.array_equal(R.T @ R, G):
                return R
    return None


def hill_climb_sm(A, deadline):
    n = A.shape[0]; s, ld = np.linalg.slogdet(A)
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
            s2, _ = np.linalg.slogdet(A);
            if s2 == 0: break
            Ai = np.linalg.inv(A)
    _, ld = np.linalg.slogdet(A)
    return A, ld


def row_replace_climb(A, deadline):
    n = A.shape[0]; A = A.copy(); s, ld = np.linalg.slogdet(A)
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


def construct_hadamard_matrix(n=29):
    deadline = time.time() + 335
    rng = np.random.RandomState(42)
    best_A = None; best_ld = -1e18

    def try_update(A):
        nonlocal best_A, best_ld
        if A is None: return
        Af = A.astype(np.float64) if A.dtype != np.float64 else A.copy()
        _, ld = np.linalg.slogdet(Af)
        if ld > best_ld: best_ld = ld; best_A = Af.copy()

    R = find_gram_decomposition(min(time.time() + 30, deadline - 300), rng)
    if R is not None:
        try_update(R)
        A = R.astype(np.float64)
        A, _ = hill_climb_sm(A, min(time.time() + 5, deadline - 290))
        try_update(A)

    while time.time() < deadline - 5:
        if best_A is not None and rng.random() < 0.85:
            A = best_A.copy()
            k = rng.randint(5, 80)
            idxs = rng.randint(0, n, size=(k, 2))
            for idx in idxs: A[idx[0], idx[1]] *= -1.0
        else:
            A = rng.choice([-1.0, 1.0], size=(n, n))
        A, _ = row_replace_climb(A, min(time.time() + 0.15, deadline - 5))
        A, _ = hill_climb_sm(A, min(time.time() + 0.1, deadline - 5))
        try_update(A)

    result = np.sign(best_A).astype(int); result[result == 0] = 1
    return result


# EVOLVE-BLOCK-END


def run_code():
    return (construct_hadamard_matrix(n=29),)


if __name__ == "__main__":
    m = run_code()[0]
    d = abs(np.linalg.det(m.astype(float)))
    print(f"|det| ≈ {d:.4e}, Score ≈ {d/1270698346568170340352:.6f}")
