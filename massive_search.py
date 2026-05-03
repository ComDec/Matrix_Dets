"""Massive long-running optimization search for 29x29 maxdet matrix.

Goal: Find a {-1,+1} matrix with det > 0.94 * THEORETICAL_MAX ≈ 1.194e21.
Current best: score 0.9357 (c=320, det ≈ 1.189e21).

Strategies:
1. Generate many different Solomon Gram decompositions (different random seeds)
2. For each decomposition, apply smart perturbation + optimization
3. Crossover between decompositions (mix rows)
4. Continuous search until breakthrough or timeout
"""

import numpy as np
import time
import sys
import os
from fractions import Fraction
from math import lcm as math_lcm
from collections import Counter

N = 29
THEORETICAL_MAX = 1270698346568170340352
TARGET_SCORE = 0.94
TARGET_DET = TARGET_SCORE * THEORETICAL_MAX  # ~1.194e21
BREAKTHROUGH_DET = 0.9357 * THEORETICAL_MAX  # anything above current best

# ============================================================================
# Import core functions from init_program.py
# ============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def det_bareiss(A):
    n = len(A)
    if n == 0:
        return 1
    M = [row.copy() for row in A]
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return M[-1][-1]


def get_exact_det(A):
    """Get exact integer determinant using Bareiss."""
    Ai = np.sign(A).astype(int)
    Ai[Ai == 0] = 1
    return abs(det_bareiss(Ai.tolist()))


def get_score(A):
    s, ld = np.linalg.slogdet(A.astype(float))
    if s == 0:
        return 0.0
    return np.exp(ld) / THEORETICAL_MAX


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
            s2, _ = np.linalg.slogdet(A)
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


def optimize_matrix(A, time_budget=0.3):
    """Row replace + hill climb."""
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10: return A, ld
    dl = time.time() + time_budget * 0.6
    A, _ = row_replace_climb(A, dl)
    dl = time.time() + time_budget * 0.4
    A, ld = hill_climb_sm(A, dl)
    return A, ld


# ============================================================================
# STRATEGY 1: Generate many Gram decompositions and smart-perturb each
# ============================================================================

def compute_flip_ratios(A):
    """For each entry, compute ratio of det change if we flip it.
    Returns matrix where ratio[i,j] = |det(A_flipped)| / |det(A)|.
    Uses Sherman-Morrison: flipping A[i,j] multiplies det by |1 - 2*A[i,j]*Ainv[j,i]|.
    """
    Ai = np.linalg.inv(A.astype(float))
    ratios = np.abs(1.0 - 2.0 * A * Ai.T)
    return ratios


def smart_perturb(A, rng, n_flips=2):
    """Flip entries that are closest to being det-neutral (ratio ~1),
    hoping to break out of current basin without destroying too much structure."""
    Af = A.astype(float)
    ratios = compute_flip_ratios(Af)
    # Sort by closeness to ratio=1 (entries where flipping barely changes det)
    flat_ratios = ratios.flatten()
    # Filter: only consider entries where ratio > 0.7 (not too destructive)
    mask = flat_ratios > 0.5
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        candidates = np.arange(N*N)

    # Sort by closeness to 1.0
    dists = np.abs(flat_ratios[candidates] - 1.0)
    sorted_idx = np.argsort(dists)

    # Pick n_flips entries near ratio=1 (with some randomness)
    B = A.copy()
    n_cand = min(len(sorted_idx), max(n_flips * 10, 50))
    chosen = rng.choice(n_cand, size=min(n_flips, n_cand), replace=False)
    for c in chosen:
        flat_idx = candidates[sorted_idx[c]]
        i, j = flat_idx // N, flat_idx % N
        B[i, j] *= -1
    return B


def aggressive_perturb(A, rng, n_flips=3):
    """Flip entries where ratio is SLIGHTLY above 1 (det-increasing flips exist
    but hill climb missed them due to greedy order). Then also flip some near-1."""
    Af = A.astype(float)
    ratios = compute_flip_ratios(Af)

    B = A.copy()
    flat = ratios.flatten()

    # First: flip entries with ratio just above 1 (these improve det)
    improving = np.where(flat > 1.0 + 1e-12)[0]
    if len(improving) > 0:
        # This shouldn't happen if hill_climb finished, but floating point...
        for idx in improving[:2]:
            i, j = idx // N, idx % N
            B[i, j] *= -1
        return B

    # Otherwise: targeted multi-flip
    # Pick pairs of entries in same row where combined flip might help
    row = rng.randint(N)
    row_ratios = ratios[row, :]
    # Sort columns by ratio descending within this row
    col_order = np.argsort(-row_ratios)
    # Flip top n_flips entries in this row
    for c in col_order[:n_flips]:
        B[row, c] *= -1
    return B


def row_swap_perturb(A, rng):
    """Swap two rows and re-optimize. Can escape local optima."""
    B = A.copy()
    i, j = rng.choice(N, size=2, replace=False)
    B[[i, j]] = B[[j, i]]
    return B


def column_replace_perturb(A, rng):
    """Replace a column with a random +-1 column. Violent but can escape basins."""
    B = A.copy()
    j = rng.randint(N)
    B[:, j] = rng.choice([-1, 1], size=N)
    return B


def crossover(A1, A2, rng):
    """Mix rows from two matrices."""
    B = A1.copy()
    # Take some rows from A2
    n_rows = rng.randint(3, N // 2)
    rows = rng.choice(N, size=n_rows, replace=False)
    for r in rows:
        B[r] = A2[r].copy()
    return B


def block_swap(A1, A2, rng):
    """Swap a rectangular block between two matrices."""
    B = A1.copy()
    r1 = rng.randint(0, N - 3)
    c1 = rng.randint(0, N - 3)
    h = rng.randint(3, min(N - r1, 12))
    w = rng.randint(3, min(N - c1, 12))
    B[r1:r1+h, c1:c1+w] = A2[r1:r1+h, c1:c1+w]
    return B


# ============================================================================
# STRATEGY 2: Exhaustive k-flip neighborhood search
# ============================================================================

def exhaustive_2flip(A, deadline):
    """Try ALL pairs of flips. If any improves det, take it."""
    Af = A.astype(float)
    s, ld = np.linalg.slogdet(Af)
    if s == 0:
        return A, False
    Ai = np.linalg.inv(Af)
    best_ratio = 1.0
    best_ij = None

    n = N
    # For 2-flip: flip (i1,j1) and (i2,j2)
    # det ratio = det(A + d1*e_i1*e_j1^T + d2*e_i2*e_j2^T) / det(A)
    # = det(I + Ainv @ (d1*e_i1*e_j1^T + d2*e_i2*e_j2^T))
    # For same row: can use rank-2 update

    # Simpler: just iterate over all pairs of entries in same row
    # Flipping A[r,c1] and A[r,c2]:
    # d1 = -2*A[r,c1], d2 = -2*A[r,c2]
    # det ratio = 1 + d1*Ai[c1,r] + d2*Ai[c2,r] + d1*d2*(Ai[c1,r]*Ai[c2,r] - Ai[c2,r]*Ai[c1,r])
    # ... actually for same-row 2-flip, need proper rank-2 formula

    # Let's just do it entry by entry with the fast Sherman-Morrison check
    # For each entry (i,j), compute det ratio if flipped
    single_ratios = 1.0 - 2.0 * Af * Ai.T  # ratio[i,j] = det(A_flip) / det(A)

    # For 2-flip: ratio = r1 * r2 approximately (if the two flips are "independent")
    # More precisely, need rank-2 update
    # But as approximation, look for pairs where r1 * r2 > 1

    # Better: look for pairs in same row
    count = 0
    for row in range(n):
        if time.time() > deadline:
            break
        for c1 in range(n):
            if time.time() > deadline:
                break
            d1 = -2.0 * Af[row, c1]
            dn1 = 1.0 + d1 * Ai[c1, row]
            if abs(dn1) < 1e-15:
                continue
            # After first flip, new inverse column changes
            # Ai_new = Ai - (d1/dn1) * Ai[:,c1] * Ai[row,:]
            # For second flip at (row, c2):
            # d2 = -2 * A_new[row, c2] = -2 * Af[row, c2] (if c2 != c1) or 2*Af[row,c1] (if c2==c1, undo)
            # dn2_new = 1 + d2 * Ai_new[c2, row]
            # Ai_new[c2, row] = Ai[c2, row] - (d1/dn1) * Ai[c1, row] * Ai[row, row] ... no wait
            # Ai_new[c2, row] = Ai[c2, row] - (d1/dn1) * Ai[c2, row] ... hmm, need to think
            #
            # Sherman-Morrison: Ai_new = Ai - (d1/dn1) * outer(Ai[:,c1], Ai[row,:])
            # So Ai_new[c2, row] = Ai[c2, row] - (d1/dn1) * Ai[c2, row] ... no
            # Ai_new[alpha, beta] = Ai[alpha, beta] - (d1/dn1) * Ai[alpha, c1] * Ai[row, beta]
            # So Ai_new[c2, row] = Ai[c2, row] - (d1/dn1) * Ai[c2, c1] * Ai[row, row]
            # Hmm, this is getting complicated. Let me just use the matrix determinant lemma for rank-2.

            # For rank-2 update: det(A + UV^T) = det(A) * det(I + V^T A^{-1} U)
            # U = [d1*e_{row}, d2*e_{row}] (29x2), V = [e_{c1}, e_{c2}] (29x2)
            # But this only works for different columns in same row
            for c2 in range(c1 + 1, n):
                count += 1
                d2 = -2.0 * Af[row, c2]
                # 2x2 matrix: M = I + V^T Ai U
                # V^T Ai = [[Ai[c1,:]], [Ai[c2,:]]]
                # V^T Ai U = [[Ai[c1,row]*d1, Ai[c1,row]*d2], [Ai[c2,row]*d1, Ai[c2,row]*d2]]
                # Wait: U = d1*e_row for col1, d2*e_row for col2
                # So V^T Ai U = [[d1*Ai[c1,row], d2*Ai[c1,row]], [d1*Ai[c2,row], d2*Ai[c2,row]]]
                # det(I + that) = (1+d1*Ai[c1,row])*(1+d2*Ai[c2,row]) - d2*Ai[c1,row]*d1*Ai[c2,row]
                m11 = 1.0 + d1 * Ai[c1, row]
                m12 = d2 * Ai[c1, row]
                m21 = d1 * Ai[c2, row]
                m22 = 1.0 + d2 * Ai[c2, row]
                det_ratio = m11 * m22 - m12 * m21

                if abs(det_ratio) > best_ratio + 1e-12:
                    best_ratio = abs(det_ratio)
                    best_ij = (row, c1, c2, det_ratio)

    if best_ij is not None and best_ratio > 1.0 + 1e-12:
        row, c1, c2, dr = best_ij
        B = A.copy()
        B[row, c1] *= -1
        B[row, c2] *= -1
        return B, True

    return A, False


def exhaustive_2flip_crossrow(A, deadline, max_pairs=50000):
    """Try pairs of flips across different rows."""
    Af = A.astype(float)
    s, ld = np.linalg.slogdet(Af)
    if s == 0:
        return A, False
    Ai = np.linalg.inv(Af)

    single_ratios = np.abs(1.0 - 2.0 * Af * Ai.T)

    # Find entries closest to ratio=1 (near-neutral flips)
    flat = single_ratios.flatten()
    near_one = np.argsort(np.abs(flat - 1.0))

    best_ratio = 1.0
    best_pair = None

    n_check = min(200, N*N)  # top 200 near-neutral entries
    candidates = near_one[:n_check]

    count = 0
    for idx1 in range(len(candidates)):
        if time.time() > deadline:
            break
        e1 = candidates[idx1]
        r1, c1 = e1 // N, e1 % N
        d1 = -2.0 * Af[r1, c1]

        for idx2 in range(idx1 + 1, len(candidates)):
            count += 1
            if count > max_pairs:
                break
            if time.time() > deadline:
                break
            e2 = candidates[idx2]
            r2, c2 = e2 // N, e2 % N
            d2 = -2.0 * Af[r2, c2]

            # Rank-2 update: det ratio
            # U = [[d1, 0], [0, d2]] applied to rows r1, r2
            # But more precisely:
            # flip1: e_{r1} e_{c1}^T * d1
            # flip2: e_{r2} e_{c2}^T * d2
            # det(I + V^T Ai U) where U cols are d1*e_{r1} and d2*e_{r2}, V cols are e_{c1} and e_{c2}
            m11 = 1.0 + d1 * Ai[c1, r1]
            m12 = d2 * Ai[c1, r2]
            m21 = d1 * Ai[c2, r1]
            m22 = 1.0 + d2 * Ai[c2, r2]
            det_ratio = abs(m11 * m22 - m12 * m21)

            if det_ratio > best_ratio + 1e-12:
                best_ratio = det_ratio
                best_pair = (r1, c1, r2, c2)
        if count > max_pairs:
            break

    if best_pair is not None and best_ratio > 1.0 + 1e-12:
        r1, c1, r2, c2 = best_pair
        B = A.copy()
        B[r1, c1] *= -1
        B[r2, c2] *= -1
        return B, True

    return A, False


# ============================================================================
# MAIN SEARCH LOOP
# ============================================================================

def save_breakthrough(A, score, det_val, path="/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy"):
    """Save matrix that exceeds current best."""
    M = np.sign(A).astype(int)
    M[M == 0] = 1
    np.save(path, M)
    print(f"\n{'='*70}")
    print(f"!!! BREAKTHROUGH SAVED to {path} !!!")
    print(f"Score: {score:.8f}, |det| = {det_val:.6e}")
    print(f"{'='*70}\n")
    sys.stdout.flush()


def main():
    start_time = time.time()
    TOTAL_TIME = 3 * 3600  # 3 hours max
    deadline = start_time + TOTAL_TIME

    print(f"{'='*70}")
    print(f"MASSIVE SEARCH: 29x29 maxdet matrix")
    print(f"Target: score > 0.94 (det > {TARGET_DET:.4e})")
    print(f"Current best: 0.9357 (det ~ 1.189e21)")
    print(f"Time limit: {TOTAL_TIME/3600:.1f} hours")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Global tracking
    best_score = 0.0
    best_matrix = None
    decompositions = []  # list of (matrix, score) from Gram decompositions
    all_good_matrices = []  # matrices with score > 0.92

    last_report = start_time
    total_decomps_found = 0
    total_perturbs_tried = 0
    total_crossovers_tried = 0
    total_2flips_tried = 0

    def report_progress():
        nonlocal last_report
        now = time.time()
        if now - last_report < 60:
            return
        last_report = now
        elapsed = now - start_time
        print(f"\n--- Progress at {elapsed:.0f}s ({elapsed/60:.1f}m) ---")
        print(f"  Best score: {best_score:.8f}")
        print(f"  Decompositions found: {total_decomps_found}")
        print(f"  Perturbations tried: {total_perturbs_tried}")
        print(f"  Crossovers tried: {total_crossovers_tried}")
        print(f"  2-flip searches: {total_2flips_tried}")
        print(f"  Good matrices (>0.92): {len(all_good_matrices)}")
        sys.stdout.flush()

    def check_and_update(A, source="unknown"):
        nonlocal best_score, best_matrix
        if A is None:
            return 0.0
        Af = A.astype(float) if A.dtype != np.float64 else A.copy()
        sc = get_score(Af)
        if sc > 0.92:
            M = np.sign(Af).astype(int)
            M[M == 0] = 1
            all_good_matrices.append((M.copy(), sc))
        if sc > best_score + 1e-8:
            best_score = sc
            best_matrix = np.sign(Af).astype(int)
            best_matrix[best_matrix == 0] = 1
            print(f"  NEW BEST: {sc:.8f} (source: {source})")
            sys.stdout.flush()

            if sc > 0.9357 + 1e-6:
                # Verify with exact determinant
                exact_det = get_exact_det(Af)
                exact_score = exact_det / THEORETICAL_MAX
                print(f"  EXACT CHECK: |det| = {exact_det}, exact score = {exact_score:.8f}")
                if exact_score > 0.9357:
                    save_breakthrough(best_matrix, exact_score, exact_det)
                sys.stdout.flush()
        return sc

    # =========================================================================
    # PHASE 1: Generate many Gram decompositions
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Generating Gram decompositions")
    print(f"{'='*70}")
    sys.stdout.flush()

    phase1_deadline = min(start_time + 1800, deadline)  # up to 30 min for decompositions
    seed_counter = 0

    while time.time() < phase1_deadline and total_decomps_found < 200:
        seed_counter += 1
        rng = np.random.RandomState(seed_counter * 7 + 13)

        t0 = time.time()
        per_attempt_deadline = min(t0 + 30, phase1_deadline)

        R = find_gram_decomposition(per_attempt_deadline, rng)

        if R is not None:
            total_decomps_found += 1
            sc = check_and_update(R, f"gram_decomp_seed{seed_counter}")
            if sc > 0.92:
                decompositions.append((R.copy(), sc))
                print(f"  Decomposition #{total_decomps_found} (seed={seed_counter}): score={sc:.8f}")
                sys.stdout.flush()

            # Immediately try optimization on this decomposition
            A = R.astype(float)
            A, _ = row_replace_climb(A, min(time.time() + 0.5, phase1_deadline))
            A, _ = hill_climb_sm(A, min(time.time() + 0.5, phase1_deadline))
            sc2 = check_and_update(A, f"gram_optimized_seed{seed_counter}")

        report_progress()

        # Also do some perturbation+optimization in between decomposition attempts
        if len(decompositions) > 0 and seed_counter % 3 == 0:
            # Pick a random decomposition and perturb
            idx = np.random.randint(len(decompositions))
            base_R, base_sc = decompositions[idx]

            for n_flips in [2, 3, 4, 5]:
                if time.time() > phase1_deadline:
                    break
                for _ in range(20):
                    total_perturbs_tried += 1
                    B = smart_perturb(base_R, np.random.RandomState(total_perturbs_tried), n_flips=n_flips)
                    Bf = B.astype(float)
                    Bf, _ = row_replace_climb(Bf, time.time() + 0.2)
                    Bf, _ = hill_climb_sm(Bf, time.time() + 0.15)
                    check_and_update(Bf, f"smart_perturb_{n_flips}flip")

    print(f"\nPhase 1 complete: {total_decomps_found} decompositions, {len(decompositions)} good ones")
    sys.stdout.flush()

    # =========================================================================
    # PHASE 2: Aggressive perturbation + optimization on all decompositions
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Aggressive perturbation + optimization")
    print(f"{'='*70}")
    sys.stdout.flush()

    phase2_deadline = min(start_time + 3600, deadline)  # up to 1 hour total

    while time.time() < phase2_deadline:
        report_progress()

        if len(decompositions) == 0:
            # No decompositions found, try random + optimize
            rng = np.random.RandomState(total_perturbs_tried)
            A = rng.choice([-1.0, 1.0], size=(N, N))
            A, _ = row_replace_climb(A, time.time() + 0.15)
            A, _ = hill_climb_sm(A, time.time() + 0.1)
            total_perturbs_tried += 1
            check_and_update(A, "random_init")
            continue

        # Pick a base matrix
        if len(all_good_matrices) > 0 and np.random.random() < 0.3:
            idx = np.random.randint(len(all_good_matrices))
            base, _ = all_good_matrices[idx]
        else:
            idx = np.random.randint(len(decompositions))
            base, _ = decompositions[idx]

        strategy = np.random.randint(8)

        if strategy == 0:
            # Smart perturbation (2-5 flips near ratio=1)
            n_flips = np.random.randint(2, 6)
            B = smart_perturb(base, np.random.RandomState(total_perturbs_tried), n_flips=n_flips)
            total_perturbs_tried += 1

        elif strategy == 1:
            # Aggressive row-based perturbation
            B = aggressive_perturb(base, np.random.RandomState(total_perturbs_tried), n_flips=np.random.randint(2, 5))
            total_perturbs_tried += 1

        elif strategy == 2:
            # Row swap
            B = row_swap_perturb(base, np.random.RandomState(total_perturbs_tried))
            total_perturbs_tried += 1

        elif strategy == 3:
            # Random flips (varying intensity)
            B = base.copy()
            k = np.random.randint(3, 40)
            rng_p = np.random.RandomState(total_perturbs_tried)
            idxs = rng_p.randint(0, N, size=(k, 2))
            for ij in idxs:
                B[ij[0], ij[1]] *= -1
            total_perturbs_tried += 1

        elif strategy == 4:
            # Crossover between two decompositions
            if len(decompositions) >= 2:
                i1, i2 = np.random.choice(len(decompositions), size=2, replace=False)
                B = crossover(decompositions[i1][0], decompositions[i2][0],
                             np.random.RandomState(total_crossovers_tried))
                total_crossovers_tried += 1
            else:
                continue

        elif strategy == 5:
            # Block swap between decompositions
            if len(decompositions) >= 2:
                i1, i2 = np.random.choice(len(decompositions), size=2, replace=False)
                B = block_swap(decompositions[i1][0], decompositions[i2][0],
                              np.random.RandomState(total_crossovers_tried))
                total_crossovers_tried += 1
            else:
                continue

        elif strategy == 6:
            # Column replacement
            B = column_replace_perturb(base, np.random.RandomState(total_perturbs_tried))
            total_perturbs_tried += 1

        elif strategy == 7:
            # Crossover with a good matrix
            if len(all_good_matrices) >= 2:
                i1 = np.random.randint(len(all_good_matrices))
                i2 = np.random.randint(len(decompositions)) if len(decompositions) > 0 else np.random.randint(len(all_good_matrices))
                A1 = all_good_matrices[i1][0]
                A2 = decompositions[i2][0] if len(decompositions) > 0 else all_good_matrices[min(i2, len(all_good_matrices)-1)][0]
                B = crossover(A1, A2, np.random.RandomState(total_crossovers_tried))
                total_crossovers_tried += 1
            else:
                continue

        # Optimize the perturbed matrix
        Bf = B.astype(float)
        s_check, _ = np.linalg.slogdet(Bf)
        if s_check != 0:
            Bf, _ = row_replace_climb(Bf, time.time() + 0.2)
            Bf, _ = hill_climb_sm(Bf, time.time() + 0.15)
            check_and_update(Bf, f"strategy_{strategy}")

    # =========================================================================
    # PHASE 3: Exhaustive 2-flip search on best matrices
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Exhaustive 2-flip neighborhood search")
    print(f"{'='*70}")
    sys.stdout.flush()

    phase3_deadline = min(start_time + 5400, deadline)  # up to 1.5 hours total

    # Collect unique good matrices
    unique_matrices = []
    seen_hashes = set()
    for M, sc in sorted(all_good_matrices + [(d[0], d[1]) for d in decompositions],
                         key=lambda x: -x[1]):
        Mi = np.sign(M).astype(int)
        Mi[Mi == 0] = 1
        h = hash(Mi.tobytes())
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_matrices.append((Mi, sc))

    print(f"Unique good matrices to search: {len(unique_matrices)}")
    sys.stdout.flush()

    for mat_idx, (M, sc) in enumerate(unique_matrices):
        if time.time() > phase3_deadline:
            break

        print(f"\n  2-flip search on matrix {mat_idx+1}/{len(unique_matrices)} (score={sc:.8f})")
        sys.stdout.flush()

        # Same-row 2-flip
        B, improved = exhaustive_2flip(M, min(time.time() + 30, phase3_deadline))
        total_2flips_tried += 1
        if improved:
            Bf = B.astype(float)
            Bf, _ = hill_climb_sm(Bf, time.time() + 1.0)
            new_sc = check_and_update(Bf, f"2flip_samerow_mat{mat_idx}")
            if new_sc > sc + 1e-6:
                print(f"  2-FLIP IMPROVEMENT: {sc:.8f} -> {new_sc:.8f}")
                # If improved, do another round
                B2, improved2 = exhaustive_2flip(np.sign(Bf).astype(int), min(time.time() + 30, phase3_deadline))
                if improved2:
                    B2f = B2.astype(float)
                    B2f, _ = hill_climb_sm(B2f, time.time() + 1.0)
                    check_and_update(B2f, f"2flip_chain_mat{mat_idx}")

        # Cross-row 2-flip
        B, improved = exhaustive_2flip_crossrow(M, min(time.time() + 30, phase3_deadline))
        total_2flips_tried += 1
        if improved:
            Bf = B.astype(float)
            Bf, _ = hill_climb_sm(Bf, time.time() + 1.0)
            new_sc = check_and_update(Bf, f"2flip_crossrow_mat{mat_idx}")
            if new_sc > sc + 1e-6:
                print(f"  CROSS-ROW 2-FLIP IMPROVEMENT: {sc:.8f} -> {new_sc:.8f}")

        report_progress()

    # =========================================================================
    # PHASE 4: Continued random search with best matrices as seeds
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: Continued random search")
    print(f"{'='*70}")
    sys.stdout.flush()

    while time.time() < deadline:
        report_progress()

        # More decomposition generation
        if np.random.random() < 0.2:
            seed_counter += 1
            rng = np.random.RandomState(seed_counter * 7 + 13)
            R = find_gram_decomposition(min(time.time() + 20, deadline), rng)
            if R is not None:
                total_decomps_found += 1
                sc = check_and_update(R, f"gram_decomp_phase4_seed{seed_counter}")
                if sc > 0.92:
                    decompositions.append((R.copy(), sc))
                # Optimize
                A = R.astype(float)
                A, _ = row_replace_climb(A, min(time.time() + 0.5, deadline))
                A, _ = hill_climb_sm(A, min(time.time() + 0.5, deadline))
                check_and_update(A, f"gram_opt_phase4_seed{seed_counter}")
            continue

        # Pick base and perturb
        if best_matrix is not None and np.random.random() < 0.5:
            base = best_matrix.copy()
        elif len(all_good_matrices) > 0:
            idx = np.random.randint(len(all_good_matrices))
            base = all_good_matrices[idx][0].copy()
        elif len(decompositions) > 0:
            idx = np.random.randint(len(decompositions))
            base = decompositions[idx][0].copy()
        else:
            base = np.random.choice([-1, 1], size=(N, N)).astype(int)

        # Vary perturbation intensity
        k = np.random.choice([2, 3, 5, 8, 12, 20, 40, 80])
        B = base.copy()
        rng_p = np.random.RandomState(total_perturbs_tried + 999999)
        idxs = rng_p.randint(0, N, size=(k, 2))
        for ij in idxs:
            B[ij[0], ij[1]] *= -1
        total_perturbs_tried += 1

        Bf = B.astype(float)
        Bf, _ = row_replace_climb(Bf, time.time() + 0.15)
        Bf, _ = hill_climb_sm(Bf, time.time() + 0.1)
        check_and_update(Bf, f"phase4_k{k}")

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE after {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*70}")
    print(f"Best score: {best_score:.8f}")
    print(f"Decompositions found: {total_decomps_found}")
    print(f"Perturbations tried: {total_perturbs_tried}")
    print(f"Crossovers tried: {total_crossovers_tried}")
    print(f"2-flip searches: {total_2flips_tried}")
    print(f"Good matrices (>0.92): {len(all_good_matrices)}")

    if best_matrix is not None:
        exact_det = get_exact_det(best_matrix.astype(float))
        exact_score = exact_det / THEORETICAL_MAX
        print(f"Exact final score: {exact_score:.8f}")
        print(f"Exact |det|: {exact_det}")

        # Save best matrix regardless
        save_path = "/home/xiwang/project/AutoMath/tasks/matrix_det/best_search_matrix.npy"
        np.save(save_path, best_matrix)
        print(f"Best matrix saved to {save_path}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
