"""Targeted modification of OS matrix to change specific Gram entries.

The OS matrix R has R^T R = G_OS. We want to find R' with R'^T R' = G'
where G' differs from G_OS by a few entries and has higher det.

Strategy: Start from the OS matrix. To change G[i,j] from 5 to 3,
we need to change columns i and j of R such that their inner product
decreases by 2, while maintaining all other inner product constraints.

This is a constrained combinatorial optimization:
- Fix most columns of R
- Re-solve columns i and j subject to the new constraints
- Use column backtracking on the modified columns
"""

import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm

N = 29
THEORETICAL_MAX = 1270698346568170340352

# The known OS matrix (from fast_decompose.py)
OS_MATRIX = np.array([
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


def build_os_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for i in range(3):
        for j in range(3, 6): G[i, j] = 5; G[j, i] = 5
    for j in range(7, 11): G[6, j] = -3; G[j, 6] = -3
    return G


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


def try_modify_columns(R, G_target, cols_to_modify, time_limit=60, rng=None):
    """Modify specific columns of R to satisfy a new Gram matrix G_target.

    Fix all columns except cols_to_modify.
    Use column backtracking on the modified columns.
    """
    if rng is None:
        rng = np.random.RandomState()

    deadline = time.time() + time_limit
    n = N
    fixed_cols = [j for j in range(n) if j not in cols_to_modify]

    placed_cols = [R[:, j].copy() for j in fixed_cols]
    placed_indices = list(fixed_cols)

    # Now backtrack on cols_to_modify
    modify_list = list(cols_to_modify)
    rng.shuffle(modify_list)

    def backtrack(step):
        if time.time() > deadline:
            return False
        if step == len(modify_list):
            return True

        target_idx = modify_list[step]
        ms = 500  # max solutions to try

        solutions = solve_column_constraints(
            placed_cols, placed_indices, target_idx, G_target, rng,
            max_sol=ms, deadline=min(time.time() + 10, deadline)
        )

        if not solutions:
            return False

        rng.shuffle(solutions)
        for s in solutions[:min(ms, 200)]:
            if time.time() > deadline:
                return False
            placed_cols.append(s)
            placed_indices.append(target_idx)
            if backtrack(step + 1):
                return True
            placed_cols.pop()
            placed_indices.pop()

        return False

    for attempt in range(50):
        if time.time() > deadline:
            break

        # Reset to fixed cols only
        while len(placed_cols) > len(fixed_cols):
            placed_cols.pop()
            placed_indices.pop()

        rng.shuffle(modify_list)

        if backtrack(0):
            # Success! Reconstruct R
            R_new = np.zeros((n, n), dtype=np.int64)
            for i, ci in enumerate(placed_indices):
                R_new[:, ci] = placed_cols[i]
            if np.array_equal(R_new.T @ R_new, G_target):
                return R_new

    return None


def main():
    print("=" * 70)
    print("TARGETED GRAM MODIFICATION SEARCH")
    print("=" * 70)
    sys.stdout.flush()

    # Verify OS matrix
    G_os = build_os_gram()
    R = OS_MATRIX.copy()
    G_check = R.T @ R
    assert np.array_equal(G_check, G_os), "OS matrix does not match OS Gram!"
    base_det = abs(det_bareiss(R.astype(int).tolist()))
    print(f"OS |det| = {base_det}, score = {base_det/THEORETICAL_MAX:.8f}")
    sys.stdout.flush()

    # Strategy: For each single Gram entry change that increases det(G),
    # try to find a modified R matrix.

    # Enumerate all possible single-entry changes
    best_score = base_det / THEORETICAL_MAX
    best_matrix = R.copy()

    print(f"\n{'='*70}")
    print("Phase 1: Single Gram entry modifications")
    print(f"{'='*70}")
    sys.stdout.flush()

    allowed_values = [-5, -3, -1, 1, 3, 5]
    modifications = []

    for i in range(N):
        for j in range(i+1, N):
            old_val = int(G_os[i, j])
            for new_val in allowed_values:
                if new_val == old_val:
                    continue
                G_new = G_os.copy()
                G_new[i, j] = new_val
                G_new[j, i] = new_val

                # Check PSD
                eigvals = np.linalg.eigvalsh(G_new.astype(float))
                if np.min(eigvals) < -0.01:
                    continue

                det_new = det_bareiss(G_new.tolist())
                if abs(det_new) > abs(det_bareiss(G_os.tolist())):
                    score = np.sqrt(float(abs(det_new))) / THEORETICAL_MAX
                    modifications.append((i, j, old_val, new_val, abs(det_new), score))

    modifications.sort(key=lambda x: -x[4])
    print(f"Found {len(modifications)} single-entry changes that increase det(G):")
    for i, j, old, new, det_val, score in modifications[:20]:
        print(f"  G[{i},{j}]: {old} -> {new}, Gram score = {score:.8f}")
    sys.stdout.flush()

    # Try decomposition for each modification
    deadline = time.time() + 3600  # 1 hour

    for mod_idx, (i, j, old_val, new_val, det_val, gram_score) in enumerate(modifications):
        if time.time() > deadline:
            break

        G_target = G_os.copy()
        G_target[i, j] = new_val
        G_target[j, i] = new_val

        # The modified columns are i and j
        # Strategy 1: Only modify columns i and j (fix all others)
        print(f"\n  [{mod_idx+1}/{len(modifications)}] G[{i},{j}]: {old_val} -> {new_val}, "
              f"target score = {gram_score:.8f}")
        sys.stdout.flush()

        rng = np.random.RandomState(mod_idx * 37 + 7)

        # Try modifying just 2 columns
        time_per = min(120, max(30, (deadline - time.time()) / max(len(modifications) - mod_idx, 1)))
        R_new = try_modify_columns(R, G_target, [i, j], time_limit=time_per, rng=rng)

        if R_new is not None:
            actual_det = abs(det_bareiss(R_new.astype(int).tolist()))
            actual_score = actual_det / THEORETICAL_MAX
            print(f"    DECOMPOSITION FOUND!")
            print(f"    |det(R)| = {actual_det}, score = {actual_score:.8f}")

            if actual_score > best_score:
                best_score = actual_score
                best_matrix = R_new.copy()

                if actual_score > 0.9357 + 1e-6:
                    print(f"\n{'='*70}")
                    print(f"!!! BREAKTHROUGH: score = {actual_score:.8f} !!!")
                    print(f"{'='*70}")
                    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy",
                            R_new.astype(int))
                    print("Saved!")
            sys.stdout.flush()
        else:
            print(f"    Could not decompose (modifying cols {i},{j})")

            # Strategy 2: Also modify neighboring columns
            # If G[i,j] changes, also try modifying columns that interact with i and j
            neighbors = set()
            for k in range(N):
                if k != i and k != j:
                    if G_os[i, k] != 1 or G_os[j, k] != 1:
                        neighbors.add(k)
            if len(neighbors) > 0:
                extra_cols = list(neighbors)[:3]  # limit to 3 extra columns
                cols_mod = [i, j] + extra_cols
                print(f"    Trying with extra cols {extra_cols}...")
                sys.stdout.flush()
                R_new2 = try_modify_columns(R, G_target, cols_mod,
                                           time_limit=min(60, time_per/2), rng=rng)
                if R_new2 is not None:
                    actual_det = abs(det_bareiss(R_new2.astype(int).tolist()))
                    actual_score = actual_det / THEORETICAL_MAX
                    print(f"    DECOMPOSITION FOUND (with neighbors)!")
                    print(f"    |det(R)| = {actual_det}, score = {actual_score:.8f}")
                    if actual_score > best_score:
                        best_score = actual_score
                        best_matrix = R_new2.copy()
                        if actual_score > 0.9357 + 1e-6:
                            print(f"\n!!! BREAKTHROUGH !!!")
                            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy",
                                    R_new2.astype(int))
                    sys.stdout.flush()
                else:
                    print(f"    Still could not decompose")

    # Phase 2: Double modifications
    if time.time() < deadline:
        print(f"\n{'='*70}")
        print("Phase 2: Double Gram entry modifications")
        print(f"{'='*70}")
        sys.stdout.flush()

        # Take top single modifications and try combining them
        top_mods = modifications[:10]
        for m1_idx in range(len(top_mods)):
            if time.time() > deadline:
                break
            i1, j1, old1, new1, _, _ = top_mods[m1_idx]
            for m2_idx in range(m1_idx + 1, len(top_mods)):
                if time.time() > deadline:
                    break
                i2, j2, old2, new2, _, _ = top_mods[m2_idx]

                G_target = G_os.copy()
                G_target[i1, j1] = new1; G_target[j1, i1] = new1
                G_target[i2, j2] = new2; G_target[j2, i2] = new2

                eigvals = np.linalg.eigvalsh(G_target.astype(float))
                if np.min(eigvals) < -0.01:
                    continue

                det_val = det_bareiss(G_target.tolist())
                if abs(det_val) <= abs(det_bareiss(G_os.tolist())):
                    continue

                gram_score = np.sqrt(float(abs(det_val))) / THEORETICAL_MAX
                if gram_score <= best_score + 0.001:
                    continue

                print(f"\n  Double mod: G[{i1},{j1}]={new1}, G[{i2},{j2}]={new2}, "
                      f"score = {gram_score:.8f}")
                sys.stdout.flush()

                cols_mod = list(set([i1, j1, i2, j2]))
                rng = np.random.RandomState(m1_idx * 100 + m2_idx)
                R_new = try_modify_columns(R, G_target, cols_mod,
                                          time_limit=60, rng=rng)
                if R_new is not None:
                    actual_det = abs(det_bareiss(R_new.astype(int).tolist()))
                    actual_score = actual_det / THEORETICAL_MAX
                    print(f"    DECOMPOSITION FOUND!")
                    print(f"    |det| = {actual_det}, score = {actual_score:.8f}")
                    if actual_score > best_score:
                        best_score = actual_score
                        best_matrix = R_new.copy()
                        if actual_score > 0.9357 + 1e-6:
                            print(f"\n!!! BREAKTHROUGH !!!")
                            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy",
                                    R_new.astype(int))
                    sys.stdout.flush()

    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"Best score: {best_score:.8f}")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
