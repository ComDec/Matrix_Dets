"""
Systematic decomposition search across all 18 Gram variants.

Each variant has:
- Diagonal: 29
- Two specific entries = 9 (and symmetric) from {0,1,2}x{3,4,5}
- G[6,j]=-3 for j in {7,8,9,10}
- All other off-diagonal: 1

The two "9" pairs must be in different rows AND different columns of the 3x3 sub-block.
"""
import numpy as np
import time
import sys
import itertools
from fractions import Fraction
from math import lcm as math_lcm

N = 29
THEORETICAL_MAX = 1270698346568170340352


def enumerate_variants():
    """Enumerate all 18 variants.
    Each variant: two (i,j) pairs in {0,1,2}x{3,4,5}, different rows, different columns."""
    rows = [0, 1, 2]
    cols = [3, 4, 5]
    variants = []
    # Choose 2 rows from 3, choose 2 cols from 3, then match them (2 ways)
    for r_pair in itertools.combinations(rows, 2):
        for c_pair in itertools.combinations(cols, 2):
            # Two matchings: (r0,c0),(r1,c1) and (r0,c1),(r1,c0)
            variants.append(((r_pair[0], c_pair[0]), (r_pair[1], c_pair[1])))
            variants.append(((r_pair[0], c_pair[1]), (r_pair[1], c_pair[0])))
    return variants


def build_gram(nine_pairs):
    """Build the Gram matrix for a given variant."""
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for (i, j) in nine_pairs:
        G[i, j] = 9
        G[j, i] = 9
    for j in range(7, 11):
        G[6, j] = -3
        G[j, 6] = -3
    return G


def verify_gram(G):
    """Verify Gram is positive definite and compute det."""
    eigvals = np.linalg.eigvalsh(G.astype(float))
    if np.min(eigvals) < -1e-6:
        return False, 0
    det = np.linalg.det(G.astype(float))
    return True, det


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng, max_sol=500, deadline=None):
    """Corrected RREF-based column constraint solver from init_program.py."""
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
        pv = icons[np.newaxis, :] - fm @ ic.T  # CORRECTED: minus sign
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


def compute_bottom_constraints(G):
    """Compute what the bottom 18 entries of columns 0-10 must satisfy.

    For the new Gram:
    - Cols 0-2 (among themselves): ip = G[i,j] - top_ip(i,j) for i,j in {0,1,2}
    - Cols 3-5 (among themselves): similar
    - Cross cols 0-2 vs 3-5: depends on which variant (9 or 1)
    - Col 6 vs everything: determined
    - Cols 7-10 vs everything: determined
    """
    pass  # Will be computed dynamically based on the top block


def find_top_block(G, rng, max_time=30.0):
    """Find a valid 11x11 top block for a given Gram variant.

    The top block determines the first 11 entries of each column.
    It must satisfy: top^T @ top gives the correct 11x11 sub-Gram.

    For our Gram, the 11x11 sub-Gram G[:11,:11] is:
    - Diagonal: 29
    - G[i,j] for i in {0,1,2}, j in {3,4,5}: either 9 or 1
    - G[6,j]=-3 for j in {7,8,9,10}
    - All other off-diagonal: 1
    """
    G11 = G[:11, :11].copy()
    deadline = time.time() + max_time

    best_top = None
    best_err = float('inf')

    attempt = 0
    while time.time() < deadline:
        attempt += 1

        # Strategy: Start from random +-1 matrix and use greedy row-replacement
        # to match the target Gram
        if attempt == 1:
            # Try Cholesky-based initialization
            L = np.linalg.cholesky(G11.astype(float))
            T = np.sign(L.T + rng.randn(11, 11) * 0.3).astype(np.int64)
            T[T == 0] = 1
        elif best_top is not None and rng.random() < 0.5:
            # Perturb best
            T = best_top.copy()
            n_flip = rng.randint(1, 8)
            for _ in range(n_flip):
                r, c = rng.randint(0, 11), rng.randint(0, 11)
                T[r, c] *= -1
        else:
            # Random
            T = rng.choice([-1, 1], size=(11, 11)).astype(np.int64)

        # Row-replacement optimization for the 11x11 block
        # Minimize ||T^T T - G11||^2
        for iteration in range(200):
            gram = T.T @ T
            err = int(np.sum((gram - G11) ** 2))
            if err == 0:
                return T
            if err < best_err:
                best_err = err
                best_top = T.copy()

            # Try replacing each row
            improved = False
            for row in rng.permutation(11):
                old_row = T[row].copy()
                gram_minus = gram - np.outer(old_row, old_row)
                target = G11 - gram_minus

                # Greedy: for each col, choose sign to minimize error
                new_row = np.ones(11, dtype=np.int64)
                for col in rng.permutation(11):
                    # dot products with all columns
                    others = np.sum(new_row[:col] * target[:col, col]) if col > 0 else 0
                    # remaining columns contribute their dot product
                    target_val = target[col, col]
                    # Simple: choose sign of target row element
                    new_row[col] = 1 if target[col, col] > 0 else -1

                # Better: solve greedily column by column
                new_row = np.ones(11, dtype=np.int64)
                perm = list(rng.permutation(11))
                for idx in perm:
                    # If we set new_row[idx] = +1 or -1, what's the impact?
                    partial = target[idx, :] * new_row
                    partial[idx] = 0
                    m = np.sum(partial)
                    new_row[idx] = 1 if m + target[idx, idx] > abs(m - target[idx, idx]) else -1

                new_gram = gram_minus + np.outer(new_row, new_row)
                new_err = int(np.sum((new_gram - G11) ** 2))
                if new_err < err:
                    T[row] = new_row
                    gram = new_gram
                    err = new_err
                    improved = True
                    if err == 0:
                        return T

            if not improved:
                break

    if best_err == 0:
        return best_top
    return None


def find_top_block_v2(G, rng, max_time=60.0):
    """Find a valid 11x11 top block using a smarter approach.

    We know the structure must have:
    - 29 rows total, top 11 rows
    - Columns 0-2, 3-5, 6, 7-10 have specific relationships

    For the original Solomon, the top block has:
    - 6x6 circulant block (cols 0-5, rows 0-5)
    - Separator row/col 6
    - 4x4 Hadamard block (cols 7-10, rows 7-10)
    - Cross blocks between these

    The key question: what top block gives G[:11,:11] with the NEW cross-block values?
    """
    G11 = G[:11, :11].copy()
    deadline = time.time() + max_time

    # The top block T (11 rows x 11 cols) must satisfy T^T T = G11
    # where G11 has specific structure.
    # The bottom 18 rows will be found later.
    # BUT: T^T T is only the TOP contribution. The full Gram is
    # G[i,j] = sum_{row=0}^{28} R[row,i] * R[row,j]
    #        = T^T T (rows 0-10) + B^T B (rows 11-28)
    # So T^T T does NOT need to equal G11.
    # What we need is: T^T T + B^T B = G (for all 29x29)
    # Restricting to cols 0-10: T[:,:11]^T T[:,:11] + B[:,:11]^T B[:,:11] = G[:11,:11]
    #
    # Actually, R is 29x29. R[:,i] is column i.
    # R[:,i] has entries (R[0,i], R[1,i], ..., R[28,i])
    # The top 11 entries of column i: R[0:11, i]
    # The bottom 18 entries: R[11:29, i]
    #
    # G[i,j] = R[:,i] . R[:,j] = R[0:11,i].R[0:11,j] + R[11:29,i].R[11:29,j]
    #
    # So for each pair of columns i,j:
    #   top_ip[i,j] + bottom_ip[i,j] = G[i,j]
    #   bottom_ip[i,j] = G[i,j] - top_ip[i,j]
    #
    # For the TOP BLOCK: rows 0-10, this IS the 11x11 block of columns.
    # We need to find T (11x11, +-1) and B (18x11, +-1) for columns 0-10
    # such that T^T T + B^T B = G[:11,:11].
    #
    # Additionally, for columns 11-28 (the remaining 18):
    # The top 11 entries are determined by the backtracker later.

    # Strategy: enumerate top blocks that satisfy known constraints.
    # We need T^T T <= G[:11,:11] entry-wise is NOT required.
    # We need bottom_ip = G - top_ip to be achievable by +-1 vectors of length 18.

    # Key constraint: bottom_ip[i,j] = G[i,j] - top_ip[i,j]
    # For diagonal: bottom_ip[i,i] = 29 - top_ip[i,i] = 29 - 11 = 18 (always, since all entries +-1)
    # For off-diagonal: bottom_ip[i,j] must be achievable
    # Specifically, two +-1 vectors of length 18 can have inner product in {-18, -16, ..., 16, 18}
    # and the sum constraint: if col i has k ones in bottom, sum = 2k-18, ip depends on overlap.

    # For the ORIGINAL Solomon:
    # top_ip for cols in {0,1,2}:
    #   Solomon KNOWN_TOP columns 0-2 are the first 3 cols of BLOCK_6x6 padded
    #   cols 0-2 in KNOWN_TOP: rows 0-5 from BLOCK_6x6, row 6 = -1, rows 7-10 = 1

    best_top = None
    best_err = float('inf')

    # Try random search with Gram-guided row replacement
    attempt = 0
    while time.time() < deadline:
        attempt += 1

        # Initialize
        if attempt <= 3:
            L = np.linalg.cholesky(G11.astype(float))
            noise_scale = 0.1 * attempt
            T = np.sign(L.T + rng.randn(11, 11) * noise_scale).astype(np.int64)
            T[T == 0] = 1
        elif best_top is not None and rng.random() < 0.6:
            T = best_top.copy()
            n_flip = rng.randint(1, 12)
            for _ in range(n_flip):
                r, c = rng.randint(0, 11), rng.randint(0, 11)
                T[r, c] *= -1
        else:
            T = rng.choice([-1, 1], size=(11, 11)).astype(np.int64)

        # Gram-guided optimization of the 11x11 block
        for iteration in range(500):
            gram = T.T @ T
            # Compute required bottom gram
            bottom_gram = G11 - gram

            # Check: bottom_gram must be achievable by 18-entry +-1 vectors
            # Diagonal must be 18 (always true for 11-entry +-1 top block)
            diag_ok = np.all(np.diag(bottom_gram) == 18)

            # Off-diagonal: must be achievable and have same parity as 18
            # (18-entry vectors have ip = 18 - 2*disagreements, so ip is even)
            off_diag = bottom_gram.copy()
            np.fill_diagonal(off_diag, 0)
            parity_ok = np.all(off_diag % 2 == 0)
            range_ok = np.all(np.abs(off_diag[np.triu_indices(11, 1)]) <= 18)

            if diag_ok and parity_ok and range_ok:
                return T, bottom_gram

            # Error: how far from feasible
            err = 0
            if not diag_ok:
                err += np.sum((np.diag(bottom_gram) - 18) ** 2) * 100
            upper_idx = np.triu_indices(11, 1)
            off_vals = bottom_gram[upper_idx]
            err += np.sum(np.maximum(0, np.abs(off_vals) - 18) ** 2) * 10
            err += np.sum((off_vals % 2) ** 2)

            if err < best_err:
                best_err = err
                best_top = T.copy()

            if err == 0:
                return T, bottom_gram

            # Row replacement
            improved = False
            for row in rng.permutation(11):
                old_row = T[row].copy()
                gram_minus_row = gram - np.outer(old_row, old_row)

                # Target for new row: want T^T T = G11 - achievable_bottom
                # Simplify: try to make bottom_gram have all even off-diag, diag=18, |off|<=18
                # That means gram[i,j] = G11[i,j] - bottom[i,j] where bottom is constrained
                # Target gram for (row, col):
                # gram[row, col] should be such that G11[row,col] - gram[row,col] is even, in [-18,18]
                # i.e., gram[row,col] should have same parity as G11[row,col] and
                # G11[row,col] - 18 <= gram[row,col] <= G11[row,col] + 18

                # Greedy construction
                new_row = np.ones(11, dtype=np.int64)
                perm = list(rng.permutation(11))
                for idx in perm:
                    partial_gram = gram_minus_row + np.outer(new_row, new_row)
                    bg = G11 - partial_gram
                    # Score: sum of constraint violations
                    # Try +1 and -1 for this position
                    scores = []
                    for val in [1, -1]:
                        new_row[idx] = val
                        pg = gram_minus_row + np.outer(new_row, new_row)
                        bg_test = G11 - pg
                        diag_err = np.sum((np.diag(bg_test) - 18) ** 2) * 100
                        ov = bg_test[upper_idx]
                        range_err = np.sum(np.maximum(0, np.abs(ov) - 18) ** 2) * 10
                        parity_err = np.sum((ov % 2) ** 2)
                        scores.append(diag_err + range_err + parity_err)
                    new_row[idx] = 1 if scores[0] <= scores[1] else -1

                new_gram = gram_minus_row + np.outer(new_row, new_row)
                bg_new = G11 - new_gram
                diag_err = np.sum((np.diag(bg_new) - 18) ** 2) * 100
                ov = bg_new[upper_idx]
                range_err = np.sum(np.maximum(0, np.abs(ov) - 18) ** 2) * 10
                parity_err = np.sum((ov % 2) ** 2)
                new_err = diag_err + range_err + parity_err

                if new_err < err:
                    T[row] = new_row
                    gram = new_gram
                    err = new_err
                    improved = True
                    if err == 0:
                        return T, G11 - gram

            if not improved:
                break

    return None, None


def find_top_block_v3(G, rng, max_time=30.0):
    """Find a valid 11x11 top block by random search with constraint checking.

    The constraint is: for every pair (i,j) of columns 0-10,
    the bottom inner product G[i,j] - top_ip(i,j) must be:
    1. Even (18-entry +-1 vectors always have even inner product)
    2. In [-18, 18]
    3. Diagonal bottom = 18 (automatic if top has 11 entries +-1)

    Additionally, the bottom vectors must have certain sum constraints
    for the full column solver to work.
    """
    G11 = G[:11, :11].copy()
    deadline = time.time() + max_time

    best_top = None
    best_err = float('inf')
    upper_idx = np.triu_indices(11, 1)

    attempt = 0
    while time.time() < deadline:
        attempt += 1

        # Generate random 11x11 +-1 matrix
        T = rng.choice([-1, 1], size=(11, 11)).astype(np.int64)

        # Quick local optimization: flip entries to reduce constraint violations
        for opt_iter in range(1000):
            gram = T.T @ T
            bottom_gram = G11 - gram

            # Check constraints
            off_vals = bottom_gram[upper_idx]
            parity_violations = np.sum(off_vals % 2 != 0)
            range_violations = np.sum(np.abs(off_vals) > 18)
            # Diagonal should be 18
            diag_err = np.sum(np.diag(bottom_gram) != 18)

            err = parity_violations * 10 + range_violations * 100 + diag_err * 1000

            if err == 0:
                return T, bottom_gram

            if err < best_err:
                best_err = err
                best_top = T.copy()

            # Try random flip
            r, c = rng.randint(0, 11), rng.randint(0, 11)
            T[r, c] *= -1

            new_gram = T.T @ T
            new_bottom = G11 - new_gram
            new_off = new_bottom[upper_idx]
            new_parity = np.sum(new_off % 2 != 0)
            new_range = np.sum(np.abs(new_off) > 18)
            new_diag = np.sum(np.diag(new_bottom) != 18)
            new_err = new_parity * 10 + new_range * 100 + new_diag * 1000

            if new_err > err:
                T[r, c] *= -1  # revert

    if best_err == 0:
        return best_top, G[:11, :11] - best_top.T @ best_top
    return None, None


def construct_bottom_for_columns(top_block, G, rng, max_time=5.0):
    """Given a valid 11x11 top block T, find 18-entry bottom vectors for columns 0-10
    that satisfy all pairwise inner product constraints.

    For each pair (i,j) of columns 0-10:
      bottom_ip[i,j] = G[i,j] - (T[:,i] . T[:,j])
    """
    T = top_block
    G11 = G[:11, :11]
    top_gram = T.T @ T
    bottom_target = G11 - top_gram

    # Verify feasibility
    for i in range(11):
        if bottom_target[i, i] != 18:
            return None
    upper_idx = np.triu_indices(11, 1)
    off_vals = bottom_target[upper_idx]
    if np.any(off_vals % 2 != 0) or np.any(np.abs(off_vals) > 18):
        return None

    deadline = time.time() + max_time

    # Build bottom vectors one by one using constraint satisfaction
    while time.time() < deadline:
        v = [None] * 11

        # Col 0: any 18-entry +-1 vector (sum doesn't matter for Gram, only pairwise)
        # But bottom_target[0,0]=18, so all entries must... no, 18 entries with ip with self = 18
        # That's always true: any +-1 vector of length 18 has self-ip = 18. Good.

        # For col 0: just random
        v[0] = rng.choice([-1, 1], size=18).astype(np.int64)

        # Col 1: must satisfy v[0].v[1] = bottom_target[0,1]
        target_01 = int(bottom_target[0, 1])
        # v[0].v[1] = sum(v[0][k]*v[1][k]) = 18 - 2*d where d = # disagreements
        # So d = (18 - target_01) / 2
        d01 = (18 - target_01) // 2
        if d01 < 0 or d01 > 18:
            continue

        found = False
        for _ in range(10000):
            if time.time() > deadline:
                break
            v[1] = v[0].copy()
            flip_idx = rng.choice(18, size=d01, replace=False)
            v[1][flip_idx] *= -1
            if np.dot(v[0], v[1]) == target_01:
                found = True
                break
        if not found:
            continue

        # Col 2: must satisfy v[0].v[2]=bt[0,2], v[1].v[2]=bt[1,2]
        target_02 = int(bottom_target[0, 2])
        target_12 = int(bottom_target[1, 2])

        found2 = False
        for _ in range(50000):
            if time.time() > deadline:
                break
            v[2] = rng.choice([-1, 1], size=18).astype(np.int64)
            if np.dot(v[0], v[2]) == target_02 and np.dot(v[1], v[2]) == target_12:
                found2 = True
                break
        if not found2:
            continue

        # Cols 3-5: similar incremental construction
        all_ok = True
        for c in range(3, 6):
            targets = [int(bottom_target[prev, c]) for prev in range(c)]
            found_c = False
            for _ in range(50000):
                if time.time() > deadline:
                    all_ok = False
                    break
                vc = rng.choice([-1, 1], size=18).astype(np.int64)
                ok = True
                for prev in range(c):
                    if np.dot(v[prev], vc) != targets[prev]:
                        ok = False
                        break
                if ok:
                    v[c] = vc
                    found_c = True
                    break
            if not found_c:
                all_ok = False
                break
        if not all_ok:
            continue

        # Col 6
        target_6 = [int(bottom_target[prev, 6]) for prev in range(6)]
        found6 = False
        for _ in range(50000):
            if time.time() > deadline:
                break
            v[6] = rng.choice([-1, 1], size=18).astype(np.int64)
            ok = True
            for prev in range(6):
                if np.dot(v[prev], v[6]) != target_6[prev]:
                    ok = False
                    break
            if ok:
                found6 = True
                break
        if not found6:
            continue

        # Cols 7-10: must satisfy IPs with all previous
        all_ok2 = True
        for c in range(7, 11):
            targets_c = [int(bottom_target[prev, c]) for prev in range(c)]
            found_c = False
            for _ in range(100000):
                if time.time() > deadline:
                    all_ok2 = False
                    break
                vc = rng.choice([-1, 1], size=18).astype(np.int64)
                ok = True
                for prev in range(c):
                    if np.dot(v[prev], vc) != targets_c[prev]:
                        ok = False
                        break
                if ok:
                    v[c] = vc
                    found_c = True
                    break
            if not found_c:
                all_ok2 = False
                break
        if not all_ok2:
            continue

        return v

    return None


def construct_bottom_systematic(top_block, G, rng, max_time=10.0):
    """Use the RREF solver to construct bottom vectors systematically.

    Build bottom vectors for columns 0-10, one at a time, using the
    column constraint solver on the 18-dimensional sub-problem.
    """
    T = top_block
    G11 = G[:11, :11]
    top_gram = T.T @ T
    bottom_target = G11 - top_gram

    # Verify feasibility
    for i in range(11):
        if bottom_target[i, i] != 18:
            return None
    upper_idx = np.triu_indices(11, 1)
    off_vals = bottom_target[upper_idx]
    if np.any(off_vals % 2 != 0) or np.any(np.abs(off_vals) > 18):
        return None

    deadline = time.time() + max_time

    # Use backtracking on the bottom vectors
    # Each bottom vector is 18-entry +-1
    # Constraints: pairwise inner products = bottom_target

    def find_bottoms():
        placed = []
        for c in range(11):
            if time.time() > deadline:
                return None

            if len(placed) == 0:
                # First column: random
                candidates = [rng.choice([-1, 1], size=18).astype(np.int64) for _ in range(100)]
            else:
                # Use RREF solver adapted for 18 dimensions
                # Constraints: placed[i] . x = bottom_target[i, c] for each placed column i
                n_placed = len(placed)
                A_rows = [[Fraction(int(placed[i][j])) for j in range(18)] for i in range(n_placed)]
                b_vec = [Fraction(int(bottom_target[i, c])) for i in range(n_placed)]

                # RREF
                aug = [A_rows[i] + [b_vec[i]] for i in range(n_placed)]
                pivot_cols_list = []
                row_idx = 0
                for col in range(18):
                    if row_idx >= n_placed:
                        break
                    piv = -1
                    for r in range(row_idx, n_placed):
                        if aug[r][col] != 0:
                            piv = r
                            break
                    if piv == -1:
                        continue
                    if piv != row_idx:
                        aug[row_idx], aug[piv] = aug[piv], aug[row_idx]
                    pv = aug[row_idx][col]
                    for j in range(19):
                        aug[row_idx][j] /= pv
                    for r in range(row_idx + 1, n_placed):
                        if aug[r][col] != 0:
                            f = aug[r][col]
                            for j in range(19):
                                aug[r][j] -= f * aug[row_idx][j]
                    pivot_cols_list.append(col)
                    row_idx += 1

                # Check consistency
                inconsistent = False
                for r in range(len(pivot_cols_list), n_placed):
                    if aug[r][18] != 0:
                        inconsistent = True
                        break
                if inconsistent:
                    return None

                # Back-substitution
                for i in range(len(pivot_cols_list) - 1, -1, -1):
                    pc = pivot_cols_list[i]
                    for r2 in range(i):
                        if aug[r2][pc] != 0:
                            f = aug[r2][pc]
                            for j in range(19):
                                aug[r2][j] -= f * aug[i][j]

                free_cols = [col for col in range(18) if col not in pivot_cols_list]
                nf = len(free_cols)
                np_ = len(pivot_cols_list)

                ic = np.zeros((np_, nf), dtype=np.int64)
                icons = np.zeros(np_, dtype=np.int64)
                iden = np.zeros(np_, dtype=np.int64)
                for i in range(np_):
                    ds = [aug[i][18].denominator] + [aug[i][fc].denominator for fc in free_cols]
                    lcd = 1
                    for d in ds:
                        lcd = math_lcm(lcd, d)
                    iden[i] = lcd
                    icons[i] = int(aug[i][18] * lcd)
                    for fi, fc in enumerate(free_cols):
                        ic[i, fi] = int(aug[i][fc] * lcd)

                candidates = []
                if nf <= 20:
                    total = 1 << nf
                    bits = np.arange(total, dtype=np.int32)
                    fm = np.empty((total, nf), dtype=np.int64)
                    for fi in range(nf):
                        fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
                    pv_vals = icons[np.newaxis, :] - fm @ ic.T
                    vm = np.ones(total, dtype=bool)
                    for i in range(np_):
                        vm &= (np.abs(pv_vals[:, i]) == iden[i])
                    si = np.where(vm)[0]
                    rng.shuffle(si)
                    for idx in si[:500]:
                        x = np.zeros(18, dtype=np.int64)
                        for fi, fc in enumerate(free_cols):
                            x[fc] = fm[idx, fi]
                        for i in range(np_):
                            x[pivot_cols_list[i]] = pv_vals[idx, i] // iden[i]
                        candidates.append(x)
                else:
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        bs = min(1 << 20, 100000)
                        fm = rng.choice([-1, 1], size=(bs, nf)).astype(np.int64)
                        pv_vals = icons[np.newaxis, :] - fm @ ic.T
                        vm = np.ones(bs, dtype=bool)
                        for i in range(np_):
                            vm &= (np.abs(pv_vals[:, i]) == iden[i])
                        for idx in np.where(vm)[0]:
                            if len(candidates) >= 500:
                                break
                            x = np.zeros(18, dtype=np.int64)
                            for fi, fc in enumerate(free_cols):
                                x[fc] = fm[idx, fi]
                            for i in range(np_):
                                x[pivot_cols_list[i]] = pv_vals[idx, i] // iden[i]
                            candidates.append(x)
                        if len(candidates) >= 500:
                            break

            if not candidates:
                return None

            # Pick a random candidate
            placed.append(candidates[rng.randint(len(candidates))])

        return placed

    for _ in range(100):
        if time.time() > deadline:
            break
        result = find_bottoms()
        if result is not None:
            return result

    return None


def backtrack_remaining_columns(placed_cols, placed_indices, G, rng, max_time=120.0, verbose=True):
    """Backtrack to find the remaining 18 columns (indices 11-28)."""
    deadline = time.time() + max_time
    max_depth = 0

    def dfs(pc, pi, step):
        nonlocal max_depth
        if time.time() > deadline:
            return False
        if step == 18:
            return True
        ti = 11 + step

        if step > max_depth:
            max_depth = step
            if verbose:
                print(f"    Backtracker depth: {step}/{18}", flush=True)

        ms = min(30, max(5, 500 // (step + 1)))
        sols = solve_column_constraints(pc, pi, ti, G, rng, max_sol=ms,
                                        deadline=min(time.time() + 5, deadline))
        if not sols:
            return False
        rng.shuffle(sols)
        for s in sols[:max(5, ms // 2)]:
            if time.time() > deadline:
                return False
            pc.append(s)
            pi.append(ti)
            if dfs(pc, pi, step + 1):
                return True
            pc.pop()
            pi.pop()
        return False

    cols = [c.copy() for c in placed_cols]
    indices = list(placed_indices)

    if dfs(cols, indices, 0):
        R = np.zeros((N, N), dtype=np.int64)
        for i, ci in enumerate(indices):
            R[:, ci] = cols[i]
        return R, max_depth

    return None, max_depth


def search_variant(variant_id, nine_pairs, rng, time_limit=120.0, verbose=True):
    """Search for a decomposition of a specific variant."""
    G = build_gram(nine_pairs)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Variant {variant_id}: G[{nine_pairs[0][0]},{nine_pairs[0][1]}]=9, G[{nine_pairs[1][0]},{nine_pairs[1][1]}]=9")
        print(f"{'='*60}", flush=True)

    # Verify Gram
    pd, det_val = verify_gram(G)
    if not pd:
        print(f"  NOT positive definite - skip")
        return None, 0

    deadline = time.time() + time_limit
    best_depth = 0

    # Phase 1: Find valid top blocks
    top_found = 0
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        remaining = deadline - time.time()
        if remaining < 5:
            break

        if verbose and attempt % 10 == 1:
            print(f"  Attempt {attempt}, best depth so far: {best_depth}/{18}", flush=True)

        # Find top block
        top_time = min(5.0, remaining * 0.2)
        top, bottom_gram = find_top_block_v3(G, rng, max_time=top_time)

        if top is None:
            # Try v2
            top, bottom_gram = find_top_block_v2(G, rng, max_time=top_time)

        if top is None:
            continue

        top_found += 1
        if verbose:
            print(f"  Top block #{top_found} found (attempt {attempt})", flush=True)

        # Phase 2: Construct bottom for columns 0-10
        bottom_time = min(10.0, remaining * 0.3)
        bottom = construct_bottom_systematic(top, G, rng, max_time=bottom_time)

        if bottom is None:
            bottom = construct_bottom_for_columns(top, G, rng, max_time=bottom_time)

        if bottom is None:
            if verbose:
                print(f"    Bottom construction failed", flush=True)
            continue

        if verbose:
            print(f"    Bottom constructed!", flush=True)

        # Assemble full columns 0-10
        cols = []
        indices = list(range(11))
        valid = True
        for j in range(11):
            c = np.zeros(N, dtype=np.int64)
            c[:11] = top[:, j]
            c[11:] = bottom[j]
            cols.append(c)

        # Verify partial Gram
        pg = np.column_stack(cols).T @ np.column_stack(cols)
        gram_err = np.sum((pg - G[:11, :11]) ** 2)
        if gram_err != 0:
            if verbose:
                print(f"    Partial Gram error: {gram_err} - skipping", flush=True)
            continue

        if verbose:
            print(f"    Partial Gram EXACT - starting backtracker", flush=True)

        # Phase 3: Backtrack remaining 18 columns
        bt_time = min(60.0, remaining - 2)
        R, depth = backtrack_remaining_columns(cols, indices, G, rng, max_time=bt_time, verbose=verbose)

        if depth > best_depth:
            best_depth = depth

        if R is not None:
            # VERIFY
            if np.array_equal(R.T @ R, G) and np.all(np.abs(R) == 1):
                print(f"\n*** DECOMPOSITION FOUND for variant {variant_id}! ***", flush=True)
                return R, 18
            else:
                print(f"    Verification FAILED", flush=True)

    if verbose:
        print(f"  Variant {variant_id}: best depth = {best_depth}/{18}, top blocks found = {top_found}")

    return None, best_depth


def main():
    print("="*70)
    print("SYSTEMATIC VARIANT DECOMPOSITION SEARCH")
    print("="*70)

    variants = enumerate_variants()
    print(f"\nEnumerated {len(variants)} variants:")
    for i, (p1, p2) in enumerate(variants):
        print(f"  Variant {i}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9")

    # Verify all have same determinant
    print("\nVerifying determinants...")
    for i, (p1, p2) in enumerate(variants):
        G = build_gram([p1, p2])
        _, det_val = verify_gram(G)
        if i == 0:
            ref_det = det_val
        print(f"  Variant {i}: det ratio to ref = {det_val/ref_det:.10f}")

    # Phase 1: Quick scan of all variants (2 min each)
    print(f"\n{'='*70}")
    print("PHASE 1: Quick scan (2 min per variant)")
    print(f"{'='*70}")
    sys.stdout.flush()

    results = {}
    for i, (p1, p2) in enumerate(variants):
        rng = np.random.RandomState(42 + i * 1000)
        R, depth = search_variant(i, [p1, p2], rng, time_limit=120.0, verbose=True)
        results[i] = (R, depth)

        if R is not None:
            print(f"\n*** BREAKTHROUGH: Variant {i} decomposed! ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_R = abs(int(np.round(np.linalg.det(R.astype(float)))))
            score = det_R / THEORETICAL_MAX
            print(f"Score = {score:.6f}")
            sys.stdout.flush()
            # Don't stop - try to find others too, but save this one

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for i in sorted(results.keys()):
        R, depth = results[i]
        p1, p2 = variants[i]
        status = "SOLVED!" if R is not None else f"depth {depth}/{18}"
        print(f"  Variant {i}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9 -> {status}")

    # Phase 2: Focus on most promising
    best_variant = max(results.keys(), key=lambda k: results[k][1])
    best_depth = results[best_variant][1]
    print(f"\nMost promising: Variant {best_variant} (depth {best_depth})")

    if best_depth < 18:
        print(f"\n{'='*70}")
        print(f"PHASE 2: Focused search on variant {best_variant}")
        print(f"{'='*70}")
        sys.stdout.flush()

        p1, p2 = variants[best_variant]
        rng = np.random.RandomState(12345)
        R, depth = search_variant(best_variant, [p1, p2], rng, time_limit=600.0, verbose=True)

        if R is not None:
            print(f"\n*** BREAKTHROUGH: Variant {best_variant} decomposed! ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)


if __name__ == "__main__":
    main()
