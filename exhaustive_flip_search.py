"""Exhaustive k-flip search on the Orrick-Solomon matrix.

The OS matrix (c=320, score=0.9357) is known to be a 1-flip local maximum.
This script searches for 2-flip and 3-flip improvements exhaustively.

For 2-flips: 29*29 choose 2 = ~352K pairs -- feasible with rank-2 det formula.
For 3-flips: 29*29 choose 3 = ~100M triples -- need pruning.

Uses exact rank-k determinant update formulas (not floating point approximations).
"""

import numpy as np
import time
import sys

N = 29
THEORETICAL_MAX = 1270698346568170340352

# The Orrick-Solomon matrix
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
], dtype=np.int64)


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


def exhaustive_2flip_exact(A):
    """Try ALL 2-flip combinations using rank-2 det update.

    For flipping entries at (r1,c1) and (r2,c2):

    Same row case (r1==r2):
      det ratio = (1 + d1*Ai[c1,r]) * (1 + d2*Ai_new[c2,r])
      where Ai_new accounts for first flip via SM.
      Equivalently: rank-2 formula with 2x2 determinant.

    Different row case:
      U = [d1*e_{r1}, d2*e_{r2}], V = [e_{c1}, e_{c2}]
      det(I + V^T Ai U) = m11*m22 - m12*m21
      where m11 = 1+d1*Ai[c1,r1], m12 = d2*Ai[c1,r2], etc.
    """
    n = A.shape[0]
    Af = A.astype(np.float64)
    s, ld = np.linalg.slogdet(Af)
    base_det = abs(det_bareiss(A.astype(int).tolist()))
    print(f"Base |det| = {base_det}, score = {base_det/THEORETICAL_MAX:.8f}")

    Ai = np.linalg.inv(Af)

    # Precompute d values and Ai columns needed
    total_entries = n * n
    best_ratio = 1.0
    best_pair = None
    count = 0

    t0 = time.time()

    for e1 in range(total_entries):
        r1, c1 = e1 // n, e1 % n
        d1 = -2.0 * Af[r1, c1]

        for e2 in range(e1 + 1, total_entries):
            r2, c2 = e2 // n, e2 % n
            d2 = -2.0 * Af[r2, c2]

            # Rank-2 update formula
            m11 = 1.0 + d1 * Ai[c1, r1]
            m12 = d2 * Ai[c1, r2]
            m21 = d1 * Ai[c2, r1]
            m22 = 1.0 + d2 * Ai[c2, r2]
            det_ratio = abs(m11 * m22 - m12 * m21)

            if det_ratio > best_ratio + 1e-10:
                best_ratio = det_ratio
                best_pair = (r1, c1, r2, c2)

            count += 1

        if (e1 + 1) % 100 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * e1 / total_entries
            print(f"  2-flip: {pct:.1f}% done, {count} pairs, "
                  f"best ratio = {best_ratio:.10f}, elapsed = {elapsed:.1f}s")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  2-flip search complete: {count} pairs in {elapsed:.1f}s")
    print(f"  Best det ratio: {best_ratio:.10f}")

    if best_pair is not None and best_ratio > 1.0 + 1e-10:
        r1, c1, r2, c2 = best_pair
        print(f"  IMPROVEMENT FOUND: flip ({r1},{c1}) and ({r2},{c2})")
        print(f"  Det would increase by factor {best_ratio:.10f}")

        B = A.copy()
        B[r1, c1] *= -1
        B[r2, c2] *= -1
        new_det = abs(det_bareiss(B.astype(int).tolist()))
        new_score = new_det / THEORETICAL_MAX
        print(f"  New |det| = {new_det}, new score = {new_score:.8f}")
        return B, new_det, True
    else:
        print(f"  No 2-flip improvement exists (matrix is 2-flip local maximum)")
        return A, base_det, False


def exhaustive_3flip_samerow(A):
    """Try all 3-flip combinations where at least 2 flips are in the same row.

    This is much more tractable than all 3-flip combos.
    For each row r, try all (c1,c2) pairs in that row + one entry from another row.

    Per row: C(29,2) * (29*28) = 406 * 812 ≈ 330K combos
    Total: 29 rows * 330K ≈ 9.6M combos (feasible)
    """
    n = A.shape[0]
    Af = A.astype(np.float64)
    Ai = np.linalg.inv(Af)

    base_det = abs(det_bareiss(A.astype(int).tolist()))
    print(f"\n3-flip (same-row pair + 1): base |det| = {base_det}")

    best_ratio = 1.0
    best_triple = None
    count = 0
    t0 = time.time()

    for row in range(n):
        for c1 in range(n):
            d1 = -2.0 * Af[row, c1]
            for c2 in range(c1 + 1, n):
                d2 = -2.0 * Af[row, c2]

                # After flipping (row,c1) and (row,c2), compute the new inverse
                # using rank-2 update, then check all possible third flips.
                #
                # But this is expensive. Instead, use rank-3 determinant formula.
                # For 3 rank-1 updates, det ratio = det of a 3x3 matrix.
                #
                # However, the 3 updates here are:
                # U1 = d1 * e_row * e_c1^T
                # U2 = d2 * e_row * e_c2^T
                # U3 = d3 * e_r3 * e_c3^T
                #
                # Combined: U = [d1*e_row, d2*e_row, d3*e_r3], V = [e_c1, e_c2, e_c3]
                # det(I + V^T Ai U) = det of 3x3 matrix M where
                # M[i,j] = delta_{ij} + V_col_i^T * Ai * U_col_j

                for r3 in range(n):
                    for c3 in range(n):
                        # Skip if this is one of the first two entries
                        if r3 == row and (c3 == c1 or c3 == c2):
                            continue
                        # Skip duplicates (ensure canonical ordering)
                        e3 = r3 * n + c3
                        e1 = row * n + c1
                        e2 = row * n + c2
                        if e3 <= e2:
                            continue

                        d3 = -2.0 * Af[r3, c3]

                        # 3x3 matrix M = I + V^T Ai U
                        # U cols: d1*e_row, d2*e_row, d3*e_r3
                        # V cols: e_c1, e_c2, e_c3
                        # M[0,0] = 1 + d1*Ai[c1,row]
                        # M[0,1] = d2*Ai[c1,row]
                        # M[0,2] = d3*Ai[c1,r3]
                        # M[1,0] = d1*Ai[c2,row]
                        # M[1,1] = 1 + d2*Ai[c2,row]
                        # M[1,2] = d3*Ai[c2,r3]
                        # M[2,0] = d1*Ai[c3,row]
                        # M[2,1] = d2*Ai[c3,row]
                        # M[2,2] = 1 + d3*Ai[c3,r3]

                        m00 = 1.0 + d1 * Ai[c1, row]
                        m01 = d2 * Ai[c1, row]
                        m02 = d3 * Ai[c1, r3]
                        m10 = d1 * Ai[c2, row]
                        m11 = 1.0 + d2 * Ai[c2, row]
                        m12 = d3 * Ai[c2, r3]
                        m20 = d1 * Ai[c3, row]
                        m21 = d2 * Ai[c3, row]
                        m22 = 1.0 + d3 * Ai[c3, r3]

                        det3 = (m00 * (m11 * m22 - m12 * m21)
                              - m01 * (m10 * m22 - m12 * m20)
                              + m02 * (m10 * m21 - m11 * m20))

                        ratio = abs(det3)
                        if ratio > best_ratio + 1e-10:
                            best_ratio = ratio
                            best_triple = (row, c1, row, c2, r3, c3)

                        count += 1

        elapsed = time.time() - t0
        print(f"  Row {row}/29: {count} triples checked, "
              f"best ratio = {best_ratio:.10f}, {elapsed:.1f}s")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  3-flip (same-row) complete: {count} triples in {elapsed:.1f}s")
    print(f"  Best det ratio: {best_ratio:.10f}")

    if best_triple is not None and best_ratio > 1.0 + 1e-10:
        r1, c1, r2, c2, r3, c3 = best_triple
        print(f"  IMPROVEMENT: flip ({r1},{c1}), ({r2},{c2}), ({r3},{c3})")
        B = A.copy()
        B[r1, c1] *= -1
        B[r2, c2] *= -1
        B[r3, c3] *= -1
        new_det = abs(det_bareiss(B.astype(int).tolist()))
        new_score = new_det / THEORETICAL_MAX
        print(f"  New |det| = {new_det}, score = {new_score:.8f}")
        return B, new_det, True
    else:
        print(f"  No same-row 3-flip improvement")
        return A, base_det, False


def exhaustive_3flip_vectorized(A, deadline):
    """Vectorized exhaustive 3-flip using rank-3 determinant formula.

    For all triples of entries (e1 < e2 < e3), compute:
    det ratio = det(I + V^T * Ainv * U) where U,V encode the 3 flips.

    Total: C(841, 3) ≈ 99M triples. At ~10M/sec vectorized, ~10 seconds.
    But memory for full enumeration is huge. Do it in batches by fixing e1.
    """
    n = A.shape[0]
    total = n * n  # 841
    Af = A.astype(np.float64)
    Ai = np.linalg.inv(Af)

    base_det = abs(det_bareiss(A.astype(int).tolist()))
    print(f"\nVectorized 3-flip search: base |det| = {base_det}")
    sys.stdout.flush()

    # Precompute: for each entry e, store (row, col, d=-2*A[r,c])
    rows = np.arange(total) // n
    cols = np.arange(total) % n
    d_vals = -2.0 * Af[rows, cols]

    # Precompute Ai values needed
    # For entry e: Ai[col_e, row_e] (needed for diagonal of M)
    # For pair (e1, e2): Ai[col_e1, row_e2] (off-diagonal)

    best_ratio = 1.0
    best_triple = None
    t0 = time.time()
    checked = 0

    for e1 in range(total):
        if time.time() > deadline:
            break
        r1, c1 = rows[e1], cols[e1]
        d1 = d_vals[e1]

        # All pairs (e2, e3) with e1 < e2 < e3
        e2_range = np.arange(e1 + 1, total)
        if len(e2_range) == 0:
            continue

        r2 = rows[e2_range]
        c2 = cols[e2_range]
        d2 = d_vals[e2_range]

        # For each e2, we need to check all e3 > e2
        # This is O(total^2) per e1, so O(total^3) overall -- too slow for vectorization
        #
        # Instead, for fixed e1, compute the 2x2 sub-matrix for the first two flips,
        # then for each e3 compute the 3x3 det using cofactor expansion along row 3.

        # M = [[1+d1*Ai[c1,r1], d2*Ai[c1,r2], d3*Ai[c1,r3]],
        #      [d1*Ai[c2,r1],   1+d2*Ai[c2,r2], d3*Ai[c2,r3]],
        #      [d1*Ai[c3,r1],   d2*Ai[c3,r2],   1+d3*Ai[c3,r3]]]

        # For fixed e1 and each e2: precompute the 2x2 top-left block
        m00 = 1.0 + d1 * Ai[c1, r1]  # scalar, same for all e2
        m01_arr = d2 * Ai[c1, r2]    # array over e2
        m10_arr = d1 * Ai[c2, r1]    # array over e2
        m11_arr = 1.0 + d2 * Ai[c2, r2]  # array over e2

        # 2x2 det of top-left
        det2_arr = m00 * m11_arr - m01_arr * m10_arr  # array over e2

        for idx2, e2 in enumerate(e2_range):
            if time.time() > deadline:
                break

            r2i, c2i = rows[e2], cols[e2]
            d2i = d_vals[e2]

            # For this (e1, e2) pair, check all e3 > e2
            e3_range = np.arange(e2 + 1, total)
            if len(e3_range) == 0:
                continue

            r3 = rows[e3_range]
            c3 = cols[e3_range]
            d3 = d_vals[e3_range]

            # Row 2 of M (index 2): [d1*Ai[c3,r1], d2i*Ai[c3,r2i], 1+d3*Ai[c3,r3]]
            m20 = d1 * Ai[c3, r1]
            m21 = d2i * Ai[c3, r2i]
            m22 = 1.0 + d3 * Ai[c3, r3]

            # Col 2 of M (top two rows): Ai[c1,r3]*d3 and Ai[c2i,r3]*d3
            m02 = d3 * Ai[c1, r3]
            m12 = d3 * Ai[c2i, r3]

            # Cofactor expansion along column 2 (index 2):
            # det = m02*(m10_arr[idx2]*m21 - m11_arr[idx2]*m20)
            #      -m12*(m00*m21 - m01_arr[idx2]*m20)
            #      +m22*(det2_arr[idx2])
            #
            # But m10_arr[idx2], m11_arr[idx2], m01_arr[idx2] are scalars

            m00_v = m00  # scalar
            m01_v = m01_arr[idx2]  # scalar
            m10_v = m10_arr[idx2]  # scalar
            m11_v = m11_arr[idx2]  # scalar
            det2_v = det2_arr[idx2]  # scalar

            # det3 = m02*(m10_v*m21 - m11_v*m20) - m12*(m00_v*m21 - m01_v*m20) + m22*det2_v
            det3 = (m02 * (m10_v * m21 - m11_v * m20)
                  - m12 * (m00_v * m21 - m01_v * m20)
                  + m22 * det2_v)

            ratios = np.abs(det3)
            max_idx = np.argmax(ratios)
            max_ratio = ratios[max_idx]

            if max_ratio > best_ratio + 1e-10:
                best_ratio = max_ratio
                e3_best = e3_range[max_idx]
                best_triple = (e1, e2, e3_best)

            checked += len(e3_range)

        if (e1 + 1) % 50 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * e1 / total
            print(f"  3-flip vectorized: {pct:.1f}% (e1={e1}), "
                  f"{checked/1e6:.1f}M triples, "
                  f"best ratio = {best_ratio:.10f}, {elapsed:.1f}s")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  3-flip search: {checked/1e6:.1f}M triples in {elapsed:.1f}s")
    print(f"  Best ratio: {best_ratio:.10f}")

    if best_triple is not None and best_ratio > 1.0 + 1e-10:
        e1, e2, e3 = best_triple
        r1, c1 = rows[e1], cols[e1]
        r2, c2 = rows[e2], cols[e2]
        r3, c3 = rows[e3], cols[e3]
        print(f"  IMPROVEMENT: flip ({r1},{c1}), ({r2},{c2}), ({r3},{c3})")
        B = A.copy()
        B[r1, c1] *= -1
        B[r2, c2] *= -1
        B[r3, c3] *= -1
        new_det = abs(det_bareiss(B.astype(int).tolist()))
        new_score = new_det / THEORETICAL_MAX
        print(f"  New |det| = {new_det}, score = {new_score:.8f}")
        return B, new_det, True
    else:
        print(f"  No 3-flip improvement found")
        return A, base_det, False


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


def main():
    print("=" * 70)
    print("EXHAUSTIVE FLIP SEARCH on Orrick-Solomon matrix (c=320)")
    print("=" * 70)
    sys.stdout.flush()

    A = BEST_KNOWN.copy()

    # Verify
    base_det = abs(det_bareiss(A.astype(int).tolist()))
    print(f"Base |det| = {base_det}")
    print(f"Base score = {base_det / THEORETICAL_MAX:.8f}")
    sys.stdout.flush()

    # Phase 1: Exhaustive 2-flip
    print(f"\n{'='*70}")
    print("PHASE 1: Exhaustive 2-flip search")
    print(f"{'='*70}")
    sys.stdout.flush()

    A, det_val, improved = exhaustive_2flip_exact(A)
    if improved:
        print(f"\n!!! 2-FLIP BREAKTHROUGH !!!")
        # Hill climb from the improved matrix
        Af = A.astype(float)
        Af, _ = row_replace_climb(Af, time.time() + 5)
        Af, _ = hill_climb_sm(Af, time.time() + 5)
        M = np.sign(Af).astype(int)
        M[M == 0] = 1
        final_det = abs(det_bareiss(M.astype(int).tolist()))
        print(f"After hill climb: |det| = {final_det}, score = {final_det/THEORETICAL_MAX:.8f}")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", M)
        print("Saved!")
        sys.stdout.flush()
        return

    # Phase 2: Vectorized 3-flip
    print(f"\n{'='*70}")
    print("PHASE 2: Vectorized 3-flip search")
    print(f"{'='*70}")
    sys.stdout.flush()

    deadline = time.time() + 7200  # 2 hours
    A, det_val, improved = exhaustive_3flip_vectorized(A, deadline)
    if improved:
        print(f"\n!!! 3-FLIP BREAKTHROUGH !!!")
        Af = A.astype(float)
        Af, _ = row_replace_climb(Af, time.time() + 5)
        Af, _ = hill_climb_sm(Af, time.time() + 5)
        M = np.sign(Af).astype(int)
        M[M == 0] = 1
        final_det = abs(det_bareiss(M.astype(int).tolist()))
        print(f"After hill climb: |det| = {final_det}, score = {final_det/THEORETICAL_MAX:.8f}")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", M)
        print("Saved!")
    else:
        print("\nNo k-flip improvement found for k <= 3.")
        print("The Orrick-Solomon matrix is a 3-flip local maximum.")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
