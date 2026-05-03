"""
2-entry modification search on Solomon Gram matrix for n=29.
Searches all pairs of off-diagonal modifications for Gram matrices with:
  1. det(G_new) > det(G_old)
  2. det(G_new) is a perfect square
  3. G_new is positive definite
"""
import numpy as np
import math
import time
import sys
from itertools import combinations

# ---- Bareiss exact integer determinant ----
def det_bareiss(A):
    n = len(A)
    M = [list(row) for row in A]
    sign = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            found = False
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    sign *= -1
                    found = True
                    break
            if not found:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return sign * M[-1][-1]

# ---- Build Solomon Gram ----
def build_solomon_gram():
    G = np.ones((29, 29), dtype=np.int64)
    np.fill_diagonal(G, 29)
    for i in range(3):
        for j in range(3, 6):
            G[i, j] = 5
            G[j, i] = 5
    for j in range(7, 11):
        G[6, j] = -3
        G[j, 6] = -3
    return G

# ---- Main search ----
def main():
    G0 = build_solomon_gram()

    # Compute reference det
    known_det = (320 * 7**12 * 2**28)**2
    log_known = np.log(float(known_det))
    print(f"Solomon det = {known_det}")
    print(f"log(det) = {log_known:.6f}")

    # Verify with slogdet
    sign0, logdet0 = np.linalg.slogdet(G0.astype(np.float64))
    print(f"slogdet: sign={sign0}, logdet={logdet0:.6f}")
    assert abs(logdet0 - log_known) < 0.01, "Det mismatch!"

    # Also verify with Bareiss
    print("Verifying with Bareiss (this takes a moment)...")
    t0 = time.time()
    exact_det = det_bareiss(G0.astype(int).tolist())
    print(f"Bareiss det = {exact_det} (took {time.time()-t0:.1f}s)")
    assert exact_det == known_det, f"Bareiss mismatch: {exact_det} vs {known_det}"
    print("Bareiss verified!\n")

    # Enumerate all off-diagonal pairs (i < j)
    n = 29
    pairs = []
    pair_values = []  # current value
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            pair_values.append(int(G0[i, j]))

    print(f"Total off-diagonal pairs: {len(pairs)}")

    # Allowed off-diagonal values (≡ 1 mod 4)
    allowed = [-7, -3, 1, 5, 9]

    # For each pair, what are the possible NEW values (excluding current)?
    pair_new_vals = []
    for idx, (i, j) in enumerate(pairs):
        cur = pair_values[idx]
        new_vals = [v for v in allowed if v != cur]
        pair_new_vals.append(new_vals)

    # Precompute inverse of G0 for fast rank-2 update det formula
    # det(G + U) where U is a rank-2 symmetric update can use matrix determinant lemma
    # For changing entries (i1,j1) and (i2,j2):
    #   G_new = G0 + d1*(e_{i1}e_{j1}^T + e_{j1}e_{i1}^T) + d2*(e_{i2}e_{j2}^T + e_{j2}e_{i2}^T)
    # This is a rank-4 update at most, but often rank-2 or rank-4
    # Use: det(A + UV^T) = det(A) * det(I + V^T A^{-1} U)

    G0_float = G0.astype(np.float64)
    G0_inv = np.linalg.inv(G0_float)
    det_G0 = np.linalg.det(G0_float)

    print(f"det(G0) via numpy = {det_G0:.6e}")
    print()

    # For a 2-entry modification, we change entries (i1,j1) by d1 and (i2,j2) by d2.
    # The update matrix is:
    #   Delta = d1*(e_{i1}e_{j1}^T + e_{j1}e_{i1}^T) + d2*(e_{i2}e_{j2}^T + e_{j2}e_{i2}^T)
    # This has rank at most 4. We can write Delta = U @ V^T where U, V are 29x4.
    # det(G0 + Delta) = det(G0) * det(I_4 + V^T @ G0_inv @ U)

    # Strategy: Use the rank-update formula for speed.
    # For each pair of modifications, build the 4x4 matrix and compute its det.

    total_combos = sum(len(pair_new_vals[a]) * len(pair_new_vals[b])
                       for a, b in combinations(range(len(pairs)), 2))
    print(f"Total 2-entry combinations: {total_combos}")

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"

    candidates = []
    checked = 0
    det_higher_count = 0
    exact_checked = 0
    perfect_sq_count = 0

    start_time = time.time()

    # Precompute G0_inv columns for fast access
    # For entry (i,j) changed by delta, the update vectors are:
    #   u1 = delta * e_i, v1 = e_j  and  u2 = delta * e_j, v2 = e_i
    # Combined: U = [e_i, e_j], V = [delta*e_j, delta*e_i]
    # So for single entry change: U = [e_i, e_j], V^T = delta * [[0,1],[1,0]] applied to [e_i, e_j]^T

    # Actually let's just use the 4x4 formula directly.
    # For 2 entry changes (i1,j1,d1) and (i2,j2,d2):
    # U = n x 4 matrix: columns are e_{i1}, e_{j1}, e_{i2}, e_{j2}
    # V^T = 4 x n matrix defined so that U @ V^T = Delta
    # Delta = d1*(e_{i1}e_{j1}^T + e_{j1}e_{i1}^T) + d2*(e_{i2}e_{j2}^T + e_{j2}e_{i2}^T)
    #
    # Let U = [e_{i1}, e_{j1}, e_{i2}, e_{j2}]  (29 x 4)
    # Then V^T should satisfy U @ V^T = Delta
    # V^T = [[0, d1, 0, 0],   <- multiplied by e_{i1}: d1*e_{j1}^T row
    #         [d1, 0, 0, 0],   <- multiplied by e_{j1}: d1*e_{i1}^T row
    #         [0, 0, 0, d2],   <- multiplied by e_{i2}: d2*e_{j2}^T row
    #         [0, 0, d2, 0]]   <- multiplied by e_{j2}: d2*e_{i2}^T row
    # Wait, that's not right. Let me think again.
    #
    # U @ V^T where U[:,0]=e_{i1}, U[:,1]=e_{j1}, U[:,2]=e_{i2}, U[:,3]=e_{j2}
    # and V^T[0,:] = d1 * e_{j1}^T  =>  U[:,0] @ V^T[0,:] = e_{i1} * d1 * e_{j1}^T = d1 * e_{i1}e_{j1}^T  ✓
    #     V^T[1,:] = d1 * e_{i1}^T  =>  e_{j1} * d1 * e_{i1}^T = d1 * e_{j1}e_{i1}^T  ✓
    #     V^T[2,:] = d2 * e_{j2}^T  =>  e_{i2} * d2 * e_{j2}^T  ✓
    #     V^T[3,:] = d2 * e_{i2}^T  =>  e_{j2} * d2 * e_{i2}^T  ✓
    #
    # So V[:,0] = d1 * e_{j1}, V[:,1] = d1 * e_{i1}, V[:,2] = d2 * e_{j2}, V[:,3] = d2 * e_{i2}

    # det(G0 + U V^T) = det(G0) * det(I_4 + V^T G0_inv U)
    #
    # The 4x4 matrix M = I_4 + V^T @ G0_inv @ U
    # V^T @ G0_inv is 4 x 29, then @ U is 4 x 4
    #
    # V^T @ G0_inv @ U where:
    # Row 0 of V^T = d1 * e_{j1}^T  =>  (V^T @ G0_inv)[0,:] = d1 * G0_inv[j1, :]
    # Row 1 of V^T = d1 * e_{i1}^T  =>  d1 * G0_inv[i1, :]
    # Row 2 of V^T = d2 * e_{j2}^T  =>  d2 * G0_inv[j2, :]
    # Row 3 of V^T = d2 * e_{i2}^T  =>  d2 * G0_inv[i2, :]
    #
    # Then @ U (columns e_{i1}, e_{j1}, e_{i2}, e_{j2}):
    # (V^T G0_inv U)[r, c] = (V^T G0_inv)[r, col_index_of_U_col_c]
    #
    # So the 4x4 matrix K = V^T @ G0_inv @ U is:
    # K[0,0] = d1 * G0_inv[j1, i1]
    # K[0,1] = d1 * G0_inv[j1, j1]
    # K[0,2] = d1 * G0_inv[j1, i2]
    # K[0,3] = d1 * G0_inv[j1, j2]
    # K[1,0] = d1 * G0_inv[i1, i1]
    # K[1,1] = d1 * G0_inv[i1, j1]
    # K[1,2] = d1 * G0_inv[i1, i2]
    # K[1,3] = d1 * G0_inv[i1, j2]
    # K[2,0] = d2 * G0_inv[j2, i1]
    # K[2,1] = d2 * G0_inv[j2, j1]
    # K[2,2] = d2 * G0_inv[j2, i2]
    # K[2,3] = d2 * G0_inv[j2, j2]
    # K[3,0] = d2 * G0_inv[i2, i1]
    # K[3,1] = d2 * G0_inv[i2, j1]
    # K[3,2] = d2 * G0_inv[i2, i2]
    # K[3,3] = d2 * G0_inv[i2, j2]
    #
    # det_ratio = det(I_4 + K)
    # det_new = det_G0 * det_ratio

    # Precompute G0_inv as a regular array for fast indexing
    Ginv = G0_inv  # 29x29

    # For efficiency, precompute all needed Ginv entries
    # We need Ginv[a, b] for a, b in {i1, j1, i2, j2}

    log_det_G0 = logdet0
    threshold_ratio = 1.0  # We want det_new > det_old, so ratio > 1
    # For exact check, use slightly higher threshold to filter noise
    exact_threshold_ratio = 0.999  # be generous to not miss due to float error

    print(f"Starting 2-entry search...")
    print(f"Will check exact det for candidates with ratio > {exact_threshold_ratio}")
    sys.stdout.flush()

    batch_size = 100000
    last_report = start_time

    # Iterate over all pairs of pair indices
    npairs = len(pairs)

    for a in range(npairs):
        i1, j1 = pairs[a]
        cur1 = pair_values[a]
        new_vals_a = pair_new_vals[a]

        # Precompute Ginv rows for i1, j1
        Ginv_i1 = Ginv[i1]
        Ginv_j1 = Ginv[j1]

        for b in range(a + 1, npairs):
            i2, j2 = pairs[b]
            cur2 = pair_values[b]
            new_vals_b = pair_new_vals[b]

            # Precompute the 16 Ginv entries we need
            Ginv_i2 = Ginv[i2]
            Ginv_j2 = Ginv[j2]

            # Base entries (indexed by {i1,j1,i2,j2} x {i1,j1,i2,j2})
            g_j1_i1 = Ginv_j1[i1]
            g_j1_j1 = Ginv_j1[j1]
            g_j1_i2 = Ginv_j1[i2]
            g_j1_j2 = Ginv_j1[j2]
            g_i1_i1 = Ginv_i1[i1]
            g_i1_j1 = Ginv_i1[j1]
            g_i1_i2 = Ginv_i1[i2]
            g_i1_j2 = Ginv_i1[j2]
            g_j2_i1 = Ginv_j2[i1]
            g_j2_j1 = Ginv_j2[j1]
            g_j2_i2 = Ginv_j2[i2]
            g_j2_j2 = Ginv_j2[j2]
            g_i2_i1 = Ginv_i2[i1]
            g_i2_j1 = Ginv_i2[j1]
            g_i2_i2 = Ginv_i2[i2]
            g_i2_j2 = Ginv_i2[j2]

            for v1 in new_vals_a:
                d1 = v1 - cur1

                # Build rows of K that depend on d1
                k00 = d1 * g_j1_i1
                k01 = d1 * g_j1_j1
                k02_base = d1 * g_j1_i2
                k03_base = d1 * g_j1_j2
                k10 = d1 * g_i1_i1
                k11 = d1 * g_i1_j1
                k12_base = d1 * g_i1_i2
                k13_base = d1 * g_i1_j2

                for v2 in new_vals_b:
                    d2 = v2 - cur2
                    checked += 1

                    # Build 4x4 matrix I + K
                    # M = I_4 + K
                    m00 = 1.0 + k00
                    m01 = k01
                    m02 = k02_base
                    m03 = k03_base
                    m10 = k10
                    m11 = 1.0 + k11
                    m12 = k12_base
                    m13 = k13_base
                    m20 = d2 * g_j2_i1
                    m21 = d2 * g_j2_j1
                    m22 = 1.0 + d2 * g_j2_i2
                    m23 = d2 * g_j2_j2
                    m30 = d2 * g_i2_i1
                    m31 = d2 * g_i2_j1
                    m32 = d2 * g_i2_i2
                    m33 = 1.0 + d2 * g_i2_j2

                    # Compute 4x4 determinant via cofactor expansion
                    # det = m00*(m11*(m22*m33-m23*m32) - m12*(m21*m33-m23*m31) + m13*(m21*m32-m22*m31))
                    #      - m01*(m10*(m22*m33-m23*m32) - m12*(m20*m33-m23*m30) + m13*(m20*m32-m22*m30))
                    #      + m02*(m10*(m21*m33-m23*m31) - m11*(m20*m33-m23*m30) + m13*(m20*m31-m21*m30))
                    #      - m03*(m10*(m21*m32-m22*m31) - m11*(m20*m32-m22*m30) + m12*(m20*m31-m21*m30))

                    # Sub-expressions
                    s0 = m22 * m33 - m23 * m32
                    s1 = m21 * m33 - m23 * m31
                    s2 = m21 * m32 - m22 * m31
                    s3 = m20 * m33 - m23 * m30
                    s4 = m20 * m32 - m22 * m30
                    s5 = m20 * m31 - m21 * m30

                    det_ratio = (m00 * (m11 * s0 - m12 * s1 + m13 * s2)
                               - m01 * (m10 * s0 - m12 * s3 + m13 * s4)
                               + m02 * (m10 * s1 - m11 * s3 + m13 * s5)
                               - m03 * (m10 * s2 - m11 * s4 + m12 * s5))

                    if det_ratio > exact_threshold_ratio:
                        det_higher_count += 1

                        # Build modified Gram and check exact det
                        G_new = G0.copy()
                        G_new[i1, j1] = v1
                        G_new[j1, i1] = v1
                        G_new[i2, j2] = v2
                        G_new[j2, i2] = v2

                        # Quick positive-definite check via Cholesky
                        try:
                            np.linalg.cholesky(G_new.astype(np.float64))
                        except np.linalg.LinAlgError:
                            continue

                        # Exact determinant
                        exact_det_new = det_bareiss(G_new.astype(int).tolist())
                        exact_checked += 1

                        if exact_det_new <= known_det:
                            continue

                        # Check perfect square
                        sqrt_det = math.isqrt(exact_det_new)
                        is_perfect_sq = (sqrt_det * sqrt_det == exact_det_new)

                        ratio = exact_det_new / known_det

                        result_str = (f"CANDIDATE: ({i1},{j1}): {cur1}->{v1}, ({i2},{j2}): {cur2}->{v2} | "
                                     f"det={exact_det_new} | ratio={ratio:.6f} | "
                                     f"perfect_sq={is_perfect_sq}")

                        if is_perfect_sq:
                            perfect_sq_count += 1
                            result_str = "*** PERFECT SQUARE *** " + result_str
                            print(result_str)
                            candidates.append({
                                'pairs': [(i1, j1, cur1, v1), (i2, j2, cur2, v2)],
                                'det': exact_det_new,
                                'sqrt_det': sqrt_det,
                                'ratio': ratio,
                            })
                        else:
                            # Still report it if det is notably higher
                            if ratio > 1.01:
                                print(f"  (non-square) {result_str}")

                    # Progress reporting
                    if checked % batch_size == 0:
                        now = time.time()
                        elapsed = now - start_time
                        rate = checked / elapsed
                        eta = (total_combos - checked) / rate if rate > 0 else 0
                        pct = 100.0 * checked / total_combos
                        print(f"  [{pct:5.1f}%] checked={checked}/{total_combos} | "
                              f"det_higher={det_higher_count} | exact_checked={exact_checked} | "
                              f"perfect_sq={perfect_sq_count} | "
                              f"rate={rate:.0f}/s | ETA={eta:.0f}s | pair_a={a}/{npairs}")
                        sys.stdout.flush()

    elapsed = time.time() - start_time

    # Write results
    with open(results_file, "w") as f:
        f.write(f"Solomon Gram 2-entry modification search\n")
        f.write(f"========================================\n\n")
        f.write(f"Solomon det = {known_det}\n")
        f.write(f"Total combinations searched: {checked} / {total_combos}\n")
        f.write(f"Candidates with det > det_old (approx): {det_higher_count}\n")
        f.write(f"Exact determinants computed: {exact_checked}\n")
        f.write(f"Perfect square determinants found: {perfect_sq_count}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")

        if candidates:
            f.write("PERFECT SQUARE CANDIDATES:\n")
            f.write("-" * 60 + "\n")
            for c in candidates:
                f.write(f"Pairs: {c['pairs']}\n")
                f.write(f"det = {c['det']}\n")
                f.write(f"sqrt(det) = {c['sqrt_det']}\n")
                f.write(f"ratio = {c['ratio']:.6f}\n\n")
        else:
            f.write("No perfect square determinants found in 2-entry search.\n")

    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"Total checked: {checked}")
    print(f"Det higher (approx): {det_higher_count}")
    print(f"Exact checked: {exact_checked}")
    print(f"Perfect squares: {perfect_sq_count}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Results saved to {results_file}")

    return candidates

if __name__ == "__main__":
    candidates = main()
