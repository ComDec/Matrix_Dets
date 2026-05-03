"""
3-entry modification search: 3 normal entries (value=1) changed.
Since full search is ~700M, we use systematic sampling + focus on
modifications that give det_new > det_old.

Key observation: For normal entries (value=1), the possible changes are to -7, -3, 5, 9.
The delta values are -8, -4, 4, 8.

The most promising single-entry changes (from 2-entry search) are delta = +4 (1->5)
and delta = -4 (1->-3). Let's focus on these.
"""
import numpy as np
import math
import time
import sys
from itertools import combinations

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

def build_solomon_gram():
    G = np.ones((29, 29), dtype=np.int64)
    np.fill_diagonal(G, 29)
    for i in range(3):
        for j in range(3, 6):
            G[i, j] = 5; G[j, i] = 5
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G

def main():
    G0 = build_solomon_gram()
    known_det = (320 * 7**12 * 2**28)**2

    n = 29
    # Identify normal pairs (value = 1)
    normal_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if G0[i, j] == 1:
                normal_pairs.append((i, j))

    print(f"Normal pairs: {len(normal_pairs)}")

    G0_float = G0.astype(np.float64)
    Ginv = np.linalg.inv(G0_float)
    sign0, logdet0 = np.linalg.slogdet(G0_float)

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"
    candidates = []
    total_checked = 0
    total_higher = 0
    total_exact = 0
    total_perfect_sq = 0

    start_time = time.time()

    # Focus on delta combinations:
    # Most promising: 3 entries changed by delta in {-4, 4}
    # This gives 2^3 = 8 combinations per triplet of pairs
    # C(393, 3) * 8 = 10,116,648 * 8 = ~80M -- still large but manageable
    # with the 6x6 inline determinant

    # Actually let's be smarter. For 3 normal entries each changed by delta_k,
    # use the rank-6 update approach.
    # But first, let's check: how many triplets give det > known_det?
    # From the 2-entry search, only entries changed to 5 or -3 (delta=4 or -4) gave improvements.
    # And the best 2-entry ratio was ~1.043 (two fives added).

    # Let's focus: all 3 entries changed to 5 (delta=+4) or all to -3 (delta=-4)
    # or mixed. That's 8 combinations per triplet.

    # But C(393,3) = 10,116,648 -- still huge for a 6x6 det per combo.
    # Let's use the G1 approach: fix one entry change, update inverse, do 2-entry on rest.

    deltas_to_try = [(-4, -3), (4, 5)]  # (delta, new_value)

    nnorm = len(normal_pairs)
    print(f"Deltas to try: {deltas_to_try}")

    for d1_delta, d1_val in deltas_to_try:
        for a_pos in range(nnorm):
            i1, j1 = normal_pairs[a_pos]

            # Build G1 with first modification
            G1 = G0_float.copy()
            G1[i1, j1] += d1_delta
            G1[j1, i1] += d1_delta

            sign1, logdet1 = np.linalg.slogdet(G1)
            if sign1 <= 0:
                total_checked += (nnorm - a_pos - 1) * (nnorm - a_pos - 2) // 2 * 4
                continue

            ratio_1_over_0 = np.exp(logdet1 - logdet0)
            G1_inv = np.linalg.inv(G1)

            for b_pos in range(a_pos + 1, nnorm):
                i2, j2 = normal_pairs[b_pos]

                Ginv_i2 = G1_inv[i2]
                Ginv_j2 = G1_inv[j2]

                for c_pos in range(b_pos + 1, nnorm):
                    i3, j3 = normal_pairs[c_pos]

                    Ginv_i3 = G1_inv[i3]
                    Ginv_j3 = G1_inv[j3]

                    g_j2_i2 = Ginv_j2[i2]
                    g_j2_j2 = Ginv_j2[j2]
                    g_j2_i3 = Ginv_j2[i3]
                    g_j2_j3 = Ginv_j2[j3]
                    g_i2_i2 = Ginv_i2[i2]
                    g_i2_j2 = Ginv_i2[j2]
                    g_i2_i3 = Ginv_i2[i3]
                    g_i2_j3 = Ginv_i2[j3]
                    g_j3_i2 = Ginv_j3[i2]
                    g_j3_j2 = Ginv_j3[j2]
                    g_j3_i3 = Ginv_j3[i3]
                    g_j3_j3 = Ginv_j3[j3]
                    g_i3_i2 = Ginv_i3[i2]
                    g_i3_j2 = Ginv_i3[j2]
                    g_i3_i3 = Ginv_i3[i3]
                    g_i3_j3 = Ginv_i3[j3]

                    for d2_delta, d2_val in deltas_to_try:
                        d2 = d2_delta

                        k00 = d2 * g_j2_i2
                        k01 = d2 * g_j2_j2
                        k10 = d2 * g_i2_i2
                        k11 = d2 * g_i2_j2

                        for d3_delta, d3_val in deltas_to_try:
                            d3 = d3_delta
                            total_checked += 1

                            m00 = 1.0 + k00
                            m01 = k01
                            m02 = d2 * g_j2_i3
                            m03 = d2 * g_j2_j3
                            m10 = k10
                            m11 = 1.0 + k11
                            m12 = d2 * g_i2_i3
                            m13 = d2 * g_i2_j3
                            m20 = d3 * g_j3_i2
                            m21 = d3 * g_j3_j2
                            m22 = 1.0 + d3 * g_j3_i3
                            m23 = d3 * g_j3_j3
                            m30 = d3 * g_i3_i2
                            m31 = d3 * g_i3_j2
                            m32 = d3 * g_i3_i3
                            m33 = 1.0 + d3 * g_i3_j3

                            s0 = m22 * m33 - m23 * m32
                            s1 = m21 * m33 - m23 * m31
                            s2 = m21 * m32 - m22 * m31
                            s3 = m20 * m33 - m23 * m30
                            s4 = m20 * m32 - m22 * m30
                            s5 = m20 * m31 - m21 * m30

                            det_ratio_2 = (m00 * (m11 * s0 - m12 * s1 + m13 * s2)
                                         - m01 * (m10 * s0 - m12 * s3 + m13 * s4)
                                         + m02 * (m10 * s1 - m11 * s3 + m13 * s5)
                                         - m03 * (m10 * s2 - m11 * s4 + m12 * s5))

                            overall_ratio = ratio_1_over_0 * det_ratio_2

                            if overall_ratio <= 0.999:
                                continue

                            total_higher += 1

                            G_new = G0.copy()
                            G_new[i1, j1] = d1_val; G_new[j1, i1] = d1_val
                            G_new[i2, j2] = d2_val; G_new[j2, i2] = d2_val
                            G_new[i3, j3] = d3_val; G_new[j3, i3] = d3_val

                            # Positive definite check via Cholesky
                            try:
                                np.linalg.cholesky(G_new.astype(np.float64))
                            except np.linalg.LinAlgError:
                                continue

                            exact_det = det_bareiss(G_new.astype(int).tolist())
                            total_exact += 1

                            if exact_det <= known_det:
                                continue

                            ratio_exact = exact_det / known_det
                            sqrt_det = math.isqrt(exact_det)
                            is_sq = (sqrt_det * sqrt_det == exact_det)

                            if is_sq:
                                total_perfect_sq += 1
                                print(f"*** PERFECT SQUARE *** ({i1},{j1})->5, "
                                      f"({i2},{j2})->{d2_val}, ({i3},{j3})->{d3_val} | "
                                      f"det={exact_det} | ratio={ratio_exact:.6f} | sqrt={sqrt_det}")
                                candidates.append({
                                    'mods': [(i1, j1, 1, d1_val),
                                             (i2, j2, 1, d2_val),
                                             (i3, j3, 1, d3_val)],
                                    'det': exact_det,
                                    'sqrt_det': sqrt_det,
                                    'ratio': ratio_exact,
                                })

                if total_checked % 5000000 == 0 and total_checked > 0:
                    elapsed = time.time() - start_time
                    rate = total_checked / elapsed
                    # Rough estimate of total
                    total_est = 2 * nnorm * nnorm * nnorm * 4 // 6  # very rough
                    pct = min(100, 100.0 * total_checked / max(total_est, 1))
                    print(f"  [{pct:.0f}%] checked={total_checked/1e6:.1f}M | "
                          f"higher={total_higher} | exact={total_exact} | "
                          f"sq={total_perfect_sq} | rate={rate:.0f}/s | "
                          f"d1={d1_val},a={a_pos}/{nnorm}")
                    sys.stdout.flush()

            if a_pos % 50 == 0 and a_pos > 0:
                elapsed = time.time() - start_time
                rate = total_checked / elapsed if elapsed > 0 else 0
                print(f"  d1={d1_val}: a_pos={a_pos}/{nnorm} | "
                      f"total={total_checked/1e6:.1f}M | higher={total_higher} | "
                      f"exact={total_exact} | sq={total_perfect_sq} | "
                      f"rate={rate:.0f}/s | {elapsed:.0f}s elapsed")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"3-ENTRY NORMAL SEARCH COMPLETE (focused deltas)")
    print(f"Total checked: {total_checked} ({total_checked/1e6:.1f}M)")
    print(f"Higher det: {total_higher}")
    print(f"Exact checked: {total_exact}")
    print(f"Perfect squares: {total_perfect_sq}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    with open(results_file, "a") as f:
        f.write(f"\n\n3-entry normal search (focused deltas +-4)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total checked: {total_checked}\n")
        f.write(f"Higher det: {total_higher}\n")
        f.write(f"Exact checked: {total_exact}\n")
        f.write(f"Perfect squares: {total_perfect_sq}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")
        if candidates:
            f.write("PERFECT SQUARE CANDIDATES:\n")
            for c in candidates:
                f.write(f"Mods: {c['mods']}\n")
                f.write(f"det = {c['det']}\n")
                f.write(f"sqrt(det) = {c['sqrt_det']}\n")
                f.write(f"ratio = {c['ratio']:.6f}\n\n")
        else:
            f.write("No perfect square determinants found.\n")

    return candidates

if __name__ == "__main__":
    main()
