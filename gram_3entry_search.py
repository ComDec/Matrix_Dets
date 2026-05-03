"""
3-entry modification search on Solomon Gram matrix for n=29.
Uses rank-6 matrix determinant lemma for fast screening,
then Bareiss for exact verification.

Strategy:
  Pass 1: Best 2-entry bases + sweep all 3rd entries
  Pass 2: All 3-entry with 2+ special entries (2M combos)
  Pass 3: All 3-entry with 1 special entry + 2 non-special (68M, vectorized)
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

def compute_6x6_det_ratio(Ginv, i1, j1, d1, i2, j2, d2, i3, j3, d3):
    """
    Compute det(G + Delta)/det(G) for a rank-6 symmetric update
    changing entries (i1,j1), (i2,j2), (i3,j3) by d1, d2, d3.

    Delta = sum_k dk * (e_{ik} e_{jk}^T + e_{jk} e_{ik}^T)
    This is rank-6 update: Delta = U @ V^T
    where U = [e_{i1}, e_{j1}, e_{i2}, e_{j2}, e_{i3}, e_{j3}] (n x 6)
    and V columns: d1*e_{j1}, d1*e_{i1}, d2*e_{j2}, d2*e_{i2}, d3*e_{j3}, d3*e_{i3}

    det_ratio = det(I_6 + V^T @ Ginv @ U)
    """
    idx = [i1, j1, i2, j2, i3, j3]
    # Build the 6x6 matrix M = I_6 + V^T @ Ginv @ U
    # V^T @ Ginv is 6 x n: row k = d_factor[k] * Ginv[v_idx[k], :]
    # Then @ U extracts columns at idx
    # V^T rows: d1*e_{j1}^T, d1*e_{i1}^T, d2*e_{j2}^T, d2*e_{i2}^T, d3*e_{j3}^T, d3*e_{i3}^T
    d_factors = [d1, d1, d2, d2, d3, d3]
    v_rows = [j1, i1, j2, i2, j3, i3]  # which Ginv row each V^T row selects

    M = np.eye(6)
    for r in range(6):
        for c in range(6):
            M[r, c] += d_factors[r] * Ginv[v_rows[r], idx[c]]

    return np.linalg.det(M)


def main():
    G0 = build_solomon_gram()
    known_det = (320 * 7**12 * 2**28)**2
    log_known = np.log(float(known_det))

    G0_float = G0.astype(np.float64)
    Ginv = np.linalg.inv(G0_float)

    sign0, logdet0 = np.linalg.slogdet(G0_float)
    print(f"Solomon det = {known_det}")
    print(f"slogdet check: {abs(logdet0 - log_known) < 0.01}")

    n = 29
    pairs = []
    pair_values = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            pair_values.append(int(G0[i, j]))

    allowed = [-7, -3, 1, 5, 9]
    pair_new_vals = []
    for idx_p in range(len(pairs)):
        cur = pair_values[idx_p]
        pair_new_vals.append([v for v in allowed if v != cur])

    # Identify special pairs
    special_5_idx = []  # pair indices with value 5
    special_m3_idx = []  # pair indices with value -3
    normal_idx = []  # pair indices with value 1
    for idx_p, (i, j) in enumerate(pairs):
        v = pair_values[idx_p]
        if v == 5:
            special_5_idx.append(idx_p)
        elif v == -3:
            special_m3_idx.append(idx_p)
        else:
            normal_idx.append(idx_p)

    special_idx = special_5_idx + special_m3_idx
    print(f"Special pairs: {len(special_idx)} (5-entries: {len(special_5_idx)}, -3-entries: {len(special_m3_idx)})")
    print(f"Normal pairs: {len(normal_idx)}")

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"

    candidates = []
    total_checked = 0
    total_det_higher = 0
    total_exact = 0
    total_perfect_sq = 0

    start_time = time.time()

    def check_and_record(i1, j1, v1, i2, j2, v2, i3, j3, v3):
        nonlocal total_checked, total_det_higher, total_exact, total_perfect_sq
        total_checked += 1

        cur1 = int(G0[i1, j1])
        cur2 = int(G0[i2, j2])
        cur3 = int(G0[i3, j3])
        d1 = v1 - cur1
        d2 = v2 - cur2
        d3 = v3 - cur3

        # Fast approximate det ratio using 6x6 determinant
        det_ratio = compute_6x6_det_ratio(Ginv, i1, j1, d1, i2, j2, d2, i3, j3, d3)

        if det_ratio <= 0.999:
            return

        total_det_higher += 1

        # Build modified Gram
        G_new = G0.copy()
        G_new[i1, j1] = v1; G_new[j1, i1] = v1
        G_new[i2, j2] = v2; G_new[j2, i2] = v2
        G_new[i3, j3] = v3; G_new[j3, i3] = v3

        # Positive definite check
        try:
            np.linalg.cholesky(G_new.astype(np.float64))
        except np.linalg.LinAlgError:
            return

        # Exact det
        exact_det = det_bareiss(G_new.astype(int).tolist())
        total_exact += 1

        if exact_det <= known_det:
            return

        ratio = exact_det / known_det
        sqrt_det = math.isqrt(exact_det)
        is_sq = (sqrt_det * sqrt_det == exact_det)

        if is_sq:
            total_perfect_sq += 1
            desc = (f"*** PERFECT SQUARE *** ({i1},{j1}): {cur1}->{v1}, "
                   f"({i2},{j2}): {cur2}->{v2}, ({i3},{j3}): {cur3}->{v3} | "
                   f"det={exact_det} | ratio={ratio:.6f} | sqrt={sqrt_det}")
            print(desc)
            candidates.append({
                'mods': [(i1, j1, cur1, v1), (i2, j2, cur2, v2), (i3, j3, cur3, v3)],
                'det': exact_det,
                'sqrt_det': sqrt_det,
                'ratio': ratio,
            })
        elif ratio > 1.03:
            print(f"  (non-sq, high) ({i1},{j1}): {cur1}->{v1}, "
                  f"({i2},{j2}): {cur2}->{v2}, ({i3},{j3}): {cur3}->{v3} | "
                  f"ratio={ratio:.6f}")

    # ======== PASS 1: Best 2-entry bases + sweep 3rd entry ========
    print("\n" + "="*60)
    print("PASS 1: Best 2-entry bases + sweep all 3rd entries")
    print("="*60)

    # The best 2-entry modifications (by det ratio):
    # 1.026719: changing two -3 entries to 1 (from the C(4,2)=6 pairs among {(6,7),(6,8),(6,9),(6,10)})
    # 1.024995: one 1->5 in {0,1,2}x{0,1,2\{same}}, one 5->1 in other-row x {3,4,5}
    # 1.018297: one 1->5, one 5->1 in same-row

    best_2entry_bases = []

    # Type 1: two -3 -> 1
    m3_pairs = [(6, 7), (6, 8), (6, 9), (6, 10)]
    for a_idx in range(len(m3_pairs)):
        for b_idx in range(a_idx + 1, len(m3_pairs)):
            best_2entry_bases.append((m3_pairs[a_idx], 1, m3_pairs[b_idx], 1))

    # Type 2: 1->5 between {0,1,2} rows, 5->1 in other row
    for i_from in range(3):
        for j_from in range(i_from + 1, 3):
            # (i_from, j_from): 1->5
            other = 3 - i_from - j_from  # the third element of {0,1,2}
            for j_to in range(3, 6):
                # (other, j_to): 5->1
                best_2entry_bases.append(((i_from, j_from), 5, (other, j_to), 1))

    # Type 3: 1->5 within same row, 5->1
    for i in range(3):
        other_in_group = [x for x in range(3) if x != i]
        for j_new5 in other_in_group:
            for j_rem5 in range(3, 6):
                best_2entry_bases.append(((i, j_new5), 5, (i, j_rem5), 1))

    # Also add: changing -3 entries to other values
    for j in range(7, 11):
        for v in [-7, 5, 9]:  # not 1 (already covered), not -3 (current)
            for j2 in range(j + 1, 11):
                for v2 in [-7, 1, 5, 9]:
                    best_2entry_bases.append(((6, j), v, (6, j2), v2))

    print(f"Base 2-entry configs: {len(best_2entry_bases)}")

    pass1_checked = 0
    for base in best_2entry_bases:
        (bi1, bj1), bv1, (bi2, bj2), bv2 = base

        for c_idx in range(len(pairs)):
            ci, cj = pairs[c_idx]
            # Skip if same as base entries
            if (ci, cj) == (bi1, bj1) or (ci, cj) == (bi2, bj2):
                continue

            cur_c = pair_values[c_idx]
            for vc in pair_new_vals[c_idx]:
                check_and_record(bi1, bj1, bv1, bi2, bj2, bv2, ci, cj, vc)
                pass1_checked += 1

                if pass1_checked % 500000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Pass1: {pass1_checked} checked, {total_det_higher} higher, "
                          f"{total_exact} exact, {total_perfect_sq} perfect_sq, "
                          f"{elapsed:.0f}s elapsed")
                    sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"Pass 1 done: {pass1_checked} checked, {total_det_higher} higher, "
          f"{total_exact} exact, {total_perfect_sq} perfect_sq, {elapsed:.0f}s")

    # ======== PASS 2: All 3-entry with 3 special entries ========
    print("\n" + "="*60)
    print("PASS 2: All 3-entry with 3 special entries")
    print("="*60)

    pass2_checked = 0
    for combo in combinations(special_idx, 3):
        a, b, c = combo
        i1, j1 = pairs[a]
        i2, j2 = pairs[b]
        i3, j3 = pairs[c]

        for v1 in pair_new_vals[a]:
            for v2 in pair_new_vals[b]:
                for v3 in pair_new_vals[c]:
                    check_and_record(i1, j1, v1, i2, j2, v2, i3, j3, v3)
                    pass2_checked += 1

    elapsed = time.time() - start_time
    print(f"Pass 2 done: {pass2_checked} checked, {total_det_higher} higher, "
          f"{total_exact} exact, {total_perfect_sq} perfect_sq, {elapsed:.0f}s")

    # ======== PASS 3: All 3-entry with 2 special + 1 normal ========
    print("\n" + "="*60)
    print("PASS 3: 2 special + 1 normal entry")
    print("="*60)

    pass3_checked = 0
    for s_combo in combinations(special_idx, 2):
        sa, sb = s_combo
        i1, j1 = pairs[sa]
        i2, j2 = pairs[sb]

        for na in normal_idx:
            i3, j3 = pairs[na]

            for v1 in pair_new_vals[sa]:
                for v2 in pair_new_vals[sb]:
                    for v3 in pair_new_vals[na]:
                        check_and_record(i1, j1, v1, i2, j2, v2, i3, j3, v3)
                        pass3_checked += 1

            if pass3_checked % 500000 == 0:
                elapsed = time.time() - start_time
                print(f"  Pass3: {pass3_checked} checked, {total_det_higher} higher, "
                      f"{total_exact} exact, {total_perfect_sq} perfect_sq, "
                      f"{elapsed:.0f}s elapsed")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"Pass 3 done: {pass3_checked} checked, {total_det_higher} higher, "
          f"{total_exact} exact, {total_perfect_sq} perfect_sq, {elapsed:.0f}s")

    # ======== PASS 4: 1 special + 2 normal (large, vectorized approach) ========
    print("\n" + "="*60)
    print("PASS 4: 1 special + 2 normal entries (vectorized)")
    print("="*60)

    # This is ~68M combinations. We need to be faster.
    # Strategy: For each special entry modification, precompute the rank-2 updated inverse,
    # then use rank-4 update for the 2 normal entries (same as 2-entry search on updated matrix).

    # Actually, let's just use the same approach but be smart about batching.
    # The 6x6 det computation per combo is the bottleneck.
    # With numpy, we can vectorize: for fixed special entry + first normal entry,
    # sweep all second normal entries in a batch.

    pass4_checked = 0
    pass4_start = time.time()

    for s_idx in special_idx:
        si, sj = pairs[s_idx]

        for sv in pair_new_vals[s_idx]:
            sd = sv - pair_values[s_idx]

            # For this special modification, compute the rank-2 updated matrix
            # G1 = G0 + sd * (e_si e_sj^T + e_sj e_si^T)
            # Use Sherman-Morrison-Woodbury to get G1_inv
            # Then for each pair of normal mods, do a rank-4 update

            # But for simplicity and correctness, let's compute G1 and its inverse once
            G1 = G0_float.copy()
            G1[si, sj] += sd
            G1[sj, si] += sd

            sign1, logdet1 = np.linalg.slogdet(G1)
            if sign1 <= 0:
                # Not positive definite after first mod
                pass4_checked += len(list(combinations(normal_idx, 2))) * 16
                continue

            G1_inv = np.linalg.inv(G1)

            # Now sweep all pairs of normal entries
            # This is identical to the 2-entry search but on G1
            nnorm = len(normal_idx)

            for na_pos in range(nnorm):
                na = normal_idx[na_pos]
                i2, j2 = pairs[na]
                cur2 = pair_values[na]

                Ginv_i2 = G1_inv[i2]
                Ginv_j2 = G1_inv[j2]

                for nb_pos in range(na_pos + 1, nnorm):
                    nb = normal_idx[nb_pos]
                    i3, j3 = pairs[nb]
                    cur3 = pair_values[nb]

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

                    for v2 in pair_new_vals[na]:
                        d2 = v2 - cur2

                        k00 = d2 * g_j2_i2
                        k01 = d2 * g_j2_j2
                        k10 = d2 * g_i2_i2
                        k11 = d2 * g_i2_j2

                        for v3 in pair_new_vals[nb]:
                            d3 = v3 - cur3
                            pass4_checked += 1

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

                            det_ratio = (m00 * (m11 * s0 - m12 * s1 + m13 * s2)
                                       - m01 * (m10 * s0 - m12 * s3 + m13 * s4)
                                       + m02 * (m10 * s1 - m11 * s3 + m13 * s5)
                                       - m03 * (m10 * s2 - m11 * s4 + m12 * s5))

                            # det_ratio is relative to G1 (with one special mod)
                            # We need det_new / det_G0 > 1
                            # det_new = det_G1 * det_ratio
                            # We need det_G1 * det_ratio > det_G0
                            # i.e. det_ratio > det_G0 / det_G1 = 1/det_ratio_1
                            # But we check: overall det > known_det
                            # Actually, the total det = sign0 * exp(logdet0) for G0
                            # and det(G_new) = det(G1) * det_ratio = sign1 * exp(logdet1) * det_ratio
                            # We want det(G_new) > known_det
                            # So det_ratio > known_det / (sign1 * exp(logdet1))

                            # Simpler: det(G_new) / det(G0) = (det(G1)/det(G0)) * det_ratio
                            # = exp(logdet1 - logdet0) * det_ratio

                            overall_ratio = np.exp(logdet1 - logdet0) * det_ratio

                            if overall_ratio <= 0.999:
                                continue

                            total_det_higher += 1

                            G_new = G0.copy()
                            G_new[si, sj] = sv; G_new[sj, si] = sv
                            G_new[i2, j2] = v2; G_new[j2, i2] = v2
                            G_new[i3, j3] = v3; G_new[j3, i3] = v3

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
                                print(f"*** PERFECT SQUARE *** ({si},{sj}): {pair_values[s_idx]}->{sv}, "
                                      f"({i2},{j2}): {cur2}->{v2}, ({i3},{j3}): {cur3}->{v3} | "
                                      f"det={exact_det} | ratio={ratio_exact:.6f} | sqrt={sqrt_det}")
                                candidates.append({
                                    'mods': [(si, sj, pair_values[s_idx], sv),
                                             (i2, j2, cur2, v2),
                                             (i3, j3, cur3, v3)],
                                    'det': exact_det,
                                    'sqrt_det': sqrt_det,
                                    'ratio': ratio_exact,
                                })
                            elif ratio_exact > 1.04:
                                print(f"  (non-sq, high) ({si},{sj})->{sv}, ({i2},{j2})->{v2}, "
                                      f"({i3},{j3})->{v3} | ratio={ratio_exact:.6f}")

                if pass4_checked % 2000000 == 0:
                    elapsed = time.time() - pass4_start
                    rate = pass4_checked / elapsed if elapsed > 0 else 0
                    total_expected = 13 * 4 * (393 * 392 // 2) * 16  # approximate
                    pct = 100.0 * pass4_checked / max(total_expected, 1)
                    print(f"  Pass4: {pass4_checked/1e6:.1f}M checked, "
                          f"{total_det_higher} higher, {total_exact} exact, "
                          f"{total_perfect_sq} perfect_sq, rate={rate:.0f}/s, "
                          f"~{pct:.1f}% done")
                    sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nPass 4 done: {pass4_checked} checked, {total_det_higher} higher, "
          f"{total_exact} exact, {total_perfect_sq} perfect_sq, {elapsed:.0f}s")

    # ======== Write results ========
    print(f"\n{'='*60}")
    print(f"ALL SEARCHES COMPLETE")
    print(f"Total checked: {total_checked + pass4_checked}")
    print(f"Det higher: {total_det_higher}")
    print(f"Exact checked: {total_exact}")
    print(f"Perfect squares: {total_perfect_sq}")
    print(f"Time: {elapsed:.1f}s")

    with open(results_file, "a") as f:
        f.write(f"\n\n3-entry modification search\n")
        f.write(f"{'='*60}\n")
        f.write(f"Pass 1 (best 2-entry + sweep): included\n")
        f.write(f"Pass 2 (3 special): included\n")
        f.write(f"Pass 3 (2 special + 1 normal): included\n")
        f.write(f"Pass 4 (1 special + 2 normal): included\n")
        f.write(f"Total checked: {total_checked + pass4_checked}\n")
        f.write(f"Det higher: {total_det_higher}\n")
        f.write(f"Exact checked: {total_exact}\n")
        f.write(f"Perfect squares: {total_perfect_sq}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")

        if candidates:
            f.write("PERFECT SQUARE CANDIDATES:\n")
            f.write("-" * 60 + "\n")
            for c in candidates:
                f.write(f"Mods: {c['mods']}\n")
                f.write(f"det = {c['det']}\n")
                f.write(f"sqrt(det) = {c['sqrt_det']}\n")
                f.write(f"ratio = {c['ratio']:.6f}\n\n")
        else:
            f.write("No perfect square determinants found in 3-entry search.\n")

    print(f"Results appended to {results_file}")
    return candidates

if __name__ == "__main__":
    main()
