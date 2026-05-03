"""
Fast Pass 4: 1 special + 2 normal entry modifications.
For each special mod, compute updated inverse G1_inv,
then run 2-entry search on G1 using fast inline 4x4 det.

This replaces the slow 6x6 numpy det in the original script.
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

    # Identify special and normal pairs
    special_5_idx = []
    special_m3_idx = []
    normal_idx = []
    for idx_p, (i, j) in enumerate(pairs):
        v = pair_values[idx_p]
        if v == 5:
            special_5_idx.append(idx_p)
        elif v == -3:
            special_m3_idx.append(idx_p)
        else:
            normal_idx.append(idx_p)

    special_idx = special_5_idx + special_m3_idx
    print(f"Special: {len(special_idx)}, Normal: {len(normal_idx)}")

    G0_float = G0.astype(np.float64)
    sign0, logdet0 = np.linalg.slogdet(G0_float)

    results_file = "/home/xiwang/project/AutoMath/tasks/matrix_det/gram_search_results.txt"
    candidates = []

    total_checked = 0
    total_higher = 0
    total_exact = 0
    total_perfect_sq = 0

    start_time = time.time()

    # For each special entry modification
    special_configs = []
    for s_idx in special_idx:
        si, sj = pairs[s_idx]
        for sv in pair_new_vals[s_idx]:
            special_configs.append((s_idx, si, sj, sv))

    print(f"Special configs: {len(special_configs)}")
    nnorm = len(normal_idx)
    combos_per_config = nnorm * (nnorm - 1) // 2 * 16  # C(393,2)*4*4
    total_expected = len(special_configs) * combos_per_config
    print(f"Expected total: {total_expected} = {total_expected/1e6:.1f}M")

    for cfg_num, (s_idx, si, sj, sv) in enumerate(special_configs):
        sd = sv - pair_values[s_idx]

        # Build G1 = G0 + sd*(e_si e_sj^T + e_sj e_si^T)
        G1 = G0_float.copy()
        G1[si, sj] += sd
        G1[sj, si] += sd

        sign1, logdet1 = np.linalg.slogdet(G1)
        if sign1 <= 0:
            total_checked += combos_per_config
            continue

        # The ratio det(G1)/det(G0)
        ratio_1_over_0 = np.exp(logdet1 - logdet0)

        G1_inv = np.linalg.inv(G1)

        # Run 2-entry search on G1 over normal pairs
        cfg_checked = 0
        cfg_higher = 0

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
                        total_checked += 1
                        cfg_checked += 1

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
                        cfg_higher += 1

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
                            desc = (f"*** PERFECT SQUARE *** ({si},{sj}): {pair_values[s_idx]}->{sv}, "
                                   f"({i2},{j2}): {cur2}->{v2}, ({i3},{j3}): {cur3}->{v3} | "
                                   f"det={exact_det} | ratio={ratio_exact:.6f} | sqrt={sqrt_det}")
                            print(desc)
                            candidates.append({
                                'mods': [(si, sj, pair_values[s_idx], sv),
                                         (i2, j2, cur2, v2), (i3, j3, cur3, v3)],
                                'det': exact_det,
                                'sqrt_det': sqrt_det,
                                'ratio': ratio_exact,
                            })

            if cfg_checked % 500000 == 0 and cfg_checked > 0:
                elapsed = time.time() - start_time
                rate = total_checked / elapsed
                pct = 100.0 * total_checked / total_expected
                eta = (total_expected - total_checked) / rate if rate > 0 else 0
                print(f"  [{pct:5.1f}%] config {cfg_num+1}/{len(special_configs)} | "
                      f"total={total_checked/1e6:.1f}M | higher={total_higher} | "
                      f"exact={total_exact} | sq={total_perfect_sq} | "
                      f"rate={rate:.0f}/s | ETA={eta/60:.0f}min")
                sys.stdout.flush()

        elapsed = time.time() - start_time
        rate = total_checked / elapsed if elapsed > 0 else 0
        pct = 100.0 * total_checked / total_expected
        eta = (total_expected - total_checked) / rate if rate > 0 else 0
        print(f"Config {cfg_num+1}/{len(special_configs)}: ({si},{sj})->{sv} | "
              f"cfg_higher={cfg_higher} | total={total_checked/1e6:.1f}M "
              f"[{pct:.1f}%] | rate={rate:.0f}/s | ETA={eta/60:.0f}min")
        sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PASS 4 (FAST) COMPLETE")
    print(f"Total checked: {total_checked} ({total_checked/1e6:.1f}M)")
    print(f"Higher det: {total_higher}")
    print(f"Exact checked: {total_exact}")
    print(f"Perfect squares: {total_perfect_sq}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    with open(results_file, "a") as f:
        f.write(f"\n\nPass 4 (fast): 1 special + 2 normal\n")
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
