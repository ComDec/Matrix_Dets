"""Alternative constructions for the n=29 Hadamard maximal determinant problem.

Implements 5 different starting-matrix constructions, each followed by
row-replacement + Sherman-Morrison hill climbing + perturbation restarts.

Constructions:
  1. GF(27) Paley H28 from different irreducible polynomials
  2. Williamson-type H28 (order 28 = 4*7)
  3. Bordered H28 with different border insertion positions
  4. Conference matrix C30 with row/column deletion -> 29x29
  5. Circulant matrices from supplementary difference sets in Z_29

Each construction gets 60s of optimization time.
"""
import numpy as np
import time
import sys

N = 29
THEORETICAL_MAX = 1270698346568170340352


# ============================================================================
# Utility: scoring
# ============================================================================

def det_bareiss(A):
    """Bareiss algorithm for exact integer determinant."""
    n = len(A)
    if n == 0:
        return 1
    M = [row[:] for row in A]
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


def score_matrix(A):
    """Return (abs_det, ratio) for a +-1 matrix A."""
    int_mat = np.sign(A).astype(int)
    int_mat[int_mat == 0] = 1
    abs_det = abs(det_bareiss(int_mat.tolist()))
    return abs_det, abs_det / THEORETICAL_MAX


def logdet(A):
    """Return log|det(A)| (float)."""
    s, ld = np.linalg.slogdet(A.astype(np.float64))
    if s == 0:
        return -1e18
    return ld


# ============================================================================
# Optimization engines (copied from init_program.py, self-contained)
# ============================================================================

def hill_climb_sm(A, deadline):
    """Sherman-Morrison single-flip hill climbing."""
    n = A.shape[0]
    A = A.astype(np.float64).copy()
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
        return A, ld
    Ai = np.linalg.inv(A)
    steps = 0
    while time.time() < deadline:
        ratios = 1.0 - 2.0 * A * Ai.T
        ar = np.abs(ratios)
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
        if steps % 30 == 0:
            s2, _ = np.linalg.slogdet(A)
            if s2 == 0:
                break
            Ai = np.linalg.inv(A)
    _, ld = np.linalg.slogdet(A)
    return A, ld


def row_replace_climb(A, deadline):
    """Row-replacement algorithm: replace entire rows to increase |det|."""
    n = A.shape[0]
    A = A.astype(np.float64).copy()
    s, ld = np.linalg.slogdet(A)
    if s == 0 or ld < 10:
        return A, ld
    Ai = np.linalg.inv(A)
    while time.time() < deadline:
        improved = False
        for k in range(n):
            col_k = Ai[:, k]
            r_new = (np.sign(col_k) * s).astype(float)
            r_new[r_new == 0] = 1.0
            d = r_new - A[k, :]
            if np.all(d == 0):
                continue
            dn = 1.0 + d @ col_k
            if abs(dn) <= 1.0 + 1e-10:
                continue
            Ai -= np.outer(Ai[:, k], d @ Ai) / dn
            A[k, :] = r_new
            s2, ld = np.linalg.slogdet(A)
            s = s2
            improved = True
        Ai = np.linalg.inv(A)
        s, ld = np.linalg.slogdet(A)
        if s == 0 or not improved:
            break
    return A, ld


def optimize_matrix(A_start, time_limit, rng, label=""):
    """Full optimization: row-replace + hill-climb + perturbation restarts."""
    deadline = time.time() + time_limit
    n = A_start.shape[0]

    A = A_start.astype(np.float64).copy()
    A, _ = row_replace_climb(A, min(time.time() + 0.5, deadline))
    A, _ = hill_climb_sm(A, min(time.time() + 0.5, deadline))
    best_A = A.copy()
    best_ld = logdet(A)

    restart_count = 0
    while time.time() < deadline - 1:
        restart_count += 1
        if rng.random() < 0.85 and best_ld > 30:
            A = best_A.copy()
            k = rng.randint(5, min(20 + restart_count // 50, 80))
            idxs = rng.randint(0, n, size=(k, 2))
            for idx in idxs:
                A[idx[0], idx[1]] *= -1.0
        else:
            A = rng.choice([-1.0, 1.0], size=(n, n))
        A, _ = row_replace_climb(A, min(time.time() + 0.15, deadline - 1))
        A, _ = hill_climb_sm(A, min(time.time() + 0.1, deadline - 1))
        ld = logdet(A)
        if ld > best_ld:
            best_ld = ld
            best_A = A.copy()

    # Final polish
    best_A, best_ld = hill_climb_sm(best_A, min(time.time() + 2, deadline))

    result = np.sign(best_A).astype(int)
    result[result == 0] = 1
    abs_det, ratio = score_matrix(result)
    print(f"  [{label}] restarts={restart_count}, score={ratio:.6f}, "
          f"|det|={abs_det:.6e}")
    sys.stdout.flush()
    return result, ratio


# ============================================================================
# Construction 1: GF(27) from different irreducible polynomials
# ============================================================================

def build_gf27_poly(poly_coeffs):
    """Build GF(27) = GF(3)[x]/(p(x)) for a given monic cubic p(x).

    poly_coeffs = (c0, c1, c2) means p(x) = x^3 + c2*x^2 + c1*x + c0.
    The reduction rule is: x^3 = -(c2*x^2 + c1*x + c0) mod 3.

    Elements are (a, b, c) representing a + b*x + c*x^2.
    """
    c0, c1, c2 = poly_coeffs
    # x^3 = (-c2)*x^2 + (-c1)*x + (-c0)  mod 3
    r0 = (-c0) % 3
    r1 = (-c1) % 3
    r2 = (-c2) % 3

    elements = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    idx = {e: i for i, e in enumerate(elements)}

    add_t = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            s = tuple((elements[i][k] + elements[j][k]) % 3 for k in range(3))
            add_t[i, j] = idx[s]

    def mul(a, b):
        a0, a1, a2 = a
        b0, b1, b2 = b
        # Product before reduction: d0 + d1*x + d2*x^2 + d3*x^3 + d4*x^4
        d0 = a0 * b0
        d1 = a0 * b1 + a1 * b0
        d2 = a0 * b2 + a1 * b1 + a2 * b0
        d3 = a1 * b2 + a2 * b1
        d4 = a2 * b2
        # Reduce x^4 = x * x^3 = x*(r0 + r1*x + r2*x^2) = r0*x + r1*x^2 + r2*x^3
        #       x^4 = r0*x + r1*x^2 + r2*(r0 + r1*x + r2*x^2)
        #            = r2*r0 + (r0 + r2*r1)*x + (r1 + r2*r2)*x^2
        # Reduce x^4 first
        d0 += d4 * (r2 * r0)
        d1 += d4 * (r0 + r2 * r1)
        d2 += d4 * (r1 + r2 * r2)
        # Reduce x^3: x^3 = r0 + r1*x + r2*x^2
        d0 += d3 * r0
        d1 += d3 * r1
        d2 += d3 * r2
        return (d0 % 3, d1 % 3, d2 % 3)

    mul_t = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            mul_t[i, j] = idx[mul(elements[i], elements[j])]

    zi = idx[(0, 0, 0)]
    oi = idx[(1, 0, 0)]

    # Find generator of GF(27)* (order 26)
    gen = None
    for g in range(27):
        if g == zi:
            continue
        cur = oi
        is_gen = True
        for k in range(1, 26):
            cur = mul_t[cur, g]
            if cur == oi:
                is_gen = False
                break
        if is_gen:
            cur = mul_t[cur, g]
            if cur == oi:
                gen = g
                break

    if gen is None:
        return None  # Not an irreducible polynomial

    # Discrete log
    dlog = np.full(27, -1, dtype=int)
    cur = oi
    for k in range(26):
        dlog[cur] = k
        cur = mul_t[cur, gen]

    # Quadratic character
    chi = np.zeros(27, dtype=int)
    for i in range(27):
        if i == zi:
            chi[i] = 0
        elif dlog[i] % 2 == 0:
            chi[i] = 1
        else:
            chi[i] = -1

    neg = np.array([idx[tuple((-x) % 3 for x in elements[i])] for i in range(27)], dtype=int)
    return add_t, chi, neg


def build_h28_from_poly(poly_coeffs):
    """Build 28x28 Hadamard from GF(27) with given irreducible polynomial."""
    result = build_gf27_poly(poly_coeffs)
    if result is None:
        return None
    add_t, chi, neg = result
    Q = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            Q[i, j] = chi[add_t[i, neg[j]]]
    H = np.zeros((28, 28), dtype=int)
    H[0, 0] = 1
    H[0, 1:] = 1
    H[1:, 0] = -1
    H[1:, 1:] = Q + np.eye(27, dtype=int)
    return H


def find_irreducible_cubics_gf3():
    """Find all monic irreducible cubics over GF(3).

    A monic cubic x^3 + c2*x^2 + c1*x + c0 over GF(3) is irreducible iff
    it has no roots in GF(3).
    For degree 3: irreducible <=> no roots in GF(3).
    There are (3^3 - 3)/3 = 8 monic irreducible cubics over GF(3).
    """
    irreducibles = []
    for c0 in range(3):
        for c1 in range(3):
            for c2 in range(3):
                # p(x) = x^3 + c2*x^2 + c1*x + c0
                has_root = False
                for x in range(3):
                    val = (x**3 + c2 * x**2 + c1 * x + c0) % 3
                    if val == 0:
                        has_root = True
                        break
                if not has_root:
                    irreducibles.append((c0, c1, c2))
    return irreducibles


def construction_1_different_polynomials(time_per_poly=8.0):
    """Construction 1: H28 from different irreducible polynomials over GF(3)."""
    print("\n" + "=" * 70)
    print("CONSTRUCTION 1: GF(27) from different irreducible polynomials")
    print("=" * 70)

    polys = find_irreducible_cubics_gf3()
    print(f"Found {len(polys)} irreducible cubics over GF(3):")
    for p in polys:
        print(f"  x^3 + {p[2]}x^2 + {p[1]}x + {p[0]}")

    best_ratio = 0
    best_matrix = None
    results = []

    for poly in polys:
        label = f"poly({poly[0]},{poly[1]},{poly[2]})"
        print(f"\n--- {label}: x^3 + {poly[2]}x^2 + {poly[1]}x + {poly[0]} ---")
        sys.stdout.flush()

        H28 = build_h28_from_poly(poly)
        if H28 is None:
            print(f"  Failed to build GF(27) -- skipping")
            continue

        # Verify Hadamard property
        HHT = H28.astype(float) @ H28.astype(float).T
        is_hadamard = np.allclose(HHT, 28 * np.eye(28))
        print(f"  H28*H28^T = 28*I: {is_hadamard}")

        if not is_hadamard:
            print(f"  NOT Hadamard -- skipping")
            continue

        # Border H28 to 29x29
        rng = np.random.RandomState(hash(poly) % (2**31))
        H28f = H28.astype(np.float64)
        H28T = H28f.T

        # Find best border
        best_border = None
        best_schur = 0
        for _ in range(50000):
            s = rng.choice([-1, 1], size=28).astype(np.float64)
            w = H28T @ s
            saw = np.sum(np.abs(w))
            sp = 1.0 + saw / 28.0
            sn = abs(-1.0 + saw / 28.0)
            if sp >= sn:
                sc, r, c = sp, -np.sign(w), 1
            else:
                sc, r, c = sn, np.sign(w), -1
            r[r == 0] = 1.0
            if sc > best_schur:
                best_schur = sc
                M = np.zeros((29, 29), dtype=np.float64)
                M[0, 0] = c
                M[0, 1:] = r
                M[1:, 0] = s
                M[1:, 1:] = H28f
                best_border = M

        if best_border is None:
            continue

        print(f"  Best Schur complement: {best_schur:.4f}")

        # Optimize
        result, ratio = optimize_matrix(best_border, time_per_poly,
                                        np.random.RandomState(42), label)
        results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    print(f"\nConstruction 1 best: {best_ratio:.6f}")
    return results, best_matrix, best_ratio


# ============================================================================
# Construction 2: Williamson-type H28 (order 28 = 4*7)
# ============================================================================

def build_williamson_h28():
    """Build Hadamard matrices of order 28 using Williamson construction.

    A Williamson matrix of order 4t uses 4 symmetric circulant {+1,-1}
    matrices A, B, C, D of order t satisfying:
        A^2 + B^2 + C^2 + D^2 = 4t * I_t

    For t=7, we need A, B, C, D as 7x7 symmetric circulants with
    A^2 + B^2 + C^2 + D^2 = 28 * I_7.

    The Williamson Hadamard matrix is then:
        W = [[A, B, C, D],
             [-B, A, -D, C],
             [-C, D, A, -B],
             [-D, -C, B, A]]

    Known Williamson sequences for t=7 (first row of each circulant):
    Several solutions exist; we search for them.
    """
    t = 7
    results = []

    # For t=7, a symmetric circulant is determined by entries [a0, a1, a2, a3]
    # since a_k = a_{t-k}. The row is [a0, a1, a2, a3, a3, a2, a1].

    # The condition A^2 + B^2 + C^2 + D^2 = 4t*I means
    # for each lag m: sum of autocorrelation at lag m = 0 (for m != 0)
    # and sum at lag 0 = 4t.

    # Autocorrelation of a symmetric circulant with first half [a0,...,a3]:
    # R_A(m) = sum_{k=0}^{t-1} a_k * a_{(k+m) mod t}

    def circulant_from_half(half):
        """Build 7x7 circulant from first-half [a0, a1, a2, a3]."""
        row = list(half) + list(reversed(half[1:]))
        C = np.zeros((t, t), dtype=int)
        for i in range(t):
            for j in range(t):
                C[i, j] = row[(j - i) % t]
        return C

    def autocorr(row, m, t):
        """Circular autocorrelation at lag m."""
        return sum(row[k] * row[(k + m) % t] for k in range(t))

    # Enumerate all possible half-rows (4 entries, each +-1)
    halves = []
    for bits in range(16):
        half = [1 if (bits >> i) & 1 else -1 for i in range(4)]
        halves.append(tuple(half))

    # For each half, compute the full row and its autocorrelation
    half_autocorrs = {}
    for half in halves:
        row = list(half) + list(reversed(half[1:]))
        acf = [autocorr(row, m, t) for m in range(t)]
        half_autocorrs[half] = acf

    # Search for (A, B, C, D) with sum of autocorrelations = 4t*delta
    # R_A(0) + R_B(0) + R_C(0) + R_D(0) = 4*7 = 28
    # R_A(m) + R_B(m) + R_C(m) + R_D(m) = 0 for m = 1, ..., 6

    # Since rows have 7 entries of +-1, R(0) = 7 always.
    # So we need R_A(0)+R_B(0)+R_C(0)+R_D(0) = 4*7 = 28. Check: 7*4=28. Good.

    # We need: for m=1,2,3: R_A(m)+R_B(m)+R_C(m)+R_D(m) = 0
    # (m=4,5,6 are same as m=3,2,1 by symmetry of the circulant)

    solutions = []
    for ia, a_half in enumerate(halves):
        acf_a = half_autocorrs[a_half]
        for ib, b_half in enumerate(halves):
            acf_b = half_autocorrs[b_half]
            ab1 = acf_a[1] + acf_b[1]
            ab2 = acf_a[2] + acf_b[2]
            ab3 = acf_a[3] + acf_b[3]
            for ic, c_half in enumerate(halves):
                acf_c = half_autocorrs[c_half]
                abc1 = ab1 + acf_c[1]
                abc2 = ab2 + acf_c[2]
                abc3 = ab3 + acf_c[3]
                # Need D with acf_d[m] = -abc_m for m=1,2,3
                target = (-abc1, -abc2, -abc3)
                for id_, d_half in enumerate(halves):
                    acf_d = half_autocorrs[d_half]
                    if (acf_d[1], acf_d[2], acf_d[3]) == target:
                        solutions.append((a_half, b_half, c_half, d_half))

    print(f"  Found {len(solutions)} Williamson quadruples for t=7")

    # Remove duplicates up to permutation of (A,B,C,D) and sign
    seen = set()
    unique_solutions = []
    for sol in solutions:
        key = tuple(sorted(sol))
        if key not in seen:
            seen.add(key)
            unique_solutions.append(sol)

    print(f"  Unique (up to permutation): {len(unique_solutions)}")

    # Build Hadamard matrices from each solution
    for sol_idx, (a_half, b_half, c_half, d_half) in enumerate(unique_solutions):
        A = circulant_from_half(a_half)
        B = circulant_from_half(b_half)
        C = circulant_from_half(c_half)
        D = circulant_from_half(d_half)

        # Verify: A^2+B^2+C^2+D^2 = 28*I
        check = A @ A.T + B @ B.T + C @ C.T + D @ D.T
        if not np.allclose(check, 28 * np.eye(t)):
            continue

        # Build Williamson Hadamard matrix
        W = np.zeros((28, 28), dtype=int)
        W[:7, :7] = A;     W[:7, 7:14] = B;     W[:7, 14:21] = C;     W[:7, 21:28] = D
        W[7:14, :7] = -B;  W[7:14, 7:14] = A;   W[7:14, 14:21] = -D;  W[7:14, 21:28] = C
        W[14:21, :7] = -C;  W[14:21, 7:14] = D;  W[14:21, 14:21] = A;  W[14:21, 21:28] = -B
        W[21:28, :7] = -D;  W[21:28, 7:14] = -C; W[21:28, 14:21] = B;  W[21:28, 21:28] = A

        # Verify Hadamard
        WWT = W.astype(float) @ W.astype(float).T
        if np.allclose(WWT, 28 * np.eye(28)):
            results.append(W)

    return results


def construction_2_williamson(time_limit=60.0):
    """Construction 2: Williamson-type H28."""
    print("\n" + "=" * 70)
    print("CONSTRUCTION 2: Williamson-type Hadamard matrices of order 28")
    print("=" * 70)
    sys.stdout.flush()

    h28_list = build_williamson_h28()
    print(f"  Built {len(h28_list)} valid Williamson H28 matrices")
    sys.stdout.flush()

    if not h28_list:
        print("  No Williamson solutions found.")
        return [], None, 0

    best_ratio = 0
    best_matrix = None
    results = []
    time_per = max(5.0, time_limit / max(len(h28_list), 1))

    for wi, H28 in enumerate(h28_list):
        label = f"Williamson-{wi}"
        print(f"\n--- {label} ---")
        sys.stdout.flush()

        # Border to 29x29
        rng = np.random.RandomState(100 + wi)
        H28f = H28.astype(np.float64)
        H28T = H28f.T

        best_border = None
        best_schur = 0
        for _ in range(50000):
            s = rng.choice([-1, 1], size=28).astype(np.float64)
            w = H28T @ s
            saw = np.sum(np.abs(w))
            sp = 1.0 + saw / 28.0
            sn = abs(-1.0 + saw / 28.0)
            if sp >= sn:
                sc, r, c = sp, -np.sign(w), 1
            else:
                sc, r, c = sn, np.sign(w), -1
            r[r == 0] = 1.0
            if sc > best_schur:
                best_schur = sc
                M = np.zeros((29, 29), dtype=np.float64)
                M[0, 0] = c
                M[0, 1:] = r
                M[1:, 0] = s
                M[1:, 1:] = H28f
                best_border = M

        if best_border is None:
            continue

        print(f"  Best Schur complement: {best_schur:.4f}")

        result, ratio = optimize_matrix(best_border, time_per,
                                        np.random.RandomState(42), label)
        results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    print(f"\nConstruction 2 best: {best_ratio:.6f}")
    return results, best_matrix, best_ratio


# ============================================================================
# Construction 3: Bordered H28 with different border insertion positions
# ============================================================================

def construction_3_different_border_positions(time_limit=60.0):
    """Construction 3: Insert border at different positions in H28.

    Instead of always bordering as row 0 / col 0, delete row i and col j
    from a 29x29 identity-bordered structure, giving different 29x29 matrices.

    Concretely: take the standard H28, insert a new row at position k
    and a new column at position k, creating a 29x29 matrix. The new row
    and column are optimized for maximum Schur complement.
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTION 3: Different border insertion positions")
    print("=" * 70)
    sys.stdout.flush()

    # Build standard Paley H28
    result_gf = build_gf27_poly((1, 2, 0))  # x^3 + 2x + 1 (standard)
    add_t, chi, neg = result_gf
    Q = np.zeros((27, 27), dtype=int)
    for i in range(27):
        for j in range(27):
            Q[i, j] = chi[add_t[i, neg[j]]]
    H28 = np.zeros((28, 28), dtype=int)
    H28[0, 0] = 1
    H28[0, 1:] = 1
    H28[1:, 0] = -1
    H28[1:, 1:] = Q + np.eye(27, dtype=int)

    best_ratio = 0
    best_matrix = None
    all_results = []

    # Try inserting the border at different positions
    positions = list(range(29))  # 0 through 28
    time_per = max(2.0, time_limit / len(positions))

    for pos in positions:
        label = f"border-pos-{pos}"
        rng = np.random.RandomState(200 + pos)

        # Insert border at position `pos`:
        # M is 29x29. Row pos and col pos are the new border.
        # The remaining 28x28 submatrix (deleting row pos, col pos) is H28.

        # Strategy: search for the best border row/col at position `pos`
        H28f = H28.astype(np.float64)
        H28inv = np.linalg.inv(H28f)

        best_border = None
        best_schur = 0

        for _ in range(20000):
            # border column (28 entries, the part of column `pos` outside row `pos`)
            s = rng.choice([-1, 1], size=28).astype(np.float64)
            # Schur complement: corner - r^T H28^{-1} s
            # Optimal r = -sign(H28^{-T} s) * corner_sign
            w = H28inv.T @ s  # H28^{-T} s
            saw = np.sum(np.abs(w))
            sp = 1.0 + saw
            sn = abs(-1.0 + saw)
            if sp >= sn:
                sc, r, c = sp, -np.sign(w), 1
            else:
                sc, r, c = sn, np.sign(w), -1
            r[r == 0] = 1.0
            if sc > best_schur:
                best_schur = sc
                # Build 29x29 matrix with border at position `pos`
                M = np.zeros((29, 29), dtype=np.float64)
                # Place H28 into M, skipping row `pos` and col `pos`
                rows_before = list(range(pos))
                rows_after = list(range(pos + 1, 29))
                all_rows = rows_before + rows_after
                for mi, ri in enumerate(all_rows):
                    for mj, cj in enumerate(all_rows):
                        M[ri, cj] = H28f[mi, mj]
                # Place border
                M[pos, pos] = c
                for mi, ri in enumerate(all_rows):
                    M[pos, ri] = r[mi]
                    M[ri, pos] = s[mi]
                best_border = M

        if best_border is None:
            continue

        result, ratio = optimize_matrix(best_border, time_per, rng, label)
        all_results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    print(f"\nConstruction 3 best: {best_ratio:.6f}")
    return all_results, best_matrix, best_ratio


# ============================================================================
# Construction 4: Conference matrix C30 with row/column deletion
# ============================================================================

def build_conference_30():
    """Build the 30x30 bordered Paley conference matrix.

    For p=29 (prime, p = 1 mod 4), the Paley conference matrix is:
    C30 = [[0, e^T], [e, Q29]]
    where Q29[i,j] = chi(i-j) (Legendre symbol), e = all-ones.

    C30 satisfies C30 * C30^T = 29 * I (since C30 is a conference matrix).
    Actually, C30^T * C30 = 29*I_30 for the normalized version.
    """
    p = 29
    # Quadratic residues mod 29
    qr = set()
    for i in range(1, p):
        qr.add((i * i) % p)

    # Legendre symbol
    def legendre(a):
        a = a % p
        if a == 0:
            return 0
        return 1 if a in qr else -1

    # Build Q29 (29x29 Jacobsthal/Paley core)
    Q = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            Q[i, j] = legendre(i - j)

    # Bordered conference matrix C30
    C = np.zeros((p + 1, p + 1), dtype=int)
    C[0, 0] = 0
    C[0, 1:] = 1
    C[1:, 0] = 1
    C[1:, 1:] = Q

    return C


def construction_4_conference_deletion(time_limit=60.0):
    """Construction 4: Delete one row/column from 30x30 conference matrix.

    The 30x30 Paley conference matrix C30 has entries in {0, +1, -1}.
    Deleting row k and column k gives a 29x29 matrix. Replace the 0s with +/-1
    and optimize.
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTION 4: Conference matrix C30 minus one row/column")
    print("=" * 70)
    sys.stdout.flush()

    C30 = build_conference_30()
    print(f"  C30 built: {C30.shape}, entries in {{-1, 0, 1}}")
    print(f"  Number of zeros: {np.sum(C30 == 0)}")

    # C30 has zeros on the diagonal (30 zeros) plus possibly row 0 has a 0
    # Actually: C[0,0]=0, C[i,i]=0 for all i (Legendre(0)=0). Total: 30 zeros.

    best_ratio = 0
    best_matrix = None
    all_results = []
    time_per = max(2.0, time_limit / 30)

    for k in range(30):
        label = f"conf30-del-{k}"
        rng = np.random.RandomState(300 + k)

        # Delete row k and col k
        rows = [i for i in range(30) if i != k]
        M29 = C30[np.ix_(rows, rows)].astype(np.float64)

        # Replace zeros with random +/-1
        zeros = (M29 == 0)
        num_zeros = np.sum(zeros)

        # Try multiple random replacements of zeros
        best_start = None
        best_start_ld = -1e18

        for trial in range(200):
            M = M29.copy()
            M[zeros] = rng.choice([-1.0, 1.0], size=num_zeros)
            ld = logdet(M)
            if ld > best_start_ld:
                best_start_ld = ld
                best_start = M.copy()

        if best_start is None:
            continue

        result, ratio = optimize_matrix(best_start, time_per, rng, label)
        all_results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    print(f"\nConstruction 4 best: {best_ratio:.6f}")
    return all_results, best_matrix, best_ratio


# ============================================================================
# Construction 5: Supplementary difference sets / circulant constructions
# ============================================================================

def construction_5_circulant(time_limit=60.0):
    """Construction 5: Circulant-based matrices for n=29.

    Approach A: Build a 29x29 circulant matrix C where C[i,j] = f((i-j) mod 29),
    using the Legendre symbol of Z_29. Since 29 is prime, 29 = 1 mod 4,
    the Paley-type matrix Q29[i,j] = chi((i-j) mod 29) is symmetric.
    Then Q29 + I is a good starting point.

    Approach B: Use difference sets. A (v, k, lambda)-difference set D in Z_v
    gives a circulant 0/1 matrix; convert to +/-1.

    Approach C: Use the Legendre symbol to build a circulant core, then
    try different diagonal modifications.
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTION 5: Circulant and difference-set constructions")
    print("=" * 70)
    sys.stdout.flush()

    p = 29
    qr = set()
    for i in range(1, p):
        qr.add((i * i) % p)

    best_ratio = 0
    best_matrix = None
    all_results = []

    # --- Approach A: Paley core Q29 + alpha*I ---
    Q29 = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            d = (i - j) % p
            if d == 0:
                Q29[i, j] = 0
            elif d in qr:
                Q29[i, j] = 1
            else:
                Q29[i, j] = -1

    for diag_val in [1, -1]:
        label = f"Q29+{diag_val}I"
        M = Q29.copy().astype(np.float64)
        np.fill_diagonal(M, diag_val)

        rng = np.random.RandomState(400 + diag_val)
        result, ratio = optimize_matrix(M, 10.0, rng, label)
        all_results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    # --- Approach B: Difference sets ---
    # For v=29, look for (29, k, lambda) difference sets
    # A classic one: quadratic residues form a (29, 14, 6) difference set
    # D = {1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}
    diff_sets = [
        sorted(qr),  # quadratic residues
        sorted(set(range(1, p)) - qr),  # quadratic non-residues
    ]

    # Also try some random difference-set-like constructions
    # (random subsets of Z_29 of various sizes)
    rng_ds = np.random.RandomState(500)
    for size in [13, 14, 15]:
        for _ in range(5):
            D = sorted(rng_ds.choice(p, size=size, replace=False))
            diff_sets.append(D)

    for di, D in enumerate(diff_sets):
        label = f"diffset-{di}-sz{len(D)}"
        D_set = set(D)

        # Build circulant +-1 matrix from difference set
        M = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            for j in range(p):
                d = (i - j) % p
                M[i, j] = 1.0 if d in D_set else -1.0

        rng = np.random.RandomState(500 + di)
        result, ratio = optimize_matrix(M, 5.0, rng, label)
        all_results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    # --- Approach C: Negaperiodic / two-circulant constructions ---
    # Build a 29x29 matrix from two blocks:
    # Top: rows 0..14 use circulant pattern from set D1
    # Bottom: rows 15..28 use circulant pattern from set D2
    for trial in range(5):
        label = f"two-circ-{trial}"
        rng = np.random.RandomState(600 + trial)
        M = np.zeros((p, p), dtype=np.float64)

        # Random first row for each half
        r1 = rng.choice([-1.0, 1.0], size=p)
        r2 = rng.choice([-1.0, 1.0], size=p)
        for i in range(15):
            for j in range(p):
                M[i, j] = r1[(j - i) % p]
        for i in range(15, p):
            for j in range(p):
                M[i, j] = r2[(j - i) % p]

        result, ratio = optimize_matrix(M, 5.0, rng, label)
        all_results.append((label, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            best_matrix = result

    print(f"\nConstruction 5 best: {best_ratio:.6f}")
    return all_results, best_matrix, best_ratio


# ============================================================================
# Construction 6 (bonus): Best-known matrix as seed with diverse perturbations
# ============================================================================

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
], dtype=np.float64)


def construction_6_seed_perturbations(time_limit=60.0):
    """Construction 6: Start from best-known matrix with structured perturbations.

    Different perturbation strategies to escape the local optimum:
    - Row/column permutations
    - Row/column negations
    - Block swaps
    - Systematic row replacements
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTION 6 (bonus): Best-known seed with diverse perturbations")
    print("=" * 70)
    sys.stdout.flush()

    best_ratio = 0
    best_matrix = None
    all_results = []

    # Strategy A: Negate random subsets of rows
    deadline = time.time() + time_limit / 3
    label = "seed-row-negate"
    rng = np.random.RandomState(700)
    best_A = BEST_KNOWN.copy()
    best_ld = logdet(best_A)
    restarts = 0
    while time.time() < deadline - 1:
        restarts += 1
        A = BEST_KNOWN.copy()
        # Negate a random subset of rows
        n_negate = rng.randint(1, 15)
        rows_to_negate = rng.choice(29, size=n_negate, replace=False)
        A[rows_to_negate] *= -1
        # Also permute some rows
        if rng.random() < 0.5:
            perm = rng.permutation(29)
            A = A[perm]
        A, _ = row_replace_climb(A, min(time.time() + 0.15, deadline - 1))
        A, ld = hill_climb_sm(A, min(time.time() + 0.1, deadline - 1))
        if ld > best_ld:
            best_ld = ld
            best_A = A.copy()

    result = np.sign(best_A).astype(int)
    result[result == 0] = 1
    _, ratio = score_matrix(result)
    all_results.append((label, ratio))
    print(f"  [{label}] restarts={restarts}, score={ratio:.6f}")
    if ratio > best_ratio:
        best_ratio = ratio
        best_matrix = result

    # Strategy B: Permute rows AND columns
    deadline = time.time() + time_limit / 3
    label = "seed-perm"
    rng = np.random.RandomState(800)
    best_A = BEST_KNOWN.copy()
    best_ld = logdet(best_A)
    restarts = 0
    while time.time() < deadline - 1:
        restarts += 1
        A = BEST_KNOWN.copy()
        row_perm = rng.permutation(29)
        col_perm = rng.permutation(29)
        A = A[row_perm][:, col_perm]
        # Flip some entries
        k = rng.randint(5, 50)
        idxs = rng.randint(0, 29, size=(k, 2))
        for idx in idxs:
            A[idx[0], idx[1]] *= -1
        A = A.astype(np.float64)
        A, _ = row_replace_climb(A, min(time.time() + 0.15, deadline - 1))
        A, ld = hill_climb_sm(A, min(time.time() + 0.1, deadline - 1))
        if ld > best_ld:
            best_ld = ld
            best_A = A.copy()

    result = np.sign(best_A).astype(int)
    result[result == 0] = 1
    _, ratio = score_matrix(result)
    all_results.append((label, ratio))
    print(f"  [{label}] restarts={restarts}, score={ratio:.6f}")
    if ratio > best_ratio:
        best_ratio = ratio
        best_matrix = result

    # Strategy C: Block-structured perturbations
    deadline = time.time() + time_limit / 3
    label = "seed-block"
    rng = np.random.RandomState(900)
    best_A = BEST_KNOWN.copy()
    best_ld = logdet(best_A)
    restarts = 0
    while time.time() < deadline - 1:
        restarts += 1
        A = BEST_KNOWN.copy().astype(np.float64)
        # Replace a random block
        r_start = rng.randint(0, 25)
        c_start = rng.randint(0, 25)
        r_size = rng.randint(2, 6)
        c_size = rng.randint(2, 6)
        A[r_start:r_start+r_size, c_start:c_start+c_size] = rng.choice(
            [-1.0, 1.0], size=(min(r_size, 29-r_start), min(c_size, 29-c_start)))
        A, _ = row_replace_climb(A, min(time.time() + 0.15, deadline - 1))
        A, ld = hill_climb_sm(A, min(time.time() + 0.1, deadline - 1))
        if ld > best_ld:
            best_ld = ld
            best_A = A.copy()

    result = np.sign(best_A).astype(int)
    result[result == 0] = 1
    _, ratio = score_matrix(result)
    all_results.append((label, ratio))
    print(f"  [{label}] restarts={restarts}, score={ratio:.6f}")
    if ratio > best_ratio:
        best_ratio = ratio
        best_matrix = result

    print(f"\nConstruction 6 best: {best_ratio:.6f}")
    return all_results, best_matrix, best_ratio


# ============================================================================
# Main: run all constructions
# ============================================================================

def main():
    print("=" * 70)
    print("ALTERNATIVE CONSTRUCTIONS FOR n=29 MAXIMAL DETERMINANT")
    print(f"Theoretical max: {THEORETICAL_MAX}")
    print("=" * 70)
    sys.stdout.flush()

    # Score the best-known matrix first
    _, bk_ratio = score_matrix(BEST_KNOWN)
    print(f"\nBest-known (Orrick-Solomon) score: {bk_ratio:.6f}")
    sys.stdout.flush()

    overall_best_ratio = 0
    overall_best_matrix = None
    all_construction_results = {}

    # Construction 1: Different irreducible polynomials
    t0 = time.time()
    results1, mat1, ratio1 = construction_1_different_polynomials(time_per_poly=7.0)
    all_construction_results["1_polynomials"] = results1
    if ratio1 > overall_best_ratio:
        overall_best_ratio = ratio1
        overall_best_matrix = mat1
    print(f"  (Construction 1 took {time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Construction 2: Williamson
    t0 = time.time()
    results2, mat2, ratio2 = construction_2_williamson(time_limit=60.0)
    all_construction_results["2_williamson"] = results2
    if ratio2 > overall_best_ratio:
        overall_best_ratio = ratio2
        overall_best_matrix = mat2
    print(f"  (Construction 2 took {time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Construction 3: Different border positions
    t0 = time.time()
    results3, mat3, ratio3 = construction_3_different_border_positions(time_limit=60.0)
    all_construction_results["3_border_positions"] = results3
    if ratio3 > overall_best_ratio:
        overall_best_ratio = ratio3
        overall_best_matrix = mat3
    print(f"  (Construction 3 took {time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Construction 4: Conference matrix deletion
    t0 = time.time()
    results4, mat4, ratio4 = construction_4_conference_deletion(time_limit=60.0)
    all_construction_results["4_conference"] = results4
    if ratio4 > overall_best_ratio:
        overall_best_ratio = ratio4
        overall_best_matrix = mat4
    print(f"  (Construction 4 took {time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Construction 5: Circulant / difference sets
    t0 = time.time()
    results5, mat5, ratio5 = construction_5_circulant(time_limit=60.0)
    all_construction_results["5_circulant"] = results5
    if ratio5 > overall_best_ratio:
        overall_best_ratio = ratio5
        overall_best_matrix = mat5
    print(f"  (Construction 5 took {time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Construction 6: Seed perturbations
    t0 = time.time()
    results6, mat6, ratio6 = construction_6_seed_perturbations(time_limit=60.0)
    all_construction_results["6_seed_perturbations"] = results6
    if ratio6 > overall_best_ratio:
        overall_best_ratio = ratio6
        overall_best_matrix = mat6
    print(f"  (Construction 6 took {time.time()-t0:.1f}s)")
    sys.stdout.flush()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL CONSTRUCTIONS")
    print("=" * 70)
    print(f"\nBest-known (Orrick-Solomon) baseline: {bk_ratio:.6f}")
    print()

    for name, results in all_construction_results.items():
        if not results:
            print(f"  {name}: no results")
            continue
        best_r = max(r for _, r in results)
        avg_r = np.mean([r for _, r in results])
        print(f"  {name}: best={best_r:.6f}, avg={avg_r:.6f}, count={len(results)}")
        # Show top 3
        sorted_results = sorted(results, key=lambda x: -x[1])
        for label, ratio in sorted_results[:3]:
            print(f"    {label}: {ratio:.6f}")

    print(f"\n{'='*70}")
    print(f"OVERALL BEST: {overall_best_ratio:.6f}")
    if overall_best_matrix is not None:
        abs_det, _ = score_matrix(overall_best_matrix)
        print(f"  |det| = {abs_det}")
    print(f"{'='*70}")

    return overall_best_matrix, overall_best_ratio


if __name__ == "__main__":
    main()
