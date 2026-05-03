"""
Decompose the new 29x29 Gram matrix into R with all entries +/-1 such that R^T R = G.

New Gram G (29x29):
  - Diagonal: 29
  - G[0,3] = G[3,0] = 9
  - G[1,4] = G[4,1] = 9
  - G[6,j] = G[j,6] = -3 for j in {7,8,9,10}
  - All other off-diagonal: 1

Strategy overview:
  1. Column modification from Solomon decomposition
  2. Algebraic 11-column base with adapted KNOWN_TOP
  3. Pure backtracking with better ordering
  4. Gram targeting (row replacement heuristic)
  5. Hybrid: partial backtrack + MILP completion
"""
import numpy as np
import time
import sys
import itertools
from fractions import Fraction
from math import lcm as math_lcm

N = 29

def build_new_gram():
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    G[0, 3] = 9; G[3, 0] = 9
    G[1, 4] = 9; G[4, 1] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G

def verify_decomposition(R, G):
    if R is None:
        return False
    RR = R.astype(np.int64)
    if not np.all(np.abs(RR) == 1):
        return False
    return np.array_equal(RR.T @ RR, G)

def save_result(R, G):
    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
    print("SAVED to breakthrough_matrix.npy")

    # Compute determinant via Bareiss
    def det_bareiss(A):
        n = len(A); M = [list(row) for row in A]
        sign = 1
        for k in range(n - 1):
            if M[k][k] == 0:
                found = False
                for i in range(k + 1, n):
                    if M[i][k] != 0:
                        M[k], M[i] = M[i], M[k]; sign *= -1; found = True; break
                if not found:
                    return 0
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                    den = M[k - 1][k - 1] if k > 0 else 1
                    M[i][j] = num // den
        return sign * M[-1][-1]

    det_R = abs(det_bareiss(R.astype(int).tolist()))
    target_det = 1218150670582268559360
    target_max = 1270698346568170340352
    score = det_R / target_max
    print(f"|det(R)| = {det_R}")
    print(f"Expected = {target_det}")
    print(f"Match: {det_R == target_det}")
    print(f"Score = {score:.6f}")


# ============================================================
# APPROACH 1: Column modification from Solomon decomposition
# ============================================================
def approach_solomon_modification(G, time_limit=600, verbose=True):
    """
    Start from Solomon R. Modify columns 0-5 so that the cross-block
    inner products change from all-5 to the new pattern.

    Changes needed to the {0,1,2} x {3,4,5} cross-block:
      Solomon: all pairs = 5
      New: (0,3)=9, (1,4)=9, all other 7 pairs = 1

    So: (0,3) and (1,4) need +4; 7 other pairs need -4.

    All other inner products (within {0,1,2}, within {3,4,5},
    cols 0-5 vs cols 6-28) must remain unchanged.
    """
    if verbose:
        print("APPROACH 1: Solomon modification")
        sys.stdout.flush()

    R_sol = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)
    G_sol = R_sol.T @ R_sol

    deadline = time.time() + time_limit
    rng = np.random.RandomState(42)

    # Fixed columns (7-28, and 6)
    fixed_cols = list(range(6, 29))
    fixed_R = R_sol[:, fixed_cols]  # 29 x 23

    # We need to find 6 new columns (for indices 0-5) that satisfy:
    # 1) All entries +/-1
    # 2) col_i . col_j matches new Gram for i,j in {0..5}
    # 3) col_i . fixed_col_k = G[i, fixed_cols[k]] for all k

    # Target cross products with fixed columns
    target_with_fixed = np.array([G[i, fixed_cols] for i in range(6)])  # 6 x 23

    # Target 6x6 block
    target_6x6 = G[:6, :6].copy()

    # Strategy: Try many random configurations for columns 0-5
    # Each column is a +-1 vector of length 29
    # Constraint: col_i . fixed_col_k = target_with_fixed[i,k]

    # For each of the 6 columns independently, the constraint with fixed columns
    # gives us a system of 23 equations in 29 unknowns (all +/-1)
    # That's 29-23 = 6 free variables

    # But we also need the 6 columns to satisfy mutual inner product constraints
    # That's 15 additional constraints (6 choose 2)

    # Let's solve column by column using the existing RREF solver
    # Place columns in order, accumulating constraints

    best_result = None
    n_attempts = 0

    col_orders = [
        [0, 3, 1, 4, 2, 5],
        [3, 0, 4, 1, 5, 2],
        [0, 1, 2, 3, 4, 5],
        [3, 4, 5, 0, 1, 2],
    ]

    for order_idx, col_order in enumerate(col_orders):
        if time.time() > deadline:
            break

        for seed in range(10000):
            if time.time() > deadline:
                break
            n_attempts += 1
            rng_local = np.random.RandomState(seed + order_idx * 10000)

            # Place columns one by one
            placed = {}  # col_index -> vector
            success = True

            for step, ci in enumerate(col_order):
                # Build constraint system:
                # For each fixed column k: col_ci . fixed_col_k = target_with_fixed[ci, k]
                # For each already-placed column cj: col_ci . placed[cj] = G[ci, cj]

                constraint_vecs = []
                constraint_vals = []

                # Constraints from fixed columns
                for k in range(len(fixed_cols)):
                    constraint_vecs.append(fixed_R[:, k])
                    constraint_vals.append(int(target_with_fixed[ci, k]))

                # Constraints from already-placed columns
                for cj in col_order[:step]:
                    constraint_vecs.append(placed[cj])
                    constraint_vals.append(int(G[ci, cj]))

                # Solve using RREF
                solutions = solve_pm1_system(
                    constraint_vecs, constraint_vals, N, rng_local,
                    max_sol=200, deadline=deadline
                )

                if not solutions:
                    success = False
                    break

                # Pick a random solution
                rng_local.shuffle(solutions)
                placed[ci] = solutions[0]

            if success:
                R_new = R_sol.copy()
                for ci in range(6):
                    R_new[:, ci] = placed[ci]

                if verify_decomposition(R_new, G):
                    if verbose:
                        print(f"  SUCCESS after {n_attempts} attempts!")
                    return R_new

    if verbose:
        print(f"  No solution after {n_attempts} attempts")
    return None


def approach_solomon_modification_backtrack(G, time_limit=600, verbose=True):
    """
    Like approach 1, but with backtracking over column choices.
    """
    if verbose:
        print("APPROACH 1b: Solomon modification with backtracking")
        sys.stdout.flush()

    R_sol = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    deadline = time.time() + time_limit
    rng = np.random.RandomState(137)

    fixed_cols = list(range(6, 29))
    fixed_R = R_sol[:, fixed_cols]
    target_with_fixed = np.array([G[i, fixed_cols] for i in range(6)])

    col_orders = [
        [0, 3, 1, 4, 2, 5],
        [3, 0, 4, 1, 5, 2],
        [0, 1, 2, 3, 4, 5],
    ]

    best_depth = 0
    n_attempts = 0

    for order_idx, col_order in enumerate(col_orders):
        for seed_base in range(0, 100000, 1):
            if time.time() > deadline:
                break
            n_attempts += 1
            rng_local = np.random.RandomState(seed_base + order_idx * 100000)

            placed = {}

            def backtrack(step):
                nonlocal best_depth
                if time.time() > deadline:
                    return None
                if step == 6:
                    # All 6 columns placed - verify
                    R_new = R_sol.copy()
                    for ci in range(6):
                        R_new[:, ci] = placed[ci]
                    if verify_decomposition(R_new, G):
                        return R_new
                    return None

                ci = col_order[step]

                constraint_vecs = []
                constraint_vals = []

                for k in range(len(fixed_cols)):
                    constraint_vecs.append(fixed_R[:, k])
                    constraint_vals.append(int(target_with_fixed[ci, k]))

                for cj in col_order[:step]:
                    constraint_vecs.append(placed[cj])
                    constraint_vals.append(int(G[ci, cj]))

                n_constraints = len(constraint_vecs)
                n_free = N - n_constraints

                max_sol = 500 if step < 4 else 2000
                solutions = solve_pm1_system(
                    constraint_vecs, constraint_vals, N, rng_local,
                    max_sol=max_sol, deadline=deadline
                )

                if not solutions:
                    return None

                if step > best_depth:
                    best_depth = step
                    if verbose:
                        print(f"    Attempt {n_attempts}, depth {step}/6, {len(solutions)} solutions for col {ci}")
                        sys.stdout.flush()

                rng_local.shuffle(solutions)
                n_try = min(len(solutions), 50 if step < 3 else 200)

                for sol in solutions[:n_try]:
                    if time.time() > deadline:
                        return None
                    placed[ci] = sol
                    result = backtrack(step + 1)
                    if result is not None:
                        return result
                    del placed[ci]

                return None

            result = backtrack(0)
            if result is not None:
                if verbose:
                    print(f"  SUCCESS after {n_attempts} attempts!")
                return result

    if verbose:
        print(f"  No solution after {n_attempts} attempts, best depth {best_depth}/6")
    return None


# ============================================================
# APPROACH 2: Algebraic 11-column base
# ============================================================
def approach_algebraic_base(G, time_limit=600, verbose=True):
    """
    Build first 11 columns algebraically, then backtrack for remaining 18.

    The new Gram has a different {0..5} x {0..5} block than Solomon.
    We need a new 11x11 top block and 18-entry bottom parts.
    """
    if verbose:
        print("APPROACH 2: Algebraic 11-column base")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    rng = np.random.RandomState(2024)

    # The new 6x6 block must satisfy:
    # Top[:6,:6]^T Top[:6,:6] contributes to G[:6,:6]
    # But we also need bottom 18 entries to contribute
    #
    # For the Solomon structure:
    # Top[:6,:6] = BLOCK_6x6 gives inner products within {0..5} from rows 0-10
    # Bottom 18 entries give the rest

    # New approach: design a KNOWN_TOP for the new Gram
    # The key difference: in the old Solomon, cols 0-2 are "related" to cols 3-5 with IP=5
    # In new Gram: col0-col3 have IP=9, col1-col4 have IP=9, rest have IP=1

    # For the top 11 rows, we need columns c0..c10 each of length 11, all +/-1
    # with: c_i . c_j (11-dim) matching the partial inner products
    # The bottom 18 entries must make up the difference

    # Let's think about what the top 11 entries contribute:
    # For cols i,j: G[i,j] = sum_k=0^28 R[k,i]*R[k,j]
    #   = sum_k=0^10 R[k,i]*R[k,j] + sum_k=11^28 R[k,i]*R[k,j]
    #   = top_ip[i,j] + bottom_ip[i,j]

    # For the Solomon KNOWN_TOP:
    # top_ip for (0,3) = col0[:11] . col3[:11] = ?

    BLOCK_6x6 = np.array([
        [-1,-1, 1, 1, 1, 1],[-1, 1,-1, 1, 1, 1],[ 1,-1,-1, 1, 1, 1],
        [ 1, 1, 1, 1,-1,-1],[ 1, 1, 1,-1,-1, 1],[ 1, 1, 1,-1, 1,-1]], dtype=np.int64)
    BLOCK_H4 = np.array([
        [-1,-1, 1,-1],[-1,-1,-1, 1],[-1, 1,-1,-1],[ 1,-1,-1,-1]], dtype=np.int64)

    KNOWN_TOP_SOL = np.zeros((11, 11), dtype=np.int64)
    KNOWN_TOP_SOL[:6, :6] = BLOCK_6x6
    KNOWN_TOP_SOL[6, :] = -1; KNOWN_TOP_SOL[:6, 6] = -1
    KNOWN_TOP_SOL[:6, 7:11] = 1; KNOWN_TOP_SOL[7:11, :6] = 1
    KNOWN_TOP_SOL[7:11, 6] = -1; KNOWN_TOP_SOL[7:11, 7:11] = BLOCK_H4

    # Check Solomon top inner products
    sol_top_ip = KNOWN_TOP_SOL.T @ KNOWN_TOP_SOL

    if verbose:
        print(f"  Solomon top IP for (0,3): {sol_top_ip[0,3]}")
        print(f"  Solomon top IP for (0,1): {sol_top_ip[0,1]}")

    # For the new Gram, we need to find a new 11x11 top block.
    # Key idea: make col0 and col3 more aligned (IP=9 needs them very similar)

    # With 29 entries and IP=9: n_agree - n_disagree = 9, n_agree + n_disagree = 29
    # => n_agree = 19, n_disagree = 10
    # So 19 entries agree, 10 differ.

    # With 29 entries and IP=1: n_agree = 15, n_disagree = 14

    # In the top 11 entries: if we want most of the "agreement" in the top...
    # For (0,3) with IP=9: top 11 must contribute significantly
    # For (0,4) with IP=1: top 11 should contribute less

    # Let me try a systematic construction.
    # For columns 0 and 3: we need them to be very similar.
    # For columns 1 and 4: we need them to be very similar.
    # For columns 2 and 5: we need them to have IP=1 (mildly correlated).

    # One approach: Search for the top 11x6 block directly
    # That's 66 +/-1 entries with many constraints.

    # Alternative: use the structure from Solomon but modify the 6x6 block

    # In Solomon's 6x6 block:
    # Row i of BLOCK_6x6 is column i restricted to rows 0-5
    # Columns 0-2 have pattern: two -1's and one +1 in first 3 rows, then mixed in rows 3-5
    # Columns 3-5 have pattern: +1 in rows 0-2, then mixed in rows 3-5

    # For the new Gram, I need col0.col3=9, which means in 29 entries, 19 agree.
    # In Solomon, col0.col3 = 5 means 17 agree.
    # So I need 2 more agreements.

    # Let me try a different 6x6 top block.
    # I want:
    #   col0[:6] . col3[:6] to be as large as possible (to help with IP=9)
    #   col1[:6] . col4[:6] to be as large as possible (to help with IP=9)
    #   col0[:6] . col4[:6] to be moderate (IP=1)

    # The 6x6 block also must satisfy:
    #   col_i[:6] . col_j[:6] for i,j within {0,1,2} and within {3,4,5}
    #   These internal IPs combine with bottom to give G[i,j] = 1

    # This is getting complex. Let me use a search approach for the full 11-column structure.
    # I'll search for the top block and bottom vectors simultaneously.

    # Actually, the most productive approach might be:
    # Use construct_first_11_columns adapted for the new Gram.

    # In the Solomon version, the bottom 18 entries of cols 0-2 each have 6 minus-ones (sum=6)
    # and the bottom 18 entries of cols 3-5 each have 6 minus-ones (sum=6)
    # Cols 6 have all +1 bottom (sum=18)
    # Cols 7-10 have 9 minus-ones (sum=0)

    # The inner product constraints for bottom 18 entries:
    # For cols i,j: bottom_ip[i,j] = G[i,j] - top_ip[i,j]

    # Let me try: keep the Solomon KNOWN_TOP but change the bottom parts
    # to satisfy the new Gram.

    # With Solomon KNOWN_TOP:
    # top_ip[0,3] = sum of KNOWN_TOP[:,0] * KNOWN_TOP[:,3] over 11 rows

    top_ip_sol = KNOWN_TOP_SOL.T @ KNOWN_TOP_SOL

    # For the new Gram, bottom_ip[i,j] = G[i,j] - top_ip[i,j]
    # bottom_ip[0,3] = 9 - top_ip_sol[0,3]
    # bottom_ip[1,4] = 9 - top_ip_sol[1,4]
    # For other cross pairs: bottom_ip = 1 - top_ip_sol[i,j]

    if verbose:
        print(f"\n  Top IP matrix (Solomon KNOWN_TOP, 11x11):")
        for pair_name, i, j, g_target in [
            ('(0,3)', 0, 3, 9), ('(1,4)', 1, 4, 9),
            ('(0,4)', 0, 4, 1), ('(0,5)', 0, 5, 1),
            ('(1,3)', 1, 3, 1), ('(1,5)', 1, 5, 1),
            ('(2,3)', 2, 3, 1), ('(2,4)', 2, 4, 1), ('(2,5)', 2, 5, 1),
            ('(0,1)', 0, 1, 1), ('(0,2)', 0, 2, 1), ('(1,2)', 1, 2, 1),
            ('(3,4)', 3, 4, 1), ('(3,5)', 3, 5, 1), ('(4,5)', 4, 5, 1),
        ]:
            tip = top_ip_sol[i, j]
            bip = g_target - tip
            print(f"    {pair_name}: top={tip}, need_bottom={bip}")

    # Now construct first 11 columns using Solomon top + adapted bottom
    # Key: bottom vectors v[0]..v[10] each of length 18, +/-1
    # v[0] has 6 minus-ones (sum = 18 - 12 = 6)... wait, let me check

    # For cols 0-5 in Solomon: KNOWN_TOP col sums
    for ci in range(11):
        col_top = KNOWN_TOP_SOL[:, ci]
        top_sum = np.sum(col_top)
        # Total column sum for a col with inner product 1 with all-ones:
        # Actually, G doesn't have an all-ones column necessarily
        # The self-IP is 29 (all entries +/-1, 29 entries)
        # The "sum" of column entries doesn't directly matter
        # What matters are the pairwise IPs
        if verbose and ci < 6:
            print(f"  Col {ci} top sum: {top_sum}")

    # Let me construct bottom parts for the new Gram
    # using the adapted version of construct_first_11_columns

    result = construct_first_11_columns_new(
        KNOWN_TOP_SOL, G, rng, deadline, verbose
    )

    if result is None:
        if verbose:
            print("  Failed to construct first 11 columns")
        return None

    v_bottom = result
    cols = []
    indices = list(range(11))
    for j in range(11):
        c = np.zeros(N, dtype=np.int64)
        c[:11] = KNOWN_TOP_SOL[:, j]
        c[11:] = v_bottom[j]
        cols.append(c)

    # Verify the partial Gram
    R_partial = np.column_stack(cols)
    pg = R_partial.T @ R_partial
    if not np.array_equal(pg, G[:11, :11]):
        if verbose:
            err = np.sum((pg - G[:11,:11])**2)
            print(f"  Partial Gram error: {err}")
            # Show which entries differ
            diffs = np.argwhere(pg != G[:11,:11])
            for d in diffs[:20]:
                print(f"    ({d[0]},{d[1]}): got {pg[d[0],d[1]]}, want {G[d[0],d[1]]}")
        return None

    if verbose:
        print(f"  First 11 columns valid! Starting backtrack for remaining 18...")
        sys.stdout.flush()

    # Now backtrack for remaining 18 columns
    return complete_remaining_columns(cols, indices, G, rng, deadline, verbose)


def construct_first_11_columns_new(KNOWN_TOP, G, rng, deadline, verbose=False):
    """
    Construct bottom 18 entries for the first 11 columns to match the new Gram.

    Bottom inner products needed:
    For i,j in {0..10}: bottom_ip[i,j] = G[i,j] - top_ip[i,j]
    where top_ip[i,j] = KNOWN_TOP[:,i] . KNOWN_TOP[:,j]
    """
    top_ip = KNOWN_TOP.T @ KNOWN_TOP

    # Compute required bottom IPs
    bottom_target = np.zeros((11, 11), dtype=np.int64)
    for i in range(11):
        for j in range(11):
            bottom_target[i, j] = G[i, j] - top_ip[i, j]

    if verbose:
        print(f"  Bottom target IPs:")
        print(bottom_target)

    # Each bottom vector v[i] has 18 entries, all +/-1
    # v[i] . v[j] = bottom_target[i,j]
    # v[i] . v[i] = 18 (automatically since all +/-1)
    # Check: bottom_target diagonal should be 18
    for i in range(11):
        if bottom_target[i, i] != 18:
            if verbose:
                print(f"  ERROR: bottom_target[{i},{i}] = {bottom_target[i,i]}, expected 18")
            return None

    # Number of +1's in v[i]: let p_i = (18 + sum_i) / 2 where sum_i = v[i].ones
    # v[i].v[j] = 2 * (agreements) - 18 = bottom_target[i,j]
    # agreements = (18 + bottom_target[i,j]) / 2

    # For feasibility, bottom_target[i,j] must have same parity as 18 (even)
    for i in range(11):
        for j in range(i+1, 11):
            bt = bottom_target[i, j]
            if bt % 2 != 0:
                if verbose:
                    print(f"  ERROR: bottom_target[{i},{j}] = {bt} is odd")
                return None

    if verbose:
        print(f"  All bottom targets have correct parity")

    # The number of -1's in v[i]:
    # v[i] . v[i] = 18 always
    # For cols 0-2: in Solomon, 6 minus-ones (sum = 6)
    # For cols 3-5: in Solomon, 6 minus-ones (sum = 6)
    # For col 6: all +1 (sum = 18)
    # For cols 7-10: 9 minus-ones (sum = 0)

    # In the new Gram with Solomon KNOWN_TOP:
    # bottom_target[i,i] = 18 for all i (good, that's just 18 entries of +/-1)

    # Let me figure out the number of -1's needed in each bottom vector
    # from the bottom_target matrix
    # v[i] has sum s_i. v[i].v[j] = (agree - disagree) = 2*agree - 18
    # agree = (18 + bottom_target[i,j]) / 2

    # For i=0,j=3: bottom_target[0,3] = 9 - top_ip[0,3]
    for i in range(6):
        top_self = top_ip[i, i]
        if verbose:
            print(f"  Col {i}: top self-IP = {top_self}, bottom self-IP = {bottom_target[i,i]}")

    # Construct using randomized search
    while time.time() < deadline:
        v = [None] * 11

        # Determine number of -1's for each column
        # For col i: v[i] has m_i minus-ones
        # v[i] sum = 18 - 2*m_i
        # For cols 0-2 and 3-5: need to figure out from constraints

        # Let's start with cols 0,1,2 (which are internally constrained)
        # bottom_target within {0,1,2}:
        bt01 = int(bottom_target[0, 1])
        bt02 = int(bottom_target[0, 2])
        bt12 = int(bottom_target[1, 2])

        # v[0].v[1] = bt01 => agree(0,1) = (18 + bt01) / 2

        # Number of -1's: from top_ip[i,i] = 11 (all entries +/-1 in 11 rows)
        # self IP in top = 11 always. So bottom self IP = 29 - 11 = 18. OK.

        # For col 0: how many -1's in bottom?
        # We need sum constraints from column 6:
        # v[0].v[6] = bottom_target[0,6]
        # v[6] is all +1 (18 ones) since KNOWN_TOP has col 6 all -1,
        # and G[0,6] = 1, top_ip[0,6] = KNOWN_TOP[:,0].KNOWN_TOP[:,6]

        # Actually v[6] doesn't have to be all +1. Let me check.
        # In Solomon: col 6 bottom is all +1 (sum 18)
        # KNOWN_TOP[:,6] = all -1 (11 entries)
        # So col 6 has: [-1]*11 + [+1]*18, sum = -11 + 18 = 7
        # Col 6 self IP = 29 OK
        # Col 6 IP with others: G[6,j] for j in 0-5 is 1 (in both Solomon and new)
        # G[6,j] for j in 7-10 is -3

        # bottom_target[6,6] = 18 OK
        # v[6] must satisfy: v[6].v[j] = bottom_target[6,j] for all j

        # For Solomon: v[6] = all +1, so v[6].v[j] = sum(v[j]) for all j
        # bottom_target[6,0] = G[6,0] - top_ip[6,0]

        bt6 = bottom_target[6, :]
        if verbose:
            print(f"  Bottom targets for col 6: {bt6}")

        # If v[6] = all +1: v[6].v[j] = sum(v[j])
        # So sum(v[j]) = bt6[j] for each j
        # m_j = (18 - bt6[j]) / 2 minus-ones in v[j]

        # Let's check if v[6] = all +1 works:
        for j in range(11):
            if (18 - bt6[j]) % 2 != 0:
                if verbose:
                    print(f"  Col 6 all-ones won't work: bt6[{j}]={bt6[j]} gives fractional m")
                break

        # Let's try v[6] = all +1 and see what constraints we get
        v[6] = np.ones(18, dtype=np.int64)

        # Number of -1's for each column:
        m = {}
        for j in range(11):
            if j == 6:
                m[j] = 0
                continue
            needed_sum = int(bt6[j])
            if (18 - needed_sum) % 2 != 0:
                if verbose:
                    print(f"  Cannot use v[6]=all-ones: col {j} needs sum {needed_sum} (not matching parity)")
                v[6] = None
                break
            m[j] = (18 - needed_sum) // 2
            if m[j] < 0 or m[j] > 18:
                if verbose:
                    print(f"  Cannot use v[6]=all-ones: col {j} needs {m[j]} minus-ones")
                v[6] = None
                break

        if v[6] is None:
            if verbose:
                print("  Trying different v[6]...")
            # Try with different v[6] - skip for now, try the adapted approach
            return None

        if verbose:
            for j in range(11):
                if j == 6:
                    print(f"  Col {j}: 0 minus-ones (all +1)")
                else:
                    print(f"  Col {j}: {m[j]} minus-ones, sum = {18 - 2*m[j]}")

        # Now construct bottom vectors via randomized search
        # Similar to Solomon's construct_first_11_columns but with new target IPs

        # Cols 0,1,2: each has m[0], m[1], m[2] minus-ones
        # Mutual IPs: v[0].v[1] = bt01, v[0].v[2] = bt02, v[1].v[2] = bt12

        # Try constructing v[0]
        for _ in range(5000):
            if time.time() > deadline:
                return None
            v[0] = np.ones(18, dtype=np.int64)
            if m[0] > 0:
                v[0][rng.choice(18, m[0], replace=False)] = -1

            # Try v[1] compatible with v[0]
            ok1 = False
            for __ in range(5000):
                v[1] = np.ones(18, dtype=np.int64)
                if m[1] > 0:
                    v[1][rng.choice(18, m[1], replace=False)] = -1
                if np.dot(v[0], v[1]) == bt01:
                    ok1 = True
                    break
            if not ok1:
                continue

            # Try v[2] compatible with v[0] and v[1]
            ok2 = False
            for __ in range(50000):
                if time.time() > deadline:
                    return None
                v[2] = np.ones(18, dtype=np.int64)
                if m[2] > 0:
                    v[2][rng.choice(18, m[2], replace=False)] = -1
                if np.dot(v[0], v[2]) == bt02 and np.dot(v[1], v[2]) == bt12:
                    ok2 = True
                    break
            if not ok2:
                continue

            # Try v[3],v[4],v[5]
            bt_cross = bottom_target[:6, :6]
            ok35 = False
            for ___ in range(50000):
                if time.time() > deadline:
                    return None
                v[3] = np.ones(18, dtype=np.int64)
                if m[3] > 0:
                    v[3][rng.choice(18, m[3], replace=False)] = -1
                if not all(np.dot(v[i], v[3]) == int(bt_cross[i, 3]) for i in range(3)):
                    continue

                for ____ in range(5000):
                    v[4] = np.ones(18, dtype=np.int64)
                    if m[4] > 0:
                        v[4][rng.choice(18, m[4], replace=False)] = -1
                    if not (all(np.dot(v[i], v[4]) == int(bt_cross[i, 4]) for i in range(3))
                            and np.dot(v[3], v[4]) == int(bt_cross[3, 4])):
                        continue

                    for _____ in range(5000):
                        v[5] = np.ones(18, dtype=np.int64)
                        if m[5] > 0:
                            v[5][rng.choice(18, m[5], replace=False)] = -1
                        if (all(np.dot(v[i], v[5]) == int(bt_cross[i, 5]) for i in range(3))
                            and np.dot(v[3], v[5]) == int(bt_cross[3, 5])
                            and np.dot(v[4], v[5]) == int(bt_cross[4, 5])):
                            ok35 = True
                            break
                    if ok35:
                        break
                if ok35:
                    break

            if not ok35:
                continue

            # Cols 7-10
            bt_710 = bottom_target[:11, :11]
            ok710 = False
            for ___ in range(50000):
                if time.time() > deadline:
                    return None
                v[7] = np.ones(18, dtype=np.int64)
                if m[7] > 0:
                    v[7][rng.choice(18, m[7], replace=False)] = -1
                if not all(np.dot(v[i], v[7]) == int(bt_710[i, 7]) for i in range(7)):
                    continue

                for ____ in range(5000):
                    v[8] = np.ones(18, dtype=np.int64)
                    if m[8] > 0:
                        v[8][rng.choice(18, m[8], replace=False)] = -1
                    if not (all(np.dot(v[i], v[8]) == int(bt_710[i, 8]) for i in range(7))
                            and np.dot(v[7], v[8]) == int(bt_710[7, 8])):
                        continue

                    for _____ in range(5000):
                        v[9] = np.ones(18, dtype=np.int64)
                        if m[9] > 0:
                            v[9][rng.choice(18, m[9], replace=False)] = -1
                        if not (all(np.dot(v[i], v[9]) == int(bt_710[i, 9]) for i in range(7))
                                and np.dot(v[7], v[9]) == int(bt_710[7, 9])
                                and np.dot(v[8], v[9]) == int(bt_710[8, 9])):
                            continue

                        for ______ in range(5000):
                            v[10] = np.ones(18, dtype=np.int64)
                            if m[10] > 0:
                                v[10][rng.choice(18, m[10], replace=False)] = -1
                            if (all(np.dot(v[i], v[10]) == int(bt_710[i, 10]) for i in range(7))
                                and all(np.dot(v[j], v[10]) == int(bt_710[j, 10]) for j in [7, 8, 9])):
                                ok710 = True
                                break
                        if ok710:
                            break
                    if ok710:
                        break
                if ok710:
                    break

            if ok710:
                if verbose:
                    print("  Found valid bottom vectors!")
                return v

    return None


# ============================================================
# APPROACH 3: Pure backtracking
# ============================================================
def approach_pure_backtrack(G, time_limit=600, verbose=True):
    """Full column-by-column backtracking with smart ordering."""
    if verbose:
        print("APPROACH 3: Pure backtracking")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    rng = np.random.RandomState(7777)

    # Order: place most constrained columns first
    # cols 0,3 have IP=9 with each other
    # cols 1,4 have IP=9 with each other
    # col 6 has IP=-3 with cols 7,8,9,10
    # Start with the high-IP pairs, then the -3 group, then the rest

    col_orders = [
        [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29)),
        [6, 7, 8, 9, 10, 0, 3, 1, 4, 2, 5] + list(range(11, 29)),
        [0, 3, 6, 7, 8, 9, 10, 1, 4, 2, 5] + list(range(11, 29)),
    ]

    best_depth = 0
    n_attempts = 0

    for order_idx, col_order in enumerate(col_orders):
        while time.time() < deadline:
            n_attempts += 1
            rng_local = np.random.RandomState(rng.randint(10**9))

            placed_cols = []
            placed_indices = []

            def recurse(step):
                nonlocal best_depth
                if time.time() > deadline:
                    return False
                if step == N:
                    return True

                if step > best_depth:
                    best_depth = step
                    if verbose:
                        elapsed = time.time() - (deadline - time_limit)
                        print(f"  Order {order_idx}, attempt {n_attempts}: depth {step}/{N}, {elapsed:.1f}s")
                        sys.stdout.flush()

                target_idx = col_order[step]
                ms = 50 if step <= 3 else (200 if step <= 8 else 500)

                solutions = solve_pm1_system(
                    placed_cols, [G[placed_indices[i], target_idx] for i in range(len(placed_cols))],
                    N, rng_local, max_sol=ms, deadline=deadline
                )

                if not solutions:
                    return False

                rng_local.shuffle(solutions)
                n_try = min(len(solutions), 10 if step < 5 else (5 if step < 15 else 3))

                for s in solutions[:n_try]:
                    if time.time() > deadline:
                        return False
                    placed_cols.append(s)
                    placed_indices.append(target_idx)
                    if recurse(step + 1):
                        return True
                    placed_cols.pop()
                    placed_indices.pop()

                return False

            if recurse(0):
                R = np.zeros((N, N), dtype=np.int64)
                for i, ci in enumerate(placed_indices):
                    R[:, ci] = placed_cols[i]
                if verify_decomposition(R, G):
                    if verbose:
                        print(f"  SUCCESS! Order {order_idx}, attempt {n_attempts}")
                    return R

    if verbose:
        print(f"  No solution after {n_attempts} attempts, best depth {best_depth}")
    return None


# ============================================================
# APPROACH 4: Gram targeting (row replacement)
# ============================================================
def approach_gram_targeting(G, time_limit=600, verbose=True):
    """Optimize R row by row to minimize ||R^T R - G||."""
    if verbose:
        print("APPROACH 4: Gram targeting")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    rng = np.random.RandomState(31415)
    start_time = time.time()

    best_R = None
    best_error = 10**18
    n_restarts = 0

    # Start from Solomon R with random perturbations
    R_sol = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    def compute_error(R):
        return int(np.sum((R.T @ R - G) ** 2))

    def optimize_row(R, gram, k):
        """Find best +/-1 row k to minimize gram error."""
        old_row = R[k].copy()
        gram_minus = gram - np.outer(old_row, old_row)
        T = G - gram_minus

        # Greedy optimization of row k
        best_row = None
        best_score = -10**18

        for trial in range(30):
            r = np.ones(N, dtype=np.int64)
            perm = list(range(N))
            rng.shuffle(perm)
            for idx in perm:
                margin = int(np.dot(T[idx], r)) - int(T[idx, idx]) * int(r[idx])
                r[idx] = 1 if margin > 0 else (-1 if margin < 0 else rng.choice([-1, 1]))

            Tr = T @ r
            score = int(r @ Tr)

            for _ in range(10):
                imp = False
                for idx in range(N):
                    delta = -4 * int(r[idx]) * int(Tr[idx]) + 4 * int(T[idx, idx])
                    if delta > 0:
                        old_v = r[idx]
                        r[idx] = -old_v
                        Tr += T[:, idx] * (-2 * old_v)
                        score += delta
                        imp = True
                if not imp:
                    break

            if score > best_score:
                best_score = score
                best_row = r.copy()

        return best_row

    while time.time() < deadline:
        n_restarts += 1

        if n_restarts <= 3:
            R = R_sol.copy()
            # Perturb a few rows
            for pr in rng.choice(N, size=rng.randint(1, 8), replace=False):
                R[pr] = rng.choice([-1, 1], size=N).astype(np.int64)
        elif best_R is not None and rng.random() < 0.7:
            R = best_R.copy()
            num_perturb = rng.randint(2, 10)
            for pr in rng.choice(N, size=num_perturb, replace=False):
                R[pr] = rng.choice([-1, 1], size=N).astype(np.int64)
        else:
            R = rng.choice([-1, 1], size=(N, N)).astype(np.int64)

        gram = R.T @ R
        error = compute_error(R)

        if error == 0:
            if verbose:
                print(f"  EXACT! restart {n_restarts}")
            return R

        stale = 0
        for iteration in range(500):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(N))
            rng.shuffle(order)

            for k in order:
                if time.time() > deadline:
                    break
                new_row = optimize_row(R, gram, k)
                new_gram = gram - np.outer(R[k], R[k]) + np.outer(new_row, new_row)
                new_error = int(np.sum((new_gram - G) ** 2))

                if new_error < error:
                    R[k] = new_row
                    gram = new_gram
                    error = new_error
                    improved = True

                    if error == 0:
                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"  EXACT! restart={n_restarts}, iter={iteration}, {elapsed:.1f}s")
                        return R

            if not improved:
                stale += 1
                if stale >= 5:
                    break
            else:
                stale = 0

        if error < best_error:
            best_error = error
            best_R = R.copy()
            if verbose:
                elapsed = time.time() - start_time
                print(f"  Restart {n_restarts}: error={error}, best={best_error}, {elapsed:.1f}s")
                sys.stdout.flush()

    if verbose:
        print(f"  Best error: {best_error} after {n_restarts} restarts")
    return None


# ============================================================
# APPROACH 5: Hybrid backtrack + MILP
# ============================================================
def approach_hybrid_milp(G, time_limit=600, verbose=True):
    """Place columns via backtracker, finish with MILP."""
    if verbose:
        print("APPROACH 5: Hybrid backtrack + MILP")
        sys.stdout.flush()

    deadline = time.time() + time_limit
    rng = np.random.RandomState(999)

    # First try to get as many columns as possible via backtracking
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))

    best_placed_cols = []
    best_placed_indices = []
    best_depth = 0

    for seed in range(1000):
        if time.time() > deadline - 60:  # Leave 60s for MILP
            break

        rng_local = np.random.RandomState(seed)
        placed_cols = []
        placed_indices = []

        for step in range(N):
            if time.time() > deadline - 60:
                break

            target_idx = col_order[step]
            ms = 100 if step <= 5 else 500

            constraint_vecs = placed_cols
            constraint_vals = [int(G[placed_indices[i], target_idx]) for i in range(len(placed_cols))]

            solutions = solve_pm1_system(
                constraint_vecs, constraint_vals, N, rng_local,
                max_sol=ms, deadline=min(time.time() + 2, deadline - 60)
            )

            if not solutions:
                break

            placed_cols.append(solutions[rng_local.randint(len(solutions))])
            placed_indices.append(target_idx)

        if len(placed_cols) > best_depth:
            best_depth = len(placed_cols)
            best_placed_cols = [c.copy() for c in placed_cols]
            best_placed_indices = placed_indices.copy()
            if verbose:
                print(f"  Seed {seed}: placed {best_depth}/{N} columns")
                sys.stdout.flush()

        if best_depth == N:
            R = np.zeros((N, N), dtype=np.int64)
            for i, ci in enumerate(best_placed_indices):
                R[:, ci] = best_placed_cols[i]
            if verify_decomposition(R, G):
                if verbose:
                    print(f"  Backtrack complete!")
                return R

    if best_depth < 15:
        if verbose:
            print(f"  Only placed {best_depth} columns, MILP unlikely to help")
        return None

    # Try MILP for remaining columns
    remaining = [col_order[i] for i in range(best_depth, N)]
    if verbose:
        print(f"  Trying MILP for {len(remaining)} remaining columns: {remaining}")
        sys.stdout.flush()

    result = milp_complete(best_placed_cols, best_placed_indices, remaining, G,
                           deadline, verbose)
    return result


def milp_complete(placed_cols, placed_indices, remaining_indices, G, deadline, verbose):
    """Use PuLP CBC to find remaining columns."""
    try:
        from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD, LpStatus
    except ImportError:
        if verbose:
            print("  PuLP not available")
        return None

    n_remaining = len(remaining_indices)
    if n_remaining == 0:
        return None

    time_left = deadline - time.time()
    if time_left < 5:
        return None

    if verbose:
        print(f"  MILP: {n_remaining} columns to place, {time_left:.0f}s left")

    prob = LpProblem("gram_decompose", LpMinimize)

    # Binary variables: x[j][k] in {0,1} where R[k, remaining[j]] = 2*x[j][k] - 1
    x = {}
    for j in range(n_remaining):
        for k in range(N):
            x[j, k] = LpVariable(f"x_{j}_{k}", 0, 1, cat='Binary')

    # Dummy objective
    prob += 0

    # Constraints:
    # 1) Inner product with placed columns
    for j in range(n_remaining):
        rj = remaining_indices[j]
        for pi, ci in enumerate(placed_indices):
            target_ip = int(G[ci, rj])
            # sum_k placed_col[pi][k] * (2*x[j,k] - 1) = target_ip
            # sum_k 2*placed_col[pi][k]*x[j,k] - sum_k placed_col[pi][k] = target_ip
            # sum_k 2*placed_col[pi][k]*x[j,k] = target_ip + sum_k placed_col[pi][k]
            col_sum = int(np.sum(placed_cols[pi]))
            rhs = target_ip + col_sum
            # rhs must be even
            if rhs % 2 != 0:
                if verbose:
                    print(f"  MILP infeasible: parity error for col {rj} vs placed col {ci}")
                return None
            prob += (lpSum(2 * int(placed_cols[pi][k]) * x[j, k] for k in range(N)) == rhs,
                     f"ip_placed_{j}_{pi}")

    # 2) Inner products between remaining columns
    # This requires quadratic constraints - linearize using auxiliary variables
    # For now, skip this (hope the placed constraints are sufficient)
    # Actually, we need these for correctness.

    # For pairs of remaining columns j1, j2:
    # sum_k (2*x[j1,k]-1)(2*x[j2,k]-1) = G[remaining[j1], remaining[j2]]
    # = sum_k (4*x[j1,k]*x[j2,k] - 2*x[j1,k] - 2*x[j2,k] + 1)
    # = 4*sum_k x[j1,k]*x[j2,k] - 2*sum_k x[j1,k] - 2*sum_k x[j2,k] + N

    # Linearize x[j1,k]*x[j2,k] with y[j1,j2,k]
    y = {}
    for j1 in range(n_remaining):
        for j2 in range(j1 + 1, n_remaining):
            r1 = remaining_indices[j1]
            r2 = remaining_indices[j2]
            target_ip = int(G[r1, r2])

            for k in range(N):
                y[j1, j2, k] = LpVariable(f"y_{j1}_{j2}_{k}", 0, 1, cat='Binary')
                # y = x[j1,k] * x[j2,k]
                prob += (y[j1, j2, k] <= x[j1, k], f"y_ub1_{j1}_{j2}_{k}")
                prob += (y[j1, j2, k] <= x[j2, k], f"y_ub2_{j1}_{j2}_{k}")
                prob += (y[j1, j2, k] >= x[j1, k] + x[j2, k] - 1, f"y_lb_{j1}_{j2}_{k}")

            prob += (
                lpSum(4 * y[j1, j2, k] - 2 * x[j1, k] - 2 * x[j2, k] for k in range(N)) + N == target_ip,
                f"ip_remaining_{j1}_{j2}"
            )

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=max(1, int(deadline - time.time()) - 5))
    prob.solve(solver)

    if LpStatus[prob.status] != 'Optimal':
        if verbose:
            print(f"  MILP status: {LpStatus[prob.status]}")
        return None

    # Extract solution
    R = np.zeros((N, N), dtype=np.int64)
    for pi, ci in enumerate(placed_indices):
        R[:, ci] = placed_cols[pi]

    for j in range(n_remaining):
        rj = remaining_indices[j]
        for k in range(N):
            val = x[j, k].varValue
            R[k, rj] = 1 if val > 0.5 else -1

    if verify_decomposition(R, G):
        if verbose:
            print("  MILP SUCCESS!")
        return R
    else:
        if verbose:
            err = int(np.sum((R.T @ R - G) ** 2))
            print(f"  MILP solution does not verify, error = {err}")
        return None


# ============================================================
# Shared utilities
# ============================================================
def solve_pm1_system(constraint_vecs, constraint_vals, n, rng, max_sol=500, deadline=None):
    """
    Find +/-1 vectors x of length n satisfying:
    constraint_vecs[i] . x = constraint_vals[i] for all i.

    Uses RREF to find free variables, then enumerate/sample.
    """
    k = len(constraint_vecs)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]

    # Build augmented matrix in Fraction arithmetic
    A_rows = [[Fraction(int(constraint_vecs[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(constraint_vals[i])) for i in range(k)]
    aug = [A_rows[i] + [b_vec[i]] for i in range(k)]

    # Forward elimination
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

    # Check consistency
    for r in range(len(pivot_cols), k):
        if aug[r][n] != 0:
            return []

    # Back-substitution
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

    # Convert to integer arithmetic
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
        pv = icons[np.newaxis, :] - fm @ ic.T
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
        batch_size = min(1 << 20, max_sol * 200)
        n_batches = max(1, (max_sol * 500) // batch_size)
        for _ in range(n_batches):
            if deadline and time.time() > deadline:
                break
            if len(solutions) >= max_sol:
                break
            fm = rng.choice([-1, 1], size=(batch_size, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(batch_size, dtype=bool)
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


def complete_remaining_columns(placed_cols, placed_indices, G, rng, deadline, verbose):
    """Use DFS to fill remaining columns after first 11."""
    remaining = [i for i in range(N) if i not in placed_indices]
    best_depth = len(placed_indices)

    def dfs(step):
        nonlocal best_depth
        if time.time() > deadline:
            return False
        if step == len(remaining):
            return True

        cur_depth = len(placed_indices)
        if cur_depth > best_depth:
            best_depth = cur_depth
            if verbose:
                print(f"    DFS depth {cur_depth}/{N}")
                sys.stdout.flush()

        ti = remaining[step]
        ms = min(20, max(3, 500 // (step + 1)))

        constraint_vals = [int(G[placed_indices[i], ti]) for i in range(len(placed_cols))]
        sols = solve_pm1_system(
            placed_cols, constraint_vals, N, rng,
            max_sol=ms, deadline=min(time.time() + 5, deadline)
        )

        if not sols:
            return False

        rng.shuffle(sols)
        for s in sols[:max(3, ms // 2)]:
            if time.time() > deadline:
                return False
            placed_cols.append(s)
            placed_indices.append(ti)
            if dfs(step + 1):
                return True
            placed_cols.pop()
            placed_indices.pop()

        return False

    if dfs(0):
        R = np.zeros((N, N), dtype=np.int64)
        for i, ci in enumerate(placed_indices):
            R[:, ci] = placed_cols[i]
        if verify_decomposition(R, G):
            if verbose:
                print("    DFS complete!")
            return R

    if verbose:
        print(f"    DFS best depth: {best_depth}/{N}")
    return None


# ============================================================
# APPROACH 6: Direct 6-column search via SAT/constraint enumeration
# ============================================================
def approach_direct_6col_search(G, time_limit=600, verbose=True):
    """
    The only difference between Solomon Gram and new Gram is in the
    {0,1,2}x{3,4,5} cross-block. Fix Solomon columns 6-28 and search
    for 6 new columns 0-5.

    Each new column must be +/-1, length 29, with:
    - Correct inner product with all 23 fixed columns
    - Correct mutual inner products (the new 6x6 block)

    Strategy: solve each column's constraints with fixed cols to get
    a low-dimensional solution space, then search through combinations.
    """
    if verbose:
        print("APPROACH 6: Direct 6-column search")
        sys.stdout.flush()

    R_sol = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/solomon_R.npy").astype(np.int64)

    deadline = time.time() + time_limit
    rng = np.random.RandomState(54321)

    fixed_col_indices = list(range(6, 29))
    fixed_R = R_sol[:, fixed_col_indices]  # 29 x 23

    # For each column i in {0..5}, find ALL +/-1 solutions satisfying
    # inner products with fixed columns
    # That's 23 constraints on 29 variables => 6 free variables => 2^6 = 64 candidates

    candidate_sets = {}
    for ci in range(6):
        constraint_vecs = [fixed_R[:, k] for k in range(23)]
        constraint_vals = [int(G[ci, fixed_col_indices[k]]) for k in range(23)]

        # Get ALL solutions
        candidates = solve_pm1_system(
            constraint_vecs, constraint_vals, N, rng,
            max_sol=100000, deadline=deadline
        )

        if verbose:
            print(f"  Col {ci}: {len(candidates)} candidates satisfying fixed-col constraints")
            sys.stdout.flush()

        if not candidates:
            if verbose:
                print(f"  No candidates for col {ci}!")
            return None

        candidate_sets[ci] = candidates

    # Now search for a combination of 6 columns with correct mutual IPs
    # Target 6x6 block:
    target_6x6 = G[:6, :6].copy()

    if verbose:
        print(f"  Target 6x6 block:")
        print(target_6x6)
        print(f"  Searching through {' x '.join(str(len(candidate_sets[i])) for i in range(6))} = ??? combinations")
        sys.stdout.flush()

    # Precompute inner products between candidate vectors
    # Build incrementally: first pick col 0, then filter col 1 candidates, etc.

    col_order = [0, 3, 1, 4, 2, 5]  # Place paired columns together

    n_searched = 0

    def search(step, chosen):
        nonlocal n_searched
        if time.time() > deadline:
            return None

        if step == 6:
            # Verify full 6x6 block
            R_new = R_sol.copy()
            for ci in range(6):
                R_new[:, ci] = chosen[ci]
            if verify_decomposition(R_new, G):
                return R_new
            return None

        ci = col_order[step]
        candidates = candidate_sets[ci]

        # Filter candidates that satisfy IPs with already-chosen columns
        valid = []
        for cand in candidates:
            ok = True
            for prev_step in range(step):
                cj = col_order[prev_step]
                ip = int(np.dot(cand, chosen[cj]))
                if ip != int(target_6x6[ci, cj]):
                    ok = False
                    break
            if ok:
                valid.append(cand)

        if verbose and step <= 2:
            print(f"    Step {step}, col {ci}: {len(valid)} valid candidates")
            sys.stdout.flush()

        rng.shuffle(valid)
        for cand in valid:
            if time.time() > deadline:
                return None
            n_searched += 1
            chosen[ci] = cand
            result = search(step + 1, chosen)
            if result is not None:
                return result
            del chosen[ci]

        return None

    result = search(0, {})

    if verbose:
        if result is not None:
            print(f"  SUCCESS! Searched {n_searched} combinations")
        else:
            print(f"  No solution found. Searched {n_searched} combinations")

    return result


# ============================================================
# Main execution loop
# ============================================================
def main():
    G = build_new_gram()
    print("=" * 70)
    print("TARGET GRAM DECOMPOSITION: 29x29 matrix R with R^T R = G")
    print("=" * 70)
    print(f"G[0,3] = G[3,0] = 9")
    print(f"G[1,4] = G[4,1] = 9")
    print(f"G[6,7..10] = -3")
    print(f"All other off-diagonal = 1")
    print()
    sys.stdout.flush()

    # Approach 6 first - most promising since we know Solomon R works for everything except 6 cols
    print("=" * 70)
    R = approach_direct_6col_search(G, time_limit=3600, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! Decomposition found! ***")
        save_result(R, G)
        return R

    # Approach 1b: Solomon modification with backtracking
    print("\n" + "=" * 70)
    R = approach_solomon_modification_backtrack(G, time_limit=3600, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! Decomposition found! ***")
        save_result(R, G)
        return R

    # Approach 4: Gram targeting (can run indefinitely)
    print("\n" + "=" * 70)
    R = approach_gram_targeting(G, time_limit=7200, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! Decomposition found! ***")
        save_result(R, G)
        return R

    # Approach 2: Algebraic base
    print("\n" + "=" * 70)
    R = approach_algebraic_base(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! Decomposition found! ***")
        save_result(R, G)
        return R

    # Approach 5: Hybrid MILP
    print("\n" + "=" * 70)
    R = approach_hybrid_milp(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! Decomposition found! ***")
        save_result(R, G)
        return R

    print("\nAll approaches exhausted without finding exact decomposition.")
    return None


if __name__ == "__main__":
    main()
