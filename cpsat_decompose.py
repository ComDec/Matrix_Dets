"""
CP-SAT solver for decomposing 29x29 Gram matrix G into R^T R where R has all entries +/-1.

Improvements over previous attempts:
1. XOR-based disagreement encoding (avoids linearization with auxiliary vars)
2. Symmetry breaking constraints
3. Warm start from partial 11-column solution
4. Automorphism-aware symmetry breaking
5. Incremental column addition strategy
"""
import numpy as np
import time
import sys
from ortools.sat.python import cp_model

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


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, n):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.x_vars = x_vars
        self.n = n
        self.solution = None
        self.solution_count = 0

    def on_solution_callback(self):
        self.solution_count += 1
        self.solution = np.zeros((self.n, self.n), dtype=np.int64)
        for k in range(self.n):
            for a in range(self.n):
                val = self.Value(self.x_vars[k, a])
                self.solution[k, a] = 2 * val - 1  # 0->-1, 1->1
        print(f"  Solution #{self.solution_count} found!")
        sys.stdout.flush()


def approach_cpsat_xor(G, time_limit=3600, verbose=True):
    """
    CP-SAT with XOR-based disagreement encoding.

    Variables: x[k,a] in {0,1} for k=0..28, a=0..28
    R[k,a] = 2*x[k,a] - 1

    For each pair (a,b) with a < b:
      R[:,a] . R[:,b] = G[a,b]
      => #agree - #disagree = G[a,b]
      => #disagree = (29 - G[a,b]) / 2

    Encode disagreement: x[k,a] XOR x[k,b] = 1 iff they disagree
    sum_k (x[k,a] XOR x[k,b]) = (29 - G[a,b]) / 2

    XOR encoding: x[k,a] + x[k,b] - 2*z[k,a,b] = d[k,a,b] where d is disagreement
    But simpler: x[k,a] + x[k,b] - 2*z[k,a,b] is 0 when agree, 1 when disagree
    where z[k,a,b] = x[k,a] AND x[k,b]

    Actually even simpler: XOR = x[k,a] + x[k,b] - 2*x[k,a]*x[k,b]
    We can avoid the product by noting:

    For each pair (a,b), define d[k] = |x[k,a] - x[k,b]|
    d[k] = 1 iff they disagree.
    sum_k d[k] = target_disagree

    In CP-SAT, |x[k,a] - x[k,b]| for binary vars can be encoded as:
    d[k] = x[k,a] + x[k,b] - 2*z[k] where z[k] = min(x[k,a], x[k,b])
    Or equivalently: d[k] = 1 - (x[k,a] == x[k,b])

    Simplest: sum_k (x[k,a] + x[k,b] - 2*x[k,a]*x[k,b]) = target_disagree

    But products need linearization. Alternative approach:

    Let agree[k] = 1 if x[k,a] == x[k,b]
    agree[k] = 1 - (x[k,a] XOR x[k,b])

    For binary vars: x[k,a] XOR x[k,b] can be encoded directly in CP-SAT!
    Actually CP-SAT doesn't have XOR directly, but we can use:

    d[k,a,b] = BoolVar
    d[k,a,b] = 1 iff x[k,a] != x[k,b]

    Encode with: AddBoolXOr([x[k,a], x[k,b], d[k,a,b].Not()])
    Wait: AddBoolXOr(literals) means XOR of literals = True
    XOR(x, y, NOT d) = True means x XOR y XOR NOT d = True
    which means x XOR y = d.

    Then sum_k d[k,a,b] = target_disagree.
    """
    if verbose:
        print("CP-SAT with XOR disagreement encoding")
        sys.stdout.flush()

    model = cp_model.CpModel()

    # Variables: x[k,a] in {0,1}, R[k,a] = 2*x[k,a] - 1
    x = {}
    for k in range(N):
        for a in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # Compute target disagreements
    target_disagree = {}
    for a in range(N):
        for b in range(a + 1, N):
            td = (N - int(G[a, b])) // 2
            target_disagree[a, b] = td

    # Disagreement variables and constraints
    n_pairs = 0
    for a in range(N):
        for b in range(a + 1, N):
            td = target_disagree[a, b]
            d_vars = []
            for k in range(N):
                d = model.NewBoolVar(f'd_{k}_{a}_{b}')
                # d = x[k,a] XOR x[k,b]
                # Encode: XOR(x[k,a], x[k,b], NOT d) = True
                model.AddBoolXOr([x[k, a], x[k, b], d.Not()])
                d_vars.append(d)
            # Sum of disagreements = target
            model.Add(sum(d_vars) == td)
            n_pairs += 1

    if verbose:
        n_xor_vars = N * n_pairs
        print(f"  {n_pairs} pair constraints, {N*N} x-vars, {n_xor_vars} d-vars")
        print(f"  Total vars: {N*N + n_xor_vars}")
        sys.stdout.flush()

    # Symmetry breaking
    add_symmetry_breaking(model, x, G, verbose)

    # Warm start from partial solution
    add_warm_start(model, x, verbose)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = verbose
    solver.parameters.cp_model_presolve = True

    callback = SolutionCallback(x, N)

    if verbose:
        print(f"  Starting CP-SAT solve with {time_limit}s time limit...")
        sys.stdout.flush()

    status = solver.Solve(model, callback)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }.get(status, f"STATUS_{status}")

    if verbose:
        print(f"  Status: {status_name}")
        print(f"  Solutions found: {callback.solution_count}")
        print(f"  Wall time: {solver.WallTime():.1f}s")
        sys.stdout.flush()

    if callback.solution is not None:
        R = callback.solution
        if verify_decomposition(R, G):
            return R
        else:
            if verbose:
                err = int(np.sum((R.T @ R - G) ** 2))
                print(f"  WARNING: Solution does not verify! Error = {err}")

    return None


def approach_cpsat_linear(G, time_limit=3600, verbose=True):
    """
    CP-SAT with linear inner product constraints (no auxiliary d-vars).

    For each pair (a,b):
      sum_k R[k,a]*R[k,b] = G[a,b]

    R[k,a] = 2*x[k,a] - 1 (x in {0,1})

    (2*x[k,a]-1)(2*x[k,b]-1) = 4*x[k,a]*x[k,b] - 2*x[k,a] - 2*x[k,b] + 1

    sum_k [4*x[k,a]*x[k,b] - 2*x[k,a] - 2*x[k,b] + 1] = G[a,b]
    4*sum_k x[k,a]*x[k,b] - 2*sum_k x[k,a] - 2*sum_k x[k,b] + N = G[a,b]

    Let S_a = sum_k x[k,a], S_b = sum_k x[k,b], P_ab = sum_k x[k,a]*x[k,b]
    4*P_ab - 2*S_a - 2*S_b + N = G[a,b]
    P_ab = (G[a,b] - N + 2*S_a + 2*S_b) / 4

    For a=b: P_aa = S_a (since x^2 = x for binary)
    4*S_a - 4*S_a + N = N. Trivially satisfied.

    The product x[k,a]*x[k,b] can be linearized with z[k,a,b].
    But that's a LOT of variables.

    Alternative: Use the AddMultiplicationEquality or AddBoolAnd.

    Actually, for boolean variables, x[k,a] AND x[k,b] can be done efficiently:
    z[k,a,b] = model.NewBoolVar(...)
    model.AddBoolAnd([x[k,a], x[k,b]]).OnlyEnforceIf(z[k,a,b])
    model.AddBoolOr([x[k,a].Not(), x[k,b].Not()]).OnlyEnforceIf(z[k,a,b].Not())

    Then sum_k z[k,a,b] = P_ab = (G[a,b] - N + 2*S_a + 2*S_b) / 4

    But P_ab depends on S_a and S_b which are also variables...

    Better: work directly with the original constraint.
    sum_k [4*z[k,a,b] - 2*x[k,a] - 2*x[k,b]] = G[a,b] - N

    This is what I'll use.
    """
    if verbose:
        print("CP-SAT with linear AND encoding")
        sys.stdout.flush()

    model = cp_model.CpModel()

    x = {}
    for k in range(N):
        for a in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # For each pair, add constraint
    n_pairs = 0
    for a in range(N):
        for b in range(a + 1, N):
            target = int(G[a, b]) - N  # G[a,b] - 29

            z_vars = []
            for k in range(N):
                z = model.NewBoolVar(f'z_{k}_{a}_{b}')
                # z = x[k,a] AND x[k,b]
                model.AddBoolAnd([x[k, a], x[k, b]]).OnlyEnforceIf(z)
                model.AddBoolOr([x[k, a].Not(), x[k, b].Not()]).OnlyEnforceIf(z.Not())
                z_vars.append(z)

            # 4*sum(z) - 2*sum(x[k,a]) - 2*sum(x[k,b]) = target
            model.Add(
                4 * sum(z_vars)
                - 2 * sum(x[k, a] for k in range(N))
                - 2 * sum(x[k, b] for k in range(N))
                == target
            )
            n_pairs += 1

    if verbose:
        print(f"  {n_pairs} pair constraints, {N*N} x-vars, {N*n_pairs} z-vars")
        sys.stdout.flush()

    add_symmetry_breaking(model, x, G, verbose)
    add_warm_start(model, x, verbose)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = verbose

    callback = SolutionCallback(x, N)

    if verbose:
        print(f"  Starting CP-SAT solve with {time_limit}s time limit...")
        sys.stdout.flush()

    status = solver.Solve(model, callback)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }.get(status, f"STATUS_{status}")

    if verbose:
        print(f"  Status: {status_name}")
        print(f"  Solutions found: {callback.solution_count}")
        print(f"  Wall time: {solver.WallTime():.1f}s")
        sys.stdout.flush()

    if callback.solution is not None:
        R = callback.solution
        if verify_decomposition(R, G):
            return R

    return None


def add_symmetry_breaking(model, x, G, verbose=True):
    """
    Add symmetry-breaking constraints exploiting Gram automorphisms.

    Automorphism groups of G:
    - Permutations within {0,1,2}: col 2 is "generic" (IP=1 with all),
      but cols 0,1 are paired. Actually {2} and {5} are interchangeable with
      the "generic" columns {11..28}. Within {0,1,2}: cols 0 and 1 are
      symmetric (both have one IP=9 partner). Col 2 is different.
      So: swap(0,1) with swap(3,4) is an automorphism.
    - Permutations within {3,4,5}: same reasoning, swap(3,4) with swap(0,1).
    - Permutations within {7,8,9,10}: all equivalent (each has IP=-3 with col 6)
    - Permutations within {11,...,28}: all equivalent (IP=1 with everything)
    - Sign changes: can negate any column (R[:,a] -> -R[:,a]) without changing
      R^T R. But this changes ALL inner products involving a... wait, no:
      (R^T R)[a,b] = R[:,a].R[:,b]. Negating column a: R[:,a] -> -R[:,a]
      gives -(R[:,a].R[:,b]) for b!=a, but (R[:,a].R[:,a]) stays same.
      This DOES change off-diagonal, so it's NOT an automorphism of the problem.
      UNLESS we also negate the corresponding row. Hmm.

      Actually, negating ALL entries in row k (R[k,:] -> -R[k,:]) is an automorphism:
      (R^T R)[a,b] = sum_k R[k,a]R[k,b]. Negating row k: (-R[k,a])(-R[k,b]) = R[k,a]R[k,b].
      So row sign changes preserve the Gram. We have 2^29 row sign symmetries.

    Symmetry breaking constraints:
    1. Fix R[0,0] = 1 (WLOG, negate row 0 if needed)
       => x[0,0] = 1
    2. Fix R[0,6] = 1 (WLOG, negate row 0... wait, already fixed row 0)
       Actually, we can fix column signs by choosing R[0,a] = 1 for all a.
       But that might over-constrain. Let's think:

       Row sign change: negate row k. This flips R[k,a] for all a.
       So we can WLOG fix the sign of any one entry per row.
       E.g., fix x[k, 0] = 1 for all k (R[k,0] = 1 for all k).
       But that means column 0 is all +1, which gives col0.col0 = 29 (OK)
       and col0.colb = sum_k R[k,b] = 1 for most b, = 9 for b=3, etc.

       Actually this is a very strong constraint: it fixes column 0 entirely!
       sum_k R[k,b] = G[0,b] for all b.
       For b=3: sum = 9, so 19 entries are +1, 10 are -1.
       For b in {1,2,4,5,6,7,...,28}\{3}: sum = 1, so 15 entries are +1, 14 are -1.

       This is valid and dramatically reduces the search space (kills 2^29 row symmetries).

    3. Column permutation symmetries:
       - Swap 0 <-> 1 with 3 <-> 4: break by sum(col0) >= sum(col1)
         But col0 is fixed (all +1). So sum(col0) = 29.
         We need sum(col1) <= 29, which is always true.
         Hmm, that doesn't help because col 0 is all +1 while col 1 isn't.

       - Actually, with col 0 fixed to all +1, the swap(0,1)(3,4) symmetry
         is already broken. Good.

       - Permutations within {7,8,9,10}: break by ordering.
         sum(col7) >= sum(col8) >= sum(col9) >= sum(col10)
         where sum(col_a) = sum_k R[k,a] = sum_k (2*x[k,a] - 1) = 2*sum_k x[k,a] - 29
         So: sum_k x[k,7] >= sum_k x[k,8] >= sum_k x[k,9] >= sum_k x[k,10]

       - Permutations within {11,...,28}: break by lexicographic ordering.
         For columns 11 through 28, enforce col_{a} >= col_{a+1} lexicographically.
         This is a LOT of constraints but CP-SAT handles it via AddLexLess.
    """
    if verbose:
        print("  Adding symmetry-breaking constraints:")
        sys.stdout.flush()

    # 1. Fix column 0 to all +1 (absorbs 2^29 row sign symmetries)
    for k in range(N):
        model.Add(x[k, 0] == 1)
    if verbose:
        print("    Fixed col 0 = all +1 (kills 2^29 row sign symmetries)")

    # 2. Order columns 7,8,9,10 by column sum
    for a_idx in range(3):
        a = 7 + a_idx
        b = 8 + a_idx
        # sum x[k,a] >= sum x[k,b]
        model.Add(
            sum(x[k, a] for k in range(N)) >= sum(x[k, b] for k in range(N))
        )
    if verbose:
        print("    Ordered cols 7 >= 8 >= 9 >= 10 by column sum")

    # 3. Lexicographic ordering on columns 11..28
    # AddLexLessOrEqual: col_{a+1} <=lex col_a
    for a in range(11, 28):
        col_a = [x[k, a] for k in range(N)]
        col_b = [x[k, a + 1] for k in range(N)]
        model.AddLexLessOrEqual(col_b, col_a)
    if verbose:
        print(f"    Lexicographic ordering on cols 11..28 ({28-11} constraints)")

    # 4. Additional: with col 0 = all +1, we know:
    # R[:,0] . R[:,a] = sum_k R[k,a] = G[0,a]
    # For a=3: sum R[k,3] = 9 => sum x[k,3] = (29+9)/2 = 19
    # For other a: sum R[k,a] = 1 => sum x[k,a] = (29+1)/2 = 15
    # For a=6 with IP=-3 not applicable since G[0,6]=1... wait
    # G[0,6] = 1 (not special). So sum x[k,6] = 15.
    # THESE ARE ALREADY IMPLIED by the pair constraints + col 0 fixed.
    # But making them explicit might help propagation.
    for a in range(1, N):
        target_sum = (N + int(G[0, a])) // 2
        model.Add(sum(x[k, a] for k in range(N)) == target_sum)
    if verbose:
        print(f"    Explicit column sum constraints from col 0 (all implied)")

    sys.stdout.flush()


def add_warm_start(model, x, verbose=True):
    """Add solution hints from partial 11-column solution."""
    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)

    if verbose:
        print("  Adding warm start hints from partial 11-column solution")
        sys.stdout.flush()

    # partial is 29 x 11, columns 0..10
    # But with col 0 fixed to all +1, we need to check if partial col 0 is all +1
    # If not, we need to negate the appropriate rows

    col0 = partial[:, 0]
    # We fixed col 0 to all +1. If partial has col0 != all+1, negate rows where col0 = -1
    negate_rows = (col0 == -1)

    for col_idx in range(11):
        col = partial[:, col_idx].copy()
        col[negate_rows] *= -1  # Apply row negations to make col 0 all +1
        for k in range(N):
            hint_val = 1 if col[k] == 1 else 0
            model.AddHint(x[k, col_idx], hint_val)

    # For columns 11..28, we don't have hints from partial solution
    # Use a heuristic: set to +1 (since col 0 is all +1 and IP=1 means ~15 agree)
    # Actually, random hints might be better than biased ones
    rng = np.random.RandomState(42)
    for a in range(11, N):
        target_ones = (N + int(build_new_gram()[0, a])) // 2  # = 15 for IP=1
        hint = np.ones(N, dtype=int)
        hint[rng.choice(N, N - target_ones, replace=False)] = 0
        for k in range(N):
            model.AddHint(x[k, a], int(hint[k]))

    if verbose:
        print(f"    Hints set for all {N} columns")
        sys.stdout.flush()


def approach_cpsat_incremental(G, time_limit=3600, verbose=True):
    """
    Incremental CP-SAT: Start with a subset of columns, solve, then add more.

    Strategy:
    1. Fix the first 11 columns from partial solution
    2. Add columns one at a time, solving the linear constraints first
    3. Only add bilinear (between-new-column) constraints as needed
    """
    if verbose:
        print("CP-SAT Incremental approach")
        sys.stdout.flush()

    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)

    # Negate rows to make col 0 all +1
    col0 = partial[:, 0]
    negate_rows = (col0 == -1)
    fixed_cols = {}
    for col_idx in range(11):
        col = partial[:, col_idx].copy()
        col[negate_rows] *= -1
        fixed_cols[col_idx] = col

    # Verify fixed columns
    for a in fixed_cols:
        for b in fixed_cols:
            if a >= b:
                continue
            ip = int(np.dot(fixed_cols[a], fixed_cols[b]))
            expected = int(G[a, b])
            if ip != expected:
                print(f"  ERROR: IP({a},{b}) = {ip}, expected {expected}")
                return None

    if verbose:
        print(f"  Fixed columns 0..10 from partial solution")
        print(f"  Remaining: columns 11..28 (18 columns)")
        sys.stdout.flush()

    # Now solve for columns 11..28
    remaining = list(range(11, N))

    model = cp_model.CpModel()

    # Variables for remaining columns only
    x = {}
    for a in remaining:
        for k in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # Constraints: IP with fixed columns (linear constraints)
    for a in remaining:
        for b in range(11):
            target_ip = int(G[a, b])
            # sum_k R[k,a]*fixed_cols[b][k] = target_ip
            # sum_k (2*x[k,a]-1)*fixed_cols[b][k] = target_ip
            # 2*sum_k x[k,a]*fixed_cols[b][k] - sum_k fixed_cols[b][k] = target_ip
            col_b_sum = int(np.sum(fixed_cols[b]))
            # sum_k x[k,a]*fixed_cols[b][k]:
            # when fixed_cols[b][k] = 1: contributes x[k,a]
            # when fixed_cols[b][k] = -1: contributes -x[k,a]
            # So: sum_{k: col_b[k]=1} x[k,a] - sum_{k: col_b[k]=-1} x[k,a]
            pos_k = [k for k in range(N) if fixed_cols[b][k] == 1]
            neg_k = [k for k in range(N) if fixed_cols[b][k] == -1]
            # 2*(sum_pos x - sum_neg x) - col_b_sum = target_ip
            # 2*sum_pos x - 2*sum_neg x = target_ip + col_b_sum
            rhs = target_ip + col_b_sum
            assert rhs % 2 == 0, f"Parity error: IP({a},{b}) target={target_ip}, col_sum={col_b_sum}"
            model.Add(
                2 * sum(x[k, a] for k in pos_k) - 2 * sum(x[k, a] for k in neg_k) == rhs
            )

    # Constraints: IP between remaining columns (XOR encoding)
    for idx_a, a in enumerate(remaining):
        for idx_b, b in enumerate(remaining):
            if a >= b:
                continue
            target_ip = int(G[a, b])  # = 1 for all remaining pairs
            td = (N - target_ip) // 2  # = 14

            d_vars = []
            for k in range(N):
                d = model.NewBoolVar(f'd_{k}_{a}_{b}')
                model.AddBoolXOr([x[k, a], x[k, b], d.Not()])
                d_vars.append(d)
            model.Add(sum(d_vars) == td)

    # Symmetry breaking: lexicographic ordering on remaining columns
    for idx in range(len(remaining) - 1):
        a = remaining[idx]
        b = remaining[idx + 1]
        col_a = [x[k, a] for k in range(N)]
        col_b = [x[k, b] for k in range(N)]
        model.AddLexLessOrEqual(col_b, col_a)

    if verbose:
        print(f"  Model: {18*N} x-vars + XOR d-vars for {18*17//2} pairs")
        sys.stdout.flush()

    # Warm start for remaining columns
    rng = np.random.RandomState(42)
    for a in remaining:
        target_ones = (N + 1) // 2  # IP=1 with col 0 (all +1)
        hint = np.ones(N, dtype=int)
        hint[rng.choice(N, N - target_ones, replace=False)] = 0
        for k in range(N):
            model.AddHint(x[k, a], int(hint[k]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = verbose

    if verbose:
        print(f"  Starting solve ({time_limit}s)...")
        sys.stdout.flush()

    status = solver.Solve(model)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
    }.get(status, f"STATUS_{status}")

    if verbose:
        print(f"  Status: {status_name}")
        print(f"  Wall time: {solver.WallTime():.1f}s")
        sys.stdout.flush()

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        R = np.zeros((N, N), dtype=np.int64)
        for b in range(11):
            R[:, b] = fixed_cols[b]
        for a in remaining:
            for k in range(N):
                val = solver.Value(x[k, a])
                R[k, a] = 2 * val - 1

        if verify_decomposition(R, G):
            if verbose:
                print("  VERIFIED!")
            return R
        else:
            err = int(np.sum((R.T @ R - G) ** 2))
            if verbose:
                print(f"  Solution does not verify, error = {err}")

    return None


def approach_cpsat_column_by_column(G, time_limit=3600, verbose=True):
    """
    Solve column by column: fix 11 columns, then add one column at a time.
    Each step is a small CP-SAT that should be very fast.
    Use backtracking if a step fails.
    """
    if verbose:
        print("CP-SAT column-by-column with backtracking")
        sys.stdout.flush()

    deadline = time.time() + time_limit

    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)

    col0 = partial[:, 0]
    negate_rows = (col0 == -1)
    fixed_cols = {}
    for col_idx in range(11):
        col = partial[:, col_idx].copy()
        col[negate_rows] *= -1
        fixed_cols[col_idx] = col

    remaining = list(range(11, N))
    placed = dict(fixed_cols)  # col_index -> vector

    best_depth = 11

    def solve_one_column(col_idx, placed_cols_dict, time_limit_single=60):
        """Find all valid +-1 vectors for column col_idx given placed columns."""
        model = cp_model.CpModel()
        xvars = [model.NewBoolVar(f'x_{k}') for k in range(N)]

        for b, col_b in placed_cols_dict.items():
            target_ip = int(G[col_idx, b])
            col_b_sum = int(np.sum(col_b))
            pos_k = [k for k in range(N) if col_b[k] == 1]
            neg_k = [k for k in range(N) if col_b[k] == -1]
            rhs = target_ip + col_b_sum
            assert rhs % 2 == 0
            model.Add(
                2 * sum(xvars[k] for k in pos_k) - 2 * sum(xvars[k] for k in neg_k) == rhs
            )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = min(time_limit_single, max(1, deadline - time.time()))
        solver.parameters.num_workers = 4
        solver.parameters.enumerate_all_solutions = True

        solutions = []

        class AllSolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.count = 0

            def on_solution_callback(self):
                self.count += 1
                if self.count > 10000:
                    self.StopSearch()
                    return
                vec = np.zeros(N, dtype=np.int64)
                for k in range(N):
                    vec[k] = 2 * self.Value(xvars[k]) - 1
                solutions.append(vec)

        collector = AllSolutionCollector()
        solver.Solve(model, collector)

        return solutions

    def backtrack(step):
        nonlocal best_depth
        if time.time() > deadline:
            return False
        if step == len(remaining):
            return True

        col_idx = remaining[step]
        current_depth = 11 + step

        if current_depth > best_depth:
            best_depth = current_depth
            if verbose:
                elapsed = time.time() - (deadline - time_limit)
                print(f"  Depth {current_depth}/{N} (col {col_idx}), {elapsed:.1f}s")
                sys.stdout.flush()

        solutions = solve_one_column(col_idx, placed, time_limit_single=30)

        if verbose and step <= 5:
            print(f"    Col {col_idx}: {len(solutions)} solutions")
            sys.stdout.flush()

        if not solutions:
            return False

        # Shuffle for diversity
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
        rng.shuffle(solutions)

        # Try solutions
        n_try = min(len(solutions), 50 if step < 5 else (20 if step < 10 else 5))
        for sol in solutions[:n_try]:
            if time.time() > deadline:
                return False
            placed[col_idx] = sol
            if backtrack(step + 1):
                return True
            del placed[col_idx]

        return False

    if backtrack(0):
        R = np.zeros((N, N), dtype=np.int64)
        for ci, col in placed.items():
            R[:, ci] = col
        if verify_decomposition(R, G):
            if verbose:
                print("  SUCCESS! Full decomposition found!")
            return R

    if verbose:
        print(f"  Best depth: {best_depth}/{N}")
    return None


def approach_cpsat_isomorphic_grams(G_original, time_limit=3600, verbose=True):
    """
    Try different isomorphic Gram matrices.

    The two "9" entries can be placed at different pairs from the cross-block
    {0,1,2} x {3,4,5}. There are C(9,2) = 36 ways to choose 2 pairs, but
    we need the 2 pairs to be "matchable" (each row/col appears at most once
    in the high-IP pairs for the structure to be isomorphic).

    Actually, the structure is: two disjoint pairs from {0,1,2}x{3,4,5} where
    each pair connects one from {0,1,2} to one from {3,4,5}.
    The number of such matchings: 3*2 = 6 (but removing the original leaves 5).

    But we can also permute which indices are "special" by choosing different
    groups of indices for the roles. The key structure is:
    - 2 pairs of columns with IP=9
    - 1 column (6) with IP=-3 to 4 others (7-10)
    - All others have IP=1

    We can permute ALL column indices, keeping the structure.
    """
    if verbose:
        print("CP-SAT on isomorphic Gram variants")
        sys.stdout.flush()

    deadline = time.time() + time_limit

    # Generate different valid placements of the two "9" pairs
    # Original: (0,3), (1,4)
    # Other options for the cross-block pattern:
    cross_pairs = [
        [(0, 3), (1, 4)],  # original
        [(0, 3), (2, 4)],
        [(0, 3), (1, 5)],
        [(0, 3), (2, 5)],
        [(0, 4), (1, 3)],
        [(0, 4), (2, 3)],
        [(0, 4), (1, 5)],
        [(0, 4), (2, 5)],
        [(0, 5), (1, 3)],
        [(0, 5), (2, 3)],
        [(0, 5), (1, 4)],
        [(0, 5), (2, 4)],
        [(1, 3), (2, 4)],
        [(1, 3), (2, 5)],
        [(1, 4), (2, 3)],
        [(1, 4), (2, 5)],
        [(1, 5), (2, 3)],
        [(1, 5), (2, 4)],
    ]

    time_per_variant = max(60, time_limit // len(cross_pairs))

    for variant_idx, pairs in enumerate(cross_pairs):
        if time.time() > deadline:
            break

        # Build this variant's Gram
        G = np.ones((N, N), dtype=np.int64)
        np.fill_diagonal(G, N)
        for i, j in pairs:
            G[i, j] = 9; G[j, i] = 9
        for j in range(7, 11):
            G[6, j] = -3; G[j, 6] = -3

        if verbose:
            print(f"\n  Variant {variant_idx}: pairs = {pairs}")
            sys.stdout.flush()

        # Use the incremental approach (fastest)
        R = approach_cpsat_incremental(G, time_limit=min(time_per_variant, deadline - time.time()), verbose=verbose)

        if R is not None and verify_decomposition(R, G):
            if verbose:
                print(f"  FOUND for variant {variant_idx}!")

            # Now convert back to original Gram via permutation
            # The original has (0,3) and (1,4).
            # This variant has pairs[0] and pairs[1].
            # We need a permutation sigma such that G_original = P^T G_variant P
            # where P is the permutation matrix for sigma.
            # Then R_original = R_variant @ P

            # Actually, if G_variant is just G_original with columns/rows permuted,
            # we need to find the permutation and apply it.
            # For now, if pairs == [(0,3),(1,4)], R works directly for original.

            if np.array_equal(G, G_original):
                return R

            # Need to find permutation from variant to original
            # The permutation swaps indices to map variant pairs to original pairs
            # This is straightforward for the cross-block structure

            # Map: variant pair (a1,b1) -> original pair (0,3)
            #       variant pair (a2,b2) -> original pair (1,4)
            a1, b1 = pairs[0]
            a2, b2 = pairs[1]

            # Build permutation
            perm = list(range(N))
            # We need: perm[0] = a1, perm[3] = b1, perm[1] = a2, perm[4] = b2
            # And perm[2] = remaining from {0,1,2}, perm[5] = remaining from {3,4,5}

            used_left = {a1, a2}
            used_right = {b1, b2}
            rem_left = [x for x in [0, 1, 2] if x not in used_left][0]
            rem_right = [x for x in [3, 4, 5] if x not in used_right][0]

            # perm maps original index -> variant index
            perm[0] = a1; perm[3] = b1
            perm[1] = a2; perm[4] = b2
            perm[2] = rem_left; perm[5] = rem_right
            # Indices 6..28 stay the same

            # R_original[:, original_col] = R_variant[:, variant_col]
            # R_original[:, j] = R[:, perm[j]]
            R_original = np.zeros_like(R)
            for j in range(N):
                R_original[:, j] = R[:, perm[j]]

            if verify_decomposition(R_original, G_original):
                return R_original
            else:
                if verbose:
                    print(f"  Permutation mapping failed, trying direct variant")
                # Return the variant solution anyway - it decomposes SOME valid Gram
                return R

    return None


def main():
    G = build_new_gram()
    print("=" * 70)
    print("CP-SAT GRAM DECOMPOSITION: 29x29 matrix R with R^T R = G")
    print("=" * 70)
    sys.stdout.flush()

    # Approach 1: Incremental (fix 11 cols, solve 18)
    # This is the most promising since it has far fewer variables
    print("\n" + "=" * 70)
    print("APPROACH 1: CP-SAT Incremental (fix 11 cols from partial solution)")
    print("=" * 70)
    sys.stdout.flush()
    R = approach_cpsat_incremental(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
        return R

    # Approach 2: Column-by-column with CP-SAT solving each column
    print("\n" + "=" * 70)
    print("APPROACH 2: CP-SAT column-by-column")
    print("=" * 70)
    sys.stdout.flush()
    R = approach_cpsat_column_by_column(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
        return R

    # Approach 3: Full XOR encoding
    print("\n" + "=" * 70)
    print("APPROACH 3: CP-SAT Full XOR encoding")
    print("=" * 70)
    sys.stdout.flush()
    R = approach_cpsat_xor(G, time_limit=1800, verbose=True)
    if R is not None and verify_decomposition(R, G):
        print("\n*** BREAKTHROUGH! ***")
        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
        return R

    print("\nAll CP-SAT approaches exhausted.")
    return None


if __name__ == "__main__":
    main()
