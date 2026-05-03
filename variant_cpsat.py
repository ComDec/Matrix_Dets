"""
CP-SAT solver for Gram matrix decomposition across all 18 variants.

For each variant, formulate: find R (29x29, all entries +/-1) such that R^T R = G.
This is a constraint satisfaction problem with 29*29 = 841 binary variables
and 29*30/2 = 435 inner product constraints.

Key CP-SAT optimizations:
1. Use bool vars (x[i][j] = 1 means R[i][j] = +1, 0 means R[i][j] = -1)
2. Inner product R[:,a] . R[:,b] = sum_i R[i][a]*R[i][b] = G[a][b]
   Using x: R[i][j] = 2*x[i][j] - 1
   Product: R[i][a]*R[i][b] = (2*x[i][a]-1)(2*x[i][b]-1)
                              = 4*x[i][a]*x[i][b] - 2*x[i][a] - 2*x[i][b] + 1
   Sum over i: sum = 4*sum(x[i][a]*x[i][b]) - 2*sum(x[i][a]) - 2*sum(x[i][b]) + 29
   Define match[a][b] = sum_i (x[i][a] == x[i][b]) = number of rows where cols a,b agree
   Then inner product = 2*match - 29 (since agree contributes +1, disagree -1)
   So match[a][b] = (G[a][b] + 29) / 2

3. Symmetry breaking: fix first column to all +1 (or use other symmetry breaking)
4. Propagation hints from partial solutions
"""
import numpy as np
import time
import sys
import itertools
from ortools.sat.python import cp_model

N = 29
THEORETICAL_MAX = 1270698346568170340352


def enumerate_variants():
    rows = [0, 1, 2]
    cols = [3, 4, 5]
    variants = []
    for r_pair in itertools.combinations(rows, 2):
        for c_pair in itertools.combinations(cols, 2):
            variants.append(((r_pair[0], c_pair[0]), (r_pair[1], c_pair[1])))
            variants.append(((r_pair[0], c_pair[1]), (r_pair[1], c_pair[0])))
    return variants


def build_gram(nine_pairs):
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for (i, j) in nine_pairs:
        G[i, j] = 9; G[j, i] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, n):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.x = x_vars
        self.n = n
        self.solutions = []
        self.start_time = time.time()

    def on_solution_callback(self):
        R = np.zeros((self.n, self.n), dtype=np.int64)
        for i in range(self.n):
            for j in range(self.n):
                R[i][j] = 2 * self.Value(self.x[i][j]) - 1
        self.solutions.append(R)
        elapsed = time.time() - self.start_time
        print(f"    Solution found! ({elapsed:.1f}s)", flush=True)
        self.StopSearch()


def solve_cpsat(G, time_limit=600, n_workers=8, hints=None, variant_id=-1):
    """Solve R^T R = G using CP-SAT."""
    model = cp_model.CpModel()

    # Variables: x[i][j] in {0, 1} representing R[i][j] = 2*x[i][j] - 1
    x = [[model.NewBoolVar(f'x_{i}_{j}') for j in range(N)] for i in range(N)]

    # Constraints: for each pair of columns (a, b) where a <= b:
    # sum_i R[i][a]*R[i][b] = G[a][b]
    # Using match count: match[a][b] = (G[a][b] + N) / 2
    for a in range(N):
        for b in range(a, N):
            target_ip = int(G[a, b])
            if (target_ip + N) % 2 != 0:
                print(f"  ERROR: G[{a},{b}]={target_ip} gives non-integer match count")
                return None
            match_count = (target_ip + N) // 2

            if a == b:
                # Self inner product = N (always satisfied for +-1 vectors)
                continue

            # match[a][b] = number of rows i where x[i][a] == x[i][b]
            # Define agree[i] = (x[i][a] == x[i][b]) for each row i
            # agree[i] = x[i][a]*x[i][b] + (1-x[i][a])*(1-x[i][b])
            #           = 2*x[i][a]*x[i][b] - x[i][a] - x[i][b] + 1
            # sum_i agree[i] = 2*sum(x[i][a]*x[i][b]) - sum(x[i][a]) - sum(x[i][b]) + N
            #
            # Alternative: use auxiliary variables
            # agree_i = 1 if x[i][a] == x[i][b], 0 otherwise
            # But this requires N auxiliary variables per pair.
            #
            # Better: define the product directly.
            # Let p[i] = x[i][a] * x[i][b] (AND)
            # agree[i] = p[i] + (1 - x[i][a] - x[i][b] + p[i]) = 2*p[i] - x[i][a] - x[i][b] + 1
            # sum agree = 2*sum(p) - sum(x[,a]) - sum(x[,b]) + N = match_count
            # So: 2*sum(p) - sum(x[,a]) - sum(x[,b]) = match_count - N

            # Using CP-SAT's native support for linear constraints with products:
            # sum_i (2*x[i][a]-1)*(2*x[i][b]-1) = G[a][b]
            # Expand: sum_i (4*x[i][a]*x[i][b] - 2*x[i][a] - 2*x[i][b] + 1) = G[a][b]
            # 4*sum(p) - 2*sum(x[,a]) - 2*sum(x[,b]) + N = G[a][b]
            # sum(p) = (G[a][b] + 2*sum(x[,a]) + 2*sum(x[,b]) - N) / 4
            # This is hard because it couples with other constraints.

            # Simpler approach: define y[i][a][b] = AND(x[i][a], x[i][b])
            # Then: 4*sum(y) - 2*sum(x[,a]) - 2*sum(x[,b]) + N = G[a][b]

            # But 29*29*28/2 = 11,774 auxiliary variables. That's manageable.

            # Actually, CP-SAT can handle multiplication of bool vars natively:
            # model.Add(sum(2*x[i][a]*x[i][b] for i) == something)
            # BUT: multiplication isn't directly supported. Need AddMultiplicationEquality.
            # Or use the BoolAnd/AddImplication approach.

            # Most efficient: use the half-reification approach
            # For each row i, define p_i = x[i][a] AND x[i][b]
            # Then sum(p_i) = count of rows where both are 1
            # Similarly, define q_i = (1-x[i][a]) AND (1-x[i][b])  (both are 0/both are -1)
            # Then match_count = sum(p_i) + sum(q_i)
            #
            # q_i = 1 - x[i][a] - x[i][b] + p_i  (DeMorgan)
            # Actually no: q_i = NOT(x[i][a]) AND NOT(x[i][b])
            # sum(q_i) = sum(1 - x[i][a] - x[i][b] + x[i][a]*x[i][b])
            #          = N - sum(x[,a]) - sum(x[,b]) + sum(p_i)
            # match = sum(p) + sum(q) = sum(p) + N - sum(x[,a]) - sum(x[,b]) + sum(p)
            #       = 2*sum(p) + N - sum(x[,a]) - sum(x[,b])
            # So: 2*sum(p) = match - N + sum(x[,a]) + sum(x[,b])
            #     2*sum(p) = (G[a,b]+N)/2 - N + sum(x[,a]) + sum(x[,b])
            #     2*sum(p) = (G[a,b]-N)/2 + sum(x[,a]) + sum(x[,b])

            # Let's just use the p_i approach:
            p_vars = []
            for i in range(N):
                p_i = model.NewBoolVar(f'p_{a}_{b}_{i}')
                # p_i = x[i][a] AND x[i][b]
                model.AddBoolAnd([x[i][a], x[i][b]]).OnlyEnforceIf(p_i)
                model.AddBoolOr([x[i][a].Not(), x[i][b].Not()]).OnlyEnforceIf(p_i.Not())
                p_vars.append(p_i)

            # Constraint: 2*sum(p) + N - sum(x[,a]) - sum(x[,b]) = match_count
            model.Add(
                2 * sum(p_vars) + N - sum(x[i][a] for i in range(N)) - sum(x[i][b] for i in range(N))
                == match_count
            )

    # Symmetry breaking: fix first row to all +1
    # (Any solution can be multiplied by a diagonal +-1 matrix to make first row all +1)
    for j in range(N):
        model.Add(x[0][j] == 1)

    # Additional symmetry breaking: fix column 0, row 1 = +1
    # Actually, we can fix first column to be non-decreasing (in terms of 0/1)
    # But be careful not to over-constrain.

    # Hints from partial solution
    if hints is not None:
        for i in range(N):
            for j in range(N):
                if hints[i][j] != 0:
                    val = 1 if hints[i][j] == 1 else 0
                    model.AddHint(x[i][j], val)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = n_workers
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 2

    callback = SolutionCallback(x, N)
    print(f"  Solving CP-SAT (time limit {time_limit}s, {n_workers} workers)...", flush=True)
    status = solver.Solve(model, callback)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }

    print(f"  Status: {status_name.get(status, status)}", flush=True)
    print(f"  Wall time: {solver.WallTime():.1f}s", flush=True)

    if callback.solutions:
        return callback.solutions[0]
    return None


def solve_cpsat_efficient(G, time_limit=600, n_workers=8, variant_id=-1):
    """More efficient CP-SAT formulation using linear constraints only."""
    model = cp_model.CpModel()

    # Variables: x[i][j] in {0, 1} where R[i][j] = 2*x[i][j] - 1
    x = [[model.NewBoolVar(f'x_{i}_{j}') for j in range(N)] for i in range(N)]

    # For each pair of columns (a, b), a < b:
    # Inner product = sum_i R[i][a]*R[i][b] = G[a][b]
    #
    # Define y[i] = 1 if x[i][a] == x[i][b], 0 otherwise
    # Inner product = sum_i (2*y[i] - 1) = 2*sum(y) - N
    # So sum(y) = (G[a][b] + N) / 2
    #
    # y[i] = x[i][a] XNOR x[i][b] = 1 - (x[i][a] XOR x[i][b])
    # XOR: z[i] = x[i][a] XOR x[i][b]
    # sum(z) = N - (G[a][b] + N)/2 = (N - G[a][b])/2

    for a in range(N):
        for b in range(a + 1, N):
            target_ip = int(G[a, b])
            n_disagree = (N - target_ip) // 2  # number of rows where cols disagree

            # sum of XOR(x[i][a], x[i][b]) = n_disagree
            # XOR(a,b) = a + b - 2*a*b
            # So sum(x[i][a] + x[i][b] - 2*x[i][a]*x[i][b]) = n_disagree
            # sum(x[,a]) + sum(x[,b]) - 2*sum(AND(x[i][a], x[i][b])) = n_disagree

            # Define p[i] = AND(x[i][a], x[i][b])
            p_vars = []
            for i in range(N):
                p_i = model.NewBoolVar(f'p_{a}_{b}_{i}')
                model.AddBoolAnd([x[i][a], x[i][b]]).OnlyEnforceIf(p_i)
                model.AddBoolOr([x[i][a].Not(), x[i][b].Not()]).OnlyEnforceIf(p_i.Not())
                p_vars.append(p_i)

            model.Add(
                sum(x[i][a] for i in range(N)) +
                sum(x[i][b] for i in range(N)) -
                2 * sum(p_vars) == n_disagree
            )

    # Symmetry breaking
    for j in range(N):
        model.Add(x[0][j] == 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = n_workers
    solver.parameters.log_search_progress = False

    callback = SolutionCallback(x, N)
    print(f"  Solving CP-SAT (time limit {time_limit}s)...", flush=True)
    t0 = time.time()
    status = solver.Solve(model, callback)
    elapsed = time.time() - t0

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"  Status: {status_name.get(status, status)} ({elapsed:.1f}s)", flush=True)

    if callback.solutions:
        return callback.solutions[0]
    return None


def solve_cpsat_compact(G, time_limit=600, n_workers=8, variant_id=-1):
    """Most compact CP-SAT formulation.

    Use IntVar for column sums and product sums directly,
    avoiding per-row auxiliary variables.
    """
    model = cp_model.CpModel()

    # Variables: x[i][j] in {0, 1}
    x = [[model.NewBoolVar(f'x_{i}_{j}') for j in range(N)] for i in range(N)]

    # Column sums
    col_sum = []
    for j in range(N):
        s = model.NewIntVar(0, N, f'cs_{j}')
        model.Add(s == sum(x[i][j] for i in range(N)))
        col_sum.append(s)

    # For each pair (a, b) with a < b:
    # We need: 4*sum(AND(x[i][a], x[i][b])) - 2*cs[a] - 2*cs[b] + N = G[a][b]
    # sum(AND) = (G[a][b] - N + 2*cs[a] + 2*cs[b]) / 4
    # This requires auxiliary AND vars.

    # Alternative: use the AddAllowedAssignments for pairs of columns
    # But that's 2^29 tuples per pair -- way too many.

    # Let's just use the AND approach but be smarter about which pairs matter.
    # Most pairs have G[a][b] = 1, so n_disagree = (29-1)/2 = 14
    # Some have G[a][b] = 9, n_disagree = (29-9)/2 = 10
    # Some have G[a][b] = -3, n_disagree = (29+3)/2 = 16

    for a in range(N):
        for b in range(a + 1, N):
            target_ip = int(G[a, b])
            n_disagree = (N - target_ip) // 2

            # XOR approach: sum(XOR(x[i][a], x[i][b])) = n_disagree
            # Without auxiliary vars, use the relation:
            # XOR(a,b) = a + b - 2*AND(a,b)
            # sum(XOR) = cs[a] + cs[b] - 2*sum(AND)
            # sum(AND) = (cs[a] + cs[b] - n_disagree) / 2

            # We need sum(AND) to be integer, so cs[a]+cs[b]-n_disagree must be even.
            # This is a parity constraint on column sums.

            # Actually, let's define and_sum[a][b] = sum(AND(x[i][a], x[i][b]))
            and_vars = []
            for i in range(N):
                p = model.NewBoolVar(f'a_{a}_{b}_{i}')
                model.AddBoolAnd([x[i][a], x[i][b]]).OnlyEnforceIf(p)
                model.AddBoolOr([x[i][a].Not(), x[i][b].Not()]).OnlyEnforceIf(p.Not())
                and_vars.append(p)

            and_sum = model.NewIntVar(0, N, f'as_{a}_{b}')
            model.Add(and_sum == sum(and_vars))

            # Constraint: cs[a] + cs[b] - 2*and_sum = n_disagree
            model.Add(col_sum[a] + col_sum[b] - 2 * and_sum == n_disagree)

    # Symmetry breaking
    for j in range(N):
        model.Add(x[0][j] == 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = n_workers
    solver.parameters.log_search_progress = False

    callback = SolutionCallback(x, N)
    print(f"  Solving CP-SAT compact (time limit {time_limit}s)...", flush=True)
    t0 = time.time()
    status = solver.Solve(model, callback)
    elapsed = time.time() - t0

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"  Status: {status_name.get(status, status)} ({elapsed:.1f}s)", flush=True)

    if callback.solutions:
        return callback.solutions[0]
    return None


def main():
    print("="*70)
    print("CP-SAT VARIANT DECOMPOSITION SEARCH")
    print("="*70)
    sys.stdout.flush()

    variants = enumerate_variants()

    # Try variant 0 first as a test
    variant_id = 0
    p1, p2 = variants[variant_id]
    G = build_gram([p1, p2])
    print(f"\nVariant {variant_id}: G[{p1[0]},{p1[1]}]=9, G[{p2[0]},{p2[1]}]=9")

    R = solve_cpsat_efficient(G, time_limit=3600, n_workers=8, variant_id=variant_id)

    if R is not None:
        if np.array_equal(R.T @ R, G):
            print("VERIFIED!")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            det_val = abs(np.linalg.det(R.astype(float)))
            score = det_val / THEORETICAL_MAX
            print(f"Score = {score:.6f}")
        else:
            print("Verification FAILED")
    else:
        print("No solution found")


if __name__ == "__main__":
    main()
