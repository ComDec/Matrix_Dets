"""
Full 29x29 CP-SAT Gram decomposition with optimal encoding.

Key insights:
1. Fix column 0 = all +1 (kills 2^29 row-sign symmetries)
2. This implies column sum constraints: sum(col_a) = G[0,a] for all a
3. Use XOR encoding for pairwise inner product constraints
4. Strong symmetry breaking for interchangeable columns
5. Warm start from best known partial solution
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


class BreakthroughCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, G):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.x = x_vars
        self.G = G
        self.solution = None
        self.count = 0

    def on_solution_callback(self):
        self.count += 1
        R = np.zeros((N, N), dtype=np.int64)
        for k in range(N):
            for a in range(N):
                R[k, a] = 2 * self.Value(self.x[k, a]) - 1
        if verify_decomposition(R, self.G):
            self.solution = R
            print(f"  *** VALID SOLUTION #{self.count} ***")
            np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
        else:
            err = int(np.sum((R.T @ R - self.G) ** 2))
            print(f"  Solution #{self.count}: verification error = {err}")
        sys.stdout.flush()
        self.StopSearch()


def build_and_solve(G, time_limit=3600, verbose=True):
    """Build full CP-SAT model with XOR encoding."""

    if verbose:
        print(f"Building full 29x29 CP-SAT model...")
        sys.stdout.flush()

    model = cp_model.CpModel()

    # Variables: x[k,a] in {0,1}, R[k,a] = 2*x[k,a] - 1
    x = {}
    for k in range(N):
        for a in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # === CONSTRAINT 1: Fix column 0 = all +1 ===
    # This absorbs all 2^29 row-sign symmetries
    for k in range(N):
        model.Add(x[k, 0] == 1)

    if verbose:
        print("  Fixed col 0 = all +1")

    # === CONSTRAINT 2: Column sum constraints ===
    # With col 0 = all +1: R[:,0] . R[:,a] = sum_k R[k,a] = G[0,a]
    # sum_k (2*x[k,a] - 1) = G[0,a]
    # 2*sum_k x[k,a] - N = G[0,a]
    # sum_k x[k,a] = (N + G[0,a]) / 2
    for a in range(1, N):
        target = (N + int(G[0, a])) // 2
        model.Add(sum(x[k, a] for k in range(N)) == target)

    if verbose:
        print(f"  Added column sum constraints for cols 1..{N-1}")

    # === CONSTRAINT 3: Pairwise inner product constraints ===
    # For each pair (a,b) with 1 <= a < b <= 28:
    # (we skip a=0 since col 0 is already handled by column sums)
    #
    # R[:,a] . R[:,b] = G[a,b]
    # Disagreements: d = (N - G[a,b]) / 2
    # sum_k (x[k,a] XOR x[k,b]) = d

    n_pair_constraints = 0
    for a in range(1, N):
        for b in range(a + 1, N):
            target_ip = int(G[a, b])
            td = (N - target_ip) // 2  # number of disagreements

            # Create disagreement variables
            d_vars = []
            for k in range(N):
                d = model.NewBoolVar(f'd_{k}_{a}_{b}')
                # d = x[k,a] XOR x[k,b]
                model.AddBoolXOr([x[k, a], x[k, b], d.Not()])
                d_vars.append(d)

            model.Add(sum(d_vars) == td)
            n_pair_constraints += 1

    if verbose:
        print(f"  Added {n_pair_constraints} pairwise XOR constraints")

    # === SYMMETRY BREAKING ===

    # Group 1: Columns {1,2,5} and columns {11..28} are all "generic"
    # (IP=1 with everything, no special structure)
    # Wait, let me recheck:
    # Col 1: IP=9 with col 4, IP=1 with all others
    # Col 2: IP=1 with all others (truly generic among cols 0-10)
    # Col 5: IP=1 with all others (truly generic among cols 0-10)
    # Cols 11-28: IP=1 with all others

    # So cols {2, 5, 11, 12, ..., 28} = 20 columns are all generic (IP=1 with everything)
    # These are interchangeable! We can break symmetry by ordering them.

    generic_cols = [2, 5] + list(range(11, 29))  # 20 generic columns
    if verbose:
        print(f"  Generic (fully interchangeable) columns: {generic_cols}")

    # Light symmetry breaking: fix first row entry ordering for interchangeable cols
    # For generic cols: x[1, col_a] >= x[1, col_b] (breaks some symmetry)
    # This is very light but avoids adding many auxiliary variables
    for i in range(len(generic_cols) - 1):
        a = generic_cols[i]
        b = generic_cols[i + 1]
        # x[1,a] >= x[1,b]: if b's row-1 entry is 1, then a's must be too
        model.AddImplication(x[1, b], x[1, a])

    if verbose:
        print(f"  Row-1 ordering on {len(generic_cols)} generic columns")

    # Group 2: Columns 7,8,9,10
    for i in range(3):
        a = 7 + i
        b = 8 + i
        model.AddImplication(x[1, b], x[1, a])

    if verbose:
        print(f"  Lexicographic ordering on cols 7,8,9,10")

    # Group 3: (col0, col3) and (col1, col4) are structurally similar pairs
    # Swap (0<->1, 3<->4) is a symmetry. Break by: col 0 is fixed,
    # so compare col 3 vs col 4 lexicographically (since col 0 > col 1 in some sense)
    # Actually (0,3) and (1,4) both have IP=9, and the rest of their
    # relationships are identical. So swapping (0<->1)(3<->4) is a symmetry.
    # But col 0 is fixed to all-ones. After this fixing, col 1 is not fixed.
    # The symmetry (0<->1)(3<->4) would map col0=all-ones to col1, which is NOT all-ones.
    # So this symmetry is already broken by fixing col 0. Good.

    # === WARM START ===
    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)
    col0_partial = partial[:, 0]
    negate_mask = (col0_partial == -1)

    for col_idx in range(11):
        col = partial[:, col_idx].copy()
        col[negate_mask] *= -1
        for k in range(N):
            model.AddHint(x[k, col_idx], 1 if col[k] == 1 else 0)

    # For remaining columns, use random hints consistent with column sums
    rng = np.random.RandomState(42)
    for a in range(11, N):
        target_ones = (N + int(G[0, a])) // 2
        hint = np.zeros(N, dtype=int)
        hint[rng.choice(N, target_ones, replace=False)] = 1
        for k in range(N):
            model.AddHint(x[k, a], int(hint[k]))

    if verbose:
        print(f"  Warm start hints set")
        total_vars = N * N + N * n_pair_constraints
        print(f"  Total model: {N*N} x-vars + {N * n_pair_constraints} d-vars = {total_vars} vars")
        print(f"  Constraints: {n_pair_constraints} pair constraints + {N-1} sum constraints + symmetry")
        sys.stdout.flush()

    # === SOLVE ===
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = verbose
    solver.parameters.cp_model_presolve = True
    # Try different search strategies
    # solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    callback = BreakthroughCallback(x, G)

    if verbose:
        print(f"\n  Starting solve ({time_limit}s limit, 8 workers)...")
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
        print(f"\n  Status: {status_name}")
        print(f"  Wall time: {solver.WallTime():.1f}s")
        if callback.solution is not None:
            print(f"  SOLUTION FOUND!")
        sys.stdout.flush()

    return callback.solution


def main():
    G = build_new_gram()
    print("=" * 70)
    print("FULL CP-SAT GRAM DECOMPOSITION")
    print("29x29 matrix R with R^T R = G, all entries +/-1")
    print("=" * 70)
    sys.stdout.flush()

    R = build_and_solve(G, time_limit=3600, verbose=True)

    if R is not None:
        print("\n*** BREAKTHROUGH! ***")
        det_val = abs(np.linalg.det(R.astype(float)))
        target_max = 1270698346568170340352
        print(f"|det(R)| = {det_val:.6e}")
        print(f"Score = {det_val / target_max:.6f}")
        return R
    else:
        print("\nNo solution found.")
        return None


if __name__ == "__main__":
    main()
