"""
Structured construction: build R from scratch using algebraic structure.

The Gram G = R^T R has structure:
- G = J + 28I + C where C is sparse (only 6 nonzero off-diagonal entries)
- C[0,3] = C[3,0] = 8
- C[1,4] = C[4,1] = 8
- C[6,7..10] = C[7..10,6] = -4

Key idea: R can be thought of as 29 rows of length 29, each +/-1.
The ALL-ONES column (col 0 = all +1 after row normalization) gives
the column sums. Most columns have sum 1, col 3 has sum 9.

Strategy: Build R row by row, using the Gram structure to guide construction.
Each row r is a +/-1 vector of length 29. The contribution of row r to the Gram is:
G += r * r^T (outer product)

After adding all 29 rows, we need G = target.

Alternative approach: Use the matrix factorization R = Q * L^T where L is the
Cholesky factor of G and Q is an orthogonal matrix. We need R to have all +/-1
entries. This is a rounding problem.

Actually, the most promising approach for this specific structure:
Use CP-SAT but with a MUCH smaller formulation.

Key observation: With column 0 = all +1, each row r_k = (1, r_k[1], ..., r_k[28]).
The Gram entry G[a,b] = sum_{k=0}^{28} r_k[a] * r_k[b].

For a=0: G[0,b] = sum_k r_k[0] * r_k[b] = sum_k r_k[b] (since r_k[0]=1).
So sum_k r_k[b] = G[0,b] = 1 for most b, 9 for b=3.

For a,b >= 1: G[a,b] = sum_k r_k[a] * r_k[b].
Let y_k be the 28-vector (r_k[1], ..., r_k[28]).
Then for a,b >= 1: G[a,b] = sum_k y_k[a-1] * y_k[b-1].

So the 28x28 submatrix G[1:, 1:] = Y^T Y where Y is 29 x 28 with +/-1 entries
and Y_k = y_k = (r_k[1], ..., r_k[28]).

The 28x28 submatrix is:
- Diagonal: 29
- (0,2) = (2,0) = 9 [was (1,4) in original, shifted by -1]...
  Wait, let me be more careful. In the original Gram, after fixing col 0:
  G[1,4] = 9, so in the shifted indices (a-1 for a>=1), this is G_sub[0,3] = 9.
  G[0,3] = 9 in original doesn't involve the shifted submatrix directly.
  Actually, G[a,b] for a,b >= 1 gives us G_sub where G_sub[a-1,b-1] = G[a,b].

So G_sub (28x28):
- Diagonal: 29
- G_sub[0,2] = G[1,3] = 1 (NOT 9)
- G_sub[2,3] = G[3,4] = 1
Wait, let me just extract it.
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
    RR = R.astype(np.int64)
    if not np.all(np.abs(RR) == 1):
        return False
    return np.array_equal(RR.T @ RR, G)


def approach_row_cpsat(G, time_limit=3600, verbose=True):
    """
    Formulate as: find 29 rows, each a +/-1 vector of length 29,
    with col 0 = all +1 (so each row starts with +1).

    Variables: y[k, a] in {0,1} for k=0..28, a=0..27
    (representing the 28 non-fixed entries of each row)
    R[k, 0] = 1 (fixed), R[k, a+1] = 2*y[k,a] - 1

    Constraints:
    For each pair of column indices (a,b) with 1 <= a < b <= 28:
      G[a,b] = sum_k R[k,a]*R[k,b] = sum_k (2*y[k,a-1]-1)(2*y[k,b-1]-1)
    And for a >= 1:
      G[0,a] = sum_k R[k,a] = sum_k (2*y[k,a-1]-1) (already handled by col sums)

    Actually this is the same formulation as before, just a different perspective.
    Let me try a different encoding that might work better.

    NEW IDEA: Row-based encoding.
    Each row is a +/-1 vector. There are 2^28 possible rows (first entry fixed to +1).
    We need to choose 29 rows (with repetition) such that their Gram gives G.

    But 2^28 is too many row types.

    BETTER IDEA: Instead of variables for individual entries, use variables
    for the row type. Group rows by their "signature" - which columns they
    agree/disagree on. But this is combinatorial.

    ALTERNATIVE: Use a different Gram decomposition strategy.

    The Gram G has a specific spectral structure:
    - 22 eigenvalues = 28 (multiplicity 22, from the J + 28I base)
    - 2 eigenvalues = 20 (from the -3 block)
    - 2 eigenvalues = 36 (from the 9 block)
    - Other eigenvalues from the specific structure

    Let me just run the full CP-SAT but with a row-based formulation
    that groups rows efficiently.
    """
    if verbose:
        print("Row-based CP-SAT formulation")
        sys.stdout.flush()

    # Actually, let me try something different: an integer programming approach.
    # Variables: count[t] = number of rows of type t.
    # But 2^28 types is too many.

    # Let me instead try: build R column by column using CP-SAT,
    # but with INCREMENTAL SOLVING and solution-guided search.

    # Start from an empty model and add columns one at a time.
    # After placing each column, verify compatibility and add constraints.

    model = cp_model.CpModel()

    # Variables for all columns (except col 0 which is all +1)
    x = {}
    for k in range(N):
        for a in range(1, N):  # columns 1..28
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # Column sum constraints (from IP with col 0 = all +1)
    for a in range(1, N):
        target_ones = (N + int(G[0, a])) // 2
        model.Add(sum(x[k, a] for k in range(N)) == target_ones)

    # Pairwise constraints for columns 1..28
    # Use the XOR encoding
    for a in range(1, N):
        for b in range(a + 1, N):
            target_ip = int(G[a, b])
            td = (N - target_ip) // 2

            d_vars = []
            for k in range(N):
                d = model.NewBoolVar(f'd_{k}_{a}_{b}')
                model.AddBoolXOr([x[k, a], x[k, b], d.Not()])
                d_vars.append(d)
            model.Add(sum(d_vars) == td)

    # Symmetry breaking
    # Generic columns: {2, 5, 11..28} but shifted to {1, 4, 10..27} in 0-indexed col space
    # Wait, column indices in our x are 1..28, corresponding to original cols 1..28.
    # Generic columns in original: {2, 5, 11, 12, ..., 28}
    generic = [2, 5] + list(range(11, 29))
    for i in range(len(generic) - 1):
        a = generic[i]
        b = generic[i + 1]
        # x[1, a] >= x[1, b]
        model.AddImplication(x[1, b], x[1, a])

    # Cols 7,8,9,10 interchangeable
    for i in range(7, 10):
        model.AddImplication(x[1, i + 1], x[1, i])

    # Warm start from partial 11-column solution
    partial = np.load("/home/xiwang/project/AutoMath/tasks/matrix_det/partial_11_cols_new.npy").astype(np.int64)
    col0_partial = partial[:, 0]
    negate_mask = (col0_partial == -1)

    for col_idx in range(1, 11):
        col = partial[:, col_idx].copy()
        col[negate_mask] *= -1
        for k in range(N):
            model.AddHint(x[k, col_idx], 1 if col[k] == 1 else 0)

    # Random hints for cols 11..28
    rng = np.random.RandomState(42)
    for a in range(11, N):
        target_ones = (N + 1) // 2  # = 15
        hint = np.zeros(N, dtype=int)
        hint[rng.choice(N, target_ones, replace=False)] = 1
        for k in range(N):
            model.AddHint(x[k, a], int(hint[k]))

    if verbose:
        n_pairs = (N - 1) * (N - 2) // 2
        print(f"  {(N-1)*N} x-vars, {N * n_pairs} d-vars")
        print(f"  {n_pairs} pair constraints + {N-1} sum constraints")
        sys.stdout.flush()

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = verbose

    class Callback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.solution = None
            self.count = 0

        def on_solution_callback(self):
            self.count += 1
            R = np.ones((N, N), dtype=np.int64)  # col 0 = all +1
            for k in range(N):
                for a in range(1, N):
                    R[k, a] = 2 * self.Value(x[k, a]) - 1
            if verify_decomposition(R, G):
                self.solution = R
                print(f"  *** VALID SOLUTION #{self.count} ***")
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
            else:
                err = int(np.sum((R.T @ R - G) ** 2))
                print(f"  Solution #{self.count}: error = {err}")
            sys.stdout.flush()
            self.StopSearch()

    callback = Callback()
    if verbose:
        print(f"\n  Starting solve ({time_limit}s, 8 workers)...")
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
        sys.stdout.flush()

    return callback.solution


def main():
    G = build_new_gram()
    print("=" * 70)
    print("STRUCTURED CONSTRUCTION via CP-SAT")
    print("=" * 70)
    sys.stdout.flush()

    R = approach_row_cpsat(G, time_limit=3600, verbose=True)

    if R is not None:
        print("\n*** BREAKTHROUGH! ***")
        det_val = abs(np.linalg.det(R.astype(float)))
        target_max = 1270698346568170340352
        print(f"|det(R)| = {det_val:.6e}")
        print(f"Score = {det_val / target_max:.6f}")
    else:
        print("\nNo solution found.")


if __name__ == "__main__":
    main()
