"""
Hybrid approach: use RREF backtracker for first ~15 columns,
then CP-SAT for the remaining ~14 columns.

The key insight is that with 15 columns placed, each remaining column
has only about 29-15=14 free variables, and we need 14 more columns.
CP-SAT may handle this remaining problem much better than sequential
backtracking because it can propagate constraints across ALL remaining
columns simultaneously.
"""
import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm
from ortools.sat.python import cp_model

N = 29
THEORETICAL_MAX = 1270698346568170340352


def build_gram(nine_pairs):
    G = np.ones((N, N), dtype=np.int64)
    np.fill_diagonal(G, N)
    for (i, j) in nine_pairs:
        G[i, j] = 9; G[j, i] = 9
    for j in range(7, 11):
        G[6, j] = -3; G[j, 6] = -3
    return G


def solve_column_constraints(placed_cols, placed_indices, target_idx, G, rng,
                             max_sol=500, deadline=None):
    n = N; k = len(placed_cols)
    if k == 0:
        return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(min(max_sol, 100))]
    A_rows = [[Fraction(int(placed_cols[i][j])) for j in range(n)] for i in range(k)]
    b_vec = [Fraction(int(G[placed_indices[i], target_idx])) for i in range(k)]
    aug = [A_rows[i] + [b_vec[i]] for i in range(k)]
    pivot_cols = []; row_idx = 0
    for col in range(n):
        if row_idx >= k: break
        piv = -1
        for r in range(row_idx, k):
            if aug[r][col] != 0: piv = r; break
        if piv == -1: continue
        if piv != row_idx: aug[row_idx], aug[piv] = aug[piv], aug[row_idx]
        pv = aug[row_idx][col]
        for j in range(n + 1): aug[row_idx][j] /= pv
        for r in range(row_idx + 1, k):
            if aug[r][col] != 0:
                f = aug[r][col]
                for j in range(n + 1): aug[r][j] -= f * aug[row_idx][j]
        pivot_cols.append(col); row_idx += 1
    for r in range(len(pivot_cols), k):
        if aug[r][n] != 0: return []
    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                f = aug[r2][pc]
                for j in range(n + 1): aug[r2][j] -= f * aug[i][j]
    free_cols = [c for c in range(n) if c not in pivot_cols]
    nf = len(free_cols); np_ = len(pivot_cols)
    ic = np.zeros((np_, nf), dtype=np.int64)
    icons = np.zeros(np_, dtype=np.int64)
    iden = np.zeros(np_, dtype=np.int64)
    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free_cols]
        lcd = 1
        for d in ds: lcd = math_lcm(lcd, d)
        iden[i] = lcd; icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free_cols): ic[i, fi] = int(aug[i][fc] * lcd)
    solutions = []
    if nf == 0:
        x = np.zeros(n, dtype=np.int64); valid = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]: valid = False; break
            x[pivot_cols[i]] = icons[i] // iden[i]
        if valid: solutions.append(x)
    elif nf <= 20:
        total = 1 << nf; bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]; rng.shuffle(si)
        for idx in si[:max_sol]:
            if deadline and time.time() > deadline: break
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
            for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
            solutions.append(x)
    else:
        for _ in range(max(1, max_sol * 500 // (1 << 20))):
            if deadline and time.time() > deadline: break
            if len(solutions) >= max_sol: break
            bs = min(1 << 20, max_sol * 200)
            fm = rng.choice([-1, 1], size=(bs, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(bs, dtype=bool)
            for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(solutions) >= max_sol: break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free_cols): x[fc] = fm[idx, fi]
                for i in range(np_): x[pivot_cols[i]] = pv[idx, i] // iden[i]
                solutions.append(x)
    return solutions


def place_columns_greedy(G, col_order, rng, n_cols=15):
    """Place the first n_cols columns greedily."""
    cols = []
    indices = []
    for step, ci in enumerate(col_order[:n_cols]):
        ms = 50
        sols = solve_column_constraints(cols, indices, ci, G, rng, max_sol=ms,
                                        deadline=time.time() + 5)
        if not sols:
            return None, None, step
        cols.append(sols[rng.randint(len(sols))])
        indices.append(ci)
    return cols, indices, n_cols


class SolCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, x_vars, remaining_indices, placed_cols, placed_indices, n):
        super().__init__()
        self.x = x_vars
        self.remaining = remaining_indices
        self.placed_cols = placed_cols
        self.placed_indices = placed_indices
        self.n = n
        self.solution = None
        self.t0 = time.time()

    def on_solution_callback(self):
        R = np.zeros((self.n, self.n), dtype=np.int64)
        # Fill placed columns
        for col_vec, ci in zip(self.placed_cols, self.placed_indices):
            R[:, ci] = col_vec
        # Fill remaining columns from CP-SAT solution
        for j_idx, ci in enumerate(self.remaining):
            for i in range(self.n):
                R[i, ci] = 2 * self.Value(self.x[j_idx][i]) - 1
        self.solution = R
        elapsed = time.time() - self.t0
        print(f"  CP-SAT found solution! ({elapsed:.1f}s)", flush=True)
        self.StopSearch()


def cpsat_remaining(placed_cols, placed_indices, remaining_indices, G, time_limit=600):
    """Use CP-SAT to find the remaining columns."""
    n_remaining = len(remaining_indices)
    n_placed = len(placed_indices)

    print(f"  CP-SAT: {n_remaining} remaining columns, {n_placed} placed")

    model = cp_model.CpModel()

    # Variables for remaining columns: x[j][i] where j is col index in remaining, i is row
    x = [[model.NewBoolVar(f'x_{j}_{i}') for i in range(N)] for j in range(n_remaining)]

    # Constraints among remaining columns
    for j1 in range(n_remaining):
        for j2 in range(j1 + 1, n_remaining):
            ci1 = remaining_indices[j1]
            ci2 = remaining_indices[j2]
            target_ip = int(G[ci1, ci2])
            n_disagree = (N - target_ip) // 2

            # XOR constraint: sum(XOR(x[j1][i], x[j2][i])) = n_disagree
            p_vars = []
            for i in range(N):
                p = model.NewBoolVar(f'p_{j1}_{j2}_{i}')
                model.AddBoolAnd([x[j1][i], x[j2][i]]).OnlyEnforceIf(p)
                model.AddBoolOr([x[j1][i].Not(), x[j2][i].Not()]).OnlyEnforceIf(p.Not())
                p_vars.append(p)

            model.Add(
                sum(x[j1][i] for i in range(N)) +
                sum(x[j2][i] for i in range(N)) -
                2 * sum(p_vars) == n_disagree
            )

    # Constraints between placed and remaining columns
    for j in range(n_remaining):
        ci_new = remaining_indices[j]
        for p_idx, ci_old in enumerate(placed_indices):
            target_ip = int(G[ci_old, ci_new])
            n_disagree = (N - target_ip) // 2

            # The placed column is fixed, so we know which rows are +1 and which are -1
            placed_col = placed_cols[p_idx]

            # Count XOR with fixed column:
            # XOR(placed[i], x[j][i]) = 1 if they differ
            # If placed[i] = +1 (x=1): XOR = NOT x[j][i]
            # If placed[i] = -1 (x=0): XOR = x[j][i]
            # sum(XOR) = sum_{placed[i]=+1} (1 - x[j][i]) + sum_{placed[i]=-1} x[j][i]
            #          = n_plus - sum_{placed[i]=+1} x[j][i] + sum_{placed[i]=-1} x[j][i]

            plus_rows = [i for i in range(N) if placed_col[i] == 1]
            minus_rows = [i for i in range(N) if placed_col[i] == -1]
            n_plus = len(plus_rows)

            # n_plus - sum(x[j][plus]) + sum(x[j][minus]) = n_disagree
            model.Add(
                n_plus
                - sum(x[j][i] for i in plus_rows)
                + sum(x[j][i] for i in minus_rows)
                == n_disagree
            )

    # Symmetry breaking: fix row 0 of all remaining columns
    # Since row 0 can be freely chosen (or we can use the convention that
    # multiplying a row by -1 is a valid transformation)
    # Actually, row 0 is already fixed by the placed columns' values.
    # We can't do simple symmetry breaking here without care.

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    callback = SolCallback(x, remaining_indices, placed_cols, placed_indices, N)
    t0 = time.time()
    status = solver.Solve(model, callback)
    elapsed = time.time() - t0

    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    print(f"  CP-SAT status: {status_map.get(status, status)} ({elapsed:.1f}s)", flush=True)

    if callback.solution is not None:
        return callback.solution
    return None


def main():
    t_start = time.time()
    print("="*70)
    print("HYBRID BACKTRACK + CP-SAT SOLVER")
    print("Variant 0: G[0,3]=9, G[1,4]=9")
    print("="*70)
    sys.stdout.flush()

    G = build_gram([(0, 3), (1, 4)])

    # Column orderings to try
    orderings = [
        [6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5] + list(range(11, 29)),
        [6, 0, 3, 7, 1, 4, 8, 2, 5, 9, 10] + list(range(11, 29)),
        [6, 7, 8, 9, 10, 0, 3, 1, 4, 2, 5] + list(range(11, 29)),
    ]

    # Strategy: place 15-18 columns with backtracker, then CP-SAT for the rest
    for n_place in [18, 16, 14, 12]:
        print(f"\n--- Placing {n_place} columns, CP-SAT for {29-n_place} ---")
        sys.stdout.flush()

        for oi, base_order in enumerate(orderings):
            for seed in range(20):
                rng = np.random.RandomState(seed * 997 + oi * 31337)
                order = base_order.copy()
                # Shuffle the generic part
                generic = order[11:]
                rng.shuffle(generic)
                order = order[:11] + list(generic)

                cols, indices, depth = place_columns_greedy(G, order, rng, n_cols=n_place)
                if cols is None:
                    continue

                # Verify partial Gram
                C = np.column_stack(cols)
                pg = C.T @ C
                Gsub = G[np.ix_(indices, indices)]
                if not np.array_equal(pg, Gsub):
                    print(f"  Partial Gram mismatch!")
                    continue

                remaining = [c for c in range(N) if c not in indices]
                print(f"  Order {oi}, seed {seed}: placed {len(indices)} cols, "
                      f"remaining {len(remaining)}", flush=True)

                # Run CP-SAT on remaining columns
                cpsat_time = min(300, 7200 - (time.time() - t_start))
                if cpsat_time < 10:
                    print("  Out of time")
                    return

                R = cpsat_remaining(cols, indices, remaining, G, time_limit=cpsat_time)

                if R is not None:
                    if np.array_equal(R.T @ R, G) and np.all(np.abs(R) == 1):
                        print(f"\n*** SOLUTION FOUND AND VERIFIED! ***")
                        np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                        det_val = abs(np.linalg.det(R.astype(float)))
                        score = det_val / THEORETICAL_MAX
                        print(f"Score = {score:.6f}")
                        return
                    else:
                        print(f"  Verification FAILED")
                else:
                    print(f"  CP-SAT did not find solution", flush=True)

                if time.time() - t_start > 7200:
                    return

    elapsed = time.time() - t_start
    print(f"\nDone. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
