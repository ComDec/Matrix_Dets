"""Final attempt: Full CP-SAT with best possible warm start."""
import numpy as np
import time
import sys
from fractions import Fraction
from math import lcm as math_lcm
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

def enum_pm1(cvecs, cvals, rng, max_sol=10000):
    k = len(cvecs); n = N
    if k == 0: return [rng.choice([-1, 1], size=n).astype(np.int64) for _ in range(100)]
    A = [[Fraction(int(cvecs[i][j])) for j in range(n)] for i in range(k)]
    b = [Fraction(int(cvals[i])) for i in range(k)]
    aug = [A[i] + [b[i]] for i in range(k)]
    pivots = []; ri = 0
    for c in range(n):
        if ri >= k: break
        p = next((r for r in range(ri, k) if aug[r][c] != 0), -1)
        if p == -1: continue
        if p != ri: aug[ri], aug[p] = aug[p], aug[ri]
        pv = aug[ri][c]
        for j in range(n + 1): aug[ri][j] /= pv
        for r in range(ri + 1, k):
            if aug[r][c] != 0:
                f = aug[r][c]
                for j in range(n + 1): aug[r][j] -= f * aug[ri][j]
        pivots.append(c); ri += 1
    for r in range(len(pivots), k):
        if aug[r][n] != 0: return []
    for i in range(len(pivots) - 1, -1, -1):
        pc = pivots[i]
        for r2 in range(i):
            if aug[r2][pc] != 0:
                f = aug[r2][pc]
                for j in range(n + 1): aug[r2][j] -= f * aug[i][j]
    free = [c for c in range(n) if c not in pivots]
    nf = len(free); np_ = len(pivots)
    ic = np.zeros((np_, nf), dtype=np.int64)
    icons = np.zeros(np_, dtype=np.int64); iden = np.zeros(np_, dtype=np.int64)
    for i in range(np_):
        ds = [aug[i][n].denominator] + [aug[i][fc].denominator for fc in free]
        lcd = 1
        for d in ds: lcd = math_lcm(lcd, d)
        iden[i] = lcd; icons[i] = int(aug[i][n] * lcd)
        for fi, fc in enumerate(free): ic[i, fi] = int(aug[i][fc] * lcd)
    sols = []
    if nf == 0:
        x = np.zeros(n, dtype=np.int64); ok = True
        for i in range(np_):
            if abs(icons[i]) != iden[i]: ok = False; break
            x[pivots[i]] = icons[i] // iden[i]
        if ok: sols.append(x)
    elif (1 << nf) <= 4194304:
        total = 1 << nf; bits = np.arange(total, dtype=np.int32)
        fm = np.empty((total, nf), dtype=np.int64)
        for fi in range(nf): fm[:, fi] = np.where((bits >> fi) & 1, 1, -1)
        pv = icons[np.newaxis, :] - fm @ ic.T
        vm = np.ones(total, dtype=bool)
        for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
        si = np.where(vm)[0]; rng.shuffle(si)
        for idx in si[:max_sol]:
            x = np.zeros(n, dtype=np.int64)
            for fi, fc in enumerate(free): x[fc] = fm[idx, fi]
            for i in range(np_): x[pivots[i]] = pv[idx, i] // iden[i]
            sols.append(x)
    else:
        batch = min(1 << 20, max_sol * 500)
        for _ in range(10):
            if len(sols) >= max_sol: break
            fm = rng.choice([-1, 1], size=(batch, nf)).astype(np.int64)
            pv = icons[np.newaxis, :] - fm @ ic.T
            vm = np.ones(batch, dtype=bool)
            for i in range(np_): vm &= (np.abs(pv[:, i]) == iden[i])
            for idx in np.where(vm)[0]:
                if len(sols) >= max_sol: break
                x = np.zeros(n, dtype=np.int64)
                for fi, fc in enumerate(free): x[fc] = fm[idx, fi]
                for i in range(np_): x[pivots[i]] = pv[idx, i] // iden[i]
                sols.append(x)
    return sols


def build_greedy_solution(G, seed=1):
    """Build best greedy solution (depth 15-17 typically)."""
    rng = np.random.RandomState(seed)
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))
    placed = {0: np.ones(N, dtype=np.int64)}
    order = [0]
    for step in range(1, N):
        ti = col_order[step]
        cvecs = [placed[order[i]] for i in range(len(order))]
        cvals = [int(G[order[i], ti]) for i in range(len(order))]
        sols = enum_pm1(cvecs, cvals, rng, max_sol=1000)
        if not sols: break
        placed[ti] = sols[rng.randint(len(sols))]
        order.append(ti)
    return placed, order


def main():
    G = build_new_gram()
    print("=" * 70)
    print("FINAL CP-SAT: Full solve with greedy warm start")
    print("=" * 70)
    sys.stdout.flush()

    # Build warm start
    print("Building greedy warm start...")
    best_placed = None
    best_depth = 0
    for seed in range(20):
        placed, order = build_greedy_solution(G, seed)
        if len(order) > best_depth:
            best_depth = len(order)
            best_placed = dict(placed)
            print(f"  Seed {seed}: depth {len(order)}")
    print(f"  Best greedy depth: {best_depth}/{N}")
    sys.stdout.flush()

    # Build R matrix for warm start (fill unplaced with random)
    R_hint = np.zeros((N, N), dtype=np.int64)
    rng = np.random.RandomState(42)
    for ci in range(N):
        if ci in best_placed:
            R_hint[:, ci] = best_placed[ci]
        else:
            # Random column with correct column sum (15 ones = (29+1)/2)
            col = np.ones(N, dtype=np.int64)
            col[rng.choice(N, 14, replace=False)] = -1
            R_hint[:, ci] = col

    # Build CP-SAT model
    print("\nBuilding CP-SAT model...")
    model = cp_model.CpModel()

    x = {}
    for k in range(N):
        for a in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # Fix col 0 = all +1
    for k in range(N):
        model.Add(x[k, 0] == 1)

    # Column sum constraints
    for a in range(1, N):
        target = (N + int(G[0, a])) // 2
        model.Add(sum(x[k, a] for k in range(N)) == target)

    # Pairwise XOR constraints
    n_pairs = 0
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
            n_pairs += 1

    # Light symmetry breaking
    generic = [2, 5] + list(range(11, 29))
    for i in range(len(generic) - 1):
        a, b = generic[i], generic[i+1]
        model.AddImplication(x[1, b], x[1, a])
    for i in range(7, 10):
        model.AddImplication(x[1, i+1], x[1, i])

    # Warm start from greedy solution
    for a in range(N):
        for k in range(N):
            val = 1 if R_hint[k, a] == 1 else 0
            model.AddHint(x[k, a], val)

    print(f"  {N*N} x-vars, {N*n_pairs} d-vars = {N*N + N*n_pairs} total")
    print(f"  {n_pairs} pair constraints")
    print(f"  Warm start from depth-{best_depth} greedy solution")
    sys.stdout.flush()

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3500
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_presolve = True

    class Callback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.solution = None
            self.count = 0
        def on_solution_callback(self):
            self.count += 1
            R = np.zeros((N, N), dtype=np.int64)
            for k in range(N):
                for a in range(N):
                    R[k, a] = 2 * self.Value(x[k, a]) - 1
            if verify_decomposition(R, G):
                self.solution = R
                print(f"\n*** VALID SOLUTION #{self.count}! ***")
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                det_val = abs(np.linalg.det(R.astype(float)))
                target_max = 1270698346568170340352
                print(f"|det(R)| = {det_val:.6e}")
                print(f"Score = {det_val / target_max:.6f}")
            sys.stdout.flush()
            self.StopSearch()

    cb = Callback()
    print(f"\nStarting solve (3500s, 8 workers)...")
    sys.stdout.flush()

    status = solver.Solve(model, cb)
    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, str(status))

    print(f"\nStatus: {status_name}")
    print(f"Wall time: {solver.WallTime():.1f}s")
    if cb.solution is not None:
        print("SOLUTION FOUND!")
    elif status_name == "INFEASIBLE":
        print("PROVED INFEASIBLE - this Gram matrix cannot be decomposed as R^T R with +/-1 entries!")
    else:
        print("No solution found within time limit.")

if __name__ == "__main__":
    main()
