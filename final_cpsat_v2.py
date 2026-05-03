"""CP-SAT v2: Try different search strategies and encodings."""
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


def main():
    G = build_new_gram()
    print("=" * 70)
    print("CP-SAT v2: AND-product encoding (no XOR d-vars)")
    print("=" * 70)
    sys.stdout.flush()

    # Build greedy warm start
    print("Building warm start...")
    best_placed = None; best_depth = 0
    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))
    for seed in range(50):
        rng = np.random.RandomState(seed)
        placed = {0: np.ones(N, dtype=np.int64)}; order = [0]
        for step in range(1, N):
            ti = col_order[step]
            cvecs = [placed[order[i]] for i in range(len(order))]
            cvals = [int(G[order[i], ti]) for i in range(len(order))]
            sols = enum_pm1(cvecs, cvals, rng, max_sol=500)
            if not sols: break
            placed[ti] = sols[rng.randint(len(sols))]; order.append(ti)
        if len(order) > best_depth:
            best_depth = len(order); best_placed = dict(placed)
    print(f"  Best greedy: {best_depth}/{N}")

    # Build model with AND-product encoding (fewer vars than XOR)
    # For each pair (a,b): sum_k R[k,a]*R[k,b] = G[a,b]
    # R[k,a] = 2*x[k,a] - 1
    # (2xa-1)(2xb-1) = 4*xa*xb - 2*xa - 2*xb + 1
    # sum_k (4*z[k,a,b] - 2*x[k,a] - 2*x[k,b] + 1) = G[a,b]
    # 4*sum z - 2*sum xa - 2*sum xb + N = G[a,b]
    # where z[k,a,b] = x[k,a] AND x[k,b]

    print("\nBuilding model...")
    model = cp_model.CpModel()

    x = {}
    for k in range(N):
        for a in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # Fix col 0
    for k in range(N):
        model.Add(x[k, 0] == 1)

    # Column sums
    for a in range(1, N):
        target = (N + int(G[0, a])) // 2
        model.Add(sum(x[k, a] for k in range(N)) == target)

    # Pairwise with AND encoding
    n_pairs = 0
    for a in range(1, N):
        for b in range(a + 1, N):
            target_val = int(G[a, b]) - N  # 4*sum_z - 2*sum_xa - 2*sum_xb = G[a,b]-N
            z_vars = []
            for k in range(N):
                z = model.NewBoolVar(f'z_{k}_{a}_{b}')
                model.AddBoolAnd([x[k, a], x[k, b]]).OnlyEnforceIf(z)
                model.AddBoolOr([x[k, a].Not(), x[k, b].Not()]).OnlyEnforceIf(z.Not())
                z_vars.append(z)
            model.Add(
                4 * sum(z_vars)
                - 2 * sum(x[k, a] for k in range(N))
                - 2 * sum(x[k, b] for k in range(N))
                == target_val
            )
            n_pairs += 1

    # Symmetry breaking
    generic = [2, 5] + list(range(11, 29))
    for i in range(len(generic) - 1):
        model.AddImplication(x[1, generic[i+1]], x[1, generic[i]])
    for i in range(7, 10):
        model.AddImplication(x[1, i+1], x[1, i])

    # Warm start
    rng = np.random.RandomState(42)
    for a in range(N):
        if a in best_placed:
            col = best_placed[a]
        else:
            col = np.ones(N, dtype=np.int64)
            col[rng.choice(N, 14, replace=False)] = -1
        for k in range(N):
            model.AddHint(x[k, a], 1 if col[k] == 1 else 0)

    print(f"  {N*N} x + {N*n_pairs} z = {N*N + N*n_pairs} vars, {n_pairs} pair constraints")
    sys.stdout.flush()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3500
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True
    # Try automatic search strategy
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    class Cb(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.solution = None
        def on_solution_callback(self):
            R = np.zeros((N, N), dtype=np.int64)
            for k in range(N):
                for a in range(N):
                    R[k, a] = 2 * self.Value(x[k, a]) - 1
            if verify_decomposition(R, G):
                self.solution = R
                print(f"\n*** BREAKTHROUGH! ***")
                np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                det_val = abs(np.linalg.det(R.astype(float)))
                print(f"Score = {det_val / 1270698346568170340352:.6f}")
            sys.stdout.flush()
            self.StopSearch()

    cb = Cb()
    print(f"\nSolving (3500s, 8 workers, AND encoding)...")
    sys.stdout.flush()

    status = solver.Solve(model, cb)
    sn = {cp_model.OPTIMAL:"OPTIMAL", cp_model.FEASIBLE:"FEASIBLE",
          cp_model.INFEASIBLE:"INFEASIBLE", cp_model.UNKNOWN:"UNKNOWN"}.get(status, str(status))
    print(f"\nStatus: {sn}, Wall: {solver.WallTime():.1f}s")
    if sn == "INFEASIBLE":
        print("PROVED INFEASIBLE!")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
