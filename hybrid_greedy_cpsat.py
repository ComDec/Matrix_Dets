"""Hybrid: greedy for first 15 columns, CP-SAT for remaining 14."""
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


def cpsat_complete(fixed_cols_dict, remaining_indices, G, time_limit=300):
    """Use CP-SAT to find remaining columns given fixed ones."""
    n_rem = len(remaining_indices)
    if n_rem == 0: return {}

    model = cp_model.CpModel()
    x = {}
    for a in remaining_indices:
        for k in range(N):
            x[k, a] = model.NewBoolVar(f'x_{k}_{a}')

    # Constraints with fixed columns (linear)
    for a in remaining_indices:
        for b, col_b in fixed_cols_dict.items():
            target_ip = int(G[a, b])
            pos_k = [k for k in range(N) if col_b[k] == 1]
            neg_k = [k for k in range(N) if col_b[k] == -1]
            rhs = target_ip + int(np.sum(col_b))
            model.Add(
                2 * sum(x[k, a] for k in pos_k) - 2 * sum(x[k, a] for k in neg_k) == rhs
            )

    # Constraints between remaining columns (XOR)
    for idx_a, a in enumerate(remaining_indices):
        for b in remaining_indices:
            if a >= b: continue
            target_ip = int(G[a, b])
            td = (N - target_ip) // 2
            d_vars = []
            for k in range(N):
                d = model.NewBoolVar(f'd_{k}_{a}_{b}')
                model.AddBoolXOr([x[k, a], x[k, b], d.Not()])
                d_vars.append(d)
            model.Add(sum(d_vars) == td)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = {}
        for a in remaining_indices:
            col = np.zeros(N, dtype=np.int64)
            for k in range(N):
                col[k] = 2 * solver.Value(x[k, a]) - 1
            result[a] = col
        return result
    return None


def main():
    G = build_new_gram()
    print("=" * 70)
    print("HYBRID: Greedy + CP-SAT completion")
    print("=" * 70)
    sys.stdout.flush()

    deadline = time.time() + 3600
    start_time = time.time()

    col_order = [0, 3, 1, 4, 6, 7, 8, 9, 10, 2, 5] + list(range(11, 29))

    # Target: get 15+ columns greedily, then CP-SAT for the rest
    TARGET_GREEDY_DEPTH = 15
    n_attempts = 0

    while time.time() < deadline - 600:  # Leave 10 min for CP-SAT
        n_attempts += 1
        rng = np.random.RandomState(n_attempts)

        placed = {0: np.ones(N, dtype=np.int64)}
        placed_order = [0]

        for step in range(1, N):
            ti = col_order[step]
            cvecs = [placed[placed_order[i]] for i in range(len(placed_order))]
            cvals = [int(G[placed_order[i], ti]) for i in range(len(placed_order))]
            sols = enum_pm1(cvecs, cvals, rng, max_sol=1000)
            if not sols:
                break
            placed[ti] = sols[rng.randint(len(sols))]
            placed_order.append(ti)

        depth = len(placed_order)

        if depth >= TARGET_GREEDY_DEPTH:
            remaining = [col_order[s] for s in range(depth, N) if col_order[s] not in placed]
            # Actually remaining should be the unplaced columns
            remaining = [c for c in range(N) if c not in placed]

            elapsed = time.time() - start_time
            print(f"  Attempt {n_attempts}: depth {depth}/{N}, {len(remaining)} remaining, trying CP-SAT... ({elapsed:.0f}s)")
            sys.stdout.flush()

            cpsat_time = min(300, (deadline - time.time()) / 2)
            result = cpsat_complete(placed, remaining, G, time_limit=cpsat_time)

            if result is not None:
                # Build full R
                R = np.zeros((N, N), dtype=np.int64)
                for ci, col in placed.items():
                    R[:, ci] = col
                for ci, col in result.items():
                    R[:, ci] = col

                if verify_decomposition(R, G):
                    elapsed = time.time() - start_time
                    print(f"\n*** BREAKTHROUGH! Attempt {n_attempts}, {elapsed:.1f}s ***")
                    np.save("/home/xiwang/project/AutoMath/tasks/matrix_det/breakthrough_matrix.npy", R)
                    det_val = abs(np.linalg.det(R.astype(float)))
                    target_max = 1270698346568170340352
                    print(f"|det(R)| = {det_val:.6e}")
                    print(f"Score = {det_val / target_max:.6f}")
                    return R
                else:
                    err = int(np.sum((R.T @ R - G) ** 2))
                    print(f"    CP-SAT solution doesn't verify, err={err}")
            else:
                print(f"    CP-SAT: no solution for this partial")

        if n_attempts % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  {n_attempts} attempts, {elapsed:.0f}s")
            sys.stdout.flush()

    print(f"\nNo solution after {n_attempts} attempts")
    return None

if __name__ == "__main__":
    main()
