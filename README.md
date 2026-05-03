# Matrix_Dets

Hadamard Maximal Determinant Problem for n=29: Finding 29×29 {-1,+1} matrices with maximum |det|.

## Problem

Find a 29×29 matrix H with entries +1 or -1 that maximizes |det(H)|.
- Score = |det(H)| / THEORETICAL_MAX
- THEORETICAL_MAX = 2^29 × 3^2 × 7^12 × 19 = 1,270,698,346,568,170,340,352
- **Target: score > 0.94**

## Current Best: score = 0.9357 (matches 23-year world record)

| Version | Score | Method |
|---------|-------|--------|
| v0 | 0.143 | QR circulant + Bareiss hill climb |
| v4 | 0.860 | GF(27) → H28 → bordered 29×29 + optimization |
| v5 | 0.921 | + large perturbation restarts |
| **v6** | **0.9357** | **Algebraic Solomon Gram decomposition (2.1s)** |

## Key Innovation

Algorithmically discovers the world-record matrix (Orrick-Solomon 2003, c=320) without any hardcoded matrices:
1. Construct Solomon Gram matrix from algebraic description
2. Build first 11 columns via block structure (6×6 circulant + 4×4 Hadamard + separator)
3. Complete remaining 18 columns via corrected RREF-based DFS backtracker
4. Total time: ~2.1 seconds

## Breakthrough Candidate: score = 0.959

Discovered 18 modified Gram matrices with perfect-square det and score = 0.9587 > 0.94:
- Pattern: change 7 of 9 "five-entries" to 1, change 2 to 9
- |det| = 2^28 × 3^3 × 5 × 7^11 × 17
- **Decomposition status: OPEN** (CP-SAT solver inconclusive after 1hr)

## Files

- `init_program.py` — Main code (evaluator-compatible, score 0.9357)
- `evaluator.py` — Scoring evaluator (DO NOT MODIFY)
- `RESULTS.md` — Detailed results and analysis
- `CHANGELOG.md` — Version history and lessons learned
- `gram_decompose.py` — Gram-targeting + column backtracking
- `fast_decompose.py` — Corrected RREF-based column solver
- `decompose_new_gram.py` — Score-0.959 Gram decomposition attempts
- `smart_perturb.py` — Algebraic block construction + perturbation strategies
- `bordered_h28.py` — GF(27) Paley construction
- `alt_constructions.py` — Williamson, conference matrix, etc.
- `massive_search.py` — Long-running multi-strategy search
- `gram_search_results.txt` — 148M Gram modification results

## Computational Verification

- 3.5M+ matrix optimization trials (zero above c=320)
- 148M+ Gram 1-3 entry modifications (zero decomposable)
- 531 CP-SAT feasibility checks on score-0.959 Gram
- 1hr full CP-SAT solve (inconclusive)
- 89 higher-det Gram decomposition attempts (zero successful)

## Mathematical Context

- Best known for n=29: c=320 (Orrick-Solomon 2003)
- Proven upper bound: c < 329 (Brent et al. 2012)
- Score > 0.94 requires c ≥ 322 (new world record)
- Open problem for 23+ years
