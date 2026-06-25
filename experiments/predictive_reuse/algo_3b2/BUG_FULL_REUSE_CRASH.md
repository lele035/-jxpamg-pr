# Bug: 3B-2 Full Reuse Path Crash

Date: 2026-06-20
Status: Unfixed

## Symptom

--predictive-theta 0.5 triggers JXPAMG_PR_AdaptiveReuseAlgorithm full-reuse path, crash on second solve:

  [predictive-reuse] full reuse across 6 levels
   Out of memory trying to allocate -41183520 bytes
  Segmentation fault

## Trigger

- Binary: solver_pr  
- Flags: -sid 12 --predictive-theta 0.5 -mat2
- All matrix pairs: probe returns sigma=0 on all 6 levels -> full reuse -> crash

## Isolation Tests

| Condition | Result |
|-----------|--------|
| No --predictive-theta | OK (reuse, no crash) |
| --predictive-theta 0.0 | OK (skips adaptive path) |
| --predictive-theta 0.5 | 100% crash |
| Added RecreateWorkVectors call | Did not fix (rolled back) |

## Root Cause

Debug print in jx_CAlloc shows first failure:
  count=18446744073706977646 elt=16 size=18446744073668368096

- count = size_t(-2573970) = 2^64 - 2573970
- 2573970 is EXACTLY the nnz of old A[2] (Velx level-3 coarse operator)
- elt=16 = sizeof(JX_Real)
- So code reads nnz from a destroyed/old matrix and passes it to CTAlloc(JX_Real, ...)

JXPAMG_PR_AdaptiveReuseAlgorithm full-reuse path:
1. Replaces A_array[0] = new_A_fine (new fine matrix)
2. Builds A_array[1..5] via BuildReuseCoarseOperator (new fine * old P/R -> new coarse)
3. Returns 0

But does NOT update:
- P_array / R_array (still from old Velx setup)
- F_array / U_array coarse-level vectors (old dimensions)
- Smoother (still configured for old matrices)
- CF_marker_array / dof_func_array etc (stale data)

Crash occurs AFTER full-reuse return, during JX_PCGSetup or JX_PCGSolve.
32-bit signed nnz value -2573970 from old A[2] leaks into a CTAlloc call.

## Suspicious: sigma=0 on all levels

Probe compares new A_H (built from Vely * old P/R) with saved old coarse ops.
For different physics fields (Velx vs Vely), diagonals should differ -> sigma should be non-zero.

Level 0 probe: compares A_array[0] (old) with saved_A[0] (old) -> must be 0 (trivial)
Level 1+: uses JXPAMG_PR_DiagRelativeDiff which may have double/long_double precision issue causing false zero

## Key Files

File | Line | Description
-----|------|------------
par_amg_setup.c | 2734-2793 | JXPAMG_PR_AdaptiveReuseAlgorithm (probe + full-reuse)
par_amg_setup.c | 461-517 | JXPAMG_PR_BuildReuseCoarseOperator (build coarse with old P/R)
par_amg_setup.c | 777-835 | JXPAMG_PR_FullReuseRAPUpdate (contrast: calls RecreateWorkVectors)
par_amg_setup.c | 527-577 | JXPAMG_PR_RecreateWorkVectors (should be called by full-reuse path)
par_amg_setup.c | 892-930 | JXPAMG_PR_DiagRelativeDiff (all zeros suspicious)
jxpamg_interface.c | 1262-1263 | Call site: JXPAMG_PR_AdaptiveReuseAlgorithm(...)
par_amg_setup_3b2.inc | 1-518 | 3B-2 algorithm (defined but NOT integrated into this code path)

## Possible Fix Directions (not implemented)

1. Remove spmt_rap_type guard in 3b2.inc (done, but 3B-2 not integrated, no effect)
2. Full-reuse path call RecreateWorkVectors (tried, did not fix, rolled back)
3. Full-reuse path call JX_PAMGSetup for complete rebuild (cost: loses reuse benefit)
4. Use JXPAMG_PR_FullReuseRAPUpdate instead of current full-reuse logic
5. Integrate JXPAMG_PR_AdaptiveReuse3B2_Residual into jx_PAMGSetup with predictive_mode branch
