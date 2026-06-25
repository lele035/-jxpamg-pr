# Phase 2 Stable Incremental Reuse Design

**Date:** 2026-06-21

**Status:** Approved design, pending implementation plan

**Project:** `/home/xiyudawang/jxpamg-0220-pr`

## 1. Goal

Phase 2 turns the Phase 1 full-rebuild baseline into a stable incremental-reuse implementation. It must support MPI `np=1/2/4`, read-only multilevel prediction, transactional tail rebuild, and full reuse without allowing a failed optimization to corrupt the active AMG hierarchy.

3B-2 integration is explicitly outside Phase 2.

## 2. Non-Negotiable Invariants

1. Prediction never changes the active hierarchy.
2. A rebuild is prepared outside the active hierarchy and committed only after complete validation.
3. A failed prediction or rebuild falls back to `JXPAMG_PR_FullSetupOnNewMatrix`.
4. No solve begins after a failed setup unless fallback setup succeeds.
5. Every rank makes the same reuse decision and uses the same `cut_k`.
6. Every matrix, prolongation, restriction, vector, smoother, and auxiliary allocation has one explicit owner.
7. Phase 1 full rebuild remains available through a runtime switch throughout Phase 2.

## 3. Runtime Decisions

The predictor returns one of three decisions:

| Decision | Meaning | Action |
|---|---|---|
| `FULL_REBUILD` | Level 0 exceeds the threshold or prediction fails | Run the Phase 1 full setup path |
| `TAIL_REBUILD(k)` | Levels below `k` remain valid and level `k` or deeper changed | Preserve prefix `[0, k)` and transactionally rebuild `[k, end)` |
| `FULL_REUSE` | Every probed level is within threshold | Commit the new operator chain and retain compatible transfer/smoother state |

`cut_k` uses one convention everywhere: it is the first level that must be rebuilt. `cut_k == 0` means full rebuild. `cut_k >= num_levels` means full reuse.

## 4. Architecture

### 4.1 Read-Only Predictor

`JXPAMG_PR_PredictCutReadOnly` starts with `new_A_fine`, builds each candidate coarse operator into temporary storage, compares it with the corresponding active `A_array[level]`, and proceeds only while `sigma <= theta`.

The predictor must not write `A_array`, `P_array`, `R_array`, smoother arrays, ownership flags, or level counts. Temporary coarse matrices are destroyed before return unless ownership is explicitly transferred to a rebuild transaction.

Local comparison values are reduced with MPI collectives. The final status and `cut_k` are checked with `MPI_Allreduce`; disagreement is treated as prediction failure and triggers full rebuild on every rank.

### 4.2 Rebuild Transaction

A transaction object records:

- the active hierarchy and original level count;
- the requested `cut_k`;
- temporary `A/P/R/F/U` arrays for the rebuilt tail;
- temporary smoothers and auxiliary data;
- ownership flags for every temporary object;
- preparation and validation status.

The transaction has three operations:

1. `Prepare`: build the complete candidate tail without destroying active objects.
2. `Validate`: check non-null arrays, dimensions, partitions, level count, ownership, and MPI-wide success.
3. `Commit`: replace the old tail in one controlled operation, then destroy superseded objects.

`Abort` destroys only temporary objects and leaves the active hierarchy unchanged. If prepare, validation, or commit preconditions fail, all ranks abort and execute full rebuild.

### 4.3 Full Reuse

Full reuse is not implemented as pointer substitution of `A_array[0]` alone. The new fine operator and every candidate coarse operator form a complete temporary operator chain. The chain is validated and committed together while retaining `P/R` and only those smoother objects whose setup contract remains valid for the new operators.

If a smoother cannot be safely refreshed in place, it is rebuilt. Unsupported smoother types force full rebuild until an explicit refresh implementation and test exist.

### 4.4 Fallback and Error Propagation

Each optimization returns a status; it never prints an error and continues. The caller uses this sequence:

1. Run predictor.
2. Attempt the selected optimized path.
3. On any nonzero status, discard temporary state and run full rebuild.
4. If full rebuild fails, return failure through `JXPAMG_Solver`, `JXPAMG_Solver_Interface`, and `solver.c`.
5. Run PCG setup and solve only after a successful hierarchy commit.

## 5. Delivery Gates

### Gate A: MPI Baseline

Run the unchanged Phase 1 full-rebuild path with `np=1/2/4`. Record convergence, iteration counts, residuals, exit codes, and hangs/timeouts. No incremental code is enabled until this baseline passes.

### Gate B: Multilevel Read-Only Probe

Repair the coarse-operator heap corruption and enable probing one level at a time. Compare predictor-created operators against operators produced by a full setup. Run under ASan or Valgrind before enabling tail rebuild.

### Gate C: Transactional Tail Rebuild

Implement prepare, validate, abort, and commit. Test forced failures at each stage. Enable only `TAIL_REBUILD(k)` initially; keep full reuse disabled.

### Gate D: Full Reuse

Enable full reuse only after tail rebuild passes all memory, MPI, and numerical tests. Validate the complete operator chain and smoother refresh behavior.

### Gate E: Integrated Acceptance

Run all three decisions under the same executable and verify that runtime counters prove each path was exercised.

## 6. Testing Strategy

### 6.1 Required Configurations

- MPI ranks: `1`, `2`, and `4`.
- Theta values: `0`, `1e-12`, `1e-7`, and `0.5`.
- Existing 14 available matrix pairs from the Phase 1 suite.
- Repeated Vely run: at least 10 rounds.
- Velz cases that previously exposed `k=1` failure.

### 6.2 Required Assertions

- No crash, hang, invalid read/write, double free, or definite/possible leak.
- Every rank reports the same decision and `cut_k`.
- Optimized solve convergence matches full rebuild.
- Solution and final residual agree with full rebuild within the existing solver tolerance.
- Forced prepare/validate failures preserve the old hierarchy and successfully fall back.
- Failure of both optimization and fallback propagates a nonzero process exit code.

### 6.3 Evidence

Every acceptance run stores the exact command, binary checksum, rank count, matrix names, theta, selected decision, `cut_k`, iteration count, residual, wall time, and exit code. Memory-tool logs are stored with their binary checksum.

## 7. Runtime Controls

Use independent runtime controls during development:

- full rebuild only;
- predictor diagnostics only;
- tail rebuild enabled;
- full reuse enabled.

The default changes only after its gate passes. A production run can always force the Phase 1 full-rebuild path.

## 8. Phase 2 Acceptance Criteria

Phase 2 is complete only when:

1. Phase 1 baseline passes for `np=1/2/4`.
2. The multilevel predictor completes without mutating the active hierarchy.
3. Tail rebuild and full reuse both use transactional commit and tested fallback.
4. Full rebuild, tail rebuild, and full reuse are each observed in test logs.
5. The complete test matrix converges without crashes or MPI disagreement.
6. ASan or Valgrind reports no AMG-origin invalid access or definite/possible leak.
7. Failure injection proves the old hierarchy remains valid before fallback.
8. All fatal failures propagate to a nonzero process exit code.

## 9. Out of Scope

- Integration of `par_amg_setup_3b2.inc`.
- New prediction metrics beyond the existing sigma/theta model.
- Performance tuning before correctness gates pass.
- Removing the Phase 1 full-rebuild fallback.
