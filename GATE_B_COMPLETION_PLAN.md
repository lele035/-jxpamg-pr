# Gate B Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and verify the MPI-safe, multilevel, read-only predictor before any Gate C tail-rebuild integration.

**Architecture:** Candidate coarse operators are built from isolated matrix copies. KT ownership transfer is respected by the standard ParCSR destructor. Every level synchronizes build status across ranks, prediction failure falls back to the Phase 1 full rebuild, and only full-setup failure terminates the solve.

**Tech Stack:** C, JXPAMG ParCSR/AMG APIs, Intel MPI, Valgrind, existing `solver_pr` integration driver.

---

## 1. Scope and Current Findings

Gate B is responsible only for prediction. It must not commit `A/P/R`, rebuild the AMG tail, or activate full reuse.

The earlier “KT has persistent process-wide side effects” conclusion is not supported. The concrete ownership defect was:

1. `jx_PAMGBuildCoarseOperatorKT` transfers `par_P/RT` `col_starts` ownership to the output RAP matrix.
2. The former custom deep-copy destructor ignored ownership flags and freed `col_starts` again.
3. Replacing it with `jx_ParCSRMatrixDestroy` fixes that ownership-contract violation.

The current implementation still has five blockers:

- copied matrices have no communication package for `np>1`;
- the KT return status is ignored;
- a failed coarse build can be reported as full reuse;
- `cut_k=0` can be overwritten by later probing;
- prediction failure currently aborts the solve instead of falling back to full setup.

## 2. Files

**Modify:**

- `csrc/amg/par_amg_setup.c`: deep copy, coarse-build status, predictor control flow, MPI consensus.
- `src/jxpamg_interface.c`: prediction-failure fallback behavior.
- `GATE_B_STATUS.md`: replace the obsolete KT-side-effect conclusion with verified results.

**Create during verification:**

- `experiments/predictive_reuse/algo_3b2/gate_b_np1.log`
- `experiments/predictive_reuse/algo_3b2/gate_b_np2.log`
- `experiments/predictive_reuse/algo_3b2/gate_b_np4.log`
- `experiments/predictive_reuse/algo_3b2/gate_b_valgrind.log`
- `experiments/predictive_reuse/algo_3b2/gate_b_results.tsv`

## 3. Required Semantics

`cut_k` has one meaning everywhere:

- `cut_k == 0`: level 0 changed; full rebuild is required.
- `0 < cut_k < num_levels`: this is the first level that would need rebuilding.
- `cut_k == num_levels`: every probed level passed; predicted full reuse.

Prediction status is separate from `cut_k`:

- status `0`: prediction completed and `cut_k` is valid.
- status nonzero: prediction is unavailable; caller must discard the prediction and run full setup.

Prediction failure is not solver failure. Solver failure occurs only when the fallback full setup, PCG setup, or PCG solve fails.

## 4. Task 1: Add MPI-Valid Deep Copies

**Files:**

- Modify: `csrc/amg/par_amg_setup.c:523-605`

- [ ] **Step 1: Preserve the current `np=1` reproduction command and output**

Run the existing binary once and save its output before editing:

```bash
cd /home/spring/ljc/jxpamg_702_sharesetup
md5sum solver_pr
mpirun -np 1 ./solver_pr \
  /home/spring/l_s/jxpamg_all/data_trans/A_Velx-0.csr \
  /home/spring/l_s/jxpamg_all/data_trans/b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 /home/spring/l_s/jxpamg_all/data_trans/A_Vely-0.csr \
  -rhs2 /home/spring/l_s/jxpamg_all/data_trans/b_Vely-0.csr \
  -ptheta 1e-7 \
  > experiments/predictive_reuse/algo_3b2/gate_b_before.log 2>&1
echo "EXIT=$?" >> experiments/predictive_reuse/algo_3b2/gate_b_before.log
```

Expected: the log preserves the pre-change behavior and exit status.

- [ ] **Step 2: Create a communication package for parallel copies**

After all ParCSR fields and partition arrays have been initialized, replace the unconditional `CommPkg = NULL` assignment with:

```c
{
   JX_Int np = 1;
   jx_MPI_Comm_size(jx_ParCSRMatrixComm(dst), &np);
   jx_ParCSRMatrixCommPkg(dst) = NULL;
   if (np > 1)
   {
      if (jx_MatvecCommPkgCreate(dst) != 0)
      {
         jx_ParCSRMatrixDestroy(dst);
         return NULL;
      }
   }
}
```

Do not copy the source `comm_pkg` pointer. It contains allocations owned by the source matrix and must not be shared with the copy.

- [ ] **Step 3: Keep standard ownership-aware destruction**

All deep copies must be destroyed with:

```c
jx_ParCSRMatrixDestroy(copy);
```

Do not restore `JXPAMG_PR_DestroyDeepCopy`. The standard destructor checks `OwnsRowStarts`, `OwnsColStarts`, `OwnsData`, and communication-package ownership.

- [ ] **Step 4: Build the library and driver**

```bash
cd /home/spring/ljc/jxpamg_702_sharesetup
source /opt/intel/oneapi/setvars.sh --force
make -C jxpamg-0220-pr clean
make -C jxpamg-0220-pr -j8
make -f Makefile_pr clean
make -f Makefile_pr
```

Expected: all commands exit `0`; no implicit-declaration or incompatible-pointer warnings related to the changed functions.

## 5. Task 2: Make Coarse Build Return an Explicit Status

**Files:**

- Modify: `csrc/amg/par_amg_setup.c:608-637`

- [ ] **Step 1: Change the helper contract**

Replace the pointer-only helper with a status-returning helper:

```c
static JX_Int
JXPAMG_PR_BuildReuseCoarseOperatorFrom(jx_ParAMGData *amg_data,
                                       JX_Int level,
                                       jx_ParCSRMatrix *A_level,
                                       jx_ParCSRMatrix **A_H_ptr)
{
   jx_ParCSRMatrix **P_array;
   jx_ParCSRMatrix *A_copy = NULL;
   jx_ParCSRMatrix *P_copy = NULL;
   jx_ParCSRMatrix *A_H = NULL;
   JX_Int status = 1;

   if (A_H_ptr == NULL) return 1;
   *A_H_ptr = NULL;
   if (amg_data == NULL || A_level == NULL) return 1;

   P_array = jx_ParAMGDataPArray(amg_data);
   if (P_array == NULL || P_array[level] == NULL) return 1;

   A_copy = JXPAMG_PR_DeepCopyParCSR(A_level);
   P_copy = JXPAMG_PR_DeepCopyParCSR(P_array[level]);
   if (A_copy == NULL || P_copy == NULL) goto cleanup;

   status = jx_PAMGBuildCoarseOperatorKT(P_copy, A_copy, P_copy, 0, &A_H);
   if (status != 0 || A_H == NULL)
   {
      status = status != 0 ? status : 1;
      goto cleanup;
   }

   jx_ParCSRMatrixSetNumNonzeros(A_H);
   jx_ParCSRMatrixSetDNumNonzeros(A_H);
   *A_H_ptr = A_H;
   A_H = NULL;
   status = 0;

cleanup:
   if (A_H != NULL) jx_ParCSRMatrixDestroy(A_H);
   if (P_copy != NULL) jx_ParCSRMatrixDestroy(P_copy);
   if (A_copy != NULL) jx_ParCSRMatrixDestroy(A_copy);
   return status;
}
```

This cleanup order is intentional. KT may transfer `P_copy` partition ownership to `A_H`; the standard destructor honors the updated flags.

- [ ] **Step 2: Remove unused `opt_icor` plumbing**

KT does not use `opt_icor`. Remove it from this helper and remove the predictor allocation/free if no other predictor code uses it.

- [ ] **Step 3: Rebuild**

Run the Task 1 build commands.

Expected: build exits `0` and no stale call uses the old helper signature.

## 6. Task 3: Correct Predictor Control Flow

**Files:**

- Modify: `csrc/amg/par_amg_setup.c:643-703`

- [ ] **Step 1: Validate before dereferencing**

Move all `amg_data` field access below argument validation:

```c
if (amg_data == NULL || new_A_fine == NULL || cut_k == NULL) return 1;

A_array = jx_ParAMGDataAArray(amg_data);
P_array = jx_ParAMGDataPArray(amg_data);
num_levels = jx_ParAMGDataNumLevels(amg_data);
if (A_array == NULL || A_array[0] == NULL || P_array == NULL) return 1;
```

- [ ] **Step 2: End prediction immediately on `cut_k=0`**

After the level-0 sigma comparison:

```c
if (sigma > theta_pred)
{
   k = 0;
   goto prediction_consensus;
}
```

Do not build coarse operators after level 0 has already required a full rebuild.

- [ ] **Step 3: Synchronize build status at every level**

Use one collective after each local build attempt so every rank follows the same loop:

```c
JX_Int local_status;
JX_Int global_status;
jx_ParCSRMatrix *A_H = NULL;

local_status = JXPAMG_PR_BuildReuseCoarseOperatorFrom(
   amg_data, level, cur_A, &A_H);
jx_MPI_Allreduce(&local_status, &global_status, 1,
                 JX_MPI_INT, MPI_MAX, comm);

if (global_status != 0)
{
   if (A_H != NULL) jx_ParCSRMatrixDestroy(A_H);
   status = global_status;
   goto prediction_cleanup;
}
```

Every rank must execute the collective once per entered level. No rank may `break` before the status collective.

- [ ] **Step 4: Treat a missing coarse matrix as failure**

`A_H == NULL` must produce nonzero prediction status. It must never fall through with `k == num_levels`, because that would falsely report full reuse.

- [ ] **Step 5: Verify final `cut_k` consensus**

At `prediction_consensus`, calculate minimum and maximum:

```c
JX_Int k_min;
JX_Int k_max;
jx_MPI_Allreduce(&k, &k_min, 1, JX_MPI_INT, MPI_MIN, comm);
jx_MPI_Allreduce(&k, &k_max, 1, JX_MPI_INT, MPI_MAX, comm);
if (k_min != k_max)
{
   status = 1;
   goto prediction_cleanup;
}
k = k_min;
```

- [ ] **Step 6: Use a single cleanup exit**

The predictor must destroy `cur_A` exactly once when it is temporary, assign `*cut_k` only on success, and return nonzero on any failed allocation, KT call, compatibility check, or MPI disagreement.

Required structure:

```c
prediction_cleanup:
   if (cur_A != NULL && cur_A != new_A_fine)
      jx_ParCSRMatrixDestroy(cur_A);
   if (status == 0)
      *cut_k = k;
   return status;
```

## 7. Task 4: Make Prediction Failure Fall Back

**Files:**

- Modify: `src/jxpamg_interface.c:1270-1303`

- [ ] **Step 1: Preserve full setup after predictor failure**

Replace the current fatal branch at lines 1285-1290. The control flow must be:

```c
status = JXPAMG_PR_PredictCutReadOnly(
   (jx_ParAMGData *)amg_solver, par_matrix,
   predictive_theta, predictive_lreuse,
   (JX_Real)1e-6, &cut_k);

if (myid == 0)
{
   if (status == 0)
      jx_printf("[predictive-reuse] probe ok, predicted cut_k=%d\n", cut_k);
   else
      jx_printf("[predictive-reuse] probe failed (status=%d), fallback to full setup\n",
                status);
}

JX_PCGSetPrecond(solver,
   (JX_PtrToSolverFcn)JX_PAMGPrecond,
   (JX_PtrToSolverFcn)JX_PAMGSetup,
   amg_solver);

status = JXPAMG_PR_FullSetupOnNewMatrix(
   (jx_ParAMGData *)amg_solver, par_matrix);
if (status != 0)
{
   JX_ParCSRPCGDestroy(solver);
   solve_status = 1;
   solver = NULL;
   break;
}
```

Prediction status must not be reused as the final setup status without first calling full setup.

- [ ] **Step 2: Verify existing fatal propagation remains intact**

Confirm that full-setup, PCG-setup, and PCG-solve failures still propagate through:

```text
JXPAMG_Solver -> JXPAMG_Solver_Interface -> solver.c ret -> process exit code
```

## 8. Task 5: Gate B Functional and MPI Verification

**Files:**

- Create: `experiments/predictive_reuse/algo_3b2/gate_b_results.tsv`

- [ ] **Step 1: Record the tested binary**

```bash
cd /home/spring/ljc/jxpamg_702_sharesetup
md5sum solver_pr | tee experiments/predictive_reuse/algo_3b2/gate_b_binary.md5
```

- [ ] **Step 2: Run the same case at `np=1/2/4`**

For each `N` in `1 2 4`, run:

```bash
timeout 180 mpirun -np N ./solver_pr \
  /home/spring/l_s/jxpamg_all/data_trans/A_Velx-0.csr \
  /home/spring/l_s/jxpamg_all/data_trans/b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 /home/spring/l_s/jxpamg_all/data_trans/A_Vely-0.csr \
  -rhs2 /home/spring/l_s/jxpamg_all/data_trans/b_Vely-0.csr \
  -ptheta 1e-7 \
  > experiments/predictive_reuse/algo_3b2/gate_b_npN.log 2>&1
echo "EXIT=$?" >> experiments/predictive_reuse/algo_3b2/gate_b_npN.log
```

Replace `N` literally with `1`, `2`, and `4` in the command and filename.

Expected for each run:

- `EXIT=0`;
- no timeout, BAD TERMINATION, or MPI abort;
- probe output reaches every expected level or reports a synchronized prediction failure;
- fallback full setup and PCG solve complete;
- all ranks agree on `cut_k`.

- [ ] **Step 3: Exercise all prediction outcomes**

Run theta values `1e-12`, `1e-7`, and `0.5` for representative matrix pairs. The collected logs must include:

- at least one `cut_k=0`;
- at least one `0 < cut_k < num_levels`;
- at least one `cut_k=num_levels`.

If the existing data cannot naturally produce one category, add a diagnostic-only test mode that forces the comparison result, but do not use forced decisions in acceptance performance measurements.

- [ ] **Step 4: Run all available Phase 1 matrix pairs**

Use the same 14 available combinations and acceptance tolerance documented in `PHASE1_FINAL.md`. Record rank count, theta, `cut_k`, iterations, residual, and exit code in TSV columns:

```text
binary_md5	np	matrix1	matrix2	theta	probe_status	cut_k	iterations	residual	exit
```

- [ ] **Step 5: Run the 10-round stability case**

Run the Vely case ten times with `np=1`, then ten times with `np=2`. All runs must select the same decision for identical inputs and finish without a crash or timeout.

## 9. Task 6: Memory Verification

**Files:**

- Create: `experiments/predictive_reuse/algo_3b2/gate_b_valgrind.log`

- [ ] **Step 1: Run Valgrind on the multilevel path**

```bash
cd /home/spring/ljc/jxpamg_702_sharesetup
{
  echo "=== Binary ==="
  md5sum solver_pr
  echo "=== Valgrind ==="
  valgrind --leak-check=full --show-leak-kinds=all \
    --track-origins=yes --error-exitcode=99 \
    mpirun -np 1 ./solver_pr \
      /home/spring/l_s/jxpamg_all/data_trans/A_Velx-0.csr \
      /home/spring/l_s/jxpamg_all/data_trans/b_Velx-0.csr 1e-6 \
      -sid 12 -maxiter 200 \
      -mat2 /home/spring/l_s/jxpamg_all/data_trans/A_Vely-0.csr \
      -rhs2 /home/spring/l_s/jxpamg_all/data_trans/b_Vely-0.csr \
      -ptheta 1e-7
  echo "EXIT=$?"
} > experiments/predictive_reuse/algo_3b2/gate_b_valgrind.log 2>&1
```

Expected:

```text
definitely lost: 0 bytes
indirectly lost: 0 bytes
possibly lost: 0 bytes
ERROR SUMMARY: 0 errors
EXIT=0
```

MPI/MKL `still reachable` memory may be documented but must not be described as a JXPAMG leak without an allocation stack in JXPAMG.

- [ ] **Step 2: Inspect specifically for ownership failures**

```bash
rg -n "Invalid free|Invalid read|Invalid write|double free|ERROR SUMMARY|LEAK SUMMARY|EXIT=" \
  experiments/predictive_reuse/algo_3b2/gate_b_valgrind.log
```

Expected: no invalid access/free lines; zero-error summary; `EXIT=0`.

## 10. Task 7: Failure-Path Verification

**Files:**

- Modify temporarily during test: `csrc/amg/par_amg_setup.c`
- Update: `GATE_B_STATUS.md`

- [ ] **Step 1: Force predictor failure after level 0**

Temporarily return nonzero from `JXPAMG_PR_BuildReuseCoarseOperatorFrom` at level 0. Rebuild and run `np=1/2/4`.

Expected:

- every rank reports predictor failure;
- every rank enters full setup;
- solve succeeds and process exit is `0`;
- no rank continues probing alone.

- [ ] **Step 2: Restore the production implementation**

Remove the injected failure, rebuild, and rerun the normal `np=1` case. Compare the binary checksum with the checksum recorded for final acceptance; only the restored binary may be used in final logs.

- [ ] **Step 3: Confirm fatal full-setup failure propagation separately**

Use the existing failure-injection mechanism or a temporary single return in `JXPAMG_PR_FullSetupOnNewMatrix`.

Expected:

- PCG setup and solve are not called;
- solver cleanup executes;
- process exit is nonzero.

Restore production code and rebuild immediately after this test.

## 11. Task 8: Replace `GATE_B_STATUS.md`

**Files:**

- Modify: `GATE_B_STATUS.md`

- [ ] **Step 1: Record the verified root cause**

State that the demonstrated defect was an ownership-contract violation in the custom deep-copy destructor. Do not retain the unproven claim that KT modifies global or static process state.

- [ ] **Step 2: Record auditable evidence**

Include:

- final binary checksum;
- exact build command;
- `np=1/2/4` exit results;
- theta and `cut_k` coverage;
- Valgrind summary and log path;
- failure-injection fallback result;
- remaining limitations.

- [ ] **Step 3: Set status from evidence**

Use `Status: COMPLETE` only if every criterion in Section 12 passes. Otherwise state the exact blocker and keep Gate C disabled.

## 12. Gate B Release Checklist

Gate C may begin only when every item is checked:

- [ ] Deep copies own independent CSR data, partitions, maps, and MPI communication packages.
- [ ] KT return status and `A_H` are both validated.
- [ ] Partial allocation and KT failure paths free every temporary object exactly once.
- [ ] Level-0 mismatch returns `cut_k=0` without coarse probing.
- [ ] Coarse-build failure cannot be reported as full reuse.
- [ ] Every entered level performs MPI-wide status synchronization.
- [ ] Final `cut_k` minimum and maximum agree across ranks.
- [ ] Predictor failure falls back to full setup on all ranks.
- [ ] Full-setup failure still propagates as nonzero process exit.
- [ ] `np=1/2/4` functional runs exit `0` without timeout or MPI abort.
- [ ] `cut_k=0`, middle cut, and full-reuse prediction are represented in logs.
- [ ] All 14 available matrix combinations converge within the Phase 1 tolerance.
- [ ] Vely repeated runs are stable.
- [ ] Valgrind reports zero errors and no definite/indirect/possible leak.
- [ ] `GATE_B_STATUS.md` matches the final source and evidence.

## 13. Stop Conditions

Stop and do not enter Gate C if any of these occurs:

- rank disagreement or a collective hang;
- prediction failure aborts instead of falling back;
- any invalid free/read/write involving copied matrices or RAP;
- `A_H == NULL` produces status `0`;
- test logs do not identify the exact binary;
- only `np=1` passes;
- only one prediction outcome has been exercised.

Gate C must consume a proven `cut_k`; it must not compensate for an unreliable Gate B predictor.
