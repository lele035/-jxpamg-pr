# Gate C Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current np=1 experimental in-place tail rebuild into a correctness-audited Gate C path that commits the correct `A[cut_k]` and does not call itself transactional until rollback semantics are real.

**Architecture:** Keep Gate C limited to `np=1` and `0 < cut_k < num_levels`. First fix the semantic gap by passing the candidate coarse operator generated during prediction into tail rebuild, then add validation/fallback around the existing in-place rebuild. A later task can upgrade from in-place fallback to true transaction-by-staging; do not mix that with MPI or full reuse.

**Tech Stack:** C, MPI, JXPAMG AMG hierarchy (`jx_ParAMGData`, `jx_ParCSRMatrix`), Valgrind, existing `solver_pr` case 12 predictive reuse path.

---

## 0. Current State

The current document says Gate C is complete as "transactional tail rebuild". That wording is too strong.

Evidence that does hold:

- `gate_c_Velz_1e-7.log`: `cut_k=1`, entered `JXPAMG_PR_SetupFromLevelExperimental`, solved in 5 iterations, Valgrind `ERROR SUMMARY: 0`, `EXIT=0`.
- `gate_c_Vely_1e-7.log`: `cut_k=0`, full setup path, Valgrind `ERROR SUMMARY: 0`, `EXIT=0`.
- `gate_c_Vely_0.5.log`: `cut_k=6`, full setup path, Valgrind `ERROR SUMMARY: 0`, `EXIT=0`.

The gap:

- `JXPAMG_PR_SetupFromLevelExperimental` first calls `JXPAMG_PR_FreeTailForSetupFromLevel(amg_data, k)`, then calls `JXPAMG_PR_RebuildTailFromLevelDraft(amg_data, k)`.
- That means old tail state is destroyed before rebuild succeeds. This is not strict transactional commit.
- For `cut_k=1`, `JXPAMG_PR_FreeTailForSetupFromLevel` frees `P[1..]`, `A[2..]`, `F/U[1..]`, smoother/finalization state, but keeps old `A[1]`.
- The rebuild wrapper starts from `A_array[k]`, so the current `cut_k=1` path may rebuild from old `A[1]`, not from the candidate `A[1]` induced by the new fine matrix.

Correct next status wording:

```text
Gate C: np=1 experimental in-place tail rebuild path runs under Valgrind, but strict transactional commit and correct A[cut_k] replacement remain to be implemented.
```

---

## 1. Files To Touch

### `csrc/amg/par_amg_setup.c`

Responsibilities:

- Predict cut level without modifying live hierarchy.
- Optionally build and return the candidate coarse operator at the cut boundary.
- Rebuild tail from a supplied `A_cut` instead of silently reusing old `A_array[cut_k]`.
- Validate hierarchy after rebuild.

### `include/jx_pamg.h`

Responsibilities:

- Declare any new Gate C helper used by `src/jxpamg_interface.c`.

### `src/jxpamg_interface.c`

Responsibilities:

- Route `cut_k` cases.
- Call the corrected Gate C helper only for `np=1` and `0 < cut_k < num_levels`.
- Fall back to `FullSetupOnNewMatrix` on prediction or tail rebuild failure.

### `OVERALL_STATUS.md`

Responsibilities:

- Downgrade current Gate C wording until the corrected implementation passes.
- Record exact log evidence and checksum meaning.

---

## 2. Target Control Flow

Gate C should use this flow:

```text
PredictCutReadOnlyWithCandidate(new_A):
  if probe fails:
      FullSetupOnNewMatrix(new_A)
  else if cut_k == 0:
      FullSetupOnNewMatrix(new_A)
  else if cut_k == num_levels:
      FullSetupOnNewMatrix(new_A)   # Gate D will optimize full reuse
  else if np != 1:
      FullSetupOnNewMatrix(new_A)   # MPI Gate C not released
  else:
      TailSetupFromCandidate(cut_k, candidate_A_cut)
      if tail setup fails:
          FullSetupOnNewMatrix(new_A)
```

Important rule:

```text
For cut_k > 0, tail rebuild must start from candidate_A_cut, not old A_array[cut_k].
```

---

## 3. Task 1: Fix Documentation Status Before More Code

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/OVERALL_STATUS.md`

- [ ] **Step 1: Change Gate C status line**

Replace:

```text
Gate C | ✅ np=1 | 事务性 tail rebuild (3 种 cut_k 全覆盖, Valgrind 0 errors)
```

with:

```text
Gate C | ⚠ np=1 experimental | 原地 tail rebuild 路径跑通；严格事务提交和 A[cut_k] 正确替换待完成
```

- [ ] **Step 2: Change Gate C description**

Replace wording that says Gate C has completed transactional tail rebuild with:

```text
Gate C 当前实现为原地 tail rebuild：先释放旧 tail，再从 A_array[cut_k] 重建尾部。该路径在 np=1 的 cut_k=1/0/N 三类路径下通过 Valgrind，但还没有证明 cut_k>0 时使用的是新矩阵诱导出的 A[cut_k]，也不是严格意义上的事务提交。
```

- [ ] **Step 3: Keep the valid evidence**

Keep these evidence lines:

```text
gate_c_Velz_1e-7.log: cut_k=1, tail rebuild, ERROR SUMMARY: 0, EXIT=0
gate_c_Vely_1e-7.log: cut_k=0, full setup, ERROR SUMMARY: 0, EXIT=0
gate_c_Vely_0.5.log: cut_k=6, full setup, ERROR SUMMARY: 0, EXIT=0
```

- [ ] **Step 4: Add an explicit next-gate statement**

Add:

```text
下一步不是 Gate D。下一步是 Gate C-correctness：让 cut_k>0 的 tail rebuild 从候选 A[cut_k] 开始，并补上失败回退语义说明。Gate D full reuse 和 MPI 释放继续等待。
```

---

## 4. Task 2: Add Candidate-A Return Path To Prediction

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/csrc/amg/par_amg_setup.c`
- Modify: `/home/xiyudawang/jxpamg-0220-pr/include/jx_pamg.h`

- [ ] **Step 1: Add a new public helper declaration**

Add to `include/jx_pamg.h` near the existing predictive reuse declarations:

```c
JX_Int JXPAMG_PR_PredictCutReadOnlyWithCandidate(jx_ParAMGData *,
                                                 jx_ParCSRMatrix *,
                                                 JX_Real,
                                                 JX_Int,
                                                 JX_Real,
                                                 JX_Int *,
                                                 jx_ParCSRMatrix **);
```

- [ ] **Step 2: Implement wrapper skeleton in `par_amg_setup.c`**

Add a function next to `JXPAMG_PR_PredictCutReadOnly`:

```c
JX_Int
JXPAMG_PR_PredictCutReadOnlyWithCandidate(jx_ParAMGData *amg_data,
                                          jx_ParCSRMatrix *new_A_fine,
                                          JX_Real theta,
                                          JX_Int Lreuse,
                                          JX_Real delta,
                                          JX_Int *cut_k,
                                          jx_ParCSRMatrix **candidate_A_cut)
{
   JX_Int status;

   if (candidate_A_cut == NULL)
   {
      return 1;
   }
   *candidate_A_cut = NULL;

   status = JXPAMG_PR_PredictCutReadOnly(amg_data, new_A_fine,
                                         theta, Lreuse, delta, cut_k);
   if (status != 0)
   {
      return status;
   }

   return 0;
}
```

Expected after this step:

- Build succeeds.
- Behavior is unchanged.
- `candidate_A_cut` is always NULL.

- [ ] **Step 3: Run build**

Run:

```bash
cd /home/xiyudawang/jxpamg-0220-pr
source /opt/intel/oneapi/setvars.sh --force
make -j8
make -f Makefile_pr clean
make -f Makefile_pr
```

Expected:

```text
exit code 0
solver_pr rebuilt
```

---

## 5. Task 3: Make Candidate-A Real For The First Failed Coarse Level

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/csrc/amg/par_amg_setup.c`

- [ ] **Step 1: Extend the prediction loop**

In the coarse probing loop, when a candidate `A_H` is built for level `level+1`, compare it with `A_array[level+1]`.

Use this behavior:

```c
if (sigma > theta)
{
   k = level + 1;
   if (candidate_A_cut != NULL)
   {
      *candidate_A_cut = A_H;
      A_H = NULL;
   }
   goto prediction_consensus;
}
```

Do not destroy `A_H` after assigning it to `candidate_A_cut`.

- [ ] **Step 2: Preserve current cleanup on non-cut levels**

For levels that do not become the cut:

```c
if (A_H != NULL)
{
   jx_ParCSRMatrixDestroy(A_H);
   A_H = NULL;
}
```

- [ ] **Step 3: On prediction failure, destroy candidate**

Before returning nonzero:

```c
if (status != 0 && candidate_A_cut != NULL && *candidate_A_cut != NULL)
{
   jx_ParCSRMatrixDestroy(*candidate_A_cut);
   *candidate_A_cut = NULL;
}
```

- [ ] **Step 4: Add diagnostic logging**

When candidate is captured:

```c
if (my_id == 0)
{
   jx_printf("[predictive-reuse] captured candidate A[%d] for tail rebuild\n", k);
}
```

Expected log for `Velz theta=1e-7`:

```text
probe ok, predicted cut_k=1
captured candidate A[1] for tail rebuild
```

---

## 6. Task 4: Rebuild Tail From Candidate A[cut_k]

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/csrc/amg/par_amg_setup.c`
- Modify: `/home/xiyudawang/jxpamg-0220-pr/include/jx_pamg.h`

- [ ] **Step 1: Add new helper declaration**

Add to `include/jx_pamg.h`:

```c
JX_Int JXPAMG_PR_SetupFromLevelWithACut(jx_ParAMGData *,
                                        JX_Int,
                                        jx_ParCSRMatrix *);
```

- [ ] **Step 2: Implement helper**

Add in `par_amg_setup.c` near `JXPAMG_PR_SetupFromLevelExperimental`:

```c
JX_Int
JXPAMG_PR_SetupFromLevelWithACut(jx_ParAMGData *amg_data,
                                 JX_Int k,
                                 jx_ParCSRMatrix *A_cut)
{
   jx_ParCSRMatrix **A_array;
   JX_Int num_levels;
   JX_Int my_id = 0;

   if (amg_data == NULL || A_cut == NULL)
   {
      return 1;
   }

   A_array = jx_ParAMGDataAArray(amg_data);
   if (A_array == NULL || A_array[0] == NULL)
   {
      return 1;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   num_levels = jx_ParAMGDataNumLevels(amg_data);
   if (k <= 0 || k >= num_levels)
   {
      return 1;
   }

   if (A_array[k] != NULL)
   {
      jx_ParCSRMatrixDestroy(A_array[k]);
      A_array[k] = NULL;
   }

   A_array[k] = A_cut;
   if (my_id == 0)
   {
      jx_printf("[predictive-reuse] Gate C: installed candidate A[%d]\n", k);
   }

   return JXPAMG_PR_SetupFromLevelExperimental(amg_data, k, 0);
}
```

Ownership rule:

```text
On success or failure after A_array[k] is assigned, amg_data owns A_cut.
The caller must set candidate_A_cut = NULL immediately after calling this helper.
```

- [ ] **Step 3: Add a comment documenting non-transactional behavior**

Above the helper:

```c
/* Gate C correctness helper.
 * This installs the candidate A[k] before the existing in-place tail rebuild.
 * It is not strict transactional commit: the old tail is freed before rebuild
 * completes. Failure must be handled by falling back to FullSetupOnNewMatrix.
 */
```

---

## 7. Task 5: Update Case 12 Routing

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/src/jxpamg_interface.c`

- [ ] **Step 1: Replace prediction call**

Change:

```c
JX_Int cut_k = 0;
```

to:

```c
JX_Int cut_k = 0;
jx_ParCSRMatrix *candidate_A_cut = NULL;
```

Change the prediction call to:

```c
probe_status = JXPAMG_PR_PredictCutReadOnlyWithCandidate(
   (jx_ParAMGData *)amg_solver, par_matrix,
   predictive_theta, predictive_lreuse,
   (JX_Real)1e-6, &cut_k, &candidate_A_cut);
```

- [ ] **Step 2: Gate tail rebuild to np=1 only**

Use this branch:

```c
if (probe_status == 0 &&
    cut_k > 0 &&
    cut_k < jx_ParAMGDataNumLevels((jx_ParAMGData *)amg_solver) &&
    num_procs == 1 &&
    candidate_A_cut != NULL)
{
   if (myid == 0)
      jx_printf("[predictive-reuse] Gate C: attempting tail rebuild from candidate A[%d]\n", cut_k);

   probe_status = JXPAMG_PR_SetupFromLevelWithACut((jx_ParAMGData *)amg_solver,
                                                   cut_k, candidate_A_cut);
   candidate_A_cut = NULL;

   if (probe_status == 0)
   {
      if (myid == 0) jx_printf("[predictive-reuse] Gate C: tail rebuild succeeded\n");
   }
   else
   {
      if (myid == 0)
         jx_printf("[predictive-reuse] Gate C: tail rebuild failed (status=%d), falling back to full setup\n", probe_status);
      probe_status = JXPAMG_PR_FullSetupOnNewMatrix((jx_ParAMGData *)amg_solver, par_matrix);
   }
}
else
{
   if (candidate_A_cut != NULL)
   {
      jx_ParCSRMatrixDestroy(candidate_A_cut);
      candidate_A_cut = NULL;
   }
   probe_status = JXPAMG_PR_FullSetupOnNewMatrix((jx_ParAMGData *)amg_solver, par_matrix);
}
```

- [ ] **Step 3: Ensure no candidate leaks on failure before setup branch**

Before any `break` in this block, add:

```c
if (candidate_A_cut != NULL)
{
   jx_ParCSRMatrixDestroy(candidate_A_cut);
   candidate_A_cut = NULL;
}
```

Expected behavior:

- `cut_k=1`, `np=1`, candidate exists: tail rebuild uses candidate.
- `cut_k=0`: full setup, no candidate.
- `cut_k=N`: full setup, candidate destroyed if any.
- `np>1`: full setup, no Gate C release.

---

## 8. Task 6: Add Correctness Diagnostics

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/csrc/amg/par_amg_setup.c`

- [ ] **Step 1: Log A[k] pointer before and after candidate install**

Inside `JXPAMG_PR_SetupFromLevelWithACut`:

```c
if (my_id == 0)
{
   jx_printf("[predictive-reuse] Gate C: replacing A[%d] old=%p candidate=%p\n",
             k, (void *)A_array[k], (void *)A_cut);
}
```

- [ ] **Step 2: Log hierarchy dimensions after rebuild**

After successful tail rebuild:

```c
JXPAMG_PR_PrintHierarchyShape("after Gate C candidate tail rebuild", amg_data);
JXPAMG_PR_CheckLinkDimensions("after Gate C candidate tail rebuild", amg_data);
```

Expected log for `Velz theta=1e-7`:

```text
captured candidate A[1] for tail rebuild
Gate C: replacing A[1] old=... candidate=...
Gate C: installed candidate A[1]
Gate C: tail rebuild succeeded
```

---

## 9. Task 7: Verification Matrix

**Files:**

- Create/update logs in `/home/xiyudawang/jxpamg-0220-pr/`

- [ ] **Step 1: Build**

Run:

```bash
cd /home/xiyudawang/jxpamg-0220-pr
source /opt/intel/oneapi/setvars.sh --force
make -j8
make -f Makefile_pr clean
make -f Makefile_pr
md5sum solver_pr
```

Expected:

```text
build exit code 0
new solver_pr checksum recorded
```

- [ ] **Step 2: Run cut_k=1 Valgrind**

Run:

```bash
cd /home/xiyudawang/jxpamg-0220-pr
valgrind --leak-check=full --show-leak-kinds=all \
  /opt/intel/oneapi/mpi/2021.6.0/bin/mpirun -np 1 ./solver_pr \
  /home/spring/l_s/jxpamg_all/data_trans/A_Velx-0.csr \
  /home/spring/l_s/jxpamg_all/data_trans/b_Velx-0.csr \
  1e-6 -sid 12 -maxiter 200 \
  -mat2 /home/spring/l_s/jxpamg_all/data_trans/A_Velz-0.csr \
  -rhs2 /home/spring/l_s/jxpamg_all/data_trans/b_Velz-0.csr \
  -ptheta 1e-7 \
  > gate_c_candidate_Velz_1e-7.log 2>&1
echo "EXIT=$?" >> gate_c_candidate_Velz_1e-7.log
```

Expected log must contain:

```text
probe ok, predicted cut_k=1
captured candidate A[1] for tail rebuild
Gate C: replacing A[1]
Gate C: installed candidate A[1]
Gate C: tail rebuild succeeded
iteration = 5
ERROR SUMMARY: 0 errors
EXIT=0
```

- [ ] **Step 3: Run cut_k=0 Valgrind**

Run:

```bash
cd /home/xiyudawang/jxpamg-0220-pr
valgrind --leak-check=full --show-leak-kinds=all \
  /opt/intel/oneapi/mpi/2021.6.0/bin/mpirun -np 1 ./solver_pr \
  /home/spring/l_s/jxpamg_all/data_trans/A_Velx-0.csr \
  /home/spring/l_s/jxpamg_all/data_trans/b_Velx-0.csr \
  1e-6 -sid 12 -maxiter 200 \
  -mat2 /home/spring/l_s/jxpamg_all/data_trans/A_Vely-0.csr \
  -rhs2 /home/spring/l_s/jxpamg_all/data_trans/b_Vely-0.csr \
  -ptheta 1e-7 \
  > gate_c_candidate_Vely_1e-7.log 2>&1
echo "EXIT=$?" >> gate_c_candidate_Vely_1e-7.log
```

Expected:

```text
probe ok, predicted cut_k=0
no Gate C tail rebuild
ERROR SUMMARY: 0 errors
EXIT=0
```

- [ ] **Step 4: Run cut_k=N Valgrind**

Run:

```bash
cd /home/xiyudawang/jxpamg-0220-pr
valgrind --leak-check=full --show-leak-kinds=all \
  /opt/intel/oneapi/mpi/2021.6.0/bin/mpirun -np 1 ./solver_pr \
  /home/spring/l_s/jxpamg_all/data_trans/A_Velx-0.csr \
  /home/spring/l_s/jxpamg_all/data_trans/b_Velx-0.csr \
  1e-6 -sid 12 -maxiter 200 \
  -mat2 /home/spring/l_s/jxpamg_all/data_trans/A_Vely-0.csr \
  -rhs2 /home/spring/l_s/jxpamg_all/data_trans/b_Vely-0.csr \
  -ptheta 0.5 \
  > gate_c_candidate_Vely_0.5.log 2>&1
echo "EXIT=$?" >> gate_c_candidate_Vely_0.5.log
```

Expected:

```text
probe ok, predicted cut_k=6
no Gate C tail rebuild
ERROR SUMMARY: 0 errors
EXIT=0
```

- [ ] **Step 5: Run np=1 batch**

Run the same 5 matrix pairs used by `gate_b_results.tsv`:

```text
Velx -> PresCorr, theta=1e-7
Velx -> TurbK,    theta=1e-7
Velx -> Vely,     theta=1e-7
Velx -> Velz,     theta=1e-7
Velx -> Vof,      theta=1e-7
```

Expected:

```text
all exit 0
PresCorr iterations around 13
TurbK iterations around 2
Vely iterations around 4
Velz iterations around 5
Vof iterations around 4
```

---

## 10. Task 8: Update Final Status After Verification

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/OVERALL_STATUS.md`

Only after Task 7 passes, update Gate C status to:

```text
Gate C | ✅ np=1 correctness | candidate A[cut_k] tail rebuild verified for cut_k=1; cut_k=0/N full setup fallback verified; Valgrind 0 errors
```

Do not call it strict transactional unless a staged temporary hierarchy and commit/rollback implementation exists.

Keep this limitation:

```text
Gate C is still in-place after candidate A[cut_k] install. On tail rebuild failure, fallback to FullSetupOnNewMatrix repairs the hierarchy, but old tail is not preserved. Strict transaction-by-staging remains future work.
```

Do not mark any of these complete:

```text
Gate D full reuse
3B-2 integration
np>1 Gate C
MPI release
```

---

## 11. Completion Criteria

Gate C correctness can be accepted when all are true:

- `cut_k=1` path logs candidate capture and candidate install.
- `cut_k=1` path solves and Valgrind reports 0 errors.
- `cut_k=0` path remains full setup and Valgrind reports 0 errors.
- `cut_k=N` path remains full setup and Valgrind reports 0 errors.
- No Gate C tail rebuild is attempted for `np>1`.
- `OVERALL_STATUS.md` does not claim strict transaction semantics.
- `Gate D`, `3B-2`, and MPI release remain explicitly pending.

---

## 12. Why This Is The Right Next Step

This focuses on the real correctness gap instead of adding new features. The current code has proved that the rebuilt tail can run without memory errors in one `np=1` case. The next risk is semantic: whether the rebuilt tail corresponds to the new matrix or merely rebuilds below an old coarse operator. Fixing `A[cut_k]` first makes later performance work meaningful. Only after that should full reuse or MPI be reconsidered.
