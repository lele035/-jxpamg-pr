# Gate D Full Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the first safe full-reuse path for predictive reuse when `cut_k == num_levels`, limited to `np=1`, while preserving Gate C behavior for `cut_k=0` and `0 < cut_k < num_levels`.

**Architecture:** Treat full reuse as a separate setup route, not as tail rebuild. When the read-only predictor returns `cut_k == num_levels`, keep the existing AMG hierarchy unchanged and make PCG use it as the preconditioner for the new fine-level matrix. Avoid calling `JX_PAMGSetup` in this path by adding an explicit no-op setup function for PCG.

**Tech Stack:** C, MPI, JXPAMG AMG hierarchy (`jx_ParAMGData`, `jx_ParCSRMatrix`), PCG preconditioner interface, Valgrind, existing `solver_pr` case 12 predictive reuse path.

---

## 0. Current State

Current proven baseline:

- Phase 1: complete, stable full rebuild path.
- Gate B: `np=1` multi-level read-only predictor works and returns `cut_k`.
- Gate C: `np=1` correctness path works:
  - `cut_k=0`: full rebuild.
  - `0 < cut_k < num_levels`: install candidate `A[cut_k]`, then in-place tail rebuild.
  - `cut_k == num_levels`: currently still full rebuild.

Gate D should only change the last case:

```text
cut_k == num_levels  ->  full reuse
```

It must not change:

```text
probe failure        ->  full rebuild
cut_k == 0           ->  full rebuild
0 < cut_k < N        ->  Gate C candidate tail rebuild
np > 1               ->  full rebuild
```

---

## 1. Why Full Reuse Needs A Separate Setup Path

The current code calls:

```c
JX_PCGSetPrecond(solver,
                 (JX_PtrToSolverFcn)JX_PAMGPrecond,
                 (JX_PtrToSolverFcn)JX_PAMGSetup,
                 amg_solver);
```

If PCG setup invokes the registered setup function, then passing `JX_PAMGSetup` will rebuild AMG and defeat full reuse.

Gate D therefore needs a no-op setup function:

```c
JX_Int JXPAMG_PR_NoOpSetup(JX_Solver solver, JX_ParCSRMatrix A);
```

For the full reuse path only, register:

```c
JX_PCGSetPrecond(solver,
                 (JX_PtrToSolverFcn)JX_PAMGPrecond,
                 (JX_PtrToSolverFcn)JXPAMG_PR_NoOpSetup,
                 amg_solver);
```

This lets PCG solve the new matrix while applying the old AMG hierarchy as a reused preconditioner.

---

## 2. Safety Scope

Gate D v1 is intentionally narrow:

```text
Allowed:
  np == 1
  probe_status == 0
  cut_k == num_levels
  predictive_theta > 0
  fine matrix dimensions match old A_array[0]

Rejected:
  np > 1
  probe failure
  cut_k < num_levels
  dimension mismatch
  missing AMG hierarchy arrays
```

Do not implement:

```text
MPI full reuse
3B-2
strict transaction-by-staging
full reuse for partial Lreuse truncation
full reuse when predictor did not probe all levels
```

The `Lreuse` caveat matters: if only part of the hierarchy was probed, `cut_k == num_levels` may be a false full-reuse signal. Gate D must require full-depth probing.

---

## 3. Files To Touch

### `include/jx_pamg.h`

Declare:

```c
JX_Int JXPAMG_PR_NoOpSetup(JX_Solver, JX_ParCSRMatrix);
JX_Int JXPAMG_PR_CanFullReuse(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Int, JX_Int);
```

### `csrc/amg/par_amg_setup.c`

Implement:

- no-op setup function
- full-reuse eligibility checker
- diagnostic logging helper if useful

### `src/jxpamg_interface.c`

Modify case 12 routing:

- Call no-op setup only for safe full reuse.
- Preserve full setup fallback for all other paths.
- Log which setup path was selected.

### `OVERALL_STATUS.md`

After verification, update Gate D status and evidence files.

---

## 4. Task 0: Archive Gate C Batch Evidence

**Files:**

- Create: `/home/xiyudawang/jxpamg-0220-pr/gate_c_candidate_results.tsv`
- Modify: `/home/xiyudawang/jxpamg-0220-pr/OVERALL_STATUS.md`

- [ ] **Step 1: Create Gate C result table**

Write a TSV with this exact schema:

```text
binary_md5 np matrix1 matrix2 theta probe_status cut_k path candidate iterations residual exit
```

Expected rows:

```text
60439a1a 1 Velx-0 PresCorr-0 1e-7 0 0 full_setup no 13 <residual> 0
60439a1a 1 Velx-0 TurbK-0    1e-7 0 0 full_setup no 2  <residual> 0
60439a1a 1 Velx-0 Vely-0     1e-7 0 0 full_setup no 4  <residual> 0
60439a1a 1 Velx-0 Velz-0     1e-7 0 1 tail_rebuild yes 5 <residual> 0
60439a1a 1 Velx-0 Vof-0      1e-7 0 0 full_setup no 4  <residual> 0
```

- [ ] **Step 2: Update status doc**

In `OVERALL_STATUS.md`, make the Gate C 5/5 statement point to this TSV:

```text
批测证据: gate_c_candidate_results.tsv
```

Do this before Gate D so later changes do not blur Gate C evidence.

---

## 5. Task 1: Add No-Op Setup

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/include/jx_pamg.h`
- Modify: `/home/xiyudawang/jxpamg-0220-pr/csrc/amg/par_amg_setup.c`

- [ ] **Step 1: Add declaration**

Add near predictive reuse declarations:

```c
JX_Int JXPAMG_PR_NoOpSetup(JX_Solver solver, JX_ParCSRMatrix A);
```

- [ ] **Step 2: Add implementation**

Add to `par_amg_setup.c`:

```c
JX_Int
JXPAMG_PR_NoOpSetup(JX_Solver solver, JX_ParCSRMatrix A)
{
   jx_ParAMGData *amg_data = (jx_ParAMGData *) solver;
   jx_ParCSRMatrix *A_par = (jx_ParCSRMatrix *) A;
   jx_ParCSRMatrix **A_array;
   JX_Int my_id = 0;

   if (amg_data == NULL || A_par == NULL)
   {
      return 1;
   }

   A_array = jx_ParAMGDataAArray(amg_data);
   if (A_array == NULL || A_array[0] == NULL)
   {
      return 1;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (my_id == 0)
   {
      jx_printf("[predictive-reuse] Gate D: no-op AMG setup (full reuse)\n");
   }

   return 0;
}
```

- [ ] **Step 3: Build**

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
```

---

## 6. Task 2: Add Full-Reuse Eligibility Check

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/include/jx_pamg.h`
- Modify: `/home/xiyudawang/jxpamg-0220-pr/csrc/amg/par_amg_setup.c`

- [ ] **Step 1: Add declaration**

```c
JX_Int JXPAMG_PR_CanFullReuse(jx_ParAMGData *amg_data,
                              jx_ParCSRMatrix *new_A,
                              JX_Int cut_k,
                              JX_Int Lreuse);
```

- [ ] **Step 2: Add implementation**

```c
JX_Int
JXPAMG_PR_CanFullReuse(jx_ParAMGData *amg_data,
                       jx_ParCSRMatrix *new_A,
                       JX_Int cut_k,
                       JX_Int Lreuse)
{
   jx_ParCSRMatrix **A_array;
   jx_ParCSRMatrix **P_array;
   jx_ParVector **F_array;
   jx_ParVector **U_array;
   JX_Int num_levels;
   JX_Int level;

   if (amg_data == NULL || new_A == NULL)
   {
      return 0;
   }

   A_array = jx_ParAMGDataAArray(amg_data);
   P_array = jx_ParAMGDataPArray(amg_data);
   F_array = jx_ParAMGDataFArray(amg_data);
   U_array = jx_ParAMGDataUArray(amg_data);
   num_levels = jx_ParAMGDataNumLevels(amg_data);

   if (num_levels <= 1 || cut_k != num_levels)
   {
      return 0;
   }

   if (Lreuse > 0 && Lreuse < num_levels)
   {
      return 0;
   }

   if (A_array == NULL || P_array == NULL || F_array == NULL || U_array == NULL)
   {
      return 0;
   }

   if (A_array[0] == NULL)
   {
      return 0;
   }

   if (jx_ParCSRMatrixGlobalNumRows(new_A) != jx_ParCSRMatrixGlobalNumRows(A_array[0]) ||
       jx_ParCSRMatrixGlobalNumCols(new_A) != jx_ParCSRMatrixGlobalNumCols(A_array[0]) ||
       jx_ParCSRMatrixNumRows(new_A) != jx_ParCSRMatrixNumRows(A_array[0]))
   {
      return 0;
   }

   for (level = 0; level < num_levels; level++)
   {
      if (A_array[level] == NULL)
      {
         return 0;
      }
   }

   for (level = 0; level < num_levels - 1; level++)
   {
      if (P_array[level] == NULL)
      {
         return 0;
      }
   }

   for (level = 1; level < num_levels; level++)
   {
      if (F_array[level] == NULL || U_array[level] == NULL)
      {
         return 0;
      }
   }

   return 1;
}
```

Rationale:

```text
Full reuse is allowed only if the old hierarchy is complete and dimensions match.
Partial Lreuse cannot release full reuse because unprobed levels may differ.
```

---

## 7. Task 3: Route cut_k == num_levels To Full Reuse

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/src/jxpamg_interface.c`

- [ ] **Step 1: Add full-reuse flag**

Inside case 12 second-call block:

```c
JX_Int use_full_reuse = 0;
```

- [ ] **Step 2: Compute full-reuse eligibility after prediction**

After successful prediction:

```c
if (probe_status == 0 &&
    num_procs == 1 &&
    JXPAMG_PR_CanFullReuse((jx_ParAMGData *)amg_solver,
                           par_matrix,
                           cut_k,
                           predictive_lreuse))
{
   use_full_reuse = 1;
}
```

- [ ] **Step 3: Register preconditioner setup based on route**

Replace the unconditional `JX_PCGSetPrecond(... JX_PAMGSetup ...)` with:

```c
if (use_full_reuse)
{
   JX_PCGSetPrecond(solver,
                    (JX_PtrToSolverFcn)JX_PAMGPrecond,
                    (JX_PtrToSolverFcn)JXPAMG_PR_NoOpSetup,
                    amg_solver);
   if (myid == 0)
   {
      jx_printf("[predictive-reuse] Gate D: full reuse selected (cut_k=%d)\n", cut_k);
   }
}
else
{
   JX_PCGSetPrecond(solver,
                    (JX_PtrToSolverFcn)JX_PAMGPrecond,
                    (JX_PtrToSolverFcn)JX_PAMGSetup,
                    amg_solver);
}
```

- [ ] **Step 4: Skip setup mutation on full reuse**

Before the existing setup routing:

```c
if (use_full_reuse)
{
   probe_status = 0;
}
else if (probe_status == 0 &&
         cut_k > 0 &&
         cut_k < jx_ParAMGDataNumLevels((jx_ParAMGData *)amg_solver) &&
         num_procs == 1 &&
         candidate_A_cut != NULL)
{
   ...
}
else
{
   ...
}
```

Important:

```text
Full reuse must not call FullSetupOnNewMatrix.
Full reuse must not call SetupFromLevelWithACut.
Full reuse must destroy candidate_A_cut if any unexpected candidate exists.
```

- [ ] **Step 5: Add defensive candidate cleanup**

Inside the `use_full_reuse` branch:

```c
if (candidate_A_cut != NULL)
{
   jx_ParCSRMatrixDestroy(candidate_A_cut);
   candidate_A_cut = NULL;
}
```

For a true `cut_k == num_levels` path, candidate should normally be NULL.

---

## 8. Task 4: Verify Gate D Does Not Break Gate C

**Files:**

- Create logs in `/home/xiyudawang/jxpamg-0220-pr/`

- [ ] **Step 1: Build and record checksum**

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

- [ ] **Step 2: Full reuse case, cut_k=N**

Use the known `Vely theta=0.5` case.

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
  > gate_d_full_reuse_Vely_0.5.log 2>&1
echo "EXIT=$?" >> gate_d_full_reuse_Vely_0.5.log
```

Expected log:

```text
probe ok, predicted cut_k=6
Gate D: full reuse selected
Gate D: no-op AMG setup (full reuse)
ERROR SUMMARY: 0 errors
EXIT=0
```

Expected solve:

```text
PCG converges within maxiter
iteration count recorded
```

If this case reaches maxiter or fails, do not keep Gate D enabled by default.

- [ ] **Step 3: Gate C tail rebuild still works**

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
  > gate_d_regression_Velz_1e-7.log 2>&1
echo "EXIT=$?" >> gate_d_regression_Velz_1e-7.log
```

Expected:

```text
predicted cut_k=1
captured candidate A[1]
Gate C: tail rebuild succeeded
ERROR SUMMARY: 0 errors
EXIT=0
```

- [ ] **Step 4: cut_k=0 still full rebuilds**

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
  > gate_d_regression_Vely_1e-7.log 2>&1
echo "EXIT=$?" >> gate_d_regression_Vely_1e-7.log
```

Expected:

```text
predicted cut_k=0
no Gate D full reuse selected
ERROR SUMMARY: 0 errors
EXIT=0
```

---

## 9. Task 5: Add Gate D Result Table

**Files:**

- Create: `/home/xiyudawang/jxpamg-0220-pr/gate_d_results.tsv`

Use schema:

```text
binary_md5 np matrix1 matrix2 theta probe_status cut_k path setup_fn iterations residual exit
```

Minimum rows:

```text
<md5> 1 Velx-0 Vely-0 0.5  0 6 full_reuse noop_setup <iters> <residual> 0
<md5> 1 Velx-0 Velz-0 1e-7 0 1 tail_rebuild JX_PAMGSetup 5 <residual> 0
<md5> 1 Velx-0 Vely-0 1e-7 0 0 full_setup JX_PAMGSetup 4 <residual> 0
```

Acceptance:

```text
all exit 0
all Valgrind logs have ERROR SUMMARY: 0
```

---

## 10. Task 6: Update Overall Status

**Files:**

- Modify: `/home/xiyudawang/jxpamg-0220-pr/OVERALL_STATUS.md`

Only after Task 4 and Task 5 pass, update:

```text
Gate D | ✅ np=1 minimal | cut_k=N full reuse via no-op AMG setup; Valgrind 0 errors
```

Keep limitations:

```text
np>1 full reuse not released
3B-2 not integrated
strict transaction-by-staging not implemented
partial Lreuse cannot trigger full reuse
```

Add evidence:

```text
gate_d_full_reuse_Vely_0.5.log
gate_d_regression_Velz_1e-7.log
gate_d_regression_Vely_1e-7.log
gate_d_results.tsv
```

---

## 11. Completion Criteria

Gate D v1 is complete only when all are true:

- `cut_k == num_levels` selects full reuse.
- Full reuse registers `JXPAMG_PR_NoOpSetup`, not `JX_PAMGSetup`.
- Full reuse does not call `FullSetupOnNewMatrix`.
- `Vely theta=0.5` converges with `EXIT=0`.
- Valgrind reports `ERROR SUMMARY: 0` for the full-reuse case.
- Gate C `cut_k=1` regression still captures candidate `A[1]` and succeeds.
- Gate C `cut_k=0` regression still full rebuilds and succeeds.
- `np>1` remains forced to full rebuild.
- `gate_d_results.tsv` records the exact binary md5 and outputs.

---

## 12. Next After Gate D

If Gate D v1 passes, the next decision is:

```text
Option A: performance comparison for full setup vs tail rebuild vs full reuse
Option B: strict transaction-by-staging for Gate C
Option C: np=2 investigation starting from CommPkg/deep-copy reliability
Option D: 3B-2 integration
```

Recommended order:

```text
1. Gate D v1 full reuse np=1
2. Performance/evidence table for 3A paths
3. Strict transaction-by-staging if needed
4. MPI reliability work
5. 3B-2 integration
```
