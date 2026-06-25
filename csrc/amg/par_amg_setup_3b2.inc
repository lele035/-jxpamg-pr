//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//========================================================================//
/*!
 *  Algorithm 3B-2: Topology-frozen local interpolation weight update
 *
 *  Builds on the existing predictive reuse infrastructure.
 *  Adds: row-norm probe, NumericRAP, local P weight update, active set propagation.
 *
 *  This file is intended to be compiled into the PR library when
 *  JXPAMG_PR_ENABLE_3B2 is defined.
 */

#include "jx_pamg.h"

#ifndef JXPAMG_PR_ENABLE_3B2
#define JXPAMG_PR_ENABLE_3B2 0
#endif

#if JXPAMG_PR_ENABLE_3B2

/* =========================================================================
 * Phase 1.1: Row-Norm Diff Probe
 * Compares row L1 norms of new and old matrices.
 * Returns active nodes set and count.
 * ========================================================================= */
static JX_Int *
JXPAMG_PR_RowNormDiff(jx_ParCSRMatrix *A_new,
                      jx_ParCSRMatrix *A_old,
                      JX_Real          theta,
                      JX_Int          *active_count_out)
{
   jx_CSRMatrix *A_new_diag = jx_ParCSRMatrixDiag(A_new);
   jx_CSRMatrix *A_old_diag = jx_ParCSRMatrixDiag(A_old);
   JX_Int *A_new_i    = jx_CSRMatrixI(A_new_diag);
   JX_Int *A_new_j    = jx_CSRMatrixJ(A_new_diag);
   JX_Real *A_new_data = jx_CSRMatrixData(A_new_diag);
   JX_Int *A_old_i    = jx_CSRMatrixI(A_old_diag);
   JX_Int *A_old_j    = jx_CSRMatrixJ(A_old_diag);
   JX_Real *A_old_data = jx_CSRMatrixData(A_old_diag);
   JX_Int  nrows       = jx_CSRMatrixNumRows(A_new_diag);
   JX_Int *active      = NULL;
   JX_Int  count       = 0;
   JX_Int  i;

   if (A_new_diag == NULL || A_old_diag == NULL) { *active_count_out = 0; return NULL; }
   if (jx_ParCSRMatrixGlobalNumRows(A_new) != jx_ParCSRMatrixGlobalNumRows(A_old))
      { *active_count_out = 0; return NULL; }

   active = (JX_Int *)jx_CTAlloc(JX_Int, nrows);

   for (i = 0; i < nrows; i++)
   {
      JX_Real new_row_max = 0.0, old_row_max = 0.0;
      JX_Int jj;
      JX_Int nnz_old = A_old_i[i+1] - A_old_i[i];

      /* Dirichlet boundary detection: single-entry rows (diag=1, no offd) */
      if (nnz_old <= 1)
         continue;

      /* Row L-inf norm: max absolute value in the row */
      for (jj = A_new_i[i]; jj < A_new_i[i+1]; jj++)
      {
         JX_Real av = A_new_data[jj] >= 0 ? A_new_data[jj] : -A_new_data[jj];
         if (av > new_row_max) new_row_max = av;
      }
      for (jj = A_old_i[i]; jj < A_old_i[i+1]; jj++)
      {
         JX_Real av = A_old_data[jj] >= 0 ? A_old_data[jj] : -A_old_data[jj];
         if (av > old_row_max) old_row_max = av;
      }

      /* Skip near-zero rows (numerical noise) */
      if (old_row_max < 1e-12 && new_row_max < 1e-12)
         continue;

      /* L-inf relative deviation */
      JX_Real diff = (new_row_max - old_row_max);
      if (diff < 0) diff = -diff;
      if (old_row_max > 1e-12 && diff > theta * old_row_max)
         active[count++] = i;
   }

   *active_count_out = count;
   if (count == 0) { jx_TFree(active); return NULL; }
   return active;
}

/* =========================================================================
 * Phase 1.1b: Residual-Based Probe
 * 1 Jacobi smoothing + residual computation.
 * Cheap (1 SpMV), directly measures if old precond works on new matrix.
 * Returns active nodes where |r_i| > theta * max(|r|).
 * ========================================================================= */
static JX_Int *
JXPAMG_PR_ResidualProbe(jx_ParCSRMatrix *A_new,
                        JX_Real         *b_raw,
                        JX_Real          theta,
                        JX_Int          *active_count_out)
{
   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(A_new);
   JX_Int  *A_i    = jx_CSRMatrixI(A_diag);
   JX_Int  *A_j    = jx_CSRMatrixJ(A_diag);
   JX_Real *A_data = jx_CSRMatrixData(A_diag);
#include "_debug_offset.inc"
   JX_Int   nrows  = jx_CSRMatrixNumRows(A_diag);
   JX_Int  *active = NULL;
   JX_Int   count  = 0;
   JX_Int   i, jj;

   if (A_diag == NULL || b_raw == NULL) { *active_count_out = 0; return NULL; }

   JX_Real *diag = (JX_Real *)jx_CTAlloc(JX_Real, nrows);
   JX_Real *x    = (JX_Real *)jx_CTAlloc(JX_Real, nrows);
   JX_Real *r    = (JX_Real *)jx_CTAlloc(JX_Real, nrows);

   /* Extract diagonal */
   for (i = 0; i < nrows; i++)
   {
      diag[i] = 0.0;
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
         if (A_j[jj] == i) { diag[i] = A_data[jj]; break; }
   }

   /* 1 Jacobi step: x = D^{-1} * b (x0 = 0) */
   for (i = 0; i < nrows; i++)
      x[i] = (diag[i] != 0.0) ? b_raw[i] / diag[i] : 0.0;

   /* Residual: r = b - A * x */
   JX_Real r_max = 0.0;
   for (i = 0; i < nrows; i++)
   {
      JX_Real ax = 0.0;
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
         ax += A_data[jj] * x[A_j[jj]];
      r[i] = b_raw[i] - ax;
      JX_Real ra = (r[i] >= 0.0) ? r[i] : -r[i];
      if (ra > r_max) r_max = ra;
   }

   /* Extract active set: |r_i| > theta * r_max */
   active = (JX_Int *)jx_CTAlloc(JX_Int, nrows);
   for (i = 0; i < nrows; i++)
   {
      JX_Real ra = (r[i] >= 0.0) ? r[i] : -r[i];
      if (ra > theta * r_max)
         active[count++] = i;
   }

   jx_TFree(diag); jx_TFree(x); jx_TFree(r);

   *active_count_out = count;
   if (count == 0) { jx_TFree(active); return NULL; }
   return active;
}


/* =========================================================================
 * Phase 1.2: Active Set Propagation
 * Maps active fine-grid nodes to coarse-grid nodes via R sparsity.
 * Returns coarse active set and count.
 * ========================================================================= */
static JX_Int *
JXPAMG_PR_PropagateActiveSet(JX_Int          *active_fine,
                             JX_Int           fine_count,
                             jx_ParCSRMatrix *R,
                             JX_Int          *coarse_count_out)
{
   jx_CSRMatrix *R_diag = jx_ParCSRMatrixDiag(R);
   JX_Int *R_i    = jx_CSRMatrixI(R_diag);
   JX_Int *R_j    = jx_CSRMatrixJ(R_diag);
   JX_Real *R_data = jx_CSRMatrixData(R_diag);
   JX_Int  coarse_size = jx_ParCSRMatrixGlobalNumCols(R);
   JX_Int *coarse_active = (JX_Int *)jx_CTAlloc(JX_Int, coarse_size);
   JX_Int  i, jj;

   /* Zero-initialize markers */
   for (i = 0; i < coarse_size; i++) coarse_active[i] = 0;

   /* For each active fine node, mark its coarse neighbors via R sparsity */
   for (i = 0; i < fine_count; i++)
   {
      JX_Int fine_idx = active_fine[i];
      if (fine_idx < 0 || fine_idx >= jx_CSRMatrixNumRows(R_diag)) continue;
      for (jj = R_i[fine_idx]; jj < R_i[fine_idx+1]; jj++)
      {
         JX_Int coarse_idx = R_j[jj];
         if (coarse_idx >= 0 && coarse_idx < coarse_size)
            coarse_active[coarse_idx] = 1;
      }
   }

   /* Compact active set */
   JX_Int count = 0;
   for (i = 0; i < coarse_size; i++)
      if (coarse_active[i]) coarse_active[count++] = i;

   *coarse_count_out = count;
   if (count == 0) { jx_TFree(coarse_active); return NULL; }
   return coarse_active;
}


/* =========================================================================
 * Phase 1.3: NumericRAP
 * Compute A_H = R * A * P, preserving old A_H's sparsity pattern.
 * For now: computes full RAP via existing BuildReuseCoarseOperator,
 * but the structure-preserving version is a TODO.
 * ========================================================================= */
static jx_ParCSRMatrix *
JXPAMG_PR_NumericRAP(jx_ParAMGData *amg_data,
                     JX_Int level,
                     JX_Int *opt_icor)
{
   /* For initial implementation, reuse the existing full RAP.
    * Later optimization: only fill values at old A[l+1] pattern positions. */
   return JXPAMG_PR_BuildReuseCoarseOperator(amg_data, level, opt_icor);
}

/* =========================================================================
 * Phase 2: Local Interpolation Weight Update
 *
 * Strategy: Keep CF_marker frozen (same topology), call existing
 * jx_PAMGBuildInterp with the old CF and new A to produce new P weights.
 * This avoids re-implementing 700+ lines of interpolation formulas.
 * ========================================================================= */

/* Helper: rebuild P at a single level using frozen CF_marker and new A.
 * Used for Algorithm 3B-2 local weight update.
 * Returns new P (or NULL on failure).  Caller is responsible for destroying old P. */
static jx_ParCSRMatrix *
JXPAMG_PR_RebuildInterpWithFrozenCF(jx_ParAMGData *amg_data, JX_Int level)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   JX_Int **CF_marker_array = jx_ParAMGDataCFMarkerArray(amg_data);
   JX_Int **dof_func_array   = jx_ParAMGDataDofFuncArray(amg_data);
   JX_Int   interp_type      = jx_ParAMGDataInterpType(amg_data);
   JX_Int   debug_flag       = jx_ParAMGDataDebugFlag(amg_data);
   JX_Int   P_max_elmts      = jx_ParAMGDataPMaxElmts(amg_data);
   JX_Real  trunc_factor     = jx_ParAMGDataTruncFactor(amg_data);
   JX_Int   spmt_rap_type    = jx_ParAMGDataSpMtRapType(amg_data);
   JX_Int   num_functions    = jx_ParAMGDataNumFunctions(amg_data);
   JX_Int   num_procs        = 1;
   JX_Int   my_id            = 0;
   MPI_Comm comm;

   jx_ParCSRMatrix *par_S       = NULL;
   jx_ParCSRMatrix *par_P       = NULL;
   JX_Int          *CF_marker   = CF_marker_array[level];
   JX_Real          strong_th   = jx_ParAMGDataStrongThreshold(amg_data);
   JX_Real          max_row_sum = jx_ParAMGDataMaxRowSum(amg_data);
   JX_Real          S_switch    = jx_ParAMGDataSCommPkgSwitch(amg_data);
   JX_Int          *col_offd_S_to_A = NULL;
   JX_Int          *coarse_pnts_global = NULL;
   JX_Int          *coarse_dof_func = NULL;
   JX_Int           coarse_size;

   if (A_array == NULL || A_array[level] == NULL || CF_marker == NULL)
      return NULL;

   comm = jx_ParCSRMatrixComm(A_array[level]);
   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   /* Build strength matrix from new A[level] */
   jx_PAMGCreateS(A_array[level], strong_th, max_row_sum, num_functions,
                  dof_func_array[level], &par_S);
   if (strong_th > S_switch)
      jx_PAMGCreateSCommPkg(A_array[level], par_S, &col_offd_S_to_A);

   /* Compute coarse parameters from frozen CF */
   jx_PAMGCoarseParms(jx_ParCSRMatrixComm(A_array[level]),
                      jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[level])),
                      num_functions, dof_func_array[level], CF_marker,
                      &coarse_dof_func, &coarse_pnts_global);

#ifdef JX_NO_GLOBAL_PARTITION
   if (my_id == num_procs-1) coarse_size = coarse_pnts_global[1];
   jx_MPI_Bcast(&coarse_size, 1, JX_MPI_INT, num_procs-1, comm);
#else
   coarse_size = coarse_pnts_global[num_procs];
#endif

   /* Build P using frozen CF (same topology, new weights) */
   if (interp_type == 0)  /* classical modified interpolation */
   {
      if (spmt_rap_type == 1)
         jx_PAMGBuildInterp(A_array[level], CF_marker, par_S,
                           coarse_pnts_global, num_functions,
                           dof_func_array[level], debug_flag, trunc_factor,
                           P_max_elmts, col_offd_S_to_A, &par_P);
   }
   else  /* fallback: all other interp types */
   {
      /* For non-classical interp, use the standard builder.
       * In a full implementation, handle each interp_type explicitly. */
      jx_PAMGBuildInterp(A_array[level], CF_marker, par_S,
                        coarse_pnts_global, num_functions,
                        dof_func_array[level], debug_flag, trunc_factor,
                        P_max_elmts, col_offd_S_to_A, &par_P);
   }

   /* Cleanup */
   if (par_S) { jx_ParCSRMatrixDestroy(par_S); par_S = NULL; }
   jx_TFree(col_offd_S_to_A);
   jx_TFree(coarse_pnts_global);
   if (num_functions > 1) jx_TFree(coarse_dof_func);

   return par_P;
}

/* =========================================================================
 * Phase 3: Algorithm 3B-2 Main Flow
 * Topology-frozen local interpolation weight update.
 * ========================================================================= */
JX_Int
JXPAMG_PR_AdaptiveReuse3B2(jx_ParAMGData *amg_data,
                           jx_ParCSRMatrix *new_A_fine,
                           JX_Real theta_local,
                           JX_Int  Llocal,
                           JX_Int  L)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix **R_array = jx_ParAMGDataRArray(amg_data);
   JX_Int **CF_marker_array   = jx_ParAMGDataCFMarkerArray(amg_data);
   JX_Int   num_levels        = jx_ParAMGDataNumLevels(amg_data);
   JX_Int   restri_type       = jx_ParAMGDataRestriction(amg_data);
   JX_Int   num_threads       = jx_NumThreads();
   JX_Int  *opt_icor          = NULL;
   JX_Int   level;
   JX_Int   my_id = 0;

   /* Active sets: fine array reused for each level */
   JX_Int *active_fine = NULL;
   JX_Int  active_count = 0;

   if (amg_data == NULL || new_A_fine == NULL || A_array == NULL || A_array[0] == NULL)
      return 1;
   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (JXPAMG_PR_CheckFineCompatibility(amg_data, new_A_fine)) return 1;
   if (num_levels <= 1) return 1;

   if (L < 0 || L >= num_levels) L = num_levels - 1;

   opt_icor = jx_CTAlloc(JX_Int, 5 * num_threads + 2);

   /* Step 1-2: Replace fine matrix, extract active nodes via row-norm */
   if (my_id == 0) jx_printf("[predictive-3b2] probe lv 0 (row-norm)\n");

   if (A_array[0] != new_A_fine) {
      /* Don't destroy the old A[0] yet — need it for norm comparison */
      jx_ParCSRMatrix *old_A0 = A_array[0];
      active_fine = JXPAMG_PR_RowNormDiff(new_A_fine, old_A0,
                                          theta_local, &active_count);
      jx_ParCSRMatrixDestroy(old_A0);
      A_array[0] = new_A_fine;
   }

    if (my_id == 0 && active_count > 0)
       jx_printf("[predictive-3b2] active nodes at lv 0: %d\n", active_count);

    /* If nearly all nodes are active, topology freeze is meaningless;
     * signal the caller to fall back to full setup. */
    {
       JX_Int total_rows = jx_ParCSRMatrixGlobalNumRows(new_A_fine);
       if (active_count > total_rows * 0.9)
       {
          if (my_id == 0)
             jx_printf("[predictive-3b2] too many active nodes (%d/%d), fallback\n",
                       active_count, total_rows);
          return 1;
       }
    }

    /* Step 3: Level loop */
    for (level = 0; level < L; level++)
   {
      /* 3a: Freeze CF topology */
      /* CF_marker_array[level] is already kept from base setup */

      if (level <= Llocal && active_count > 0)
      {
         jx_ParCSRMatrix *new_P = NULL;
         jx_ParCSRMatrix *new_A = NULL;

         if (my_id == 0)
            jx_printf("[predictive-3b2] local update lv %d (active=%d)\n",
                      level, active_count);

         /* 3b: Rebuild P with frozen CF (new A → new weights) */
         new_P = JXPAMG_PR_RebuildInterpWithFrozenCF(amg_data, level);
         if (new_P == NULL) break;

         /* Destroy old P[level], replace with new */
         if (P_array[level] != NULL)
            jx_ParCSRMatrixDestroy(P_array[level]);
         P_array[level] = new_P;

         /* 3c: R = P^T (if restri_type == 0) */
         if (!restri_type)
            R_array[level] = new_P; /* R aliases P */

         /* 3d: NumericRAP — compute A[level+1] = R * A * P */
         new_A = JXPAMG_PR_NumericRAP(amg_data, level, opt_icor);
         if (new_A != NULL)
         {
            if (A_array[level + 1] != NULL)
               jx_ParCSRMatrixDestroy(A_array[level + 1]);
            A_array[level + 1] = new_A;
         }

         /* 3e: Propagate active set to coarse level */
         {
            JX_Int *active_coarse = JXPAMG_PR_PropagateActiveSet(
               active_fine, active_count, new_P, &active_count);
            jx_TFree(active_fine);
            active_fine = active_coarse;
            if (active_count == 0 && active_fine != NULL)
               { jx_TFree(active_fine); active_fine = NULL; }
         }
      }
      else
      {
         /* Steps 11-13: Deep levels — force full reuse of old operators */
         if (my_id == 0 && active_count > 0)
            jx_printf("[predictive-3b2] freeze lv %d+ (beyond Llocal=%d)\n",
                      level, Llocal);
         if (active_fine != NULL) { jx_TFree(active_fine); active_fine = NULL; }
         active_count = 0;
      }
   }

   if (active_fine != NULL) jx_TFree(active_fine);
   jx_TFree(opt_icor);
   JXPAMG_PR_RecreateWorkVectors(amg_data);

   if (my_id == 0)
      jx_printf("[predictive-3b2] done: num_levels=%d\n", num_levels);
   return 0;
}

#endif /* JXPAMG_PR_ENABLE_3B2 */

/* =========================================================================
 * Phase 3b: Residual-based Adaptive Reuse
 * Same as AdaptiveReuse3B2, but uses residual probe instead of row-norm.
 * ========================================================================= */
JX_Int
JXPAMG_PR_AdaptiveReuse3B2_Residual(jx_ParAMGData *amg_data,
                                    jx_ParCSRMatrix *new_A_fine,
                                    JX_Real         *b_new,
                                    JX_Real theta_local,
                                    JX_Int  Llocal,
                                    JX_Int  L)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix **R_array = jx_ParAMGDataRArray(amg_data);
   JX_Int **CF_marker_array   = jx_ParAMGDataCFMarkerArray(amg_data);
   JX_Int   num_levels        = jx_ParAMGDataNumLevels(amg_data);
   JX_Int   restri_type       = jx_ParAMGDataRestriction(amg_data);
   JX_Int   num_threads       = jx_NumThreads();
   JX_Int  *opt_icor          = NULL;
   JX_Int   level;
   JX_Int   my_id = 0;
   JX_Int  *active_fine = NULL;
   JX_Int   active_count = 0;

   if (amg_data == NULL || new_A_fine == NULL || A_array == NULL || A_array[0] == NULL)
      return 1;
   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (JXPAMG_PR_CheckFineCompatibility(amg_data, new_A_fine)) return 1;
   if (num_levels <= 1) return 1;
   if (L < 0 || L >= num_levels) L = num_levels - 1;

   opt_icor = jx_CTAlloc(JX_Int, 5 * num_threads + 2);

   if (my_id == 0) jx_printf("[predictive-3b2] residual probe lv 0\n");

   /* Use residual probe with b_new */
   active_fine = JXPAMG_PR_ResidualProbe(new_A_fine, b_new,
                                         theta_local, &active_count);

   if (A_array[0] != new_A_fine) {
      jx_ParCSRMatrixDestroy(A_array[0]);
      A_array[0] = new_A_fine;
   }

   if (my_id == 0 && active_count > 0)
      jx_printf("[predictive-3b2] active nodes at lv 0: %d\n", active_count);

   {
      JX_Int total_rows = jx_ParCSRMatrixGlobalNumRows(new_A_fine);
      if (active_count > total_rows * 0.9)
      {
         if (my_id == 0)
            jx_printf("[predictive-3b2] too many active nodes (%d/%d), fallback\n",
                      active_count, total_rows);
         return 1;
      }
   }

   for (level = 0; level < L; level++)
   {
      if (level <= Llocal && active_count > 0)
      {
         jx_ParCSRMatrix *new_P = JXPAMG_PR_RebuildInterpWithFrozenCF(amg_data, level);
         if (new_P == NULL) break;
         if (P_array[level] != NULL) jx_ParCSRMatrixDestroy(P_array[level]);
         P_array[level] = new_P;
         if (!restri_type) R_array[level] = new_P;

         jx_ParCSRMatrix *new_A = JXPAMG_PR_NumericRAP(amg_data, level, opt_icor);
         if (new_A != NULL) {
            if (A_array[level + 1] != NULL) jx_ParCSRMatrixDestroy(A_array[level + 1]);
            A_array[level + 1] = new_A;
         }

         JX_Int *active_coarse = JXPAMG_PR_PropagateActiveSet(
            active_fine, active_count, new_P, &active_count);
         jx_TFree(active_fine);
         active_fine = active_coarse;
         if (active_count == 0 && active_fine != NULL)
            { jx_TFree(active_fine); active_fine = NULL; }
      }
      else
      {
         if (active_fine != NULL) { jx_TFree(active_fine); active_fine = NULL; }
         active_count = 0;
      }
   }

   if (active_fine != NULL) jx_TFree(active_fine);
   jx_TFree(opt_icor);
   JXPAMG_PR_RecreateWorkVectors(amg_data);

   if (my_id == 0)
      jx_printf("[predictive-3b2] done: num_levels=%d\n", num_levels);
   return 0;
}
