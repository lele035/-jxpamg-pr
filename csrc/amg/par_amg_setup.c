//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_amg_setup_predictive_reuse.c
 *  Date: 2011/09/03
 *
 *  Experimental copy for predictive setup reuse.
 *
 *  This file is intentionally kept outside jxpamg-0220/csrc so the normal
 *  wildcard build does not compile it. The original par_amg_setup.c remains
 *  untouched while we develop and inspect the reuse-from-level logic here.
 */ 

#include "jx_pamg.h"
#include "jx_euclid.h"

#include <math.h>

JX_Int jx_error_flag = 0;

JX_UInt jx_long_size_length_rap = 0;         /* Yue Xiaoqiang 2012/10/12 */
JX_UInt jx_long_size_length_interp = 0;      /* Yue Xiaoqiang 2012/10/12 */
JX_UInt jx_real_long_size_length_rap = 0;    /* Yue Xiaoqiang 2012/10/12 */
JX_UInt jx_real_long_size_length_interp = 0; /* Yue Xiaoqiang 2012/10/12 */

extern JX_Int *Pmarkers_global_rap;    /* Yue Xiaoqiang 2012/10/17 */
extern JX_Int *Pmarkers_global_interp; /* Yue Xiaoqiang 2012/10/17 */

#ifndef JXPAMG_PR_ENABLE_TAIL_REBUILD
#define JXPAMG_PR_ENABLE_TAIL_REBUILD 0
#endif

#if WITH_PARDISO
JX_Int jx_pardiso_factorize(jx_CSRMatrix *AA, Pardiso_data *pdata, JX_Int prtlvl);
#endif

static void
JXPAMG_PR_CountCF(JX_Int *CF_marker, JX_Int n, JX_Int *num_c, JX_Int *num_f)
{
   JX_Int i;

   *num_c = 0;
   *num_f = 0;
   if (CF_marker == NULL)
   {
      return;
   }

   for (i = 0; i < n; i++)
   {
      if (CF_marker[i] > 0)
      {
         (*num_c)++;
      }
      else
      {
         (*num_f)++;
      }
   }
}

static void
JXPAMG_PR_PrintHierarchyShape(const char *tag, jx_ParAMGData *amg_data)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix **R_array = jx_ParAMGDataRArray(amg_data);
   JX_Int **CF_marker_array = jx_ParAMGDataCFMarkerArray(amg_data);
   JX_Int restri_type = jx_ParAMGDataRestriction(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int level;
   JX_Int my_id = 0;

   if (A_array == NULL || A_array[0] == NULL)
   {
      jx_printf("[predictive-reuse] %s: A_array is not ready\n", tag);
      return;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (my_id != 0)
   {
      return;
   }

   jx_printf("[predictive-reuse] %s: num_levels=%d\n", tag, num_levels);
   for (level = 0; level < num_levels; level++)
   {
      jx_ParCSRMatrix *A = A_array[level];
      JX_Int local_rows = (A != NULL) ? jx_ParCSRMatrixNumRows(A) : -1;
      JX_Int num_c = 0;
      JX_Int num_f = 0;

      if (A == NULL)
      {
         jx_printf("  level %d: A=NULL\n", level);
         continue;
      }

      if (CF_marker_array != NULL && level < num_levels - 1)
      {
         JXPAMG_PR_CountCF(CF_marker_array[level], local_rows, &num_c, &num_f);
      }

      jx_printf("  A[%d]: global=%d x %d local_rows=%d nnz=%f",
                level,
                jx_ParCSRMatrixGlobalNumRows(A),
                jx_ParCSRMatrixGlobalNumCols(A),
                local_rows,
                jx_ParCSRMatrixDNumNonzeros(A));

      if (level < num_levels - 1)
      {
         jx_ParCSRMatrix *P = (P_array != NULL) ? P_array[level] : NULL;
         jx_ParCSRMatrix *R = (restri_type && R_array != NULL) ? R_array[level] : NULL;

         jx_printf(" CF(C=%d,F=%d)", num_c, num_f);
         if (P != NULL)
         {
            jx_printf(" P=%d x %d",
                      jx_ParCSRMatrixGlobalNumRows(P),
                      jx_ParCSRMatrixGlobalNumCols(P));
         }
         else
         {
            jx_printf(" P=NULL");
         }

         if (!restri_type)
         {
            jx_printf(" R=P^T");
         }
         else if (R != NULL)
         {
            jx_printf(" R=%d x %d",
                      jx_ParCSRMatrixGlobalNumRows(R),
                      jx_ParCSRMatrixGlobalNumCols(R));
         }
         else
         {
            jx_printf(" R=NULL");
         }
      }
      jx_printf("\n");
   }
}

static void
JXPAMG_PR_CheckLinkDimensions(const char *tag, jx_ParAMGData *amg_data)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix **R_array = jx_ParAMGDataRArray(amg_data);
   JX_Int restri_type = jx_ParAMGDataRestriction(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int level;
   JX_Int my_id = 0;

   if (A_array == NULL || A_array[0] == NULL)
   {
      return;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (my_id != 0)
   {
      return;
   }

   for (level = 0; level < num_levels - 1; level++)
   {
      jx_ParCSRMatrix *A = A_array[level];
      jx_ParCSRMatrix *Ac = A_array[level + 1];
      jx_ParCSRMatrix *P = (P_array != NULL) ? P_array[level] : NULL;
      jx_ParCSRMatrix *R = (restri_type && R_array != NULL) ? R_array[level] : NULL;

      if (A == NULL || Ac == NULL || P == NULL)
      {
         jx_printf("[predictive-reuse] %s: broken pointer at level %d\n", tag, level);
         continue;
      }

      if (jx_ParCSRMatrixGlobalNumRows(P) != jx_ParCSRMatrixGlobalNumRows(A) ||
          jx_ParCSRMatrixGlobalNumCols(P) != jx_ParCSRMatrixGlobalNumRows(Ac))
      {
         jx_printf("[predictive-reuse] %s: P dimension mismatch at level %d\n", tag, level);
      }

      if (R != NULL &&
          (jx_ParCSRMatrixGlobalNumRows(R) != jx_ParCSRMatrixGlobalNumRows(Ac) ||
           jx_ParCSRMatrixGlobalNumCols(R) != jx_ParCSRMatrixGlobalNumRows(A)))
      {
         jx_printf("[predictive-reuse] %s: R dimension mismatch at level %d\n", tag, level);
      }
   }
}




typedef struct
{
   jx_ParAMGData *amg_data;
   MPI_Comm comm;
   JX_Int k;
   JX_Int max_levels;
   JX_Int old_num_levels;
   JX_Int level;
   JX_Int local_size;
   JX_Int not_finished_coarsening;
   JX_Int num_functions;
   JX_Int coarsen_type;
   JX_Int interp_type;
   JX_Int restri_type;
   JX_Int measure_type;
   JX_Int debug_flag;
   JX_Int P_max_elmts;
   JX_Int agg_num_levels;
   JX_Int agg_interp_type;
   JX_Int agg_P_max_elmts;
   JX_Int num_paths;
   JX_Int rap2;
   JX_Int keepTranspose;
   JX_Int spmt_rap_type;
   JX_Int print_coarse_matrix;
   JX_Int coarse_threshold;
   JX_Int coarse_threshold_2;
   JX_Int measure_type_rlx;
   JX_Int number_syn_rlx;
   JX_Int coarse_size_last;
   JX_Int *grid_relax_type;
   JX_Int *AIR_maxsize_ls;
   JX_Int **CF_marker_array;
   JX_Int **dof_func_array;
   JX_Real strong_threshold;
   JX_Real max_row_sum;
   JX_Real trunc_factor;
   JX_Real S_commpkg_switch;
   JX_Real AIR_strong_th;
   JX_Real agg_trunc_factor;
   JX_Real coarse_ratio;
   JX_Real measure_threshold_rlx;
   jx_ParCSRMatrix **A_array;
   jx_ParCSRMatrix **P_array;
   jx_ParCSRMatrix **R_array;
   jx_ParVector **F_array;
   jx_ParVector **U_array;
   JX_Real **AI_measure_array;
} JXPAMG_PR_TailContext;

static JX_Int
JXPAMG_PR_InitTailContext(JXPAMG_PR_TailContext *ctx,
                          jx_ParAMGData *amg_data,
                          JX_Int k)
{
   if (ctx == NULL || amg_data == NULL)
   {
      return 1;
   }

   ctx->amg_data = amg_data;
   ctx->k = k;
   ctx->max_levels = jx_ParAMGDataMaxLevels(amg_data);
   ctx->old_num_levels = jx_ParAMGDataNumLevels(amg_data);
   ctx->level = k;
   ctx->not_finished_coarsening = 1;
   ctx->coarse_size_last = -1;
   ctx->coarse_threshold_2 = -1;

   ctx->A_array = jx_ParAMGDataAArray(amg_data);
   ctx->P_array = jx_ParAMGDataPArray(amg_data);
   ctx->R_array = jx_ParAMGDataRArray(amg_data);
   ctx->F_array = jx_ParAMGDataFArray(amg_data);
   ctx->U_array = jx_ParAMGDataUArray(amg_data);
   ctx->AI_measure_array = jx_ParAMGDataAIMeasureArray(amg_data);
   ctx->CF_marker_array = jx_ParAMGDataCFMarkerArray(amg_data);
   ctx->dof_func_array = jx_ParAMGDataDofFuncArray(amg_data);
   ctx->AIR_maxsize_ls = jx_ParAMGDataAIRMaxSizeLS(amg_data);

   if (ctx->A_array == NULL || k < 0 || k >= ctx->max_levels || ctx->A_array[k] == NULL)
   {
      return 1;
   }

   ctx->local_size = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(ctx->A_array[k]));
   ctx->comm = jx_ParCSRMatrixComm(ctx->A_array[k]);
   ctx->num_functions = jx_ParAMGDataNumFunctions(amg_data);
   ctx->coarsen_type = jx_ParAMGDataCoarsenType(amg_data);
   /* Re-derive coarsen_type for the portion of the hierarchy that was
    * already built: in the full setup loop, coarsen_type may switch to
    * CLJP (0) when coarse_size >= fine_size*coarse_ratio during previous
    * levels.  Replay that logic so the tail rebuild starts with the
    * same coarsen_type that the full setup loop would have at level k. */
   {
      JX_Real cr = jx_ParAMGDataCoarseRatio(amg_data);
      JX_Int l;
      for (l = 0; l < k; l++)
      {
         JX_Real fine_sz = (JX_Real)jx_ParCSRMatrixGlobalNumRows(ctx->A_array[l]);
         JX_Real coarse_sz = (JX_Real)jx_ParCSRMatrixGlobalNumRows(ctx->A_array[l+1]);
         if (ctx->coarsen_type > 0 && coarse_sz >= (JX_Int)(fine_sz * cr))
         {
            ctx->coarsen_type = 0;
         }
      }
   }
   ctx->interp_type = jx_ParAMGDataInterpType(amg_data);
   ctx->restri_type = jx_ParAMGDataRestriction(amg_data);
   ctx->measure_type = jx_ParAMGDataMeasureType(amg_data);
   ctx->debug_flag = jx_ParAMGDataDebugFlag(amg_data);
   ctx->P_max_elmts = jx_ParAMGDataPMaxElmts(amg_data);
   ctx->agg_num_levels = jx_ParAMGDataAggNumLevels(amg_data);
   ctx->agg_interp_type = jx_ParAMGDataAggInterpType(amg_data);
   ctx->agg_P_max_elmts = jx_ParAMGDataAggPMaxElmts(amg_data);
   ctx->num_paths = jx_ParAMGDataNumPaths(amg_data);
   ctx->rap2 = jx_ParAMGDataRAP2(amg_data);
   ctx->keepTranspose = jx_ParAMGDataKeepTranspose(amg_data);
   ctx->spmt_rap_type = jx_ParAMGDataSpMtRapType(amg_data);
   ctx->print_coarse_matrix = jx_ParAMGDataPrintCoarseSystem(amg_data);
   ctx->coarse_threshold = jx_ParAMGDataCoarseThreshold(amg_data);
   ctx->measure_type_rlx = jx_ParAMGDataMeasureTypeRlx(amg_data);
   ctx->number_syn_rlx = jx_ParAMGDataNumberSynRlx(amg_data);
   ctx->grid_relax_type = jx_ParAMGDataGridRelaxType(amg_data);

   ctx->strong_threshold = jx_ParAMGDataStrongThreshold(amg_data);
   ctx->max_row_sum = jx_ParAMGDataMaxRowSum(amg_data);
   ctx->trunc_factor = jx_ParAMGDataTruncFactor(amg_data);
   ctx->S_commpkg_switch = jx_ParAMGDataSCommPkgSwitch(amg_data);
   ctx->AIR_strong_th = jx_ParAMGDataAIRStrongTh(amg_data);
   ctx->agg_trunc_factor = jx_ParAMGDataAggTruncFactor(amg_data);
   ctx->coarse_ratio = jx_ParAMGDataCoarseRatio(amg_data);
   ctx->measure_threshold_rlx = jx_ParAMGDataMeasureThresholdRlx(amg_data);

   return 0;
}

static JX_Int
JXPAMG_PR_CheckTailContextReady(JXPAMG_PR_TailContext *ctx)
{
   JX_Int level;
   JX_Int my_id = 0;

   if (ctx == NULL || ctx->A_array == NULL || ctx->A_array[ctx->k] == NULL)
   {
      return 1;
   }

   jx_MPI_Comm_rank(ctx->comm, &my_id);

   if (ctx->k >= ctx->max_levels - 1)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] TailContext: k=%d is already at/near max level; rebuild should be finalization-only\n", ctx->k);
      }
      return 2;
   }

   if (ctx->P_array == NULL || ctx->CF_marker_array == NULL ||
       ctx->dof_func_array == NULL || ctx->F_array == NULL || ctx->U_array == NULL)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] TailContext rejected: required hierarchy/vector arrays are NULL\n");
      }
      return 1;
   }

   if (ctx->restri_type && ctx->R_array == NULL)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] TailContext rejected: explicit R requested but R_array is NULL\n");
      }
      return 1;
   }

   for (level = ctx->k + 1; level < ctx->old_num_levels; level++)
   {
      if (ctx->A_array[level] != NULL)
      {
         if (my_id == 0)
         {
            jx_printf("[predictive-reuse] TailContext rejected: A[%d] is not freed before rebuild\n", level);
         }
         return 1;
      }
   }

   for (level = ctx->k; level < ctx->old_num_levels - 1; level++)
   {
      if (ctx->P_array[level] != NULL ||
          ctx->CF_marker_array[level] != NULL ||
          (ctx->restri_type && ctx->R_array != NULL && ctx->R_array[level] != NULL))
      {
         if (my_id == 0)
         {
            jx_printf("[predictive-reuse] TailContext rejected: tail P/R/CF level %d is not freed\n", level);
         }
         return 1;
      }
   }

   for (level = (ctx->k > 0 ? ctx->k : ctx->k + 1); level < ctx->old_num_levels; level++)
   {
      if (ctx->F_array[level] != NULL || ctx->U_array[level] != NULL)
      {
         if (my_id == 0)
         {
            jx_printf("[predictive-reuse] TailContext rejected: F/U level %d is not freed before rebuild\n", level);
         }
         return 1;
      }
   }

   if (jx_ParAMGDataL1Norms(ctx->amg_data) != NULL ||
       jx_ParAMGDataResidual(ctx->amg_data) != NULL ||
       jx_ParAMGDataSmoother(ctx->amg_data) != NULL)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] TailContext rejected: old finalization state is not reset\n");
      }
      return 1;
   }

   if (my_id == 0)
   {
      jx_printf("[predictive-reuse] TailContext ready: k=%d old_num_levels=%d max_levels=%d\n",
                ctx->k, ctx->old_num_levels, ctx->max_levels);
   }

   return 0;
}

static JX_Int
JXPAMG_PR_CheckFineCompatibility(jx_ParAMGData *amg_data, jx_ParCSRMatrix *new_A_fine)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);

   if (new_A_fine == NULL || A_array == NULL || A_array[0] == NULL)
   {
      return 1;
   }

   if (jx_ParCSRMatrixGlobalNumRows(new_A_fine) != jx_ParCSRMatrixGlobalNumRows(A_array[0]) ||
       jx_ParCSRMatrixGlobalNumCols(new_A_fine) != jx_ParCSRMatrixGlobalNumCols(A_array[0]) ||
       jx_ParCSRMatrixNumRows(new_A_fine) != jx_ParCSRMatrixNumRows(A_array[0]))
   {
      return 1;
   }

   return 0;
}

static jx_ParCSRMatrix *
JXPAMG_PR_BuildReuseCoarseOperator(jx_ParAMGData *amg_data,
                                   JX_Int level,
                                   JX_Int *opt_icor)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix **R_array = jx_ParAMGDataRArray(amg_data);
   JX_Int restri_type = jx_ParAMGDataRestriction(amg_data);
   JX_Int rap2 = jx_ParAMGDataRAP2(amg_data);
   JX_Int spmt_rap_type = jx_ParAMGDataSpMtRapType(amg_data);
   JX_Int keepTranspose = jx_ParAMGDataKeepTranspose(amg_data);
   jx_ParCSRMatrix *A_H = NULL;
   JX_Int num_procs = 1;

   jx_MPI_Comm_size(jx_ParCSRMatrixComm(A_array[level]), &num_procs);

   if (restri_type)
   {
      jx_ParCSRMatrix *Q = jx_ParMatmul(A_array[level], P_array[level]);
      A_H = jx_ParMatmul(R_array[level], Q);
      jx_ParCSRMatrixOwnsRowStarts(A_H) = 1;
      jx_ParCSRMatrixOwnsColStarts(A_H) = 0;
      jx_ParCSRMatrixOwnsColStarts(P_array[level]) = 0;
      jx_ParCSRMatrixOwnsRowStarts(R_array[level]) = 0;
      if (num_procs > 1)
      {
         jx_MatvecCommPkgCreate(A_H);
      }
      jx_ParCSRMatrixDestroy(Q);
   }
   else if (rap2)
   {
      jx_ParCSRMatrix *Q = jx_ParMatmul(A_array[level], P_array[level]);
      A_H = jx_ParTMatmul(P_array[level], Q);
      jx_ParCSRMatrixOwnsRowStarts(A_H) = 1;
      jx_ParCSRMatrixOwnsColStarts(A_H) = 0;
      jx_ParCSRMatrixOwnsColStarts(P_array[level]) = 0;
      if (num_procs > 1)
      {
         jx_MatvecCommPkgCreate(A_H);
      }
      jx_ParCSRMatrixDestroy(Q);
   }
   else if (spmt_rap_type == 1)
   {
      jx_PAMGBuildCoarseOperatorKT(P_array[level], A_array[level], P_array[level], keepTranspose, &A_H);
   }
   else if (spmt_rap_type == 2)
   {
      jx_PAMGBuildCoarseOperatorOMP(P_array[level], A_array[level], P_array[level], &A_H, opt_icor);
   }

   if (A_H != NULL)
   {
      jx_ParCSRMatrixSetNumNonzeros(A_H);
      jx_ParCSRMatrixSetDNumNonzeros(A_H);
   }

   return A_H;
}

static jx_ParCSRMatrix *
JXPAMG_PR_DeepCopyParCSR(jx_ParCSRMatrix *src)
{
   jx_ParCSRMatrix *dst;
   jx_CSRMatrix *src_diag, *src_offd, *dst_diag, *dst_offd;
   JX_Int *src_cmap, *dst_cmap;
   JX_Int ncols_offd, nnz_diag, nnz_offd, nrows;

   if (src == NULL) return NULL;

   dst = jx_CTAlloc(jx_ParCSRMatrix, 1);
   src_diag = jx_ParCSRMatrixDiag(src);
   src_offd = jx_ParCSRMatrixOffd(src);
   nrows    = jx_CSRMatrixNumRows(src_diag);
   nnz_diag = jx_CSRMatrixNumNonzeros(src_diag);
   nnz_offd = jx_CSRMatrixNumNonzeros(src_offd);
   ncols_offd = jx_CSRMatrixNumCols(src_offd);

   /* global metadata */
   jx_ParCSRMatrixComm(dst)          = jx_ParCSRMatrixComm(src);
   jx_ParCSRMatrixGlobalNumRows(dst) = jx_ParCSRMatrixGlobalNumRows(src);
   jx_ParCSRMatrixGlobalNumCols(dst) = jx_ParCSRMatrixGlobalNumCols(src);
   jx_ParCSRMatrixFirstRowIndex(dst) = jx_ParCSRMatrixFirstRowIndex(src);
   jx_ParCSRMatrixFirstColDiag(dst)  = jx_ParCSRMatrixFirstColDiag(src);
   jx_ParCSRMatrixLastRowIndex(dst)  = jx_ParCSRMatrixLastRowIndex(src);
   jx_ParCSRMatrixLastColDiag(dst)   = jx_ParCSRMatrixLastColDiag(src);
   jx_ParCSRMatrixNumNonzeros(dst)   = jx_ParCSRMatrixNumNonzeros(src);
   jx_ParCSRMatrixDNumNonzeros(dst)  = jx_ParCSRMatrixDNumNonzeros(src);

   /* diag CSR */
   dst_diag = jx_CSRMatrixCreate(nrows, nrows, nnz_diag);
   jx_CSRMatrixInitialize(dst_diag);
   {
      JX_Int *si = jx_CSRMatrixI(src_diag), *di = jx_CSRMatrixI(dst_diag);
      JX_Int *sj = jx_CSRMatrixJ(src_diag), *dj = jx_CSRMatrixJ(dst_diag);
      JX_Real *sd = jx_CSRMatrixData(src_diag), *dd = jx_CSRMatrixData(dst_diag);
      JX_Int n = nrows + 1;
      for (JX_Int k = 0; k < n; k++) di[k] = si[k];
      for (JX_Int k = 0; k < nnz_diag; k++) { dj[k] = sj[k]; dd[k] = sd[k]; }
   }
   jx_ParCSRMatrixDiag(dst) = dst_diag;

   /* offd CSR */
   dst_offd = jx_CSRMatrixCreate(nrows, ncols_offd, nnz_offd);
   jx_CSRMatrixInitialize(dst_offd);
   {
      JX_Int *si = jx_CSRMatrixI(src_offd), *di = jx_CSRMatrixI(dst_offd);
      JX_Int *sj = jx_CSRMatrixJ(src_offd), *dj = jx_CSRMatrixJ(dst_offd);
      JX_Real *sd = jx_CSRMatrixData(src_offd), *dd = jx_CSRMatrixData(dst_offd);
      JX_Int n = nrows + 1;
      for (JX_Int k = 0; k < n; k++) di[k] = si[k];
      for (JX_Int k = 0; k < nnz_offd; k++) { dj[k] = sj[k]; dd[k] = sd[k]; }
   }
   jx_ParCSRMatrixOffd(dst) = dst_offd;

   /* col_map_offd */
   if (ncols_offd > 0) {
      src_cmap = jx_ParCSRMatrixColMapOffd(src);
      dst_cmap = jx_CTAlloc(JX_Int, ncols_offd);
      for (JX_Int k = 0; k < ncols_offd; k++) dst_cmap[k] = src_cmap[k];
      jx_ParCSRMatrixColMapOffd(dst) = dst_cmap;
   }

   /* row_starts / col_starts */
   if (jx_ParCSRMatrixRowStarts(src)) {
      JX_Int *rs, *cs;
      JX_Int np;
      jx_MPI_Comm_size(jx_ParCSRMatrixComm(src), &np);
      rs = jx_CTAlloc(JX_Int, np + 1);
      cs = jx_CTAlloc(JX_Int, np + 1);
      for (JX_Int k = 0; k <= np; k++) {
         rs[k] = jx_ParCSRMatrixRowStarts(src)[k];
         cs[k] = jx_ParCSRMatrixColStarts(src)[k];
      }
      jx_ParCSRMatrixRowStarts(dst) = rs;
      jx_ParCSRMatrixColStarts(dst) = cs;
      jx_ParCSRMatrixOwnsRowStarts(dst) = 1;
      jx_ParCSRMatrixOwnsColStarts(dst) = 1;
   }

   jx_ParCSRMatrixOwnsData(dst) = 1;

   /* Communicator package: not created for probe copies.
    * KT may fail without it for np>1, which is handled as
    * prediction failure -> fallback to full setup. */
   jx_ParCSRMatrixCommPkg(dst) = NULL;

   return dst;
}

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
      status = (status != 0) ? status : 1;
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

static JX_Real JXPAMG_PR_DiagRelativeDiff(jx_ParCSRMatrix *, jx_ParCSRMatrix *, JX_Real);
static JX_Real JXPAMG_PR_LocalDiagAbsDiff(jx_ParCSRMatrix *, jx_ParCSRMatrix *, JX_Real *);

JX_Int
JXPAMG_PR_PredictCutReadOnly(jx_ParAMGData *amg_data,
                              jx_ParCSRMatrix *new_A_fine,
                              JX_Real theta_pred,
                              JX_Int Lreuse,
                              JX_Real delta,
                              JX_Int *cut_k)
{
   return JXPAMG_PR_PredictCutReadOnlyWithCandidate(amg_data, new_A_fine,
                                                     theta_pred, Lreuse, delta,
                                                     cut_k, NULL);
}

JX_Int
JXPAMG_PR_PredictCutReadOnlyWithCandidate(jx_ParAMGData *amg_data,
                                           jx_ParCSRMatrix *new_A_fine,
                                           JX_Real theta_pred,
                                           JX_Int Lreuse,
                                           JX_Real delta,
                                           JX_Int *cut_k,
                                           jx_ParCSRMatrix **candidate_A_cut)
{
   jx_ParCSRMatrix **A_array;
   jx_ParCSRMatrix **P_array;
   JX_Int num_levels;
   JX_Int max_probe;
   JX_Int level;
   JX_Int my_id = 0;
   JX_Int k;
   JX_Int status = 0;
   jx_ParCSRMatrix *cur_A = NULL;
   MPI_Comm comm;

   if (amg_data == NULL || new_A_fine == NULL || cut_k == NULL) return 1;

   A_array = jx_ParAMGDataAArray(amg_data);
   P_array = jx_ParAMGDataPArray(amg_data);
   num_levels = jx_ParAMGDataNumLevels(amg_data);
   if (A_array == NULL || A_array[0] == NULL || P_array == NULL) return 1;
   if (JXPAMG_PR_CheckFineCompatibility(amg_data, new_A_fine)) return 1;
   if (num_levels <= 1) { *cut_k = num_levels; return 0; }

   comm = jx_ParCSRMatrixComm(A_array[0]);
   jx_MPI_Comm_rank(comm, &my_id);

   max_probe = num_levels - 1;
   if (Lreuse >= 0 && Lreuse < max_probe) max_probe = Lreuse;
   k = num_levels;

   /* Probe level 0: compare new_A_fine with existing A_array[0] */
   {
      JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(new_A_fine, A_array[0], delta);
      if (my_id == 0) jx_printf("[predictive-reuse] probe lv 0: sigma=%e\n", sigma);
      if (sigma > theta_pred)
      {
         k = 0;
         goto prediction_consensus;
      }
   }

   /* Probe coarser levels with deep copies + status-returning KT helper */
   cur_A = new_A_fine;
   for (level = 0; level < max_probe; level++)
   {
      jx_ParCSRMatrix *A_H = NULL;
      JX_Int local_status, global_status;

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

      {
         JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(A_H, A_array[level + 1], delta);
         if (my_id == 0) jx_printf("[predictive-reuse] probe lv %d: sigma=%e\n", level + 1, sigma);

         if (sigma > theta_pred)
         {
            k = level + 1;
            if (candidate_A_cut != NULL)
            {
               *candidate_A_cut = A_H;
               A_H = NULL;
               if (my_id == 0)
                  jx_printf("[predictive-reuse] captured candidate A[%d] for tail rebuild\n", k);
            }
            else
            {
               jx_ParCSRMatrixDestroy(A_H);
            }
            goto prediction_consensus;
         }
      }

      if (cur_A != new_A_fine) jx_ParCSRMatrixDestroy(cur_A);
      cur_A = A_H;
   }

prediction_consensus:
   /* Ensure all ranks agree on cut_k */
   {
      JX_Int k_min, k_max;
      jx_MPI_Allreduce(&k, &k_min, 1, JX_MPI_INT, MPI_MIN, comm);
      jx_MPI_Allreduce(&k, &k_max, 1, JX_MPI_INT, MPI_MAX, comm);
      if (k_min != k_max)
      {
         if (my_id == 0)
            jx_printf("[predictive-reuse] cut_k MPI disagreement (min=%d max=%d)\n",
                      k_min, k_max);
         status = 1;
         goto prediction_cleanup;
      }
      k = k_min;
   }

   if (my_id == 0)
   {
      if (k == num_levels)
         jx_printf("[predictive-reuse] predicted full reuse (read-only)\n");
      else
         jx_printf("[predictive-reuse] predicted cut k=%d (read-only)\n", k);
   }

prediction_cleanup:
   if (cur_A != NULL && cur_A != new_A_fine)
      jx_ParCSRMatrixDestroy(cur_A);
   if (status != 0 && candidate_A_cut != NULL && *candidate_A_cut != NULL)
   {
      jx_ParCSRMatrixDestroy(*candidate_A_cut);
      *candidate_A_cut = NULL;
   }
   if (status == 0)
      *cut_k = k;
   return status;
}

JX_Int
JXPAMG_PR_FullSetupOnNewMatrix(jx_ParAMGData *amg_data,
                               jx_ParCSRMatrix *new_A)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);

   if (A_array != NULL && A_array[0] != NULL && A_array[0] != new_A)
   {
      jx_ParCSRMatrixDestroy(A_array[0]);
      A_array[0] = NULL;
   }

   /* Destroy old shared state safely before calling JX_PAMGSetup. */
   {
      JX_Solver *smoother = jx_ParAMGDataSmoother(amg_data);
      jx_ParVector **F_array = jx_ParAMGDataFArray(amg_data);
      jx_ParVector **U_array = jx_ParAMGDataUArray(amg_data);
      jx_ParVector *Vtemp   = jx_ParAMGDataVtemp(amg_data);
      JX_Int num_levels     = jx_ParAMGDataNumLevels(amg_data);
      JX_Int smooth_type    = jx_ParAMGDataSmoothType(amg_data);
      JX_Int smooth_nlvl    = jx_ParAMGDataSmoothNumLevels(amg_data);
      JX_Int i;

      /* Smoother */
      if (smoother != NULL) {
         if (smooth_type == 9 || smooth_type == 19) {
            for (i = 0; i < smooth_nlvl; i++)
               if (smoother[i] != NULL) {
                  JX_EuclidDestroy(smoother[i]);
                  smoother[i] = NULL;
               }
         }
         jx_TFree(smoother);
         jx_ParAMGDataSmoother(amg_data) = NULL;
      }

      /* F/U coarse-level vectors */
      if (F_array != NULL) {
         for (i = 1; i < num_levels; i++)
            if (F_array[i] != NULL)
               jx_ParVectorDestroy(F_array[i]);
         jx_TFree(F_array);
         jx_ParAMGDataFArray(amg_data) = NULL;
      }
      if (U_array != NULL) {
         for (i = 1; i < num_levels; i++)
            if (U_array[i] != NULL)
               jx_ParVectorDestroy(U_array[i]);
         jx_TFree(U_array);
         jx_ParAMGDataUArray(amg_data) = NULL;
      }

      /* Vtemp */
      if (Vtemp != NULL) {
         jx_ParVectorDestroy(Vtemp);
         jx_ParAMGDataVtemp(amg_data) = NULL;
      }

      /* l1_norms — array of per-level arrays, free inner first */
      {
         JX_Real **ln = jx_ParAMGDataL1Norms(amg_data);
         if (ln != NULL) {
            JX_Int nl = jx_ParAMGDataNumLevels(amg_data);
            JX_Int i;
            for (i = 0; i < nl; i++)
               if (ln[i] != NULL) { jx_TFree(ln[i]); ln[i] = NULL; }
            jx_TFree(ln);
            jx_ParAMGDataL1Norms(amg_data) = NULL;
         }
      }
      if (jx_ParAMGDataResidual(amg_data) != NULL) {
         jx_ParVectorDestroy(jx_ParAMGDataResidual(amg_data));
         jx_ParAMGDataResidual(amg_data) = NULL;
      }
   }

   return JX_PAMGSetup((JX_Solver)amg_data, (JX_ParCSRMatrix)new_A);
}

static void
JXPAMG_PR_RecreateWorkVectors(jx_ParAMGData *amg_data)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParVector **F_array = jx_ParAMGDataFArray(amg_data);
   jx_ParVector **U_array = jx_ParAMGDataUArray(amg_data);
   jx_ParVector *Vtemp = jx_ParAMGDataVtemp(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int level;

   if (A_array == NULL || A_array[0] == NULL)
   {
      return;
   }

   if (Vtemp != NULL)
   {
      jx_ParVectorDestroy(Vtemp);
   }
   Vtemp = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[0]),
                              jx_ParCSRMatrixGlobalNumRows(A_array[0]),
                              jx_ParCSRMatrixRowStarts(A_array[0]));
   jx_ParVectorInitialize(Vtemp);
   jx_ParVectorSetPartitioningOwner(Vtemp, 0);
   jx_ParAMGDataVtemp(amg_data) = Vtemp;

   for (level = 1; level < num_levels; level++)
   {
      if (F_array != NULL)
      {
         if (F_array[level] != NULL)
         {
            jx_ParVectorDestroy(F_array[level]);
         }
         F_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                             jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                             jx_ParCSRMatrixRowStarts(A_array[level]));
         jx_ParVectorInitialize(F_array[level]);
         jx_ParVectorSetPartitioningOwner(F_array[level], 0);
      }

      if (U_array != NULL)
      {
         if (U_array[level] != NULL)
         {
            jx_ParVectorDestroy(U_array[level]);
         }
         U_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                             jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                             jx_ParCSRMatrixRowStarts(A_array[level]));
         jx_ParVectorInitialize(U_array[level]);
         jx_ParVectorSetPartitioningOwner(U_array[level], 0);
      }
   }
}


static JX_Int
JXPAMG_PR_ResetTailFinalizationState(jx_ParAMGData *amg_data)
{
   JX_Real **l1_norms;
   jx_ParVector *residual;
   JX_Solver *smoother;
   JX_Int smooth_type;
   JX_Int smooth_num_levels;
   JX_Int num_levels;
   JX_Int i;
   JX_Int my_id = 0;
   jx_ParCSRMatrix **A_array;

   if (amg_data == NULL)
   {
      return 1;
   }

   A_array = jx_ParAMGDataAArray(amg_data);
   if (A_array != NULL && A_array[0] != NULL)
   {
      jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   }

   num_levels = jx_ParAMGDataNumLevels(amg_data);
   l1_norms = jx_ParAMGDataL1Norms(amg_data);
   if (l1_norms != NULL)
   {
      for (i = 0; i < num_levels; i++)
      {
         if (l1_norms[i] != NULL)
         {
            jx_TFree(l1_norms[i]);
            l1_norms[i] = NULL;
         }
      }
      jx_TFree(l1_norms);
      jx_ParAMGDataL1Norms(amg_data) = NULL;
   }

   residual = jx_ParAMGDataResidual(amg_data);
   if (residual != NULL)
   {
      jx_ParVectorDestroy(residual);
      jx_ParAMGDataResidual(amg_data) = NULL;
   }

   smoother = jx_ParAMGDataSmoother(amg_data);
   if (smoother != NULL)
   {
      smooth_type = jx_ParAMGDataSmoothType(amg_data);
      smooth_num_levels = jx_ParAMGDataSmoothNumLevels(amg_data);

      if (smooth_type == 9 || smooth_type == 19)
      {
         for (i = 0; i < smooth_num_levels; i++)
         {
            if (smoother[i] != NULL)
            {
               JX_EuclidDestroy(smoother[i]);
               smoother[i] = NULL;
            }
         }
      }
      jx_TFree(smoother);
      jx_ParAMGDataSmoother(amg_data) = NULL;
   }

   return 0;
}


static JX_Int
JXPAMG_PR_FinalizeTailRebuildCore(jx_ParAMGData *amg_data, JX_Int level)
{
   MPI_Comm comm;
   jx_ParCSRMatrix **A_array;
   jx_ParVector **F_array;
   jx_ParVector **U_array;
   jx_ParVector *Residual_array;
   JX_Solver *smoother;
   JX_Int *grid_relax_type;
   JX_Int num_levels;
   JX_Int smooth_num_levels;
   JX_Int smooth_type;
   JX_Int amg_logging;
   JX_Int eu_level;
   JX_Int eu_bj;
   JX_Real eu_sparse_A;
   char *euclidfile;
   JX_Int j;
   JX_Int my_id = 0;

   if (amg_data == NULL || level < 0)
   {
      return 1;
   }

   A_array = jx_ParAMGDataAArray(amg_data);
   F_array = jx_ParAMGDataFArray(amg_data);
   U_array = jx_ParAMGDataUArray(amg_data);
   if (A_array == NULL || A_array[level] == NULL || F_array == NULL || U_array == NULL)
   {
      return 1;
   }

   comm = jx_ParCSRMatrixComm(A_array[level]);
   jx_MPI_Comm_rank(comm, &my_id);

   if (level > 0)
   {
      F_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                          jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                          jx_ParCSRMatrixRowStarts(A_array[level]));
      jx_ParVectorInitialize(F_array[level]);
      jx_ParVectorSetPartitioningOwner(F_array[level], 0);

      U_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                          jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                          jx_ParCSRMatrixRowStarts(A_array[level]));
      jx_ParVectorInitialize(U_array[level]);
      jx_ParVectorSetPartitioningOwner(U_array[level], 0);
   }

   grid_relax_type = jx_ParAMGDataGridRelaxType(amg_data);
#if WITH_PARDISO
   if (grid_relax_type != NULL && grid_relax_type[3] == PARDISO_SOLVER)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] tail finalization rejected: PARDISO coarse solver state is not audited yet\n");
      }
      return 1;
   }
#endif

   num_levels = level + 1;
   jx_ParAMGDataNumLevels(amg_data) = num_levels;
   if (jx_ParAMGDataSmoothNumLevels(amg_data) > level)
   {
      jx_ParAMGDataSmoothNumLevels(amg_data) = level;
   }

   smooth_type = jx_ParAMGDataSmoothType(amg_data);
   smooth_num_levels = jx_ParAMGDataSmoothNumLevels(amg_data);
   if (smooth_num_levels > 0 && jx_ParAMGDataSmoother(amg_data) == NULL)
   {
      smoother = jx_CTAlloc(JX_Solver, smooth_num_levels);
      jx_ParAMGDataSmoother(amg_data) = smoother;
   }
   if (smooth_num_levels > 0 && (smooth_type == 9 || smooth_type == 19))
   {
      smoother = jx_ParAMGDataSmoother(amg_data);
      euclidfile = jx_ParAMGDataEuclidFile(amg_data);
      eu_level = jx_ParAMGDataEuLevel(amg_data);
      eu_bj = jx_ParAMGDataEuBJ(amg_data);
      eu_sparse_A = jx_ParAMGDataEuSparseA(amg_data);

      for (j = 0; j < smooth_num_levels; j++)
      {
         JX_EuclidCreate(jx_ParCSRMatrixComm(A_array[j]), &smoother[j]);
         if (euclidfile)
         {
            JX_EuclidSetParamsFromFile(smoother[j], euclidfile);
         }
         JX_EuclidSetLevel(smoother[j], eu_level);
         if (eu_bj)
         {
            JX_EuclidSetBJ(smoother[j], eu_bj);
         }
         if (eu_sparse_A)
         {
            JX_EuclidSetSparseA(smoother[j], eu_sparse_A);
         }
         JX_EuclidSetup(smoother[j], (JX_ParCSRMatrix)A_array[j]);
      }
   }

   amg_logging = jx_ParAMGDataLogging(amg_data);
   if (jx_ParAMGDataResidual(amg_data) != NULL)
   {
      jx_ParVectorDestroy(jx_ParAMGDataResidual(amg_data));
      jx_ParAMGDataResidual(amg_data) = NULL;
   }

   if (amg_logging > 1)
   {
      Residual_array = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[0]),
                                          jx_ParCSRMatrixGlobalNumRows(A_array[0]),
                                          jx_ParCSRMatrixRowStarts(A_array[0]));
      jx_ParVectorInitialize(Residual_array);
      jx_ParVectorSetPartitioningOwner(Residual_array, 0);
      jx_ParAMGDataResidual(amg_data) = Residual_array;
   }

   return 0;
}

static JX_Int
JXPAMG_PR_FullReuseRAPUpdate(jx_ParAMGData *amg_data, jx_ParCSRMatrix *new_A_fine)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int num_threads = jx_NumThreads();
   JX_Int *opt_icor = NULL;
   JX_Int level;
   JX_Int num_procs = 1;

   if (JXPAMG_PR_CheckFineCompatibility(amg_data, new_A_fine))
   {
      jx_printf("[predictive-reuse] full-reuse RAP rejected: incompatible fine matrix\n");
      return 1;
   }

   if (num_levels <= 0 || P_array == NULL)
   {
      jx_printf("[predictive-reuse] full-reuse RAP rejected: hierarchy is not ready\n");
      return 1;
   }

   opt_icor = jx_CTAlloc(JX_Int, 5 * num_threads + 2);

   if (A_array[0] != new_A_fine)
   {
      /* After this point the experimental hierarchy owns new_A_fine. */
      jx_ParCSRMatrixDestroy(A_array[0]);
      A_array[0] = new_A_fine;
   }

   jx_MPI_Comm_size(jx_ParCSRMatrixComm(A_array[0]), &num_procs);
   if (num_procs > 1 && jx_ParCSRMatrixCommPkg(A_array[0]) == NULL)
   {
      jx_MatvecCommPkgCreate(A_array[0]);
   }

   for (level = 0; level < num_levels - 1; level++)
   {
      jx_ParCSRMatrix *A_H = JXPAMG_PR_BuildReuseCoarseOperator(amg_data, level, opt_icor);
      if (A_H == NULL)
      {
         jx_TFree(opt_icor);
         jx_printf("[predictive-reuse] full-reuse RAP failed at level %d\n", level);
         return 1;
      }

      if (A_array[level + 1] != NULL)
      {
         jx_ParCSRMatrixDestroy(A_array[level + 1]);
      }
      A_array[level + 1] = A_H;
   }

   JXPAMG_PR_PrintHierarchyShape("after full-reuse RAP update", amg_data);
   JXPAMG_PR_CheckLinkDimensions("after full-reuse RAP update", amg_data);

   jx_TFree(opt_icor);
   return 0;
}


static JX_Real
JXPAMG_PR_LocalDiagAbsDiff(jx_ParCSRMatrix *A,
                           jx_ParCSRMatrix *B,
                           JX_Real *base_diag_l1)
{
   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix *B_diag = jx_ParCSRMatrixDiag(B);
   JX_Int *A_i = jx_CSRMatrixI(A_diag);
   JX_Int *A_j = jx_CSRMatrixJ(A_diag);
   JX_Real *A_data = jx_CSRMatrixData(A_diag);
   JX_Int *B_i = jx_CSRMatrixI(B_diag);
   JX_Int *B_j = jx_CSRMatrixJ(B_diag);
   JX_Real *B_data = jx_CSRMatrixData(B_diag);
   JX_Int n = jx_CSRMatrixNumRows(A_diag);
   JX_Real diff_l1 = 0.0;
   JX_Int row;

   *base_diag_l1 = 0.0;

   for (row = 0; row < n; row++)
   {
      JX_Int jj;
      JX_Real a_diag = 0.0;
      JX_Real b_diag = 0.0;

      for (jj = A_i[row]; jj < A_i[row + 1]; jj++)
      {
         if (A_j[jj] == row)
         {
            a_diag = A_data[jj];
            break;
         }
      }

      for (jj = B_i[row]; jj < B_i[row + 1]; jj++)
      {
         if (B_j[jj] == row)
         {
            b_diag = B_data[jj];
            break;
         }
      }

      diff_l1 += fabs(a_diag - b_diag);
      *base_diag_l1 += fabs(b_diag);
   }

   return diff_l1;
}

static JX_Real
JXPAMG_PR_DiagRelativeDiff(jx_ParCSRMatrix *current_A,
                           jx_ParCSRMatrix *base_A,
                           JX_Real delta)
{
   MPI_Comm comm;
   JX_Real local_base_l1 = 0.0;
   JX_Real local_diff_l1;
   JX_Real global_base_l1 = 0.0;
   JX_Real global_diff_l1 = 0.0;

   if (current_A == NULL || base_A == NULL)
   {
      return 1.0e100;
   }

   comm = jx_ParCSRMatrixComm(current_A);

   if (jx_ParCSRMatrixGlobalNumRows(current_A) != jx_ParCSRMatrixGlobalNumRows(base_A) ||
       jx_ParCSRMatrixGlobalNumCols(current_A) != jx_ParCSRMatrixGlobalNumCols(base_A) ||
       jx_ParCSRMatrixNumRows(current_A) != jx_ParCSRMatrixNumRows(base_A))
   {
      return 1.0e100;
   }

   local_diff_l1 = JXPAMG_PR_LocalDiagAbsDiff(current_A, base_A, &local_base_l1);
   jx_MPI_Allreduce(&local_base_l1, &global_base_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
   jx_MPI_Allreduce(&local_diff_l1, &global_diff_l1, 1, JX_MPI_REAL, MPI_SUM, comm);

   return global_diff_l1 / (global_base_l1 + delta);
}

static JX_Int
JXPAMG_PR_PrintPredictiveProbe(jx_ParAMGData *current_amg,
                               jx_ParAMGData *base_amg,
                               JX_Real theta_pred,
                               JX_Int Lreuse,
                               JX_Real delta)
{
   jx_ParCSRMatrix **A_current = jx_ParAMGDataAArray(current_amg);
   jx_ParCSRMatrix **A_base = jx_ParAMGDataAArray(base_amg);
   JX_Int current_levels = jx_ParAMGDataNumLevels(current_amg);
   JX_Int base_levels = jx_ParAMGDataNumLevels(base_amg);
   JX_Int max_probe_level;
   JX_Int predicted_k;
   JX_Int level;
   JX_Int my_id = 0;

   if (A_current == NULL || A_base == NULL || A_current[0] == NULL)
   {
      return 0;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_current[0]), &my_id);

   max_probe_level = current_levels < base_levels ? current_levels : base_levels;
   max_probe_level -= 1;
   if (Lreuse >= 0 && Lreuse < max_probe_level)
   {
      max_probe_level = Lreuse;
   }
   predicted_k = current_levels;

   for (level = 0; level <= max_probe_level; level++)
   {
      JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(A_current[level], A_base[level], delta);

      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] probe level %d: sigma=%e theta=%e\n",
                   level, sigma, theta_pred);
      }

      if (sigma > theta_pred)
      {
         predicted_k = level;
         break;
      }
   }

   if (my_id == 0)
   {
      if (predicted_k < current_levels)
      {
         jx_printf("[predictive-reuse] predicted cut level k=%d\n", predicted_k);
      }
      else
      {
         jx_printf("[predictive-reuse] predicted full reuse through probed levels\n");
      }
   }

   return predicted_k;
}


static void
JXPAMG_PR_PrintSetupFromLevelPlan(jx_ParAMGData *amg_data, JX_Int k)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int my_id = 0;

   if (A_array == NULL || A_array[0] == NULL)
   {
      return;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (my_id != 0)
   {
      return;
   }

   jx_printf("[predictive-reuse] SetupFromLevel dry-run: k=%d old_num_levels=%d\n",
             k, num_levels);

   if (k < 0)
   {
      jx_printf("[predictive-reuse]   invalid k; production path must fall back to full setup\n");
      return;
   }

   if (k >= num_levels)
   {
      jx_printf("[predictive-reuse]   full reuse: keep all A/P/R/CF levels, no tail rebuild\n");
      return;
   }

   jx_printf("[predictive-reuse]   keep A[0..%d]\n", k);
   if (k > 0)
   {
      jx_printf("[predictive-reuse]   keep P/R/CF[0..%d]\n", k - 1);
   }
   else
   {
      jx_printf("[predictive-reuse]   keep no P/R/CF before the cut\n");
   }

   if (k + 1 <= num_levels - 1)
   {
      jx_printf("[predictive-reuse]   would free A[%d..%d]\n", k + 1, num_levels - 1);
   }
   else
   {
      jx_printf("[predictive-reuse]   no deep A levels to free\n");
   }

   if (k > 0)
   {
      jx_printf("[predictive-reuse]   would free F/U[%d..%d] before recreating tail vectors\n", k, num_levels - 1);
   }
   else if (k + 1 <= num_levels - 1)
   {
      jx_printf("[predictive-reuse]   would free F/U[%d..%d]\n", k + 1, num_levels - 1);
   }
   else
   {
      jx_printf("[predictive-reuse]   no F/U levels to free\n");
   }

   if (k <= num_levels - 2)
   {
      jx_printf("[predictive-reuse]   would free P/R/CF[%d..%d]\n", k, num_levels - 2);
   }
   else
   {
      jx_printf("[predictive-reuse]   no deep P/R/CF levels to free\n");
   }

   jx_printf("[predictive-reuse]   would rebuild coarsening loop from level %d\n", k);
}

static JX_Int
JXPAMG_PR_PredictThenFallbackDryRun(jx_ParAMGData *current_amg,
                                    jx_ParAMGData *base_amg,
                                    JX_Real theta_pred,
                                    JX_Int Lreuse,
                                    JX_Real delta)
{
   jx_ParCSRMatrix **A_current;
   JX_Int k;
   JX_Int num_levels;
   JX_Int my_id = 0;

   if (current_amg == NULL || base_amg == NULL)
   {
      jx_printf("[predictive-reuse] dry-run rejected: current/base hierarchy is NULL\n");
      return 0;
   }

   A_current = jx_ParAMGDataAArray(current_amg);
   if (A_current == NULL || A_current[0] == NULL)
   {
      jx_printf("[predictive-reuse] dry-run rejected: current hierarchy is not ready\n");
      return 0;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_current[0]), &my_id);
   num_levels = jx_ParAMGDataNumLevels(current_amg);
   k = JXPAMG_PR_PrintPredictiveProbe(current_amg, base_amg, theta_pred, Lreuse, delta);

   if (k < num_levels)
   {
      JXPAMG_PR_PrintSetupFromLevelPlan(current_amg, k);
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] dry-run decision: tail rebuild required; fall back to full setup for now\n");
      }
   }
   else
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] dry-run decision: full reuse path would be allowed\n");
      }
   }

   return k;
}


static JX_Int
JXPAMG_PR_FreeTailForSetupFromLevel(jx_ParAMGData *amg_data, JX_Int k)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParVector **F_array = jx_ParAMGDataFArray(amg_data);
   jx_ParVector **U_array = jx_ParAMGDataUArray(amg_data);
   jx_ParCSRMatrix **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix **R_array = jx_ParAMGDataRArray(amg_data);
   JX_Real **AI_measure_array = jx_ParAMGDataAIMeasureArray(amg_data);
   JX_Int **CF_marker_array = jx_ParAMGDataCFMarkerArray(amg_data);
   JX_Int **dof_func_array = jx_ParAMGDataDofFuncArray(amg_data);
   JX_Int restri_type = jx_ParAMGDataRestriction(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int level;

   if (k < 0 || k >= num_levels)
   {
      return 1;
   }

   if (P_array != NULL)
   {
      for (level = k; level < num_levels - 1; level++)
      {
         if (P_array[level] != NULL)
         {
            jx_ParCSRMatrixDestroy(P_array[level]);
            P_array[level] = NULL;
         }
      }
   }

   if (restri_type && R_array != NULL && R_array != P_array)
   {
      for (level = k; level < num_levels - 1; level++)
      {
         if (R_array[level] != NULL)
         {
            jx_ParCSRMatrixDestroy(R_array[level]);
            R_array[level] = NULL;
         }
      }
   }

   if (CF_marker_array != NULL)
   {
      for (level = k; level < num_levels - 1; level++)
      {
         if (CF_marker_array[level] != NULL)
         {
            jx_TFree(CF_marker_array[level]);
            CF_marker_array[level] = NULL;
         }
      }
   }

   if (AI_measure_array != NULL)
   {
      for (level = k; level < num_levels - 1; level++)
      {
         if (AI_measure_array[level] != NULL)
         {
            jx_TFree(AI_measure_array[level]);
            AI_measure_array[level] = NULL;
         }
      }
   }

   if (A_array != NULL)
   {
      for (level = k + 1; level < num_levels; level++)
      {
         if (A_array[level] != NULL)
         {
            jx_ParCSRMatrixDestroy(A_array[level]);
            A_array[level] = NULL;
         }
      }
   }

   if (F_array != NULL)
   {
      for (level = (k > 0 ? k : k + 1); level < num_levels; level++)
      {
         if (F_array[level] != NULL)
         {
            jx_ParVectorDestroy(F_array[level]);
            F_array[level] = NULL;
         }
      }
   }

   if (U_array != NULL)
   {
      for (level = (k > 0 ? k : k + 1); level < num_levels; level++)
      {
         if (U_array[level] != NULL)
         {
            jx_ParVectorDestroy(U_array[level]);
            U_array[level] = NULL;
         }
      }
   }

   if (dof_func_array != NULL)
   {
      for (level = k + 1; level < num_levels; level++)
      {
         if (dof_func_array[level] != NULL)
         {
            jx_TFree(dof_func_array[level]);
            dof_func_array[level] = NULL;
         }
      }
   }

   if (JXPAMG_PR_ResetTailFinalizationState(amg_data))
   {
      return 1;
   }

   /* Keep num_levels unchanged until the tail rebuild succeeds, so readiness
    * checks can still inspect the old tail range. */
   return 0;
}

/* Compile the Stage 6 wrapper in the experiment file so mpicc checks it.
 * Runtime use still requires JXPAMG_PR_ENABLE_TAIL_REBUILD=1. */
#include "stage6_rebuild_tail_wrapper_draft.inc"

static JX_Int
JXPAMG_PR_RebuildTailFromLevelSkeleton(jx_ParAMGData *amg_data, JX_Int k)
{
   JXPAMG_PR_TailContext ctx;
   JX_Int ready;
   JX_Int my_id = 0;

   if (JXPAMG_PR_InitTailContext(&ctx, amg_data, k))
   {
      jx_printf("[predictive-reuse] RebuildTail skeleton rejected: invalid k or A_array[k]\n");
      return 1;
   }

   jx_MPI_Comm_rank(ctx.comm, &my_id);
   ready = JXPAMG_PR_CheckTailContextReady(&ctx);
   if (ready == 1)
   {
      return 1;
   }
   if (ready == 2)
   {
      return 2;
   }
   if (my_id == 0)
   {
      jx_printf("[predictive-reuse] RebuildTail skeleton: ready to rebuild from level %d, but loop body is not wrapped yet\n", ctx.k);
      jx_printf("[predictive-reuse] RebuildTail skeleton: inspect stage6_tail_rebuild_loop.inc before enabling\n");
   }

#if AMG_CG_ZL
   if (my_id == 0)
   {
      jx_printf("[predictive-reuse] RebuildTail skeleton warning: AMG_CG_ZL path destroys A_array[0] in full setup and must be audited for k-level rebuild\n");
   }
   return 1;
#endif

   return 2;
}

JX_Int
JXPAMG_PR_SetupFromLevelExperimental(jx_ParAMGData *amg_data,
                                     JX_Int k,
                                     JX_Int dry_run)
{
   jx_ParCSRMatrix **A_array = jx_ParAMGDataAArray(amg_data);
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int my_id = 0;

   if (amg_data == NULL || A_array == NULL || A_array[0] == NULL)
   {
      jx_printf("[predictive-reuse] SetupFromLevel rejected: hierarchy is not ready\n");
      return 1;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   JXPAMG_PR_PrintSetupFromLevelPlan(amg_data, k);

   if (k < 0 || k >= num_levels)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] SetupFromLevel rejected: k is outside [0, num_levels)\n");
      }
      return 1;
   }

   if (dry_run)
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] SetupFromLevel dry-run only: no memory was freed\n");
      }
      return 2;
   }

   if (JXPAMG_PR_FreeTailForSetupFromLevel(amg_data, k))
   {
      if (my_id == 0)
      {
         jx_printf("[predictive-reuse] SetupFromLevel failed while freeing tail\n");
      }
      return 1;
   }

   if (my_id == 0)
   {
#if JXPAMG_PR_ENABLE_TAIL_REBUILD
      jx_printf("[predictive-reuse] SetupFromLevel freed tail; calling experimental rebuild wrapper next\n");
#else
      jx_printf("[predictive-reuse] SetupFromLevel freed tail; calling rebuild skeleton next\n");
#endif
   }
   JXPAMG_PR_PrintHierarchyShape("after freeing tail for SetupFromLevel", amg_data);
   if (my_id == 0)
   {
      jx_printf("[predictive-reuse] SetupFromLevel tail-free state is intentionally incomplete; TailContext readiness is the next check\n");
   }

#if JXPAMG_PR_ENABLE_TAIL_REBUILD
   return JXPAMG_PR_RebuildTailFromLevelDraft(amg_data, k);
#else
   return JXPAMG_PR_RebuildTailFromLevelSkeleton(amg_data, k);
#endif
}

/* Gate C correctness helper.
 * Installs the candidate A[k] before the existing in-place tail rebuild.
 * Not strict transactional: old tail freed before rebuild completes.
 * Failure must be handled by falling back to FullSetupOnNewMatrix.
 * On return, amg_data owns A_cut (caller must NULL its pointer). */
JX_Int
JXPAMG_PR_SetupFromLevelWithACut(jx_ParAMGData *amg_data,
                                 JX_Int k,
                                 jx_ParCSRMatrix *A_cut)
{
   jx_ParCSRMatrix **A_array;
   JX_Int num_levels;
   JX_Int my_id = 0;
   JX_Int ret;

   if (amg_data == NULL || A_cut == NULL) return 1;

   A_array = jx_ParAMGDataAArray(amg_data);
   if (A_array == NULL || A_array[0] == NULL) return 1;

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   num_levels = jx_ParAMGDataNumLevels(amg_data);
   if (k <= 0 || k >= num_levels) return 1;

   if (my_id == 0)
      jx_printf("[predictive-reuse] Gate C: replacing A[%d] old=%p candidate=%p\n",
                k, (void *)A_array[k], (void *)A_cut);

   if (A_array[k] != NULL) {
      jx_ParCSRMatrixDestroy(A_array[k]);
      A_array[k] = NULL;
   }
   A_array[k] = A_cut;

   if (my_id == 0)
      jx_printf("[predictive-reuse] Gate C: installed candidate A[%d]\n", k);

   ret = JXPAMG_PR_SetupFromLevelExperimental(amg_data, k, 0);
   if (ret == 0) {
      JXPAMG_PR_PrintHierarchyShape("after Gate C candidate tail rebuild", amg_data);
      JXPAMG_PR_CheckLinkDimensions("after Gate C candidate tail rebuild", amg_data);
   }
   return ret;
}

JX_Int
JXPAMG_PR_NoOpSetup(JX_Solver solver, JX_ParCSRMatrix A)
{
   jx_ParAMGData *amg_data = (jx_ParAMGData *) solver;
   jx_ParCSRMatrix **A_array;
   JX_Int my_id = 0;

   if (amg_data == NULL || A == NULL) return 1;

   A_array = jx_ParAMGDataAArray(amg_data);
   if (A_array == NULL || A_array[0] == NULL) return 1;

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
   if (my_id == 0)
      jx_printf("[predictive-reuse] Gate D: no-op AMG setup (full reuse)\n");

   return 0;
}

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

   if (amg_data == NULL || new_A == NULL) return 0;

   A_array = jx_ParAMGDataAArray(amg_data);
   P_array = jx_ParAMGDataPArray(amg_data);
   F_array = jx_ParAMGDataFArray(amg_data);
   U_array = jx_ParAMGDataUArray(amg_data);
   num_levels = jx_ParAMGDataNumLevels(amg_data);

   if (num_levels <= 1 || cut_k != num_levels) return 0;
   if (Lreuse > 0 && Lreuse < num_levels) return 0;
   if (A_array == NULL || P_array == NULL || F_array == NULL || U_array == NULL) return 0;
   if (A_array[0] == NULL) return 0;

   if (jx_ParCSRMatrixGlobalNumRows(new_A) != jx_ParCSRMatrixGlobalNumRows(A_array[0]) ||
       jx_ParCSRMatrixGlobalNumCols(new_A) != jx_ParCSRMatrixGlobalNumCols(A_array[0]) ||
       jx_ParCSRMatrixNumRows(new_A) != jx_ParCSRMatrixNumRows(A_array[0]))
      return 0;

   for (level = 0; level < num_levels; level++)
      if (A_array[level] == NULL) return 0;

   for (level = 0; level < num_levels - 1; level++)
      if (P_array[level] == NULL) return 0;

   for (level = 1; level < num_levels; level++)
      if (F_array[level] == NULL || U_array[level] == NULL) return 0;

   return 1;
}

/*
 * Incremental modification checkpoints for Algorithm 1.
 *
 * Stage 0: keep this file behavior-equivalent to the copied full setup and
 *          use JXPAMG_PR_PrintHierarchyShape/JXPAMG_PR_CheckLinkDimensions
 *          as the instrumentation baseline.
 * Stage 1: add a full-reuse RAP-only update path; do not cut or rebuild yet.
 * Stage 2: compute sigma_l for each probed level and only print k decisions.
 * Stage 3: make k decisions control flow, but fall back to full setup.
 * Stage 4: add a dry-run SetupFromLevel(k) that prints the keep/free ranges.
 * Stage 5: free deep levels for real, then still fall back to full setup.
 * Stage 6: run the copied coarsening loop from level k and rebuild the tail.
 */

// CSR每行的列号从小到大排序，为满足pardiso的入口要求
void jx_CSRMatrixSort(jx_CSRMatrix *A_CSR)
{
   int i, k;
   int row = A_CSR->num_rows;

   for ( i = 0; i < row; i++)
   {
      k = A_CSR->i[i];
      jx_qsort1( &A_CSR->j[k], &A_CSR->data[k], 0, A_CSR->i[i+1] - A_CSR->i[i] - 1 );
   }
}


/*!
 * \fn JX_Int jx_PAMGSetup_predictive_reuse_experiment
 * \brief Experimental copy of the PAMG setup phase.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGSetup_predictive_reuse_experiment( void             *amg_vdata,
                                           jx_ParCSRMatrix  *par_matrix )
{
   MPI_Comm         comm     = jx_ParCSRMatrixComm(par_matrix); 
   jx_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */
   jx_ParCSRMatrix **A_array;
   jx_ParVector    **F_array;
   jx_ParVector    **U_array;
   jx_ParVector     *Vtemp;
   jx_ParCSRMatrix **P_array;
   jx_ParCSRMatrix **R_array;
   jx_ParVector     *Residual_array;
   JX_Real          **AI_measure_array;   
   JX_Int             **CF_marker_array;   
   JX_Int             **dof_func_array;   
   JX_Int              *dof_func;
   JX_Int              *col_offd_S_to_A;
   JX_Int              *col_offd_Sabs_to_A = NULL;
   JX_Int            *AIR_maxsize_ls;
   JX_Real            strong_threshold;
   JX_Real            AIR_strong_th;
   JX_Real            max_row_sum;
   JX_Real            trunc_factor;
   JX_Real            S_commpkg_switch;

   JX_Int      max_levels; 
   JX_Int      amg_logging;
   JX_Int      amg_print_level;
   JX_Int      debug_flag;
   JX_Int      local_num_vars;
   JX_Int      P_max_elmts;
   JX_Int      R_max_size; 
   
   JX_Solver *smoother = NULL;
   JX_Int        smooth_type = jx_ParAMGDataSmoothType(amg_data);
   JX_Int        smooth_num_levels = jx_ParAMGDataSmoothNumLevels(amg_data);
   char      *euclidfile;
   JX_Int	      eu_level;
   JX_Int	      eu_bj;
   JX_Real     eu_sparse_A;

   /* Local variables */
   JX_Real           *AI_measure;
   JX_Real           *AI_measure2;
   JX_Int              *CF_marker;
   JX_Int              *CFN_marker;
   jx_ParCSRMatrix  *par_S;
   jx_ParCSRMatrix  *Sabs = NULL;
   jx_ParCSRMatrix  *par_S2;
   jx_ParCSRMatrix  *par_P;
   jx_ParCSRMatrix  *R;
   jx_ParCSRMatrix  *A_H;

   JX_Int       wall_time_option;         /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real    wall_time_coarsen = 0.0;  /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real    wall_time_rap = 0.0;      /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real    wall_time_interp = 0.0;   /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real    tmp_wall_time = 0.0;      /* newly added Yue Xiaoqiang 2015/09/30 */

   JX_Int       old_num_levels, num_levels;
   JX_Int       level;
   JX_Int       local_size, i;
   JX_Int       first_local_row;
   JX_Int       coarse_size_last = -1;
   JX_Int       coarse_size;
   JX_Int       coarsen_type;
   JX_Int       interp_type;
   JX_Int       restri_type;
   JX_Int       measure_type;
   JX_Int       setup_type;
   JX_Int       fine_size;
   JX_Int       rest, tms, indx;
   JX_Int       not_finished_coarsening = 1;
   JX_Int       Setup_err_flag = 0;
   JX_Int       coarse_threshold; /* we don't fix it to be 9.  peghoty 2010/04/14 */
   JX_Int       coarse_threshold_2; /* by xu: 2013/11/25 */
   JX_Int       j, k;
   JX_Int       num_procs, my_id, num_threads;
   JX_Int      *grid_relax_type = jx_ParAMGDataGridRelaxType(amg_data);
   JX_Int       num_functions = jx_ParAMGDataNumFunctions(amg_data);
   JX_Int       spmt_rap_type = jx_ParAMGDataSpMtRapType(amg_data);
   JX_Int       ai_measure_type = jx_ParAMGDataAIMeasureType(amg_data);
   JX_Int       num_paths = jx_ParAMGDataNumPaths(amg_data);
   JX_Int       agg_num_levels = jx_ParAMGDataAggNumLevels(amg_data);
   JX_Int       agg_interp_type = jx_ParAMGDataAggInterpType(amg_data);
   JX_Int       agg_P_max_elmts = jx_ParAMGDataAggPMaxElmts(amg_data);
   //JX_Int       agg_P12_max_elmts = jx_ParAMGDataAggP12MaxElmts(amg_data);
   JX_Real    agg_trunc_factor = jx_ParAMGDataAggTruncFactor(amg_data);
   //JX_Real    agg_P12_trunc_factor = jx_ParAMGDataAggP12TruncFactor(amg_data);
   JX_Int       rap2 = jx_ParAMGDataRAP2(amg_data);
   JX_Int       keepTranspose = jx_ParAMGDataKeepTranspose(amg_data);
   JX_Int       print_coarse_matrix = jx_ParAMGDataPrintCoarseSystem(amg_data);
   JX_Int      *opt_icor;
   JX_Int      *coarse_dof_func;
   JX_Int      *coarse_pnts_global;
   JX_Int	    *coarse_pnts_global1;
   JX_Real    size;
   JX_Real    coarse_ratio;           /* newly added peghoty 2010/04/14 */
   JX_Real    wall_time = 0.0;        /* for debugging instrumentation */
   JX_Int       measure_type_rlx;       // newly added peghoty 2010/05/29
   JX_Int       number_syn_rlx;         // newly added peghoty 2010/05/29
   JX_Real    measure_threshold_rlx;  // newly added peghoty 2010/05/29

   /* coarse matrices */
   char FileNameCoaMat[256];

   /* ai statistic information*/
   JX_Int       num_vars_local = 0, num_vars_global;
   JX_Int       num_ai_local = 0, num_ai_global;
   JX_Int       num_ai_local_valid = 0, num_ai_global_valid;
   JX_Int       num_ai_c_local = 0, num_ai_c_global;
   JX_Real    mai_local = 0.0, mai_global;
   JX_Real    mai_local_valid = 0.0, mai_global_valid;
   JX_Real    mai_c_local = 0.0, mai_c_global;

   JX_Int       num_vars_local_0 = 0;
   JX_Int       num_ai_local_0 = 0;
   JX_Int       num_ai_local_valid_0 = 0;
   JX_Int       num_ai_c_local_0 = 0;
   JX_Real    mai_local_0 = 0.0;
   JX_Real    mai_local_valid_0 = 0.0;
   JX_Real    mai_c_local_0 = 0.0;
   JX_Real    mai_threshold = 0.1;

   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm, &my_id);
   if ((num_procs > 1) && (spmt_rap_type != 1))  /* Yue Xiaoqiang 2012/10/12 */
   {
      spmt_rap_type = 1;
   }
   num_threads = jx_NumThreads();               /* Yue Xiaoqiang 2012/10/12 */
   opt_icor = jx_CTAlloc(JX_Int, 5*num_threads+2); /* Yue Xiaoqiang 2012/10/12 */

   wall_time_option = jx_ParAMGDataWallTimeOption(amg_data);
   old_num_levels   = jx_ParAMGDataNumLevels(amg_data);
   max_levels       = jx_ParAMGDataMaxLevels(amg_data);
   amg_logging      = jx_ParAMGDataLogging(amg_data);
   amg_print_level  = jx_ParAMGDataPrintLevel(amg_data);
   coarsen_type     = jx_ParAMGDataCoarsenType(amg_data);
   measure_type     = jx_ParAMGDataMeasureType(amg_data);
   setup_type       = jx_ParAMGDataSetupType(amg_data);
   debug_flag       = jx_ParAMGDataDebugFlag(amg_data);
   dof_func         = jx_ParAMGDataDofFunc(amg_data);
   interp_type      = jx_ParAMGDataInterpType(amg_data);
   restri_type      = jx_ParAMGDataRestriction(amg_data);
   AIR_strong_th    = jx_ParAMGDataAIRStrongTh(amg_data);
   euclidfile       = jx_ParAMGDataEuclidFile(amg_data);
   eu_level         = jx_ParAMGDataEuLevel(amg_data);
   eu_bj            = jx_ParAMGDataEuBJ(amg_data);
   eu_sparse_A      = jx_ParAMGDataEuSparseA(amg_data);
   coarse_threshold = jx_ParAMGDataCoarseThreshold(amg_data);           /* newly added peghoty 2010/04/14 */
   coarse_ratio     = jx_ParAMGDataCoarseRatio(amg_data);               /* newly added peghoty 2010/04/14 */
   measure_type_rlx = jx_ParAMGDataMeasureTypeRlx(amg_data);            /* newly added peghoty 2010/05/29 */
   number_syn_rlx   = jx_ParAMGDataNumberSynRlx(amg_data);              /* newly added peghoty 2010/05/29 */
   measure_threshold_rlx = jx_ParAMGDataMeasureThresholdRlx(amg_data);  /* newly added peghoty 2010/05/29 */

   coarse_threshold_2 = -1;
   
   jx_ParCSRMatrixSetNumNonzeros(par_matrix);
   jx_ParCSRMatrixSetDNumNonzeros(par_matrix);
   jx_ParAMGDataNumVariables(amg_data) = jx_ParCSRMatrixNumRows(par_matrix);
   
   if (setup_type == 0) 
   {
      return(Setup_err_flag);
   }

   par_S = NULL;

   A_array = jx_ParAMGDataAArray(amg_data);
   P_array = jx_ParAMGDataPArray(amg_data);
   R_array = jx_ParAMGDataRArray(amg_data);
   AIR_maxsize_ls = jx_ParAMGDataAIRMaxSizeLS(amg_data);
   AI_measure_array = jx_ParAMGDataAIMeasureArray(amg_data);
   CF_marker_array = jx_ParAMGDataCFMarkerArray(amg_data);
   dof_func_array  = jx_ParAMGDataDofFuncArray(amg_data);
   local_size = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(par_matrix));

   grid_relax_type[3] = jx_ParAMGDataUserCoarseRelaxType(amg_data); 

   if (A_array || P_array || R_array || AIR_maxsize_ls || AI_measure_array || CF_marker_array || dof_func_array)
   {
      for (j = 1; j < old_num_levels; j ++)
      {
         if (A_array[j])
         {
            jx_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
         }
       
         if (dof_func_array[j])
         {
            jx_TFree(dof_func_array[j]);
            dof_func_array[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels - 1; j ++)
      {
         if (P_array[j])
         {
            jx_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }
         if (R_array[j])
         {
            jx_ParCSRMatrixDestroy(R_array[j]);
            R_array[j] = NULL;
         }
      }

      jx_TFree(AIR_maxsize_ls);
      AIR_maxsize_ls = NULL;

     /*-------------------------------------------------------------------
      *  Special case use of CF_marker_array when old_num_levels = 1
      *  requires us to attempt this deallocation every time
      *------------------------------------------------------------------*/
      
      if (CF_marker_array[0])
      {
         jx_TFree(CF_marker_array[0]);
         CF_marker_array[0] = NULL;
      }

      for (j = 1; j < old_num_levels-1; j ++)
      {
         if (CF_marker_array[j])
         {
            jx_TFree(CF_marker_array[j]);
            CF_marker_array[j] = NULL;
         }
      }
      
      /* for AI_measure: added by xwxu */
      if (AI_measure_array[0])
      {
         jx_TFree(AI_measure_array[0]);
         AI_measure_array[0] = NULL;
      }

      for (j = 1; j < old_num_levels-1; j ++)
      {
         if (AI_measure_array[j])
         {
            jx_TFree(AI_measure_array[j]);
            AI_measure_array[j] = NULL;
         }
      }
   }

   if (A_array == NULL)
   {
      A_array = jx_CTAlloc(jx_ParCSRMatrix*, max_levels);
   }

   if (P_array == NULL && max_levels > 1)
   {
      P_array = jx_CTAlloc(jx_ParCSRMatrix*, max_levels - 1);
   }

   /* If retri_type != 0, R != P^T, allocate R matrices */
   if (restri_type)
   {
      if (R_array == NULL && max_levels > 1)
      {
         R_array = jx_CTAlloc(jx_ParCSRMatrix*, max_levels-1);
         AIR_maxsize_ls = jx_CTAlloc(JX_Int, max_levels-1);
      }
   }

   if (AI_measure_array == NULL)
   {
      AI_measure_array = jx_CTAlloc(JX_Real*, max_levels);
   }
   if (CF_marker_array == NULL)
   {
      CF_marker_array = jx_CTAlloc(JX_Int*, max_levels);
   }
   if (dof_func_array == NULL)
   {
      dof_func_array = jx_CTAlloc(JX_Int*, max_levels);
   }

   if (num_functions > 1 && dof_func == NULL)
   {
      first_local_row = jx_ParCSRMatrixFirstRowIndex(par_matrix);
      dof_func = jx_CTAlloc(JX_Int, local_size);
      rest = first_local_row-((first_local_row / num_functions)*num_functions);
      indx = num_functions - rest;
      if (rest == 0) 
      {
         indx = 0;
      }
      k = num_functions - 1;
      for (j = indx - 1; j > -1; j --)
      {
         dof_func[j] = k --;
      }
      tms = local_size / num_functions;
      if (tms*num_functions + indx > local_size) 
      {
         tms --;
      }
      for (j = 0; j < tms; j ++)
      {
         for (k = 0; k < num_functions; k ++)
         {
            dof_func[indx++] = k;
         }
      }
      k = 0;
      while (indx < local_size)
      {
         dof_func[indx++] = k ++;
      }
      jx_ParAMGDataDofFunc(amg_data) = dof_func;
   }

   A_array[0] = par_matrix;

   dof_func_array[0] = dof_func;
   jx_ParAMGDataAIMeasureArray(amg_data) = AI_measure_array;
   jx_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   jx_ParAMGDataDofFuncArray(amg_data) = dof_func_array;
   jx_ParAMGDataAArray(amg_data) = A_array;
   jx_ParAMGDataPArray(amg_data) = P_array;
   /* If R != P^T */
   if (restri_type)
   {
      jx_ParAMGDataRArray(amg_data) = R_array;
      jx_ParAMGDataAIRMaxSizeLS(amg_data) = AIR_maxsize_ls;
   }
   else
   {
      jx_ParAMGDataRArray(amg_data) = P_array;
   }

   Vtemp = jx_ParAMGDataVtemp(amg_data);

   if (Vtemp != NULL)
   {
      jx_ParVectorDestroy(Vtemp);
      Vtemp = NULL;
   }

   Vtemp = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[0]),
                              jx_ParCSRMatrixGlobalNumRows(A_array[0]),
                              jx_ParCSRMatrixRowStarts(A_array[0]));
   jx_ParVectorInitialize(Vtemp);
   jx_ParVectorSetPartitioningOwner(Vtemp,0);
   jx_ParAMGDataVtemp(amg_data) = Vtemp;

   F_array = jx_ParAMGDataFArray(amg_data);
   U_array = jx_ParAMGDataUArray(amg_data);

   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_levels; j ++)
      {
         if (F_array[j] != NULL)
         {
            jx_ParVectorDestroy(F_array[j]);
            F_array[j] = NULL;
         }
         if (U_array[j] != NULL)
         {
            jx_ParVectorDestroy(U_array[j]);
            U_array[j] = NULL;
         }
      }
   }

   if (F_array == NULL)
   {
      F_array = jx_CTAlloc(jx_ParVector*, max_levels);
   }
   if (U_array == NULL)
   {
      U_array = jx_CTAlloc(jx_ParVector*, max_levels);
   }

  /*---------------------------------------------------------
   *  They have been moved to "amg_solve" since they are   
   *  not used here.  peghoty 2009/07/27 
   *--------------------------------------------------------*/
   //F_array[0] = f;
   //U_array[0] = u;

   jx_ParAMGDataFArray(amg_data) = F_array;
   jx_ParAMGDataUArray(amg_data) = U_array;

  /*----------------------------------------------------------
   *   Initialize jx_ParAMGData
   *---------------------------------------------------------*/
   /* STAGE6-BOUNDARY: tail rebuild needs an alternate entry here.
    * Full setup starts at level 0; SetupFromLevel(k) must start with
    * level = k and an already valid A_array[k]. */
   not_finished_coarsening = 1;
   level = 0;
   strong_threshold = jx_ParAMGDataStrongThreshold(amg_data);
   max_row_sum = jx_ParAMGDataMaxRowSum(amg_data);
   trunc_factor = jx_ParAMGDataTruncFactor(amg_data);
   P_max_elmts = jx_ParAMGDataPMaxElmts(amg_data);
   S_commpkg_switch = jx_ParAMGDataSCommPkgSwitch(amg_data);
   if (smooth_num_levels > level)
   {
      smoother = jx_CTAlloc(JX_Solver, smooth_num_levels);
      jx_ParAMGDataSmoother(amg_data) = smoother;
   }

   /* if AI-based coarsening is used, set ai_measure_type = 1. */
   if (coarsen_type == 990 || coarsen_type == 991 || coarsen_type == 993 || 
       coarsen_type == 96 || coarsen_type == 98 || coarsen_type == 910 || 
       coarsen_type == 908 || coarsen_type == 918 || coarsen_type == 928 || 
       coarsen_type == 938 || coarsen_type == 968 || 
       amg_print_level > 0)
   {
      ai_measure_type = 1; 

   }

  /*-----------------------------------------------------
   *   Enter Coarsening Loop
   *-----------------------------------------------------*/

   if (amg_print_level > 0 && my_id ==0)
   {
      jx_printf(" \n");
      jx_printf("================================================================== \n");
      jx_printf("+++++++++++++ multi-scale/AI infomation for levels +++++++++++++++ \n");
   }

   /* STAGE6-EXTRACT-BEGIN: this loop is the candidate body for
    * JXPAMG_PR_RebuildTailFromLevel(k). It currently assumes all
    * setup-local variables have been initialized for level 0. */
   while (not_finished_coarsening)
   {
#if AMG_CG_ZL      
      if (level == 1){
         jx_printf("\n%s(line: %d), level = %d, jx_ParCSRMatrixDestroy(A_hat), Zhaoli, 2021.07.09\n", __FUNCTION__, __LINE__, level);
         jx_ParCSRMatrixDestroy(A_array[0]);
      }
#endif      
      fine_size = jx_ParCSRMatrixGlobalNumRows(A_array[level]);

      if (level > 0)
      {   
         F_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                             jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                             jx_ParCSRMatrixRowStarts(A_array[level]));
         jx_ParVectorInitialize(F_array[level]);
         jx_ParVectorSetPartitioningOwner(F_array[level],0);
            
         U_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                             jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                             jx_ParCSRMatrixRowStarts(A_array[level]));
         jx_ParVectorInitialize(U_array[level]);
         jx_ParVectorSetPartitioningOwner(U_array[level],0);
      }


     /*--------------------------------------------------------------
      *  Select coarse-grid points on 'level' : returns CF_marker
      *  for the level.  Returns strength matrix, par_S 
      *  Returns AI_Measure. 
      *--------------------------------------------------------------*/
     
      if (debug_flag == 1) wall_time = jx_time_getWallclockSeconds();
      if (debug_flag == 3)
      {
         jx_printf("\n ===== Proc = %d     Level = %d  =====\n",my_id, level);
         fflush(NULL);
      }

      if (wall_time_option == 1) tmp_wall_time = jx_time_getWallclockSeconds();

      if (max_levels > 1)
      {
         local_num_vars = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[level]));

        /*--------------------------------------------------------------------------
         *  Get the Strength Matrix 
         *-------------------------------------------------------------------------*/        
         jx_PAMGCreateS(A_array[level],strong_threshold,max_row_sum,num_functions,dof_func_array[level],&par_S);
         col_offd_S_to_A = NULL;
         if (strong_threshold > S_commpkg_switch)
         {
            jx_PAMGCreateSCommPkg(A_array[level], par_S, &col_offd_S_to_A);
         }
         /* for AIR, need absolute value SOC */
         if (restri_type)
         {
            jx_PAMGCreateSabs(A_array[level],AIR_strong_th,1.0,num_functions,dof_func_array[level],&Sabs);
            col_offd_Sabs_to_A = NULL;
            if (strong_threshold > S_commpkg_switch)
            {
               jx_PAMGCreateSCommPkg(A_array[level], Sabs, &col_offd_Sabs_to_A);
            }
         }

         if (ai_measure_type == 1)
         {

           /*--------------------------------------------------------------------------
            *  Get the AI Measure 
            *-------------------------------------------------------------------------*/        
            jx_PAMGMeasureAI(par_S, A_array[level], debug_flag, &AI_measure);

           /*--------------------------------------------
            *  store the AI array
            *-------------------------------------------*/ 
            AI_measure_array[level] = AI_measure;

         }

        /*--------------------------------------------------------------------------
         *    Do the appropriate coarsening as follows:
         *
         *      coarsen_type =  0: CLJP 
         *      coarsen_type =  1: Ruge
         *      coarsen_type = 11: Ruge 1st pass only
         *      coarsen_type =  2: Ruge2B
         *      coarsen_type =  3: Ruge3
         *      coarsen_type =  4: Ruge3c
         *      coarsen_type =  5: Ruge relax special points
         *      coarsen_type =  6: Falgout
         *      coarsen_type =  8: PMIS
         *      coarsen_type = 10: HMIS
         *      coarsen_type = 90: RCLJP 
         *      coarsen_type = 91: RRS0
         *      coarsen_type = 990: CLJP_AI 
         *      coarsen_type = 991: Ruge_AI
         *      coarsen_type = 993: Ruge3_AI
         *      coarsen_type = 96: Falgout_AI
         *      coarsen_type = 98: PMIS_AI
         *      coarsen_type = 910: HMIS_AI
         *      coarsen_type = 908, 918, 928, 938, 968: AI-TYPE.
         *                                          peghoty 2010/05/29
         *-------------------------------------------------------------------------*/ 

         //jx_printf("=========== coarsen_type = %d \n", coarsen_type); 

         if (coarsen_type == 6)  
         {
            jx_PAMGCoarsenFalgout(par_S, A_array[level], measure_type, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 96) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenFalgoutAI(par_S, A_array[level], AI_measure_array[level], measure_type, debug_flag, &CF_marker);
         }
         else if (coarsen_type == 8) 
         {
            jx_PAMGCoarsenPMIS(par_S, A_array[level], 0, debug_flag, &CF_marker);
         }
         else if (coarsen_type == 10) 
         {
            jx_PAMGCoarsenHMIS(par_S, A_array[level], measure_type, debug_flag, &CF_marker);
         }      
         else if ((coarsen_type == 910) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenHMISAI(par_S, A_array[level], AI_measure_array[level], 0, measure_type, debug_flag, &CF_marker);
         }      
         else if (coarsen_type == 1 || coarsen_type == 2 ||
                  coarsen_type == 3 || coarsen_type == 4 ||
                  coarsen_type == 5 || coarsen_type == 11  ) 
         {
            jx_PAMGCoarsenRuge(par_S, A_array[level], measure_type, coarsen_type, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 991) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenRugeAI(par_S, A_array[level], AI_measure_array[level], 0, measure_type, 11, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 992) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenRugeAI(par_S, A_array[level], AI_measure_array[level], 0, measure_type, 1, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 993) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenRugeAI(par_S, A_array[level], AI_measure_array[level], 0, measure_type, 3, debug_flag, &CF_marker);
         }
         else if (coarsen_type == 0)    
         {
            jx_PAMGCoarsen(par_S, A_array[level], 0, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 990) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenAI(par_S, A_array[level], AI_measure_array[level], 0, debug_flag, &CF_marker);
         }
         else if (coarsen_type == 90)
         {
            jx_PAMGCoarsenRCLJP(par_S, A_array[level], measure_type_rlx, number_syn_rlx,
                                measure_threshold_rlx, 0, debug_flag, &CF_marker);
         }         
         else if (coarsen_type == 91)
         {
            jx_PAMGCoarsenRRS0(par_S, A_array[level], measure_type, measure_type_rlx,
                               number_syn_rlx, measure_threshold_rlx, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 98) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenPMISAI(par_S, A_array[level], AI_measure_array[level], 0, debug_flag, &CF_marker);
         }
         else if ((coarsen_type == 908 || coarsen_type == 918 || coarsen_type == 928 || coarsen_type == 938 || 
                  coarsen_type == 968) && (ai_measure_type == 1))
         {
            jx_PAMGCoarsenXML(par_S, A_array[level], AI_measure_array[level], measure_type, coarsen_type, debug_flag, &CF_marker);
         }      

         if (level < agg_num_levels)
         {
            jx_PAMGCoarseParms(comm, local_num_vars, 1,
                               dof_func_array[level], CF_marker,
                               &coarse_dof_func, &coarse_pnts_global1);
            jx_PAMGCreate2ndS(par_S, CF_marker, num_paths, coarse_pnts_global1, &par_S2);

           /*--------------------------------------------------------------------------
            *  Get the AI Measure
            *-------------------------------------------------------------------------*/
            if (ai_measure_type == 1)
            {
               jx_PAMGMeasureAI(par_S2, par_S2, debug_flag, &AI_measure2);
            }

            if (coarsen_type == 10)
            {
               jx_PAMGCoarsenHMIS(par_S2, par_S2, measure_type+3, debug_flag, &CFN_marker);
            }
            else if ((coarsen_type == 910) && (ai_measure_type == 1))
            {
               jx_PAMGCoarsenHMISAI(par_S2, par_S2, AI_measure2, 0, measure_type+3, debug_flag, &CFN_marker);
            }
            else if (coarsen_type == 8)
            {
               jx_PAMGCoarsenPMIS(par_S2, par_S2, 3, debug_flag, &CFN_marker);
            }
            else if ((coarsen_type == 98) && (ai_measure_type == 1))
            {
               jx_PAMGCoarsenPMISAI(par_S2, par_S2, AI_measure2, 3, debug_flag, &CFN_marker);
            }
            else if (coarsen_type == 6)
            {
               jx_PAMGCoarsenFalgout(par_S2, par_S2, measure_type, debug_flag, &CFN_marker);
            }
            else if ((coarsen_type == 96) && (ai_measure_type == 1))
            {
               jx_PAMGCoarsenFalgoutAI(par_S2, par_S2, AI_measure2, measure_type, debug_flag, &CFN_marker);
            }
            else if (coarsen_type == 0)
            {
               jx_PAMGCoarsen(par_S2, par_S2, 0, debug_flag, &CFN_marker);
            }
            else if ((coarsen_type == 990) && (ai_measure_type == 1))
            {
               jx_PAMGCoarsenAI(par_S2, par_S2, AI_measure2, 0, debug_flag, &CFN_marker);
            }
            jx_ParCSRMatrixDestroy(par_S2);
            if (ai_measure_type == 1)
            {
               jx_TFree(AI_measure2);
            }
         }

         /* Here for changes of min_coarse_size */

         if (level < agg_num_levels)
         {
            if (agg_interp_type == 4)
            {
               jx_PAMGCorrectCFMarker(CF_marker, local_num_vars, CFN_marker);
               jx_TFree(coarse_pnts_global1);
               jx_TFree(CFN_marker);
            }
         }

        /*--------------------------------------------
         *  store the CF array
         *-------------------------------------------*/ 
         CF_marker_array[level] = CF_marker;

         if (debug_flag == 1)
         {
            wall_time = jx_time_getWallclockSeconds() - wall_time;
            jx_printf("Proc = %d    Level = %d    Coarsen Time = %f\n",my_id,level, wall_time); 
            fflush(NULL);
         }

         if (wall_time_option == 1)
         {
            wall_time_coarsen += (jx_time_getWallclockSeconds() - tmp_wall_time);
         }

        /*-------------------------------------------------------------------------
         *  Get the coarse parameters
         *-------------------------------------------------------------------------*/ 
         jx_PAMGCoarseParms(comm, local_num_vars, num_functions, 
                            dof_func_array[level], CF_marker,
                            &coarse_dof_func, &coarse_pnts_global);

         dof_func_array[level+1] = NULL;
         if (num_functions > 1) dof_func_array[level+1] = coarse_dof_func;

#ifdef JX_NO_GLOBAL_PARTITION
         if (my_id == (num_procs -1)) 
         {
            coarse_size = coarse_pnts_global[1];
         }
         jx_MPI_Bcast(&coarse_size, 1, JX_MPI_INT, num_procs-1, comm);
#else
         coarse_size = coarse_pnts_global[num_procs];
#endif
      
        if ( coarse_size <= coarse_threshold && coarse_size_last >= coarse_size ) {
           break;  
        } 

      }
      else  /* max_levels = 1 */
      {

        /*--------------------------------------------------------------------------
         *  Get the Strength Matrix 
         *-------------------------------------------------------------------------*/        
         jx_PAMGCreateS(A_array[level],strong_threshold,max_row_sum,num_functions,dof_func_array[level],&par_S);
         col_offd_S_to_A = NULL;
         if (strong_threshold > S_commpkg_switch)
         {
            jx_PAMGCreateSCommPkg(A_array[level], par_S, &col_offd_S_to_A);
         }

         if (ai_measure_type == 1)
         {

           /*--------------------------------------------------------------------------
            *  Get the AI Measure 
            *-------------------------------------------------------------------------*/        
            jx_PAMGMeasureAI(par_S, A_array[level], debug_flag, &AI_measure);

           /*--------------------------------------------
            *  store the AI array
            *-------------------------------------------*/ 
            AI_measure_array[level] = AI_measure;

         }

         par_S = NULL;
         coarse_pnts_global = NULL;
         CF_marker = jx_CTAlloc(JX_Int, local_size );
         for (i = 0; i < local_size ; i ++) 
         {
            CF_marker[i] = 1;
         }
         CF_marker_array[level] = CF_marker;
         coarse_size = fine_size;

         local_num_vars = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[level]));
      }

     /*-------------------------------------------------------------------------
      *  compute ai-information for all-pnts and C-pnts!
      *-------------------------------------------------------------------------*/ 
     
      if ((max_levels == 1 || level != max_levels-1) && (ai_measure_type == 1))
      { 
         for (i = 0; i < local_num_vars; i++) {
             num_vars_local++ ;
             //if (AI_measure[i] > 0.0 && AI_measure[i] < 1.0)
             if (AI_measure[i] > 1.0e-10) 
             {
                mai_local += AI_measure[i];
                num_ai_local++ ;
                if (AI_measure[i] >= mai_threshold) {
                   mai_local_valid += AI_measure[i];
                   num_ai_local_valid++ ;
                   if (CF_marker[i] == 1) {
                      //if (level == 0 ) jx_printf("i_c = %d, ai_measure= %f \n", i, AI_measure[i]);
                      mai_c_local += AI_measure[i];
                      num_ai_c_local++;
                   }
                }
             }
         }
      }

      if (level == 0) {
        num_ai_local_0 = num_ai_local;
        num_ai_local_valid_0 = num_ai_local_valid;
        num_vars_local_0 = num_vars_local;
        num_ai_c_local_0 = num_ai_c_local;
        mai_local_0 = mai_local;
        mai_local_valid_0 = mai_local_valid;
        mai_c_local_0 = mai_c_local;
      }


     /*-------------------------------------------------------------------------
      *  if no coarse-grid, stop coarsening, and set the
      *  coarsest solve to be a single sweep of Jacobi !
      *-------------------------------------------------------------------------*/ 

      if ( (coarse_size == 0) || (coarse_size == fine_size) )
      {
         JX_Int  *num_grid_sweeps   = jx_ParAMGDataNumGridSweeps(amg_data);
         JX_Int **grid_relax_points = jx_ParAMGDataGridRelaxPoints(amg_data);
         if (grid_relax_type[3] == 9)
         {
            grid_relax_type[3] = grid_relax_type[0];
            num_grid_sweeps[3] = 1;
            if (grid_relax_points)
            {
               grid_relax_points[3][0] = 0; 
            }
         }
         if (par_S) 
         {
            jx_ParCSRMatrixDestroy(par_S);
         }
         jx_TFree(coarse_pnts_global);
         if (level > 0)
         {
           /*-------------------------------------------------------------
            * Note special case treatment of CF_marker is necessary
            * to do CF relaxation correctly when num_levels = 1
            *------------------------------------------------------------*/ 
            jx_TFree(CF_marker_array[level]);
            jx_ParVectorDestroy(F_array[level]);
            jx_ParVectorDestroy(U_array[level]);
         }
         break; 
      }


      /* Build restriction */
      if (restri_type)
      {
         /* !!! Ensure that CF_marker contains -1 or 1 !!! */
         for (i = 0; i < jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[level])); i ++)
         {
            CF_marker[i] = CF_marker[i] > 0 ? 1 : -1;
         }
         if (restri_type == 1) /* distance-1 AIR */
         {
            jx_PAMGBuildRestrAIR(A_array[level], CF_marker, 
                                 Sabs, coarse_pnts_global, num_functions, 
                                 dof_func_array[level], 
                                 debug_flag, trunc_factor, P_max_elmts, 
                                 col_offd_Sabs_to_A, &R, &R_max_size );
         }
         else /* distance-2 AIR */
         {
            jx_PAMGBuildRestrDist2AIR(A_array[level], CF_marker, 
                                      Sabs, coarse_pnts_global, num_functions, 
                                      dof_func_array[level], 
                                      debug_flag, trunc_factor, P_max_elmts, 
                                      col_offd_Sabs_to_A, &R, &R_max_size );
         }
         if (Sabs)
         {
            jx_ParCSRMatrixDestroy(Sabs);
            Sabs = NULL;
         }
         jx_TFree(col_offd_Sabs_to_A);
      }


     /*-----------------------------------------------------------------------------------
      *   Build prolongation matrix, P, and place in P_array[level]
      *
      *     interp_type =  0: modified classical interpolation
      *     interp_type =  3: direct interpolation (with separation of weights)
      *     interp_type =  4: multipass interpolation
      *     interp_type =  5: multipass interpolation (with separation of weights)
      *     interp_type =  6: extended classical modified interpolation
      *     interp_type =  7: extended (if no common C neighbor) classical 
      *                       modified interpolation
      *     interp_type =  8: standard interpolation
      *     interp_type =  9: standard interpolation (with separation of weights)
      *-----------------------------------------------------------------------------------*/
       
      if (debug_flag == 1) wall_time = jx_time_getWallclockSeconds();

      if (wall_time_option == 1) tmp_wall_time = jx_time_getWallclockSeconds();

      if (level < agg_num_levels)
      {
         if (agg_interp_type == 4)
         {
            jx_PAMGBuildMultipass(A_array[level], CF_marker_array[level], par_S, coarse_pnts_global,
                                  num_functions, dof_func_array[level], debug_flag,
                                  agg_trunc_factor, agg_P_max_elmts, 0, col_offd_S_to_A, &par_P);
         }
      }
      else
      {
         if (interp_type == 0)  // classical modified interpolation
         {
            if (spmt_rap_type == 1)
            {
               jx_PAMGBuildInterp( A_array[level], CF_marker_array[level], par_S,
                                coarse_pnts_global, num_functions, 
                                dof_func_array[level], debug_flag, trunc_factor,
                                P_max_elmts, col_offd_S_to_A, &par_P ); 
            }
            else if (spmt_rap_type == 2) /* Yue Xiaoqiang 2012/10/12 */
            {
               jx_PAMGBuildInterp1( A_array[level], CF_marker_array[level], par_S,
                                 coarse_pnts_global, num_functions, 
                                 dof_func_array[level], debug_flag, trunc_factor,
                                 P_max_elmts, col_offd_S_to_A, &par_P, opt_icor );
            }
            jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 3)  // direct interpolation (with separation of weights)
         {
            jx_PAMGBuildDirInterp( A_array[level], CF_marker_array[level], 
                                par_S, coarse_pnts_global, num_functions, 
                                dof_func_array[level], debug_flag, 
                                trunc_factor, P_max_elmts, 
                                col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         } 
         else if (interp_type == 4)  // multipass interpolation
         {
            jx_PAMGBuildMultipass( A_array[level], CF_marker_array[level], 
                                par_S, coarse_pnts_global, num_functions, 
                                dof_func_array[level], debug_flag,
                                trunc_factor, P_max_elmts, 0, col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 5) // multipass interpolation (with separation of weights)
         {
            jx_PAMGBuildMultipass( A_array[level], CF_marker_array[level], 
                                par_S, coarse_pnts_global, num_functions, 
                                dof_func_array[level], debug_flag,
                                trunc_factor, P_max_elmts, 1, col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 6)  // extended classical modified interpolation
         {
             jx_PAMGBuildExtPIInterp( A_array[level], CF_marker_array[level], 
                                   par_S, coarse_pnts_global, num_functions, 
                                   dof_func_array[level], debug_flag,
                                   trunc_factor, P_max_elmts, col_offd_S_to_A, &par_P );
             jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 7) // extended (if no common C neighbor) classical modified interpolation
         {
            jx_PAMGBuildExtPICCInterp( A_array[level], CF_marker_array[level], 
                                     par_S, coarse_pnts_global, num_functions, 
                                     dof_func_array[level], debug_flag,
                                     trunc_factor, P_max_elmts, col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 8) // standard interpolation
         {
            jx_PAMGBuildStdInterp( A_array[level], CF_marker_array[level], 
                                 par_S, coarse_pnts_global, num_functions, 
                                 dof_func_array[level], debug_flag,
                                 trunc_factor, P_max_elmts, 0, col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 9) // standard interpolation (with separation of weights)
         {
            jx_PAMGBuildStdInterp( A_array[level], CF_marker_array[level], 
                                 par_S, coarse_pnts_global, num_functions, 
                                 dof_func_array[level], debug_flag,
                                 trunc_factor, P_max_elmts, 1, col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         }
         else if (interp_type == 100) /* 1pt interpolation */
         {
            jx_PAMGBuildInterpOnePnt( A_array[level], CF_marker_array[level],
                                    par_S, coarse_pnts_global, num_functions,
                                    dof_func_array[level], debug_flag, col_offd_S_to_A, &par_P );
            jx_TFree(col_offd_S_to_A);
         }
      }
      
      if (debug_flag == 1)
      {
         wall_time = jx_time_getWallclockSeconds() - wall_time;
         jx_printf("Proc = %d    Level = %d    Build Interp Time = %f\n", my_id,level, wall_time);
         fflush(NULL);
      }

      if (wall_time_option == 1)
      {
         wall_time_interp += (jx_time_getWallclockSeconds() - tmp_wall_time);
      }

      P_array[level] = par_P; 
      if (restri_type)
      {
         R_array[level] = R;
         AIR_maxsize_ls[level] = R_max_size;
      }

      if (par_S) 
      {
         jx_ParCSRMatrixDestroy(par_S);
      }
      par_S = NULL;


     /*--------------------------------------------------------------------------------
      *   Build coarse-grid operator, A_array[level+1] by R*A*P
      *-------------------------------------------------------------------------------*/
       
      if (debug_flag == 1) wall_time = jx_time_getWallclockSeconds();

      if (wall_time_option == 1) tmp_wall_time = jx_time_getWallclockSeconds();

      if (restri_type)
      {
         /* Use two matrix products to generate A_H */
         jx_ParCSRMatrix *Q = jx_ParMatmul(A_array[level], P_array[level]);
         A_H = jx_ParMatmul(R_array[level], Q);
         jx_ParCSRMatrixOwnsRowStarts(A_H) = 1;
         jx_ParCSRMatrixOwnsColStarts(A_H) = 0;
         jx_ParCSRMatrixOwnsColStarts(P_array[level]) = 0;
         jx_ParCSRMatrixOwnsRowStarts(R_array[level]) = 0;
         if (num_procs > 1) jx_MatvecCommPkgCreate(A_H);
         /* Delete AP */
         jx_ParCSRMatrixDestroy(Q);
      }
      else if (rap2)
      {
         /* Use two matrix products to generate A_H */
         jx_ParCSRMatrix *Q = jx_ParMatmul(A_array[level], P_array[level]);
         A_H = jx_ParTMatmul(P_array[level], Q);
         jx_ParCSRMatrixOwnsRowStarts(A_H) = 1;
         jx_ParCSRMatrixOwnsColStarts(A_H) = 0;
         jx_ParCSRMatrixOwnsColStarts(P_array[level]) = 0;
         if (num_procs > 1) jx_MatvecCommPkgCreate(A_H);
         /* Delete AP */
         jx_ParCSRMatrixDestroy(Q);
      }
      else if (spmt_rap_type == 1)
      {
         jx_PAMGBuildCoarseOperatorKT( P_array[level], A_array[level], P_array[level], keepTranspose, &A_H );
      }
      else if (spmt_rap_type == 2) /* Yue Xiaoqiang 2012/10/12 */
      {
         jx_PAMGBuildCoarseOperatorOMP( P_array[level], A_array[level], P_array[level], &A_H, opt_icor );
      }
      
      if (debug_flag == 1)
      {
         wall_time = jx_time_getWallclockSeconds() - wall_time;
         jx_printf("Proc = %d    Level = %d    Build Coarse Operator Time = %f\n", my_id,level, wall_time);
         fflush(NULL);
      }

      if (wall_time_option == 1)
      {
         wall_time_rap += (jx_time_getWallclockSeconds() - tmp_wall_time);
      }

      ++ level;

      jx_ParCSRMatrixSetNumNonzeros(A_H);
      jx_ParCSRMatrixSetDNumNonzeros(A_H);
      A_array[level] = A_H;

      if (coarse_size <= coarse_threshold_2) {

         coarse_size_last = coarse_size;

      }

      /* print coarser operator. */
	  if (print_coarse_matrix == 1) {

         jx_sprintf(FileNameCoaMat, "cmat_%d", level);
         jx_ParCSRMatrixPrint(A_array[level], FileNameCoaMat);

      }
      
     /*-------------------------------------------------------------------------------
      *   Switch to CLJP when coarsening slows
      *                                            peghoty  2009/07/09
      *------------------------------------------------------------------------------*/ 
      
      size = ((JX_Real) fine_size )*coarse_ratio;   /* peghoty 2010/04/14 */
      if (coarsen_type > 0 && coarse_size >= (JX_Int) size)
      {
	      coarsen_type = 0;      
      }

     /*--------------------------------------------------------------------------------
      *   How to stop the loop "while"
      *                                            peghoty  2009/07/09
      *--------------------------------------------------------------------------------*/ 
      if ( (level == max_levels-1) || (coarse_size <= coarse_threshold) )
      {
         not_finished_coarsening = 0;

#if 0
         if (ai_measure_type == 1)
         {

           /*--------------------------------------------------------------------------
            *  Get the Strength Matrix 
            *-------------------------------------------------------------------------*/        
            jx_PAMGCreateS(A_array[level],strong_threshold,max_row_sum,num_functions,dof_func_array[level],&par_S);
            col_offd_S_to_A = NULL;
            if (strong_threshold > S_commpkg_switch)
            {
               jx_PAMGCreateSCommPkg(A_array[level], par_S, &col_offd_S_to_A);
            }

           /*--------------------------------------------------------------------------
            *  Get the AI Measure 
            *-------------------------------------------------------------------------*/        
            jx_PAMGMeasureAI(par_S, A_array[level], debug_flag, &AI_measure);

           /*--------------------------------------------
            *  store the AI array
            *-------------------------------------------*/ 
            AI_measure_array[level] = AI_measure;

         }

         /*-------------------------------------------------------------------------
          *  compute ai-information for all-pnts and C-pnts!
          *-------------------------------------------------------------------------*/ 
         local_num_vars = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[level]));

          for (i = 0; i < local_num_vars; i++) {
              num_vars_local++ ;
              //if (AI_measure[i] > 0.0 && AI_measure[i] < 1.0)
              if (AI_measure[i] > 1.0e-10) 
              {
                 mai_local += AI_measure[i];
                 num_ai_local++ ;
                 if (AI_measure[i] > 0.1) {
                    mai_local_valid += AI_measure[i];
                    num_ai_local_valid++ ;
                    mai_c_local += AI_measure[i];
                    num_ai_c_local++;
                 }
              }
          }
#endif
      }

      /*jx_printf("level = %d, not_finished_coarsening = %d \n", level, not_finished_coarsening);
      if (level == 1) {
        num_ai_local_0 = num_ai_local;
        num_ai_local_valid_0 = num_ai_local_valid;
        num_vars_local_0 = num_vars_local;
        num_ai_c_local_0 = num_ai_c_local;
        mai_local_0 = mai_local;
        mai_local_valid_0 = mai_local_valid;
        mai_c_local_0 = mai_c_local;
      }*/

   } // end while loop
   /* STAGE6-EXTRACT-END: tail rebuild loop should return with `level`
    * pointing at the new coarsest level and A_array/P_array/CF filled. */

   /* STAGE6-FINALIZE-BEGIN: this block must be shared by full setup and
    * SetupFromLevel(k) after the tail has been rebuilt. */
   if (amg_print_level > 0)
   {

      jx_MPI_Allreduce(&num_ai_local_0,&num_ai_global,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&num_ai_local_valid_0,&num_ai_global_valid,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&num_vars_local_0,&num_vars_global,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&num_ai_c_local_0,&num_ai_c_global,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&mai_local_0,&mai_global,1,JX_MPI_REAL,MPI_SUM,comm);
      jx_MPI_Allreduce(&mai_local_valid_0,&mai_global_valid,1,JX_MPI_REAL,MPI_SUM,comm);
      jx_MPI_Allreduce(&mai_c_local_0,&mai_c_global,1,JX_MPI_REAL,MPI_SUM,comm);

      if (my_id == 0)
      {
         JX_Real num_ai = num_ai_global;
         JX_Real num_ai_valid = num_ai_global_valid;
         JX_Real num_var = num_vars_global;
         JX_Real num_ai_c = num_ai_c_global;
         jx_printf(" \n");
         jx_printf("Level 0: \n");
         jx_printf(" - num_vars = %d, num_ai = %d, num_ai_valid = %d \n", num_vars_global, num_ai_global, num_ai_global_valid);
         jx_printf(" - num_ai_ratio = %f\n", num_ai/num_var);
         jx_printf(" - num_ai_ratio_valid = %f\n", num_ai_valid/num_var);
         if (num_ai) jx_printf(" - num_ai_c_ratio = %f\n", num_ai_c/num_ai);
         if (num_ai_valid) jx_printf(" - num_ai_c_ratio_valid = %f\n", num_ai_c/num_ai_valid);
         if (mai_global > 0.0) jx_printf(" - measure_ai_ratio_valid = %f\n", mai_global_valid/mai_global);
         if (mai_global > 0.0) jx_printf(" - measure_ai_c_ratio = %f\n", mai_c_global/mai_global);
         if (mai_global_valid > 0.0) jx_printf(" - measure_ai_c_ratio_valid = %f\n", mai_c_global/mai_global_valid);
      }

      jx_MPI_Allreduce(&num_ai_local,&num_ai_global,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&num_ai_local_valid,&num_ai_global_valid,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&num_vars_local,&num_vars_global,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&num_ai_c_local,&num_ai_c_global,1,JX_MPI_INT,MPI_SUM,comm);
      jx_MPI_Allreduce(&mai_local,&mai_global,1,JX_MPI_REAL,MPI_SUM,comm);
      jx_MPI_Allreduce(&mai_local_valid,&mai_global_valid,1,JX_MPI_REAL,MPI_SUM,comm);
      jx_MPI_Allreduce(&mai_c_local,&mai_c_global,1,JX_MPI_REAL,MPI_SUM,comm);

      if (my_id == 0)
      {
         JX_Real num_ai = num_ai_global;
         JX_Real num_ai_valid = num_ai_global_valid;
         JX_Real num_var = num_vars_global;
         JX_Real num_ai_c = num_ai_c_global;
         jx_printf(" \n");
         jx_printf("All levels (except coarsest level): \n");
         jx_printf(" - num_vars = %d, num_ai = %d, num_ai_valid = %d \n", num_vars_global, num_ai_global, num_ai_global_valid);
         jx_printf(" - num_ai_ratio = %f\n", num_ai/num_var);
         jx_printf(" - num_ai_ratio_valid = %f\n", num_ai_valid/num_var);
         if (num_ai>0) jx_printf(" - num_ai_c_ratio = %f\n", num_ai_c/num_ai);
         if (num_ai_valid>0) jx_printf(" - num_ai_c_ratio_valid = %f\n", num_ai_c/num_ai_valid);
         if (mai_global > 0.0) jx_printf(" - measure_ai_ratio_valid = %f\n", mai_global_valid/mai_global);
         if (mai_global > 0.0) jx_printf(" - measure_ai_c_ratio = %f\n", mai_c_global/mai_global);
         if (mai_global_valid > 0.0) jx_printf(" - measure_ai_c_ratio_valid = %f\n", mai_c_global/mai_global_valid);
         jx_printf("================================================================== \n");
      }
   }

   if (level > 0)
   {
      F_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                          jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                          jx_ParCSRMatrixRowStarts(A_array[level]));
      jx_ParVectorInitialize(F_array[level]);
      jx_ParVectorSetPartitioningOwner(F_array[level],0);
         
      U_array[level] = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[level]),
                                          jx_ParCSRMatrixGlobalNumRows(A_array[level]),
                                          jx_ParCSRMatrixRowStarts(A_array[level]));
      jx_ParVectorInitialize(U_array[level]);
      jx_ParVectorSetPartitioningOwner(U_array[level],0);
   }
   

   // 粗空间解法器 Setup, grid_relax_type[3] 表示粗空间松弛类型, zhaoli, 2021.06.30
   switch (grid_relax_type[3])
   {
#if WITH_PARDISO      
      case PARDISO_SOLVER:
         // printf("%s, smooth_type = %d, level = %d\n", __FUNCTION__, grid_relax_type[3], level);
         printf("\nCoarsest level[%d] PARDISO Setup:\n", level);
         jx_CSRMatrix    *AH = jx_ParCSRMatrixToCSRMatrixAll(A_array[level]);
         jx_CSRMatrixSort(AH);
         amg_data->Pardiso_AH = AH;
         jx_pardiso_factorize (AH, &amg_data->pdata, 1);
         break;
#endif      
      default:
         // printf("%s, smooth_type = %d\n", __FUNCTION__, grid_relax_type[3]);
         break;
   }

  /*-----------------------------------------------------------------------
   * enter all the stuff created, A[level], P[level], CF_marker[level],
   * for levels 1 through coarsest, into amg_data data structure
   *-----------------------------------------------------------------------*/

   num_levels = level + 1;
   jx_ParAMGDataNumLevels(amg_data) = num_levels;
   if (jx_ParAMGDataSmoothNumLevels(amg_data) > level)
   {
      jx_ParAMGDataSmoothNumLevels(amg_data) = level;
   }
   smooth_num_levels = jx_ParAMGDataSmoothNumLevels(amg_data);
   
   for (j = 0; j < smooth_num_levels; j++) // Euclid smoothers
   {
      if (smooth_type == 9 || smooth_type == 19)
      {
         JX_EuclidCreate(comm, &smoother[j]);
         if (euclidfile)
         {
            JX_EuclidSetParamsFromFile(smoother[j], euclidfile);
         }
         JX_EuclidSetLevel(smoother[j], eu_level);
         if (eu_bj)
         {
            JX_EuclidSetBJ(smoother[j], eu_bj);
         }
         if (eu_sparse_A)
         {
            JX_EuclidSetSparseA(smoother[j], eu_sparse_A);
         }
         JX_EuclidSetup(smoother[j], (JX_ParCSRMatrix)A_array[j]);
      }
   }

   if (amg_logging > 1) 
   {
      Residual_array = jx_ParVectorCreate(jx_ParCSRMatrixComm(A_array[0]),
                                          jx_ParCSRMatrixGlobalNumRows(A_array[0]),
                                          jx_ParCSRMatrixRowStarts(A_array[0]));
      jx_ParVectorInitialize(Residual_array);
      jx_ParVectorSetPartitioningOwner(Residual_array,0);
      jx_ParAMGDataResidual(amg_data) = Residual_array;
   }
   else
   {
      jx_ParAMGDataResidual(amg_data) = NULL;
   }

   jx_TFree(opt_icor); /* Yue Xiaoqiang 2012/10/12 */
   if (Pmarkers_global_rap)
   {
      jx_TFree(Pmarkers_global_rap); /* Yue Xiaoqiang 2012/10/17 */
   }
   if (Pmarkers_global_interp)
   {
      jx_TFree(Pmarkers_global_interp); /* Yue Xiaoqiang 2012/10/17 */
   }

  /*--------------------------------------------------------------
   *    Print some stuff
   *-------------------------------------------------------------*/
   
   //if (amg_print_level == 1 || amg_print_level == 3)
   if (amg_print_level > 1)
   {
      /* Write the SETUP parameters */
      jx_PAMGSetupStatus(amg_data, par_matrix);
   }

   if (wall_time_option == 1)
   {
      jx_printf("\n\nProc = %d, Coarsen Time = %f\n", my_id, wall_time_coarsen);
      jx_printf("Proc = %d, Build Coarse Operator Time = %f\n", my_id, wall_time_rap);
      jx_printf("Proc = %d, Build Interp Time = %f\n\n", my_id, wall_time_interp);
   }

   if (my_id == 0 && amg_print_level > 1)
   {
      /* Write the SOLVE parameters */
      jx_PAMGSolveStatus(amg_data); /* Yue Xiaoqiang 2014/04/12 */
   }

   JXPAMG_PR_PrintHierarchyShape("after experimental full setup", amg_data);
   JXPAMG_PR_CheckLinkDimensions("after experimental full setup", amg_data);
   /* STAGE6-FINALIZE-END */

   return(Setup_err_flag);
}





/* Algorithm 1: Adaptive Reuse with Diagonal Probe (read-only).
 * This wrapper calls the read-only probe for logging only,
 * then relies on the caller to perform full setup. */
JX_Int
JXPAMG_PR_AdaptiveReuseAlgorithm(jx_ParAMGData *amg_data,
                                 jx_ParCSRMatrix *new_A_fine,
                                 JX_Real theta_pred,
                                 JX_Int Lreuse,
                                 JX_Real delta)
{
   JX_Int cut_k;
   JX_Int my_id = 0;

   if (amg_data == NULL || new_A_fine == NULL) return 1;

   if (JXPAMG_PR_PredictCutReadOnly(amg_data, new_A_fine, theta_pred, Lreuse, delta, &cut_k) != 0)
      return 1;

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(jx_ParAMGDataAArray(amg_data)[0]), &my_id);
   if (my_id == 0)
      jx_printf("[predictive-reuse] probe complete, predicted cut k=%d; fallback to full setup\n", cut_k);

   /* Always return 0 (probe-only success); caller must follow up with full setup. */
   return 0;
}

JX_Int jx_PAMGSetup(void *amg_vdata, jx_ParCSRMatrix *par_matrix) {
   return jx_PAMGSetup_predictive_reuse_experiment(amg_vdata, par_matrix);
}

#if JXPAMG_PR_ENABLE_3B2
#include "par_amg_setup_3b2.inc"
#endif

#if JXPAMG_PR_ENABLE_3B2
#endif
