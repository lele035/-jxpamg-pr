/*
 * Draft wrapper for Stage 6 tail rebuild.
 *
 * This file is compiled through par_amg_setup_predictive_reuse.c, but runtime
 * use remains disabled unless JXPAMG_PR_ENABLE_TAIL_REBUILD is set to 1.
 * It is kept as an include-style draft so the large copied loop stays isolated
 * while mpicc can still syntax-check it.
 */

static JX_Int
JXPAMG_PR_RebuildTailFromLevelDraft(jx_ParAMGData *amg_data, JX_Int k)
{
   JXPAMG_PR_TailContext ctx;
   JX_Int ready;
   JX_Int finalize_status;

   /* Variables copied from the original full setup loop context. */
   MPI_Comm comm;
   jx_ParCSRMatrix **A_array;
   jx_ParVector **F_array;
   jx_ParVector **U_array;
   jx_ParCSRMatrix **P_array;
   jx_ParCSRMatrix **R_array;
   JX_Real **AI_measure_array;
   JX_Int **CF_marker_array;
   JX_Int **dof_func_array;
   JX_Int *AIR_maxsize_ls;

   JX_Int *col_offd_S_to_A;
   JX_Int *col_offd_Sabs_to_A = NULL;
   JX_Real strong_threshold;
   JX_Real AIR_strong_th;
   JX_Real max_row_sum;
   JX_Real trunc_factor;
   JX_Real S_commpkg_switch;

   JX_Int max_levels;
   JX_Int amg_print_level;
   JX_Int debug_flag;
   JX_Int local_num_vars;
   JX_Int P_max_elmts;
   JX_Int R_max_size;

   JX_Real *AI_measure;
   JX_Real *AI_measure2;
   JX_Int *CF_marker;
   JX_Int *CFN_marker = NULL;
   jx_ParCSRMatrix *par_S;
   jx_ParCSRMatrix *Sabs = NULL;
   jx_ParCSRMatrix *par_S2;
   jx_ParCSRMatrix *par_P;
   jx_ParCSRMatrix *R;
   jx_ParCSRMatrix *A_H;

   JX_Int wall_time_option;
   JX_Real wall_time_coarsen = 0.0;
   JX_Real wall_time_rap = 0.0;
   JX_Real wall_time_interp = 0.0;
   JX_Real tmp_wall_time = 0.0;

   JX_Int level;
   JX_Int local_size;
   JX_Int coarse_size_last;
   JX_Int coarse_size;
   JX_Int coarsen_type;
   JX_Int interp_type;
   JX_Int restri_type;
   JX_Int measure_type;
   JX_Int fine_size;
   JX_Int not_finished_coarsening;
   JX_Int coarse_threshold;
   JX_Int coarse_threshold_2;
   JX_Int i;
   JX_Int j;
   JX_Int num_procs;
   JX_Int my_id;
   JX_Int num_threads;
   JX_Int *grid_relax_type;
   JX_Int num_functions;
   JX_Int spmt_rap_type;
   JX_Int ai_measure_type;
   JX_Int num_paths;
   JX_Int agg_num_levels;
   JX_Int agg_interp_type;
   JX_Int agg_P_max_elmts;
   JX_Real agg_trunc_factor;
   JX_Int rap2;
   JX_Int keepTranspose;
   JX_Int print_coarse_matrix;
   JX_Int *opt_icor;
   JX_Int *coarse_dof_func;
   JX_Int *coarse_pnts_global;
   JX_Int *coarse_pnts_global1;
   JX_Real size;
   JX_Real coarse_ratio;
   JX_Real wall_time = 0.0;
   JX_Int measure_type_rlx;
   JX_Int number_syn_rlx;
   JX_Real measure_threshold_rlx;

   char FileNameCoaMat[256];

   /* Tail-local AI statistics. These are not full-hierarchy statistics. */
   JX_Int num_vars_local = 0, num_vars_global;
   JX_Int num_ai_local = 0, num_ai_global;
   JX_Int num_ai_local_valid = 0, num_ai_global_valid;
   JX_Int num_ai_c_local = 0, num_ai_c_global;
   JX_Real mai_local = 0.0, mai_global;
   JX_Real mai_local_valid = 0.0, mai_global_valid;
   JX_Real mai_c_local = 0.0, mai_c_global;

   JX_Int num_vars_local_0 = 0;
   JX_Int num_ai_local_0 = 0;
   JX_Int num_ai_local_valid_0 = 0;
   JX_Int num_ai_c_local_0 = 0;
   JX_Real mai_local_0 = 0.0;
   JX_Real mai_local_valid_0 = 0.0;
   JX_Real mai_c_local_0 = 0.0;
   JX_Real mai_threshold = 0.1;

   if (JXPAMG_PR_InitTailContext(&ctx, amg_data, k))
   {
      return 1;
   }
   ready = JXPAMG_PR_CheckTailContextReady(&ctx);
   if (ready != 0)
   {
      return ready;
   }

#if AMG_CG_ZL
   /* The extracted full-setup loop destroys A_array[0] under AMG_CG_ZL.
    * Do not enable the draft wrapper until that branch is audited. */
   return 1;
#endif

   comm = ctx.comm;
   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   num_threads = jx_NumThreads();

   A_array = ctx.A_array;
   F_array = ctx.F_array;
   U_array = ctx.U_array;
   P_array = ctx.P_array;
   R_array = ctx.R_array;
   AI_measure_array = ctx.AI_measure_array;
   CF_marker_array = ctx.CF_marker_array;
   dof_func_array = ctx.dof_func_array;
   AIR_maxsize_ls = ctx.AIR_maxsize_ls;

   max_levels = ctx.max_levels;
   level = ctx.level;
   local_size = ctx.local_size;
   not_finished_coarsening = ctx.not_finished_coarsening;
   coarse_size_last = ctx.coarse_size_last;
   coarse_threshold_2 = ctx.coarse_threshold_2;

   strong_threshold = ctx.strong_threshold;
   max_row_sum = ctx.max_row_sum;
   trunc_factor = ctx.trunc_factor;
   S_commpkg_switch = ctx.S_commpkg_switch;
   AIR_strong_th = ctx.AIR_strong_th;
   agg_trunc_factor = ctx.agg_trunc_factor;
   coarse_ratio = ctx.coarse_ratio;
   measure_threshold_rlx = ctx.measure_threshold_rlx;

   num_functions = ctx.num_functions;
   coarsen_type = ctx.coarsen_type;
   interp_type = ctx.interp_type;
   restri_type = ctx.restri_type;
   measure_type = ctx.measure_type;
   debug_flag = ctx.debug_flag;
   P_max_elmts = ctx.P_max_elmts;
   agg_num_levels = ctx.agg_num_levels;
   agg_interp_type = ctx.agg_interp_type;
   agg_P_max_elmts = ctx.agg_P_max_elmts;
   num_paths = ctx.num_paths;
   rap2 = ctx.rap2;
   keepTranspose = ctx.keepTranspose;
   spmt_rap_type = ctx.spmt_rap_type;
   print_coarse_matrix = ctx.print_coarse_matrix;
   coarse_threshold = ctx.coarse_threshold;
   measure_type_rlx = ctx.measure_type_rlx;
   number_syn_rlx = ctx.number_syn_rlx;
   grid_relax_type = ctx.grid_relax_type;

   wall_time_option = jx_ParAMGDataWallTimeOption(amg_data);
   amg_print_level = jx_ParAMGDataPrintLevel(amg_data);
   ai_measure_type = jx_ParAMGDataAIMeasureType(amg_data);
   if (coarsen_type == 990 || coarsen_type == 991 || coarsen_type == 993 ||
       coarsen_type == 96 || coarsen_type == 98 || coarsen_type == 910 ||
       coarsen_type == 908 || coarsen_type == 918 || coarsen_type == 928 ||
       coarsen_type == 938 || coarsen_type == 968 ||
       amg_print_level > 0)
   {
      ai_measure_type = 1;
   }

   opt_icor = jx_CTAlloc(JX_Int, 5 * num_threads + 2);
   par_S = NULL;
   Sabs = NULL;
   A_H = NULL;
   col_offd_S_to_A = NULL;
   col_offd_Sabs_to_A = NULL;

#include "stage6_tail_rebuild_loop.inc"

   finalize_status = JXPAMG_PR_FinalizeTailRebuildCore(amg_data, level);
   if (finalize_status == 0)
   {
      JXPAMG_PR_PrintHierarchyShape("after experimental tail rebuild", amg_data);
      JXPAMG_PR_CheckLinkDimensions("after experimental tail rebuild", amg_data);
   }
   jx_TFree(opt_icor);
   return finalize_status;
}
