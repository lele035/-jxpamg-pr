//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_cycle.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"


/*!
 * \fn JX_Int jx_PAMGCycle
 * \brief Cycle of PAMG.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGCycle( void           *amg_vdata, 
              jx_ParVector  **F_array,
              jx_ParVector  **U_array  )
{
   jx_ParAMGData *amg_data = amg_vdata;
   MPI_Comm       comm;

   jx_ParVector *Aux_U;
   jx_ParVector *Aux_F;
   jx_ParVector *Ttemp = NULL;
   
   JX_Solver *smoother;
 
   JX_Real *num_coeffs;

   /* Local variables  */ 
   JX_Int      *lev_counter;
   JX_Int       Solve_err_flag;
   JX_Int       i, j, k;
   JX_Int       level;
   JX_Int       cycle_param;
   JX_Int       coarse_grid;
   JX_Int       fine_grid;
   JX_Int       Not_Finished;
   JX_Int       num_sweep;
   JX_Int       relax_type;
   JX_Int       relax_points = 0;
   JX_Int       relax_local;
   JX_Int       old_version = 0;
   JX_Int       local_size = 0;
   JX_Real    alpha, beta;
   JX_Int     **relax_marker_ai = NULL;
   JX_Int     **relax_marker_ess = NULL;
   JX_Int       n, num_ai_th_step, num_ai_th, num_ess_th;
   JX_Real    ai_measure_th_min = 0.2;
   JX_Real    ai_measure_th_max = 2.01;
//   JX_Real    ai_factor = 0.5;
//   JX_Real    ai_measure_th, ai_measure_th_l, ai_measure_th_h;
   JX_Real    ai_measure_th_l, ai_measure_th_h;
   JX_Real    ai_measure;


  /*----------------------------------------------------------------------
   * Get the amg_data data
   *---------------------------------------------------------------------*/
   JX_Int      wall_time_option = jx_ParAMGDataWallTimeOption(amg_data);  /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Int      num_levels  = jx_ParAMGDataNumLevels(amg_data);
   JX_Int      max_levels  = jx_ParAMGDataMaxLevels(amg_data); 
   JX_Int      cycle_type  = jx_ParAMGDataCycleType(amg_data);
   JX_Int      restri_type = jx_ParAMGDataRestriction(amg_data);
   JX_Int      smooth_type = jx_ParAMGDataSmoothType(amg_data);
   JX_Int      smooth_num_levels = jx_ParAMGDataSmoothNumLevels(amg_data);
   JX_Real   wall_time_coarsest = 0.0;    /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real   wall_time_prolong = 0.0;     /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real   wall_time_restrict = 0.0;    /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real   wall_time_relaxation = 0.0;  /* newly added Yue Xiaoqiang 2015/09/30 */
   JX_Real   tmp_wall_time = 0.0;         /* newly added Yue Xiaoqiang 2015/09/30 */

   JX_Int     *num_grid_sweeps   = jx_ParAMGDataNumGridSweeps(amg_data);  
   JX_Int     *grid_relax_type   = jx_ParAMGDataGridRelaxType(amg_data);
   JX_Int    **grid_relax_points = jx_ParAMGDataGridRelaxPoints(amg_data);
   JX_Real  *relax_weight      = jx_ParAMGDataRelaxWeight(amg_data);
   JX_Real  *omega             = jx_ParAMGDataOmega(amg_data); 
   JX_Real   cycle_op_count    = jx_ParAMGDataCycleOpCount(amg_data);
   JX_Real **AI_measure_array  = jx_ParAMGDataAIMeasureArray(amg_data);
   JX_Int    **CF_marker_array   = jx_ParAMGDataCFMarkerArray(amg_data);
   JX_Int      relax_order       = jx_ParAMGDataRelaxOrder(amg_data);

   jx_ParCSRMatrix  **A_array = jx_ParAMGDataAArray(amg_data);
   jx_ParCSRMatrix  **P_array = jx_ParAMGDataPArray(amg_data);
   jx_ParCSRMatrix  **R_array = jx_ParAMGDataRArray(amg_data);
   jx_ParVector      *Vtemp   = jx_ParAMGDataVtemp(amg_data);

   /* for ai-prior smoothing .*/
   jx_Vector               *local_vector_ai = NULL;
   JX_Real                  *local_data_ai = NULL;
   jx_Vector               *local_vector_ess = NULL;
   JX_Real                  *local_data_ess = NULL;
   JX_Real   relative       = 0.0;
   JX_Real   rhs_norm       = 0.0;
   JX_Real   resid_nrm      = 0.0;
   JX_Real   resid_nrm_init = 0.0;
   JX_Real   resid_nrm_0 = 0.0;
   JX_Real   relative_ai       = 0.0;
   JX_Real   rhs_norm_ai       = 0.0;
   JX_Real   resid_nrm_ai      = 0.0;
   JX_Real   resid_nrm_init_ai = 0.0;
   JX_Real   relative_ess       = 0.0;
   JX_Real   rhs_norm_ess       = 0.0;
   JX_Real   resid_nrm_ess      = 0.0;
   JX_Real   resid_nrm_init_ess = 0.0;
 
   JX_Int           *partitioning   = NULL;
   jx_ParVector  *Utemp    = NULL;
   JX_Int           size;
   JX_Int           my_id;
   JX_Int           conv_criteria = 0;  /* relative residual Norm-2 */
   JX_Int           print_level = 0, print_level_ai = 0, print_level_ess = 0;
   JX_Real        tol = 1.0e-6;
//   JX_Real        convfac_threshold = 0.95;
   JX_Real        convfac_threshold_ai = 0.3;
   JX_Real        convfac_threshold_ess = 0.3;
   JX_Real        conv_factor, conv_factor_ai, conv_factor_ess;
   JX_Int           iter, iter_ai, iter_ess, iter_ai_tot, iter_ess_tot;
   JX_Int           ai_relax_type = jx_ParAMGDataAIRelaxType(amg_data);

   lev_counter = jx_CTAlloc(JX_Int, num_levels);

   /* Initialize */
   Solve_err_flag = 0;
   comm = jx_ParCSRMatrixComm(A_array[0]);

   if (grid_relax_points) 
   {
      old_version = 1;
   }

   num_coeffs  = jx_CTAlloc(JX_Real, num_levels);
   num_coeffs[0] = (JX_Real)jx_ParCSRMatrixNumNonzeros(A_array[0]);

   for (j = 1; j < num_levels; j ++)
   {
      num_coeffs[j] = (JX_Real)jx_ParCSRMatrixNumNonzeros(A_array[j]);
   }


  /*---------------------------------------------------------------------
   *    Initialize cycling control counter
   *
   *     Cycling is controlled using a level counter: lev_counter[k]
   *     
   *     Each time relaxation is performed on level k, the
   *     counter is decremented by 1. If the counter is then
   *     negative, we go to the next finer level. If non-
   *     negative, we go to the next coarser level. The
   *     following actions control cycling:
   *     
   *     a. lev_counter[0] is initialized to 1.
   *     b. lev_counter[k] is initialized to cycle_type for k>0.
   *     
   *     c. During cycling, when going down to level k, lev_counter[k]
   *        is set to the max of (lev_counter[k],cycle_type)
   *---------------------------------------------------------------------*/

   Not_Finished = 1;

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++ k) 
   {
      lev_counter[k] = cycle_type;
   }

   if (ai_relax_type == 1) { // only for AI-AMG Yue Xiaoqiang 2014/07/03

      //relax_order = 0; // temporal setting by xu 2013/05/30.
      //old_version = 0; // temporal setting by xu 2013/05/30.
      
      /* initialize arrays for AI-prior smoothing. */
      relax_marker_ai = jx_CTAlloc(JX_Int*, num_levels);
      relax_marker_ess = jx_CTAlloc(JX_Int*, num_levels);
      for (k = 0; k < num_levels; k++) 
      {
         n = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[k]));
         relax_marker_ai[k] = jx_CTAlloc(JX_Int, n);
         relax_marker_ess[k] = jx_CTAlloc(JX_Int, n);
         for (i=0; i<n; i++) {
             relax_marker_ai[k][i] = 0;
             relax_marker_ess[k][i] = 1;
         }
      }
      alpha = -1.0; beta = 1.0;

   }

   level = 0;
   cycle_param = 1;

   smoother = jx_ParAMGDataSmoother(amg_data);

   if (smooth_num_levels > 0)
   {
      if (smooth_type == 9 || smooth_type == 19)
      {
         Ttemp = jx_ParVectorCreate(comm, jx_ParVectorGlobalSize(Vtemp),
                           jx_ParVectorPartitioning(Vtemp));
         jx_ParVectorOwnsPartitioning(Ttemp) = 0;
         jx_ParVectorInitialize(Ttemp);
      }
   }

  /*---------------------------------------------------------------------
   * Main loop of cycling
   *--------------------------------------------------------------------*/
  
   while (Not_Finished)
   {
      if (num_levels > 1) 
      {
         local_size = jx_VectorSize(jx_ParVectorLocalVector(F_array[level]));
         jx_VectorSize(jx_ParVectorLocalVector(Vtemp)) = local_size;

         num_sweep = num_grid_sweeps[cycle_param];
         Aux_U = U_array[level];
         Aux_F = F_array[level];

         relax_type = grid_relax_type[cycle_param];

         /* initialize Utemp. */
         comm = jx_ParCSRMatrixComm(A_array[level]);
         jx_MPI_Comm_rank(comm, &my_id);

         partitioning = jx_ParVectorPartitioning(Aux_F);
         size = jx_ParVectorGlobalSize(Aux_F);
         Utemp = jx_ParVectorCreate(comm, size, partitioning);
         jx_ParVectorSetPartitioningOwner(Utemp, 0);
         jx_ParVectorInitialize(Utemp);
      }
      else /* AB: 4/08: removed the max_levels > 1 check - should do this when max-levels = 1 also */
      {
         /* If no coarsening occurred, apply a simple smoother once */
         Aux_U = U_array[level];
         Aux_F = F_array[level];
         num_sweep  = 1;
         /* TK: Use the user relax type (instead of 0) to allow for setting a
            convergent smoother (e.g. in the solution of singular problems). */
         //relax_type = 0;
         relax_type = jx_ParAMGDataUserRelaxType(amg_data);
         if (relax_type == -1) relax_type = 6;
      }

      if (ai_relax_type == 0) { // not for AI-AMG Yue Xiaoqiang 2014/07/03

         if (wall_time_option == 1) tmp_wall_time = jx_time_getWallclockSeconds();

        /*------------------------------------------------------------------
         * Do the relaxation num_sweep times
         *-----------------------------------------------------------------*/
         for (j = 0; j < num_sweep; j ++)
         {
            if (num_levels == 1 && max_levels > 1)
            {
               relax_points = 0;
               relax_local = 0;
            }
            else
            {
               if (old_version)
               {
                  relax_points = grid_relax_points[cycle_param][j];
               }
               relax_local = relax_order;
            }

           /*----------------------------------------------------
            * VERY sloppy approximation to cycle complexity
            *---------------------------------------------------*/

            if (old_version && level < num_levels - 1)
            {
               switch (relax_points)
               {
                  case 1:
                  cycle_op_count += num_coeffs[level+1];
                  break;
  
                  case -1: 
                  cycle_op_count += (num_coeffs[level]-num_coeffs[level+1]); 
                  break;
               }
            }
            else
            {
               cycle_op_count += num_coeffs[level]; 
            }

            if (smooth_num_levels > level && (smooth_type == 9 || smooth_type == 19))
            {
               jx_VectorSize(jx_ParVectorLocalVector(Utemp)) = local_size;
               jx_ParVectorCopy(Aux_F, Vtemp);
               jx_ParCSRMatrixMatvec(-1.0, A_array[level], U_array[level], 1.0, Vtemp);
               JX_EuclidSolve(smoother[level], (JX_ParCSRMatrix) A_array[level],
                                         (JX_ParVector) Vtemp, (JX_ParVector) Utemp);
               jx_ParVectorAxpy(relax_weight[level], Utemp, Aux_U);
            }
            else if (old_version)
            {
               // printf("jx_PAMGRelax, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
#if WITH_PARDISO                
               if (relax_type != PARDISO_SOLVER)
               {
#endif                  
                   Solve_err_flag = jx_PAMGRelax( A_array[level], 
                                              Aux_F,
                                              CF_marker_array[level],
                                              relax_type,
                                              relax_points,
                                              relax_weight[level],
                                              omega[level],
                                              Aux_U,
                                              Vtemp );
#if WITH_PARDISO                                              
               }else // pardiso zhaoli, 2021.06.30
               {
                  // printf("jx_PAMGRelax10(Direct Solve), %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
                  jx_Vector   *u_local  = jx_ParVectorLocalVector(Aux_U);
                  jx_Vector   *f_vector = jx_ParVectorToVectorAll(Aux_F);
                  Solve_err_flag = jx_pardiso_solve (amg_data->Pardiso_AH, f_vector, u_local, &amg_data->pdata, 0);
                  jx_SeqVectorDestroy(f_vector);
                  f_vector = NULL;
               }             
#endif               
              
            }
            else
            {
#if WITH_PARDISO                
               if (relax_type != PARDISO_SOLVER)
               {
#endif          
               // printf("jx_PAMGRelaxIF, level:%d, %s, %d\n", level, __FUNCTION__, __LINE__);       
               Solve_err_flag = jx_PAMGRelaxIF( A_array[level], 
                                                Aux_F,
                                                CF_marker_array[level],
                                                relax_type,
                                                relax_local,
                                                cycle_param,
                                                relax_weight[level],
                                                omega[level],
                                                Aux_U,
                                                Vtemp );
#if WITH_PARDISO                                              
               }else // pardiso zhaoli, 2021.06.30
               {
                  // printf("jx_PAMGRelax10, level: %d, %s, %d\n", level, __FUNCTION__, __LINE__);
                  // jx_CSRMatrix *A_CSR    = jx_ParCSRMatrixToCSRMatrixAll(A_array[level]);
                  jx_Vector    *u_local  = jx_ParVectorLocalVector(Aux_U);
                  jx_Vector    *f_vector = jx_ParVectorToVectorAll(Aux_F);
                  // jx_CSRMatrixSort(A_CSR);
                  // Solve_err_flag = jx_pardiso_solve (A_CSR, f_vector, u_local, &amg_data->pdata, 0);
                  Solve_err_flag = jx_pardiso_solve (amg_data->Pardiso_AH, f_vector, u_local, &amg_data->pdata, 0);
                  jx_SeqVectorDestroy(f_vector);
                  f_vector = NULL;
               }             
#endif                                                
            }
    
            if (Solve_err_flag != 0) 
            {
               return(Solve_err_flag);
            }
         }

         if (wall_time_option == 1)
         {
            if (relax_type == 9)
            {
               wall_time_coarsest += (jx_time_getWallclockSeconds() - tmp_wall_time);
            }
            else
            {
               wall_time_relaxation += (jx_time_getWallclockSeconds() - tmp_wall_time);
            }
         }
      }
      else if (ai_relax_type == 1) { // only for AI-AMG Yue Xiaoqiang 2014/07/03

         //jx_printf("level = %d, relax_type in amg_cycle = %d\n", level, relax_type);

         iter = 0;
         conv_factor = 0.0;

         /* local rhs_norm */
         jx_ParVectorCopy(Aux_F, Vtemp);
         rhs_norm = jx_ParVectorNorm2(Vtemp);

         /* compute initial local resid_nrm. */
         jx_ParVectorCopy(Aux_F, Vtemp);
         alpha = -1.0; beta = 1.0;
         jx_ParCSRMatrixMatvec(alpha, A_array[level], Aux_U, beta, Vtemp);
         resid_nrm = jx_ParVectorNorm2(Vtemp);
         resid_nrm_init = resid_nrm;
         //jx_printf("rhs_norm = %f\n", rhs_norm);
         //jx_printf("resid_nrm_init = %f\n", resid_nrm_init);
         //jx_printf("relax_type = %d\n", relax_type);
         relative = 2*tol;

        /*------------------------------------------------------------------
         * Do the relaxation num_sweep times
         *-----------------------------------------------------------------*/

         if (relax_type >=90 ) {

            relax_type = relax_type - 90;

            n = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(A_array[level]));

            // following code need to be optimized.
            //ai_measure_th = ai_measure_th_max;
            ai_measure_th_l = ai_measure_th_min;
            ai_measure_th_h = ai_measure_th_max;
            num_ai_th = 0;
            num_ess_th = n;

            num_ai_th_step = 0;
            for (i=0; i<n; i++) {
               ai_measure = AI_measure_array[level][i];
               if  ( ai_measure >= ai_measure_th_l && ai_measure <= ai_measure_th_h 
                  //&& CF_marker_array[level][i] == 1
                   ) {
                   relax_marker_ai[level][i] = 1;
                   relax_marker_ess[level][i] = 0;
                   num_ai_th++;
                   num_ai_th_step++;
                   num_ess_th--;
               }
            }

#if 1 
            /* local rhs_norm */
            jx_ParVectorCopy(Aux_F, Vtemp);
            jx_ParVectorCopy(Aux_F, Utemp);

            local_vector_ai = jx_ParVectorLocalVector(Vtemp);
            local_data_ai   = jx_VectorData(local_vector_ai);
            local_size   = jx_VectorSize(local_vector_ai);

            local_vector_ess = jx_ParVectorLocalVector(Utemp);
            local_data_ess   = jx_VectorData(local_vector_ess);

            if (local_size != n) jx_printf("ERROR: local_size != n. \n");
            for (i=0; i<local_size; i++) {
                if (relax_marker_ai[level][i] == 0) {
                   local_data_ai[i] = 0.0;
                } else if (relax_marker_ess[level][i] == 0){
                   local_data_ess[i] = 0.0;
                } else {
                   jx_printf("ERROR: relax_marker_ai does not match relax_marker_ess.\n");
                   exit(0);
                }
            }
            rhs_norm_ai = jx_ParVectorNorm2(Vtemp);
            rhs_norm_ess = jx_ParVectorNorm2(Utemp);

#endif

            iter_ai_tot = 0;
            iter_ess_tot = 0;
            //jx_printf(" level = %d, num_points = %d, num_ai_points = %d, num_ess_pionts = %d \n",
            //         level, n, num_ai_th, num_ess_th);
            //for (j = 0; j < num_sweep; j ++)
            while ( iter < num_sweep && relative >= tol)
            //while ( iter < num_sweep && conv_factor <= convfac_threshold)
            { 
#if 1 
               ++iter;
               resid_nrm_0 = resid_nrm;
               /* compute initial local resid_nrm: ai-parts. */
               jx_ParVectorCopy(Aux_F, Vtemp);
               alpha = -1.0; beta = 1.0;
               jx_ParCSRMatrixMatvec(alpha, A_array[level], Aux_U, beta, Vtemp);

               local_vector_ai = jx_ParVectorLocalVector(Vtemp);
               local_data_ai   = jx_VectorData(local_vector_ai);
               local_size   = jx_VectorSize(local_vector_ai);

               for (i=0; i<local_size; i++) {
                   if (relax_marker_ai[level][i] == 0) {
                      local_data_ai[i] = 0.0;
                   }
               }
               resid_nrm_ai = jx_ParVectorNorm2(Vtemp);
               resid_nrm_init_ai = resid_nrm_ai;

               if (rhs_norm_ai)
               {
                  relative_ai = resid_nrm_init_ai / rhs_norm_ai;
               }
               else
               {
                  relative_ai = resid_nrm_init_ai;
               }
               relative_ai = 2*tol;

#endif 
               /* AI-part smoothing. */
               conv_factor_ai = 0.0;
               iter_ai = 0;
               if (print_level_ai) jx_printf("Initial_ai    %e                %e\n", resid_nrm_init_ai, relative_ai);
               //jx_printf("relative_ai = %f, num_ai_th = %d, conv_factor_ai = %f \n",
               //        relative_ai, num_ai_th, conv_factor_ai);
               while ( iter_ai < 1  && relative_ai >= tol && num_ai_th>0 && conv_factor_ai <= convfac_threshold_ai)
               //while ( iter_ai < 1 )
               {
                  //jx_printf("ai-part smoothing...\n");
                  if (num_levels == 1 && max_levels > 1)
                  {
                     relax_points = 0;
                     relax_local = 0;
                  }
                  else
                  {
                     if (old_version)
                     {
                        relax_points = grid_relax_points[cycle_param][j];
                     }
                     relax_local = relax_order;
                  }

                  /*----------------------------------------------------
                   * VERY sloppy approximation to cycle complexity
                   *---------------------------------------------------*/

                  if (old_version && level < num_levels - 1)
                  {
                     switch (relax_points)
                     {
                        case 1:
                        cycle_op_count += (num_coeffs[level+1]*num_ai_th)/n; 
                        break;

                        case -1: 
                        cycle_op_count += ((num_coeffs[level]-num_coeffs[level+1])*num_ai_th)/n;
                        break;
                     }
                  }
                  else
                  {
                     cycle_op_count += (num_coeffs[level]*num_ai_th)/n; 
                  }

                  //relax_local = 0; // temporal setting by xu 2013/05/30.
                  if (old_version)
                  {
                     Solve_err_flag = jx_PAMGRelaxAI( A_array[level], 
                                                      Aux_F,
                                                      CF_marker_array[level],
                                                      relax_marker_ai[level],
                                                      relax_type,
                                                      relax_points,
                                                      relax_weight[level],
                                                      omega[level],
                                                      Aux_U,
                                                      Vtemp );
                  }
                  else
                  {
                     Solve_err_flag = jx_PAMGRelaxIFAI( A_array[level], 
                                                        Aux_F,
                                                        CF_marker_array[level],
                                                        relax_marker_ai[level],
                                                        relax_type,
                                                        relax_local,
                                                        cycle_param,
                                                        relax_weight[level],
                                                        omega[level],
                                                        Aux_U,
                                                        Vtemp );
                  }

                  if (Solve_err_flag != 0) 
                  {
                     return(Solve_err_flag);
                  }

                  ++ iter_ai;

                  jx_check_convergence( relax_marker_ai[level], A_array[level], Aux_F,
                                        alpha, beta, iter_ai, Aux_U, Vtemp, NULL,
                                        my_id, print_level_ai, conv_criteria, rhs_norm_ai, resid_nrm_init_ai,
                                        &resid_nrm_ai, &relative_ai, &conv_factor_ai );

                  //ai_measure_th = ai_measure_th_l;
                  //ai_measure_th_h = ai_measure_th_l;
                  //ai_factor = 0.1;
               } // end for j: AI-parts.
      
#if 1 
               /* compute initial local resid_nrm: ess-part. */
               jx_ParVectorCopy(Aux_F, Vtemp);
               alpha = -1.0; beta = 1.0;
               jx_ParCSRMatrixMatvec(alpha, A_array[level], Aux_U, beta, Vtemp);

               local_vector_ess = jx_ParVectorLocalVector(Vtemp);
               local_data_ess   = jx_VectorData(local_vector_ess);
               local_size   = jx_VectorSize(local_vector_ess);

               for (i=0; i<local_size; i++) {
                   if (relax_marker_ess[level][i] == 0) {
                      local_data_ess[i] = 0.0;
                   }
               }
               resid_nrm_ess = jx_ParVectorNorm2(Vtemp);
               resid_nrm_init_ess = resid_nrm_ess;

               if (rhs_norm_ess)
               {
                  relative_ess = resid_nrm_init_ess / rhs_norm_ess;
               }
               else
               {
                  relative_ess = resid_nrm_init_ess;
               }
               relative_ess = 2*tol;

#endif 
               /* ESS-part smoothing. */
               conv_factor_ess = 0.0;
               iter_ess = 0;
               if (print_level_ess) jx_printf("Initial_ess    %e                %e\n",
                resid_nrm_init_ess, relative_ess);
               while ( iter_ess < 1 && relative_ess >= tol &&
                       num_ess_th>0 && conv_factor_ess <= convfac_threshold_ess)
               //while ( iter_ess < 6)
               {
                  if (num_levels == 1 && max_levels > 1)
                  {
                     relax_points = 0;
                     relax_local = 0;
                  }
                  else
                  {
                     if (old_version)
                     {
                        relax_points = grid_relax_points[cycle_param][j];
                     }
                     relax_local = relax_order;
                  }

                 /*----------------------------------------------------
                  * VERY sloppy approximation to cycle complexity
                  *---------------------------------------------------*/

                  if (old_version && level < num_levels - 1)
                  {
                     switch (relax_points)
                     {
                        case 1:
                        cycle_op_count += (num_coeffs[level+1]*num_ess_th)/n;
                        break;
  
                        case -1: 
                        cycle_op_count += ((num_coeffs[level]-num_coeffs[level+1])*num_ess_th)/n;
                        break;
                     }
                  }
                  else
                  {
                     cycle_op_count += (num_coeffs[level]*num_ess_th)/n;
                  }

                  //relax_local = 0; // temporal setting by xu 2013/05/30.
                  if (old_version)
                  {
                     Solve_err_flag = jx_PAMGRelaxAI( A_array[level], 
                                                      Aux_F,
                                                      CF_marker_array[level],
                                                      relax_marker_ess[level],
                                                      relax_type,
                                                      relax_points,
                                                      relax_weight[level],
                                                      omega[level],
                                                      Aux_U,
                                                      Vtemp );
                  }
                  else
                  {
                     Solve_err_flag = jx_PAMGRelaxIFAI( A_array[level], 
                                                        Aux_F,
                                                        CF_marker_array[level],
                                                        relax_marker_ess[level],
                                                        relax_type,
                                                        relax_local,
                                                        cycle_param,
                                                        relax_weight[level],
                                                        omega[level],
                                                        Aux_U,
                                                        Vtemp );
                  }
  
                  if (Solve_err_flag != 0) 
                  {
                     return(Solve_err_flag);
                  }
                  ++ iter_ess;

                  jx_check_convergence( relax_marker_ess[level], A_array[level], Aux_F,
                                        alpha,beta, iter_ess, Aux_U, Vtemp,NULL, 
                                        my_id, print_level_ess, conv_criteria, rhs_norm_ess, resid_nrm_init_ess,
                                        &resid_nrm_ess, &relative_ess, &conv_factor_ess );

               } //end for j: ESS-parts.

               iter_ai_tot = iter_ai_tot + iter_ai;
               iter_ess_tot = iter_ess_tot + iter_ess;
               //jx_printf("iter = %d, iter_ai = %d, iter_ess = %d\n", iter, iter_ai, iter_ess);
               jx_check_convergence( NULL, A_array[level], Aux_F, alpha, beta, iter, Aux_U, Vtemp, NULL, 
                                     my_id, print_level, conv_criteria, rhs_norm, resid_nrm_init,
                                     &resid_nrm_0, &relative, &conv_factor );
               resid_nrm = resid_nrm_0;

            } // end for j.

            //jx_printf("iter_ai_tot = %d, iter_ess_tot = %d\n", iter_ai_tot, iter_ess_tot);
         } // end for ai-prior relax.
         /* for general smoothing strategy. */
         else 
         { 

            //for (j = 0; j < num_sweep; j ++)
            //while ( iter < num_sweep && conv_factor <= convfac_threshold)
            while ( iter < num_sweep && relative >= tol)
            {
               ++iter;
               resid_nrm_0 = resid_nrm;
               if (num_levels == 1 && max_levels > 1)
               {
                  relax_points = 0;
                  relax_local = 0;
               }
               else
               {
                  if (old_version)
                  {
                     relax_points = grid_relax_points[cycle_param][j];
                  }
                  relax_local = relax_order;
               }

              /*----------------------------------------------------
               * VERY sloppy approximation to cycle complexity
               *---------------------------------------------------*/

               if (old_version && level < num_levels - 1)
               {
                  switch (relax_points)
                  {
                     case 1:
                     cycle_op_count += num_coeffs[level+1];
                     break;

                     case -1: 
                     cycle_op_count += (num_coeffs[level]-num_coeffs[level+1]); 
                     break;
                  }
               }
               else
               {
                  cycle_op_count += num_coeffs[level]; 
               }

               //relax_local = 0; // temporal setting by xu 2013/05/30.
               //jx_printf("old_version = %d, cycle_param = %d, relax_type = %d\n",
               //        old_version, cycle_param, relax_type);
               /* Choose Smoother */
               if (smooth_num_levels > level && (smooth_type == 9 || smooth_type == 19))
               {
                  jx_VectorSize(jx_ParVectorLocalVector(Utemp)) = local_size;
                  jx_ParVectorCopy(Aux_F, Vtemp);
                  jx_ParCSRMatrixMatvec(-1.0, A_array[level], U_array[level], 1.0, Vtemp);
                  JX_EuclidSolve(smoother[level], (JX_ParCSRMatrix) A_array[level],
                                            (JX_ParVector) Vtemp, (JX_ParVector) Utemp);
                  jx_ParVectorAxpy(relax_weight[level], Utemp, Aux_U);
               }
               else if (old_version)
               {
                  Solve_err_flag = jx_PAMGRelax( A_array[level], 
                                                 Aux_F,
                                                 CF_marker_array[level],
                                                 relax_type,
                                                 relax_points,
                                                 relax_weight[level],
                                                 omega[level],
                                                 Aux_U,
                                                 Vtemp );
               }
               else
               {
                  Solve_err_flag = jx_PAMGRelaxIF( A_array[level], 
                                                   Aux_F,
                                                   CF_marker_array[level],
                                                   relax_type,
                                                   relax_local,
                                                   cycle_param,
                                                   relax_weight[level],
                                                   omega[level],
                                                   Aux_U,
                                                   Vtemp );
               }

               if (Solve_err_flag != 0) 
               {
                  return(Solve_err_flag);
               }
               jx_check_convergence( NULL, A_array[level], Aux_F, alpha, beta, iter, Aux_U, Vtemp, NULL, 
                                     my_id, print_level, conv_criteria, rhs_norm, resid_nrm_init,
                                     &resid_nrm_0, &relative, &conv_factor );
               resid_nrm = resid_nrm_0;

            } //end for j

         }
      }
      //jx_printf("iters of smoother = %d\n", iter);

     /*--------------------------------------------------------------------------
      * Decrement the control counter and determine which grid to visit next
      *------------------------------------------------------------------------*/

      -- lev_counter[level];
       
      if (lev_counter[level] >= 0 && level != num_levels-1)
      {                 
        
        /*---------------------------------------------------------------
         * Visit coarser level next.  
         * Compute residual using jx_ParCSRMatrixMatvec.
         * Perform restriction using jx_ParCSRMatrixMatvecT.
         * Reset counters and cycling parameters for coarse level
         *--------------------------------------------------------------*/
         fine_grid = level;
         coarse_grid = level + 1;

         if (wall_time_option == 1) tmp_wall_time = jx_time_getWallclockSeconds();

         jx_ParVectorSetConstantValues(U_array[coarse_grid], 0.0); 
         
         jx_ParVectorCopy(F_array[fine_grid], Vtemp);
         
         alpha = -1.0; beta = 1.0;
         jx_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid], beta, Vtemp);

         alpha = 1.0; beta = 0.0;
         if (restri_type)
         {
            /* No transpose for R */
            jx_ParCSRMatrixMatvec(alpha, R_array[fine_grid], Vtemp, beta, F_array[coarse_grid]);
         }
         else
         {
            jx_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp, beta, F_array[coarse_grid]);
         }

         if (wall_time_option == 1)
         {
            wall_time_restrict += (jx_time_getWallclockSeconds() - tmp_wall_time);
         }

         ++ level;

         lev_counter[level] = jx_max(lev_counter[level], cycle_type);
         
         cycle_param = 1;
         
         if (level == num_levels-1) 
         {
            cycle_param = 3;
         }
      }

      else if (level != 0)
      {
         
        /*-------------------------------------------------------------------
         * Visit finer level next.
         * Interpolate and add correction using jx_ParCSRMatrixMatvec.
         * Reset counters and cycling parameters for finer level.
         *------------------------------------------------------------------*/
         fine_grid = level - 1;
         coarse_grid = level;
         
         if (wall_time_option == 1) tmp_wall_time = jx_time_getWallclockSeconds();

         alpha = 1.0; beta = 1.0;
         jx_ParCSRMatrixMatvec(alpha, P_array[fine_grid], U_array[coarse_grid], beta, U_array[fine_grid]);

         if (wall_time_option == 1)
         {
            wall_time_prolong += (jx_time_getWallclockSeconds() - tmp_wall_time);
         }

         -- level;

         cycle_param = 2;
      }
      else
      {
         Not_Finished = 0;
      }

      jx_ParVectorDestroy(Utemp);

   }

   jx_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

   if (ai_relax_type == 1) { // only for AI-AMG Yue Xiaoqiang 2014/07/03

      for (k = 0; k < num_levels; k++) 
      {
         jx_TFree(relax_marker_ai[k]);
         jx_TFree(relax_marker_ess[k]);
      }
      jx_TFree(relax_marker_ai);
      jx_TFree(relax_marker_ess);

   }

   jx_TFree(lev_counter);
   jx_TFree(num_coeffs);
   
   if (smooth_num_levels > 0)
   {
     if (smooth_type == 9 || smooth_type == 19)
     {
       jx_ParVectorDestroy(Ttemp);
     }
   }

   if (wall_time_option == 1)
   {
      jx_MPI_Comm_rank(jx_ParCSRMatrixComm(A_array[0]), &my_id);
      jx_printf("\n\nProc = %d, Relaxation except Coarsest Level Time = %f\n", my_id, wall_time_relaxation);
      jx_printf("Proc = %d, Relaxation on Coarsest Level Time = %f\n", my_id, wall_time_coarsest);
      jx_printf("Proc = %d, Prolongation and Correction Time = %f\n", my_id, wall_time_prolong);
      jx_printf("Proc = %d, Residual and Restriction Time = %f\n\n", my_id, wall_time_restrict);
   }

   return(Solve_err_flag);
}
