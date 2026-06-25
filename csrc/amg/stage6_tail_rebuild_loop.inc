/*
 * Extracted Stage 6 coarsening loop candidate.
 *
 * Source: par_amg_setup_predictive_reuse.c between
 * STAGE6-EXTRACT-BEGIN and STAGE6-EXTRACT-END.
 *
 * This file is not compiled. It is a working copy used to wrap the copied
 * full-setup coarsening loop into JXPAMG_PR_RebuildTailFromLevel(k).
 */

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

