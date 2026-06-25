//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_amg_solver.c -- SOLUTION phase of PAMG.
 *  Date: 2011/09/27
 */ 
 
#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGSolve
 * \brief Solution phase of PAMG.
 * \param amg_vdata pointer to the amgdata object
 * \param par_matrix pointer to the coefficient matrix
 * \param par_rhs pointer to the right hand side vector
 * \param par_app pointer to the approxamation vector 
 * \remark (peghoty  2009/07/24)
 *   Compared with the hypre code, there are two main alterations in 
 *   this function as follows:
 *     1. dealing with the cases "max_levels = 1" and "max_levels > 1" separately;
 *     2. add a parameter "conv_criteria", the user can choose a proper convergence 
 *        control by providing the value of the parameter
 *          conv_criteria =  0:  ||r^{k}||_2 / ||b||_2
 *          conv_criteria =  1:  ||u^{k} - u^{k-1}||_1 / ||u^{k}||_1
 *          conv_criteria = 11:  max { |u_i^{k} - u_i^{k-1}| / |u_i^{k}| }
 *          conv_criteria =  2:  ||u^{k} - u^{k-1}||_2 / ||u^{k}||_2
 *   where 
 *         ||u||_1 = max{|u_i|}， ||u||_2 = sqrt((u_1)^2 + ... + (u_n)^2)
 * \date 2011/09/28
 */
JX_Int
jx_PAMGSolve( void            *amg_vdata,
              jx_ParCSRMatrix *par_matrix,
              jx_ParVector    *par_rhs,
              jx_ParVector    *par_app  )
{
   //MPI_Comm        comm     = MPI_COMM_WORLD;
   MPI_Comm         comm     = jx_ParCSRMatrixComm(par_matrix); 
   jx_ParAMGData  *amg_data = amg_vdata;

   /* Data Structure variables */
   JX_Int    amg_print_level;
   JX_Int    amg_logging;
   JX_Int    cycle_count;
   JX_Int    num_levels;
   JX_Int    max_levels;
   JX_Real tol;
   JX_Real rhsnrm_threshold;

   jx_ParCSRMatrix **A_array;
   jx_ParVector    **F_array;
   jx_ParVector    **U_array;

   /*  Local variables  */
   JX_Int      j;
   JX_Int      Solve_err_flag;
   JX_Int      min_iter;
   JX_Int      max_iter;
   JX_Int      num_procs, my_id;

   JX_Real   alpha = 1.0;
   JX_Real   beta = -1.0;
   JX_Real   cycle_op_count;
   JX_Real   total_coeffs;
   JX_Real   total_variables;
   JX_Real  *num_coeffs;
   JX_Real  *num_variables;
   JX_Real   cycle_cmplxty  = 0.0;
   JX_Real   operat_cmplxty = 0.0;
   JX_Real   grid_cmplxty   = 0.0;
   JX_Real   conv_factor    = 0.0;
   JX_Real   resid_nrm      = 0.0;
   JX_Real   resid_nrm_init = 0.0;
   JX_Real   relative       = 0.0;
   JX_Real   old_relative   = 0.0;
   JX_Real   rhs_norm       = 0.0;
   JX_Real   old_resid      = 0.0;
   JX_Real   ieee_check     = 0.0;

   jx_ParVector  *Vtemp    = NULL;
   jx_ParVector  *Residual = NULL;
   jx_ParVector  *Utemp    = NULL;

   JX_Real  normup   = 0.0;
   JX_Real  normdown = 0.0;
   JX_Int    *partitioning;
   JX_Int     size;
   
   /* peghoty 2009/07/25 */
   JX_Int conv_criteria;
   JX_Real convfac_threshold;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   /* peghoty 2009/07/27 */
   max_levels = jx_ParAMGDataMaxLevels(amg_data);
   num_levels = jx_ParAMGDataNumLevels(amg_data);
   if (max_levels == 1)
   {
      jx_CoarsestSolver( amg_data, par_matrix, par_rhs, par_app );
      return(0);
   }

   conv_criteria    = jx_ParAMGDataConvCriteria(amg_data);
   convfac_threshold = jx_ParAMGDataConvFacThreshold(amg_data);
   amg_print_level  = jx_ParAMGDataPrintLevel(amg_data);
   amg_logging      = jx_ParAMGDataLogging(amg_data);
   tol              = jx_ParAMGDataTol(amg_data);
   rhsnrm_threshold = jx_ParAMGDataRhsNrmThreshold(amg_data);
   min_iter         = jx_ParAMGDataMinIter(amg_data);
   max_iter         = jx_ParAMGDataMaxIter(amg_data);
   Vtemp            = jx_ParAMGDataVtemp(amg_data);

#if 0 // not robust. peghoty, 2012/02/23
   size = jx_ParVectorGlobalSize(par_rhs);
   partitioning = jx_ParVectorPartitioning(par_rhs);
   Utemp = jx_ParVectorCreate(comm, size, partitioning);
   jx_ParVectorSetPartitioningOwner(Utemp, 0);
   jx_ParVectorInitialize(Utemp);
#else
   size = jx_ParCSRMatrixGlobalNumRows(par_matrix);
   partitioning = jx_ParCSRMatrixRowStarts(par_matrix);
   Utemp = jx_ParVectorCreate(comm, size, partitioning);
   jx_ParVectorInitialize(Utemp);
   jx_ParVectorSetPartitioningOwner(Utemp, 0); 
#endif

   /* Get the amg_data data */
   A_array = jx_ParAMGDataAArray(amg_data);
   F_array = jx_ParAMGDataFArray(amg_data);
   U_array = jx_ParAMGDataUArray(amg_data);

   /* they are moved from "amg_setup".  peghoty, 2009/07/27 */
   F_array[0] = par_rhs;
   U_array[0] = par_app;

   if (amg_logging > 1)
   {
      Residual = jx_ParAMGDataResidual(amg_data);
   }
   
   num_coeffs       = jx_CTAlloc(JX_Real, num_levels);
   num_variables    = jx_CTAlloc(JX_Real, num_levels);
   num_coeffs[0]    = (JX_Real) jx_ParCSRMatrixNumNonzeros(A_array[0]);
   num_variables[0] = (JX_Real) jx_ParCSRMatrixGlobalNumRows(A_array[0]);

   for (j = 1; j < num_levels; j ++)
   {
      num_coeffs[j]    = (JX_Real) jx_ParCSRMatrixNumNonzeros(A_array[j]);
      num_variables[j] = (JX_Real) jx_ParCSRMatrixGlobalNumRows(A_array[j]);
   }

   /* Write the solver parameters */ // set the solve status into SETUP phase, Yue Xiaoqiang, 2014/04/12
   //if (my_id == 0 && amg_print_level > 1)
   //{ 
   //   jx_PAMGSolveStatus(amg_data);
   //}

   /* Initialize the solver error flag and assorted bookkeeping variables */
   Solve_err_flag  = 0;
   total_coeffs    = 0;
   total_variables = 0;
   cycle_count     = 0;
   operat_cmplxty  = 0;
   grid_cmplxty    = 0;

   /* write some initial info */
   if (my_id == 0 && amg_print_level > 1 && tol > 0.)
   {
      jx_printf("\n\nJXPAMG SOLUTION INFO:\n");
   }


   //-----------------------------------------------------------------------//
   //       Compute initial fine-grid norm and print the table header       //
   //-----------------------------------------------------------------------//
   
   if (conv_criteria == 0)
   {
      if (tol >= 0.)
      {
          if (amg_logging > 1) 
          {
             jx_ParVectorCopy(F_array[0], Residual);
             jx_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual);
             resid_nrm = jx_ParVectorNorm2(Residual);
          }
          else 
          {
             jx_ParVectorCopy(F_array[0], Vtemp);
             jx_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
             resid_nrm = jx_ParVectorNorm2(Vtemp);
          }

         /*--------------------------------------------------------------------------------------
          * Since it does not diminish performance, attempt to return an error flag
          * and notify users when they supply bad input. 
          *-------------------------------------------------------------------------------------*/
          if (resid_nrm != 0.) ieee_check = resid_nrm / resid_nrm; /* INF -> NaN conversion */
          
          if (ieee_check != ieee_check)
          {
            /*----------------------------------------------------------------------------
             * ...INFs or NaNs in input can make ieee_check a NaN.  This test
             *   for ieee_check self-equality works on all IEEE-compliant compilers/
             *   machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
             *   by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
             *   found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF 
             *---------------------------------------------------------------------------*/
             if (amg_print_level > 0)
             {
                jx_printf("\n\nERROR detected by JXPAMG ...  BEGIN\n");
                jx_printf("ERROR -- jx_PAMGSolve: INFs and/or NaNs detected in input.\n");
                jx_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
                jx_printf("ERROR detected by JXPAMG ...  END\n\n\n");
//                break;
             }
             jx_error(JX_ERROR_GENERIC);
             return jx_error_flag;
          }

          resid_nrm_init = resid_nrm;
          rhs_norm = jx_ParVectorNorm2(F_array[0]);
          if (rhsnrm_threshold) /* peghoty, 2011/09/28 */
          {
             if (rhs_norm > rhsnrm_threshold)
             {
                relative = resid_nrm_init / rhs_norm;
             }
             else
             {
                relative = resid_nrm_init;
             }
          }
          else
          {
             if (rhs_norm)
             {
                relative = resid_nrm_init / rhs_norm;
             }
             else
             {
                relative = resid_nrm_init;
             }
          }
      }
      else
      {
         relative = 1.0;
      }
   }

   if (conv_criteria == 1 || conv_criteria == 11 || conv_criteria == 2)
   {
     /*---------------------------------------------------------------------------
      * To guarantee entering the "while loop" once at least peghoty, 2009/07/24 
      *--------------------------------------------------------------------------*/
      relative = 2*tol;  
   }

   if (my_id == 0 && amg_print_level > 1 && tol >= 0.)
   {    
      if (conv_criteria == 0)
      { 
         jx_printf("                                             Relative  \n");
         jx_printf("                Res Norm-2       Factor      Res Norm-2\n");
         jx_printf("                ----------      -------     -----------\n");
         jx_printf("    Initial     %e                \n",resid_nrm_init);
         //jx_printf("    Initial     %e                %e\n",resid_nrm_init,relative);
      }
      if (conv_criteria == 1)
      {
         jx_printf("                                           Relative   \n");
         jx_printf("                Error Norm-1    Factor    Error Norm-1\n");
         jx_printf("                ------------   --------  -------------\n");
      }
      if (conv_criteria == 11)
      {
         jx_printf("                         Relative Point-wise \n");
         jx_printf("                Factor      Error Norm-1     \n");
         jx_printf("                -------    --------------    \n");
      }
      if (conv_criteria == 2)
      {
         jx_printf("                                           Relative   \n");
         jx_printf("                Error Norm-2    Factor    Error Norm-2\n");
         jx_printf("                ------------   --------  -------------\n");           
      }
   }


   //-----------------------------------------------------------------------//
   //                    M A I N   V  C Y C L E   L O O P                   //
   //-----------------------------------------------------------------------//
   
//   jx_ParAMGDataCycleOpCount(amg_data) = 0; 
   while ( (relative >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      jx_ParAMGDataCycleOpCount(amg_data) = 0;
      /* Op count only needed for one cycle */

      if (conv_criteria == 1 || conv_criteria == 11 || conv_criteria == 2)
      {
         jx_ParVectorCopy(U_array[0], Utemp);
      }


     /*----------------------------------------------
      *  Excute One Cycle
      *---------------------------------------------*/
      jx_PAMGCycle(amg_data, F_array, U_array);  


     /*------------------------------------------------------------------
      *  Compute the residual norm on fine-grid  peghoty, 2009/07/23
      *-----------------------------------------------------------------*/
     
      if (conv_criteria == 0)
      {
         if (tol >= 0.)
         {
            old_resid = resid_nrm;

            if (amg_logging > 1) 
            {
               jx_ParVectorCopy(F_array[0], Residual);
               jx_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual);
               resid_nrm = jx_ParVectorNorm2(Residual);
            }
            else 
            {
               jx_ParVectorCopy(F_array[0], Vtemp);
               jx_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
               resid_nrm = jx_ParVectorNorm2(Vtemp);
            }

            if (old_resid)
            { 
               conv_factor = resid_nrm / old_resid;
            }   
            else
            { 
               conv_factor = resid_nrm;
            }
            
            if (rhsnrm_threshold) /* peghoty, 2011/09/28 */
            {
               if (rhs_norm > rhsnrm_threshold)
               {
                  relative = resid_nrm / rhs_norm;
               }
               else
               {
                  relative = resid_nrm / resid_nrm_init;
               }
            }
            else
            {
               if (rhs_norm)
               {
                  relative = resid_nrm / rhs_norm;
               }
               else
               {
                  relative = resid_nrm / resid_nrm_init;
               }
            }
         }
      } // end if (conv_criteria == 0)

      if (conv_criteria == 1)
      {
         if (cycle_count > 0) old_relative = relative;
         jx_ParVectorAxpy(-1.0, U_array[0], Utemp);  // Utemp = Utemp - U_array[0]
         normup   = jx_ParVectorNorm1(Utemp);
         normdown = jx_ParVectorNorm1(U_array[0]);
         if (normdown < EPS)
         {
            relative = normup;
         }
         else
         {
            relative = normup / normdown;
         }
         if (cycle_count > 0) 
         {
            conv_factor = relative / old_relative;
         }
      } // end if (conv_criteria == 1)

      if (conv_criteria == 11)
      {
         if (cycle_count > 0) 
         {
            old_relative = relative;
         }
         jx_ParVectorAxpy(-1.0, U_array[0], Utemp);  // Utemp = Utemp - U_array[0]
         relative = jx_ParVectorPointWiseRelNorm1(Utemp,U_array[0]);
         if (cycle_count > 0) 
         {
            conv_factor = relative / old_relative;
         }
      } // end if (conv_criteria == 11)

      if(conv_criteria == 2)
      {
         if (cycle_count > 0) old_relative = relative;
         jx_ParVectorAxpy(-1.0, U_array[0], Utemp);  // Utemp = Utemp - U_array[0]
         normup   = jx_ParVectorNorm2(Utemp);
         normdown = jx_ParVectorNorm2(U_array[0]);
         if (normdown < EPS)
         {
            relative = normup;
         }
         else
         {
            relative = normup / normdown;
         }
         if (cycle_count > 0) 
         {
            conv_factor = relative / old_relative;
         }
      } // end if (conv_criteria == 2)

      ++ cycle_count;

#ifdef CUMNUMIT
      ++ jx_ParAMGDataCumNumIterations(amg_data);
#endif

      if (my_id == 0 && amg_print_level > 1 && tol >= 0.)
      { 
         if (conv_criteria == 0)
         {
            jx_printf("    Cycle %3d   %e    %6f    %e \n", cycle_count, resid_nrm, conv_factor, relative);
         }
         else if (conv_criteria == 1)
         {
            if (cycle_count == 1)
            {
               jx_printf("    Cycle %3d   %e             %e \n", cycle_count, normup, relative);
            }
            else
            {
               jx_printf("    Cycle %3d   %e   %6f  %e \n", cycle_count, normup, conv_factor, relative);
            }
         }
         else if (conv_criteria == 11)
         {
            if (cycle_count == 1)
            {
               jx_printf("    Cycle %3d               %e \n", cycle_count, relative);
            }
            else
            {
               jx_printf("    Cycle %3d   %6f    %e \n", cycle_count, conv_factor, relative);
            }
         }
         else if (conv_criteria == 2)
         {
            if (cycle_count == 1)
            {
               jx_printf("    Cycle %3d   %e             %e \n", cycle_count, normup, relative);
            }
            else
            {
               jx_printf("    Cycle %3d   %e   %6f  %e \n", cycle_count, normup, conv_factor, relative);
            }
         }
      }

      if (conv_factor > convfac_threshold)
      {
         if (my_id == 0 && amg_print_level > 1)
         {
            jx_printf("\n  Warning: iteration has terminated because the\n");
            jx_printf("           current conv_factor > convfac_threshold(%lf) !!\n\n",convfac_threshold);
         }
         break;
      }

   }  

   jx_ParAMGDataRelativeResidualNorm(amg_data) = relative;
   jx_ParAMGDataNumIterations(amg_data) = cycle_count;

   if (cycle_count == max_iter && tol > 0.) 
   {
      Solve_err_flag = 1;
   }


   //-----------------------------------------------------------------------//
   //                    Compute closing statistics                         //
   //-----------------------------------------------------------------------//    
    
   if (conv_criteria == 0)
   {
      if (cycle_count > 0 && tol >= 0. && resid_nrm_init) 
      {
         conv_factor = pow((resid_nrm/resid_nrm_init),(1.0/(JX_Real) cycle_count));
      }
      else
      {
         conv_factor = 1.0;
      }
   }
   else
   {
      if (cycle_count > 0 && tol >= 0)
      {
         conv_factor = pow(relative,(1.0/(JX_Real) cycle_count));
      }
      else
      {
         conv_factor = 1.;
      }        
   }

   for (j = 0; j < num_levels; j ++)
   {
      total_coeffs    += num_coeffs[j];
      total_variables += num_variables[j];
   }

   cycle_op_count = jx_ParAMGDataCycleOpCount(amg_data);

   if (num_variables[0])
   {
      grid_cmplxty = total_variables / num_variables[0];
   }
   
   if (num_coeffs[0])
   {
      operat_cmplxty = total_coeffs / num_coeffs[0];
      cycle_cmplxty  = cycle_op_count / num_coeffs[0];
   }

   if (my_id == 0 && amg_print_level > 1)
   {
      if (Solve_err_flag == 1)
      {
         jx_printf("\n==================================================");
         jx_printf("\n Warning: Convergence tolerance was not achieved\n");
         jx_printf("          within the allowed %d V-cycles!\n",max_iter);
         jx_printf("==================================================\n");
      }
      if (tol >= 0.)
      {
         jx_printf("\n\n Average Convergence Factor = %f",conv_factor);
      }
      jx_printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
      jx_printf("                operator = %f\n",  operat_cmplxty);
      jx_printf("                   cycle = %f\n\n", cycle_cmplxty);
   }

   jx_TFree(num_coeffs);
   jx_TFree(num_variables);
   jx_ParVectorDestroy(Utemp);

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGPrecond
 * \brief PAMG as a preconditioner.
 * \param amg_vdata pointer to the amgdata object
 * \param par_matrix pointer to the coefficient matrix 
 * \param par_rhs pointer to the right hand side vector
 * \param par_app pointer to the approxamation vector
 * \author peghoty 
 * \date 2011/09/28
 */
JX_Int
jx_PAMGPrecond( void            *amg_vdata,
                jx_ParCSRMatrix *par_matrix,
                jx_ParVector    *par_rhs,
                jx_ParVector    *par_app  )
{
   jx_ParAMGData  *amg_data = amg_vdata;
   
   jx_ParVector  **F_array = jx_ParAMGDataFArray(amg_data);
   jx_ParVector  **U_array = jx_ParAMGDataUArray(amg_data);   

   JX_Int max_iter    = jx_ParAMGDataMaxIter(amg_data);       
   JX_Int cycle_count = 0;

   F_array[0] = par_rhs;
   U_array[0] = par_app;

  /*----------------------------------------------------------------
   * Generally, max_iter should be set to 1 when AMG is used as  
   * a preconditioner, here, we keep the parameter max_iter for  
   * convenience if max_iter should be larger than 1.
   *---------------------------------------------------------------*/
   while (cycle_count < max_iter)
   {
      jx_PAMGCycle(amg_data, F_array, U_array); 
      cycle_count ++;
   }
   
   return 0; 
}            
