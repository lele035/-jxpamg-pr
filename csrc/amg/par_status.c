//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_status.c -- Information outputing of the Setup and Solution 
 *                  phase of PAMG.
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGSetupStatus
 * \brief Information outputing of the Setup phase of PAMG.
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGSetupStatus( void            *amg_vdata,
                    jx_ParCSRMatrix *par_matrix )
{
   MPI_Comm       comm     = jx_ParCSRMatrixComm(par_matrix);   
   jx_ParAMGData *amg_data = amg_vdata;

   /* Data Structure variables */

   jx_ParCSRMatrix  **A_array;
   jx_ParCSRMatrix  **P_array;
   jx_ParCSRMatrix  **R_array;

   jx_CSRMatrix *A_diag;
   JX_Real       *A_diag_data;
   JX_Int          *A_diag_i;

   jx_CSRMatrix *A_offd;   
   JX_Real       *A_offd_data;
   JX_Int          *A_offd_i;

   jx_CSRMatrix *P_diag;
   JX_Real       *P_diag_data;
   JX_Int          *P_diag_i;

   jx_CSRMatrix *P_offd;   
   JX_Real       *P_offd_data;
   JX_Int          *P_offd_i;

   JX_Int	    numrows;
   JX_Int	   *row_starts;
   JX_Int      num_levels; 
   JX_Int      coarsen_type;
   JX_Int      interp_type;
   JX_Int      restri_type;
   JX_Int      measure_type;
   JX_Int      agg_interp_type;
   JX_Int      agg_num_levels;
   JX_Real   global_nonzeros;

   JX_Real  *send_buff;
   JX_Real  *gather_buff;
 
   /* Local variables */
   JX_Int      level;
   JX_Int      i,j;
   JX_Int      fine_size;
 
   JX_Int      min_entries;
   JX_Int      max_entries;

   JX_Int      num_procs,my_id;

   JX_Real   min_rowsum;
   JX_Real   max_rowsum;
   JX_Real   sparse;

   JX_Int      coarse_size;
   JX_Int      entries;

   JX_Real   avg_entries;
   JX_Real   rowsum;

   JX_Real   min_weight;
   JX_Real   max_weight;

   JX_Int      global_min_e;
   JX_Int      global_max_e;
   JX_Real   global_min_rsum;
   JX_Real   global_max_rsum;
   JX_Real   global_min_wt;
   JX_Real   global_max_wt;

   JX_Real  *num_coeffs;
   JX_Real  *num_variables;
   JX_Real   total_variables; 
   JX_Real   operat_cmplxty = 0.0;
   JX_Real   grid_cmplxty   = 0.0;

   /* amg solve params */
   JX_Int      max_iter;
   JX_Int      cycle_type;    
   JX_Int     *num_grid_sweeps;  
   JX_Int     *grid_relax_type;   
   JX_Int      relax_order;
   JX_Int    **grid_relax_points; 
   JX_Real  *relax_weight;
   JX_Real  *omega;
   JX_Real   tol;
   JX_Int      conv_criteria; /* peghoty 2009/07/27 */
   //JX_Int      amg_print_level;
   JX_Int     *AIR_maxsize_ls;

   JX_Int one       =  1;
   JX_Int minus_one = -1;
   JX_Int zero      =  0;
 
   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm, &my_id);

   A_array = jx_ParAMGDataAArray(amg_data);
   P_array = jx_ParAMGDataPArray(amg_data);
   R_array = jx_ParAMGDataRArray(amg_data);
   num_levels = jx_ParAMGDataNumLevels(amg_data);
   coarsen_type = jx_ParAMGDataCoarsenType(amg_data);
   interp_type = jx_ParAMGDataInterpType(amg_data);
   restri_type = jx_ParAMGDataRestriction(amg_data);
   measure_type = jx_ParAMGDataMeasureType(amg_data);
   agg_num_levels = jx_ParAMGDataAggNumLevels(amg_data);
   agg_interp_type = jx_ParAMGDataAggInterpType(amg_data);
   AIR_maxsize_ls = jx_ParAMGDataAIRMaxSizeLS(amg_data);

  /*----------------------------------------------------------
   * Get the amg_data data
   *--------------------------------------------------------*/

   num_levels        = jx_ParAMGDataNumLevels(amg_data);
   max_iter          = jx_ParAMGDataMaxIter(amg_data);
   cycle_type        = jx_ParAMGDataCycleType(amg_data);    
   num_grid_sweeps   = jx_ParAMGDataNumGridSweeps(amg_data);  
   grid_relax_type   = jx_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points = jx_ParAMGDataGridRelaxPoints(amg_data);
   relax_weight      = jx_ParAMGDataRelaxWeight(amg_data); 
   relax_order       = jx_ParAMGDataRelaxOrder(amg_data); 
   omega             = jx_ParAMGDataOmega(amg_data); 
   tol               = jx_ParAMGDataTol(amg_data);
   conv_criteria     = jx_ParAMGDataConvCriteria(amg_data); /* peghoty 2009/07/27 */
   //amg_print_level   = jx_ParAMGDataPrintLevel(amg_data);

   send_buff   = jx_CTAlloc(JX_Real, 6);
#ifdef JX_NO_GLOBAL_PARTITION
   gather_buff = jx_CTAlloc(JX_Real, 6);    
#else
   gather_buff = jx_CTAlloc(JX_Real, 6*num_procs);    
#endif

   if (my_id == 0)
   {
      jx_printf("\njx_PAMG SETUP PARAMETERS:\n\n");
      jx_printf(" Max levels = %d\n",jx_ParAMGDataMaxLevels(amg_data));
      jx_printf(" Num levels = %d\n\n",num_levels);
      jx_printf(" Strength Threshold = %f\n", jx_ParAMGDataStrongThreshold(amg_data));
      if (restri_type > 0)
      {
     jx_printf(" AIR Strength Threshold = %f for restriction\n", jx_ParAMGDataAIRStrongTh(amg_data));
      }
      jx_printf(" Interpolation Truncation Factor = %f\n", jx_ParAMGDataTruncFactor(amg_data));
      jx_printf(" Maximum Row Sum Threshold for Dependency Weakening = %f\n\n", jx_ParAMGDataMaxRowSum(amg_data));

      if (coarsen_type == 0)
      {
	 jx_printf(" Coarsening Type = Cleary-Luby-Jones-Plassman\n");
      }
      else if (abs(coarsen_type) == 1) 
      {
	 jx_printf(" Coarsening Type = Ruge\n");
      }
      else if (abs(coarsen_type) == 2) 
      {
	 jx_printf(" Coarsening Type = Ruge2B\n");
      }
      else if (abs(coarsen_type) == 3) 
      {
	 jx_printf(" Coarsening Type = Ruge3\n");
      }
      else if (abs(coarsen_type) == 4) 
      {
	 jx_printf(" Coarsening Type = Ruge 3c \n");
      }
      else if (abs(coarsen_type) == 5) 
      {
	 jx_printf(" Coarsening Type = Ruge relax special points \n");
      }
      else if (abs(coarsen_type) == 6) 
      {
	 jx_printf(" Coarsening Type = Falgout-CLJP \n");
      }
      else if (abs(coarsen_type) == 8) 
      {
	 jx_printf(" Coarsening Type = PMIS \n");
      }
      else if (abs(coarsen_type) == 10) 
      {
	 jx_printf(" Coarsening Type = HMIS \n");
      }
      else if (abs(coarsen_type) == 11) 
      {
	 jx_printf(" Coarsening Type = Ruge 1st pass only \n");
      }
      else if (coarsen_type == 990)
      {
	 jx_printf(" Coarsening Type = Cleary-Luby-Jones-Plassman - AI\n");
      }
      else if (coarsen_type == 991)
      {
	 jx_printf(" Coarsening Type = Ruge 1st pass only - AI \n");
      }
      else if (coarsen_type == 992)
      {
	 jx_printf(" Coarsening Type = Ruge - AI\n");
      }
      else if (coarsen_type == 993)
      {
	 jx_printf(" Coarsening Type = Ruge3 - AI\n");
      }
      else if (abs(coarsen_type) == 96)
      {
	 jx_printf(" Coarsening Type = Falgout-CLJP - AI \n");
      }
      else if (abs(coarsen_type) == 98)
      {
	 jx_printf(" Coarsening Type = PMIS - AI \n");
      }
      else if (abs(coarsen_type) == 910)
      {
	 jx_printf(" Coarsening Type = HMIS - AI \n");
      }
      else if (coarsen_type == 908)
      {
	 jx_printf(" Coarsening Type = CLJP -AI + PMIS - AI \n");
      }
      else if (coarsen_type == 918)
      {
         jx_printf(" Coarsening Type = Ruge 1st pass only - AI + PMIS - AI \n");
      }
      else if (coarsen_type == 928)
      {
         jx_printf(" Coarsening Type = Ruge - AI + PMIS - AI \n");
      }
      else if (coarsen_type == 938)
      {
         jx_printf(" Coarsening Type = Ruge3 - AI + PMIS - AI \n");
      }
      else if (coarsen_type == 968)
      {
         jx_printf(" Coarsening Type = Falgout-CLJP - AI + PMIS - AI \n");
      }

      if (agg_num_levels > 0)
      {
         jx_printf("\n No. of levels of aggressive coarsening: %d\n\n", agg_num_levels);
         if (agg_interp_type == 4)
            jx_printf(" Interpolation on agg. levels = multipass interpolation\n");
      }

      if (coarsen_type)
      {
         jx_printf(" measures are determined %s\n\n", (measure_type ? "globally" : "locally"));
      }

#ifdef JX_NO_GLOBAL_PARTITION
      jx_printf("\n No global partition option chosen.\n\n");
#endif

      if (interp_type == 0)
      {
	 jx_printf(" Interpolation = modified classical interpolation\n");
      }
      else if (interp_type == 3) 
      {
	 jx_printf(" Interpolation = direct interpolation (with separation of weights)\n");
      }
      else if (interp_type == 4) 
      {
	 jx_printf(" Interpolation = multipass interpolation\n");
      }
      else if (interp_type == 5) 
      {
	 jx_printf(" Interpolation = multipass interpolation (with separation of weights)\n");
      }
      else if (interp_type == 6) 
      {
	 jx_printf(" Interpolation = extended classical modified interpolation\n");
      }
      else if (interp_type == 7) 
      {
	 jx_printf(" Interpolation = extended (if no common C neighbor) classical modified interpolation\n");
      }
      else if (interp_type == 8) 
      {
	 jx_printf(" Interpolation = standard interpolation\n");
      }
      else if (interp_type == 9) 
      {
	 jx_printf(" Interpolation = standard interpolation (with separation of weights)\n");
      }
      else if (interp_type == 100) 
      {
	 jx_printf(" Interpolation = one-point interpolation (strongest C neighbor)\n");
      }

      if (restri_type == 1)
      {
     jx_printf(" Restriction = local approximate ideal restriction (AIR-1)\n");
      }
      else if (restri_type == 2)
      {
     jx_printf(" Restriction = local approximate ideal restriction (AIR-2)\n");
      }

      jx_printf("\nOperator Matrix Information:\n\n");

      jx_printf("            nonzero         entries p");
      jx_printf("er row        row sums\n");
      jx_printf("lev   rows  entries  sparse  min  max   ");
      jx_printf("avg       min         max\n");
      jx_printf("===================================================================\n");
   }
  
  
  /*-----------------------------------------------------
   *  Enter Statistics Loop
   *-----------------------------------------------------*/

   num_coeffs = jx_CTAlloc(JX_Real, num_levels);

   num_variables = jx_CTAlloc(JX_Real, num_levels);

   for (level = 0; level < num_levels; level ++)
   { 
       A_diag      = jx_ParCSRMatrixDiag(A_array[level]);
       A_diag_data = jx_CSRMatrixData(A_diag);
       A_diag_i    = jx_CSRMatrixI(A_diag);
         
       A_offd      = jx_ParCSRMatrixOffd(A_array[level]);   
       A_offd_data = jx_CSRMatrixData(A_offd);
       A_offd_i    = jx_CSRMatrixI(A_offd);
         
       row_starts = jx_ParCSRMatrixRowStarts(A_array[level]);
         
       fine_size         = jx_ParCSRMatrixGlobalNumRows(A_array[level]);
       global_nonzeros   = jx_ParCSRMatrixDNumNonzeros(A_array[level]);
       num_coeffs[level] = global_nonzeros;
       num_variables[level] = (JX_Real) fine_size;
         
       sparse = global_nonzeros / ((JX_Real) fine_size * (JX_Real) fine_size);

       min_entries = 0;
       max_entries = 0;
       min_rowsum  = 0.0;
       max_rowsum  = 0.0;
         
       if (jx_CSRMatrixNumRows(A_diag))
       {
          min_entries = (A_diag_i[1] - A_diag_i[0]) + (A_offd_i[1] - A_offd_i[0]);
          for (j = A_diag_i[0]; j < A_diag_i[1]; j ++)
          {
             min_rowsum += A_diag_data[j];
          }
          for (j = A_offd_i[0]; j < A_offd_i[1]; j ++)
          {
             min_rowsum += A_offd_data[j];
          }
            
          //jx_printf("rowsum_init = %e\n", min_rowsum);

          max_rowsum = min_rowsum;
            
          for (j = 0; j < jx_CSRMatrixNumRows(A_diag); j ++)
          {
             entries = (A_diag_i[j+1] - A_diag_i[j]) + (A_offd_i[j+1] - A_offd_i[j]);
             min_entries = jx_min(entries, min_entries);
             max_entries = jx_max(entries, max_entries);
               
             rowsum = 0.0;
             for (i = A_diag_i[j]; i < A_diag_i[j+1]; i ++)
             {
                rowsum += A_diag_data[i];
             }
               
             for (i = A_offd_i[j]; i < A_offd_i[j+1]; i ++)
             {
                rowsum += A_offd_data[i];
             }

#if 0            
             if (rowsum < -1.0e-1) {
                jx_printf("i= %d, rowsum = %e\n", j, rowsum);
                for (i = A_diag_i[j]; i < A_diag_i[j+1]; i ++)
                {
                   jx_printf("ia= %d, A_diag = %e\n", i, A_diag_data[i]);
                }
               
                for (i = A_offd_i[j]; i < A_offd_i[j+1]; i ++)
                {
                   jx_printf("ia= %d, A_offd = %e\n", i, A_offd_data[i]);
                }

             }
#endif
               
             min_rowsum = jx_min(rowsum, min_rowsum);
             max_rowsum = jx_max(rowsum, max_rowsum);
          }
       }
       avg_entries = global_nonzeros / ((JX_Real) fine_size);

      
#ifdef JX_NO_GLOBAL_PARTITION       

       numrows = row_starts[1] - row_starts[0];
       
      /*------------------------------------------------------------- 
       * if we don't have any rows, then don't have this 
       * count toward min row sum or min num entries. 
       *------------------------------------------------------------*/       
       if (!numrows) 
       {
          min_entries = 1000000;
          min_rowsum  = 1.0e7;
       }
       
       send_buff[0] = - (JX_Real) min_entries;
       send_buff[1] = (JX_Real) max_entries;
       send_buff[2] = - min_rowsum;
       send_buff[3] = max_rowsum;

       jx_MPI_Reduce(send_buff, gather_buff, 4, JX_MPI_REAL, MPI_MAX, 0, comm);
       
       if (my_id == 0)
       {
          global_min_e = - gather_buff[0];
          global_max_e =   gather_buff[1];
          global_min_rsum = - gather_buff[2];
          global_max_rsum =   gather_buff[3];
          
          jx_printf("%2d %7d %8.0f  %0.3f  %4d %4d",
                 level, fine_size, global_nonzeros, sparse, global_min_e, global_max_e);
          jx_printf("  %4.1f  %10.3e  %10.3e\n", avg_entries, global_min_rsum, global_max_rsum);
       }
       
#else

       send_buff[0] = (JX_Real) min_entries;
       send_buff[1] = (JX_Real) max_entries;
       send_buff[2] = min_rowsum;
       send_buff[3] = max_rowsum;
       
       jx_MPI_Gather(send_buff, 4, JX_MPI_REAL, gather_buff, 4, JX_MPI_REAL, 0, comm);

       if (my_id == 0)
       {
          global_min_e = 1000000;
          global_max_e = 0;
          global_min_rsum = 1.0e7;
          global_max_rsum = 0.0;
          for (j = 0; j < num_procs; j ++)
          {
             numrows = row_starts[j+1] - row_starts[j];
             if (numrows)
             {
                global_min_e    = jx_min(global_min_e, (JX_Int) gather_buff[j*4]);
                global_min_rsum = jx_min(global_min_rsum, gather_buff[j*4 + 2]);
             }
             global_max_e    = jx_max(global_max_e, (JX_Int) gather_buff[j*4 + 1]);
             global_max_rsum = jx_max(global_max_rsum, gather_buff[j*4 + 3]);
          }

          jx_printf("%2d %7d %8.0f  %0.3f  %4d %4d",
                  level, fine_size, global_nonzeros, sparse, global_min_e, global_max_e);
          jx_printf("  %4.1f  %10.3e  %10.3e\n", avg_entries, global_min_rsum, global_max_rsum);
       }

#endif
    
   }

   if (my_id == 0)
   {
      jx_printf("\n\nInterpolation Matrix Information:\n");
      jx_printf("                 entries/row    min     max");
      jx_printf("         row sums\n");
      jx_printf("lev  rows cols    min max  ");
      jx_printf("   weight   weight     min       max \n");
      jx_printf("=================================================================\n");
   }
  

  /*-----------------------------------------------------
   *  Enter Statistics Loop
   *-----------------------------------------------------*/

   for (level = 0; level < num_levels-1; level ++)
   {
      P_diag      = jx_ParCSRMatrixDiag(P_array[level]);
      P_diag_data = jx_CSRMatrixData(P_diag);
      P_diag_i    = jx_CSRMatrixI(P_diag);
         
      P_offd      = jx_ParCSRMatrixOffd(P_array[level]);   
      P_offd_data = jx_CSRMatrixData(P_offd);
      P_offd_i    = jx_CSRMatrixI(P_offd);
         
      row_starts = jx_ParCSRMatrixRowStarts(P_array[level]);
         
      fine_size       = jx_ParCSRMatrixGlobalNumRows(P_array[level]);
      coarse_size     = jx_ParCSRMatrixGlobalNumCols(P_array[level]);
      global_nonzeros = jx_ParCSRMatrixNumNonzeros(P_array[level]);
         
      min_weight  = 1.0;
      max_weight  = 0.0;
      max_rowsum  = 0.0;
      min_rowsum  = 0.0;
      min_entries = 0;
      max_entries = 0;
         
      if (jx_CSRMatrixNumRows(P_diag))
      {
         if (jx_CSRMatrixNumCols(P_diag)) 
         {
            min_weight = P_diag_data[0];
         }
         
         for (j = P_diag_i[0]; j < P_diag_i[1]; j ++)
         {
            min_weight = jx_min(min_weight, P_diag_data[j]);
            if (P_diag_data[j] != 1.0)
            {
               max_weight = jx_max(max_weight, P_diag_data[j]);
            }
            min_rowsum += P_diag_data[j];
         }
         for (j = P_offd_i[0]; j < P_offd_i[1]; j ++)
         {        
            min_weight = jx_min(min_weight, P_offd_data[j]); 
            if (P_offd_data[j] != 1.0)
            {
               max_weight = jx_max(max_weight, P_offd_data[j]);  
            }   
            min_rowsum += P_offd_data[j];
         }
            
         max_rowsum = min_rowsum;
            
         min_entries = (P_diag_i[1] - P_diag_i[0]) + (P_offd_i[1] - P_offd_i[0]); 
         max_entries = 0;
            
         for (j = 0; j < jx_CSRMatrixNumRows(P_diag); j ++)
         {
            entries = (P_diag_i[j+1] - P_diag_i[j]) + (P_offd_i[j+1] - P_offd_i[j]);
            min_entries = jx_min(entries, min_entries);
            max_entries = jx_max(entries, max_entries);
               
            rowsum = 0.0;
            for (i = P_diag_i[j]; i < P_diag_i[j+1]; i ++)
            {
               min_weight = jx_min(min_weight, P_diag_data[i]);
               if (P_diag_data[i] != 1.0)
               {
                  max_weight = jx_max(max_weight, P_diag_data[i]);
               }
               rowsum += P_diag_data[i];
            }
               
            for (i = P_offd_i[j]; i < P_offd_i[j+1]; i ++)
            {
               min_weight = jx_min(min_weight, P_offd_data[i]);
               if (P_offd_data[i] != 1.0)
               { 
                  max_weight = jx_max(max_weight, P_offd_data[i]);
               }
               rowsum += P_offd_data[i];
            }
               
            min_rowsum = jx_min(rowsum, min_rowsum);
            max_rowsum = jx_max(rowsum, max_rowsum);
         }
         
      }
      avg_entries = ((JX_Real) global_nonzeros) / ((JX_Real) fine_size);


#ifdef JX_NO_GLOBAL_PARTITION

      numrows = row_starts[1] - row_starts[0];

      /*------------------------------------------------------------- 
       * if we don't have any rows, then don't have this 
       * count toward min row sum or min num entries. 
       *------------------------------------------------------------*/        
      if (!numrows)
      {
         min_entries = 1000000;
         min_rowsum  = 1.0e7;
         min_weight  = 1.0e7;
       }
       
      send_buff[0] = - (JX_Real) min_entries;
      send_buff[1] = (JX_Real) max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = - min_weight;
      send_buff[5] = max_weight;

      jx_MPI_Reduce(send_buff, gather_buff, 6, JX_MPI_REAL, MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e    = -gather_buff[0];
         global_max_e    =  gather_buff[1];
         global_min_rsum = -gather_buff[2];
         global_max_rsum =  gather_buff[3];
         global_min_wt   = -gather_buff[4];
         global_max_wt   =  gather_buff[5];
         jx_printf("%2d %5d x %-5d %3d %3d", level, fine_size, coarse_size,  global_min_e, global_max_e);
         jx_printf("  %10.3e %9.3e %9.3e %9.3e\n", global_min_wt, global_max_wt, global_min_rsum, global_max_rsum);
      }

#else
      
      send_buff[0] = (JX_Real) min_entries;
      send_buff[1] = (JX_Real) max_entries;
      send_buff[2] = min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = min_weight;
      send_buff[5] = max_weight;
      
      jx_MPI_Gather(send_buff, 6, JX_MPI_REAL, gather_buff, 6, JX_MPI_REAL, 0, comm);
      
      if (my_id == 0)
      {
         global_min_e    = 1000000;
         global_max_e    = 0;
         global_min_rsum = 1.0e7;
         global_max_rsum = 0.0;
         global_min_wt   = 1.0e7;
         global_max_wt   = 0.0;
         
         for (j = 0; j < num_procs; j ++)
         {
            numrows = row_starts[j+1] - row_starts[j];
            if (numrows)
            {
               global_min_e = jx_min(global_min_e, (JX_Int) gather_buff[j*6]);
               global_min_rsum = jx_min(global_min_rsum, gather_buff[j*6+2]);
               global_min_wt = jx_min(global_min_wt, gather_buff[j*6+4]);
            }
            global_max_e = jx_max(global_max_e, (JX_Int) gather_buff[j*6+1]);
            global_max_rsum = jx_max(global_max_rsum, gather_buff[j*6+3]);
            global_max_wt = jx_max(global_max_wt, gather_buff[j*6+5]);
         }
         jx_printf("%2d %5d x %-5d %3d %3d", level, fine_size, coarse_size,  global_min_e, global_max_e);
         jx_printf("  %10.3e %9.3e %9.3e %9.3e\n", global_min_wt, global_max_wt, global_min_rsum, global_max_rsum);
      }

#endif

   }

   if (restri_type)
   {

   if (my_id == 0)
   {
      jx_printf("\n\nRestriction Matrix Information:\n");
      jx_printf("                 entries/row    min     max");
      jx_printf("         row sums       size(ls)\n");
      jx_printf("lev  rows cols    min max  ");
      jx_printf("   weight   weight     min       max      max\n");
      jx_printf("=========================================================================\n");
   }


  /*-----------------------------------------------------
   *  Enter Statistics Loop
   *-----------------------------------------------------*/

   for (level = 0; level < num_levels-1; level ++)
   {
      P_diag      = jx_ParCSRMatrixDiag(R_array[level]);
      P_diag_data = jx_CSRMatrixData(P_diag);
      P_diag_i    = jx_CSRMatrixI(P_diag);
         
      P_offd      = jx_ParCSRMatrixOffd(R_array[level]);   
      P_offd_data = jx_CSRMatrixData(P_offd);
      P_offd_i    = jx_CSRMatrixI(P_offd);
         
      row_starts = jx_ParCSRMatrixRowStarts(R_array[level]);
         
      fine_size       = jx_ParCSRMatrixGlobalNumRows(R_array[level]);
      coarse_size     = jx_ParCSRMatrixGlobalNumCols(R_array[level]);
      global_nonzeros = jx_ParCSRMatrixNumNonzeros(R_array[level]);
         
      min_weight  = 1.0;
      max_weight  = 0.0;
      max_rowsum  = 0.0;
      min_rowsum  = 0.0;
      min_entries = 0;
      max_entries = 0;
         
      if (jx_CSRMatrixNumRows(P_diag))
      {
         if (jx_CSRMatrixNumCols(P_diag)) 
         {
            min_weight = P_diag_data[0];
         }
         
         for (j = P_diag_i[0]; j < P_diag_i[1]; j ++)
         {
            min_weight = jx_min(min_weight, P_diag_data[j]);
            if (P_diag_data[j] != 1.0)
            {
               max_weight = jx_max(max_weight, P_diag_data[j]);
            }
            min_rowsum += P_diag_data[j];
         }
         for (j = P_offd_i[0]; j < P_offd_i[1]; j ++)
         {        
            min_weight = jx_min(min_weight, P_offd_data[j]); 
            if (P_offd_data[j] != 1.0)
            {
               max_weight = jx_max(max_weight, P_offd_data[j]);  
            }   
            min_rowsum += P_offd_data[j];
         }
            
         max_rowsum = min_rowsum;
            
         min_entries = (P_diag_i[1] - P_diag_i[0]) + (P_offd_i[1] - P_offd_i[0]); 
         max_entries = 0;
            
         for (j = 0; j < jx_CSRMatrixNumRows(P_diag); j ++)
         {
            entries = (P_diag_i[j+1] - P_diag_i[j]) + (P_offd_i[j+1] - P_offd_i[j]);
            min_entries = jx_min(entries, min_entries);
            max_entries = jx_max(entries, max_entries);
               
            rowsum = 0.0;
            for (i = P_diag_i[j]; i < P_diag_i[j+1]; i ++)
            {
               min_weight = jx_min(min_weight, P_diag_data[i]);
               if (P_diag_data[i] != 1.0)
               {
                  max_weight = jx_max(max_weight, P_diag_data[i]);
               }
               rowsum += P_diag_data[i];
            }
               
            for (i = P_offd_i[j]; i < P_offd_i[j+1]; i ++)
            {
               min_weight = jx_min(min_weight, P_offd_data[i]);
               if (P_offd_data[i] != 1.0)
               { 
                  max_weight = jx_max(max_weight, P_offd_data[i]);
               }
               rowsum += P_offd_data[i];
            }
               
            min_rowsum = jx_min(rowsum, min_rowsum);
            max_rowsum = jx_max(rowsum, max_rowsum);
         }
         
      }
      avg_entries = ((JX_Real) global_nonzeros) / ((JX_Real) fine_size);


#ifdef JX_NO_GLOBAL_PARTITION

      numrows = row_starts[1] - row_starts[0];

      /*------------------------------------------------------------- 
       * if we don't have any rows, then don't have this 
       * count toward min row sum or min num entries. 
       *------------------------------------------------------------*/        
      if (!numrows)
      {
         min_entries = 1000000;
         min_rowsum  = 1.0e7;
         min_weight  = 1.0e7;
       }
       
      send_buff[0] = - (JX_Real) min_entries;
      send_buff[1] = (JX_Real) max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = - min_weight;
      send_buff[5] = max_weight;

      jx_MPI_Reduce(send_buff, gather_buff, 6, JX_MPI_REAL, MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e    = -gather_buff[0];
         global_max_e    =  gather_buff[1];
         global_min_rsum = -gather_buff[2];
         global_max_rsum =  gather_buff[3];
         global_min_wt   = -gather_buff[4];
         global_max_wt   =  gather_buff[5];
         jx_printf("%2d %5d x %-5d %3d %3d", level, fine_size, coarse_size,  global_min_e, global_max_e);
         jx_printf("  %10.3e %9.3e %9.3e %9.3e", global_min_wt, global_max_wt, global_min_rsum, global_max_rsum);
         jx_printf("  %3d\n", AIR_maxsize_ls[level]);
      }

#else
      
      send_buff[0] = (JX_Real) min_entries;
      send_buff[1] = (JX_Real) max_entries;
      send_buff[2] = min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = min_weight;
      send_buff[5] = max_weight;
      
      jx_MPI_Gather(send_buff, 6, JX_MPI_REAL, gather_buff, 6, JX_MPI_REAL, 0, comm);
      
      if (my_id == 0)
      {
         global_min_e    = 1000000;
         global_max_e    = 0;
         global_min_rsum = 1.0e7;
         global_max_rsum = 0.0;
         global_min_wt   = 1.0e7;
         global_max_wt   = 0.0;
         
         for (j = 0; j < num_procs; j ++)
         {
            numrows = row_starts[j+1] - row_starts[j];
            if (numrows)
            {
               global_min_e = jx_min(global_min_e, (JX_Int) gather_buff[j*6]);
               global_min_rsum = jx_min(global_min_rsum, gather_buff[j*6+2]);
               global_min_wt = jx_min(global_min_wt, gather_buff[j*6+4]);
            }
            global_max_e = jx_max(global_max_e, (JX_Int) gather_buff[j*6+1]);
            global_max_rsum = jx_max(global_max_rsum, gather_buff[j*6+3]);
            global_max_wt = jx_max(global_max_wt, gather_buff[j*6+5]);
         }
         jx_printf("%2d %5d x %-5d %3d %3d", level, fine_size, coarse_size,  global_min_e, global_max_e);
         jx_printf("  %10.3e %9.3e %9.3e %9.3e", global_min_wt, global_max_wt, global_min_rsum, global_max_rsum);
         jx_printf("  %3d\n", AIR_maxsize_ls[level]);
      }

#endif

   }

   }

   total_variables = 0;
   operat_cmplxty  = 0;
   
   for (j = 0; j < jx_ParAMGDataNumLevels(amg_data); j ++)
   {
      operat_cmplxty  +=  num_coeffs[j] / num_coeffs[0];
      total_variables += num_variables[j];
   }
   
   if (num_variables[0] != 0)
   {
      grid_cmplxty = total_variables / num_variables[0];
   }
 
   if (my_id == 0)
   {
      jx_printf("\n\n     Complexity:    grid = %f\n", grid_cmplxty);
      jx_printf("                operator = %f\n", operat_cmplxty);
   }

   if (my_id == 0) jx_printf("\n\n");

   //if (my_id == 0 && amg_print_level == 1)
   if (my_id == 0)
   { 
      jx_printf("\n\njx_PAMG SOLVER PARAMETERS:\n\n");
      jx_printf("  Maximum number of cycles:         %d \n",max_iter);
      jx_printf("  Stopping Tolerance:               %e \n",tol); 
      /* peghoty 2009/07/27 */
      jx_printf("  Convergence Criteria:");
      switch(conv_criteria)
      {
         case 0:
         jx_printf("  Relative Residual Norm-2 (default)\n");
         break;
         case 1:
         jx_printf("  Relative Error Norm-1\n");
         break;
         case 11:
         jx_printf("  Relative Point-wise Error Norm-1\n");
         break;
         case 2:
         jx_printf("  Relative Error Norm-2\n");
         break;
      }
      jx_printf("  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      jx_printf("  Relaxation Parameters:\n");
      jx_printf("   Visiting Grid:                     down   up  coarse\n");
      jx_printf("            Number of sweeps:         %4d   %2d  %4d \n",
              num_grid_sweeps[1], num_grid_sweeps[2], num_grid_sweeps[3]);
      jx_printf("   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
              grid_relax_type[1], grid_relax_type[2], grid_relax_type[3]);
      jx_printf("   Point types, partial sweeps (1=C, -1=F):\n");

      if (grid_relax_points && grid_relax_type[1] != 8)
      {
         jx_printf("                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j ++)
         {
            jx_printf("  %2d", grid_relax_points[1][j]);
         }
         jx_printf("\n");
         jx_printf("                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j ++)
         {
            jx_printf("  %2d", grid_relax_points[2][j]);
         }
         jx_printf("\n");
         jx_printf("                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j ++)
         {
            jx_printf("  %2d", grid_relax_points[3][j]);
         }
         jx_printf("\n\n");
      }
      else if (relax_order == 1 && grid_relax_type[1] != 8)
      {
         jx_printf("                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j ++)
         {
            jx_printf("  %2d  %2d", one, minus_one);
         }
         jx_printf("\n");
         jx_printf("                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j ++)
         {
            jx_printf("  %2d  %2d", minus_one, one);
         }
         jx_printf("\n");
         jx_printf("                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n\n");
      }
      else 
      {
         jx_printf("                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n");
         jx_printf("                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n");
         jx_printf("                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n\n");
      }

      for (j = 0; j < num_levels; j ++)
      {
         if (relax_weight[j] != 1)
         {
	    jx_printf(" Relaxation Weight %f level %d\n", relax_weight[j], j);
	 }
      }
      for (j = 0; j < num_levels; j ++)
      {
         if (omega[j] != 1)
         {
            jx_printf(" Outer relaxation weight %f level %d\n", omega[j], j);
         }
      }
   }

   jx_TFree(num_coeffs);
   jx_TFree(num_variables);
   jx_TFree(send_buff);
   jx_TFree(gather_buff);
   
   return(0);
}


/*!
 * \fn JX_Int jx_PAMGSolveStatus
 * \brief Information outputing of the Solution phase of PAMG.
 * \date 2011/09/03
 */ 
JX_Int    
jx_PAMGSolveStatus( void *data )
{ 
   jx_ParAMGData  *amg_data = data;
   JX_Int j;
   JX_Int one = 1;
   JX_Int minus_one = -1;
   JX_Int zero = 0;

  /*-------------------------------------------------------------------------
   * Get the amg_data data
   *------------------------------------------------------------------------*/
   JX_Int     amg_print_level   = jx_ParAMGDataPrintLevel(amg_data);
   JX_Int     max_iter          = jx_ParAMGDataMaxIter(amg_data);
   JX_Int     cycle_type        = jx_ParAMGDataCycleType(amg_data);
   JX_Int    *num_grid_sweeps   = jx_ParAMGDataNumGridSweeps(amg_data);  
   JX_Int    *grid_relax_type   = jx_ParAMGDataGridRelaxType(amg_data);
   JX_Int   **grid_relax_points = jx_ParAMGDataGridRelaxPoints(amg_data);
   JX_Int     relax_order       = jx_ParAMGDataRelaxOrder(amg_data);
   JX_Real  tol               = jx_ParAMGDataTol(amg_data);
   JX_Int     conv_criteria     = jx_ParAMGDataConvCriteria(amg_data); /* peghoty 2009/07/27 */

   if (amg_print_level == 2 || amg_print_level == 3)
   { 
      jx_printf("\n\nJX_PAMG SOLVER PARAMETERS:\n\n");
      jx_printf("  Maximum number of cycles:         %d \n", max_iter);
      jx_printf("  Stopping Tolerance:               %e \n", tol);
      jx_printf("  Convergence Criteria:"); /* peghoty 2009/07/27 */
      switch (conv_criteria)
      {
         case 0:
         jx_printf("  Relative Residual Norm-2 (default)\n");
         break;
         case 1:
         jx_printf("  Relative Error Norm-1\n");
         break;
         case 11:
         jx_printf("  Relative Point-wise Error Norm-1\n");
         break;
         case 2:
         jx_printf("  Relative Error Norm-2\n");
         break;
      }
      jx_printf("  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      jx_printf("  Relaxation Parameters:\n");
      jx_printf("   Visiting Grid:                     down   up  coarse\n");
      jx_printf("            Number of sweeps:         %4d   %2d  %4d \n",
              num_grid_sweeps[1], num_grid_sweeps[2], num_grid_sweeps[3]);
      jx_printf("   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
              grid_relax_type[1], grid_relax_type[2], grid_relax_type[3]);
      jx_printf("   Point types, partial sweeps (1=C, -1=F):\n");
      if (grid_relax_points)
      {
         jx_printf("                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j ++)
         {
            jx_printf("  %2d", grid_relax_points[1][j]);
         }
         jx_printf("\n");
         jx_printf("                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j ++)
         {
            jx_printf("  %2d", grid_relax_points[2][j]);
         }
         jx_printf("\n");
         jx_printf("                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j ++)
         {
            jx_printf("  %2d", grid_relax_points[3][j]);
         }
         jx_printf("\n\n");
      }
      else if (relax_order == 1)
      {
         jx_printf("                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j ++)
         {
            jx_printf("  %2d  %2d", one, minus_one);
         }
         jx_printf("\n");
         jx_printf("                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j ++)
         {
            jx_printf("  %2d  %2d", minus_one, one);
         }
         jx_printf("\n");
         jx_printf("                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n\n");
      }
      else 
      {
         jx_printf("                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n");
         jx_printf("                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n");
         jx_printf("                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j ++)
         {
            jx_printf("  %2d", zero);
         }
         jx_printf("\n\n");
      }
      jx_printf(" Output flag (print_level): %d \n", amg_print_level);
   }
 
   return 0;
}
