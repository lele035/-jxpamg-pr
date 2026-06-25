//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_coarsen_rcljp.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*==========================================================================
 * 
 * Relax CLJP (RCLJP) coarsening,a parallel coarsening algorithm for AMG
 * based on CLJP algorithm, proposed by Zeyao Mo & Xiaowen Xu, May, 2005.                  
 * The code was devoloped by Dr. X.Xu, June,2005.
 * 
 *==========================================================================*/

#define C_PT  1
#define F_PT -1
#define COMMON_C_PT  2
#define Z_PT -2 

/*!
 * \fn JX_Int jx_PAMGCoarsenRRS0
 * \brief RCLJP Coarsening routine.
 * \author Xu Xiaowen
 * \date June, 2005
 */ 
JX_Int
jx_PAMGCoarsenRCLJP( jx_ParCSRMatrix    *S,
                     jx_ParCSRMatrix    *A,
                     JX_Int                 measure_type_rlx,
                     JX_Int                 number_syn_rlx,
                     JX_Real              measure_threshold_rlx,
                     JX_Int                 CF_init,
                     JX_Int                 debug_flag,
                     JX_Int               **CF_marker_ptr )
{
   MPI_Comm 	         comm      = jx_ParCSRMatrixComm(S);
   jx_ParCSRCommPkg     *comm_pkg  = jx_ParCSRMatrixCommPkg(A);
   jx_ParCSRCommHandle  *comm_handle = NULL;


   jx_CSRMatrix    *S_diag   = jx_ParCSRMatrixDiag(S);
   JX_Int             *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int             *S_diag_j = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix    *S_offd   = jx_ParCSRMatrixOffd(S);
   JX_Int             *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int             *S_offd_j = NULL;

   JX_Int 		   *col_map_offd  = jx_ParCSRMatrixColMapOffd(S);
   JX_Int              num_variables = jx_CSRMatrixNumRows(S_diag);
   JX_Int		    col_1 = jx_ParCSRMatrixFirstColDiag(S);
   JX_Int		    col_n = col_1 + jx_CSRMatrixNumCols(S_diag);
   JX_Int 		    num_cols_offd = 0;
                  
   jx_CSRMatrix    *S_ext   = NULL;
   JX_Int             *S_ext_i = NULL;
   JX_Int             *S_ext_j = NULL;

   JX_Int		    num_sends = 0;
   JX_Int  	   *int_buf_data;
   JX_Real	   *buf_data;

   JX_Int             *CF_marker;
   JX_Int             *CF_marker_offd;
                      
   JX_Real          *measure_array;
   JX_Int             *graph_array;
   JX_Int             *graph_array_offd;
   JX_Int              graph_size;
   JX_Int              graph_offd_size;
   JX_Int              global_graph_size;
                      
   JX_Int              i, j, k, kc, jS, kS, ig;
   JX_Int		    index, start, my_id, num_procs, jrow, cnt;
                      
   JX_Int              ierr = 0;
   JX_Int              break_var = 1;

   JX_Real	    wall_time  = 0.0;
   JX_Real           wall_time1 = 0.0;
   JX_Real           inset_time = 0.0;
   JX_Real           measure_max, measure_global, measure_th;
   JX_Int              iter = 0;

   S_ext = NULL;
   
   if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();
   
   jx_MPI_Comm_size(comm,&num_procs);
   jx_MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
   buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
 
   num_cols_offd = jx_CSRMatrixNumCols(S_offd);

   S_diag_j = jx_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = jx_CSRMatrixJ(S_offd);
   }
   
  /*----------------------------------------------------------
   * Compute the measures
   *
   * The measures are currently given by the column sums of S.
   * Hence, measure_array[i] is the number of influences
   * of variable i.
   *
   * The measures are augmented by a random number
   * between 0 and 1.
   *----------------------------------------------------------*/

   measure_array = jx_CTAlloc(JX_Real, num_variables + num_cols_offd);
   for (i = 0; i < S_offd_i[num_variables]; i ++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }

   if (num_procs > 1)
   {
      comm_handle = jx_ParCSRCommHandleCreate(2, comm_pkg, &measure_array[num_variables], buf_data);
   }

   for (i = 0; i < S_diag_i[num_variables]; i ++)
   { 
      measure_array[S_diag_j[i]] += 1.0;
   }

   if (num_procs > 1)
   {
      jx_ParCSRCommHandleDestroy(comm_handle);
   }

   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
      {
         measure_array[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)] += buf_data[index++];
      }
   }

   for (i = num_variables; i < num_variables + num_cols_offd; i ++)
   { 
      measure_array[i] = 0;
   }

   /* this augments the measures */
   jx_PAMGIndepSetRelaxInit(S, measure_array);
   
  /*--------------------------------------------------------------------------
   * Initialize the graph array
   * graph_array contains interior points in elements 0 ... num_variables-1
   * followed by boundary values
   *-------------------------------------------------------------------------*/

   graph_array = jx_CTAlloc(JX_Int, num_variables);
   if (num_cols_offd) 
   {
      graph_array_offd = jx_CTAlloc(JX_Int, num_cols_offd);
   }
   else
   {
      graph_array_offd = NULL;
   }

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_variables; ig ++)
   {
      graph_array[ig] = ig;
   }
   for (ig = 0; ig < num_cols_offd; ig ++)
   {
      graph_array_offd[ig] = ig;
   }

   /*-----------------------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ... 
    * num_variables-1  followed by boundary values
    *---------------------------------------------------------------*/

   graph_size = num_variables;
   graph_offd_size = num_cols_offd;

   if (CF_init)
   {
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i = 0; i < num_variables; i ++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0 || CF_marker[i] == -1)
         {
            CF_marker[i] = 0;
         }
         if ( CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 || (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               graph_size --;
               CF_marker[i] = F_PT;
            }
         }
         else
         {
            graph_array[cnt++] = i;
         }
      }
   }
   else
   {
      CF_marker = jx_CTAlloc(JX_Int, num_variables);
      for (i = 0; i < num_variables; i ++)
      {
         CF_marker[i] = 0;
      }
   }
   if (num_cols_offd)
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd);
   }
   else
   {
      CF_marker_offd = NULL;
   }
   for (i = 0; i < num_cols_offd; i ++)
   {
      CF_marker_offd[i] = 0;
   }

  /*---------------------------------------------------
   * Loop until all points are either fine or coarse.
   *---------------------------------------------------*/

   if (num_procs > 1)
   {
      S_ext   = jx_ParCSRMatrixExtractBExt(S, A, 0);
      S_ext_i = jx_CSRMatrixI(S_ext);
      S_ext_j = jx_CSRMatrixJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/
   index = 0;
   for (i = 0; i < num_cols_offd; i ++)
   {
      for (j = S_ext_i[i]; j < S_ext_i[i+1]; j ++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = jx_BinarySearch(col_map_offd, k, num_cols_offd);
	    if (kc > -1) 
	    {
	       S_ext_j[index++] = -kc - 1;
	    }
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i --)
   {
      S_ext_i[i] = S_ext_i[i-1];
   }
   
   if (num_procs > 1) 
   {
      S_ext_i[0] = 0;
   }

   if (debug_flag == 3)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d    Initialize CLJP phase = %f\n",my_id, wall_time); 
   }


   while (1)
   {
     /*-------------------------------------------------------------
      * Exchange boundary data, i.i. get measures and S_ext_data
      *------------------------------------------------------------*/

      if (num_procs > 1)
      {
   	 comm_handle = jx_ParCSRCommHandleCreate(2, comm_pkg, &measure_array[num_variables], buf_data);
      }
      if (num_procs > 1)
      {
   	 jx_ParCSRCommHandleDestroy(comm_handle);
      }

      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
         {
            measure_array[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)] += buf_data[index++];
         }
      }

     /*------------------------------------------------
      * Set F-pts and update subgraph
      *------------------------------------------------*/
      if (iter || !CF_init)
      {
         for (ig = 0; ig < graph_size; ig ++)
         {
            i = graph_array[ig];

            if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;
 
	       /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
            }
            if (CF_marker[i])
            {
               measure_array[i] = 0;
 
               /* take point out of the subgraph */
               graph_size --;
               graph_array[ig] = graph_array[graph_size];
               graph_array[graph_size] = i;
               ig --;
            }
         }
      }

     /*------------------------------------------------
      * Exchange boundary data, i.i. get measures 
      *------------------------------------------------*/

      if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();

      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
        start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
        {
            jrow = jx_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      { 
         comm_handle = jx_ParCSRCommHandleCreate(1, comm_pkg, buf_data, &measure_array[num_variables]);
         jx_ParCSRCommHandleDestroy(comm_handle);   
      } 


     /*------------------------------------------------
      * Test for convergence
      *------------------------------------------------*/

      jx_MPI_Allreduce(&graph_size,&global_graph_size,1,JX_MPI_INT,MPI_SUM,comm);


      if (global_graph_size == 0)
      {
         break;
      }

     /*------------------------------------------------
      * Pick an independent set of points with
      * maximal measure.
      *------------------------------------------------*/
      
      measure_max = 0.0;
      for (ig = 0; ig < graph_size; ig ++)
      {
      	i = graph_array[ig];
      	if (measure_array[i] > measure_max) 
      	{
      	   measure_max = measure_array[i];
      	}
      }
                 
      if (measure_type_rlx)
      {
         jx_MPI_Allreduce(&measure_max, &measure_global, 1, JX_MPI_REAL, MPI_MAX, comm); // Feb,24,2005
      }
      else
      {
         measure_global = measure_max; 
      }
         
      measure_th = measure_threshold_rlx*measure_global;     

      if (debug_flag == 1) wall_time1 = jx_time_getWallclockSeconds();
      
      if (iter || !CF_init)
      {
         jx_PAMGIndepSetRelax(S, S_ext, measure_array, graph_array, 
                              graph_size, graph_array_offd, graph_offd_size,
                              CF_marker, CF_marker_offd, number_syn_rlx, measure_th);			
      }		
      	
      if (debug_flag == 1) inset_time = inset_time + jx_time_getWallclockSeconds() - wall_time1;
      
      iter ++;
      
     /*------------------------------------------------
      * Exchange boundary data for CF_marker
      *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
         {
            int_buf_data[index++] = CF_marker[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
         jx_ParCSRCommHandleDestroy(comm_handle); 
      }
      
      for (ig = 0; ig < graph_offd_size; ig ++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] == F_PT)
         {
            /* take point out of the subgraph */
            graph_offd_size --;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig --;
         }
      }
      
      if (debug_flag == 3)
      {
         wall_time = jx_time_getWallclockSeconds() - wall_time;
         jx_printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",my_id, iter, wall_time); 
      }
      
     /*------------------------------------------------
      * Set C_pts and apply heuristics.
      *------------------------------------------------*/

      for (i = num_variables; i < num_variables+num_cols_offd; i ++)
      { 
         measure_array[i] = 0;
      }

      if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();
      
      for (ig = 0; ig < graph_size; ig ++)
      {
         i = graph_array[ig];

        /*---------------------------------------------
         * Heuristic: C-pts don't interpolate from
         * neighbors that influence them.
         *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {  
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {
                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS] - 1;
             
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j] --;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {
                  /* "remove" edge from S */
                  S_offd_j[jS] = - S_offd_j[jS] - 1;
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j+num_variables] --;
                  }
               }
            }
         }
	 else
    	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++)
            {
               j = S_diag_j[jS];
               
	       if (j < 0) 
	       {
	          j = - j - 1;
	       }
   
               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS] - 1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++)
            {
               j = S_offd_j[jS];
               
	       if (j < 0) 
	       {
	          j = - j - 1;
	       }
   
               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS] - 1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
            }
   
            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
   	          break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS ++)
                  {
                     k = S_diag_j[kS];
                     
		     if (k < 0) 
		     {
		        k = - k - 1;
		     }
   
                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = - S_diag_j[jS] - 1;
                        measure_array[j] --;
                        break_var = 0;
                        break;
                     }
                  }
   		  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS ++)
                     {
                        k = S_offd_j[kS];
                        
		        if (k < 0) 
		        {
		           k = - k - 1;
		        }
   
                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = - S_diag_j[jS] - 1;
                           measure_array[j] --;
                           break;
                        }
                     }
                  }
               }
            }
            
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];
   
                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS ++)
                  {
                     k = S_ext_j[kS];
   	             if (k >= 0)
   		     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = - S_offd_j[jS] - 1;
                           measure_array[j+num_variables] --;
                           break;
                        }
                     }
   		     else
   		     {
   		        kc = - k - 1;
   		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = - S_offd_j[jS] - 1;
                           measure_array[j+num_variables] --;
                           break;
   		        }
   		     }
                  }
               }
            }
         }

         /* reset CF_marker */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++)
	 {
            j = S_diag_j[jS];
            
	    if (j < 0) 
	    {
	       j = - j - 1;
	    }

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++)
         {
            j = S_offd_j[jS];
            
	    if (j < 0) 
	    {
	       j = - j - 1;
	    }

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }

      if (debug_flag == 3)
      {
         wall_time = jx_time_getWallclockSeconds() - wall_time;
         jx_printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                 my_id, wall_time, graph_size, num_cols_offd); 
      }      
   }
   
   if (debug_flag == 1)
   {
      jx_printf("Proc = %d   Inset_time = %f\n",my_id, inset_time);
   }

   /* Reset S_matrix */
   for (i = 0; i < S_diag_i[num_variables]; i ++)
   {
      if (S_diag_j[i] < 0)
      {
         S_diag_j[i] = - S_diag_j[i] - 1;
      }
   }
   for (i = 0; i < S_offd_i[num_variables]; i ++)
   {
      if (S_offd_j[i] < 0)
      {
         S_offd_j[i] = - S_offd_j[i] - 1;
      }
   }
   
  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   jx_TFree(measure_array);
   jx_TFree(graph_array);
   if (num_cols_offd) 
   {
      jx_TFree(graph_array_offd);
   }
   jx_TFree(buf_data);
   jx_TFree(int_buf_data);
   jx_TFree(CF_marker_offd);
   if (num_procs > 1) 
   {
      jx_CSRMatrixDestroy(S_ext);
   }

   *CF_marker_ptr = CF_marker;

   return (ierr);
}
