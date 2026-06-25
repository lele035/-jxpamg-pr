//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_coarsen_ai.c
 *  Date: 2013/01/21
 */ 

#include "jx_pamg.h"

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

/*!
 * \fn JX_Int jx_PAMGCoarsenAI
 * \brief AI-based PMIS (Modified Independent Set ) coarsening routine.
 * \note Don't worry about strong F-F connections without a common C point.
 * SF_PT: special fine points, no strongly depend points (special case: isolated points). 
 * Z_PT: have strongly depends points, however, no strongly influence points. (depends points)
 * CF_init == 9: for XML coarsening,
 *               reset zero for CF_marker on ESS-parts while preversing CF_marker on AI-parts) 
 * \date 2011/09/03
 */
JX_Int
jx_PAMGCoarsenPMISAI( jx_ParCSRMatrix  *par_S,
                  jx_ParCSRMatrix  *par_A,
                  JX_Real           *AI_measure,
                  JX_Int               CF_init,
                  JX_Int               debug_flag,
                  JX_Int             **CF_marker_ptr )
{
   MPI_Comm               comm        = jx_ParCSRMatrixComm(par_S);
   jx_ParCSRCommPkg      *comm_pkg    = jx_ParCSRMatrixCommPkg(par_S);
   jx_ParCSRCommHandle   *comm_handle = NULL;

   jx_CSRMatrix    *S_diag    = jx_ParCSRMatrixDiag(par_S);
   JX_Int             *S_diag_i  = jx_CSRMatrixI(S_diag);
   JX_Int             *S_diag_j  = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix    *S_offd    = jx_ParCSRMatrixOffd(par_S);
   JX_Int             *S_offd_i  = jx_CSRMatrixI(S_offd);
   JX_Int             *S_offd_j  = NULL;

   JX_Int              num_variables = jx_CSRMatrixNumRows(S_diag);
   JX_Int 		    num_cols_offd = 0;

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
                      
   JX_Int              i, j, jS, ig;
   JX_Int              index, start, my_id, num_procs, jrow, cnt, elmt;
                      
   JX_Int              ierr = 0;

   JX_Real           wall_time = 0.0;
   JX_Int              iter = 0;
   JX_Real           mai_threshold = 0.1;


  /***********************************************************************
   *  BEFORE THE INDEPENDENT SET COARSENING LOOP:
   *   measure_array: calculate the measures, and communicate them
   *     (this array contains measures for both local and external nodes)
   *   CF_marker, CF_marker_offd: initialize CF_marker
   *     (separate arrays for local and external; 
   *      0=unassigned, negative=F point, positive=C point)
   ***********************************************************************/      

  /*-------------------------------------------------------------------------
   * Use the ParCSR strength matrix, par_S.
   *
   * For now, the "strength" of dependence/influence is defined in
   * the following way: i depends on j if
   *     aij > jx_max (k != i) aik,    aii < 0
   * or
   *     aij < jx_min (k != i) aik,    aii >= 0
   * Then S_ij = 1, else S_ij = 0.
   *
   * NOTE: S_data is not used; in stead, only strong columns are retained
   *       in S_j, which can then be used like S_data
   *-----------------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();
   
   jx_MPI_Comm_size(comm,&num_procs);
   jx_MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
   buf_data     = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
 
   num_cols_offd = jx_CSRMatrixNumCols(S_offd);

   S_diag_j = jx_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = jx_CSRMatrixJ(S_offd);
   }

  /*---------------------------------------------------------------
   * Compute the measures
   *
   * The measures are currently given by the column sums of par_S.
   * Hence, measure_array[i] is the number of influences
   * of variable i.
   *
   * The measures are augmented by a random number
   * between 0 and 1.
   *-------------------------------------------------------------*/

   measure_array = jx_CTAlloc(JX_Real, num_variables + num_cols_offd);

   /* first calculate the local part of the sums for the external nodes */
   for (i = 0; i < S_offd_i[num_variables]; i ++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
   {
      comm_handle = jx_ParCSRCommHandleCreate(2, comm_pkg, &measure_array[num_variables], buf_data);
   }

   /* calculate the local part for the local nodes */
   for (i = 0; i < S_diag_i[num_variables]; i ++)
   { 
      measure_array[S_diag_j[i]] += 1.0;
   }

   /* finish the communication */
   if (num_procs > 1)
   {
      jx_ParCSRCommHandleDestroy(comm_handle);
   }
       
   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
      {
         measure_array[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)] += buf_data[index++];
      }
   }

   /* set the measures of the external nodes to zero */
   for (i = num_variables; i < num_variables + num_cols_offd; i ++)
   { 
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   /* this augments the measures */
   if (CF_init == 2)
   {
      jx_PAMGIndepSetInit(par_S, measure_array, 1);
   }
   else
   {
      jx_PAMGIndepSetInit(par_S, measure_array, 0);
   }

  /*---------------------------------------------------
   * Add AI_measure to measure_array.
   *---------------------------------------------------*/
   for (i = 0; i < num_variables; i ++)
   { 
      //if (AI_measure[i] > 0.0)
      if (AI_measure[i] >= mai_threshold) 
         measure_array[i] += 1000*AI_measure[i];
   }

  /*---------------------------------------------------
   * Initialize the graph arrays, and CF_marker arrays
   *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd) 
   {
      graph_array_offd = jx_CTAlloc(JX_Int, num_cols_offd);
   }
   else
   {
      graph_array_offd = NULL;
   }

   for (ig = 0; ig < num_cols_offd; ig ++)
   {
      graph_array_offd[ig] = ig;
   }

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = jx_CTAlloc(JX_Int, num_variables);

   if (CF_init == 1)
   { 
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i = 0; i < num_variables; i ++)
      {
         if ( (S_offd_i[i+1] - S_offd_i[i]) > 0 || CF_marker[i] == -1)
	 {
            CF_marker[i] = 0;
	 }
         if (CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 || (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
         {
            measure_array[i] = 0;
         }
         else
         {
            graph_array[cnt++] = i;
         }
      }
   }
   else if (CF_init == 9)
   {
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i = 0; i < num_variables; i ++)
      {
         //if ( S_offd_i[i+1] - S_offd_i[i]) > 0 || CF_marker[i] == -1)
         //if ( AI_measure[i] < 0.1 || CF_marker[i] == -1)
         //if ( AI_measure[i] < 0.1)
         if (AI_measure[i] < mai_threshold || CF_marker[i] == -1) 
	 {
            CF_marker[i] = 0;
	 } 
         if (CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 || (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
         {
            measure_array[i] = 0;
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
      cnt = 0;
      for (i = 0; i < num_variables; i ++)
      {
         CF_marker[i] = 0;
         //if ( (S_diag_i[i+1] - S_diag_i[i]) == 0 && (S_offd_i[i+1] - S_offd_i[i]) == 0)
	 if ( (S_diag_i[i+1] - S_diag_i[i]) == 0 && (S_offd_i[i+1]-S_offd_i[i]) == 0 && AI_measure[i] < mai_threshold)
         {
            CF_marker[i] = SF_PT;
            measure_array[i] = 0;
         }
         else
         {
            graph_array[cnt++] = i;
         }
         // graph_array[cnt++] = i;
      }
   }
   graph_size = cnt;

   /* now the off-diagonal part of CF_marker */
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
  
  
  /*--------------------------------------------------------
   * Communicate the local measures, which are complete,
   * to the external nodes
   *------------------------------------------------------*/
   
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
 
   if (debug_flag == 3)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d    Initialize CLJP phase = %f\n", my_id, wall_time); 
   }


  /*****************************************************************************
   * THE INDEPENDENT SET COARSENING LOOP:
   *****************************************************************************/      

  /*----------------------------------------------------
   * Loop until all points are either fine or coarse.
   *---------------------------------------------------*/

   while (1)
   {
   
      /* stop the coarsening if nothing left to be coarsened */
      jx_MPI_Allreduce(&graph_size, &global_graph_size, 1, JX_MPI_INT, MPI_SUM, comm);

      if (global_graph_size == 0)
      {
         break;
      }

     /*----------------------------------------------------------------
      * Pick an independent set of points with
      * maximal measure.
      * At the end, CF_marker is complete, but still needs to be
      * communicated to CF_marker_offd
      *---------------------------------------------------------------*/
      if (!CF_init || iter)
      {
         jx_PAMGIndepSet( par_S, measure_array, graph_array, 
                          graph_size, 
                          graph_array_offd, graph_offd_size, 
                          CF_marker, CF_marker_offd );

      /*-------------------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
       *  points to external points
       *------------------------------------------------------------*/

         if (num_procs > 1)
         {
            comm_handle = jx_ParCSRCommHandleCreate(12, comm_pkg, CF_marker_offd, int_buf_data);
            jx_ParCSRCommHandleDestroy(comm_handle);   
         }

         index = 0;
         for (i = 0; i < num_sends; i ++)
         {
            start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j=start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
            {
               elmt = jx_ParCSRCommPkgSendMapElmt(comm_pkg, j);
               if (!int_buf_data[index] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0; 
                  index ++;
               }
               else
               {
                  int_buf_data[index++] = CF_marker[elmt];
               }
            }
         }
         
 
         if (num_procs > 1)
         {
            comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
            jx_ParCSRCommHandleDestroy(comm_handle);   
         }
         
      } // end (!CF_init || iter)

      iter ++;


     /*------------------------------------------------
      * Set C-pts and F-pts.
      *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++) 
      {
      
         i = graph_array[ig];

        /*------------------------------------------------------
	 * If the measure of i is smaller than 1, then
         * make i and F point (because it does not influence
         * any other point), and remove all edges of
	 * equation i.
	 *-----------------------------------------------------*/

	 if (measure_array[i] < 1.0)
	 {
	    /* make point i an F point*/
	    CF_marker[i]= F_PT;

	    /* remove the edges in equation i */
	    /* first the local part */
	    for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) 
	    {
	       j = S_diag_j[jS];
	       if (j > -1)
	       { 
	          /* column number is still positive; not accounted for yet */
	          S_diag_j[jS]  = -S_diag_j[jS]-1;
	       }
	    }
	    
	    /* now the external part */
	    for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) 
	    {
	       j = S_offd_j[jS];
	       if (j > -1)
	       { 
	          /* column number is still positive; not accounted for yet */
	          S_offd_j[jS]  = -S_offd_j[jS]-1;
	       }
	    }
	    
	 } // end if (measure_array[i] < 1.0)


        /*--------------------------------------------------
         * First treat the case where point i is in the
         * independent set: make i a C point, 
         * take out all the graph edges for
         * equation i.
         *------------------------------------------------*/
       
         if (CF_marker[i] > 0) 
         {
            /* set to be a C-pt */
	    CF_marker[i] = C_PT;

            /* remove the edges in equation i */
	    /* first the local part */
	    for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++) 
	    {
               j = S_diag_j[jS];
               if (j > -1)
               { 
                  /* column number is still positive; not accounted for yet */
	          S_diag_j[jS]  = - S_diag_j[jS] - 1;
               }
            }
         
	    /* now the external part */
	    for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++) 
	    {
               j = S_offd_j[jS];
               if (j > -1)
               { 
                  /* column number is still positive; not accounted for yet */
                  S_offd_j[jS]  = - S_offd_j[jS] - 1;
               }
            }
         
         }


        /*-------------------------------------------------------
         * Now treat the case where point i is not in the
         * independent set: loop over
         * all the points j that influence equation i; if
         * j is a C point, then make i an F point.
         * If i is a new F point, then remove all the edges
         * from the graph for equation i.
         *-------------------------------------------------------*/

         else 
         {

            /* first the local part */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++) 
            {
               /* j is the column number, or the local number of the point influencing i */
               j = S_diag_j[jS];
               
               if (j < 0) 
               {
                  j = - j - 1;
               }

               if (CF_marker[j] > 0)
               { 
                  /* j is a C-point */
                  CF_marker[i] = F_PT;
               }
            }
            
            /* now the external part */
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++) 
            {
               j = S_offd_j[jS];
               
               if (j < 0)
               {
                  j = - j - 1;
               }
               
               if (CF_marker_offd[j] > 0)
               { 
                  /* j is a C-point */
                  CF_marker[i] = F_PT;
               }
            }

            /* remove all the edges for equation i if i is a new F point */
            if (CF_marker[i] == F_PT)
            {
               /* first the local part */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++) 
               {
                  j = S_diag_j[jS];
                  if (j > -1)
                  { 
                     /* column number is still positive; not accounted for yet */
                     S_diag_j[jS]  = - S_diag_j[jS] - 1;
                  }
               }
               
               /* now the external part */
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++) 
               {
                  j = S_offd_j[jS];
                  if (j > -1)
                  { 
                     /* column number is still positive; not accounted for yet */
                     S_offd_j[jS]  = -S_offd_j[jS] - 1;
                  }
               }
            }   
	    
         } // end if (CF_marker[i] > 0)
       
      } /* end first loop over graph */


     /*****************************************************************
      * now communicate CF_marker to CF_marker_offd, to make sure
      * that new external F points are known on this processor 
      *****************************************************************/

     /*------------------------------------------------
      * Exchange boundary data for CF_marker: send internal
      *   points to external points
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


     /*-------------------------------------------------------
      * Now loop over the points i in the unassigned
      * graph again. For all points i that are no new C or
      * F points, remove the edges in equation i that
      * connect to C or F points.
      * (We have removed the rows for the new C and F
      * points above; now remove the columns.)
      *------------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++) 
      {
         i = graph_array[ig];

         if (CF_marker[i] == 0) 
         {
            /* first the local part */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++) 
            {
               j = S_diag_j[jS];
               
               if (j < 0) 
               {
                  j = - j - 1;
               }

               if (!CF_marker[j] == 0 && S_diag_j[jS] > -1)
               { 
                  /* connection to C or F point, and column number is still positive; not accounted for yet */
                  S_diag_j[jS]  = -S_diag_j[jS]-1;
               }
            }
            
            /* now the external part */
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++) 
            {
               j = S_offd_j[jS];
               
               if (j < 0) 
               {
                  j = - j - 1;
               }

               if (!CF_marker_offd[j]==0 && S_offd_j[jS] > -1)
               { 
                  /* connection to C or F point, and column number is still positive; not accounted for yet */
                  S_offd_j[jS]  = -S_offd_j[jS]-1;
               }
            }
            
         } // end if (CF_marker[i] == 0) 
         
      } /* end second loop over graph */


     /*------------------------------------------------
      * Update subgraph
      *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++) 
      {
         i = graph_array[ig];
       
         if (!CF_marker[i]==0) /* C or F point */
	 {
            /* the independent set subroutine needs measure 0 for removed nodes */
            measure_array[i] = 0;
            /* take point out of the subgraph */
            graph_size --;
            graph_array[ig] = graph_array[graph_size];
            graph_array[graph_size] = i;
            ig --;
         }
      }
     
      for (ig = 0; ig < graph_offd_size; ig ++) 
      {
         i = graph_array_offd[ig];
       
         if (!CF_marker_offd[i] == 0) /* C or F point */
         {
            /* the independent set subroutine needs measure 0 for removed nodes */
            measure_array[i+num_variables] = 0;
            /* take point out of the subgraph */
            graph_offd_size --;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig --;
         }
      }
     
   } /* end while */


  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

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
   
   /*
   for (i = 0; i < num_variables; i ++)
   {
      if (CF_marker[i] == SF_PT) 
      {
         CF_marker[i] = F_PT;
      }
   }
   */

   jx_TFree(measure_array);
   jx_TFree(graph_array);
   if (num_cols_offd) 
   {
      jx_TFree(graph_array_offd);
   }
   jx_TFree(buf_data);
   jx_TFree(int_buf_data);
   jx_TFree(CF_marker_offd);
   
   /*
   if (num_procs > 1) 
   {
      jx_CSRMatrixDestroy(S_ext);
   }
   */

   *CF_marker_ptr = CF_marker;

   return (ierr);
}
