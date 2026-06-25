//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_coarsen_ruge.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0 

/*!
 * \fn JX_Int jx_PAMGCoarsenRuge
 * \brief Ruge's coarsening algorithm.
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGCoarsenRuge( jx_ParCSRMatrix    *par_S,
                    jx_ParCSRMatrix    *par_A,
                    JX_Int                 measure_type,
                    JX_Int                 coarsen_type,
                    JX_Int                 debug_flag,
                    JX_Int               **CF_marker_ptr )
{
   MPI_Comm             comm     = jx_ParCSRMatrixComm(par_S);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_S);
   jx_ParCSRCommHandle *comm_handle;
   
   jx_CSRMatrix *S_diag        = jx_ParCSRMatrixDiag(par_S);
   jx_CSRMatrix *S_offd        = jx_ParCSRMatrixOffd(par_S);
   JX_Int          *S_i           = jx_CSRMatrixI(S_diag);
   JX_Int          *S_j           = jx_CSRMatrixJ(S_diag);
   JX_Int          *S_offd_i      = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j      = NULL;
   JX_Int           num_variables = jx_CSRMatrixNumRows(S_diag);
   JX_Int           num_cols_offd = jx_CSRMatrixNumCols(S_offd);
   JX_Int          *col_map_offd  = jx_ParCSRMatrixColMapOffd(par_S);
                  
   jx_CSRMatrix *S_ext   = NULL;
   JX_Int          *S_ext_i = NULL;
   JX_Int          *S_ext_j = NULL;
                 
   jx_CSRMatrix *ST;
   JX_Int          *ST_i;
   JX_Int          *ST_j;
                 
   JX_Int          *CF_marker;
   JX_Int          *CF_marker_offd = NULL;
   JX_Int           ci_tilde = -1;
   JX_Int           ci_tilde_mark = -1;
   JX_Int           ci_tilde_offd = -1;
   JX_Int           ci_tilde_offd_mark = -1;

   JX_Int          *measure_array;
   JX_Int          *graph_array;
   JX_Int 	        *int_buf_data = NULL;
   JX_Int 	        *ci_array = NULL;

   JX_Int           i, j, k, jS;
   JX_Int           ji, jj, jk, jm, index;
   JX_Int		 set_empty = 1;
   JX_Int		 C_i_nonempty = 0;
   JX_Int		 num_nonzeros;
   JX_Int		 num_procs, my_id;
   JX_Int		 num_sends = 0;
   JX_Int		 first_col = 0;
   JX_Int		 start = 0;
   JX_Int		 col_0 = 0;
   JX_Int		 col_n = 0;

   jx_LinkList   LoL_head;
   jx_LinkList   LoL_tail;

   JX_Int          *lists, *where;
   JX_Int           measure, new_meas;
   JX_Int           num_left, elmt;
   JX_Int           nabor, nabor_two;

   JX_Int           ierr = 0;
   JX_Int           use_commpkg_A = 0;
   JX_Int           break_var = 0;
   JX_Int           f_pnt = F_PT;
   JX_Real        wall_time = 0.0;

   if (coarsen_type < 0) coarsen_type = -coarsen_type;

  /*-------------------------------------------------------
   * Initialize the C/F marker, LoL_head, LoL_tail  arrays
   *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = jx_CTAlloc(JX_Int, num_variables);
   where = jx_CTAlloc(JX_Int, num_variables);

  /*--------------------------------------------------------------
   * Use a CSR strength matrix, par_S.
   *
   * For now, the "strength" of dependence/influence is defined in
   * the following way: i depends on j if
   *     aij > jx_max (k != i) aik,    aii < 0
   * or
   *     aij < jx_min (k != i) aik,    aii >= 0
   * Then S_ij = 1, else S_ij = 0.
   *
   * NOTE: the entries are negative initially, corresponding
   * to "unaccounted-for" dependence.
   *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      use_commpkg_A = 1;
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd) S_offd_j = jx_CSRMatrixJ(S_offd);

   jS = S_i[num_variables];

   ST = jx_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = jx_CTAlloc(JX_Int, num_variables + 1);
   ST_j = jx_CTAlloc(JX_Int, jS);
   jx_CSRMatrixI(ST) = ST_i;
   jx_CSRMatrixJ(ST) = ST_j;

  /*----------------------------------------------------------
   * generate transpose of S, ST
   *----------------------------------------------------------*/

   for (i = 0; i <= num_variables; i ++)
   {
      ST_i[i] = 0;
   }
 
   for (i = 0; i < jS; i ++)
   {
      ST_i[S_j[i]+1] ++;
   }
   
   for (i = 0; i < num_variables; i ++)
   {
      ST_i[i+1] += ST_i[i];
   }
   
   for (i = 0; i < num_variables; i ++)
   {
      for (j = S_i[i]; j < S_i[i+1]; j ++)
      {
	 index = S_j[j];
       	 ST_j[ST_i[index]] = i;
       	 ST_i[index] ++;
      }
   }
         
   for (i = num_variables; i > 0; i --)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;


  /*----------------------------------------------------------
   * Compute the measures
   *
   * The measures are given by the row sums of ST.
   * Hence, measure_array[i] is the number of influences
   * of variable i.
   * correct actual measures through adding influences from
   * neighbor processors
   *----------------------------------------------------------*/

   measure_array = jx_CTAlloc(JX_Int, num_variables);

   for (i = 0; i < num_variables; i ++)
   {
      measure_array[i] = ST_i[i+1] - ST_i[i];
   }

   /* special case for Falgout coarsening */
   if (coarsen_type == 6) 
   {
      f_pnt = Z_PT;
      coarsen_type = 1;
   }
   if (coarsen_type == 10)
   {
      f_pnt = Z_PT;
      coarsen_type = 11;
   }

   if ( (measure_type || (coarsen_type != 1 && coarsen_type != 11)) && num_procs > 1 ) 
   {
      if (use_commpkg_A)
      {
         S_ext = jx_ParCSRMatrixExtractBExt(par_S, par_A, 0);
      }
      else
      {
         S_ext = jx_ParCSRMatrixExtractBExt(par_S, par_S, 0);
      }
      S_ext_i = jx_CSRMatrixI(S_ext);
      S_ext_j = jx_CSRMatrixJ(S_ext);
      num_nonzeros = S_ext_i[num_cols_offd];
      first_col = jx_ParCSRMatrixFirstColDiag(par_S);
      col_0 = first_col - 1;
      col_n = col_0 + num_variables;
      if (measure_type)
      {
	 for (i = 0; i < num_nonzeros; i ++)
         {
            index = S_ext_j[i] - first_col;
            if (index > -1 && index < num_variables)
            {
               measure_array[index] ++;
            }
         } 
      } 
   }

  /*---------------------------------------------------
   * Loop until all points are either fine or coarse.
   *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();


   //================================//
   //    first coarsening phase      //
   //================================//

  /*----------------------------------------------------------------
   *   Initialize the lists
   *---------------------------------------------------------------*/

   CF_marker = jx_CTAlloc(JX_Int, num_variables);
   
   num_left = 0;
   for (j = 0; j < num_variables; j ++)
   {
      if ( (S_i[j+1] - S_i[j]) == 0 && (S_offd_i[j+1] - S_offd_i[j]) == 0 )
      {
         CF_marker[j] = SF_PT;
         measure_array[j] = 0;
      }
      else
      {
         CF_marker[j] = UNDECIDED;
         num_left ++;
      }
   } 

   for (j = 0; j < num_variables; j ++) 
   {    
      measure = measure_array[j];
      if (CF_marker[j] != SF_PT)
      {
         if (measure > 0)
         {
            jx_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
         }
         else
         {
            if (measure < 0) jx_printf("negative measure!\n");
            CF_marker[j] = f_pnt;
            for (k = S_i[j]; k < S_i[j+1]; k ++)
            {
               nabor = S_j[k];
               if (CF_marker[nabor] != SF_PT)
               {
                  if (nabor < j)
                  {
                     new_meas = measure_array[nabor];
	             if (new_meas > 0)
	             {
                        jx_remove_point(&LoL_head, &LoL_tail, new_meas, nabor, lists, where);
                     }

                     new_meas = ++(measure_array[nabor]);
                     jx_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor, lists, where);
                  }
	          else
                  {
                     new_meas = ++(measure_array[nabor]);
                  }
               }
            }
            --num_left;
         }
      }
   }

  
  /***********************************************************************************
   *
   *  Main loop of Ruge-Stueben first coloring pass.
   *
   *  WHILE there are still points to classify DO:
   *        1) find first point, i,  on list with max_measure
   *           make i a C-point, remove it from the lists
   *        2) For each point, j,  in S_i^T,
   *           a) Set j to be an F-point
   *           b) For each point, k, in S_j
   *                  move k to the list in LoL with measure one
   *                  greater than it occupies (creating new LoL
   *                  entry if necessary)
   *        3) For each point, j,  in S_i,
   *                  move j to the list in LoL with measure one
   *                  smaller than it occupies (creating new LoL
   *                  entry if necessary)
   *
   *************************************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = C_PT;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;
      
      jx_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
  
      for (j = ST_i[index]; j < ST_i[index+1]; j ++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = F_PT;
            measure = measure_array[nabor];

            jx_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor]; k < S_i[nabor+1]; k ++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  jx_remove_point(&LoL_head, &LoL_tail, measure, nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  jx_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index]; j < S_i[index+1]; j ++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            jx_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;
	
	    if (measure > 0)
	    {
               jx_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            }
	    else
	    {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor]; k < S_i[nabor+1]; k ++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     jx_remove_point(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);
                 
                     jx_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
                  }
               }
	    }
         }
      }
   }

   jx_TFree(measure_array);
   jx_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d    Coarsen 1st pass = %f\n", my_id, wall_time); 
   }

   jx_TFree(lists);
   jx_TFree(where);
   jx_TFree(LoL_head);
   jx_TFree(LoL_tail);

   if (coarsen_type == 11)
   {
      *CF_marker_ptr = CF_marker;
      return 0;
   }


   //==============================================================//
   //   second pass, check fine points for coarse neighbors        //
   //   for coarsen_type = 2, the second pass includes             //
   //   off-processore boundary points                             // 
   //==============================================================//
   
  /*---------------------------------------------------
   * Initialize the graph array
   *---------------------------------------------------*/

   graph_array = jx_CTAlloc(JX_Int, num_variables);

   for (i = 0; i < num_variables; i ++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();

   if (coarsen_type == 2)
   {
     /*------------------------------------------------
      * Exchange boundary data for CF_marker
      *------------------------------------------------*/
    
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg,num_sends));
    
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
    
      ci_array = jx_CTAlloc(JX_Int, num_cols_offd);
      for (i = 0; i < num_cols_offd; i ++)
      {
         ci_array[i] = -1;
      }
	
      for (i = 0; i < num_variables; i ++)
      {
	 if (ci_tilde_mark != i) ci_tilde = -1;
	 if (ci_tilde_offd_mark != i) ci_tilde_offd = -1;
         if (CF_marker[i] == -1)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
               {
                  graph_array[j] = i;
               }
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji ++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
               {
                  ci_array[j] = i;
               }
            }
            for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
            {
               j = S_j[ji];
               if (CF_marker[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_i[j]; jj < S_i[j+1]; jj ++)
                  {
                     index = S_j[jj];
                     if (graph_array[index] == i)
                     {
                        set_empty = 0;
                        break;
                     }
                  }
		  if (set_empty)
                  {
                     for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj ++)
                     {
                        index = S_offd_j[jj];
                        if (ci_array[index] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                  } 
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = 1;
                        if (ci_tilde > -1)
                        {
                           CF_marker[ci_tilde] = -1;
                           ci_tilde = -1;
                        }
                        if (ci_tilde_offd > -1)
                        {
                           CF_marker_offd[ci_tilde_offd] = -1;
                           ci_tilde_offd = -1;
                        }
                        C_i_nonempty = 0;
                        break_var = 0;
                        break;
                     }
                     else
                     {
                        ci_tilde = j;
                        ci_tilde_mark = i;
                        CF_marker[j] = 1;
                        C_i_nonempty = 1;
                        i --;
                        break_var = 0;
                        break;
                     }
                  }
               }
            }
            if (break_var)
            {
               for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji ++)
               {
                  j = S_offd_j[ji];
                  if (CF_marker_offd[j] == -1)
                  {
                     set_empty = 1;
                     for (jj = S_ext_i[j]; jj < S_ext_i[j+1]; jj ++)
                     {
                        index = S_ext_j[jj];
                        if (index > col_0 && index < col_n) /* index interior */
                        {
                           if (graph_array[index-first_col] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                        else
                        {
   		           jk = jx_BinarySearch(col_map_offd,index,num_cols_offd);
                           if (jk != -1)
                           {
                              if (ci_array[jk] == i)
                              {
                                 set_empty = 0;
                                 break;
                              }
                           }
                        }
                     }
                     if (set_empty)
                     {
                        if (C_i_nonempty)
                        {
                           CF_marker[i] = 1;
                           if (ci_tilde > -1)
                           {
                              CF_marker[ci_tilde] = -1;
                              ci_tilde = -1;
                           }
                           if (ci_tilde_offd > -1)
                           {
                              CF_marker_offd[ci_tilde_offd] = -1;
                              ci_tilde_offd = -1;
                           }
                           C_i_nonempty = 0;
                           break;
                        }
                        else
                        {
                           ci_tilde_offd = j;
                           ci_tilde_offd_mark = i;
                           CF_marker_offd[j] = 1;
                           C_i_nonempty = 1;
                           i --;
                           break;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   else
   {
      for (i = 0; i < num_variables; i ++)
      {
	 if (ci_tilde_mark != i) ci_tilde = -1;
         if (CF_marker[i] == -1)
         {
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] > 0)
   	       {
   	          graph_array[j] = i;
   	       }
    	    }
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] == -1)
   	       {
   	          set_empty = 1;
   	          for (jj = S_i[j]; jj < S_i[j+1]; jj ++)
   	          {
   		     index = S_j[jj];
   		     if (graph_array[index] == i)
   		     {
   		        set_empty = 0;
   		        break;
   		     }
   	          }
   	          if (set_empty)
   	          {
   		     if (C_i_nonempty)
   		     {
   		        CF_marker[i] = 1;
   		        if (ci_tilde > -1)
   		        {
   			   CF_marker[ci_tilde] = -1;
   		           ci_tilde = -1;
   		        }
   	    		C_i_nonempty = 0;
   		        break;
   		     }
   		     else
   		     {
   		        ci_tilde = j;
   		        ci_tilde_mark = i;
   		        CF_marker[j] = 1;
   		        C_i_nonempty = 1;
		        i --;
		        break;
		     }
	          }
	       }
	    }
	 }
      }
   }

   if (debug_flag == 3 && coarsen_type != 2)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d    Coarsen 2nd pass = %f\n", my_id, wall_time); 
   }



   //===============================================================//
   //  third pass, check boundary fine points for coarse neighbors  //
   //===============================================================//

   if (coarsen_type == 3 || coarsen_type == 4)
   {
      if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();

      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

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

      ci_array = jx_CTAlloc(JX_Int, num_cols_offd);
      for (i = 0; i < num_cols_offd; i ++)
      {
         ci_array[i] = -1;
      }
   }

   if (coarsen_type > 1 && coarsen_type < 5)
   { 
      for (i = 0; i < num_variables; i ++)
      {
         graph_array[i] = -1;
      }
      for (i = 0; i < num_cols_offd; i ++)
      {
	 if (ci_tilde_mark != i) ci_tilde = -1;
	 if (ci_tilde_offd_mark != i) ci_tilde_offd = -1;
         if (CF_marker_offd[i] == -1)
         {
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji ++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j > col_0 && j < col_n)
   	       {
   	          j = j - first_col;
   	          if (CF_marker[j] > 0)
   	          {
   	             graph_array[j] = i;
   	          }
   	       }
   	       else
   	       {
   		  jj = jx_BinarySearch(col_map_offd, j, num_cols_offd);
   		  if (jj != -1 && CF_marker_offd[jj] > 0)
   		  {
   	             ci_array[jj] = i;
   	          }
    	       }	
    	    }
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji ++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j > col_0 && j < col_n)
   	       {
   	          j = j - first_col;
   	          if ( CF_marker[j] == -1)
   	          {
   	             set_empty = 1;
   	             for (jj = S_i[j]; jj < S_i[j+1]; jj ++)
   	             {
   		        index = S_j[jj];
   		        if (graph_array[index] == i)
   		        {
   		           set_empty = 0;
   		           break;
   		        }
   	             }
   	             for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj ++)
   	             {
   		        index = S_offd_j[jj];
   		        if (ci_array[index] == i)
   		        {
   		           set_empty = 0;
   		           break;
   		        }
   	             }
   	             if (set_empty)
   	             {
   		        if (C_i_nonempty)
   		        {
   		           CF_marker_offd[i] = 1;
   		           if (ci_tilde > -1)
   		           {
   			      CF_marker[ci_tilde] = -1;
			      ci_tilde = -1;
   		           }
   		           if (ci_tilde_offd > -1)
   		           {
   			      CF_marker_offd[ci_tilde_offd] = -1;
			      ci_tilde_offd = -1;
   		           }
                           C_i_nonempty = 0;
   		           break;
   		        }
   		        else
   		        {
   		           ci_tilde = j;
   		           ci_tilde_mark = i;
   		           CF_marker[j] = 1;
   		           C_i_nonempty = 1;
   		           i --;
   		           break;
   		        }
   	             }
   	          }
   	       }
   	       else
   	       {
   		  jm = jx_BinarySearch(col_map_offd, j, num_cols_offd);
   		  if (jm != -1 && CF_marker_offd[jm] == -1)
   	          {
   	             set_empty = 1;
   	             for (jj = S_ext_i[jm]; jj < S_ext_i[jm+1]; jj ++)
   	             {
   		        index = S_ext_j[jj];
   		        if (index > col_0 && index < col_n) 
   		  	{
   		           if (graph_array[index-first_col] == i)
   		           {
   		              set_empty = 0;
   		              break;
   		           }
   	                }
   			else
   			{
   		           jk = jx_BinarySearch(col_map_offd, index, num_cols_offd);
   			   if (jk != -1)
   			   {
   		              if (ci_array[jk] == i)
   		              {
   		                 set_empty = 0;
   		                 break;
   		              }
   		           }
   	                }
   	             }
   	             if (set_empty)
   	             {
   		        if (C_i_nonempty)
   		        {
   		           CF_marker_offd[i] = 1;
   		           if (ci_tilde > -1)
   		           {
   			      CF_marker[ci_tilde] = -1;
   			      ci_tilde = -1;
   		           }
   		           if (ci_tilde_offd > -1)
   		           {
   			      CF_marker_offd[ci_tilde_offd] = -1;
   			      ci_tilde_offd = -1;
   		           }
                           C_i_nonempty = 0;
   		           break;
   		        }
   		        else
   		        {
   		           ci_tilde_offd = jm;
   		           ci_tilde_offd_mark = i;
   		           CF_marker_offd[jm] = 1;
   		           C_i_nonempty = 1;
   		           i --;
   		           break;
   		        }
   		     }
   	          }
   	       }
   	    }
         }
      }
      
     /*---------------------------------------------------------------------------------
      * Send boundary data for CF_marker back
      *--------------------------------------------------------------------------------*/
      if (num_procs > 1)
      {
         comm_handle = jx_ParCSRCommHandleCreate(12, comm_pkg, CF_marker_offd, int_buf_data);
         jx_ParCSRCommHandleDestroy(comm_handle);   
      }
   
      
     /* only CF_marker entries from larger procs are accepted  
	if coarsen_type = 4 coarse points are not overwritten  */
 
      index = 0;
      if (coarsen_type != 4)
      {
         for (i = 0; i < num_sends; i ++)
         {
	    start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (jx_ParCSRCommPkgSendProc(comm_pkg,i) > my_id)
	    {
               for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
               {
                  CF_marker[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)] = int_buf_data[index++]; 
               }
            }
	    else
	    {
	       index += jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start;
	    }
         }
      }
      else
      {
         for (i = 0; i < num_sends; i ++)
         {
	    start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (jx_ParCSRCommPkgSendProc(comm_pkg,i) > my_id)
	    {
              for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
              {
                 elmt = jx_ParCSRCommPkgSendMapElmt(comm_pkg, j);
                 if (CF_marker[elmt] != 1)
                 {
                    CF_marker[elmt] = int_buf_data[index];
                 }
		 index ++; 
              }
            }
	    else
	    {
	       index += jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start;
	    }
         }
      }
      
      if (debug_flag == 3)
      {
         wall_time = jx_time_getWallclockSeconds() - wall_time;
         if (coarsen_type == 4)
         {
            jx_printf("Proc = %d    Coarsen 3rd pass = %f\n", my_id, wall_time); 
         }
         if (coarsen_type == 3)
         {
            jx_printf("Proc = %d    Coarsen 3rd pass = %f\n", my_id, wall_time); 
         }
         if (coarsen_type == 2)
         {
            jx_printf("Proc = %d    Coarsen 2nd pass = %f\n", my_id, wall_time);
         } 
      }
   }

   if (coarsen_type == 5)
   {
     /*------------------------------------------------
      * Exchange boundary data for CF_marker
      *------------------------------------------------*/

      if (debug_flag == 3) wall_time = jx_time_getWallclockSeconds();
    
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
    
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
    
      ci_array = jx_CTAlloc(JX_Int, num_cols_offd);
      for (i = 0; i < num_cols_offd; i ++)
      {
         ci_array[i] = -1;
      }
      for (i = 0; i < num_variables; i ++)
      {
         graph_array[i] = -1;
      }

      for (i = 0; i < num_variables; i ++)
      {
         if (CF_marker[i] == -1 && (S_offd_i[i+1]-S_offd_i[i]) > 0)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
               {
                  graph_array[j] = i;
               }
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji ++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
               {
                  ci_array[j] = i;
               }
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji ++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_ext_i[j]; jj < S_ext_i[j+1]; jj ++)
                  {
                     index = S_ext_j[jj];
                     if (index > col_0 && index < col_n) /* index interior */
                     {
                        if (graph_array[index-first_col] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     else
                     {
   		        jk = jx_BinarySearch(col_map_offd, index, num_cols_offd);
                        if (jk != -1)
                        {
                           if (ci_array[jk] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                     }
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = -2;
                        C_i_nonempty = 0;
                        break;
                     }
                     else
                     {
                        C_i_nonempty = 1;
                        i --;
                        break;
                     }
                  }
               }
            }
         }
      }
      
      if (debug_flag == 3)
      {
         wall_time = jx_time_getWallclockSeconds() - wall_time;
         jx_printf("Proc = %d    Coarsen special points = %f\n", my_id, wall_time); 
      }

   }
   
   
  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   if (coarsen_type != 1)
   {   
      jx_TFree(CF_marker_offd);
      jx_TFree(int_buf_data);
      jx_TFree(ci_array);
   }   
   jx_TFree(graph_array);
   if ( (measure_type || (coarsen_type != 1 && coarsen_type != 11)) && num_procs > 1 )
   {
      jx_CSRMatrixDestroy(S_ext);
   } 
   
   *CF_marker_ptr = CF_marker;
   
   return (ierr);
}
