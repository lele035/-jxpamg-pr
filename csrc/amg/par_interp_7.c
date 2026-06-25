//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_7.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGBuildExtPICCInterp
 * \note: Only use FF when there is no common c point.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGBuildExtPICCInterp( jx_ParCSRMatrix   *par_A, 
                           JX_Int               *CF_marker,
			   jx_ParCSRMatrix   *par_S, 
			   JX_Int               *num_cpts_global,
			   JX_Int                num_functions, 
			   JX_Int               *dof_func, 
			   JX_Int                debug_flag,
			   JX_Real             trunc_factor, 
			   JX_Int                max_elmts, 
			   JX_Int               *col_offd_S_to_A,
			   jx_ParCSRMatrix  **P_ptr )
{
   /* Communication Variables */
   MPI_Comm 	     comm     = jx_ParCSRMatrixComm(par_A);   
   jx_ParCSRCommPkg  *comm_pkg = jx_ParCSRMatrixCommPkg(par_A);

   JX_Int my_id, num_procs;

   /* Variables to store input variables */
   jx_CSRMatrix *A_diag      = jx_ParCSRMatrixDiag(par_A); 
   JX_Real       *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int          *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int          *A_diag_j    = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix *A_offd      = jx_ParCSRMatrixOffd(par_A);   
   JX_Real       *A_offd_data = jx_CSRMatrixData(A_offd);
   JX_Int          *A_offd_i    = jx_CSRMatrixI(A_offd);
   JX_Int          *A_offd_j    = jx_CSRMatrixJ(A_offd);

   JX_Int           num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
   JX_Int          *col_map_offd    = jx_ParCSRMatrixColMapOffd(par_A);
   JX_Int           n_fine          = jx_CSRMatrixNumRows(A_diag);
   JX_Int           col_1           = jx_ParCSRMatrixFirstRowIndex(par_A);
   JX_Int           local_numrows   = jx_CSRMatrixNumRows(A_diag);
   JX_Int           col_n           = col_1 + local_numrows;
   JX_Int           total_global_cpts, my_first_cpt;

   /* Variables to store strong connection matrix info */
   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(par_S);
   JX_Int          *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int          *S_diag_j = jx_CSRMatrixJ(S_diag);
   
   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(par_S);   
   JX_Int          *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j = jx_CSRMatrixJ(S_offd);

   /* Interpolation matrix par_P */
   jx_ParCSRMatrix *par_P  = NULL;
   jx_CSRMatrix    *P_diag = NULL;
   jx_CSRMatrix    *P_offd = NULL;   

   JX_Real          *P_diag_data = NULL;
   JX_Int             *P_diag_i, *P_diag_j = NULL;
   JX_Real          *P_offd_data = NULL;
   JX_Int             *P_offd_i, *P_offd_j = NULL;

   JX_Int		   *col_map_offd_P = NULL;
   JX_Int              P_diag_size; 
   JX_Int              P_offd_size;
   JX_Int             *P_marker = NULL; 
   JX_Int             *P_marker_offd = NULL;
   JX_Int             *CF_marker_offd = NULL;
   JX_Int             *tmp_CF_marker_offd = NULL;
   JX_Int             *dof_func_offd = NULL;
//   JX_Int              ccounter_offd;
   JX_Int              common_c;

   /* Full row information for columns of A that are off diag*/
   jx_CSRMatrix *A_ext = NULL;   
   JX_Real       *A_ext_data = NULL;
   JX_Int          *A_ext_i = NULL;
   JX_Int          *A_ext_j = NULL;
  
   JX_Int          *fine_to_coarse = NULL;
   JX_Int          *fine_to_coarse_offd = NULL;
   JX_Int          *found = NULL;

   JX_Int           num_cols_P_offd = 0;
   JX_Int           newoff, loc_col;
   JX_Int           A_ext_rows, full_off_procNodes;
  
   jx_CSRMatrix *Sop   = NULL;
   JX_Int          *Sop_i = NULL;
   JX_Int          *Sop_j = NULL;
  
   JX_Int           Soprows, sgn;
  
   /* Variables to keep count of interpolatory points */
   JX_Int           jj_counter, jj_counter_offd;
   JX_Int           jj_begin_row, jj_end_row;
   JX_Int           jj_begin_row_offd = 0;
   JX_Int           jj_end_row_offd = 0;
   JX_Int           coarse_counter;
    
   /* Interpolation weight variables */
   JX_Real        sum, diagonal, distribute;
   JX_Int           strong_f_marker = -2;

   /* Loop variables */
   JX_Int           index;
   JX_Int           start_indexing = 0;
   JX_Int           i, i1, i2, j, jj, kk, k1, jj1;
//   JX_Int           ccounter;

   /* Definitions */
   JX_Real        zero = 0.0;
   JX_Real        one  = 1.0;

   jx_ParCSRCommPkg *extend_comm_pkg = NULL;

   /* BEGIN */
   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm,&my_id);

#ifdef JX_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) 
   {
      total_global_cpts = num_cpts_global[1];
   }
   jx_MPI_Bcast(&total_global_cpts, 1, JX_MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   /* Set up off processor information (specifically for neighbors of neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
      A_ext      = jx_ParCSRMatrixExtractBExt(par_A,par_A,1);
      A_ext_i    = jx_CSRMatrixI(A_ext);
      A_ext_j    = jx_CSRMatrixJ(A_ext);
      A_ext_data = jx_CSRMatrixData(A_ext);
      A_ext_rows = jx_CSRMatrixNumRows(A_ext);
     
      Sop        = jx_ParCSRMatrixExtractBExt(par_S,par_A,0);
      Sop_i      = jx_CSRMatrixI(Sop);
      Sop_j      = jx_CSRMatrixJ(Sop);
      Soprows    = jx_CSRMatrixNumRows(Sop);

      /* Find nodes that are neighbors of neighbors, not found in offd */
      newoff = jx_new_offd_nodes( &found, A_ext_rows, A_ext_i, A_ext_j, 
			          Soprows, col_map_offd, col_1, col_n, 
			          Sop_i, Sop_j, CF_marker, comm_pkg );
      if (newoff >= 0)
      {
         full_off_procNodes = newoff + num_cols_A_offd;
      }
      else
      {
         return jx_error_flag;
      }

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */

     /* AHB - create a new comm package just for extended info -
        this will work better with the assumed partition*/
      jx_ParCSRFindExtendCommPkg(par_A, newoff, found, &extend_comm_pkg);
     
      if (full_off_procNodes)
      {
         CF_marker_offd = jx_CTAlloc(JX_Int, full_off_procNodes);
      }
     
      if (num_functions > 1 && full_off_procNodes > 0)
      {
         dof_func_offd = jx_CTAlloc(JX_Int, full_off_procNodes);
      }
     
      jx_alt_insert_new_nodes(comm_pkg, extend_comm_pkg, CF_marker, full_off_procNodes, CF_marker_offd);
     
      if (num_functions > 1)
      {
         jx_alt_insert_new_nodes(comm_pkg, extend_comm_pkg, dof_func, full_off_procNodes, dof_func_offd);
      }

   } // end if (num_procs > 1)


  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/
   P_diag_i = jx_CTAlloc(JX_Int, n_fine + 1);
   P_offd_i = jx_CTAlloc(JX_Int, n_fine + 1);

   if (n_fine)
   {
      fine_to_coarse = jx_CTAlloc(JX_Int, n_fine);
      P_marker = jx_CTAlloc(JX_Int, n_fine);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = jx_CTAlloc(JX_Int, full_off_procNodes);
      fine_to_coarse_offd = jx_CTAlloc(JX_Int, full_off_procNodes);
      tmp_CF_marker_offd = jx_CTAlloc(JX_Int, full_off_procNodes);
   }

   jx_initialize_vecs( n_fine, full_off_procNodes, fine_to_coarse, 
		       fine_to_coarse_offd, P_marker, P_marker_offd, tmp_CF_marker_offd );
   
   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;

  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/
  
   for (i = 0; i < n_fine; i ++)
   {
      P_diag_i[i] = jj_counter;
      if (num_procs > 1)
      {
         P_offd_i[i] = jj_counter_offd;
      }
     
      if (CF_marker[i] >= 0)
      {
         jj_counter ++;
         fine_to_coarse[i] = coarse_counter;
         coarse_counter ++;
      }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
      else if (CF_marker[i] != -3)
      {
         /* Initialize ccounter for each f point */
//         ccounter = 0;
//         ccounter_offd = 0;
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
         { 
            /* search through diag to find all c neighbors */
            i1 = S_diag_j[jj];           
            if (CF_marker[i1] > 0)
            { 
               /* i1 is a C point */
               CF_marker[i1] = 2;
               if (P_marker[i1] < P_diag_i[i])
               {
                  P_marker[i1] = jj_counter;
                  jj_counter ++;
               }
            }
         } // end for jj
         
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
            { 
               /* search through offd to find all c neighbors */
               if (col_offd_S_to_A)
               {
                  i1 = col_offd_S_to_A[S_offd_j[jj]];
               }
               else
               {
                  i1 = S_offd_j[jj];
               }
            
               if (CF_marker_offd[i1] > 0)
               { 
                  /* i1 is a C point direct neighbor */
                  CF_marker_offd[i1] = 2;
                  if (P_marker_offd[i1] < P_offd_i[i])
                  {
                     tmp_CF_marker_offd[i1] = 1;
                     P_marker_offd[i1] = jj_counter_offd;
                     jj_counter_offd ++;
                  }
               }
            
            } // end if (num_procs > 1)
         
         } // if (CF_marker[i] >= 0)
      
      
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
         {
            /* Search diag to find f neighbors and determine if common c point */
            i1 = S_diag_j[jj];
	    if (CF_marker[i1] == -1)
	    { 
	       /* i1 is a F point, loop through it's strong neighbors */
	       common_c = 0;
	       for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk ++)
	       {
	          k1 = S_diag_j[kk];
	          if (CF_marker[k1] == 2)
	          {
	             common_c = 1;
	             break;
	          }
	       }
	    
	       if (num_procs > 1 && common_c == 0)
	       { 
	          /* no common c point yet, check offd */
	          for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk ++)
	          {
	             if (col_offd_S_to_A)
	             {
	                k1 = col_offd_S_to_A[S_offd_j[kk]];
	             }
	             else
	             {
	                k1 = S_offd_j[kk];
	             }

	             if (CF_marker_offd[k1] == 2)
	             { 
	                /* k1 is a c point check if it is common */
	                common_c = 1;
	                break;
	             }
	          } // end for kk
	       } // end if (num_procs > 1 && common_c == 0)
	    
	       if (!common_c)
	       { 
	          /* No common c point, extend the interp set */
	          for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk ++)
	          {
	             k1 = S_diag_j[kk];
	             if (CF_marker[k1] > 0)
	             {
	                if (P_marker[k1] < P_diag_i[i])
	                {
	                   P_marker[k1] = jj_counter;
	                   jj_counter ++;
	                }
	             } 
	          }
	       
	          if (num_procs > 1)
	          {
	             for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk ++)
	             {
	                if (col_offd_S_to_A)
	                {
	                   k1 = col_offd_S_to_A[S_offd_j[kk]];
	                }
	                else
	                {
	                   k1 = S_offd_j[kk];
	                }
	                if (CF_marker_offd[k1] >  0)
	                {
	                   if (P_marker_offd[k1] < P_offd_i[i])
	                   {
	                      tmp_CF_marker_offd[k1] = 1;
	                      P_marker_offd[k1] = jj_counter_offd;
	                      jj_counter_offd ++;
	                   }
	                }
	             } // end for kk
	          } // end if (num_procs > 1)
	       } // end if (num_procs > 1 && common_c == 0)
	    } // end if (CF_marker[i1] == -1)
         } // end for jj
      
         /* Look at off diag strong connections of i */ 
         if (num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
            { 
               i1 = S_offd_j[jj];           
               if (col_offd_S_to_A)
               {
                  i1 = col_offd_S_to_A[i1];
               }
            
               if (CF_marker_offd[i1] == -1)
               { 
            
                 /*---------------------------------------------------------------------
                  * F point; look at neighbors of i1. Sop contains global col
	          * numbers and entries that could be in S_diag or S_offd or neither. 
	          *---------------------------------------------------------------------*/
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk ++)
                  { 
                     /* Check if common c */
                     k1 = Sop_j[kk];
                     if (k1 >= col_1 && k1 < col_n)
                     { 
                        /* In S_diag */
                        loc_col = k1-col_1;
                        if (CF_marker[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                     else
                     { 
                        loc_col = - k1 - 1;
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                  }
               
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk ++)
                     { 
                        /* Check if common c */
                        k1 = Sop_j[kk];
                        if (k1 >= col_1 && k1 < col_n)
                        { 
                           /* In S_diag */
                           loc_col = k1 - col_1;
                           if (CF_marker[loc_col] >  0)
                           {
                              if (P_marker[loc_col] < P_diag_i[i])
                              {
                                 P_marker[loc_col] = jj_counter;
                                 jj_counter ++;
                              }
                           }
                        }
                        else
                        { 
                           loc_col = -k1 - 1;
                           if (CF_marker_offd[loc_col] >  0)
                           {
                              if (P_marker_offd[loc_col] < P_offd_i[i])
                              {
                                 P_marker_offd[loc_col] = jj_counter_offd;
                                 tmp_CF_marker_offd[loc_col] = 1;
                                 jj_counter_offd ++;
                              }
                           }
                        }
                     } // end for kk
                  } // end if (!common_c)
               }
            } // end for jj
         } // end if (num_procs > 1)
      
      
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
         { 
            /* search through diag to find all c neighbors */
	    i1 = S_diag_j[jj];           
	    if (CF_marker[i1] == 2)
	    {
	       CF_marker[i1] = 1;
	    }
         }
      
         if (num_procs > 1)
         {
	    for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
	    { 
	       /* search through offd to find all c neighbors */
	      if (col_offd_S_to_A)
	      {
	         i1 = col_offd_S_to_A[S_offd_j[jj]];
	      }
	      else
	      {
	         i1 = S_offd_j[jj];
	      }
	      if (CF_marker_offd[i1] == 2)
	      { 
	         /* i1 is a C point direct neighbor */
	        CF_marker_offd[i1] = 1;
	      }
	    }
         }
      }
   } // end for i
   
  /*-----------------------------------------------------------------------
   *  Allocate  arrays.
   *-----------------------------------------------------------------------*/

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {   
      P_diag_j    = jx_CTAlloc(JX_Int, P_diag_size);
      P_diag_data = jx_CTAlloc(JX_Real, P_diag_size);
   }

   if (P_offd_size)
   {   
      P_offd_j    = jx_CTAlloc(JX_Int, P_offd_size);
      P_offd_data = jx_CTAlloc(JX_Real, P_offd_size);
   }

   P_diag_i[n_fine] = jj_counter; 
   P_offd_i[n_fine] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
//   ccounter = start_indexing;
//   ccounter_offd = start_indexing;

   /* Fine to coarse mapping */
   if (num_procs > 1)
   {
      for (i = 0; i < n_fine; i ++)
      {
         fine_to_coarse[i] += my_first_cpt;
      }

      jx_alt_insert_new_nodes( comm_pkg, extend_comm_pkg, fine_to_coarse, 
		               full_off_procNodes, fine_to_coarse_offd );

      for (i = 0; i < n_fine; i ++)
      {
         fine_to_coarse[i] -= my_first_cpt;
      }
   }

   for (i = 0; i < n_fine; i ++)  
   {   
      P_marker[i] = -1;
   }
     
   for (i = 0; i < full_off_procNodes; i ++)
   {
      P_marker_offd[i] = -1;
   }

  /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
   
   for (i = 0; i < n_fine; i ++)
   {
      jj_begin_row = jj_counter;        
      
      if (num_procs > 1)
      {
         jj_begin_row_offd = jj_counter_offd;
      }

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
      if (CF_marker[i] >= 0)
      {
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter ++;
      }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
      else if (CF_marker[i] != -3)
      {
//         ccounter = 0;
//         ccounter_offd = 0;
         strong_f_marker --;

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
         { 
            /* Search C points only */
            i1 = S_diag_j[jj];   
	 
           /*--------------------------------------------------------------
            * If neighbor i1 is a C-point, set column number in P_diag_j
            * and initialize interpolation weight to zero.
            *--------------------------------------------------------------*/
	 
            if (CF_marker[i1] > 0)
            {
               CF_marker[i1]  = 2;
               if (P_marker[i1] < jj_begin_row)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter ++;
               }
            } // end if (CF_marker[i1] > 0)
         } // end for jj
         
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
            {
               if (col_offd_S_to_A)
               {
                  i1 = col_offd_S_to_A[S_offd_j[jj]];
               }
               else
               {
                  i1 = S_offd_j[jj];
               }
               
               if ( CF_marker_offd[i1] > 0)
               {
                  CF_marker_offd[i1] = 2;
                  if (P_marker_offd[i1] < jj_begin_row_offd)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd] = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd ++;
                  }
               }
            } // end for jj
         } // end if ( num_procs > 1)
         

         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
         { 
            /* Search through F points */
            i1 = S_diag_j[jj];
            
            if (CF_marker[i1] == -1) 
            {
               P_marker[i1] = strong_f_marker;
               common_c = 0;
               for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk ++)
               {
                  k1 = S_diag_j[kk];
                  if (CF_marker[k1] == 2)
                  {
                     common_c = 1;
                     break;
                  }
               }  // end for kk
                  
               if (num_procs > 1 && common_c == 0)
               { 
                  /* no common c point yet, check offd */
                  for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk ++)
                  {
                     if (col_offd_S_to_A)
                     {
                        k1 = col_offd_S_to_A[S_offd_j[kk]];
                     }
                     else
                     {
                        k1 = S_offd_j[kk];
                     }

                     if (CF_marker_offd[k1] == 2)
                     { 
                        /* k1 is a c point check if it is common */
                        common_c = 1;
                        break;
                     }
                  } // end for kk
               } // end if (num_procs > 1 && common_c == 0)
               
               if (!common_c)
               { 
                  /* No common c point, extend the interp set */
                  for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk ++)
                  {
                     k1 = S_diag_j[kk];
                     if (CF_marker[k1] >= 0)
                     {
                        if (P_marker[k1] < jj_begin_row)
                        {
                           P_marker[k1] = jj_counter;
                           P_diag_j[jj_counter] = fine_to_coarse[k1];
                           P_diag_data[jj_counter] = zero;
                           jj_counter ++;
                        }
                     }
                  }
                  
                  if (num_procs > 1)
                  {
                     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk ++)
                     {
                        if (col_offd_S_to_A)
                        {
                           k1 = col_offd_S_to_A[S_offd_j[kk]];
                        }
                        else
                        {
                           k1 = S_offd_j[kk];
                        }
                        
                        if (CF_marker_offd[k1] >= 0)
                        {
                           if (P_marker_offd[k1] < jj_begin_row_offd)
                           {
                              P_marker_offd[k1] = jj_counter_offd;
                              P_offd_j[jj_counter_offd] = k1;
                              P_offd_data[jj_counter_offd] = zero;
                              jj_counter_offd ++;
                           }
                        }
                     } // end for kk
                  } // end if (num_procs > 1)
               } // end if (!common_c)
            } // end if (CF_marker[i1] == -1) 
         } // end for jj
         
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
            {
               i1 = S_offd_j[jj];
               
               if (col_offd_S_to_A)
               {
                  i1 = col_offd_S_to_A[i1];
               }
               
               if (CF_marker_offd[i1] == -1)
               { 
                  /* F points that are off proc */
                  P_marker_offd[i1] = strong_f_marker;
                  common_c = 0;
                  for (kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk ++)
                  { 
                     /* Check if common c */
                     k1 = Sop_j[kk];
                     if (k1 >= col_1 && k1 < col_n)
                     { 
                        /* In S_diag */
                        loc_col = k1 - col_1;
                        if (CF_marker[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                     else
                     { 
                        loc_col = -k1 - 1;
                        if (CF_marker_offd[loc_col] == 2)
                        {
                           common_c = 1;
                           break;
                        }
                     }
                  } // end for kk
                  
                  if (!common_c)
                  {
                     for (kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk ++)
                     {
                        k1 = Sop_j[kk];
                        
                        /* Find local col number */
                        if (k1 >= col_1 && k1 < col_n)
                        {
                           loc_col = k1 - col_1;
                           if (CF_marker[loc_col] > 0)
                           {
                              if (P_marker[loc_col] < jj_begin_row)
                              {		
                                 P_marker[loc_col] = jj_counter;
                                 P_diag_j[jj_counter] = fine_to_coarse[loc_col];
                                 P_diag_data[jj_counter] = zero;
                                 jj_counter ++;
                              }
                           }
                        }
                        else
                        { 
                           loc_col = -k1 - 1;
                           if (CF_marker_offd[loc_col] > 0)
                           {
                              if (P_marker_offd[loc_col] < jj_begin_row_offd)
                              {
                                 P_marker_offd[loc_col] = jj_counter_offd;
                                 P_offd_j[jj_counter_offd] = loc_col;
                                 P_offd_data[jj_counter_offd] = zero;
                                 jj_counter_offd ++;
                              }
                           } // end if (CF_marker_offd[loc_col] > 0)
                        } // end if (k1 >= col_1 && k1 < col_n)
                     } // end for kk
                  } // end if (!common_c)
               }
            } // end for jj
         } // end if ( num_procs > 1)
         
         for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
         { 
            /* Search C points only */
            i1 = S_diag_j[jj];   
	 
           /*--------------------------------------------------------------
            * If neighbor i1 is a C-point, set column number in P_diag_j
            * and initialize interpolation weight to zero.
            *--------------------------------------------------------------*/
	 
            if (CF_marker[i1] == 2)
            {
               CF_marker[i1]  = 1;
            }
         } // end for jj
         
         if ( num_procs > 1)
         {
            for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
            {
               if (col_offd_S_to_A)
               {
                  i1 = col_offd_S_to_A[S_offd_j[jj]];
               }
               else
               {
                  i1 = S_offd_j[jj];
               }
               
               if ( CF_marker_offd[i1] == 2)
               {
                  CF_marker_offd[i1]  = 1;
               }
            }
         }


         jj_end_row = jj_counter;
         jj_end_row_offd = jj_counter_offd;
       
         diagonal = A_diag_data[A_diag_i[i]];
         
         for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
         { 
            /* i1 is a c-point and strongly influences i, accumulate a_(i,i1) into interpolation weight */
            i1 = A_diag_j[jj];
            if (P_marker[i1] >= jj_begin_row)
            {
               P_diag_data[P_marker[i1]] += A_diag_data[jj];
            }
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               sgn = 1;
               if (A_diag_data[A_diag_i[i1]] < 0) 
               {
                  sgn = -1;
               }
               
               for (jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1 ++)
               {
                  i2 = A_diag_j[jj1];
                  if ((P_marker[i2] >= jj_begin_row || i2 == i)  && (sgn*A_diag_data[jj1]) < 0)
                  {
                     sum += A_diag_data[jj1];
                  }
               }
               
               if (num_procs > 1)
               {
                  for (jj1 = A_offd_i[i1]; jj1< A_offd_i[i1+1]; jj1 ++)
                  {
                     i2 = A_offd_j[jj1];
                     if (P_marker_offd[i2] >= jj_begin_row_offd && (sgn*A_offd_data[jj1]) < 0)
                     {
                        sum += A_offd_data[jj1];
                     }
                  }
               }
               
               if (sum != 0)
               {
                  distribute = A_diag_data[jj] / sum;
                  
                  /* Loop over row of A for point i1 and do the distribution */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1 ++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
                     {
                        P_diag_data[P_marker[i2]] += distribute*A_diag_data[jj1];
                     }
               
                     if (i2 == i && (sgn*A_diag_data[jj1]) < 0)
                     {
                        diagonal += distribute*A_diag_data[jj1];
                     }
                  }
                  
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1 ++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd && (sgn*A_offd_data[jj1]) < 0)
                        {
                           P_offd_data[P_marker_offd[i2]] += distribute*A_offd_data[jj1];
                        }
                     }
                  }
               }
               else
               {
                  diagonal += A_diag_data[jj];
               }
            }
            
	    /* neighbor i1 weakly influences i, accumulate a_(i,i1) into diagonal */
	    else if (CF_marker[i1] != -3)
	    {
	       if (num_functions == 1 || dof_func[i] == dof_func[i1])
	       {
	          diagonal += A_diag_data[jj];
	       }
	    }
	 } // end for jj
	 
	 if (num_procs > 1)
	 {
	    for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
	    {
	       i1 = A_offd_j[jj];
	       if (P_marker_offd[i1] >= jj_begin_row_offd)
	       {
	          P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
	       }
	       else if (P_marker_offd[i1] == strong_f_marker)
	       {
	          sum = zero;
	          sgn = 1;
	          if (A_ext_data[A_ext_i[i1]] < 0) 
	          {
	             sgn = -1;
	          }
	          
	          for (jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1 ++)
	          {
                     k1 = A_ext_j[jj1];
                     if (k1 >= col_1 && k1 < col_n)
                     { 
                        /* diag */
                        loc_col = k1 - col_1;
                        if ((P_marker[loc_col] >= jj_begin_row || loc_col == i) && (sgn*A_ext_data[jj1]) < 0)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
                     else
                     { 
                        loc_col = -k1 - 1;
                        if (P_marker_offd[loc_col] >= jj_begin_row_offd && (sgn*A_ext_data[jj1]) < 0)
                        {
                           sum += A_ext_data[jj1];
                        }
                     }
	          } // end jj1
	       
	          if (sum != 0)
	          {
	             distribute = A_offd_data[jj] / sum;
	             for (jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1 ++)
	             {
	                k1 = A_ext_j[jj1];
	                if (k1 >= col_1 && k1 < col_n)
	                { 
	                   /* diag */
	                   loc_col = k1 - col_1;
	                   if (P_marker[loc_col] >= jj_begin_row && (sgn*A_ext_data[jj1]) < 0)
	                   {
	                      P_diag_data[P_marker[loc_col]] += distribute*A_ext_data[jj1];
	                   }
	                
	                   if (loc_col == i && (sgn*A_ext_data[jj1]) < 0)
	                   {
	                      diagonal += distribute*A_ext_data[jj1];
	                   }
	                }
	                else
	                { 
	                   loc_col = -k1 - 1;
	                   if (P_marker_offd[loc_col] >= jj_begin_row_offd && (sgn*A_ext_data[jj1]) < 0)
	                   {
	                      P_offd_data[P_marker_offd[loc_col]] += distribute*A_ext_data[jj1];
	                   }
	                }
	             }
	          }
	          else
	          {
	             diagonal += A_offd_data[jj];
	          }
	       }
	       else if (CF_marker_offd[i1] != -3)
	       {
	          if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	          {
	             diagonal += A_offd_data[jj];
	          }
	       }
	    }
         } // end if (num_procs > 1)
      
         for (jj = jj_begin_row; jj < jj_end_row; jj ++)
         {
            P_diag_data[jj] /= -diagonal;
         }
      
         for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
         {
            P_offd_data[jj] /= -diagonal;
         }
         strong_f_marker--;
      
      } 
   
      par_P = jx_ParCSRMatrixCreate( comm,
                                     jx_ParCSRMatrixGlobalNumRows(par_A),
                                     total_global_cpts,
                                     jx_ParCSRMatrixColStarts(par_A),
                                     num_cpts_global,
                                     0,
                                     P_diag_i[n_fine],
                                     P_offd_i[n_fine] );
               
      P_diag = jx_ParCSRMatrixDiag(par_P);
      jx_CSRMatrixData(P_diag) = P_diag_data;
      jx_CSRMatrixI(P_diag) = P_diag_i;
      jx_CSRMatrixJ(P_diag) = P_diag_j;
      P_offd = jx_ParCSRMatrixOffd(par_P);
      jx_CSRMatrixData(P_offd) = P_offd_data;
      jx_CSRMatrixI(P_offd) = P_offd_i;
      jx_CSRMatrixJ(P_offd) = P_offd_j;
      jx_ParCSRMatrixOwnsRowStarts(par_P) = 0;

      /* Compress P, removing coefficients smaller than trunc_factor * Max */
      if (trunc_factor != 0.0 || max_elmts > 0)
      {
         jx_PAMGInterpTruncation(par_P, trunc_factor, max_elmts);
         P_diag_data = jx_CSRMatrixData(P_diag);
         P_diag_i = jx_CSRMatrixI(P_diag);
         P_diag_j = jx_CSRMatrixJ(P_diag);
         P_offd_data = jx_CSRMatrixData(P_offd);
         P_offd_i = jx_CSRMatrixI(P_offd);
         P_offd_j = jx_CSRMatrixJ(P_offd);
         P_diag_size = P_diag_i[n_fine];
         P_offd_size = P_offd_i[n_fine];
      }

      /* This builds col_map, col_map should be monotone increasing and contain global numbers. */
      num_cols_P_offd = 0;
      if (P_offd_size)
      {
         jx_TFree(P_marker);
         if (full_off_procNodes)
         {
	    P_marker = jx_CTAlloc(JX_Int, full_off_procNodes);
         }     
         
         for (i = 0; i < full_off_procNodes; i ++)
         {
            P_marker[i] = 0;
         }
     
         num_cols_P_offd = 0;
         for (i = 0; i < P_offd_size; i ++)
         {
            index = P_offd_j[i];
            if (!P_marker[index])
            {
               if (tmp_CF_marker_offd[index] >= 0)
               {
                  num_cols_P_offd ++;
                  P_marker[index] = 1;
               }
            }
         }
     
         if (num_cols_P_offd)
         {
            col_map_offd_P = jx_CTAlloc(JX_Int, num_cols_P_offd);
         }
     
         index = 0;
         for (i = 0; i < num_cols_P_offd; i ++)
         {
            while( P_marker[index] == 0) 
            {
               index ++;
            }
            col_map_offd_P[i] = index ++;
         }
      
         for (i = 0; i < P_offd_size; i ++)
         {
            P_offd_j[i] = jx_BinarySearch(col_map_offd_P, P_offd_j[i], num_cols_P_offd);
         }

         index = 0;
         for (i = 0; i < num_cols_P_offd; i ++)
         {
            while (P_marker[index] == 0) 
            {
               index ++;
            }
            col_map_offd_P[i] = fine_to_coarse_offd[index];
            index ++;
         }

         /* Sort the col_map_offd_P and P_offd_j correctly */
         for (i = 0; i < num_cols_P_offd; i ++)
         {
            P_marker[i] = col_map_offd_P[i];
         }

         /* Check if sort actually changed anything */
         if (jx_ssort(col_map_offd_P,num_cols_P_offd))
         {
            for (i = 0; i < P_offd_size; i ++)
            {
               for (j = 0; j < num_cols_P_offd; j ++)
               {
                  if (P_marker[P_offd_j[i]] == col_map_offd_P[j])
                  {
                     P_offd_j[i] = j;
                     j = num_cols_P_offd;
                  }
               } // end for j
            } // end for i
         } // end if (jx_ssort(col_map_offd_P,num_cols_P_offd))
         
      } // end if (P_offd_size)
   
   } // end for i

   if (num_cols_P_offd)
   { 
      jx_ParCSRMatrixColMapOffd(par_P) = col_map_offd_P;
      jx_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   jx_MatvecCommPkgCreate(par_P);
 
   for (i = 0; i < n_fine; i ++)
   {
      if (CF_marker[i] == -3) 
      {
         CF_marker[i] = -1;
      }
   }
 
   *P_ptr = par_P;

   /* Deallocate memory */   
   jx_TFree(fine_to_coarse);
   jx_TFree(P_marker);
   
   if (num_procs > 1) 
   {
      jx_CSRMatrixDestroy(Sop);
      jx_CSRMatrixDestroy(A_ext);
      jx_TFree(fine_to_coarse_offd);
      jx_TFree(P_marker_offd);
      jx_TFree(CF_marker_offd);
      jx_TFree(tmp_CF_marker_offd);
      if (num_functions > 1)
      {
         jx_TFree(dof_func_offd);
      }
      jx_TFree(found);
     
      jx_MatvecCommPkgDestroy(extend_comm_pkg);
   }
   
   return jx_error_flag;  
}
