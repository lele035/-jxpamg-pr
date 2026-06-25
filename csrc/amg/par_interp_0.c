//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_0.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

JX_UInt jx_total_calloc_global_interp = 0; /* Yue Xiaoqiang 2012/10/13 */

JX_Int *Pmarkers_global_interp = NULL;             /* Feng Chunsheng & Yue Xiaoqiang 2012/10/17 */

/*!
 * \fn JX_Int jx_PAMGBuildInterp
 * \brief Modified Classical Interpolation.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGBuildInterp( jx_ParCSRMatrix   *par_A,
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
   MPI_Comm              comm     = jx_ParCSRMatrixComm(par_A);   
   jx_ParCSRCommPkg     *comm_pkg = jx_ParCSRMatrixCommPkg(par_A);
   jx_ParCSRCommHandle  *comm_handle;

   jx_CSRMatrix *A_diag      = jx_ParCSRMatrixDiag(par_A);
   JX_Real       *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int          *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int          *A_diag_j    = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix *A_offd      = jx_ParCSRMatrixOffd(par_A);   
   JX_Real       *A_offd_data = jx_CSRMatrixData(A_offd);
   JX_Int          *A_offd_i    = jx_CSRMatrixI(A_offd);
   JX_Int          *A_offd_j    = jx_CSRMatrixJ(A_offd);
   JX_Int           num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
   JX_Int          *col_map_offd = jx_ParCSRMatrixColMapOffd(par_A);

   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(par_S);
   JX_Int          *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int          *S_diag_j = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(par_S);   
   JX_Int          *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j = jx_CSRMatrixJ(S_offd);

   jx_ParCSRMatrix *par_P;
   JX_Int             *col_map_offd_P = NULL;

   JX_Int             *CF_marker_offd = NULL;
   JX_Int             *dof_func_offd  = NULL;

   jx_CSRMatrix    *A_ext = NULL;
   
   JX_Real          *A_ext_data = NULL;
   JX_Int             *A_ext_i    = NULL;
   JX_Int             *A_ext_j    = NULL;

   jx_CSRMatrix    *P_diag;
   jx_CSRMatrix    *P_offd;   

   JX_Real          *P_diag_data;
   JX_Int             *P_diag_i;
   JX_Int             *P_diag_j;
   JX_Real          *P_offd_data;
   JX_Int             *P_offd_i;
   JX_Int             *P_offd_j;

   JX_Int              P_diag_size, P_offd_size;
   
   JX_Int             *P_marker, *P_marker_offd;

   JX_Int              jj_counter,jj_counter_offd;
   JX_Int             *jj_count, *jj_count_offd;
   JX_Int              jj_begin_row,jj_begin_row_offd;
   JX_Int              jj_end_row,jj_end_row_offd;
   
   JX_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   JX_Int              n_fine = jx_CSRMatrixNumRows(A_diag);

   JX_Int              strong_f_marker;

   JX_Int             *fine_to_coarse;
   JX_Int             *fine_to_coarse_offd;
   JX_Int             *coarse_counter;
   JX_Int              coarse_shift;
   JX_Int              total_global_cpts;
   JX_Int              num_cols_P_offd,my_first_cpt;

   JX_Int              i, i1, i2;
   JX_Int              j, jl, jj, jj1;
   JX_Int              k, kc;
   JX_Int              start;
   JX_Int              sgn;
   JX_Int              c_num;
   
   JX_Real           diagonal;
   JX_Real           sum;
   JX_Real           distribute;          
   
   JX_Real           zero = 0.0;
   JX_Real           one  = 1.0;
   
   JX_Int              my_id;
   JX_Int              num_procs;
   JX_Int              num_threads;
   JX_Int              num_sends;
   JX_Int              index;
   JX_Int              ns, ne, size, rest;
   JX_Int             *int_buf_data;

   JX_Int              col_1 = jx_ParCSRMatrixFirstRowIndex(par_A);
   JX_Int              local_numrows = jx_CSRMatrixNumRows(A_diag);
   JX_Int              col_n = col_1 + local_numrows;

   JX_Real           wall_time = 0.0;  /* for debugging instrumentation  */

   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm, &my_id);
   num_threads = jx_NumThreads();


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


  /*-------------------------------------------------------------------
   * Get the CF_marker data for the off-processor columns
   *-------------------------------------------------------------------*/

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

   if (num_cols_A_offd) 
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }
   
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
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
	
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   jx_ParCSRCommHandleDestroy(comm_handle);
      
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
	 start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
	 {
            int_buf_data[index++] = dof_func[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
	
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n", my_id, wall_time);
      fflush(NULL);
   }

  /*----------------------------------------------------------------------
   * Get the ghost rows of A
   *---------------------------------------------------------------------*/

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

   if (num_procs > 1)
   {
      A_ext      = jx_ParCSRMatrixExtractBExt(par_A, par_A, 1);
      A_ext_i    = jx_CSRMatrixI(A_ext);
      A_ext_j    = jx_CSRMatrixJ(A_ext);
      A_ext_data = jx_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i ++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i+1]; j ++)
      {
         k = A_ext_j[j];
         if (k >= col_1 && k < col_n)
         {
            A_ext_j[index] = k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = jx_BinarySearch(col_map_offd, k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = - kc - 1;
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   
   for (i = num_cols_A_offd; i > 0; i --)
   {
      A_ext_i[i] = A_ext_i[i-1];
   }
   
   if (num_procs > 1) 
   {
      A_ext_i[0] = 0;
   }
   
   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n", my_id, wall_time);
      fflush(NULL);
   }


  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

   coarse_counter = jx_CTAlloc(JX_Int, num_threads);
   jj_count = jx_CTAlloc(JX_Int, num_threads);
   jj_count_offd = jx_CTAlloc(JX_Int, num_threads);

   fine_to_coarse = jx_CTAlloc(JX_Int, n_fine);
   
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < n_fine; i ++) 
   {
      fine_to_coarse[i] = -1;
   }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
  
  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
   
#define JX_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../../include/jx_smp_forloop.h"
   for (j = 0; j < num_threads; j ++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size*num_threads;
     
      if (j < rest)
      {
         ns = j*size + j;
         ne = (j+1)*size + j + 1;
      }
      else
      {
         ns = j*size + rest;
         ne = (j+1)*size + rest;
      }
     
      for (i = ns; i < ne; i ++)
      {
      
        /*--------------------------------------------------------------------
         *  If i is a C-point, interpolation is the identity. Also set up
         *  mapping vector.
         *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j] ++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j] ++;
         }
      
        /*--------------------------------------------------------------------
         *  If i is an F-point, interpolation is from the C-points that
         *  strongly influence i.
         *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
            {
               i1 = S_diag_j[jj];           
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j] ++;
               }
            }

            if (num_procs > 1)
            {
               if (col_offd_S_to_A)
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = col_offd_S_to_A[S_offd_j[jj]];           
                     if (CF_marker_offd[i1] >= 0)
                     {
                        jj_count_offd[j] ++;
                     }
                  }
               }
               else
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = S_offd_j[jj];           
                     if (CF_marker_offd[i1] >= 0)
                     {
                        jj_count_offd[j] ++;
                     }
                  }
               }
              
            } // end if (num_procs > 1)
           
         } // end if (CF_marker[i] >= 0)
        
      } // end for i
     
   } // end for j


  /*-----------------------------------------------------------------------
   *  Allocate  arrays.
   *-----------------------------------------------------------------------*/

   for (i=0; i < num_threads-1; i ++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = jx_CTAlloc(JX_Int, n_fine + 1);
   P_diag_j    = jx_CTAlloc(JX_Int, P_diag_size);
   P_diag_data = jx_CTAlloc(JX_Real, P_diag_size);

   P_diag_i[n_fine] = jj_counter; 

   P_offd_size = jj_counter_offd;

   P_offd_i    = jx_CTAlloc(JX_Int, n_fine + 1);
   P_offd_j    = jx_CTAlloc(JX_Int, P_offd_size);
   P_offd_data = jx_CTAlloc(JX_Real, P_offd_size);


  /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d     Interp: Internal work 1 =     %f\n", my_id, wall_time);
      fflush(NULL);
   }

  /*-----------------------------------------------------------------------
   *  Send and receive fine_to_coarse info.
   *-----------------------------------------------------------------------*/ 

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

   fine_to_coarse_offd = jx_CTAlloc(JX_Int, num_cols_A_offd); 

#define JX_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../../include/jx_smp_forloop.h"
   for (j = 0; j < num_threads; j ++)
   {
      coarse_shift = 0;
      
      if (j > 0) 
      {
         coarse_shift = coarse_counter[j-1];
      }
      
      size = n_fine / num_threads;
      rest = n_fine - size*num_threads;
      
      if (j < rest)
      {
        ns = j*size + j;
        ne = (j+1)*size + j + 1;
      }
      else
      {
        ns = j*size + rest;
        ne = (j+1)*size + rest;
      }
      
      for (i = ns; i < ne; i ++)
      {
         fine_to_coarse[i] += my_first_cpt+coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
      {
         int_buf_data[index++] = fine_to_coarse[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }
	
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, fine_to_coarse_offd);  
   jx_ParCSRCommHandleDestroy(comm_handle);   

   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n", my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < n_fine; i ++) 
   {
      fine_to_coarse[i] -= my_first_cpt;
   }

  /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
    
#define JX_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd
#include "../../include/jx_smp_forloop.h"
   for (jl = 0; jl < num_threads; jl ++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size*num_threads;
      
      if (jl < rest)
      {
         ns = jl*size + jl;
         ne = (jl+1)*size + jl + 1;
      }
      else
      {
         ns = jl*size + rest;
         ne = (jl+1)*size + rest;
      }
      
      jj_counter = 0;
      
      if (jl > 0) 
      {
         jj_counter = jj_count[jl-1];
      }
      
      jj_counter_offd = 0;
      
      if (jl > 0) 
      {
         jj_counter_offd = jj_count_offd[jl-1];
      }

      P_marker = jx_CTAlloc(JX_Int, n_fine);
      
      if (num_cols_A_offd)
      {
         P_marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
      }
      else
      {
         P_marker_offd = NULL;
      }

      for (i = 0; i < n_fine; i ++)
      {      
         P_marker[i] = -1;
      }
      
      for (i = 0; i < num_cols_A_offd; i ++)
      {      
         P_marker_offd[i] = -1;
      }
      
      strong_f_marker = -2;
 
      for (i = ns; i < ne; i ++)
      {
             
        /*--------------------------------------------------------------------
         *  If i is a c-point, interpolation is the identity.
         *--------------------------------------------------------------------*/
      
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter ++;
         }
      
        /*--------------------------------------------------------------------
         *  If i is an F-point, build interpolation.
         *--------------------------------------------------------------------*/

         else // if if (CF_marker[i] < 0)
         {         
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
            {
               i1 = S_diag_j[jj];   

              /*--------------------------------------------------------------
               * If neighbor i1 is a C-point, set column number in P_diag_j
               * and initialize interpolation weight to zero.
               *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter ++;
               }

              /*--------------------------------------------------------------
               * If neighbor i1 is an F-point, mark it as a strong F-point
               * whose connection needs to be distributed.
               *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }            
            } // end for jj
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               if (col_offd_S_to_A)
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = col_offd_S_to_A[S_offd_j[jj]];   

                    /*-----------------------------------------------------------
                     * If neighbor i1 is a C-point, set column number in P_offd_j
                     * and initialize interpolation weight to zero.
                     *-----------------------------------------------------------*/

                     if (CF_marker_offd[i1] >= 0)
                     {
                        P_marker_offd[i1] = jj_counter_offd;
                        P_offd_j[jj_counter_offd]  = i1;
                        P_offd_data[jj_counter_offd] = zero;
                        jj_counter_offd ++;
                     }

                    /*-----------------------------------------------------------
                     * If neighbor i1 is an F-point, mark it as a strong F-point
                     * whose connection needs to be distributed.
                     *-----------------------------------------------------------*/

                     else if (CF_marker_offd[i1] != -3)
                     {
                        P_marker_offd[i1] = strong_f_marker;
                     }            
                  } // end for jj
               }
               else
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = S_offd_j[jj];   

                    /*-----------------------------------------------------------
                     * If neighbor i1 is a C-point, set column number in P_offd_j
                     * and initialize interpolation weight to zero.
                     *-----------------------------------------------------------*/

                     if (CF_marker_offd[i1] >= 0)
                     {
                        P_marker_offd[i1] = jj_counter_offd;
                        P_offd_j[jj_counter_offd]  = i1;
                        P_offd_data[jj_counter_offd] = zero;
                        jj_counter_offd ++;
                     }

                    /*-----------------------------------------------------------
                     * If neighbor i1 is an F-point, mark it as a strong F-point
                     * whose connection needs to be distributed.
                     *-----------------------------------------------------------*/

                     else if (CF_marker_offd[i1] != -3)
                     {
                        P_marker_offd[i1] = strong_f_marker;
                     }        
                         
                  } // end for jj
                  
               } // end if (col_offd_S_to_A)
               
            } // end if (num_procs > 1)
            
      
            jj_end_row_offd = jj_counter_offd;
         
            diagonal = A_diag_data[A_diag_i[i]];

     
            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
            {
               i1 = A_diag_j[jj];

              /*--------------------------------------------------------------
               * Case 1: neighbor i1 is a C-point and strongly influences i,
               * accumulate a_{i,i1} into the interpolation weight.
               *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
               }

              /*--------------------------------------------------------------
               * Case 2: neighbor i1 is an F-point and strongly influences i,
               * distribute a_{i,i1} to C-points that strongly infuence i.
               * Note: currently no distribution to the diagonal in this case.
               *--------------------------------------------------------------*/
            
               else if (P_marker[i1] == strong_f_marker)
               {
                  sum = zero;
               
                 /*-----------------------------------------------------------
                  * Loop over row of A for point i1 and calculate the sum
                  * of the connections to c-points that strongly influence i.
                  *-----------------------------------------------------------*/
                  
                  sgn = 1;
                  if (A_diag_data[A_diag_i[i1]] < 0) 
                  {
                     sgn = -1;
                  }
                  
                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1 ++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
                     {
                        sum += A_diag_data[jj1];
                     }
                  }

                  /* Off-Diagonal block part of row i1 */ 
                  if (num_procs > 1)
                  {              
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1 ++)
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
 
                    /*-----------------------------------------------------------
                     * Loop over row of A for point i1 and do the distribution.
                     *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1+1]; jj1 ++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]] += distribute * A_diag_data[jj1];
                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1 ++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd && (sgn*A_offd_data[jj1]) < 0)
                           {
                              P_offd_data[P_marker_offd[i2]] += distribute * A_offd_data[jj1]; 
                           }
                        }
                     }
                  }
                  else
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     {
                        diagonal += A_diag_data[jj];
                     }
                  }
                   
               } // end if (P_marker[i1] >= jj_begin_row)
            
              /*--------------------------------------------------------------
               * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
               * into the diagonal.
               *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  {
                     diagonal += A_diag_data[jj];
                  }
               } 

            }   // end for jj 
       

           /*----------------------------------------------------------------
            * Still looping over ith row of A. Next, loop over the 
            * off-diagonal part of A 
            *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
               {
                  i1 = A_offd_j[jj];

                 /*--------------------------------------------------------------
                  * Case 1: neighbor i1 is a C-point and strongly influences i,
                  * accumulate a_{i,i1} into the interpolation weight.
                  *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                  }

                 /*------------------------------------------------------------
                  * Case 2: neighbor i1 is an F-point and strongly influences i,
                  * distribute a_{i,i1} to C-points that strongly infuence i.
                  * Note: currently no distribution to the diagonal in this case.
                  *-----------------------------------------------------------*/
            
                  else if (P_marker_offd[i1] == strong_f_marker)
                  {
                     sum = zero;
               
                    /*---------------------------------------------------------
                     * Loop over row of A_ext for point i1 and calculate the sum
                     * of the connections to c-points that strongly influence i.
                     *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     sgn = 1;
                     if (A_ext_data[A_ext_i[c_num]] < 0) 
                     {
                        sgn = -1;
                     }
                     
                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1 ++)
                     {
                        i2 = A_ext_j[jj1];
                                         
                        if (i2 > -1)
                        {                            
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row && (sgn*A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        else                       
                        {                          
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2-1] >= jj_begin_row_offd && (sgn*A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
 
                        }

                     } // end for jj1

                     if (sum != 0)
                     {
                     
                        distribute = A_offd_data[jj] / sum;
                           
                       /*---------------------------------------------------------
                        * Loop over row of A_ext for point i1 and do 
                        * the distribution.
                        *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */
                          
                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1 ++)
                        {
                           i2 = A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */           
                           {
                              if (P_marker[i2] >= jj_begin_row && (sgn*A_ext_data[jj1]) < 0)
                              {
                                 P_diag_data[P_marker[i2]] += distribute * A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2-1] >= jj_begin_row_offd && (sgn*A_ext_data[jj1]) < 0)
                              {
                                 P_offd_data[P_marker_offd[-i2-1]] += distribute * A_ext_data[jj1];
                              }
                           }
                        } // end for jj1
                     }
                     else
                     {
                        if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        {
                           diagonal += A_offd_data[jj];
                        }
                     }
                     
                  } // end  if (P_marker_offd[i1] >= jj_begin_row_offd)
            

                 /*-----------------------------------------------------------
                  * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                  * into the diagonal.
                  *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     {
                        diagonal += A_offd_data[jj];
                     }
                  } 

               } // end for jj
            
            } // end if (num_procs > 1)          


           /*-----------------------------------------------------------------
            * Set interpolation weight by dividing by the diagonal.
            *-----------------------------------------------------------------*/

#if 0 /* Yue Xiaoqiang 2012/12/28 */
            if (diagonal == 0.0)
            {
               jx_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id,i);
               diagonal = A_diag_data[A_diag_i[i]];
            }

            for (jj = jj_begin_row; jj < jj_end_row; jj ++)
            {
               P_diag_data[jj] /= -diagonal;
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
            {
               P_offd_data[jj] /= -diagonal;
            }
#else
            if (diagonal == 0.0)
            {
               if (debug_flag < 0)
               {
                  jx_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id, i);
               }
               for (jj = jj_begin_row; jj < jj_end_row; jj ++)
               {
                  P_diag_data[jj] = 0.0;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
               {
                  P_offd_data[jj] = 0.0;
               }
            }
            else
            {
               for (jj = jj_begin_row; jj < jj_end_row; jj ++)
               {
                  P_diag_data[jj] /= -diagonal;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
               {
                  P_offd_data[jj] /= -diagonal;
               }
            }
#endif
         
         } // end if (CF_marker[i] >= 0)

         strong_f_marker--; 

         P_offd_i[i+1] = jj_counter_offd;
         
      } // end for i
      
      jx_TFree(P_marker);
      jx_TFree(P_marker_offd);
      
   } // end for jl

   par_P = jx_ParCSRMatrixCreate(comm,
                                 jx_ParCSRMatrixGlobalNumRows(par_A),
                                 total_global_cpts,
                                 jx_ParCSRMatrixColStarts(par_A),
                                 num_cpts_global,
                                 0,
                                 P_diag_i[n_fine],
                                 P_offd_i[n_fine]);
                                                                                  
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

   num_cols_P_offd = 0;
   
   if (P_offd_size)
   {
      P_marker = jx_CTAlloc(JX_Int, num_cols_A_offd);

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
      for (i = 0; i < num_cols_A_offd; i ++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i ++)
      {
	 index = P_offd_j[i];
	 if (!P_marker[index])
	 {
 	    num_cols_P_offd ++;
 	    P_marker[index] = 1;
  	 }
      }

      col_map_offd_P = jx_CTAlloc(JX_Int, num_cols_P_offd);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i ++)
      {
         while (P_marker[index] == 0) 
         {
            index ++;
         }
         col_map_offd_P[i] = index ++;
      }

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
      for (i = 0; i < P_offd_size; i ++)
      {
	P_offd_j[i] = jx_BinarySearch(col_map_offd_P, P_offd_j[i], num_cols_P_offd);
      }
      
      jx_TFree(P_marker); 
   }

   for (i = 0; i < n_fine; i ++)
   {
      if (CF_marker[i] == -3) 
      {
         CF_marker[i] = -1;
      }
   }

   if (num_cols_P_offd)
   { 
      jx_ParCSRMatrixColMapOffd(par_P) = col_map_offd_P;
      jx_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   jx_GetCommPkgRTFromCommPkgA(par_P,par_A, fine_to_coarse_offd);

   *P_ptr = par_P;

   jx_TFree(CF_marker_offd);
   jx_TFree(dof_func_offd);
   jx_TFree(int_buf_data);
   jx_TFree(fine_to_coarse);
   jx_TFree(fine_to_coarse_offd);
   jx_TFree(coarse_counter);
   jx_TFree(jj_count);
   jx_TFree(jj_count_offd);

   if (num_procs > 1) 
   {
      jx_CSRMatrixDestroy(A_ext);
   }

   return(0);  
}   


/*!
 * \fn JX_Int jx_PAMGInterpTruncation
 * \brief Truncation for interpolation.
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGInterpTruncation( jx_ParCSRMatrix *par_P,
                         JX_Real           trunc_factor,        
                         JX_Int              max_elmts )        
{
   jx_CSRMatrix *P_diag = jx_ParCSRMatrixDiag(par_P);
   JX_Int *P_diag_i = jx_CSRMatrixI(P_diag);
   JX_Int *P_diag_j = jx_CSRMatrixJ(P_diag);
   JX_Real *P_diag_data = jx_CSRMatrixData(P_diag);
   JX_Int    *P_diag_j_new;
   JX_Real *P_diag_data_new;

   jx_CSRMatrix *P_offd = jx_ParCSRMatrixOffd(par_P);
   JX_Int *P_offd_i = jx_CSRMatrixI(P_offd);
   JX_Int *P_offd_j = jx_CSRMatrixJ(P_offd);
   JX_Real *P_offd_data = jx_CSRMatrixData(P_offd);
   JX_Int    *P_offd_j_new;
   JX_Real *P_offd_data_new;

   JX_Int n_fine = jx_CSRMatrixNumRows(P_diag);
   JX_Int num_cols = jx_CSRMatrixNumCols(P_diag);
   JX_Int i, j, start_j;
   JX_Int ierr = 0;
   JX_Int next_open = 0;
   JX_Int now_checking = 0;
   JX_Int num_lost = 0;
   JX_Int next_open_offd = 0;
   JX_Int now_checking_offd = 0;
   JX_Int num_lost_offd = 0;
   JX_Int P_diag_size;
   JX_Int P_offd_size;
   JX_Int num_elmts;
   JX_Int cnt, cnt_diag, cnt_offd;
   JX_Real max_coef;
   JX_Real row_sum;
   JX_Real scale;

   if (trunc_factor > 0)
   {
      for (i = 0; i < n_fine; i ++)
      {
         max_coef = 0;
         for (j = P_diag_i[i]; j < P_diag_i[i+1]; j ++)
         {
            max_coef = (max_coef < fabs(P_diag_data[j])) ? fabs(P_diag_data[j]) : max_coef;
         }
         for (j = P_offd_i[i]; j < P_offd_i[i+1]; j ++)
         {
            max_coef = (max_coef < fabs(P_offd_data[j])) ? fabs(P_offd_data[j]) : max_coef;
         }
         max_coef *= trunc_factor;

         start_j = P_diag_i[i];
         P_diag_i[i] -= num_lost;
         row_sum = 0;
         scale = 0;
         for (j = start_j; j < P_diag_i[i+1]; j ++)
         {
            row_sum += P_diag_data[now_checking];
            if (fabs(P_diag_data[now_checking]) < max_coef)
            {
               num_lost ++;
               now_checking ++;
            }
            else
            {
	       scale += P_diag_data[now_checking];
               P_diag_data[next_open] = P_diag_data[now_checking];
               P_diag_j[next_open] = P_diag_j[now_checking];
               now_checking ++;
               next_open ++;
            }
         }

         start_j = P_offd_i[i];
         P_offd_i[i] -= num_lost_offd;

         for (j = start_j; j < P_offd_i[i+1]; j ++)
         {
            row_sum += P_offd_data[now_checking_offd];
            if (fabs(P_offd_data[now_checking_offd]) < max_coef)
            {
               num_lost_offd ++;
               now_checking_offd ++;
            }
            else
            {
	       scale += P_offd_data[now_checking_offd];
               P_offd_data[next_open_offd] = P_offd_data[now_checking_offd];
               P_offd_j[next_open_offd] = P_offd_j[now_checking_offd];
               now_checking_offd ++;
               next_open_offd ++;
            }
         }
         
         
         /* normalize row of P */

         if (scale != 0.)
         {
            if (scale != row_sum)
            {
               scale = row_sum/scale;
               for (j = P_diag_i[i]; j < (P_diag_i[i+1]-num_lost); j ++)
               {
      	          P_diag_data[j] *= scale;
      	       }
               for (j = P_offd_i[i]; j < (P_offd_i[i+1]-num_lost_offd); j ++)
               {
      	          P_offd_data[j] *= scale;
      	       }
            }
         }
      }
      P_diag_i[n_fine] -= num_lost;
      P_offd_i[n_fine] -= num_lost_offd;
      
   }
   
   
   if (max_elmts > 0)
   {
      JX_Int     P_mxnum, cnt1, rowlength;
      JX_Int    *P_aux_j;
      JX_Real *P_aux_data;

      rowlength = 0;
      if (n_fine)
      { 
         rowlength = P_diag_i[1] + P_offd_i[1];
      }
    
      P_mxnum = rowlength;
      for (i = 1; i < n_fine; i ++)
      {
         rowlength = P_diag_i[i+1] - P_diag_i[i] + P_offd_i[i+1] - P_offd_i[i];
         if (rowlength > P_mxnum) 
         {
            P_mxnum = rowlength;
         }
      }
      
      if (P_mxnum > max_elmts)
      {
         P_aux_j = jx_CTAlloc(JX_Int, P_mxnum);
         P_aux_data = jx_CTAlloc(JX_Real, P_mxnum);
         cnt_diag = 0;
         cnt_offd = 0;

         for (i = 0; i < n_fine; i ++)
         {
            row_sum = 0;
            num_elmts = P_diag_i[i+1] - P_diag_i[i] + P_offd_i[i+1] - P_offd_i[i];
            if (max_elmts < num_elmts)
            {
               cnt = 0;
               for (j = P_diag_i[i]; j < P_diag_i[i+1]; j ++)
               {
                  P_aux_j[cnt] = P_diag_j[j];
                  P_aux_data[cnt++] = P_diag_data[j];
                  row_sum += P_diag_data[j];
               }
               
               num_lost += cnt;
               cnt1 = cnt;
               
               for (j = P_offd_i[i]; j < P_offd_i[i+1]; j ++)
               {
                  P_aux_j[cnt] = P_offd_j[j] + num_cols;
                  P_aux_data[cnt++] = P_offd_data[j];
                  row_sum += P_offd_data[j];
               }
               num_lost_offd += cnt - cnt1;
               
               /* sort data */
               jx_qsort2abs(P_aux_j,P_aux_data,0,cnt-1);
               scale = 0;
               P_diag_i[i] = cnt_diag;
               P_offd_i[i] = cnt_offd;
               for (j = 0; j < max_elmts; j ++)
               {
                  scale += P_aux_data[j];
                  if (P_aux_j[j] < num_cols)
                  {
                     P_diag_j[cnt_diag] = P_aux_j[j];
                     P_diag_data[cnt_diag++] = P_aux_data[j];
                  }
                  else
                  {
                     P_offd_j[cnt_offd] = P_aux_j[j]-num_cols;
                     P_offd_data[cnt_offd++] = P_aux_data[j];
                  }
               }
               
               num_lost -= cnt_diag-P_diag_i[i];
               num_lost_offd -= cnt_offd-P_offd_i[i];
                  
               /* normalize row of P */

               if (scale != 0.)
               {
                  if (scale != row_sum)
                  {
                     scale = row_sum/scale;
                     for (j = P_diag_i[i]; j < cnt_diag; j ++)
                     {
                        P_diag_data[j] *= scale;
                     }
                     for (j = P_offd_i[i]; j < cnt_offd; j ++)
                     {
                        P_offd_data[j] *= scale;
                     }
                  }
               }
               
            } 
            else  // if (max_elmts >= num_elmts)
            {
               if (P_diag_i[i] != cnt_diag)
               {
                  start_j = P_diag_i[i];
                  P_diag_i[i] = cnt_diag;
                  for (j = start_j; j < P_diag_i[i+1]; j ++)
                  {
                     P_diag_j[cnt_diag] = P_diag_j[j];
                     P_diag_data[cnt_diag++] = P_diag_data[j];
                  }
               }
               else
               {
                  cnt_diag += P_diag_i[i+1] - P_diag_i[i];
               }
               
               if (P_offd_i[i] != cnt_offd)
               {
                  start_j = P_offd_i[i];
                  P_offd_i[i] = cnt_offd;
                  for (j = start_j; j < P_offd_i[i+1]; j ++)
                  {
                     P_offd_j[cnt_offd] = P_offd_j[j];
                     P_offd_data[cnt_offd++] = P_offd_data[j];
                  }
               }
               else
               {
                  cnt_offd += P_offd_i[i+1] - P_offd_i[i];
               }
            }
         }
         P_diag_i[n_fine] = cnt_diag;
         P_offd_i[n_fine] = cnt_offd;
         jx_TFree(P_aux_j);
         jx_TFree(P_aux_data);
         
      } // end if (P_mxnum > max_elmts)
      
   } // end if (max_elmts > 0)
   
   if (num_lost)
   {
      P_diag_size = P_diag_i[n_fine];
      P_diag_j_new = jx_CTAlloc(JX_Int, P_diag_size);
      P_diag_data_new = jx_CTAlloc(JX_Real, P_diag_size);
      for (i = 0; i < P_diag_size; i ++)
      {
	 P_diag_j_new[i] = P_diag_j[i];
	 P_diag_data_new[i] = P_diag_data[i];
      }
      jx_TFree(P_diag_j);
      jx_TFree(P_diag_data);
      jx_CSRMatrixJ(P_diag) = P_diag_j_new;
      jx_CSRMatrixData(P_diag) = P_diag_data_new;
      jx_CSRMatrixNumNonzeros(P_diag) = P_diag_size;
   }
   
   if (num_lost_offd)
   {
      P_offd_size = P_offd_i[n_fine];
      P_offd_j_new = jx_CTAlloc(JX_Int, P_offd_size);
      P_offd_data_new = jx_CTAlloc(JX_Real, P_offd_size);
      for (i = 0; i < P_offd_size; i ++)
      {
         P_offd_j_new[i] = P_offd_j[i];
	 P_offd_data_new[i] = P_offd_data[i];
      }
      jx_TFree(P_offd_j);
      jx_TFree(P_offd_data);
      jx_CSRMatrixJ(P_offd) = P_offd_j_new;
      jx_CSRMatrixData(P_offd) = P_offd_data_new;
      jx_CSRMatrixNumNonzeros(P_offd) = P_offd_size;
   }
   
   return ierr;
}


/*!
 * \fn JX_Int jx_GetCommPkgRTFromCommPkgA
 * \date 2011/09/03
 */ 
JX_Int
jx_GetCommPkgRTFromCommPkgA( jx_ParCSRMatrix *RT,
                             jx_ParCSRMatrix *par_A,
                             JX_Int             *fine_to_coarse_offd )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(RT);
   jx_ParCSRCommPkg *comm_pkg_A = jx_ParCSRMatrixCommPkg(par_A);
   JX_Int num_recvs_A = jx_ParCSRCommPkgNumRecvs(comm_pkg_A);
   JX_Int *recv_procs_A = jx_ParCSRCommPkgRecvProcs(comm_pkg_A);
   JX_Int *recv_vec_starts_A = jx_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   JX_Int num_sends_A = jx_ParCSRCommPkgNumSends(comm_pkg_A);
   JX_Int *send_procs_A = jx_ParCSRCommPkgSendProcs(comm_pkg_A);

   jx_ParCSRCommPkg *comm_pkg;
   
   JX_Int  num_recvs_RT;
   JX_Int *recv_procs_RT;   
   JX_Int *recv_vec_starts_RT;   
   JX_Int  num_sends_RT;
   JX_Int *send_procs_RT;   
   JX_Int *send_map_starts_RT;   
   JX_Int *send_map_elmts_RT;   

   JX_Int *col_map_offd_RT  = jx_ParCSRMatrixColMapOffd(RT);
   JX_Int  num_cols_offd_RT = jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(RT));
   JX_Int  first_col_diag   = jx_ParCSRMatrixFirstColDiag(RT);

   JX_Int i, j;
   JX_Int vec_len, vec_start;
   JX_Int num_procs, my_id;
   JX_Int ierr = 0;
   JX_Int num_requests;
   JX_Int offd_col, proc_num;
 
   JX_Int *proc_mark;
   JX_Int *change_array;

   MPI_Request *requests;
   MPI_Status  *status;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

  /*--------------------------------------------------------------------------
   * determine num_recvs, recv_procs and recv_vec_starts for RT
   *--------------------------------------------------------------------------*/

   proc_mark = jx_CTAlloc(JX_Int, num_recvs_A);

   for (i = 0; i < num_recvs_A; i ++)
   {
      proc_mark[i] = 0;
   }

   proc_num = 0;
   num_recvs_RT = 0;
   if (num_cols_offd_RT)
   {
      for (i = 0; i < num_recvs_A; i ++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j ++)
         {
            offd_col = col_map_offd_RT[proc_num];
            if (offd_col == j)
            {
                proc_mark[i] ++;
                proc_num ++;
                if (proc_num == num_cols_offd_RT) 
                {
                   break;
                }
            }
         }
         if (proc_mark[i]) 
         {
            num_recvs_RT ++;
         }
         if (proc_num == num_cols_offd_RT) 
         {
            break;
         }
      }
   }

   for (i = 0; i < num_cols_offd_RT; i ++)
   {
      col_map_offd_RT[i] = fine_to_coarse_offd[col_map_offd_RT[i]];
   }
 
   recv_procs_RT = jx_CTAlloc(JX_Int, num_recvs_RT);
   recv_vec_starts_RT = jx_CTAlloc(JX_Int, num_recvs_RT + 1);
 
   j = 0;
   recv_vec_starts_RT[0] = 0;
   for (i = 0; i < num_recvs_A; i ++)
   {
      if (proc_mark[i])
      {
         recv_procs_RT[j] = recv_procs_A[i];
         recv_vec_starts_RT[j+1] = recv_vec_starts_RT[j] + proc_mark[i];
         j ++;
      }
   }

  /*--------------------------------------------------------------------------
   * send num_changes to recv_procs_A and receive change_array from send_procs_A
   *--------------------------------------------------------------------------*/

   num_requests = num_recvs_A + num_sends_A;
   requests = jx_CTAlloc(MPI_Request, num_requests);
   status = jx_CTAlloc(MPI_Status, num_requests);

   change_array = jx_CTAlloc(JX_Int, num_sends_A);

   j = 0;
   for (i=0; i < num_sends_A; i ++)
   {
      jx_MPI_Irecv(&change_array[i], 1, JX_MPI_INT, send_procs_A[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs_A; i ++)
   {
      jx_MPI_Isend(&proc_mark[i], 1, JX_MPI_INT, recv_procs_A[i], 0, comm, &requests[j++]);
   }
   
   jx_MPI_Waitall(num_requests,requests,status);

   jx_TFree(proc_mark);
   
  /*--------------------------------------------------------------------------
   * if change_array[i] is 0 , omit send_procs_A[i] in send_procs_RT
   *--------------------------------------------------------------------------*/

   num_sends_RT = 0;
   for (i = 0; i < num_sends_A; i ++)
   {
      if (change_array[i]) 
      {
         num_sends_RT ++;
      }
   }

   send_procs_RT = jx_CTAlloc(JX_Int, num_sends_RT);
   send_map_starts_RT = jx_CTAlloc(JX_Int, num_sends_RT + 1);

   j = 0;
   send_map_starts_RT[0] = 0;
   for (i = 0; i < num_sends_A; i ++)
   {
      if (change_array[i]) 
      {
         send_procs_RT[j] = send_procs_A[i];
         send_map_starts_RT[j+1] = send_map_starts_RT[j] + change_array[i];
         j ++;
      }
   }

  /*--------------------------------------------------------------------------
   * generate send_map_elmts
   *--------------------------------------------------------------------------*/

   send_map_elmts_RT = jx_CTAlloc(JX_Int, send_map_starts_RT[num_sends_RT]);

   j = 0;
   for (i = 0; i < num_sends_RT; i ++)
   {
      vec_start = send_map_starts_RT[i];
      vec_len = send_map_starts_RT[i+1]-vec_start;
      jx_MPI_Irecv(&send_map_elmts_RT[vec_start],vec_len,JX_MPI_INT,send_procs_RT[i],0,comm,&requests[j++]);
   }

   for (i = 0; i < num_recvs_RT; i ++)
   {
      vec_start = recv_vec_starts_RT[i];
      vec_len = recv_vec_starts_RT[i+1] - vec_start;
      jx_MPI_Isend(&col_map_offd_RT[vec_start],vec_len,JX_MPI_INT,recv_procs_RT[i],0,comm,&requests[j++]);
   }
   
   jx_MPI_Waitall(j,requests,status);

   for (i = 0; i < send_map_starts_RT[num_sends_RT]; i ++)
   {
      send_map_elmts_RT[i] -= first_col_diag;
   } 
	
   comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);

   jx_ParCSRCommPkgComm(comm_pkg) = comm;
   jx_ParCSRCommPkgNumSends(comm_pkg) = num_sends_RT;
   jx_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs_RT;
   jx_ParCSRCommPkgSendProcs(comm_pkg) = send_procs_RT;
   jx_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs_RT;
   jx_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts_RT;
   jx_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts_RT;
   jx_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts_RT;

   jx_TFree(status);
   jx_TFree(requests);

   jx_ParCSRMatrixCommPkg(RT) = comm_pkg;
   jx_TFree(change_array);

   return ierr;
}

/*!
 * \fn JX_Int jx_PAMGBuildInterp1
 * \brief Modified Classical Interpolation for single processor with multi-core
 * \date 2012/10/12
 */
JX_Int
jx_PAMGBuildInterp1( jx_ParCSRMatrix   *par_A,
                     JX_Int               *CF_marker,
                     jx_ParCSRMatrix   *par_S,
                     JX_Int               *num_cpts_global,
                     JX_Int                num_functions,
                     JX_Int               *dof_func,
                     JX_Int                debug_flag,
                     JX_Real             trunc_factor,
                     JX_Int                max_elmts,
                     JX_Int               *col_offd_S_to_A,
                     jx_ParCSRMatrix  **P_ptr,
                     JX_Int               *opt_icor )
{
   MPI_Comm              comm     = jx_ParCSRMatrixComm(par_A);   
   jx_ParCSRCommPkg     *comm_pkg = jx_ParCSRMatrixCommPkg(par_A);
   jx_ParCSRCommHandle  *comm_handle;

   jx_CSRMatrix *A_diag      = jx_ParCSRMatrixDiag(par_A);
   JX_Real       *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int          *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int          *A_diag_j    = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix *A_offd      = jx_ParCSRMatrixOffd(par_A);   
   JX_Real       *A_offd_data = jx_CSRMatrixData(A_offd);
   JX_Int          *A_offd_i    = jx_CSRMatrixI(A_offd);
   JX_Int          *A_offd_j    = jx_CSRMatrixJ(A_offd);
   JX_Int           num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
   JX_Int          *col_map_offd = jx_ParCSRMatrixColMapOffd(par_A);

   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(par_S);
   JX_Int          *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int          *S_diag_j = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(par_S);   
   JX_Int          *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j = jx_CSRMatrixJ(S_offd);

   jx_ParCSRMatrix *par_P;
   JX_Int             *col_map_offd_P = NULL;

   JX_Int             *CF_marker_offd = NULL;
   JX_Int             *dof_func_offd  = NULL;

   jx_CSRMatrix    *A_ext = NULL;
   
   JX_Real          *A_ext_data = NULL;
   JX_Int             *A_ext_i    = NULL;
   JX_Int             *A_ext_j    = NULL;

   jx_CSRMatrix    *P_diag;
   jx_CSRMatrix    *P_offd;   

   JX_Real          *P_diag_data;
   JX_Int             *P_diag_i;
   JX_Int             *P_diag_j;
   JX_Real          *P_offd_data;
   JX_Int             *P_offd_i;
   JX_Int             *P_offd_j;

   JX_Int              P_diag_size, P_offd_size;
   
   JX_Int             *P_marker, *P_marker_offd;

   JX_Int             *P_markers; /* Yue Xiaoqiang 2012/10/12 */

   JX_Int              jj_counter,jj_counter_offd;
   JX_Int             *jj_count, *jj_count_offd;
   JX_Int              jj_begin_row,jj_begin_row_offd;
   JX_Int              jj_end_row,jj_end_row_offd;
   
   JX_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   JX_Int              n_fine = jx_CSRMatrixNumRows(A_diag);

   JX_Int              strong_f_marker;

   JX_Int             *fine_to_coarse;
   JX_Int             *fine_to_coarse_offd;
   JX_Int             *coarse_counter;
   JX_Int              coarse_shift;
   JX_Int              total_global_cpts;
   JX_Int              num_cols_P_offd,my_first_cpt;

   JX_Int              i, i1, i2;
   JX_Int              j, jl, jj, jj1;
   JX_Int              k, kc;
   JX_Int              start;
   JX_Int              sgn;
   JX_Int              c_num;

   JX_Int              S_end_row;  /* Yue Xiaoqiang 2012/10/12 */
   JX_Int              A_end_row;  /* Yue Xiaoqiang 2012/10/12 */
   JX_Int              A_end_row2; /* Yue Xiaoqiang 2012/10/12 */
   
   JX_Real           diagonal;
   JX_Real           sum;
   JX_Real           distribute;          
   
   JX_Real           zero = 0.0;
   JX_Real           one  = 1.0;
   
   JX_Int              my_id;
   JX_Int              num_procs;
   JX_Int              num_threads;
   JX_Int              num_sends;
   JX_Int              index;
   JX_Int              ns, ne, size, rest;
   JX_Int             *int_buf_data;

   JX_Int              xqy_nbl, xqy_nbr;       /* Yue Xiaoqiang 2012/10/12 */
   JX_Int             *xqy_icor_interp = NULL; /* Yue Xiaoqiang 2012/10/12 */

   JX_Int              col_1 = jx_ParCSRMatrixFirstRowIndex(par_A);
   JX_Int              local_numrows = jx_CSRMatrixNumRows(A_diag);
   JX_Int              col_n = col_1 + local_numrows;

   JX_Real           wall_time = 0.0;  /* for debugging instrumentation  */

   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm, &my_id);
   num_threads = jx_NumThreads();

   xqy_icor_interp = jx_CTAlloc(JX_Int, 2*num_threads+1);                  /* Yue Xiaoqiang 2012/10/12 */
   jx_CSRMatrixGetBandWidth(A_diag, &xqy_nbl, &xqy_nbr);                /* Yue Xiaoqiang 2012/10/12 */
   jx_IntegerArrayGetInterp(xqy_icor_interp, n_fine, xqy_nbl, xqy_nbr); /* Yue Xiaoqiang 2012/10/12 */

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


  /*-------------------------------------------------------------------
   * Get the CF_marker data for the off-processor columns
   *-------------------------------------------------------------------*/

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

   if (num_cols_A_offd) 
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }
   
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
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
	
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   jx_ParCSRCommHandleDestroy(comm_handle);
      
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
	 start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
	 {
            int_buf_data[index++] = dof_func[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
	
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);   
   }

   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n", my_id, wall_time);
      fflush(NULL);
   }

  /*----------------------------------------------------------------------
   * Get the ghost rows of A
   *---------------------------------------------------------------------*/

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

   if (num_procs > 1)
   {
      A_ext      = jx_ParCSRMatrixExtractBExt(par_A, par_A, 1);
      A_ext_i    = jx_CSRMatrixI(A_ext);
      A_ext_j    = jx_CSRMatrixJ(A_ext);
      A_ext_data = jx_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i ++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i+1]; j ++)
      {
         k = A_ext_j[j];
         if (k >= col_1 && k < col_n)
         {
            A_ext_j[index] = k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = jx_BinarySearch(col_map_offd, k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = - kc - 1;
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   
   for (i = num_cols_A_offd; i > 0; i --)
   {
      A_ext_i[i] = A_ext_i[i-1];
   }
   
   if (num_procs > 1) 
   {
      A_ext_i[0] = 0;
   }
   
   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n", my_id, wall_time);
      fflush(NULL);
   }


  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

   coarse_counter = jx_CTAlloc(JX_Int, num_threads);
   jj_count = jx_CTAlloc(JX_Int, num_threads);
   jj_count_offd = jx_CTAlloc(JX_Int, num_threads);

   fine_to_coarse = jx_CTAlloc(JX_Int, n_fine);
   
#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < n_fine; i ++) 
   {
      fine_to_coarse[i] = -1;
   }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
      
  
  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
   
#define JX_SMP_PRIVATE i,j,i1,jj,ns,ne,size,rest
#include "../../include/jx_smp_forloop.h"
   for (j = 0; j < num_threads; j ++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size*num_threads;
     
      if (j < rest)
      {
         ns = j*size + j;
         ne = (j+1)*size + j + 1;
      }
      else
      {
         ns = j*size + rest;
         ne = (j+1)*size + rest;
      }
     
      for (i = ns; i < ne; i ++)
      {
      
        /*--------------------------------------------------------------------
         *  If i is a C-point, interpolation is the identity. Also set up
         *  mapping vector.
         *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j] ++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j] ++;
         }
      
        /*--------------------------------------------------------------------
         *  If i is an F-point, interpolation is from the C-points that
         *  strongly influence i.
         *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj ++)
            {
               i1 = S_diag_j[jj];           
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j] ++;
               }
            }

            if (num_procs > 1)
            {
               if (col_offd_S_to_A)
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = col_offd_S_to_A[S_offd_j[jj]];           
                     if (CF_marker_offd[i1] >= 0)
                     {
                        jj_count_offd[j] ++;
                     }
                  }
               }
               else
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = S_offd_j[jj];           
                     if (CF_marker_offd[i1] >= 0)
                     {
                        jj_count_offd[j] ++;
                     }
                  }
               }
              
            } // end if (num_procs > 1)
           
         } // end if (CF_marker[i] >= 0)
        
      } // end for i
     
   } // end for j


  /*-----------------------------------------------------------------------
   *  Allocate  arrays.
   *-----------------------------------------------------------------------*/

   for (i=0; i < num_threads-1; i ++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = jx_CTAlloc(JX_Int, n_fine + 1);
   P_diag_j    = jx_CTAlloc(JX_Int, P_diag_size);
   P_diag_data = jx_CTAlloc(JX_Real, P_diag_size);

   P_diag_i[n_fine] = jj_counter; 

   P_offd_size = jj_counter_offd;

   P_offd_i    = jx_CTAlloc(JX_Int, n_fine + 1);
   P_offd_j    = jx_CTAlloc(JX_Int, P_offd_size);
   P_offd_data = jx_CTAlloc(JX_Real, P_offd_size);


  /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d     Interp: Internal work 1 =     %f\n", my_id, wall_time);
      fflush(NULL);
   }

  /*-----------------------------------------------------------------------
   *  Send and receive fine_to_coarse info.
   *-----------------------------------------------------------------------*/ 

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

   fine_to_coarse_offd = jx_CTAlloc(JX_Int, num_cols_A_offd); 

#define JX_SMP_PRIVATE i,j,ns,ne,size,rest,coarse_shift
#include "../../include/jx_smp_forloop.h"
   for (j = 0; j < num_threads; j ++)
   {
      coarse_shift = 0;
      
      if (j > 0) 
      {
         coarse_shift = coarse_counter[j-1];
      }
      
      size = n_fine / num_threads;
      rest = n_fine - size*num_threads;
      
      if (j < rest)
      {
        ns = j*size + j;
        ne = (j+1)*size + j + 1;
      }
      else
      {
        ns = j*size + rest;
        ne = (j+1)*size + rest;
      }
      
      for (i = ns; i < ne; i ++)
      {
         fine_to_coarse[i] += my_first_cpt+coarse_shift;
      }
   }
   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
      {
         int_buf_data[index++] = fine_to_coarse[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }

   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, fine_to_coarse_offd);  
   jx_ParCSRCommHandleDestroy(comm_handle);   

   if (debug_flag == 4)
   {
      wall_time = jx_time_getWallclockSeconds() - wall_time;
      jx_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n", my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) wall_time = jx_time_getWallclockSeconds();

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < n_fine; i ++) 
   {
      fine_to_coarse[i] -= my_first_cpt;
   }

   JX_Int minus_one_length_ysk = xqy_icor_interp[2*num_threads];             /* Yue Xiaoqiang 2012/10/12 */
   JX_Int Ysk_pos;                                                           /* Yue Xiaoqiang 2012/10/12 */
#if 0
   P_markers = jx_CTAlloc(JX_Int, minus_one_length_ysk);                     /* Yue Xiaoqiang 2012/10/12 */
#else
   if (Pmarkers_global_interp == NULL)                             /* Feng Chunsheng & Yue Xiaoqiang 2012/10/17 */
   {
      Pmarkers_global_interp = jx_CTAlloc(JX_Int, minus_one_length_ysk);
      jx_total_calloc_global_interp = minus_one_length_ysk;
   }
   else if (minus_one_length_ysk > jx_total_calloc_global_interp)  /* Feng Chunsheng & Yue Xiaoqiang 2012/10/17 */
   {
      Pmarkers_global_interp = jx_TReAlloc(Pmarkers_global_interp, JX_Int, minus_one_length_ysk);
      jx_printf("<-----Pmarkers_global_interp realloc %d-----%d----->\n",
                    jx_total_calloc_global_interp, minus_one_length_ysk);
      jx_total_calloc_global_interp = minus_one_length_ysk;
   }
   P_markers = Pmarkers_global_interp;
#endif
   jx_IntegerArraySetConstantValues(minus_one_length_ysk, P_markers, -1); /* Yue Xiaoqiang 2012/10/12 */
   jx_long_size_length_interp += n_fine * num_threads;                    /* Yue Xiaoqiang 2012/10/12 */
   jx_real_long_size_length_interp += minus_one_length_ysk;               /* Yue Xiaoqiang 2012/10/12 */

  /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
    
#define JX_SMP_PRIVATE i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,Ysk_pos,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd,S_end_row,A_end_row,A_end_row2
#include "../../include/jx_smp_forloop.h"
   for (jl = 0; jl < num_threads; jl ++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size*num_threads;
      
      if (jl < rest)
      {
         ns = jl*size + jl;
         ne = (jl+1)*size + jl + 1;
      }
      else
      {
         ns = jl*size + rest;
         ne = (jl+1)*size + rest;
      }
      
      jj_counter = 0;
      
      if (jl > 0) 
      {
         jj_counter = jj_count[jl-1];
      }
      
      jj_counter_offd = 0;
      
      if (jl > 0) 
      {
         jj_counter_offd = jj_count_offd[jl-1];
      }

#if 0
      P_marker = jx_CTAlloc(JX_Int, n_fine);
#else /* Yue Xiaoqiang 2012/10/12 */
      Ysk_pos = 0;
      for (i = jl-1; i >= 0; i --)
      {
         Ysk_pos += xqy_icor_interp[2*i+1];
      }
      P_marker = P_markers + Ysk_pos - xqy_icor_interp[2*jl];
#endif
      
      if (num_cols_A_offd)
      {
         P_marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
      }
      else
      {
         P_marker_offd = NULL;
      }

#if 0 /* Yue Xiaoqiang 2012/10/12 */
      for (i = 0; i < n_fine; i ++)
      {      
         P_marker[i] = -1;
      }
#endif
      
      for (i = 0; i < num_cols_A_offd; i ++)
      {      
         P_marker_offd[i] = -1;
      }
      
      strong_f_marker = -2;
 
      for (i = ns; i < ne; i ++)
      {
             
        /*--------------------------------------------------------------------
         *  If i is a c-point, interpolation is the identity.
         *--------------------------------------------------------------------*/
      
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter ++;
         }
      
        /*--------------------------------------------------------------------
         *  If i is an F-point, build interpolation.
         *--------------------------------------------------------------------*/

         else // if if (CF_marker[i] < 0)
         {         
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            jj_begin_row = jj_counter;

            S_end_row = S_diag_i[i+1]; /* Yue Xiaoqiang 2012/10/12 */
            for (jj = S_diag_i[i]; jj < S_end_row; jj ++)
            {
               i1 = S_diag_j[jj];   

              /*--------------------------------------------------------------
               * If neighbor i1 is a C-point, set column number in P_diag_j
               * and initialize interpolation weight to zero.
               *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter ++;
               }

              /*--------------------------------------------------------------
               * If neighbor i1 is an F-point, mark it as a strong F-point
               * whose connection needs to be distributed.
               *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }            
            } // end for jj
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               if (col_offd_S_to_A)
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = col_offd_S_to_A[S_offd_j[jj]];   

                    /*-----------------------------------------------------------
                     * If neighbor i1 is a C-point, set column number in P_offd_j
                     * and initialize interpolation weight to zero.
                     *-----------------------------------------------------------*/

                     if (CF_marker_offd[i1] >= 0)
                     {
                        P_marker_offd[i1] = jj_counter_offd;
                        P_offd_j[jj_counter_offd]  = i1;
                        P_offd_data[jj_counter_offd] = zero;
                        jj_counter_offd ++;
                     }

                    /*-----------------------------------------------------------
                     * If neighbor i1 is an F-point, mark it as a strong F-point
                     * whose connection needs to be distributed.
                     *-----------------------------------------------------------*/

                     else if (CF_marker_offd[i1] != -3)
                     {
                        P_marker_offd[i1] = strong_f_marker;
                     }            
                  } // end for jj
               }
               else
               {
                  for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj ++)
                  {
                     i1 = S_offd_j[jj];   

                    /*-----------------------------------------------------------
                     * If neighbor i1 is a C-point, set column number in P_offd_j
                     * and initialize interpolation weight to zero.
                     *-----------------------------------------------------------*/

                     if (CF_marker_offd[i1] >= 0)
                     {
                        P_marker_offd[i1] = jj_counter_offd;
                        P_offd_j[jj_counter_offd]  = i1;
                        P_offd_data[jj_counter_offd] = zero;
                        jj_counter_offd ++;
                     }

                    /*-----------------------------------------------------------
                     * If neighbor i1 is an F-point, mark it as a strong F-point
                     * whose connection needs to be distributed.
                     *-----------------------------------------------------------*/

                     else if (CF_marker_offd[i1] != -3)
                     {
                        P_marker_offd[i1] = strong_f_marker;
                     }        
                         
                  } // end for jj
                  
               } // end if (col_offd_S_to_A)
               
            } // end if (num_procs > 1)
            
      
            jj_end_row_offd = jj_counter_offd;
         
            diagonal = A_diag_data[A_diag_i[i]];

     
            /* Loop over ith row of A.  First, the diagonal part of A */
            A_end_row = A_diag_i[i+1]; /* Yue Xiaoqiang 2012/10/12 */
            for (jj = A_diag_i[i]+1; jj < A_end_row; jj ++)
            {
               i1 = A_diag_j[jj];

              /*--------------------------------------------------------------
               * Case 1: neighbor i1 is a C-point and strongly influences i,
               * accumulate a_{i,i1} into the interpolation weight.
               *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
               }

              /*--------------------------------------------------------------
               * Case 2: neighbor i1 is an F-point and strongly influences i,
               * distribute a_{i,i1} to C-points that strongly infuence i.
               * Note: currently no distribution to the diagonal in this case.
               *--------------------------------------------------------------*/
            
               else if (P_marker[i1] == strong_f_marker)
               {
                  sum = zero;
               
                 /*-----------------------------------------------------------
                  * Loop over row of A for point i1 and calculate the sum
                  * of the connections to c-points that strongly influence i.
                  *-----------------------------------------------------------*/
                  
                  sgn = 1;
                  if (A_diag_data[A_diag_i[i1]] < 0) 
                  {
                     sgn = -1;
                  }
                  
                  /* Diagonal block part of row i1 */
                  A_end_row2 = A_diag_i[i1+1]; /* Yue Xiaoqiang 2012/10/12 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_end_row2; jj1 ++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
                     {
                        sum += A_diag_data[jj1];
                     }
                  }

                  /* Off-Diagonal block part of row i1 */ 
                  if (num_procs > 1)
                  {              
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1 ++)
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
 
                    /*-----------------------------------------------------------
                     * Loop over row of A for point i1 and do the distribution.
                     *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     A_end_row2 = A_diag_i[i1+1]; /* Yue Xiaoqiang 2012/10/12 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_end_row2; jj1 ++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]] += distribute * A_diag_data[jj1];
                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1 ++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd && (sgn*A_offd_data[jj1]) < 0)
                           {
                              P_offd_data[P_marker_offd[i2]] += distribute * A_offd_data[jj1]; 
                           }
                        }
                     }
                  }
                  else
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     {
                        diagonal += A_diag_data[jj];
                     }
                  }
                   
               } // end if (P_marker[i1] >= jj_begin_row)
            
              /*--------------------------------------------------------------
               * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
               * into the diagonal.
               *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  {
                     diagonal += A_diag_data[jj];
                  }
               } 

            }   // end for jj 
       

           /*----------------------------------------------------------------
            * Still looping over ith row of A. Next, loop over the 
            * off-diagonal part of A 
            *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
               {
                  i1 = A_offd_j[jj];

                 /*--------------------------------------------------------------
                  * Case 1: neighbor i1 is a C-point and strongly influences i,
                  * accumulate a_{i,i1} into the interpolation weight.
                  *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                  }

                 /*------------------------------------------------------------
                  * Case 2: neighbor i1 is an F-point and strongly influences i,
                  * distribute a_{i,i1} to C-points that strongly infuence i.
                  * Note: currently no distribution to the diagonal in this case.
                  *-----------------------------------------------------------*/
            
                  else if (P_marker_offd[i1] == strong_f_marker)
                  {
                     sum = zero;
               
                    /*---------------------------------------------------------
                     * Loop over row of A_ext for point i1 and calculate the sum
                     * of the connections to c-points that strongly influence i.
                     *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     sgn = 1;
                     if (A_ext_data[A_ext_i[c_num]] < 0) 
                     {
                        sgn = -1;
                     }
                     
                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1 ++)
                     {
                        i2 = A_ext_j[jj1];
                                         
                        if (i2 > -1)
                        {                            
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row && (sgn*A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        else                       
                        {                          
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2-1] >= jj_begin_row_offd && (sgn*A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
 
                        }

                     } // end for jj1

                     if (sum != 0)
                     {
                     
                        distribute = A_offd_data[jj] / sum;
                           
                       /*---------------------------------------------------------
                        * Loop over row of A_ext for point i1 and do 
                        * the distribution.
                        *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */
                          
                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num+1]; jj1 ++)
                        {
                           i2 = A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */           
                           {
                              if (P_marker[i2] >= jj_begin_row && (sgn*A_ext_data[jj1]) < 0)
                              {
                                 P_diag_data[P_marker[i2]] += distribute * A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2-1] >= jj_begin_row_offd && (sgn*A_ext_data[jj1]) < 0)
                              {
                                 P_offd_data[P_marker_offd[-i2-1]] += distribute * A_ext_data[jj1];
                              }
                           }
                        } // end for jj1
                     }
                     else
                     {
                        if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        {
                           diagonal += A_offd_data[jj];
                        }
                     }
                     
                  } // end  if (P_marker_offd[i1] >= jj_begin_row_offd)
            

                 /*-----------------------------------------------------------
                  * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                  * into the diagonal.
                  *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     {
                        diagonal += A_offd_data[jj];
                     }
                  } 

               } // end for jj
            
            } // end if (num_procs > 1)          


           /*-----------------------------------------------------------------
            * Set interpolation weight by dividing by the diagonal.
            *-----------------------------------------------------------------*/

#if 0 /* Yue Xiaoqiang 2012/12/28 */
            if (diagonal == 0.0)
            {
               jx_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id,i); 
               diagonal = A_diag_data[A_diag_i[i]];
            }

            for (jj = jj_begin_row; jj < jj_end_row; jj ++)
            {
               P_diag_data[jj] /= -diagonal;
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
            {
               P_offd_data[jj] /= -diagonal;
            }
#else
            if (diagonal == 0.0)
            {
               if (debug_flag < 0)
               {
                  jx_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id, i);
               }
               for (jj = jj_begin_row; jj < jj_end_row; jj ++)
               {
                  P_diag_data[jj] = 0.0;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
               {
                  P_offd_data[jj] = 0.0;
               }
            }
            else
            {
               for (jj = jj_begin_row; jj < jj_end_row; jj ++)
               {
                  P_diag_data[jj] /= -diagonal;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj ++)
               {
                  P_offd_data[jj] /= -diagonal;
               }
            }
#endif
         
         } // end if (CF_marker[i] >= 0)

         strong_f_marker--; 

         P_offd_i[i+1] = jj_counter_offd;
         
      } // end for i
#if 0 /* Yue Xiaoqiang 2012/10/12 */
      jx_TFree(P_marker);
#endif
      jx_TFree(P_marker_offd);
      
   } // end for jl
#if 0 /* Yue Xiaoqiang 2012/10/17 */
   jx_TFree(P_markers);
#endif

   par_P = jx_ParCSRMatrixCreate(comm,
                                 jx_ParCSRMatrixGlobalNumRows(par_A),
                                 total_global_cpts,
                                 jx_ParCSRMatrixColStarts(par_A),
                                 num_cpts_global,
                                 0,
                                 P_diag_i[n_fine],
                                 P_offd_i[n_fine]);
                                                                                  
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

   num_cols_P_offd = 0;
   
   if (P_offd_size)
   {
      P_marker = jx_CTAlloc(JX_Int, num_cols_A_offd);

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
      for (i = 0; i < num_cols_A_offd; i ++)
      {
         P_marker[i] = 0;
      }

      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i ++)
      {
	 index = P_offd_j[i];
	 if (!P_marker[index])
	 {
 	    num_cols_P_offd ++;
 	    P_marker[index] = 1;
  	 }
      }

      col_map_offd_P = jx_CTAlloc(JX_Int, num_cols_P_offd);

      index = 0;
      for (i = 0; i < num_cols_P_offd; i ++)
      {
         while (P_marker[index] == 0) 
         {
            index ++;
         }
         col_map_offd_P[i] = index ++;
      }

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
      for (i = 0; i < P_offd_size; i ++)
      {
	P_offd_j[i] = jx_BinarySearch(col_map_offd_P, P_offd_j[i], num_cols_P_offd);
      }
      
      jx_TFree(P_marker); 
   }

   if (jx_CSRMatrixNumCols(jx_ParCSRMatrixDiag(par_P)))  /* Yue Xiaoqiang 2012/10/12 */
   {
      jx_IntegerArrayModFine2Coarse(n_fine, fine_to_coarse);
      jx_IntegerArrayGenerateIcor(n_fine, jx_CSRMatrixNumCols(jx_ParCSRMatrixDiag(par_P)),
                                    fine_to_coarse, xqy_nbl, xqy_nbr, CF_marker, opt_icor);
   }
   jx_TFree(xqy_icor_interp);                        /* Yue Xiaoqiang 2012/10/12 */

   for (i = 0; i < n_fine; i ++)
   {
      if (CF_marker[i] == -3) 
      {
         CF_marker[i] = -1;
      }
   }

   if (num_cols_P_offd)
   { 
      jx_ParCSRMatrixColMapOffd(par_P) = col_map_offd_P;
      jx_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   jx_GetCommPkgRTFromCommPkgA(par_P,par_A, fine_to_coarse_offd);

   *P_ptr = par_P;

   jx_TFree(CF_marker_offd);
   jx_TFree(dof_func_offd);
   jx_TFree(int_buf_data);
   jx_TFree(fine_to_coarse);
   jx_TFree(fine_to_coarse_offd);
   jx_TFree(coarse_counter);
   jx_TFree(jj_count);
   jx_TFree(jj_count_offd);

   if (num_procs > 1) 
   {
      jx_CSRMatrixDestroy(A_ext);
   }

   return(0);  
}
