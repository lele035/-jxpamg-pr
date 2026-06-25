//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_rap_omp.c -- Computing the coarse operater in PAMG using OpenMP only.
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

JX_UInt jx_total_calloc_global_rap = 0; /* Yue Xiaoqiang 2012/10/13 */

JX_Int *Pmarkers_global_rap = NULL;             /* Feng Chunsheng & Yue Xiaoqiang 2012/10/17 */

/*!
 * \fn JX_Int jx_PAMGBuildCoarseOperatorOMP
 * \brief Build the coarse operator in PAMG using OpenMP only.
 * \date 2012/10/13
 */ 
JX_Int
jx_PAMGBuildCoarseOperatorOMP( jx_ParCSRMatrix  *RT,
                               jx_ParCSRMatrix  *par_A,
                               jx_ParCSRMatrix  *par_P,
                               jx_ParCSRMatrix **RAP_ptr,
                               JX_Int              *icor_yoo )

{
   MPI_Comm          comm = jx_ParCSRMatrixComm(par_A);

   jx_CSRMatrix     *RT_diag = jx_ParCSRMatrixDiag(RT);
   JX_Int               num_cols_diag_RT = jx_CSRMatrixNumCols(RT_diag);
   jx_ParCSRCommPkg *comm_pkg_RT = jx_ParCSRMatrixCommPkg(RT);

   jx_CSRMatrix     *A_diag = jx_ParCSRMatrixDiag(par_A);
   
   JX_Real           *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int              *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int              *A_diag_j    = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix     *A_offd = jx_ParCSRMatrixOffd(par_A);

   JX_Int  num_cols_diag_A = jx_CSRMatrixNumCols(A_diag);
   JX_Int  num_cols_offd_A = jx_CSRMatrixNumCols(A_offd);

   jx_CSRMatrix    *P_diag = jx_ParCSRMatrixDiag(par_P);
   
   JX_Real          *P_diag_data = jx_CSRMatrixData(P_diag);
   JX_Int             *P_diag_i    = jx_CSRMatrixI(P_diag);
   JX_Int             *P_diag_j    = jx_CSRMatrixJ(P_diag);

   //JX_Int  first_col_diag_P    = jx_ParCSRMatrixFirstColDiag(par_P);
   JX_Int  num_cols_diag_P     = jx_CSRMatrixNumCols(P_diag);
   JX_Int *coarse_partitioning = jx_ParCSRMatrixColStarts(par_P);
   JX_Int *RT_partitioning     = jx_ParCSRMatrixColStarts(RT);

   jx_ParCSRMatrix *RAP = NULL;

   jx_CSRMatrix    *RAP_diag = NULL;

   JX_Real          *RAP_diag_data = NULL;
   JX_Int             *RAP_diag_i    = NULL;
   JX_Int             *RAP_diag_j    = NULL;

   jx_CSRMatrix    *RAP_offd = NULL;

   JX_Int             *RAP_offd_i    = NULL;

   JX_Int              RAP_diag_size;
   JX_Int              RAP_offd_size;
   JX_Int              num_cols_offd_RAP = 0;
   
   jx_CSRMatrix    *R_diag = NULL;
   
   JX_Real          *R_diag_data = NULL;
   JX_Int             *R_diag_i    = NULL;
   JX_Int             *R_diag_j    = NULL;

   JX_Int             *P_ext_diag_i    = NULL;

   JX_Int             *P_ext_offd_i    = NULL;

   JX_Int             *P_marker     = NULL;
   JX_Int            **P_mark_array = NULL;
   JX_Int            **A_mark_array = NULL;
   JX_Int             *A_marker     = NULL;

   JX_Int              n_coarse, n_coarse_RT;
   JX_Int              square = 1;
   
   JX_Int              ic, i;
   JX_Int              i1, i2, i3, ii, ns, ne;
   JX_Int              jj1, jj2, jj3;
   
   JX_Int             *jj_cnt_diag = NULL;
   JX_Int             *jj_cnt_offd = NULL;
   JX_Int              jj_count_diag, jj_count_offd;
   JX_Int              jj_row_begin_diag, jj_row_begin_offd;
   JX_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */
   JX_Int              num_nz_cols_A;
   JX_Int              num_procs;
   JX_Int              num_threads;

   JX_Real           r_entry;
   JX_Real           r_a_product;
   JX_Real           r_a_p_product;
   
   JX_Real           zero = 0.0;

  /*-----------------------------------------------------------------------
   *  Copy ParCSRMatrix RT into CSRMatrix R so that we have row-wise access 
   *  to restriction .
   *-----------------------------------------------------------------------*/

   jx_MPI_Comm_size(comm,&num_procs);
   num_threads = jx_NumThreads();

   if (comm_pkg_RT)
   {
      //num_recvs_RT = jx_ParCSRCommPkgNumRecvs(comm_pkg_RT);
      //num_sends_RT = jx_ParCSRCommPkgNumSends(comm_pkg_RT);
      //send_map_starts_RT = jx_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
      //send_map_elmts_RT  = jx_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }

   jx_CSRMatrixTranspose(RT_diag, &R_diag, 1); 
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_cols_offd_RT) 
   {
      jx_CSRMatrixTranspose(RT_offd ,&R_offd, 1); 
      R_offd_data = jx_CSRMatrixData(R_offd);
      R_offd_i    = jx_CSRMatrixI(R_offd);
      R_offd_j    = jx_CSRMatrixJ(R_offd);
   }
#endif

  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for R. Also get sizes of fine and
   *  coarse grids.
   *-----------------------------------------------------------------------*/

   R_diag_data = jx_CSRMatrixData(R_diag);
   R_diag_i    = jx_CSRMatrixI(R_diag);
   R_diag_j    = jx_CSRMatrixJ(R_diag);

   n_coarse = jx_ParCSRMatrixGlobalNumCols(par_P);
   num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

   n_coarse_RT = jx_ParCSRMatrixGlobalNumCols(RT);
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (n_coarse != n_coarse_RT)
   {
      square = 0;
   }
#endif

  /*----------------------------------------------------------------------------
   *  Generate Ps_ext, i.e. portion of par_P that is stored on neighbor procs
   *  and needed locally for triple matrix product 
   *---------------------------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_procs > 1) 
   {
      Ps_ext      = jx_ParCSRMatrixExtractBExt(par_P, par_A,1);
      Ps_ext_data = jx_CSRMatrixData(Ps_ext);
      Ps_ext_i    = jx_CSRMatrixI(Ps_ext);
      Ps_ext_j    = jx_CSRMatrixJ(Ps_ext);
   }
#endif

   P_ext_diag_i = jx_CTAlloc(JX_Int, num_cols_offd_A + 1);
   P_ext_offd_i = jx_CTAlloc(JX_Int, num_cols_offd_A + 1);
   //P_ext_diag_size = 0;
   //P_ext_offd_size = 0;
   //last_col_diag_P = first_col_diag_P + num_cols_diag_P - 1;
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   for (i = 0; i < num_cols_offd_A; i ++)
   {
      for (j = Ps_ext_i[i]; j < Ps_ext_i[i+1]; j ++)
      {
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            P_ext_offd_size ++;
         }
         else
         {
            P_ext_diag_size ++;
         }
      }
      P_ext_diag_i[i+1] = P_ext_diag_size;
      P_ext_offd_i[i+1] = P_ext_offd_size;
   }
   
   if (P_ext_diag_size)
   {
      P_ext_diag_j = jx_CTAlloc(JX_Int, P_ext_diag_size);
      P_ext_diag_data = jx_CTAlloc(JX_Real, P_ext_diag_size);
   }
   if (P_ext_offd_size)
   {
      P_ext_offd_j = jx_CTAlloc(JX_Int, P_ext_offd_size);
      P_ext_offd_data = jx_CTAlloc(JX_Real, P_ext_offd_size);
   }
#endif
   //cnt_offd = 0;
   //cnt_diag = 0;
   //cnt = 0;
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   for (i = 0; i < num_cols_offd_A; i ++)
   {
      for (j = Ps_ext_i[i]; j < Ps_ext_i[i+1]; j ++)
      {
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            P_ext_offd_j[cnt_offd] = Ps_ext_j[j];
            P_ext_offd_data[cnt_offd++] = Ps_ext_data[j];
         }
         else
         {
            P_ext_diag_j[cnt_diag] = Ps_ext_j[j] - first_col_diag_P;
            P_ext_diag_data[cnt_diag++] = Ps_ext_data[j];
         }
      }
   }
   if (num_procs > 1) 
   {
      jx_CSRMatrixDestroy(Ps_ext);
      Ps_ext = NULL;
   }

   if (P_ext_offd_size || num_cols_offd_P)
   {
      temp = jx_CTAlloc(JX_Int, P_ext_offd_size + num_cols_offd_P);
      for (i = 0; i < P_ext_offd_size; i ++)
      {
         temp[i] = P_ext_offd_j[i];
      }
      cnt = P_ext_offd_size;
      for (i = 0; i < num_cols_offd_P; i ++)
      {
         temp[cnt++] = col_map_offd_P[i];
      }
   }
   if (cnt)
   {
      jx_qsort0(temp, 0, cnt-1);

      num_cols_offd_Pext = 1;
      value = temp[0];
      for (i = 1; i < cnt; i ++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_Pext++] = value;
         }
      }
   }
 
   if (num_cols_offd_Pext)
   {
      col_map_offd_Pext = jx_CTAlloc(JX_Int, num_cols_offd_Pext);
   }

   for (i = 0; i < num_cols_offd_Pext; i ++)
   {
      col_map_offd_Pext[i] = temp[i];
   }

   if (P_ext_offd_size || num_cols_offd_P)
   {
      jx_TFree(temp);
   }

   for (i = 0 ; i < P_ext_offd_size; i ++)
   {
      P_ext_offd_j[i] = jx_BinarySearch(col_map_offd_Pext,P_ext_offd_j[i],num_cols_offd_Pext);
   }
   
   if (num_cols_offd_P)
   {
      map_P_to_Pext = jx_CTAlloc(JX_Int, num_cols_offd_P);

      cnt = 0;
      for (i = 0; i < num_cols_offd_Pext; i ++)
      {
         if (col_map_offd_Pext[i] == col_map_offd_P[cnt])
         {
            map_P_to_Pext[cnt++] = i;
            if (cnt == num_cols_offd_P) break;
         }
      }
   }
#endif
  /*----------------------------------------------------------------------    
   *  First Pass: Determine size of RAP_int and set up RAP_int_i if there 
   *  are more than one processor and nonzero elements in R_offd
   *-----------------------------------------------------------------------*/

#if 0 /* Yue Xiaoqiang 2012/10/13 */
   P_mark_array = jx_CTAlloc(JX_Int *, num_threads);
   A_mark_array = jx_CTAlloc(JX_Int *, num_threads);
#else /* Yue Xiaoqiang 2012/10/13 */
   JX_Int minus_one_length_A = icor_yoo[5*num_threads];
   JX_Int minus_one_length_P = icor_yoo[5*num_threads+1];
   JX_Int minus_one_length = minus_one_length_A + minus_one_length_P;
   JX_Int *Ps_marker, *As_marker;
#if 0
   Ps_marker = jx_CTAlloc(JX_Int, minus_one_length);
#else
   if (Pmarkers_global_rap == NULL)                             /* Feng Chunsheng & Yue Xiaoqiang 2012/10/17 */
   {
      Pmarkers_global_rap = jx_CTAlloc(JX_Int, minus_one_length);
      jx_total_calloc_global_rap = minus_one_length;
   }
   else if (minus_one_length > jx_total_calloc_global_rap)      /* Feng Chunsheng & Yue Xiaoqiang 2012/10/17 */
   {
      Pmarkers_global_rap = jx_TReAlloc(Pmarkers_global_rap, JX_Int, minus_one_length);
      jx_printf("<-----Pmarkers_global_rap realloc %d-----%d----->\n",
                       jx_total_calloc_global_rap, minus_one_length);
      jx_total_calloc_global_rap = minus_one_length;
   }
   Ps_marker = Pmarkers_global_rap;
#endif
   As_marker = Ps_marker + minus_one_length_P;
   jx_long_size_length_rap += (num_cols_diag_P + num_cols_offd_RAP) * num_threads;
   jx_long_size_length_rap += num_nz_cols_A * num_threads;
   jx_real_long_size_length_rap += minus_one_length;
   jx_IntegerArraySetConstantValues(minus_one_length, Ps_marker, -1);
   P_mark_array = jx_CTAlloc(JX_Int *, num_threads);
   A_mark_array = jx_CTAlloc(JX_Int *, num_threads);
#endif

#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_cols_offd_RT)
   {
      jj_count = jx_CTAlloc(JX_Int, num_threads);

#define JX_SMP_PRIVATE i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_counter,jj_row_begining,A_marker,P_marker
#include "../../../include/jx_smp_forloop.h"

      for (ii = 0; ii < num_threads; ii ++)
      {
         size = num_cols_offd_RT/num_threads;
         rest = num_cols_offd_RT - size*num_threads;
         if (ii < rest)
         {
            ns = ii*size + ii;
            ne = (ii+1)*size + ii + 1;
         }
         else
         {
            ns = ii*size + rest;
            ne = (ii+1)*size + rest;
         }
   
        /*-----------------------------------------------------------------------
         *  Allocate marker arrays.
         *-----------------------------------------------------------------------*/

         P_marker = NULL; /* To avoid the warning. Yue Xiaoqiang 2012/09/05 */
         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_mark_array[ii] = jx_CTAlloc(JX_Int, num_cols_diag_P + num_cols_offd_Pext);
            P_marker = P_mark_array[ii];
         }
         A_mark_array[ii] = jx_CTAlloc(JX_Int, num_nz_cols_A);
         A_marker = A_mark_array[ii];

        /*-----------------------------------------------------------------------
         *  Initialize some stuff.
         *-----------------------------------------------------------------------*/

         jj_counter = start_indexing;
         for (ic = 0; ic < num_cols_diag_P+num_cols_offd_Pext; ic ++)
         {      
            P_marker[ic] = -1;
         }
         for (i = 0; i < num_nz_cols_A; i ++)
         {      
            A_marker[i] = -1;
         }   


        /*-----------------------------------------------------------------------
         *  Loop over exterior c-points
         *-----------------------------------------------------------------------*/
    
         for (ic = ns; ic < ne; ic ++)
         {
      
            jj_row_begining = jj_counter;

           /*--------------------------------------------------------------------
            *  Loop over entries in row ic of R_offd.
            *--------------------------------------------------------------------*/
   
            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic+1]; jj1 ++)
            {
               i1  = R_offd_j[jj1];

              /*-----------------------------------------------------------------
               *  Loop over entries in row i1 of A_offd.
               *-----------------------------------------------------------------*/
         
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2 ++)
               {
                  i2 = A_offd_j[jj2];

                 /*--------------------------------------------------------------
                  *  Check A_marker to see if point i2 has been previously
                  *  visited. New entries in RAP only occur from unmarked points.
                  *--------------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                    /*-----------------------------------------------------------
                     *  Mark i2 as visited.
                     *-----------------------------------------------------------*/

                     A_marker[i2] = ic;
               
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_ext.
                     *-----------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_diag_j[jj3];
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, mark it and increment
                        *  counter.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter ++;
                        }
                     } // end for jj3
                     
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not 
                        *  already been accounted for. If it has not, mark 
                        *  it and increment counter.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter ++;
                        }
                     } // end for jj3
                     
                  } // end if (A_marker[i2] != ic)
                  
               } // end if jj2
               
               
              /*-----------------------------------------------------------------
               *  Loop over entries in row i1 of A_diag.
               *-----------------------------------------------------------------*/
         
               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2 ++)
               {
                  i2 = A_diag_j[jj2];

                 /*--------------------------------------------------------------
                  *  Check A_marker to see if point i2 has been previously
                  *  visited. New entries in RAP only occur from unmarked points.
                  *--------------------------------------------------------------*/

                  if (A_marker[i2+num_cols_offd_A] != ic)
                  {

                    /*-----------------------------------------------------------
                     *  Mark i2 as visited.
                     *-----------------------------------------------------------*/

                     A_marker[i2+num_cols_offd_A] = ic;
               
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_diag.
                     *-----------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_diag_j[jj3];
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has 
                        *  not already been accounted for. If it has not, 
                        *  mark it and increment counter.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter ++;
                        }
                     } // end for jj3
                     
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_offd.
                     *-----------------------------------------------------------*/

                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, mark it and increment
                        *  counter.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter ++;
                        }
                     } // end for jj3
                     
                  } // end if (A_marker[i2+num_cols_offd_A] != ic)
                  
               } // end for jj2
               
            } // end for jj1
            
         } // end for ic

         jj_count[ii] = jj_counter;

      } // end for ii
      
  
     /*-----------------------------------------------------------------------
      *  Allocate RAP_int_data and RAP_int_j arrays.
      *-----------------------------------------------------------------------*/
      
      for (i = 0; i < num_threads - 1; i ++)
      {
         jj_count[i+1] += jj_count[i];
      }
    
      RAP_size     = jj_count[num_threads-1];
      RAP_int_i    = jx_CTAlloc(JX_Int, num_cols_offd_RT + 1);
      RAP_int_data = jx_CTAlloc(JX_Real, RAP_size);
      RAP_int_j    = jx_CTAlloc(JX_Int, RAP_size);

      RAP_int_i[num_cols_offd_RT] = RAP_size;



     /*-----------------------------------------------------------------------
      *  Second Pass: Fill in RAP_int_data and RAP_int_j.
      *-----------------------------------------------------------------------*/

#define JX_SMP_PRIVATE i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_counter,jj_row_begining,A_marker,P_marker,r_entry,r_a_product,r_a_p_product
#include "../../../include/jx_smp_forloop.h"

      for (ii = 0; ii < num_threads; ii ++)
      {
         size = num_cols_offd_RT/num_threads;
         rest = num_cols_offd_RT - size*num_threads;
         if (ii < rest)
         {
            ns = ii*size + ii;
            ne = (ii+1)*size + ii + 1;
         }
         else
         {
            ns = ii*size + rest;
            ne = (ii+1)*size + rest;
         }

        /*-----------------------------------------------------------------------
         *  Initialize some stuff.
         *-----------------------------------------------------------------------*/
         
         P_marker = NULL; /* To avoid the warning. Yue Xiaoqiang 2012/09/05 */
         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_marker = P_mark_array[ii];
         }
         A_marker = A_mark_array[ii];

         jj_counter = start_indexing;
         if (ii > 0) 
         {
            jj_counter = jj_count[ii-1];
         }

         for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic ++)
         {      
            P_marker[ic] = -1;
         }
         
         for (i = 0; i < num_nz_cols_A; i ++)
         {      
            A_marker[i] = -1;
         }   
   
        /*-----------------------------------------------------------------------
         *  Loop over exterior c-points.
         *-----------------------------------------------------------------------*/
    
         for (ic = ns; ic < ne; ic ++)
         {
      
            jj_row_begining = jj_counter;
            RAP_int_i[ic] = jj_counter;

           /*--------------------------------------------------------------------
            *  Loop over entries in row ic of R_offd.
            *--------------------------------------------------------------------*/
   
            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic+1]; jj1 ++)
            {
               i1  = R_offd_j[jj1];
               r_entry = R_offd_data[jj1];

              /*-----------------------------------------------------------------
               *  Loop over entries in row i1 of A_offd.
               *-----------------------------------------------------------------*/
         
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2 ++)
               {
                  i2 = A_offd_j[jj2];
                  r_a_product = r_entry * A_offd_data[jj2];
            
                 /*--------------------------------------------------------------
                  *  Check A_marker to see if point i2 has been previously
                  *  visited. New entries in RAP only occur from unmarked points.
                  *--------------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                    /*-----------------------------------------------------------
                     *  Mark i2 as visited.
                     *-----------------------------------------------------------*/

                     A_marker[i2] = ic;
               
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_ext.
                     *-----------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                  
                       /*-----------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *---------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                           jj_counter ++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                        
                     } // end for jj3
                     

                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                  
                       /*----------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] = col_map_offd_Pext[i3-num_cols_diag_P];
                           jj_counter ++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                        
                     } // end for jj3
                     
                  } // end if (A_marker[i2] != ic)
                  

                 /*--------------------------------------------------------------
                  *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                  *  no new entries in RAP and can just add new contributions.
                  *--------------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                     
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
                  
               } // end  for jj2
               

              /*-----------------------------------------------------------------
               *  Loop over entries in row i1 of A_diag.
               *-----------------------------------------------------------------*/
         
               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2 ++)
               {
                  i2 = A_diag_j[jj2];
                  r_a_product = r_entry * A_diag_data[jj2];
            
                 /*--------------------------------------------------------------
                  *  Check A_marker to see if point i2 has been previously
                  *  visited. New entries in RAP only occur from unmarked points.
                  *--------------------------------------------------------------*/

                  if (A_marker[i2+num_cols_offd_A] != ic)
                  {

                    /*-----------------------------------------------------------
                     *  Mark i2 as visited.
                     *-----------------------------------------------------------*/

                     A_marker[i2+num_cols_offd_A] = ic;
               
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_diag.
                     *-----------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_diag_j[jj3];
                        r_a_p_product = r_a_product * P_diag_data[jj3];
                  
                       /*----------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                           jj_counter ++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                     } // end for jj3
                     
                     
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] =  col_map_offd_Pext[i3-num_cols_diag_P];
                           jj_counter ++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                     } // end for jj3
                     
                  } // end if (A_marker[i2+num_cols_offd_A] != ic)

                 /*--------------------------------------------------------------
                  *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                  *  no new entries in RAP and can just add new contributions.
                  *--------------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_diag_j[jj3];
                        r_a_p_product = r_a_product * P_diag_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
                  
               } // end for jj2
            
            } // end for jj1
         
         } // end for ic
         
         
         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            jx_TFree(P_mark_array[ii]);
         }
         jx_TFree(A_mark_array[ii]);
         
      } // end  for ii

      RAP_int = jx_CSRMatrixCreate(num_cols_offd_RT,num_rows_offd_RT,RAP_size);
      jx_CSRMatrixI(RAP_int) = RAP_int_i;
      jx_CSRMatrixJ(RAP_int) = RAP_int_j;
      jx_CSRMatrixData(RAP_int) = RAP_int_data;
      jx_TFree(jj_count);
   
   } // end if (num_cols_offd_RT)
#endif

   //RAP_ext_size = 0;
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_sends_RT || num_recvs_RT)
   {
      RAP_ext = jx_ExchangeRAPData(RAP_int, comm_pkg_RT);
      RAP_ext_i = jx_CSRMatrixI(RAP_ext);
      RAP_ext_j = jx_CSRMatrixJ(RAP_ext);
      RAP_ext_data = jx_CSRMatrixData(RAP_ext);
      RAP_ext_size = RAP_ext_i[jx_CSRMatrixNumRows(RAP_ext)];
   }
   
   if (num_cols_offd_RT)
   {
      jx_CSRMatrixDestroy(RAP_int);
      RAP_int = NULL;
   }
#endif
   RAP_diag_i = jx_CTAlloc(JX_Int, num_cols_diag_RT + 1);
   RAP_offd_i = jx_CTAlloc(JX_Int, num_cols_diag_RT + 1);

   //first_col_diag_RAP = first_col_diag_P;
   //last_col_diag_RAP = first_col_diag_P + num_cols_diag_P - 1;


  /*-----------------------------------------------------------------------
   *  check for new nonzero columns in RAP_offd generated through RAP_ext
   *-----------------------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (RAP_ext_size || num_cols_offd_Pext)
   {
      temp = jx_CTAlloc(JX_Int, RAP_ext_size + num_cols_offd_Pext);
      cnt = 0;
      for (i = 0; i < RAP_ext_size; i ++)
      {
         if (RAP_ext_j[i] < first_col_diag_RAP || RAP_ext_j[i] > last_col_diag_RAP)
         {
            temp[cnt++] = RAP_ext_j[i];
         }
      }
      for (i = 0; i < num_cols_offd_Pext; i ++)
      {
         temp[cnt++] = col_map_offd_Pext[i];
      }

      if (cnt)
      {
         jx_qsort0(temp, 0, cnt - 1);
         value = temp[0];
         num_cols_offd_RAP = 1;
         for (i = 1; i < cnt; i ++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_RAP++] = value;
            }
         }
      }

      /* now evaluate col_map_offd_RAP */
      if (num_cols_offd_RAP)
      {
         col_map_offd_RAP = jx_CTAlloc(JX_Int, num_cols_offd_RAP);
      }

      for (i = 0 ; i < num_cols_offd_RAP; i ++)
      {
         col_map_offd_RAP[i] = temp[i];
      }
  
      jx_TFree(temp);
   }


   if (num_cols_offd_P)
   {
      map_P_to_RAP = jx_CTAlloc(JX_Int, num_cols_offd_P);

      cnt = 0;
      for (i = 0; i < num_cols_offd_RAP; i ++)
      {
         if (col_map_offd_RAP[i] == col_map_offd_P[cnt])
         {
            map_P_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_P) 
            {
               break;
            }
         }
      }
   }

   if (num_cols_offd_Pext)
   {
      map_Pext_to_RAP = jx_CTAlloc(JX_Int, num_cols_offd_Pext);

      cnt = 0;
      for (i = 0; i < num_cols_offd_RAP; i ++)
      {
         if (col_map_offd_RAP[i] == col_map_offd_Pext[cnt])
         {
            map_Pext_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_Pext) 
            {
               break;
            }
         }
      }
   }
#endif

  /*-----------------------------------------------------------------------
   *  Convert RAP_ext column indices
   *-----------------------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   for (i = 0; i < RAP_ext_size; i ++)
   {
      if (RAP_ext_j[i] < first_col_diag_RAP || RAP_ext_j[i] > last_col_diag_RAP)
      {
         RAP_ext_j[i] = num_cols_diag_P + jx_BinarySearch(col_map_offd_RAP, RAP_ext_j[i], num_cols_offd_RAP);
      }
      else
      {
         RAP_ext_j[i] -= first_col_diag_RAP;
      }
   }
#endif
   /* need to allocate new P_marker etc. and make further changes */
   
  /*-----------------------------------------------------------------------
   *  Initialize some stuff.
   *-----------------------------------------------------------------------*/
   
   jj_cnt_diag = jx_CTAlloc(JX_Int, num_threads);
   jj_cnt_offd = jx_CTAlloc(JX_Int, num_threads);

#define JX_SMP_PRIVATE i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker
#include "../../../include/jx_smp_forloop.h"
   for (ii = 0; ii < num_threads; ii ++)
   {
#if 0 /* Yue Xiaoqiang 2012/10/13 */
      size = num_cols_diag_RT/num_threads;
      rest = num_cols_diag_RT - size*num_threads;
      if (ii < rest)
      {
         ns = ii*size + ii;
         ne = (ii+1)*size + ii + 1;
      }
      else
      {
         ns = ii*size + rest;
         ne = (ii+1)*size + rest;
      }

      P_mark_array[ii] = jx_CTAlloc(JX_Int, num_cols_diag_P + num_cols_offd_RAP);
      A_mark_array[ii] = jx_CTAlloc(JX_Int, num_nz_cols_A);
      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;

      for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic ++)
      {      
         P_marker[ic] = -1;
      }
      
      for (i = 0; i < num_nz_cols_A; i ++)
      {      
         A_marker[i] = -1;
      }   
#else
      JX_Int min_A, min_P, A_pos, P_pos, R_pos;
      R_pos = ii * 5;
      ns = icor_yoo[R_pos];
      if (ii == num_threads-1)
      {
         ne = n_coarse;
      }
      else
      {
         ne = icor_yoo[R_pos+5];
      }
      min_A = icor_yoo[R_pos+2];
      min_P = icor_yoo[R_pos+4];
      
      /* Add the constant displacement */
      if (ii == 0)
      { 
         P_mark_array[ii] =  P_marker = Ps_marker - min_P;
         A_mark_array[ii] =  A_marker = As_marker - min_A;
      }
      else
      {
         A_pos = 0;
         P_pos = 0;
         for (ic = ii-1; ic >= 0; ic --)
         {
            R_pos = ic * 5;
            A_pos += icor_yoo[R_pos+1];
            P_pos += icor_yoo[R_pos+3];
         }
         P_mark_array[ii] = P_marker = Ps_marker + P_pos - min_P;
         A_mark_array[ii] = A_marker = As_marker + A_pos - min_A;
      }

      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;
#endif

     /*-----------------------------------------------------------------------
      *  Loop over interior c-points.
      *-----------------------------------------------------------------------*/
   
      for (ic = ns; ic < ne; ic ++)
      {
      
        /*--------------------------------------------------------------------
         *  Set marker for diagonal entry, RAP_{ic,ic}. and for all points
         *  being added to row ic of RAP_diag and RAP_offd through RAP_ext
         *--------------------------------------------------------------------*/

         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;

         if (square)
         {
            P_marker[ic] = jj_count_diag ++;
         }

#if 0 /* Yue Xiaoqiang 2012/10/13 */
         for (i = 0; i < num_sends_RT; i ++)
         {
         
            for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j ++)
            {
               
               if (send_map_elmts_RT[j] == ic)
               {
                  for (k=RAP_ext_i[j]; k < RAP_ext_i[j+1]; k ++)
                  {
                     jcol = RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           jj_count_diag ++;
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           jj_count_offd ++;
                        }
                     }
                  }
                  break;
                  
               } // end if (send_map_elmts_RT[j] == ic)
            
            } // end for j
         
         } // end for i
 
#endif
        /*--------------------------------------------------------------------
         *  Loop over entries in row ic of R_diag.
         *--------------------------------------------------------------------*/
         R_pos = R_diag_i[ic+1]; /* Yue Xiaoqiang 2012/10/13 */
         for (jj1 = R_diag_i[ic]; jj1 < R_pos; jj1 ++)
         {
            i1  = R_diag_j[jj1];
 
           /*-----------------------------------------------------------------
            *  Loop over entries in row i1 of A_offd.
            *-----------------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
            if (num_cols_offd_A)
            {
            
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2 ++)
               {
                  i2 = A_offd_j[jj2];
 
                  /*--------------------------------------------------------------
                  *  Check A_marker to see if point i2 has been previously
                  *  visited. New entries in RAP only occur from unmarked points.
                  *--------------------------------------------------------------*/
 
                  if (A_marker[i2] != ic)
                  {
 
                    /*-----------------------------------------------------------
                     *  Mark i2 as visited.
                     *-----------------------------------------------------------*/
 
                     A_marker[i2] = ic;
               
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_ext.
                     *-----------------------------------------------------------*/
 
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_diag_j[jj3];
                  
                       /*-----------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, mark it and increment
                        *  counter.
                        *---------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           jj_count_diag ++;
                        }
                     } // end for jj3
                     
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]]+num_cols_diag_P;
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, mark it and increment
                        *  counter.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           jj_count_offd ++;
                        }
                     } // end for jj3
                     
                  } // end if (A_marker[i2] != ic)
               
               } // end for jj2
               
            } // end if (num_cols_offd_A)
#endif
         
           /*-----------------------------------------------------------------
            *  Loop over entries in row i1 of A_diag.
            *-----------------------------------------------------------------*/
            A_pos = A_diag_i[i1+1]; /* Yue Xiaoqiang 2012/10/13 */
            for (jj2 = A_diag_i[i1]; jj2 < A_pos; jj2 ++)
            {
               i2 = A_diag_j[jj2];
 
              /*--------------------------------------------------------------
               *  Check A_marker to see if point i2 has been previously
               *  visited. New entries in RAP only occur from unmarked points.
               *--------------------------------------------------------------*/
 
               if (A_marker[i2+num_cols_offd_A] != ic)
               {
 
                 /*-----------------------------------------------------------
                  *  Mark i2 as visited.
                  *-----------------------------------------------------------*/
 
                  A_marker[i2+num_cols_offd_A] = ic;
               
                 /*-----------------------------------------------------------
                  *  Loop over entries in row i2 of P_diag.
                  *-----------------------------------------------------------*/
                  P_pos = P_diag_i[i2+1]; /* Yue Xiaoqiang 2012/10/13 */
                  for (jj3 = P_diag_i[i2]; jj3 < P_pos; jj3 ++)
                  {
                     i3 = P_diag_j[jj3];
                  
                    /*--------------------------------------------------------
                     *  Check P_marker to see that RAP_{ic,i3} has not already
                     *  been accounted for. If it has not, mark it and increment
                     *  counter.
                     *--------------------------------------------------------*/
 
                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        jj_count_diag ++;
                     }
                  } // end for jj3
                  
                  
                 /*-----------------------------------------------------------
                  *  Loop over entries in row i2 of P_offd.
                  *-----------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
                  if (num_cols_offd_P)
                  { 
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, mark it and increment
                        *  counter.
                        *--------------------------------------------------------*/
 
                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           jj_count_offd ++;
                        }
                     } // end for jj3
                  } // end if (num_cols_offd_P)
#endif
                  
               } // end if (A_marker[i2+num_cols_offd_A] != ic)
               
            } // for jj2
            
         } // end for jj1
                 
      } // end for ic
      
      jj_cnt_diag[ii] = jj_count_diag;
      jj_cnt_offd[ii] = jj_count_offd;
      
   } // end for ii

   jx_IntegerArraySetConstantValues(minus_one_length, Ps_marker, -1);

   for (i = 0; i < num_threads - 1; i ++)
   {
      jj_cnt_diag[i+1] += jj_cnt_diag[i];
      jj_cnt_offd[i+1] += jj_cnt_offd[i];
   }

   jj_count_diag = jj_cnt_diag[num_threads-1];
   jj_count_offd = jj_cnt_offd[num_threads-1];

   RAP_diag_i[num_cols_diag_RT] = jj_count_diag;
   RAP_offd_i[num_cols_diag_RT] = jj_count_offd;
 
 
  /*-----------------------------------------------------------------------
   *  Allocate RAP_diag_data and RAP_diag_j arrays.
   *  Allocate RAP_offd_data and RAP_offd_j arrays.
   *-----------------------------------------------------------------------*/
 
   RAP_diag_size = jj_count_diag;
   if (RAP_diag_size)
   { 
      RAP_diag_data = jx_CTAlloc(JX_Real, RAP_diag_size);
      RAP_diag_j    = jx_CTAlloc(JX_Int, RAP_diag_size);
   } 
 
   RAP_offd_size = jj_count_offd;
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (RAP_offd_size)
   { 
      RAP_offd_data = jx_CTAlloc(JX_Real, RAP_offd_size);
      RAP_offd_j    = jx_CTAlloc(JX_Int, RAP_offd_size);
   }
#endif
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (RAP_offd_size == 0 && num_cols_offd_RAP != 0)
   {
      num_cols_offd_RAP = 0;
      jx_TFree(col_map_offd_RAP);
   }
#endif
   
  /*-----------------------------------------------------------------------
   *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
   *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
   *-----------------------------------------------------------------------*/

#define JX_SMP_PRIVATE i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker,r_entry,r_a_product,r_a_p_product
#include "../../../include/jx_smp_forloop.h"
   for (ii = 0; ii < num_threads; ii ++)
   {
#if 0 /* Yue Xiaoqiang 2012/10/13 */
      size = num_cols_diag_RT / num_threads;
      rest = num_cols_diag_RT - size*num_threads;
      if (ii < rest)
      {
         ns = ii*size + ii;
         ne = (ii+1)*size + ii + 1;
      }
      else
      {
         ns = ii*size + rest;
         ne = (ii+1)*size + rest;
      }

     /*-----------------------------------------------------------------------
      *  Initialize some stuff.
      *-----------------------------------------------------------------------*/

      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
      for (ic = 0; ic < num_cols_diag_P+num_cols_offd_RAP; ic ++)
      {      
         P_marker[ic] = -1;
      }
      for (i = 0; i < num_nz_cols_A ; i ++)
      {      
         A_marker[i] = -1;
      }   
#else
      ns = icor_yoo[ii*5];
      if (ii == num_threads-1)
      {
         ne = n_coarse;
      }
      else
      {
         ne = icor_yoo[(ii+1)*5];
      }
     /*-----------------------------------------------------------------------
      *  Initialize some stuff.
      *-----------------------------------------------------------------------*/
      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
#endif

      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;
      if (ii > 0)
      {
         jj_count_diag = jj_cnt_diag[ii-1];
         jj_count_offd = jj_cnt_offd[ii-1];
      }

     /*-----------------------------------------------------------------------
      *  Loop over interior c-points.
      *-----------------------------------------------------------------------*/
      JX_Int R_pos, A_pos, P_pos;
      for (ic = ns; ic < ne; ic ++)
      {
      
        /*--------------------------------------------------------------------
         *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext 
         *--------------------------------------------------------------------*/

         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         RAP_diag_i[ic] = jj_row_begin_diag;
         RAP_offd_i[ic] = jj_row_begin_offd;

         if (square)
         {
            P_marker[ic] = jj_count_diag;
            RAP_diag_data[jj_count_diag] = zero;
            RAP_diag_j[jj_count_diag] = ic;
            jj_count_diag ++;
         }
#if 0 /* Yue Xiaoqiang 2012/10/13 */
         for (i = 0; i < num_sends_RT; i ++)
         {
            for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j ++)
            {
               if (send_map_elmts_RT[j] == ic)
               {
                  for (k = RAP_ext_i[j]; k < RAP_ext_i[j+1]; k ++)
                  {
                     jcol = RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           RAP_diag_data[jj_count_diag] = RAP_ext_data[k];
                           RAP_diag_j[jj_count_diag] = jcol;
                           jj_count_diag ++;
                        }
                        else
                        {
                           RAP_diag_data[P_marker[jcol]] += RAP_ext_data[k];
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           RAP_offd_data[jj_count_offd] = RAP_ext_data[k];
                           RAP_offd_j[jj_count_offd] = jcol-num_cols_diag_P;
                           jj_count_offd ++;
                        }
                        else
                        {
                           RAP_offd_data[P_marker[jcol]] += RAP_ext_data[k];
                        }
                     }
                  } // end for k
                  break;
               } // end if (send_map_elmts_RT[j] == ic)
            } // end for j
         } // end for i
#endif

        /*--------------------------------------------------------------------
         *  Loop over entries in row ic of R_diag.
         *--------------------------------------------------------------------*/
         R_pos = R_diag_i[ic+1]; /* Yue Xiaoqiang 2012/10/13 */
         for (jj1 = R_diag_i[ic]; jj1 < R_pos; jj1 ++)
         {
            i1  = R_diag_j[jj1];
            r_entry = R_diag_data[jj1];

           /*-----------------------------------------------------------------
            *  Loop over entries in row i1 of A_offd.
            *-----------------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
            if (num_cols_offd_A)
            {
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2 ++)
               {
                  i2 = A_offd_j[jj2];
                  r_a_product = r_entry * A_offd_data[jj2];
            
                 /*--------------------------------------------------------------
                  *  Check A_marker to see if point i2 has been previously
                  *  visited. New entries in RAP only occur from unmarked points.
                  *--------------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                    /*-----------------------------------------------------------
                     *  Mark i2 as visited.
                     *-----------------------------------------------------------*/

                     A_marker[i2] = ic;
               
                    /*-----------------------------------------------------------
                     *  Loop over entries in row i2 of P_ext.
                     *-----------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                  
                       /*--------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *--------------------------------------------------------*/
                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           RAP_diag_data[jj_count_diag] = r_a_p_product;
                           RAP_diag_j[jj_count_diag] = i3;
                           jj_count_diag ++;
                        }
                        else
                        {
                           RAP_diag_data[P_marker[i3]] += r_a_p_product;
                        }
                     }
                     
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                  
                       /*----------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *---------------------------------------------------------*/
                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           RAP_offd_data[jj_count_offd] = r_a_p_product;
                           RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                           jj_count_offd ++;
                        }
                        else
                        {
                           RAP_offd_data[P_marker[i3]] += r_a_p_product;
                        }
                     }
                  }

                 /*--------------------------------------------------------------
                  *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                  *  no new entries in RAP and can just add new contributions.
                  *--------------------------------------------------------------*/
                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3 ++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                        RAP_diag_data[P_marker[i3]] += r_a_p_product;
                     }
                     
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                        RAP_offd_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
                  
               } // end for jj2
               
            } // end if (num_cols_offd_A)
#endif
           /*-----------------------------------------------------------------
            *  Loop over entries in row i1 of A_diag.
            *-----------------------------------------------------------------*/
            A_pos = A_diag_i[i1+1]; /* Yue Xiaoqiang 2012/10/13 */
            for (jj2 = A_diag_i[i1]; jj2 < A_pos; jj2 ++)
            {
               i2 = A_diag_j[jj2];
               r_a_product = r_entry * A_diag_data[jj2];
            
              /*--------------------------------------------------------------
               *  Check A_marker to see if point i2 has been previously
               *  visited. New entries in RAP only occur from unmarked points.
               *--------------------------------------------------------------*/

               if (A_marker[i2+num_cols_offd_A] != ic)
               {

                 /*-----------------------------------------------------------
                  *  Mark i2 as visited.
                  *-----------------------------------------------------------*/

                  A_marker[i2+num_cols_offd_A] = ic;
               
                 /*-----------------------------------------------------------
                  *  Loop over entries in row i2 of P_diag.
                  *-----------------------------------------------------------*/
                  P_pos =  P_diag_i[i2+1];  /* Yue Xiaoqiang 2012/10/13 */
                  for (jj3 = P_diag_i[i2]; jj3 < P_pos; jj3 ++)
                  {
                     i3 = P_diag_j[jj3];
                     r_a_p_product = r_a_product * P_diag_data[jj3];
                  
                    /*----------------------------------------------------------
                     *  Check P_marker to see that RAP_{ic,i3} has not already
                     *  been accounted for. If it has not, create a new entry.
                     *  If it has, add new contribution.
                     *--------------------------------------------------------*/

                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        RAP_diag_data[jj_count_diag] = r_a_p_product;
                        RAP_diag_j[jj_count_diag] = P_diag_j[jj3];
                        jj_count_diag ++;
                     }
                     else
                     {
                        RAP_diag_data[P_marker[i3]] += r_a_p_product;
                     }
                  } // end for jj3
#if 0 /* Yue Xiaoqiang 2012/10/13 */
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];
                  
                       /*----------------------------------------------------------
                        *  Check P_marker to see that RAP_{ic,i3} has not already
                        *  been accounted for. If it has not, create a new entry.
                        *  If it has, add new contribution.
                        *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           RAP_offd_data[jj_count_offd] = r_a_p_product;
                           RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                           jj_count_offd ++;
                        }
                        else
                        {
                           RAP_offd_data[P_marker[i3]] += r_a_p_product;
                        }
                     } // end for jj3
                     
                  } // end if (num_cols_offd_P)
#endif
               }

              /*--------------------------------------------------------------
               *  If i2 is previously visited ( A_marker[12]=ic ) it yields
               *  no new entries in RAP and can just add new contributions.
               *--------------------------------------------------------------*/

               else
               {
                  P_pos =  P_diag_i[i2+1]; /* Yue Xiaoqiang 2012/10/13 */
                  for (jj3 = P_diag_i[i2]; jj3 < P_pos; jj3 ++)
                  {
                     i3 = P_diag_j[jj3];
                     r_a_p_product = r_a_product * P_diag_data[jj3];
                     RAP_diag_data[P_marker[i3]] += r_a_p_product;
                  }
#if 0 /* Yue Xiaoqiang 2012/10/13 */
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];
                        RAP_offd_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
#endif
               }  // end if (A_marker[i2+num_cols_offd_A] != ic)
               
            } // end for jj2
            
         } // end for jj1
         
      } // end for ic
#if 0 /* Yue Xiaoqiang 2012/10/13 */
      jx_TFree(P_mark_array[ii]);
      jx_TFree(A_mark_array[ii]);
#endif
   } // end for ii
#if 0 /* Yue Xiaoqiang 2012/10/17 */
   jx_TFree(Ps_marker);
#endif


  /*---------------------------------------------------------------
   *    Check if really all off-diagonal entries occurring in 
   * col_map_offd_RAP are represented and eliminate if necessary. 
   *--------------------------------------------------------------*/
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   P_marker = jx_CTAlloc(JX_Int,num_cols_offd_RAP);
   for (i = 0; i < num_cols_offd_RAP; i ++)
   {
      P_marker[i] = -1;
   }

   jj_count_offd = 0;
   for (i = 0; i < RAP_offd_size; i ++)
   {
      i3 = RAP_offd_j[i];
      if (P_marker[i3])
      {
	 P_marker[i3] = 0;
	 jj_count_offd ++;
      }
   }

   if (jj_count_offd < num_cols_offd_RAP)
   {
      new_col_map_offd_RAP = jx_CTAlloc(JX_Int, jj_count_offd);
      jj_counter = 0;
      for (i = 0; i < num_cols_offd_RAP; i ++)
      {
         if (!P_marker[i]) 
         {
	    P_marker[i] = jj_counter;
	    new_col_map_offd_RAP[jj_counter ++] = col_map_offd_RAP[i];
         }
      }
 
      for (i = 0; i < RAP_offd_size; i ++)
      {
	 i3 = RAP_offd_j[i];
	 RAP_offd_j[i] = P_marker[i3];
      }
      
      num_cols_offd_RAP = jj_count_offd;
      jx_TFree(col_map_offd_RAP);
      col_map_offd_RAP = new_col_map_offd_RAP;
   }
   jx_TFree(P_marker);
#endif
   RAP = jx_ParCSRMatrixCreate(comm, n_coarse_RT, n_coarse,
                               RT_partitioning, coarse_partitioning,
                               num_cols_offd_RAP, RAP_diag_size,
                               RAP_offd_size);

   /* Have RAP own coarse_partitioning instead of par_P */
   jx_ParCSRMatrixSetColStartsOwner(par_P,0);
   jx_ParCSRMatrixSetColStartsOwner(RT,0);

   RAP_diag = jx_ParCSRMatrixDiag(RAP);
   jx_CSRMatrixI(RAP_diag) = RAP_diag_i; 
   if (RAP_diag_size)
   {
      jx_CSRMatrixData(RAP_diag) = RAP_diag_data; 
      jx_CSRMatrixJ(RAP_diag) = RAP_diag_j; 
   }

   RAP_offd = jx_ParCSRMatrixOffd(RAP);
   jx_CSRMatrixI(RAP_offd) = RAP_offd_i; 
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_cols_offd_RAP)
   {
      jx_CSRMatrixData(RAP_offd) = RAP_offd_data; 
      jx_CSRMatrixJ(RAP_offd) = RAP_offd_j; 
      jx_ParCSRMatrixColMapOffd(RAP) = col_map_offd_RAP;
   }
#endif
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_procs > 1)
   {
      jx_MatvecCommPkgCreate(RAP); 
   }
#endif
   *RAP_ptr = RAP;

  /*-----------------------------------------------------------------------
   *  Free R, P_ext and marker arrays.
   *-----------------------------------------------------------------------*/

   jx_CSRMatrixDestroy(R_diag);
   R_diag = NULL;
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_cols_offd_RT) 
   {
      jx_CSRMatrixDestroy(R_offd);
      R_offd = NULL;
   }
#endif
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_sends_RT || num_recvs_RT) 
   {
      jx_CSRMatrixDestroy(RAP_ext);
      RAP_ext = NULL;
   }
#endif
   jx_TFree(P_mark_array);   
   jx_TFree(A_mark_array);
   jx_TFree(P_ext_diag_i);
   jx_TFree(P_ext_offd_i);
   jx_TFree(jj_cnt_diag);   
   jx_TFree(jj_cnt_offd);
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_cols_offd_P)
   {
      jx_TFree(map_P_to_Pext);
      jx_TFree(map_P_to_RAP);
   }
#endif
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (num_cols_offd_Pext)
   {
      jx_TFree(col_map_offd_Pext);
      jx_TFree(map_Pext_to_RAP);
   }
#endif
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (P_ext_diag_size)
   {
      jx_TFree(P_ext_diag_data);
      jx_TFree(P_ext_diag_j);
   }
#endif
#if 0 /* Yue Xiaoqiang 2012/10/13 */
   if (P_ext_offd_size)
   {
      jx_TFree(P_ext_offd_data);
      jx_TFree(P_ext_offd_j);
   }
#endif
   return(0);
}
