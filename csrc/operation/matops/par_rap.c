//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_rap.c -- Computing the coarse operater in PAMG.
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn jx_CSRMatrix *jx_ExchangeRAPData
 * \date 2011/09/03
 */
jx_CSRMatrix *
jx_ExchangeRAPData( jx_CSRMatrix *RAP_int, jx_ParCSRCommPkg *comm_pkg_RT )
{
   JX_Int     *RAP_int_i    = 0;
   JX_Int     *RAP_int_j    = NULL;
   JX_Real  *RAP_int_data = NULL;
   JX_Int      num_cols     = 0;

   MPI_Comm comm        = jx_ParCSRCommPkgComm(comm_pkg_RT);
   JX_Int num_recvs        = jx_ParCSRCommPkgNumRecvs(comm_pkg_RT);
   JX_Int *recv_procs      = jx_ParCSRCommPkgRecvProcs(comm_pkg_RT);
   JX_Int *recv_vec_starts = jx_ParCSRCommPkgRecvVecStarts(comm_pkg_RT);
   JX_Int num_sends        = jx_ParCSRCommPkgNumSends(comm_pkg_RT);
   JX_Int *send_procs      = jx_ParCSRCommPkgSendProcs(comm_pkg_RT);
   JX_Int *send_map_starts = jx_ParCSRCommPkgSendMapStarts(comm_pkg_RT);

   jx_CSRMatrix *RAP_ext;

   JX_Int     *RAP_ext_i    = NULL;
   JX_Int     *RAP_ext_j    = NULL;
   JX_Real  *RAP_ext_data = NULL;

   jx_ParCSRCommHandle *comm_handle = NULL;
   jx_ParCSRCommPkg *tmp_comm_pkg   = NULL;

   JX_Int *jdata_recv_vec_starts;
   JX_Int *jdata_send_map_starts;

   JX_Int num_rows;
   JX_Int num_nonzeros;
   JX_Int i, j;
   JX_Int num_procs;

   jx_MPI_Comm_size(comm, &num_procs);

   RAP_ext_i             = jx_CTAlloc(JX_Int, send_map_starts[num_sends] + 1);
   jdata_recv_vec_starts = jx_CTAlloc(JX_Int, num_recvs + 1);
   jdata_send_map_starts = jx_CTAlloc(JX_Int, num_sends + 1);
 
  /*--------------------------------------------------------------------------
   * recompute RAP_int_i so that RAP_int_i[j+1] contains the number of
   * elements of row j (to be determined through send_map_elmnts on the
   * receiving end)
   *--------------------------------------------------------------------------*/

   if (num_recvs)
   {
      RAP_int_i = jx_CSRMatrixI(RAP_int);
      RAP_int_j = jx_CSRMatrixJ(RAP_int);
      RAP_int_data = jx_CSRMatrixData(RAP_int);
      num_cols = jx_CSRMatrixNumCols(RAP_int);
   }

   jdata_recv_vec_starts[0] = 0;
   for (i = 0; i < num_recvs; i ++)
   {
      jdata_recv_vec_starts[i+1] = RAP_int_i[recv_vec_starts[i+1]];
   }
 
   for (i = num_recvs; i > 0; i --)
   {
      for (j = recv_vec_starts[i]; j > recv_vec_starts[i-1]; j --)
      {
         RAP_int_i[j] -= RAP_int_i[j-1];
      }
   }

  /*--------------------------------------------------------------------------
   * initialize communication 
   *--------------------------------------------------------------------------*/

   if (num_recvs && num_sends)
   {
      comm_handle = jx_ParCSRCommHandleCreate(12,comm_pkg_RT, &RAP_int_i[1], &RAP_ext_i[1]);
   }
   else if (num_recvs)
   {
      comm_handle = jx_ParCSRCommHandleCreate(12,comm_pkg_RT, &RAP_int_i[1], NULL);
   }
   else if (num_sends)
   {
      comm_handle = jx_ParCSRCommHandleCreate(12,comm_pkg_RT, NULL, &RAP_ext_i[1]);
   }

   tmp_comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
   jx_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_recvs;
   jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_sends;
   jx_ParCSRCommPkgSendProcs(tmp_comm_pkg) = recv_procs;
   jx_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = send_procs;

   jx_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
   * compute num_nonzeros for RAP_ext
   *--------------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i ++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
      {
         RAP_ext_i[j+1] += RAP_ext_i[j];
      }
   }

   num_rows = send_map_starts[num_sends];
   num_nonzeros = RAP_ext_i[num_rows];
   if (num_nonzeros)
   {
      RAP_ext_j = jx_CTAlloc(JX_Int, num_nonzeros);
      RAP_ext_data = jx_CTAlloc(JX_Real, num_nonzeros);
   }

   for (i = 0; i < num_sends+1; i ++)
   {
      jdata_send_map_starts[i] = RAP_ext_i[send_map_starts[i]];
   }

   jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_send_map_starts;      
   jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_recv_vec_starts;      

   comm_handle = jx_ParCSRCommHandleCreate(1, tmp_comm_pkg, RAP_int_data, RAP_ext_data);
   jx_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   comm_handle = jx_ParCSRCommHandleCreate(11, tmp_comm_pkg, RAP_int_j, RAP_ext_j);
   RAP_ext = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

   jx_CSRMatrixI(RAP_ext) = RAP_ext_i;
   if (num_nonzeros)
   {
      jx_CSRMatrixJ(RAP_ext) = RAP_ext_j;
      jx_CSRMatrixData(RAP_ext) = RAP_ext_data;
   }

   jx_ParCSRCommHandleDestroy(comm_handle); 
   comm_handle = NULL;

   jx_TFree(jdata_recv_vec_starts);
   jx_TFree(jdata_send_map_starts);
   jx_TFree(tmp_comm_pkg);

   return RAP_ext;
}


/*!
 * \fn JX_Int jx_PAMGBuildCoarseOperator
 * \brief Build the coarse operator in PAMG.
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGBuildCoarseOperator( jx_ParCSRMatrix  *RT,
                            jx_ParCSRMatrix  *par_A,
                            jx_ParCSRMatrix  *par_P,
                            jx_ParCSRMatrix **RAP_ptr )
{
   return (jx_PAMGBuildCoarseOperatorKT(RT, par_A, par_P, 0, RAP_ptr));
}

/*!
 * \fn int jx_PAMGBuildCoarseOperatorKT
 * \brief Build the coarse operator in PAMG.
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGBuildCoarseOperatorKT( jx_ParCSRMatrix  *RT,
                              jx_ParCSRMatrix  *par_A,
                              jx_ParCSRMatrix  *par_P,
                              JX_Int            keepTranspose,
                              jx_ParCSRMatrix **RAP_ptr )
{
   MPI_Comm          comm = jx_ParCSRMatrixComm(par_A);

   jx_CSRMatrix     *RT_diag = jx_ParCSRMatrixDiag(RT);
   jx_CSRMatrix     *RT_offd = jx_ParCSRMatrixOffd(RT);
   JX_Int               num_cols_diag_RT = jx_CSRMatrixNumCols(RT_diag);
   JX_Int               num_cols_offd_RT = jx_CSRMatrixNumCols(RT_offd);
   JX_Int               num_rows_offd_RT = jx_CSRMatrixNumRows(RT_offd);
   jx_ParCSRCommPkg *comm_pkg_RT = jx_ParCSRMatrixCommPkg(RT);
   JX_Int               num_recvs_RT = 0;
   JX_Int               num_sends_RT = 0;
   JX_Int              *send_map_starts_RT = NULL;
   JX_Int              *send_map_elmts_RT  = NULL;

   jx_CSRMatrix     *A_diag = jx_ParCSRMatrixDiag(par_A);
   
   JX_Real           *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int              *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int              *A_diag_j    = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix     *A_offd = jx_ParCSRMatrixOffd(par_A);
   
   JX_Real           *A_offd_data = jx_CSRMatrixData(A_offd);
   JX_Int              *A_offd_i    = jx_CSRMatrixI(A_offd);
   JX_Int              *A_offd_j    = jx_CSRMatrixJ(A_offd);

   JX_Int  num_cols_diag_A = jx_CSRMatrixNumCols(A_diag);
   JX_Int  num_cols_offd_A = jx_CSRMatrixNumCols(A_offd);

   jx_CSRMatrix    *P_diag = jx_ParCSRMatrixDiag(par_P);
   
   JX_Real          *P_diag_data = jx_CSRMatrixData(P_diag);
   JX_Int             *P_diag_i    = jx_CSRMatrixI(P_diag);
   JX_Int             *P_diag_j    = jx_CSRMatrixJ(P_diag);

   jx_CSRMatrix    *P_offd         = jx_ParCSRMatrixOffd(par_P);
   JX_Int             *col_map_offd_P = jx_ParCSRMatrixColMapOffd(par_P);
   
   JX_Real          *P_offd_data = jx_CSRMatrixData(P_offd);
   JX_Int             *P_offd_i    = jx_CSRMatrixI(P_offd);
   JX_Int             *P_offd_j    = jx_CSRMatrixJ(P_offd);

   JX_Int  first_col_diag_P    = jx_ParCSRMatrixFirstColDiag(par_P);
   JX_Int  last_col_diag_P;
   JX_Int  num_cols_diag_P     = jx_CSRMatrixNumCols(P_diag);
   JX_Int  num_cols_offd_P     = jx_CSRMatrixNumCols(P_offd);
   JX_Int *coarse_partitioning = jx_ParCSRMatrixColStarts(par_P);
   JX_Int *RT_partitioning     = jx_ParCSRMatrixColStarts(RT);

   jx_ParCSRMatrix *RAP = NULL;
   JX_Int             *col_map_offd_RAP = NULL;
   JX_Int             *new_col_map_offd_RAP = NULL;

   jx_CSRMatrix    *RAP_int = NULL;

   JX_Real          *RAP_int_data = NULL;
   JX_Int             *RAP_int_i    = NULL;
   JX_Int             *RAP_int_j    = NULL;

   jx_CSRMatrix    *RAP_ext = NULL;

   JX_Real          *RAP_ext_data = NULL;
   JX_Int             *RAP_ext_i    = NULL;
   JX_Int             *RAP_ext_j    = NULL;

   jx_CSRMatrix    *RAP_diag = NULL;

   JX_Real          *RAP_diag_data = NULL;
   JX_Int             *RAP_diag_i    = NULL;
   JX_Int             *RAP_diag_j    = NULL;

   jx_CSRMatrix    *RAP_offd = NULL;

   JX_Real          *RAP_offd_data = NULL;
   JX_Int             *RAP_offd_i    = NULL;
   JX_Int             *RAP_offd_j    = NULL;

   JX_Int              RAP_size;
   JX_Int              RAP_ext_size;
   JX_Int              RAP_diag_size;
   JX_Int              RAP_offd_size;
   JX_Int              P_ext_diag_size;
   JX_Int              P_ext_offd_size;
   JX_Int              first_col_diag_RAP;
   JX_Int              last_col_diag_RAP;
   JX_Int              num_cols_offd_RAP = 0;
   
   jx_CSRMatrix    *R_diag = NULL;
   
   JX_Real          *R_diag_data;
   JX_Int             *R_diag_i;
   JX_Int             *R_diag_j;

   jx_CSRMatrix    *R_offd = NULL;
   
   JX_Real          *R_offd_data = NULL;
   JX_Int             *R_offd_i    = NULL;
   JX_Int             *R_offd_j    = NULL;

   jx_CSRMatrix    *Ps_ext = NULL;
   
   JX_Real          *Ps_ext_data = NULL;
   JX_Int             *Ps_ext_i    = NULL;
   JX_Int             *Ps_ext_j    = NULL;

   JX_Real          *P_ext_diag_data = NULL;
   JX_Int             *P_ext_diag_i    = NULL;
   JX_Int             *P_ext_diag_j    = NULL;

   JX_Real          *P_ext_offd_data = NULL;
   JX_Int             *P_ext_offd_i    = NULL;
   JX_Int             *P_ext_offd_j    = NULL;

   JX_Int             *col_map_offd_Pext = NULL;
   JX_Int             *map_P_to_Pext     = NULL;
   JX_Int             *map_P_to_RAP      = NULL;
   JX_Int             *map_Pext_to_RAP   = NULL;

   JX_Int             *P_marker = NULL;
   JX_Int            **P_mark_array;
   JX_Int            **A_mark_array;
   JX_Int             *A_marker;
   JX_Int             *temp = NULL;

   JX_Int              n_coarse, n_coarse_RT;
   JX_Int              square = 1;
   JX_Int              num_cols_offd_Pext = 0;
   
   JX_Int              ic, i, j, k;
   JX_Int              i1, i2, i3, ii, ns, ne, size, rest;
   JX_Int              cnt, cnt_offd, cnt_diag, value;
   JX_Int              jj1, jj2, jj3, jcol;
   
   JX_Int             *jj_count, *jj_cnt_diag, *jj_cnt_offd;
   JX_Int              jj_counter, jj_count_diag, jj_count_offd;
   JX_Int              jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
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
      num_recvs_RT = jx_ParCSRCommPkgNumRecvs(comm_pkg_RT);
      num_sends_RT = jx_ParCSRCommPkgNumSends(comm_pkg_RT);
      send_map_starts_RT = jx_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
      send_map_elmts_RT  = jx_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }

   jx_CSRMatrixTranspose(RT_diag, &R_diag, 1); 
   if (num_cols_offd_RT) 
   {
      jx_CSRMatrixTranspose(RT_offd ,&R_offd, 1); 
      R_offd_data = jx_CSRMatrixData(R_offd);
      R_offd_i    = jx_CSRMatrixI(R_offd);
      R_offd_j    = jx_CSRMatrixJ(R_offd);
   }

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
   if (n_coarse != n_coarse_RT)
   {
      square = 0;
   }

  /*----------------------------------------------------------------------------
   *  Generate Ps_ext, i.e. portion of par_P that is stored on neighbor procs
   *  and needed locally for triple matrix product 
   *---------------------------------------------------------------------------*/

   if (num_procs > 1) 
   {
      Ps_ext      = jx_ParCSRMatrixExtractBExt(par_P, par_A,1);
      Ps_ext_data = jx_CSRMatrixData(Ps_ext);
      Ps_ext_i    = jx_CSRMatrixI(Ps_ext);
      Ps_ext_j    = jx_CSRMatrixJ(Ps_ext);
   }

   P_ext_diag_i = jx_CTAlloc(JX_Int, num_cols_offd_A + 1);
   P_ext_offd_i = jx_CTAlloc(JX_Int, num_cols_offd_A + 1);
   P_ext_diag_size = 0;
   P_ext_offd_size = 0;
   last_col_diag_P = first_col_diag_P + num_cols_diag_P - 1;

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

   cnt_offd = 0;
   cnt_diag = 0;
   cnt = 0;
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

  /*----------------------------------------------------------------------    
   *  First Pass: Determine size of RAP_int and set up RAP_int_i if there 
   *  are more than one processor and nonzero elements in R_offd
   *-----------------------------------------------------------------------*/

   P_mark_array = jx_CTAlloc(JX_Int *, num_threads);
   A_mark_array = jx_CTAlloc(JX_Int *, num_threads);







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



   RAP_ext_size = 0;
   
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
 
   RAP_diag_i = jx_CTAlloc(JX_Int, num_cols_diag_RT + 1);
   RAP_offd_i = jx_CTAlloc(JX_Int, num_cols_diag_RT + 1);

   first_col_diag_RAP = first_col_diag_P;
   last_col_diag_RAP = first_col_diag_P + num_cols_diag_P - 1;


  /*-----------------------------------------------------------------------
   *  check for new nonzero columns in RAP_offd generated through RAP_ext
   *-----------------------------------------------------------------------*/

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


  /*-----------------------------------------------------------------------
   *  Convert RAP_ext column indices
   *-----------------------------------------------------------------------*/

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

   /* need to allocate new P_marker etc. and make further changes */
   
  /*-----------------------------------------------------------------------
   *  Initialize some stuff.
   *-----------------------------------------------------------------------*/
   
   jj_cnt_diag = jx_CTAlloc(JX_Int, num_threads);
   jj_cnt_offd = jx_CTAlloc(JX_Int, num_threads);

#define JX_SMP_PRIVATE i,j,k,jcol,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker
#include "../../../include/jx_smp_forloop.h"
   for (ii = 0; ii < num_threads; ii ++)
   {
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
 
 
        /*--------------------------------------------------------------------
         *  Loop over entries in row ic of R_diag.
         *--------------------------------------------------------------------*/
   
         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic+1]; jj1 ++)
         {
            i1  = R_diag_j[jj1];
 
           /*-----------------------------------------------------------------
            *  Loop over entries in row i1 of A_offd.
            *-----------------------------------------------------------------*/
         
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
                  
               } // end if (A_marker[i2+num_cols_offd_A] != ic)
               
            } // for jj2
            
         } // end for jj1
                 
      } // end for ic
      
      jj_cnt_diag[ii] = jj_count_diag;
      jj_cnt_offd[ii] = jj_count_offd;
      
   } // end for ii

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
   if (RAP_offd_size)
   { 
      RAP_offd_data = jx_CTAlloc(JX_Real, RAP_offd_size);
      RAP_offd_j    = jx_CTAlloc(JX_Int, RAP_offd_size);
   } 

   if (RAP_offd_size == 0 && num_cols_offd_RAP != 0)
   {
      num_cols_offd_RAP = 0;
      jx_TFree(col_map_offd_RAP);
   }
   
   
  /*-----------------------------------------------------------------------
   *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
   *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
   *-----------------------------------------------------------------------*/

#define JX_SMP_PRIVATE i,j,k,jcol,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker,r_entry,r_a_product,r_a_p_product
#include "../../../include/jx_smp_forloop.h"
   
   for (ii = 0; ii < num_threads; ii ++)
   {
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
 
 
        /*--------------------------------------------------------------------
         *  Loop over entries in row ic of R_diag.
         *--------------------------------------------------------------------*/

         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic+1]; jj1 ++)
         {
            i1  = R_diag_j[jj1];
            r_entry = R_diag_data[jj1];

           /*-----------------------------------------------------------------
            *  Loop over entries in row i1 of A_offd.
            *-----------------------------------------------------------------*/
         
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
               
               }

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
                     RAP_diag_data[P_marker[i3]] += r_a_p_product;
                  }
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3 ++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];
                        RAP_offd_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
                  
               }  // end if (A_marker[i2+num_cols_offd_A] != ic)
               
            } // end for jj2
            
         } // end for jj1
         
      } // end for ic
      
      jx_TFree(P_mark_array[ii]);   
      jx_TFree(A_mark_array[ii]);  
       
   } // end for ii



  /*---------------------------------------------------------------
   *    Check if really all off-diagonal entries occurring in 
   * col_map_offd_RAP are represented and eliminate if necessary. 
   *--------------------------------------------------------------*/

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
   
   if (num_cols_offd_RAP)
   {
      jx_CSRMatrixData(RAP_offd) = RAP_offd_data; 
      jx_CSRMatrixJ(RAP_offd) = RAP_offd_j; 
      jx_ParCSRMatrixColMapOffd(RAP) = col_map_offd_RAP;
   }
   
   if (num_procs > 1)
   {
      jx_MatvecCommPkgCreate(RAP); 
   }

   *RAP_ptr = RAP;

  /*-----------------------------------------------------------------------
   *  Free R, P_ext and marker arrays.
   *-----------------------------------------------------------------------*/

   if (keepTranspose)
   {
      jx_ParCSRMatrixDiagT(RT) = R_diag;
   }
   else
   {
      jx_CSRMatrixDestroy(R_diag);
   }
   R_diag = NULL;

   if (num_cols_offd_RT) 
   {
      if (keepTranspose)
      {
         jx_ParCSRMatrixOffdT(RT) = R_offd;
      }
      else
      {
         jx_CSRMatrixDestroy(R_offd);
      }
      R_offd = NULL;
   }

   if (num_sends_RT || num_recvs_RT) 
   {
      jx_CSRMatrixDestroy(RAP_ext);
      RAP_ext = NULL;
   }
   jx_TFree(P_mark_array);   
   jx_TFree(A_mark_array);
   jx_TFree(P_ext_diag_i);
   jx_TFree(P_ext_offd_i);
   jx_TFree(jj_cnt_diag);   
   jx_TFree(jj_cnt_offd);

   if (num_cols_offd_P)
   {
      jx_TFree(map_P_to_Pext);
      jx_TFree(map_P_to_RAP);
   }
   
   if (num_cols_offd_Pext)
   {
      jx_TFree(col_map_offd_Pext);
      jx_TFree(map_Pext_to_RAP);
   }
   
   if (P_ext_diag_size)
   {
      jx_TFree(P_ext_diag_data);
      jx_TFree(P_ext_diag_j);
   }
   
   if (P_ext_offd_size)
   {
      jx_TFree(P_ext_offd_data);
      jx_TFree(P_ext_offd_j);
   }

   return(0);
}
