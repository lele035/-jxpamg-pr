//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_restr.c -- Build the distance-1/2 AIR.
 *  Date: 2018/05/10
 */ 

#include "jx_pamg.h"
#include "jx_blas2.h"
#include "jx_lapack2.h"

/*!
 * \fn JX_Int jx_PAMGBuildRestrAIR
 * \brief Build the distance-1 AIR.
 * \date 2018/05/10
 */
JX_Int
jx_PAMGBuildRestrAIR( jx_ParCSRMatrix   *A,
                      JX_Int            *CF_marker,
                      jx_ParCSRMatrix   *S,
                      JX_Int            *num_cpts_global,
                      JX_Int             num_functions,
                      JX_Int            *dof_func,
                      JX_Int             debug_flag,
                      JX_Real            trunc_factor,
                      JX_Int             max_elmts,
                      JX_Int            *col_offd_S_to_A,
                      jx_ParCSRMatrix  **R_ptr,
                      JX_Int            *R_max_size_ptr )
{
   MPI_Comm                 comm     = jx_ParCSRMatrixComm(A);
   jx_ParCSRCommPkg     *comm_pkg = jx_ParCSRMatrixCommPkg(A);
   jx_ParCSRCommHandle  *comm_handle;
   /* diag part of A */
   jx_CSRMatrix *A_diag      = jx_ParCSRMatrixDiag(A);
   JX_Real      *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int       *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int       *A_diag_j    = jx_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   jx_CSRMatrix *A_offd      = jx_ParCSRMatrixOffd(A);   
   JX_Real      *A_offd_data = jx_CSRMatrixData(A_offd);
   JX_Int       *A_offd_i    = jx_CSRMatrixI(A_offd);
   JX_Int       *A_offd_j    = jx_CSRMatrixJ(A_offd);

   JX_Int        num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
   JX_Int       *col_map_offd_A  = jx_ParCSRMatrixColMapOffd(A);
   /* Strength matrix S */
   /* diag part of S */
   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(S);
   JX_Int       *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int       *S_diag_j = jx_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(S);   
   JX_Int       *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int       *S_offd_j = jx_CSRMatrixJ(S_offd);
   /* Restriction matrix R */
   jx_ParCSRMatrix *R;
   /* csr's */
   jx_CSRMatrix *R_diag;
   jx_CSRMatrix *R_offd;
   /* arrays */
   JX_Real      *R_diag_data;
   JX_Int       *R_diag_i;
   JX_Int       *R_diag_j;
   JX_Real      *R_offd_data;
   JX_Int       *R_offd_i;
   JX_Int       *R_offd_j;
   JX_Int       *col_map_offd_R;
   /* CF marker off-diag part */
   JX_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   JX_Int       *dof_func_offd  = NULL;
   /* ghost rows */
   jx_CSRMatrix *A_ext      = NULL;
   JX_Real      *A_ext_data = NULL;
   JX_Int       *A_ext_i    = NULL;
   JX_Int       *A_ext_j    = NULL;

   JX_Int        i, j, k, i1, k1, k2, rr, cc, ic, index, start, local_max_size, local_size, num_cols_offd_R;

   /* LAPACK */
   JX_Real *DAi, *Dbi;
   JX_Int *Ipi, lapack_info, ione = 1;
   char charT = 'T';

   JX_Int my_id, num_procs;
   JX_Int total_global_cpts;
   JX_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   JX_Int *marker_diag, *marker_offd;
   JX_Int num_sends, *int_buf_data;
   /* local size, local num of C points */
   JX_Int n_fine = jx_CSRMatrixNumRows(A_diag);
   JX_Int n_cpts = 0;
   JX_Int col_start = jx_ParCSRMatrixFirstRowIndex(A);
   JX_Int col_end   = col_start + n_fine;

   /* MPI size and rank*/
   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
   total_global_cpts = num_cpts_global[num_procs];

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
   if (num_cols_A_offd)
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }
   /* number of sends to do (number of procs) */
   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   /* send buffer, of size send_map_starts[num_sends]), 
    * i.e., number of entries to send */
   int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
   /* copy CF markers of elements to send to buffer 
    * RL: why copy them with two for loops? Why not just loop through all in one */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         int_buf_data[index++] = CF_marker[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   /* destroy the handle to finish communication */
   jx_ParCSRCommHandleDestroy(comm_handle);   
   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            int_buf_data[index++] = dof_func[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);   
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine the nnz of R and the max local size
    *-----------------------------------------------------------------------*/
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;
   for (i = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }
      /* local number of C-pts */
      n_cpts ++;
      /* If i is a C-point, the restriction is from the F-points that
       * strongly influence i */
      local_size = 0;
      /* loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            cnt_diag ++;
            local_size ++;
         }
      }
      /* if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         /* use this mapping to have offd indices of A */
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            if (CF_marker_offd[i1] < 0)
            {
               cnt_offd ++;
               local_size ++;
            }
         }
      }
      /* keep ths max size */
      local_max_size = jx_max(local_max_size, local_size);
   }

   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;

   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;

   /*------------- allocate arrays */
   R_diag_i    = jx_CTAlloc(JX_Int,  n_cpts+1);
   R_diag_j    = jx_CTAlloc(JX_Int,  nnz_diag);
   R_diag_data = jx_CTAlloc(JX_Real, nnz_diag);

   /* not in ``if num_procs > 1'', 
    * allocation needed even for empty CSR */
   R_offd_i    = jx_CTAlloc(JX_Int,  n_cpts+1);
   R_offd_j    = jx_CTAlloc(JX_Int,  nnz_offd);
   R_offd_data = jx_CTAlloc(JX_Real, nnz_offd);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;
 
   /*----------------------------------------       .-.
    * Get the GHOST rows of A,                     (o o) boo!
    * i.e., adjacent rows to this proc             | O \
    * whose row indices are in A->col_map_offd      \   \
    *-----------------------------------------       `~~~'  */
   /* external rows of A that are needed for perform A multiplication,
    * the last arg means need data 
    * the number of rows is num_cols_A_offd */
   if (num_procs > 1)
   {
      A_ext      = jx_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = jx_CSRMatrixI(A_ext);
      A_ext_j    = jx_CSRMatrixJ(A_ext);
      A_ext_data = jx_CSRMatrixData(A_ext);
   }

   /* marker array: if this point is i's strong F neighbors
    *             >=  0: yes, and is the local dense id 
    *             == -1: no */
   marker_diag = jx_CTAlloc(JX_Int, n_fine);
   for (i = 0; i < n_fine; i++)
   {
      marker_diag[i] = -1;
   }
   marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   for (i = 0; i< num_cols_A_offd; i++)
   {
      marker_offd[i] = -1;
   }

   /* the local matrix and rhs (dense) 
    * column-major as always by BLAS/LAPACK */
   /* matrix */
   DAi = jx_CTAlloc(JX_Real, local_max_size * local_max_size);
   /* rhs */
   Dbi = jx_CTAlloc(JX_Real, local_max_size);
   /* pivot */
   Ipi = jx_CTAlloc(JX_Int, local_max_size);

   /*-----------------------------------------------------------------------
    *  Second Pass: Populate R
    *-----------------------------------------------------------------------*/
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* size of Ai, bi */
      local_size = 0;
      
      /* If i is a C-point, build the restriction, from the F-points that
       * strongly influence i
       * Access S for the first time, mark the points we want */
      /* 1: loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            jx_assert(marker_diag[i1] == -1);
            /* mark this point */
            marker_diag[i1] = local_size ++;
         }
      }
      /* 2: if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               jx_assert(marker_offd[i1] == -1);
               /* mark this point */
               marker_offd[i1] = local_size ++;
            }
         }
      }

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries 
       * we need from these rows and put them in Ai and bi*/

      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(JX_Real));
      memset(Dbi, 0, local_size * sizeof(JX_Real));

      /* we will populate Ai, bi row-by-row
       * rr is the local dense matrix row counter */
      rr = 0;
      /* 1. diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         /* row i1 */
         i1 = S_diag_j[j];
         /* i1 is an F point */
         if (CF_marker[i1] < 0)
         {
            /* go through row i1 of A: a local row */
            /* diag part of row i1 */
            for (k = A_diag_i[i1]; k < A_diag_i[i1+1]; k++)
            {
               k1 = A_diag_j[k];
               /* if this col is marked with its local dense id */
               if ((cc = marker_diag[k1]) >= 0)
               {
                  jx_assert(CF_marker[k1] < 0);
                  /* copy the value */
                  /* rr and cc: local dense ids */
                  DAi[rr + cc * local_size] = A_diag_data[k];
               }
            }
            /* if parallel, offd part of row i1 */
            if (num_procs > 1)
            {
               for (k = A_offd_i[i1]; k < A_offd_i[i1+1]; k++)
               {
                  k1 = A_offd_j[k];
                  /* if this col is marked with its local dense id */
                  if ((cc = marker_offd[k1]) >= 0)
                  {
                     jx_assert(CF_marker_offd[k1] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     DAi[rr + cc * local_size] = A_offd_data[k];
                  }
               }
            }
            /* done with row i1 */
            rr++;
         }
      } /* for (j=...), diag part of row i done */

      /* 2. if parallel, offd part of row i. The corresponding rows are
       *    in matrix A_ext */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* row i1: use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* if this is an F point */
            if (CF_marker_offd[i1] < 0)
            {
               /* loop through row i1 of A_ext, a global CSR matrix */
               for (k = A_ext_i[i1]; k < A_ext_i[i1+1]; k++)
               {
                  /* k1 is a global index! */
                  k1 = A_ext_j[k];
                  if (k1 >= col_start && k1 < col_end)
                  {
                     /* k1 is in the diag part, adjust to local index */
                     k1 -= col_start;
                     /* if this col is marked with its local dense id*/
                     if ((cc = marker_diag[k1]) >= 0)
                     {
                        jx_assert(CF_marker[k1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = A_ext_data[k];
                     }
                  }
                  else
                  {
                     /* k1 is in the offd part
                      * search k1 in A->col_map_offd */
                     k2 = jx_BinarySearch(col_map_offd_A, k1, num_cols_A_offd);
                     /* if found, k2 is the position of column id k1 in col_map_offd */
                     if (k2 > -1)
                     {
                        /* if this col is marked with its local dense id */
                        if ((cc = marker_offd[k2]) >= 0)
                        {
                           jx_assert(CF_marker_offd[k2] < 0);
                           /* copy the value */
                           /* rr and cc: local dense ids */
                           DAi[rr + cc * local_size] = A_ext_data[k];
                        }
                     }
                  }
               }
               /* done with row i1 */
               rr++;
            }
         }
      }

      jx_assert(rr == local_size);

      /* assemble rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = marker_diag[i1]) >= 0)
         {
            /* this should be true but not very important
             * what does it say is that eqn order == unknown order
             * this is true, since order in A is preserved in S */
            jx_assert(rr == cc);
            /* Note the sign change */
            Dbi[cc] = -A_diag_data[j];
            rr++;
         }
      }
      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            i1 = A_offd_j[j];
            if ((cc = marker_offd[i1]) >= 0)
            {
               /* this should be true but not very important
                * what does it say is that eqn order == unknown order
                * this is true, since order in A is preserved in S */
               jx_assert(rr == cc);
               /* Note the sign change */
               Dbi[cc] = -A_offd_data[j];
               rr++;
            }
         }
      }
      jx_assert(rr == local_size);

      if (local_size > 0)
      {
         /* we have Ai and bi build
          * solve the linear system by LAPACK : LU factorization */

         jx_dgetrf(&local_size, &local_size, DAi, &local_size, Ipi, &lapack_info);

         jx_assert(lapack_info == 0);

         if (lapack_info == 0)
         {
            /* solve A_i^T x_i = b_i,
             * solution is saved in b_i on return */
            jx_dgetrs(&charT, &local_size, &ione, DAi, &local_size, Ipi, Dbi, &local_size, &lapack_info);
            jx_assert(lapack_info == 0);
         }
      }
      
      /* now we are ready to fill this row of R */
      /* diag part */
      rr = 0;
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            jx_assert(marker_diag[i1] == rr);
            /* col idx: use i1, local idx  */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Dbi[rr++];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i''; */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;
      
      /* row ptr of the next row */
      R_diag_i[ic+1] = cnt_diag;

      /* offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               jx_assert(marker_offd[i1] == rr);
               /* col idx: use the local col id of A_offd,
                * and you will see why later (very soon!) */
               R_offd_j[cnt_offd] = i1;
               /* copy the value */
               R_offd_data[cnt_offd++] = Dbi[rr++];
            }
         }
      }
      /* row ptr of the next row */
      R_offd_i[ic+1] = cnt_offd;

      /* we must have copied all entries */
      jx_assert(rr == local_size);
      
      /* reset markers */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            jx_assert(marker_diag[i1] >= 0);
            marker_diag[i1] = -1;
         }
      }
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               jx_assert(marker_offd[i1] >= 0);
               marker_offd[i1] = -1;
            }
         }
      }

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */

   jx_assert(ic == n_cpts)
   jx_assert(cnt_diag == nnz_diag)
   jx_assert(cnt_offd == nnz_offd)
   
   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, marker_offd should be all -1 */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (marker_offd[i1] == -1)
      {
         num_cols_offd_R++;
         marker_offd[i1] = 1;
      }
   }

   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the offd-idx of A */
   col_map_offd_R = jx_CTAlloc(JX_Int, num_cols_offd_R);
   for (i = 0, i1 = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i] == 1)
      {
         col_map_offd_R[i1++] = i;
      }
   }
   jx_assert(i1 == num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t col_map_offd_R
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = jx_BinarySearch(col_map_offd_R, i1, num_cols_offd_R);
      /* search must succeed */
      jx_assert(k1 >= 0 && k1 < num_cols_offd_R);
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = col_map_offd_A[col_map_offd_R[i]];
   }

   /* Now, we should have everything of Parcsr matrix R */
   R = jx_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                jx_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                jx_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = jx_ParCSRMatrixDiag(R);
   jx_CSRMatrixData(R_diag) = R_diag_data;
   jx_CSRMatrixI(R_diag)    = R_diag_i;
   jx_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = jx_ParCSRMatrixOffd(R);
   jx_CSRMatrixData(R_offd) = R_offd_data;
   jx_CSRMatrixI(R_offd)    = R_offd_i;
   jx_CSRMatrixJ(R_offd)    = R_offd_j;
   /* R does not own ColStarts, since A does */
   jx_ParCSRMatrixOwnsColStarts(R) = 0;
   
   jx_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   jx_MatvecCommPkgCreate(R);

  *R_ptr = R;
  *R_max_size_ptr = local_max_size;

   /* free workspace */
   jx_TFree(CF_marker_offd);
   jx_TFree(dof_func_offd);
   jx_TFree(int_buf_data);
   jx_TFree(marker_diag);
   jx_TFree(marker_offd);
   jx_TFree(DAi);
   jx_TFree(Dbi);
   jx_TFree(Ipi);
   if (num_procs > 1)
   {
      jx_CSRMatrixDestroy(A_ext);
   }

   return 0;
}

/*!
 * \fn JX_Int jx_PAMGBuildRestrDist2AIR
 * \brief Build the distance-2 AIR.
 * \date 2018/05/10
 */
JX_Int
jx_PAMGBuildRestrDist2AIR( jx_ParCSRMatrix   *A,
                           JX_Int            *CF_marker,
                           jx_ParCSRMatrix   *S,
                           JX_Int            *num_cpts_global,
                           JX_Int             num_functions,
                           JX_Int            *dof_func,
                           JX_Int             debug_flag,
                           JX_Real            trunc_factor,
                           JX_Int             max_elmts,
                           JX_Int            *col_offd_S_to_A,
                           jx_ParCSRMatrix  **R_ptr,
                           JX_Int            *R_max_size_ptr )
{
   MPI_Comm              comm     = jx_ParCSRMatrixComm(A);
   jx_ParCSRCommPkg     *comm_pkg = jx_ParCSRMatrixCommPkg(A);
   jx_ParCSRCommHandle  *comm_handle;

   jx_ParCSRCommPkg     *comm_pkg_SF;
   
   /* diag part of A */
   jx_CSRMatrix *A_diag   = jx_ParCSRMatrixDiag(A);
   JX_Real      *A_diag_a = jx_CSRMatrixData(A_diag);
   JX_Int       *A_diag_i = jx_CSRMatrixI(A_diag);
   JX_Int       *A_diag_j = jx_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   jx_CSRMatrix *A_offd   = jx_ParCSRMatrixOffd(A);   
   JX_Real      *A_offd_a = jx_CSRMatrixData(A_offd);
   JX_Int       *A_offd_i = jx_CSRMatrixI(A_offd);
   JX_Int       *A_offd_j = jx_CSRMatrixJ(A_offd);

   JX_Int        num_cols_A_offd = jx_CSRMatrixNumCols(A_offd);
   JX_Int       *col_map_offd_A  = jx_ParCSRMatrixColMapOffd(A);
   /* Strength matrix S */
   /* diag part of S */
   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(S);
   JX_Int       *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int       *S_diag_j = jx_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(S);   
   JX_Int       *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int       *S_offd_j = jx_CSRMatrixJ(S_offd);
   /* Restriction matrix R */
   jx_ParCSRMatrix *R;
   /* csr's */
   jx_CSRMatrix *R_diag;
   jx_CSRMatrix *R_offd;
   /* arrays */
   JX_Real      *R_diag_data;
   JX_Int       *R_diag_i;
   JX_Int       *R_diag_j;
   JX_Real      *R_offd_data;
   JX_Int       *R_offd_i;
   JX_Int       *R_offd_j;
   JX_Int       *col_map_offd_R;
   /* CF marker off-diag part */
   JX_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   JX_Int       *dof_func_offd  = NULL;
   
   JX_Int        i, j, j1, j2, k, i1, i2, k1, k2, k3, rr, cc, ic, index, start, end, local_max_size, local_size, num_cols_offd_R;
   /*JX_Int        i6;*/
   JX_Int        *FF2_offd, FF2_offd_len;

   /* LAPACK */
   JX_Real *DAi, *Dbi;
   JX_Int *Ipi, lapack_info, ione = 1, *RRi, *KKi;
   char charT = 'T';

   JX_Int my_id, num_procs;
   JX_Int total_global_cpts;
   JX_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   JX_Int *Marker_diag, *Marker_offd;
   JX_Int *Marker_diag_j, Marker_diag_count;
   JX_Int num_sends, num_recvs, num_elems_send;
   /* local size, local num of C points */
   JX_Int n_fine = jx_CSRMatrixNumRows(A_diag);
   JX_Int n_cpts = 0;
   JX_Int col_start = jx_ParCSRMatrixFirstRowIndex(A);
   JX_Int col_end   = col_start + n_fine;

   JX_Int  *send_buf_i;
   
   /* recv_SF means the Strong F-neighbors of offd elements in col_map_offd */
   JX_Int *send_SF_i, *send_SF_j, send_SF_jlen;
   JX_Int *recv_SF_i, *recv_SF_j, *recv_SF_j2, recv_SF_jlen;
   JX_Int *send_SF_jstarts, *recv_SF_jstarts;
   JX_Int *recv_SF_offd_list, recv_SF_offd_list_len;
   JX_Int *Mapper_recv_SF_offd_list, *Mapper_offd_A, *Marker_recv_SF_offd_list;
   JX_Int *Marker_FF2_offd;
   JX_Int *Marker_FF2_offd_j, Marker_FF2_offd_count;

   /* for communication of offd F and F^2 rows of A */
   jx_ParCSRCommPkg *comm_pkg_FF2_i, *comm_pkg_FF2_j;
   JX_Int num_sends_FF2, *send_FF2_i, send_FF2_ilen, *send_FF2_j, send_FF2_jlen,
             num_recvs_FF2, *recv_FF2_i, recv_FF2_ilen, *recv_FF2_j, recv_FF2_jlen,
             *send_FF2_jstarts, *recv_FF2_jstarts;
   JX_Real *send_FF2_a, *recv_FF2_a;

   /* ghost rows: offd F and F2-pts */
   jx_CSRMatrix *A_offd_FF2   = NULL;
   
   /* MPI size and rank*/
   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
   total_global_cpts = num_cpts_global[num_procs];
 
   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
   if (num_cols_A_offd)
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }

   /* init markers to zeros */
   Marker_diag = jx_CTAlloc(JX_Int, n_fine);
   Marker_offd = jx_CTAlloc(JX_Int, num_cols_A_offd);

   /* number of sends (number of procs) */
   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);

   /* number of recvs (number of procs) */
   num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
 
   /* number of elements to send */
   num_elems_send = jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* send buffer, of size send_map_starts[num_sends]), 
    * i.e., number of entries to send */
   send_buf_i = jx_CTAlloc(JX_Int , num_elems_send);
   
   /* copy CF markers of elements to send to buffer 
    * RL: why copy them with two for loops? Why not just loop through all in one */
   for (i = 0, index = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         send_buf_i[index++] = CF_marker[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, CF_marker_offd);
   /* destroy the handle to finish communication */
   jx_ParCSRCommHandleDestroy(comm_handle);

   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      for (i = 0, index = 0; i < num_sends; i++)
      {
         start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            send_buf_i[index++] = dof_func[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, dof_func_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);   
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *        Send/Recv Offd F-neighbors' strong F-neighbors
    *        F^2: OffdF - F
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
   send_SF_i = jx_CTAlloc(JX_Int, num_elems_send);
   recv_SF_i = jx_CTAlloc(JX_Int, num_cols_A_offd + 1);
   
   /* for each F-elem to send, find the number of strong F-neighbors */
   for (i = 0, send_SF_jlen = 0; i < num_elems_send; i++)
   {
      /* number of strong F-pts */
      send_SF_i[i] = 0;
      /* elem i1 */
      i1 = jx_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      /* ignore C-pts */
      if (CF_marker[i1] >= 0)
      {
         continue;
      }
      /* diag part of row i1 */
      for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
      {
         if (CF_marker[S_diag_j[j]] < 0)
         {
            send_SF_i[i] ++;
         }
      }
      /* offd part of row i1 */
      for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
      {
         j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
         if (CF_marker_offd[j1] < 0)
         {
            send_SF_i[i] ++;
         }
      }

      /* add to the num of elems going to be sent */
      send_SF_jlen += send_SF_i[i];
   }

   /* do communication */
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, send_SF_i, recv_SF_i+1);
   /* ... */
   jx_ParCSRCommHandleDestroy(comm_handle);
   
   send_SF_j = jx_CTAlloc(JX_Int, send_SF_jlen);
   send_SF_jstarts = jx_CTAlloc(JX_Int, num_sends + 1);

   for (i = 0, i1 = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* 1-past-the-end pos */
      end   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1);

      for (j = start; j < end; j++)
      {
         /* strong F-pt, j1 */
         j1 = jx_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         /* ignore C-pts */
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* diag part of row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1+1]; k++)
         {
            k1 = S_diag_j[k];
            if (CF_marker[k1] < 0)
            {
               send_SF_j[i1++] = col_start + k1;
            }
         }
         /* offd part of row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
         {
            k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];
            if (CF_marker_offd[k1] < 0)
            {
               send_SF_j[i1++] = col_map_offd_A[k1];
            }
         }
      }
      send_SF_jstarts[i+1] = i1;
   }

   jx_assert(i1 == send_SF_jlen);

   /* adjust recv_SF_i to ptrs */
   for (i = 1; i <= num_cols_A_offd; i++)
   {
      recv_SF_i[i] += recv_SF_i[i-1];
   }

   recv_SF_jlen = recv_SF_i[num_cols_A_offd];
   recv_SF_j = jx_CTAlloc(JX_Int, recv_SF_jlen);
   recv_SF_jstarts = jx_CTAlloc(JX_Int, num_recvs + 1);

   for (i = 1; i <= num_recvs; i++)
   {
      start = jx_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_SF_jstarts[i] = recv_SF_i[start];
   }

   /* create a communication package for SF_j */
   comm_pkg_SF = jx_CTAlloc(jx_ParCSRCommPkg, 1);
   jx_ParCSRCommPkgComm         (comm_pkg_SF) = comm;
   jx_ParCSRCommPkgNumSends     (comm_pkg_SF) = num_sends;
   jx_ParCSRCommPkgSendProcs    (comm_pkg_SF) = jx_ParCSRCommPkgSendProcs(comm_pkg);
   jx_ParCSRCommPkgSendMapStarts(comm_pkg_SF) = send_SF_jstarts;
   jx_ParCSRCommPkgNumRecvs     (comm_pkg_SF) = num_recvs;
   jx_ParCSRCommPkgRecvProcs    (comm_pkg_SF) = jx_ParCSRCommPkgRecvProcs(comm_pkg);
   jx_ParCSRCommPkgRecvVecStarts(comm_pkg_SF) = recv_SF_jstarts;

   /* do communication */
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg_SF, send_SF_j, recv_SF_j);
   /* ... */
   jx_ParCSRCommHandleDestroy(comm_handle);

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * recv_SF_offd_list: a sorted list of offd elems in recv_SF_j 
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   recv_SF_offd_list = jx_CTAlloc(JX_Int, recv_SF_jlen);
   for (i = 0, j = 0; i < recv_SF_jlen; i++)
   {
      i1 = recv_SF_j[i];
      /* offd */
      if (i1 < col_start || i1 >= col_end)
      {
         recv_SF_offd_list[j++] = i1;
      }
   }

   /* remove redundancy after sorting */
   jx_qsort0(recv_SF_offd_list, 0, j-1);

   for (i = 0, recv_SF_offd_list_len = 0; i < j; i++)
   {
      if (i == 0 || recv_SF_offd_list[i] != recv_SF_offd_list[i-1])
      {
         recv_SF_offd_list[recv_SF_offd_list_len++] = recv_SF_offd_list[i];
      }
   }

   /* make a copy of recv_SF_j in which
    * adjust the offd indices corresponding to recv_SF_offd_list */
   recv_SF_j2 = jx_CTAlloc(JX_Int, recv_SF_jlen);
   for (i = 0; i < recv_SF_jlen; i++)
   {
      i1 = recv_SF_j[i];
      if (i1 < col_start || i1 >= col_end)
      {
         j = jx_BinarySearch(recv_SF_offd_list, i1, recv_SF_offd_list_len);
         jx_assert(j >= 0 && j < recv_SF_offd_list_len);
         recv_SF_j2[i] = j;
      }
      else
      {
         recv_SF_j2[i] = -1;
      }
   }

   /* mapping to col_map_offd_A */
   Mapper_recv_SF_offd_list = jx_CTAlloc(JX_Int, recv_SF_offd_list_len);
   Marker_recv_SF_offd_list = jx_CTAlloc(JX_Int, recv_SF_offd_list_len);
   
   /* create a mapping from recv_SF_offd_list to col_map_offd_A for their intersections */
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      i1 = recv_SF_offd_list[i];
      jx_assert(i1 < col_start || i1 >= col_end);
      j = jx_BinarySearch(col_map_offd_A, i1, num_cols_A_offd);
      /* mapping to col_map_offd_A, if not found equal to -1 */
      Mapper_recv_SF_offd_list[i] = j;      
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *       Find offd F and F-F (F^2) neighboring points for C-pts
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
   for (i = 0, FF2_offd_len = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* diag(F)-offd(F) */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         j1 = S_diag_j[j];
         /* if it is F */
         if (CF_marker[j1] < 0)
         {
            /* go through its offd part */
            for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
            {
               k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];
               if (CF_marker_offd[k1] < 0)
               {
                  /* mark F pts */
                  if (!Marker_offd[k1])
                  {
                     FF2_offd_len ++;
                     Marker_offd[k1] = 1;
                  }
               }
            }
         }
      }

      /* offd(F) and offd(F)-offd(F) 
       * NOTE: we are working with two marker arrays here: Marker_offd and Marker_recv_SF_offd_list
       * which may have overlap.
       * So, we always check the first marker array */
      for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
      {
         j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];
         /* offd F pts */
         if (CF_marker_offd[j1] < 0)
         {
            if (!Marker_offd[j1])
            {
               FF2_offd_len ++;
               Marker_offd[j1] = 1;
            }
            /* offd(F)-offd(F), need to open recv_SF */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1+1]; k++)
            {
               /* k1: global index */
               k1 = recv_SF_j[k];
               /* if k1 is not in my range */
               if (k1 < col_start || k1 >= col_end)
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  jx_assert(recv_SF_offd_list[k2] == k1);
                  
                  /* map to offd_A */
                  k3 = Mapper_recv_SF_offd_list[k2];
                  if (k3 >= 0)
                  {
                     if (!Marker_offd[k3])
                     {
                        FF2_offd_len ++;
                        Marker_offd[k3] = 1;
                     }
                  }
                  else
                  {
                     if (!Marker_recv_SF_offd_list[k2])
                     {
                        FF2_offd_len ++;
                        Marker_recv_SF_offd_list[k2] = 1;
                     }
                  }
               }
            }
         }
      }
   }

   /* create a list of offd F, F2 points
    * and RESET the markers to ZEROs*/
   FF2_offd = jx_CTAlloc(JX_Int, FF2_offd_len);
   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (Marker_offd[i])
      {
         FF2_offd[k++] = col_map_offd_A[i];
         Marker_offd[i] = 0;
      }
   }

   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      /* debug: if mapping exists, this marker should not be set */
      if (Mapper_recv_SF_offd_list[i] >= 0)
      {
         jx_assert(Marker_recv_SF_offd_list[i] == 0);
      }

      if (Marker_recv_SF_offd_list[i])
      {
         j = recv_SF_offd_list[i];
         jx_assert(j < col_start || j >= col_end);
         FF2_offd[k++] = j;
         Marker_recv_SF_offd_list[i] = 0;
      }
   }
   jx_assert(k == FF2_offd_len);

   /* sort the list */
   jx_qsort0(FF2_offd, 0, FF2_offd_len-1);

   /* there must be no repetition in FF2_offd */
   for (i = 1; i < FF2_offd_len; i++)
   {
      jx_assert(FF2_offd[i] != FF2_offd[i-1]);
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *    Create CommPkgs for exchanging offd F and F2 rows of A
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* we will create TWO commPkg: one for row lengths and one for row data,
    * similar to what we have done above for SF_i, SF_j */
   jx_ParCSRFindExtendCommPkg(A, FF2_offd_len, FF2_offd, &comm_pkg_FF2_i);
   /* number of sends (#procs) */
   num_sends_FF2 = jx_ParCSRCommPkgNumSends(comm_pkg_FF2_i);
   /* number of rows to send */
   send_FF2_ilen = jx_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, num_sends_FF2);
   /* number of recvs (#procs) */
   num_recvs_FF2 = jx_ParCSRCommPkgNumRecvs(comm_pkg_FF2_i);
   /* number of rows to recv */
   recv_FF2_ilen = jx_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, num_recvs_FF2);

   jx_assert(FF2_offd_len == recv_FF2_ilen);

   send_FF2_i = jx_CTAlloc(JX_Int, send_FF2_ilen);
   recv_FF2_i = jx_CTAlloc(JX_Int, recv_FF2_ilen + 1);
   for (i = 0, send_FF2_jlen = 0; i < send_FF2_ilen; i++)
   {
      j = jx_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, i);
      send_FF2_i[i] = A_diag_i[j+1] - A_diag_i[j] + A_offd_i[j+1] - A_offd_i[j];
      send_FF2_jlen += send_FF2_i[i];
   }
 
   /* do communication */
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg_FF2_i, send_FF2_i, recv_FF2_i+1);
   /* ... */
   jx_ParCSRCommHandleDestroy(comm_handle);

   send_FF2_j = jx_CTAlloc(JX_Int, send_FF2_jlen);
   send_FF2_a = jx_CTAlloc(JX_Real, send_FF2_jlen);
   send_FF2_jstarts = jx_CTAlloc(JX_Int, num_sends_FF2 + 1);
   
   for (i = 0, i1 = 0; i < num_sends_FF2; i++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i);
      end   = jx_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i+1);
      for (j = start; j < end; j++)
      {
         /* will send row j1 to send_proc[i] */
         j1 = jx_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, j);
         /* open row j1 and fill ja and a */
         for (k = A_diag_i[j1]; k < A_diag_i[j1+1]; k++)
         {
            send_FF2_j[i1] = col_start + A_diag_j[k];
            send_FF2_a[i1] = A_diag_a[k];
            i1++;
         }
         if (num_procs > 1)
         {
            for (k = A_offd_i[j1]; k < A_offd_i[j1+1]; k++)
            {
               send_FF2_j[i1] = col_map_offd_A[A_offd_j[k]];
               send_FF2_a[i1] = A_offd_a[k];
               i1++;
            }
         }
      }
      send_FF2_jstarts[i+1] = i1;
   }
   jx_assert(i1 == send_FF2_jlen);
   
   /* adjust recv_FF2_i to ptrs */
   for (i = 1; i <= recv_FF2_ilen; i++)
   {
      recv_FF2_i[i] += recv_FF2_i[i-1];
   }

   recv_FF2_jlen = recv_FF2_i[recv_FF2_ilen];
   recv_FF2_j = jx_CTAlloc(JX_Int, recv_FF2_jlen);
   recv_FF2_a = jx_CTAlloc(JX_Real, recv_FF2_jlen);
   recv_FF2_jstarts = jx_CTAlloc(JX_Int, num_recvs_FF2 + 1);

   for (i = 1; i <= num_recvs_FF2; i++)
   {
      start = jx_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, i);
      recv_FF2_jstarts[i] = recv_FF2_i[start];
   }

   /* create a communication package for FF2_j */
   comm_pkg_FF2_j = jx_CTAlloc(jx_ParCSRCommPkg, 1);
   jx_ParCSRCommPkgComm         (comm_pkg_FF2_j) = comm;
   jx_ParCSRCommPkgNumSends     (comm_pkg_FF2_j) = num_sends_FF2;
   jx_ParCSRCommPkgSendProcs    (comm_pkg_FF2_j) = jx_ParCSRCommPkgSendProcs(comm_pkg_FF2_i);
   jx_ParCSRCommPkgSendMapStarts(comm_pkg_FF2_j) = send_FF2_jstarts;
   jx_ParCSRCommPkgNumRecvs     (comm_pkg_FF2_j) = num_recvs_FF2;
   jx_ParCSRCommPkgRecvProcs    (comm_pkg_FF2_j) = jx_ParCSRCommPkgRecvProcs(comm_pkg_FF2_i);
   jx_ParCSRCommPkgRecvVecStarts(comm_pkg_FF2_j) = recv_FF2_jstarts;
   
   /* do communication */
   /* ja */
   comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg_FF2_j, send_FF2_j, recv_FF2_j);
   /* ... */
   jx_ParCSRCommHandleDestroy(comm_handle);
  
   /* a */
   comm_handle = jx_ParCSRCommHandleCreate( 1, comm_pkg_FF2_j, send_FF2_a, recv_FF2_a);
   /* ... */
   jx_ParCSRCommHandleDestroy(comm_handle);

   /* A_offd_FF2 is ready ! */
   A_offd_FF2 = jx_CSRMatrixCreate(recv_FF2_ilen, jx_ParCSRMatrixGlobalNumCols(A),
                                      recv_FF2_jlen);

   jx_CSRMatrixI   (A_offd_FF2) = recv_FF2_i;
   jx_CSRMatrixJ   (A_offd_FF2) = recv_FF2_j;
   jx_CSRMatrixData(A_offd_FF2) = recv_FF2_a;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * FF2_offd contains all the offd indices and corresponds to matrix A_offd_FF2
    * So, we are able to use indices in terms of FF2_offd to bookkeeping all offd
    * information.
    * [ FF2_offd is a subset of col_map_offd_A UNION recv_SF_offd_list ]
    * Mappings from col_map_offd_A and recv_SF_offd_list will be created
    * markers for FF2_offd will also be created
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   /* Mapping from col_map_offd_A */
   Mapper_offd_A = jx_CTAlloc(JX_Int, num_cols_A_offd);
   for (i = 0; i < num_cols_A_offd; i++)
   {
      Mapper_offd_A[i] = jx_BinarySearch(FF2_offd, col_map_offd_A[i], FF2_offd_len);
   }
   
   /* Mapping from recv_SF_offd_list, overwrite the old one*/
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      Mapper_recv_SF_offd_list[i] = jx_BinarySearch(FF2_offd, recv_SF_offd_list[i], FF2_offd_len);
   }

   /* marker */
   Marker_FF2_offd = jx_CTAlloc(JX_Int, FF2_offd_len);

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *  First Pass: Determine the nnz of R and the max local size
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;
      
   for (i = 0; i < n_fine; i++)
   {
      JX_Int MARK = i + 1;

      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }
      
      /* size of the local dense problem */
      local_size = 0;

      /* i is a C-pt, increase the number of C-pts */
      n_cpts ++;

      /* diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* j1, F: D1 */
         if (Marker_diag[j1] != MARK)
         {
            Marker_diag[j1] = MARK;
            local_size ++;
            cnt_diag ++;
         }
         /* F^2: D1-D2. Open row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1+1]; k++)
         {
            k1 = S_diag_j[k];
            /* F-pt and never seen before */
            if (CF_marker[k1] < 0 && Marker_diag[k1] != MARK)
            {
               Marker_diag[k1] = MARK;
               local_size ++;
               cnt_diag ++;
            }
         }
         /* F^2: D1-O2. Open row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
         {
            k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];
            
            if (CF_marker_offd[k1] < 0)
            {
               /* map to FF2_offd */
               k2 = Mapper_offd_A[k1];

               /* this mapping must be successful */
               jx_assert(k2 >= 0 && k2 < FF2_offd_len);

               /* an F-pt and never seen before */
               if (Marker_FF2_offd[k2] != MARK)
               {
                  Marker_FF2_offd[k2] = MARK;
                  local_size ++;
                  cnt_offd ++;
               }
            }
         }
      }

      /* offd part of row i */
      for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
      {
         j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];

         if (CF_marker_offd[j1] >= 0)
         {
            continue;
         }
      
         /* map to FF2_offd */
         j2 = Mapper_offd_A[j1];

         /* this mapping must be successful */
         jx_assert(j2 >= 0 && j2 < FF2_offd_len);

         /* j1, F: O1 */
         if (Marker_FF2_offd[j2] != MARK)
         {
            Marker_FF2_offd[j2] = MARK;
            local_size ++;
            cnt_offd ++;
         }

         /* F^2: O1-D2, O1-O2 */
         /* row j1 is an external row. check recv_SF for strong F-neighbors  */
         for (k = recv_SF_i[j1]; k < recv_SF_i[j1+1]; k++)
         {
            /* k1: global index */
            k1 = recv_SF_j[k];
            /* if k1 is in the diag part */
            if (k1 >= col_start && k1 < col_end)
            {
               k3 = k1 - col_start;
               jx_assert(CF_marker[k3] < 0);
               if (Marker_diag[k3] != MARK)
               {
                  Marker_diag[k3] = MARK;
                  local_size ++;
                  cnt_diag ++;
               }
            }
            else /* k1 is in the offd part */
            {
               /* index in recv_SF_offd_list */
               k2 = recv_SF_j2[k];

               jx_assert(recv_SF_offd_list[k2] == k1);

               /* map to FF2_offd */
               k3 = Mapper_recv_SF_offd_list[k2];
          
               /* this mapping must be successful */
               jx_assert(k3 >= 0 && k3 < FF2_offd_len);
               
               if (Marker_FF2_offd[k3] != MARK)
               {
                  Marker_FF2_offd[k3] = MARK;
                  local_size ++;
                  cnt_offd ++;
               }
            }
         }
      }

      /* keep ths max size */
      local_max_size = jx_max(local_max_size, local_size);
   } /* for (i=0,...) */
 
   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;
 
   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;
 
   /*------------- allocate arrays */
   R_diag_i    = jx_CTAlloc(JX_Int,  n_cpts+1);
   R_diag_j    = jx_CTAlloc(JX_Int,  nnz_diag);
   R_diag_data = jx_CTAlloc(JX_Real, nnz_diag);

   /* not in ``if num_procs > 1'', 
    * allocation needed even for empty CSR */
   R_offd_i    = jx_CTAlloc(JX_Int,  n_cpts+1);
   R_offd_j    = jx_CTAlloc(JX_Int,  nnz_offd);
   R_offd_data = jx_CTAlloc(JX_Real, nnz_offd);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   /* RESET marker arrays */
   for (i = 0; i < n_fine; i++)
   {
      Marker_diag[i] = -1;
   }
   Marker_diag_j = jx_CTAlloc(JX_Int, n_fine);

   for (i = 0; i < FF2_offd_len; i++)
   {
      Marker_FF2_offd[i] = -1;
   }
   Marker_FF2_offd_j = jx_CTAlloc(JX_Int, FF2_offd_len);

   //for (i = 0; i < num_cols_A_offd; i++)
   //{
   //   Marker_offd[i] = -1;
   //}
   //for (i = 0; i < recv_SF_offd_list_len; i++)
   //{
   //   Marker_recv_SF_list[i] = -1;
   //}

   /* the local matrix and rhs (dense) 
    * column-major as always by BLAS/LAPACK */
   /* matrix */
   DAi = jx_CTAlloc(JX_Real, local_max_size * local_max_size);
   /* rhs */
   Dbi = jx_CTAlloc(JX_Real, local_max_size);
   /* pivot */
   Ipi = jx_CTAlloc(JX_Int, local_max_size);
   /*- - - - - - - - - - - - - - - - - - - - - - - - - 
    * space to save row indices of the local problem,
    * if diag, save the local indices,
    * if offd, save the indices in FF2_offd, 
    *          since we will use it to access A_offd_FF2
    *- - - - - - - - - - - - - - - - - - - - - - - - - */
   RRi = jx_CTAlloc(JX_Int, local_max_size);
   /* indicators for RRi of being local (0) or offd (1) */
   KKi = jx_CTAlloc(JX_Int, local_max_size);

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *                        Second Pass: Populate R
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      Marker_diag_count = 0;
      Marker_FF2_offd_count = 0;

      /* size of Ai, bi */
      local_size = 0;
       
      /* Access matrices for the First time, mark the points we want */
      /* diag part of row i */
//JX_Real t1 = jx_MPI_Wtime();
      for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* j1, F: D1 */
         if (Marker_diag[j1] == -1)
         {
            RRi[local_size] = j1;
            KKi[local_size] = 0;
            Marker_diag_j[Marker_diag_count++] = j1;
            Marker_diag[j1] = local_size ++;
         }
         /* F^2: D1-D2. Open row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1+1]; k++)
         {
            k1 = S_diag_j[k];
            /* F-pt and never seen before */
            if (CF_marker[k1] < 0 && Marker_diag[k1] == -1)
            {
               RRi[local_size] = k1;
               KKi[local_size] = 0;
               Marker_diag_j[Marker_diag_count++] = k1;
               Marker_diag[k1] = local_size ++;
            }
         }
         /* F^2: D1-O2. Open row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1+1]; k++)
         {
            k1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[k]] : S_offd_j[k];

            if (CF_marker_offd[k1] < 0)
            {
               /* map to FF2_offd */
               k2 = Mapper_offd_A[k1];
               
               /* this mapping must be successful */
               jx_assert(k2 >= 0 && k2 < FF2_offd_len);
               
               /* an F-pt and never seen before */
               if (Marker_FF2_offd[k2] == -1)
               {
                  /* NOTE: we save this mapped index */
                  RRi[local_size] = k2;
                  KKi[local_size] = 1;
                  Marker_FF2_offd_j[Marker_FF2_offd_count++] = k2;
                  Marker_FF2_offd[k2] = local_size ++;
               }
            }
         }
      }

      /* offd part of row i */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            j1 = col_offd_S_to_A ? col_offd_S_to_A[S_offd_j[j]] : S_offd_j[j];

            if (CF_marker_offd[j1] >= 0)
            {
               continue;
            }

            /* map to FF2_offd */
            j2 = Mapper_offd_A[j1];

            /* this mapping must be successful */
            jx_assert(j2 >= 0 && j2 < FF2_offd_len);

            /* j1, F: O1 */
            if (Marker_FF2_offd[j2] == -1)
            {
               /* NOTE: we save this mapped index */
               RRi[local_size] = j2;
               KKi[local_size] = 1;
               Marker_FF2_offd_j[Marker_FF2_offd_count++] = j2;
               Marker_FF2_offd[j2] = local_size ++;
            }

            /* F^2: O1-D2, O1-O2 */
            /* row j1 is an external row. check recv_SF for strong F-neighbors  */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1+1]; k++)
            {
               /* k1: global index */
               k1 = recv_SF_j[k];
               /* if k1 is in the diag part */
               if (k1 >= col_start && k1 < col_end)
               {
                  k3 = k1 - col_start;

                  jx_assert(CF_marker[k3] < 0);

                  if (Marker_diag[k3] == -1)
                  {
                     RRi[local_size] = k3;
                     KKi[local_size] = 0;
                     Marker_diag_j[Marker_diag_count++] = k3;
                     Marker_diag[k3] = local_size ++;
                  }
               }
               else /* k1 is in the offd part */
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  jx_assert(recv_SF_offd_list[k2] == k1);

                  /* map to FF2_offd */
                  k3 = Mapper_recv_SF_offd_list[k2];

                  /* this mapping must be successful */
                  jx_assert(k3 >= 0 && k3 < FF2_offd_len);

                  if (Marker_FF2_offd[k3] == -1)
                  {
                     /* NOTE: we save this mapped index */
                     RRi[local_size] = k3;
                     KKi[local_size] = 1;
                     Marker_FF2_offd_j[Marker_FF2_offd_count++] = k3;
                     Marker_FF2_offd[k3] = local_size ++;
                  }
               }
            }
         }
      }

//t1 = jx_MPI_Wtime() - t1;
//time1 += t1;

      jx_assert(local_size <= local_max_size);

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries 
       * we need from these rows and put them in Ai and bi*/

//JX_Real t2 = jx_MPI_Wtime();
      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(JX_Real));
      memset(Dbi, 0, local_size * sizeof(JX_Real));

      /* we will populate Ai row-by-row */
      for (rr = 0; rr < local_size; rr++)
      {
         /* row index */
         i1 = RRi[rr];
         /* diag-offd indicator */
         i2 = KKi[rr];

         if (i2)  /* i2 == 1, i1 is an offd row */
         {
            /* open row i1, a remote row */
            for (j = jx_CSRMatrixI(A_offd_FF2)[i1]; j < jx_CSRMatrixI(A_offd_FF2)[i1+1]; j++)
            {
               /* j1 is a global index */
               j1 = jx_CSRMatrixJ(A_offd_FF2)[j];

               /* if j1 is in the diag part */
               if (j1 >= col_start && j1 < col_end)
               {
                  j2 = j1 - col_start;
                  /* if this col is marked with its local dense id */
                  if ((cc = Marker_diag[j2]) >= 0)
                  {
                     jx_assert(CF_marker[j2] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     DAi[rr + cc * local_size] = jx_CSRMatrixData(A_offd_FF2)[j];
                  }
               }
               else
               {
                  /* j1 is in offd part, search it in FF2_offd */
                  j2 =  jx_BinarySearch(FF2_offd, j1, FF2_offd_len);
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = jx_CSRMatrixData(A_offd_FF2)[j];
                     }
                  }
               }
            }
         }
         else /* i2 == 0, i1 is a local row */
         {
            /* open row i1, a local row */
            for (j = A_diag_i[i1]; j < A_diag_i[i1+1]; j++)
            {
               /* j1 is a local index */
               j1 = A_diag_j[j];
               /* if this col is marked with its local dense id */
               if ((cc = Marker_diag[j1]) >= 0)
               {
                  jx_assert(CF_marker[j1] < 0);
                  
                  /* copy the value */
                  /* rr and cc: local dense ids */
                  DAi[rr + cc * local_size] = A_diag_a[j];
               }
            }

            if (num_procs > 1)
            {
               for (j = A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
               {
                  j1 = A_offd_j[j];
                  /* map to FF2_offd */
                  j2 = Mapper_offd_A[j1];
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        jx_assert(CF_marker_offd[j1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = A_offd_a[j];
                     }
                  }
               }
            }
         }
      }

      /* rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = Marker_diag[i1]) >= 0)
         {
            jx_assert(i1 == RRi[cc] && KKi[cc] == 0);
            /* Note the sign change */
            Dbi[cc] = -A_diag_a[j];
            rr++;
         }
      }

      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            i1 = A_offd_j[j];
            i2 = Mapper_offd_A[i1];
            if (i2 > -1)
            {
               if ((cc = Marker_FF2_offd[i2]) >= 0)
               {
                  jx_assert(i2 == RRi[cc] && KKi[cc] == 1);
                  /* Note the sign change */
                  Dbi[cc] = -A_offd_a[j];
                  rr++;
               }
            }
         }
      }

      jx_assert(rr <= local_size);

//t2 = jx_MPI_Wtime() - t2;
//time2 += t2;

      if (local_size > 0)
      {
         /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          * we have Ai and bi build 
          * Solve the linear system by LAPACK : LU factorization
          *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
         jx_dgetrf(&local_size, &local_size, DAi, &local_size, Ipi, &lapack_info);

         jx_assert(lapack_info == 0);

         if (lapack_info == 0)
         {
            /* solve A_i^T x_i = b_i,
             * solution is saved in b_i on return */
            jx_dgetrs(&charT, &local_size, &ione, DAi, &local_size, Ipi, Dbi, &local_size, &lapack_info);
            jx_assert(lapack_info == 0);
         }
      }

      /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       * Now we are ready to fill this row of R
       *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
      for (rr = 0; rr < local_size; rr++)
      {
         /* row index */
         i1 = RRi[rr];
         /* diag-offd indicator */
         i2 = KKi[rr];

         if (i2) /* offd */
         {
            jx_assert(Marker_FF2_offd[i1] == rr);

            /* col idx: use the index in FF2_offd,
             * and you will see why later (very soon!) */
            R_offd_j[cnt_offd] = i1;
            /* copy the value */
            R_offd_data[cnt_offd++] = Dbi[rr];
         }
         else /* diag */
         {
            jx_assert(Marker_diag[i1] == rr);

            /* col idx: use local index i1 */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Dbi[rr];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i'' */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;

      /* row ptr of the next row */
      R_diag_i[ic+1] = cnt_diag;

      R_offd_i[ic+1] = cnt_offd;

//JX_Real t3 = jx_MPI_Wtime();
      /* RESET marker arrays */
      for (j = 0; j < Marker_diag_count; j++)
      {
         Marker_diag[Marker_diag_j[j]] = -1;
      }

      for (j = 0; j < Marker_FF2_offd_count; j++)
      {
         Marker_FF2_offd[Marker_FF2_offd_j[j]] = -1;
      }

      /* never turn them on !!! */
      /*
      for (j = 0; j < n_fine; j++)
      {
         jx_assert(Marker_diag[j] == -1);
      }

      for (j = 0; j < FF2_offd_len; j++)
      {
         jx_assert(Marker_FF2_offd[j] == -1);
      }
      */

//t3 = jx_MPI_Wtime() - t3;
//time3 += t3;

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */


   jx_assert(ic == n_cpts)
   jx_assert(cnt_diag == nnz_diag)
   jx_assert(cnt_offd == nnz_offd)
   
   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, Marker_FF2_offd should be all -1 */
   /*
   for (i = 0; i < FF2_offd_len; i++)
   {
      jx_assert(Marker_FF2_offd[i] == - 1);
   }
   */

   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (Marker_FF2_offd[i1] == -1)
      {
         num_cols_offd_R++;
         Marker_FF2_offd[i1] = 1;
      }
   }

   col_map_offd_R = jx_CTAlloc(JX_Int, num_cols_offd_R);
   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the local indices in FF2_offd [will be changed] */
   for (i = 0, i1 = 0; i < FF2_offd_len; i++)
   {
      if (Marker_FF2_offd[i] == 1)
      {
         col_map_offd_R[i1++] = i;
      }
   }

   jx_assert(i1 == num_cols_offd_R);
   //printf("FF2_offd_len %d, num_cols_offd_R %d\n", FF2_offd_len, num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t FF2_offd
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = jx_BinarySearch(col_map_offd_R, i1, num_cols_offd_R);
      /* searching must succeed */
      jx_assert(k1 >= 0 && k1 < num_cols_offd_R);
      /* change index */
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids [guaranteed to be sorted] */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = FF2_offd[col_map_offd_R[i]];
   }

      /* Now, we should have everything of Parcsr matrix R */
   R = jx_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                jx_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                jx_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = jx_ParCSRMatrixDiag(R);
   jx_CSRMatrixData(R_diag) = R_diag_data;
   jx_CSRMatrixI(R_diag)    = R_diag_i;
   jx_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = jx_ParCSRMatrixOffd(R);
   jx_CSRMatrixData(R_offd) = R_offd_data;
   jx_CSRMatrixI(R_offd)    = R_offd_i;
   jx_CSRMatrixJ(R_offd)    = R_offd_j;
   /* R does not own ColStarts, since A does */
   jx_ParCSRMatrixOwnsColStarts(R) = 0;
   
   jx_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   jx_MatvecCommPkgCreate(R);

  *R_ptr = R;
  *R_max_size_ptr = local_max_size;

   jx_TFree(CF_marker_offd);
   jx_TFree(dof_func_offd);
   jx_TFree(Marker_diag);
   jx_TFree(Marker_offd);
   jx_TFree(send_buf_i);
   jx_TFree(send_SF_i);
   jx_TFree(recv_SF_i);
   jx_TFree(send_SF_j);
   jx_TFree(send_SF_jstarts);
   jx_TFree(recv_SF_j);
   jx_TFree(recv_SF_jstarts);
   jx_TFree(comm_pkg_SF);
   jx_TFree(recv_SF_offd_list);
   jx_TFree(recv_SF_j2);
   jx_TFree(Mapper_recv_SF_offd_list);
   jx_TFree(Marker_recv_SF_offd_list);
   jx_TFree(FF2_offd);
   jx_TFree(send_FF2_i);
   /* jx_TFree(recv_FF2_i); */
   jx_TFree(send_FF2_j);
   jx_TFree(send_FF2_a);
   jx_TFree(send_FF2_jstarts);
   /* jx_TFree(recv_FF2_j); */
   /* jx_TFree(recv_FF2_a); */
   jx_CSRMatrixDestroy(A_offd_FF2);
   jx_TFree(recv_FF2_jstarts);
   jx_MatvecCommPkgDestroy(comm_pkg_FF2_i);
   jx_TFree(comm_pkg_FF2_j);
   jx_TFree(Mapper_offd_A);
   jx_TFree(Marker_FF2_offd);
   jx_TFree(Marker_diag_j);
   jx_TFree(Marker_FF2_offd_j);
   jx_TFree(DAi);
   jx_TFree(Dbi);
   jx_TFree(Ipi);
   jx_TFree(RRi);
   jx_TFree(KKi);

   return 0;
}
