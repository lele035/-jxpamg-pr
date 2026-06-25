//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_csr_matop.c -- basic operations for parallel matrix.
 *  Date: 2011/09/05
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_ParCSRMatrixCopy
 * \brief Copies A to B.
 * \note If copy_data = 0, only the structure of A is copied to B.
 *       This routine does not check whether the dimensions of
 *       A and B are compatible.
 * \date 2011/09/08
 */
JX_Int 
jx_ParCSRMatrixCopy( jx_ParCSRMatrix *A, 
                     jx_ParCSRMatrix *B, 
                     JX_Int              copy_data )
{
   jx_CSRMatrix *A_diag;
   jx_CSRMatrix *A_offd;
   JX_Int *col_map_offd_A;
   jx_CSRMatrix *B_diag;
   jx_CSRMatrix *B_offd;
   JX_Int *col_map_offd_B;
   JX_Int num_cols_offd;
   JX_Int i;

   if (!A)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (!B)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   A_diag = jx_ParCSRMatrixDiag(A);
   A_offd = jx_ParCSRMatrixOffd(A);
   col_map_offd_A = jx_ParCSRMatrixColMapOffd(A);
   B_diag = jx_ParCSRMatrixDiag(B);
   B_offd = jx_ParCSRMatrixOffd(B);
   col_map_offd_B = jx_ParCSRMatrixColMapOffd(B);
   num_cols_offd = jx_CSRMatrixNumCols(A_offd);

   jx_CSRMatrixCopy(A_diag, B_diag, copy_data);
   jx_CSRMatrixCopy(A_offd, B_offd, copy_data);
   if (num_cols_offd && col_map_offd_B == NULL)
   {
      col_map_offd_B = jx_CTAlloc(JX_Int,num_cols_offd);
      jx_ParCSRMatrixColMapOffd(B) = col_map_offd_B;
   }
   for (i = 0; i < num_cols_offd; i ++)
   {
      col_map_offd_B[i] = col_map_offd_A[i];
   }
        
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixTranspose
 * \brief Transpose a parallel CSR matrix.
 * \param *A pointer to the matrix to be transposed.
 * \param **AT_ptr pointer to pointer to the transposed matrix.
 * \param data flag to indicate whether transpose the data part
 *        data = 0: Only transpose the sparse pattern;
 *        data = 1: Transpose both the sparse pattern and the data part.
 * \date 2011/09/05
 */
JX_Int
jx_ParCSRMatrixTranspose( jx_ParCSRMatrix  *A,
                          jx_ParCSRMatrix **AT_ptr,
                          JX_Int               data ) 
{
   jx_ParCSRCommHandle *comm_handle = NULL;
   MPI_Comm             comm = jx_ParCSRMatrixComm(A);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(A);
   jx_CSRMatrix        *A_diag   = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix        *A_offd   = jx_ParCSRMatrixOffd(A);
   JX_Int  num_cols = jx_ParCSRMatrixNumCols(A);
   JX_Int  first_row_index = jx_ParCSRMatrixFirstRowIndex(A);
   JX_Int *row_starts = jx_ParCSRMatrixRowStarts(A);
   JX_Int *col_starts = jx_ParCSRMatrixColStarts(A);

   JX_Int	      num_cols_offd = jx_CSRMatrixNumCols(A_offd);
   JX_Int        ierr = 0;
   JX_Int	      num_sends = 0, num_recvs = 0, num_cols_offd_AT; 
   JX_Int	      i, j, k, index, counter, j_row;
   JX_Int        value;

   jx_ParCSRMatrix *AT;
   jx_CSRMatrix    *AT_diag;
   jx_CSRMatrix    *AT_offd;
   jx_CSRMatrix    *AT_tmp;
   
   JX_Int first_row_index_AT, first_col_diag_AT;
   JX_Int local_num_rows_AT, local_num_cols_AT;
   
   JX_Int    *AT_tmp_i    = NULL;
   JX_Int    *AT_tmp_j    = NULL;
   JX_Real *AT_tmp_data = NULL;

   JX_Int    *AT_buf_i    = NULL;
   JX_Int    *AT_buf_j    = NULL;
   JX_Real *AT_buf_data = NULL;

   JX_Int    *AT_offd_i;
   JX_Int    *AT_offd_j;
   JX_Real *AT_offd_data;
   JX_Int    *col_map_offd_AT;
   JX_Int    *row_starts_AT;
   JX_Int    *col_starts_AT;

   JX_Int num_procs, my_id;

   JX_Int *recv_procs      = NULL;
   JX_Int *send_procs      = NULL;
   JX_Int *recv_vec_starts = NULL;
   JX_Int *send_map_starts = NULL;
   JX_Int *send_map_elmts  = NULL;
   JX_Int *tmp_recv_vec_starts;
   JX_Int *tmp_send_map_starts;
   
   jx_ParCSRCommPkg *tmp_comm_pkg;

   jx_MPI_Comm_size(comm, &num_procs);   
   jx_MPI_Comm_rank(comm, &my_id);
  
   num_cols_offd_AT = 0;
   counter = 0;
   AT_offd_j = NULL;
   AT_offd_data = NULL;
   col_map_offd_AT = NULL;
 
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
    
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }

   if (num_procs > 1)
   {
      jx_CSRMatrixTranspose (A_offd, &AT_tmp, data);

      AT_tmp_i = jx_CSRMatrixI(AT_tmp);
      AT_tmp_j = jx_CSRMatrixJ(AT_tmp);
      if (data) AT_tmp_data = jx_CSRMatrixData(AT_tmp);

      num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = jx_ParCSRCommPkgRecvProcs(comm_pkg);
      send_procs = jx_ParCSRCommPkgSendProcs(comm_pkg);
      recv_vec_starts = jx_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_map_starts = jx_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = jx_ParCSRCommPkgSendMapElmts(comm_pkg);

      AT_buf_i = jx_CTAlloc(JX_Int,send_map_starts[num_sends]); 

      for (i = 0; i < AT_tmp_i[num_cols_offd]; i ++)
      {
         AT_tmp_j[i] += first_row_index;
      }

      for (i = 0; i < num_cols_offd; i ++)
      {
         AT_tmp_i[i] = AT_tmp_i[i+1] - AT_tmp_i[i];
      }
	
      comm_handle = jx_ParCSRCommHandleCreate(12, comm_pkg, AT_tmp_i, AT_buf_i);
   }

   jx_CSRMatrixTranspose(A_diag, &AT_diag, data);

   AT_offd_i = jx_CTAlloc(JX_Int, num_cols+1);

   if (num_procs > 1)
   {   
      jx_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      tmp_send_map_starts = jx_CTAlloc(JX_Int, num_sends+1);
      tmp_recv_vec_starts = jx_CTAlloc(JX_Int, num_recvs+1);

      tmp_send_map_starts[0] = send_map_starts[0];
      for (i = 0; i < num_sends; i ++)
      {
	 tmp_send_map_starts[i+1] = tmp_send_map_starts[i];
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
 	 {
	    tmp_send_map_starts[i+1] += AT_buf_i[j];
	    AT_offd_i[send_map_elmts[j]+1] += AT_buf_i[j];
	 }
      }
      for (i = 0; i < num_cols; i ++)
      {
         AT_offd_i[i+1] += AT_offd_i[i];
      }

      tmp_recv_vec_starts[0] = recv_vec_starts[0];
      for (i = 0; i < num_recvs; i ++)
      {
	 tmp_recv_vec_starts[i+1] = tmp_recv_vec_starts[i];
         for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j ++)
         {
            tmp_recv_vec_starts[i+1] += AT_tmp_i[j];
         }
      }

      tmp_comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
      jx_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
      jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
      jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
      jx_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = recv_procs;
      jx_ParCSRCommPkgSendProcs(tmp_comm_pkg) = send_procs;
      jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = tmp_recv_vec_starts;
      jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = tmp_send_map_starts;

      AT_buf_j = jx_CTAlloc(JX_Int,tmp_send_map_starts[num_sends]);
      comm_handle = jx_ParCSRCommHandleCreate(12, tmp_comm_pkg, AT_tmp_j, AT_buf_j);
      jx_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      if (data)
      {
         AT_buf_data = jx_CTAlloc(JX_Real,tmp_send_map_starts[num_sends]);
         comm_handle = jx_ParCSRCommHandleCreate(2,tmp_comm_pkg,AT_tmp_data,AT_buf_data);
         jx_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
      }

      jx_TFree(tmp_recv_vec_starts);
      jx_TFree(tmp_send_map_starts);
      jx_TFree(tmp_comm_pkg);
      jx_CSRMatrixDestroy(AT_tmp);

      if (AT_offd_i[num_cols])
      {
         AT_offd_j = jx_CTAlloc(JX_Int, AT_offd_i[num_cols]);
         if (data) AT_offd_data = jx_CTAlloc(JX_Real, AT_offd_i[num_cols]);
      }
      else
      {
         AT_offd_j = NULL;
         AT_offd_data = NULL;
      }
	 
      counter = 0;
      for (i = 0; i < num_sends; i ++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
	 {
	    j_row = send_map_elmts[j];
	    index = AT_offd_i[j_row];
	    for (k = 0; k < AT_buf_i[j]; k ++)
	    {
	       if (data) AT_offd_data[index] = AT_buf_data[counter];
	       AT_offd_j[index++] = AT_buf_j[counter++];
	    }
	    AT_offd_i[j_row] = index;
	 }
      }
      for (i = num_cols; i > 0; i --)
      {
         AT_offd_i[i] = AT_offd_i[i-1];
      }
      AT_offd_i[0] = 0;

      if (counter)
      {
         jx_qsort0(AT_buf_j, 0, counter - 1);
         num_cols_offd_AT = 1;
	 value = AT_buf_j[0];
         for (i = 1; i < counter; i ++)
	 {
	    if (value < AT_buf_j[i])
	    {
	       AT_buf_j[num_cols_offd_AT++] = AT_buf_j[i];
	       value = AT_buf_j[i];
	    }
	 }
      }

      if (num_cols_offd_AT)
      {
         col_map_offd_AT = jx_CTAlloc(JX_Int, num_cols_offd_AT);
      }
      else
      {
         col_map_offd_AT = NULL;
      }
      
      for (i = 0; i < num_cols_offd_AT; i ++)
      {
	 col_map_offd_AT[i] = AT_buf_j[i];
      }

      jx_TFree(AT_buf_i);
      jx_TFree(AT_buf_j);
      if (data) jx_TFree(AT_buf_data);

      for (i = 0; i < counter; i ++)
      {
	 AT_offd_j[i] = jx_BinarySearch(col_map_offd_AT, AT_offd_j[i], num_cols_offd_AT);
      }
   }

   AT_offd = jx_CSRMatrixCreate(num_cols,num_cols_offd_AT,counter);
   jx_CSRMatrixI(AT_offd) = AT_offd_i;
   jx_CSRMatrixJ(AT_offd) = AT_offd_j;
   jx_CSRMatrixData(AT_offd) = AT_offd_data;
   

#ifdef JX_NO_GLOBAL_PARTITION

   row_starts_AT = jx_CTAlloc(JX_Int, 2);
   for (i = 0; i < 2; i ++)
   {
      row_starts_AT[i] = col_starts[i];
   }   

   if (row_starts != col_starts)
   {
      col_starts_AT = jx_CTAlloc(JX_Int, 2);
      for (i = 0; i < 2; i++)
      {
         col_starts_AT[i] = row_starts[i];
      }
   }
   else
   {
      col_starts_AT = row_starts_AT;
   }

   first_row_index_AT =  row_starts_AT[0];
   first_col_diag_AT  =  col_starts_AT[0];

   local_num_rows_AT = row_starts_AT[1] - first_row_index_AT ;
   local_num_cols_AT = col_starts_AT[1] - first_col_diag_AT;

#else

   row_starts_AT = jx_CTAlloc(JX_Int, num_procs+1);
   for (i = 0; i < num_procs + 1; i ++)
   {
      row_starts_AT[i] = col_starts[i];
   }

   if (row_starts != col_starts)
   {
      col_starts_AT = jx_CTAlloc(JX_Int, num_procs + 1);
      for (i = 0; i < num_procs+1; i ++)
      {
         col_starts_AT[i] = row_starts[i];
      }
   }
   else
   {
      col_starts_AT = row_starts_AT;
   }
   first_row_index_AT = row_starts_AT[my_id];
   first_col_diag_AT  = col_starts_AT[my_id];

   local_num_rows_AT = row_starts_AT[my_id+1] - first_row_index_AT ;
   local_num_cols_AT = col_starts_AT[my_id+1] - first_col_diag_AT;

#endif


   AT = jx_CTAlloc(jx_ParCSRMatrix,1);
   jx_ParCSRMatrixComm(AT) = comm;
   jx_ParCSRMatrixDiag(AT) = AT_diag;
   jx_ParCSRMatrixOffd(AT) = AT_offd;
   jx_ParCSRMatrixGlobalNumRows(AT) = jx_ParCSRMatrixGlobalNumCols(A);
   jx_ParCSRMatrixGlobalNumCols(AT) = jx_ParCSRMatrixGlobalNumRows(A);
   jx_ParCSRMatrixRowStarts(AT)  = row_starts_AT;
   jx_ParCSRMatrixColStarts(AT)  = col_starts_AT;
   jx_ParCSRMatrixColMapOffd(AT) = col_map_offd_AT;
 
   jx_ParCSRMatrixFirstRowIndex(AT) = first_row_index_AT;
   jx_ParCSRMatrixFirstColDiag(AT)  = first_col_diag_AT;

   jx_ParCSRMatrixLastRowIndex(AT) = first_row_index_AT + local_num_rows_AT - 1;
   jx_ParCSRMatrixLastColDiag(AT)  = first_col_diag_AT  + local_num_cols_AT - 1;

   jx_ParCSRMatrixOwnsData(AT) = 1;
   jx_ParCSRMatrixOwnsRowStarts(AT) = 1;
   jx_ParCSRMatrixOwnsColStarts(AT) = 1;
   if (row_starts_AT == col_starts_AT)
   {
      jx_ParCSRMatrixOwnsColStarts(AT) = 0;
   }

   jx_ParCSRMatrixCommPkg(AT) = NULL;
   jx_ParCSRMatrixCommPkgT(AT) = NULL;

   jx_ParCSRMatrixRowindices(AT) = NULL;
   jx_ParCSRMatrixRowvalues(AT)  = NULL;
   jx_ParCSRMatrixGetrowactive(AT) = 0;

   *AT_ptr = AT;
  
   return ierr;
}

/*!
 * \fn jx_CSRMatrix *jx_ParCSRMatrixExtractBExt
 * \brief Extracts rows from B which are located on other processors
 *        and needed for multiplication with A locally. The rows
 *        are returned as CSRMatrix.
 * \date 2011/09/05
 */ 
jx_CSRMatrix * 
jx_ParCSRMatrixExtractBExt( jx_ParCSRMatrix *B, 
                            jx_ParCSRMatrix *A, 
                            JX_Int              data )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(B);
   JX_Int first_col_diag = jx_ParCSRMatrixFirstColDiag(B);
   JX_Int first_row_index = jx_ParCSRMatrixFirstRowIndex(B);
   JX_Int *col_map_offd = jx_ParCSRMatrixColMapOffd(B);

   jx_ParCSRCommPkg *comm_pkg = jx_ParCSRMatrixCommPkg(A);
   JX_Int  num_recvs;
   JX_Int *recv_vec_starts;
   JX_Int  num_sends;
   JX_Int *send_map_starts;
   JX_Int *send_map_elmts;
 
   jx_CSRMatrix *diag = jx_ParCSRMatrixDiag(B);

   JX_Int    *diag_i    = jx_CSRMatrixI(diag);
   JX_Int    *diag_j    = jx_CSRMatrixJ(diag);
   JX_Real *diag_data = jx_CSRMatrixData(diag);

   jx_CSRMatrix *offd = jx_ParCSRMatrixOffd(B);

   JX_Int    *offd_i    = jx_CSRMatrixI(offd);
   JX_Int    *offd_j    = jx_CSRMatrixJ(offd);
   JX_Real *offd_data = jx_CSRMatrixData(offd);

   JX_Int num_cols_B, num_nonzeros;
   JX_Int num_rows_B_ext;

   jx_CSRMatrix *B_ext;

   JX_Int    *B_ext_i;
   JX_Int    *B_ext_j;
   JX_Real *B_ext_data;
   JX_Int    *idummy;

  /*---------------------------------------------------------------------
   * If there exists no CommPkg for A, a CommPkg is generated using
   * equally load balanced partitionings 
   *--------------------------------------------------------------------*/
   
   if (!jx_ParCSRMatrixCommPkg(A))
   {
      jx_MatvecCommPkgCreate(A);
   }
    
   comm_pkg = jx_ParCSRMatrixCommPkg(A);
   num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = jx_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   send_map_starts = jx_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elmts = jx_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   num_cols_B = jx_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   jx_ParCSRMatrixExtractBExt_Arrays( &B_ext_i, &B_ext_j, &B_ext_data, &idummy,
                                      &num_nonzeros,
                                      data, 0, comm, comm_pkg,
                                      num_cols_B, num_recvs, num_sends,
                                      first_col_diag, first_row_index,
                                      recv_vec_starts, send_map_starts, send_map_elmts,
                                      diag_i, diag_j, offd_i, offd_j, col_map_offd,
                                      diag_data, offd_data );

   B_ext = jx_CSRMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   jx_CSRMatrixI(B_ext) = B_ext_i;
   jx_CSRMatrixJ(B_ext) = B_ext_j;
   if (data) jx_CSRMatrixData(B_ext) = B_ext_data;

   return B_ext;
}

/*!
 * \fn void jx_ParCSRMatrixExtractBExt_Arrays
 * \brief This function was formerly part of jx_ParCSRMatrixExtractBExt
 *        but the code was removed so it can be used for a corresponding function
 *        for Boolean matrices.
 * \date 2011/09/05
 */ 
void 
jx_ParCSRMatrixExtractBExt_Arrays( JX_Int              **pB_ext_i, 
                                   JX_Int              **pB_ext_j, 
                                   JX_Real           **pB_ext_data, 
                                   JX_Int              **pB_ext_row_map,
                                   JX_Int               *num_nonzeros,
                                   JX_Int                data, 
                                   JX_Int                find_row_map, 
                                   MPI_Comm           comm, 
                                   jx_ParCSRCommPkg  *comm_pkg,
                                   JX_Int                num_cols_B, 
                                   JX_Int                num_recvs, 
                                   JX_Int                num_sends,
                                   JX_Int                first_col_diag, 
                                   JX_Int                first_row_index,
                                   JX_Int               *recv_vec_starts, 
                                   JX_Int               *send_map_starts, 
                                   JX_Int               *send_map_elmts,
                                   JX_Int               *diag_i, 
                                   JX_Int               *diag_j, 
                                   JX_Int               *offd_i, 
                                   JX_Int               *offd_j, 
                                   JX_Int               *col_map_offd,
                                   JX_Real            *diag_data, 
                                   JX_Real            *offd_data )
{
   jx_ParCSRCommHandle *comm_handle;
   jx_ParCSRCommPkg    *tmp_comm_pkg;
   JX_Int *B_int_i;
   JX_Int *B_int_j;
   JX_Int *B_ext_i;
   JX_Int *B_ext_j;
   JX_Real *B_ext_data = NULL;
   JX_Real *B_int_data = NULL;
   JX_Int *B_int_row_map = NULL;
   JX_Int *B_ext_row_map = NULL;
   JX_Int num_procs, my_id;
   JX_Int *jdata_recv_vec_starts;
   JX_Int *jdata_send_map_starts;
 
   JX_Int i, j, k, counter;
   JX_Int start_index;
   JX_Int j_cnt, j_cnt_rm, jrow;
   JX_Int num_rows_B_ext;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   num_rows_B_ext = recv_vec_starts[num_recvs];
   
   if (num_rows_B_ext < 0)  /* no B_ext, no communication */
   {  
      *pB_ext_i = NULL;
      *pB_ext_j = NULL;
      if ( data ) *pB_ext_data = NULL;
      if ( find_row_map ) *pB_ext_row_map = NULL;
      *num_nonzeros = 0;
      return;
   }
   
   B_int_i = jx_CTAlloc(JX_Int, send_map_starts[num_sends]+1);
   B_ext_i = jx_CTAlloc(JX_Int, num_rows_B_ext+1);
   *pB_ext_i = B_ext_i;
   
   if (find_row_map) 
   {
      B_int_row_map = jx_CTAlloc(JX_Int, send_map_starts[num_sends]+1);
      B_ext_row_map = jx_CTAlloc(JX_Int, num_rows_B_ext+1);
      *pB_ext_row_map = B_ext_row_map;
   }

  /*--------------------------------------------------------------------------
   * generate B_int_i through adding number of row-elements of offd and diag
   * for corresponding rows. B_int_i[j+1] contains the number of elements of
   * a row j (which is determined through send_map_elmts) 
   *--------------------------------------------------------------------------*/
   
   B_int_i[0] = 0;
   j_cnt = 0;
   j_cnt_rm = 0;
   *num_nonzeros = 0;
   for (i = 0; i < num_sends; i ++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
      {
         jrow = send_map_elmts[j];
         B_int_i[++j_cnt] = offd_i[jrow+1] - offd_i[jrow] + diag_i[jrow+1] - diag_i[jrow];
	 *num_nonzeros += B_int_i[j_cnt];
      }
      if (find_row_map) 
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++) 
         {
            jrow = send_map_elmts[j];
            B_int_row_map[j_cnt_rm++] = jrow + first_row_index;
         }
      }
   }

  /*---------------------------------------------------------------------------------
   * initialize communication 
   *--------------------------------------------------------------------------------*/
   comm_handle = jx_ParCSRCommHandleCreate( 11,comm_pkg, &B_int_i[1],&(B_ext_i[1]) );
   if ( find_row_map ) 
   {
      /* scatter/gather B_int row numbers to form array of B_ext row numbers */
      jx_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, B_int_row_map, B_ext_row_map);
   }

   B_int_j = jx_CTAlloc(JX_Int, *num_nonzeros);
   if (data) B_int_data = jx_CTAlloc(JX_Real, *num_nonzeros);

   jdata_send_map_starts = jx_CTAlloc(JX_Int, num_sends+1);
   jdata_recv_vec_starts = jx_CTAlloc(JX_Int, num_recvs+1);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
   counter = 0;
   for (i = 0; i < num_sends; i ++)
   {
	*num_nonzeros = counter;
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
	{
	    jrow = send_map_elmts[j];
	    for (k = diag_i[jrow]; k < diag_i[jrow+1]; k ++) 
	    {
		B_int_j[counter] = diag_j[k] + first_col_diag;
		if (data) B_int_data[counter] = diag_data[k];
		counter ++;
  	    }
	    for (k = offd_i[jrow]; k < offd_i[jrow+1]; k ++) 
	    {
		B_int_j[counter] = col_map_offd[offd_j[k]];
		if (data) B_int_data[counter] = offd_data[k];
		counter ++;
  	    }
	   
	}
	*num_nonzeros = counter - *num_nonzeros;

	start_index += *num_nonzeros;
        jdata_send_map_starts[i+1] = start_index;
   }

   tmp_comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
   jx_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
   jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   jx_ParCSRCommPkgSendProcs(tmp_comm_pkg) = jx_ParCSRCommPkgSendProcs(comm_pkg);
   jx_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = jx_ParCSRCommPkgRecvProcs(comm_pkg);
   jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

   jx_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

  /*--------------------------------------------------------------------------
   * after communication exchange B_ext_i[j+1] contains the number of elements
   * of a row j ! 
   * evaluate B_ext_i and compute *num_nonzeros for B_ext 
   *--------------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i ++)
   {
      for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j ++)
      {
         B_ext_i[j+1] += B_ext_i[j];
      }
   }
   *num_nonzeros = B_ext_i[num_rows_B_ext];

   *pB_ext_j = jx_CTAlloc(JX_Int, *num_nonzeros);
   B_ext_j = *pB_ext_j;
   if (data) 
   {
      *pB_ext_data = jx_CTAlloc(JX_Real, *num_nonzeros);
      B_ext_data = *pB_ext_data;
   }

   for (i = 0; i < num_recvs; i ++)
   {
	start_index = B_ext_i[recv_vec_starts[i]];
	*num_nonzeros = B_ext_i[recv_vec_starts[i+1]] - start_index;
	jdata_recv_vec_starts[i+1] = B_ext_i[recv_vec_starts[i+1]];
   }

   jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   comm_handle = jx_ParCSRCommHandleCreate(11, tmp_comm_pkg, B_int_j, B_ext_j);
   jx_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
      comm_handle = jx_ParCSRCommHandleCreate(1,tmp_comm_pkg,B_int_data,B_ext_data);
      jx_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   jx_TFree(jdata_send_map_starts);
   jx_TFree(jdata_recv_vec_starts);
   jx_TFree(tmp_comm_pkg);
   jx_TFree(B_int_i);
   jx_TFree(B_int_j);
   if (data) jx_TFree(B_int_data);
   if (find_row_map) jx_TFree(B_int_row_map);
}

/*!
 * \fn jx_ParCSRMatrix *jx_ParMatmul
 * \brief multiplies two ParCSRMatrices A and B and returns the product in ParCSRMatrix C
 * \note C does not own the partitionings since its row_starts is owned by A and col_starts by B
 * \date 2017/02/25
 */
jx_ParCSRMatrix *
jx_ParMatmul( jx_ParCSRMatrix *A, jx_ParCSRMatrix *B )
{
   MPI_Comm      comm = jx_ParCSRMatrixComm(A);

   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(A);
   
   JX_Real       *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int          *A_diag_i = jx_CSRMatrixI(A_diag);
   JX_Int          *A_diag_j = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix *A_offd = jx_ParCSRMatrixOffd(A);
   
   JX_Real       *A_offd_data = jx_CSRMatrixData(A_offd);
   JX_Int          *A_offd_i = jx_CSRMatrixI(A_offd);
   JX_Int          *A_offd_j = jx_CSRMatrixJ(A_offd);

   JX_Int          *row_starts_A = jx_ParCSRMatrixRowStarts(A);
   JX_Int           num_rows_diag_A = jx_CSRMatrixNumRows(A_diag);
   JX_Int           num_cols_diag_A = jx_CSRMatrixNumCols(A_diag);
   JX_Int           num_cols_offd_A = jx_CSRMatrixNumCols(A_offd);
   
   jx_CSRMatrix *B_diag = jx_ParCSRMatrixDiag(B);
   
   JX_Real    *B_diag_data = jx_CSRMatrixData(B_diag);
   JX_Int       *B_diag_i = jx_CSRMatrixI(B_diag);
   JX_Int       *B_diag_j = jx_CSRMatrixJ(B_diag);

   jx_CSRMatrix *B_offd = jx_ParCSRMatrixOffd(B);
   JX_Int          *col_map_offd_B = jx_ParCSRMatrixColMapOffd(B);
   
   JX_Real    *B_offd_data = jx_CSRMatrixData(B_offd);
   JX_Int       *B_offd_i = jx_CSRMatrixI(B_offd);
   JX_Int       *B_offd_j = jx_CSRMatrixJ(B_offd);

   JX_Int        first_col_diag_B = jx_ParCSRMatrixFirstColDiag(B);
   JX_Int        last_col_diag_B;
   JX_Int       *col_starts_B = jx_ParCSRMatrixColStarts(B);
   JX_Int        num_rows_diag_B = jx_CSRMatrixNumRows(B_diag);
   JX_Int        num_cols_diag_B = jx_CSRMatrixNumCols(B_diag);
   JX_Int        num_cols_offd_B = jx_CSRMatrixNumCols(B_offd);

   jx_ParCSRMatrix *C;
   JX_Int          *col_map_offd_C = NULL;
   JX_Int          *map_B_to_C = NULL;

   jx_CSRMatrix *C_diag;

   JX_Real   *C_diag_data;
   JX_Int       *C_diag_i;
   JX_Int       *C_diag_j;

   jx_CSRMatrix *C_offd;

   JX_Real   *C_offd_data=NULL;
   JX_Int       *C_offd_i=NULL;
   JX_Int       *C_offd_j=NULL;

   JX_Int        C_diag_size;
   JX_Int        C_offd_size;
   JX_Int        num_cols_offd_C = 0;
   
   jx_CSRMatrix *Bs_ext = NULL;
   
   JX_Real    *Bs_ext_data = NULL;
   JX_Int       *Bs_ext_i = NULL;
   JX_Int       *Bs_ext_j = NULL;

   JX_Real    *B_ext_diag_data = NULL;
   JX_Int       *B_ext_diag_i;
   JX_Int       *B_ext_diag_j = NULL;
   JX_Int        B_ext_diag_size;

   JX_Real    *B_ext_offd_data = NULL;
   JX_Int       *B_ext_offd_i;
   JX_Int       *B_ext_offd_j = NULL;
   JX_Int        B_ext_offd_size;

   JX_Int        n_rows_A, n_cols_A;
   JX_Int        n_rows_B, n_cols_B;
   JX_Int        allsquare = 0;
   JX_Int        num_procs;
   JX_Int       *my_diag_array;
   JX_Int       *my_offd_array;
   JX_Int        max_num_threads;

   JX_Real    zero = 0.0;

   n_rows_A = jx_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = jx_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = jx_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = jx_ParCSRMatrixGlobalNumCols(B);

   max_num_threads = jx_NumThreads();
   my_diag_array = jx_CTAlloc(JX_Int, max_num_threads);
   my_offd_array = jx_CTAlloc(JX_Int, max_num_threads);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      jx_error_w_msg(JX_ERROR_GENERIC," Error! Incompatible matrix dimensions!\n");
      return NULL;
   }
   if ( num_rows_diag_A==num_cols_diag_B) allsquare = 1;

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   jx_MPI_Comm_size(comm, &num_procs);

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within 
       * jx_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext = jx_ParCSRMatrixExtractBExt(B,A,1);
      Bs_ext_data = jx_CSRMatrixData(Bs_ext);
      Bs_ext_i    = jx_CSRMatrixI(Bs_ext);
      Bs_ext_j    = jx_CSRMatrixJ(Bs_ext);
   }
   B_ext_diag_i = jx_CTAlloc(JX_Int, num_cols_offd_A+1);
   B_ext_offd_i = jx_CTAlloc(JX_Int, num_cols_offd_A+1);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + num_cols_diag_B -1;

   JX_Int *temp = NULL;
#if JX_USING_OPENMP
#pragma omp parallel 
#endif
   {
     JX_Int size, rest, ii;
     JX_Int ns, ne;
     JX_Int i1, i, j;
     JX_Int my_offd_size, my_diag_size;
     JX_Int cnt_offd, cnt_diag;

     JX_Int num_threads = jx_NumActiveThreads();

     size = num_cols_offd_A/num_threads;
     rest = num_cols_offd_A - size*num_threads;
     ii = jx_GetThreadNum();
     if (ii < rest)
     {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
     }
     else
     {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
     }

     my_diag_size = 0;
     my_offd_size = 0;
     for (i=ns; i < ne; i++)
     {
       B_ext_diag_i[i] = my_diag_size;
       B_ext_offd_i[i] = my_offd_size;
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            my_offd_size++;
         else
            my_diag_size++;
     }
     my_diag_array[ii] = my_diag_size;
     my_offd_array[ii] = my_offd_size;

#if JX_USING_OPENMP
#pragma omp barrier
#endif

     if (ii)
     {
       my_diag_size = my_diag_array[0];
       my_offd_size = my_offd_array[0];
       for (i1 = 1; i1 < ii; i1++)
       {
          my_diag_size += my_diag_array[i1];
          my_offd_size += my_offd_array[i1];
       }

       for (i1 = ns; i1 < ne; i1++)
       {
          B_ext_diag_i[i1] += my_diag_size;
          B_ext_offd_i[i1] += my_offd_size;
       }
     }
     else
     {
       B_ext_diag_size = 0;
       B_ext_offd_size = 0;
       for (i1 = 0; i1 < num_threads; i1++)
       {
          B_ext_diag_size += my_diag_array[i1];
          B_ext_offd_size += my_offd_array[i1];
       }
       B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
       B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

       if (B_ext_diag_size)
       {
          B_ext_diag_j = jx_CTAlloc(JX_Int, B_ext_diag_size);
          B_ext_diag_data = jx_CTAlloc(JX_Real, B_ext_diag_size);
       }
       if (B_ext_offd_size)
       {
          B_ext_offd_j = jx_CTAlloc(JX_Int, B_ext_offd_size);
          B_ext_offd_data = jx_CTAlloc(JX_Real, B_ext_offd_size);
       }
       if (B_ext_offd_size || num_cols_offd_B)
          temp = jx_CTAlloc(JX_Int, B_ext_offd_size+num_cols_offd_B);
     }

#if JX_USING_OPENMP
#pragma omp barrier
#endif

     cnt_offd = B_ext_offd_i[ns];
     cnt_diag = B_ext_diag_i[ns];
     for (i=ns; i < ne; i++)
     {
       for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            temp[cnt_offd] = Bs_ext_j[j];
            B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
            B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
         }
         else
         {
            B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
            B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
         }
     }

#if JX_USING_OPENMP
#pragma omp barrier
#endif

     if (ii == 0)
     {
      JX_Int        cnt;

      if (num_procs > 1)
      {
         jx_CSRMatrixDestroy(Bs_ext);
         Bs_ext = NULL;
      }

      cnt = 0;
      if (B_ext_offd_size || num_cols_offd_B)
      {
         cnt = B_ext_offd_size;
         for (i=0; i < num_cols_offd_B; i++)
            temp[cnt++] = col_map_offd_B[i];
         if (cnt)
         {
            JX_Int        value;
            jx_qsort0(temp, 0, cnt-1);
            num_cols_offd_C = 1;
            value = temp[0];
            for (i=1; i < cnt; i++)
            {
               if (temp[i] > value)
               {
                  value = temp[i];
                  temp[num_cols_offd_C++] = value;
               }
            }
         }

         if (num_cols_offd_C)
            col_map_offd_C = jx_CTAlloc(JX_Int,num_cols_offd_C);

         for (i=0; i < num_cols_offd_C; i++)
            col_map_offd_C[i] = temp[i];

         jx_TFree(temp);
      }
     }

#if JX_USING_OPENMP
#pragma omp barrier
#endif

     for (i=ns; i < ne; i++)
        for (j=B_ext_offd_i[i]; j < B_ext_offd_i[i+1]; j++)
            B_ext_offd_j[j] = jx_BinarySearch(col_map_offd_C, B_ext_offd_j[j], num_cols_offd_C);

    } /* end parallel region */

    jx_TFree(my_diag_array);
    jx_TFree(my_offd_array);

     if (num_cols_offd_B)
     {
         JX_Int i, cnt;
         map_B_to_C = jx_CTAlloc(JX_Int,num_cols_offd_B);

         cnt = 0;
         for (i=0; i < num_cols_offd_C; i++)
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B) break;
            }
      }

   jx_ParMatmul_RowSizes(
      /*&C_diag_i, &C_offd_i, &B_marker,*/
      &C_diag_i, &C_offd_i, 
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_diag_i, B_ext_diag_j, B_ext_offd_i, B_ext_offd_j,
      map_B_to_C,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A, allsquare,
      num_cols_diag_B, num_cols_offd_B,
      num_cols_offd_C
      );

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;
   C_diag_data = jx_CTAlloc(JX_Real, C_diag_size);
   C_diag_j    = jx_CTAlloc(JX_Int, C_diag_size);
   if (C_offd_size)
   { 
      C_offd_data = jx_CTAlloc(JX_Real, C_offd_size);
      C_offd_j    = jx_CTAlloc(JX_Int, C_offd_size);
   } 

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/
#if JX_USING_OPENMP
#pragma omp parallel 
#endif
   {
    JX_Int *B_marker = NULL;
    JX_Int ns, ne, size, rest, ii;
    JX_Int i1, i2, i3, jj2, jj3;
    JX_Int jj_row_begin_diag, jj_count_diag;
    JX_Int jj_row_begin_offd, jj_count_offd;
    JX_Int num_threads;
    JX_Real a_entry; /*, a_b_product;*/

    ii = jx_GetThreadNum();
    num_threads = jx_NumActiveThreads();
    size = num_rows_diag_A/num_threads;
    rest = num_rows_diag_A - size*num_threads;
    if (ii < rest)
    {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
    }
    else
    {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
    }
    jj_count_diag = C_diag_i[ns];
    jj_count_offd = C_offd_i[ns];
    if (num_cols_diag_B || num_cols_offd_C)
    B_marker = jx_CTAlloc(JX_Int, num_cols_diag_B+num_cols_offd_C);
    for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
      B_marker[i1] = -1;

    /*-----------------------------------------------------------------------
     *  Loop over interior c-points.
     *-----------------------------------------------------------------------*/

    for (i1 = ns; i1 < ne; i1++)
    {

      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1} 
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) 
      {
         B_marker[i1] = jj_count_diag;
         C_diag_data[jj_count_diag] = zero;
         C_diag_j[jj_count_diag] = i1;
         jj_count_diag++;
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/
         
      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            a_entry = A_offd_data[jj2];
            
            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_ext.
             *-----------------------------------------------------------*/

            for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_data[jj_count_offd] = a_entry*B_ext_offd_data[jj3];
                  C_offd_j[jj_count_offd] = i3-num_cols_diag_B;
                  jj_count_offd++;
               }
               else
                  C_offd_data[B_marker[i3]] += a_entry*B_ext_offd_data[jj3];
            }
            for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
            {
               i3 = B_ext_diag_j[jj3];
               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  C_diag_data[jj_count_diag] = a_entry*B_ext_diag_data[jj3];
                  C_diag_j[jj_count_diag] = i3;
                  jj_count_diag++;
               }
               else
                  C_diag_data[B_marker[i3]] += a_entry*B_ext_diag_data[jj3];
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_diag.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
      {
         i2 = A_diag_j[jj2];
         a_entry = A_diag_data[jj2];
            
         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_diag.
          *-----------------------------------------------------------*/

         for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
         {
            i3 = B_diag_j[jj3];
                  
            /*--------------------------------------------------------
             *  Check B_marker to see that C_{i1,i3} has not already
             *  been accounted for. If it has not, create a new entry.
             *  If it has, add new contribution.
             *--------------------------------------------------------*/

            if (B_marker[i3] < jj_row_begin_diag)
            {
               B_marker[i3] = jj_count_diag;
               C_diag_data[jj_count_diag] = a_entry*B_diag_data[jj3];
               C_diag_j[jj_count_diag] = i3;
               jj_count_diag++;
            }
            else
            {
               C_diag_data[B_marker[i3]] += a_entry*B_diag_data[jj3];
            }
         }
         if (num_cols_offd_B)
         {
            for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_data[jj_count_offd] = a_entry*B_offd_data[jj3];
                  C_offd_j[jj_count_offd] = i3-num_cols_diag_B;
                  jj_count_offd++;
               }
               else
               {
                  C_offd_data[B_marker[i3]] += a_entry*B_offd_data[jj3];
               }
            }
         }
      }
    }
    jx_TFree(B_marker);
   } /*end parallel region */

   C = jx_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                 col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

   /* Note that C does not own the partitionings */
   jx_ParCSRMatrixSetRowStartsOwner(C,0);
   jx_ParCSRMatrixSetColStartsOwner(C,0);

   C_diag = jx_ParCSRMatrixDiag(C);
   jx_CSRMatrixData(C_diag) = C_diag_data; 
   jx_CSRMatrixI(C_diag) = C_diag_i; 
   jx_CSRMatrixJ(C_diag) = C_diag_j; 

   C_offd = jx_ParCSRMatrixOffd(C);
   jx_CSRMatrixI(C_offd) = C_offd_i; 
   jx_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      jx_CSRMatrixData(C_offd) = C_offd_data; 
      jx_CSRMatrixJ(C_offd) = C_offd_j; 
      jx_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   jx_TFree(B_ext_diag_i);
   if (B_ext_diag_size)
   {
      jx_TFree(B_ext_diag_j);
      jx_TFree(B_ext_diag_data);
   }
   jx_TFree(B_ext_offd_i);
   if (B_ext_offd_size)
   {
      jx_TFree(B_ext_offd_j);
      jx_TFree(B_ext_offd_data);
   }
   if (num_cols_offd_B) jx_TFree(map_B_to_C);

   return C;
}

/*!
 * \fn jx_ParCSRMatrix *jx_ParTMatmul
 * \brief multiplies two ParCSRMatrices transpose(A) and B and returns the product in ParCSRMatrix C
 * \note C does not own the partitionings since its row_starts is owned by A and col_starts by B.
 * \date 2017/02/25
 */
jx_ParCSRMatrix *
jx_ParTMatmul( jx_ParCSRMatrix *A, jx_ParCSRMatrix *B )
{
   MPI_Comm 	     comm = jx_ParCSRMatrixComm(A);
   jx_ParCSRCommPkg *comm_pkg_A = jx_ParCSRMatrixCommPkg(A);

   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix *AT_diag = NULL;

   jx_CSRMatrix *A_offd = jx_ParCSRMatrixOffd(A);
   jx_CSRMatrix *AT_offd = NULL;
   
   JX_Int	num_rows_diag_A = jx_CSRMatrixNumRows(A_diag);
   JX_Int	num_cols_diag_A = jx_CSRMatrixNumCols(A_diag);
   
   jx_CSRMatrix *B_diag = jx_ParCSRMatrixDiag(B);
   
   jx_CSRMatrix *B_offd = jx_ParCSRMatrixOffd(B);
   JX_Int	        *col_map_offd_B = jx_ParCSRMatrixColMapOffd(B);
   
   JX_Int	first_col_diag_B = jx_ParCSRMatrixFirstColDiag(B);
   JX_Int *col_starts_A = jx_ParCSRMatrixColStarts(A);
   JX_Int *col_starts_B = jx_ParCSRMatrixColStarts(B);
   JX_Int	num_rows_diag_B = jx_CSRMatrixNumRows(B_diag);
   JX_Int	num_cols_diag_B = jx_CSRMatrixNumCols(B_diag);
   JX_Int	num_cols_offd_B = jx_CSRMatrixNumCols(B_offd);

   jx_ParCSRMatrix *C;
   JX_Int	           *col_map_offd_C = NULL;
   JX_Int	           *map_B_to_C = NULL;

   jx_CSRMatrix *C_diag = NULL;
   jx_CSRMatrix *C_tmp_diag = NULL;

   JX_Real    *C_diag_data = NULL;
   JX_Int       *C_diag_i = NULL;
   JX_Int       *C_diag_j = NULL;
   JX_Int	first_col_diag_C;
   JX_Int	last_col_diag_C;

   jx_CSRMatrix *C_offd = NULL;
   jx_CSRMatrix *C_tmp_offd = NULL;
   jx_CSRMatrix *C_int = NULL;
   jx_CSRMatrix *C_ext = NULL;
   JX_Int   *C_ext_i = NULL;
   JX_Int   *C_ext_j = NULL;
   JX_Real   *C_ext_data = NULL;
   JX_Int   *C_ext_diag_i = NULL;
   JX_Int   *C_ext_diag_j = NULL;
   JX_Real   *C_ext_diag_data = NULL;
   JX_Int   *C_ext_offd_i = NULL;
   JX_Int   *C_ext_offd_j = NULL;
   JX_Real   *C_ext_offd_data = NULL;
   JX_Int	C_ext_size = 0;
   JX_Int	C_ext_diag_size = 0;
   JX_Int	C_ext_offd_size = 0;

   JX_Int   *C_tmp_diag_i;
   JX_Int   *C_tmp_diag_j = NULL;
   JX_Real   *C_tmp_diag_data = NULL;
   JX_Int   *C_tmp_offd_i = NULL;
   JX_Int   *C_tmp_offd_j = NULL;
   JX_Real   *C_tmp_offd_data = NULL;

   JX_Real          *C_offd_data=NULL;
   JX_Int       *C_offd_i=NULL;
   JX_Int       *C_offd_j=NULL;

   JX_Int       *temp;
   JX_Int       *send_map_starts_A = NULL;
   JX_Int       *send_map_elmts_A = NULL;
   JX_Int        num_sends_A = 0;

   JX_Int		    num_cols_offd_C = 0;
   
   JX_Int		   *P_marker;

   JX_Int              i, j;
   JX_Int              i1, j_indx;
   
   JX_Int		    n_rows_A, n_cols_A;
   JX_Int		    n_rows_B, n_cols_B;
   /*JX_Int              allsquare = 0;*/
   JX_Int              cnt, cnt_offd, cnt_diag;
   JX_Int 		    value;
   JX_Int 		    num_procs, my_id;
   JX_Int                max_num_threads;
   JX_Int               *C_diag_array = NULL;
   JX_Int               *C_offd_array = NULL;

   JX_Int first_row_index, first_col_diag;
   JX_Int local_num_rows, local_num_cols;

   n_rows_A = jx_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = jx_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = jx_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = jx_ParCSRMatrixGlobalNumCols(B);

   jx_MPI_Comm_size(comm,&num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   max_num_threads = jx_NumThreads();

   if (n_rows_A != n_rows_B || num_rows_diag_A != num_rows_diag_B)
   {
        jx_error_w_msg(JX_ERROR_GENERIC," Error! Incompatible matrix dimensions!\n");
	return NULL;
   }

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   jx_CSRMatrixTranspose(A_diag, &AT_diag, 1);
   jx_CSRMatrixTranspose(A_offd, &AT_offd, 1);

   C_tmp_diag = jx_CSRMatrixMultiply(AT_diag, B_diag);
   C_ext_size = 0;
   if (num_procs > 1) 
   {
      jx_CSRMatrix *C_int_diag;
      jx_CSRMatrix *C_int_offd;
      C_tmp_offd = jx_CSRMatrixMultiply(AT_diag, B_offd);
      C_int_diag = jx_CSRMatrixMultiply(AT_offd, B_diag);
      C_int_offd = jx_CSRMatrixMultiply(AT_offd, B_offd);
      jx_ParCSRMatrixDiag(B) = C_int_diag;
      jx_ParCSRMatrixOffd(B) = C_int_offd;
      C_int = jx_MergeDiagAndOffd(B);
      jx_ParCSRMatrixDiag(B) = B_diag;
      jx_ParCSRMatrixOffd(B) = B_offd;
      C_ext = jx_ExchangeRAPData(C_int, comm_pkg_A);
      C_ext_i = jx_CSRMatrixI(C_ext);
      C_ext_j = jx_CSRMatrixJ(C_ext);
      C_ext_data = jx_CSRMatrixData(C_ext);
      C_ext_size = C_ext_i[jx_CSRMatrixNumRows(C_ext)];
      jx_CSRMatrixDestroy(C_int);
      jx_CSRMatrixDestroy(C_int_diag);
      jx_CSRMatrixDestroy(C_int_offd);
   }
   else
   {
     C_tmp_offd = jx_CSRMatrixCreate(num_cols_diag_A, 0, 0);
     jx_CSRMatrixInitialize(C_tmp_offd);
   }
   jx_CSRMatrixDestroy(AT_diag);
   jx_CSRMatrixDestroy(AT_offd);

   /*-----------------------------------------------------------------------
    *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
    *  to obtain C_diag and C_offd
    *-----------------------------------------------------------------------*/

   /* check for new nonzero columns in C_offd generated through C_ext */

   first_col_diag_C = first_col_diag_B;
   last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

   C_tmp_diag_i = jx_CSRMatrixI(C_tmp_diag);
   if (C_ext_size || num_cols_offd_B)
   {
      JX_Int C_ext_num_rows;
      num_sends_A = jx_ParCSRCommPkgNumSends(comm_pkg_A);
      send_map_starts_A = jx_ParCSRCommPkgSendMapStarts(comm_pkg_A);
      send_map_elmts_A = jx_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_ext_num_rows =  send_map_starts_A[num_sends_A];
     
      C_ext_diag_i = jx_CTAlloc(JX_Int, C_ext_num_rows+1);
      C_ext_offd_i = jx_CTAlloc(JX_Int, C_ext_num_rows+1);
      temp = jx_CTAlloc(JX_Int, C_ext_size+num_cols_offd_B);
      C_ext_diag_size = 0;
      C_ext_offd_size = 0;
      for (i=0; i < C_ext_num_rows; i++)
      {
         for (j=C_ext_i[i]; j < C_ext_i[i+1]; j++)
            if (C_ext_j[j] < first_col_diag_C || C_ext_j[j] > last_col_diag_C)
	       temp[C_ext_offd_size++] = C_ext_j[j];
            else
               C_ext_diag_size++;
         C_ext_diag_i[i+1] = C_ext_diag_size;
         C_ext_offd_i[i+1] = C_ext_offd_size;
      }
      cnt = C_ext_offd_size;
      for (i=0; i < num_cols_offd_B; i++)
         temp[cnt++] = col_map_offd_B[i];

      if (cnt)
      {
	  jx_qsort0(temp,0,cnt-1);
          value = temp[0];
          num_cols_offd_C = 1;
          for (i=1; i < cnt; i++)
          {
 	     if (temp[i] > value)
	     {
		value = temp[i];
		temp[num_cols_offd_C++] = value;
	     }
  	  }
       }

       if (num_cols_offd_C)
	  col_map_offd_C = jx_CTAlloc(JX_Int, num_cols_offd_C);
       for (i=0; i < num_cols_offd_C; i++)
	  col_map_offd_C[i] = temp[i];

       jx_TFree(temp);
   
      if (C_ext_diag_size)
      {
         C_ext_diag_j = jx_CTAlloc(JX_Int, C_ext_diag_size);
         C_ext_diag_data = jx_CTAlloc(JX_Real, C_ext_diag_size);
      }
      if (C_ext_offd_size)
      {
         C_ext_offd_j = jx_CTAlloc(JX_Int, C_ext_offd_size);
         C_ext_offd_data = jx_CTAlloc(JX_Real, C_ext_offd_size);
      }

      C_tmp_diag_j = jx_CSRMatrixJ(C_tmp_diag);
      C_tmp_diag_data = jx_CSRMatrixData(C_tmp_diag);

      C_tmp_offd_i = jx_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = jx_CSRMatrixJ(C_tmp_offd);
      C_tmp_offd_data = jx_CSRMatrixData(C_tmp_offd);

      cnt_offd = 0;
      cnt_diag = 0;
      for (i=0; i < C_ext_num_rows; i++)
      {
         for (j=C_ext_i[i]; j < C_ext_i[i+1]; j++)
            if (C_ext_j[j] < first_col_diag_C || C_ext_j[j] > last_col_diag_C)
            {
               C_ext_offd_j[cnt_offd] = jx_BinarySearch(col_map_offd_C,
                                           C_ext_j[j],
                                           num_cols_offd_C);
               C_ext_offd_data[cnt_offd++] = C_ext_data[j];
            }
            else
            {
               C_ext_diag_j[cnt_diag] = C_ext_j[j] - first_col_diag_C;
               C_ext_diag_data[cnt_diag++] = C_ext_data[j];
            }
      }
   }

   if (C_ext)
   {
      jx_CSRMatrixDestroy(C_ext);
      C_ext = NULL;
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = jx_CTAlloc(JX_Int,num_cols_offd_B);

      cnt = 0;
      for (i=0; i < num_cols_offd_C; i++)
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) break;
         }
      for (i=0; 
	i < jx_CSRMatrixI(C_tmp_offd)[jx_CSRMatrixNumRows(C_tmp_offd)]; i++)
      {
         j_indx = C_tmp_offd_j[i];
         C_tmp_offd_j[i] = map_B_to_C[j_indx];
      }
   }

   /*-----------------------------------------------------------------------
    *  Need to compute C_diag = C_tmp_diag + C_ext_diag
    *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
    *  First generate structure
    *-----------------------------------------------------------------------*/

   if (C_ext_size || num_cols_offd_B)
   {
     C_diag_i = jx_CTAlloc(JX_Int, num_cols_diag_A+1);
     C_offd_i = jx_CTAlloc(JX_Int, num_cols_diag_A+1);

     C_diag_array = jx_CTAlloc(JX_Int, max_num_threads);
     C_offd_array = jx_CTAlloc(JX_Int, max_num_threads);

#if JX_USING_OPENMP
#pragma omp parallel
#endif
     {
        JX_Int *B_marker = NULL;
        JX_Int *B_marker_offd = NULL;
        JX_Int ik, jk, j1, j2, jcol;
        JX_Int ns, ne, ii, nnz_d, nnz_o;
        JX_Int rest, size;
        JX_Int num_threads = jx_NumActiveThreads();

        size = num_cols_diag_A/num_threads;
        rest = num_cols_diag_A - size*num_threads;
        ii = jx_GetThreadNum();
        if (ii < rest)
        {
           ns = ii*size+ii;
           ne = (ii+1)*size+ii+1;
        }
        else
        {
           ns = ii*size+rest;
           ne = (ii+1)*size+rest;
        }

        B_marker = jx_CTAlloc(JX_Int, num_cols_diag_B);
        B_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd_C);

        for (ik = 0; ik < num_cols_diag_B; ik++)
           B_marker[ik] = -1;

        for (ik = 0; ik < num_cols_offd_C; ik++)
           B_marker_offd[ik] = -1;

        nnz_d = 0;
        nnz_o = 0;
        for (ik = ns; ik < ne; ik++)
        {
          for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik+1]; jk++)
          {
             jcol = C_tmp_diag_j[jk];
             B_marker[jcol] = ik;
	     nnz_d++;
          }
          for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik+1]; jk++)
          {
             jcol = C_tmp_offd_j[jk];
             B_marker_offd[jcol] = ik;
	     nnz_o++;
          }
          for (jk = 0; jk < num_sends_A; jk++)
            for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk+1]; j1++)
             if (send_map_elmts_A[j1] == ik)
             {
                for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1+1]; j2++)
                {
                    jcol = C_ext_diag_j[j2];
 	            if (B_marker[jcol] < ik)
                    {
                       B_marker[jcol] = ik;
	               nnz_d++;
                    }
                }
                for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1+1]; j2++)
                {
                    jcol = C_ext_offd_j[j2];
 	            if (B_marker_offd[jcol] < ik)
                    {
                       B_marker_offd[jcol] = ik;
	               nnz_o++;
                    }
                }
                break;
             }
               C_diag_array[ii] = nnz_d;
               C_offd_array[ii] = nnz_o;
        }
#if JX_USING_OPENMP
#pragma omp barrier
#endif
       if (ii == 0)
       {
          nnz_d = 0;
          nnz_o = 0;
          for (ik = 0; ik < num_threads-1; ik++)
          {
             C_diag_array[ik+1] += C_diag_array[ik];
             C_offd_array[ik+1] += C_offd_array[ik];
          }
          nnz_d = C_diag_array[num_threads-1];
          nnz_o = C_offd_array[num_threads-1];
          C_diag_i[num_cols_diag_A] = nnz_d;
          C_offd_i[num_cols_diag_A] = nnz_o;

          C_diag = jx_CSRMatrixCreate(num_cols_diag_A, num_cols_diag_A, nnz_d);
          C_offd = jx_CSRMatrixCreate(num_cols_diag_A, num_cols_offd_C, nnz_o);
          jx_CSRMatrixI(C_diag) = C_diag_i;
          jx_CSRMatrixInitialize(C_diag);
          C_diag_j = jx_CSRMatrixJ(C_diag);
          C_diag_data = jx_CSRMatrixData(C_diag);
          jx_CSRMatrixI(C_offd) = C_offd_i;
          jx_CSRMatrixInitialize(C_offd);
          C_offd_j = jx_CSRMatrixJ(C_offd);
          C_offd_data = jx_CSRMatrixData(C_offd); 
       }
#if JX_USING_OPENMP
#pragma omp barrier
#endif

   /*-----------------------------------------------------------------------
    *  Need to compute C_diag = C_tmp_diag + C_ext_diag
    *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
    *  Now fill in values
    *-----------------------------------------------------------------------*/
   
     for (ik = 0; ik < num_cols_diag_B; ik++)
        B_marker[ik] = -1;

     for (ik = 0; ik < num_cols_offd_C; ik++)
        B_marker_offd[ik] = -1;

   /*-----------------------------------------------------------------------
    *  Populate matrices
    *-----------------------------------------------------------------------*/

      nnz_d = 0;
      nnz_o = 0;
        nnz_o = 0;
        if (ii)
        {
           nnz_d = C_diag_array[ii-1];
           nnz_o = C_offd_array[ii-1];
        }
        for (ik = ns; ik < ne; ik++)
        {
           C_diag_i[ik] = nnz_d;
           C_offd_i[ik] = nnz_o;
           for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik+1]; jk++)
           {
              jcol = C_tmp_diag_j[jk];
              C_diag_j[nnz_d] = jcol;
              C_diag_data[nnz_d] = C_tmp_diag_data[jk];
              B_marker[jcol] = nnz_d;
              nnz_d++;
           }
           for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik+1]; jk++)
           {
              jcol = C_tmp_offd_j[jk];
              C_offd_j[nnz_o] = jcol;
              C_offd_data[nnz_o] = C_tmp_offd_data[jk];
              B_marker_offd[jcol] = nnz_o;
              nnz_o++;
           }
           for (jk = 0; jk < num_sends_A; jk++)
              for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk+1]; j1++)
                 if (send_map_elmts_A[j1] == ik)
                 {
                    for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1+1]; j2++)
                    {
                       jcol = C_ext_diag_j[j2];
                       if (B_marker[jcol] < C_diag_i[ik])
                       {
                          C_diag_j[nnz_d] = jcol;
                          C_diag_data[nnz_d] = C_ext_diag_data[j2];
                          B_marker[jcol] = nnz_d;
                          nnz_d++;
                       }
                       else
                          C_diag_data[B_marker[jcol]] += C_ext_diag_data[j2];
                    }
                    for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1+1]; j2++)
                    {
                       jcol = C_ext_offd_j[j2];
                       if (B_marker_offd[jcol] < C_offd_i[ik])
                       {
                          C_offd_j[nnz_o] = jcol;
                          C_offd_data[nnz_o] = C_ext_offd_data[j2];
                          B_marker_offd[jcol] = nnz_o;
                          nnz_o++;
                       }
                       else
                          C_offd_data[B_marker_offd[jcol]] += C_ext_offd_data[j2];
                    }
                    break;
                 }
        }
        jx_TFree(B_marker);
        jx_TFree(B_marker_offd);
     } /*end parallel region */
     jx_TFree(C_diag_array);
     jx_TFree(C_offd_array);
   }

   /*C = jx_ParCSRMatrixCreate(comm, n_cols_A, n_cols_B, col_starts_A,
	col_starts_B, num_cols_offd_C, nnz_diag, nnz_offd);

   jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(C));
   jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(C)); */
#ifdef JX_NO_GLOBAL_PARTITION
   /* row_starts[0] is start of local rows.  row_starts[1] is start of next 
      processor's rows */
   first_row_index = col_starts_A[0];
   local_num_rows = col_starts_A[1]-first_row_index ;
   first_col_diag = col_starts_B[0];
   local_num_cols = col_starts_B[1]-first_col_diag;
#else
   first_row_index = col_starts_A[my_id];
   local_num_rows = col_starts_A[my_id+1]-first_row_index;
   first_col_diag = col_starts_B[my_id];
   local_num_cols = col_starts_B[my_id+1]-first_col_diag;
#endif

   C = jx_CTAlloc(jx_ParCSRMatrix, 1);
   jx_ParCSRMatrixComm(C) = comm;
   jx_ParCSRMatrixGlobalNumRows(C) = n_cols_A;
   jx_ParCSRMatrixGlobalNumCols(C) = n_cols_B;
   jx_ParCSRMatrixFirstRowIndex(C) = first_row_index;
   jx_ParCSRMatrixFirstColDiag(C) = first_col_diag;
   jx_ParCSRMatrixLastRowIndex(C) = first_row_index + local_num_rows - 1;
   jx_ParCSRMatrixLastColDiag(C) = first_col_diag + local_num_cols - 1;

   jx_ParCSRMatrixColMapOffd(C) = NULL;

   jx_ParCSRMatrixAssumedPartition(C) = NULL;

   jx_ParCSRMatrixRowStarts(C) = col_starts_A;
   jx_ParCSRMatrixColStarts(C) = col_starts_B;

   jx_ParCSRMatrixCommPkg(C) = NULL;
   jx_ParCSRMatrixCommPkgT(C) = NULL;

   /* set defaults */
   jx_ParCSRMatrixOwnsData(C) = 1;
   jx_ParCSRMatrixRowindices(C) = NULL;
   jx_ParCSRMatrixRowvalues(C) = NULL;
   jx_ParCSRMatrixGetrowactive(C) = 0;

/* Note that C does not own the partitionings */
   jx_ParCSRMatrixSetRowStartsOwner(C,0);
   jx_ParCSRMatrixSetColStartsOwner(C,0);

   if (C_diag) jx_ParCSRMatrixDiag(C) = C_diag;
   else jx_ParCSRMatrixDiag(C) = C_tmp_diag;
   if (C_offd) jx_ParCSRMatrixOffd(C) = C_offd;
   else jx_ParCSRMatrixOffd(C) = C_tmp_offd;

   if (num_cols_offd_C)
   {
      JX_Int jj_count_offd, nnz_offd;
      JX_Int *new_col_map_offd_C = NULL;

      P_marker = jx_CTAlloc(JX_Int,num_cols_offd_C);
      for (i=0; i < num_cols_offd_C; i++)
         P_marker[i] = -1;

      jj_count_offd = 0;
      nnz_offd = C_offd_i[num_cols_diag_A];
      for (i=0; i < nnz_offd; i++)
      {
         i1 = C_offd_j[i];
         if (P_marker[i1])
         {
            P_marker[i1] = 0;
            jj_count_offd++;
         }
      }

      if (jj_count_offd < num_cols_offd_C)
      {
         new_col_map_offd_C = jx_CTAlloc(JX_Int,jj_count_offd);
         jj_count_offd = 0;
         for (i=0; i < num_cols_offd_C; i++)
            if (!P_marker[i])
            {
               P_marker[i] = jj_count_offd;
               new_col_map_offd_C[jj_count_offd++] = col_map_offd_C[i];
            }

         for (i=0; i < nnz_offd; i++)
         {
            i1 = C_offd_j[i];
            C_offd_j[i] = P_marker[i1];
         }

         num_cols_offd_C = jj_count_offd;
         jx_TFree(col_map_offd_C);
         col_map_offd_C = new_col_map_offd_C;
         jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(C)) = num_cols_offd_C;
      }
      jx_TFree(P_marker);
   }
   jx_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   if (C_ext_size || num_cols_offd_B)
   {
      jx_TFree(C_ext_diag_i);
      jx_TFree(C_ext_offd_i);
   }
   if (C_ext_diag_size)
   {
      jx_TFree(C_ext_diag_j);
      jx_TFree(C_ext_diag_data);
   }
   if (C_ext_offd_size)
   {
      jx_TFree(C_ext_offd_j);
      jx_TFree(C_ext_offd_data);
   }
   if (num_cols_offd_B) jx_TFree(map_B_to_C);

   if (C_diag) jx_CSRMatrixDestroy(C_tmp_diag);
   if (C_offd) jx_CSRMatrixDestroy(C_tmp_offd);

   return C;
}

/*!
 * \fn void jx_ParMatmul_RowSizes
 * \brief The following function was formerly part of jx_ParMatmul
 *        but was removed so it can also be used for
 *        multiplication of Boolean matrices
 * \date 2017/02/25
 */
void
jx_ParMatmul_RowSizes( JX_Int **C_diag_i,
                       JX_Int **C_offd_i,
                       JX_Int  *A_diag_i,
                       JX_Int  *A_diag_j,
                       JX_Int  *A_offd_i,
                       JX_Int  *A_offd_j,
                       JX_Int  *B_diag_i,
                       JX_Int  *B_diag_j,
                       JX_Int  *B_offd_i,
                       JX_Int  *B_offd_j,
                       JX_Int  *B_ext_diag_i,
                       JX_Int  *B_ext_diag_j, 
                       JX_Int  *B_ext_offd_i,
                       JX_Int  *B_ext_offd_j,
                       JX_Int  *map_B_to_C,
                       JX_Int  *C_diag_size,
                       JX_Int  *C_offd_size,
                       JX_Int   num_rows_diag_A,
                       JX_Int   num_cols_offd_A,
                       JX_Int   allsquare,
                       JX_Int   num_cols_diag_B,
                       JX_Int   num_cols_offd_B,
                       JX_Int   num_cols_offd_C )
{
   JX_Int i1, i2, i3, jj2, jj3;
   JX_Int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
   JX_Int start_indexing = 0; /* start indexing for C_data at 0 */
   JX_Int num_threads = jx_NumThreads();
   JX_Int *jj_count_diag_array;
   JX_Int *jj_count_offd_array;
   JX_Int ii, size, rest;

   *C_diag_i = jx_CTAlloc(JX_Int, num_rows_diag_A+1);
   *C_offd_i = jx_CTAlloc(JX_Int, num_rows_diag_A+1);
   jj_count_diag_array = jx_CTAlloc(JX_Int, num_threads);
   jj_count_offd_array = jx_CTAlloc(JX_Int, num_threads);
   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
   size = num_rows_diag_A/num_threads;
   rest = num_rows_diag_A - size*num_threads;

#if JX_USING_OPENMP
#pragma omp parallel private(ii, i1, jj_row_begin_diag, jj_row_begin_offd, jj_count_diag, jj_count_offd, jj2, i2, jj3, i3) 
#endif
   /*for (ii=0; ii < num_threads; ii++)*/
   {
    JX_Int *B_marker = NULL;
    JX_Int ns, ne;
    ii = jx_GetThreadNum();
    if (ii < rest)
    {
       ns = ii*size+ii;
       ne = (ii+1)*size+ii+1;
    }
    else
    {
       ns = ii*size+rest;
       ne = (ii+1)*size+rest;
    }
    jj_count_diag = start_indexing;
    jj_count_offd = start_indexing;

    if (num_cols_diag_B || num_cols_offd_C)
    B_marker = jx_CTAlloc(JX_Int, num_cols_diag_B+num_cols_offd_C);
    for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
      B_marker[i1] = -1;

    for (i1 = ns; i1 < ne; i1++)
    {
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1} (for square matrices). 
       *--------------------------------------------------------------------*/
 
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) {
         B_marker[i1] = jj_count_diag;
         jj_count_diag++;
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/
         
      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
 
            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_ext.
             *-----------------------------------------------------------*/
 
            for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, mark it and increment
                *  counter.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  jj_count_offd++;
               } 
            }
            for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
            {
               i3 = B_ext_diag_j[jj3];
                  
               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  jj_count_diag++;
               } 
            }
         }
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_diag.
       *-----------------------------------------------------------------*/
         
      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
      {
         i2 = A_diag_j[jj2];
 
         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_diag.
          *-----------------------------------------------------------*/
 
         for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
         {
            i3 = B_diag_j[jj3];
                  
            /*--------------------------------------------------------
             *  Check B_marker to see that C_{i1,i3} has not already
             *  been accounted for. If it has not, mark it and increment
             *  counter.
             *--------------------------------------------------------*/
 
            if (B_marker[i3] < jj_row_begin_diag)
            {
               B_marker[i3] = jj_count_diag;
               jj_count_diag++;
            }
         }

         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_offd.
          *-----------------------------------------------------------*/

         if (num_cols_offd_B)
         { 
            for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
            {
               i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  
               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, mark it and increment
                *  counter.
                *--------------------------------------------------------*/
 
               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  jj_count_offd++;
               }
            }
         }
      }
            
      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      (*C_diag_i)[i1] = jj_row_begin_diag;
      (*C_offd_i)[i1] = jj_row_begin_offd;
      
    }
    jj_count_diag_array[ii] = jj_count_diag;
    jj_count_offd_array[ii] = jj_count_offd;

    jx_TFree(B_marker);
#if JX_USING_OPENMP
#pragma omp barrier
#endif

    if (ii)
    {
       jj_count_diag = jj_count_diag_array[0];
       jj_count_offd = jj_count_offd_array[0];
       for (i1 = 1; i1 < ii; i1++)
       {
          jj_count_diag += jj_count_diag_array[i1];
          jj_count_offd += jj_count_offd_array[i1];
       }

       for (i1 = ns; i1 < ne; i1++)
       {
          (*C_diag_i)[i1] += jj_count_diag;
          (*C_offd_i)[i1] += jj_count_offd;
       }
    }
    else
    {
       (*C_diag_i)[num_rows_diag_A] = 0;
       (*C_offd_i)[num_rows_diag_A] = 0;
       for (i1 = 0; i1 < num_threads; i1++)
       {
          (*C_diag_i)[num_rows_diag_A] += jj_count_diag_array[i1];
          (*C_offd_i)[num_rows_diag_A] += jj_count_offd_array[i1];
       }
    }
   } /* end parallel loop */
 
   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   *C_diag_size = (*C_diag_i)[num_rows_diag_A];
   *C_offd_size = (*C_offd_i)[num_rows_diag_A];

   jx_TFree(jj_count_diag_array);
   jx_TFree(jj_count_offd_array);
}
