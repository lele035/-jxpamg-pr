//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_csr_matrix.c -- basic operations for parallel matrices.
 *  Date: 2011/09/07
 */ 

#include "jx_mv.h"

/*!
 * \fn jx_ParCSRMatrix *jx_BuildMatParFromOneFile
 * \brief Build a parallel matrix by reading data from a given file.
 * \param filename pointer to the matrix file.
 * \param num_functions number of unknown functions
 * \note The storage format of the matrix is expected to be CSR, i.e.
 *         the 1st row:                           n                          // JX_Int
 *         the 2 to (n+2)-th row:                ia[k]     k=0,1,...,n       // JX_Int
 *         the (n+3) to (n+nz+2)-th row:         ja[k]     k=0,1,...,nz-1    // JX_Int
 *         the (n+nz+3) to ((n+2nz+2))-th row:    a[k]     k=0,1,...,nz-1    // JX_Real 
 *       where n is the order of the matrix, nz is the number of non-zeros of the matrix.
 * \author peghoty, Yue Xiaoqiang
 * \date 2009/07/10, 2017/10/31
 */
jx_ParCSRMatrix *
jx_BuildMatParFromOneFile( char *filename, JX_Int num_functions, JX_Int file_base )
{
   jx_ParCSRMatrix  *A     = NULL;
   jx_CSRMatrix     *A_CSR = NULL;

   JX_Int my_id, num_procs;
   JX_Int i, rest, size, num_nodes, num_dofs;
   
   JX_Int *row_part = NULL;
   JX_Int *col_part = NULL;

   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   jx_MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (my_id == 0)
   {
      A_CSR = jx_CSRMatrixRead(filename, file_base);
      jx_CSRMatrixReorder(A_CSR); 
   }

   if ((my_id == 0) && (num_functions > 1))
   {
      num_dofs = jx_CSRMatrixNumRows(A_CSR);
      num_nodes = num_dofs / num_functions;
      if (num_dofs != num_functions*num_nodes)
      {
         row_part = NULL;
         col_part = NULL;
      }
      else
      {
         row_part = jx_CTAlloc(JX_Int, num_procs+1);
         row_part[0] = 0;
         size = num_nodes / num_procs;
         rest = num_nodes - size * num_procs;
         for (i = 0; i < num_procs; i ++)
         {
            row_part[i+1] = row_part[i] + size * num_functions;
            if (i < rest) row_part[i+1] += num_functions;
         }
         col_part = row_part;
      }
   }

   /* ser -> par transfering */
   A = jx_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, row_part, col_part);

   if (my_id == 0) 
   {
      jx_CSRMatrixDestroy(A_CSR);
   }

   return (A);
}

jx_ParCSRMatrix *
jx_BuildMatParFromOneFile5( char *filename, JX_Int num_functions, JX_Int file_base )
{
   jx_ParCSRMatrix  *A     = NULL;
   jx_CSRMatrix     *A_CSR = NULL;

   JX_Int my_id, num_procs;
   JX_Int i, rest, size, num_nodes, num_dofs;
   
   JX_Int *row_part = NULL;
   JX_Int *col_part = NULL;

   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   jx_MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   if (my_id == 0)
   {
      A_CSR = jx_CSRMatrixRead2(filename, file_base);
      jx_CSRMatrixReorder(A_CSR); 
   }

   if ((my_id == 0) && (num_functions > 1))
   {
      num_dofs = jx_CSRMatrixNumRows(A_CSR);
      num_nodes = num_dofs / num_functions;
      if (num_dofs != num_functions*num_nodes)
      {
         row_part = NULL;
         col_part = NULL;
      }
      else
      {
         row_part = jx_CTAlloc(JX_Int, num_procs+1);
         row_part[0] = 0;
         size = num_nodes / num_procs;
         rest = num_nodes - size * num_procs;
         for (i = 0; i < num_procs; i ++)
         {
            row_part[i+1] = row_part[i] + size * num_functions;
            if (i < rest) row_part[i+1] += num_functions;
         }
         col_part = row_part;
      }
   }

   /* ser -> par transfering */
   A = jx_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, row_part, col_part);

   if (my_id == 0) 
   {
      jx_CSRMatrixDestroy(A_CSR);
   }

   return (A);
}

/*!
 * \fn jx_ParCSRMatrix *jx_ParCSRMatrixCreate
 * \brief Create a parallel CSR Matrix.
 * \note If create is called for JX_NO_GLOBAL_PARTITION and row_starts and col_starts are NOT 
 *       null, then it is assumed that they are array of length 2 containing the start row of 
 *       the calling processor followed by the start row of the next processor. - AHB 6/05 
 * \date 2009/07/10
 */
jx_ParCSRMatrix *
jx_ParCSRMatrixCreate( MPI_Comm   comm,
                       JX_Int        global_num_rows,
                       JX_Int        global_num_cols,
                       JX_Int       *row_starts,
                       JX_Int       *col_starts,
                       JX_Int        num_cols_offd,
                       JX_Int        num_nonzeros_diag,
                       JX_Int        num_nonzeros_offd )
{
   jx_ParCSRMatrix  *matrix;
   
   JX_Int num_procs, my_id;
   JX_Int local_num_rows, local_num_cols;
   JX_Int first_row_index, first_col_diag;
   
   matrix = jx_CTAlloc(jx_ParCSRMatrix, 1);

   jx_MPI_Comm_rank(comm, &my_id);
   jx_MPI_Comm_size(comm, &num_procs);

   if (!row_starts)
   {
#ifdef JX_NO_GLOBAL_PARTITION  
      jx_GenerateLocalPartitioning(global_num_rows, num_procs, my_id, &row_starts);
#else
      jx_GeneratePartitioning(global_num_rows, num_procs, &row_starts);
#endif
   }
   
   if (!col_starts)
   {
      if (global_num_rows == global_num_cols)
      {
         col_starts = row_starts;
      }
      else
      {
#ifdef JX_NO_GLOBAL_PARTITION   
         jx_GenerateLocalPartitioning(global_num_cols, num_procs, my_id, &col_starts);
#else
         jx_GeneratePartitioning(global_num_cols, num_procs, &col_starts);
#endif
      }
   }


#ifdef JX_NO_GLOBAL_PARTITION
   /* row_starts[0] is start of local rows.  row_starts[1] is start of next processor's rows */
   first_row_index = row_starts[0];
   local_num_rows  = row_starts[1] - first_row_index ;
   first_col_diag  = col_starts[0];
   local_num_cols  = col_starts[1] - first_col_diag;
#else
   first_row_index = row_starts[my_id];
   local_num_rows  = row_starts[my_id+1] - first_row_index;
   first_col_diag  = col_starts[my_id];
   local_num_cols  = col_starts[my_id+1] - first_col_diag;
#endif


   jx_ParCSRMatrixComm(matrix) = comm;
   jx_ParCSRMatrixDiag(matrix) = jx_CSRMatrixCreate(local_num_rows, local_num_cols, num_nonzeros_diag);
   jx_ParCSRMatrixOffd(matrix) = jx_CSRMatrixCreate(local_num_rows, num_cols_offd, num_nonzeros_offd);
   jx_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   jx_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   jx_ParCSRMatrixFirstRowIndex(matrix) = first_row_index;
   jx_ParCSRMatrixFirstColDiag(matrix)  = first_col_diag;
 
   jx_ParCSRMatrixLastRowIndex(matrix) = first_row_index + local_num_rows - 1;
   jx_ParCSRMatrixLastColDiag(matrix)  = first_col_diag  + local_num_cols - 1;

   jx_ParCSRMatrixColMapOffd(matrix)       = NULL;
   jx_ParCSRMatrixAssumedPartition(matrix) = NULL;


  /*------------------------------------------------------------------------- 
   *   When NO_GLOBAL_PARTITION is set we could make these null, instead
   * of leaving the range.  If that change is made, then when this create
   * is called from functions like the matrix-matrix multiply, be careful
   * not to generate a new partition.
   *-------------------------------------------------------------------------*/

   jx_ParCSRMatrixRowStarts(matrix) = row_starts;
   jx_ParCSRMatrixColStarts(matrix) = col_starts;

   jx_ParCSRMatrixDiagT(matrix) = NULL;
   jx_ParCSRMatrixOffdT(matrix) = NULL;

   jx_ParCSRMatrixCommPkg(matrix)  = NULL;
   jx_ParCSRMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   jx_ParCSRMatrixOwnsData(matrix) = 1;
   jx_ParCSRMatrixOwnsRowStarts(matrix) = 1;
   jx_ParCSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   {
      jx_ParCSRMatrixOwnsColStarts(matrix) = 0;
   }
   jx_ParCSRMatrixRowindices(matrix)   = NULL;
   jx_ParCSRMatrixRowvalues(matrix)    = NULL;
   jx_ParCSRMatrixGetrowactive(matrix) = 0;

   return matrix;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixInitialize
 * \brief Initialize a parallel CSR Matrix. 
 * \date 2009/07/10
 */
JX_Int 
jx_ParCSRMatrixInitialize( jx_ParCSRMatrix *matrix )
{
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_CSRMatrixInitialize(jx_ParCSRMatrixDiag(matrix));
   jx_CSRMatrixInitialize(jx_ParCSRMatrixOffd(matrix));
   jx_ParCSRMatrixColMapOffd(matrix) 
   = jx_CTAlloc(JX_Int, jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(matrix)));

   return jx_error_flag;
}


/*!
 * \fn jx_CSRMatrix *jx_ParCSRMatrixToCSRMatrixAll
 * \brief Generates a CSRMatrix from a ParCSRMatrix on all  
 *        processors that have parts of the ParCSRMatrix. 
 * \date 2009/07/10
 */
jx_CSRMatrix *
jx_ParCSRMatrixToCSRMatrixAll( jx_ParCSRMatrix *par_matrix )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(par_matrix);
   jx_CSRMatrix *matrix;
   jx_CSRMatrix *local_matrix;
   JX_Int num_rows = jx_ParCSRMatrixGlobalNumRows(par_matrix);
   JX_Int num_cols = jx_ParCSRMatrixGlobalNumCols(par_matrix);
#ifndef JX_NO_GLOBAL_PARTITION
   JX_Int *row_starts = jx_ParCSRMatrixRowStarts(par_matrix);
#endif
   JX_Int     *matrix_i;
   JX_Int     *matrix_j;
   JX_Real  *matrix_data;
  
   JX_Int    *local_matrix_i;
   JX_Int    *local_matrix_j;
   JX_Real *local_matrix_data;
  
   JX_Int  i, j;
   JX_Int  local_num_rows;
   JX_Int  local_num_nonzeros;
   JX_Int  num_nonzeros;
   JX_Int  num_data;
   JX_Int  num_requests;
   JX_Int  vec_len, offset;
   JX_Int  start_index;
   JX_Int  proc_id;
   JX_Int  num_procs, my_id;
   JX_Int  num_types;
   JX_Int *used_procs;

   MPI_Request *requests;
   MPI_Status *status;

#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int *new_vec_starts;
   JX_Int  num_contacts;
   JX_Int  contact_proc_list[1];
   JX_Int  contact_send_buf[1];
   JX_Int  contact_send_buf_starts[2];
   JX_Int  max_response_size;
   JX_Int *response_recv_buf = NULL;
   JX_Int *response_recv_buf_starts = NULL;
   
   jx_DataExchangeResponse response_obj;
   jx_ProcListElements send_proc_obj;
   
   JX_Int *send_info = NULL;
   MPI_Status  status1;
   JX_Int count, tag1 = 11112, tag2 = 22223, tag3 = 33334;
   JX_Int start;
#endif

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);


#ifdef JX_NO_GLOBAL_PARTITION

   local_num_rows = jx_ParCSRMatrixLastRowIndex(par_matrix)  - 
                    jx_ParCSRMatrixFirstRowIndex(par_matrix) + 1;
   

   local_matrix   = jx_MergeDiagAndOffd(par_matrix); /* creates matrix */
   local_matrix_i = jx_CSRMatrixI(local_matrix);
   local_matrix_j = jx_CSRMatrixJ(local_matrix);
   local_matrix_data = jx_CSRMatrixData(local_matrix);

 
  /*-----------------------------------------------------------------------------
   * Determine procs that have vector data and store their ids in used_procs
   * we need to do an exchange data for this. If I own row then I will contact
   * processor 0 with the endpoint of my local range。
   *-----------------------------------------------------------------------------*/

   if (local_num_rows > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0]  = jx_ParCSRMatrixLastRowIndex(par_matrix);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;
   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }
   
  /*--------------------------------------------------------------
   * build the response object
   * send_proc_obj will  be for saving info from contacts 
   *------------------------------------------------------------*/
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = jx_CTAlloc(JX_Int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = jx_CTAlloc(JX_Int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements = jx_CTAlloc(JX_Int, send_proc_obj.element_storage_length);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = jx_FillResponseParToCSRMatrix;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /* this is where we keep info from contacts */
  
   jx_DataExchangeList( num_contacts, 
                        contact_proc_list, contact_send_buf, 
                        contact_send_buf_starts, sizeof(JX_Int), 
                        sizeof(JX_Int), &response_obj, 
                        max_response_size, 1,
                        comm, (void**) &response_recv_buf,	   
                        &response_recv_buf_starts );
   

  /*-------------------------------------------------------------------------------
   * now processor 0 should have a list of ranges for processors that have rows -
   * these are in send_proc_obj - it needs to create the new list of processors
   * and also an array of vec starts - and send to those who own row.
   *-------------------------------------------------------------------------------*/
   
   if (my_id)
   {
      if (local_num_rows)      
      {
         /* look for a message from processor 0 */         
         jx_MPI_Probe(0, tag1, comm, &status1);
         jx_MPI_Get_count(&status1, JX_MPI_INT, &count);
         
         send_info = jx_CTAlloc(JX_Int, count);
         jx_MPI_Recv(send_info, count, JX_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */  
         num_types = send_info[0];
         used_procs = jx_CTAlloc(JX_Int, num_types);  
         new_vec_starts = jx_CTAlloc(JX_Int, num_types+1);

         for (i = 1; i <= num_types; i ++)
         {
            used_procs[i-1] = send_info[i];
         }
         for (i = num_types + 1; i < count; i ++)
         {
            new_vec_starts[i-num_types-1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         jx_TFree(send_proc_obj.vec_starts);
         jx_TFree(send_proc_obj.id);
         jx_TFree(send_proc_obj.elements);
         if(response_recv_buf)        jx_TFree(response_recv_buf);
         if(response_recv_buf_starts) jx_TFree(response_recv_buf_starts);

         if (jx_CSRMatrixOwnsData(local_matrix))
         {
            jx_CSRMatrixDestroy(local_matrix);
         }
         else
         {
            jx_TFree(local_matrix);
         }
         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types = send_proc_obj.length;
      used_procs = jx_CTAlloc(JX_Int, num_types);  
      new_vec_starts = jx_CTAlloc(JX_Int, num_types+1);
      
      new_vec_starts[0] = 0;
      for (i = 0; i < num_types; i ++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i+1] = send_proc_obj.elements[i] + 1;
      }
      jx_qsort0(used_procs, 0, num_types-1);
      jx_qsort0(new_vec_starts, 0, num_types);
      
      /*now we need to put into an array to send */
      count = 2*num_types + 2;
      send_info = jx_CTAlloc(JX_Int, count);
      send_info[0] = num_types;
      for (i = 1; i <= num_types; i ++)
      {
         send_info[i] = used_procs[i-1];
      }
      for (i = num_types + 1; i < count; i ++)
      {
         send_info[i] = new_vec_starts[i-num_types-1];
      }
      requests = jx_CTAlloc(MPI_Request, num_types);
      status   = jx_CTAlloc(MPI_Status, num_types);

      /* don't send to myself - these are sorted so my id would be first */
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }
   
      for (i = start; i < num_types; i ++)
      {
         jx_MPI_Isend(send_info, count, JX_MPI_INT, used_procs[i], tag1, comm, &requests[i-start]);
      }
      jx_MPI_Waitall(num_types-start, requests, status);

      jx_TFree(status);
      jx_TFree(requests);
   }
   
   /* clean up */
   jx_TFree(send_proc_obj.vec_starts);
   jx_TFree(send_proc_obj.id);
   jx_TFree(send_proc_obj.elements);
   jx_TFree(send_info);
   if(response_recv_buf)        jx_TFree(response_recv_buf);
   if(response_recv_buf_starts) jx_TFree(response_recv_buf_starts);

   /* now proc 0 can exit if it has no rows */
   if (!local_num_rows) 
   { 
      if (jx_CSRMatrixOwnsData(local_matrix))
      {
         jx_CSRMatrixDestroy(local_matrix);
      }
      else
      {
         jx_TFree(local_matrix);
      }
      
      jx_TFree(new_vec_starts);
      jx_TFree(used_procs);

      return NULL;
   }
   
   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

   /* this matrix should be rather small */
   matrix_i = jx_CTAlloc(JX_Int, num_rows+1);

   num_requests = 4*num_types;
   requests = jx_CTAlloc(MPI_Request, num_requests);
   status = jx_CTAlloc(MPI_Status, num_requests);


   /* exchange contents of local_matrix_i - here we are sending to ourself also */

   j = 0;
   for (i = 0; i < num_types; i ++)
   {
      proc_id = used_procs[i];
      vec_len = new_vec_starts[i+1] - new_vec_starts[i];
      jx_MPI_Irecv( &matrix_i[new_vec_starts[i]+1], vec_len, JX_MPI_INT,
                 proc_id, tag2, comm, &requests[j++] );
   }
   for (i = 0; i < num_types; i ++)
   {
      proc_id = used_procs[i];
      jx_MPI_Isend( &local_matrix_i[1], local_num_rows, JX_MPI_INT,
                 proc_id, tag2, comm, &requests[j++] );
   }

   jx_MPI_Waitall(j, requests, status);

   /* generate matrix_i from received data */
   /* global numbering? */
   offset = matrix_i[new_vec_starts[1]];
   for (i = 1; i < num_types; i ++)
   {
      for (j = new_vec_starts[i]; j < new_vec_starts[i+1]; j ++)
      {
         matrix_i[j+1] += offset;
      }
      offset = matrix_i[new_vec_starts[i+1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixI(matrix) = matrix_i;
   jx_CSRMatrixInitialize(matrix);
   matrix_j = jx_CSRMatrixJ(matrix);
   matrix_data = jx_CSRMatrixData(matrix);

  /*-------------------------------------------------------------- 
   * Generate datatypes for further data exchange and exchange 
   * remaining data, i.e. column info and actual data. 
   *-------------------------------------------------------------*/

   j = 0;
   for (i = 0; i < num_types; i ++)
   {
        proc_id = used_procs[i];
        start_index = matrix_i[new_vec_starts[i]];
        num_data = matrix_i[new_vec_starts[i+1]] - start_index; 
        jx_MPI_Irecv( &matrix_data[start_index], num_data, JX_MPI_REAL,
                   used_procs[i], tag1, comm, &requests[j++] );
        jx_MPI_Irecv( &matrix_j[start_index], num_data, JX_MPI_INT,
                   used_procs[i], tag3, comm, &requests[j++] );
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   for (i = 0; i < num_types; i ++)
   {
        jx_MPI_Isend( local_matrix_data, local_num_nonzeros, JX_MPI_REAL,
                   used_procs[i], tag1, comm, &requests[j++] );
        jx_MPI_Isend( local_matrix_j, local_num_nonzeros, JX_MPI_INT,
                   used_procs[i], tag3, comm, &requests[j++] );
   }

   jx_MPI_Waitall(num_requests, requests, status);

   jx_TFree(new_vec_starts);
   
#else

   local_num_rows = row_starts[my_id+1] - row_starts[my_id];

   /* if my_id contains no data, return NULL */
 
   if (!local_num_rows)
   {
      return NULL;
   }     
 
   local_matrix   = jx_MergeDiagAndOffd(par_matrix);
   local_matrix_i = jx_CSRMatrixI(local_matrix);
   local_matrix_j = jx_CSRMatrixJ(local_matrix);
   local_matrix_data = jx_CSRMatrixData(local_matrix);

   matrix_i = jx_CTAlloc(JX_Int, num_rows + 1);

  /* determine procs that have vector data and store their ids in used_procs */

   num_types = 0;
   for (i = 0; i < num_procs; i ++)
   {
      if (row_starts[i+1] - row_starts[i] && i - my_id)
      {
         num_types ++;
      }
   }
   num_requests = 4*num_types;

   used_procs = jx_CTAlloc(JX_Int, num_types);
   j = 0;
   for (i = 0; i < num_procs; i ++)
   {
      if (row_starts[i+1] - row_starts[i] && i - my_id)
      {
         used_procs[j++] = i;
      }
   }

   requests = jx_CTAlloc(MPI_Request, num_requests);
   status = jx_CTAlloc(MPI_Status, num_requests);

   /* exchange contents of local_matrix_i */

   j = 0;
   for (i = 0; i < num_types; i ++)
   {
      proc_id = used_procs[i];
      vec_len = row_starts[proc_id+1] - row_starts[proc_id];
      jx_MPI_Irecv( &matrix_i[row_starts[proc_id]+1], vec_len, JX_MPI_INT,
                 proc_id, 0, comm, &requests[j++] );
   }
   for (i = 0; i < num_types; i ++)
   {
      proc_id = used_procs[i];
      jx_MPI_Isend( &local_matrix_i[1], local_num_rows, JX_MPI_INT,
                 proc_id, 0, comm, &requests[j++] );
   }

   vec_len = row_starts[my_id+1] - row_starts[my_id];
   for (i = 1; i <= vec_len; i ++)
   {
      matrix_i[row_starts[my_id]+i] = local_matrix_i[i];
   }

   jx_MPI_Waitall(j, requests, status);

   /* generate matrix_i from received data */

   offset = matrix_i[row_starts[1]];
   for (i = 1; i < num_procs; i ++)
   {
      for (j = row_starts[i]; j < row_starts[i+1]; j ++)
      {
         matrix_i[j+1] += offset;
      }
      offset = matrix_i[row_starts[i+1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixI(matrix) = matrix_i;
   jx_CSRMatrixInitialize(matrix);
   matrix_j = jx_CSRMatrixJ(matrix);
   matrix_data = jx_CSRMatrixData(matrix);

  /*-------------------------------------------------------------
   * Generate datatypes for further data exchange and exchange 
   * remaining data, i.e. column info and actual data 
   *------------------------------------------------------------*/

   j = 0;
   for (i = 0; i < num_types; i ++)
   {
      proc_id = used_procs[i];
      start_index = matrix_i[row_starts[proc_id]];
      num_data = matrix_i[row_starts[proc_id+1]] - start_index; 
      jx_MPI_Irecv( &matrix_data[start_index], num_data, JX_MPI_REAL,
                 used_procs[i], 0, comm, &requests[j++] );
      jx_MPI_Irecv( &matrix_j[start_index], num_data, JX_MPI_INT,
                 used_procs[i], 0, comm, &requests[j++] );
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   for (i = 0; i < num_types; i ++)
   {
      jx_MPI_Isend( local_matrix_data, local_num_nonzeros, JX_MPI_REAL,
                 used_procs[i], 0, comm, &requests[j++] );
      jx_MPI_Isend( local_matrix_j, local_num_nonzeros, JX_MPI_INT,
                 used_procs[i], 0, comm, &requests[j++] );
   }

   start_index = matrix_i[row_starts[my_id]];
   for (i = 0; i < local_num_nonzeros; i ++)
   {
      matrix_j[start_index+i] = local_matrix_j[i];
      matrix_data[start_index+i] = local_matrix_data[i];
   }
   jx_MPI_Waitall(num_requests, requests, status);

   start_index = matrix_i[row_starts[my_id]];
   for (i = 0; i < local_num_nonzeros; i ++)
   {
      matrix_j[start_index+i] = local_matrix_j[i];
      matrix_data[start_index+i] = local_matrix_data[i];
   }
   jx_MPI_Waitall(num_requests, requests, status);

#endif

   if (jx_CSRMatrixOwnsData(local_matrix))
   {
      jx_CSRMatrixDestroy(local_matrix);
   }
   else
   {
      jx_TFree(local_matrix);
   }

   if (num_requests)
   {
      jx_TFree(requests);
      jx_TFree(status);
      jx_TFree(used_procs);
   }

   return matrix;
}

/*!
 * \fn jx_ParCSRMatrix *jx_CSRMatrixToParCSRMatrix
 * \brief Generates a ParCSRMatrix distributed across the 
 *        processors in comm from a CSRMatrix on proc 0.
 * \note This shouldn't be used with the JX_NO_GLOBAL_PARTITON option!  
 * \date 2009/07/10
 */
jx_ParCSRMatrix *
jx_CSRMatrixToParCSRMatrix( MPI_Comm       comm,
                            jx_CSRMatrix  *A,
                            JX_Int           *row_starts,
                            JX_Int           *col_starts )
{
   JX_Int         *global_data;
   JX_Int          global_size;
   JX_Int          global_num_rows;
   JX_Int          global_num_cols;
   JX_Int         *local_num_rows;

   JX_Int          num_procs, my_id;
   JX_Int         *local_num_nonzeros = NULL;
   JX_Int          num_nonzeros;
  
   JX_Real       *a_data = NULL;
   JX_Int          *a_i    = NULL;
   JX_Int          *a_j    = NULL;
  
   jx_CSRMatrix *local_A;

   MPI_Request  *requests;
   MPI_Status   *status, status0;
   MPI_Datatype *csr_matrix_datatypes;

   jx_ParCSRMatrix *par_matrix;

   JX_Int first_col_diag;
   JX_Int last_col_diag;
 
   JX_Int i, j, ind;

   jx_MPI_Comm_rank(comm, &my_id);
   jx_MPI_Comm_size(comm, &num_procs);

   global_data = jx_CTAlloc(JX_Int, 2*num_procs + 6);
   if (my_id == 0) 
   {
      global_size = 3;
      if (row_starts) 
      {
	 if (col_starts)
	 {
	    if (col_starts != row_starts)
	    {
              /*----------------------------------------------------------
               * contains code for what to expect, 
               * if 0: row_starts = col_starts, only row_starts given
               * if 1: only row_starts given, col_starts = NULL
               * if 2: both row_starts and col_starts given 
               * if 3: only col_starts given, row_starts = NULL 
               *---------------------------------------------------------*/
               global_data[3] = 2;
               global_size = 2*num_procs+6;
               for (i = 0; i < num_procs + 1; i ++)
               {
                  global_data[i+4] = row_starts[i];
               }
               for (i = 0; i < num_procs + 1; i ++)
               {
                  global_data[i+num_procs+5] = col_starts[i];
               }
	    }
	    else
	    {
               global_data[3] = 0;
               global_size = num_procs+5;
               for (i = 0; i < num_procs + 1; i ++)
               {
                  global_data[i+4] = row_starts[i];
               }
	    }
	 }
	 else
	 {
            global_data[3] = 1;
            global_size = num_procs+5;
	    for (i = 0; i < num_procs + 1; i ++)
	    {
	       global_data[i+4] = row_starts[i];
	    }
	 }
      }
      else 
      {
	 if (col_starts)
	 {
            global_data[3] = 3;
	    global_size = num_procs + 5;
	    for (i = 0; i < num_procs + 1; i ++)
	    {
	       global_data[i+4] = col_starts[i];
	    }
	 }
      }
      global_data[0] = jx_CSRMatrixNumRows(A);
      global_data[1] = jx_CSRMatrixNumCols(A);
      global_data[2] = global_size;
      a_data = jx_CSRMatrixData(A);
      a_i = jx_CSRMatrixI(A);
      a_j = jx_CSRMatrixJ(A);
   }
   jx_MPI_Bcast(global_data, 3, JX_MPI_INT, 0, comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];

   global_size = global_data[2];
   if (global_size > 3)
   {
      jx_MPI_Bcast(&global_data[3], global_size - 3, JX_MPI_INT, 0, comm);
      if (my_id > 0)
      {
	 if (global_data[3] < 3)
	 {
	    row_starts = jx_CTAlloc(JX_Int, num_procs + 1);
	    for (i = 0; i < num_procs + 1; i ++)
	    {
	       row_starts[i] = global_data[i+4];
	    }
	    if (global_data[3] == 0)
	    {
	       col_starts = row_starts;
	    }
	    if (global_data[3] == 2)
	    {
	       col_starts = jx_CTAlloc(JX_Int, num_procs+1);
	       for (i = 0; i < num_procs + 1; i ++)
	       {
	          col_starts[i] = global_data[i+num_procs+5];
	       }
	    }
	 }
	 else
	 {
	    col_starts = jx_CTAlloc(JX_Int, num_procs + 1);
	    for (i = 0; i < num_procs + 1; i ++)
	    {
	       col_starts[i] = global_data[i+4];
	    }
	 }
      }
   }
   jx_TFree(global_data);

   local_num_rows = jx_CTAlloc(JX_Int, num_procs);
   csr_matrix_datatypes = jx_CTAlloc(MPI_Datatype, num_procs);

   par_matrix = jx_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts, 0, 0, 0);

   row_starts = jx_ParCSRMatrixRowStarts(par_matrix);
   col_starts = jx_ParCSRMatrixColStarts(par_matrix);

   for (i = 0; i < num_procs; i ++)
   {
      local_num_rows[i] = row_starts[i+1] - row_starts[i];
   }

   if (my_id == 0)
   {
      local_num_nonzeros = jx_CTAlloc(JX_Int, num_procs);
      for (i = 0; i < num_procs - 1; i ++)
      {
         local_num_nonzeros[i] = a_i[row_starts[i+1]] - a_i[row_starts[i]];
      }
      local_num_nonzeros[num_procs-1] = a_i[global_num_rows] - a_i[row_starts[num_procs-1]];
   }
   jx_MPI_Scatter(local_num_nonzeros, 1, JX_MPI_INT, &num_nonzeros, 1, JX_MPI_INT, 0, comm);

   if (my_id == 0) 
   {
      num_nonzeros = local_num_nonzeros[0];
   }

   local_A = jx_CSRMatrixCreate(local_num_rows[my_id], global_num_cols, num_nonzeros);

   if (my_id == 0)
   {
        requests = jx_CTAlloc (MPI_Request, num_procs - 1);
        status = jx_CTAlloc(MPI_Status, num_procs - 1);
        j = 0;
        for ( i = 1; i < num_procs; i ++)
        {
                ind = a_i[row_starts[i]];
                jx_BuildCSRMatrixMPIDataType( local_num_nonzeros[i], 
                                              local_num_rows[i],
                                              &a_data[ind],
                                              &a_i[row_starts[i]],
                                              &a_j[ind],
                                              &csr_matrix_datatypes[i] );
                jx_MPI_Isend( MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm, &requests[j++] );
                jx_MPI_Type_free(&csr_matrix_datatypes[i]);
        }
        jx_CSRMatrixData(local_A) = a_data;
        jx_CSRMatrixI(local_A) = a_i;
        jx_CSRMatrixJ(local_A) = a_j;
        jx_CSRMatrixOwnsData(local_A) = 0;
        jx_MPI_Waitall(num_procs-1, requests, status);
        jx_TFree(requests);
        jx_TFree(status);
        jx_TFree(local_num_nonzeros);
   }
   else
   {
        jx_CSRMatrixInitialize(local_A);
        jx_BuildCSRMatrixMPIDataType( num_nonzeros, 
                                      local_num_rows[my_id],
                                      jx_CSRMatrixData(local_A),
                                      jx_CSRMatrixI(local_A),
                                      jx_CSRMatrixJ(local_A),
                                      csr_matrix_datatypes );
        jx_MPI_Recv( MPI_BOTTOM, 1, csr_matrix_datatypes[0], 0, 0, comm, &status0 );
        jx_MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag  = col_starts[my_id+1] - 1;

   jx_GenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {      
      jx_CSRMatrixData(local_A) = NULL;
      jx_CSRMatrixI(local_A) = NULL;
      jx_CSRMatrixJ(local_A) = NULL; 
   }      
   jx_CSRMatrixDestroy(local_A);
   jx_TFree(local_num_rows);
   jx_TFree(csr_matrix_datatypes);

   return par_matrix;
}

/*!
 * \fn jx_ParCSRMatrix *jx_CSRMatrixToParCSRMatrix_FromGivenPro
 * \brief Generates a ParCSRMatrix distributed across the 
 *        processors in comm from a CSRMatrix on proc srcid.
 * \note This shouldn't be used with the JX_NO_GLOBAL_PARTITON option!
 * \note 在jx_CSRMatrixToParCSRMatrix的基础上改写的程序还有问题，这里重新写.
 * \author peghoty  
 * \date 2012/03/03
 */
jx_ParCSRMatrix *
jx_CSRMatrixToParCSRMatrix_FromGivenPro( MPI_Comm       comm,
                                         JX_Int            srcid,
                                         jx_CSRMatrix  *A,
                                         JX_Int           *row_starts,
                                         JX_Int           *col_starts )
{
   JX_Int         *global_data = NULL;
   JX_Int          global_num_rows;
   JX_Int          global_num_cols;
   JX_Int         *local_num_rows = NULL;
   JX_Int         *local_num_nonzeros = NULL;
   
   JX_Int          num_procs, my_id;
   JX_Int          num_rows;
   JX_Int          num_nonzeros;
  
   JX_Real       *a_data = NULL;
   JX_Int          *a_i    = NULL;
   JX_Int          *a_j    = NULL;
  
   jx_CSRMatrix *local_A = NULL;

   MPI_Request  *requests;
   MPI_Status   *status;   
   MPI_Status    status0;

   jx_ParCSRMatrix *par_matrix = NULL;

   JX_Int first_col_diag;
   JX_Int last_col_diag;
 
   JX_Int i, j, comsize;
   
   JX_Int start_ia, start_ja;
   JX_Int tag_ia = 111;
   JX_Int tag_ja = 222;
   JX_Int tag_aa = 333;

   jx_MPI_Comm_rank(comm, &my_id);
   jx_MPI_Comm_size(comm, &num_procs);
   
   comsize = 3*(num_procs - 1);

   global_data = jx_CTAlloc(JX_Int, 2);
   if (my_id == srcid) 
   {
      global_data[0] = jx_CSRMatrixNumRows(A);
      global_data[1] = jx_CSRMatrixNumCols(A);
      a_data = jx_CSRMatrixData(A);
      a_i = jx_CSRMatrixI(A);
      a_j = jx_CSRMatrixJ(A);
   }
   jx_MPI_Bcast(global_data, 2, JX_MPI_INT, srcid, comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];
   jx_TFree(global_data);
  
   local_num_rows = jx_CTAlloc(JX_Int, num_procs);
   local_num_nonzeros = jx_CTAlloc(JX_Int, num_procs);
   
   par_matrix = jx_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols, row_starts, col_starts, 0, 0, 0);

   row_starts = jx_ParCSRMatrixRowStarts(par_matrix);
   col_starts = jx_ParCSRMatrixColStarts(par_matrix);

   for (i = 0; i < num_procs; i ++)
   {
      local_num_rows[i] = row_starts[i+1] - row_starts[i];
   }
   
   if (my_id == srcid)
   {
      for (i = 0; i < num_procs; i ++)
      {
         local_num_nonzeros[i] = a_i[row_starts[i+1]] - a_i[row_starts[i]];
      }
   }
   jx_MPI_Bcast(local_num_nonzeros, num_procs, JX_MPI_INT, srcid, comm);

   num_nonzeros = local_num_nonzeros[my_id];
   num_rows = local_num_rows[my_id];

   local_A = jx_CSRMatrixCreate(num_rows, global_num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(local_A);

   if (my_id == srcid)
   {
        requests = jx_CTAlloc(MPI_Request, comsize);
        status = jx_CTAlloc(MPI_Status, comsize);        
        j = 0;
        for ( i = 0; i < num_procs; i ++)
        {
           if (i != srcid)
           {
              start_ia = row_starts[i];
              start_ja = a_i[start_ia]; 
              jx_MPI_Isend(&a_i[start_ia], local_num_rows[i]+1, JX_MPI_INT, i, tag_ia, comm, &requests[j++]);
              jx_MPI_Isend(&a_j[start_ja], local_num_nonzeros[i], JX_MPI_INT, i, tag_ja, comm, &requests[j++]);
              jx_MPI_Isend(&a_data[start_ja], local_num_nonzeros[i], JX_MPI_REAL, i, tag_aa, comm, &requests[j++]);        
           }     
        }

        start_ia = row_starts[my_id];
        start_ja = a_i[start_ia];
        for (i = 0; i <= num_rows; i ++)
        {
           jx_CSRMatrixI(local_A)[i] = a_i[start_ia+i]; 
        }
        for (i = 0; i < num_nonzeros; i ++)
        {
           jx_CSRMatrixData(local_A)[i] = a_data[start_ja + i];
           jx_CSRMatrixJ(local_A)[i] = a_j[start_ja + i];
        }
      
        jx_MPI_Waitall(comsize, requests, status);     

        jx_TFree(requests);
        jx_TFree(status);        
   }
   else
   {
      jx_MPI_Recv(jx_CSRMatrixI(local_A), num_rows+1, JX_MPI_INT, srcid, tag_ia, comm, &status0);
      jx_MPI_Recv(jx_CSRMatrixJ(local_A), num_nonzeros, JX_MPI_INT, srcid, tag_ja, comm, &status0);
      jx_MPI_Recv(jx_CSRMatrixData(local_A), num_nonzeros, JX_MPI_REAL, srcid, tag_aa, comm, &status0);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag  = col_starts[my_id+1] - 1;
   jx_GenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);

   jx_CSRMatrixDestroy(local_A);
   jx_TFree(local_num_rows);
   jx_TFree(local_num_nonzeros);

   return (par_matrix);  
}

/*!
 * \fn JX_Int jx_CSRMatrixToParCSRMatrix_sp
 * \brief Transfer a serial CSR matrix to a parallel one (only for single-processor case). 
 * \author peghoty 
 * \date 2011/09/27
 */
jx_ParCSRMatrix *
jx_CSRMatrixToParCSRMatrix_sp( MPI_Comm comm, jx_CSRMatrix *A )
{
   jx_ParCSRMatrix *par_matrix = NULL;
   
   JX_Int num_rows     = jx_CSRMatrixNumRows(A);
   JX_Int num_cols     = jx_CSRMatrixNumCols(A);
   JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);

   /* Create a paralle matrix */
   par_matrix = jx_ParCSRMatrixCreate(comm, num_rows, num_cols, NULL, NULL, 0, num_nonzeros, 0);   

   /* deal with the diag */
   jx_CSRMatrix *diag = jx_ParCSRMatrixDiag(par_matrix);
   jx_CSRMatrixData(diag) = jx_CSRMatrixData(A);
   jx_CSRMatrixI(diag)    = jx_CSRMatrixI(A);
   jx_CSRMatrixJ(diag)    = jx_CSRMatrixJ(A);
        
   /* deal with the offd */
   jx_CSRMatrix *offd = jx_ParCSRMatrixOffd(par_matrix);  
   JX_Int *offd_i = jx_CTAlloc(JX_Int, num_rows + 1);
   JX_Int i;
   for (i = 0; i < num_rows + 1; i ++)
   {
      offd_i[i] = 0;
   }
   jx_CSRMatrixNumCols(offd) = 0;
   jx_CSRMatrixI(offd) = offd_i;
   
   return (par_matrix);  
}  

/*!
 * \fn JX_Int jx_GenerateDiagAndOffd
 * \brief Generate the Diag and Offd for a parallel CSR matrix.
 * \date 2011/09/05
 */
JX_Int
jx_GenerateDiagAndOffd( jx_CSRMatrix     *A,
                        jx_ParCSRMatrix  *matrix,
                        JX_Int               first_col_diag,
                        JX_Int               last_col_diag )
{
   JX_Int  i, j;
   JX_Int  jo, jd;
   JX_Int  num_rows = jx_CSRMatrixNumRows(A);
   JX_Int  num_cols = jx_CSRMatrixNumCols(A);
   JX_Real *a_data = jx_CSRMatrixData(A);
   JX_Int *a_i = jx_CSRMatrixI(A);
   JX_Int *a_j = jx_CSRMatrixJ(A);

   jx_CSRMatrix *diag = jx_ParCSRMatrixDiag(matrix);
   jx_CSRMatrix *offd = jx_ParCSRMatrixOffd(matrix);

   JX_Int *col_map_offd;

   JX_Real *diag_data, *offd_data;
   JX_Int  *diag_i, *offd_i;
   JX_Int  *diag_j, *offd_j;
   JX_Int  *marker;
   JX_Int num_cols_diag, num_cols_offd;
   JX_Int first_elmt = a_i[0];
   JX_Int num_nonzeros = a_i[num_rows] - first_elmt;
   JX_Int counter;

   num_cols_diag = last_col_diag - first_col_diag + 1;
   num_cols_offd = 0;

   if (num_cols - num_cols_diag)
   {
        jx_CSRMatrixInitialize(diag);
        diag_i = jx_CSRMatrixI(diag);

        jx_CSRMatrixInitialize(offd);
        offd_i = jx_CSRMatrixI(offd);
        marker = jx_CTAlloc(JX_Int, num_cols);

        for (i = 0; i < num_cols; i ++)
        {
           marker[i] = 0;
        }
        
        jo = 0;
        jd = 0;
        for (i = 0; i < num_rows; i ++)
        {
            offd_i[i] = jo;
            diag_i[i] = jd;
   
            for (j = a_i[i] - first_elmt; j < a_i[i+1] - first_elmt; j ++)
            {
                if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
                {
                        if (!marker[a_j[j]])
                        {
                                marker[a_j[j]] = 1;
                                num_cols_offd ++;
                        }
                        jo ++;
                }
                else
                {
                        jd ++;
                }
            }
        }
        offd_i[num_rows] = jo;
        diag_i[num_rows] = jd;

        jx_ParCSRMatrixColMapOffd(matrix) = jx_CTAlloc(JX_Int, num_cols_offd);
        col_map_offd = jx_ParCSRMatrixColMapOffd(matrix);

        counter = 0;
        for (i = 0; i < num_cols; i ++)
        {
                if (marker[i])
                {
                        col_map_offd[counter] = i;
                        marker[i] = counter;
                        counter ++;
                }
        }
        jx_CSRMatrixNumNonzeros(diag) = jd;
        jx_CSRMatrixInitialize(diag);
        diag_data = jx_CSRMatrixData(diag);
        diag_j = jx_CSRMatrixJ(diag);

        jx_CSRMatrixNumNonzeros(offd) = jo;
        jx_CSRMatrixNumCols(offd) = num_cols_offd;
        jx_CSRMatrixInitialize(offd);
        offd_data = jx_CSRMatrixData(offd);
        offd_j = jx_CSRMatrixJ(offd);

        jo = 0;
        jd = 0;
        for (i = 0; i < num_rows; i ++)
        {
            for (j = a_i[i] - first_elmt; j < a_i[i+1] - first_elmt; j ++)
            {
                if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
                {
                        offd_data[jo] = a_data[j];
                        offd_j[jo++] = marker[a_j[j]];
                }
                else
                {
                        diag_data[jd] = a_data[j];
                        diag_j[jd++] = a_j[j] - first_col_diag;
                }
            }    
        }
        jx_TFree(marker);
   }
   else 
   {
        jx_CSRMatrixNumNonzeros(diag) = num_nonzeros;
        jx_CSRMatrixInitialize(diag);
        diag_data = jx_CSRMatrixData(diag);
        diag_i = jx_CSRMatrixI(diag);
        diag_j = jx_CSRMatrixJ(diag);

        for (i = 0; i < num_nonzeros; i ++)
        {
                diag_data[i] = a_data[i];
                diag_j[i] = a_j[i];
        }
        offd_i = jx_CTAlloc(JX_Int, num_rows + 1);

        for (i = 0; i < num_rows + 1; i ++)
        {
                diag_i[i] = a_i[i];
                offd_i[i] = 0;
        }

        jx_CSRMatrixNumCols(offd) = 0;
        jx_CSRMatrixI(offd) = offd_i;
   }
   
   return jx_error_flag;
}


/*!
 * \fn jx_CSRMatrix *jx_MergeDiagAndOffd
 * \brief Merge the Diag and Offd for a parallel CSR matrix.
 * \date 2011/09/08
 */
jx_CSRMatrix *
jx_MergeDiagAndOffd( jx_ParCSRMatrix *par_matrix )
{
   jx_CSRMatrix  *diag = jx_ParCSRMatrixDiag(par_matrix);
   jx_CSRMatrix  *offd = jx_ParCSRMatrixOffd(par_matrix);
   jx_CSRMatrix  *matrix;

   JX_Int           num_cols = jx_ParCSRMatrixGlobalNumCols(par_matrix);
   JX_Int           first_col_diag = jx_ParCSRMatrixFirstColDiag(par_matrix);
   JX_Int          *col_map_offd = jx_ParCSRMatrixColMapOffd(par_matrix);
   JX_Int           num_rows = jx_CSRMatrixNumRows(diag);

   JX_Int          *diag_i = jx_CSRMatrixI(diag);
   JX_Int          *diag_j = jx_CSRMatrixJ(diag);
   JX_Real       *diag_data = jx_CSRMatrixData(diag);
   JX_Int          *offd_i = jx_CSRMatrixI(offd);
   JX_Int          *offd_j = jx_CSRMatrixJ(offd);
   JX_Real       *offd_data = jx_CSRMatrixData(offd);

   JX_Int          *matrix_i;
   JX_Int          *matrix_j;
   JX_Real       *matrix_data;

   JX_Int          num_nonzeros, i, j;
   JX_Int          count;

   num_nonzeros = diag_i[num_rows] + offd_i[num_rows];

   matrix = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(matrix);

   matrix_i = jx_CSRMatrixI(matrix);
   matrix_j = jx_CSRMatrixJ(matrix);
   matrix_data = jx_CSRMatrixData(matrix);

   count = 0;
   matrix_i[0] = 0;
   for (i = 0; i < num_rows; i ++)
   {
        for (j = diag_i[i]; j < diag_i[i+1]; j ++)
        {
                matrix_data[count] = diag_data[j];
                matrix_j[count++] = diag_j[j] + first_col_diag;
        }
        for (j = offd_i[i]; j < offd_i[i+1]; j ++)
        {
                matrix_data[count] = offd_data[j];
                matrix_j[count++]  = col_map_offd[offd_j[j]];
        }
        matrix_i[i+1] = count;
   }

   return matrix;
}


/*!
 * \fn jx_ParCSRMatrix *jx_ParCSRMatrixRead
 * \brief Read a parallel matrix from given file(s).
 * \date 2011/09/08
 */
jx_ParCSRMatrix *
jx_ParCSRMatrixRead( MPI_Comm comm, const char *file_name, JX_Int file_base )
{
   jx_ParCSRMatrix  *matrix;
   jx_CSRMatrix  *diag;
   jx_CSRMatrix  *offd;
   JX_Int   my_id, i, num_procs;
   char  new_file_d[80], new_file_o[80], new_file_info[80];
   JX_Int   global_num_rows, global_num_cols, num_cols_offd;
   JX_Int   local_num_rows;
   JX_Int  *row_starts;
   JX_Int  *col_starts;
   JX_Int  *col_map_offd;
   FILE *fp;
   JX_Int   equal = 1;
   
#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int   row_s, row_e, col_s, col_e;
#endif
   

   jx_MPI_Comm_rank(comm,&my_id);
   jx_MPI_Comm_size(comm,&num_procs);
   
#ifdef JX_NO_GLOBAL_PARTITION
   row_starts = jx_CTAlloc(JX_Int, 2);
   col_starts = jx_CTAlloc(JX_Int, 2);
#else
   row_starts = jx_CTAlloc(JX_Int, num_procs + 1);
   col_starts = jx_CTAlloc(JX_Int, num_procs + 1);
#endif

   jx_sprintf(new_file_d,"%s.D.%d", file_name, my_id);
   jx_sprintf(new_file_o,"%s.O.%d", file_name, my_id);
   jx_sprintf(new_file_info,"%s.INFO.%d", file_name, my_id);
   fp = fopen(new_file_info, "r");
   jx_fscanf(fp, "%d", &global_num_rows);
   jx_fscanf(fp, "%d", &global_num_cols);
   jx_fscanf(fp, "%d", &num_cols_offd);
   
#ifdef JX_NO_GLOBAL_PARTITION
   /* the bgl input file should only contain the EXACT range for local processor */
   jx_fscanf(fp, "%d %d %d %d", &row_s, &row_e, &col_s, &col_e);
   row_starts[0] = row_s;
   row_starts[1] = row_e;
   col_starts[0] = col_s;
   col_starts[1] = col_e;
#else
   for (i = 0; i < num_procs; i ++)
   {
      jx_fscanf(fp, "%d %d", &row_starts[i], &col_starts[i]);
   }
   row_starts[num_procs] = global_num_rows;
   col_starts[num_procs] = global_num_cols;
#endif

   col_map_offd = jx_CTAlloc(JX_Int, num_cols_offd);

   for (i = 0; i < num_cols_offd; i ++)
   {
      jx_fscanf(fp, "%d", &col_map_offd[i]);
   }
        
   fclose(fp);


#ifdef JX_NO_GLOBAL_PARTITION
   for (i = 1; i >= 0; i --)
      if (row_starts[i] != col_starts[i])
      {
         equal = 0;
         break;
      }
#else
   for (i = num_procs; i >= 0; i --)
   {
        if (row_starts[i] != col_starts[i])
        {
                equal = 0;
                break;
        }
   }
#endif
   if (equal)
   {
        jx_TFree(col_starts);
        col_starts = row_starts;
   }


   diag = jx_CSRMatrixRead(new_file_d, file_base);
   local_num_rows = jx_CSRMatrixNumRows(diag);

   if (num_cols_offd)
   {
        offd = jx_CSRMatrixRead(new_file_o, file_base);
   }
   else
   {
        offd = jx_CSRMatrixCreate(local_num_rows, 0, 0);
        jx_CSRMatrixInitialize(offd);
   }

        
   matrix = jx_CTAlloc(jx_ParCSRMatrix, 1);
   
   jx_ParCSRMatrixComm(matrix) = comm;
   jx_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   jx_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   
#ifdef JX_NO_GLOBAL_PARTITION
   jx_ParCSRMatrixFirstRowIndex(matrix) = row_s;
   jx_ParCSRMatrixFirstColDiag(matrix)  = col_s;
   jx_ParCSRMatrixLastRowIndex(matrix)  = row_e - 1;
   jx_ParCSRMatrixLastColDiag(matrix)   = col_e - 1;
#else
   jx_ParCSRMatrixFirstRowIndex(matrix) = row_starts[my_id];
   jx_ParCSRMatrixFirstColDiag(matrix)  = col_starts[my_id];
   jx_ParCSRMatrixLastRowIndex(matrix)  = row_starts[my_id+1] - 1;
   jx_ParCSRMatrixLastColDiag(matrix)   = col_starts[my_id+1] - 1;
#endif

   jx_ParCSRMatrixRowStarts(matrix) = row_starts;
   jx_ParCSRMatrixColStarts(matrix) = col_starts;
   jx_ParCSRMatrixCommPkg(matrix)   = NULL;

   /* set defaults */
   jx_ParCSRMatrixOwnsData(matrix) = 1;
   jx_ParCSRMatrixOwnsRowStarts(matrix) = 1;
   jx_ParCSRMatrixOwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   {
      jx_ParCSRMatrixOwnsColStarts(matrix) = 0;
   }

   jx_ParCSRMatrixDiag(matrix) = diag;
   jx_ParCSRMatrixOffd(matrix) = offd;
   if (num_cols_offd)
   {
        jx_ParCSRMatrixColMapOffd(matrix) = col_map_offd;
   }
   else
   {
        jx_ParCSRMatrixColMapOffd(matrix) = NULL;
   }

   return matrix;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixPrint
 * \brief Print a parallel matrix into given file(s).
 * \date 2011/09/08
 */
JX_Int
jx_ParCSRMatrixPrint( jx_ParCSRMatrix *matrix, const char *file_name )
{
   MPI_Comm comm;
   JX_Int  global_num_rows;
   JX_Int  global_num_cols;
   JX_Int *col_map_offd;
#ifndef JX_NO_GLOBAL_PARTITION
   JX_Int *row_starts;
   JX_Int *col_starts;
#endif
   JX_Int   my_id, i, num_procs;
   char  new_file_d[80], new_file_o[80], new_file_info[80];
   FILE *fp;
   JX_Int num_cols_offd = 0;
#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int row_s, row_e, col_s, col_e;
#endif
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   comm = jx_ParCSRMatrixComm(matrix);
   global_num_rows = jx_ParCSRMatrixGlobalNumRows(matrix);
   global_num_cols = jx_ParCSRMatrixGlobalNumCols(matrix);
   col_map_offd = jx_ParCSRMatrixColMapOffd(matrix);
   
#ifndef JX_NO_GLOBAL_PARTITION
   row_starts = jx_ParCSRMatrixRowStarts(matrix);
   col_starts = jx_ParCSRMatrixColStarts(matrix);
#endif

   if (jx_ParCSRMatrixOffd(matrix))
   {
        num_cols_offd = jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(matrix));
   }

   jx_MPI_Comm_rank(comm, &my_id);
   jx_MPI_Comm_size(comm, &num_procs);
   
   jx_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   jx_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   jx_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   jx_CSRMatrixPrint(jx_ParCSRMatrixDiag(matrix), new_file_d);
   if (num_cols_offd != 0)
   {
      jx_CSRMatrixPrint(jx_ParCSRMatrixOffd(matrix),new_file_o);
   }
  
   fp = fopen(new_file_info, "w");
   jx_fprintf(fp, "%d\n", global_num_rows);
   jx_fprintf(fp, "%d\n", global_num_cols);
   jx_fprintf(fp, "%d\n", num_cols_offd);
   
#ifdef JX_NO_GLOBAL_PARTITION
   row_s = jx_ParCSRMatrixFirstRowIndex(matrix);
   row_e = jx_ParCSRMatrixLastRowIndex(matrix);
   col_s =  jx_ParCSRMatrixFirstColDiag(matrix);
   col_e =  jx_ParCSRMatrixLastColDiag(matrix);
   /* add 1 to the ends because this is a starts partition */
   jx_fprintf(fp, "%d %d %d %d\n", row_s, row_e + 1, col_s, col_e + 1);
#else
   for (i = 0; i < num_procs; i ++)
   {
      jx_fprintf(fp, "%d %d\n", row_starts[i], col_starts[i]);
   }
#endif

   for (i = 0; i < num_cols_offd; i ++)
   {
      jx_fprintf(fp, "%d\n", col_map_offd[i]);
   }
   fclose(fp);

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixDestroy
 * \brief Destroy a parallel CSR matrix.
 * \date 2011/09/08
 */
JX_Int 
jx_ParCSRMatrixDestroy(jx_ParCSRMatrix *matrix)
{
   if (matrix)
   {
      if ( jx_ParCSRMatrixOwnsData(matrix) )
      {
         jx_CSRMatrixDestroy(jx_ParCSRMatrixDiag(matrix));
         jx_CSRMatrixDestroy(jx_ParCSRMatrixOffd(matrix));
         if (jx_ParCSRMatrixDiagT(matrix))
         {
              jx_CSRMatrixDestroy(jx_ParCSRMatrixDiagT(matrix));
         }
         if (jx_ParCSRMatrixOffdT(matrix))
         {
              jx_CSRMatrixDestroy(jx_ParCSRMatrixOffdT(matrix));
         }
         if (jx_ParCSRMatrixColMapOffd(matrix))
         {
              jx_TFree(jx_ParCSRMatrixColMapOffd(matrix));
         }
         if (jx_ParCSRMatrixCommPkg(matrix))
         {
              jx_MatvecCommPkgDestroy(jx_ParCSRMatrixCommPkg(matrix));
         }
         if (jx_ParCSRMatrixCommPkgT(matrix))
         {
              jx_MatvecCommPkgDestroy(jx_ParCSRMatrixCommPkgT(matrix));
         }
      }
      if ( jx_ParCSRMatrixOwnsRowStarts(matrix) )
      {
          jx_TFree(jx_ParCSRMatrixRowStarts(matrix));
      }
      if ( jx_ParCSRMatrixOwnsColStarts(matrix) )
      {
          jx_TFree(jx_ParCSRMatrixColStarts(matrix));
      }

      jx_TFree(jx_ParCSRMatrixRowindices(matrix));
      jx_TFree(jx_ParCSRMatrixRowvalues(matrix));

      if (jx_ParCSRMatrixAssumedPartition(matrix))
      {
         jx_ParCSRMatrixDestroyAssumedPartition(matrix);
      }

      jx_TFree(matrix);
   }

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixGetLocalRange
 * \brief Returns the row numbers of the rows stored on this processor.
 *       "End" is actually the row number of the last row on this processor.
 * \date 2011/09/08
 */
JX_Int 
jx_ParCSRMatrixGetLocalRange( jx_ParCSRMatrix   *matrix,
                              JX_Int               *row_start,
                              JX_Int               *row_end,
                              JX_Int               *col_start,
                              JX_Int               *col_end )
{  
   JX_Int my_id;

   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(matrix), &my_id);

#ifdef JX_NO_GLOBAL_PARTITION
  *row_start = jx_ParCSRMatrixFirstRowIndex(matrix);
  *row_end   = jx_ParCSRMatrixLastRowIndex(matrix);
  *col_start = jx_ParCSRMatrixFirstColDiag(matrix);
  *col_end   = jx_ParCSRMatrixLastColDiag(matrix);
#else
   *row_start = jx_ParCSRMatrixRowStarts(matrix)[my_id];
   *row_end   = jx_ParCSRMatrixRowStarts(matrix)[my_id + 1] - 1;
   *col_start = jx_ParCSRMatrixColStarts(matrix)[my_id];
   *col_end   = jx_ParCSRMatrixColStarts(matrix)[my_id + 1] - 1;
#endif

   return jx_error_flag;
}


JX_Int
JX_ParCSRMatrixGetRow( JX_ParCSRMatrix  matrix,
                       JX_Int              row,
                       JX_Int             *size,
                       JX_Int            **col_ind,
                       JX_Real         **values )
{
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParCSRMatrixGetRow((jx_ParCSRMatrix *) matrix, row, size, col_ind, values);
   
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixGetRow
 * \brief Returns global column indices and/or values for a given row in the global
 * matrix. Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the hypre_ParCSRMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.
 * Either one of col_ind and values can be left null, and those values will
 * not be returned.
 * All indices are returned in 0-based indexing, no matter what is used under
 * the hood. EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * \date 2013/01/23
 */
JX_Int 
jx_ParCSRMatrixGetRow( jx_ParCSRMatrix *matrix,
                       JX_Int row,
                       JX_Int *size,
                       JX_Int **col_ind,
                       JX_Real **values )
{
   JX_Int my_id;
   JX_Int row_start, row_end;
   jx_CSRMatrix *Aa;
   jx_CSRMatrix *Ba;
   
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   
   Aa = (jx_CSRMatrix *)jx_ParCSRMatrixDiag(matrix);
   Ba = (jx_CSRMatrix *)jx_ParCSRMatrixOffd(matrix);
   
   if (jx_ParCSRMatrixGetrowactive(matrix)) return -1;
   
   jx_MPI_Comm_rank(jx_ParCSRMatrixComm(matrix), &my_id);
   
   jx_ParCSRMatrixGetrowactive(matrix) = 1;
#ifdef JX_NO_GLOBAL_PARTITION
   row_start = jx_ParCSRMatrixFirstRowIndex(matrix);
   row_end = jx_ParCSRMatrixLastRowIndex(matrix) + 1;
#else
   row_end = jx_ParCSRMatrixRowStarts(matrix)[my_id+1];
   row_start = jx_ParCSRMatrixRowStarts(matrix)[my_id];
#endif
   if (row < row_start || row >= row_end) return -1;
   
   if (!jx_ParCSRMatrixRowvalues(matrix) && (col_ind || values))
   {
      JX_Int max = 1, tmp;
      JX_Int i;
      JX_Int m = row_end - row_start;
      
      for (i = 0; i < m; i ++)
      {
         tmp = jx_CSRMatrixI(Aa)[i+1] - jx_CSRMatrixI(Aa)[i] + jx_CSRMatrixI(Ba)[i+1] - jx_CSRMatrixI(Ba)[i];
         if (max < tmp)
         {
            max = tmp;
         }
      }
      jx_ParCSRMatrixRowvalues(matrix) = (JX_Real *)jx_CTAlloc(JX_Real, max);
      jx_ParCSRMatrixRowindices(matrix) = (JX_Int *)jx_CTAlloc(JX_Int, max);
   }
   
   JX_Real *vworkA, *vworkB, *v_p;
   JX_Int i, *cworkA, *cworkB, cstart = jx_ParCSRMatrixFirstColDiag(matrix);
   JX_Int nztot, nzA, nzB, lrow = row - row_start;
   JX_Int *cmap, *idx_p;
   
   nzA = jx_CSRMatrixI(Aa)[lrow+1] - jx_CSRMatrixI(Aa)[lrow];
   cworkA = &(jx_CSRMatrixJ(Aa)[jx_CSRMatrixI(Aa)[lrow]]);
   vworkA = &(jx_CSRMatrixData(Aa)[jx_CSRMatrixI(Aa)[lrow]]);
   
   nzB = jx_CSRMatrixI(Ba)[lrow+1] - jx_CSRMatrixI(Ba)[lrow];
   cworkB = &(jx_CSRMatrixJ(Ba)[jx_CSRMatrixI(Ba)[lrow]]);
   vworkB = &(jx_CSRMatrixData(Ba)[jx_CSRMatrixI(Ba)[lrow]]);
   
   nztot = nzA + nzB;
   
   cmap = jx_ParCSRMatrixColMapOffd(matrix);
   
   if (values || col_ind)
   {
      if (nztot)
      {
         JX_Int imark = -1;
         
         if (values)
         {
           *values = v_p = jx_ParCSRMatrixRowvalues(matrix);
            for (i = 0; i < nzB; i ++)
            {
               if (cmap[cworkB[i]] < cstart) v_p[i] = vworkB[i];
               else break;
            }
            imark = i;
            for (i = 0; i < nzA; i ++) v_p[imark+i] = vworkA[i];
            for (i = imark; i < nzB; i ++) v_p[nzA+i] = vworkB[i];
         }
         if (col_ind)
         {
           *col_ind = idx_p = jx_ParCSRMatrixRowindices(matrix);
            if (imark > -1)
            {
               for (i = 0; i < imark; i ++)
               {
                  idx_p[i] = cmap[cworkB[i]];
               }
            }
            else
            {
               for (i = 0; i < nzB; i ++)
               {
                  if (cmap[cworkB[i]] < cstart) idx_p[i] = cmap[cworkB[i]];
                  else break;
               }
               imark = i;
            }
            for (i = 0; i < nzA; i ++) idx_p[imark+i] = cstart + cworkA[i];
            for (i = imark; i < nzB; i ++) idx_p[nzA+i] = cmap[cworkB[i]];
         }
      }
      else
      {
         if (col_ind) *col_ind = 0;
         if (values) *values = 0;
      }
   }
  *size = nztot;
   
   return jx_error_flag;
}


JX_Int
JX_ParCSRMatrixRestoreRow( JX_ParCSRMatrix  matrix,
                           JX_Int              row,
                           JX_Int             *size,
                           JX_Int            **col_ind,
                           JX_Real         **values )
{
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParCSRMatrixRestoreRow((jx_ParCSRMatrix *) matrix, row, size, col_ind, values);

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixRestoreRow
 * \brief Restore Row
 * \date 2013/01/23
 */
JX_Int 
jx_ParCSRMatrixRestoreRow( jx_ParCSRMatrix *matrix,
                           JX_Int row,
                           JX_Int *size,
                           JX_Int **col_ind,
                           JX_Real **values )
{  
   if (!jx_ParCSRMatrixGetrowactive(matrix))
   {
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }
   
   jx_ParCSRMatrixGetrowactive(matrix) = 0;
   
   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixCreateAssumedPartition
 * \brief  Each proc gets it own range. Then each needs to
 *         reconcile its actual range with its assumed
 *         range - the result is essentila a partition of its 
 *         assumed range - this is the assumed partition.
 * \date 2011/09/08
 */
JX_Int
jx_ParCSRMatrixCreateAssumedPartition( jx_ParCSRMatrix *matrix ) 
{
   JX_Int  global_num_cols;
   JX_Int  myid;
   JX_Int  row_start = 0, row_end = 0, col_start = 0, col_end = 0;

   MPI_Comm  comm;
   
   jx_IJAssumedPart *apart;

   global_num_cols = jx_ParCSRMatrixGlobalNumCols(matrix); 
   comm = jx_ParCSRMatrixComm(matrix);
   
   /* find out my actualy range of rows and columns */
   jx_ParCSRMatrixGetLocalRange(matrix, &row_start, &row_end , &col_start, &col_end);
   jx_MPI_Comm_rank(comm, &myid);

   /* allocate space */
   apart = jx_CTAlloc(jx_IJAssumedPart, 1);

   /* get my assumed partitioning - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   jx_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols, &(apart->row_start), &(apart->row_end));

  /* allocate some space for the partition of the assumed partition */
   apart->length = 0;
  /* room for 10 owners of the assumed partition */ 
   apart->storage_length = 10; /* need to be >=1 */ 
   apart->proc_list      = jx_TAlloc(JX_Int, apart->storage_length);
   apart->row_start_list = jx_TAlloc(JX_Int, apart->storage_length);
   apart->row_end_list   = jx_TAlloc(JX_Int, apart->storage_length);

  /* now we want to reconcile our actual partition with the assumed partition */
   jx_LocateAssummedPartition(comm, col_start, col_end, 0, global_num_cols, apart, myid);

  /* this partition will be saved in the matrix data structure until the matrix is destroyed */
   jx_ParCSRMatrixAssumedPartition(matrix) = apart;

   return jx_error_flag;
}


#define CONTACT(a,b)  (contact_list[(a)*3+(b)])

/*!
 * \fn JX_Int jx_LocateAssummedPartition
 * \brief Reconcile assumed partition with actual partition. Essentially
 *        each processor ends of with a partition of its assumed partition.
 * \date 2011/09/08
 */
JX_Int 
jx_LocateAssummedPartition( MPI_Comm          comm,
                            JX_Int               row_start,
                            JX_Int               row_end,
                            JX_Int               global_first_row,
                            JX_Int               global_num_rows,
                            jx_IJAssumedPart *part,
                            JX_Int               myid )
{
   JX_Int        i;

   JX_Int       *contact_list;
   JX_Int        contact_list_length, contact_list_storage;
   
   JX_Int        contact_row_start[2], contact_row_end[2], contact_ranges;
   JX_Int        owner_start, owner_end;
   JX_Int        tmp_row_start, tmp_row_end, complete;

//   JX_Int        locate_row_start[2], locate_ranges;
   JX_Int        locate_ranges;

   JX_Int        locate_row_count, rows_found;  
 
   JX_Int        tmp_range[2];
   JX_Int       *si, *sortme;   

   const JX_Int  flag1 = 17;  

   MPI_Request  *requests;
   MPI_Status   status0, *statuses;

  /*------------------------------------------------------------------
   *  Contact ranges - which rows do I have that others  are assumed
   *  responsible for? (at most two ranges - maybe none)
   *----------------------------------------------------------------*/

   contact_row_start[0] = 0;
   contact_row_end[0] = 0;
   contact_row_start[1] = 0;
   contact_row_end[1] = 0;
   contact_ranges = 0;
 
   if (row_start <= row_end ) /* must own at least one row */
   { 
      /* no overlap - so all of my rows and only one range */
      if ( part->row_end < row_start  || row_end < part->row_start )   
      {  
         contact_row_start[0] = row_start;
         contact_row_end[0] = row_end;
         contact_ranges ++;
      }
      else /* the two regions overlap - so one or two ranges */
      {  
         /* check for contact rows on the low end of the local range */
	 if (row_start < part->row_start) 
	 {
            contact_row_start[0] = row_start;
	    contact_row_end[0]   = part->row_start - 1;
	    contact_ranges ++;
	 } 
	 if (part->row_end < row_end) /* check the high end */
         {
            if (contact_ranges) /* already found one range */ 
            {
	       contact_row_start[1] = part->row_end + 1;
	       contact_row_end[1]   = row_end;
	    }
	    else
	    {
	       contact_row_start[0] = part->row_end + 1;
	       contact_row_end[0]   = row_end;
	    } 
	    contact_ranges ++;
	 }
      }
   }

  /*-----------------------------------------------------------
   *  Contact: find out who is assumed responsible for  
   *  these ranges of contact rows and contact them 
   *-----------------------------------------------------------*/
   contact_list_length = 0;
   contact_list_storage = 5; 
   contact_list = jx_TAlloc(JX_Int, contact_list_storage*3); /* each contact needs 3 ints */

   for (i = 0; i < contact_ranges; i ++)
   {  
      /* get start and end row owners */
      jx_GetAssumedPartitionProcFromRow(comm, contact_row_start[i], global_first_row,
                                        global_num_rows, &owner_start);
      jx_GetAssumedPartitionProcFromRow(comm, contact_row_end[i], global_first_row,
                                        global_num_rows, &owner_end);

      if (owner_start == owner_end) /* same processor owns the whole range */
      {

         if (contact_list_length == contact_list_storage)
         {
            /*allocate more space*/
            contact_list_storage += 5;
            contact_list = jx_TReAlloc(contact_list, JX_Int, (contact_list_storage*3));
         }
         CONTACT(contact_list_length, 0) = owner_start;           /* proc # */
         CONTACT(contact_list_length, 1) = contact_row_start[i];  /* start row */
         CONTACT(contact_list_length, 2) = contact_row_end[i];    /* end row */
         contact_list_length ++;
      }
      else
      { 
        complete = 0;
        while (!complete) 
        {
            jx_GetAssumedPartitionRowRange(comm, owner_start, global_first_row,
                                     global_num_rows, &tmp_row_start, &tmp_row_end);
            if (tmp_row_end >= contact_row_end[i])
	    {
	       tmp_row_end = contact_row_end[i];
               complete = 1; 
	    }     
            if (tmp_row_start < contact_row_start[i])
	    {
	      tmp_row_start = contact_row_start[i];
            }

            if (contact_list_length == contact_list_storage)
            {
               /*allocate more space*/
               contact_list_storage += 5;
               contact_list = jx_TReAlloc(contact_list, JX_Int, (contact_list_storage*3));
            }

            CONTACT(contact_list_length, 0) = owner_start;    /* proc # */
            CONTACT(contact_list_length, 1) = tmp_row_start;  /* start row */
            CONTACT(contact_list_length, 2) = tmp_row_end;    /* end row */
            contact_list_length ++;
	    owner_start ++;  /* processors are seqential */
        }
      }
   }

   requests = jx_CTAlloc(MPI_Request, contact_list_length);
   statuses = jx_CTAlloc(MPI_Status, contact_list_length);

   /* send out messages */
   for (i = 0; i < contact_list_length; i ++) 
   {
      jx_MPI_Isend( &CONTACT(i,1) ,2, JX_MPI_INT, CONTACT(i,0), flag1, comm, &requests[i] );
   }


  /*-----------------------------------------------------------
   *  Locate ranges - 
   *  which rows in my assumed range do I not own
   *  (at most two ranges - maybe none)
   *  locate_row_count = total number of rows I must locate
   *-----------------------------------------------------------*/
   
   locate_row_count    = 0;
   //locate_row_start[0] = 0;
   //locate_row_start[1] = 0;

   locate_ranges = 0;

   /* no overlap - so all of my assumed rows */
   if ( part->row_end < row_start  || row_end < part->row_start )
   {
      //locate_row_start[0] = part->row_start;
      locate_ranges ++;
      locate_row_count += part->row_end - part->row_start + 1; 
   }
   else /* the two regions overlap */
   {
      /* check for locate rows on the low end of the local range */
      if (part->row_start < row_start) 
      {
         //locate_row_start[0] = part->row_start;
         locate_ranges ++;
         locate_row_count += (row_start-1) - part->row_start + 1;
      } 
      if (row_end < part->row_end) /* check the high end */
      {
         if (locate_ranges) /* already have one range */ 
         {
	    //locate_row_start[1] = row_end + 1;
	 }
         else
         {
	    //locate_row_start[0] = row_end + 1;
         } 
         locate_ranges ++;
         locate_row_count += part->row_end - (row_end + 1) + 1;
      }
   }

   /*-----------------------------------------------------------
    * Receive messages from other procs telling us where
    * all our  locate rows actually reside 
    *-----------------------------------------------------------*/

    /* we will keep a partition of our assumed partition - list  
       ourselves first.  We will sort later with an additional index.
       In practice, this should only contain a few processors */
 
   /* which part do I own? */
   tmp_row_start = jx_max(part->row_start, row_start);
   tmp_row_end = jx_min(row_end, part->row_end);

   if (tmp_row_start <= tmp_row_end)
   {
      part->proc_list[0] = myid;
      part->row_start_list[0] = tmp_row_start;
      part->row_end_list[0] = tmp_row_end;
      part->length ++;
   }
  
   /* now look for messages that tell us which processor has our locate rows */
   /* these will be blocking receives as we know how many to expect and they 
      should be waiting (and we don't want to continue on without them) */

   rows_found = 0;

   while (rows_found != locate_row_count) 
   {
      jx_MPI_Recv( tmp_range, 2, JX_MPI_INT, MPI_ANY_SOURCE, flag1 , comm, &status0 );
     
      if (part->length == part->storage_length)
      {
	part->storage_length += 10;
        part->proc_list = jx_TReAlloc(part->proc_list, JX_Int, part->storage_length);
        part->row_start_list = jx_TReAlloc(part->row_start_list, JX_Int, part->storage_length);
        part->row_end_list   = jx_TReAlloc(part->row_end_list, JX_Int, part->storage_length);

      }
      part->row_start_list[part->length] = tmp_range[0];
      part->row_end_list[part->length] = tmp_range[1]; 

      part->proc_list[part->length] = status0.MPI_SOURCE;
      rows_found += tmp_range[1]- tmp_range[0] + 1;
      
      part->length ++;
   } 

   /* In case the partition of the assumed partition is longish, 
      we would like to know the sorted order */
   si = jx_CTAlloc(JX_Int, part->length); 
   sortme = jx_CTAlloc(JX_Int, part->length); 

   for (i = 0; i < part->length; i ++) 
   {
       si[i] = i;
       sortme[i] = part->row_start_list[i];
   }
   jx_qsort2i(sortme, si, 0, (part->length) - 1);
   part->sort_index = si;

   /* free the requests */
   jx_MPI_Waitall(contact_list_length, requests, statuses);

   jx_TFree(statuses);
   jx_TFree(requests);
   
   jx_TFree(sortme);
   jx_TFree(contact_list);

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_GetAssumedPartitionProcFromRow
 * \brief Assumed partition for IJ case. Given a particular row j, 
 *        return the processor that is assumed to own that row.
 * \date 2011/09/08
 */
JX_Int
jx_GetAssumedPartitionProcFromRow( MPI_Comm comm,
                                   JX_Int      row,
                                   JX_Int      global_first_row,
                                   JX_Int      global_num_rows,
                                   JX_Int     *proc_id )
{
   JX_Int num_procs;
   JX_Int size, switch_row, extra;
   
   jx_MPI_Comm_size(comm, &num_procs);
        
   /* this looks a bit odd, but we have to be very careful that
      this function and the next are inverses - and rounding 
      errors make this difficult!!!!! */

   size = global_num_rows / num_procs;
   extra = global_num_rows - size*num_procs;
   switch_row = global_first_row + (size + 1)*extra;
   
   if (row >= switch_row)
   {
      *proc_id = extra + (row - switch_row) / size;
   }
   else
   {
      *proc_id = (row - global_first_row)/(size+1);
   }

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_FillResponseParToCSRMatrix
 * \brief Fill response function for determining the send 
 *        processors data exchange.
 * \date 2011/09/08
 */
JX_Int
jx_FillResponseParToCSRMatrix( void      *p_recv_contact_buf, 
                               JX_Int        contact_size, 
                               JX_Int        contact_proc, 
                               void      *ro, 
                               MPI_Comm   comm, 
                               void     **p_send_response_buf, 
                               JX_Int       *response_message_size )
{
   JX_Int    myid, i, index, count, elength;
   JX_Int   *recv_contact_buf = (JX_Int *) p_recv_contact_buf;

   jx_DataExchangeResponse  *response_obj = ro;  
   jx_ProcListElements      *send_proc_obj = response_obj->data2;   

   jx_MPI_Comm_rank(comm, &myid);

   /* check to see if we need to allocate more space in send_proc_obj for ids */
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 10; /* add space for 10 more processors */
      send_proc_obj->id = jx_TReAlloc(send_proc_obj->id, JX_Int, send_proc_obj->storage_length);
      send_proc_obj->vec_starts 
      = jx_TReAlloc(send_proc_obj->vec_starts, JX_Int, send_proc_obj->storage_length + 1);
   }
  
   /* initialize */ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /* this is the number of elements */

   /* send proc */ 
   send_proc_obj->id[count] = contact_proc; 

   /* do we need more storage for the elements? */
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = jx_max(contact_size, 10);   
      elength += index;
      send_proc_obj->elements = jx_TReAlloc(send_proc_obj->elements, JX_Int, elength);
      send_proc_obj->element_storage_length = elength; 
   }
   
   /* populate send_proc_obj */
   for (i = 0; i < contact_size; i ++) 
   { 
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count+1] = index;
   send_proc_obj->length ++;
   
   /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixDestroyAssumedPartition
 * \date 2011/09/08
 */
JX_Int 
jx_ParCSRMatrixDestroyAssumedPartition( jx_ParCSRMatrix *matrix )
{
   jx_IJAssumedPart *apart;
   apart = jx_ParCSRMatrixAssumedPartition(matrix);
   
   if (apart->storage_length > 0) 
   {      
      jx_TFree(apart->proc_list);
      jx_TFree(apart->row_start_list);
      jx_TFree(apart->row_end_list);
      jx_TFree(apart->sort_index);
   }

   jx_TFree(apart);
   
   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixGetRowPartitioning
 * \author peghoty
 * \date 2009/07/10
 */ 
JX_Int
jx_ParCSRMatrixGetRowPartitioning( jx_ParCSRMatrix *matrix, JX_Int **row_partitioning_ptr )
{  
   JX_Int *row_partitioning, *row_starts;
   JX_Int  num_procs, i;

   if (!matrix) 
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_MPI_Comm_size(jx_ParCSRMatrixComm(matrix), &num_procs);
   row_starts = jx_ParCSRMatrixRowStarts(matrix);
   if (!row_starts) return -1;
   row_partitioning = jx_CTAlloc(JX_Int, num_procs + 1);
   for (i = 0; i < num_procs + 1; i ++)
   {
      row_partitioning[i] = row_starts[i];
   }

   *row_partitioning_ptr = row_partitioning;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixGetColPartitioning
 * \author peghoty
 * \date 2011/08/27
 */  
JX_Int
jx_ParCSRMatrixGetColPartitioning( jx_ParCSRMatrix *matrix, JX_Int **col_partitioning_ptr )
{  
   JX_Int *col_partitioning, *col_starts;
   JX_Int num_procs, i;

   if (!matrix) 
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_MPI_Comm_size(jx_ParCSRMatrixComm(matrix), &num_procs);
   col_starts = jx_ParCSRMatrixColStarts(matrix);
   if (!col_starts) return -1;
   col_partitioning = jx_CTAlloc(JX_Int, num_procs+1);
   for (i = 0; i < num_procs + 1; i ++)
   {
      col_partitioning[i] = col_starts[i];
   }

   *col_partitioning_ptr = col_partitioning;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixSetNumNonzeros
 * \date 2009/07/10
 */ 
JX_Int 
jx_ParCSRMatrixSetNumNonzeros( jx_ParCSRMatrix *matrix )
{
   MPI_Comm comm;
   jx_CSRMatrix *diag;
   JX_Int *diag_i;
   jx_CSRMatrix *offd;
   JX_Int *offd_i;
   JX_Int local_num_rows;
   JX_Int total_num_nonzeros;
   JX_Int local_num_nonzeros;
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   comm = jx_ParCSRMatrixComm(matrix);
   diag = jx_ParCSRMatrixDiag(matrix);
   diag_i = jx_CSRMatrixI(diag);
   offd = jx_ParCSRMatrixOffd(matrix);
   offd_i = jx_CSRMatrixI(offd);
   local_num_rows = jx_CSRMatrixNumRows(diag);

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   jx_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, JX_MPI_INT, MPI_SUM, comm);
   jx_ParCSRMatrixNumNonzeros(matrix) = total_num_nonzeros;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixSetDNumNonzeros
 * \date 2009/07/10
 */ 
JX_Int 
jx_ParCSRMatrixSetDNumNonzeros( jx_ParCSRMatrix *matrix )
{
   MPI_Comm comm;
   jx_CSRMatrix *diag;
   JX_Int *diag_i;
   jx_CSRMatrix *offd;
   JX_Int *offd_i;
   JX_Int local_num_rows;
   JX_Real total_num_nonzeros;
   JX_Real local_num_nonzeros;
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   comm = jx_ParCSRMatrixComm(matrix);
   diag = jx_ParCSRMatrixDiag(matrix);
   diag_i = jx_CSRMatrixI(diag);
   offd = jx_ParCSRMatrixOffd(matrix);
   offd_i = jx_CSRMatrixI(offd);
   local_num_rows = jx_CSRMatrixNumRows(diag);
   local_num_nonzeros = (JX_Real) diag_i[local_num_rows] + (JX_Real) offd_i[local_num_rows];
   jx_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, JX_MPI_REAL, MPI_SUM, comm);
   jx_ParCSRMatrixDNumNonzeros(matrix) = total_num_nonzeros;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixSetDataOwner
 * \date 2009/07/10
 */
JX_Int 
jx_ParCSRMatrixSetDataOwner( jx_ParCSRMatrix *matrix,
                             JX_Int              owns_data )
{
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_ParCSRMatrixOwnsData(matrix) = owns_data;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixSetRowStartsOwner
 * \date 2009/07/10
 */
JX_Int 
jx_ParCSRMatrixSetRowStartsOwner( jx_ParCSRMatrix *matrix, JX_Int owns_row_starts )
{
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_ParCSRMatrixOwnsRowStarts(matrix) = owns_row_starts;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_ParCSRMatrixSetColStartsOwner
 * \date 2009/07/10
 */
JX_Int 
jx_ParCSRMatrixSetColStartsOwner( jx_ParCSRMatrix *matrix, JX_Int owns_col_starts )
{
   if (!matrix)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_ParCSRMatrixOwnsColStarts(matrix) = owns_col_starts;

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_BuildCSRMatrixMPIDataType
 * \date 2009/07/10
 */
JX_Int
jx_BuildCSRMatrixMPIDataType( JX_Int            num_nonzeros, 
                              JX_Int            num_rows,
                              JX_Real        *a_data, 
                              JX_Int           *a_i, 
                              JX_Int           *a_j, 
                              MPI_Datatype  *csr_matrix_datatype )
{
   JX_Int		  block_lens[3];
   MPI_Aint	  displ[3];
   MPI_Datatype   types[3];

   block_lens[0] = num_nonzeros;
   block_lens[1] = num_rows+1;
   block_lens[2] = num_nonzeros;

   types[0] = JX_MPI_REAL;
   types[1] = JX_MPI_INT;
   types[2] = JX_MPI_INT;

   jx_MPI_Address(a_data, &displ[0]);
   jx_MPI_Address(a_i, &displ[1]);
   jx_MPI_Address(a_j, &displ[2]);
   jx_MPI_Type_struct(3,block_lens,displ,types,csr_matrix_datatype);
   jx_MPI_Type_commit(csr_matrix_datatype);

   return jx_error_flag;
}


/*!
 * \fn JX_Int jx_Build_Par_LinearSystem
 * \brief Build a parallel linear system by reading data
 *        from given files.
 * \author peghoty
 * \date 2009/08/28
 */
JX_Int      
jx_Build_Par_LinearSystem( MPI_Comm          comm, 
                           char             *filenamemat, 
                           char             *filenamerhs, 
                           jx_ParCSRMatrix **par_matrix_ptr,
                           jx_ParVector    **par_rhs_ptr )
{
     FILE   *fp = NULL;
     JX_Int     nrows,nonzeros;
     JX_Int     i;
     JX_Int    *ia,*ja;
     JX_Real *a,*f;

     JX_Int *row_part,*col_part;  /* 并行矩阵的行列分划 */
     JX_Int *partitioning;        /* 并行向量的分划，必须与并行矩阵的行分划一致 */

     jx_CSRMatrix    *matrix;
     jx_Vector       *rhs;

     jx_ParCSRMatrix *par_matrix;
     jx_ParVector    *par_rhs;

    /*---------------------------------------------------
     *     read the matrix from file 
     *--------------------------------------------------*/ 
     
     if((fp = fopen(filenamemat,"r")) == NULL)
     {
        jx_printf("Can't open the file %s !\n",filenamemat);
        exit(0);
     }

     jx_fscanf(fp,"%d\n", &nrows);
     ia = (JX_Int *)malloc((nrows+1)*sizeof(JX_Int));

     for (i = 0; i < nrows + 1; i ++)
     { 
        jx_fscanf(fp, "%d\n", &ia[i]); 
        ia[i] --; 
     }

     nonzeros = ia[nrows];
     ja = (JX_Int *)malloc(nonzeros*sizeof(JX_Int));
     a  = (JX_Real *)malloc(nonzeros*sizeof(JX_Real));
     
     for (i = 0; i < nonzeros; i ++)
     { 
        jx_fscanf(fp, "%d\n", &ja[i]); 
        ja[i] --;
     }

     for (i = 0; i < nonzeros; i ++) 
     {
        jx_fscanf(fp, "%le\n", &a[i]);
     }

     fclose(fp);

    /*---------------------------------------------------
     *     Read the rhs from file
     *--------------------------------------------------*/ 
     
     f = (JX_Real *)malloc(nrows*sizeof(JX_Real));
     if((fp = fopen(filenamerhs,"r")) == NULL)
     {
        jx_printf(" Can't open the file %s!\n", filenamerhs);
        exit(0);
     }
     jx_fscanf(fp, "%d\n", &nrows);
     for (i = 0; i < nrows; i ++) 
     {
        jx_fscanf(fp, "%lf\n", &f[i]);
     }
     fclose(fp);

    /*--------------------------------------------------------------------------
     * build a parallel matrix and rhs  
     *--------------------------------------------------------------------------*/
     
     /* 1. Make the diag element to be the first of each row */
     jx_PutDiagFirst(ia,ja,a,nrows);
     
     /* 2. build a parallel matrix */
     matrix = jx_CSRMatrixCreate(nrows, nrows, nonzeros);
     jx_CSRMatrixI(matrix) = ia;
     jx_CSRMatrixJ(matrix) = ja;
     jx_CSRMatrixData(matrix) = a;
     row_part = NULL;
     col_part = NULL;
     par_matrix = jx_CSRMatrixToParCSRMatrix(comm, matrix, row_part, col_part);
     
     /* 3. build a parallel rhs */
     rhs = jx_SeqVectorCreate(nrows);
     jx_VectorData(rhs) = f;
     jx_ParCSRMatrixGetRowPartitioning(par_matrix,&partitioning);
     par_rhs = jx_VectorToParVector(comm, rhs, partitioning);

     *par_matrix_ptr = par_matrix;
     *par_rhs_ptr = par_rhs;

     jx_TFree(matrix);
     jx_TFree(rhs);
  
      return 0;
}


/*!
 * \fn JX_Int jx_PutDiagFirst
 * \brief Reorder the matrix in order to make the diagonal entries
 *        are stored firstly in each row.
 * \author peghoty
 * \date 2009/06/17
 */
JX_Int 
jx_PutDiagFirst( JX_Int *ia, JX_Int *ja, JX_Real *a, JX_Int n )
{
   JX_Int i,j;
   JX_Int col,tmpcol;
   JX_Real tmpa;

   for (i = 0; i < n; i ++)
   {
       for (j = ia[i]; j < ia[i+1]; j ++)
       {
          col = ja[j];
          if(col == i && j != ia[i])
          {
             tmpcol    = ja[ia[i]];
             ja[ia[i]] = col;
             ja[j]     = tmpcol;
             
             tmpa     = a[ia[i]];
             a[ia[i]] = a[j];
             a[j]     = tmpa;
          }
       }
   }

   return 0;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixGetDims
 * \brief Get num_rows and num_cols.
 * \author Yue Xiaoqiang
 * \date 2013/01/23
 */
JX_Int
jx_ParCSRMatrixGetDims( jx_ParCSRMatrix *A, JX_Int *M, JX_Int *N )
{
   if (!A)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   
   *M = jx_ParCSRMatrixGlobalNumRows(A);
   *N = jx_ParCSRMatrixGlobalNumCols(A);
  
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParCSRMatrixGetComm
 * \brief Get comm.
 * \author Yue Xiaoqiang
 * \date 2013/02/27
 */
JX_Int
jx_ParCSRMatrixGetComm( jx_ParCSRMatrix *matrix, MPI_Comm *comm )
{
    if (!matrix)
    {
        jx_error_in_arg(1);
        return jx_error_flag;
    }
   *comm = jx_ParCSRMatrixComm(matrix);

    return jx_error_flag;
}
