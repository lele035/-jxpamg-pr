//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_csr_matrix.c -- operations for parallel matrices.
 *  Date: 2014/07/05
 */

#include "jx_mv.h"
#include "jx_ilu.h"

/*!
 * \fn jx_ParCSRMatrix *jx_BuildMatParFromOneFile2
 * \brief Build a parallel matrix by reading data from a given file,
 *        including reorder by variables and along y-axis
 * \param filename pointer to the matrix file.
 * \param row_part pointer to the row-start partitioning.
 * \param row_part pointer to the col-start partitioning.
 * \note The storage format of the matrix is expected to be CSR, i.e.
 *         the 1st row:                           n                          // JX_Int
 *         the 2 to (n+2)-th row:                ia[k]     k=0,1,...,n       // JX_Int
 *         the (n+3) to (n+nz+2)-th row:         ja[k]     k=0,1,...,nz-1    // JX_Int
 *         the (n+nz+3) to ((n+2nz+2))-th row:    a[k]     k=0,1,...,nz-1    // JX_Real 
 *       where n is the order of the matrix, nz is the number of non-zeros of the matrix.
 * \author Yue Xiaoqiang
 * \date 2014/04/10
 */
jx_ParCSRMatrix *
jx_BuildMatParFromOneFile2( char *filename, 
                            JX_Int   file_base,
                            JX_Int  *row_part, 
                            JX_Int  *col_part,
                            JX_Int   num_equns,
                            JX_Int   nx,
                            JX_Int   ny )
{
   jx_ParCSRMatrix  *A     = NULL;
   jx_CSRMatrix     *A_CSR = NULL;
   jx_CSRMatrix     *B_CSR = NULL;

   JX_Int my_id;
   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (my_id == 0)
   {
      A_CSR = jx_CSRMatrixRead(filename, file_base);
//      jx_CSRMatrixReorder(A_CSR);
      B_CSR = jx_CSRMatrixTDMGReorderByNodes(A_CSR, num_equns);
//      jx_CSRMatrixReorder(B_CSR);
      jx_CSRMatrixDestroy(A_CSR);
      A_CSR = jx_CSRMatrixTDMGReorderAlongY(B_CSR, num_equns, nx, ny);
      jx_CSRMatrixDestroy(B_CSR);
      jx_CSRMatrixReorder(A_CSR);
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
 * \fn jx_ParCSRMatrix *jx_BuildMatParFromOneFile3
 * \brief Build a parallel matrix by reading data from a given file,
 *        including reorder by variables and along y-axis, block-partitioned
 * \param filename pointer to the matrix file.
 * \param num_equns number of equations.
 * \param nx number of segments in x-direction.
 * \param ny number of segments in y-direction.
 * \note The storage format of the matrix is expected to be CSR, i.e.
 *         the 1st row:                           n                          // JX_Int
 *         the 2 to (n+2)-th row:                ia[k]     k=0,1,...,n       // JX_Int
 *         the (n+3) to (n+nz+2)-th row:         ja[k]     k=0,1,...,nz-1    // JX_Int
 *         the (n+nz+3) to ((n+2nz+2))-th row:    a[k]     k=0,1,...,nz-1    // JX_Real 
 *       where n is the order of the matrix, nz is the number of non-zeros of the matrix.
 * \author Yue Xiaoqiang
 * \date 2014/04/13
 */
jx_ParCSRMatrix *
jx_BuildMatParFromOneFile3( char *filename, JX_Int file_base, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
   jx_ParCSRMatrix  *A     = NULL;
   jx_CSRMatrix     *A_CSR = NULL;
   jx_CSRMatrix     *B_CSR = NULL;

   JX_Int my_id;
   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (my_id == 0)
   {
      A_CSR = jx_CSRMatrixRead(filename, file_base);
//      jx_CSRMatrixReorder(A_CSR);
      B_CSR = jx_CSRMatrixTDMGReorderByNodes(A_CSR, num_equns);
//      jx_CSRMatrixReorder(B_CSR);
      jx_CSRMatrixDestroy(A_CSR);
      A_CSR = jx_CSRMatrixTDMGReorderAlongY(B_CSR, num_equns, nx, ny);
      jx_CSRMatrixDestroy(B_CSR);
      jx_CSRMatrixReorder(A_CSR);
   }
   
   /* ser -> par transfering */
   A = jx_CSRMatrixToParCSRMatrix2(MPI_COMM_WORLD, A_CSR, num_equns, nx, ny);

   if (my_id == 0) 
   {
      jx_CSRMatrixDestroy(A_CSR);
   }

   return (A);
}

/*!
 * \fn jx_ParCSRMatrix *jx_ParCSRMatrixCreate2
 * \brief Create a parallel CSR Matrix.
 * \author Yue Xiaoqiang
 * \date 2014/04/13
 */
jx_ParCSRMatrix *
jx_ParCSRMatrixCreate2( MPI_Comm   comm,
                        JX_Int        global_num_rows,
                        JX_Int        global_num_cols,
                        JX_Int       *row_starts,
                        JX_Int       *col_starts,
                        JX_Int        num_cols_offd,
                        JX_Int        num_nonzeros_diag,
                        JX_Int        num_nonzeros_offd,
                        JX_Int        num_equns,
                        JX_Int        nx,
                        JX_Int        ny )
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
      jx_GeneratePartitioning2(num_procs, &row_starts, num_equns, nx, ny);
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
         jx_GeneratePartitioning2(num_procs, &col_starts, num_equns, nx, ny);
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
 * \fn jx_ParCSRMatrix *jx_CSRMatrixToParCSRMatrix2
 * \brief Generates a ParCSRMatrix distributed across the 
 *        processors in comm from a CSRMatrix on proc 0, block-partitioned
 * \note This shouldn't be used with the JX_NO_GLOBAL_PARTITON option
 * \author Yue Xiaoqiang
 * \date 2014/04/13
 */
jx_ParCSRMatrix *
jx_CSRMatrixToParCSRMatrix2( MPI_Comm comm, jx_CSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
   JX_Int         *global_data;
   JX_Int          global_size;
   JX_Int          global_num_rows;
   JX_Int          global_num_cols;
   JX_Int         *local_num_rows;
   JX_Int         *row_starts = NULL;
   JX_Int         *col_starts = NULL;

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

   par_matrix = jx_ParCSRMatrixCreate2(comm, global_num_rows, global_num_cols,
                                       row_starts, col_starts, 0, 0, 0, num_equns, nx, ny);

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
 * \fn jx_CSRMatrix *jx_MergeDiagAndOffdDropSmall
 * \brief Merge the Diag and Offd for a parallel CSR matrix, except small items
 * \author Yue Xiaoqiang
 * \date 2014/04/24
 */
jx_CSRMatrix *
jx_MergeDiagAndOffdDropSmall( jx_ParCSRMatrix *par_matrix, JX_Real drop_tol )
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
   
   JX_Real       tmpval;

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
      tmpval = drop_tol * fabs(diag_data[diag_i[i]]); // Treat the diagonal as the maximal
      for (j = diag_i[i]; j < diag_i[i+1]; j ++)
      {
         if (fabs(diag_data[j]) > tmpval)
         {
            matrix_data[count] = diag_data[j];
            matrix_j[count++] = diag_j[j] + first_col_diag;
         }
      }
      for (j = offd_i[i]; j < offd_i[i+1]; j ++)
      {
         if (fabs(offd_data[j]) > tmpval)
         {
            matrix_data[count] = offd_data[j];
            matrix_j[count++] = col_map_offd[offd_j[j]];
         }
      }
      matrix_i[i+1] = count;
   }
   jx_CSRMatrixNumNonzeros(matrix) = count; /* Reset num_nonzeros, while mat_j and mat_data needn't */

   return matrix;
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixMergeReorderIntUpRtDnLtQuasiBdy
 * \brief Merge and reorder in JX_Int-up-rgt-dwn-lft
 * \author Yue Xiaoqiang
 * \date 2014/08/25
 */
jx_CSRMatrix *
jx_CSRMatrixMergeReorderIntUpRtDnLtQuasiBdy( MPI_Comm              comm,
                                             JX_Int                  *row_starts,
                                             JX_Int                  *permute,
                                             jx_CSRMatrix         *ser_B,
                                             JX_Int                   nz_srt,
                                             JX_Int                   ng_pt,
                                             jx_GridPartitionData *grid_data,
                                             JX_Int                  *ex_len,
                                             JX_Int                  *ey_len,
                                             JX_Int                  *dx_len,
                                             JX_Int                  *kx_len,
                                             JX_Int                  *dy_len,
                                             JX_Int                  *lx_len,
                                             JX_Int                  *ly_len,
                                             JX_Int                  *postn_a,
                                             JX_Int                  *postn_b,
                                             JX_Int                  *postn_c )
{
    JX_Int num_rows = jx_CSRMatrixNumRows(ser_B);
    JX_Int num_cols = jx_CSRMatrixNumCols(ser_B);
    JX_Int *IB = jx_CSRMatrixI(ser_B);
    JX_Int *JB = jx_CSRMatrixJ(ser_B);
    JX_Real *BB = jx_CSRMatrixData(ser_B);
    
    JX_Int x_part_len = grid_data->x_part_len;
    JX_Int y_part_len = grid_data->y_part_len;
    JX_Int num_sideprocs = grid_data->num_nocrossside; // YUE: Needn't to consider the cross ones in 5-point stencil
    JX_Int num_smallside = grid_data->num_smallside;
    JX_Int num_largeside = grid_data->num_largeside;
    JX_Int *xlo_array = grid_data->xlo_array;
    JX_Int *xup_array = grid_data->xup_array;
    JX_Int *ylo_array = grid_data->ylo_array;
    JX_Int *yup_array = grid_data->yup_array;
    JX_Int *smallprocs = grid_data->sideprocs;
    JX_Int *largeprocs = smallprocs + num_smallside;
    JX_Int *smallprcpos = grid_data->sideprcpos;
    JX_Int *largeprcpos = smallprcpos + num_smallside;
    
    jx_CSRMatrix *ser_A = NULL;
    JX_Int *IA = NULL;
    JX_Int *JA = NULL;
    JX_Real *AA = NULL;
    
    JX_Int ng_pt_mx = ng_pt * x_part_len;
    JX_Int ng_pt_my = ng_pt * y_part_len;
    JX_Int ng_pt_m2x = 2 * ng_pt_mx;
    JX_Int ng_pt_m_px = ng_pt_mx + ng_pt;
    JX_Int ng_pt_m_xo = ng_pt_mx - ng_pt;
    
    JX_Int num_procs, my_id, recv_rows, rnt, cnt, ctt, i, j, k, l, pid, rele, col;
    JX_Int plle, pile, pkle, pkke, plme, nloc, nvar, xloc, yloc, row_end, pyln, pxln;
    JX_Int erow, jsrt, jend, pke, ple, psrt, pend, povr, psse, puue, ptte, pjje;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    
    if (grid_data->y_lower_idx == 0) // without DOWN
    {
        if (grid_data->x_lower_idx == 0) // without LEFT
        {
            recv_rows = 0;
           *ex_len = *ey_len = recv_rows;
            /* Attention the offsets */
            //jx_assert(nz_srt == 0);
            ser_A = jx_CSRMatrixCreate(num_rows+recv_rows, num_cols, jx_CSRMatrixNumNonzeros(ser_B)+nz_srt);
            jx_CSRMatrixInitialize(ser_A);
            IA = jx_CSRMatrixI(ser_A) + recv_rows;
            JA = jx_CSRMatrixJ(ser_A) + nz_srt;
            AA = jx_CSRMatrixData(ser_A) + nz_srt;
            IA[0] = nz_srt;
            rnt = cnt = ctt = 0;
            plle = row_starts[my_id];
            pkle = row_starts[my_id+1];
            for (j = 0; j < num_rows; j ++) // No connection with LEFT or DOWN
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= plle) && (col < pkle)) // its interior, up and right points
                    {
                        JA[ctt] = col;
                    }
                    else // connections with UP and RIGHT only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_largeside; l ++)
                        {
                            pid = largeprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (largeprcpos[l] == 1) // my_id as pid's DOWN
                                {
                                    if (xloc == 0) // left-down corner
                                    {
                                        erow = 1;
                                    }
                                    else
                                    {
                                        erow = pxln - xloc + 1;
                                        if (xlo_array[pid] != 0) // LEFT exists
                                        {
                                            erow += (pyln - 1);
                                        }
                                    }
                                    JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                    break;
                                }
                                else if (largeprcpos[l] == 3) // my_id as pid's LEFT
                                {
                                    if (yloc == 0) // left-down corner
                                    {
                                        erow = 1;
                                    }
                                    else
                                    {
                                        erow = pyln - yloc + 1;
                                    }
                                    JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                    break;
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
           *postn_a = *postn_b = *postn_c = cnt;
            //jx_assert(cnt == num_rows);
           *dx_len = ng_pt_mx + ng_pt_m_xo; // senting start to RIGHT for TYPE 3
           *kx_len = ng_pt_mx; // senting interval to RIGHT for TYPE 3
           *dy_len = num_rows - ng_pt_m_xo; // senting start to UP for TYPE 2
           *lx_len = ng_pt_m_xo; // senting start to RIGHT for TYPE 1
           *ly_len = num_rows - ng_pt_mx; // senting start to UP for TYPE 1
        }
        else // with LEFT
        {
            recv_rows = ng_pt_my;
           *ex_len = *ey_len = recv_rows;
            /* Attention the offsets */
            ser_A = jx_CSRMatrixCreate(num_rows+recv_rows, num_cols, jx_CSRMatrixNumNonzeros(ser_B)+nz_srt);
            jx_CSRMatrixInitialize(ser_A);
            IA = jx_CSRMatrixI(ser_A) + recv_rows;
            JA = jx_CSRMatrixJ(ser_A) + nz_srt;
            AA = jx_CSRMatrixData(ser_A) + nz_srt;
            IA[0] = nz_srt;
            rnt = cnt = ctt = 0;
            pke = row_starts[my_id];
            ple = num_rows - recv_rows - ng_pt;
            ptte = pke + ng_pt;
            plme = num_rows - ng_pt;
            pkle = row_starts[my_id+1];
            psrt = 0;
            for (i = 0; i < y_part_len; i ++)
            {
                jsrt = i * ng_pt_mx;
                jend = jsrt + ng_pt_mx;
                psrt += ng_pt;
                pend = psrt + ng_pt;
                povr = psrt - ng_pt;
                pile = ple - i * ng_pt_m_xo;
                pkke = pke + jsrt;
                plle = pkke + ng_pt;
                psse = pkke + ng_pt_mx;
                for (j = jsrt+ng_pt; j < jend; j ++) // No connection with LEFT or DOWN
                {
                    permute[rnt++] = j;
                    IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                    cnt ++;
                    row_end = IB[j+1];
                    for (k = IB[j]; k < row_end; k ++)
                    {
                        col = JB[k]; // globally
                        if ((col >= pke) && (col < ptte)) // sent its left-down corner points to back, firstly
                        {
                            JA[ctt] = col + plme;
                        }
                        else if ((col >= pkke) && (col < plle)) // sent its left points to back, secondly
                        {
                            JA[ctt] = col + pile;
                        }
                        else if ((col >= plle) && (col < psse)) // its interior points
                        {
                            JA[ctt] = col - psrt;
                        }
                        else if ((col >= psse) && (col < pkle)) // its up points
                        {
                            JA[ctt] = col - pend;
                        }
                        else if ((col >= ptte) && (col < pkke)) // its down points
                        {
                            JA[ctt] = col - povr;
                        }
                        else // connections with UP and RIGHT only, if any. Yue Xiaoqiang on 2014/08/15
                        {
                            // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                            for (l = 0; l < num_largeside; l ++)
                            {
                                pid = largeprocs[l];
                                if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                                {
                                    pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                    pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                    rele = col - row_starts[pid]; // local index
                                    nloc = rele / ng_pt; // local grid-block index
                                    nvar = rele - nloc * ng_pt; // local variable index
                                    yloc = nloc / pxln;
                                    xloc = nloc - yloc * pxln;
                                    if (largeprcpos[l] == 1) // my_id as pid's DOWN
                                    {
                                        if (xloc == 0) // left-down corner
                                        {
                                            erow = 1;
                                        }
                                        else
                                        {
                                            erow = pxln - xloc + 1;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow += (pyln - 1);
                                            }
                                        }
                                        JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                        break;
                                    }
                                    else if (largeprcpos[l] == 3) // my_id as pid's LEFT
                                    {
                                        if (yloc == 0)
                                        {
                                            erow = 1;
                                        }
                                        else
                                        {
                                            erow = pyln - yloc + 1;
                                        }
                                        JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                        break;
                                    }
                                }
                            }
                        }
                        AA[ctt++] = BB[k];
                    }
                }
            }
           *postn_a = *postn_b = cnt;
            psrt = ng_pt;
            for (i = 1; i < y_part_len; i ++)
            {
                jsrt = i * ng_pt_mx;
                jend = jsrt + ng_pt;
                psrt += ng_pt;
                pile = ple - i * ng_pt_m_xo;
                pend = pile - ng_pt_m_xo;
                povr = pile + ng_pt_m_xo;
                pkke = pke + jsrt;
                plle = pkke + ng_pt;
                psse = pkke + ng_pt_mx;
                for (j = jsrt; j < jend; j ++) // Connections with LEFT and UP
                {
                    permute[rnt++] = j;
                    IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                    cnt ++;
                    row_end = IB[j+1];
                    for (k = IB[j]; k < row_end; k ++)
                    {
                        col = JB[k]; // globally
                        if ((col >= pke) && (col < ptte)) // its left-down corner points
                        {
                            JA[ctt] = col + plme;
                        }
                        else if ((col >= pkke) && (col < plle)) // its interior points
                        {
                            JA[ctt] = col + pile;
                        }
                        else if ((col >= plle) && (col < psse)) // its right points
                        {
                            JA[ctt] = col - psrt;
                        }
                        else if ((col >= psse) && (col < pkle)) // its up points
                        {
                            JA[ctt] = col + pend;
                        }
                        else if ((col >= ptte) && (col < pkke)) // its down points
                        {
                            JA[ctt] = col + povr;
                        }
                        else // connections with LEFT and UP only, if any. Yue Xiaoqiang on 2014/08/15
                        {
                            // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                            for (l = 0; l < num_sideprocs; l ++)
                            {
                                pid = smallprocs[l];
                                if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                                {
                                    pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                    pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                    rele = col - row_starts[pid]; // local index
                                    nloc = rele / ng_pt; // local grid-block index
                                    nvar = rele - nloc * ng_pt; // local variable index
                                    yloc = nloc / pxln;
                                    xloc = nloc - yloc * pxln;
                                    if (smallprcpos[l] == 1) // my_id as pid's DOWN
                                    {
                                        if (xloc == 0) // left-down corner
                                        {
                                            erow = 1;
                                        }
                                        else
                                        {
                                            erow = pxln - xloc + 1;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow += (pyln - 1);
                                            }
                                        }
                                        JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                        break;
                                    }
                                    else if (smallprcpos[l] == 4) // my_id as pid's RIGHT
                                    {
                                        if (ylo_array[pid] == 0) // no DOWN
                                        {
                                            erow = nloc;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= (yloc + 1);
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                        else // DOWN exists
                                        {
                                            if (yloc == 0) // bottom-block of grid
                                            {
                                                erow = 2;
                                                if (xlo_array[pid] != 0) // LEFT exists
                                                {
                                                    erow += (pyln - 1);
                                                }
                                                JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                                break;
                                            }
                                            else
                                            {
                                                erow = nloc - pxln;
                                                if (xlo_array[pid] != 0) // LEFT exists
                                                {
                                                    erow -= yloc;
                                                }
                                                JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        AA[ctt++] = BB[k];
                    }
                }
            }
           *postn_c = cnt;
            pend = ple - ng_pt_m_xo;
            psse = pke + ng_pt_mx;
            for (j = 0; j < ng_pt; j ++) // Connections with LEFT only
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= pke) && (col < ptte)) // its interior points
                    {
                        JA[ctt] = col + plme;
                    }
                    else if ((col >= ptte) && (col < psse)) // its right points
                    {
                        JA[ctt] = col - ng_pt;
                    }
                    else if ((col >= psse) && (col < pkle)) // its up points
                    {
                        JA[ctt] = col + pend;
                    }
                    else // connections with LEFT only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_smallside; l ++)
                        {
                            pid = smallprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (smallprcpos[l] == 4) // my_id as pid's RIGHT
                                {
                                    if (ylo_array[pid] == 0) // no DOWN
                                    {
                                        erow = nloc;
                                        if (xlo_array[pid] != 0) // LEFT exists
                                        {
                                            erow -= (yloc + 1);
                                        }
                                        JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                        break;
                                    }
                                    else // DOWN exists
                                    {
                                        if (yloc == 0) // bottom-block of grid
                                        {
                                            erow = 2;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow += (pyln - 1);
                                            }
                                            JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                            break;
                                        }
                                        else
                                        {
                                            erow = nloc - pxln;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= yloc;
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
            //jx_assert(cnt == num_rows);
           *dx_len = 2 * ng_pt_m_xo - ng_pt; // senting start to RIGHT for TYPE 3
           *kx_len = ng_pt_m_xo; // senting interval to RIGHT for TYPE 3
           *dy_len = num_rows - ng_pt_m_xo - ng_pt_my; // senting start to UP for TYPE 2
           *lx_len = ng_pt_m_xo - ng_pt; // senting start to RIGHT for TYPE 1
           *ly_len = num_rows - 2 * ng_pt; // senting start to UP for TYPE 1
        }
    }
    else // with DOWN
    {
        if (grid_data->x_lower_idx == 0) // without LEFT
        {
            recv_rows = ng_pt_mx;
           *ex_len = *ey_len = recv_rows;
            /* Attention the offsets */
            ser_A = jx_CSRMatrixCreate(num_rows+recv_rows, num_cols, jx_CSRMatrixNumNonzeros(ser_B)+nz_srt);
            jx_CSRMatrixInitialize(ser_A);
            IA = jx_CSRMatrixI(ser_A) + recv_rows;
            JA = jx_CSRMatrixJ(ser_A) + nz_srt;
            AA = jx_CSRMatrixData(ser_A) + nz_srt;
            IA[0] = nz_srt;
            rnt = cnt = ctt = 0;
            pkke = row_starts[my_id];
            pjje = pkke + ng_pt;
            plle = pkke + recv_rows;
            ptte = num_rows - ng_pt;
            pile = ptte - recv_rows;
            pkle = row_starts[my_id+1];
            for (j = recv_rows; j < num_rows; j ++) // No connection with DOWN
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= pkke) && (col < pjje)) // sent its left-down corner points to back
                    {
                        JA[ctt] = col + ptte;
                    }
                    else if ((col >= pjje) && (col < plle)) // sent its down points to back
                    {
                        JA[ctt] = col + pile;
                    }
                    else if ((col >= plle) && (col < pkle)) // its interior, up and right points
                    {
                        JA[ctt] = col - recv_rows;
                    }
                    else // connections with UP and RIGHT only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_largeside; l ++)
                        {
                            pid = largeprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (largeprcpos[l] == 1) // my_id as pid's DOWN
                                {
                                    if (xloc == 0) // left-down corner
                                    {
                                        erow = 1;
                                    }
                                    else
                                    {
                                        erow = pxln - xloc + 1;
                                        if (xlo_array[pid] != 0) // LEFT exists
                                        {
                                            erow += (pyln - 1);
                                        }
                                    }
                                    JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                    break;
                                }
                                else if (largeprcpos[l] == 3) // my_id as pid's LEFT
                                {
                                    if (yloc == 0)
                                    {
                                        erow = 1;
                                    }
                                    else
                                    {
                                        erow = pyln - yloc + 1;
                                    }
                                    JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                    break;
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
           *postn_a = cnt;
            for (j = ng_pt; j < recv_rows; j ++) // Connections with DOWN and RIGHT
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= pkke) && (col < pjje)) // its left-down corner points
                    {
                        JA[ctt] = col + ptte;
                    }
                    else if ((col >= pjje) && (col < plle)) // its interior points
                    {
                        JA[ctt] = col + pile;
                    }
                    else if ((col >= plle) && (col < pkle)) // its up points
                    {
                        JA[ctt] = col - recv_rows;
                    }
                    else // connections with RIGHT and DOWN only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_sideprocs; l ++)
                        {
                            pid = smallprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (smallprcpos[l] == 3) // my_id as pid's LEFT
                                {
                                    if (yloc == 0)
                                    {
                                        erow = 1;
                                    }
                                    else
                                    {
                                        erow = pyln - yloc + 1;
                                    }
                                    JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                    break;
                                }
                                else if (smallprcpos[l] == 2) // my_id as pid's UP
                                {
                                    if (ylo_array[pid] == 0) // no DOWN
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_mx + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= (yloc + 1);
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                    else // DOWN exists
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_m2x + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc - ng_pt_mx;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= yloc;
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
           *postn_b = *postn_c = cnt;
            for (j = 0; j < ng_pt; j ++) // Connections with DOWN only
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= pkke) && (col < pjje)) // its interior points
                    {
                        JA[ctt] = col + ptte;
                    }
                    else if ((col >= pjje) && (col < plle)) // its right points
                    {
                        JA[ctt] = col + pile;
                    }
                    else if ((col >= plle) && (col < pkle)) // its up points
                    {
                        JA[ctt] = col - recv_rows;
                    }
                    else // connections with DOWN only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_smallside; l ++)
                        {
                            pid = smallprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (smallprcpos[l] == 2) // my_id as pid's UP
                                {
                                    if (ylo_array[pid] == 0) // no DOWN
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_mx + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= (yloc + 1);
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                    else // DOWN exists
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_m2x + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc - ng_pt_mx;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= yloc;
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
            //jx_assert(cnt == num_rows);
           *dx_len = ng_pt_m_xo; // senting start to RIGHT for TYPE 3
           *kx_len = ng_pt_mx; // senting interval to RIGHT for TYPE 3
           *dy_len = num_rows - ng_pt_m_xo - ng_pt_mx; // senting start to UP for TYPE 2
           *lx_len = num_rows - 2 * ng_pt; // senting start to RIGHT for TYPE 1
           *ly_len = num_rows - 2 * ng_pt_mx; // senting start to UP for TYPE 1
        }
        else // with LEFT
        {
            recv_rows = ng_pt_mx + ng_pt_my;
           *ex_len = recv_rows;
           *ey_len = recv_rows - ng_pt; // YUE: exclude the reduplicate block
            /* Attention the offsets */
            ser_A = jx_CSRMatrixCreate(num_rows+recv_rows, num_cols, jx_CSRMatrixNumNonzeros(ser_B)+nz_srt);
            jx_CSRMatrixInitialize(ser_A);
            IA = jx_CSRMatrixI(ser_A) + recv_rows;
            JA = jx_CSRMatrixJ(ser_A) + nz_srt;
            AA = jx_CSRMatrixData(ser_A) + nz_srt;
            IA[0] = nz_srt;
            rnt = cnt = ctt = 0;
            pke = row_starts[my_id];
            ple = num_rows - recv_rows;
            pile = ple + ng_pt_m_xo;
            plme = num_rows - ng_pt;
            pkle = row_starts[my_id+1];
            psrt = ng_pt_mx;
            puue = pke + ng_pt;
            ptte = pke + ng_pt_mx;
            for (i = 1; i < y_part_len; i ++)
            {
                jsrt = i * ng_pt_mx;
                jend = jsrt + ng_pt_mx;
                psrt += ng_pt;
                pend = psrt + ng_pt;
                povr = psrt - ng_pt;
                pile -= ng_pt_m_xo;
                pkke = pke + jsrt;
                plle = pkke + ng_pt;
                psse = pkke + ng_pt_mx;
                for (j = jsrt+ng_pt; j < jend; j ++) // No connection with DOWN or LEFT
                {
                    permute[rnt++] = j;
                    IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                    cnt ++;
                    row_end = IB[j+1];
                    for (k = IB[j]; k < row_end; k ++)
                    {
                        col = JB[k]; // globally
                        if ((col >= pkke) && (col < plle)) // sent its left points to back, without LEFT
                        {
                            JA[ctt] = col + pile;
                        }
                        else if ((col >= plle) && (col < psse)) // its interior points
                        {
                            JA[ctt] = col - psrt;
                        }
                        else if ((col >= psse) && (col < pkle)) // its up points
                        {
                            JA[ctt] = col - pend;
                        }
                        else if ((col >= puue) && (col < ptte)) // its downest points
                        {
                            JA[ctt] = col + ple;
                        }
                        else if ((col >= ptte) && (col < pkke)) // its down points
                        {
                            JA[ctt] = col - povr;
                        }
                        else // connections with UP and RIGHT only, if any. Yue Xiaoqiang on 2014/08/15
                        {
                            // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                            for (l = 0; l < num_largeside; l ++)
                            {
                                pid = largeprocs[l];
                                if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                                {
                                    pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                    pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                    rele = col - row_starts[pid]; // local index
                                    nloc = rele / ng_pt; // local grid-block index
                                    nvar = rele - nloc * ng_pt; // local variable index
                                    yloc = nloc / pxln;
                                    xloc = nloc - yloc * pxln;
                                    if (largeprcpos[l] == 1) // my_id as pid's DOWN
                                    {
                                        if (xloc == 0) // left-down corner
                                        {
                                            erow = 1;
                                        }
                                        else
                                        {
                                            erow = pxln - xloc + 1;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow += (pyln - 1);
                                            }
                                        }
                                        JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                        break;
                                    }
                                    else if (largeprcpos[l] == 3) // my_id as pid's LEFT
                                    {
                                        if (yloc == 0)
                                        {
                                            erow = 1;
                                        }
                                        else
                                        {
                                            erow = pyln - yloc + 1;
                                        }
                                        JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                        break;
                                    }
                                }
                            }
                        }
                        AA[ctt++] = BB[k];
                    }
                }
            }
           *postn_a = cnt;
            for (j = ng_pt; j < ng_pt_mx; j ++) // Connections with DOWN and RIGHT
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= puue) && (col < ptte)) // its interior points
                    {
                        JA[ctt] = col + ple;
                    }
                    else if ((col >= pke) && (col < puue)) // its leftest points
                    {
                        JA[ctt] = col + plme;
                    }
                    else if ((col >= ptte) && (col < pkle)) // its up points
                    {
                        JA[ctt] = col - ng_pt_m_px;
                    }
                    else // connections with DOWN and RIGHT only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_sideprocs; l ++)
                        {
                            pid = smallprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (smallprcpos[l] == 3) // my_id as pid's LEFT
                                {
                                    if (yloc == 0)
                                    {
                                        erow = 1;
                                    }
                                    else
                                    {
                                        erow = pyln - yloc + 1;
                                    }
                                    JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                    break;
                                }
                                else if (smallprcpos[l] == 2) // my_id as pid's UP
                                {
                                    if (ylo_array[pid] == 0) // no DOWN
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_mx + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= (yloc + 1);
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                    else // DOWN exists
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_m2x + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc - ng_pt_mx;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= yloc;
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
           *postn_b = cnt;
            pile = ple + ng_pt_m_xo;
            psrt = ng_pt_mx;
            for (i = 1; i < y_part_len; i ++)
            {
                jsrt = i * ng_pt_mx;
                jend = jsrt + ng_pt;
                psrt += ng_pt;
                pile -= ng_pt_m_xo;
                pend = pile - ng_pt_m_xo;
                povr = pile + ng_pt_m_xo;
                pkke = pke + jsrt;
                plle = pkke + ng_pt;
                psse = pkke + ng_pt_mx;
                for (j = jsrt; j < jend; j ++) // Connections with UP and LEFT
                {
                    permute[rnt++] = j;
                    IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                    cnt ++;
                    row_end = IB[j+1];
                    for (k = IB[j]; k < row_end; k ++)
                    {
                        col = JB[k]; // globally
                        if ((col >= pkke) && (col < plle)) // its interior points
                        {
                            JA[ctt] = col + pile;
                        }
                        else if ((col >= plle) && (col < psse)) // its right points
                        {
                            JA[ctt] = col - psrt;
                        }
                        else if ((col >= psse) && (col < pkle)) // its up points
                        {
                            JA[ctt] = col + pend;
                        }
                        else if ((col >= pke) && (col < puue)) // its downest points
                        {
                            JA[ctt] = col + plme;
                        }
                        else if ((col >= puue) && (col < pkke)) // its down points
                        {
                            JA[ctt] = col + povr;
                        }
                        else // connections with UP and LEFT only, if any. Yue Xiaoqiang on 2014/08/15
                        {
                            // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                            for (l = 0; l < num_sideprocs; l ++)
                            {
                                pid = smallprocs[l];
                                if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                                {
                                    pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                    pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                    rele = col - row_starts[pid]; // local index
                                    nloc = rele / ng_pt; // local grid-block index
                                    nvar = rele - nloc * ng_pt; // local variable index
                                    yloc = nloc / pxln;
                                    xloc = nloc - yloc * pxln;
                                    if (smallprcpos[l] == 1) // my_id as pid's DOWN
                                    {
                                        if (xloc == 0) // left-down corner
                                        {
                                            erow = 1;
                                        }
                                        else
                                        {
                                            erow = pxln - xloc + 1;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow += (pyln - 1);
                                            }
                                        }
                                        JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                        break;
                                    }
                                    else if (smallprcpos[l] == 4) // my_id as pid's RIGHT
                                    {
                                        if (ylo_array[pid] == 0) // no DOWN
                                        {
                                            erow = nloc;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= (yloc + 1);
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                        else // DOWN exists
                                        {
                                            if (yloc == 0) // bottom-block of grid
                                            {
                                                erow = 2;
                                                if (xlo_array[pid] != 0) // LEFT exists
                                                {
                                                    erow += (pyln - 1);
                                                }
                                                JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                                break;
                                            }
                                            else
                                            {
                                                erow = nloc - pxln;
                                                if (xlo_array[pid] != 0) // LEFT exists
                                                {
                                                    erow -= yloc;
                                                }
                                                JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        AA[ctt++] = BB[k];
                    }
                }
            }
           *postn_c = cnt;
            for (j = 0; j < ng_pt; j ++) // Connections with DOWN and LEFT
            {
                permute[rnt++] = j;
                IA[cnt+1] = IA[cnt] + (IB[j+1] - IB[j]);
                cnt ++;
                row_end = IB[j+1];
                for (k = IB[j]; k < row_end; k ++)
                {
                    col = JB[k]; // globally
                    if ((col >= puue) && (col < ptte)) // its right points
                    {
                        JA[ctt] = col + ple;
                    }
                    else if ((col >= pke) && (col < puue)) // its interior points
                    {
                        JA[ctt] = col + plme;
                    }
                    else if ((col >= ptte) && (col < pkle)) // its up points
                    {
                        JA[ctt] = col + ple;
                    }
                    else // connections with DOWN and LEFT only, if any. Yue Xiaoqiang on 2014/08/15
                    {
                        // Loop for the wanted PID, Yue Xiaoqiang on 2014/08/15
                        for (l = 0; l < num_smallside; l ++)
                        {
                            pid = smallprocs[l];
                            if ((col >= row_starts[pid]) && (col < row_starts[pid+1]))
                            {
                                pxln = xup_array[pid] - xlo_array[pid]; // local number of x points
                                pyln = yup_array[pid] - ylo_array[pid]; // local number of y points
                                rele = col - row_starts[pid]; // local index
                                nloc = rele / ng_pt; // local grid-block index
                                nvar = rele - nloc * ng_pt; // local variable index
                                yloc = nloc / pxln;
                                xloc = nloc - yloc * pxln;
                                if (smallprcpos[l] == 2) // my_id as pid's UP
                                {
                                    if (ylo_array[pid] == 0) // no DOWN
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_mx + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= (yloc + 1);
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                    else // DOWN exists
                                    {
                                        if (xloc == 0) // leftest-block
                                        {
                                            if (xlo_array[pid] == 0) // no LEFT
                                            {
                                                JA[ctt] = row_starts[pid+1] - ng_pt_m2x + nvar;
                                                break;
                                            }
                                            else // LEFT exists
                                            {
                                                JA[ctt] = row_starts[pid+1] - 2 * ng_pt + nvar;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            erow = nloc - ng_pt_mx;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= yloc;
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                }
                                else if (smallprcpos[l] == 4) // my_id as pid's RIGHT
                                {
                                    if (ylo_array[pid] == 0) // no DOWN
                                    {
                                        erow = nloc;
                                        if (xlo_array[pid] != 0) // LEFT exists
                                        {
                                            erow -= (yloc + 1);
                                        }
                                        JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                        break;
                                    }
                                    else // DOWN exists
                                    {
                                        if (yloc == 0) // bottom-block of grid
                                        {
                                            erow = 2;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow += (pyln - 1);
                                            }
                                            JA[ctt] = row_starts[pid+1] - ng_pt * erow + nvar;
                                            break;
                                        }
                                        else
                                        {
                                            erow = nloc - pxln;
                                            if (xlo_array[pid] != 0) // LEFT exists
                                            {
                                                erow -= yloc;
                                            }
                                            JA[ctt] = row_starts[pid] + ng_pt * erow + nvar;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AA[ctt++] = BB[k];
                }
            }
            //jx_assert(cnt == num_rows);
           *dx_len = ng_pt_m_xo - ng_pt; // senting start to RIGHT for TYPE 3
           *kx_len = ng_pt_m_xo; // senting interval to RIGHT for TYPE 3
           *dy_len = num_rows - 2 * ng_pt_m_xo - ng_pt_my; // senting start to UP for TYPE 2
           *lx_len = num_rows - ng_pt_m_px; // senting start to RIGHT for TYPE 1
           *ly_len = num_rows - 2 * ng_pt; // senting start to UP for TYPE 1
        }
    }
    
    return ser_A;
}
