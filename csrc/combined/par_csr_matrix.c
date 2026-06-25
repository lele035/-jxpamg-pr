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

/*!
 * \fn jx_ParCSRMatrix *jx_BuildMatParFromOneFile4
 * \brief Build a symmetric parallel matrix by reading data from a given file.
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
 * \date 2014/04/14
 */
jx_ParCSRMatrix *
jx_BuildMatParFromOneFile4( char *filename, JX_Int file_base, JX_Int  *row_part, JX_Int  *col_part )
{
   jx_ParCSRMatrix  *A     = NULL;
   jx_CSRMatrix     *A_CSR = NULL;
   jx_CSRMatrix     *B_CSR = NULL;

   JX_Int my_id;
   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (my_id == 0)
   {
      B_CSR = jx_CSRMatrixRead(filename, file_base);
      jx_CSRMatrixReorder(B_CSR);
      A_CSR = jx_CSRMatrixSymmetrization(B_CSR);
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
