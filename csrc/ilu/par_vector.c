//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_vector.c -- operations for parallel vectors.
 *  Date: 2014/07/05
 */

#include "jx_mv.h"
#include "jx_ilu.h"

/*!
 * \fn jx_ParVector *jx_BuildRhsParFromOneFile2
 * \brief Build a parallel vector from a given file
 *        including reorder by variables and along y-axis
 * \param *filename pointer to the file name.
 * \param *A pointer to the matrix which will provide the row partitioning.
 * \note The storage format of the vector is expected to be as follows:
 *         the 1st row:                   n                       // JX_Int-type
 *         the 2 to (n+1)-th row:         f[i]    i=0(1)n-1       // JX_Real-type
 *      where n is the total size of the vector, f[i] is the i-th entry of the vector.
 * \author Yue Xiaoqiang
 * \date 2014/04/10
 */
jx_ParVector *
jx_BuildRhsParFromOneFile2( char *filename, jx_ParCSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
   jx_ParVector *b = NULL;
   jx_Vector    *b_CSR = NULL;
   jx_Vector    *c_CSR = NULL;
   JX_Int           my_id;
   JX_Int          *partitioning;

   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (my_id == 0) 
   {
      b_CSR = jx_SeqVectorRead(filename);
      c_CSR = jx_VectorTDMGReorderByNodes(b_CSR, num_equns);
      jx_SeqVectorDestroy(b_CSR);
      b_CSR = jx_VectorTDMGReorderAlongY(c_CSR, num_equns, nx, ny);
      jx_SeqVectorDestroy(c_CSR);
   }

   jx_ParCSRMatrixGetRowPartitioning(A, &partitioning);

   b = jx_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning);

   if (my_id == 0) jx_SeqVectorDestroy(b_CSR);

   return (b);
}

/*!
 * \fn JX_Int jx_GeneratePartitioning2
 * \brief Generates load balanced partitioning of a 1-d array, block-partitioned
 * \author Yue Xiaoqiang
 * \date 2014/04/13
 */
JX_Int
jx_GeneratePartitioning2(JX_Int num_procs, JX_Int **part_ptr, JX_Int num_equns, JX_Int length, JX_Int ny)
{
   JX_Int  ierr = 0;
   JX_Int *part;
   JX_Int  size, rest;
   JX_Int  dist, eist;
   JX_Int  i;

   part = jx_CTAlloc(JX_Int, num_procs+1);
   size = length / num_procs;
   rest = length - size * num_procs;
   dist = num_equns * ny;
   eist = dist * size;
   part[0] = 0;
   for (i = 0; i < num_procs; i ++)
   {
	part[i+1] = part[i] + eist;
	if (i < rest) part[i+1] += dist;
   }

  *part_ptr = part;
   return ierr;
}
