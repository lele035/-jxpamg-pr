//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  csr_matrix.c -- operations for CSR matrices.
 *  Date: 2014/07/05
 */

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_CSRMatrixWeaklyDiagDominant
 * \brief Weakly Diag-Dominant
 * \author Yue Xiaoqiang
 * \date 2014/01/03
 */
JX_Int
jx_CSRMatrixWeaklyDiagDominant( jx_CSRMatrix *A,
                                JX_Int num_equns,
                                JX_Real theta,
                                JX_Real gamma_3,
                                JX_Real gamma_11 )
{
   JX_Int ierr = 1;
   JX_Int num_rows = jx_CSRMatrixNumRows(A);
   JX_Int block_size = num_rows / num_equns;
   JX_Int t_b_size = 2 * block_size;
   JX_Int f_b_size = 3 * block_size;
   JX_Int tt_b_size = 10 * block_size;
   JX_Int ff_b_size = 11 * block_size;
   JX_Int *IA = jx_CSRMatrixI(A);
   JX_Int *JA = jx_CSRMatrixJ(A);
   JX_Real *AA = jx_CSRMatrixData(A);
   JX_Int cnt = 0, gnt = 0;
   JX_Int row, row_srt, row_end, col;
   JX_Real row_sum;
   
   // block - 11
   for (row = tt_b_size; row < ff_b_size; row ++)
   {
      row_sum = 0.0;
      row_srt = IA[row];
      row_end = IA[row+1];
      for (col = row_srt; col < row_end; col ++)
      {
         if ((JA[col] >= tt_b_size) && (JA[col] < ff_b_size))
         {
            row_sum += AA[col];
         }
      }
      if (fabs(row_sum) <= (theta * fabs(AA[row_srt])))
      {
         gnt ++;
      }
   }
   if ((JX_Real)gnt/block_size < gamma_11)
   {
      return ierr;
   }
   // block - 3
   if (gamma_3 > 0.0)
   {
      for (row = t_b_size; row < f_b_size; row ++)
      {
         row_sum = 0.0;
         row_srt = IA[row];
         row_end = IA[row+1];
         for (col = row_srt; col < row_end; col ++)
         {
            if ((JA[col] >= t_b_size) && (JA[col] < f_b_size))
            {
               row_sum += AA[col];
            }
         }
         if (fabs(row_sum) <= (theta * fabs(AA[row_srt])))
         {
            cnt ++;
         }
      }
      if ((JX_Real)cnt/block_size >= gamma_3)
      {
          ierr = 0;
      }
   }
   else
   {
       ierr = 0;
   }
   
   return ierr;
}
