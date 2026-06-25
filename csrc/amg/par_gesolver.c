//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_gesolver.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_GaussElimination
 * \brief Gauss Elimination solver.
 * \author peghoty
 * \date 2009/07/25
 */
JX_Int 
jx_GaussElimination( JX_Int              solverid,
                     jx_ParCSRMatrix *par_matrix, 
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app )
{
   jx_CSRMatrix    *A_diag      = jx_ParCSRMatrixDiag(par_matrix);
   JX_Int              n_global    = jx_ParCSRMatrixGlobalNumRows(par_matrix);
   JX_Int              n           = jx_CSRMatrixNumRows(A_diag);
   JX_Int              first_index = jx_ParVectorFirstIndex(par_app);

   jx_CSRMatrix    *A_CSR;
   JX_Int             *A_CSR_i;  
   JX_Int             *A_CSR_j;
   JX_Real          *A_CSR_data;
   
   jx_Vector       *f_vector;
   JX_Real          *f_vector_data;

   jx_Vector       *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real          *u_data  = jx_VectorData(u_local);

   JX_Real          *A_mat;
   JX_Real          *b_vec;

   JX_Int              err_flag = 0;
   JX_Int              i, jj;
   JX_Int              column;

   if (n)   
   {
      A_CSR    = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
      f_vector = jx_ParVectorToVectorAll(par_rhs);

      A_CSR_i = jx_CSRMatrixI(A_CSR);
      A_CSR_j = jx_CSRMatrixJ(A_CSR);
      A_CSR_data    = jx_CSRMatrixData(A_CSR);
      f_vector_data = jx_VectorData(f_vector);

      A_mat = jx_CTAlloc(JX_Real, n_global*n_global);
      b_vec = jx_CTAlloc(JX_Real, n_global);    

     /*---------------------------------------------------------------
      *  Load CSR matrix into A_mat and b_vec.
      *--------------------------------------------------------------*/

      for (i = 0; i < n_global; i ++)
      {
         for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj ++)
         {
            column = A_CSR_j[jj];
            A_mat[i*n_global+column] = A_CSR_data[jj];
         }
         b_vec[i] = f_vector_data[i];
      }

      if (solverid == 9)
      {
         err_flag = jx_gselim(A_mat, b_vec, n_global);
      }
      if (solverid == 10)  /* newly-added 2009/08/08 */
      {
         err_flag = jx_gselim_piv(A_mat, b_vec, n_global);
      }

      for (i = 0; i < n; i ++)
      {
         u_data[i] = b_vec[first_index+i];
      }

      jx_TFree(A_mat); 
      jx_TFree(b_vec);
      jx_CSRMatrixDestroy(A_CSR);
      A_CSR = NULL;
      jx_SeqVectorDestroy(f_vector);
      f_vector = NULL;
   }
   
   return err_flag; 
}
