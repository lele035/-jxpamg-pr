//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_gs_piv.c
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_gselim_piv
 * \brief Gaussian Elimination - with pivoting, newly-added 2009/08/08
 * \date 2011/09/03
 */
JX_Int 
jx_gselim_piv( JX_Real *A_matrix, JX_Real *x_vector, JX_Int size )
{
   JX_Int    err_flag = 0;
   JX_Int    j, k, m, piv_row;
   JX_Real factor, piv, tmp;
   JX_Real eps = 1e-8;
   
   if (size == 1)  /* A_matrix is 1x1 */  
   {
      if (fabs(A_matrix[0]) > 1e-10)
      {
         x_vector[0] = x_vector[0] / A_matrix[0];
         return(err_flag);
      }
      else
      {
         err_flag = 1;
         return (err_flag);
      }
   }
   else  /* A_matrix is nxn.  Forward elimination */ 
   {
      for (k = 0; k < size-1; k ++)
      {
         /* we do partial pivoting for size */
         piv = A_matrix[k*size+k];
         piv_row = k;
         /* find the largest pivot in position k */
         for (j = k + 1; j < size; j ++)         
         {
            if (fabs(A_matrix[j*size+k]) > fabs(piv))
            {
               piv = A_matrix[j*size+k];
               piv_row = j;
            }
         }
         if (piv_row != k)   /* do a row exchange - rows k and piv_row */
         {
            for (j=0; j < size; j ++)
            {
               tmp = A_matrix[k*size + j];
               A_matrix[k*size + j] = A_matrix[piv_row*size + j];
               A_matrix[piv_row*size + j] = tmp;
            }
            tmp = x_vector[k];
            x_vector[k] = x_vector[piv_row];
            x_vector[piv_row] = tmp;
         }

         if (fabs(piv) > eps)
          {          
             for (j = k + 1; j < size; j ++)
             {
                 if (A_matrix[j*size+k] != 0.0)
                 {
                    factor = A_matrix[j*size+k] / A_matrix[k*size+k];
                    for (m = k+1; m < size; m ++)
                    {
                        A_matrix[j*size+m] -= factor * A_matrix[k*size+m];
                    }           
                    x_vector[j] -= factor * x_vector[k];   /* Elimination step for rhs */             
                 }
             }
          }
         else
         {
            /* jx_printf("Matrix is nearly singular: zero pivot error\n"); */
            return(-1);
         }
      }
      
      /* we also need to check the pivot in the last row to see if it is zero */  
      k = size - 1; /* last row */
      if (fabs(A_matrix[k*size+k]) < eps)
      {
         /* jx_printf("Block of matrix is nearly singular: zero pivot error\n"); */
         return(-1);
      }

      /* Back Substitution  */
      for (k = size-1; k > 0; -- k)
      {
           x_vector[k] /= A_matrix[k*size+k];
           for (j = 0; j < k; j ++)
           {
               if (A_matrix[j*size+k] != 0.0)
               {
                  x_vector[j] -= x_vector[k] * A_matrix[j*size+k];
               }
           }
      }
      x_vector[0] /= A_matrix[0];
      return(err_flag);
    }
}
