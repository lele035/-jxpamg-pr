//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_csr_matvec.c -- basic operations for mat-vec multiplication.
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_CSRMatrixMatvec
 * \brief Perform y = alpha*A*x + beta*y.
 * \date 2011/09/03
 */
JX_Int
jx_CSRMatrixMatvec( JX_Real        alpha,
                    jx_CSRMatrix *A,
                    jx_Vector    *x,
                    JX_Real        beta,
                    jx_Vector    *y     )
{
   JX_Real     *A_data   = jx_CSRMatrixData(A);
   JX_Int        *A_i      = jx_CSRMatrixI(A);
   JX_Int        *A_j      = jx_CSRMatrixJ(A);
   JX_Int         num_rows = jx_CSRMatrixNumRows(A);
   JX_Int         num_cols = jx_CSRMatrixNumCols(A);

   JX_Int        *A_rownnz = jx_CSRMatrixRownnz(A);
   JX_Int         num_rownnz = jx_CSRMatrixNumRownnz(A);

   JX_Real     *x_data = jx_VectorData(x);
   JX_Real     *y_data = jx_VectorData(y);
   JX_Int         x_size = jx_VectorSize(x);
   JX_Int         y_size = jx_VectorSize(y);
   JX_Int         num_vectors = jx_VectorNumVectors(x);
   JX_Int         idxstride_y = jx_VectorIndexStride(y);
   JX_Int         vecstride_y = jx_VectorVectorStride(y);
   JX_Int         idxstride_x = jx_VectorIndexStride(x);
   JX_Int         vecstride_x = jx_VectorVectorStride(x);

   JX_Real      temp, tempx;

   JX_Int         i, j, jj;

   JX_Int         m;

   JX_Real      xpar = 0.7;

   JX_Int         ierr = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
 
    jx_assert( num_vectors == jx_VectorNumVectors(y) );

    if (num_cols != x_size)
              ierr = 1;

    if (num_rows != y_size)
              ierr = 2;

    if (num_cols != x_size && num_rows != y_size)
              ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

    if (alpha == 0.0)
    {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
       for (i = 0; i < num_rows*num_vectors; i ++)
          y_data[i] *= beta;

       return ierr;
    }

  /*--------------------------------------------
   * y = (beta/alpha)*y
   *------------------------------------------*/
   
   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i ++)
	    y_data[i] = 0.0;
      }
      else
      {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
	 for (i = 0; i < num_rows*num_vectors; i ++)
	    y_data[i] *= temp;
      }
   }

  /*-----------------------------------------------
   * y += A*x
   *---------------------------------------------*/

   /* use rownnz pointer to do the A*x multiplication  
      when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
   
#define JX_SMP_PRIVATE i,jj,m,tempx
#include "../../../include/jx_smp_forloop.h"

      for (i = 0; i < num_rownnz; i ++)
      {
         m = A_rownnz[i];

         if (num_vectors == 1)
         {
            tempx = y_data[m];
            for (jj = A_i[m]; jj < A_i[m+1]; jj ++) 
            {
               tempx += A_data[jj] * x_data[A_j[jj]];
            }
            y_data[m] = tempx;
         }
         else
         {
            for (j = 0; j < num_vectors; ++ j)
            {
               tempx = y_data[ j*vecstride_y + m*idxstride_y ];
               for (jj = A_i[m]; jj < A_i[m+1]; jj ++) 
               {
                  tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + m*idxstride_y] = tempx;
            }
         }
      }

   }
   else
   {
   
#define JX_SMP_PRIVATE i,jj,temp
#include "../../../include/jx_smp_forloop.h"
      for (i = 0; i < num_rows; i ++)
      {
         if (num_vectors == 1)
         {
            temp = y_data[i];
            for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
            {
               temp += A_data[jj] * x_data[A_j[jj]];
            }
            y_data[i] = temp;
         }
         else
         {
            for (j = 0; j < num_vectors; ++ j)
            {
               temp = y_data[ j*vecstride_y + i*idxstride_y ];
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] = temp;
            }
         }
      }
   }


  /*----------------------------------------------------
   *  y = alpha*y
   *---------------------------------------------------*/

   if (alpha != 1.0)
   {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
      for (i = 0; i < num_rows*num_vectors; i ++)
      {
	 y_data[i] *= alpha;
      }	 
   }

   return ierr;
}

/*!
 * \fn JX_Int jx_CSRMatrixMatvecT
 * \brief Perform y = alpha*A^T*x + beta*y.
 * \date 2011/09/03
 */ 
JX_Int
jx_CSRMatrixMatvecT( JX_Real        alpha,
                     jx_CSRMatrix *A,
                     jx_Vector    *x,
                     JX_Real        beta,
                     jx_Vector    *y     )
{
   JX_Real     *A_data    = jx_CSRMatrixData(A);
   JX_Int        *A_i       = jx_CSRMatrixI(A);
   JX_Int        *A_j       = jx_CSRMatrixJ(A);
   JX_Int         num_rows  = jx_CSRMatrixNumRows(A);
   JX_Int         num_cols  = jx_CSRMatrixNumCols(A);

   JX_Real     *x_data = jx_VectorData(x);
   JX_Real     *y_data = jx_VectorData(y);
   JX_Int         x_size = jx_VectorSize(x);
   JX_Int         y_size = jx_VectorSize(y);
   JX_Int         num_vectors = jx_VectorNumVectors(x);
   JX_Int         idxstride_y = jx_VectorIndexStride(y);
   JX_Int         vecstride_y = jx_VectorVectorStride(y);
   JX_Int         idxstride_x = jx_VectorIndexStride(x);
   JX_Int         vecstride_x = jx_VectorVectorStride(x);

   JX_Real      temp;

   JX_Int         i, i1, j, jv, jj, ns, ne, size, rest;
   JX_Int         num_threads;

   JX_Int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

    jx_assert( num_vectors == jx_VectorNumVectors(y) );
 
    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
              
  /*--------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i ++)
      {
         y_data[i] *= beta;
      }	 

      return ierr;
   }

  /*------------------------------------------------
   * y = (beta/alpha)*y
   *-----------------------------------------------*/

   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i ++)
	 {
	    y_data[i] = 0.0;
	 }
      }
      else
      {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
	 for (i = 0; i < num_cols*num_vectors; i ++)
	 {
	    y_data[i] *= temp;
	 }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   num_threads = jx_NumThreads();
   if (num_threads > 1)
   {

#define JX_SMP_PRIVATE i, i1,jj,j,ns,ne,size,rest
#include "../../../include/jx_smp_forloop.h"
      for (i1 = 0; i1 < num_threads; i1 ++)
      {
         size = num_cols / num_threads;
         rest = num_cols - size*num_threads;
         if (i1 < rest)
         {
            ns = i1*size + i1 - 1;
            ne = (i1 + 1)*size + i1 + 1;
         }
         else
         {
            ns = i1*size + rest - 1;
            ne = (i1 + 1)*size + rest;
         }
         
         if (num_vectors == 1)
         {
            for (i = 0; i < num_rows; i ++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  j = A_j[jj];
                  if (j > ns && j < ne)
                  {
                     y_data[j] += A_data[jj] * x_data[i];
                  }
               }
            }
         }
         else
         {
            for (i = 0; i < num_rows; i ++)
            {
               for (jv = 0; jv < num_vectors; ++ jv)
               {
                  for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
                  {
                     j = A_j[jj];
                     if (j > ns && j < ne)
                     {
                        y_data[ j*idxstride_y + jv*vecstride_y ] +=
                           A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
                     }
                  }
               }
            }
         }

      }
   }
   else 
   {
      for (i = 0; i < num_rows; i ++)
      {
         if (num_vectors == 1)
         {
            for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
            {
               j = A_j[jj];
               y_data[j] += A_data[jj] * x_data[i];
            }
         }
         else
         {
            for (jv = 0; jv < num_vectors; ++ jv)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj ++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                     A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
               }
            }
         }
      }
   }
   

  /*--------------------------------------------
   * y = alpha*y
   *-------------------------------------------*/

   if (alpha != 1.0)
   {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
      for (i = 0; i < num_cols*num_vectors; i ++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}
