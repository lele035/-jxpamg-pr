/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/
 
/*!
 *  tool.c
 *
 *  Created by peghoty 2010/08/22
 *
 *  Xiangtan University
 *  peghoty@163.com
 *
 */ 

#include "jx_multils.h"

/*!
 * \fn JX_Int fsls_ArrayInitialize
 * \brief Zero out a JX_Real array.
 * \param *x pointer to the JX_Real array.
 * \param n size of the 'x' array.
 * \author peghoty
 * \date 2010/09/16
 */
void 
fsls_ArrayInitialize( JX_Real *x, JX_Int n )
{
   memset(x, 0X0, n*sizeof(JX_Real));
}

/*!
 * \fn JX_Int fsls_ArrayPrint
 * \brief write a vector to a given file
 * \param *u pointer to the vector array
 * \param n size of the 'u' array
 * \param *filename where the vector be saved
 * \author peghoty
 * \date 2009/11/29 
 */
JX_Int 
fsls_ArrayPrint( JX_Real *u, JX_Int n, char *filename )
{
    JX_Int j;
    FILE *fp = NULL;
   
    /* Save the vector */
    fp = fopen(filename, "w");
    jx_fprintf(fp, "%d\n", n);
    for (j = 0; j < n; j ++)
    {
       jx_fprintf(fp, "%.15le\n", u[j]);
    }
    fclose(fp);
    
    return (0);
}

/*!
 * \fn JX_Int fsls_IntArrayPrint
 * \brief write a vector of integer type to a given file
 * \param *u pointer to the vector array
 * \param n size of the 'u' array
 * \param *filename where the vector be saved
 * \author peghoty
 * \date 2011/05/14 
 */
JX_Int 
fsls_IntArrayPrint( JX_Int *u, JX_Int n, char *filename )
{
    JX_Int j;
    FILE *fp = NULL;
   
    /* Save the vector */
    fp = fopen(filename, "w");
    jx_fprintf(fp, "%d\n", n);
    for (j = 0; j < n; j ++)
    {
       jx_fprintf(fp, "%d\n", u[j]);
    }
    fclose(fp);
    
    return (0);
}

/*!
 * \fn JX_Int fsls_IntArrayIJPrint
 * \brief write a vector of integer type to a given file in IJ manner.
 * \param *u pointer to the vector array
 * \param n size of the 'u' array
 * \param *filename where the vector be saved
 * \author peghoty
 * \date 2011/05/18 
 */
JX_Int 
fsls_IntArrayIJPrint( JX_Int *u, JX_Int n, char *filename )
{
    JX_Int j;
    FILE *fp = NULL;
   
    /* Save the vector */
    fp = fopen(filename, "w");
    for (j = 0; j < n; j ++)
    {
       jx_fprintf(fp, "%d %d\n", j+1, u[j]);
    }
    fclose(fp);
    
    return (0);
}

/*!
 * \fn void fsls_ArrayCopy( JX_Real *x, JX_Real *y, JX_Int size )
 * \brief copy x to y ( x -> y ).
 * \author peghoty
 * \date 2010/04/29 
 */
void 
fsls_ArrayCopy( JX_Int size, JX_Real *x, JX_Real *y )
{
   memcpy( y, x, size*sizeof(JX_Real) );
}

/*!
 * \fn JX_Real fsls_Arrayl2Norm
 * \brief Compute the l^2 norm of a given vector
 * \param *x pointer to the vector
 * \return l2norm, the l^2 norm of x
 * \author peghoty
 * \date 2009/12/30 
 */
JX_Real   
fsls_Arrayl2Norm( JX_Real *x, JX_Int n )
{
   JX_Int i;
   JX_Real l2norm = 0.0;
   
   for (i = 0; i < n; i ++)
   {
      l2norm += x[i]*x[i];
   }
   l2norm = sqrt(l2norm); 
     
   return l2norm;
}

/*!
 * \fn JX_Real fsls_ArrayAveL2Norm
 * \brief Compute the l^2 norm of a given vector
 * \param *x pointer to the vector
 * \return l2norm, the l^2 norm of x
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Real   
fsls_ArrayAveL2Norm( JX_Real *x, JX_Int n )
{
   JX_Int i;
   JX_Real l2norm = 0.0;
   
   for (i = 0; i < n; i ++)
   {
     l2norm += x[i]*x[i];
   }
   l2norm = sqrt(l2norm/(JX_Real)n); 
     
   return l2norm;
}

/*!
 * \fn JX_Int fsls_ArrayDoubleAbsMax
 * \brief Find the abs. maximal value of 'x'
 * \return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayDoubleAbsMax( JX_Real *x, JX_Int n, JX_Real *elm_ptr )
{
   JX_Int i,index;
   JX_Real elm = 0.0;
   
   elm = fabs(x[0]); index = 0;
   for (i = 1; i < n; i ++)
   {
      if (fabs(x[i]) > elm)
      {
         elm = fabs(x[i]);
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayDoubleAbsMin
 * \brief Find the abs. minimal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayDoubleAbsMin( JX_Real *x, JX_Int n, JX_Real *elm_ptr )
{
   JX_Int i,index;
   JX_Real elm = 0.0;
   
   elm = fabs(x[0]); index = 0;
   for (i = 1; i < n; i ++)
   {
      if (fabs(x[i]) < elm)
      {
         elm = fabs(x[i]);
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayDoubleMax
 * \brief Find the maximal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayDoubleMax( JX_Real *x, JX_Int n, JX_Real *elm_ptr )
{
   JX_Int i,index;
   JX_Real elm = 0.0;
   
   elm = x[0]; index = 0;
   for (i = 1; i < n; i ++)
   {
      if (x[i] > elm)
      {
         elm = x[i];
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayDoubleMin
 * \brief Find the minimal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayDoubleMin( JX_Real *x, JX_Int n, JX_Real *elm_ptr )
{
   JX_Int i,index;
   JX_Real elm = 0.0;
   
   elm = x[0]; index = 0;
   for (i = 1; i < n; i ++)
   {
      if (x[i] < elm)
      {
         elm = x[i];
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayIntAbsMax
 * \brief Find the abs. maximal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayIntAbsMax( JX_Int *x, JX_Int n, JX_Int *elm_ptr )
{
   JX_Int i,index;
   JX_Int elm = 0;
   
   elm = abs(x[0]); index = 0;
   for (i = 1; i < n; i ++)
   {
      if (abs(x[i]) > elm)
      {
         elm = abs(x[i]);
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayIntAbsMin
 * \brief Find the abs. minimal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayIntAbsMin( JX_Int *x, JX_Int n, JX_Int *elm_ptr )
{
   JX_Int i,index;
   JX_Int elm = 0;
   
   elm = abs(x[0]); index = 0;
   for (i = 1; i < n; i ++)
   {
      if (abs(x[i]) < elm)
      {
         elm = abs(x[i]);
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayIntMax
 * \brief Find the maximal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayIntMax( JX_Int *x, JX_Int n, JX_Int *elm_ptr )
{
   JX_Int i,index;
   JX_Int elm = 0;
   
   elm = x[0]; index = 0;
   for (i = 1; i < n; i ++)
   {
      if (x[i] > elm)
      {
         elm = x[i];
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_ArrayIntMin
 * \brief Find the minimal value of 'x', and return the corresponding index
 * \author peghoty
 * \date 2010/08/26 
 */
JX_Int
fsls_ArrayIntMin( JX_Int *x, JX_Int n, JX_Int *elm_ptr )
{
   JX_Int i,index;
   JX_Int elm = 0;
   
   elm = x[0]; index = 0;
   for (i = 1; i < n; i ++)
   {
      if (x[i] < elm)
      {
         elm = x[i];
         index = i;
      } 
   }
   
   *elm_ptr = elm;
   return index;
}

/*!
 * \fn JX_Int fsls_daxpy
 * \brief Compute y := y + alpha*x
 * \author peghoty
 * \date 2010/12/30 
 */
JX_Int
fsls_daxpy( JX_Int n, JX_Real alpha, JX_Real *x, JX_Real *y )
{           
   JX_Int i,ierr = 0;

   for (i = 0; i < n; i ++)
   {
      y[i] += alpha*x[i];
   }

   return ierr;
}

/*!
 * \fn fsls_gselim
 * \brief Gaussian Elimination
 * \param *A the pointer to the matrix to be solved
 * \param *x the pointer to the rhs vector at first and the solution at last
 * \param n the size of the matrix
 * \return 0 if success, 1 if fail
 * \date 2009/12/21
 */
JX_Int 
fsls_gselim( JX_Real *A, JX_Real *x, JX_Int n )
{
   JX_Int    err_flag = 0;
   JX_Int    j,k,m;
   JX_Real factor = 0.0;
   
   if (n == 1)   /* A is 1x1 */  
   {
      if (A[0] != 0.0)
      {
         x[0] = x[0] / A[0];
         return(err_flag);
      }
      else
      {
         err_flag = 1;
         return(err_flag);
      }
   }
   else   /* A is nxn */   
   {
   
      /* Forward elimination */
      for (k = 0; k < n-1; k ++)
      {
          if (A[k*n+k] != 0.0)
          {          
             for (j = k+1; j < n; j ++)
             {
                 if (A[j*n+k] != 0.0)
                 {
                    factor = A[j*n+k] / A[k*n+k];
                    for (m = k+1; m < n; m ++)
                    {
                        A[j*n+m] -= factor * A[k*n+m];
                    }
                    /* Elimination step for rhs */ 
                    x[j] -= factor * x[k];              
                 }
             }
          }
       }
       
       /* Back Substitution */
       for (k = n-1; k > 0; -- k)
       {
           x[k] /= A[k*n+k];
           for (j = 0; j < k; j ++)
           {
               if (A[j*n+k] != 0.0)
               {
                  x[j] -= x[k] * A[j*n+k];
               }
           }
       }
       
       x[0] /= A[0];
       
       return(err_flag);
    }
}

/*!
 * \fn fsls_gselim_piv
 * \brief Gaussian Elimination with pivoting
 * \param *A the pointer to the matrix to be solved
 * \param *x the pointer to the rhs vector at first and the solution at last
 * \param n the size of the matrix
 * \return 0 if success, 1,-1 if fail
 * \date 2010/03/03
 */
JX_Int 
fsls_gselim_piv( JX_Real *A, JX_Real *x, JX_Int n )
{
   JX_Int    err_flag = 0;
   JX_Int    j,k,m, piv_row;
   JX_Real factor, piv, tmp;
   JX_Real eps = 1e-8;
   
   if (n == 1)  /* A is 1x1 */  
   {
      if (fabs(A[0]) >  1e-10)
      {
         x[0] = x[0] / A[0];
         return(err_flag);
      }
      else
      {
         err_flag = 1;
         return(err_flag);
      }
   }
   
   else  /* A is nxn.  Forward elimination */ 
   {
      for (k = 0; k < n-1; k ++)
      {
         /* we do partial pivoting for size */
         piv = A[k*n+k];
         piv_row = k;
         /* find the largest pivot in position k*/
         for (j = k+1; j < n; j ++)         
         {
            if (fabs(A[j*n+k]) > fabs(piv))
            {
               piv =  A[j*n+k];
               piv_row = j;
            }
         }
         if (piv_row != k)   /* do a row exchange  - rows k and piv_row*/
         {
            for (j = 0; j < n; j ++)
            {
               tmp = A[k*n + j];
               A[k*n + j] = A[piv_row*n + j];
               A[piv_row*n + j] = tmp;
            }
            tmp = x[k];
            x[k] = x[piv_row];
            x[piv_row] = tmp;
         }

         if (fabs(piv) > eps)
         {          
             for (j = k+1; j < n; j ++)
             {
                 if (A[j*n+k] != 0.0)
                 {
                    factor = A[j*n+k]/A[k*n+k];
                    for (m = k+1; m < n; m ++)
                    {
                        A[j*n+m] -= factor * A[k*n+m];
                    }
                    /* Elimination step for rhs */ 
                    x[j] -= factor * x[k];              
                 }
             }
         }
         else
         {
            jx_printf("Matrix is nearly singular: zero pivot error\n");
            return(-1);
         }
      }
      
      /* we also need to check the pivot in the last row to see if it is zero */  
      k = n - 1; /* last row */
      if ( fabs(A[k*n+k]) < eps)
      {
         jx_printf("Block of matrix is nearly singular: zero pivot error\n");
         return(-1);
      }

      /* Back Substitution  */
      for (k = n-1; k > 0; -- k)
      {
           x[k] /= A[k*n+k];
           for (j = 0; j < k; j ++)
           {
               if (A[j*n+k] != 0.0)
               {
                  x[j] -= x[k] * A[j*n+k];
               }
           }
      }
      x[0] /= A[0];
      return(err_flag);
   }
}

/*!
 * \fn fsls_Free
 * \brief free memory
 * \date 2010/01/03
 */
void
fsls_Free( char *ptr )
{
   if (ptr) free(ptr);
}

/*!
 * \fn fsls_CAlloc
 * \brief allocate memory
 * \date 2010/01/03
 */
char *
fsls_CAlloc( size_t count, size_t elt_size )
{
   char *ptr  = NULL;
   JX_Int   size = count*elt_size;

   if (size > 0)
   {
      ptr = calloc(count, elt_size);
      if (ptr == NULL)
      {
         fsls_OutOfMemory(size);
      }
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*!
 * \fn fsls_OutOfMemory
 * \brief print information when out of memory
 * \date 2010/01/03
 */
JX_Int
fsls_OutOfMemory( size_t size )
{
   jx_printf("\n \033[31mOut of memory\033[00m trying to allocate \033[31m%d\033[00m bytes!\n", (JX_Int) size);
   fflush(stdout);
   return 0;
}

/*!
 * \fn void fsls_ISwap
 * \brief swap the i-th and j-th element in the array 'w'(JX_Int type)
 * \param *w pointer to the array
 * \param i one position in w
 * \param j the other position in w
 * \author peghoty
 * \date 2009/11/28 
 */
void 
fsls_ISwap(JX_Int *w, JX_Int i, JX_Int j)
{
   JX_Int temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*!
 * \fn fsls_DSwap
 * \brief swap the i-th and j-th element in the array 'w'(JX_Real type)
 * \param *w pointer to the array
 * \param i one position in w
 * \param j the other position in w  
 * \author peghoty 
 * \date 2009/11/28 
 */
void 
fsls_DSwap(JX_Real *w, JX_Int i, JX_Int j)
{
   JX_Real temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*!
 * \fn void fsls_IDSwap
 * \brief swap the i-th and j-th element both in the array 
 *  'v'(JX_Int type) and 'w'(JX_Real type)
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Real array
 * \param i one position in the array
 * \param j the other position in in the array  
 * \author peghoty 
 * \date 2009/12/04
 */
void 
fsls_IDSwap( JX_Int *v, JX_Real *w, JX_Int i, JX_Int j )
{
   JX_Int    temp;
   JX_Real temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*!
 * \fn void fsls_IISwap
 * \brief swap the i-th and j-th element both in the array 
 *  'v'(JX_Int type) and 'w'(JX_Int type)
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param i one position in the array
 * \param j the other position in in the array  
 * \author peghoty 
 * \date 2010/05/06 
 */
void 
fsls_IISwap( JX_Int *v, JX_Int *w, JX_Int i, JX_Int j )
{
   JX_Int temp;
   JX_Int temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*!
 * \fn void fsls_IQuickSort
 * \brief sort the array 'a'(JX_Int type) with the quick sorting algorithm
 * \param order integer, 12 means ascendingly; 21 means descendingly
 * \param *a pointer to the array needed to be sorted
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1,respectively,where n is the 
 * length of 'a'.
 * \author peghoty
 * \date 2009/12/05 
 */
void
fsls_IQuickSort(JX_Int order, JX_Int *a, JX_Int left, JX_Int right)
{
   JX_Int i, last;

   if (left >= right) return;
   
   fsls_ISwap(a, left, (left+right)/2);
   
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (a[i] < a[left]) fsls_ISwap(a, ++last, i);
      }
      else if(order == 21)
      {
         if (a[i] > a[left]) fsls_ISwap(a, ++last, i);
      }
   }
   
   fsls_ISwap(a, left, last);
   
   fsls_IQuickSort(order, a, left, last-1);
   fsls_IQuickSort(order, a, last+1, right);
}

/*!
 * \fn void fsls_DQuickSort
 * \brief sort the array 'a'(JX_Real type) with the quick sorting algorithm
 * \param order integer, 12 means ascendingly; 21 means descendingly
 * \param *a pointer to the array needed to be sorted
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1,respectively,where n is the 
 * length of 'a'.  
 * \author peghoty
 * \date 2009/12/05  
 */
void
fsls_DQuickSort(JX_Int order, JX_Real *a, JX_Int left, JX_Int right)
{
   JX_Int i, last;

   if (left >= right) return;
   
   fsls_DSwap(a, left, (left+right)/2);
   
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (a[i] < a[left]) fsls_DSwap(a, ++last, i);
      }
      else if (order == 21)
      {
         if (a[i] > a[left]) fsls_DSwap(a, ++last, i);
      }
   }
   
   fsls_DSwap(a, left, last);
   
   fsls_DQuickSort(order, a, left, last-1);
   fsls_DQuickSort(order, a, last+1, right);
}

/*!
 * \fn void fsls_IQuickSortIndex
 * \brief reorder the index of 'a'(JX_Int type) so that 'a' is ascending 
 *  or descending in such order
 * \param order integer, 12 means ascendingly; 21 means descendingly  
 * \param *a pointer to the array 
 * \param left the starting index
 * \param right the ending index
 * \param *index the index of 'a'
 * \note 'left' and 'right' are usually set to be 0 and n-1,respectively,where n is the 
 * length of 'a'. 'index' should be initialized in the nature order and it has the
 * same length as 'a'.   
 * \author peghoty
 * \date 2009/12/05  
 */
void
fsls_IQuickSortIndex(JX_Int order, JX_Int *a, JX_Int left, JX_Int right, JX_Int *index)
{
   JX_Int i, last;

   if (left >= right) return;
   
   fsls_ISwap(index, left, (left+right)/2);
   
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (a[index[i]] < a[index[left]]) 
           fsls_ISwap(index, ++last, i);
      }
      else if (order == 21)
      {
         if (a[index[i]] > a[index[left]]) 
           fsls_ISwap(index, ++last, i);
      }
   }
   
   fsls_ISwap(index, left, last);
   
   fsls_IQuickSortIndex(order, a, left, last-1, index);
   fsls_IQuickSortIndex(order, a, last+1, right, index);
}

/*!
 * \fn void fsls_DQuickSortIndex
 * \brief reorder the index of 'a'(JX_Real type) so that 'a' is ascending 
 *  or descending in such order
 * \param order integer, 12 means ascendingly; 21 means descendingly   
 * \param *a pointer to the array 
 * \param left the starting index
 * \param right the ending index
 * \param *index the index of 'a'
 * \note 'left' and 'right' are usually set to be 0 and n-1,respectively,where n is the 
 * length of 'a'. 'index' should be initialized in the nature order and it has the
 * same length as 'a'.   
 * \author peghoty
 * \date 2009/12/05  
 */
void
fsls_DQuickSortIndex(JX_Int order, JX_Real *a, JX_Int left, JX_Int right, JX_Int *index)
{
   JX_Int i, last;

   if (left >= right) return;
   
   fsls_ISwap(index, left, (left+right)/2);
   
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (a[index[i]] < a[index[left]])
           fsls_ISwap(index, ++last, i);
      }
      else if (order == 21)
      {
         if (a[index[i]] > a[index[left]])
           fsls_ISwap(index, ++last, i);
      }
   }
   
   fsls_ISwap(index, left, last);
   
   fsls_DQuickSortIndex(order, a, left, last-1, index);
   fsls_DQuickSortIndex(order, a, last+1, right, index);
}

/*!
 * \fn void fsls_IIQuickSort
 * \brief sort the array 'w' with the quick sorting 
 *  algorithm, and meanwhile 'v'(usually be the indexes array of 'w')  
 *  is sorted in the same way.
 * \param order integer, 12 means ascendingly; 21 means descendingly  
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *  respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty  
 * \date 2010/05/06
 */
void 
fsls_IIQuickSort(JX_Int order, JX_Int *v, JX_Int *w, JX_Int left, JX_Int right)
{
   JX_Int i, last;

   if (left >= right) return;
   fsls_IISwap(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (w[i] < w[left])
           fsls_IISwap(v, w, ++last, i);
      }
      else if (order == 21)
      {
         if (w[i] > w[left])
           fsls_IISwap(v, w, ++last, i);
      }
   }
   fsls_IISwap(v, w, left, last);
   fsls_IIQuickSort(order, v, w, left, last-1);
   fsls_IIQuickSort(order, v, w, last+1, right);
}

/*!
 * \fn void fsls_IDQuickSort
 * \brief sort the array 'w' with the quick sorting 
 *  algorithm, and meanwhile 'v'(usually be the indexes array of 'w')  
 *  is sorted in the same way.
 * \param order integer, 12 means ascendingly; 21 means descendingly  
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Real array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *  respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2009/12/05  
 */
void 
fsls_IDQuickSort( JX_Int order, JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right) return;
   fsls_IDSwap(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (w[i] < w[left])
           fsls_IDSwap(v, w, ++last, i);
      }
      else if (order == 21)
      {
         if (w[i] > w[left])
           fsls_IDSwap(v, w, ++last, i);
      }
   }
   fsls_IDSwap(v, w, left, last);
   fsls_IDQuickSort(order, v, w, left, last-1);
   fsls_IDQuickSort(order, v, w, last+1, right);
}

/*!
 * \fn void fsls_DIQuickSort
 * \brief sort the array 'w'(JX_Int type) with the quick sorting 
 *  algorithm, and meanwhile 'v'(JX_Real type)  
 *  is sorted in the same way.
 * \param order integer, 12 means ascendingly; 21 means descendingly  
 * \param *v pointer to the JX_Real array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *  respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2009/12/05  
 */
void 
fsls_DIQuickSort( JX_Int order, JX_Real *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right) return;
   fsls_IDSwap(w, v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (w[i] < w[left])
           fsls_IDSwap(w, v, ++last, i);
      }
      else if (order == 21)
      {
         if (w[i] > w[left])
           fsls_IDSwap(w, v, ++last, i);
      }
   }
   fsls_IDSwap(w, v, left, last);
   fsls_DIQuickSort(order, v, w, left, last-1);
   fsls_DIQuickSort(order, v, w, last+1, right);
}

/*!
 * \fn void fsls_IDfabsQuickSort
 * \brief sort the array '|w|' with the quick sorting 
 *  algorithm, and meanwhile 'v'(usually be the indexes array of 'w')  
 *  is sorted in the same way.
 * \param order integer, 12 means ascendingly; 21 means descendingly  
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Real array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *  respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2009/12/05  
 */
void 
fsls_IDfabsQuickSort( JX_Int order, JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i, last;

   if (left >= right) return;
   fsls_IDSwap(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (order == 12)
      {
         if (fabs(w[i]) < fabs(w[left]))
           fsls_IDSwap(v, w, ++last, i);
      }
      else if (order == 21)
      {
         if (fabs(w[i]) > fabs(w[left]))
           fsls_IDSwap(v, w, ++last, i);
      }
   }
   fsls_IDSwap(v, w, left, last);
   fsls_IDfabsQuickSort(order, v, w, left, last-1);
   fsls_IDfabsQuickSort(order, v, w, last+1, right);
}

/*!
 * \fn void fsls_iQuickSort12
 * \brief Sort the array 'data'(JX_Int type) with quick sorting algorithm
 * \param *data pointer to the array to be sorted
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively,
 *       where n is the length of 'data'.  
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_iQuickSort12( JX_Int *data, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[j] <= key) break;
         }
         if (i < j)
         {
            tmp = data[i]; data[i] = data[j]; data[j] = tmp;
         }
      }
      tmp = data[left]; data[left] = data[j]; data[j] = tmp;
      fsls_iQuickSort12(data, left, j-1);
      fsls_iQuickSort12(data, i, right);
   }
}

/*!
 * \fn void fsls_iQuickSortIndex12
 * \brief Reorder the index of 'data'(JX_Int type) so that 'data' is  
 *        ascending in such order
 * \param *data pointer to the array 
 * \param left the starting index
 * \param right the ending index
 * \param *index the index of 'data'
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively, 
 *       where n is the length of 'data'. 'index', with the same length with
 *       'data', should be initialized in the nature order.   
 * \date 2010/12/10 
 * \author peghoty
 */
void 
fsls_iQuickSortIndex12( JX_Int *data, JX_Int left, JX_Int right, JX_Int *index )
{
   JX_Int i,j;
   JX_Int key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[index[left]];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[index[i]] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[index[j]] <= key) break;
         }
         if (i < j)
         {
            tmp = index[i]; index[i] = index[j]; index[j] = tmp;
         }
      }
      tmp = index[left]; index[left] = index[j]; index[j] = tmp;
      fsls_iQuickSortIndex12(data, left, j-1, index);
      fsls_iQuickSortIndex12(data, i, right, index);
   }
}

/*!
 * \fn void fsls_dQuickSort12
 * \brief Sort the array 'data'(JX_Real type) with quick sorting algorithm
 * \param *a pointer to the array to be sorted
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively,
 *       where n is the length of 'data'.  
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_dQuickSort12( JX_Real *data, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Real key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[j] <= key) break;
         }
         if (i < j)
         {
            tmp = data[i]; data[i] = data[j]; data[j] = tmp;
         }
      }
      tmp = data[left]; data[left] = data[j]; data[j] = tmp;
      fsls_dQuickSort12(data, left, j-1);
      fsls_dQuickSort12(data, i, right);
   }
}

/*!
 * \fn void fsls_dQuickSortIndex12
 * \brief Reorder the index of 'data'(JX_Real type) so that 'data' is  
 *        ascending in such order
 * \param *data pointer to the array 
 * \param left the starting index
 * \param right the ending index
 * \param *index the index of 'data'
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively, 
 *       where n is the length of 'data'. 'index', with the same length with
 *       'data', should be initialized in the nature order.   
 * \date 2010/12/10 
 * \author peghoty
 */
void 
fsls_dQuickSortIndex12( JX_Real *data, JX_Int left, JX_Int right, JX_Int *index )
{
   JX_Int i,j;
   JX_Real key;
   JX_Int tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[index[left]];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[index[i]] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[index[j]] <= key) break;
         }
         if (i < j)
         {
            tmp = index[i]; index[i] = index[j]; index[j] = tmp;
         }
      }
      tmp = index[left]; index[left] = index[j]; index[j] = tmp;
      fsls_dQuickSortIndex12(data, left, j-1, index);
      fsls_dQuickSortIndex12(data, i, right, index);
   }
}

/*!
 * \fn void fsls_iiQuickSort12
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v'(usually be the indexes array of 'w')  
 *        is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_iiQuickSort12( JX_Int *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] <= key) break;
         }
         if (i < j)
         {
            tmp = w[i]; w[i] = w[j]; w[j] = tmp;
            tmp = v[i]; v[i] = v[j]; v[j] = tmp;
         }
      }
      tmp = w[left]; w[left] = w[j]; w[j] = tmp;
      tmp = v[left]; v[left] = v[j]; v[j] = tmp;
      fsls_iiQuickSort12(v, w, left, j-1);
      fsls_iiQuickSort12(v, w, i, right);
   }
}

/*!
 * \fn void fsls_ddQuickSort12
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_ddQuickSort12( JX_Real *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Real key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] <= key) break;
         }
         if (i < j)
         {
            tmp = w[i]; w[i] = w[j]; w[j] = tmp;
            tmp = v[i]; v[i] = v[j]; v[j] = tmp;
         }
      }
      tmp = w[left]; w[left] = w[j]; w[j] = tmp;
      tmp = v[left]; v[left] = v[j]; v[j] = tmp;
      fsls_ddQuickSort12(v, w, left, j-1);
      fsls_ddQuickSort12(v, w, i, right);
   }
}

/*!
 * \fn void fsls_idQuickSort12
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_idQuickSort12( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int tmpv;
   JX_Real key,tmpw;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] <= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      fsls_idQuickSort12(v, w, left, j-1);
      fsls_idQuickSort12(v, w, i, right);
   }
}

/*!
 * \fn void fsls_idFabsQuickSort12
 * \brief Sort the array '|w|' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/20  
 */
void 
fsls_idFabsQuickSort12( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int tmpv;
   JX_Real key,tmpw;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = fabs(w[left]);
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (fabs(w[i]) >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (fabs(w[j]) <= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      fsls_idFabsQuickSort12(v, w, left, j-1);
      fsls_idFabsQuickSort12(v, w, i, right);
   }
}

/*!
 * \fn void fsls_diQuickSort12
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_diQuickSort12( JX_Real *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmpw;
   JX_Real tmpv;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] >= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] <= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      fsls_diQuickSort12(v, w, left, j-1);
      fsls_diQuickSort12(v, w, i, right);
   }
}

/*!
 * \fn void fsls_iQuickSort21
 * \brief Sort the array 'data'(JX_Int type) with quick sorting algorithm
 * \param *data pointer to the array to be sorted
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively,
 *       where n is the length of 'data'.  
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_iQuickSort21( JX_Int *data, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[j] >= key) break;
         }
         if (i < j)
         {
            tmp = data[i]; data[i] = data[j]; data[j] = tmp;
         }
      }
      tmp = data[left]; data[left] = data[j]; data[j] = tmp;
      fsls_iQuickSort21(data, left, j-1);
      fsls_iQuickSort21(data, i, right);
   }
}

/*!
 * \fn void fsls_iQuickSortIndex21
 * \brief Reorder the index of 'data'(JX_Int type) so that 'data' is  
 *        descending in such order
 * \param *data pointer to the array 
 * \param left the starting index
 * \param right the ending index
 * \param *index the index of 'data'
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively, 
 *       where n is the length of 'data'. 'index', with the same length with
 *       'data', should be initialized in the nature order.   
 * \date 2010/12/10 
 * \author peghoty
 */
void 
fsls_iQuickSortIndex21( JX_Int *data, JX_Int left, JX_Int right, JX_Int *index )
{
   JX_Int i,j;
   JX_Int key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[index[left]];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[index[i]] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[index[j]] >= key) break;
         }
         if (i < j)
         {
            tmp = index[i]; index[i] = index[j]; index[j] = tmp;
         }
      }
      tmp = index[left]; index[left] = index[j]; index[j] = tmp;
      fsls_iQuickSortIndex21(data, left, j-1, index);
      fsls_iQuickSortIndex21(data, i, right, index);
   }
}

/*!
 * \fn void fsls_dQuickSort21
 * \brief Sort the array 'data'(JX_Real type) with quick sorting algorithm
 * \param *a pointer to the array to be sorted
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively,
 *       where n is the length of 'data'.  
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_dQuickSort21( JX_Real *data, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Real key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[j] >= key) break;
         }
         if (i < j)
         {
            tmp = data[i]; data[i] = data[j]; data[j] = tmp;
         }
      }
      tmp = data[left]; data[left] = data[j]; data[j] = tmp;
      fsls_dQuickSort21(data, left, j-1);
      fsls_dQuickSort21(data, i, right);
   }
}

/*!
 * \fn void fsls_dQuickSortIndex21
 * \brief Reorder the index of 'data'(JX_Real type) so that 'data' is  
 *        descending in such order
 * \param *data pointer to the array 
 * \param left the starting index
 * \param right the ending index
 * \param *index the index of 'data'
 * \note 'left' and 'right' are usually set to be 0 and n-1, respectively, 
 *       where n is the length of 'data'. 'index', with the same length with
 *       'data', should be initialized in the nature order.   
 * \date 2010/12/10 
 * \author peghoty
 */
void 
fsls_dQuickSortIndex21( JX_Real *data, JX_Int left, JX_Int right, JX_Int *index )
{
   JX_Int i,j;
   JX_Real key;
   JX_Int tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = data[index[left]];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (data[index[i]] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (data[index[j]] >= key) break;
         }
         if (i < j)
         {
            tmp = index[i]; index[i] = index[j]; index[j] = tmp;
         }
      }
      tmp = index[left]; index[left] = index[j]; index[j] = tmp;
      fsls_dQuickSortIndex21(data, left, j-1, index);
      fsls_dQuickSortIndex21(data, i, right, index);
   }
}

/*!
 * \fn void fsls_iiQuickSort21
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v'(usually be the indexes array of 'w')  
 *        is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_iiQuickSort21( JX_Int *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] >= key) break;
         }
         if (i < j)
         {
            tmp = w[i]; w[i] = w[j]; w[j] = tmp;
            tmp = v[i]; v[i] = v[j]; v[j] = tmp;
         }
      }
      tmp = w[left]; w[left] = w[j]; w[j] = tmp;
      tmp = v[left]; v[left] = v[j]; v[j] = tmp;
      fsls_iiQuickSort21(v, w, left, j-1);
      fsls_iiQuickSort21(v, w, i, right);
   }
}

/*!
 * \fn void fsls_ddQuickSort21
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_ddQuickSort21( JX_Real *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Real key,tmp;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] >= key) break;
         }
         if (i < j)
         {
            tmp = w[i]; w[i] = w[j]; w[j] = tmp;
            tmp = v[i]; v[i] = v[j]; v[j] = tmp;
         }
      }
      tmp = w[left]; w[left] = w[j]; w[j] = tmp;
      tmp = v[left]; v[left] = v[j]; v[j] = tmp;
      fsls_ddQuickSort21(v, w, left, j-1);
      fsls_ddQuickSort21(v, w, i, right);
   }
}

/*!
 * \fn void fsls_idQuickSort21
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_idQuickSort21( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int tmpv;
   JX_Real key,tmpw;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] >= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      fsls_idQuickSort21(v, w, left, j-1);
      fsls_idQuickSort21(v, w, i, right);
   }
}

/*!
 * \fn void fsls_idFabsQuickSort21
 * \brief Sort the array '|w|' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/20  
 */
void 
fsls_idFabsQuickSort21( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int tmpv;
   JX_Real key,tmpw;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = fabs(w[left]);
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (fabs(w[i]) <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (fabs(w[j]) >= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      fsls_idFabsQuickSort21(v, w, left, j-1);
      fsls_idFabsQuickSort21(v, w, i, right);
   }
}

/*!
 * \fn void fsls_diQuickSort21
 * \brief Sort the array 'w' with quick sorting algorithm, 
 *        and meanwhile 'v' is sorted in the same way.
 * \param *v pointer to the JX_Int array
 * \param *w pointer to the JX_Int array
 * \param left the starting index
 * \param right the ending index
 * \note 'left' and 'right' are usually set to be 0 and n-1, 
 *        respectively, where n is the length of 'v' or 'w'. 
 * \author peghoty
 * \date 2010/12/10  
 */
void 
fsls_diQuickSort21( JX_Real *v, JX_Int *w, JX_Int left, JX_Int right )
{
   JX_Int i,j;
   JX_Int key,tmpw;
   JX_Real tmpv;
   if (left < right)
   {
      i = left;
      j = right + 1;
      key = w[left];
      while (i < j)
      {
         for (i = i+1; i < right; i ++)
         {
            if (w[i] <= key) break;
         }
         for (j = j-1; j > left; j --)
         {
            if (w[j] >= key) break;
         }
         if (i < j)
         {
            tmpw = w[i]; w[i] = w[j]; w[j] = tmpw;
            tmpv = v[i]; v[i] = v[j]; v[j] = tmpv;
         }
      }
      tmpw = w[left]; w[left] = w[j]; w[j] = tmpw;
      tmpv = v[left]; v[left] = v[j]; v[j] = tmpv;
      fsls_diQuickSort21(v, w, left, j-1);
      fsls_diQuickSort21(v, w, i, right);
   }
}

/*-----------------------------------
 * Static variables
 *----------------------------------*/
 
static JX_Int Seed = 13579;

#define a  16807
#define m  2147483647
#define q  127773
#define r  2836

/*!
 * \fn void fsls_SeedRand
 * \brief Initializes the pseudo-random number 
 *        generator to a place in the sequence.
 * \param seed an JX_Int containing the seed for the RNG.
 * \date 2010/08/29
 */
void  
fsls_SeedRand( JX_Int seed )
{
   Seed = seed;
}


/*!
 * \fn JX_Real fsls_Rand
 * \brief Computes the next pseudo-random number in 
 *        the sequence using the global variable Seed.
 * \return a JX_Real containing the next number 
 *         in the sequence divided by 2147483647
 *         so that the numbers are in (0, 1]
 * \date 2010/08/29
 */
JX_Real  
fsls_Rand()
{
   JX_Int low, high, test;

   high = Seed / q;
   low  = Seed % q;
   test = a * low - r * high;
   
   if (test > 0)
   {
      Seed = test;
   }
   else
   {
      Seed = test + m;
   }
   
   return ((JX_Real)(Seed) / m);
}

#undef a  
#undef m  
#undef q  
#undef r 

