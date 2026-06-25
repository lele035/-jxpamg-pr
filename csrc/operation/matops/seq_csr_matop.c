//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_csr_matop.c -- basic operations for CSR matrices.
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_CSRMatrixCopy
 * \brief Copys A to B. If copy_data = 0 only the structure of A is copied to B.
 * \note The routine does not check if the dimensions of A and B match !    
 * \date 2011/09/03
 */
JX_Int 
jx_CSRMatrixCopy( jx_CSRMatrix *A, jx_CSRMatrix *B, JX_Int copy_data )
{
   JX_Int  ierr=0;
   JX_Int  num_rows = jx_CSRMatrixNumRows(A);
   JX_Int *A_i = jx_CSRMatrixI(A);
   JX_Int *A_j = jx_CSRMatrixJ(A);
   JX_Real *A_data;
   JX_Int *B_i = jx_CSRMatrixI(B);
   JX_Int *B_j = jx_CSRMatrixJ(B);
   JX_Real *B_data;

   JX_Int i, j;

   for (i=0; i < num_rows; i++)
   {
	B_i[i] = A_i[i];
	for (j = A_i[i]; j < A_i[i+1]; j ++)
	{
		B_j[j] = A_j[j];
	}
   }
   B_i[num_rows] = A_i[num_rows];
   if (copy_data)
   {
	A_data = jx_CSRMatrixData(A);
	B_data = jx_CSRMatrixData(B);
   	for (i = 0; i < num_rows; i ++)
   	{
	   for (j = A_i[i]; j < A_i[i+1]; j ++)
	   {
		B_data[j] = A_data[j];
	   }
	}
   }
   return ierr;
}

/*!
 * \fn JX_Int jx_CSRMatrixTranspose
 * \brief Transpose a CSR matrix.
 * \param *A pointer to the original matrix.
 * \param **AT pointer to pointer to the transposed matrix.
 * \param data flag to indicate whether transpose the data part together?
 *        data = 0: only the structure of A will be transposed;
 *        data = 1: both the structure and the data part will be transposed.
 * \date 2011/09/03
 */
JX_Int 
jx_CSRMatrixTranspose( jx_CSRMatrix *A, jx_CSRMatrix **AT, JX_Int data )
{
   JX_Real       *A_data = jx_CSRMatrixData(A);
   JX_Int          *A_i    = jx_CSRMatrixI(A);
   JX_Int          *A_j    = jx_CSRMatrixJ(A);
   JX_Int           num_rowsA = jx_CSRMatrixNumRows(A);
   JX_Int           num_colsA = jx_CSRMatrixNumCols(A);
   JX_Int           num_nonzerosA = jx_CSRMatrixNumNonzeros(A);

   JX_Real       *AT_data = NULL;
   JX_Int          *AT_i;
   JX_Int          *AT_j;
   JX_Int           num_rowsAT;
   JX_Int           num_colsAT;
   JX_Int           num_nonzerosAT;

   JX_Int           max_col;
   JX_Int           i, j;

  /*-------------------------------------------------------------- 
   * First, ascertain that num_cols and num_nonzeros has been set. 
   * If not, set them.
   *--------------------------------------------------------------*/

   if (! num_nonzerosA)
   {
      num_nonzerosA = A_i[num_rowsA];
   }

   if (num_rowsA && ! num_colsA)
   {
      max_col = -1;
      for (i = 0; i < num_rowsA; ++ i)
      {
          for (j = A_i[i]; j < A_i[i+1]; j ++)
          {
              if (A_j[j] > max_col)
                 max_col = A_j[j];
          }
      }
      num_colsA = max_col + 1;
   }

   num_rowsAT = num_colsA;
   num_colsAT = num_rowsA;
   num_nonzerosAT = num_nonzerosA;

   *AT = jx_CSRMatrixCreate(num_rowsAT, num_colsAT, num_nonzerosAT);

   AT_i = jx_CTAlloc(JX_Int, num_rowsAT + 1);
   AT_j = jx_CTAlloc(JX_Int, num_nonzerosAT);
   jx_CSRMatrixI(*AT) = AT_i;
   jx_CSRMatrixJ(*AT) = AT_j;
   if (data) 
   {
      AT_data = jx_CTAlloc(JX_Real, num_nonzerosAT);
      jx_CSRMatrixData(*AT) = AT_data;
   }

  /*-----------------------------------------------------------------
   * Count the number of entries in each column of A (row of AT)
   * and fill the AT_i array.
   *-----------------------------------------------------------------*/

   for (i = 0; i < num_nonzerosA; i ++)
   {
       ++ AT_i[A_j[i]+1];
   }

   for (i = 2; i <= num_rowsAT; i ++)
   {
       AT_i[i] += AT_i[i-1];
   }

  /*-------------------------------------------------------------
   * Load the data and column numbers of AT
   *-----------------------------------------------------------*/

   for (i = 0; i < num_rowsA; i ++)
   {
      for (j = A_i[i]; j < A_i[i+1]; j ++)
      {
         jx_assert( AT_i[A_j[j]] >= 0 );
         jx_assert( AT_i[A_j[j]] < num_nonzerosAT );
         AT_j[AT_i[A_j[j]]] = i;
         if (data) 
         {
            AT_data[AT_i[A_j[j]]] = A_data[j];
         }
         AT_i[A_j[j]] ++;
      }
   }

  /*------------------------------------------------------------
   * AT_i[j] now points to the *end* of the jth row of entries
   * instead of the beginning.  Restore AT_i to front of row.
   *------------------------------------------------------------*/

   for (i = num_rowsAT; i > 0; i --)
   {
      AT_i[i] = AT_i[i-1];
   }

   AT_i[0] = 0;

   return(0);
}

/*!
 * \fn JX_Int jx_CSRMatrixReorder
 * \brief Reorders the column and data arrays of a square CSR matrix, 
 *        such that the first entry in each row is the diagonal one.
 * \date 2011/09/03
 */
JX_Int 
jx_CSRMatrixReorder( jx_CSRMatrix *A )
{
   JX_Int i, j, tempi, row_size;
   JX_Real tempd;

   JX_Real *A_data = jx_CSRMatrixData(A);
   JX_Int    *A_i = jx_CSRMatrixI(A);
   JX_Int    *A_j = jx_CSRMatrixJ(A);
   JX_Int     num_rowsA = jx_CSRMatrixNumRows(A);
   JX_Int     num_colsA = jx_CSRMatrixNumCols(A);

   /* the matrix should be square */
   if (num_rowsA != num_colsA)
      return -1;

   for (i = 0; i < num_rowsA; i ++)
   {
      row_size = A_i[i+1] - A_i[i];

      for (j = 0; j < row_size; j ++)
      {
         if (A_j[j] == i)
         {
            if (j != 0)
            {
               tempi = A_j[0];
               A_j[0] = A_j[j];
               A_j[j] = tempi;

               tempd = A_data[0];
               A_data[0] = A_data[j];
               A_data[j] = tempd;
            }
            break;
         }

         /* diagonal element is missing */
         if (j == row_size - 1)
            return -2;
      }

      A_j    += row_size;
      A_data += row_size;
   }

   return 0;
}

/*!
 * \fn jx_CSRMatrixReorderColumnNumber12
 * \brief Reorder the column number of each row ascendingly while keeping
 *        all the diagonal entries firstly-stored. 
 * \note Both the column number and data part are reordered. 
 * \author peghoty
 * \date 2010/10/20
 */
void
jx_CSRMatrixReorderColumnNumber12( jx_CSRMatrix *A )
{
   JX_Int i,begin,end;
   JX_Int n = jx_CSRMatrixNumRows(A);
   JX_Int *ia = jx_CSRMatrixI(A);
   JX_Int *ja = jx_CSRMatrixJ(A);
   JX_Real *a = jx_CSRMatrixData(A);

   for (i = 0; i < n; i ++)
   {
      begin = ia[i] + 1;
      end   = ia[i+1] - 1;
      jx_diQuickSort12(a, ja, begin, end);       
   }
}

/*!
 * \fn jx_CSRMatrixReorderColumnNumberAll
 * \brief Reorder the column number of each row ascendingly 
 *        and the diagonal entries are included. 
 * \note Both the column number and data part are reordered.  
 * \author peghoty
 * \date 2010/10/29
 */
void
jx_CSRMatrixReorderColumnNumberAll( jx_CSRMatrix *A )
{
   JX_Int i,begin,end;
   JX_Int n = jx_CSRMatrixNumRows(A);
   JX_Int *ia = jx_CSRMatrixI(A);
   JX_Int *ja = jx_CSRMatrixJ(A);
   JX_Real *a = jx_CSRMatrixData(A);

   for (i = 0; i < n; i ++)
   {
      begin = ia[i];
      end   = ia[i+1] - 1;
      jx_diQuickSort12(a, ja, begin, end);      
   }
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixAdd
 * \brief adds two CSR Matrices A and B and returns a CSR Matrix C
 * \date 2009/12/05
 */
jx_CSRMatrix *
jx_CSRMatrixAdd( jx_CSRMatrix *A, jx_CSRMatrix *B )
{
   JX_Real     *A_data   = jx_CSRMatrixData(A);
   JX_Int        *A_i      = jx_CSRMatrixI(A);
   JX_Int        *A_j      = jx_CSRMatrixJ(A);
   JX_Int         nrows_A  = jx_CSRMatrixNumRows(A);
   JX_Int         ncols_A  = jx_CSRMatrixNumCols(A);

   JX_Real     *B_data   = jx_CSRMatrixData(B);
   JX_Int        *B_i      = jx_CSRMatrixI(B);
   JX_Int        *B_j      = jx_CSRMatrixJ(B);
   JX_Int         nrows_B  = jx_CSRMatrixNumRows(B);
   JX_Int         ncols_B  = jx_CSRMatrixNumCols(B);

   jx_CSRMatrix *C      = NULL;
   JX_Real       *C_data = NULL;
   JX_Int	        *C_i    = NULL;
   JX_Int          *C_j    = NULL;

   JX_Int         ia, ib, ic, jcol, num_nonzeros;
   JX_Int	       pos;
   JX_Int        *marker;

   /* check the compatibility of A and B */
   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      jx_printf(" >>> Warning: Incompatible matrix dimensions!\n");
      return NULL;
   }

   marker = jx_CTAlloc(JX_Int, ncols_A);
   C_i = jx_CTAlloc(JX_Int, nrows_A+1);

   /* initialize the marker array */
   for (ia = 0; ia < ncols_A; ia ++)
   {
      marker[ia] = -1;
   }

   num_nonzeros = 0;
   C_i[0] = 0;
   for (ic = 0; ic < nrows_A; ic ++)
   {
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia ++)
	{
            jcol = A_j[ia];
            marker[jcol] = ic;
            num_nonzeros ++;
	}
	for (ib = B_i[ic]; ib < B_i[ic+1]; ib ++)
	{
            jcol = B_j[ib];
            if (marker[jcol] != ic)
            {
               marker[jcol] = ic;
               num_nonzeros ++;
            }
   	}
	C_i[ic+1] = num_nonzeros;
   }

   C = jx_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros);
   jx_CSRMatrixI(C) = C_i;
   jx_CSRMatrixInitialize(C);
   C_j = jx_CSRMatrixJ(C);
   C_data = jx_CSRMatrixData(C);

   /* initialize the marker array again */
   for (ia = 0; ia < ncols_A; ia ++)
   {
      marker[ia] = -1;
   }

   pos = 0;
   for (ic = 0; ic < nrows_A; ic ++)
   {
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia ++)
	{
            jcol = A_j[ia];
            C_j[pos] = jcol;
            C_data[pos] = A_data[ia];
            marker[jcol] = pos;
            pos ++;
	}
	for (ib = B_i[ic]; ib < B_i[ic+1]; ib ++)
	{
            jcol = B_j[ib];
            if (marker[jcol] < C_i[ic])  /* very skillful */
            {  /* the col number 'jcol' doesn't appear in the ic-th row of A */
               C_j[pos] = jcol;
               C_data[pos] = B_data[ib];
               marker[jcol] = pos;
               pos ++;
            }
            else
            {  /* the col number 'jcol' has appeared in the ic-th row of A */
               C_data[marker[jcol]] += B_data[ib];
            }
   	}
   }

   jx_TFree(marker);
   return C;
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixMultiply
 * \brief multiplies two CSR Matrices A and B and returns a CSR Matrix C.
 * \date 2017/02/25
 */
jx_CSRMatrix *
jx_CSRMatrixMultiply( jx_CSRMatrix *A, jx_CSRMatrix *B )
{
   JX_Real *A_data   = jx_CSRMatrixData(A);
   JX_Int    *A_i      = jx_CSRMatrixI(A);
   JX_Int    *A_j      = jx_CSRMatrixJ(A);
   JX_Int     nrows_A  = jx_CSRMatrixNumRows(A);
   JX_Int     ncols_A  = jx_CSRMatrixNumCols(A);
   JX_Real *B_data   = jx_CSRMatrixData(B);
   JX_Int    *B_i      = jx_CSRMatrixI(B);
   JX_Int    *B_j      = jx_CSRMatrixJ(B);
   JX_Int     nrows_B  = jx_CSRMatrixNumRows(B);
   JX_Int     ncols_B  = jx_CSRMatrixNumCols(B);

   jx_CSRMatrix *C;
   JX_Real       *C_data;
   JX_Int          *C_i;
   JX_Int          *C_j;

   JX_Int    ia, ib, ic, ja, jb, num_nonzeros=0;
   JX_Int    row_start, counter;
   JX_Real a_entry, b_entry;
   JX_Int    allsquare = 0;
   JX_Int    max_num_threads;
   JX_Int   *jj_count;

   if (ncols_A != nrows_B)
   {
      jx_printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   if (nrows_A == ncols_B) allsquare = 1;

   C_i = jx_CTAlloc(JX_Int, nrows_A+1);

   max_num_threads = jx_NumThreads();

   jj_count = jx_CTAlloc(JX_Int, max_num_threads);

#if JX_USING_OPENMP
#pragma omp parallel private(ia, ib, ic, ja, jb, num_nonzeros, row_start, counter, a_entry, b_entry)
#endif
   {
    JX_Int *B_marker = NULL;
    JX_Int ns, ne, ii, jj;
    JX_Int size, rest, num_threads;
    JX_Int i1;
    ii = jx_GetThreadNum();
    num_threads = jx_NumActiveThreads();

   size = nrows_A/num_threads;
   rest = nrows_A - size*num_threads;
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

    B_marker = jx_CTAlloc(JX_Int, ncols_B);

    for (ib = 0; ib < ncols_B; ib++)
      B_marker[ib] = -1;

    num_nonzeros = 0;
    for (ic = ns; ic < ne; ic++)
    {
        C_i[ic] = num_nonzeros;
	if (allsquare) 
        {
           B_marker[ic] = ic;
           num_nonzeros++;
        }
	for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	{
	   ja = A_j[ia];
	   for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
	   {
	      jb = B_j[ib];
	      if (B_marker[jb] != ic)
	      {
	 	 B_marker[jb] = ic;
		 num_nonzeros++;
	      }
	   }
   	}
   }
   jj_count[ii] = num_nonzeros;

#if JX_USING_OPENMP
#pragma omp barrier
#endif

    if (ii)
    {
       jj = jj_count[0];
       for (i1 = 1; i1 < ii; i1++)
          jj += jj_count[i1];

       for (i1 = ns; i1 < ne; i1++)
          C_i[i1] += jj;
    }
    else
    {
       C_i[nrows_A] = 0;
       for (i1 = 0; i1 < num_threads; i1++)
          C_i[nrows_A] += jj_count[i1];

       C = jx_CSRMatrixCreate(nrows_A, ncols_B, C_i[nrows_A]);
       jx_CSRMatrixI(C) = C_i;
       jx_CSRMatrixInitialize(C);
       C_j = jx_CSRMatrixJ(C);
       C_data = jx_CSRMatrixData(C);
    }

#if JX_USING_OPENMP
#pragma omp barrier
#endif

   for (ib = 0; ib < ncols_B; ib++)
      B_marker[ib] = -1;

   counter = C_i[ns];
   for (ic = ns; ic < ne; ic++)
   {
      row_start = C_i[ic];
      if (allsquare) 
      {
         B_marker[ic] = counter;
         C_data[counter] = 0;
         C_j[counter] = ic;
         counter++;
      }
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
	 ja = A_j[ia];
	 a_entry = A_data[ia];
	 for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
	 {
	    jb = B_j[ib];
	    b_entry = B_data[ib];
	    if (B_marker[jb] < row_start)
	    {
		B_marker[jb] = counter;
		C_j[B_marker[jb]] = jb;
		C_data[B_marker[jb]] = a_entry*b_entry;
		counter++;
	    }
	    else
		C_data[B_marker[jb]] += a_entry*b_entry;
	}
      }
   }
   jx_TFree(B_marker);
  } /*end parallel region */
  jx_TFree(jj_count);
  return C;
}

/*!
 * \fn void jx_CSRMatrixScale
 * \brief Scale a given matrix by multiplying each entries with alpha.
 * \author Yue Xiaoqiang
 * \date 2014/04/14
 */
void
jx_CSRMatrixScale( jx_CSRMatrix *A, JX_Real alpha )
{
   JX_Int i, nz;

   nz = jx_CSRMatrixNumNonzeros(A);
   for (i = 0; i < nz; i ++)
   {
      jx_CSRMatrixData(A)[i] *= alpha;
   }
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixSymmetrization
 * \brief Symmetrize a given matrix A by (A+A^T)/2.
 * \note Base on 'jx_CSRMatrixSymmetrization'
 * \author Yue Xiaoqiang
 * \date 2014/04/14
 */
jx_CSRMatrix *
jx_CSRMatrixSymmetrization( jx_CSRMatrix *A )
{
   jx_CSRMatrix *AT = NULL;
   jx_CSRMatrix *B  = NULL;

   jx_CSRMatrixTranspose(A, &AT, 1);
   B = jx_CSRMatrixAdd(A, AT);
   jx_CSRMatrixScale(B, 0.5);

   jx_CSRMatrixDestroy(AT);

   return B;
}
