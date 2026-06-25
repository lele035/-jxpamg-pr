//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  seq_csr_matrix.c -- basic operations for CSR matrices.
 *  Date: 2011/09/03
 */ 
 
#include "jx_mv.h"

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixCreate
 * \brief Create a CSR format matrix.
 * \param num_rows number of rows
 * \param num_cols number of columns
 * \param num_nonzeros number of nonzero entries
 * \date 2011/09/03
 */
jx_CSRMatrix *
jx_CSRMatrixCreate( JX_Int num_rows, JX_Int num_cols, JX_Int num_nonzeros )
{
   jx_CSRMatrix  *matrix;

   matrix = jx_CTAlloc(jx_CSRMatrix, 1);

   jx_CSRMatrixData(matrix) = NULL;
   jx_CSRMatrixI(matrix)    = NULL;
   jx_CSRMatrixJ(matrix)    = NULL;
   jx_CSRMatrixRownnz(matrix) = NULL;
   jx_CSRMatrixNumRows(matrix) = num_rows;
   jx_CSRMatrixNumCols(matrix) = num_cols;
   jx_CSRMatrixNumNonzeros(matrix) = num_nonzeros;

   /* set defaults */
   jx_CSRMatrixOwnsData(matrix) = 1;
   jx_CSRMatrixNumRownnz(matrix) = num_rows;

   return matrix;
}

/*!
 * \fn JX_Int jx_CSRMatrixInitialize
 * \brief Initialize a CSR format matrix.
 * \param *matrix pointer to the matrix to be initialized.
 * \date 2011/09/03
 */
JX_Int 
jx_CSRMatrixInitialize( jx_CSRMatrix *matrix )
{
   JX_Int  num_rows     = jx_CSRMatrixNumRows(matrix);
   JX_Int  num_nonzeros = jx_CSRMatrixNumNonzeros(matrix);

   JX_Int  ierr = 0;

   if ( ! jx_CSRMatrixData(matrix) && num_nonzeros )
      jx_CSRMatrixData(matrix) = jx_CTAlloc(JX_Real, num_nonzeros);
   if ( ! jx_CSRMatrixI(matrix) )
      jx_CSRMatrixI(matrix)    = jx_CTAlloc(JX_Int, num_rows + 1);
   if ( ! jx_CSRMatrixJ(matrix) && num_nonzeros )
      jx_CSRMatrixJ(matrix)    = jx_CTAlloc(JX_Int, num_nonzeros);

   return ierr;
}


/*!
 * \fn JX_Int jx_CSRMatrixDestroy
 * \brief Destroy a CSR format matrix.
 * \param *matrix pointer to the matrix to be destroyed.
 * \date 2011/09/03
 */
JX_Int 
jx_CSRMatrixDestroy( jx_CSRMatrix *matrix )
{
   JX_Int  ierr=0;

   if (matrix)
   {
      jx_TFree(jx_CSRMatrixI(matrix));
      if (jx_CSRMatrixRownnz(matrix))
         jx_TFree(jx_CSRMatrixRownnz(matrix));
      if ( jx_CSRMatrixOwnsData(matrix) )
      {
         jx_TFree(jx_CSRMatrixData(matrix));
         jx_TFree(jx_CSRMatrixJ(matrix));
      }
      jx_TFree(matrix);
   }

   return ierr;
}

/*!
 * \fn JX_Int jx_CSRMatrixPrint
 * \brief Print a CSR format matrix into given file.
 *        format of the file:
 *        --------------------------------------------
 *           nrows
 *           ia[i], i=0(1)nrows
 *           ja[i], i=0(1)nonzeros
 *            a[i], i=0(1)nonzeros
 *        -------------------------------------------- 
 *        There is only one number in each line, row and colum 
 *        index are of fortran style.    
 * \param *matrix pointer to the matrix to be printed.
 * \param *file_name pointer to the file name.
 * \date 2011/09/03
 */
JX_Int
jx_CSRMatrixPrint( jx_CSRMatrix *matrix, char *file_name )
{
   FILE    *fp;

   JX_Real  *matrix_data;
   JX_Int     *matrix_i;
   JX_Int     *matrix_j;
   JX_Int      num_rows;
   
   JX_Int      file_base = 1;
   
   JX_Int      j;

   JX_Int      ierr = 0;

  /*--------------------------------------------
   * Print the matrix data
   *-------------------------------------------*/
   
   matrix_data = jx_CSRMatrixData(matrix);
   matrix_i    = jx_CSRMatrixI(matrix);
   matrix_j    = jx_CSRMatrixJ(matrix);
   num_rows    = jx_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   jx_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j ++)
   {
      jx_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j ++)
   {
      jx_fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j ++)
      {
         jx_fprintf(fp, "%.14e\n", matrix_data[j]);
      }
   }
   else
   {
      jx_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixRead
 * \brief Read a CSR format matrix from given file.
 *        format of the file:
 *        --------------------------------------------
 *           nrows
 *           ia[i], i=0(1)nrows
 *           ja[i], i=0(1)nonzeros
 *            a[i], i=0(1)nonzeros
 *        -------------------------------------------- 
 *        There is only one number in each line, row and colum 
 *        index are of fortran (fb=1) or C (fb=0) style.    
 * \param *file_name pointer to the file name.
 * \date 2011/09/03
 */
jx_CSRMatrix *
jx_CSRMatrixRead( char *file_name, JX_Int file_base )
{
   jx_CSRMatrix  *matrix;

   FILE    *fp;

   JX_Real  *matrix_data;
   JX_Int     *matrix_i;
   JX_Int     *matrix_j;
   JX_Int      num_rows;
   JX_Int      num_nonzeros;
   JX_Int      max_col = 0;

   JX_Int      j;

  /*------------------------------------------
   *  Read in the data
   *----------------------------------------*/

   fp = fopen(file_name, "r");

   jx_fscanf(fp, "%d", &num_rows);

   matrix_i = jx_CTAlloc(JX_Int, num_rows + 1);
   for (j = 0; j < num_rows+1; j ++)
   {
      jx_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = jx_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   jx_CSRMatrixI(matrix) = matrix_i;
   jx_CSRMatrixInitialize(matrix);

   matrix_j = jx_CSRMatrixJ(matrix);
   for (j = 0; j < num_nonzeros; j ++)
   {
      jx_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = jx_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j ++)
   {
      jx_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   jx_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   jx_CSRMatrixNumCols(matrix) = ++ max_col;

   return matrix;
}

jx_CSRMatrix *
jx_CSRMatrixRead2( char *file_name, JX_Int file_base )
{
   jx_CSRMatrix  *matrix;

   FILE    *fp;

   JX_Real  *matrix_data;
   JX_Int     *matrix_i;
   JX_Int     *matrix_j;
   JX_Int      num_rows;
   JX_Int      num_nonzeros;
   JX_Int      max_col = 0;

   JX_Int      j, tmp;

  /*------------------------------------------
   *  Read in the data
   *----------------------------------------*/

   fp = fopen(file_name, "r");

   jx_fscanf(fp, "%d", &num_rows);
   jx_fscanf(fp, "%d", &num_nonzeros);
   for (j = 0; j < num_rows; j ++)
   {
      jx_fscanf(fp, "%d", &tmp);
   }

   matrix_i = jx_CTAlloc(JX_Int, num_rows + 1);
   for (j = 0; j < num_rows+1; j ++)
   {
      jx_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   jx_assert(matrix_i[num_rows] == num_nonzeros);

   matrix = jx_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   jx_CSRMatrixI(matrix) = matrix_i;
   jx_CSRMatrixInitialize(matrix);

   matrix_j = jx_CSRMatrixJ(matrix);
   for (j = 0; j < num_nonzeros; j ++)
   {
      jx_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = jx_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j ++)
   {
      jx_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   jx_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   jx_CSRMatrixNumCols(matrix) = ++ max_col;

   return matrix;
}

/*!
 * \fn JX_Int jx_CSRMatrixGetBandWidth
 * \brief Get the left and right bandwidth
 * \author Yue Xiaoqiang
 * \date 2012/10/12
 */
JX_Int
jx_CSRMatrixGetBandWidth( jx_CSRMatrix *A, JX_Int *nbl_ptr, JX_Int *nbr_ptr )
{
   JX_Int ierr = 0;
   JX_Int *IA = jx_CSRMatrixI(A);
   JX_Int *JA = jx_CSRMatrixJ(A);
   JX_Int myid, mybegin, myend;
   JX_Int max_l, max_r;
   JX_Int i, end_row_A, j;
   JX_Int num_threads;
   
   num_threads = jx_NumThreads();
   
   // num_rows must be greater than openmp_holds
   JX_Int *max_left_right = jx_CTAlloc(JX_Int, 2*num_threads);
#define JX_SMP_PRIVATE myid,mybegin,myend,max_l,max_r,i,end_row_A,j
#include "../../include/jx_smp_forloop.h"
   for (myid = 0; myid < num_threads; myid ++)
   {
      JX_OMP_GET_START_END(myid, num_threads, A->num_rows, mybegin, myend);
      max_l = 0;
      max_r = 0;
      for (i = mybegin; i < myend; i ++)
      {
         end_row_A = IA[i+1];
         for (j = IA[i]; j < end_row_A; j ++)
         {
            max_l = jx_max(i-JA[j], max_l);
            max_r = jx_max(JA[j]-i, max_r);
         }
      }
      max_left_right[myid*2] = max_l;
      max_left_right[myid*2+1] = max_r;
   }
   max_l = max_left_right[0];
   max_r = max_left_right[1];
   for (i = 1; i < num_threads; i ++)
   {
      max_l = jx_max(max_l, max_left_right[i*2]);
      max_r = jx_max(max_r, max_left_right[i*2+1]);
   }
   jx_TFree(max_left_right);
  *nbl_ptr = max_l;
  *nbr_ptr = max_r;
   
   return ierr;
}

/*!
 * \fn JX_Int jx_CSRMatrixSetRownnz
 * \brief function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 * \date 2015/11/28
 */
JX_Int
jx_CSRMatrixSetRownnz( jx_CSRMatrix *matrix )
{
   JX_Int  ierr=0;
   JX_Int  num_rows = jx_CSRMatrixNumRows(matrix);
   JX_Int *A_i = jx_CSRMatrixI(matrix);
   JX_Int *Arownnz;
   JX_Int i, adiag;
   JX_Int irownnz=0;

   for (i=0; i < num_rows; i++)
   {
      adiag = (A_i[i+1] - A_i[i]);
      if(adiag > 0) irownnz++;
   }
   jx_CSRMatrixNumRownnz(matrix) = irownnz;
   if ((irownnz == 0) || (irownnz == num_rows))
   {
      jx_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
      Arownnz = jx_CTAlloc(JX_Int, irownnz);
      irownnz = 0;
      for (i=0; i < num_rows; i++)
      {
         adiag = A_i[i+1]-A_i[i];
         if(adiag > 0) Arownnz[irownnz++] = i;
      }
      jx_CSRMatrixRownnz(matrix) = Arownnz;
   }
   return ierr;
}
