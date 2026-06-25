//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  submat.c -- subroutines to extract a or some sub-matrices from a 3tmatrix.
 *  Date: 2011/09/08
 *
 *  The 3T matrices are of the following form:
 *    /           \      /           \  
 *   | A11 A12  0  |    | Arr Are  0  |  
 *   | A21 A22 A23 | or | Aer Aee Aei |
 *   |  0  A32 A33 |    |  0  Aie Aii |
 *    \           /      \           /   
 * 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn JX_Int jx_3tGetSubBlocks
 * \brief Get all the blocks of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks( jx_CSRMatrix  *A, 
                   jx_CSRMatrix **ARR_ptr, 
                   jx_CSRMatrix **AEE_ptr, 
                   jx_CSRMatrix **AII_ptr, 
                   jx_CSRMatrix **ARE_ptr, 
                   jx_CSRMatrix **AER_ptr, 
                   jx_CSRMatrix **AEI_ptr, 
                   jx_CSRMatrix **AIE_ptr )
{
   jx_CSRMatrix *A11 = NULL; 
   jx_CSRMatrix *A22 = NULL;
   jx_CSRMatrix *A33 = NULL;
   jx_CSRMatrix *A12 = NULL; 
   jx_CSRMatrix *A21 = NULL; 
   jx_CSRMatrix *A23 = NULL; 
   jx_CSRMatrix *A32 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia11 = NULL;
   JX_Int    *ia22 = NULL;
   JX_Int    *ia33 = NULL;
   JX_Int    *ia12 = NULL;
   JX_Int    *ia21 = NULL;
   JX_Int    *ia23 = NULL;
   JX_Int    *ia32 = NULL;
   
   JX_Int    *ja11 = NULL;
   JX_Int    *ja22 = NULL;
   JX_Int    *ja33 = NULL;
   JX_Int    *ja12 = NULL;
   JX_Int    *ja21 = NULL;
   JX_Int    *ja23 = NULL;
   JX_Int    *ja32 = NULL;
   
   JX_Real *a11 = NULL;
   JX_Real *a22 = NULL;
   JX_Real *a33 = NULL;
   JX_Real *a12 = NULL;
   JX_Real *a21 = NULL;
   JX_Real *a23 = NULL;
   JX_Real *a32 = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza11,nza22,nza33;
   JX_Int nzv12,nzv21,nzv23,nzv32;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries in each diagonal block
   *------------------------------------------------------------------*/
   nza11 = 0;
   nza22 = 0;
   nza33 = 0;

   //=================================================================
   // Remark: Since Arr, Aee and Aii have the same nonzero-structure,
   //         only nza11 should be counted, nza22 = nza33 = nza11.
   //                                        peghoty 2011/06/26
   //=================================================================
       
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             nza11 ++;
          }
       }     
   }
   nza22 = nza11;
   nza33 = nza11;

  /*-----------------------------------------------------
   * Create 7 CSR submatrices
   *----------------------------------------------------*/
   A11 = jx_CSRMatrixCreate(n,n,nza11);
   jx_CSRMatrixInitialize(A11);
   ia11 = jx_CSRMatrixI(A11);
   ja11 = jx_CSRMatrixJ(A11);     
   a11  = jx_CSRMatrixData(A11);

   A22 = jx_CSRMatrixCreate(n,n,nza22);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);     
   a22  = jx_CSRMatrixData(A22);

   A33 = jx_CSRMatrixCreate(n,n,nza33);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);     
   a33  = jx_CSRMatrixData(A33);

   A12 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A12);
   ia12 = jx_CSRMatrixI(A12);
   ja12 = jx_CSRMatrixJ(A12);     
   a12  = jx_CSRMatrixData(A12);
   
   A21 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A21);
   ia21 = jx_CSRMatrixI(A21);
   ja21 = jx_CSRMatrixJ(A21);     
   a21  = jx_CSRMatrixData(A21);
   
   A23 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A23);
   ia23 = jx_CSRMatrixI(A23);
   ja23 = jx_CSRMatrixJ(A23);     
   a23  = jx_CSRMatrixData(A23);
   
   A32 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A32);
   ia32 = jx_CSRMatrixI(A32);
   ja32 = jx_CSRMatrixJ(A32);     
   a32  = jx_CSRMatrixData(A32);
   
   for (j = 0; j < n; j ++)
   {
      ia12[j] = j; ja12[j] = j;
      ia21[j] = j; ja21[j] = j;
      ia23[j] = j; ja23[j] = j;
      ia32[j] = j; ja32[j] = j;
   }
   ia12[n] = n; 
   ia21[n] = n;
   ia23[n] = n;
   ia32[n] = n;   
     
  /*-----------------------------------------------------
   * Initialization
   *----------------------------------------------------*/
   nza11 = 0; nza22 = 0; nza33 = 0;
   nzv12 = 0; nzv21 = 0; nzv23 = 0;  nzv32 = 0; 
   ia11[0] = 0; ia22[0] = 0; ia33[0] = 0;

  /*-----------------------------------------------------
   * Get the 7 submatrices
   *----------------------------------------------------*/
   
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             ja11[nza11] = col;
              a11[nza11] = a[j];
             nza11 ++;
          }
          else if (col >= n && col < nplusn)
          {
             a12[nzv12++] = a[j];
          }
          else
          {
             jx_printf("\n\nA13 != 0\n\n");
          }
       }
       ia11[row+1] = nza11;
   }
   
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             a21[nzv21++] = a[j];
          }
          else if (col >= n && col < nplusn)
          {
             ja22[nza22] = col - n;
              a22[nza22] = a[j];
             nza22 ++;
          }
          else
          {
             a23[nzv23++] = a[j];
          }
       }
       ia22[row-n+1] = nza22;   
   }   
   
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             jx_printf("\n\nA31 != 0\n\n");
          }
          else if (col >= nplusn)
          {
             ja33[nza33] = col - nplusn;
              a33[nza33] = a[j];
             nza33 ++;
          }
          else
          {
             a32[nzv32++] = a[j];
          }
       }
       ia33[row-nplusn+1] = nza33;
   }  
   
   *ARR_ptr = A11; 
   *AEE_ptr = A22;
   *AII_ptr = A33;
   *ARE_ptr = A12;
   *AER_ptr = A21;
   *AEI_ptr = A23;
   *AIE_ptr = A32;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_REIV
 * \brief Get all the blocks of a 3tmatrix, where the off-diagonal
 *        parts are stored in vectors.
 * \author peghoty
 * \date 2011/09/10
 */
JX_Int
jx_3tGetSubBlocks_REIV( jx_CSRMatrix  *A, 
                        jx_CSRMatrix **ARR_ptr, 
                        jx_CSRMatrix **AEE_ptr, 
                        jx_CSRMatrix **AII_ptr, 
                        jx_Vector    **VRE_ptr, 
                        jx_Vector    **VER_ptr, 
                        jx_Vector    **VEI_ptr, 
                        jx_Vector    **VIE_ptr )
{
   jx_CSRMatrix *A11 = NULL; 
   jx_CSRMatrix *A22 = NULL;
   jx_CSRMatrix *A33 = NULL;
   jx_Vector    *V12 = NULL; 
   jx_Vector    *V21 = NULL; 
   jx_Vector    *V23 = NULL; 
   jx_Vector    *V32 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia11 = NULL;
   JX_Int    *ia22 = NULL;
   JX_Int    *ia33 = NULL;   
   JX_Int    *ja11 = NULL;
   JX_Int    *ja22 = NULL;
   JX_Int    *ja33 = NULL;
   
   JX_Real *a11 = NULL;
   JX_Real *a22 = NULL;
   JX_Real *a33 = NULL;
   
   /* information of vectors */
   JX_Real *v12_data = NULL;
   JX_Real *v21_data = NULL;
   JX_Real *v23_data = NULL;
   JX_Real *v32_data = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza11,nza22,nza33;
   JX_Int nzv12,nzv21,nzv23,nzv32;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
   //=================================================================
   // Counter the numbers of nonzero entries in each diagonal block
   // Remark: Since Arr, Aee and Aii have the same nonzero-structure,
   //         only nza11 should be counted, nza22 = nza33 = nza11.
   //                                        peghoty 2011/06/26
   //=================================================================
    
   nza11 = 0;    
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             nza11 ++;
          }
       }     
   }
   nza22 = nza11;
   nza33 = nza11;

  /*-----------------------------------------------------
   * Create 3 CSR submatrices and 4 vectors
   *----------------------------------------------------*/
   
   A11 = jx_CSRMatrixCreate(n,n,nza11);
   jx_CSRMatrixInitialize(A11);
   ia11 = jx_CSRMatrixI(A11);
   ja11 = jx_CSRMatrixJ(A11);     
   a11  = jx_CSRMatrixData(A11);

   A22 = jx_CSRMatrixCreate(n,n,nza22);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);     
   a22  = jx_CSRMatrixData(A22);

   A33 = jx_CSRMatrixCreate(n,n,nza33);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);     
   a33  = jx_CSRMatrixData(A33);

   V12 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V12);
   v12_data = jx_VectorData(V12);  
   
   V21 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V21);
   v21_data = jx_VectorData(V21);  
   
   V23 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V23);
   v23_data = jx_VectorData(V23);  
   
   V32 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V32);
   v32_data = jx_VectorData(V32);     
     

  /*-----------------------------------------------------
   * Initialization
   *----------------------------------------------------*/
   
   nza11 = 0; nza22 = 0; nza33 = 0;
   nzv12 = 0; nzv21 = 0; nzv23 = 0;  nzv32 = 0; 
   ia11[0] = 0; ia22[0] = 0; ia33[0] = 0;

  /*-----------------------------------------------------
   * Get 3 csr submatrices and 4 vectors
   *----------------------------------------------------*/
   
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             ja11[nza11] = col;
              a11[nza11] = a[j];
             nza11 ++;
          }
          else if (col >= n && col < nplusn)
          {
             v12_data[nzv12++] = a[j];
          }
          else
          {
             jx_printf("\n\nA13 != 0\n\n");
          }
       }
       ia11[row+1] = nza11;
   }
   
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             v21_data[nzv21++] = a[j];
          }
          else if (col >= n && col < nplusn)
          {
             ja22[nza22] = col - n;
              a22[nza22] = a[j];
             nza22 ++;
          }
          else
          {
             v23_data[nzv23++] = a[j];
          }
       }
       ia22[row-n+1] = nza22;   
   }   
   
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             jx_printf("\n\nA31 != 0\n\n");
          }
          else if (col >= nplusn)
          {
             ja33[nza33] = col - nplusn;
              a33[nza33] = a[j];
             nza33 ++;
          }
          else
          {
             v32_data[nzv32++] = a[j];
          }
       }
       ia33[row-nplusn+1] = nza33;
   }  
   
   *ARR_ptr = A11; 
   *AEE_ptr = A22;
   *AII_ptr = A33;
   *VRE_ptr = V12;
   *VER_ptr = V21;
   *VEI_ptr = V23;
   *VIE_ptr = V32;
   
   return 0;  
}

JX_Int
jx_mgGetSubBlocks_REIV( jx_CSRMatrix  *A,
                        jx_CSRMatrix **ARRarray,
                        jx_CSRMatrix **AEE_ptr,
                        jx_CSRMatrix **AII_ptr,
                        jx_Vector    **VREarray,
                        jx_Vector    **VERarray,
                        jx_Vector    **VEI_ptr,
                        jx_Vector    **VIE_ptr,
                        JX_Int         ng )
{
   jx_CSRMatrix  *A22 = NULL;
   jx_CSRMatrix  *A33 = NULL;
   jx_Vector     *V23 = NULL;
   jx_Vector     *V32 = NULL;

   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);
   JX_Real *a   = jx_CSRMatrixData(A);

   JX_Int *nnzb = jx_CTAlloc(JX_Int, ng);
   JX_Int *ercnt = jx_CTAlloc(JX_Int, ng);

   /* CSR information of submatrices */
   JX_Int *ia11 = NULL;
   JX_Int *ia22 = NULL;
   JX_Int *ia33 = NULL;   
   JX_Int *ja11 = NULL;
   JX_Int *ja22 = NULL;
   JX_Int *ja33 = NULL;

   JX_Real *a11 = NULL;
   JX_Real *a22 = NULL;
   JX_Real *a33 = NULL;

   /* information of vectors */
   JX_Real *a12 = NULL;
   JX_Real *a23 = NULL;
   JX_Real *a32 = NULL;

   /* local variables */
   JX_Int row,col;
   JX_Int block_size,e_pos,i_pos;
   JX_Int i,j,srt_row,end_row,row_end;
   JX_Int nz_ee,nz_ii,nz_rr,nz_re,nz_ei,nz_ie;

   block_size = N / (ng + 2);
   e_pos = ng * block_size;
   i_pos = e_pos + block_size;

   for (i = 0; i < ng; i ++)
   {
      srt_row = i * block_size;
      end_row = srt_row + block_size;
      for (row = srt_row; row < end_row; row ++)
      {
         row_end = ia[row+1];
         for (j = ia[row]; j < row_end; j ++)
         {
            col = ja[j];
            if ((col >= srt_row) && (col < end_row))
            {
               nnzb[i] ++;
            }
         }
      }
   }

   nz_ee = 0;
   for (row = e_pos; row < i_pos; row ++)
   {
      row_end = ia[row+1];
      for (j = ia[row]; j < row_end; j ++)
      {
         col = ja[j];
         if ((col >= e_pos) && (col < i_pos))
         {
            nz_ee ++;
         }
      }
   }

   nz_ii = 0;
   for (row = i_pos; row < N; row ++)
   {
      row_end = ia[row+1];
      for (j = ia[row]; j < row_end; j ++)
      {
         if (ja[j] >= i_pos)
         {
            nz_ii ++;
         }
      }
   }

   for (i = 0; i < ng; i ++)
   {
      ARRarray[i] = jx_CSRMatrixCreate(block_size,block_size,nnzb[i]);
      jx_CSRMatrixInitialize(ARRarray[i]);
   }

   A22 = jx_CSRMatrixCreate(block_size,block_size,nz_ee);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);
   a22  = jx_CSRMatrixData(A22);

   A33 = jx_CSRMatrixCreate(block_size,block_size,nz_ii);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);
   a33  = jx_CSRMatrixData(A33);

   for (i = 0; i < ng; i ++)
   {
      VREarray[i] = jx_SeqVectorCreate(block_size);
      jx_SeqVectorInitialize(VREarray[i]);
      VERarray[i] = jx_SeqVectorCreate(block_size);
      jx_SeqVectorInitialize(VERarray[i]);
   }

   V23 = jx_SeqVectorCreate(block_size);
   jx_SeqVectorInitialize(V23);
   a23 = jx_VectorData(V23);

   V32 = jx_SeqVectorCreate(block_size);
   jx_SeqVectorInitialize(V32);
   a32 = jx_VectorData(V32);

   for (i = 0; i < ng; i ++)
   {
      ia11 = jx_CSRMatrixI(ARRarray[i]);
      ja11 = jx_CSRMatrixJ(ARRarray[i]);
      a11  = jx_CSRMatrixData(ARRarray[i]);
      a12 = jx_VectorData(VREarray[i]);
      nz_rr = 0;
      nz_re = 0;
      ia11[0] = 0;
      srt_row = i * block_size;
      end_row = srt_row + block_size;
      for (row = srt_row; row < end_row; row ++)
      {
         row_end = ia[row+1];
         for (j = ia[row]; j < row_end; j ++)
         {
            col = ja[j];
            if ((col >= srt_row) && (col < end_row))
            {
               ja11[nz_rr] = col - srt_row;
               a11[nz_rr] = a[j];
               nz_rr ++;
            }
            else if ((col >= e_pos) && (col < i_pos))
            {
               a12[nz_re++] = a[j];
            }
            else if (col >= i_pos)
            {
               jx_printf("\033[31m >>> Ar[%d]i != 0\033[00m\n", i);
               exit(0);
            }
            else
            {
               jx_printf("\033[31m >>> Ar[%d]r[%d]-[%d][%d] != 0\033[00m\n", i, col/block_size, row, col);
               exit(0);
            }
         }
         ia11[row-srt_row+1] = nz_rr;
      }
   }

   nz_ee = 0;
   nz_ei = 0;
   ia22[0] = 0;
   for (row = e_pos; row < i_pos; row ++)
   {
      row_end = ia[row+1];
      for (j = ia[row]; j < row_end; j ++)
      {
         col = ja[j];
         if (col < e_pos)
         {
            jx_VectorData(VERarray[col/block_size])[ercnt[col/block_size]++] = a[j];
         }
         else if ((col >= e_pos) && (col < i_pos))
         {
            ja22[nz_ee] = col - e_pos;
            a22[nz_ee] = a[j];
            nz_ee ++;
         }
         else
         {
            a23[nz_ei++] = a[j];
         }
      }
      ia22[row-e_pos+1] = nz_ee;
   }
   nz_ii = 0;
   nz_ie = 0;
   ia33[0] = 0;
   for (row = i_pos; row < N; row ++)
   {
      row_end = ia[row+1];
      for (j = ia[row]; j < row_end; j ++)
      {
         col = ja[j];
         if (col < e_pos)
         {
            jx_printf("\033[31m >>> Air[%d] != 0\033[00m\n", col/block_size);
            exit(0);
         }
         else if (col < i_pos)
         {
            a32[nz_ie++] = a[j];
         }
         else
         {
            ja33[nz_ii] = col - i_pos;
            a33[nz_ii] = a[j];
            nz_ii ++;
         }
      }
      ia33[row-i_pos+1] = nz_ii;
   }

   jx_TFree(nnzb);
   jx_TFree(ercnt);

   *AEE_ptr = A22;
   *AII_ptr = A33;
   *VEI_ptr = V23;
   *VIE_ptr = V32;
   
   return 0;
}

/*!
 * \fn JX_Int jx_3tGetSubDiagBlocks
 * \brief Get 3 diagonal blocks of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubDiagBlocks( jx_CSRMatrix  *A, 
                       jx_CSRMatrix **ARR_ptr, 
                       jx_CSRMatrix **AEE_ptr, 
                       jx_CSRMatrix **AII_ptr )
{
   jx_CSRMatrix *A11 = NULL; 
   jx_CSRMatrix *A22 = NULL;
   jx_CSRMatrix *A33 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia11 = NULL;
   JX_Int    *ia22 = NULL;
   JX_Int    *ia33 = NULL;
   
   JX_Int    *ja11 = NULL;
   JX_Int    *ja22 = NULL;
   JX_Int    *ja33 = NULL;
   
   JX_Real *a11 = NULL;
   JX_Real *a22 = NULL;
   JX_Real *a33 = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza11,nza22,nza33;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries in each diagonal block
   *------------------------------------------------------------------*/
   nza11 = 0;
   nza22 = 0;
   nza33 = 0;

   //=================================================================
   // Remark: Since Arr, Aee and Aii have the same nonzero-structure,
   //         only nza11 should be counted, nza22 = nza33 = nza11.
   //                                        peghoty 2011/06/26
   //=================================================================
       
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             nza11 ++;
          }
       }     
   }
   nza22 = nza11;
   nza33 = nza11;


  /*-----------------------------------------------------
   * Create 3 diagonal CSR submatrices
   *----------------------------------------------------*/
   A11 = jx_CSRMatrixCreate(n,n,nza11);
   jx_CSRMatrixInitialize(A11);
   ia11 = jx_CSRMatrixI(A11);
   ja11 = jx_CSRMatrixJ(A11);     
   a11  = jx_CSRMatrixData(A11);

   A22 = jx_CSRMatrixCreate(n,n,nza22);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);     
   a22  = jx_CSRMatrixData(A22);

   A33 = jx_CSRMatrixCreate(n,n,nza33);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);     
   a33  = jx_CSRMatrixData(A33);


  /*-----------------------------------------------------
   * Initialization
   *----------------------------------------------------*/
   nza11 = 0; nza22 = 0; nza33 = 0;
   ia11[0] = 0; ia22[0] = 0; ia33[0] = 0;

  /*-----------------------------------------------------
   * Get the 3 diagonal submatrices
   *----------------------------------------------------*/
   
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             ja11[nza11] = col;
              a11[nza11] = a[j];
             nza11 ++;
          }
       }
       ia11[row+1] = nza11;
   }
   
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= n && col < nplusn)
          {
             ja22[nza22] = col - n;
              a22[nza22] = a[j];
             nza22 ++;
          }
       }
       ia22[row-n+1] = nza22;   
   }   
   
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= nplusn)
          {
             ja33[nza33] = col - nplusn;
              a33[nza33] = a[j];
             nza33 ++;
          }
       }
       ia33[row-nplusn+1] = nza33;
   }  
   
   *ARR_ptr = A11; 
   *AEE_ptr = A22;
   *AII_ptr = A33;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_R
 * \brief Get ARR and ARE of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_R( jx_CSRMatrix  *A, 
                     jx_CSRMatrix **ARR_ptr, 
                     jx_CSRMatrix **ARE_ptr )
{
   jx_CSRMatrix *A11 = NULL; 
   jx_CSRMatrix *A12 = NULL; 

   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia11 = NULL;
   JX_Int    *ia12 = NULL;
   
   JX_Int    *ja11 = NULL;
   JX_Int    *ja12 = NULL;
   
   JX_Real *a11 = NULL;
   JX_Real *a12 = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza11;
   JX_Int nzv12;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries ARR
   *------------------------------------------------------------------*/
 
   nza11 = 0;    
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             nza11 ++;
          }
       }     
   }

  /*-----------------------------------------------------
   * Create 2 CSR submatrices
   *----------------------------------------------------*/
   A11 = jx_CSRMatrixCreate(n,n,nza11);
   jx_CSRMatrixInitialize(A11);
   ia11 = jx_CSRMatrixI(A11);
   ja11 = jx_CSRMatrixJ(A11);     
   a11  = jx_CSRMatrixData(A11);

   A12 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A12);
   ia12 = jx_CSRMatrixI(A12);
   ja12 = jx_CSRMatrixJ(A12);     
   a12  = jx_CSRMatrixData(A12);
   
   for (j = 0; j < n; j ++)
   {
      ia12[j] = j; ja12[j] = j;
   }
   ia12[n] = n;  
     
     
  /*-----------------------------------------------------
   * Get the 2 submatrices
   *----------------------------------------------------*/

   nza11 = 0;
   nzv12 = 0;
   ia11[0] = 0;
   
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             ja11[nza11] = col;
              a11[nza11] = a[j];
             nza11 ++;
          }
          else if (col >= n && col < nplusn)
          {
             a12[nzv12++] = a[j];
          }
       }
       ia11[row+1] = nza11;
   }
   
   *ARR_ptr = A11; 
   *ARE_ptr = A12;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_RV
 * \brief Get ARR and ARE of a 3tmatrix, where ARE is stored in vector.
 * \author peghoty
 * \date 2011/09/10
 */
JX_Int
jx_3tGetSubBlocks_RV( jx_CSRMatrix  *A, 
                      jx_CSRMatrix **ARR_ptr, 
                      jx_Vector    **VRE_ptr )
{
   jx_CSRMatrix *A11 = NULL; 
   jx_Vector    *V12 = NULL; 

   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);

   
   /* CSR submatrix information */
   JX_Int    *ia11 = NULL;
   JX_Int    *ja11 = NULL;
   JX_Real *a11  = NULL;
   
   /* vector information */
   JX_Real *v12_data = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza11;
   JX_Int nzv12;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries ARR
   *------------------------------------------------------------------*/
 
   nza11 = 0;    
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             nza11 ++;
          }
       }     
   }

  /*-----------------------------------------------------
   * Create a CSR submatrix and a vector
   *----------------------------------------------------*/
   
   A11 = jx_CSRMatrixCreate(n,n,nza11);
   jx_CSRMatrixInitialize(A11);
   ia11 = jx_CSRMatrixI(A11);
   ja11 = jx_CSRMatrixJ(A11);     
   a11  = jx_CSRMatrixData(A11);
   
   V12 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V12);
   v12_data = jx_VectorData(V12);
   
  
  /*-----------------------------------------------------
   * Get the 2 submatrices
   *----------------------------------------------------*/

   nza11 = 0;
   nzv12 = 0;
   ia11[0] = 0;
   
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             ja11[nza11] = col;
              a11[nza11] = a[j];
             nza11 ++;
          }
          else if (col >= n && col < nplusn)
          {
             v12_data[nzv12++] = a[j];
          }
       }
       ia11[row+1] = nza11;
   }
   
   *ARR_ptr = A11; 
   *VRE_ptr = V12;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_E
 * \brief Get AEE, AER and AEI of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_E( jx_CSRMatrix  *A, 
                     jx_CSRMatrix **AEE_ptr, 
                     jx_CSRMatrix **AER_ptr, 
                     jx_CSRMatrix **AEI_ptr )
{
   jx_CSRMatrix *A22 = NULL;
   jx_CSRMatrix *A21 = NULL; 
   jx_CSRMatrix *A23 = NULL; 
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia22 = NULL;
   JX_Int    *ia21 = NULL;
   JX_Int    *ia23 = NULL;
   
   JX_Int    *ja22 = NULL;
   JX_Int    *ja21 = NULL;
   JX_Int    *ja23 = NULL;
   
   JX_Real *a22 = NULL;
   JX_Real *a21 = NULL;
   JX_Real *a23 = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza22;
   JX_Int nzv21,nzv23;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries in AEE
   *------------------------------------------------------------------*/

   nza22 = 0;
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= n && col < nplusn)
          {
             nza22 ++;
          }
       }     
   }


  /*-----------------------------------------------------
   * Create 3 CSR submatrices
   *----------------------------------------------------*/
   
   A22 = jx_CSRMatrixCreate(n,n,nza22);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);     
   a22  = jx_CSRMatrixData(A22);
   
   A21 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A21);
   ia21 = jx_CSRMatrixI(A21);
   ja21 = jx_CSRMatrixJ(A21);     
   a21  = jx_CSRMatrixData(A21);
   
   A23 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A23);
   ia23 = jx_CSRMatrixI(A23);
   ja23 = jx_CSRMatrixJ(A23);     
   a23  = jx_CSRMatrixData(A23);
   
   for (j = 0; j < n; j ++)
   {
      ia21[j] = j; ja21[j] = j;
      ia23[j] = j; ja23[j] = j;
   }
   ia21[n] = n;
   ia23[n] = n; 

  /*-----------------------------------------------------
   * Get the 3 submatrices
   *----------------------------------------------------*/
   
   nza22 = 0;
   nzv21 = 0; nzv23 = 0; 
   ia22[0] = 0;

   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             a21[nzv21++] = a[j];
          }
          else if (col >= n && col < nplusn)
          {
             ja22[nza22] = col - n;
              a22[nza22] = a[j];
             nza22 ++;
          }
          else
          {
             a23[nzv23++] = a[j];
          }
       }
       ia22[row-n+1] = nza22;   
   }   
   

   *AEE_ptr = A22;
   *AER_ptr = A21;
   *AEI_ptr = A23;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_EV
 * \brief Get AEE, AER and AEI of a 3tmatrix, where AER and AEI
 *        are stored in vectors.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_EV( jx_CSRMatrix  *A, 
                      jx_CSRMatrix **AEE_ptr, 
                      jx_Vector    **VER_ptr, 
                      jx_Vector    **VEI_ptr )
{
   jx_CSRMatrix *A22 = NULL;
   jx_Vector    *V21 = NULL; 
   jx_Vector    *V23 = NULL; 
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR submatrix information */
   JX_Int    *ia22 = NULL;
   JX_Int    *ja22 = NULL;
   JX_Real *a22 = NULL;

   /* vectors information */
   JX_Real *v21_data = NULL;
   JX_Real *v23_data = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza22;
   JX_Int nzv21,nzv23;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries in AEE
   *------------------------------------------------------------------*/

   nza22 = 0;
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= n && col < nplusn)
          {
             nza22 ++;
          }
       }     
   }


  /*-----------------------------------------------------
   * Create a CSR submatrix and two vectors
   *----------------------------------------------------*/
   
   A22 = jx_CSRMatrixCreate(n,n,nza22);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);     
   a22  = jx_CSRMatrixData(A22);
   
   V21 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V21);
   v21_data = jx_VectorData(V21);
   
   V23 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V23);
   v23_data = jx_VectorData(V23);      

  /*-----------------------------------------------------
   * Get the csr submatrix and two vectors
   *----------------------------------------------------*/
   
   nza22 = 0;
   nzv21 = 0; nzv23 = 0; 
   ia22[0] = 0;

   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             v21_data[nzv21++] = a[j];
          }
          else if (col >= n && col < nplusn)
          {
             ja22[nza22] = col - n;
              a22[nza22] = a[j];
             nza22 ++;
          }
          else
          {
             v23_data[nzv23++] = a[j];
          }
       }
       ia22[row-n+1] = nza22;   
   }   
   
   *AEE_ptr = A22;
   *VER_ptr = V21;
   *VEI_ptr = V23;   
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_I
 * \brief Get AII, AIE of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_I( jx_CSRMatrix  *A, 
                     jx_CSRMatrix **AII_ptr, 
                     jx_CSRMatrix **AIE_ptr )
{
   jx_CSRMatrix *A33 = NULL;
   jx_CSRMatrix *A32 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia33 = NULL;
   JX_Int    *ia32 = NULL;

   JX_Int    *ja33 = NULL;
   JX_Int    *ja32 = NULL;
   
   JX_Real *a33 = NULL;
   JX_Real *a32 = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza33;
   JX_Int nzv32;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*--------------------------------------------------------------------
   * Counter the numbers of nonzero entries in AII
   *------------------------------------------------------------------*/

   nza33 = 0;    
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= nplusn)
          {
             nza33 ++;
          }
       }
   }   

  /*-----------------------------------------------------
   * Create 2 CSR submatrices
   *----------------------------------------------------*/

   A33 = jx_CSRMatrixCreate(n,n,nza33);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);     
   a33  = jx_CSRMatrixData(A33);
   
   A32 = jx_CSRMatrixCreate(n,n,n);
   jx_CSRMatrixInitialize(A32);
   ia32 = jx_CSRMatrixI(A32);
   ja32 = jx_CSRMatrixJ(A32);     
   a32  = jx_CSRMatrixData(A32);
   
   for (j = 0; j < n; j ++)
   {
      ia32[j] = j; ja32[j] = j;
   }
   ia32[n] = n;   
     

  /*-----------------------------------------------------
   * Get the 3 submatrices
   *----------------------------------------------------*/
   
   nza33 = 0;
   nzv32 = 0; 
   ia33[0] = 0; 
   
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             jx_printf("\n\nA31 != 0\n\n");
          }
          else if (col >= nplusn)
          {
             ja33[nza33] = col - nplusn;
              a33[nza33] = a[j];
             nza33 ++;
          }
          else
          {
             a32[nzv32++] = a[j];
          }
       }
       ia33[row-nplusn+1] = nza33;
   }  
   
   *AII_ptr = A33;
   *AIE_ptr = A32;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_IV
 * \brief Get AII, AIE of a 3tmatrix, where AIE 
 *        is stored in vector.
 * \author peghoty
 * \date 2011/09/10
 */
JX_Int
jx_3tGetSubBlocks_IV( jx_CSRMatrix  *A, 
                      jx_CSRMatrix **AII_ptr, 
                      jx_Vector    **VIE_ptr )
{
   jx_CSRMatrix *A33 = NULL;
   jx_Vector    *V32 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR submatrix information */
   JX_Int    *ia33 = NULL;
   JX_Int    *ja33 = NULL;
   JX_Real *a33  = NULL;
   
   /* vector information */
   JX_Real *v32_data = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza33;
   JX_Int nzv32;
   JX_Int nplusn = 2*n;
   
   jx_CSRMatrixReorder(A);
    
  /*-------------------------------------------------------
   * Counter the numbers of nonzero entries in AII
   *------------------------------------------------------*/

   nza33 = 0;    
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= nplusn)
          {
             nza33 ++;
          }
       }
   }   

  /*-----------------------------------------------------
   * Create 2 CSR submatrices
   *----------------------------------------------------*/

   A33 = jx_CSRMatrixCreate(n,n,nza33);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);     
   a33  = jx_CSRMatrixData(A33);

   V32 = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(V32);
   v32_data = jx_VectorData(V32);   


  /*-----------------------------------------------------
   * Get the a csr submatrix and a vector
   *----------------------------------------------------*/
   
   nza33 = 0;
   nzv32 = 0; 
   ia33[0] = 0; 
   
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             jx_printf("\n\nA31 != 0\n\n");
          }
          else if (col >= nplusn)
          {
             ja33[nza33] = col - nplusn;
              a33[nza33] = a[j];
             nza33 ++;
          }
          else
          {
             v32_data[nzv32++] = a[j];
          }
       }
       ia33[row-nplusn+1] = nza33;
   }  
   
   *AII_ptr = A33;
   *VIE_ptr = V32;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_ARR
 * \brief Get ARR of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_ARR( jx_CSRMatrix *A,  jx_CSRMatrix **ARR_ptr )
{
   jx_CSRMatrix *A11 = NULL; 
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia11 = NULL;
   JX_Int    *ja11 = NULL;
   JX_Real *a11  = NULL;

   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza11;
   
   /* Diagonal entries first */
   jx_CSRMatrixReorder(A);
    
   /* count how many nonzeros in ARR */ 
   nza11 = 0;    
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             nza11 ++;
          }
       }     
   }

   /* Create ARR */
   A11 = jx_CSRMatrixCreate(n,n,nza11);
   jx_CSRMatrixInitialize(A11);
   ia11 = jx_CSRMatrixI(A11);
   ja11 = jx_CSRMatrixJ(A11);     
   a11  = jx_CSRMatrixData(A11);

   /* Get the ARR from A */
   nza11 = 0;
   ia11[0] = 0;
   for (row = 0; row < n; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col < n)
          {
             ja11[nza11] = col;
              a11[nza11] = a[j];
             nza11 ++;
          }
       }
       ia11[row+1] = nza11;
   }
   
   *ARR_ptr = A11; 
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_AEE
 * \brief Get AEE of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_AEE( jx_CSRMatrix *A, jx_CSRMatrix **AEE_ptr )
{
   jx_CSRMatrix *A22 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia22 = NULL;
   JX_Int    *ja22 = NULL;
   JX_Real *a22  = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza22;
   JX_Int nplusn = 2*n;
   
   /* Diagonal entries first */
   jx_CSRMatrixReorder(A);

   /* count how many nonzeros in AEE */ 
   nza22 = 0;    
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= n && col < nplusn)
          {
             nza22 ++;
          }
       }
   }

   /* Create AEE */
   A22 = jx_CSRMatrixCreate(n,n,nza22);
   jx_CSRMatrixInitialize(A22);
   ia22 = jx_CSRMatrixI(A22);
   ja22 = jx_CSRMatrixJ(A22);     
   a22  = jx_CSRMatrixData(A22);

   /* Get AEE form A */
   nza22 = 0;
   ia22[0] = 0;
   for (row = n; row < nplusn; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= n && col < nplusn)
          {
             ja22[nza22] = col - n;
              a22[nza22] = a[j];
             nza22 ++;
          }
       }
       ia22[row-n+1] = nza22;   
   }   

   *AEE_ptr = A22;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_AII
 * \brief Get AII of a 3tmatrix.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_AII( jx_CSRMatrix *A, jx_CSRMatrix **AII_ptr )
{
   jx_CSRMatrix *A33 = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);
   JX_Int    *ja  = jx_CSRMatrixJ(A);     
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* CSR information of submatrices */
   JX_Int    *ia33 = NULL;
   JX_Int    *ja33 = NULL;
   JX_Real *a33  = NULL;
   
   /* local variables */
   JX_Int row,col;
   JX_Int j,n = N / 3;
   JX_Int nza33;
   JX_Int nplusn = 2*n;
   
   /* Diagonal entries first */
   jx_CSRMatrixReorder(A);
    
   /* count how many nonzeros in AEE */
   nza33 = 0;  
   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];

          if (col >= nplusn)
          {
             nza33 ++;
          }
       }
   }  

   /* Create AII */
   A33 = jx_CSRMatrixCreate(n,n,nza33);
   jx_CSRMatrixInitialize(A33);
   ia33 = jx_CSRMatrixI(A33);
   ja33 = jx_CSRMatrixJ(A33);     
   a33  = jx_CSRMatrixData(A33);

  /*-----------------------------------------------------
   * Get AII from A
   *----------------------------------------------------*/
   nza33 = 0;
   ia33[0] = 0;

   for (row = nplusn; row < N; row ++)
   {
       for (j = ia[row]; j < ia[row+1]; j ++)
       {
          col = ja[j];
          if (col >= nplusn)
          {
             ja33[nza33] = col - nplusn;
              a33[nza33] = a[j];
             nza33 ++;
          }
       }
       ia33[row-nplusn+1] = nza33;
   }  
   
   *AII_ptr = A33;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetSubBlocks_DII
 * \brief Get DII of a 3tmatrix, here, DII = Diag(AII)^{-1}.
 * \author peghoty
 * \date 2011/09/08
 */
JX_Int
jx_3tGetSubBlocks_DII( jx_CSRMatrix *A, jx_Vector **DII_ptr )
{
   jx_Vector *DII = NULL;
   
   /* CSR information of A */
   JX_Int     N   = jx_CSRMatrixNumRows(A);
   JX_Int    *ia  = jx_CSRMatrixI(A);    
   JX_Real *a   = jx_CSRMatrixData(A);
   
   /* value part of the vector DII */
   JX_Real *val = NULL;
   
   /* local variables */
   JX_Int row;
   JX_Int n = N / 3;
   JX_Int cnt;
   JX_Int nplusn = 2*n;
   
   /* Diagonal entries first */
   jx_CSRMatrixReorder(A);

   /* Create AII */
   DII = jx_SeqVectorCreate(n);
   jx_SeqVectorInitialize(DII); 
   val = jx_VectorData(DII);

   /* Get DII from A */
   cnt = 0;
   for (row = nplusn; row < N; row ++)
   {
      val[cnt++] = 1.0 / a[ia[row]];
   }  
   
   *DII_ptr = DII;
   
   return 0;  
}

/*!
 * \fn JX_Int jx_3tGetAD
 * \brief Get Ad = Diag(ARR,AEE,AII) or Diag(ARR,AEE,DII), where DII=Diag(AII).
 * \author peghoty
 * \date 2011/09/15
 */
jx_ParCSRMatrix *
jx_3tGetAD( jx_ParCSRMatrix *A, JX_Int AII_all )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   
   jx_ParCSRMatrix  *AD = NULL;
   jx_CSRMatrix     *As = NULL;
   
   JX_Int *row_starts = NULL;
   JX_Int *col_starts = NULL;
   
   /* local variables */
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int i, j;
   JX_Int nz, start;
   JX_Int n = N / 3;
   JX_Int nplusn = 2*n;
   JX_Int myid;

   jx_MPI_Comm_rank(comm, &myid);
   

   //==================================================================
   //  Convert parallel matrix A to a sequential one on 0# processor
   //==================================================================
      
   As = jx_ParCSRMatrixToCSRMatrixAll(A);
   jx_CSRMatrixReorder(As); 
   

   //==================================================================
   //  Build Block Diag Matrix of As (get rid of OffDiag entries, 
   //  change AII to DII) on 0# processor
   //==================================================================
     
   if (myid == 0)
   {
   
      JX_Real *aa = jx_CSRMatrixData(As);
      JX_Int    *ia = jx_CSRMatrixI(As);
      JX_Int    *ja = jx_CSRMatrixJ(As);
   
      /* get rid of the unwanted entries in the matrix */ 
      nz = 0;
      
      /* get ARR */
      for (i = 0; i < n; i ++)
      {
         start = ia[i];
         ia[i] = nz;
         for (j = start; j < ia[i+1]; j ++)
         {
            if (ja[j] < n)
            {
               ja[nz] = ja[j];
               aa[nz] = aa[j];
               nz ++;
            }
         }
      } 
      
      /* get AEE */
      for (i = n; i < nplusn; i ++)
      {
         start = ia[i];
         ia[i] = nz;
         for (j = start; j < ia[i+1]; j ++)
         {
            if (ja[j] >= n && ja[j] < nplusn)
            {
               ja[nz] = ja[j];
               aa[nz] = aa[j];
               nz ++;
            }
         }
      } 
    
      if (AII_all == 1)
      { 
         /* get AII */
         for (i = nplusn; i < N; i ++)
         {
            start = ia[i];
            ia[i] = nz;
           for (j = start; j < ia[i+1]; j ++)
            {
               if (ja[j] >= nplusn && ja[j] < N)
               {
                  ja[nz] = ja[j];
                  aa[nz] = aa[j];
                  nz ++;
               }
            }
         }   
      }  
      else
      {
         /* get DII = Diag(AII) */
         for (i = nplusn; i < N; i ++)
         {
            start  = ia[i];
            ia[i]  = nz;
            ja[nz] = ja[start];
            aa[nz] = aa[start];
            nz ++;       
         }
      }
      
      ia[N] = nz;
      jx_CSRMatrixNumNonzeros(As) = nz;
      
   }
   
   
   //=====================================================================
   //  Convert the sequential matrix AD to a parallel one
   //===================================================================== 
    
   jx_ParCSRMatrixGetRowPartitioning(A, &row_starts);
   jx_ParCSRMatrixGetColPartitioning(A, &col_starts);    
   AD = jx_CSRMatrixToParCSRMatrix(comm, As, row_starts, col_starts);
      
   return (AD);  
} 

/*!
 * \fn JX_Int jx_3tAPCTLDDCheck
 * \brief Check the diagonal-dominance of A. 
 * \author peghoty 
 * \date 2012/02/23
 */
JX_Int 
jx_3tAPCTLDDCheck( JX_Real theta_dd, JX_Real threshold_dd, jx_ParCSRMatrix *A )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
 
   jx_CSRMatrix *Diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix *Offd = jx_ParCSRMatrixOffd(A);
 
   JX_Int num_row_dd;
   JX_Int num_row_dd_sum;
   JX_Int IS_DD;
   JX_Int global_size = jx_ParCSRMatrixGlobalNumRows(A);

   num_row_dd = jx_3tDDCheckParallel(Diag, Offd, theta_dd, threshold_dd);

   jx_MPI_Allreduce(&num_row_dd, &num_row_dd_sum, 1, JX_MPI_INT, MPI_SUM, comm);

   if ( (JX_Real)num_row_dd_sum / (JX_Real)global_size >= threshold_dd )
   {
      IS_DD = 1;
   }
   else
   {
      IS_DD = 0;
   }

   return (IS_DD);
}

/*!
 * \fn JX_Int jx_3tDDCheckParallel
 * \brief Count the number of rows that s.t. the DD condition. 
 * \author peghoty 
 * \date 2012/02/23
 */
JX_Int 
jx_3tDDCheckParallel( jx_CSRMatrix  *A_diag, 
                      jx_CSRMatrix  *A_offd, 
                      JX_Real         theta_dd, 
                      JX_Real         threshold_dd )
{
   JX_Int num_row_dd;
   JX_Int n = jx_CSRMatrixNumRows(A_diag);
   
   JX_Real *aa_diag = jx_CSRMatrixData(A_diag);
   JX_Int *ia_diag = jx_CSRMatrixI(A_diag);  

   JX_Real *aa_offd = jx_CSRMatrixData(A_offd);
   JX_Int *ia_offd = jx_CSRMatrixI(A_offd); 
   
   JX_Int i, j;
   JX_Real diag, row_sum, rate;
   
   num_row_dd = 0;
   for (i = 0; i < n; i ++)
   {
      diag = fabs(aa_diag[ia_diag[i]]);
      row_sum = 0.0;
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         row_sum += aa_diag[j];
      }
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         row_sum += aa_offd[j];
      }      
      
      /* we don't have to judge diag since it is always positive!! */
      rate = fabs(row_sum) / diag;
      
      if (rate > theta_dd)
      {
         num_row_dd ++;
      }
   } 
   
   return (num_row_dd);
}

/*!
 * \fn JX_Int jx_3tAPCTLWeakCouplingE
 * \brief Check the weak-coupling between AEE and VER, VEI. 
 * \author peghoty 
 * \date 2012/02/23
 */
JX_Int 
jx_3tAPCTLWeakCouplingE( JX_Real            theta_wc_E, 
                         JX_Real            threshold_wc_E, 
                         jx_ParCSRMatrix  *AEE, 
                         jx_ParVector     *VER, 
                         jx_ParVector     *VEI )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(AEE);

   JX_Int num_row_wc_ER;
   JX_Int num_row_wc_EI; 
   JX_Int num_row_wc_ER_sum;
   JX_Int num_row_wc_EI_sum;   
   JX_Int Need_CC;
   JX_Int global_size = jx_ParCSRMatrixGlobalNumRows(AEE);
   
   //=====================================================================
   //  检查电子与光子，离子之间的弱耦合性
   //=====================================================================  

   /* check the weak coupling of the AEE and VER and VEI */
   num_row_wc_ER = jx_3tWeakCouplingParallel(AEE, VER, theta_wc_E);
   num_row_wc_EI = jx_3tWeakCouplingParallel(AEE, VEI, theta_wc_E);
   
   /* 全归约求和 */
   jx_MPI_Allreduce(&num_row_wc_ER, &num_row_wc_ER_sum, 1, JX_MPI_INT, MPI_SUM, comm);
   jx_MPI_Allreduce(&num_row_wc_EI, &num_row_wc_EI_sum, 1, JX_MPI_INT, MPI_SUM, comm);
   //jx_printf(" num_row_wc_ER = %d\n", num_row_wc_ER);
   //jx_printf(" num_row_wc_EI = %d\n", num_row_wc_EI);
 
   if ( ((JX_Real)num_row_wc_ER_sum / (JX_Real)global_size >= threshold_wc_E) &&
        ((JX_Real)num_row_wc_EI_sum / (JX_Real)global_size >= threshold_wc_E)  )
   {
      Need_CC = 0;
   }
   else
   {
      Need_CC = 1;
   } 
   //jx_printf(" Need_CC = %d\n", Need_CC);
 
   return (Need_CC); 
}

JX_Int 
jx_3tAPCTLmgWeakCouplingE( JX_Real            theta_wc_E, 
                           JX_Real            threshold_wc_E, 
                           JX_Int            ng,
                           jx_ParCSRMatrix  *AEE, 
                           jx_ParVector    **VER, 
                           jx_ParVector     *VEI )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(AEE);

   JX_Int *num_row_wc_ER = jx_CTAlloc(JX_Int, ng);
   JX_Int *num_row_wc_ER_sum = jx_CTAlloc(JX_Int, ng);

   JX_Int num_row_wc_EI;
   JX_Int num_row_wc_EI_sum;
   JX_Int Need_CC, gidx;
   JX_Int global_size = jx_ParCSRMatrixGlobalNumRows(AEE);

   //=====================================================================
   //  检查电子与光子，离子之间的弱耦合性
   //=====================================================================

   /* check the weak coupling of the AEE and VER and VEI */
   for (gidx = 0; gidx < ng; gidx ++) num_row_wc_ER[gidx] = jx_3tWeakCouplingParallel(AEE, VER[gidx], theta_wc_E);
   num_row_wc_EI = jx_3tWeakCouplingParallel(AEE, VEI, theta_wc_E);

   /* 全归约求和 */
   jx_MPI_Allreduce(num_row_wc_ER, num_row_wc_ER_sum, ng, JX_MPI_INT, MPI_SUM, comm);
   jx_MPI_Allreduce(&num_row_wc_EI, &num_row_wc_EI_sum, 1, JX_MPI_INT, MPI_SUM, comm);

   /* If AEE and AER, AEI is weakly coupled, the coarse-grid correction is unnecessary. */
   Need_CC = 0;
   if ((JX_Real)num_row_wc_EI_sum / (JX_Real)global_size < threshold_wc_E)
   {
      Need_CC = 1;
   }
   else
   {
      for (gidx = 0; gidx < ng; gidx ++)
      {
         if ((JX_Real)num_row_wc_ER_sum[gidx] / (JX_Real)global_size < threshold_wc_E)
         {
            Need_CC = 1;
            break;
         }
      }
   }

   jx_TFree(num_row_wc_ER);
   jx_TFree(num_row_wc_ER_sum);
 
   return (Need_CC);
}

/*!
 * \fn JX_Int jx_3tWeakCouplingParallel
 * \brief Count the number of rows that s.t. the weakly coupling 
 *        condition in the current processor.
 * \author peghoty
 * \date 2012/02/23 
 */
JX_Int 
jx_3tWeakCouplingParallel( jx_ParCSRMatrix  *A, 
                           jx_ParVector     *V,
                           JX_Real            theta_wc )
{
   jx_CSRMatrix *local_A = jx_ParCSRMatrixDiag(A); 
   jx_Vector    *local_V = jx_ParVectorLocalVector(V);
   JX_Int    *ia = NULL;
   JX_Real *aa = NULL;
   JX_Real *v_data = NULL;
   
   JX_Int n = jx_CSRMatrixNumRows(local_A); 
  
   JX_Int i, nwc;
   JX_Real diag;
   
   aa = jx_CSRMatrixData(local_A);
   ia = jx_CSRMatrixI(local_A);  
   v_data = jx_VectorData(local_V); 
   
   nwc = 0;
   for (i = 0; i < n; i ++)
   {
      diag = aa[ia[i]];
      /* we don't have to judge diag since it is always positive!! */
      if (fabs(v_data[i]) / fabs(diag) <= theta_wc)
      {
         nwc ++;
      }
   }
   
   return (nwc);      
}

/*!
 * \fn JX_Int jx_3tAPCTLWCDD
 * \brief Check the weak-coupling of different physical variables 
 *        and diagonal-dominance of each diagonal block. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLWCDD( jx_3tAPCTLData    *pre_3tapctl_data,
                MPI_Comm           comm,
                jx_CSRMatrix      *ARR, 
                jx_CSRMatrix      *AEE,
                jx_CSRMatrix      *AII,
                jx_Vector         *VRE,
                jx_Vector         *VER,
                jx_Vector         *VEI,
                jx_Vector         *VIE  )
{
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);
   
   /* parameters to describe the weaking coupling and diagonal dominance,
      whether the Coarse-grid Correction is necessary */
   JX_Real theta_wc_E     = jx_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JX_Real threshold_wc_E = jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JX_Real theta_dd_R     = jx_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JX_Real theta_dd_E     = jx_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JX_Real theta_dd_I     = jx_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JX_Real threshold_dd_R = jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JX_Real threshold_dd_E = jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JX_Real threshold_dd_I = jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data);    
   
   /* fixed number of iterations and tolerance for submatrices */ 
   JX_Int fixit_pctl_R, fixit_pctl_E, fixit_pctl_I;
   JX_Int fixit_brlx_R, fixit_brlx_E, fixit_brlx_I;  
   
   JX_Int IS_WC_E, IS_DD_R, IS_DD_E, IS_DD_I, Need_CC;  

   JX_Int ARR_relax_type;
   JX_Int AEE_relax_type;
   JX_Int AII_relax_type;    

   JX_Int IS_WC_ER;
   JX_Int IS_WC_EI;
   JX_Int myid;

   jx_MPI_Comm_rank(comm, &myid);
   
   if (print_level && myid == 0)
   {
      jx_printf(" === theta_wc_E     = %lf\n", theta_wc_E);
      jx_printf(" === threshold_wc_E = %lf\n", threshold_wc_E);     

      jx_printf(" === theta_dd_R     = %lf\n", theta_dd_R);
      jx_printf(" === theta_dd_E     = %lf\n", theta_dd_E);
      jx_printf(" === theta_dd_I     = %lf\n", theta_dd_I);
      jx_printf(" === threshold_dd_R = %lf\n", threshold_dd_R);
      jx_printf(" === threshold_dd_E = %lf\n", threshold_dd_E);
      jx_printf(" === threshold_dd_I = %lf\n\n", threshold_dd_I);   
   }

   //=====================================================================
   //  Step 1: 检查三个物理量之间的弱耦合性
   //=====================================================================  

   /* check the weak coupling of the AEE and VER and VEI */
   IS_WC_ER = jx_3tWeakCoupling(AEE, VER, theta_wc_E, threshold_wc_E);
   IS_WC_EI = jx_3tWeakCoupling(AEE, VEI, theta_wc_E, threshold_wc_E);
   if (IS_WC_ER == 1 && IS_WC_EI == 1)
   {
      IS_WC_E = 1;
   }
   else
   {
      IS_WC_E = 0;
   }

   if (print_level && myid == 0)
   {   
      jx_printf(" >>> IS_WC_ER = %d\n", IS_WC_ER);
      jx_printf(" >>> IS_WC_EI = %d\n", IS_WC_EI);
      jx_printf(" >>> IS_WC_E  = %d\n", IS_WC_E);
   }

   //=====================================================================
   //  Step 2: 检查三个对角块矩阵的强对角占优性
   //=====================================================================  
     
   /* check the diagonal dominance of the 3 sub-matrices */
   IS_DD_R = jx_3tDiagDominant(ARR, theta_dd_R, threshold_dd_R);
   IS_DD_E = jx_3tDiagDominant(AEE, theta_dd_E, threshold_dd_E);
   IS_DD_I = jx_3tDiagDominant(AII, theta_dd_I, threshold_dd_I);

   if (print_level && myid == 0)
   {   
      jx_printf(" >>> IS_DD_R  = %d\n", IS_DD_R);
      jx_printf(" >>> IS_DD_E  = %d\n", IS_DD_E);
      jx_printf(" >>> IS_DD_I  = %d\n\n", IS_DD_I);
   }

   //=====================================================================
   //  Step 3: 根据 Step 1 和 Step 2 的结果，确定是否需要粗网格校正
   //          以及块磨光时采用 Vcycle 还是 Jacobi.
   //=====================================================================     
   
   /* If AEE and AER, AEI is weakly coupled, 
      the coarse-grid correction is unnecessary. */
   if (IS_WC_E)
   {
      Need_CC = 0;
   }
   else
   {
      Need_CC = 1;
   }
   
   if (print_level && myid == 0)
   {   
      jx_printf(" >>> Need_CC  = %d\n\n", Need_CC);
   }   

   /* Confirm the relaxation-type and relaxation-iter of ARR. */ 
   if (IS_DD_R)
   {      
      ARR_relax_type = RELAX_WJACOBI;        
   }
   else
   {
      ARR_relax_type = RELAX_AMG;
   }

   /* Confirm the relaxation-type and relaxation-iter of AEE. */ 
   if (IS_DD_E)
   {      
      AEE_relax_type = RELAX_WJACOBI;        
   }
   else
   {
      AEE_relax_type = RELAX_AMG;
   }

   /* Confirm the relaxation-type and relaxation-iter of AII. */ 
   if (IS_DD_I)
   {      
      AII_relax_type = RELAX_WJACOBI;        
   }
   else
   {
      AII_relax_type = RELAX_AMG;
   }

   if (print_level && myid == 0)
   {   
      fixit_pctl_R   = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
      fixit_pctl_E   = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
      fixit_pctl_I   = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
      fixit_brlx_R   = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
      fixit_brlx_E   = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
      fixit_brlx_I   = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data); 
          
      if (Need_CC)
      {
         if (ARR_relax_type == RELAX_AMG) 
            jx_printf(" >>> ARR_relax_type: AMG    (%d)\n", fixit_pctl_R);
         else
            jx_printf(" >>> ARR_relax_type: Jacobi (%d)\n", fixit_pctl_R);

         if (AEE_relax_type == RELAX_AMG) 
            jx_printf(" >>> AEE_relax_type: AMG    (%d)\n", fixit_pctl_E);
         else
            jx_printf(" >>> AEE_relax_type: Jacobi (%d)\n", fixit_pctl_E);
               
         if (AII_relax_type == RELAX_AMG) 
            jx_printf(" >>> AII_relax_type: AMG    (%d)\n\n", fixit_pctl_I);
         else
            jx_printf(" >>> AII_relax_type: Jacobi (%d)\n\n", fixit_pctl_I);
      }
      else
      {
         if (ARR_relax_type == RELAX_AMG) 
            jx_printf(" >>> ARR_relax_type: AMG    (%d)\n", fixit_brlx_R);
         else
            jx_printf(" >>> ARR_relax_type: Jacobi (%d)\n", fixit_brlx_R);

         if (AEE_relax_type == RELAX_AMG) 
            jx_printf(" >>> AEE_relax_type: AMG    (%d)\n", fixit_brlx_E);
         else
            jx_printf(" >>> AEE_relax_type: Jacobi (%d)\n", fixit_brlx_E);
               
         if (AII_relax_type == RELAX_AMG) 
            jx_printf(" >>> AII_relax_type: AMG    (%d)\n\n", fixit_brlx_I);
         else
            jx_printf(" >>> AII_relax_type: Jacobi (%d)\n\n", fixit_brlx_I);
      }
   }
   
   
   //===================================================================
   //  填充结构体 pre_3tapctl_data 的部分成员
   //===================================================================

   jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
   jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
   jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
   jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);

   jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
   jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
   jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
   
   return (0);
}  

/*!
 * \fn JX_Int jx_3tWeakCoupling
 * \brief check the weak coupling of the A and V.
 * \author peghoty
 * \date 2011/09/17 
 */
JX_Int
jx_3tWeakCoupling( jx_CSRMatrix  *A, 
                   jx_Vector     *V, 
                   JX_Real         theta_wc, 
                   JX_Real         threshold_wc )
{
   JX_Int     i, n;
   JX_Int    *ia = NULL;
   JX_Real *aa = NULL;
   JX_Real *v_data = NULL;
   
   JX_Int nwc;
   JX_Int IS_WC;
   JX_Real diag;
   
   n  = jx_CSRMatrixNumRows(A);
   aa = jx_CSRMatrixData(A);
   ia = jx_CSRMatrixI(A);  
   v_data = jx_VectorData(V);
   
   nwc = 0;
   for (i = 0; i < n; i ++)
   {
      diag = aa[ia[i]];
      /* we don't have to judge diag since it is always positive!! */
      if (fabs(v_data[i]) / fabs(diag) <= theta_wc)
      {
         nwc ++;
      }
   }
   
   if ( (JX_Real)nwc / (JX_Real)n >= threshold_wc )
   {
      IS_WC = 1;
   }
   else
   {
      IS_WC = 0;
   }

   return (IS_WC); 
}

/*!
 * \fn JX_Int jx_3tDiagDominant
 * \brief Check the diagonal dominance of a CSR matrix. 
 * \author peghoty
 * \date 2011/09/12 
 */
JX_Int 
jx_3tDiagDominant( jx_CSRMatrix  *A,
                   JX_Real         theta_dd,
                   JX_Real         threshold_dd )
{
   JX_Int     i,j,n;
   JX_Int    *ia = NULL;
   JX_Real *aa = NULL;
   
   JX_Int     ndd;
   JX_Real  diag;
   JX_Real  row_sum;
   JX_Real  rate;
   
   JX_Int     IS_DD;
   
   n  = jx_CSRMatrixNumRows(A);
   aa = jx_CSRMatrixData(A);
   ia = jx_CSRMatrixI(A);
   
   ndd = 0;
   for (i = 0; i < n; i ++)
   {
      diag = fabs(aa[ia[i]]);
      row_sum = 0.0;
      for (j = ia[i]; j < ia[i+1]; j ++)
      {
         row_sum += aa[j];
      }
      
      /* we don't have to judge diag since it is always positive!! */
      rate = fabs(row_sum) / diag;
      
      if (rate > theta_dd)
      {
         ndd ++;
      }
   }
   
   if ((JX_Real)ndd / (JX_Real)n >= threshold_dd)
   {
      IS_DD = 1;
   }
   else
   {
      IS_DD = 0;
   }
   
   return (IS_DD);
}             

JX_Int
jx_BSCModifySubMat( jx_ParCSRMatrix *A, jx_ParVector *V, JX_Real *TEMP, JX_Int hsize )
{
   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(A);
   JX_Int *IA = jx_CSRMatrixI(A_diag);
   JX_Real *AA = jx_CSRMatrixData(A_diag);
   JX_Real *VV = jx_VectorData(jx_ParVectorLocalVector(V));

   JX_Int gidx;

   for (gidx = 0; gidx < hsize; gidx ++)
   {
      AA[IA[gidx]] -= (VV[gidx] * TEMP[hsize+gidx] / TEMP[gidx]);
   }

   return 0;
}

JX_Int
jx_BSCModify2SubMat( jx_ParCSRMatrix *A, jx_ParVector *V, JX_Real *sTEMP, JX_Real *vTEMP, JX_Int hsize )
{
   jx_CSRMatrix *A_diag = jx_ParCSRMatrixDiag(A);
   JX_Int *IA = jx_CSRMatrixI(A_diag);
   JX_Real *AA = jx_CSRMatrixData(A_diag);
   JX_Real *VV = jx_VectorData(jx_ParVectorLocalVector(V));

   JX_Int gidx;

   for (gidx = 0; gidx < hsize; gidx ++)
   {
      AA[IA[gidx]] -= (VV[gidx] * vTEMP[gidx] / sTEMP[gidx]);
   }

   return 0;
}
