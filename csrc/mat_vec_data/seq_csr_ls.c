//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_mv.h"

/*!
 * \fn void jx_ReadLinearSystemData
 * \brief Read the data of a linear system from given two files.
 * \date 2011/09/08
 */ 
void
jx_ReadLinearSystemData( char    *MatFile, 
                         char    *RhsFile,
                         JX_Int    **IA_ptr,
                         JX_Int    **JA_ptr,
                         JX_Real **AA_ptr,
                         JX_Real **F_ptr,
                         JX_Int     *n_ptr   )
{
   FILE   *fp = NULL;
   JX_Int     n, nz, j;

   JX_Int    *IA = NULL;
   JX_Int    *JA = NULL;
   JX_Real *AA = NULL;
   JX_Real *F  = NULL;
                            
   /* Read the matrix from file */
   
   fp = fopen(MatFile, "r");
   
   jx_fscanf(fp, "%d", &n);
   
   IA = jx_CTAlloc(JX_Int, n + 1);
   
   for (j = 0; j < n + 1; j ++)
   {
      jx_fscanf(fp, "%d", &IA[j]);
      IA[j] -= 1; // index-displacing
   }
   
   nz = IA[n];
   JA = jx_CTAlloc(JX_Int, nz);
   
   for (j = 0; j < nz; j ++)
   {
      jx_fscanf(fp, "%d", &JA[j]);
      JA[j] -= 1; // index-displacing
   }
   
   AA = jx_CTAlloc(JX_Real, nz);
   
   for (j = 0; j < nz; j ++)
   {
      jx_fscanf(fp, "%le", &AA[j]);
   }
   fclose(fp);
      
   /* Read the right hand side vector from file */
   
   fp = fopen(RhsFile, "r");
   
   jx_fscanf(fp, "%d", &n); 
   
   F = jx_CTAlloc(JX_Real, n);
   
   for (j = 0; j < n; j ++)
   {
      jx_fscanf(fp, "%le", &F[j]);
   }
   fclose(fp);
   
   *IA_ptr = IA;
   *JA_ptr = JA;
   *AA_ptr = AA;
   *F_ptr  = F;
   *n_ptr  = n;
}      

/*!
 * \fn void jx_InitializeLinearSystemStruct
 * \brief Initialize the structs of a sequential matrix and a vector for a linear system.
 * \date 2011/09/08
 */ 
void
jx_InitializeLinearSystemStruct( jx_CSRMatrix **matrix_ptr, 
                                 jx_Vector    **vectorF_ptr, 
                                 jx_Vector    **vectorX_ptr )
{
   jx_CSRMatrix *matrix  = NULL; 
   jx_Vector    *vectorF = NULL; 
   jx_Vector    *vectorX = NULL;
                                 
   matrix  = jx_CTAlloc(jx_CSRMatrix, 1);
   jx_CSRMatrixData(matrix)     = NULL;
   jx_CSRMatrixI(matrix)        = NULL;
   jx_CSRMatrixJ(matrix)        = NULL;
   jx_CSRMatrixRownnz(matrix)   = NULL;
   jx_CSRMatrixOwnsData(matrix) = 1;
   
   vectorF = jx_CTAlloc(jx_Vector, 1);
   jx_VectorData(vectorF)                  = NULL;
   jx_VectorNumVectors(vectorF)            = 1;
   jx_VectorMultiVecStorageMethod(vectorF) = 0;
   jx_VectorOwnsData(vectorF)              = 1;   

   vectorX = jx_CTAlloc(jx_Vector, 1);
   jx_VectorData(vectorX)                  = NULL;
   jx_VectorNumVectors(vectorX)            = 1;
   jx_VectorMultiVecStorageMethod(vectorX) = 0;
   jx_VectorOwnsData(vectorX)              = 1; 
   
   *matrix_ptr  = matrix;
   *vectorF_ptr = vectorF;  
   *vectorX_ptr = vectorX; 
}

/*!
 * \fn void jx_FillData2Struct
 * \brief Arrays -> structs for a linear system.
 * \note The zeros entries are all deleted during the scanning. 
 * \date 2011/09/08
 */ 
void
jx_FillData2Struct( JX_Int              n, 
                    JX_Int             *ia, 
                    JX_Int             *ja, 
                    JX_Real          *aa, 
                    JX_Real          *f, 
                    JX_Real          *x, 
                    jx_CSRMatrix    *matrix, 
                    jx_Vector       *vectorF, 
                    jx_Vector       *vectorX  )
{
   JX_Int nz;
   JX_Int i, j;
   JX_Int start;
  
   /* get rid of the zeros in the matrix */ 
   nz = 0;
   for (i = 0; i < n; i ++)
   {
      start = ia[i];
      ia[i] = nz;
      for (j = start; j < ia[i+1]; j ++)
      {
         if (aa[j])
         {
            ja[nz] = ja[j];
            aa[nz] = aa[j];
            nz ++;
         }
      }
   } 
   ia[n] = nz;

   jx_CSRMatrixNumRows(matrix)     = n;  
   jx_CSRMatrixNumCols(matrix)     = n;
   jx_CSRMatrixNumNonzeros(matrix) = nz;
   jx_CSRMatrixData(matrix)        = aa;
   jx_CSRMatrixI(matrix)           = ia;
   jx_CSRMatrixJ(matrix)           = ja;
   jx_CSRMatrixOwnsData(matrix)    = 1;
   jx_CSRMatrixRownnz(matrix)      = NULL; 
   jx_CSRMatrixNumRownnz(matrix)   = n; 
   
   jx_CSRMatrixReorder(matrix); 
   
   /* for the vectors */
   jx_VectorData(vectorF) = f;
   jx_VectorSize(vectorF) = n;
   jx_VectorNumVectors(vectorF)            = 1;
   jx_VectorMultiVecStorageMethod(vectorF) = 0;
   jx_VectorOwnsData(vectorF)              = 1; 
   
   jx_VectorData(vectorX) = x;
   jx_VectorSize(vectorX) = n;
   jx_VectorNumVectors(vectorX)            = 1;
   jx_VectorMultiVecStorageMethod(vectorX) = 0;
   jx_VectorOwnsData(vectorX)              = 1; 
} 
