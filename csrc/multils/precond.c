/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 * precond.c
 *
 * Created by peghoty 2010/12/19
 * Xiangtan University
 * peghoty@163.com
 *  
 */
 
#include "jx_multils.h"

/*!
 * \fn void *fsls_PreDiagDataInitialize
 * \brief Create a fsls_PreDiagData object.
 * \param m1 size of the first diagonal block
 * \param m2 size of the second diagonal block
 * \param level_fill_in level of fill in for ILU(k) 
 * \author peghoty 
 * \date 2010/12/19
 */
void *
fsls_PreDiagDataInitialize( JX_Int m1, JX_Int m2, JX_Int level_fill_in )
{
   fsls_PreDiagData *pre_diag_data = fsls_CTAlloc(fsls_PreDiagData, 1);
   
   fsls_PreDiagDataM1(pre_diag_data)          = m1;
   fsls_PreDiagDataM2(pre_diag_data)          = m2;
   fsls_PreDiagDataLevelFillIn(pre_diag_data) = level_fill_in;
   fsls_PreDiagDataPreMatrix(pre_diag_data)   = NULL;
   fsls_PreDiagDataA1(pre_diag_data)          = NULL;
   fsls_PreDiagDataA2(pre_diag_data)          = NULL;
   fsls_PreDiagDataAMGSolver1(pre_diag_data)  = NULL;
   fsls_PreDiagDataAMGSolver2(pre_diag_data)  = NULL;
   fsls_PreDiagDataILUData1(pre_diag_data)    = NULL;
   fsls_PreDiagDataILUData2(pre_diag_data)    = NULL;
   fsls_PreDiagDataZ1(pre_diag_data)          = NULL;
   fsls_PreDiagDataZ2(pre_diag_data)          = NULL;
   fsls_PreDiagDataR1(pre_diag_data)          = NULL;
   fsls_PreDiagDataR2(pre_diag_data)          = NULL;
   
   return pre_diag_data;
}

/*!
 * \fn JX_Int fsls_PreDiagSetup
 * \brief SETUP phase of block diagonal preconditioner.
 * \note This function deponds on how we treat each block, AMG or ILU, or something else. 
 * \author peghoty 
 * \date 2010/12/19
 */
JX_Int 
fsls_PreDiagSetup( void *pre_diag_vdata, fsls_CSRMatrix *A, fsls_Vector *f, fsls_Vector *u )
{
   fsls_PreDiagData *pre_diag_data = pre_diag_vdata; 
   
   JX_Int m1            = fsls_PreDiagDataM1(pre_diag_data);          // size of the first diagonal block
   JX_Int m2            = fsls_PreDiagDataM2(pre_diag_data);          // size of the second diagonal block
   JX_Int level_fill_in = fsls_PreDiagDataLevelFillIn(pre_diag_data); // level of fill-in for ILU(k)
  
   fsls_CSRMatrix    *A1          = NULL;  // the first diagonal block
   fsls_CSRMatrix    *A2          = NULL;  // the second diagonal block 
   fsls_AMGData      *amg_solver1 = NULL;  // amg_solver for the first diagonal block
   //fsls_AMGData      *amg_solver2 = NULL;  // amg_solver for the second diagonal block
   //fsls_PreILUpData  *ilu_data1   = NULL;  // ilu_data for the first diagonal block
   fsls_PreILUpData  *ilu_data2   = NULL;  // ilu_data for the second diagonal block   
   fsls_Vector       *z1          = NULL;
   fsls_Vector       *z2          = NULL;
   fsls_Vector       *r1          = NULL;
   fsls_Vector       *r2          = NULL;
    
   JX_Int dof = fsls_CSRMatrixNumRows(A);
   
   /* auxiliary arrays */
   JX_Int *work    = fsls_CTAlloc(JX_Int, 2*dof);
   JX_Int *map_row = work;
   JX_Int *map_col = work + dof;
   
   /* local variables or arrays */  
   JX_Int Setup_err_flag = 0;   
   JX_Int i;  
   
  /*-------------------------------------------------------
   * Define which entries in A constructs the submatrix
   *------------------------------------------------------*/
   for (i = 0; i < dof; i ++)
   {
      map_row[i] = i; 
      map_col[i] = i;
   }
   
  /*--------------------------------------------------------------
   * Abstract the submatrix
   *-------------------------------------------------------------*/   
   A1 = fsls_GetSubCSRMatrix(A, map_row, map_col, m1, m1);
   A2 = fsls_GetSubCSRMatrix(A, map_row+m1, map_col+m1, m2, m2);
   fsls_TFree(work);

  /*-------------------------------------------------------
   * AMG Setup for the diagonal block
   *------------------------------------------------------*/   
   amg_solver1 = fsls_AMGInitialize();
   fsls_AMGSetMaxIter(amg_solver1, 1);
   fsls_AMGSetPrintLevel(amg_solver1, 0);
   fsls_AMGSetup(amg_solver1, A1, NULL, NULL);

   //amg_solver2 = fsls_AMGInitialize();
   //fsls_AMGSetMaxIter(amg_solver2, 1);
   //fsls_AMGSetPrintLevel(amg_solver2, 0);
   //fsls_AMGSetup(amg_solver2, A2, NULL, NULL);
   
  /*-------------------------------------------------------
   * ILU(k) Setup for the diagonal block
   *------------------------------------------------------*/   
   //ilu_data1 = fsls_PreILUpDataInitialize(level_fill_in);
   //fsls_PreILUpSetup(ilu_data1, A1, NULL, NULL);

   ilu_data2 = fsls_PreILUpDataInitialize(level_fill_in);
   fsls_PreILUpSetup(ilu_data2, A2, NULL, NULL);

  /*-------------------------------------------------------
   * auxiliary vectors (only pointer, no data part)
   *------------------------------------------------------*/ 
   z1 = fsls_SeqVectorCreate(m1);
   z2 = fsls_SeqVectorCreate(m2);
   r1 = fsls_SeqVectorCreate(m1);
   r2 = fsls_SeqVectorCreate(m2);
   
  /*-------------------------------------------------------
   * Fill in members of 'pre_diag_data'
   *------------------------------------------------------*/  
   fsls_PreDiagDataPreMatrix(pre_diag_data)  = A;
   fsls_PreDiagDataA1(pre_diag_data)         = A1;
   fsls_PreDiagDataA2(pre_diag_data)         = A2;
   fsls_PreDiagDataAMGSolver1(pre_diag_data) = amg_solver1;
   //fsls_PreDiagDataAMGSolver2(pre_diag_data) = amg_solver2;
   //fsls_PreDiagDataILUData1(pre_diag_data)   = ilu_data1;
   fsls_PreDiagDataILUData2(pre_diag_data)   = ilu_data2;
   fsls_PreDiagDataZ1(pre_diag_data)         = z1;
   fsls_PreDiagDataZ2(pre_diag_data)         = z2;
   fsls_PreDiagDataR1(pre_diag_data)         = r1;
   fsls_PreDiagDataR2(pre_diag_data)         = r2;

   return (Setup_err_flag); 
}

/*!
 * \fn JX_Int fsls_PreDiagSolve
 * \brief Solve phase for Block Diagonal Preconditioner (z = Br)
 * \author peghoty
 * \date 2010/12/19 
 */
JX_Int
fsls_PreDiagSolve( void            *pre_diag_vdata,
                   fsls_Vector     *r,
                   fsls_Vector     *z )
{
   fsls_PreDiagData *pre_diag_data = pre_diag_vdata;
    
   /* members of 'pre_diag_data' */
   JX_Int                m1;          // size of the first diagonal block
   //JX_Int                m2;          // size of the second diagonal block
   fsls_AMGData      *amg_solver1; // amg_solver for the first diagonal block
   //fsls_AMGData      *amg_solver2; // amg_solver for the second diagonal block
   //fsls_PreILUpData  *ilu_data1;   // ilu_data for the first diagonal block
   fsls_PreILUpData  *ilu_data2;   // ilu_data for the second diagonal block   
   fsls_Vector       *z1;
   fsls_Vector       *z2;
   fsls_Vector       *r1;
   fsls_Vector       *r2;
   
   //JX_Int solve_type;
   
   m1 = fsls_PreDiagDataM1(pre_diag_data);
   //m2 = fsls_PreDiagDataM2(pre_diag_data);
   amg_solver1 = fsls_PreDiagDataAMGSolver1(pre_diag_data);
   //amg_solver2 = fsls_PreDiagDataAMGSolver2(pre_diag_data);
   //ilu_data1   = fsls_PreDiagDataILUData1(pre_diag_data);
   ilu_data2   = fsls_PreDiagDataILUData2(pre_diag_data);
   z1 = fsls_PreDiagDataZ1(pre_diag_data);
   z2 = fsls_PreDiagDataZ2(pre_diag_data);
   r1 = fsls_PreDiagDataR1(pre_diag_data);
   r2 = fsls_PreDiagDataR2(pre_diag_data);
   
  /*-------------------------------------------------------
   * prepare for the sub vectors
   *------------------------------------------------------*/ 
   fsls_VectorData(z1) = fsls_VectorData(z);
   fsls_VectorData(z2) = fsls_VectorData(z) + m1;
   fsls_VectorData(r1) = fsls_VectorData(r);
   fsls_VectorData(r2) = fsls_VectorData(r) + m1;       
   
  /*-------------------------------------------------------
   * prepare for the sub vectors
   *------------------------------------------------------*/ 
   //solve_type = 3;
   //switch (solve_type)
   //{
      //case 1:
      //{
      //   fsls_SeqVectorSetConstantValues(z1, 0.0);
      //   fsls_AMGPrecond(amg_solver1, r1, z1);
      //   fsls_SeqVectorSetConstantValues(z2, 0.0);
      //   fsls_AMGPrecond(amg_solver2, r2, z2);
      //}
      //break;
      //case 2:
      //{
      //   fsls_PreILUpSolve(ilu_data1, r1, z1);
      //   fsls_PreILUpSolve(ilu_data2, r2, z2);
      //}
      //break;
      //case 3:
      {
         fsls_SeqVectorSetConstantValues(z1, 0.0);
         fsls_AMGPrecond(amg_solver1, r1, z1);
         fsls_PreILUpSolve(ilu_data2, r2, z2);
      }
      //break;
      //case 4:
      //{
      //   fsls_PreILUpSolve(ilu_data1, r1, z1);
      //   fsls_SeqVectorSetConstantValues(z2, 0.0);
      //   fsls_AMGPrecond(amg_solver2,r2, z2);
      //}
      //break;            
   //}
    
   return 0;
}

/*!
 * \fn void fsls_PreDiagDataFinalize
 * \brief Finalize a fsls_PreDiagData object.
 * \author peghoty 
 * \date 2010/12/19
 */
void
fsls_PreDiagDataFinalize( void *pre_diag_vdata )
{
   fsls_PreDiagData *pre_diag_data = pre_diag_vdata;

   if (pre_diag_data)
   {
      if ( fsls_PreDiagDataA1(pre_diag_data) )
      {
         fsls_CSRMatrixDestroy(fsls_PreDiagDataA1(pre_diag_data));
      }
      if ( fsls_PreDiagDataA2(pre_diag_data) )
      {
         fsls_CSRMatrixDestroy(fsls_PreDiagDataA2(pre_diag_data));
      }
      
      if ( fsls_PreDiagDataAMGSolver1(pre_diag_data) )
      {
         fsls_AMGFinalize(fsls_PreDiagDataAMGSolver1(pre_diag_data));
      }
      if ( fsls_PreDiagDataAMGSolver2(pre_diag_data) )
      {
         fsls_AMGFinalize(fsls_PreDiagDataAMGSolver2(pre_diag_data));
      }
      
      if ( fsls_PreDiagDataILUData1(pre_diag_data) )
      {
         fsls_PreILUpDataFinalize(fsls_PreDiagDataILUData1(pre_diag_data));
      }
      if ( fsls_PreDiagDataILUData2(pre_diag_data) )
      {
         fsls_PreILUpDataFinalize(fsls_PreDiagDataILUData2(pre_diag_data));
      } 
      
      if ( fsls_PreDiagDataZ1(pre_diag_data) )
      {
         fsls_TFree(fsls_PreDiagDataZ1(pre_diag_data));
      }     
      if ( fsls_PreDiagDataZ2(pre_diag_data) )
      {
         fsls_TFree(fsls_PreDiagDataZ2(pre_diag_data));
      }
      if ( fsls_PreDiagDataR1(pre_diag_data) )
      {
         fsls_TFree(fsls_PreDiagDataR1(pre_diag_data));
      }  
      if ( fsls_PreDiagDataR2(pre_diag_data) )
      {
         fsls_TFree(fsls_PreDiagDataR2(pre_diag_data));
      }  
      
      fsls_TFree(pre_diag_data);
   }
}

/*!
 * \fn fsls_PreDSData *fsls_PreDSDataInitialize
 * \brief Initialize the DS for preconditioning
 * \author peghoty
 * \date 2010/12/23 
 */
void *
fsls_PreDSDataInitialize()
{
   fsls_PreDSData *pre_ds_data = fsls_CTAlloc(fsls_PreDSData, 1);
   fsls_PreDSDataPreMatrix(pre_ds_data) = NULL;
   return pre_ds_data;
}

/*!
 * \fn JX_Int fsls_PreDSSetup
 * \brief Diagonal Scaling Setup for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/10/26
 */
JX_Int 
fsls_PreDSSetup( void *vdata, void *A, void *b, void *x )
{
   fsls_PreDSData *pre_ds_data = (fsls_PreDSData *)vdata;
   fsls_CSRMatrix *prematrix = (fsls_CSRMatrix *)A;
   fsls_PreDSDataPreMatrix(pre_ds_data) = prematrix;
   return 0;
}

/*!
 * \fn JX_Int fsls_PreDSSolve
 * \brief Diagonal Scaling Solve for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/10/26
 */
JX_Int 
fsls_PreDSSolve( void *vdata, void *b, void *x )
{
   fsls_PreDSData *pre_ds_data = (fsls_PreDSData *)vdata;
   fsls_CSRMatrix *A1 = (fsls_CSRMatrix *)fsls_PreDSDataPreMatrix(pre_ds_data);
   fsls_Vector    *b1 = (fsls_Vector *)b;
   fsls_Vector    *x1 = (fsls_Vector *)x;
   
   JX_Real *b_data = fsls_VectorData(b1);
   JX_Real *x_data = fsls_VectorData(x1);
   JX_Real *A_data = fsls_CSRMatrixData(A1);
   JX_Int    *A_i    = fsls_CSRMatrixI(A1);
   JX_Int     n      = fsls_CSRMatrixNumRows(A1);
   JX_Int     i,ierr = 0;

   for (i = 0; i < n; i ++)
   {
      x_data[i] = b_data[i] / A_data[A_i[i]];
   } 
 
   return ierr;
}

/*!
 * \fn fsls_PreILUpData *fsls_PreILUpDataInitialize
 * \brief Initialize the ILUp for preconditioning
 * \author peghoty
 * \date 2010/12/07 
 */
void *
fsls_PreILUpDataInitialize( JX_Int level_fill_in )
{
   fsls_PreILUpData *pre_ilup_data = fsls_CTAlloc(fsls_PreILUpData, 1);

   fsls_PreILUpDataLevelFillIn(pre_ilup_data) = level_fill_in;
   fsls_PreILUpDataIndex(pre_ilup_data)       = NULL;
   fsls_PreILUpDataValue(pre_ilup_data)       = NULL;
   fsls_PreILUpDataWork(pre_ilup_data)        = NULL;
   fsls_PreILUpDataPreMatrix(pre_ilup_data)   = NULL;
   
   return pre_ilup_data;
}

/*!
 * \fn void fsls_PreILUpDataFinalize
 * \brief Finalize the ILUp preconditioner
 * \author peghoty
 * \date 2010/12/07 
 */
void 
fsls_PreILUpDataFinalize( void *pre_ilup_vdata )
{
   fsls_PreILUpData *pre_ilup_data = pre_ilup_vdata;
   
   if (pre_ilup_data)
   {
      if ( fsls_PreILUpDataIndex(pre_ilup_data) )
      { 
         fsls_TFree(fsls_PreILUpDataIndex(pre_ilup_data)); 
      }
      if ( fsls_PreILUpDataValue(pre_ilup_data) )
      {
         fsls_TFree(fsls_PreILUpDataValue(pre_ilup_data));
      }
      if ( fsls_PreILUpDataWork(pre_ilup_data) )
      {
         fsls_TFree(fsls_PreILUpDataWork(pre_ilup_data));
      }
      
      fsls_TFree(pre_ilup_data);
   }
}

/*!
 * \fn JX_Int fsls_PreILUpSetup
 * \brief ILUp Setup phase
 * \author peghoty
 * \date 2010/12/07 
 */
JX_Int
fsls_PreILUpSetup( void            *pre_ilup_vdata,
                   fsls_CSRMatrix  *A,
                   fsls_Vector     *b,
                   fsls_Vector     *x )
{
   fsls_PreILUpData  *pre_ilup_data = pre_ilup_vdata;
   JX_Int level_fill_in = fsls_PreILUpDataLevelFillIn(pre_ilup_data);

   JX_Int    *index = NULL;
   JX_Real *value = NULL;
   JX_Real *work  = NULL;
   
   JX_Int ierr = 0;
   
   ierr = fsls_ILUp_Decomp(A, level_fill_in, &index, &value);
   work = fsls_CTAlloc(JX_Real, fsls_CSRMatrixNumRows(A));
   
   //------------------------------------------------
   //  Fill in the members of pre_ilup_data
   //------------------------------------------------ 
   fsls_PreILUpDataIndex(pre_ilup_data)     = index;
   fsls_PreILUpDataValue(pre_ilup_data)     = value;
   fsls_PreILUpDataWork(pre_ilup_data)      = work;
   fsls_PreILUpDataPreMatrix(pre_ilup_data) = A;

   return ierr;
}

/*!
 * \fn JX_Int fsls_PreILUpSolve
 * \brief ILUp Solve phase (z = Br)
 * \author peghoty
 * \date 2010/12/07 
 */
JX_Int
fsls_PreILUpSolve( void            *pre_ilup_vdata,
                   fsls_Vector     *r,
                   fsls_Vector     *z )
{
   fsls_PreILUpData *pre_ilup_data = pre_ilup_vdata;
   JX_Int    *index  = fsls_PreILUpDataIndex(pre_ilup_data);
   JX_Real *value  = fsls_PreILUpDataValue(pre_ilup_data);
   JX_Real *e_data = fsls_PreILUpDataWork(pre_ilup_data);
   fsls_CSRMatrix *A = fsls_PreILUpDataPreMatrix(pre_ilup_data);

   JX_Real *r_data = fsls_VectorData(r);
   JX_Real *z_data = fsls_VectorData(z);
   
   JX_Int i,j,col;
   JX_Int n = fsls_CSRMatrixNumRows(A);
   JX_Real tmp = 0.0;
      
   //-----------------------------------------------------------
   // forward sweep: solve unit lower matrix equation L*e = r 
   //-----------------------------------------------------------
   
   e_data[0] = r_data[0];
   for (i = 1; i < n; i ++)
   {
      tmp = r_data[i];
      for (j = index[i]; j < index[i+1]; j ++)
      {
         col = index[j];
         if (col < i)
            tmp -= value[j]*e_data[col];
         else
            break;
      }
      e_data[i] = tmp;
   }

   //-----------------------------------------------------------
   // backward sweep: solve upper matrix equation U*z = e 
   //-----------------------------------------------------------

   z_data[n-1] = e_data[n-1]*value[n-1];
   for (i = n-2; i >= 0; i --)
   {
      tmp = e_data[i];
      for (j = index[i+1]-1; j >= index[i]; j --)
      {
         col = index[j];
         if (col > i)
            tmp -= value[j]*z_data[col];
         else
            break;
      }
      z_data[i] = tmp*value[i];
   }
   
   return 0;
}
