//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  amg_ilu.c
 *  Date: 2014/03/24
 *
 *  Created by Yue Xiaoqiang
 *
 */

#include "jx_combined.h"

/*!
 * \fn JX_Int JX_ILUZPAMGSetup
 */
JX_Int
JX_ILUZPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_ILUZPAMGSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_ILUZPAMGSetup
 */
JX_Int
jx_ILUZPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Real drop_tol = jx_CombinedPrecondDataDropTol(pre_data);
    JX_Solver ilu_solver = jx_CombinedPrecondDataILUZData(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Int *partition = NULL;
    
    // SETUP phase for PAMG
    JX_PAMGSetMaxLevels(amg_solver, 25);
    JX_PAMGSetMaxIter(amg_solver, 1);
    JX_PAMGSetCycleType(amg_solver, 1);
    JX_PAMGSetMeasureType(amg_solver, 0);
    JX_PAMGSetCoarsenType(amg_solver, 6);
    JX_PAMGSetInterpType(amg_solver, 0);
    JX_PAMGSetSpMtRapType(amg_solver, 1);
    JX_PAMGSetStrongThreshold(amg_solver, 0.25);
    JX_PAMGSetMaxRowSum(amg_solver, 0.9);
    JX_PAMGSetPrintLevel(amg_solver, 0);
    JX_PAMGSetRelaxWt(amg_solver, 1.0);
    JX_PAMGSetOuterWt(amg_solver, 1.0);
    JX_PAMGSetCoarseThreshold(amg_solver, 100);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 1);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 2);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 3);
    JX_PAMGSetCycleRelaxType(amg_solver, 3, 1);
    JX_PAMGSetCycleRelaxType(amg_solver, 3, 2);
    JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
    
    // SETUP phase for ILU
    JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
    JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);
    JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);
    
    // Allocate memory for auxiliary vector
    jx_ParCSRMatrixGetRowPartitioning(par_matrix, &partition);
    jx_CombinedPrecondDataAuxVector(pre_data) = jx_ParVectorCreate(comm,
                               jx_ParCSRMatrixGlobalNumRows(par_matrix), partition);
    jx_ParVectorSetPartitioningOwner(jx_CombinedPrecondDataAuxVector(pre_data), 0);
    jx_ParVectorInitialize(jx_CombinedPrecondDataAuxVector(pre_data));
    
    // Allocate memory for residual vector
    jx_CombinedPrecondDataResVector(pre_data) = jx_ParVectorCreate(comm,
                               jx_ParCSRMatrixGlobalNumRows(par_matrix), partition);
    jx_ParVectorSetPartitioningOwner(jx_CombinedPrecondDataResVector(pre_data), 0);
    jx_ParVectorInitialize(jx_CombinedPrecondDataResVector(pre_data));
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ILUZPAMGILUZSetup
 */
JX_Int
JX_ILUZPAMGILUZSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_ILUZPAMGILUZSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_ILUZPAMGILUZSetup
 */
JX_Int
jx_ILUZPAMGILUZSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Real drop_tol = jx_CombinedPrecondDataDropTol(pre_data);
    JX_Solver ilu_solver = jx_CombinedPrecondDataILUZData(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Int *partition = NULL;
    
    // SETUP phase for PAMG
    JX_PAMGSetMaxLevels(amg_solver, 25);
    JX_PAMGSetMaxIter(amg_solver, 1);
    JX_PAMGSetCycleType(amg_solver, 1);
    JX_PAMGSetMeasureType(amg_solver, 0);
    JX_PAMGSetCoarsenType(amg_solver, jx_CombinedPrecondDataAMGCoarsenType(pre_data));
    JX_PAMGSetInterpType(amg_solver, jx_CombinedPrecondDataAMGInterpType(pre_data));
    JX_PAMGSetSpMtRapType(amg_solver, 1);
    JX_PAMGSetStrongThreshold(amg_solver, jx_CombinedPrecondDataAMGStrongThreshold(pre_data));
    JX_PAMGSetMaxRowSum(amg_solver, 0.9);
    JX_PAMGSetPrintLevel(amg_solver, 3);
    JX_PAMGSetRelaxWt(amg_solver, 1.0);
    JX_PAMGSetOuterWt(amg_solver, 1.0);
    JX_PAMGSetCoarseThreshold(amg_solver, 100);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 1);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 2);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 3);
    JX_PAMGSetCycleRelaxType(amg_solver, jx_CombinedPrecondDataAMGRelaxType(pre_data), 1);
    JX_PAMGSetCycleRelaxType(amg_solver, jx_CombinedPrecondDataAMGRelaxType(pre_data), 2);
    JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
    
    // SETUP phase for ILU
    JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
    JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);
    JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);
    
    // Allocate memory for auxiliary vector
    jx_ParCSRMatrixGetRowPartitioning(par_matrix, &partition);
    jx_CombinedPrecondDataAuxVector(pre_data) = jx_ParVectorCreate(comm,
                               jx_ParCSRMatrixGlobalNumRows(par_matrix), partition);
    jx_ParVectorSetPartitioningOwner(jx_CombinedPrecondDataAuxVector(pre_data), 0);
    jx_ParVectorInitialize(jx_CombinedPrecondDataAuxVector(pre_data));
    
    // Allocate memory for residual vector
    jx_CombinedPrecondDataResVector(pre_data) = jx_ParVectorCreate(comm,
                               jx_ParCSRMatrixGlobalNumRows(par_matrix), partition);
    jx_ParVectorSetPartitioningOwner(jx_CombinedPrecondDataResVector(pre_data), 0);
    jx_ParVectorInitialize(jx_CombinedPrecondDataResVector(pre_data));
    
    return jx_error_flag;
}

JX_Int
JX_PAMGILUZPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_PAMGILUZPAMGSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

JX_Int
jx_PAMGILUZPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Real drop_tol = jx_CombinedPrecondDataDropTol(pre_data);
    JX_Solver ilu_solver = jx_CombinedPrecondDataILUZData(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Int *partition = NULL;
    
    // SETUP phase for PAMG
    JX_PAMGSetMaxLevels(amg_solver, 25);
    JX_PAMGSetMaxIter(amg_solver, 1);
    JX_PAMGSetCycleType(amg_solver, 1);
    JX_PAMGSetMeasureType(amg_solver, 0);
    JX_PAMGSetCoarsenType(amg_solver, jx_CombinedPrecondDataAMGCoarsenType(pre_data));
    JX_PAMGSetInterpType(amg_solver, jx_CombinedPrecondDataAMGInterpType(pre_data));
    JX_PAMGSetSpMtRapType(amg_solver, 1);
    JX_PAMGSetStrongThreshold(amg_solver, jx_CombinedPrecondDataAMGStrongThreshold(pre_data));
    JX_PAMGSetMaxRowSum(amg_solver, 0.9);
    JX_PAMGSetPrintLevel(amg_solver, 3);
    JX_PAMGSetRelaxWt(amg_solver, 1.0);
    JX_PAMGSetOuterWt(amg_solver, 1.0);
    JX_PAMGSetCoarseThreshold(amg_solver, 100);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 1);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 2);
    JX_PAMGSetCycleNumSweeps(amg_solver, 1, 3);
    JX_PAMGSetCycleRelaxType(amg_solver, jx_CombinedPrecondDataAMGRelaxType(pre_data), 1);
    JX_PAMGSetCycleRelaxType(amg_solver, jx_CombinedPrecondDataAMGRelaxType(pre_data), 2);
    JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
    
    // SETUP phase for ILU
    JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
    JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);
    JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);
    
    // Allocate memory for auxiliary vector
    jx_ParCSRMatrixGetRowPartitioning(par_matrix, &partition);
    jx_CombinedPrecondDataAuxVector(pre_data) = jx_ParVectorCreate(comm,
                               jx_ParCSRMatrixGlobalNumRows(par_matrix), partition);
    jx_ParVectorSetPartitioningOwner(jx_CombinedPrecondDataAuxVector(pre_data), 0);
    jx_ParVectorInitialize(jx_CombinedPrecondDataAuxVector(pre_data));
    
    // Allocate memory for residual vector
    jx_CombinedPrecondDataResVector(pre_data) = jx_ParVectorCreate(comm,
                               jx_ParCSRMatrixGlobalNumRows(par_matrix), partition);
    jx_ParVectorSetPartitioningOwner(jx_CombinedPrecondDataResVector(pre_data), 0);
    jx_ParVectorInitialize(jx_CombinedPrecondDataResVector(pre_data));
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ILUZPAMGSolve
 */
JX_Int
JX_ILUZPAMGSolve( JX_Solver       solver,
                  JX_ParCSRMatrix par_matrix,
                  JX_ParVector    par_b,
                  JX_ParVector    par_x )
{
    return( jx_ILUZPAMGSolve( (void *) solver,
                              (jx_ParCSRMatrix *) par_matrix,
                              (jx_ParVector *)par_b,
                              (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_ILUZPAMGSolve
 */
JX_Int
jx_ILUZPAMGSolve( void            *data,
                  jx_ParCSRMatrix *par_matrix,
                  jx_ParVector    *par_b,
                  jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver ilu_solver = jx_CombinedPrecondDataILUZData(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One ILU(0) for A w = g, where A = par_matrix, g = par_b, w = par_x
    //jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_ILUZeroFactorDataPrecond(ilu_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    // 2. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 3. One V-cycle for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_PAMGPrecond(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
    // 4. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ILUZPAMGILUZSolve
 */
JX_Int
JX_ILUZPAMGILUZSolve( JX_Solver       solver,
                      JX_ParCSRMatrix par_matrix,
                      JX_ParVector    par_b,
                      JX_ParVector    par_x )
{
    return( jx_ILUZPAMGILUZSolve( (void *) solver,
                                  (jx_ParCSRMatrix *) par_matrix,
                                  (jx_ParVector *)par_b,
                                  (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_ILUZPAMGILUZSolve
 */
JX_Int
jx_ILUZPAMGILUZSolve( void            *data,
                      jx_ParCSRMatrix *par_matrix,
                      jx_ParVector    *par_b,
                      jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver ilu_solver = jx_CombinedPrecondDataILUZData(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One ILU(0) for A w = g, where A = par_matrix, g = par_b, w = par_x
    //jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_ILUZeroFactorDataPrecond(ilu_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    // 2. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 3. One V-cycle for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_PAMGPrecond(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
    // 4. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);
    // 5. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 6. One ILU(0) for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    //jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_ILUZeroFactorDataPrecond(ilu_solver, (JX_ParCSRMatrix)par_matrix,(JX_ParVector)res_vec,(JX_ParVector)aux_vec);
    // 7. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);
    
    return jx_error_flag;
}

JX_Int
JX_PAMGILUZPAMGSolve( JX_Solver       solver,
                      JX_ParCSRMatrix par_matrix,
                      JX_ParVector    par_b,
                      JX_ParVector    par_x )
{
    return( jx_PAMGILUZPAMGSolve( (void *) solver,
                                  (jx_ParCSRMatrix *) par_matrix,
                                  (jx_ParVector *)par_b,
                                  (jx_ParVector *)par_x ) );
}

JX_Int
jx_PAMGILUZPAMGSolve( void            *data,
                      jx_ParCSRMatrix *par_matrix,
                      jx_ParVector    *par_b,
                      jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver ilu_solver = jx_CombinedPrecondDataILUZData(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One V-cycle for A w = g, where A = par_matrix, g = par_b, w = par_x
    jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_PAMGPrecond(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    // 2. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 3. One ILU(0) for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    //jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_ILUZeroFactorDataPrecond(ilu_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
    // 4. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);
    // 5. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 6. One V-cycle for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_PAMGPrecond(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
    // 7. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);

    return jx_error_flag;
}
