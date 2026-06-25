//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  amg_euclid.c
 *  Date: 2013/12/11
 *
 *  Created by Yue Xiaoqiang
 *
 */

#include "jx_combined.h"

/*!
 * \fn JX_Int JX_PAMGEuclidPAMGSetup
 */
JX_Int
JX_PAMGEuclidPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_PAMGEuclidPAMGSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_PAMGEuclidPAMGSetup
 */
JX_Int
jx_PAMGEuclidPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
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
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix) par_matrix);
    
    // SETUP phase for Euclid
    JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
    
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
 * \fn JX_Int JX_PAMGEuclidPAMGSetupB
 */
JX_Int
JX_PAMGEuclidPAMGSetupB( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_PAMGEuclidPAMGSetupB( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_PAMGEuclidPAMGSetupB
 */
JX_Int
jx_PAMGEuclidPAMGSetupB( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
    jx_ParCSRMatrix *pre_euc_mat = jx_CombinedPrecondDataPreEucMat(pre_data);
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
    
    // SETUP phase for Euclid
    JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)pre_euc_mat);
    
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
 * \fn JX_Int JX_EuclidPAMGSetup
 */
JX_Int
JX_EuclidPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_EuclidPAMGSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_EuclidPAMGSetup
 */
JX_Int
jx_EuclidPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
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
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix) par_matrix);
    
    // SETUP phase for Euclid
    JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
    
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
 * \fn JX_Int JX_PAMGEuclidSetup
 */
JX_Int
JX_PAMGEuclidSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_PAMGEuclidSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_PAMGEuclidSetup
 */
JX_Int
jx_PAMGEuclidSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
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
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix) par_matrix);
    
    // SETUP phase for Euclid
    JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
    
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
 * \fn JX_Int JX_EuclidPAMGEuclidSetup
 */
JX_Int
JX_EuclidPAMGEuclidSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_EuclidPAMGEuclidSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_EuclidPAMGEuclidSetup
 */
JX_Int
jx_EuclidPAMGEuclidSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
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
    JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix) par_matrix);
    
    // SETUP phase for Euclid
    JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
    
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
 * \fn JX_Int JX_PAMGEuclidPAMGSolve
 */
JX_Int
JX_PAMGEuclidPAMGSolve( JX_Solver       solver,
                        JX_ParCSRMatrix par_matrix,
                        JX_ParVector    par_b,
                        JX_ParVector    par_x )
{
    return( jx_PAMGEuclidPAMGSolve( (void *) solver,
                                    (jx_ParCSRMatrix *) par_matrix,
                                    (jx_ParVector *)par_b,
                                    (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_PAMGEuclidPAMGSolve
 */
JX_Int
jx_PAMGEuclidPAMGSolve( void            *data,
                        jx_ParCSRMatrix *par_matrix,
                        jx_ParVector    *par_b,
                        jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One V-cycle for A w = g, where A = par_matrix, g = par_b, w = par_x
    jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_PAMGPrecond(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    // 2. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 3. One ILU(k) for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_EuclidSolve(euclid_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
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

/*!
 * \fn JX_Int JX_EuclidPAMGSolve
 */
JX_Int
JX_EuclidPAMGSolve( JX_Solver       solver,
                    JX_ParCSRMatrix par_matrix,
                    JX_ParVector    par_b,
                    JX_ParVector    par_x )
{
    return( jx_EuclidPAMGSolve( (void *) solver,
                                (jx_ParCSRMatrix *) par_matrix,
                                (jx_ParVector *)par_b,
                                (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_EuclidPAMGSolve
 */
JX_Int
jx_EuclidPAMGSolve( void            *data,
                    jx_ParCSRMatrix *par_matrix,
                    jx_ParVector    *par_b,
                    jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One ILU(k) for A w = g, where A = par_matrix, g = par_b, w = par_x
    jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_EuclidSolve(euclid_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
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
 * \fn JX_Int JX_PAMGEuclidSolve
 */
JX_Int
JX_PAMGEuclidSolve( JX_Solver       solver,
                    JX_ParCSRMatrix par_matrix,
                    JX_ParVector    par_b,
                    JX_ParVector    par_x )
{
    return( jx_PAMGEuclidSolve( (void *) solver,
                                (jx_ParCSRMatrix *) par_matrix,
                                (jx_ParVector *)par_b,
                                (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_PAMGEuclidSolve
 */
JX_Int
jx_PAMGEuclidSolve( void            *data,
                    jx_ParCSRMatrix *par_matrix,
                    jx_ParVector    *par_b,
                    jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One V-cycle for A w = g, where A = par_matrix, g = par_b, w = par_x
    //jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_PAMGPrecond(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    // 2. r = g - Aw, where r = res_vec, g = par_b, w = par_x
    jx_ParVectorCopy(par_b, res_vec);
    jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_x, 1.0, res_vec);
    // 3. One ILU(k) for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    //jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_EuclidSolve(euclid_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
    // 4. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_EuclidPAMGEuclidSolve
 */
JX_Int
JX_EuclidPAMGEuclidSolve( JX_Solver       solver,
                          JX_ParCSRMatrix par_matrix,
                          JX_ParVector    par_b,
                          JX_ParVector    par_x )
{
    return( jx_EuclidPAMGEuclidSolve( (void *) solver,
                                      (jx_ParCSRMatrix *) par_matrix,
                                      (jx_ParVector *)par_b,
                                      (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_EuclidPAMGEuclidSolve
 */
JX_Int
jx_EuclidPAMGEuclidSolve( void            *data,
                          jx_ParCSRMatrix *par_matrix,
                          jx_ParVector    *par_b,
                          jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Solver amg_solver = jx_CombinedPrecondDataPAMGData(pre_data);
    JX_Solver euclid_solver = jx_CombinedPrecondDataEuclidData(pre_data);
    jx_ParVector *aux_vec = jx_CombinedPrecondDataAuxVector(pre_data);
    jx_ParVector *res_vec = jx_CombinedPrecondDataResVector(pre_data);
    
    // 1. One ILU(k) for A w = g, where A = par_matrix, g = par_b, w = par_x
    jx_ParVectorSetConstantValues(par_x, 0.0);
    JX_EuclidSolve(euclid_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
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
    // 6. One ILU(k) for A v = r, where A = par_matrix, r = res_vec, v = aux_vec
    jx_ParVectorSetConstantValues(aux_vec, 0.0);
    JX_EuclidSolve(euclid_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
    // 7. w = w + v, where w = par_x, v = aux_vec
    jx_ParVectorAxpy(1.0, aux_vec, par_x);
    
    return jx_error_flag;
}
