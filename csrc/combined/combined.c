//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  combined.c
 *  Date: 2013/12/11
 *
 *  Created by Yue Xiaoqiang
 *
 */

#include "jx_combined.h"

#define JX_EUCLID_ERRCHKB \
      if (jx_errFlag_dh) { \
        jx_setError_dh("", __FUNC__, __FILE__, __LINE__); \
        jx_printErrorMsg(stderr); \
        jx_MPI_Abort(jx_comm_dh, -1); \
      }

#undef ENABLE_EUCLID_LOGGING

#if !defined(ENABLE_EUCLID_LOGGING)
#undef JX_START_FUNC_DH
#undef JX_END_FUNC_VAL
#undef JX_END_FUNC_DH
#define JX_START_FUNC_DH
#define JX_END_FUNC_DH
#define JX_END_FUNC_VAL(a) return(a);
#endif

/*!
 * \fn JX_Int JX_CombinedPrecondDataCreate
 */
JX_Int
JX_CombinedPrecondDataCreate( JX_Solver *solver, MPI_Comm comm )
{
   *solver = (JX_Solver) jx_CombinedPrecondDataCreate(comm);
    if (!solver)
    {
        jx_error_in_arg(1);
    }
    
    return jx_error_flag;
}

/*!
 * \fn void *jx_CombinedPrecondDataCreate
 */
void *
jx_CombinedPrecondDataCreate( MPI_Comm comm )
{
    jx_CombinedPrecondData *pre_data;
    JX_Int pre_id = 1;
    
    pre_data = jx_CTAlloc(jx_CombinedPrecondData, 1);
    jx_CombinedPrecondDataComm(pre_data) = comm;
    jx_CombinedPrecondDataSetPreID(pre_data, pre_id);
    
    return (void *)pre_data;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataInitialize
 */
JX_Int
JX_CombinedPrecondDataInitialize( JX_Solver solver, JX_Int level )
{
    return( jx_CombinedPrecondDataInitialize( (void *) solver, level ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataInitialize
 */
JX_Int
jx_CombinedPrecondDataInitialize( void *data, JX_Int level )
{
    if (!data)
    {
        jx_error_in_arg(1);
        return jx_error_flag;
    }
    jx_CombinedPrecondData *pre_data = data;
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    jx_CSRMatrix *ser_mat = NULL;
    JX_Int nprocs, myid, stg_dd;
    
    jx_MPI_Comm_size(comm, &nprocs);
    jx_MPI_Comm_rank(comm, &myid);
    if (pre_id == 11) // PAMG-Euclid-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
    }
    else if (pre_id == 12) // Euclid-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
    }
    else if (pre_id == 13) // PAMG-Euclid
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
    }
    else if (pre_id == 14) // Euclid-PAMG-Euclid
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
    }
    else if (pre_id == 15) // Adaptive
    {
        if (nprocs == 1)
        {
            if (myid == 0)
            {
                stg_dd = jx_CSRMatrixWeaklyDiagDominant(
                                    jx_ParCSRMatrixDiag(jx_CombinedPrecondDataPreMat(pre_data)),
                                        jx_CombinedPrecondDataNumEquns(pre_data),
                                            jx_CombinedPrecondDataAdpTheta(pre_data),
                                                jx_CombinedPrecondDataAdpGammaT(pre_data),
                                                    jx_CombinedPrecondDataAdpGammaE(pre_data));
                if (stg_dd == 1)
                {
                    JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
                    JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
                    JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                    JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
                    jx_CombinedPrecondDataPreID(pre_data) = 13; // switch to PAMG-Euclid
                }
                else
                {
                    JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                    JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
                }
            }
        }
        else
        {
            ser_mat = jx_ParCSRMatrixToCSRMatrixAll(jx_CombinedPrecondDataPreMat(pre_data));
            if (myid == 0)
            {
                stg_dd = jx_CSRMatrixWeaklyDiagDominant(ser_mat,
                                        jx_CombinedPrecondDataNumEquns(pre_data),
                                            jx_CombinedPrecondDataAdpTheta(pre_data),
                                                jx_CombinedPrecondDataAdpGammaT(pre_data),
                                                    jx_CombinedPrecondDataAdpGammaE(pre_data));
            }
            jx_CSRMatrixDestroy(ser_mat);
            jx_MPI_Bcast(&stg_dd, 1, JX_MPI_INT, 0, comm);
            if (stg_dd == 1)
            {
                JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
                JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
                JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
                jx_CombinedPrecondDataPreID(pre_data) = 13; // switch to PAMG-Euclid
            }
            else
            {
                JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
            }
        }
    }
    else if (pre_id == 16) // ILU-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
    }
    else if (pre_id == 17) // ILU-PAMG-ILU
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
    }
    else if (pre_id == 18) // PAMG-Euclid-PAMG, where Euclid for (A+A^T)/2
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidLevel(jx_CombinedPrecondDataEuclidData(pre_data), level);
    }
    else if (pre_id == 19) // PAMG-ILU-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
    }
    else
    {
        jx_printf("\n Warning: Wrong preconditioner ID\n\n");
        exit(-3);
    }
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetEuclidLevel
 */
#undef __FUNC__
#define __FUNC__ "JX_CombinedPrecondDataSetEuclidLevel"
JX_Int
JX_CombinedPrecondDataSetEuclidLevel( JX_Solver solver, JX_Int level )
{
    char str_level[8];
    JX_START_FUNC_DH
    jx_sprintf(str_level,"%d",level);
    jx_Parser_dhInsert(jx_parser_dh, "-level", str_level); JX_EUCLID_ERRCHKB;
    JX_END_FUNC_VAL(0)
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataInitializeP
 */
JX_Int
JX_CombinedPrecondDataInitializeP( JX_Solver solver,
                                   JX_Int argc,
                                   char *argv[] )
{
    return( jx_CombinedPrecondDataInitializeP( (void *) solver, argc, argv ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataInitializeP
 */
JX_Int
jx_CombinedPrecondDataInitializeP( void *data,
                                   JX_Int argc,
                                   char *argv[] )
{
    if (!data)
    {
        jx_error_in_arg(1);
        return jx_error_flag;
    }
    jx_CombinedPrecondData *pre_data = data;
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    jx_CSRMatrix *ser_mat = NULL;
    JX_Int nprocs, myid, stg_dd;
    
    jx_MPI_Comm_size(comm, &nprocs);
    jx_MPI_Comm_rank(comm, &myid);
    if (pre_id == 11) // PAMG-Euclid-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
    }
    else if (pre_id == 12) // Euclid-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
    }
    else if (pre_id == 13) // PAMG-Euclid
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
    }
    else if (pre_id == 14) // Euclid-PAMG-Euclid
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
    }
    else if (pre_id == 15) // Adaptive
    {
        if (nprocs == 1)
        {
            if (myid == 0)
            {
                stg_dd = jx_CSRMatrixWeaklyDiagDominant(
                                    jx_ParCSRMatrixDiag(jx_CombinedPrecondDataPreMat(pre_data)),
                                        jx_CombinedPrecondDataNumEquns(pre_data),
                                            jx_CombinedPrecondDataAdpTheta(pre_data),
                                                jx_CombinedPrecondDataAdpGammaT(pre_data),
                                                    jx_CombinedPrecondDataAdpGammaE(pre_data));
                if (stg_dd == 1)
                {
                    JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
                    JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
                    JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                    JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
                    jx_CombinedPrecondDataPreID(pre_data) = 13; // switch to PAMG-Euclid
                }
                else
                {
                    JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                    JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
                }
            }
        }
        else
        {
            ser_mat = jx_ParCSRMatrixToCSRMatrixAll(jx_CombinedPrecondDataPreMat(pre_data));
            if (myid == 0)
            {
                stg_dd = jx_CSRMatrixWeaklyDiagDominant(ser_mat,
                                        jx_CombinedPrecondDataNumEquns(pre_data),
                                            jx_CombinedPrecondDataAdpTheta(pre_data),
                                                jx_CombinedPrecondDataAdpGammaT(pre_data),
                                                    jx_CombinedPrecondDataAdpGammaE(pre_data));
            }
            jx_CSRMatrixDestroy(ser_mat);
            jx_MPI_Bcast(&stg_dd, 1, JX_MPI_INT, 0, comm);
            if (stg_dd == 1)
            {
                JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
                JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
                JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
                jx_CombinedPrecondDataPreID(pre_data) = 13; // switch to PAMG-Euclid
            }
            else
            {
                JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
            }
        }
    }
    else if (pre_id == 16) // ILU-PAMG
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
    }
    else if (pre_id == 17) // ILU-PAMG-ILU
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
    }
    else if (pre_id == 18) // PAMG-Euclid-PAMG, where Euclid for (A+A^T)/2
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetAIMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), 0);
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_CombinedPrecondDataSetEuclidParams(jx_CombinedPrecondDataEuclidData(pre_data), argc, argv);
    }
    else
    {
        jx_printf("\n Warning: Wrong preconditioner ID\n\n");
        exit(-3);
    }
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetEuclidParams
 */
#undef __FUNC__
#define __FUNC__ "JX_CombinedPrecondDataSetEuclidParams"
JX_Int
JX_CombinedPrecondDataSetEuclidParams( JX_Solver solver,
                                       JX_Int argc,
                                       char *argv[] )
{
    jx_Parser_dhInit(jx_parser_dh, argc, argv);
    JX_CP_EUCLID_ERRCHKA;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetPreID
 */
JX_Int
JX_CombinedPrecondDataSetPreID( JX_Solver solver, JX_Int pre_id )
{
    return( jx_CombinedPrecondDataSetPreID( (void *) solver, pre_id ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetPreID
 */
JX_Int
jx_CombinedPrecondDataSetPreID( void *data, JX_Int pre_id )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataPreID(pre_data) = pre_id;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetNxyNpxyNequ
 */
JX_Int
JX_CombinedPrecondDataSetNxyNpxyNequ( JX_Solver solver, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy, JX_Int num_equns )
{
    return( jx_CombinedPrecondDataSetNxyNpxyNequ( (void *) solver, nx, ny, npx, npy, num_equns ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetNxyNpxyNequ
 */
JX_Int
jx_CombinedPrecondDataSetNxyNpxyNequ( void *data, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy, JX_Int num_equns )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataNx(pre_data) = nx;
    jx_CombinedPrecondDataNy(pre_data) = ny;
    jx_CombinedPrecondDataNpx(pre_data) = npx;
    jx_CombinedPrecondDataNpy(pre_data) = npy;
    jx_CombinedPrecondDataNumEquns(pre_data) = num_equns;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetThetaPsiRhoPhi
 */
JX_Int
JX_CombinedPrecondDataSetThetaPsiRhoPhi( JX_Solver solver,
                                         JX_Int theta_psi,
                                         JX_Int theta_rho,
                                         JX_Int theta_phi,
                                         JX_Real theta_dis )
{
    return( jx_CombinedPrecondDataSetThetaPsiRhoPhi( (void *) solver, theta_psi, theta_rho, theta_phi, theta_dis ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetThetaPsiRhoPhi
 */
JX_Int
jx_CombinedPrecondDataSetThetaPsiRhoPhi( void *data,
                                         JX_Int theta_psi,
                                         JX_Int theta_rho,
                                         JX_Int theta_phi,
                                         JX_Real theta_dis )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataThetaPsi(pre_data) = theta_psi;
    jx_CombinedPrecondDataThetaRho(pre_data) = theta_rho;
    jx_CombinedPrecondDataThetaPhi(pre_data) = theta_phi;
    jx_CombinedPrecondDataThetaDis(pre_data) = theta_dis;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetAMGParameters
 */
JX_Int
JX_CombinedPrecondDataSetAMGParameters( JX_Solver solver,
                                        JX_Int amg_max_levels,
                                        JX_Int amg_relax_type,
                                        JX_Int amg_print_level,
                                        JX_Int amg_interp_type,
                                        JX_Int amg_P_max_elmts,
                                        JX_Int amg_measure_type,
                                        JX_Int amg_coarsen_type,
                                        JX_Int amg_agg_num_levels,
                                        JX_Int amg_coarse_threshold,
                                        JX_Real amg_strong_threshold )
{
    return( jx_CombinedPrecondDataSetAMGParameters( (void *) solver, amg_max_levels, amg_relax_type,
                                                    amg_print_level, amg_interp_type, amg_P_max_elmts,
                                                    amg_measure_type, amg_coarsen_type, amg_agg_num_levels,
                                                    amg_coarse_threshold, amg_strong_threshold ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetAMGParameters
 */
JX_Int
jx_CombinedPrecondDataSetAMGParameters( void *data,
                                        JX_Int amg_max_levels,
                                        JX_Int amg_relax_type,
                                        JX_Int amg_print_level,
                                        JX_Int amg_interp_type,
                                        JX_Int amg_P_max_elmts,
                                        JX_Int amg_measure_type,
                                        JX_Int amg_coarsen_type,
                                        JX_Int amg_agg_num_levels,
                                        JX_Int amg_coarse_threshold,
                                        JX_Real amg_strong_threshold )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAMGMaxLevels(pre_data) = amg_max_levels;
    jx_CombinedPrecondDataAMGRelaxType(pre_data) = amg_relax_type;
    jx_CombinedPrecondDataAMGPrintLevel(pre_data) = amg_print_level;
    jx_CombinedPrecondDataAMGInterpType(pre_data) = amg_interp_type;
    jx_CombinedPrecondDataAMGPMaxElmts(pre_data) = amg_P_max_elmts;
    jx_CombinedPrecondDataAMGMeasureType(pre_data) = amg_measure_type;
    jx_CombinedPrecondDataAMGCoarsenType(pre_data) = amg_coarsen_type;
    jx_CombinedPrecondDataAMGAggNumLevels(pre_data) = amg_agg_num_levels;
    jx_CombinedPrecondDataAMGCoarseThreshold(pre_data) = amg_coarse_threshold;
    jx_CombinedPrecondDataAMGStrongThreshold(pre_data) = amg_strong_threshold;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetPreMat
 */
JX_Int
JX_CombinedPrecondDataSetPreMat( JX_Solver solver, jx_ParCSRMatrix *pre_mat )
{
    return( jx_CombinedPrecondDataSetPreMat( (void *) solver, pre_mat ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetPreMat
 */
JX_Int
jx_CombinedPrecondDataSetPreMat( void *data, jx_ParCSRMatrix *pre_mat )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataPreMat(pre_data) = pre_mat;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetDropTol
 */
JX_Int
JX_CombinedPrecondDataSetDropTol( JX_Solver solver, JX_Real drop_tol )
{
    return( jx_CombinedPrecondDataSetDropTol( (void *) solver, drop_tol ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetDropTol
 */
JX_Int
jx_CombinedPrecondDataSetDropTol( void *data, JX_Real drop_tol )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataDropTol(pre_data) = drop_tol;
    
    return jx_error_flag;
}

JX_Int
JX_CombinedPrecondDataSetInterpType( JX_Solver solver, JX_Int interp_type )
{
    return( jx_CombinedPrecondDataSetInterpType( (void *) solver, interp_type ) );
}

JX_Int
jx_CombinedPrecondDataSetInterpType( void *data, JX_Int interp_type )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAMGInterpType(pre_data) = interp_type;
    
    return jx_error_flag;
}

JX_Int
JX_CombinedPrecondDataSetCoarsenType( JX_Solver solver, JX_Int coarsen_type )
{
    return( jx_CombinedPrecondDataSetCoarsenType( (void *) solver, coarsen_type ) );
}

JX_Int
jx_CombinedPrecondDataSetCoarsenType( void *data, JX_Int coarsen_type )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAMGCoarsenType(pre_data) = coarsen_type;
    
    return jx_error_flag;
}

JX_Int
JX_CombinedPrecondDataSetCycleRelaxType( JX_Solver solver, JX_Int relax_type )
{
    return( jx_CombinedPrecondDataSetCycleRelaxType( (void *) solver, relax_type ) );
}

JX_Int
jx_CombinedPrecondDataSetCycleRelaxType( void *data, JX_Int relax_type )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAMGRelaxType(pre_data) = relax_type;
    
    return jx_error_flag;
}

JX_Int
JX_CombinedPrecondDataSetStrongThreshold( JX_Solver solver, JX_Real strong_threshold )
{
    return( jx_CombinedPrecondDataSetStrongThreshold( (void *) solver, strong_threshold ) );
}

JX_Int
jx_CombinedPrecondDataSetStrongThreshold( void *data, JX_Real strong_threshold )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAMGStrongThreshold(pre_data) = strong_threshold;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetAdpTheta
 */
JX_Int
JX_CombinedPrecondDataSetAdpTheta( JX_Solver solver, JX_Real adp_theta )
{
    return( jx_CombinedPrecondDataSetAdpTheta( (void *) solver, adp_theta ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetAdpTheta
 */
JX_Int
jx_CombinedPrecondDataSetAdpTheta( void *data, JX_Real adp_theta )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAdpTheta(pre_data) = adp_theta;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetAdpGammaT
 */
JX_Int
JX_CombinedPrecondDataSetAdpGammaT( JX_Solver solver, JX_Real adp_gamma_3 )
{
    return( jx_CombinedPrecondDataSetAdpGammaT( (void *) solver, adp_gamma_3 ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetAdpGammaT
 */
JX_Int
jx_CombinedPrecondDataSetAdpGammaT( void *data, JX_Real adp_gamma_3 )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAdpGammaT(pre_data) = adp_gamma_3;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetAdpGammaE
 */
JX_Int
JX_CombinedPrecondDataSetAdpGammaE( JX_Solver solver, JX_Real adp_gamma_11 )
{
    return( jx_CombinedPrecondDataSetAdpGammaE( (void *) solver, adp_gamma_11 ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetAdpGammaE
 */
JX_Int
jx_CombinedPrecondDataSetAdpGammaE( void *data, JX_Real adp_gamma_11 )
{
    jx_CombinedPrecondData *pre_data = data;
    
    jx_CombinedPrecondDataAdpGammaE(pre_data) = adp_gamma_11;
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetILUMatA
 */
JX_Int
JX_CombinedPrecondDataSetILUMatA( JX_Solver solver, jx_CSRMatrix *matA )
{
    return( jx_CombinedPrecondDataSetILUMatA( (void *) solver, matA ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetILUMatA
 */
JX_Int
jx_CombinedPrecondDataSetILUMatA( void *data, jx_CSRMatrix *matA )
{
    jx_CombinedPrecondData *pre_data = data;
    if (!pre_data)
    {
        jx_printf(" Warning: CombinedPrecond object empty!\n");
        jx_error_in_arg(1);
        return jx_error_flag;
    }
    JX_ILUZeroFactorDataSetMatA(jx_CombinedPrecondDataILUZData(pre_data), matA);
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetPreEucMat
 */
JX_Int
JX_CombinedPrecondDataSetPreEucMat( JX_Solver solver, jx_ParCSRMatrix *matA )
{
    return( jx_CombinedPrecondDataSetPreEucMat( (void *) solver, matA ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetPreEucMat
 */
JX_Int
jx_CombinedPrecondDataSetPreEucMat( void *data, jx_ParCSRMatrix *matA )
{
    jx_CombinedPrecondData *pre_data = data;
    if (!pre_data)
    {
        jx_printf(" Warning: CombinedPrecond object empty!\n");
        jx_error_in_arg(1);
        return jx_error_flag;
    }
    jx_CombinedPrecondDataPreEucMat(pre_data) = matA;
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataGetLULength
 */
JX_Int
JX_CombinedPrecondDataGetLULength( JX_Solver solver, JX_Int *lu_length )
{
    return( jx_CombinedPrecondDataGetLULength( (void *) solver, lu_length ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataGetLULength
 */
JX_Int
jx_CombinedPrecondDataGetLULength( void *data, JX_Int *lu_length )
{
    jx_CombinedPrecondData *pre_data = data;
    if (!pre_data)
    {
        jx_printf(" Warning: CombinedPrecond object empty!\n");
        jx_error_in_arg(1);
        return jx_error_flag;
    }
    JX_ILUZeroFactorDataGetLULength(jx_CombinedPrecondDataILUZData(pre_data), lu_length);
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSetup
 */
JX_Int
JX_CombinedPrecondDataSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_CombinedPrecondDataSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSetup
 */
JX_Int
jx_CombinedPrecondDataSetup( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    
    if (pre_id == 11) // PAMG-Euclid-PAMG
    {
        jx_PAMGEuclidPAMGSetup(data, par_matrix);
    }
    else if (pre_id == 12) // Euclid-PAMG
    {
        jx_EuclidPAMGSetup(data, par_matrix);
    }
    else if (pre_id == 13) // PAMG-Euclid
    {
        jx_PAMGEuclidSetup(data, par_matrix);
    }
    else if (pre_id == 14) // Euclid-PAMG-Euclid
    {
        jx_EuclidPAMGEuclidSetup(data, par_matrix);
    }
    else if (pre_id == 15) // Euclid
    {
        JX_EuclidSetup(jx_CombinedPrecondDataEuclidData(pre_data), (JX_ParCSRMatrix)par_matrix);
    }
    else if (pre_id == 16) // ILU-PAMG
    {
        jx_ILUZPAMGSetup(data, par_matrix);
    }
    else if (pre_id == 17) // ILU-PAMG-ILU
    {
        jx_ILUZPAMGILUZSetup(data, par_matrix);
    }
    else if (pre_id == 18) // PAMG-Euclid-PAMG, where Euclid for (A+A^T)/2
    {
        jx_PAMGEuclidPAMGSetupB(data, par_matrix);
    }
    else if (pre_id == 19) // PAMG-ILU-PAMG
    {
        jx_PAMGILUZPAMGSetup(data, par_matrix);
    }
    else
    {
        jx_printf("\n Warning: Wrong preconditioner ID\n\n");
        exit(-3);
    }
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataAdaptiveSetup2
 */
JX_Int
JX_CombinedPrecondDataAdaptiveSetup2( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_CombinedPrecondDataAdaptiveSetup2( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataAdaptiveSetup2
 */
JX_Int
jx_CombinedPrecondDataAdaptiveSetup2( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Int nx = jx_CombinedPrecondDataNx(pre_data);
    JX_Int ny = jx_CombinedPrecondDataNy(pre_data);
    JX_Int npx = jx_CombinedPrecondDataNpx(pre_data);
    JX_Int npy = jx_CombinedPrecondDataNpy(pre_data);
    JX_Int num_equns = jx_CombinedPrecondDataNumEquns(pre_data);
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    JX_Int euclid_level = jx_CombinedPrecondDataEuclidLevel(pre_data);
    JX_Real drop_tol = jx_CombinedPrecondDataDropTol(pre_data);
    
    if (pre_id == 1) // Solve by Euclid firstly, Yue Xiaoqiang 2014/10/24
    {
        JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
        JX_EuclidSetLevel(jx_CombinedPrecondDataEuclidData(pre_data), euclid_level);
        JX_EuclidSetSparseA(jx_CombinedPrecondDataEuclidData(pre_data), drop_tol);
        JX_EuclidSetup(jx_CombinedPrecondDataEuclidData(pre_data), (JX_ParCSRMatrix)par_matrix);
    }
    else if (pre_id == 4) // Solve by ILU(0) - crossed firstly, Yue Xiaoqiang 2014/12/03
    {
        JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
        JX_ILUZeroFactorDataSetNxy(jx_CombinedPrecondDataILUZData(pre_data), nx, ny);
        JX_ILUZeroFactorDataSetNpxy(jx_CombinedPrecondDataILUZData(pre_data), npx, npy);
        JX_ILUZeroFactorDataSetNumEquns(jx_CombinedPrecondDataILUZData(pre_data), num_equns);
        JX_ILUZeroFactorDataSetDropTol(jx_CombinedPrecondDataILUZData(pre_data), drop_tol);
        JX_ILUZeroFactorDataGenerateParGrid(jx_CombinedPrecondDataILUZData(pre_data));
        JX_ILUZeroFactorDataSetup(jx_CombinedPrecondDataILUZData(pre_data),(JX_ParCSRMatrix)par_matrix);
    }
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataAdaptiveSetup3
 */
JX_Int
JX_CombinedPrecondDataAdaptiveSetup3( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
    return( jx_CombinedPrecondDataAdaptiveSetup3( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataAdaptiveSetup3
 */
JX_Int
jx_CombinedPrecondDataAdaptiveSetup3( void *data, jx_ParCSRMatrix *par_matrix )
{
    jx_CombinedPrecondData *pre_data = data;
    MPI_Comm comm = jx_CombinedPrecondDataComm(pre_data);
    JX_Int nx = jx_CombinedPrecondDataNx(pre_data);
    JX_Int ny = jx_CombinedPrecondDataNy(pre_data);
    JX_Int npx = jx_CombinedPrecondDataNpx(pre_data);
    JX_Int npy = jx_CombinedPrecondDataNpy(pre_data);
    JX_Int num_equns = jx_CombinedPrecondDataNumEquns(pre_data);
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    JX_Int num_rows = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(par_matrix));
    JX_Int theta_psi = jx_CombinedPrecondDataThetaPsi(pre_data);
    JX_Int theta_rho = jx_CombinedPrecondDataThetaRho(pre_data);
    JX_Int theta_phi = jx_CombinedPrecondDataThetaPhi(pre_data);
    JX_Int euclid_level = jx_CombinedPrecondDataEuclidLevel(pre_data);
    JX_Int amg_max_levels = jx_CombinedPrecondDataAMGMaxLevels(pre_data);
    JX_Int amg_relax_type = jx_CombinedPrecondDataAMGRelaxType(pre_data);
    JX_Int amg_print_level = jx_CombinedPrecondDataAMGPrintLevel(pre_data);
    JX_Int amg_interp_type = jx_CombinedPrecondDataAMGInterpType(pre_data);
    JX_Int amg_P_max_elmts = jx_CombinedPrecondDataAMGPMaxElmts(pre_data);
    JX_Int amg_measure_type = jx_CombinedPrecondDataAMGMeasureType(pre_data);
    JX_Int amg_coarsen_type = jx_CombinedPrecondDataAMGCoarsenType(pre_data);
    JX_Int amg_agg_num_levels = jx_CombinedPrecondDataAMGAggNumLevels(pre_data);
    JX_Int amg_coarse_threshold = jx_CombinedPrecondDataAMGCoarseThreshold(pre_data);
    JX_Int global_num_rows = jx_ParCSRMatrixGlobalNumRows(par_matrix);
    JX_Real drop_tol = jx_CombinedPrecondDataDropTol(pre_data);
    JX_Real theta_dis = jx_CombinedPrecondDataThetaDis(pre_data);
    JX_Real amg_strong_threshold = jx_CombinedPrecondDataAMGStrongThreshold(pre_data);
    jx_CSRMatrix *ser_matrix = NULL;
    JX_Real *max_dvd_min = NULL;
    JX_Int *mat_dis_psi = NULL, *max_mat_dis = NULL;
    JX_Int row, pos_srt, pos_num, mat_flg = 0, tmp_row = 0;
    JX_Int mat_psi, max_mat_psi, mat_rho, mat_phi;
    JX_Int num_procs;
    JX_Real fabs_max, fabs_min;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_CombinedPrecondDataActualPsi(pre_data) = 0; // undecided
    jx_CombinedPrecondDataActualRho(pre_data) = 0; // undecided
    jx_CombinedPrecondDataActualPhi(pre_data) = 0; // undecided
    ser_matrix = jx_MergeDiagAndOffdDropSmall(par_matrix, 0.0); // Delete zero-elements
    max_dvd_min = jx_CTAlloc(JX_Real, num_rows);
    for (row = 0; row < num_rows; row ++)
    {
        pos_srt = jx_CSRMatrixI(ser_matrix)[row] + 1;
        pos_num = jx_CSRMatrixI(ser_matrix)[row+1] - pos_srt;
        fabs_max = jx_DoubleArrayAbsMaxElement(&jx_CSRMatrixData(ser_matrix)[pos_srt], pos_num);
        fabs_min = jx_DoubleArrayAbsMinElement(&jx_CSRMatrixData(ser_matrix)[pos_srt], pos_num);
        max_dvd_min[row] = fabs_max / fabs_min;
    }
    mat_psi = (JX_Int)log10(jx_DoubleArrayMaxElement(max_dvd_min, num_rows));
    jx_MPI_Allreduce(&mat_psi, &max_mat_psi, 1, JX_MPI_INT, MPI_MAX, comm);
    jx_CombinedPrecondDataActualPsi(pre_data) = max_mat_psi;
    if (max_mat_psi < theta_psi) // Solve by AMG, Yue Xiaoqiang 2014/10/24
    {
        JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
        JX_PAMGSetMaxLevels(jx_CombinedPrecondDataPAMGData(pre_data), amg_max_levels);
        JX_PAMGSetMaxIter(jx_CombinedPrecondDataPAMGData(pre_data), 1);
        JX_PAMGSetMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), amg_measure_type);
        JX_PAMGSetCoarsenType(jx_CombinedPrecondDataPAMGData(pre_data), amg_coarsen_type);
        JX_PAMGSetInterpType(jx_CombinedPrecondDataPAMGData(pre_data), amg_interp_type);
        JX_PAMGSetPMaxElmts(jx_CombinedPrecondDataPAMGData(pre_data), amg_P_max_elmts);
        JX_PAMGSetAggNumLevels(jx_CombinedPrecondDataPAMGData(pre_data), amg_agg_num_levels);
        JX_PAMGSetStrongThreshold(jx_CombinedPrecondDataPAMGData(pre_data), amg_strong_threshold);
        JX_PAMGSetPrintLevel(jx_CombinedPrecondDataPAMGData(pre_data), amg_print_level);
        JX_PAMGSetCoarseThreshold(jx_CombinedPrecondDataPAMGData(pre_data), amg_coarse_threshold);
        JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 1);
        JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 2);
        JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 3);
        JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), amg_relax_type, 1);
        JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), amg_relax_type, 2);
        JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), 9, 3);
        JX_PAMGSetup(jx_CombinedPrecondDataPAMGData(pre_data), (JX_ParCSRMatrix)par_matrix);
        jx_CombinedPrecondDataPreID(pre_data) = 3;
    }
    else
    {
        mat_psi = max_mat_psi + 1;
        mat_dis_psi = jx_CTAlloc(JX_Int, mat_psi);
        memset(mat_dis_psi, 0, mat_psi*sizeof(JX_Int));
        for (row = 0; row < num_rows; row ++)
        {
            mat_dis_psi[(JX_Int)log10(max_dvd_min[row])] ++;
        }
        max_mat_dis = jx_CTAlloc(JX_Int, mat_psi);
        memset(max_mat_dis, 0, mat_psi*sizeof(JX_Int));
        jx_MPI_Allreduce(mat_dis_psi, max_mat_dis, mat_psi, JX_MPI_INT, MPI_SUM, comm);
        mat_rho = 0;
        for (row = 0; row < mat_psi; row ++)
        {
            if (max_mat_dis[row] >= theta_dis*global_num_rows)
            {
                mat_rho ++;
            }
        }
        jx_CombinedPrecondDataActualRho(pre_data) = mat_rho;
        if (mat_rho < theta_rho || mat_rho == 1) // Solve by AMG, Yue Xiaoqiang 2014/10/24
        {
            JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
            JX_PAMGSetMaxLevels(jx_CombinedPrecondDataPAMGData(pre_data), amg_max_levels);
            JX_PAMGSetMaxIter(jx_CombinedPrecondDataPAMGData(pre_data), 1);
            JX_PAMGSetMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), amg_measure_type);
            JX_PAMGSetCoarsenType(jx_CombinedPrecondDataPAMGData(pre_data), amg_coarsen_type);
            JX_PAMGSetInterpType(jx_CombinedPrecondDataPAMGData(pre_data), amg_interp_type);
            JX_PAMGSetPMaxElmts(jx_CombinedPrecondDataPAMGData(pre_data), amg_P_max_elmts);
            JX_PAMGSetAggNumLevels(jx_CombinedPrecondDataPAMGData(pre_data), amg_agg_num_levels);
            JX_PAMGSetStrongThreshold(jx_CombinedPrecondDataPAMGData(pre_data), amg_strong_threshold);
            JX_PAMGSetPrintLevel(jx_CombinedPrecondDataPAMGData(pre_data), amg_print_level);
            JX_PAMGSetCoarseThreshold(jx_CombinedPrecondDataPAMGData(pre_data), amg_coarse_threshold);
            JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 1);
            JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 2);
            JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 3);
            JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), amg_relax_type, 1);
            JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), amg_relax_type, 2);
            JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), 9, 3);
            JX_PAMGSetup(jx_CombinedPrecondDataPAMGData(pre_data), (JX_ParCSRMatrix)par_matrix);
            jx_CombinedPrecondDataPreID(pre_data) = 3;
        }
        else
        {
            mat_phi = 0;
            for (row = 0; row < mat_psi; row ++)
            {
                if (max_mat_dis[row] >= theta_dis*global_num_rows)
                {
                    if (mat_flg == 0)
                    {
                        mat_flg = 1;
                        tmp_row = row;
                    }
                    else
                    {
                        mat_phi += (row - tmp_row - 1);
                        tmp_row = row;
                    }
                }
            }
            jx_CombinedPrecondDataActualPhi(pre_data) = mat_phi;
            if (mat_phi < theta_phi) // Solve by AMG, Yue Xiaoqiang 2014/10/24
            {
                JX_PAMGCreate(&jx_CombinedPrecondDataPAMGData(pre_data));
                JX_PAMGSetMaxLevels(jx_CombinedPrecondDataPAMGData(pre_data), amg_max_levels);
                JX_PAMGSetMaxIter(jx_CombinedPrecondDataPAMGData(pre_data), 1);
                JX_PAMGSetMeasureType(jx_CombinedPrecondDataPAMGData(pre_data), amg_measure_type);
                JX_PAMGSetCoarsenType(jx_CombinedPrecondDataPAMGData(pre_data), amg_coarsen_type);
                JX_PAMGSetInterpType(jx_CombinedPrecondDataPAMGData(pre_data), amg_interp_type);
                JX_PAMGSetPMaxElmts(jx_CombinedPrecondDataPAMGData(pre_data), amg_P_max_elmts);
                JX_PAMGSetAggNumLevels(jx_CombinedPrecondDataPAMGData(pre_data), amg_agg_num_levels);
                JX_PAMGSetStrongThreshold(jx_CombinedPrecondDataPAMGData(pre_data), amg_strong_threshold);
                JX_PAMGSetPrintLevel(jx_CombinedPrecondDataPAMGData(pre_data), amg_print_level);
                JX_PAMGSetCoarseThreshold(jx_CombinedPrecondDataPAMGData(pre_data), amg_coarse_threshold);
                JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 1);
                JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 2);
                JX_PAMGSetCycleNumSweeps(jx_CombinedPrecondDataPAMGData(pre_data), 1, 3);
                JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), amg_relax_type, 1);
                JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), amg_relax_type, 2);
                JX_PAMGSetCycleRelaxType(jx_CombinedPrecondDataPAMGData(pre_data), 9, 3);
                JX_PAMGSetup(jx_CombinedPrecondDataPAMGData(pre_data), (JX_ParCSRMatrix)par_matrix);
                jx_CombinedPrecondDataPreID(pre_data) = 3;
            }
            else if (pre_id == 1) // Solve by Euclid firstly, Yue Xiaoqiang 2014/10/24
            {
                JX_EuclidCreate(comm, &jx_CombinedPrecondDataEuclidData(pre_data));
                JX_EuclidSetLevel(jx_CombinedPrecondDataEuclidData(pre_data), euclid_level);
                JX_EuclidSetSparseA(jx_CombinedPrecondDataEuclidData(pre_data), drop_tol);
                JX_EuclidSetup(jx_CombinedPrecondDataEuclidData(pre_data), (JX_ParCSRMatrix)par_matrix);
            }
            else if (pre_id == 4) // Solve by ILU(0) - crossed firstly, Yue Xiaoqiang 2014/12/03
            {
                JX_ILUZeroFactorDataCreate(&jx_CombinedPrecondDataILUZData(pre_data), comm);
                JX_ILUZeroFactorDataSetNxy(jx_CombinedPrecondDataILUZData(pre_data), nx, ny);
                JX_ILUZeroFactorDataSetNpxy(jx_CombinedPrecondDataILUZData(pre_data), npx, npy);
                JX_ILUZeroFactorDataSetNumEquns(jx_CombinedPrecondDataILUZData(pre_data), num_equns);
                JX_ILUZeroFactorDataSetDropTol(jx_CombinedPrecondDataILUZData(pre_data), drop_tol);
                JX_ILUZeroFactorDataGenerateParGrid(jx_CombinedPrecondDataILUZData(pre_data));
                JX_ILUZeroFactorDataSetup(jx_CombinedPrecondDataILUZData(pre_data),(JX_ParCSRMatrix)par_matrix);
            }
        }
    }
    jx_TFree(max_dvd_min);
    if (mat_dis_psi)
    {
        jx_TFree(mat_dis_psi);
    }
    if (max_mat_dis)
    {
        jx_TFree(max_mat_dis);
    }
    jx_CSRMatrixDestroy(ser_matrix);
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataSolve
 */
JX_Int
JX_CombinedPrecondDataSolve( JX_Solver       solver,
                             JX_ParCSRMatrix par_matrix,
                             JX_ParVector    par_b,
                             JX_ParVector    par_x )
{
    return( jx_CombinedPrecondDataSolve( (void *) solver,
                                         (jx_ParCSRMatrix *) par_matrix,
                                         (jx_ParVector *)par_b,
                                         (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataSolve
 */
JX_Int
jx_CombinedPrecondDataSolve( void            *data,
                             jx_ParCSRMatrix *par_matrix,
                             jx_ParVector    *par_b,
                             jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    
    if (pre_id == 11) // PAMG-Euclid-PAMG
    {
        jx_PAMGEuclidPAMGSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 12) // Euclid-PAMG
    {
        jx_EuclidPAMGSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 13) // PAMG-Euclid
    {
        jx_PAMGEuclidSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 14) // Euclid-PAMG-Euclid
    {
        jx_EuclidPAMGEuclidSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 15) // Euclid
    {
        JX_EuclidSolve(jx_CombinedPrecondDataEuclidData(pre_data),
                (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    }
    else if (pre_id == 16) // ILU-PAMG
    {
        jx_ILUZPAMGSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 17) // ILU-PAMG-ILU
    {
        jx_ILUZPAMGILUZSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 18) // PAMG-Euclid-PAMG, where Euclid for (A+A^T)/2
    {
        jx_PAMGEuclidPAMGSolve(data, par_matrix, par_b, par_x);
    }
    else if (pre_id == 19) // PAMG-ILU-PAMG
    {
        jx_PAMGILUZPAMGSolve(data, par_matrix, par_b, par_x);
    }
    else
    {
        jx_printf("\n Warning: Wrong preconditioner ID\n\n");
        exit(-3);
    }
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataAdaptiveSolve
 */
JX_Int
JX_CombinedPrecondDataAdaptiveSolve( JX_Solver       solver,
                                     JX_ParCSRMatrix par_matrix,
                                     JX_ParVector    par_b,
                                     JX_ParVector    par_x )
{
    return( jx_CombinedPrecondDataAdaptiveSolve( (void *) solver,
                                                 (jx_ParCSRMatrix *) par_matrix,
                                                 (jx_ParVector *)par_b,
                                                 (jx_ParVector *)par_x ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataAdaptiveSolve
 */
JX_Int
jx_CombinedPrecondDataAdaptiveSolve( void            *data,
                                     jx_ParCSRMatrix *par_matrix,
                                     jx_ParVector    *par_b,
                                     jx_ParVector    *par_x )
{
    jx_CombinedPrecondData *pre_data = data;
    JX_Int pre_id = jx_CombinedPrecondDataPreID(pre_data);
    
    if (pre_id == 1) // Euclid
    {
        JX_EuclidSolve(jx_CombinedPrecondDataEuclidData(pre_data),
                   (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    }
    else if (pre_id == 3) // AMG
    {
        JX_PAMGPrecond(jx_CombinedPrecondDataPAMGData(pre_data),
                   (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    }
    else if (pre_id == 4) // ILU(0) - crossed
    {
        JX_ILUZeroFactorDataPrecond(jx_CombinedPrecondDataILUZData(pre_data),
                   (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_b, (JX_ParVector)par_x);
    }
    else
    {
        jx_printf("\n Warning: Wrong adaptive preconditioner ID\n\n");
        exit(-3);
    }
    
    return jx_error_flag;
}

/*!
 * \fn JX_Int JX_CombinedPrecondDataDestroy
 */
JX_Int
JX_CombinedPrecondDataDestroy( JX_Solver solver )
{
    return( jx_CombinedPrecondDataDestroy( (void *) solver ) );
}

/*!
 * \fn JX_Int jx_CombinedPrecondDataDestroy
 */
JX_Int
jx_CombinedPrecondDataDestroy( void *data )
{
    jx_CombinedPrecondData *pre_data = data;
    
    if ( jx_CombinedPrecondDataPAMGData(pre_data) )
    {
        JX_PAMGDestroy( jx_CombinedPrecondDataPAMGData(pre_data) );
    }
    if ( jx_CombinedPrecondDataEuclidData(pre_data) )
    {
        JX_EuclidDestroy( jx_CombinedPrecondDataEuclidData(pre_data) );
    }
    if ( jx_CombinedPrecondDataILUZData(pre_data) )
    {
        JX_ILUZeroFactorDataDestroy( jx_CombinedPrecondDataILUZData(pre_data) );
    }
    if ( jx_CombinedPrecondDataAuxVector(pre_data) )
    {
        jx_ParVectorDestroy( jx_CombinedPrecondDataAuxVector(pre_data) );
    }
    if ( jx_CombinedPrecondDataResVector(pre_data) )
    {
        jx_ParVectorDestroy( jx_CombinedPrecondDataResVector(pre_data) );
    }
    
    jx_TFree(pre_data);
    
    return jx_error_flag;
}
