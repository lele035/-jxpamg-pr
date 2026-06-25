//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_combined.h -- head file for combined preconditioners
 *  Date: 2013/12/11
 *
 *  Created by Yue Xiaoqiang
 */

#ifndef JX_COMBINED_HEADER
#define JX_COMBINED_HEADER

#ifndef JX_PAMG_HEADER
#include "jx_pamg.h"
#endif

#ifndef JX_EUCLID_HEADER
#include "jx_euclid.h"
#endif

#ifndef JX_ILU_HEADER
#include "jx_ilu.h"
#endif

#define JX_CP_EUCLID_ERRCHKA \
      if (jx_errFlag_dh) { \
        jx_setError_dh("", __FUNC__, __FILE__, __LINE__); \
        jx_printErrorMsg(stderr); \
        jx_MPI_Abort(jx_comm_dh, -1); \
      }

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jx_CombinedPrecondData
 */
typedef struct
{
    JX_Int nx;
    JX_Int ny;
    JX_Int npx;
    JX_Int npy;
    JX_Int num_equns;
    
    JX_Int pre_id;
    JX_Int theta_psi;
    JX_Int theta_rho;
    JX_Int theta_phi;
    JX_Int actual_psi;
    JX_Int actual_rho;
    JX_Int actual_phi;
    JX_Int euclid_level;
    JX_Int amg_max_levels;
    JX_Int amg_relax_type;
    JX_Int amg_print_level;
    JX_Int amg_interp_type;
    JX_Int amg_P_max_elmts;
    JX_Int amg_measure_type;
    JX_Int amg_coarsen_type;
    JX_Int amg_agg_num_levels;
    JX_Int amg_coarse_threshold;
    
    JX_Real drop_tol;
    JX_Real adp_theta;
    JX_Real theta_dis;
    JX_Real adp_gamma_3;
    JX_Real adp_gamma_11;
    JX_Real amg_strong_threshold;
    
    MPI_Comm comm;
    
    jx_ParCSRMatrix *pre_mat;
    jx_ParCSRMatrix *pre_euc_mat;
    
    jx_ParVector *aux_vec;
    jx_ParVector *res_vec;
    
    JX_Solver ilu_data;
    JX_Solver pamg_data;
    JX_Solver euclid_data;
    
} jx_CombinedPrecondData;

#define jx_CombinedPrecondDataNx(pre_data)                 ((pre_data)->nx)
#define jx_CombinedPrecondDataNy(pre_data)                 ((pre_data)->ny)
#define jx_CombinedPrecondDataNpx(pre_data)                ((pre_data)->npx)
#define jx_CombinedPrecondDataNpy(pre_data)                ((pre_data)->npy)
#define jx_CombinedPrecondDataNumEquns(pre_data)           ((pre_data)->num_equns)
#define jx_CombinedPrecondDataPreID(pre_data)              ((pre_data)->pre_id)
#define jx_CombinedPrecondDataThetaPsi(pre_data)           ((pre_data)->theta_psi)
#define jx_CombinedPrecondDataThetaRho(pre_data)           ((pre_data)->theta_rho)
#define jx_CombinedPrecondDataThetaPhi(pre_data)           ((pre_data)->theta_phi)
#define jx_CombinedPrecondDataActualPsi(pre_data)          ((pre_data)->actual_psi)
#define jx_CombinedPrecondDataActualRho(pre_data)          ((pre_data)->actual_rho)
#define jx_CombinedPrecondDataActualPhi(pre_data)          ((pre_data)->actual_phi)
#define jx_CombinedPrecondDataEuclidLevel(pre_data)        ((pre_data)->euclid_level)
#define jx_CombinedPrecondDataAMGMaxLevels(pre_data)       ((pre_data)->amg_max_levels)
#define jx_CombinedPrecondDataAMGRelaxType(pre_data)       ((pre_data)->amg_relax_type)
#define jx_CombinedPrecondDataAMGPrintLevel(pre_data)      ((pre_data)->amg_print_level)
#define jx_CombinedPrecondDataAMGInterpType(pre_data)      ((pre_data)->amg_interp_type)
#define jx_CombinedPrecondDataAMGPMaxElmts(pre_data)       ((pre_data)->amg_P_max_elmts)
#define jx_CombinedPrecondDataAMGMeasureType(pre_data)     ((pre_data)->amg_measure_type)
#define jx_CombinedPrecondDataAMGCoarsenType(pre_data)     ((pre_data)->amg_coarsen_type)
#define jx_CombinedPrecondDataAMGAggNumLevels(pre_data)    ((pre_data)->amg_agg_num_levels)
#define jx_CombinedPrecondDataAMGCoarseThreshold(pre_data) ((pre_data)->amg_coarse_threshold)
#define jx_CombinedPrecondDataDropTol(pre_data)            ((pre_data)->drop_tol)
#define jx_CombinedPrecondDataAdpTheta(pre_data)           ((pre_data)->adp_theta)
#define jx_CombinedPrecondDataThetaDis(pre_data)           ((pre_data)->theta_dis)
#define jx_CombinedPrecondDataAdpGammaT(pre_data)          ((pre_data)->adp_gamma_3)
#define jx_CombinedPrecondDataAdpGammaE(pre_data)          ((pre_data)->adp_gamma_11)
#define jx_CombinedPrecondDataAMGStrongThreshold(pre_data) ((pre_data)->amg_strong_threshold)
#define jx_CombinedPrecondDataComm(pre_data)               ((pre_data)->comm)
#define jx_CombinedPrecondDataPreMat(pre_data)             ((pre_data)->pre_mat)
#define jx_CombinedPrecondDataPreEucMat(pre_data)          ((pre_data)->pre_euc_mat)
#define jx_CombinedPrecondDataAuxVector(pre_data)          ((pre_data)->aux_vec)
#define jx_CombinedPrecondDataResVector(pre_data)          ((pre_data)->res_vec)
#define jx_CombinedPrecondDataILUZData(pre_data)           ((pre_data)->ilu_data)
#define jx_CombinedPrecondDataPAMGData(pre_data)           ((pre_data)->pamg_data)
#define jx_CombinedPrecondDataEuclidData(pre_data)         ((pre_data)->euclid_data)

/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/combined/amg_euclid.c */
JX_Int
JX_PAMGEuclidPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_PAMGEuclidPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_PAMGEuclidPAMGSetupB( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_PAMGEuclidPAMGSetupB( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_EuclidPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_EuclidPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_PAMGEuclidSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_PAMGEuclidSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_EuclidPAMGEuclidSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_EuclidPAMGEuclidSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_PAMGEuclidPAMGSolve( JX_Solver       solver,
                        JX_ParCSRMatrix par_matrix,
                        JX_ParVector    par_b,
                        JX_ParVector    par_x );
JX_Int
jx_PAMGEuclidPAMGSolve( void            *data,
                        jx_ParCSRMatrix *par_matrix,
                        jx_ParVector    *par_b,
                        jx_ParVector    *par_x );
JX_Int
JX_EuclidPAMGSolve( JX_Solver       solver,
                    JX_ParCSRMatrix par_matrix,
                    JX_ParVector    par_b,
                    JX_ParVector    par_x );
JX_Int
jx_EuclidPAMGSolve( void            *data,
                    jx_ParCSRMatrix *par_matrix,
                    jx_ParVector    *par_b,
                    jx_ParVector    *par_x );
JX_Int
JX_PAMGEuclidSolve( JX_Solver       solver,
                    JX_ParCSRMatrix par_matrix,
                    JX_ParVector    par_b,
                    JX_ParVector    par_x );
JX_Int
jx_PAMGEuclidSolve( void            *data,
                    jx_ParCSRMatrix *par_matrix,
                    jx_ParVector    *par_b,
                    jx_ParVector    *par_x );
JX_Int
JX_EuclidPAMGEuclidSolve( JX_Solver       solver,
                          JX_ParCSRMatrix par_matrix,
                          JX_ParVector    par_b,
                          JX_ParVector    par_x );
JX_Int
jx_EuclidPAMGEuclidSolve( void            *data,
                          jx_ParCSRMatrix *par_matrix,
                          jx_ParVector    *par_b,
                          jx_ParVector    *par_x );

/* csrc/combined/amg_ilu.c */
JX_Int
JX_ILUZPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
JX_ILUZPAMGILUZSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
JX_PAMGILUZPAMGSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_ILUZPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
jx_ILUZPAMGILUZSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
jx_PAMGILUZPAMGSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_ILUZPAMGSolve( JX_Solver       solver,
                  JX_ParCSRMatrix par_matrix,
                  JX_ParVector    par_b,
                  JX_ParVector    par_x );
JX_Int
jx_ILUZPAMGSolve( void            *data,
                  jx_ParCSRMatrix *par_matrix,
                  jx_ParVector    *par_b,
                  jx_ParVector    *par_x );
JX_Int
JX_ILUZPAMGILUZSolve( JX_Solver       solver,
                      JX_ParCSRMatrix par_matrix,
                      JX_ParVector    par_b,
                      JX_ParVector    par_x );
JX_Int
jx_ILUZPAMGILUZSolve( void            *data,
                      jx_ParCSRMatrix *par_matrix,
                      jx_ParVector    *par_b,
                      jx_ParVector    *par_x );
JX_Int
JX_PAMGILUZPAMGSolve( JX_Solver       solver,
                      JX_ParCSRMatrix par_matrix,
                      JX_ParVector    par_b,
                      JX_ParVector    par_x );
JX_Int
jx_PAMGILUZPAMGSolve( void            *data,
                      jx_ParCSRMatrix *par_matrix,
                      jx_ParVector    *par_b,
                      jx_ParVector    *par_x );

/* csrc/combined/combined.c */
JX_Int
JX_CombinedPrecondDataCreate( JX_Solver *solver, MPI_Comm comm );
void *
jx_CombinedPrecondDataCreate( MPI_Comm comm );
JX_Int
JX_CombinedPrecondDataInitialize( JX_Solver solver, JX_Int level );
JX_Int
jx_CombinedPrecondDataInitialize( void *data, JX_Int level );
JX_Int
JX_CombinedPrecondDataSetEuclidLevel( JX_Solver solver, JX_Int level );
JX_Int
JX_CombinedPrecondDataInitializeP( JX_Solver solver,
                                   JX_Int argc,
                                   char *argv[] );
JX_Int
jx_CombinedPrecondDataInitializeP( void *data,
                                   JX_Int argc,
                                   char *argv[] );
JX_Int
JX_CombinedPrecondDataSetEuclidParams( JX_Solver solver,
                                       JX_Int argc,
                                       char *argv[] );
JX_Int
JX_CombinedPrecondDataSetPreID( JX_Solver solver, JX_Int pre_id );
JX_Int
jx_CombinedPrecondDataSetPreID( void *data, JX_Int pre_id );
JX_Int
JX_CombinedPrecondDataSetNxyNpxyNequ( JX_Solver solver, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy, JX_Int num_equns );
JX_Int
jx_CombinedPrecondDataSetNxyNpxyNequ( void *data, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy, JX_Int num_equns );
JX_Int
JX_CombinedPrecondDataSetThetaPsiRhoPhi( JX_Solver solver,
                                         JX_Int theta_psi,
                                         JX_Int theta_rho,
                                         JX_Int theta_phi,
                                         JX_Real theta_dis );
JX_Int
jx_CombinedPrecondDataSetThetaPsiRhoPhi( void *data,
                                         JX_Int theta_psi,
                                         JX_Int theta_rho,
                                         JX_Int theta_phi,
                                         JX_Real theta_dis );
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
                                        JX_Real amg_strong_threshold );
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
                                        JX_Real amg_strong_threshold );
JX_Int
JX_CombinedPrecondDataSetPreMat( JX_Solver solver, jx_ParCSRMatrix *pre_mat );
JX_Int
jx_CombinedPrecondDataSetPreMat( void *data, jx_ParCSRMatrix *pre_mat );
JX_Int
JX_CombinedPrecondDataSetDropTol( JX_Solver solver, JX_Real drop_tol );
JX_Int
jx_CombinedPrecondDataSetDropTol( void *data, JX_Real drop_tol );
JX_Int
JX_CombinedPrecondDataSetInterpType( JX_Solver solver, JX_Int interp_type );
JX_Int
jx_CombinedPrecondDataSetInterpType( void *data, JX_Int interp_type );
JX_Int
JX_CombinedPrecondDataSetCoarsenType( JX_Solver solver, JX_Int coarsen_type );
JX_Int
jx_CombinedPrecondDataSetCoarsenType( void *data, JX_Int coarsen_type );
JX_Int
JX_CombinedPrecondDataSetCycleRelaxType( JX_Solver solver, JX_Int relax_type );
JX_Int
jx_CombinedPrecondDataSetCycleRelaxType( void *data, JX_Int relax_type );
JX_Int
JX_CombinedPrecondDataSetStrongThreshold( JX_Solver solver, JX_Real strong_threshold );
JX_Int
jx_CombinedPrecondDataSetStrongThreshold( void *data, JX_Real strong_threshold );
JX_Int
JX_CombinedPrecondDataSetAdpTheta( JX_Solver solver, JX_Real adp_theta );
JX_Int
jx_CombinedPrecondDataSetAdpTheta( void *data, JX_Real adp_theta );
JX_Int
JX_CombinedPrecondDataSetAdpGammaT( JX_Solver solver, JX_Real adp_gamma_3 );
JX_Int
jx_CombinedPrecondDataSetAdpGammaT( void *data, JX_Real adp_gamma_3 );
JX_Int
JX_CombinedPrecondDataSetAdpGammaE( JX_Solver solver, JX_Real adp_gamma_11 );
JX_Int
jx_CombinedPrecondDataSetAdpGammaE( void *data, JX_Real adp_gamma_11 );
JX_Int
JX_CombinedPrecondDataSetILUMatA( JX_Solver solver, jx_CSRMatrix *matA );
JX_Int
jx_CombinedPrecondDataSetILUMatA( void *data, jx_CSRMatrix *matA );
JX_Int
JX_CombinedPrecondDataSetPreEucMat( JX_Solver solver, jx_ParCSRMatrix *matA );
JX_Int
jx_CombinedPrecondDataSetPreEucMat( void *data, jx_ParCSRMatrix *matA );
JX_Int
JX_CombinedPrecondDataGetLULength( JX_Solver solver, JX_Int *lu_length );
JX_Int
jx_CombinedPrecondDataGetLULength( void *data, JX_Int *lu_length );
JX_Int
JX_CombinedPrecondDataSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_CombinedPrecondDataSetup( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_CombinedPrecondDataAdaptiveSetup2( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_CombinedPrecondDataAdaptiveSetup2( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_CombinedPrecondDataAdaptiveSetup3( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int
jx_CombinedPrecondDataAdaptiveSetup3( void *data, jx_ParCSRMatrix *par_matrix );
JX_Int
JX_CombinedPrecondDataSolve( JX_Solver       solver,
                             JX_ParCSRMatrix par_matrix,
                             JX_ParVector    par_b,
                             JX_ParVector    par_x );
JX_Int
jx_CombinedPrecondDataSolve( void            *data,
                             jx_ParCSRMatrix *par_matrix,
                             jx_ParVector    *par_b,
                             jx_ParVector    *par_x );
JX_Int
JX_CombinedPrecondDataAdaptiveSolve( JX_Solver       solver,
                                     JX_ParCSRMatrix par_matrix,
                                     JX_ParVector    par_b,
                                     JX_ParVector    par_x );
JX_Int
jx_CombinedPrecondDataAdaptiveSolve( void            *data,
                                     jx_ParCSRMatrix *par_matrix,
                                     jx_ParVector    *par_b,
                                     jx_ParVector    *par_x );
JX_Int
JX_CombinedPrecondDataDestroy( JX_Solver solver );
JX_Int
jx_CombinedPrecondDataDestroy( void *data );

/* csrc/combined/csr_matrix.c */
JX_Int
jx_CSRMatrixWeaklyDiagDominant( jx_CSRMatrix *A,
                                JX_Int num_equns,
                                JX_Real theta,
                                JX_Real gamma_3,
                                JX_Real gamma_11 );

/* csrc/combined/par_csr_matrix.c */
jx_ParCSRMatrix *
jx_BuildMatParFromOneFile4( char *filename, JX_Int file_base, JX_Int  *row_part, JX_Int  *col_part );

#endif
