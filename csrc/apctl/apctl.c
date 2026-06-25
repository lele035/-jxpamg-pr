//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  apctl.c -- Adaptive PCTL iterattion or preconditioner for 3t linear systems.
 *             PCTL, Physical-variable based Coarsening Two Level iterative Method,
 *             is proposed by Xu Xiaowen, Mo Zeyao etc.
 *  Date: 2011/09/17
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
 * \fn JX_Int JX_3tAPCTLSetXXXXX
 * \brief Set parameters for jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/17
 */ 
JX_Int 
JX_3tAPCTLSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetMaxiter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetMaxiter( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetNumRlxPre( JX_Solver solver, JX_Int num_relax )
{
   return( jx_3tAPCTLSetNumRlxPre( (void *) solver, num_relax ) );
}

JX_Int 
JX_3tAPCTLSetNumRlxPost( JX_Solver solver, JX_Int num_relax )
{
   return( jx_3tAPCTLSetNumRlxPost( (void *) solver, num_relax ) );
}

JX_Int 
JX_3tAPCTLSetPrintLevel( JX_Solver solver, JX_Int print_level )
{
   return( jx_3tAPCTLSetPrintLevel( (void *) solver, print_level ) );
}

JX_Int 
JX_3tAPCTLSetIsDiagElmFirst( JX_Solver solver, JX_Int is_diagelm_first )
{
   return( jx_3tAPCTLSetIsDiagElmFirst( (void *) solver, is_diagelm_first ) );
}

JX_Int 
JX_3tAPCTLSetNumGroup( JX_Solver solver, JX_Int num_group )
{
   return( jx_3tAPCTLSetNumGroup( (void *) solver, num_group ) );
}

JX_Int 
JX_3tAPCTLSetNumIterations( JX_Solver solver, JX_Int num_iterations )
{
   return( jx_3tAPCTLSetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int 
JX_3tAPCTLSetLastRelNrm( JX_Solver solver, JX_Real last_rel_nrm )
{
   return( jx_3tAPCTLSetLastRelNrm( (void *) solver, last_rel_nrm ) );
}

JX_Int 
JX_3tAPCTLSetAveConvFactor( JX_Solver solver, JX_Real ave_conv_factor )
{
   return( jx_3tAPCTLSetAveConvFactor( (void *) solver, ave_conv_factor ) );
}

JX_Int 
JX_3tAPCTLGetNumIterations( JX_Solver solver, JX_Int *num_iterations )
{
   return( jx_3tAPCTLGetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int 
JX_3tAPCTLGetLastRelNrm( JX_Solver solver, JX_Real *rel_resid_norm )
{
   return( jx_3tAPCTLGetLastRelNrm( (void *) solver, rel_resid_norm ) );
}

JX_Int 
JX_3tAPCTLSetNpR( JX_Solver solver, JX_Int np )
{
   return( jx_3tAPCTLSetNpR( (void *) solver, np ) );
}

JX_Int 
JX_3tAPCTLSetNpE( JX_Solver solver, JX_Int np )
{
   return( jx_3tAPCTLSetNpE( (void *) solver, np ) );
}

JX_Int 
JX_3tAPCTLSetNpI( JX_Solver solver, JX_Int np )
{
   return( jx_3tAPCTLSetNpI( (void *) solver, np ) );
}

JX_Int 
JX_3tAPCTLSetARRInterpMaxIt( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetARRInterpMaxIt( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetAIIInterpMaxIt( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetAIIInterpMaxIt( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetARRRelaxMaxIt( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetARRRelaxMaxIt( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetAEERelaxMaxIt( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetAEERelaxMaxIt( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetAIIRelaxMaxIt( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetAIIRelaxMaxIt( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetACCRelaxMaxIt( JX_Solver solver, JX_Int max_iter )
{
   return( jx_3tAPCTLSetACCRelaxMaxIt( (void *) solver, max_iter ) );
}

JX_Int 
JX_3tAPCTLSetARRInterpTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetARRInterpTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetAIIInterpTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetAIIInterpTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetARRRelaxTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetARRRelaxTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetAEERelaxTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetAEERelaxTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetAIIRelaxTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetAIIRelaxTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetACCRelaxTol( JX_Solver solver, JX_Real tol )
{
   return( jx_3tAPCTLSetACCRelaxTol( (void *) solver, tol ) );
}

JX_Int 
JX_3tAPCTLSetARRRelaxType( JX_Solver solver, JX_Int ARR_relax_type )
{
   return( jx_3tAPCTLSetARRRelaxType( (void *) solver, ARR_relax_type ) );
}

JX_Int 
JX_3tAPCTLSetAEERelaxType( JX_Solver solver, JX_Int AEE_relax_type )
{
   return( jx_3tAPCTLSetAEERelaxType( (void *) solver, AEE_relax_type ) );
}

JX_Int 
JX_3tAPCTLSetAIIRelaxType( JX_Solver solver, JX_Int AII_relax_type )
{
   return( jx_3tAPCTLSetAIIRelaxType( (void *) solver, AII_relax_type ) );
}

JX_Int 
JX_3tAPCTLSetThetaWCR( JX_Solver solver, JX_Real theta )
{
   return( jx_3tAPCTLSetThetaWCR( (void *) solver, theta ) );
} 

JX_Int 
JX_3tAPCTLSetThetaWCE( JX_Solver solver, JX_Real theta )
{
   return( jx_3tAPCTLSetThetaWCE( (void *) solver, theta ) );
} 
 
JX_Int 
JX_3tAPCTLSetThetaWCI( JX_Solver solver, JX_Real theta )
{
   return( jx_3tAPCTLSetThetaWCI( (void *) solver, theta ) );
}
  
JX_Int 
JX_3tAPCTLSetThresholdWCR( JX_Solver solver, JX_Real threshold )
{
   return( jx_3tAPCTLSetThresholdWCR( (void *) solver, threshold ) );
}
 
JX_Int 
JX_3tAPCTLSetThresholdWCE( JX_Solver solver, JX_Real threshold )
{
   return( jx_3tAPCTLSetThresholdWCE( (void *) solver, threshold ) );
}
 
JX_Int 
JX_3tAPCTLSetThresholdWCI( JX_Solver solver, JX_Real threshold )
{
   return( jx_3tAPCTLSetThresholdWCI( (void *) solver, threshold ) );
}
   
JX_Int 
JX_3tAPCTLSetISWCR( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetISWCR( (void *) solver, flag ) );
}

JX_Int 
JX_3tAPCTLSetISWCE( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetISWCE( (void *) solver, flag ) );
}  

JX_Int 
JX_3tAPCTLSetISWCI( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetISWCI( (void *) solver, flag ) );
}      

JX_Int 
JX_3tAPCTLSetThetaDDR( JX_Solver solver, JX_Real theta )
{
   return( jx_3tAPCTLSetThetaDDR( (void *) solver, theta ) );
} 

JX_Int 
JX_3tAPCTLSetThetaDDE( JX_Solver solver, JX_Real theta )
{
   return( jx_3tAPCTLSetThetaDDE( (void *) solver, theta ) );
} 
 
JX_Int 
JX_3tAPCTLSetThetaDDI( JX_Solver solver, JX_Real theta )
{
   return( jx_3tAPCTLSetThetaDDI( (void *) solver, theta ) );
} 
 
JX_Int 
JX_3tAPCTLSetThresholdDDR( JX_Solver solver, JX_Real threshold )
{
   return( jx_3tAPCTLSetThresholdDDR( (void *) solver, threshold ) );
}
 
JX_Int 
JX_3tAPCTLSetThresholdDDE( JX_Solver solver, JX_Real threshold )
{
   return( jx_3tAPCTLSetThresholdDDE( (void *) solver, threshold ) );
} 

JX_Int 
JX_3tAPCTLSetThresholdDDI( JX_Solver solver, JX_Real threshold )
{
   return( jx_3tAPCTLSetThresholdDDI( (void *) solver, threshold ) );
}
   
JX_Int 
JX_3tAPCTLSetISDDR( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetISDDR( (void *) solver, flag ) );
}

JX_Int 
JX_3tAPCTLSetISDDE( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetISDDE( (void *) solver, flag ) );
}  

JX_Int 
JX_3tAPCTLSetISDDI( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetISDDI( (void *) solver, flag ) );
}  
  
JX_Int 
JX_3tAPCTLSetNeedCC( JX_Solver solver, JX_Int flag )
{
   return( jx_3tAPCTLSetNeedCC( (void *) solver, flag ) );
}

JX_Int 
JX_3tAPCTLSetBlockSmoothType( JX_Solver solver, JX_Int blocksmooth_type )
{
   return( jx_3tAPCTLSetBlockSmoothType( (void *) solver, blocksmooth_type ) );
}
  
JX_Int 
JX_3tAPCTLSetFixItPCTLR( JX_Solver solver, JX_Int fixit )
{
   return( jx_3tAPCTLSetFixItPCTLR( (void *) solver, fixit ) );
}

JX_Int 
JX_3tAPCTLSetFixItPCTLE( JX_Solver solver, JX_Int fixit )
{
   return( jx_3tAPCTLSetFixItPCTLE( (void *) solver, fixit ) );
}

JX_Int 
JX_3tAPCTLSetFixItPCTLI( JX_Solver solver, JX_Int fixit )
{
   return( jx_3tAPCTLSetFixItPCTLI( (void *) solver, fixit ) );
}

JX_Int 
JX_3tAPCTLSetFixItBRLXR( JX_Solver solver, JX_Int fixit )
{
   return( jx_3tAPCTLSetFixItBRLXR( (void *) solver, fixit ) );
}

JX_Int 
JX_3tAPCTLSetFixItBRLXE( JX_Solver solver, JX_Int fixit )
{
   return( jx_3tAPCTLSetFixItBRLXE( (void *) solver, fixit ) );
}

JX_Int 
JX_3tAPCTLSetFixItBRLXI( JX_Solver solver, JX_Int fixit )
{
   return( jx_3tAPCTLSetFixItBRLXI( (void *) solver, fixit ) );
}

JX_Int 
JX_3tAPCTLSetUseFixedModeR( JX_Solver solver, JX_Int usefixedmode )
{
   return( jx_3tAPCTLSetUseFixedModeR( (void *) solver, usefixedmode ) );
}

JX_Int 
JX_3tAPCTLSetUseFixedModeE( JX_Solver solver, JX_Int usefixedmode )
{
   return( jx_3tAPCTLSetUseFixedModeE( (void *) solver, usefixedmode ) );
}   

JX_Int 
JX_3tAPCTLSetUseFixedModeI( JX_Solver solver, JX_Int usefixedmode )
{
   return( jx_3tAPCTLSetUseFixedModeI( (void *) solver, usefixedmode ) );
}

JX_Int 
JX_3tAPCTLSetUsePPCTL( JX_Solver solver, JX_Int use_ppctl )
{
   return( jx_3tAPCTLSetUsePPCTL( (void *) solver, use_ppctl ) );
}

JX_Int 
JX_3tAPCTLSetARRSolverID( JX_Solver solver, JX_Int solverid )
{
   return( jx_3tAPCTLSetARRSolverID( (void *) solver, solverid ) );
}

JX_Int 
JX_3tAPCTLSetACCSolverID( JX_Solver solver, JX_Int solverid )
{
   return( jx_3tAPCTLSetACCSolverID( (void *) solver, solverid ) );
}

JX_Int 
JX_3tAPCTLSetARRKDim( JX_Solver solver, JX_Int kdim )
{
   return( jx_3tAPCTLSetARRKDim( (void *) solver, kdim ) );
}

JX_Int 
JX_3tAPCTLSetACCKDim( JX_Solver solver, JX_Int kdim )
{
   return( jx_3tAPCTLSetACCKDim( (void *) solver, kdim ) );
}

JX_Int 
JX_3tAPCTLSetTestSubLSIter( JX_Solver solver, JX_Int test_subls_iter )
{
   return( jx_3tAPCTLSetTestSubLSIter( (void *) solver, test_subls_iter ) );
}

JX_Int 
JX_3tAPCTLSetDebugFlag( JX_Solver solver, JX_Int debug_flag )
{
   return( jx_3tAPCTLSetDebugFlag( (void *) solver, debug_flag ) );
}

JX_Int 
JX_3tAPCTLSetResetZero( JX_Solver solver, JX_Int reset_zero )
{
   return( jx_3tAPCTLSetResetZero( (void *) solver, reset_zero ) );
}

JX_Int 
JX_3tAPCTLSetInterpType( JX_Solver solver, JX_Int interp_type )
{
   return( jx_3tAPCTLSetInterpType( (void *) solver, interp_type ) );
}

JX_Int 
JX_3tAPCTLSetCoarsenType( JX_Solver solver, JX_Int coarsen_type )
{
   return( jx_3tAPCTLSetCoarsenType( (void *) solver, coarsen_type ) );
}

JX_Int 
JX_3tAPCTLSetAggNumLevels( JX_Solver solver, JX_Int agg_num_levels )
{
   return( jx_3tAPCTLSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

JX_Int 
JX_3tAPCTLSetCoarseThreshold( JX_Solver solver, JX_Int coarse_threshold )
{
   return( jx_3tAPCTLSetCoarseThreshold( (void *) solver, coarse_threshold ) );
}

JX_Int 
JX_3tAPCTLSetPrintLevelAMG( JX_Solver solver, JX_Int print_level_amg )
{
   return( jx_3tAPCTLSetPrintLevelAMG( (void *) solver, print_level_amg ) );
}

JX_Int 
JX_3tAPCTLSetStrongThreshold( JX_Solver solver, JX_Real strong_threshold )
{
   return( jx_3tAPCTLSetStrongThreshold( (void *) solver, strong_threshold ) );
}

JX_Int 
JX_3tAPCTLSetNumIterAiSetup( JX_Solver solver, JX_Int num_iter_Ai_pctl_setup )
{
   return( jx_3tAPCTLSetNumIterAiSetup( (void *) solver, num_iter_Ai_pctl_setup ) );
}

JX_Int 
JX_3tAPCTLSetNumIterArSetup( JX_Solver solver, JX_Int num_iter_Ar_pctl_setup )
{
   return( jx_3tAPCTLSetNumIterArSetup( (void *) solver, num_iter_Ar_pctl_setup ) );
}

JX_Int 
JX_3tAPCTLSetNumIterAePrecond( JX_Solver solver, JX_Int num_iter_Ae_pctl_precond )
{
   return( jx_3tAPCTLSetNumIterAePrecond( (void *) solver, num_iter_Ae_pctl_precond ) );
}

JX_Int 
JX_3tAPCTLSetNumIterAiPrecond( JX_Solver solver, JX_Int num_iter_Ai_pctl_precond )
{
   return( jx_3tAPCTLSetNumIterAiPrecond( (void *) solver, num_iter_Ai_pctl_precond ) );
}

JX_Int 
JX_3tAPCTLSetNumIterArPrecond( JX_Solver solver, JX_Int num_iter_Ar_pctl_precond )
{
   return( jx_3tAPCTLSetNumIterArPrecond( (void *) solver, num_iter_Ar_pctl_precond ) );
}

JX_Int 
JX_3tAPCTLSetNumIterAcPrecond( JX_Solver solver, JX_Int num_iter_Ac_pctl_precond )
{
   return( jx_3tAPCTLSetNumIterAcPrecond( (void *) solver, num_iter_Ac_pctl_precond ) );
}

JX_Int 
JX_3tAPCTLSetA( JX_Solver solver, jx_ParCSRMatrix *A )
{
   return( jx_3tAPCTLSetA( (void *) solver, A ) );
}

JX_Int 
JX_3tAPCTLSetSubBlocks( JX_Solver solver )
{
   return( jx_3tAPCTLSetSubBlocks( (void *) solver ) );
}

JX_Int 
JX_3tAPCTLSetARR( JX_Solver solver, jx_ParCSRMatrix *ARR )
{
   return( jx_3tAPCTLSetARR( (void *) solver, ARR ) );
}

JX_Int 
JX_3tAPCTLSetAEE( JX_Solver solver, jx_ParCSRMatrix *AEE )
{
   return( jx_3tAPCTLSetAEE( (void *) solver, AEE ) );
}

JX_Int 
JX_3tAPCTLSetAII( JX_Solver solver, jx_ParCSRMatrix *AII )
{
   return( jx_3tAPCTLSetAII( (void *) solver, AII ) );
}

JX_Int 
JX_3tAPCTLSetVRE( JX_Solver solver, jx_ParVector *VRE )
{
   return( jx_3tAPCTLSetVRE( (void *) solver, VRE ) );
}

JX_Int 
JX_3tAPCTLSetVER( JX_Solver solver, jx_ParVector *VER )
{
   return( jx_3tAPCTLSetVER( (void *) solver, VER ) );
}

JX_Int 
JX_3tAPCTLSetVER2( JX_Solver solver, jx_ParVector **VER )
{
   return( jx_3tAPCTLSetVER2( (void *) solver, VER ) );
}

JX_Int 
JX_3tAPCTLSetVEI( JX_Solver solver, jx_ParVector *VEI )
{
   return( jx_3tAPCTLSetVEI( (void *) solver, VEI ) );
}

JX_Int 
JX_3tAPCTLSetVIE( JX_Solver solver, jx_ParVector *VIE )
{
   return( jx_3tAPCTLSetVIE( (void *) solver, VIE ) );
}

JX_Int 
JX_3tAPCTLSetARRAll( JX_Solver solver, jx_ParCSRMatrix *ARR )
{
   return( jx_3tAPCTLSetARRAll( (void *) solver, ARR ) );
}

JX_Int 
JX_3tAPCTLSetARRAll2( JX_Solver solver, jx_ParCSRMatrix **ARR )
{
   return( jx_3tAPCTLSetARRAll2( (void *) solver, ARR ) );
}

JX_Int 
JX_3tAPCTLSetAEEAll( JX_Solver solver, jx_ParCSRMatrix *AEE )
{
   return( jx_3tAPCTLSetAEEAll( (void *) solver, AEE ) );
}

JX_Int 
JX_3tAPCTLSetAIIAll( JX_Solver solver, jx_ParCSRMatrix *AII )
{
   return( jx_3tAPCTLSetAIIAll( (void *) solver, AII ) );
}

JX_Int 
JX_3tAPCTLSetVREAll( JX_Solver solver, jx_ParVector *VRE )
{
   return( jx_3tAPCTLSetVREAll( (void *) solver, VRE ) );
}

JX_Int 
JX_3tAPCTLSetVREAll2( JX_Solver solver, jx_ParVector **VRE )
{
   return( jx_3tAPCTLSetVREAll2( (void *) solver, VRE ) );
}

JX_Int 
JX_3tAPCTLSetVERAll( JX_Solver solver, jx_ParVector *VER )
{
   return( jx_3tAPCTLSetVERAll( (void *) solver, VER ) );
}

JX_Int 
JX_3tAPCTLSetVERAll2( JX_Solver solver, jx_ParVector **VER )
{
   return( jx_3tAPCTLSetVERAll2( (void *) solver, VER ) );
}

JX_Int 
JX_3tAPCTLSetVEIAll( JX_Solver solver, jx_ParVector *VEI )
{
   return( jx_3tAPCTLSetVEIAll( (void *) solver, VEI ) );
}

JX_Int 
JX_3tAPCTLSetVIEAll( JX_Solver solver, jx_ParVector *VIE )
{
   return( jx_3tAPCTLSetVIEAll( (void *) solver, VIE ) );
}

JX_Int
JX_3tAPCTLIterCount( JX_Solver solver )
{
   return( jx_3tAPCTLIterCount( (void *) solver) );
}

JX_Int 
JX_3tAPCTLSetCommX( JX_Solver solver, MPI_Comm comm )
{
   return( jx_3tAPCTLSetCommX( (void *) solver, comm ) );
}

JX_Int 
JX_3tAPCTLSetComm( JX_Solver solver, MPI_Comm comm )
{
   return( jx_3tAPCTLSetComm( (void *) solver, comm ) );
}

JX_Int 
JX_3tAPCTLSetCommY( JX_Solver solver, MPI_Comm comm )
{
   return( jx_3tAPCTLSetCommY( (void *) solver, comm ) );
}

JX_Int 
JX_3tAPCTLSetGroupIdX( JX_Solver solver, JX_Int groupid )
{
   return( jx_3tAPCTLSetGroupIdX( (void *) solver, groupid ) );
}

JX_Int 
JX_3tAPCTLSetGroupIdY( JX_Solver solver, JX_Int groupid )
{
   return( jx_3tAPCTLSetGroupIdY( (void *) solver, groupid ) );
}

/*!
 * \fn JX_Int JX_3tAPCTLDataInitialize
 * \brief Initialize a 3tAPCTLData object.
 * \author peghoty
 * \date 2011/09/17
 */
JX_Int
JX_3tAPCTLDataInitialize( JX_Solver *solver )
{
   *solver = (JX_Solver) jx_3tAPCTLDataInitialize( );
   if (!solver)
   {
      jx_error_in_arg(1);
   }
   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_3tAPCTLDestroy
 * \brief Destroy a 3tAPCTLData object.
 * \author peghoty
 * \date 2011/09/17
 */
JX_Int 
JX_3tAPCTLDestroy( JX_Solver solver )
{
   return( jx_3tAPCTLDestroy( (void *) solver ) );
}

/*!
 * \fn JX_Int jx_3tAPCTLSetXXXXXX
 * \brief Set parameters for PCTL.
 * \author peghoty
 * \date 2011/09/17
 */
JX_Int 
jx_3tAPCTLSetTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetMaxiter( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataMaxIter(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumRlxPre( void *solver, JX_Int num_relax )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumRlxPre(pre_3tapctl_data) = num_relax;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumRlxPost( void *solver, JX_Int num_relax )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumRlxPost(pre_3tapctl_data) = num_relax;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetPrintLevel( void *solver, JX_Int print_level )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataPrintLevel(pre_3tapctl_data) = print_level;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetIsDiagElmFirst( void *solver, JX_Int is_diagelm_first )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataIsDiagElmFirst(pre_3tapctl_data) = is_diagelm_first;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumGroup( void *solver, JX_Int num_group )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_3tAPCTLDataNumGroup(pre_3tapctl_data) = num_group;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumIterations( void *solver, JX_Int num_iterations )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterations(pre_3tapctl_data) = num_iterations;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetLastRelNrm( void *solver, JX_Real last_rel_nrm )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataLastRelNrm(pre_3tapctl_data) = last_rel_nrm;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAveConvFactor( void *solver, JX_Real ave_conv_factor )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAveConvFactor(pre_3tapctl_data) = ave_conv_factor;

   return jx_error_flag;
}

JX_Int
jx_3tAPCTLGetNumIterations( void *solver, JX_Int *num_iterations )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *num_iterations = jx_3tAPCTLDataNumIterations(pre_3tapctl_data);

   return jx_error_flag;
} 

JX_Int  
jx_3tAPCTLGetLastRelNrm( void *solver, JX_Real *rel_resid_norm ) 
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *rel_resid_norm = jx_3tAPCTLDataLastRelNrm(pre_3tapctl_data);

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNpR( void *solver, JX_Int np )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNpR(pre_3tapctl_data) = np;

   return jx_error_flag;
}


JX_Int 
jx_3tAPCTLSetNpE( void *solver, JX_Int np )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNpE(pre_3tapctl_data) = np;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNpI( void *solver, JX_Int np )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNpI(pre_3tapctl_data) = np;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRInterpMaxIt( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAIIInterpMaxIt( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRRelaxMaxIt( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAEERelaxMaxIt( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAIIRelaxMaxIt( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetACCRelaxMaxIt( void *solver, JX_Int max_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data) = max_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetMaxItDefault( void *solver, JX_Int maxit_default )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataMaxItDefault(pre_3tapctl_data) = maxit_default;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRInterpTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAIIInterpTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRRelaxTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAEERelaxTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAIIRelaxTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetACCRelaxTol( void *solver, JX_Real tol )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data) = tol;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRRelaxType( void *solver, JX_Int ARR_relax_type )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAEERelaxType( void *solver, JX_Int AEE_relax_type )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAIIRelaxType( void *solver, JX_Int AII_relax_type )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetTolDefault( void *solver, JX_Real tol_default )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataTolDefault(pre_3tapctl_data) = tol_default;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThetaWCR( void *solver, JX_Real theta )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThetaWCR(pre_3tapctl_data) = theta;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThetaWCE( void *solver, JX_Real theta )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThetaWCE(pre_3tapctl_data) = theta;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThetaWCI( void *solver, JX_Real theta )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThetaWCI(pre_3tapctl_data) = theta;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThresholdWCR( void *solver, JX_Real threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThresholdWCR(pre_3tapctl_data) = threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThresholdWCE( void *solver, JX_Real threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data) = threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThresholdWCI( void *solver, JX_Real threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThresholdWCI(pre_3tapctl_data) = threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetISWCR( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataISWCR(pre_3tapctl_data) = flag;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetISWCE( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataISWCE(pre_3tapctl_data) = flag;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetISWCI( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataISWCI(pre_3tapctl_data) = flag;

   return jx_error_flag;
}
 
JX_Int 
jx_3tAPCTLSetThetaDDR( void *solver, JX_Real theta )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThetaDDR(pre_3tapctl_data) = theta;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThetaDDE( void *solver, JX_Real theta )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThetaDDE(pre_3tapctl_data) = theta;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThetaDDI( void *solver, JX_Real theta )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThetaDDI(pre_3tapctl_data) = theta;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThresholdDDR( void *solver, JX_Real threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data) = threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThresholdDDE( void *solver, JX_Real threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data) = threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetThresholdDDI( void *solver, JX_Real threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data) = threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetISDDR( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataISDDR(pre_3tapctl_data) = flag;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetISDDE( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataISDDE(pre_3tapctl_data) = flag;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetISDDI( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataISDDI(pre_3tapctl_data) = flag;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNeedCC( void *solver, JX_Int flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNeedCC(pre_3tapctl_data) = flag;

   return jx_error_flag;
} 


JX_Int 
jx_3tAPCTLSetBlockSmoothType( void *solver, JX_Int blocksmooth_type )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data) = blocksmooth_type;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetFixItPCTLR( void *solver, JX_Int fixit )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data) = fixit;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetFixItPCTLE( void *solver, JX_Int fixit )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data) = fixit;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetFixItPCTLI( void *solver, JX_Int fixit )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data) = fixit;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetFixItBRLXR( void *solver, JX_Int fixit )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data) = fixit;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetFixItBRLXE( void *solver, JX_Int fixit )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data) = fixit;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetFixItBRLXI( void *solver, JX_Int fixit )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data) = fixit;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetUseFixedModeR( void *solver, JX_Int usefixedmode )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = usefixedmode;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetUseFixedModeE( void *solver, JX_Int usefixedmode )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = usefixedmode;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetUseFixedModeI( void *solver, JX_Int usefixedmode )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = usefixedmode;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetUsePPCTL( void *solver, JX_Int use_ppctl )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataUsePPCTL(pre_3tapctl_data) = use_ppctl;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRSolverID( void *solver, JX_Int solverid )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRSolverID(pre_3tapctl_data) = solverid;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetACCSolverID( void *solver, JX_Int solverid )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataACCSolverID(pre_3tapctl_data) = solverid;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARRKDim( void *solver, JX_Int kdim )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRKDim(pre_3tapctl_data) = kdim;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetACCKDim( void *solver, JX_Int kdim )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataACCKDim(pre_3tapctl_data) = kdim;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetTestSubLSIter( void *solver, JX_Int test_subls_iter )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data) = test_subls_iter;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetDebugFlag( void *solver, JX_Int debug_flag )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataDebugFlag(pre_3tapctl_data) = debug_flag;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetResetZero( void *solver, JX_Int reset_zero )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataResetZero(pre_3tapctl_data) = reset_zero;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetInterpType( void *solver, JX_Int interp_type )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataInterpType(pre_3tapctl_data) = interp_type;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetCoarsenType( void *solver, JX_Int coarsen_type )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataCoarsenType(pre_3tapctl_data) = coarsen_type;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAggNumLevels( void *solver, JX_Int agg_num_levels )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAggNumLevels(pre_3tapctl_data) = agg_num_levels;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetCoarseThreshold( void *solver, JX_Int coarse_threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataCoarseThreshold(pre_3tapctl_data) = coarse_threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetPrintLevelAMG( void *solver, JX_Int print_level_amg )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data) = print_level_amg;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetStrongThreshold( void *solver, JX_Real strong_threshold )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataStrongThreshold(pre_3tapctl_data) = strong_threshold;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumIterAiSetup( void *solver, JX_Int num_iter_Ai_pctl_setup )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = num_iter_Ai_pctl_setup;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetNumIterArSetup( void *solver, JX_Int num_iter_Ar_pctl_setup )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = num_iter_Ar_pctl_setup;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumIterAePrecond( void *solver, JX_Int num_iter_Ae_pctl_precond )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) = num_iter_Ae_pctl_precond;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumIterAiPrecond( void *solver, JX_Int num_iter_Ai_pctl_precond )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) = num_iter_Ai_pctl_precond;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetNumIterArPrecond( void *solver, JX_Int num_iter_Ar_pctl_precond )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) = num_iter_Ar_pctl_precond;

   return jx_error_flag;
}                

JX_Int 
jx_3tAPCTLSetNumIterAcPrecond( void *solver, JX_Int num_iter_Ac_pctl_precond )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) = num_iter_Ac_pctl_precond;

   return jx_error_flag;
}  

JX_Int 
jx_3tAPCTLSetA( void *solver, jx_ParCSRMatrix *A )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataA(pre_3tapctl_data) = A;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetSubBlocks( void *solver )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;
   jx_ParCSRMatrix *A = jx_3tAPCTLDataA(pre_3tapctl_data);
   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   MPI_Comm comm_x, comm_y;
   JX_Int myid, nprocs, np_R;

   JX_Int groupid_x = MPI_UNDEFINED;
   JX_Int groupid_y = MPI_UNDEFINED;

   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL;
   jx_ParVector     *VRE = NULL;
   jx_ParVector    **VER = NULL;
   jx_ParVector     *VEI = NULL;
   jx_ParVector     *VIE = NULL;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (!jx_3tAPCTLDataIsDiagElmFirst(pre_3tapctl_data))
   {
      jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(A));
   }
   np_R = nprocs / (ng + 2);
   groupid_x = myid / np_R;
   jx_MPI_Comm_split(comm, groupid_x, myid, &comm_x);
   groupid_y = myid % np_R;
   jx_MPI_Comm_split(comm, groupid_y, myid, &comm_y);
   jx_mgGenerateSubBlocks(comm, comm_x, groupid_x, ng, A, &ARR, &AEE, &AII, &VRE, &VER, &VEI, &VIE);
   jx_3tAPCTLSetARR(solver, ARR);
   jx_3tAPCTLSetAEE(solver, AEE);
   jx_3tAPCTLSetAII(solver, AII);
   jx_3tAPCTLSetVRE(solver, VRE);
   jx_3tAPCTLSetVER2(solver, VER);
   jx_3tAPCTLSetVEI(solver, VEI);
   jx_3tAPCTLSetVIE(solver, VIE);
   jx_3tAPCTLSetNpR(solver, np_R);
   jx_3tAPCTLSetComm(solver, comm);
   jx_3tAPCTLSetCommX(solver, comm_x);
   jx_3tAPCTLSetCommY(solver, comm_y);
   jx_3tAPCTLSetGroupIdX(solver, groupid_x);

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetARR( void *solver, jx_ParCSRMatrix *ARR )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARR(pre_3tapctl_data) = ARR;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetAEE( void *solver, jx_ParCSRMatrix *AEE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAEE(pre_3tapctl_data) = AEE;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAII( void *solver, jx_ParCSRMatrix *AII )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAII(pre_3tapctl_data) = AII;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVRE( void *solver, jx_ParVector *VRE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVRE(pre_3tapctl_data) = VRE;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVER( void *solver, jx_ParVector *VER )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVER(pre_3tapctl_data) = VER;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVER2( void *solver, jx_ParVector **VER )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_3tAPCTLDataVER2(pre_3tapctl_data) = VER;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVEI( void *solver, jx_ParVector *VEI )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVEI(pre_3tapctl_data) = VEI;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVIE( void *solver, jx_ParVector *VIE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVIE(pre_3tapctl_data) = VIE;

   return jx_error_flag;
}                

JX_Int 
jx_3tAPCTLSetARRAll( void *solver, jx_ParCSRMatrix *ARR )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataARRAll(pre_3tapctl_data) = ARR;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetARRAll2( void *solver, jx_ParCSRMatrix **ARR )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_3tAPCTLDataARRAll2(pre_3tapctl_data) = ARR;

   return jx_error_flag;
} 

JX_Int 
jx_3tAPCTLSetAEEAll( void *solver, jx_ParCSRMatrix *AEE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAEEAll(pre_3tapctl_data) = AEE;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetAIIAll( void *solver, jx_ParCSRMatrix *AII )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataAIIAll(pre_3tapctl_data) = AII;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVREAll( void *solver, jx_ParVector *VRE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVREAll(pre_3tapctl_data) = VRE;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVREAll2( void *solver, jx_ParVector **VRE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVREAll2(pre_3tapctl_data) = VRE;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVERAll( void *solver, jx_ParVector *VER )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVERAll(pre_3tapctl_data) = VER;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVERAll2( void *solver, jx_ParVector **VER )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_3tAPCTLDataVERAll2(pre_3tapctl_data) = VER;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVEIAll( void *solver, jx_ParVector *VEI )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVEIAll(pre_3tapctl_data) = VEI;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetVIEAll( void *solver, jx_ParVector *VIE )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataVIEAll(pre_3tapctl_data) = VIE;

   return jx_error_flag;
}                

JX_Int
jx_3tAPCTLIterCount( void *solver )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jx_3tAPCTLIterCount_sp(pre_3tapctl_data);
   }
   else if (nprocs > 1)
   {
      jx_3tAPCTLIterCount_mp(pre_3tapctl_data);
   }
   
   return 0;
}

JX_Int 
jx_3tAPCTLSetComm( void *solver, MPI_Comm comm )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataComm(pre_3tapctl_data) = comm;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetCommX( void *solver, MPI_Comm comm )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataCommX(pre_3tapctl_data) = comm;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetCommY( void *solver, MPI_Comm comm )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataCommY(pre_3tapctl_data) = comm;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetGroupIdX( void *solver, JX_Int groupid )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataGroupIdX(pre_3tapctl_data) = groupid;

   return jx_error_flag;
}

JX_Int 
jx_3tAPCTLSetGroupIdY( void *solver, JX_Int groupid )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;

   if (!pre_3tapctl_data)
   {
      jx_printf("Warning! 3tAPCTL object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_3tAPCTLDataGroupIdY(pre_3tapctl_data) = groupid;

   return jx_error_flag;
}   
         
/*!
 * \fn void *jx_3tAPCTLDataInitialize
 * \brief Create a jx_3tAPCTLData object.
 * \author peghoty 
 * \date 2011/09/17
 */
void *
jx_3tAPCTLDataInitialize()
{
   jx_3tAPCTLData *pre_3tapctl_data = NULL;  
   
   JX_Real   tol;               // tolerance    
   JX_Int      max_iter;          // maximal number of iteration
   JX_Int      num_relax_pre;     // number of pre-relaxations 
   JX_Int      num_relax_post;    // number of post-relaxations     
   JX_Int      print_level;       // 0: switch off; 1: print time; 2: print number of iter; 3: both 1 and 2 

   JX_Int      num_group;

   JX_Int      reset_zero;

   JX_Int      is_diagelm_first;

   /* parameters to describe the weaking coupling */
   JX_Real theta_wc_R;
   JX_Real theta_wc_E;
   JX_Real theta_wc_I;
   JX_Real threshold_wc_R;
   JX_Real threshold_wc_E;
   JX_Real threshold_wc_I;

   /* parameters to describe the diagonal dominance */
   JX_Real theta_dd_R;
   JX_Real theta_dd_E;
   JX_Real theta_dd_I;
   JX_Real threshold_dd_R;
   JX_Real threshold_dd_E;
   JX_Real threshold_dd_I;
   
   /* type of the block smoothing */
   JX_Int    blocksmooth_type; 
   
   /* maximal number of iterations and tolerance for submatrices */ 
   JX_Int    maxit_default;
   JX_Real tol_default;
   
   /* fixed number of iterations and tolerance for submatrices */ 
   JX_Int    fixit_pctl_R;
   JX_Int    fixit_pctl_E;
   JX_Int    fixit_pctl_I;
   JX_Int    fixit_brlx_R;
   JX_Int    fixit_brlx_E;
   JX_Int    fixit_brlx_I;

   /* whether employ the fixed-number-of-iterations mode? peghoty, 2012/02/15 */
   JX_Int    use_fixedmode_R;
   JX_Int    use_fixedmode_E;
   JX_Int    use_fixedmode_I;
   
   /* Whether use the pure PCTL? */
   JX_Int    use_ppctl;    
   
   /* solver type for interpolation-building of PRR */
   JX_Int    ARR_solver_id; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;

   /* solver type for PCTL iteration */
   JX_Int    ACC_solver_id; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   
   /* restart parameters for GMRES solver */
   JX_Int    ARR_kdim;
   JX_Int    ACC_kdim;
   
   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JX_Int    test_subls_iter;

   /* variables to count the number of iteration */
   JX_Int    num_iter_Ai_pctl_setup;
   JX_Int    num_iter_Ar_pctl_setup;
   JX_Int    num_iter_Ae_pctl_precond;
   JX_Int    num_iter_Ai_pctl_precond;
   JX_Int    num_iter_Ar_pctl_precond;
   JX_Int    num_iter_Ac_pctl_precond;                 

   JX_Int    debug_flag;

   /* set some parameters by default */
   tol               = 1.0e-6;            
   max_iter          = 100; 
   num_relax_pre     = 1;
   num_relax_post    = 0;                   
   print_level       = 0;    

   num_group         = 1;

   theta_wc_R        = 1.0e-4;
   theta_wc_E        = 1.0e-4;
   theta_wc_I        = 1.0e-4;
   threshold_wc_R    = 0.5;
   threshold_wc_E    = 0.5;
   threshold_wc_I    = 0.5;
   
   theta_dd_R        = 0.9;
   theta_dd_E        = 0.9;
   theta_dd_I        = 0.9;
   threshold_dd_R    = 1.0;
   threshold_dd_E    = 1.0;
   threshold_dd_I    = 1.0; 
   
   blocksmooth_type  = BLOCKSMOOTH_BD; 
   
   maxit_default     = 200;
   tol_default       = 1.0e-6;
   
   fixit_pctl_R      = 1;
   fixit_pctl_E      = 1;
   fixit_pctl_I      = 1;
   fixit_brlx_R      = 3;
   fixit_brlx_E      = 1;
   fixit_brlx_I      = 1;
   
   use_fixedmode_R   = 1;
   use_fixedmode_E   = 1;
   use_fixedmode_I   = 1; 
   
   use_ppctl         = 0;     
   
   ARR_solver_id     = SOLVER_AMGGMRES;  // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   ACC_solver_id     = SOLVER_AMGGMRES;  // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   ARR_kdim          = 20;
   ACC_kdim          = 20;
   
   test_subls_iter   = 0;

   num_iter_Ai_pctl_setup   = 0;
   num_iter_Ar_pctl_setup   = 0;
   num_iter_Ae_pctl_precond = 0;
   num_iter_Ai_pctl_precond = 0;
   num_iter_Ar_pctl_precond = 0;
   num_iter_Ac_pctl_precond = 0;   

   debug_flag = 0;

   reset_zero = 1;

   is_diagelm_first = 0;

   /* create a 3tAPCTLData object */
   pre_3tapctl_data = jx_CTAlloc(jx_3tAPCTLData, 1);
   
   /* set some parameters by default */
   jx_3tAPCTLSetTol(pre_3tapctl_data, tol);
   jx_3tAPCTLSetMaxiter(pre_3tapctl_data, max_iter);       
   jx_3tAPCTLSetNumRlxPre(pre_3tapctl_data, num_relax_pre);
   jx_3tAPCTLSetNumRlxPost(pre_3tapctl_data, num_relax_post);
   jx_3tAPCTLSetPrintLevel(pre_3tapctl_data, print_level);
   
   jx_3tAPCTLSetNumGroup(pre_3tapctl_data, num_group);

   /* max_iter for each subblock */
   jx_3tAPCTLSetARRInterpMaxIt(pre_3tapctl_data, maxit_default);
   jx_3tAPCTLSetAIIInterpMaxIt(pre_3tapctl_data, maxit_default);
   jx_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, maxit_default);
   jx_3tAPCTLSetMaxItDefault(pre_3tapctl_data, maxit_default);

   /* tolerance for each subblock */
   jx_3tAPCTLSetARRInterpTol(pre_3tapctl_data, tol_default);
   jx_3tAPCTLSetAIIInterpTol(pre_3tapctl_data, tol_default);
   jx_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
   jx_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
   jx_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
   jx_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);
   jx_3tAPCTLSetTolDefault(pre_3tapctl_data, tol_default);
   
   jx_3tAPCTLSetThetaWCR(pre_3tapctl_data, theta_wc_R);
   jx_3tAPCTLSetThetaWCE(pre_3tapctl_data, theta_wc_E);
   jx_3tAPCTLSetThetaWCI(pre_3tapctl_data, theta_wc_I);
   jx_3tAPCTLSetThresholdWCR(pre_3tapctl_data, threshold_wc_R);
   jx_3tAPCTLSetThresholdWCE(pre_3tapctl_data, threshold_wc_E);
   jx_3tAPCTLSetThresholdWCI(pre_3tapctl_data, threshold_wc_I);
   jx_3tAPCTLSetThetaDDR(pre_3tapctl_data, theta_dd_R);
   jx_3tAPCTLSetThetaDDE(pre_3tapctl_data, theta_dd_E);
   jx_3tAPCTLSetThetaDDI(pre_3tapctl_data, theta_dd_I);
   jx_3tAPCTLSetThresholdDDR(pre_3tapctl_data, threshold_dd_R);
   jx_3tAPCTLSetThresholdDDE(pre_3tapctl_data, threshold_dd_E);
   jx_3tAPCTLSetThresholdDDI(pre_3tapctl_data, threshold_dd_I);
   
   jx_3tAPCTLSetBlockSmoothType(pre_3tapctl_data, blocksmooth_type);

   jx_3tAPCTLSetFixItPCTLR(pre_3tapctl_data, fixit_pctl_R);
   jx_3tAPCTLSetFixItPCTLE(pre_3tapctl_data, fixit_pctl_E);
   jx_3tAPCTLSetFixItPCTLI(pre_3tapctl_data, fixit_pctl_I);
   jx_3tAPCTLSetFixItBRLXR(pre_3tapctl_data, fixit_brlx_R);
   jx_3tAPCTLSetFixItBRLXE(pre_3tapctl_data, fixit_brlx_E);
   jx_3tAPCTLSetFixItBRLXI(pre_3tapctl_data, fixit_brlx_I);
   
   jx_3tAPCTLSetUseFixedModeR(pre_3tapctl_data, use_fixedmode_R);
   jx_3tAPCTLSetUseFixedModeE(pre_3tapctl_data, use_fixedmode_E);
   jx_3tAPCTLSetUseFixedModeI(pre_3tapctl_data, use_fixedmode_I);
   
   jx_3tAPCTLSetUsePPCTL(pre_3tapctl_data, use_ppctl); // peghoty, 2012/03/06
   
   jx_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);
   jx_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id);
   jx_3tAPCTLSetARRKDim(pre_3tapctl_data, ARR_kdim);
   jx_3tAPCTLSetACCKDim(pre_3tapctl_data, ACC_kdim);
   
   jx_3tAPCTLSetTestSubLSIter(pre_3tapctl_data, test_subls_iter);  // peghoty, 2012/02/23           
   jx_3tAPCTLSetNumIterAiSetup(pre_3tapctl_data, num_iter_Ai_pctl_setup); // peghoty, 2012/02/23
   jx_3tAPCTLSetNumIterArSetup(pre_3tapctl_data, num_iter_Ar_pctl_setup); // peghoty, 2012/02/23
   jx_3tAPCTLSetNumIterAePrecond(pre_3tapctl_data, num_iter_Ae_pctl_precond); // peghoty, 2012/02/23
   jx_3tAPCTLSetNumIterAiPrecond(pre_3tapctl_data, num_iter_Ai_pctl_precond); // peghoty, 2012/02/23
   jx_3tAPCTLSetNumIterArPrecond(pre_3tapctl_data, num_iter_Ar_pctl_precond); // peghoty, 2012/02/23
   jx_3tAPCTLSetNumIterAcPrecond(pre_3tapctl_data, num_iter_Ac_pctl_precond); // peghoty, 2012/02/23

   jx_3tAPCTLSetDebugFlag(pre_3tapctl_data, debug_flag);
   jx_3tAPCTLSetResetZero(pre_3tapctl_data, reset_zero);
   jx_3tAPCTLSetIsDiagElmFirst(pre_3tapctl_data, is_diagelm_first);

   /* initialize some members */
   jx_3tAPCTLDataA(pre_3tapctl_data)              = NULL;
   jx_3tAPCTLDataARR(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataAEE(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataAII(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataVRE(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataVER(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataVER2(pre_3tapctl_data)           = NULL;
   jx_3tAPCTLDataVEI(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataVIE(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataARRAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataARRAll2(pre_3tapctl_data)        = NULL;
   jx_3tAPCTLDataAEEAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataAIIAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataVREAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataVREAll2(pre_3tapctl_data)        = NULL;
   jx_3tAPCTLDataVERAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataVERAll2(pre_3tapctl_data)        = NULL;
   jx_3tAPCTLDataVEIAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataVIEAll(pre_3tapctl_data)         = NULL;
   jx_3tAPCTLDataP(pre_3tapctl_data)              = NULL;
   jx_3tAPCTLDataPRR(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataPII(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataACC(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = NULL;
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = NULL;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = NULL;
   jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = NULL;
   jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = NULL;
   jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = NULL;
   jx_3tAPCTLDataWRR(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataWEE(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataWII(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataWCC(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataGCC(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataRES(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataRHS(pre_3tapctl_data)            = NULL;
   jx_3tAPCTLDataJAC(pre_3tapctl_data)            = NULL;

   return pre_3tapctl_data;
} 

/*!
 * \fn JX_Int jx_3tAPCTLDestroy
 * \brief Destroy the jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLDestroy( void *vdata )
{
   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jx_3tAPCTLDestroy_sp(pre_3tapctl_data);
   }
   else if (nprocs > 1)
   {
      jx_3tAPCTLDestroy_mp(pre_3tapctl_data);
   }
   
   return 0;
}

/*!
 * \fn JX_Int jx_3tAPCTLDestroy_sp
 * \brief Destroy the jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/27
 */
JX_Int
jx_3tAPCTLDestroy_sp( void *vdata )
{
   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   
   if (pre_3tapctl_data)
   {  
 
      // ARR
      if ( jx_3tAPCTLDataARR(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataARR(pre_3tapctl_data) );
      }
      
      // VRE
      if ( jx_3tAPCTLDataVRE(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataVRE(pre_3tapctl_data) );
      }  
                      
      // ARR_amg_solver
      if ( jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) );
      }  
           
      // ARR_gmres_solver  peghoty, 2011/10/29
      if ( jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) )
      {
         jx_GMRESDestroy(jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data));
      } 
         
      // AEE
      if ( jx_3tAPCTLDataAEE(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataAEE(pre_3tapctl_data) );
      }
      
      // VER
      if ( jx_3tAPCTLDataVER(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataVER(pre_3tapctl_data) );
      }  
             
      // VEI
      if ( jx_3tAPCTLDataVEI(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataVEI(pre_3tapctl_data) );
      } 
      
      // AEE_amg_solver
      if ( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
      }        
   
      // AII
      if ( jx_3tAPCTLDataAII(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataAII(pre_3tapctl_data) );
      }
      
      // VIE
      if ( jx_3tAPCTLDataVIE(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataVIE(pre_3tapctl_data) );
      }
      
      // AII_amg_solver
      if ( jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) );
      }       

      // RHS
      if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
      } 
       
      // WRR
      if ( jx_3tAPCTLDataWRR(pre_3tapctl_data) )
      {
         jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWRR(pre_3tapctl_data)) );
         jx_ParVectorDestroy( jx_3tAPCTLDataWRR(pre_3tapctl_data) );
      }
            
      // WEE
      if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
      {
         jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWEE(pre_3tapctl_data)) );
         jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
      }
      
      // WII
      if ( jx_3tAPCTLDataWII(pre_3tapctl_data) )
      {
         jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWII(pre_3tapctl_data)) );
         jx_ParVectorDestroy( jx_3tAPCTLDataWII(pre_3tapctl_data) );
      } 

      // ACC
      if ( jx_3tAPCTLDataACC(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataACC(pre_3tapctl_data) );
      }
      
      // PRR
      if ( jx_3tAPCTLDataPRR(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataPRR(pre_3tapctl_data) );
      }
      
      // PII
      if ( jx_3tAPCTLDataPII(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataPII(pre_3tapctl_data) );
      }   
                
      // ACC_amg_solver
      if ( jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) );
      } 

      // ACC_gmres_solver  peghoty, 2011/10/29
      if ( jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) )
      {
         jx_GMRESDestroy(jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data));
      }  
      
      // WCC
      if ( jx_3tAPCTLDataWCC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataWCC(pre_3tapctl_data) );
      }    
        
      // RES
      if ( jx_3tAPCTLDataRES(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataRES(pre_3tapctl_data) );
      }       
   
      // JAC
      if ( jx_3tAPCTLDataJAC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataJAC(pre_3tapctl_data) );
      }
          
      jx_TFree(pre_3tapctl_data);
   }
 
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_3tAPCTLDestroy_mp
 * \brief Destroy the jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLDestroy_mp( void *vdata )
{
   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   JX_Int groupid_x;
   JX_Int blocksmooth_type;
   JX_Int  Need_CC;    
   
   if (pre_3tapctl_data)
   {  
      groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
      Need_CC = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);   
   
      if (groupid_x == 0)
      {  
         // ARR
         if ( jx_3tAPCTLDataARR(pre_3tapctl_data) )
         {
            jx_ParCSRMatrixDestroy( jx_3tAPCTLDataARR(pre_3tapctl_data) );
         }
         
         // VRE
         if ( jx_3tAPCTLDataVRE(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataVRE(pre_3tapctl_data) );
         } 
                          
         // ARR_amg_solver
         if ( jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) )
         {
            jx_PAMGDestroy( jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) );
         }
          
         // ARR_gmres_solver  peghoty, 2011/10/29
         if ( jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) )
         {
            jx_GMRESDestroy(jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data));
         } 

         // RHS
         if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
           
         // WRR
         if ( jx_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWRR(pre_3tapctl_data)) );
            jx_ParVectorDestroy( jx_3tAPCTLDataWRR(pre_3tapctl_data) );
         }
               
         // WEE
         if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         
         // WII
         if ( jx_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWII(pre_3tapctl_data) );
         }    
      }      
      else if (groupid_x == 1)
      { 
       
         // AEE
         if ( jx_3tAPCTLDataAEE(pre_3tapctl_data) )
         {
            jx_ParCSRMatrixDestroy( jx_3tAPCTLDataAEE(pre_3tapctl_data) );
         }

         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            // VER
            if ( jx_3tAPCTLDataVER(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVER(pre_3tapctl_data) );
            }
               
            // VEI
            if ( jx_3tAPCTLDataVEI(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVEI(pre_3tapctl_data) );
            }
         }
         
         // AEE_amg_solver
         if ( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
         {
            jx_PAMGDestroy( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
         } 
  
         // RHS
         if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
         }

         // WRR
         if ( jx_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWRR(pre_3tapctl_data) );
         }
       
         // WEE
         if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWEE(pre_3tapctl_data)) );
            jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
         }

         // WII
         if ( jx_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWII(pre_3tapctl_data) );
         } 
      }
      else if (groupid_x == 2)
      { 
        
         // AII
         if ( jx_3tAPCTLDataAII(pre_3tapctl_data) )
         {
            jx_ParCSRMatrixDestroy( jx_3tAPCTLDataAII(pre_3tapctl_data) );
         }
         
         // VIE
         if ( jx_3tAPCTLDataVIE(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataVIE(pre_3tapctl_data) );
         }
         
         // AII_amg_solver
         if ( jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) )
         {
            jx_PAMGDestroy( jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) );
         }    
         
         // RHS
         if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
           
         // WRR
         if ( jx_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWRR(pre_3tapctl_data) );
         }
               
         // WEE
         if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         
         // WII
         if ( jx_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWII(pre_3tapctl_data)) );
            jx_ParVectorDestroy( jx_3tAPCTLDataWII(pre_3tapctl_data) );
         }           
      }
  
      // ACC
      if ( jx_3tAPCTLDataACC(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataACC(pre_3tapctl_data) );
      }
      
      // P
      if ( jx_3tAPCTLDataP(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataP(pre_3tapctl_data) );
      } 
           
      // ACC_amg_solver
      if ( jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) );
      }
       
      // ACC_gmres_solver  peghoty, 2011/10/29
      if ( jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) )
      {
         jx_GMRESDestroy(jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data));
      }
             
      // WCC
      if ( jx_3tAPCTLDataWCC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataWCC(pre_3tapctl_data) );
      }
            
      // GCC
      if ( jx_3tAPCTLDataGCC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataGCC(pre_3tapctl_data) );
      }
      
      // RES
      if ( jx_3tAPCTLDataRES(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataRES(pre_3tapctl_data) );
      }       
   
      // JAC
      if ( jx_3tAPCTLDataJAC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataJAC(pre_3tapctl_data) );
      }
      
      // comm_x and comm_y
      jx_MPI_Comm_free( &jx_3tAPCTLDataCommX(pre_3tapctl_data) ); 
      jx_MPI_Comm_free( &jx_3tAPCTLDataCommY(pre_3tapctl_data) );        
          
      jx_TFree(pre_3tapctl_data);
   }
 
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_3tAPCTLIterCount_mp
 * \brief Count the number of iterations for each sub-matrices (multi-processor case).
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLIterCount_mp( void *solver )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   MPI_Comm comm;
   MPI_Comm comm_x;
   JX_Int groupid_x;
   JX_Int ARR_relax_type;
   JX_Int AEE_relax_type;
   JX_Int AII_relax_type;
   JX_Int ARR_solver_id;
   JX_Int ACC_solver_id;
   JX_Int Need_CC;
   JX_Int myid, myid_x;   
   
   if (test_subls_iter)
   { 
      comm   = jx_3tAPCTLDataComm(pre_3tapctl_data);
      comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);
      groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
      AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
      AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
      ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);
      ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);
      Need_CC = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   
      jx_MPI_Comm_rank(comm, &myid);
      jx_MPI_Comm_rank(comm_x, &myid_x);
   
      jx_MPI_Barrier(comm);
   
      if (Need_CC && groupid_x == 0 && myid_x == 0)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_printf(" === num_iter_Ar_pctl_setup   (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_printf(" === num_iter_Ar_pctl_setup   (AMGGMRES) = %d\n", 
                     jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }
      }
   
      if (Need_CC && groupid_x == 2 && myid_x == 0)
      {
         jx_printf(" === num_iter_Ai_pctl_setup   (AMG)      = %d\n", 
                  jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data));
      } 
   
      jx_MPI_Barrier(comm);
   
      if (groupid_x == 0 && myid_x == 0)
      {
         if (ARR_relax_type == RELAX_AMG)
         {
            jx_printf(" === num_iter_Ar_pctl_precond (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jx_printf(" === num_iter_Ar_pctl_precond (JAC)      = %d\n", 
                     jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
         }
      }     

      if (groupid_x == 1 && myid_x == 0)
      {
         if (AEE_relax_type == RELAX_AMG)
         {
            jx_printf(" === num_iter_Ae_pctl_precond (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            jx_printf(" === num_iter_Ae_pctl_precond (JAC)      = %d\n", 
                     jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
         }
      }  

      if (groupid_x == 2 && myid_x == 0)
      {
         if (AII_relax_type == RELAX_AMG)
         {
            jx_printf(" === num_iter_Ai_pctl_precond (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jx_printf(" === num_iter_Ai_pctl_precond (JAC)      = %d\n", 
                     jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
         }
      } 
   
      if (Need_CC && myid == 0)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jx_printf(" === num_iter_Ac_pctl_precond (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jx_printf(" === num_iter_Ac_pctl_precond (AMGGMRES) = %d\n", 
                     jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
      }   
   
      jx_MPI_Barrier(comm);
      jx_printf("\n"); 
   }

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLIterCount_sp
 * \brief Count the number of iterations for each sub-matrices (single-processor case).
 * \author peghoty 
 * \date 2011/09/26
 */
JX_Int
jx_3tAPCTLIterCount_sp( void *solver )
{
   jx_3tAPCTLData *pre_3tapctl_data = solver;
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   JX_Int ARR_solver_id;
   JX_Int ACC_solver_id;
   JX_Int ARR_relax_type;
   JX_Int AEE_relax_type;
   JX_Int AII_relax_type;
   JX_Int Need_CC;
      
   if (test_subls_iter)
   {
      ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);
      ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);
      ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
      AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
      AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
      Need_CC = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);

      if (Need_CC)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_printf(" === num_iter_Ar_pctl_setup   (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_printf(" === num_iter_Ar_pctl_setup   (AMGGMRES) = %d\n", 
                     jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data));
         }      
         jx_printf(" === num_iter_Ai_pctl_setup   (AMG)      = %d\n", 
                  jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data));
      } 
      
      if (ARR_relax_type == RELAX_AMG)
      {
         jx_printf(" === num_iter_Ar_pctl_precond (AMG)      = %d\n", 
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jx_printf(" === num_iter_Ar_pctl_precond (JAC)      = %d\n", 
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data));
      }

      if (AEE_relax_type == RELAX_AMG)
      {
         jx_printf(" === num_iter_Ae_pctl_precond (AMG)      = %d\n", 
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         jx_printf(" === num_iter_Ae_pctl_precond (JAC)      = %d\n", 
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data));
      }

      if (AII_relax_type == RELAX_AMG)
      {
         jx_printf(" === num_iter_Ai_pctl_precond (AMG)      = %d\n", 
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jx_printf(" === num_iter_Ai_pctl_precond (JAC)      = %d\n", 
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data));
      }
   
      if (Need_CC)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jx_printf(" === num_iter_Ac_pctl_precond (AMG)      = %d\n", 
                     jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jx_printf(" === num_iter_Ac_pctl_precond (AMGGMRES) = %d\n", 
                     jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data));
         }
      }   
      jx_printf("\n"); 
   }

   return (0);
}

/*!
 * \fn JX_Int jx_GetAPCTLNumIterOfSubLS
 * \brief Get the number of iterations of the sub-linear systems solution.
 * \author peghoty 
 * \date 2012/03/23
 */
JX_Int 
jx_GetAPCTLNumIterOfSubLS( JX_Solver precond, jx_APCTLKrylovParam *apctlkrylov_param )
{
   jx_3tAPCTLData *pre_3tapctl_data = (jx_3tAPCTLData *)precond;
   
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);
   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int np_E = jx_3tAPCTLDataNpE(pre_3tapctl_data);
   JX_Int rootR_id = 0;
   JX_Int rootE_id = np_R;
   JX_Int rootI_id = np_R + np_E; 

   JX_Int num_iter_Ai_pctl_setup = jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data);
   JX_Int num_iter_Ar_pctl_setup = jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data);
   JX_Int num_iter_Ae_pctl_precond = jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data);
   JX_Int num_iter_Ai_pctl_precond = jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data);
   JX_Int num_iter_Ar_pctl_precond = jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data);
   JX_Int num_iter_Ac_pctl_precond = jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data); 
  
   // 从各进程组的根进程广播给所有进程
   jx_MPI_Bcast(&num_iter_Ar_pctl_setup,   1, JX_MPI_INT, rootR_id, comm);
   jx_MPI_Bcast(&num_iter_Ar_pctl_precond, 1, JX_MPI_INT, rootR_id, comm);
   jx_MPI_Bcast(&num_iter_Ae_pctl_precond, 1, JX_MPI_INT, rootE_id, comm);   
   jx_MPI_Bcast(&num_iter_Ai_pctl_setup,   1, JX_MPI_INT, rootI_id, comm);
   jx_MPI_Bcast(&num_iter_Ai_pctl_precond, 1, JX_MPI_INT, rootI_id, comm);   
   
   jx_APCTLKrylovParamNumIterAiSetup(apctlkrylov_param)   = num_iter_Ai_pctl_setup;
   jx_APCTLKrylovParamNumIterArSetup(apctlkrylov_param)   = num_iter_Ar_pctl_setup;
   jx_APCTLKrylovParamNumIterAePrecond(apctlkrylov_param) = num_iter_Ae_pctl_precond;
   jx_APCTLKrylovParamNumIterAiPrecond(apctlkrylov_param) = num_iter_Ai_pctl_precond;
   jx_APCTLKrylovParamNumIterArPrecond(apctlkrylov_param) = num_iter_Ar_pctl_precond;
   jx_APCTLKrylovParamNumIterAcPrecond(apctlkrylov_param) = num_iter_Ac_pctl_precond;
   jx_APCTLKrylovParamNeedCC(apctlkrylov_param) = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   
   return (0);
}

JX_Int
jx_GetAPCTLmgNumIterOfSubLS( JX_Solver precond, JX_Int groupid_x, JX_Int ng, jx_APCTLKrylovParam *apctlkrylov_param )
{
   jx_3tAPCTLData *pre_3tapctl_data = (jx_3tAPCTLData *)precond;

   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);
   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int num_iter_Ai_pctl_setup = jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data);
   JX_Int num_iter_Ar_pctl_setup = jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data);
   JX_Int num_iter_Ae_pctl_precond = jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data);
   JX_Int num_iter_Ai_pctl_precond = jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data);
   JX_Int num_iter_Ar_pctl_precond = jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data);
   JX_Int num_iter_Ac_pctl_precond = jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data);
   JX_Int rootE_id = ng * np_R;
   JX_Int rootI_id = rootE_id + np_R;
   JX_Int myid, nprocs, gidx, temp_sum;

   JX_Int *gather_buff = NULL;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   gather_buff = jx_CTAlloc(JX_Int, nprocs);

   jx_MPI_Gather(&num_iter_Ar_pctl_setup, 1, JX_MPI_INT, gather_buff, 1, JX_MPI_INT, 0, comm);
   if (myid == 0)
   {
      temp_sum = 0;
      for (gidx = 0; gidx < ng; gidx ++)
      {
         temp_sum += gather_buff[gidx*np_R];
      }
   }
   jx_MPI_Bcast(&temp_sum, 1, JX_MPI_INT, 0, comm);
   jx_APCTLKrylovParamNumIterArSetup(apctlkrylov_param) = temp_sum;

   jx_MPI_Gather(&num_iter_Ar_pctl_precond, 1, JX_MPI_INT, gather_buff, 1, JX_MPI_INT, 0, comm);
   if (myid == 0)
   {
      temp_sum = 0;
      for (gidx = 0; gidx < ng; gidx ++)
      {
         temp_sum += gather_buff[gidx*np_R];
      }
   }
   jx_MPI_Bcast(&temp_sum, 1, JX_MPI_INT, 0, comm);
   jx_APCTLKrylovParamNumIterArPrecond(apctlkrylov_param) = temp_sum;

   // 从各进程组的根进程广播给所有进程
   jx_MPI_Bcast(&num_iter_Ae_pctl_precond, 1, JX_MPI_INT, rootE_id, comm);
   jx_MPI_Bcast(&num_iter_Ai_pctl_setup,   1, JX_MPI_INT, rootI_id, comm);
   jx_MPI_Bcast(&num_iter_Ai_pctl_precond, 1, JX_MPI_INT, rootI_id, comm);

   jx_APCTLKrylovParamNumIterAiSetup(apctlkrylov_param)   = num_iter_Ai_pctl_setup;
   jx_APCTLKrylovParamNumIterAePrecond(apctlkrylov_param) = num_iter_Ae_pctl_precond;
   jx_APCTLKrylovParamNumIterAiPrecond(apctlkrylov_param) = num_iter_Ai_pctl_precond;
   jx_APCTLKrylovParamNumIterAcPrecond(apctlkrylov_param) = num_iter_Ac_pctl_precond;
   jx_APCTLKrylovParamNeedCC(apctlkrylov_param) = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);

   jx_TFree(gather_buff);

   return (0);
}
