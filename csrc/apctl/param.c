//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  param.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn jx_APCTLKrylovParam *jx_APCTLKrylovParamInitialize
 * \brief Create a jx_APCTLKrylovParam object. 
 * \author peghoty 
 * \date 2012/02/25
 */
jx_APCTLKrylovParam *
jx_APCTLKrylovParamInitialize()
{
   jx_APCTLKrylovParam *apctlkrylov_param = jx_CTAlloc(jx_APCTLKrylovParam, 1);

   JX_Int       solver_id;           // 1: PCG; 2: PGMRES; 3: PBicgStab
   JX_Int       precond_id;          // 1: APCTL; 2: Schur1; 3: Schur2
   JX_Real    tol;                 // tolerance of the APCTL-Krylov method
   JX_Int       max_iter;            // maximal number of iteration
   JX_Int       k_dim;               // number of restart
   JX_Int       is_check_restarted;  // peghoty, 2011/11/08
   JX_Int       two_norm;            // for PCG
   JX_Int       print_level;         // how many info to be output?
   JX_Int       TTest;               // whether timing the program?
   JX_Int       keepsol;             // whether save the solution?

   JX_Int    print_level_apctl; // how much info to be output in apctl?
                             // 1: CPU information
                             // 2: inner iteration information
                             // 3: both CPU and inner iteration information
   JX_Int    blocksmooth_type;  // BD or GS type preconditioner when Coarse Correction is not needed?
                             // BLOCKSMOOTH_BD:
                             // BLOCKSMOOTH_GS:
   JX_Int    num_relax_pre;     // number of sweeps for pre-block-smoothing
   JX_Int    num_relax_post;    // number of sweeps for post-block-smoothing

   /* solver type and restart number for interpolation-building of PRR */
   JX_Int    interp_solver_ARR; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;   
   JX_Int    interp_kdim_ARR;   // restart parameters for GMRES solver

   JX_Int    interp_maxit_ARR;  // maximal number of iteration for ARR to build PRR
   JX_Int    interp_maxit_AII;  // maximal number of iteration for AII to build PII
   JX_Real interp_tol_ARR;    // tolerance for ARR to build PRR
   JX_Real interp_tol_AII;    // tolerance for AII to build PII
   
   JX_Int    fixit_pctl_R;      // fixed number of iterations for ARR in PCTL
   JX_Int    fixit_pctl_E;      // fixed number of iterations for AEE in PCTL
   JX_Int    fixit_pctl_I;      // fixed number of iterations for AII in PCTL
   JX_Int    fixit_brlx_R;      // fixed number of iterations for ARR in Block Relaxation
   JX_Int    fixit_brlx_E;      // fixed number of iterations for AEE in Block Relaxation
   JX_Int    fixit_brlx_I;      // fixed number of iterations for AII in Block Relaxation     

   JX_Int    ARR_relax_type;
   JX_Int    AEE_relax_type;
   JX_Int    AII_relax_type;

   /* whether employ the fixed-number-of-iterations mode? peghoty, 2012/02/15 */
   JX_Int    use_fixedmode_R;
   JX_Int    use_fixedmode_E;
   JX_Int    use_fixedmode_I; 
   
   /* parameters to describe the weaking coupling between AEE and VER, VEI */
   JX_Real theta_wc_E;
   JX_Real threshold_wc_E;
   
   /* parameters to describe the diagonal dominance */
   JX_Real theta_dd_R;
   JX_Real theta_dd_E;
   JX_Real theta_dd_I;
   JX_Real threshold_dd_R;
   JX_Real threshold_dd_E;
   JX_Real threshold_dd_I;
   
   /* flag to indicate whether diagonal elements of the three DiagonalBlock
      matrices are firstly stored in each row for the CSR format. peghoty, 2012/03/06 */
   JX_Int    is_diagelm_first;

   /* Whether use the pure PCTL? */
   JX_Int    use_ppctl; 
 
   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JX_Int    test_subls_iter;

   /* CPU time of each phase */
   JX_Real cpu_trans;
   JX_Real cpu_setup;
   JX_Real cpu_solve;
   JX_Real cpu_total;   

   JX_Int debug_flag;
      
   JX_Int reset_zero;

   solver_id          = 2;
   precond_id         = 1;
   tol                = 1.0e-7;
   max_iter           = 200;
   k_dim              = 5;
   is_check_restarted = 0;
   two_norm           = 0;
   print_level        = 3;  
   TTest              = 1;
   keepsol            = 0;            

   print_level_apctl = 0; 
   blocksmooth_type  = BLOCKSMOOTH_BD; // BLOCKSMOOTH_GS;
   num_relax_pre     = 1;
   num_relax_post    = 0;  
   
   interp_solver_ARR = SOLVER_AMG;   // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   interp_kdim_ARR   = 20;
  
   interp_maxit_ARR  = 200;
   interp_maxit_AII  = 200;
   interp_tol_ARR    = 1.0e-6;   
   interp_tol_AII    = 1.0e-6; 
   
   fixit_pctl_R      = 1;      
   fixit_pctl_E      = 1;
   fixit_pctl_I      = 1;
   fixit_brlx_R      = 3;
   fixit_brlx_E      = 1;
   fixit_brlx_I      = 1;

   ARR_relax_type    = RELAX_AMG;
   AEE_relax_type    = RELAX_AMG;
   AII_relax_type    = RELAX_AMG;

   use_fixedmode_R   = 1;
   use_fixedmode_E   = 1;
   use_fixedmode_I   = 1;      
   
   theta_wc_E        = 1.0e-4;
   threshold_wc_E    = 0.5;
   theta_dd_R        = 0.9;
   theta_dd_E        = 0.9;
   theta_dd_I        = 0.9;
   threshold_dd_R    = 1.0;
   threshold_dd_E    = 1.0;
   threshold_dd_I    = 1.0; 
   
   is_diagelm_first  = 0; 
   use_ppctl         = 0;
   test_subls_iter   = 0; 
 
   cpu_trans         = 0.0;
   cpu_setup         = 0.0;
   cpu_solve         = 0.0;
   cpu_total         = 0.0;  

   debug_flag        = 0;

   reset_zero        = 1;
          
   jx_APCTLKrylovParamSetSolverID(apctlkrylov_param, solver_id);
   jx_APCTLKrylovParamSetPrecondID(apctlkrylov_param, precond_id);
   jx_APCTLKrylovParamSetTol(apctlkrylov_param, tol);
   jx_APCTLKrylovParamSetMaxIter(apctlkrylov_param, max_iter);
   jx_APCTLKrylovParamSetKDim(apctlkrylov_param, k_dim);
   jx_APCTLKrylovParamSetISCheckRestarted(apctlkrylov_param, is_check_restarted);
   jx_APCTLKrylovParamSetTwoNorm(apctlkrylov_param, two_norm);
   jx_APCTLKrylovParamSetPrintLevel(apctlkrylov_param, print_level);
   jx_APCTLKrylovParamSetTTest(apctlkrylov_param, TTest);
   jx_APCTLKrylovParamSetKeepSol(apctlkrylov_param, keepsol);
   
   jx_APCTLKrylovParamSetPrintLevelAPCTL(apctlkrylov_param, print_level_apctl );
   jx_APCTLKrylovParamSetBlockSmoothType(apctlkrylov_param, blocksmooth_type );
   jx_APCTLKrylovParamSetNumRelaxPre(apctlkrylov_param, num_relax_pre );
   jx_APCTLKrylovParamSetNumRelaxPost(apctlkrylov_param, num_relax_post );
   jx_APCTLKrylovParamSetInterpSolverARR(apctlkrylov_param, interp_solver_ARR );
   jx_APCTLKrylovParamSetInterpKdimARR(apctlkrylov_param, interp_kdim_ARR );
   jx_APCTLKrylovParamSetInterpMaxitARR(apctlkrylov_param, interp_maxit_ARR );
   jx_APCTLKrylovParamSetInterpMaxitAII(apctlkrylov_param, interp_maxit_AII );
   jx_APCTLKrylovParamSetInterpTolARR(apctlkrylov_param, interp_tol_ARR );
   jx_APCTLKrylovParamSetInterpTolAII(apctlkrylov_param, interp_tol_AII );
   jx_APCTLKrylovParamSetFixitPCTLR(apctlkrylov_param, fixit_pctl_R );
   jx_APCTLKrylovParamSetFixitPCTLE(apctlkrylov_param, fixit_pctl_E );
   jx_APCTLKrylovParamSetFixitPCTLI(apctlkrylov_param, fixit_pctl_I );
   jx_APCTLKrylovParamSetFixitBrlxR(apctlkrylov_param, fixit_brlx_R );
   jx_APCTLKrylovParamSetFixitBrlxE(apctlkrylov_param, fixit_brlx_E );
   jx_APCTLKrylovParamSetFixitBrlxI(apctlkrylov_param, fixit_brlx_I );
   jx_APCTLKrylovParamSetARRRelaxType(apctlkrylov_param, ARR_relax_type);
   jx_APCTLKrylovParamSetAEERelaxType(apctlkrylov_param, AEE_relax_type);
   jx_APCTLKrylovParamSetAIIRelaxType(apctlkrylov_param, AII_relax_type);
   jx_APCTLKrylovParamSetUseFixedModeR(apctlkrylov_param, use_fixedmode_R );
   jx_APCTLKrylovParamSetUseFixedModeE(apctlkrylov_param, use_fixedmode_E );
   jx_APCTLKrylovParamSetUseFixedModeI(apctlkrylov_param, use_fixedmode_I );
   jx_APCTLKrylovParamSetThetaWCE(apctlkrylov_param, theta_wc_E );
   jx_APCTLKrylovParamSetParamThresholdWCE(apctlkrylov_param, threshold_wc_E );
   jx_APCTLKrylovParamSetThetaDDR(apctlkrylov_param, theta_dd_R );
   jx_APCTLKrylovParamSetThetaDDE(apctlkrylov_param, theta_dd_E );
   jx_APCTLKrylovParamSetThetaDDI(apctlkrylov_param, theta_dd_I );
   jx_APCTLKrylovParamSetThresholdDDR(apctlkrylov_param, threshold_dd_R );
   jx_APCTLKrylovParamSetThresholdDDE(apctlkrylov_param, threshold_dd_E );
   jx_APCTLKrylovParamSetThresholdDDI(apctlkrylov_param, threshold_dd_I );
   jx_APCTLKrylovParamSetISDiagElmFirst(apctlkrylov_param, is_diagelm_first);
   jx_APCTLKrylovParamSetUsePPCTL(apctlkrylov_param, use_ppctl);
   jx_APCTLKrylovParamSetTestSubLSIter(apctlkrylov_param, test_subls_iter);    
   // peghoty, 2012/03/25
   jx_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans);
   jx_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jx_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jx_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total); 

   jx_APCTLKrylovParamSetDebugFlag(apctlkrylov_param, debug_flag);
   jx_APCTLKrylovParamSetResetZero(apctlkrylov_param, reset_zero);

   return (apctlkrylov_param);
}

/*!
 * \fn JX_Int jx_APCTLKrylovParamSetXXXX
 * \brief Set parameters for the jx_APCTLKrylovParam object. 
 * \author peghoty 
 * \date 2012/02/25
 */
JX_Int 
jx_APCTLKrylovParamSetNumGroup( void *param, JX_Int ng )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamNumGroup(apctlkrylov_param) = ng;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetSolverID( void *param, JX_Int solver_id )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamSolverID(apctlkrylov_param) = solver_id;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetPrecondID( void *param, JX_Int precond_id )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamPrecondID(apctlkrylov_param) = precond_id;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetTol( void *param, JX_Real tol )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamTol(apctlkrylov_param) = tol;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetMaxIter( void *param, JX_Int max_iter )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamMaxIter(apctlkrylov_param) = max_iter;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetKDim( void *param, JX_Int k_dim )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamKDim(apctlkrylov_param) = k_dim;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetISCheckRestarted( void *param, JX_Int is_check_restarted )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamISCheckRestarted(apctlkrylov_param) = is_check_restarted;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetTwoNorm( void *param, JX_Int two_norm )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamTwoNorm(apctlkrylov_param) = two_norm;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetPrintLevel( void *param, JX_Int print_level )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamPrintLevel(apctlkrylov_param) = print_level;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetTTest( void *param, JX_Int TTest )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamTTest(apctlkrylov_param) = TTest;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetKeepSol( void *param, JX_Int keepsol )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamKeepSol(apctlkrylov_param) = keepsol;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetNumIterations( void *param, JX_Int num )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamNumIterations(apctlkrylov_param) = num;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetLastRelNrm( void *param, JX_Real last_rel_nrm )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamLastRelNrm(apctlkrylov_param) = last_rel_nrm;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetPrintLevelAPCTL( void *param, JX_Int print_level_apctl )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param) = print_level_apctl;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetBlockSmoothType( void *param, JX_Int blocksmooth_type )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamBlockSmoothType(apctlkrylov_param) = blocksmooth_type;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetNumRelaxPre( void *param, JX_Int num_relax_pre )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamNumRelaxPre(apctlkrylov_param) = num_relax_pre;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetNumRelaxPost( void *param, JX_Int num_relax_post )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamNumRelaxPost(apctlkrylov_param) = num_relax_post;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpSolverARR( void *param, JX_Int interp_solver_ARR )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpSolverARR(apctlkrylov_param) = interp_solver_ARR;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpKdimARR( void *param, JX_Int interp_kdim_ARR )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpKdimARR(apctlkrylov_param) = interp_kdim_ARR;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpMaxitARR( void *param, JX_Int interp_maxit_ARR )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param) = interp_maxit_ARR;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpMaxitAII( void *param, JX_Int interp_maxit_AII )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param) = interp_maxit_AII;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpTolARR( void *param, JX_Real interp_tol_ARR )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpTolARR(apctlkrylov_param) = interp_tol_ARR;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpTolAII( void *param, JX_Real interp_tol_AII )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpTolAII(apctlkrylov_param) = interp_tol_AII;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetARRRelaxTol( void *param, JX_Real ARR_relax_tol )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamARRRelaxTol(apctlkrylov_param) = ARR_relax_tol;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetAEERelaxTol( void *param, JX_Real AEE_relax_tol )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamAEERelaxTol(apctlkrylov_param) = AEE_relax_tol;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetAIIRelaxTol( void *param, JX_Real AII_relax_tol )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamAIIRelaxTol(apctlkrylov_param) = AII_relax_tol;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetACCRelaxTol( void *param, JX_Real ACC_relax_tol )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamACCRelaxTol(apctlkrylov_param) = ACC_relax_tol;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetFixitPCTLR( void *param, JX_Int fixit_pctl_R )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamFixitPCTLR(apctlkrylov_param) = fixit_pctl_R;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetFixitPCTLE( void *param, JX_Int fixit_pctl_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamFixitPCTLE(apctlkrylov_param) = fixit_pctl_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetFixitPCTLI( void *param, JX_Int fixit_pctl_I )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamFixitPCTLI(apctlkrylov_param) = fixit_pctl_I;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetFixitBrlxR( void *param, JX_Int fixit_brlx_R )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamFixitBrlxR(apctlkrylov_param) = fixit_brlx_R;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetFixitBrlxE( void *param, JX_Int fixit_brlx_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamFixitBrlxE(apctlkrylov_param) = fixit_brlx_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetFixitBrlxI( void *param, JX_Int fixit_brlx_I )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamFixitBrlxI(apctlkrylov_param) = fixit_brlx_I;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetARRRelaxType( void *param, JX_Int ARR_relax_type )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamARRRelaxType(apctlkrylov_param) = ARR_relax_type;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetAEERelaxType( void *param, JX_Int AEE_relax_type )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamAEERelaxType(apctlkrylov_param) = AEE_relax_type;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetAIIRelaxType( void *param, JX_Int AII_relax_type )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamAIIRelaxType(apctlkrylov_param) = AII_relax_type;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetUseFixedModeR( void *param, JX_Int use_fixedmode_R )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamUseFixedModeR(apctlkrylov_param) = use_fixedmode_R;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetUseFixedModeE( void *param, JX_Int use_fixedmode_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamUseFixedModeE(apctlkrylov_param) = use_fixedmode_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetUseFixedModeI( void *param, JX_Int use_fixedmode_I )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamUseFixedModeI(apctlkrylov_param) = use_fixedmode_I;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThetaWCE( void *param, JX_Real theta_wc_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThetaWCE(apctlkrylov_param) = theta_wc_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetParamThresholdWCE( void *param, JX_Real threshold_wc_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThresholdWCE(apctlkrylov_param) = threshold_wc_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThetaDDR( void *param, JX_Real theta_dd_R )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThetaDDR(apctlkrylov_param) = theta_dd_R;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThetaDDE( void *param, JX_Real theta_dd_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThetaDDE(apctlkrylov_param) = theta_dd_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThetaDDI( void *param, JX_Real theta_dd_I )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThetaDDI(apctlkrylov_param) = theta_dd_I;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThresholdDDR( void *param, JX_Real threshold_dd_R )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThresholdDDR(apctlkrylov_param) = threshold_dd_R;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThresholdDDE( void *param, JX_Real threshold_dd_E )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThresholdDDE(apctlkrylov_param) = threshold_dd_E;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetThresholdDDI( void *param, JX_Real threshold_dd_I )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamThresholdDDI(apctlkrylov_param) = threshold_dd_I;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetISDiagElmFirst( void *param, JX_Int is_diagelm_first )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param) = is_diagelm_first;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetUsePPCTL( void *param, JX_Int use_ppctl )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamUsePPCTL(apctlkrylov_param) = use_ppctl;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetDebugFlag( void *param, JX_Int debug_flag )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamDebugFlag(apctlkrylov_param) = debug_flag;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetResetZero( void *param, JX_Int reset_zero )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamResetZero(apctlkrylov_param) = reset_zero;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetStrongThreshold( void *param, JX_Real strong_threshold )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamStrongThreshold(apctlkrylov_param) = strong_threshold;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetInterpType( void *param, JX_Int interp_type )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamInterpType(apctlkrylov_param) = interp_type;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetCoarsenType( void *param, JX_Int coarsen_type )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamCoarsenType(apctlkrylov_param) = coarsen_type;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetAggNumLevels( void *param, JX_Int agg_num_levels )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamAggNumLevels(apctlkrylov_param) = agg_num_levels;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetCoarseThreshold( void *param, JX_Int coarse_threshold )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamCoarseThreshold(apctlkrylov_param) = coarse_threshold;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetPrintLevelAMG( void *param, JX_Int print_level_amg )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param) = print_level_amg;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetTestSubLSIter( void *param, JX_Int test_subls_iter )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param) = test_subls_iter;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetCPUTrans( void *param, JX_Real cpu_trans )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamCPUTrans(apctlkrylov_param) = cpu_trans;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetCPUSetup( void *param, JX_Real cpu_setup )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamCPUSetup(apctlkrylov_param) = cpu_setup;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetCPUSolve( void *param, JX_Real cpu_solve )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamCPUSolve(apctlkrylov_param) = cpu_solve;
   return jx_error_flag;
}

JX_Int 
jx_APCTLKrylovParamSetCPUTotal( void *param, JX_Real cpu_total )
{
   jx_APCTLKrylovParam *apctlkrylov_param = param;
   jx_APCTLKrylovParamCPUTotal(apctlkrylov_param) = cpu_total;
   return jx_error_flag;
}
