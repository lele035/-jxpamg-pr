//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_apctl.h -- head files for 3t solver and preconditioners
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

#ifndef JX_3T_HEADER
#define JX_3T_HEADER 

#ifndef JX_KRYLOV_HEADER
#include "jx_krylov.h"
#endif

/* relax type for a submatrix */
#define RELAX_AMG      0
#define RELAX_WJACOBI  1

/* solver type for a submatrix */
#define SOLVER_AMG      0
#define SOLVER_AMGGMRES 1

/* block smoothing type for a 3t matrix */
#define BLOCKSMOOTH_GS 0
#define BLOCKSMOOTH_BD 1

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jx_3tAPCTLData
 */
typedef struct
{
   JX_Real   tol;                    // tolerance    
   JX_Int      max_iter;               // maximal number of iteration
   JX_Int      num_relax_pre;          // number of pre-relaxations 
   JX_Int      num_relax_post;         // number of post-relaxations    
   JX_Int      print_level;            // 1: print CPU time 2: print number of iterations 

   JX_Int      num_iterations;         // number of iterations
   JX_Real   last_rel_nrm;           // last relative residual norm
   JX_Real   ave_conv_factor;        // average convergence factor
   
   /* for processors-grouping */
   MPI_Comm comm;
   MPI_Comm comm_x;
   MPI_Comm comm_y;
   
   JX_Int      groupid_x; 
   JX_Int      groupid_y;
 
   JX_Int      np_R;     // number of processors used for ARR
   JX_Int      np_E;     // number of processors used for AEE
   JX_Int      np_I;     // number of processors used for AII      

   JX_Int      num_group;

   JX_Int      reset_zero;

   /* flag to indicate whether diagonal elements of the three DiagonalBlock
      matrices are firstly stored in each row for the CSR format. peghoty, 2012/03/06 */
   JX_Int    is_diagelm_first;

   /* the 3t matrix to be solved */
   jx_ParCSRMatrix *A;
   
   /* Blocks of the 3t matrix A (基于进程组) */
   jx_ParCSRMatrix *ARR;
   jx_ParCSRMatrix *AEE;
   jx_ParCSRMatrix *AII;

   jx_ParVector    *VRE;
   jx_ParVector    *VER;
   jx_ParVector    *VEI;
   jx_ParVector    *VIE;   

   jx_ParVector   **VER2;

   /* Blocks of the 3t matrix A (基于所有进程) peghoty, 2012/02/26 */
   jx_ParCSRMatrix *ARR_all;
   jx_ParCSRMatrix *AEE_all;
   jx_ParCSRMatrix *AII_all;

   jx_ParCSRMatrix **ARR_all2;

   jx_ParVector    *VRE_all;
   jx_ParVector    *VER_all;
   jx_ParVector    *VEI_all;
   jx_ParVector    *VIE_all; 

   jx_ParVector   **VRE_all2;
   jx_ParVector   **VER_all2;

   /* prolongation operator */
   jx_ParCSRMatrix *P;  
   jx_ParVector    *PRR;
   jx_ParVector    *PII;
       
   /* coarse matrix ACC */
   jx_ParCSRMatrix *ACC; 
   
   /* relaxation type of each subblock */
   JX_Int     ARR_relax_type;
   JX_Int     AEE_relax_type;
   JX_Int     AII_relax_type;           
     
   /* max_iter for each subblock */
   JX_Int     ARR_interp_maxit;     
   JX_Int     AII_interp_maxit;
   JX_Int     ARR_relax_maxit;
   JX_Int     AEE_relax_maxit;
   JX_Int     AII_relax_maxit;   
   JX_Int     ACC_relax_maxit;

   JX_Int     maxit_default;

   /* tolerace for each subblock */
   JX_Real  ARR_interp_tol;   
   JX_Real  AII_interp_tol;
   JX_Real  ARR_relax_tol;
   JX_Real  AEE_relax_tol;
   JX_Real  AII_relax_tol;
   JX_Real  ACC_relax_tol;

   JX_Real  tol_default;

   /* solver type for interpolation-building of PRR.  peghoty,2011/10/29 */
   JX_Int ARR_solver_id; // ARR sometimes is very hard to converge by PAMG, PAMG-GMRES is provided as an alternative

   /* solver type for PCTL iteration.  peghoty,2011/10/29 */
   JX_Int ACC_solver_id; // ACC sometimes is very hard to converge by PAMG, PAMG-GMRES is provided as an alternative
   
   /* restart parameters for GMRES solver. peghoty,2011/10/29 */
   JX_Int ARR_kdim;
   JX_Int ACC_kdim;
    
   /* AMG solver data for each subblock */
   jx_ParAMGData    *ARR_amg_solver;
   jx_ParAMGData    *AEE_amg_solver;
   jx_ParAMGData    *AII_amg_solver;
   jx_ParAMGData    *ACC_amg_solver;
   
   /* gmres solver data for ARR and ACC. peghoty,2011/10/29 */
   jx_GMRESData     *ARR_gmres_solver;
   jx_GMRESData     *ACC_gmres_solver;   

   /* Auxialiary vectors */
   jx_ParVector     *WRR;
   jx_ParVector     *WEE;
   jx_ParVector     *WII;
   jx_ParVector     *WCC;

   jx_ParVector     *GCC;

   jx_ParVector     *RES;
   jx_ParVector     *RHS;
   
   jx_ParVector     *JAC;  // used in Jacobi relaxation

   JX_Real *tTEMP;

   /* parameters to describe the weaking coupling */
   JX_Real theta_wc_R;
   JX_Real theta_wc_E;
   JX_Real theta_wc_I;
   JX_Real threshold_wc_R;
   JX_Real threshold_wc_E;
   JX_Real threshold_wc_I;
   JX_Int    IS_WC_R;
   JX_Int    IS_WC_E;    
   JX_Int    IS_WC_I;
   
   /* parameters to describe the diagonal dominance */
   JX_Real theta_dd_R;
   JX_Real theta_dd_E;
   JX_Real theta_dd_I;
   JX_Real threshold_dd_R;
   JX_Real threshold_dd_E;
   JX_Real threshold_dd_I; 
   JX_Int    IS_DD_R;
   JX_Int    IS_DD_E;    
   JX_Int    IS_DD_I;
   
   /* flag to indicate whether the Coarse-grid Correction is necessary */
   JX_Int    Need_CC;
   
   /* type of the block smoothing */
   JX_Int    blocksmooth_type;
   
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

   JX_Real strong_threshold;

   JX_Int interp_type;
   JX_Int coarsen_type;
   JX_Int agg_num_levels;
   JX_Int coarse_threshold;
   JX_Int print_level_amg;
    
} jx_3tAPCTLData;

#define jx_3tAPCTLDatatTEMP(pre_3tapctl_data)               ((pre_3tapctl_data) -> tTEMP)
#define jx_3tAPCTLDataTol(pre_3tapctl_data)                 ((pre_3tapctl_data) -> tol)
#define jx_3tAPCTLDataMaxIter(pre_3tapctl_data)             ((pre_3tapctl_data) -> max_iter)
#define jx_3tAPCTLDataNumRlxPre(pre_3tapctl_data)           ((pre_3tapctl_data) -> num_relax_pre)
#define jx_3tAPCTLDataNumRlxPost(pre_3tapctl_data)          ((pre_3tapctl_data) -> num_relax_post)
#define jx_3tAPCTLDataPrintLevel(pre_3tapctl_data)          ((pre_3tapctl_data) -> print_level)
#define jx_3tAPCTLDataNumIterations(pre_3tapctl_data)       ((pre_3tapctl_data) -> num_iterations)
#define jx_3tAPCTLDataLastRelNrm(pre_3tapctl_data)          ((pre_3tapctl_data) -> last_rel_nrm)
#define jx_3tAPCTLDataAveConvFactor(pre_3tapctl_data)       ((pre_3tapctl_data) -> ave_conv_factor)
#define jx_3tAPCTLDataComm(pre_3tapctl_data)                ((pre_3tapctl_data) -> comm)
#define jx_3tAPCTLDataCommX(pre_3tapctl_data)               ((pre_3tapctl_data) -> comm_x)
#define jx_3tAPCTLDataCommY(pre_3tapctl_data)               ((pre_3tapctl_data) -> comm_y)
#define jx_3tAPCTLDataGroupIdX(pre_3tapctl_data)            ((pre_3tapctl_data) -> groupid_x)
#define jx_3tAPCTLDataGroupIdY(pre_3tapctl_data)            ((pre_3tapctl_data) -> groupid_y)
#define jx_3tAPCTLDataNpR(pre_3tapctl_data)                 ((pre_3tapctl_data) -> np_R)
#define jx_3tAPCTLDataNpE(pre_3tapctl_data)                 ((pre_3tapctl_data) -> np_E)
#define jx_3tAPCTLDataNpI(pre_3tapctl_data)                 ((pre_3tapctl_data) -> np_I)
#define jx_3tAPCTLDataNumGroup(pre_3tapctl_data)            ((pre_3tapctl_data) -> num_group)
#define jx_3tAPCTLDataResetZero(pre_3tapctl_data)           ((pre_3tapctl_data) -> reset_zero)
#define jx_3tAPCTLDataIsDiagElmFirst(pre_3tapctl_data)      ((pre_3tapctl_data) -> is_diagelm_first)
#define jx_3tAPCTLDataA(pre_3tapctl_data)                   ((pre_3tapctl_data) -> A)
#define jx_3tAPCTLDataARR(pre_3tapctl_data)                 ((pre_3tapctl_data) -> ARR)
#define jx_3tAPCTLDataAEE(pre_3tapctl_data)                 ((pre_3tapctl_data) -> AEE)
#define jx_3tAPCTLDataAII(pre_3tapctl_data)                 ((pre_3tapctl_data) -> AII)
#define jx_3tAPCTLDataVRE(pre_3tapctl_data)                 ((pre_3tapctl_data) -> VRE)
#define jx_3tAPCTLDataVER(pre_3tapctl_data)                 ((pre_3tapctl_data) -> VER)
#define jx_3tAPCTLDataVER2(pre_3tapctl_data)                ((pre_3tapctl_data) -> VER2)
#define jx_3tAPCTLDataVEI(pre_3tapctl_data)                 ((pre_3tapctl_data) -> VEI)
#define jx_3tAPCTLDataVIE(pre_3tapctl_data)                 ((pre_3tapctl_data) -> VIE)
#define jx_3tAPCTLDataARRAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> ARR_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataARRAll2(pre_3tapctl_data)             ((pre_3tapctl_data) -> ARR_all2)
#define jx_3tAPCTLDataAEEAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> AEE_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataAIIAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> AII_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataVREAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> VRE_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataVREAll2(pre_3tapctl_data)             ((pre_3tapctl_data) -> VRE_all2)
#define jx_3tAPCTLDataVERAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> VER_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataVERAll2(pre_3tapctl_data)             ((pre_3tapctl_data) -> VER_all2)
#define jx_3tAPCTLDataVEIAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> VEI_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataVIEAll(pre_3tapctl_data)              ((pre_3tapctl_data) -> VIE_all) /* peghoty, 2012/02/26 */
#define jx_3tAPCTLDataP(pre_3tapctl_data)                   ((pre_3tapctl_data) -> P)
#define jx_3tAPCTLDataPRR(pre_3tapctl_data)                 ((pre_3tapctl_data) -> PRR)
#define jx_3tAPCTLDataPII(pre_3tapctl_data)                 ((pre_3tapctl_data) -> PII)
#define jx_3tAPCTLDataACC(pre_3tapctl_data)                 ((pre_3tapctl_data) -> ACC)
#define jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data)        ((pre_3tapctl_data) -> ARR_relax_type)
#define jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data)        ((pre_3tapctl_data) -> AEE_relax_type)
#define jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data)        ((pre_3tapctl_data) -> AII_relax_type)
#define jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data)      ((pre_3tapctl_data) -> ARR_interp_maxit)
#define jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data)      ((pre_3tapctl_data) -> AII_interp_maxit)
#define jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data)       ((pre_3tapctl_data) -> ARR_relax_maxit)
#define jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data)       ((pre_3tapctl_data) -> AEE_relax_maxit)
#define jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data)       ((pre_3tapctl_data) -> AII_relax_maxit)
#define jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data)       ((pre_3tapctl_data) -> ACC_relax_maxit)
#define jx_3tAPCTLDataMaxItDefault(pre_3tapctl_data)        ((pre_3tapctl_data) -> maxit_default)
#define jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data)        ((pre_3tapctl_data) -> ARR_interp_tol)
#define jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data)        ((pre_3tapctl_data) -> AII_interp_tol)
#define jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data)         ((pre_3tapctl_data) -> ARR_relax_tol)
#define jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data)         ((pre_3tapctl_data) -> AEE_relax_tol)
#define jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data)         ((pre_3tapctl_data) -> AII_relax_tol)
#define jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data)         ((pre_3tapctl_data) -> ACC_relax_tol)
#define jx_3tAPCTLDataTolDefault(pre_3tapctl_data)          ((pre_3tapctl_data) -> tol_default)
#define jx_3tAPCTLDataARRSolverID(pre_3tapctl_data)         ((pre_3tapctl_data) -> ARR_solver_id)
#define jx_3tAPCTLDataACCSolverID(pre_3tapctl_data)         ((pre_3tapctl_data) -> ACC_solver_id)
#define jx_3tAPCTLDataARRKDim(pre_3tapctl_data)             ((pre_3tapctl_data) -> ARR_kdim)
#define jx_3tAPCTLDataACCKDim(pre_3tapctl_data)             ((pre_3tapctl_data) -> ACC_kdim)  
#define jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)        ((pre_3tapctl_data) -> ARR_amg_solver)  
#define jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)        ((pre_3tapctl_data) -> AEE_amg_solver)
#define jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)        ((pre_3tapctl_data) -> AII_amg_solver)
#define jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)        ((pre_3tapctl_data) -> ACC_amg_solver) 
#define jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data)      ((pre_3tapctl_data) -> ARR_gmres_solver)
#define jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data)      ((pre_3tapctl_data) -> ACC_gmres_solver)
#define jx_3tAPCTLDataWRR(pre_3tapctl_data)                 ((pre_3tapctl_data) -> WRR)
#define jx_3tAPCTLDataWEE(pre_3tapctl_data)                 ((pre_3tapctl_data) -> WEE)
#define jx_3tAPCTLDataWII(pre_3tapctl_data)                 ((pre_3tapctl_data) -> WII)
#define jx_3tAPCTLDataWCC(pre_3tapctl_data)                 ((pre_3tapctl_data) -> WCC)
#define jx_3tAPCTLDataGCC(pre_3tapctl_data)                 ((pre_3tapctl_data) -> GCC)
#define jx_3tAPCTLDataRES(pre_3tapctl_data)                 ((pre_3tapctl_data) -> RES)
#define jx_3tAPCTLDataRHS(pre_3tapctl_data)                 ((pre_3tapctl_data) -> RHS)
#define jx_3tAPCTLDataJAC(pre_3tapctl_data)                 ((pre_3tapctl_data) -> JAC)
#define jx_3tAPCTLDataThetaWCR(pre_3tapctl_data)            ((pre_3tapctl_data) -> theta_wc_R)
#define jx_3tAPCTLDataThetaWCE(pre_3tapctl_data)            ((pre_3tapctl_data) -> theta_wc_E)
#define jx_3tAPCTLDataThetaWCI(pre_3tapctl_data)            ((pre_3tapctl_data) -> theta_wc_I)
#define jx_3tAPCTLDataThresholdWCR(pre_3tapctl_data)        ((pre_3tapctl_data) -> threshold_wc_R)
#define jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data)        ((pre_3tapctl_data) -> threshold_wc_E)
#define jx_3tAPCTLDataThresholdWCI(pre_3tapctl_data)        ((pre_3tapctl_data) -> threshold_wc_I)
#define jx_3tAPCTLDataISWCR(pre_3tapctl_data)               ((pre_3tapctl_data) -> IS_WC_R)
#define jx_3tAPCTLDataISWCE(pre_3tapctl_data)               ((pre_3tapctl_data) -> IS_WC_E)
#define jx_3tAPCTLDataISWCI(pre_3tapctl_data)               ((pre_3tapctl_data) -> IS_WC_I)
#define jx_3tAPCTLDataThetaDDR(pre_3tapctl_data)            ((pre_3tapctl_data) -> theta_dd_R)
#define jx_3tAPCTLDataThetaDDE(pre_3tapctl_data)            ((pre_3tapctl_data) -> theta_dd_E)
#define jx_3tAPCTLDataThetaDDI(pre_3tapctl_data)            ((pre_3tapctl_data) -> theta_dd_I)
#define jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data)        ((pre_3tapctl_data) -> threshold_dd_R)
#define jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data)        ((pre_3tapctl_data) -> threshold_dd_E)
#define jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data)        ((pre_3tapctl_data) -> threshold_dd_I)
#define jx_3tAPCTLDataISDDR(pre_3tapctl_data)               ((pre_3tapctl_data) -> IS_DD_R)
#define jx_3tAPCTLDataISDDE(pre_3tapctl_data)               ((pre_3tapctl_data) -> IS_DD_E)
#define jx_3tAPCTLDataISDDI(pre_3tapctl_data)               ((pre_3tapctl_data) -> IS_DD_I)
#define jx_3tAPCTLDataNeedCC(pre_3tapctl_data)              ((pre_3tapctl_data) -> Need_CC)
#define jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data)     ((pre_3tapctl_data) -> blocksmooth_type)
#define jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data)          ((pre_3tapctl_data) -> fixit_pctl_R)
#define jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data)          ((pre_3tapctl_data) -> fixit_pctl_E)
#define jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data)          ((pre_3tapctl_data) -> fixit_pctl_I)
#define jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data)          ((pre_3tapctl_data) -> fixit_brlx_R)
#define jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data)          ((pre_3tapctl_data) -> fixit_brlx_E)
#define jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data)          ((pre_3tapctl_data) -> fixit_brlx_I)
#define jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data)       ((pre_3tapctl_data) -> use_fixedmode_R)
#define jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data)       ((pre_3tapctl_data) -> use_fixedmode_E)
#define jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data)       ((pre_3tapctl_data) -> use_fixedmode_I)
#define jx_3tAPCTLDataUsePPCTL(pre_3tapctl_data)            ((pre_3tapctl_data) -> use_ppctl)
// peghoty, 2012/03/23
#define jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data)       ((pre_3tapctl_data) -> test_subls_iter)
#define jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data)      ((pre_3tapctl_data) -> num_iter_Ai_pctl_setup)
#define jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data)      ((pre_3tapctl_data) -> num_iter_Ar_pctl_setup)
#define jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data)    ((pre_3tapctl_data) -> num_iter_Ae_pctl_precond)
#define jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data)    ((pre_3tapctl_data) -> num_iter_Ai_pctl_precond)
#define jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data)    ((pre_3tapctl_data) -> num_iter_Ar_pctl_precond)
#define jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data)    ((pre_3tapctl_data) -> num_iter_Ac_pctl_precond)
#define jx_3tAPCTLDataDebugFlag(pre_3tapctl_data)           ((pre_3tapctl_data) -> debug_flag)
#define jx_3tAPCTLDataStrongThreshold(pre_3tapctl_data)     ((pre_3tapctl_data) -> strong_threshold)
#define jx_3tAPCTLDataInterpType(pre_3tapctl_data)          ((pre_3tapctl_data) -> interp_type)
#define jx_3tAPCTLDataCoarsenType(pre_3tapctl_data)         ((pre_3tapctl_data) -> coarsen_type)
#define jx_3tAPCTLDataAggNumLevels(pre_3tapctl_data)        ((pre_3tapctl_data) -> agg_num_levels)
#define jx_3tAPCTLDataCoarseThreshold(pre_3tapctl_data)     ((pre_3tapctl_data) -> coarse_threshold)
#define jx_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data)     ((pre_3tapctl_data) -> print_level_amg)


/*!
 * \struct jx_APCTLKrylovParam
 * \brief parameters for APCTL-Krylov method
 */
typedef struct
{
   JX_Int       num_group;

   JX_Int       reset_zero;

   //----------------------------------------------------------------------------
   //  parameters for P-Krylov methods
   //--------------------------------------------------------------------------
   JX_Int       solver_id;           // 1: PCG; 2: PGMRES; 3: PBicgStab
   JX_Int       precond_id;           // 1: APCTL; 2: Schur1; 3: Schur2
   JX_Real    tol;                 // tolerance of the APCTL-Krylov method
   JX_Int       max_iter;            // maximal number of iteration
   JX_Int       k_dim;               // number of restart
   JX_Int       is_check_restarted;  // peghoty, 2011/11/08
   JX_Int       two_norm;            // for PCG
   JX_Int       print_level;         // how many info to be output?
   JX_Int       TTest;               // whether timing the program?
   JX_Int       keepsol;             // whether save the solution?
   JX_Int       num_iterations;      // number of iterations of the APCTL-Krylov method
   JX_Real    last_rel_nrm;        // last relative residual norm     peghoty,2012/03/22

   //----------------------------------------------------------------------------
   //  parameters for PCTL iteration or preconditioner
   //--------------------------------------------------------------------------   
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
   
   JX_Real  ARR_relax_tol;
   JX_Real  AEE_relax_tol;
   JX_Real  AII_relax_tol;
   JX_Real  ACC_relax_tol;

   JX_Int    ARR_relax_type;
   JX_Int    AEE_relax_type;
   JX_Int    AII_relax_type;

   JX_Int    fixit_pctl_R;      // fixed number of iterations for ARR in PCTL
   JX_Int    fixit_pctl_E;      // fixed number of iterations for AEE in PCTL
   JX_Int    fixit_pctl_I;      // fixed number of iterations for AII in PCTL
   JX_Int    fixit_brlx_R;      // fixed number of iterations for ARR in Block Relaxation
   JX_Int    fixit_brlx_E;      // fixed number of iterations for AEE in Block Relaxation
   JX_Int    fixit_brlx_I;      // fixed number of iterations for AII in Block Relaxation     

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
   
   /* variables to count the number of iteration */
   JX_Int    num_iter_Ai_pctl_setup;
   JX_Int    num_iter_Ar_pctl_setup;
   JX_Int    num_iter_Ae_pctl_precond;
   JX_Int    num_iter_Ai_pctl_precond;
   JX_Int    num_iter_Ar_pctl_precond;
   JX_Int    num_iter_Ac_pctl_precond;
   
   /* whether need coarse-grid correction in apctl preconditioner? */
   JX_Int    Need_CC;
   
   /* CPU time of each phase */
   JX_Real cpu_trans;
   JX_Real cpu_setup;
   JX_Real cpu_solve;
   JX_Real cpu_total;                

   JX_Int debug_flag;

   JX_Real strong_threshold;

   JX_Int interp_type;
   JX_Int coarsen_type;
   JX_Int agg_num_levels;
   JX_Int coarse_threshold;
   JX_Int print_level_amg;

} jx_APCTLKrylovParam;

#define jx_APCTLKrylovParamNumGroup(apctlkrylov_param)           ((apctlkrylov_param) -> num_group)
#define jx_APCTLKrylovParamResetZero(apctlkrylov_param)          ((apctlkrylov_param) -> reset_zero)
#define jx_APCTLKrylovParamSolverID(apctlkrylov_param)           ((apctlkrylov_param) -> solver_id)
#define jx_APCTLKrylovParamPrecondID(apctlkrylov_param)          ((apctlkrylov_param) -> precond_id)
#define jx_APCTLKrylovParamTol(apctlkrylov_param)                ((apctlkrylov_param) -> tol)
#define jx_APCTLKrylovParamMaxIter(apctlkrylov_param)            ((apctlkrylov_param) -> max_iter)
#define jx_APCTLKrylovParamKDim(apctlkrylov_param)               ((apctlkrylov_param) -> k_dim)
#define jx_APCTLKrylovParamISCheckRestarted(apctlkrylov_param)   ((apctlkrylov_param) -> is_check_restarted)
#define jx_APCTLKrylovParamTwoNorm(apctlkrylov_param)            ((apctlkrylov_param) -> two_norm)
#define jx_APCTLKrylovParamPrintLevel(apctlkrylov_param)         ((apctlkrylov_param) -> print_level)
#define jx_APCTLKrylovParamTTest(apctlkrylov_param)              ((apctlkrylov_param) -> TTest)
#define jx_APCTLKrylovParamKeepSol(apctlkrylov_param)            ((apctlkrylov_param) -> keepsol)
#define jx_APCTLKrylovParamNumIterations(apctlkrylov_param)      ((apctlkrylov_param) -> num_iterations)
#define jx_APCTLKrylovParamLastRelNrm(apctlkrylov_param)         ((apctlkrylov_param) -> last_rel_nrm) // 2012/03/22
#define jx_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param)    ((apctlkrylov_param) -> print_level_apctl)
#define jx_APCTLKrylovParamBlockSmoothType(apctlkrylov_param)    ((apctlkrylov_param) -> blocksmooth_type)
#define jx_APCTLKrylovParamNumRelaxPre(apctlkrylov_param)        ((apctlkrylov_param) -> num_relax_pre)
#define jx_APCTLKrylovParamNumRelaxPost(apctlkrylov_param)       ((apctlkrylov_param) -> num_relax_post)
#define jx_APCTLKrylovParamInterpSolverARR(apctlkrylov_param)    ((apctlkrylov_param) -> interp_solver_ARR)
#define jx_APCTLKrylovParamInterpKdimARR(apctlkrylov_param)      ((apctlkrylov_param) -> interp_kdim_ARR)
#define jx_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param)     ((apctlkrylov_param) -> interp_maxit_ARR)
#define jx_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param)     ((apctlkrylov_param) -> interp_maxit_AII)
#define jx_APCTLKrylovParamInterpTolARR(apctlkrylov_param)       ((apctlkrylov_param) -> interp_tol_ARR)
#define jx_APCTLKrylovParamInterpTolAII(apctlkrylov_param)       ((apctlkrylov_param) -> interp_tol_AII)
#define jx_APCTLKrylovParamARRRelaxTol(apctlkrylov_param)        ((apctlkrylov_param) -> ARR_relax_tol)
#define jx_APCTLKrylovParamAEERelaxTol(apctlkrylov_param)        ((apctlkrylov_param) -> AEE_relax_tol)
#define jx_APCTLKrylovParamAIIRelaxTol(apctlkrylov_param)        ((apctlkrylov_param) -> AII_relax_tol)
#define jx_APCTLKrylovParamACCRelaxTol(apctlkrylov_param)        ((apctlkrylov_param) -> ACC_relax_tol)
#define jx_APCTLKrylovParamFixitPCTLR(apctlkrylov_param)         ((apctlkrylov_param) -> fixit_pctl_R)
#define jx_APCTLKrylovParamFixitPCTLE(apctlkrylov_param)         ((apctlkrylov_param) -> fixit_pctl_E)
#define jx_APCTLKrylovParamFixitPCTLI(apctlkrylov_param)         ((apctlkrylov_param) -> fixit_pctl_I)
#define jx_APCTLKrylovParamFixitBrlxR(apctlkrylov_param)         ((apctlkrylov_param) -> fixit_brlx_R)
#define jx_APCTLKrylovParamFixitBrlxE(apctlkrylov_param)         ((apctlkrylov_param) -> fixit_brlx_E)
#define jx_APCTLKrylovParamFixitBrlxI(apctlkrylov_param)         ((apctlkrylov_param) -> fixit_brlx_I)
#define jx_APCTLKrylovParamARRRelaxType(apctlkrylov_param)       ((apctlkrylov_param) -> ARR_relax_type)
#define jx_APCTLKrylovParamAEERelaxType(apctlkrylov_param)       ((apctlkrylov_param) -> AEE_relax_type)
#define jx_APCTLKrylovParamAIIRelaxType(apctlkrylov_param)       ((apctlkrylov_param) -> AII_relax_type)
#define jx_APCTLKrylovParamUseFixedModeR(apctlkrylov_param)      ((apctlkrylov_param) -> use_fixedmode_R)
#define jx_APCTLKrylovParamUseFixedModeE(apctlkrylov_param)      ((apctlkrylov_param) -> use_fixedmode_E)
#define jx_APCTLKrylovParamUseFixedModeI(apctlkrylov_param)      ((apctlkrylov_param) -> use_fixedmode_I)
#define jx_APCTLKrylovParamThetaWCE(apctlkrylov_param)           ((apctlkrylov_param) -> theta_wc_E)
#define jx_APCTLKrylovParamThresholdWCE(apctlkrylov_param)       ((apctlkrylov_param) -> threshold_wc_E)
#define jx_APCTLKrylovParamThetaDDR(apctlkrylov_param)           ((apctlkrylov_param) -> theta_dd_R)
#define jx_APCTLKrylovParamThetaDDE(apctlkrylov_param)           ((apctlkrylov_param) -> theta_dd_E)
#define jx_APCTLKrylovParamThetaDDI(apctlkrylov_param)           ((apctlkrylov_param) -> theta_dd_I)
#define jx_APCTLKrylovParamThresholdDDR(apctlkrylov_param)       ((apctlkrylov_param) -> threshold_dd_R)
#define jx_APCTLKrylovParamThresholdDDE(apctlkrylov_param)       ((apctlkrylov_param) -> threshold_dd_E)
#define jx_APCTLKrylovParamThresholdDDI(apctlkrylov_param)       ((apctlkrylov_param) -> threshold_dd_I)
#define jx_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param)     ((apctlkrylov_param) -> is_diagelm_first)
#define jx_APCTLKrylovParamUsePPCTL(apctlkrylov_param)           ((apctlkrylov_param) -> use_ppctl)
// peghoty, 2012/03/23
#define jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param)      ((apctlkrylov_param) -> test_subls_iter)
#define jx_APCTLKrylovParamNumIterAiSetup(apctlkrylov_param)     ((apctlkrylov_param) -> num_iter_Ai_pctl_setup)
#define jx_APCTLKrylovParamNumIterArSetup(apctlkrylov_param)     ((apctlkrylov_param) -> num_iter_Ar_pctl_setup)
#define jx_APCTLKrylovParamNumIterAePrecond(apctlkrylov_param)   ((apctlkrylov_param) -> num_iter_Ae_pctl_precond)
#define jx_APCTLKrylovParamNumIterAiPrecond(apctlkrylov_param)   ((apctlkrylov_param) -> num_iter_Ai_pctl_precond)
#define jx_APCTLKrylovParamNumIterArPrecond(apctlkrylov_param)   ((apctlkrylov_param) -> num_iter_Ar_pctl_precond)
#define jx_APCTLKrylovParamNumIterAcPrecond(apctlkrylov_param)   ((apctlkrylov_param) -> num_iter_Ac_pctl_precond)
#define jx_APCTLKrylovParamNeedCC(apctlkrylov_param)             ((apctlkrylov_param) -> Need_CC)
// peghoty, 2012/03/25
#define jx_APCTLKrylovParamCPUTrans(apctlkrylov_param)           ((apctlkrylov_param) -> cpu_trans)
#define jx_APCTLKrylovParamCPUSetup(apctlkrylov_param)           ((apctlkrylov_param) -> cpu_setup)
#define jx_APCTLKrylovParamCPUSolve(apctlkrylov_param)           ((apctlkrylov_param) -> cpu_solve)
#define jx_APCTLKrylovParamCPUTotal(apctlkrylov_param)           ((apctlkrylov_param) -> cpu_total)
#define jx_APCTLKrylovParamDebugFlag(apctlkrylov_param)          ((apctlkrylov_param) -> debug_flag)
#define jx_APCTLKrylovParamStrongThreshold(apctlkrylov_param)    ((apctlkrylov_param) -> strong_threshold)
#define jx_APCTLKrylovParamInterpType(apctlkrylov_param)         ((apctlkrylov_param) -> interp_type)
#define jx_APCTLKrylovParamCoarsenType(apctlkrylov_param)        ((apctlkrylov_param) -> coarsen_type)
#define jx_APCTLKrylovParamAggNumLevels(apctlkrylov_param)       ((apctlkrylov_param) -> agg_num_levels)
#define jx_APCTLKrylovParamCoarseThreshold(apctlkrylov_param)    ((apctlkrylov_param) -> coarse_threshold)
#define jx_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param)    ((apctlkrylov_param) -> print_level_amg)

/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/
 
/* /csrc/3t/subvec.c */
JX_Int
jx_3tGetSubVecs( jx_Vector  *f, 
                 jx_Vector **fR_ptr, 
                 jx_Vector **fE_ptr, 
                 jx_Vector **fI_ptr );
JX_Int
jx_mgGetSubVecs( jx_Vector  *f,
                 jx_Vector **fR,
                 jx_Vector **fE_ptr,
                 jx_Vector **fI_ptr,
                 JX_Int      ng );

/* /csrc/3t/submat.c */                   
JX_Int
jx_3tGetSubBlocks( jx_CSRMatrix  *A, 
                   jx_CSRMatrix **ARR_ptr, 
                   jx_CSRMatrix **AEE_ptr, 
                   jx_CSRMatrix **AII_ptr, 
                   jx_CSRMatrix **ARE_ptr, 
                   jx_CSRMatrix **AER_ptr, 
                   jx_CSRMatrix **AEI_ptr, 
                   jx_CSRMatrix **AIE_ptr );
JX_Int
jx_3tGetSubBlocks_REIV( jx_CSRMatrix  *A, 
                        jx_CSRMatrix **ARR_ptr, 
                        jx_CSRMatrix **AEE_ptr, 
                        jx_CSRMatrix **AII_ptr, 
                        jx_Vector    **VRE_ptr, 
                        jx_Vector    **VER_ptr, 
                        jx_Vector    **VEI_ptr, 
                        jx_Vector    **VIE_ptr );
JX_Int
jx_mgGetSubBlocks_REIV( jx_CSRMatrix  *A,
                        jx_CSRMatrix **ARRarray,
                        jx_CSRMatrix **AEE_ptr,
                        jx_CSRMatrix **AII_ptr,
                        jx_Vector    **VREarray,
                        jx_Vector    **VERarray,
                        jx_Vector    **VEI_ptr,
                        jx_Vector    **VIE_ptr,
                        JX_Int         ng );
JX_Int
jx_3tGetSubDiagBlocks( jx_CSRMatrix  *A, 
                       jx_CSRMatrix **ARR_ptr, 
                       jx_CSRMatrix **AEE_ptr, 
                       jx_CSRMatrix **AII_ptr );                                        
JX_Int
jx_3tGetSubBlocks_R( jx_CSRMatrix  *A, 
                     jx_CSRMatrix **ARR_ptr, 
                     jx_CSRMatrix **ARE_ptr );
JX_Int
jx_3tGetSubBlocks_RV( jx_CSRMatrix  *A, 
                      jx_CSRMatrix **ARR_ptr, 
                      jx_Vector    **VRE_ptr );                                   
JX_Int
jx_3tGetSubBlocks_E( jx_CSRMatrix  *A, 
                     jx_CSRMatrix **AEE_ptr, 
                     jx_CSRMatrix **AER_ptr, 
                     jx_CSRMatrix **AEI_ptr );
JX_Int
jx_3tGetSubBlocks_EV( jx_CSRMatrix  *A, 
                      jx_CSRMatrix **AEE_ptr, 
                      jx_Vector    **VER_ptr, 
                      jx_Vector    **VEI_ptr );                   
JX_Int
jx_3tGetSubBlocks_I( jx_CSRMatrix  *A, 
                     jx_CSRMatrix **AII_ptr, 
                     jx_CSRMatrix **AIE_ptr );
JX_Int
jx_3tGetSubBlocks_IV( jx_CSRMatrix  *A, 
                      jx_CSRMatrix **AII_ptr, 
                      jx_Vector    **VIE_ptr );
JX_Int jx_3tGetSubBlocks_ARR( jx_CSRMatrix *A, jx_CSRMatrix **ARR_ptr );                 
JX_Int jx_3tGetSubBlocks_AEE( jx_CSRMatrix *A, jx_CSRMatrix **AEE_ptr );
JX_Int jx_3tGetSubBlocks_AII( jx_CSRMatrix *A, jx_CSRMatrix **AII_ptr );                    
JX_Int jx_3tGetSubBlocks_DII( jx_CSRMatrix *A, jx_Vector **DII_ptr );  
jx_ParCSRMatrix *jx_3tGetAD( jx_ParCSRMatrix *A, JX_Int AII_all );
JX_Int jx_3tAPCTLDDCheck( JX_Real theta_dd, JX_Real threshold_dd, jx_ParCSRMatrix *A );
JX_Int 
jx_3tDDCheckParallel( jx_CSRMatrix  *A_diag, 
                      jx_CSRMatrix  *A_offd, 
                      JX_Real         theta_dd, 
                      JX_Real         threshold_dd );                                            

JX_Int 
jx_3tAPCTLWeakCouplingE( JX_Real            theta_wc_E, 
                         JX_Real            threshold_wc_E, 
                         jx_ParCSRMatrix  *AEE, 
                         jx_ParVector     *VER, 
                         jx_ParVector     *VEI );
JX_Int 
jx_3tAPCTLmgWeakCouplingE( JX_Real            theta_wc_E, 
                           JX_Real            threshold_wc_E, 
                           JX_Int            ng,
                           jx_ParCSRMatrix  *AEE, 
                           jx_ParVector    **VER, 
                           jx_ParVector     *VEI );
JX_Int 
jx_3tWeakCouplingParallel( jx_ParCSRMatrix  *A, 
                           jx_ParVector     *V,
                           JX_Real            theta_wc );                     
JX_Int
jx_3tAPCTLWCDD( jx_3tAPCTLData   *pre_3tapctl_data,
                MPI_Comm          comm,
                jx_CSRMatrix      *ARR, 
                jx_CSRMatrix      *AEE,
                jx_CSRMatrix      *AII,
                jx_Vector         *VRE,
                jx_Vector         *VER,
                jx_Vector         *VEI,
                jx_Vector         *VIE  );
JX_Int
jx_3tWeakCoupling( jx_CSRMatrix  *A, 
                   jx_Vector     *V, 
                   JX_Real         theta_wc, 
                   JX_Real         threshold_wc );
JX_Int 
jx_3tDiagDominant( jx_CSRMatrix  *A,
                   JX_Real         theta_dd,
                   JX_Real         threshold_dd );
JX_Int
jx_BSCModifySubMat( jx_ParCSRMatrix *A, jx_ParVector *V, JX_Real *TEMP, JX_Int hsize );
JX_Int
jx_BSCModify2SubMat( jx_ParCSRMatrix *A, jx_ParVector *V, JX_Real *sTEMP, JX_Real *vTEMP, JX_Int hsize );

/* /csrc/3t/transf.c */                 
JX_Int 
jx_APCTLKrylovSolBack4Jasmin( MPI_Comm      comm,
                              MPI_Comm      comm_x, 
                              JX_Int           groupid_x, 
                              jx_ParVector *par_sol, 
                              jx_ParVector *uR_p, 
                              jx_ParVector *uE_p, 
                              jx_ParVector *uI_p );
JX_Int 
jx_APCTLKrylovSolBack4mgJasmin( MPI_Comm      comm,
                                MPI_Comm      comm_x,
                                JX_Int           groupid_x,
                                JX_Int           ng,
                                jx_ParVector  *par_sol,
                                jx_ParVector **uR_p,
                                jx_ParVector  *uE_p,
                                jx_ParVector  *uI_p );
JX_Int 
jx_APCTLKrylovSolBack4Jasmin2( MPI_Comm      comm,
                               jx_ParVector *par_sol, 
                               jx_ParVector *uR_p, 
                               jx_ParVector *uE_p, 
                               jx_ParVector *uI_p );
JX_Int 
jx_MatVecGroupR( MPI_Comm           comm_x,
                 jx_ParCSRMatrix   *A, 
                 jx_ParCSRMatrix  **ARR_ptr, 
                 jx_ParVector     **VRE_ptr );
JX_Int 
jx_MatGroupR( MPI_Comm           comm_x,
              jx_ParCSRMatrix   *A, 
              jx_ParCSRMatrix  **ARR_ptr );
JX_Int 
jx_MatVecGroupI( MPI_Comm           comm_x,
                 jx_ParCSRMatrix   *A, 
                 jx_ParCSRMatrix  **AII_ptr, 
                 jx_ParVector     **VIE_ptr );
JX_Int 
jx_MatGroupI( MPI_Comm           comm_x,
              jx_ParCSRMatrix   *A, 
              jx_ParCSRMatrix  **AII_ptr );                                
JX_Int 
jx_MatVecGroupE( MPI_Comm           comm_x,
                 jx_ParCSRMatrix   *A, 
                 jx_ParCSRMatrix  **AEE_ptr, 
                 jx_ParVector     **VER_ptr,
                 jx_ParVector     **VEI_ptr );
JX_Int 
jx_DataCombine4ApctlKrylov( // input:
                            jx_ParCSRMatrix   *ARR_p, 
                            jx_ParCSRMatrix   *AEE_p, 
                            jx_ParCSRMatrix   *AII_p, 
                            jx_ParVector      *VRE_p, 
                            jx_ParVector      *VER_p, 
                            jx_ParVector      *VEI_p, 
                            jx_ParVector      *VIE_p, 
                            jx_ParVector      *fR_p, 
                            jx_ParVector      *fE_p, 
                            jx_ParVector      *fI_p,
                            jx_ParVector      *uR_p, 
                            jx_ParVector      *uE_p, 
                            jx_ParVector      *uI_p,
                            // output:    
                            jx_ParCSRMatrix  **A_ptr,  
                            jx_ParVector     **f_ptr, 
                            jx_ParVector     **u_ptr );
JX_Int 
jx_3tGlobalSystem( jx_CSRMatrix  *ARR, 
                   jx_CSRMatrix  *AEE, 
                   jx_CSRMatrix  *AII, 
                   jx_Vector     *VRE, 
                   jx_Vector     *VER, 
                   jx_Vector     *VEI, 
                   jx_Vector     *VIE, 
                   jx_Vector     *fR, 
                   jx_Vector     *fE, 
                   jx_Vector     *fI,
                   jx_Vector     *uR, 
                   jx_Vector     *uE, 
                   jx_Vector     *uI,                       
                   jx_CSRMatrix **A_ptr, 
                   jx_Vector    **f_ptr,
                   jx_Vector    **u_ptr );
JX_Int 
jx_ParaDataTrans4ApctlKrylov( // input:
                              MPI_Comm           comm, 
                              MPI_Comm           comm_x, 
                              JX_Int                groupid_x,
                              jx_ParCSRMatrix   *ARR_p, 
                              jx_ParCSRMatrix   *AEE_p, 
                              jx_ParCSRMatrix   *AII_p, 
                              jx_ParVector      *VRE_p, 
                              jx_ParVector      *VER_p, 
                              jx_ParVector      *VEI_p, 
                              jx_ParVector      *VIE_p, 
                              jx_ParVector      *fR_p, 
                              jx_ParVector      *fE_p, 
                              jx_ParVector      *fI_p,
                              jx_ParVector      *uR_p, 
                              jx_ParVector      *uE_p, 
                              jx_ParVector      *uI_p,
                              // output:    
                              jx_ParCSRMatrix  **ARR_ptr,
                              jx_ParCSRMatrix  **AEE_ptr,
                              jx_ParCSRMatrix  **AII_ptr, 
                              jx_ParVector     **VRE_ptr, 
                              jx_ParVector     **VER_ptr, 
                              jx_ParVector     **VEI_ptr, 
                              jx_ParVector     **VIE_ptr,
                              jx_ParCSRMatrix  **A_ptr,  
                              jx_ParVector     **f_ptr, 
                              jx_ParVector     **u_ptr );
JX_Int 
jx_ParaDataTrans4ApctlmgKrylov( // input:
                                MPI_Comm           comm, 
                                MPI_Comm           comm_x, 
                                JX_Int                groupid_x,
                                jx_ParCSRMatrix  **ARR_p, 
                                jx_ParCSRMatrix   *AEE_p, 
                                jx_ParCSRMatrix   *AII_p, 
                                jx_ParVector     **VRE_p, 
                                jx_ParVector     **VER_p, 
                                jx_ParVector      *VEI_p, 
                                jx_ParVector      *VIE_p, 
                                jx_ParVector     **fR_p, 
                                jx_ParVector      *fE_p, 
                                jx_ParVector      *fI_p,
                                jx_ParVector     **uR_p, 
                                jx_ParVector      *uE_p, 
                                jx_ParVector      *uI_p,
                                JX_Int             ng,
                                // output:    
                                jx_ParCSRMatrix  **ARR_ptr,
                                jx_ParCSRMatrix  **AEE_ptr,
                                jx_ParCSRMatrix  **AII_ptr, 
                                jx_ParVector     **VRE_ptr, 
                                jx_ParVector    ***VER_ptr, 
                                jx_ParVector     **VEI_ptr, 
                                jx_ParVector     **VIE_ptr,
                                jx_ParCSRMatrix  **A_ptr,  
                                jx_ParVector     **f_ptr, 
                                jx_ParVector     **u_ptr );
JX_Int 
jx_MatVecGroup2All( MPI_Comm          comm, 
                    JX_Int               groupid_x,
                    jx_ParCSRMatrix  *ARR, 
                    jx_ParCSRMatrix  *AEE, 
                    jx_ParCSRMatrix  *AII, 
                    jx_ParVector     *VRE, 
                    jx_ParVector     *VER, 
                    jx_ParVector     *VEI, 
                    jx_ParVector     *VIE,
                    jx_ParCSRMatrix **ARR_all_ptr, 
                    jx_ParCSRMatrix **AEE_all_ptr, 
                    jx_ParCSRMatrix **AII_all_ptr, 
                    jx_ParVector    **VRE_all_ptr, 
                    jx_ParVector    **VER_all_ptr, 
                    jx_ParVector    **VEI_all_ptr, 
                    jx_ParVector    **VIE_all_ptr );
JX_Int
jx_3tDataTransFromSeq2SubPara( JX_Int               iniguess,
                               MPI_Comm          comm,
                               jx_CSRMatrix     *A_s, 
                               jx_Vector        *f_s, 
                               jx_Vector        *u_s, 
                               jx_ParCSRMatrix **ARR_p_ptr, 
                               jx_ParCSRMatrix **AEE_p_ptr, 
                               jx_ParCSRMatrix **AII_p_ptr, 
                               jx_ParVector    **VRE_p_ptr, 
                               jx_ParVector    **VER_p_ptr, 
                               jx_ParVector    **VEI_p_ptr, 
                               jx_ParVector    **VIE_p_ptr, 
                               jx_ParVector    **fR_p_ptr, 
                               jx_ParVector    **fE_p_ptr, 
                               jx_ParVector    **fI_p_ptr,
                               jx_ParVector    **uR_p_ptr, 
                               jx_ParVector    **uE_p_ptr, 
                               jx_ParVector    **uI_p_ptr );
JX_Int
jx_3tDataTransFromSeq2SubPar0( JX_Int               iniguess,
                               MPI_Comm          comm,
                               jx_CSRMatrix     *A_s, 
                               jx_Vector        *f_s, 
                               jx_Vector        *u_s, 
                               jx_ParCSRMatrix **ARR_p_ptr, 
                               jx_ParCSRMatrix **AEE_p_ptr, 
                               jx_ParCSRMatrix **AII_p_ptr, 
                               jx_ParVector    **VRE_p_ptr, 
                               jx_ParVector    **VER_p_ptr, 
                               jx_ParVector    **VEI_p_ptr, 
                               jx_ParVector    **VIE_p_ptr, 
                               jx_ParVector    **fR_p_ptr, 
                               jx_ParVector    **fE_p_ptr, 
                               jx_ParVector    **fI_p_ptr,
                               jx_ParVector    **uR_p_ptr, 
                               jx_ParVector    **uE_p_ptr, 
                               jx_ParVector    **uI_p_ptr );                              
JX_Int  
jx_ParaDataTransSEQIF_mp( // input:
                          MPI_Comm           comm, 
                          MPI_Comm           comm_x, 
                          JX_Int                groupid_x,
                          JX_Real             theta_wc_E, 
                          JX_Real             threshold_wc_E,
                          /* 串行离散系统数据 */
                          jx_CSRMatrix      *A_s, 
                          jx_Vector         *f_s, 
                          jx_Vector         *u_s, 
                          // output: 
                          /* 系数矩阵各子块在整个进程组上的并行数据 */
                          jx_ParCSRMatrix  **ARR_p_ptr, 
                          jx_ParCSRMatrix  **AEE_p_ptr, 
                          jx_ParCSRMatrix  **AII_p_ptr, 
                          jx_ParVector     **VRE_p_ptr, 
                          jx_ParVector     **VER_p_ptr, 
                          jx_ParVector     **VEI_p_ptr, 
                          jx_ParVector     **VIE_p_ptr, 
                          /* 系数矩阵各子块基于进程组的并行数据 */                         
                          jx_ParCSRMatrix  **ARR_ptr,
                          jx_ParCSRMatrix  **AEE_ptr,
                          jx_ParCSRMatrix  **AII_ptr, 
                          jx_ParVector     **VRE_ptr, 
                          jx_ParVector     **VER_ptr, 
                          jx_ParVector     **VEI_ptr, 
                          jx_ParVector     **VIE_ptr,
                          /* 并行离散系统数据 */
                          jx_ParCSRMatrix  **A_ptr,  
                          jx_ParVector     **f_ptr, 
                          jx_ParVector     **u_ptr,
                          /* 是否需要粗网格校正的标志变量 */
                          JX_Int               *Need_CC_ptr );
JX_Int  
jx_ParaDataTransSEQIF_sp( // input:
                          MPI_Comm           comm, 
                          /* 串行离散系统数据 */
                          jx_CSRMatrix      *A_s, 
                          jx_Vector         *f_s, 
                          jx_Vector         *u_s, 
                          // output:
                          /* 系数矩阵各子块基于进程组的并行数据 */                         
                          jx_ParCSRMatrix  **ARR_ptr,
                          jx_ParCSRMatrix  **AEE_ptr,
                          jx_ParCSRMatrix  **AII_ptr, 
                          jx_ParVector     **VRE_ptr, 
                          jx_ParVector     **VER_ptr, 
                          jx_ParVector     **VEI_ptr, 
                          jx_ParVector     **VIE_ptr,
                          /* 并行离散系统数据 */
                          jx_ParCSRMatrix  **A_ptr,  
                          jx_ParVector     **f_ptr, 
                          jx_ParVector     **u_ptr );
JX_Int jx_APCTLKrylovSolGather( jx_ParVector *par_sol, jx_Vector *u_s );
JX_Int 
jx_mgGenerateSubBlocks( MPI_Comm           comm,
                        MPI_Comm           comm_x,
                        JX_Int             groupid_x,
                        JX_Int             ng,
                        jx_ParCSRMatrix   *par_mat,
                        jx_ParCSRMatrix  **ARR_ptr,
                        jx_ParCSRMatrix  **AEE_ptr,
                        jx_ParCSRMatrix  **AII_ptr,
                        jx_ParVector     **VRE_ptr,
                        jx_ParVector    ***VER_ptr,
                        jx_ParVector     **VEI_ptr,
                        jx_ParVector     **VIE_ptr );
                          
                          

/* /csrc/3t/partition.c */
JX_Int jx_3tCheckNprocs( MPI_Comm comm, JX_Int print_level, JX_Int *np_R_ptr, JX_Int *np_E_ptr, JX_Int *np_I_ptr );
JX_Int jx_3tGetMyPartition( JX_Int nprocs, JX_Int np_R, JX_Int np_E, JX_Int np_I, JX_Int N, JX_Int **mypartition_ptr );   



/* /csrc/3t/apctl.c */
JX_Int JX_3tAPCTLSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetMaxiter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetNumRlxPre( JX_Solver solver, JX_Int num_relax );
JX_Int JX_3tAPCTLSetNumRlxPost( JX_Solver solver, JX_Int num_relax );
JX_Int JX_3tAPCTLSetPrintLevel( JX_Solver solver, JX_Int print_level );
JX_Int JX_3tAPCTLSetIsDiagElmFirst( JX_Solver solver, JX_Int is_diagelm_first );
JX_Int JX_3tAPCTLSetNumGroup( JX_Solver solver, JX_Int num_group );
JX_Int JX_3tAPCTLSetNumIterations( JX_Solver solver, JX_Int num_iterations );
JX_Int JX_3tAPCTLSetLastRelNrm( JX_Solver solver, JX_Real last_rel_nrm );
JX_Int JX_3tAPCTLSetAveConvFactor( JX_Solver solver, JX_Real ave_conv_factor );
JX_Int JX_3tAPCTLGetNumIterations( JX_Solver solver, JX_Int *num_iterations );
JX_Int JX_3tAPCTLGetLastRelNrm( JX_Solver solver, JX_Real *rel_resid_norm );
JX_Int JX_3tAPCTLSetNpR( JX_Solver solver, JX_Int np );
JX_Int JX_3tAPCTLSetNpE( JX_Solver solver, JX_Int np );
JX_Int JX_3tAPCTLSetNpI( JX_Solver solver, JX_Int np );
JX_Int JX_3tAPCTLSetARRInterpMaxIt( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetAIIInterpMaxIt( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetARRRelaxMaxIt( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetAEERelaxMaxIt( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetAIIRelaxMaxIt( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetACCRelaxMaxIt( JX_Solver solver, JX_Int max_iter );
JX_Int JX_3tAPCTLSetARRInterpTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetAIIInterpTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetARRRelaxTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetAEERelaxTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetAIIRelaxTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetACCRelaxTol( JX_Solver solver, JX_Real tol );
JX_Int JX_3tAPCTLSetARRRelaxType( JX_Solver solver, JX_Int ARR_relax_type );
JX_Int JX_3tAPCTLSetAEERelaxType( JX_Solver solver, JX_Int AEE_relax_type );
JX_Int JX_3tAPCTLSetAIIRelaxType( JX_Solver solver, JX_Int AII_relax_type );
JX_Int JX_3tAPCTLSetThetaWCR( JX_Solver solver, JX_Real theta );
JX_Int JX_3tAPCTLSetThetaWCE( JX_Solver solver, JX_Real theta ); 
JX_Int JX_3tAPCTLSetThetaWCI( JX_Solver solver, JX_Real theta ); 
JX_Int JX_3tAPCTLSetThresholdWCR( JX_Solver solver, JX_Real threshold );
JX_Int JX_3tAPCTLSetThresholdWCE( JX_Solver solver, JX_Real threshold );
JX_Int JX_3tAPCTLSetThresholdWCI( JX_Solver solver, JX_Real threshold );
JX_Int JX_3tAPCTLSetISWCR( JX_Solver solver, JX_Int flag );
JX_Int JX_3tAPCTLSetISWCE( JX_Solver solver, JX_Int flag );
JX_Int JX_3tAPCTLSetISWCI( JX_Solver solver, JX_Int flag );
JX_Int JX_3tAPCTLSetThetaDDR( JX_Solver solver, JX_Real theta ); 
JX_Int JX_3tAPCTLSetThetaDDE( JX_Solver solver, JX_Real theta );  
JX_Int JX_3tAPCTLSetThetaDDI( JX_Solver solver, JX_Real theta );  
JX_Int JX_3tAPCTLSetThresholdDDR( JX_Solver solver, JX_Real threshold ); 
JX_Int JX_3tAPCTLSetThresholdDDE( JX_Solver solver, JX_Real threshold );
JX_Int JX_3tAPCTLSetThresholdDDI( JX_Solver solver, JX_Real threshold );   
JX_Int JX_3tAPCTLSetISDDR( JX_Solver solver, JX_Int flag );
JX_Int JX_3tAPCTLSetISDDE( JX_Solver solver, JX_Int flag ); 
JX_Int JX_3tAPCTLSetISDDI( JX_Solver solver, JX_Int flag );
JX_Int JX_3tAPCTLSetNeedCC( JX_Solver solver, JX_Int flag );
JX_Int JX_3tAPCTLSetBlockSmoothType( JX_Solver solver, JX_Int blocksmooth_type );
JX_Int JX_3tAPCTLSetFixItPCTLR( JX_Solver solver, JX_Int fixit );
JX_Int JX_3tAPCTLSetFixItPCTLE( JX_Solver solver, JX_Int fixit );
JX_Int JX_3tAPCTLSetFixItPCTLI( JX_Solver solver, JX_Int fixit );
JX_Int JX_3tAPCTLSetFixItBRLXR( JX_Solver solver, JX_Int fixit );
JX_Int JX_3tAPCTLSetFixItBRLXE( JX_Solver solver, JX_Int fixit );
JX_Int JX_3tAPCTLSetFixItBRLXI( JX_Solver solver, JX_Int fixit );
JX_Int JX_3tAPCTLSetUseFixedModeR( JX_Solver solver, JX_Int usefixedmode ); /* peghoty  2012/02/15 */
JX_Int JX_3tAPCTLSetUseFixedModeE( JX_Solver solver, JX_Int usefixedmode ); /* peghoty  2012/02/15 */
JX_Int JX_3tAPCTLSetUseFixedModeI( JX_Solver solver, JX_Int usefixedmode ); /* peghoty  2012/02/15 */
JX_Int JX_3tAPCTLSetUsePPCTL( JX_Solver solver, JX_Int use_ppctl );   /* peghoty  2012/03/06 */
JX_Int JX_3tAPCTLSetARRSolverID( JX_Solver solver, JX_Int solverid ); /* peghoty  2011/10/29 */
JX_Int JX_3tAPCTLSetACCSolverID( JX_Solver solver, JX_Int solverid ); /* peghoty  2011/10/29 */
JX_Int JX_3tAPCTLSetARRKDim( JX_Solver solver, JX_Int kdim ); /* peghoty  2011/10/29 */
JX_Int JX_3tAPCTLSetACCKDim( JX_Solver solver, JX_Int kdim ); /* peghoty  2011/10/29 */
JX_Int JX_3tAPCTLSetTestSubLSIter( JX_Solver solver, JX_Int test_subls_iter ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetDebugFlag( JX_Solver solver, JX_Int debug_flag );
JX_Int JX_3tAPCTLSetResetZero( JX_Solver solver, JX_Int reset_zero );
JX_Int JX_3tAPCTLSetInterpType( JX_Solver solver, JX_Int interp_type );
JX_Int JX_3tAPCTLSetCoarsenType( JX_Solver solver, JX_Int coarsen_type );
JX_Int JX_3tAPCTLSetAggNumLevels( JX_Solver solver, JX_Int agg_num_levels );
JX_Int JX_3tAPCTLSetCoarseThreshold( JX_Solver solver, JX_Int coarse_threshold );
JX_Int JX_3tAPCTLSetPrintLevelAMG( JX_Solver solver, JX_Int print_level_amg );
JX_Int JX_3tAPCTLSetStrongThreshold( JX_Solver solver, JX_Real strong_threshold );
JX_Int JX_3tAPCTLSetNumIterAiSetup( JX_Solver solver, JX_Int num_iter_Ai_pctl_setup ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetNumIterArSetup( JX_Solver solver, JX_Int num_iter_Ar_pctl_setup ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetNumIterAePrecond( JX_Solver solver, JX_Int num_iter_Ae_pctl_precond ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetNumIterAiPrecond( JX_Solver solver, JX_Int num_iter_Ai_pctl_precond ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetNumIterArPrecond( JX_Solver solver, JX_Int num_iter_Ar_pctl_precond ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetNumIterAcPrecond( JX_Solver solver, JX_Int num_iter_Ac_pctl_precond ); // peghoty, 2012/02/23
JX_Int JX_3tAPCTLSetA( JX_Solver solver, jx_ParCSRMatrix *A );
JX_Int JX_3tAPCTLSetSubBlocks( JX_Solver solver );
JX_Int JX_3tAPCTLSetARR( JX_Solver solver, jx_ParCSRMatrix *ARR );
JX_Int JX_3tAPCTLSetAEE( JX_Solver solver, jx_ParCSRMatrix *AEE );
JX_Int JX_3tAPCTLSetAII( JX_Solver solver, jx_ParCSRMatrix *AII );
JX_Int JX_3tAPCTLSetVRE( JX_Solver solver, jx_ParVector *VRE );
JX_Int JX_3tAPCTLSetVER( JX_Solver solver, jx_ParVector *VER );
JX_Int JX_3tAPCTLSetVER2( JX_Solver solver, jx_ParVector **VER );
JX_Int JX_3tAPCTLSetVEI( JX_Solver solver, jx_ParVector *VEI );
JX_Int JX_3tAPCTLSetVIE( JX_Solver solver, jx_ParVector *VIE );
JX_Int JX_3tAPCTLSetARRAll( JX_Solver solver, jx_ParCSRMatrix *ARR );
JX_Int JX_3tAPCTLSetARRAll2( JX_Solver solver, jx_ParCSRMatrix **ARR );
JX_Int JX_3tAPCTLSetAEEAll( JX_Solver solver, jx_ParCSRMatrix *AEE );
JX_Int JX_3tAPCTLSetAIIAll( JX_Solver solver, jx_ParCSRMatrix *AII );
JX_Int JX_3tAPCTLSetVREAll( JX_Solver solver, jx_ParVector *VRE );
JX_Int JX_3tAPCTLSetVREAll2( JX_Solver solver, jx_ParVector **VRE );
JX_Int JX_3tAPCTLSetVERAll( JX_Solver solver, jx_ParVector *VER );
JX_Int JX_3tAPCTLSetVERAll2( JX_Solver solver, jx_ParVector **VER );
JX_Int JX_3tAPCTLSetVEIAll( JX_Solver solver, jx_ParVector *VEI );
JX_Int JX_3tAPCTLSetVIEAll( JX_Solver solver, jx_ParVector *VIE );
JX_Int JX_3tAPCTLIterCount( JX_Solver solver );
JX_Int JX_3tAPCTLSetComm( JX_Solver solver, MPI_Comm comm );
JX_Int JX_3tAPCTLSetCommX( JX_Solver solver, MPI_Comm comm );
JX_Int JX_3tAPCTLSetCommY( JX_Solver solver, MPI_Comm comm );
JX_Int JX_3tAPCTLSetGroupIdX( JX_Solver solver, JX_Int groupid );
JX_Int JX_3tAPCTLSetGroupIdY( JX_Solver solver, JX_Int groupid );
JX_Int JX_3tAPCTLDataInitialize( JX_Solver *solver);
JX_Int JX_3tAPCTLDestroy( JX_Solver solver );
JX_Int jx_3tAPCTLSetTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetMaxiter( void *solver, JX_Int max_iter );       
JX_Int jx_3tAPCTLSetNumRlxPre( void *solver, JX_Int num_relax );
JX_Int jx_3tAPCTLSetNumRlxPost( void *solver, JX_Int num_relax );
JX_Int jx_3tAPCTLSetPrintLevel( void *solver, JX_Int print_level );
JX_Int jx_3tAPCTLSetIsDiagElmFirst( void *solver, JX_Int is_diagelm_first );
JX_Int jx_3tAPCTLSetNumGroup( void *solver, JX_Int num_group );
JX_Int jx_3tAPCTLSetNumIterations( void *solver, JX_Int num_iterations );
JX_Int jx_3tAPCTLSetLastRelNrm( void *solver, JX_Real last_rel_nrm );
JX_Int jx_3tAPCTLSetAveConvFactor( void *solver, JX_Real ave_conv_factor );
JX_Int jx_3tAPCTLGetNumIterations( void *solver, JX_Int *num_iterations );
JX_Int jx_3tAPCTLGetLastRelNrm( void *solver, JX_Real *rel_resid_norm );
JX_Int jx_3tAPCTLSetNpR( void *solver, JX_Int np );
JX_Int jx_3tAPCTLSetNpE( void *solver, JX_Int np );
JX_Int jx_3tAPCTLSetNpI( void *solver, JX_Int np );
JX_Int jx_3tAPCTLSetARRInterpMaxIt( void *solver, JX_Int max_iter );
JX_Int jx_3tAPCTLSetAIIInterpMaxIt( void *solver, JX_Int max_iter );
JX_Int jx_3tAPCTLSetARRRelaxMaxIt( void *solver, JX_Int max_iter );
JX_Int jx_3tAPCTLSetAEERelaxMaxIt( void *solver, JX_Int max_iter );
JX_Int jx_3tAPCTLSetAIIRelaxMaxIt( void *solver, JX_Int max_iter );
JX_Int jx_3tAPCTLSetACCRelaxMaxIt( void *solver, JX_Int max_iter );
JX_Int jx_3tAPCTLSetMaxItDefault( void *solver, JX_Int maxit_default );
JX_Int jx_3tAPCTLSetARRInterpTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetAIIInterpTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetARRRelaxTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetAEERelaxTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetAIIRelaxTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetACCRelaxTol( void *solver, JX_Real tol );
JX_Int jx_3tAPCTLSetARRRelaxType( void *solver, JX_Int ARR_relax_type );
JX_Int jx_3tAPCTLSetAEERelaxType( void *solver, JX_Int AEE_relax_type );
JX_Int jx_3tAPCTLSetAIIRelaxType( void *solver, JX_Int AII_relax_type );
JX_Int jx_3tAPCTLSetTolDefault( void *solver, JX_Real tol_default );
JX_Int jx_3tAPCTLSetThetaWCR( void *solver, JX_Real theta );
JX_Int jx_3tAPCTLSetThetaWCE( void *solver, JX_Real theta );
JX_Int jx_3tAPCTLSetThetaWCI( void *solver, JX_Real theta );
JX_Int jx_3tAPCTLSetThresholdWCR( void *solver, JX_Real threshold );
JX_Int jx_3tAPCTLSetThresholdWCE( void *solver, JX_Real threshold );
JX_Int jx_3tAPCTLSetThresholdWCI( void *solver, JX_Real threshold );
JX_Int jx_3tAPCTLSetISWCR( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetISWCE( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetISWCI( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetThetaDDR( void *solver, JX_Real theta );
JX_Int jx_3tAPCTLSetThetaDDE( void *solver, JX_Real theta );
JX_Int jx_3tAPCTLSetThetaDDI( void *solver, JX_Real theta );
JX_Int jx_3tAPCTLSetThresholdDDR( void *solver, JX_Real threshold );
JX_Int jx_3tAPCTLSetThresholdDDE( void *solver, JX_Real threshold );
JX_Int jx_3tAPCTLSetThresholdDDI( void *solver, JX_Real threshold );
JX_Int jx_3tAPCTLSetISDDR( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetISDDE( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetISDDI( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetNeedCC( void *solver, JX_Int flag );
JX_Int jx_3tAPCTLSetBlockSmoothType( void *solver, JX_Int blocksmooth_type );
JX_Int jx_3tAPCTLSetFixItPCTLR( void *solver, JX_Int fixit );
JX_Int jx_3tAPCTLSetFixItPCTLE( void *solver, JX_Int fixit );
JX_Int jx_3tAPCTLSetFixItPCTLI( void *solver, JX_Int fixit );
JX_Int jx_3tAPCTLSetFixItBRLXR( void *solver, JX_Int fixit );
JX_Int jx_3tAPCTLSetFixItBRLXE( void *solver, JX_Int fixit );
JX_Int jx_3tAPCTLSetFixItBRLXI( void *solver, JX_Int fixit );
JX_Int jx_3tAPCTLSetUseFixedModeR( void *solver, JX_Int usefixedmode ); // peghoty, 2012/02/15
JX_Int jx_3tAPCTLSetUseFixedModeE( void *solver, JX_Int usefixedmode ); // peghoty, 2012/02/15
JX_Int jx_3tAPCTLSetUseFixedModeI( void *solver, JX_Int usefixedmode ); // peghoty, 2012/02/15
JX_Int jx_3tAPCTLSetUsePPCTL( void *solver, JX_Int use_ppctl ); // peghoty, 2012/03/06
JX_Int jx_APCTLKrylovParamSetDebugFlag( void *param, JX_Int debug_flag );
JX_Int jx_APCTLKrylovParamSetResetZero( void *param, JX_Int reset_zero );
JX_Int jx_APCTLKrylovParamSetStrongThreshold( void *param, JX_Real strong_threshold );
JX_Int jx_APCTLKrylovParamSetInterpType( void *param, JX_Int interp_type );
JX_Int jx_APCTLKrylovParamSetCoarsenType( void *param, JX_Int coarsen_type );
JX_Int jx_APCTLKrylovParamSetAggNumLevels( void *param, JX_Int agg_num_levels );
JX_Int jx_APCTLKrylovParamSetCoarseThreshold( void *param, JX_Int coarse_threshold );
JX_Int jx_APCTLKrylovParamSetPrintLevelAMG( void *param, JX_Int print_level_amg );
JX_Int jx_3tAPCTLSetARRSolverID( void *solver, JX_Int solverid );
JX_Int jx_3tAPCTLSetACCSolverID( void *solver, JX_Int solverid );
JX_Int jx_3tAPCTLSetARRKDim( void *solver, JX_Int kdim );
JX_Int jx_3tAPCTLSetACCKDim( void *solver, JX_Int kdim );
JX_Int jx_3tAPCTLSetTestSubLSIter( void *solver, JX_Int test_subls_iter );  // peghoty, 2012/02/23           
JX_Int jx_3tAPCTLSetDebugFlag( void *solver, JX_Int debug_flag );
JX_Int jx_3tAPCTLSetResetZero( void *solver, JX_Int reset_zero );
JX_Int jx_3tAPCTLSetInterpType( void *solver, JX_Int interp_type );
JX_Int jx_3tAPCTLSetCoarsenType( void *solver, JX_Int coarsen_type );
JX_Int jx_3tAPCTLSetAggNumLevels( void *solver, JX_Int agg_num_levels );
JX_Int jx_3tAPCTLSetCoarseThreshold( void *solver, JX_Int coarse_threshold );
JX_Int jx_3tAPCTLSetPrintLevelAMG( void *solver, JX_Int print_level_amg );
JX_Int jx_3tAPCTLSetStrongThreshold( void *solver, JX_Real strong_threshold );
JX_Int jx_3tAPCTLSetNumIterAiSetup( void *solver, JX_Int num_iter_Ai_pctl_setup ); // peghoty, 2012/02/23
JX_Int jx_3tAPCTLSetNumIterArSetup( void *solver, JX_Int num_iter_Ar_pctl_setup ); // peghoty, 2012/02/23
JX_Int jx_3tAPCTLSetNumIterAePrecond( void *solver, JX_Int num_iter_Ae_pctl_precond ); // peghoty, 2012/02/23
JX_Int jx_3tAPCTLSetNumIterAiPrecond( void *solver, JX_Int num_iter_Ai_pctl_precond ); // peghoty, 2012/02/23
JX_Int jx_3tAPCTLSetNumIterArPrecond( void *solver, JX_Int num_iter_Ar_pctl_precond ); // peghoty, 2012/02/23
JX_Int jx_3tAPCTLSetNumIterAcPrecond( void *solver, JX_Int num_iter_Ac_pctl_precond ); // peghoty, 2012/02/23
JX_Int jx_3tAPCTLSetA( void *solver, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetSubBlocks( void *solver );
JX_Int jx_3tAPCTLSetARR( void *solver, jx_ParCSRMatrix *ARR );
JX_Int jx_3tAPCTLSetAEE( void *solver, jx_ParCSRMatrix *AEE );
JX_Int jx_3tAPCTLSetAII( void *solver, jx_ParCSRMatrix *AII );
JX_Int jx_3tAPCTLSetVRE( void *solver, jx_ParVector *VRE );
JX_Int jx_3tAPCTLSetVER( void *solver, jx_ParVector *VER );
JX_Int jx_3tAPCTLSetVER2( void *solver, jx_ParVector **VER );
JX_Int jx_3tAPCTLSetVEI( void *solver, jx_ParVector *VEI );
JX_Int jx_3tAPCTLSetVIE( void *solver, jx_ParVector *VIE );
JX_Int jx_3tAPCTLSetARRAll( void *solver, jx_ParCSRMatrix *ARR );
JX_Int jx_3tAPCTLSetARRAll2( void *solver, jx_ParCSRMatrix **ARR );
JX_Int jx_3tAPCTLSetAEEAll( void *solver, jx_ParCSRMatrix *AEE );
JX_Int jx_3tAPCTLSetAIIAll( void *solver, jx_ParCSRMatrix *AII );
JX_Int jx_3tAPCTLSetVREAll( void *solver, jx_ParVector *VRE );
JX_Int jx_3tAPCTLSetVREAll2( void *solver, jx_ParVector **VRE );
JX_Int jx_3tAPCTLSetVERAll( void *solver, jx_ParVector *VER );
JX_Int jx_3tAPCTLSetVERAll2( void *solver, jx_ParVector **VER );
JX_Int jx_3tAPCTLSetVEIAll( void *solver, jx_ParVector *VEI );
JX_Int jx_3tAPCTLSetVIEAll( void *solver, jx_ParVector *VIE );
JX_Int jx_3tAPCTLSetComm( void *solver, MPI_Comm comm );
JX_Int jx_3tAPCTLSetCommX( void *solver, MPI_Comm comm );
JX_Int jx_3tAPCTLSetCommY( void *solver, MPI_Comm comm );
JX_Int jx_3tAPCTLSetGroupIdX( void *solver, JX_Int groupid );
JX_Int jx_3tAPCTLSetGroupIdY( void *solver, JX_Int groupid );
void *jx_3tAPCTLDataInitialize();                  
JX_Int jx_3tAPCTLDestroy( void *vdata );
JX_Int jx_3tAPCTLDestroy_sp( void *vdata );           
JX_Int jx_3tAPCTLDestroy_mp( void *vdata ); 
JX_Int jx_3tAPCTLIterCount( void *solver );
JX_Int jx_3tAPCTLIterCount_sp( void *solver );
JX_Int jx_3tAPCTLIterCount_mp( void *solver );
JX_Int jx_GetAPCTLNumIterOfSubLS( JX_Solver precond, jx_APCTLKrylovParam *apctlkrylov_param ); 
JX_Int jx_GetAPCTLmgNumIterOfSubLS( JX_Solver precond, JX_Int groupid_x, JX_Int ng, jx_APCTLKrylovParam *apctlkrylov_param ); 



/* /csrc/3t/setup.c */
JX_Int JX_3tAPCTLSetup( JX_Solver solver, JX_ParCSRMatrix A );
JX_Int jx_3tAPCTLSetup( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A ); 
JX_Int jx_3tAPCTLSetup_sp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );                
JX_Int jx_3tAPCTLSetup_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );                    

                              

/* /csrc/3t/solve.c */
JX_Int 
JX_3tAPCTLSolve( JX_Solver        solver,
                 JX_ParCSRMatrix  A,
                 JX_ParVector     f,
                 JX_ParVector     u  );
JX_Int 
JX_3tAPCTLPrecond( JX_Solver        solver,
                   JX_ParCSRMatrix  A,
                   JX_ParVector     b,
                   JX_ParVector     x  );
JX_Int 
JX_3tAPCTLmgPrecond( JX_Solver        solver,
                     JX_ParCSRMatrix  A,
                     JX_ParVector     b,
                     JX_ParVector     x  );
JX_Int 
JX_3tABSC1mgPrecond( JX_Solver        solver,
                     JX_ParCSRMatrix  A,
                     JX_ParVector     b,
                     JX_ParVector     x  );
JX_Int 
JX_3tABSC2mgPrecond( JX_Solver        solver,
                     JX_ParCSRMatrix  A,
                     JX_ParVector     b,
                     JX_ParVector     x  );
JX_Int
jx_3tAPCTLOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                        jx_ParCSRMatrix   *A,
                        jx_ParVector      *g,
                        jx_ParVector      *w ); 
JX_Int
jx_3tAPCTLmgOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                          jx_ParCSRMatrix   *A,
                          jx_ParVector      *g,
                          jx_ParVector      *w );
JX_Int
jx_3tABSC1mgOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                          jx_ParCSRMatrix   *A,
                          jx_ParVector      *g,
                          jx_ParVector      *w );
JX_Int
jx_3tABSC2mgOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                          jx_ParCSRMatrix   *A,
                          jx_ParVector      *g,
                          jx_ParVector      *w );
JX_Int
jx_3tAPCTLOneIteration_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w );  
JX_Int
jx_3tAPCTLmgOneIteration_mp( jx_3tAPCTLData  *pre_3tapctl_data,
                             jx_ParCSRMatrix *A,
                             jx_ParVector    *g,
                             jx_ParVector    *w );
JX_Int
jx_3tABSC1mgOneIteration_mp( jx_3tAPCTLData  *pre_3tapctl_data,
                             jx_ParCSRMatrix *A,
                             jx_ParVector    *g,
                             jx_ParVector    *w );
JX_Int
jx_3tABSC2mgOneIteration_mp( jx_3tAPCTLData  *pre_3tapctl_data,
                             jx_ParCSRMatrix *A,
                             jx_ParVector    *g,
                             jx_ParVector    *w );
JX_Int
jx_3tAPCTLOneIteration_sp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w );                                                  
JX_Int
jx_3tAPCTLPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                   jx_ParCSRMatrix   *A,
                   jx_ParVector      *f,
                   jx_ParVector      *u );                                        
JX_Int
jx_3tAPCTLmgPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                     jx_ParCSRMatrix   *A,
                     jx_ParVector      *f,
                     jx_ParVector      *u );
JX_Int
jx_3tABSC1mgPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                     jx_ParCSRMatrix   *A,
                     jx_ParVector      *f,
                     jx_ParVector      *u );
JX_Int
jx_3tABSC2mgPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                     jx_ParCSRMatrix   *A,
                     jx_ParVector      *f,
                     jx_ParVector      *u );
JX_Int
jx_3tAPCTLSolve( jx_3tAPCTLData    *pre_3tapctl_data,
                 jx_ParCSRMatrix   *A,
                 jx_ParVector      *f,
                 jx_ParVector      *u );



/* /csrc/3t/relax.c */ 
JX_Int
jx_3tAPCTLRelax_GSType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w );
JX_Int
jx_3tAPCTLmgRelax_GSType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                             jx_ParCSRMatrix   *A,
                             jx_ParVector      *g,
                             jx_ParVector      *w );
JX_Int
jx_3tAPCTLRelax_BDType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w ); 
JX_Int
jx_3tAPCTLmgRelax_BDType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                             jx_ParCSRMatrix   *A,
                             jx_ParVector      *g,
                             jx_ParVector      *w );
JX_Int
jx_3tAPCTLRelax_GSType_sp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w );
JX_Int
jx_3tAPCTLRelax_BDType_sp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w );         


/* /csrc/3t/param.c */                         
jx_APCTLKrylovParam *jx_APCTLKrylovParamInitialize();
JX_Int jx_APCTLKrylovParamSetNumGroup( void *param, JX_Int ng );
JX_Int jx_APCTLKrylovParamSetSolverID( void *param, JX_Int solver_id );
JX_Int jx_APCTLKrylovParamSetPrecondID( void *param, JX_Int precond_id );
JX_Int jx_APCTLKrylovParamSetTol( void *param, JX_Real tol );
JX_Int jx_APCTLKrylovParamSetMaxIter( void *param, JX_Int max_iter );
JX_Int jx_APCTLKrylovParamSetKDim( void *param, JX_Int k_dim );
JX_Int jx_APCTLKrylovParamSetISCheckRestarted( void *param, JX_Int is_check_restarted );
JX_Int jx_APCTLKrylovParamSetTwoNorm( void *param, JX_Int two_norm );
JX_Int jx_APCTLKrylovParamSetPrintLevel( void *param, JX_Int print_level );
JX_Int jx_APCTLKrylovParamSetTTest( void *param, JX_Int TTest );
JX_Int jx_APCTLKrylovParamSetKeepSol( void *param, JX_Int keepsol );
JX_Int jx_APCTLKrylovParamSetNumIterations( void *param, JX_Int num );
JX_Int jx_APCTLKrylovParamSetLastRelNrm( void *param, JX_Real last_rel_nrm );
JX_Int jx_APCTLKrylovParamSetPrintLevelAPCTL( void *param, JX_Int print_level_apctl );
JX_Int jx_APCTLKrylovParamSetBlockSmoothType( void *param, JX_Int blocksmooth_type );
JX_Int jx_APCTLKrylovParamSetNumRelaxPre( void *param, JX_Int num_relax_pre );
JX_Int jx_APCTLKrylovParamSetNumRelaxPost( void *param, JX_Int num_relax_post );
JX_Int jx_APCTLKrylovParamSetInterpSolverARR( void *param, JX_Int interp_solver_ARR );
JX_Int jx_APCTLKrylovParamSetInterpKdimARR( void *param, JX_Int interp_kdim_ARR );
JX_Int jx_APCTLKrylovParamSetInterpMaxitARR( void *param, JX_Int interp_maxit_ARR );
JX_Int jx_APCTLKrylovParamSetInterpMaxitAII( void *param, JX_Int interp_maxit_AII );
JX_Int jx_APCTLKrylovParamSetInterpTolARR( void *param, JX_Real interp_tol_ARR );
JX_Int jx_APCTLKrylovParamSetInterpTolAII( void *param, JX_Real interp_tol_AII );
JX_Int jx_APCTLKrylovParamSetARRRelaxTol( void *param, JX_Real ARR_relax_tol );
JX_Int jx_APCTLKrylovParamSetAEERelaxTol( void *param, JX_Real AEE_relax_tol );
JX_Int jx_APCTLKrylovParamSetAIIRelaxTol( void *param, JX_Real AII_relax_tol );
JX_Int jx_APCTLKrylovParamSetACCRelaxTol( void *param, JX_Real ACC_relax_tol );
JX_Int jx_APCTLKrylovParamSetFixitPCTLR( void *param, JX_Int fixit_pctl_R );
JX_Int jx_APCTLKrylovParamSetFixitPCTLE( void *param, JX_Int fixit_pctl_E );
JX_Int jx_APCTLKrylovParamSetFixitPCTLI( void *param, JX_Int fixit_pctl_I );
JX_Int jx_APCTLKrylovParamSetFixitBrlxR( void *param, JX_Int fixit_brlx_R );
JX_Int jx_APCTLKrylovParamSetFixitBrlxE( void *param, JX_Int fixit_brlx_E );
JX_Int jx_APCTLKrylovParamSetFixitBrlxI( void *param, JX_Int fixit_brlx_I );
JX_Int jx_APCTLKrylovParamSetARRRelaxType( void *param, JX_Int ARR_relax_type );
JX_Int jx_APCTLKrylovParamSetAEERelaxType( void *param, JX_Int AEE_relax_type );
JX_Int jx_APCTLKrylovParamSetAIIRelaxType( void *param, JX_Int AII_relax_type );
JX_Int jx_APCTLKrylovParamSetUseFixedModeR( void *param, JX_Int use_fixedmode_R );
JX_Int jx_APCTLKrylovParamSetUseFixedModeE( void *param, JX_Int use_fixedmode_E );
JX_Int jx_APCTLKrylovParamSetUseFixedModeI( void *param, JX_Int use_fixedmode_I );
JX_Int jx_APCTLKrylovParamSetThetaWCE( void *param, JX_Real theta_wc_E );
JX_Int jx_APCTLKrylovParamSetParamThresholdWCE( void *param, JX_Real threshold_wc_E );
JX_Int jx_APCTLKrylovParamSetThetaDDR( void *param, JX_Real theta_dd_R );
JX_Int jx_APCTLKrylovParamSetThetaDDE( void *param, JX_Real theta_dd_E );
JX_Int jx_APCTLKrylovParamSetThetaDDI( void *param, JX_Real theta_dd_I );
JX_Int jx_APCTLKrylovParamSetThresholdDDR( void *param, JX_Real threshold_dd_R );
JX_Int jx_APCTLKrylovParamSetThresholdDDE( void *param, JX_Real threshold_dd_E );
JX_Int jx_APCTLKrylovParamSetThresholdDDI( void *param, JX_Real threshold_dd_I );
JX_Int jx_APCTLKrylovParamSetISDiagElmFirst( void *param, JX_Int is_diagelm_first );
JX_Int jx_APCTLKrylovParamSetUsePPCTL( void *param, JX_Int use_ppctl );
JX_Int jx_APCTLKrylovParamSetTestSubLSIter( void *param, JX_Int test_subls_iter );
JX_Int jx_APCTLKrylovParamSetCPUTrans( void *param, JX_Real cpu_trans );
JX_Int jx_APCTLKrylovParamSetCPUSetup( void *param, JX_Real cpu_setup );
JX_Int jx_APCTLKrylovParamSetCPUSolve( void *param, JX_Real cpu_solve );
JX_Int jx_APCTLKrylovParamSetCPUTotal( void *param, JX_Real cpu_total );



/* /csrc/3t/jasminIF.c */
JX_Int 
jx_ApctlKrylov_JASMIN( jx_ParCSRMatrix     *ARR_p, 
                       jx_ParCSRMatrix     *AEE_p, 
                       jx_ParCSRMatrix     *AII_p, 
                       jx_ParVector        *VRE_p, 
                       jx_ParVector        *VER_p, 
                       jx_ParVector        *VEI_p, 
                       jx_ParVector        *VIE_p, 
                       jx_ParVector        *fR_p, 
                       jx_ParVector        *fE_p, 
                       jx_ParVector        *fI_p,
                       jx_ParVector        *uR_p, 
                       jx_ParVector        *uE_p, 
                       jx_ParVector        *uI_p,                      
                       jx_APCTLKrylovParam *apctlkrylov_param );
JX_Int
jx_ApctlKrylov_mgJASMIN( jx_ParCSRMatrix    **ARR_p,
                         jx_ParCSRMatrix     *AEE_p,
                         jx_ParCSRMatrix     *AII_p,
                         jx_ParVector       **VRE_p,
                         jx_ParVector       **VER_p,
                         jx_ParVector        *VEI_p,
                         jx_ParVector        *VIE_p,
                         jx_ParVector       **fR_p,
                         jx_ParVector        *fE_p,
                         jx_ParVector        *fI_p,
                         jx_ParVector       **uR_p,
                         jx_ParVector        *uE_p,
                         jx_ParVector        *uI_p,
                         jx_APCTLKrylovParam *apctlkrylov_param );
JX_Int 
jx_ApctlKrylov_JASMIN_mp( jx_ParCSRMatrix     *ARR_p, 
                          jx_ParCSRMatrix     *AEE_p, 
                          jx_ParCSRMatrix     *AII_p, 
                          jx_ParVector        *VRE_p, 
                          jx_ParVector        *VER_p, 
                          jx_ParVector        *VEI_p, 
                          jx_ParVector        *VIE_p, 
                          jx_ParVector        *fR_p, 
                          jx_ParVector        *fE_p, 
                          jx_ParVector        *fI_p,
                          jx_ParVector        *uR_p, 
                          jx_ParVector        *uE_p, 
                          jx_ParVector        *uI_p,
                          jx_APCTLKrylovParam *apctlkrylov_param ); 
JX_Int
jx_ApctlKrylov_mgJASMIN_mp( jx_ParCSRMatrix    **ARR_p, 
                            jx_ParCSRMatrix     *AEE_p, 
                            jx_ParCSRMatrix     *AII_p, 
                            jx_ParVector       **VRE_p, 
                            jx_ParVector       **VER_p, 
                            jx_ParVector        *VEI_p, 
                            jx_ParVector        *VIE_p, 
                            jx_ParVector       **fR_p, 
                            jx_ParVector        *fE_p, 
                            jx_ParVector        *fI_p,
                            jx_ParVector       **uR_p, 
                            jx_ParVector        *uE_p, 
                            jx_ParVector        *uI_p,
                            jx_APCTLKrylovParam *apctlkrylov_param );
JX_Int 
jx_ApctlKrylov_JASMIN_sp( jx_ParCSRMatrix     *ARR_p, 
                          jx_ParCSRMatrix     *AEE_p, 
                          jx_ParCSRMatrix     *AII_p, 
                          jx_ParVector        *VRE_p, 
                          jx_ParVector        *VER_p, 
                          jx_ParVector        *VEI_p, 
                          jx_ParVector        *VIE_p, 
                          jx_ParVector        *fR_p, 
                          jx_ParVector        *fE_p, 
                          jx_ParVector        *fI_p,
                          jx_ParVector        *uR_p, 
                          jx_ParVector        *uE_p, 
                          jx_ParVector        *uI_p,
                          jx_APCTLKrylovParam *apctlkrylov_param );                       
JX_Int JX_3tAPCTLSetup4Jasmin( JX_Solver solver, JX_ParCSRMatrix A );
JX_Int JX_3tAPCTLSetup4mgJasmin( JX_Solver solver, JX_ParCSRMatrix A );
JX_Int JX_3tABSC1Setup4mgJasmin( JX_Solver solver, JX_ParCSRMatrix A );
JX_Int JX_3tABSC2Setup4mgJasmin( JX_Solver solver, JX_ParCSRMatrix A );
JX_Int jx_3tAPCTLSetup4Jasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetup4mgJasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tABSC1Setup4mgJasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tABSC2Setup4mgJasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetup4Jasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetup4mgJasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tABSC1Setup4mgJasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tABSC2Setup4mgJasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetup4Jasmin_sp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int JX_3tAPCTLDestroy4Jasmin( JX_Solver solver );  
JX_Int JX_3tAPCTLDestroy4mgJasmin( JX_Solver solver );
JX_Int jx_3tAPCTLDestroy4Jasmin( void *vdata );
JX_Int jx_3tAPCTLDestroy4mgJasmin( void *vdata );
JX_Int jx_3tAPCTLDestroy4Jasmin_sp( void *vdata );
JX_Int jx_3tAPCTLDestroy4Jasmin_mp( void *vdata );
JX_Int jx_3tAPCTLDestroy4mgJasmin_mp( void *vdata );



/* /csrc/3t/interp.c */
jx_ParCSRMatrix *
jx_3tApctlInterpolation( MPI_Comm         comm, 
                         MPI_Comm         comm_x,
                         JX_Int              N,
                         JX_Int              groupid_x,
                         JX_Int             *row_starts, 
                         JX_Int             *col_starts,
                         jx_ParVector    *WEE );
jx_ParCSRMatrix *
jx_3tApctlmgInterpolation( MPI_Comm         comm, 
                           MPI_Comm         comm_x,
                           JX_Int              N,
                           JX_Int              groupid_x,
                           JX_Int              ng,
                           JX_Int             *row_starts, 
                           JX_Int             *col_starts,
                           jx_ParVector    *WEE );
JX_Int 
jx_3tAPCTLParaInterpVec( MPI_Comm       comm,
                         MPI_Comm       comm_x, 
                         JX_Int            groupid_x,
                         JX_Int           *vec_starts, 
                         jx_ParVector  *WEE, 
                         jx_ParVector **PRR_ptr, 
                         jx_ParVector **PII_ptr );
JX_Int 
jx_3tAPCTLmgParaInterpVec( MPI_Comm        comm,
                           MPI_Comm        comm_x, 
                           JX_Int            groupid_x,
                           JX_Int           *vec_starts, 
                           JX_Int          ng,
                           jx_ParVector   *WEE, 
                           jx_ParVector ***PRR_ptr, 
                           jx_ParVector  **PII_ptr );


/* /csrc/3t/coarseop.c */
jx_ParCSRMatrix *
jx_3tApctlCoarseOperator_sp( MPI_Comm       comm,
                             jx_CSRMatrix  *ARR_s, 
                             jx_CSRMatrix  *AEE_s, 
                             jx_CSRMatrix  *AII_s, 
                             jx_Vector     *VRE_s, 
                             jx_Vector     *VER_s, 
                             jx_Vector     *VEI_s, 
                             jx_Vector     *VIE_s,
                             jx_ParVector  *PRR,
                             jx_ParVector  *PII  );
jx_ParCSRMatrix *
jx_3tApctlCoarseOperator_mp( MPI_Comm          comm,
                             jx_ParCSRMatrix  *ARR, 
                             jx_ParCSRMatrix  *AEE, 
                             jx_ParCSRMatrix  *AII, 
                             jx_ParVector     *VRE, 
                             jx_ParVector     *VER, 
                             jx_ParVector     *VEI, 
                             jx_ParVector     *VIE,
                             jx_ParVector     *PRR, 
                             jx_ParVector     *PII ); 
jx_ParCSRMatrix *
jx_3tApctlmgCoarseOperator_mp( MPI_Comm          comm,
                               JX_Int            ng,
                               jx_ParCSRMatrix **ARR, 
                               jx_ParCSRMatrix  *AEE, 
                               jx_ParCSRMatrix  *AII, 
                               jx_ParVector    **VRE, 
                               jx_ParVector    **VER, 
                               jx_ParVector     *VEI, 
                               jx_ParVector     *VIE,
                               jx_ParVector    **PRR, 
                               jx_ParVector     *PII );


/* /csrc/3t/seqIF.c */
JX_Int 
jx_ApctlKrylov_SEQ( MPI_Comm              comm,
                    jx_CSRMatrix         *A_s,  
                    jx_Vector            *f_s, 
                    jx_Vector            *u_s,                      
                    jx_APCTLKrylovParam  *apctlkrylov_param );
JX_Int 
jx_ApctlKrylov_SEQ_mp( MPI_Comm              comm,
                       jx_CSRMatrix         *A_s,  
                       jx_Vector            *f_s, 
                       jx_Vector            *u_s,                      
                       jx_APCTLKrylovParam  *apctlkrylov_param );
JX_Int 
jx_ApctlKrylov_SEQ_sp( MPI_Comm              comm,
                       jx_CSRMatrix         *A_s,  
                       jx_Vector            *f_s, 
                       jx_Vector            *u_s,                      
                       jx_APCTLKrylovParam  *apctlkrylov_param );                    
JX_Int JX_3tAPCTLSetupSEQIF( JX_Solver solver, JX_ParCSRMatrix A );
JX_Int jx_3tAPCTLSetupSEQIF( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetupSEQIF_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int jx_3tAPCTLSetupSEQIF_sp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A );
JX_Int JX_3tAPCTLDestroySEQIF( JX_Solver solver );
JX_Int jx_3tAPCTLDestroySEQIF( void *vdata );
JX_Int jx_3tAPCTLDestroySEQIF_mp( void *vdata );
JX_Int jx_3tAPCTLDestroySEQIF_sp( void *vdata );

                             
#endif                                                     
