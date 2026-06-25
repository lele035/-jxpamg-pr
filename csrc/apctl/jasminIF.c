//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jasminIF.c -- Adaptive PCTL iterattion or preconditioner for 3t linear systems.
 *                PCTL, Physical-variable based Coarsening Two Level iterative Method,
 *                is proposed by Xu Xiaowen, Mo Zeyao etc.
 *
 *  These functions are specially designed for JASMIN interface.  
 *
 *  Date: 2012/02/26
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
 * \fn JX_Int jx_ApctlKrylov_JASMIN
 * \brief APCTL-Krylov 法的接口程序
 * \note 湘潭大学课题组和北京应用物理与计算数学研究所合作研制
 * \ref [1] 周志阳,徐小文,舒适,冯春生,莫则尧，二维三温辐射扩散方程组两层预条件子的自适应求解，计算物理，2012，已接收.
 *      [2] 周志阳,几种求解辐射扩散问题和弹性问题的代数多层网格法与区域分解法，博士学位论文，湘潭：湘潭大学，2012. 
 * \date 2012/02/25
 */
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
                       jx_APCTLKrylovParam *apctlkrylov_param )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(ARR_p);
   JX_Int myid, nprocs;
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs > 1 && nprocs % 3 == 0)
   {
      if (!jx_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param))
      {
         jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(ARR_p));
         jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(AEE_p));
         jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(AII_p));
      }  
      jx_ApctlKrylov_JASMIN_mp( ARR_p, AEE_p, AII_p, 
                                VRE_p, VER_p, VEI_p, VIE_p, 
                                fR_p, fE_p, fI_p,
                                uR_p, uE_p, uI_p,
                                apctlkrylov_param );
   }
   else if (nprocs == 1)
   {
      if (!jx_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param))
      {
         jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(ARR_p));
         jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(AEE_p));
         jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(AII_p));
      } 
      
      jx_ApctlKrylov_JASMIN_sp( ARR_p, AEE_p, AII_p, 
                                VRE_p, VER_p, VEI_p, VIE_p, 
                                fR_p, fE_p, fI_p,
                                uR_p, uE_p, uI_p,
                                apctlkrylov_param );
   }
   else
   {
      if (myid == 0)
      {
         jx_printf(" Warning: np should be 1 or a multiple of 3, please try again!\n\n");
      }
   } 
  
   return (0);  
}

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
                         jx_APCTLKrylovParam *apctlkrylov_param )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(ARR_p[0]);
   JX_Int ng = jx_APCTLKrylovParamNumGroup(apctlkrylov_param);
   JX_Int myid, nprocs, idx;
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (!jx_APCTLKrylovParamISDiagElmFirst(apctlkrylov_param))
   {
      for (idx = 0; idx < ng; idx ++) jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(ARR_p[idx]));
      jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(AEE_p));
      jx_CSRMatrixReorder(jx_ParCSRMatrixDiag(AII_p));
   }

   if (nprocs > 1 && nprocs % (ng+2) == 0)
   {
      jx_ApctlKrylov_mgJASMIN_mp(ARR_p, AEE_p, AII_p, VRE_p, VER_p, VEI_p, VIE_p, fR_p, fE_p, fI_p, uR_p, uE_p, uI_p, apctlkrylov_param);
   }
   else
   {
      if (myid == 0)
      {
         jx_printf(" Warning: np should be a multiple of %d, please try again!\n\n", ng+2);
      }
   }

   return (0);
}

/*!
 * \fn JX_Int jx_ApctlKrylov_JASMIN_mp
 * \brief APCTL-GMRES solvers for multi-processor case.
 * \author peghoty
 * \date 2012/02/25 
 */
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
                          jx_APCTLKrylovParam *apctlkrylov_param )
{                    
   JX_Int    solver_id          = jx_APCTLKrylovParamSolverID(apctlkrylov_param);
   JX_Real tol                = jx_APCTLKrylovParamTol(apctlkrylov_param);
   JX_Int    max_iter           = jx_APCTLKrylovParamMaxIter(apctlkrylov_param);
   JX_Int    k_dim              = jx_APCTLKrylovParamKDim(apctlkrylov_param);
   JX_Int    is_check_restarted = jx_APCTLKrylovParamISCheckRestarted(apctlkrylov_param);
   JX_Int    two_norm           = jx_APCTLKrylovParamTwoNorm(apctlkrylov_param);
   JX_Int    print_level        = jx_APCTLKrylovParamPrintLevel(apctlkrylov_param);  
   JX_Int    TTest              = jx_APCTLKrylovParamTTest(apctlkrylov_param);       
   JX_Int    keepsol            = jx_APCTLKrylovParamKeepSol(apctlkrylov_param);
   
   JX_Int    print_level_apctl  = jx_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param);   
   JX_Int    blocksmooth_type   = jx_APCTLKrylovParamBlockSmoothType(apctlkrylov_param);   
   JX_Int    num_relax_pre      = jx_APCTLKrylovParamNumRelaxPre(apctlkrylov_param);       
   JX_Int    num_relax_post     = jx_APCTLKrylovParamNumRelaxPost(apctlkrylov_param);      
   JX_Int    interp_solver_ARR  = jx_APCTLKrylovParamInterpSolverARR(apctlkrylov_param);   
   JX_Int    interp_kdim_ARR    = jx_APCTLKrylovParamInterpKdimARR(apctlkrylov_param);     
   JX_Int    interp_maxit_ARR   = jx_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param);    
   JX_Int    interp_maxit_AII   = jx_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param);    
   JX_Real interp_tol_ARR     = jx_APCTLKrylovParamInterpTolARR(apctlkrylov_param);      
   JX_Real interp_tol_AII     = jx_APCTLKrylovParamInterpTolAII(apctlkrylov_param);      
   JX_Int    fixit_pctl_R       = jx_APCTLKrylovParamFixitPCTLR(apctlkrylov_param);        
   JX_Int    fixit_pctl_E       = jx_APCTLKrylovParamFixitPCTLE(apctlkrylov_param);        
   JX_Int    fixit_pctl_I       = jx_APCTLKrylovParamFixitPCTLI(apctlkrylov_param);        
   JX_Int    fixit_brlx_R       = jx_APCTLKrylovParamFixitBrlxR(apctlkrylov_param);        
   JX_Int    fixit_brlx_E       = jx_APCTLKrylovParamFixitBrlxE(apctlkrylov_param);        
   JX_Int    fixit_brlx_I       = jx_APCTLKrylovParamFixitBrlxI(apctlkrylov_param);        
   JX_Int    use_fixedmode_R    = jx_APCTLKrylovParamUseFixedModeR(apctlkrylov_param);     
   JX_Int    use_fixedmode_E    = jx_APCTLKrylovParamUseFixedModeE(apctlkrylov_param);     
   JX_Int    use_fixedmode_I    = jx_APCTLKrylovParamUseFixedModeI(apctlkrylov_param);     
   JX_Real theta_wc_E         = jx_APCTLKrylovParamThetaWCE(apctlkrylov_param);          
   JX_Real threshold_wc_E     = jx_APCTLKrylovParamThresholdWCE(apctlkrylov_param);      
   JX_Real theta_dd_R         = jx_APCTLKrylovParamThetaDDR(apctlkrylov_param);          
   JX_Real theta_dd_E         = jx_APCTLKrylovParamThetaDDE(apctlkrylov_param);          
   JX_Real theta_dd_I         = jx_APCTLKrylovParamThetaDDI(apctlkrylov_param);          
   JX_Real threshold_dd_R     = jx_APCTLKrylovParamThresholdDDR(apctlkrylov_param);      
   JX_Real threshold_dd_E     = jx_APCTLKrylovParamThresholdDDE(apctlkrylov_param);      
   JX_Real threshold_dd_I     = jx_APCTLKrylovParamThresholdDDI(apctlkrylov_param); 
   JX_Int    use_ppctl          = jx_APCTLKrylovParamUsePPCTL(apctlkrylov_param);
   JX_Int    test_subls_iter    = jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param);     
   JX_Real tol_default        = 1.0e-6;
   JX_Int    debug_flag         = jx_APCTLKrylovParamDebugFlag(apctlkrylov_param);

   JX_Real strong_threshold = jx_APCTLKrylovParamStrongThreshold(apctlkrylov_param);
   JX_Int interp_type = jx_APCTLKrylovParamInterpType(apctlkrylov_param);
   JX_Int coarsen_type = jx_APCTLKrylovParamCoarsenType(apctlkrylov_param);
   JX_Int agg_num_levels = jx_APCTLKrylovParamAggNumLevels(apctlkrylov_param);
   JX_Int coarse_threshold = jx_APCTLKrylovParamCoarseThreshold(apctlkrylov_param);
   JX_Int print_level_amg = jx_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param);
   
   JX_Int    num_iterations;
   JX_Real final_res_norm; 
   JX_Real starttime  = 0.0;
   JX_Real endtime    = 0.0;  
   JX_Real starttimeT = 0.0;
   JX_Real endtimeT   = 0.0; 
   
   JX_Real cpu_trans_in  = 0.0;
   JX_Real cpu_trans_out = 0.0;
   JX_Real cpu_setup     = 0.0;
   JX_Real cpu_solve     = 0.0;
   JX_Real cpu_total     = 0.0;         
   
   jx_ParCSRMatrix  *ARR = NULL; 
   jx_ParCSRMatrix  *AEE = NULL; 
   jx_ParCSRMatrix  *AII = NULL; 
   jx_ParVector     *VRE = NULL; 
   jx_ParVector     *VER = NULL; 
   jx_ParVector     *VEI = NULL; 
   jx_ParVector     *VIE = NULL; 
                   
   jx_ParCSRMatrix  *par_mat = NULL;
   jx_ParVector     *par_rhs = NULL;
   jx_ParVector     *par_sol = NULL;

   JX_Solver solver  = NULL;
   JX_Solver precond = NULL;

   JX_Int myid, nprocs;
   JX_Int np_R, np_E, np_I;
   JX_Int group_num_x = 3;
   JX_Int group_num_y;
   JX_Int groupid_x = MPI_UNDEFINED;
   JX_Int groupid_y = MPI_UNDEFINED; 
   MPI_Comm comm = jx_ParCSRMatrixComm(ARR_p);
   MPI_Comm comm_x, comm_y;
   MPI_Comm comm_bak; 
         
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   np_R = nprocs / 3;
   np_E = np_R;
   np_I = np_R;  

   if (myid < np_R)
   {
      groupid_x = 0;
   }
   else if (myid < np_R + np_E)
   {
      groupid_x = 1;
   }
   else if (myid < np_R + np_E + np_I)
   {
      groupid_x = 2;
   }
   jx_MPI_Comm_split(comm, groupid_x, myid, &comm_x); 
   jx_MPI_Comm_dup(comm_x, &comm_bak);

   group_num_y = nprocs / group_num_x; 
   groupid_y   = myid % group_num_y;
   jx_MPI_Comm_split(comm, groupid_y, myid, &comm_y);

   if (TTest) starttime = jx_MPI_Wtime();
                
   jx_ParaDataTrans4ApctlKrylov( comm, comm_x, groupid_x, 
                                 ARR_p, AEE_p, AII_p, 
                                 VRE_p, VER_p, VEI_p, VIE_p,
                                 fR_p, fE_p, fI_p,
                                 uR_p, uE_p, uI_p,
                                 &ARR, &AEE, &AII, &VRE, &VER, &VEI, &VIE,
                                 &par_mat, &par_rhs, &par_sol ); 

   if (TTest)
   {
      endtime = jx_MPI_Wtime();
      cpu_trans_in = jx_GetWallTimeMax(comm, starttime, endtime);  
   }     

   if (TTest) starttimeT = jx_MPI_Wtime();
                                          
   switch (solver_id)
   {

      case 1: 
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jx_printf("\n Solver: \033[31mAPCTL-CG\033[00m\n\n");
            else
               jx_printf("\n Solver: \033[31mPPCTL-CG\033[00m\n\n");
         }
         
         if (TTest) starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&precond);

         JX_3tAPCTLSetA(precond, par_mat);
         JX_3tAPCTLSetARR(precond, ARR);
         JX_3tAPCTLSetAEE(precond, AEE);
         JX_3tAPCTLSetAII(precond, AII);
         JX_3tAPCTLSetVRE(precond, VRE);
         JX_3tAPCTLSetVER(precond, VER);
         JX_3tAPCTLSetVEI(precond, VEI);
         JX_3tAPCTLSetVIE(precond, VIE);
         JX_3tAPCTLSetARRAll(precond, ARR_p);
         JX_3tAPCTLSetAEEAll(precond, AEE_p);
         JX_3tAPCTLSetAIIAll(precond, AII_p);
         JX_3tAPCTLSetVREAll(precond, VRE_p);
         JX_3tAPCTLSetVERAll(precond, VER_p);
         JX_3tAPCTLSetVEIAll(precond, VEI_p);
         JX_3tAPCTLSetVIEAll(precond, VIE_p);         
         JX_3tAPCTLSetMaxiter(precond, 1);
         JX_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JX_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
         JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
         JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */          
         JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JX_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JX_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JX_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JX_3tAPCTLSetACCRelaxTol(precond, tol_default);  
         
         JX_3tAPCTLSetNpR(precond, np_R);
         JX_3tAPCTLSetNpE(precond, np_E);
         JX_3tAPCTLSetNpI(precond, np_I); 
         JX_3tAPCTLSetComm(precond, comm);          
         JX_3tAPCTLSetCommX(precond, comm_x);
         JX_3tAPCTLSetCommY(precond, comm_y);
         JX_3tAPCTLSetGroupIdX(precond, groupid_x);
         JX_3tAPCTLSetGroupIdY(precond, groupid_y);
  
         JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */ 

         JX_3tAPCTLSetDebugFlag(precond, debug_flag);
         JX_3tAPCTLSetStrongThreshold(precond, strong_threshold);
         JX_3tAPCTLSetInterpType(precond, interp_type);
         JX_3tAPCTLSetCoarsenType(precond, coarsen_type);
         JX_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
         JX_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
         JX_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

         JX_ParCSRPCGCreate(comm, &solver); 
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, two_norm);  // 0: B 范数； 1：l2 范数 
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);
         
         JX_PCGSetPrecond( solver,
                           (JX_PtrToSolverFcn) JX_3tAPCTLPrecond,
                           (JX_PtrToSolverFcn) NULL,
                           precond ); 
         
         JX_3tAPCTLSetup4Jasmin( precond, (JX_ParCSRMatrix)par_mat );                   
         
         JX_PCGSetup ( solver, 
                       (JX_Matrix) par_mat, 
                       (JX_Vector) par_rhs, 
                       (JX_Vector) par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PCGSolve ( solver, 
                       (JX_Matrix) par_mat, // preOperater
                       (JX_Matrix) par_mat, 
                       (JX_Vector) par_rhs, 
                       (JX_Vector) par_sol );  

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime); 
         }

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);

         if (print_level_apctl)
         { 
            JX_3tAPCTLIterCount(precond);
         }

         JX_ParCSRPCGDestroy(solver);
      }
      break;
      

      case 2:  // APCTL-GMRES(m)
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jx_printf("\n Solver: \033[31mAPCTL-GMRES(%d)\033[00m\n\n", k_dim);
            else
               jx_printf("\n Solver: \033[31mPPCTL-GMRES(%d)\033[00m\n\n", k_dim);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&precond);
         JX_3tAPCTLSetA(precond, par_mat);
         JX_3tAPCTLSetARR(precond, ARR);
         JX_3tAPCTLSetAEE(precond, AEE);
         JX_3tAPCTLSetAII(precond, AII);
         JX_3tAPCTLSetVRE(precond, VRE);
         JX_3tAPCTLSetVER(precond, VER);
         JX_3tAPCTLSetVEI(precond, VEI);
         JX_3tAPCTLSetVIE(precond, VIE);
         JX_3tAPCTLSetARRAll(precond, ARR_p);
         JX_3tAPCTLSetAEEAll(precond, AEE_p);
         JX_3tAPCTLSetAIIAll(precond, AII_p);
         JX_3tAPCTLSetVREAll(precond, VRE_p);
         JX_3tAPCTLSetVERAll(precond, VER_p);
         JX_3tAPCTLSetVEIAll(precond, VEI_p);
         JX_3tAPCTLSetVIEAll(precond, VIE_p);            
         JX_3tAPCTLSetMaxiter(precond, 1);
         JX_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JX_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
         JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
         JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */          
         JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JX_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JX_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JX_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JX_3tAPCTLSetACCRelaxTol(precond, tol_default);  
                  
         JX_3tAPCTLSetNpR(precond, np_R);
         JX_3tAPCTLSetNpE(precond, np_E);
         JX_3tAPCTLSetNpI(precond, np_I);
         JX_3tAPCTLSetComm(precond, comm);           
         JX_3tAPCTLSetCommX(precond, comm_x);
         JX_3tAPCTLSetCommY(precond, comm_y);
         JX_3tAPCTLSetGroupIdX(precond, groupid_x);
         JX_3tAPCTLSetGroupIdY(precond, groupid_y);
         
         JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         
         JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

         JX_3tAPCTLSetDebugFlag(precond, debug_flag);
         JX_3tAPCTLSetStrongThreshold(precond, strong_threshold);
         JX_3tAPCTLSetInterpType(precond, interp_type);
         JX_3tAPCTLSetCoarsenType(precond, coarsen_type);
         JX_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
         JX_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
         JX_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);
                  
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);

         JX_GMRESSetPrecond( solver,
                             (JX_PtrToSolverFcn) JX_3tAPCTLPrecond,
                             (JX_PtrToSolverFcn) NULL,
                             precond );            

         JX_3tAPCTLSetup4Jasmin( precond, (JX_ParCSRMatrix)par_mat );

         JX_GMRESSetup( solver, 
                        (JX_Matrix)par_mat,
                        (JX_Vector)par_rhs,
                        (JX_Vector)par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve( solver,
                        (JX_Matrix)par_mat, // preOperator
                        (JX_Matrix)par_mat,
                        (JX_Vector)par_rhs,
                        (JX_Vector)par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime); 
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
         if (print_level_apctl)
         { 
            JX_3tAPCTLIterCount(precond);
         }

         JX_ParCSRGMRESDestroy(solver);
      }
      break;


      case 3:  // APCTL-BiCGSTab
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jx_printf("\n Solver: \033[31mAPCTL-BiCGSTab\033[00m\n\n");
            else
               jx_printf("\n Solver: \033[31mPPCTL-BiCGSTab\033[00m\n\n");
         }
         
         if (TTest) starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&precond);

         JX_3tAPCTLSetA(precond, par_mat);
         JX_3tAPCTLSetARR(precond, ARR);
         JX_3tAPCTLSetAEE(precond, AEE);
         JX_3tAPCTLSetAII(precond, AII);
         JX_3tAPCTLSetVRE(precond, VRE);
         JX_3tAPCTLSetVER(precond, VER);
         JX_3tAPCTLSetVEI(precond, VEI);
         JX_3tAPCTLSetVIE(precond, VIE);
         JX_3tAPCTLSetARRAll(precond, ARR_p);
         JX_3tAPCTLSetAEEAll(precond, AEE_p);
         JX_3tAPCTLSetAIIAll(precond, AII_p);
         JX_3tAPCTLSetVREAll(precond, VRE_p);
         JX_3tAPCTLSetVERAll(precond, VER_p);
         JX_3tAPCTLSetVEIAll(precond, VEI_p);
         JX_3tAPCTLSetVIEAll(precond, VIE_p);         
         JX_3tAPCTLSetMaxiter(precond, 1);
         JX_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JX_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
         JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
         JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */          
         JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JX_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JX_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JX_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JX_3tAPCTLSetACCRelaxTol(precond, tol_default);  
         
         JX_3tAPCTLSetNpR(precond, np_R);
         JX_3tAPCTLSetNpE(precond, np_E);
         JX_3tAPCTLSetNpI(precond, np_I); 
         JX_3tAPCTLSetComm(precond, comm);          
         JX_3tAPCTLSetCommX(precond, comm_x);
         JX_3tAPCTLSetCommY(precond, comm_y);
         JX_3tAPCTLSetGroupIdX(precond, groupid_x);
         JX_3tAPCTLSetGroupIdY(precond, groupid_y);
         
         JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         
         JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

         JX_3tAPCTLSetDebugFlag(precond, debug_flag);
         JX_3tAPCTLSetStrongThreshold(precond, strong_threshold);
         JX_3tAPCTLSetInterpType(precond, interp_type);
         JX_3tAPCTLSetCoarsenType(precond, coarsen_type);
         JX_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
         JX_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
         JX_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);
                  
         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond ( solver,
                                 (JX_PtrToSolverFcn) JX_3tAPCTLPrecond,
                                 (JX_PtrToSolverFcn) NULL,
                                 precond ); 

         JX_3tAPCTLSetup4Jasmin( precond, (JX_ParCSRMatrix)par_mat );

         JX_BiCGSTABSetup ( solver, 
                            (JX_Matrix) par_mat, 
		            (JX_Vector) par_rhs, 
		            (JX_Vector) par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_BiCGSTABSolve ( solver, 
                            (JX_Matrix) par_mat, // preOperater
                            (JX_Matrix) par_mat, 
		            (JX_Vector) par_rhs, 
		            (JX_Vector) par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime); 
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
 
         if (print_level_apctl)
         { 
            JX_3tAPCTLIterCount(precond);
         }
         
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

   }  // end switch
   
    
   if (TTest)
   {
      endtimeT = jx_MPI_Wtime();
      cpu_total = jx_GetWallTimeMax(comm, starttimeT, endtimeT);   
   }
   
   
   //=========================================================
   //  Step 4: 获取子系统迭代次数统计信息
   //=========================================================
   if (jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param))
   {   
      jx_GetAPCTLNumIterOfSubLS(precond, apctlkrylov_param);
   }
   JX_3tAPCTLDestroy4Jasmin(precond);     

  
   //=========================================================
   //  Step 5: 将近似解向量保存到文件中
   //=========================================================
   
   if (keepsol)
   {
      jx_Vector *ser_sol = NULL;
      ser_sol = jx_ParVectorToVectorAll(par_sol);
      if (myid == 0) 
      {
#if 1      
         jx_SeqVectorPrint(ser_sol, "./app");
#else       
         jx_Vector *solR = NULL;
         jx_Vector *solE = NULL;
         jx_Vector *solI = NULL;
         jx_3tGetSubVecs(ser_sol, &solR, &solE, &solI);
         jx_SeqVectorPrint(solR, "./uR-1");
         jx_SeqVectorPrint(solE, "./uE-1");
         jx_SeqVectorPrint(solI, "./uI-1");
         jx_SeqVectorDestroy(solR);
         jx_SeqVectorDestroy(solE);
         jx_SeqVectorDestroy(solI);               
#endif
      }
      jx_SeqVectorDestroy(ser_sol);
   }


   //=========================================================
   //  Step 6: 将近似解向量从 par_sol 还原到 uR_p, uE_p, uI_p
   //=========================================================
   
   if (TTest) starttime = jx_MPI_Wtime();
      
   jx_APCTLKrylovSolBack4Jasmin(comm, comm_bak, groupid_x, par_sol, uR_p, uE_p, uI_p);

   if (TTest)
   {
      endtime = jx_MPI_Wtime();
      cpu_trans_out = jx_GetWallTimeMax(comm, starttime, endtime); 
   }
   
   
   //==================================================================================
   //  Step 7: 获取数据转换，Setup 和 Solve 过程的 CPU 时间
   //==================================================================================  
   
   jx_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans_in + cpu_trans_out);
   jx_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jx_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jx_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total);    
   
   
   //=========================================================
   //  Step 8: 释放内存
   //=========================================================
   
   jx_ParCSRMatrixDestroy(par_mat);
   jx_ParVectorDestroy(par_rhs);
   jx_ParVectorDestroy(par_sol);
   jx_MPI_Comm_free(&comm_bak);
   
   return (0);
}                      

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
                            jx_APCTLKrylovParam *apctlkrylov_param )
{
   JX_Int    ng                 = jx_APCTLKrylovParamNumGroup(apctlkrylov_param);
   JX_Int    solver_id          = jx_APCTLKrylovParamSolverID(apctlkrylov_param);
   JX_Int    precond_id         = jx_APCTLKrylovParamPrecondID(apctlkrylov_param);
   JX_Real tol                = jx_APCTLKrylovParamTol(apctlkrylov_param);
   JX_Int    max_iter           = jx_APCTLKrylovParamMaxIter(apctlkrylov_param);
   JX_Int    k_dim              = jx_APCTLKrylovParamKDim(apctlkrylov_param);
   JX_Int    is_check_restarted = jx_APCTLKrylovParamISCheckRestarted(apctlkrylov_param);
   JX_Int    print_level        = jx_APCTLKrylovParamPrintLevel(apctlkrylov_param);
   JX_Int    TTest              = jx_APCTLKrylovParamTTest(apctlkrylov_param);
   JX_Int    keepsol            = jx_APCTLKrylovParamKeepSol(apctlkrylov_param);

   JX_Int    print_level_apctl  = jx_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param);
   JX_Int    blocksmooth_type   = jx_APCTLKrylovParamBlockSmoothType(apctlkrylov_param);
   JX_Int    interp_solver_ARR  = jx_APCTLKrylovParamInterpSolverARR(apctlkrylov_param);
   JX_Int    interp_kdim_ARR    = jx_APCTLKrylovParamInterpKdimARR(apctlkrylov_param);
   JX_Int    interp_maxit_ARR   = jx_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param);
   JX_Int    interp_maxit_AII   = jx_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param);
   JX_Real interp_tol_ARR     = jx_APCTLKrylovParamInterpTolARR(apctlkrylov_param);
   JX_Real interp_tol_AII     = jx_APCTLKrylovParamInterpTolAII(apctlkrylov_param);
   JX_Int    fixit_pctl_R       = jx_APCTLKrylovParamFixitPCTLR(apctlkrylov_param);
   JX_Int    fixit_pctl_E       = jx_APCTLKrylovParamFixitPCTLE(apctlkrylov_param);
   JX_Int    fixit_pctl_I       = jx_APCTLKrylovParamFixitPCTLI(apctlkrylov_param);
   JX_Int    fixit_brlx_R       = jx_APCTLKrylovParamFixitBrlxR(apctlkrylov_param);
   JX_Int    fixit_brlx_E       = jx_APCTLKrylovParamFixitBrlxE(apctlkrylov_param);
   JX_Int    fixit_brlx_I       = jx_APCTLKrylovParamFixitBrlxI(apctlkrylov_param);
   JX_Int    ARR_relax_type     = jx_APCTLKrylovParamARRRelaxType(apctlkrylov_param);
   JX_Int    AEE_relax_type     = jx_APCTLKrylovParamAEERelaxType(apctlkrylov_param);
   JX_Int    AII_relax_type     = jx_APCTLKrylovParamAIIRelaxType(apctlkrylov_param);
   JX_Int    use_fixedmode_R    = jx_APCTLKrylovParamUseFixedModeR(apctlkrylov_param);
   JX_Int    use_fixedmode_E    = jx_APCTLKrylovParamUseFixedModeE(apctlkrylov_param);
   JX_Int    use_fixedmode_I    = jx_APCTLKrylovParamUseFixedModeI(apctlkrylov_param);
   JX_Real theta_wc_E         = jx_APCTLKrylovParamThetaWCE(apctlkrylov_param);
   JX_Real threshold_wc_E     = jx_APCTLKrylovParamThresholdWCE(apctlkrylov_param);
   JX_Real theta_dd_R         = jx_APCTLKrylovParamThetaDDR(apctlkrylov_param);
   JX_Real theta_dd_E         = jx_APCTLKrylovParamThetaDDE(apctlkrylov_param);
   JX_Real theta_dd_I         = jx_APCTLKrylovParamThetaDDI(apctlkrylov_param);
   JX_Real threshold_dd_R     = jx_APCTLKrylovParamThresholdDDR(apctlkrylov_param);
   JX_Real threshold_dd_E     = jx_APCTLKrylovParamThresholdDDE(apctlkrylov_param);
   JX_Real threshold_dd_I     = jx_APCTLKrylovParamThresholdDDI(apctlkrylov_param);
   JX_Int    use_ppctl          = jx_APCTLKrylovParamUsePPCTL(apctlkrylov_param);
   JX_Int    test_subls_iter    = jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param);
   JX_Int    debug_flag         = jx_APCTLKrylovParamDebugFlag(apctlkrylov_param);
   JX_Int    reset_zero         = jx_APCTLKrylovParamResetZero(apctlkrylov_param);
   JX_Real ARR_relax_tol      = jx_APCTLKrylovParamARRRelaxTol(apctlkrylov_param);
   JX_Real AEE_relax_tol      = jx_APCTLKrylovParamAEERelaxTol(apctlkrylov_param);
   JX_Real AII_relax_tol      = jx_APCTLKrylovParamAIIRelaxTol(apctlkrylov_param);
   JX_Real ACC_relax_tol      = jx_APCTLKrylovParamACCRelaxTol(apctlkrylov_param);

   JX_Real strong_threshold = jx_APCTLKrylovParamStrongThreshold(apctlkrylov_param);
   JX_Int interp_type = jx_APCTLKrylovParamInterpType(apctlkrylov_param);
   JX_Int coarsen_type = jx_APCTLKrylovParamCoarsenType(apctlkrylov_param);
   JX_Int agg_num_levels = jx_APCTLKrylovParamAggNumLevels(apctlkrylov_param);
   JX_Int coarse_threshold = jx_APCTLKrylovParamCoarseThreshold(apctlkrylov_param);
   JX_Int print_level_amg = jx_APCTLKrylovParamPrintLevelAMG(apctlkrylov_param);

   JX_Int    num_iterations;
   JX_Real final_res_norm;
   JX_Real starttime  = 0.0;
   JX_Real endtime    = 0.0;
   JX_Real starttimeT = 0.0;
   JX_Real endtimeT   = 0.0;

   JX_Real cpu_trans_in  = 0.0;
   JX_Real cpu_trans_out = 0.0;
   JX_Real cpu_setup     = 0.0;
   JX_Real cpu_solve     = 0.0;
   JX_Real cpu_total     = 0.0;

   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL;
   jx_ParVector     *VRE = NULL;
   jx_ParVector    **VER = NULL;
   jx_ParVector     *VEI = NULL;
   jx_ParVector     *VIE = NULL;

   jx_ParCSRMatrix  *par_mat = NULL;
   jx_ParVector     *par_rhs = NULL;
   jx_ParVector     *par_sol = NULL;

   JX_Solver solver  = NULL;
   JX_Solver precond = NULL;

   JX_Int myid, nprocs, np_R;
   JX_Int groupid_x = MPI_UNDEFINED;
   JX_Int groupid_y = MPI_UNDEFINED;
   MPI_Comm comm = jx_ParCSRMatrixComm(ARR_p[0]);
   MPI_Comm comm_x, comm_bak, comm_y;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   np_R = nprocs / (ng + 2);
   groupid_x = myid / np_R;
   jx_MPI_Comm_split(comm, groupid_x, myid, &comm_x);
   jx_MPI_Comm_dup(comm_x, &comm_bak);

   groupid_y = myid % np_R;
   jx_MPI_Comm_split(comm, groupid_y, myid, &comm_y);

   if (TTest) starttime = jx_MPI_Wtime();

   jx_ParaDataTrans4ApctlmgKrylov( comm, comm_x, groupid_x,
                                   ARR_p, AEE_p, AII_p, VRE_p, VER_p, VEI_p, VIE_p,
                                   fR_p, fE_p, fI_p, uR_p, uE_p, uI_p, ng,
                                  &ARR, &AEE, &AII, &VRE, &VER, &VEI, &VIE,
                                  &par_mat, &par_rhs, &par_sol );

   if (TTest)
   {
      endtime = jx_MPI_Wtime();
      cpu_trans_in = jx_GetWallTimeMax(comm, starttime, endtime);
   }

   if (TTest) starttimeT = jx_MPI_Wtime();

   if (precond_id == 1) // APCTL
   {
      switch (solver_id)
      {
         case 2:  // APCTL-GMRES(m)
         {
            if (myid == 0 && print_level)
            {
               if (!use_ppctl)
               {
                  jx_printf("\n Solver: \033[31mAPCTL-GMRES(%d)\033[00m\n\n", k_dim);
               }
               else
               {
                  jx_printf("\n Solver: \033[31mPPCTL-GMRES(%d)\033[00m\n\n", k_dim);
               }
            }

            if (TTest) starttime = jx_MPI_Wtime();

            JX_3tAPCTLDataInitialize(&precond);

            JX_3tAPCTLSetNumGroup(precond, ng);
            JX_3tAPCTLSetA(precond, par_mat);
            JX_3tAPCTLSetARR(precond, ARR);
            JX_3tAPCTLSetAEE(precond, AEE);
            JX_3tAPCTLSetAII(precond, AII);
            JX_3tAPCTLSetVRE(precond, VRE);
            JX_3tAPCTLSetVER2(precond, VER);
            JX_3tAPCTLSetVEI(precond, VEI);
            JX_3tAPCTLSetVIE(precond, VIE);
            JX_3tAPCTLSetARRAll2(precond, ARR_p);
            JX_3tAPCTLSetAEEAll(precond, AEE_p);
            JX_3tAPCTLSetAIIAll(precond, AII_p);
            JX_3tAPCTLSetVREAll2(precond, VRE_p);
            JX_3tAPCTLSetVERAll2(precond, VER_p);
            JX_3tAPCTLSetVEIAll(precond, VEI_p);
            JX_3tAPCTLSetVIEAll(precond, VIE_p);
            JX_3tAPCTLSetMaxiter(precond, 1);
            JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
            JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type);
            JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR); /* peghoty 2011/10/30 */
            JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG); /* peghoty 2011/10/30 */
            JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR); /* peghoty 2011/10/30 */
            JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
            JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
            JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
            JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
            JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
            JX_3tAPCTLSetARRRelaxTol(precond, ARR_relax_tol);
            JX_3tAPCTLSetAEERelaxTol(precond, AEE_relax_tol);
            JX_3tAPCTLSetAIIRelaxTol(precond, AII_relax_tol);
            JX_3tAPCTLSetACCRelaxTol(precond, ACC_relax_tol);
            JX_3tAPCTLSetARRRelaxType(precond, ARR_relax_type);
            JX_3tAPCTLSetAEERelaxType(precond, AEE_relax_type);
            JX_3tAPCTLSetAIIRelaxType(precond, AII_relax_type);

            JX_3tAPCTLSetNpR(precond, np_R);
            JX_3tAPCTLSetComm(precond, comm);
            JX_3tAPCTLSetCommX(precond, comm_x);
            JX_3tAPCTLSetCommY(precond, comm_y);
            JX_3tAPCTLSetGroupIdX(precond, groupid_x);

            JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
            JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);

            JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
            JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
            JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
            JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
            JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
            JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);

            JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
            JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
            JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
            JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
            JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
            JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

            JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
            JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
            JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */

            JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
            JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

            JX_3tAPCTLSetDebugFlag(precond, debug_flag);
            JX_3tAPCTLSetResetZero(precond, reset_zero);
            JX_3tAPCTLSetStrongThreshold(precond, strong_threshold);
            JX_3tAPCTLSetInterpType(precond, interp_type);
            JX_3tAPCTLSetCoarsenType(precond, coarsen_type);
            JX_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
            JX_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
            JX_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

            JX_ParCSRGMRESCreate(comm, &solver);
            JX_GMRESSetKDim(solver, k_dim);
            JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
            JX_GMRESSetMaxIter(solver, max_iter);
            JX_GMRESSetTol(solver, tol);
            JX_GMRESSetLogging(solver, 1);
            JX_GMRESSetPrintLevel(solver, print_level);

            JX_GMRESSetPrecond( solver,
                                (JX_PtrToSolverFcn) JX_3tAPCTLmgPrecond,
                                (JX_PtrToSolverFcn) NULL,
                                precond );

            JX_3tAPCTLSetup4mgJasmin(precond, (JX_ParCSRMatrix)par_mat);

            JX_GMRESSetup( solver, 
                           (JX_Matrix)par_mat,
                           (JX_Vector)par_rhs,
                           (JX_Vector)par_sol );

            if (TTest)
            {
               endtime = jx_MPI_Wtime();
               cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);
            }

            if (TTest) starttime = jx_MPI_Wtime();

            JX_GMRESSolve( solver,
                           (JX_Matrix)par_mat, // preOperator
                           (JX_Matrix)par_mat,
                           (JX_Vector)par_rhs,
                           (JX_Vector)par_sol );

            if (TTest)
            {
               endtime = jx_MPI_Wtime();
               cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime);
            }

            JX_GMRESGetNumIterations(solver, &num_iterations);
            JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
            jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
            if (print_level_apctl)
            {
               JX_3tAPCTLIterCount(precond);
            }

            JX_ParCSRGMRESDestroy(solver);
         }
         break;
      }
   }
   else if (precond_id == 2) // ABSC1
   {
      switch (solver_id)
      {
         case 2:  // ABSC1-GMRES(m)
         {
            if (myid == 0 && print_level)
            {
               jx_printf("\n Solver: \033[31mABSC1-GMRES(%d)\033[00m\n\n", k_dim);
            }

            if (TTest) starttime = jx_MPI_Wtime();

            JX_3tAPCTLDataInitialize(&precond);

            JX_3tAPCTLSetNumGroup(precond, ng);
            JX_3tAPCTLSetA(precond, par_mat);
            JX_3tAPCTLSetARR(precond, ARR);
            JX_3tAPCTLSetAEE(precond, AEE);
            JX_3tAPCTLSetAII(precond, AII);
            JX_3tAPCTLSetVRE(precond, VRE);
            JX_3tAPCTLSetVER2(precond, VER);
            JX_3tAPCTLSetVEI(precond, VEI);
            JX_3tAPCTLSetVIE(precond, VIE);
            JX_3tAPCTLSetMaxiter(precond, 1);
            JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
            JX_3tAPCTLSetARRRelaxTol(precond, ARR_relax_tol);
            JX_3tAPCTLSetAEERelaxTol(precond, AEE_relax_tol);
            JX_3tAPCTLSetAIIRelaxTol(precond, AII_relax_tol);
            JX_3tAPCTLSetARRRelaxType(precond, ARR_relax_type);
            JX_3tAPCTLSetAEERelaxType(precond, AEE_relax_type);
            JX_3tAPCTLSetAIIRelaxType(precond, AII_relax_type);

            JX_3tAPCTLSetNpR(precond, np_R);
            JX_3tAPCTLSetComm(precond, comm);
            JX_3tAPCTLSetCommX(precond, comm_x);
            JX_3tAPCTLSetCommY(precond, comm_y);
            JX_3tAPCTLSetGroupIdX(precond, groupid_x);

            JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
            JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);

            JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
            JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
            JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
            JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
            JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
            JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);

            JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
            JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
            JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
            JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
            JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
            JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

            JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
            JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
            JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */

            JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

            JX_3tAPCTLSetDebugFlag(precond, debug_flag);
            JX_3tAPCTLSetResetZero(precond, reset_zero);
            JX_3tAPCTLSetStrongThreshold(precond, strong_threshold);
            JX_3tAPCTLSetInterpType(precond, interp_type);
            JX_3tAPCTLSetCoarsenType(precond, coarsen_type);
            JX_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
            JX_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
            JX_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

            JX_ParCSRGMRESCreate(comm, &solver);
            JX_GMRESSetKDim(solver, k_dim);
            JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
            JX_GMRESSetMaxIter(solver, max_iter);
            JX_GMRESSetTol(solver, tol);
            JX_GMRESSetLogging(solver, 1);
            JX_GMRESSetPrintLevel(solver, print_level);

            JX_GMRESSetPrecond( solver,
                                (JX_PtrToSolverFcn) JX_3tABSC1mgPrecond,
                                (JX_PtrToSolverFcn) NULL,
                                precond );

            JX_3tABSC1Setup4mgJasmin(precond, (JX_ParCSRMatrix)par_mat);

            JX_GMRESSetup( solver, 
                           (JX_Matrix)par_mat,
                           (JX_Vector)par_rhs,
                           (JX_Vector)par_sol );

            if (TTest)
            {
               endtime = jx_MPI_Wtime();
               cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);
            }

            if (TTest) starttime = jx_MPI_Wtime();

            JX_GMRESSolve( solver,
                           (JX_Matrix)par_mat, // preOperator
                           (JX_Matrix)par_mat,
                           (JX_Vector)par_rhs,
                           (JX_Vector)par_sol );

            if (TTest)
            {
               endtime = jx_MPI_Wtime();
               cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime);
            }

            JX_GMRESGetNumIterations(solver, &num_iterations);
            JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
            jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
            if (print_level_apctl)
            {
               JX_3tAPCTLIterCount(precond);
            }

            JX_ParCSRGMRESDestroy(solver);
         }
         break;
      }
   }
   else if (precond_id == 3) // ABSC2
   {
      switch (solver_id)
      {
         case 2:  // ABSC2-GMRES(m)
         {
            if (myid == 0 && print_level)
            {
               jx_printf("\n Solver: \033[31mABSC2-GMRES(%d)\033[00m\n\n", k_dim);
            }

            if (TTest) starttime = jx_MPI_Wtime();

            JX_3tAPCTLDataInitialize(&precond);

            JX_3tAPCTLSetNumGroup(precond, ng);
            JX_3tAPCTLSetA(precond, par_mat);
            JX_3tAPCTLSetARR(precond, ARR);
            JX_3tAPCTLSetAEE(precond, AEE);
            JX_3tAPCTLSetAII(precond, AII);
            JX_3tAPCTLSetVRE(precond, VRE);
            JX_3tAPCTLSetVER2(precond, VER);
            JX_3tAPCTLSetVEI(precond, VEI);
            JX_3tAPCTLSetVIE(precond, VIE);
            JX_3tAPCTLSetMaxiter(precond, 1);
            JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
            JX_3tAPCTLSetARRRelaxTol(precond, ARR_relax_tol);
            JX_3tAPCTLSetAEERelaxTol(precond, AEE_relax_tol);
            JX_3tAPCTLSetAIIRelaxTol(precond, AII_relax_tol);
            JX_3tAPCTLSetARRRelaxType(precond, ARR_relax_type);
            JX_3tAPCTLSetAEERelaxType(precond, AEE_relax_type);
            JX_3tAPCTLSetAIIRelaxType(precond, AII_relax_type);

            JX_3tAPCTLSetNpR(precond, np_R);
            JX_3tAPCTLSetComm(precond, comm);
            JX_3tAPCTLSetCommX(precond, comm_x);
            JX_3tAPCTLSetCommY(precond, comm_y);
            JX_3tAPCTLSetGroupIdX(precond, groupid_x);

            JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
            JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);

            JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
            JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
            JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
            JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
            JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
            JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);

            JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
            JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
            JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
            JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
            JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
            JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

            JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
            JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
            JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */

            JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

            JX_3tAPCTLSetDebugFlag(precond, debug_flag);
            JX_3tAPCTLSetResetZero(precond, reset_zero);
            JX_3tAPCTLSetStrongThreshold(precond, strong_threshold);
            JX_3tAPCTLSetInterpType(precond, interp_type);
            JX_3tAPCTLSetCoarsenType(precond, coarsen_type);
            JX_3tAPCTLSetAggNumLevels(precond, agg_num_levels);
            JX_3tAPCTLSetCoarseThreshold(precond, coarse_threshold);
            JX_3tAPCTLSetPrintLevelAMG(precond, print_level_amg);

            JX_ParCSRGMRESCreate(comm, &solver);
            JX_GMRESSetKDim(solver, k_dim);
            JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
            JX_GMRESSetMaxIter(solver, max_iter);
            JX_GMRESSetTol(solver, tol);
            JX_GMRESSetLogging(solver, 1);
            JX_GMRESSetPrintLevel(solver, print_level);

            JX_GMRESSetPrecond( solver,
                                (JX_PtrToSolverFcn) JX_3tABSC2mgPrecond,
                                (JX_PtrToSolverFcn) NULL,
                                precond );

            JX_3tABSC2Setup4mgJasmin(precond, (JX_ParCSRMatrix)par_mat);

            JX_GMRESSetup( solver, 
                           (JX_Matrix)par_mat,
                           (JX_Vector)par_rhs,
                           (JX_Vector)par_sol );

            if (TTest)
            {
               endtime = jx_MPI_Wtime();
               cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);
            }

            if (TTest) starttime = jx_MPI_Wtime();

            JX_GMRESSolve( solver,
                           (JX_Matrix)par_mat, // preOperator
                           (JX_Matrix)par_mat,
                           (JX_Vector)par_rhs,
                           (JX_Vector)par_sol );

            if (TTest)
            {
               endtime = jx_MPI_Wtime();
               cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime);
            }

            JX_GMRESGetNumIterations(solver, &num_iterations);
            JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
            jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
            if (print_level_apctl)
            {
               JX_3tAPCTLIterCount(precond);
            }

            JX_ParCSRGMRESDestroy(solver);
         }
         break;
      }
   }

   if (TTest)
   {
      endtimeT = jx_MPI_Wtime();
      cpu_total = jx_GetWallTimeMax(comm, starttimeT, endtimeT);
   }

   //=========================================================
   //  Step 4: 获取子系统迭代次数统计信息
   //=========================================================
   if (jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param))
   {
      jx_GetAPCTLmgNumIterOfSubLS(precond, groupid_x, ng, apctlkrylov_param);
   }
   JX_3tAPCTLDestroy4mgJasmin(precond);

   //=========================================================
   //  Step 5: 将近似解向量保存到文件中
   //=========================================================

   if (keepsol)
   {
      jx_Vector *ser_sol = NULL;
      ser_sol = jx_ParVectorToVectorAll(par_sol);
      if (myid == 0)
      {
         jx_SeqVectorPrint(ser_sol, "./app");
      }
      jx_SeqVectorDestroy(ser_sol);
   }

   //=========================================================
   //  Step 6: 将近似解向量从 par_sol 还原到 uR_p, uE_p, uI_p
   //=========================================================

   if (TTest) starttime = jx_MPI_Wtime();
 
   jx_APCTLKrylovSolBack4mgJasmin(comm, comm_bak, groupid_x, ng, par_sol, uR_p, uE_p, uI_p);

   if (TTest)
   {
      endtime = jx_MPI_Wtime();
      cpu_trans_out = jx_GetWallTimeMax(comm, starttime, endtime);
   }

   //==================================================================================
   //  Step 7: 获取数据转换，Setup 和 Solve 过程的 CPU 时间
   //==================================================================================

   jx_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans_in + cpu_trans_out);
   jx_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jx_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jx_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total);

   //=========================================================
   //  Step 8: 释放内存
   //=========================================================

   jx_ParCSRMatrixDestroy(par_mat);
   jx_ParVectorDestroy(par_rhs);
   jx_ParVectorDestroy(par_sol);
   jx_MPI_Comm_free(&comm_bak);

   return (0);
}

/*!
 * \fn JX_Int jx_ApctlKrylov_JASMIN_sp
 * \brief APCTL-GMRES solvers for single-processor case.
 * \author peghoty
 * \date 2012/02/25 
 */
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
                          jx_APCTLKrylovParam *apctlkrylov_param )
{                   
   JX_Int    solver_id          = jx_APCTLKrylovParamSolverID(apctlkrylov_param);
   JX_Real tol                = jx_APCTLKrylovParamTol(apctlkrylov_param);
   JX_Int    max_iter           = jx_APCTLKrylovParamMaxIter(apctlkrylov_param);
   JX_Int    k_dim              = jx_APCTLKrylovParamKDim(apctlkrylov_param);
   JX_Int    is_check_restarted = jx_APCTLKrylovParamISCheckRestarted(apctlkrylov_param);
   JX_Int    two_norm           = jx_APCTLKrylovParamTwoNorm(apctlkrylov_param);
   JX_Int    print_level        = jx_APCTLKrylovParamPrintLevel(apctlkrylov_param);  
   JX_Int    TTest              = jx_APCTLKrylovParamTTest(apctlkrylov_param);       
   JX_Int    keepsol            = jx_APCTLKrylovParamKeepSol(apctlkrylov_param);

   JX_Int    print_level_apctl  = jx_APCTLKrylovParamPrintLevelAPCTL(apctlkrylov_param);   
   JX_Int    blocksmooth_type   = jx_APCTLKrylovParamBlockSmoothType(apctlkrylov_param);   
   JX_Int    num_relax_pre      = jx_APCTLKrylovParamNumRelaxPre(apctlkrylov_param);       
   JX_Int    num_relax_post     = jx_APCTLKrylovParamNumRelaxPost(apctlkrylov_param);      
   JX_Int    interp_solver_ARR  = jx_APCTLKrylovParamInterpSolverARR(apctlkrylov_param);   
   JX_Int    interp_kdim_ARR    = jx_APCTLKrylovParamInterpKdimARR(apctlkrylov_param);     
   JX_Int    interp_maxit_ARR   = jx_APCTLKrylovParamInterpMaxitARR(apctlkrylov_param);    
   JX_Int    interp_maxit_AII   = jx_APCTLKrylovParamInterpMaxitAII(apctlkrylov_param);    
   JX_Real interp_tol_ARR     = jx_APCTLKrylovParamInterpTolARR(apctlkrylov_param);      
   JX_Real interp_tol_AII     = jx_APCTLKrylovParamInterpTolAII(apctlkrylov_param);      
   JX_Int    fixit_pctl_R       = jx_APCTLKrylovParamFixitPCTLR(apctlkrylov_param);        
   JX_Int    fixit_pctl_E       = jx_APCTLKrylovParamFixitPCTLE(apctlkrylov_param);        
   JX_Int    fixit_pctl_I       = jx_APCTLKrylovParamFixitPCTLI(apctlkrylov_param);        
   JX_Int    fixit_brlx_R       = jx_APCTLKrylovParamFixitBrlxR(apctlkrylov_param);        
   JX_Int    fixit_brlx_E       = jx_APCTLKrylovParamFixitBrlxE(apctlkrylov_param);        
   JX_Int    fixit_brlx_I       = jx_APCTLKrylovParamFixitBrlxI(apctlkrylov_param);        
   JX_Int    use_fixedmode_R    = jx_APCTLKrylovParamUseFixedModeR(apctlkrylov_param);     
   JX_Int    use_fixedmode_E    = jx_APCTLKrylovParamUseFixedModeE(apctlkrylov_param);     
   JX_Int    use_fixedmode_I    = jx_APCTLKrylovParamUseFixedModeI(apctlkrylov_param);     
   JX_Real theta_wc_E         = jx_APCTLKrylovParamThetaWCE(apctlkrylov_param);          
   JX_Real threshold_wc_E     = jx_APCTLKrylovParamThresholdWCE(apctlkrylov_param);      
   JX_Real theta_dd_R         = jx_APCTLKrylovParamThetaDDR(apctlkrylov_param);          
   JX_Real theta_dd_E         = jx_APCTLKrylovParamThetaDDE(apctlkrylov_param);          
   JX_Real theta_dd_I         = jx_APCTLKrylovParamThetaDDI(apctlkrylov_param);          
   JX_Real threshold_dd_R     = jx_APCTLKrylovParamThresholdDDR(apctlkrylov_param);      
   JX_Real threshold_dd_E     = jx_APCTLKrylovParamThresholdDDE(apctlkrylov_param);      
   JX_Real threshold_dd_I     = jx_APCTLKrylovParamThresholdDDI(apctlkrylov_param);      
   JX_Int    use_ppctl          = jx_APCTLKrylovParamUsePPCTL(apctlkrylov_param);
   JX_Int    test_subls_iter    = jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param);
   JX_Real tol_default        = 1.0e-6;

   JX_Int    num_iterations;
   JX_Real final_res_norm; 
   JX_Real starttime  = 0.0;
   JX_Real endtime    = 0.0;  
   JX_Real starttimeT = 0.0;
   JX_Real endtimeT   = 0.0;      

   JX_Real cpu_trans_in  = 0.0;
   JX_Real cpu_trans_out = 0.0;
   JX_Real cpu_setup     = 0.0;
   JX_Real cpu_solve     = 0.0;
   JX_Real cpu_total     = 0.0;  
                 
   jx_ParCSRMatrix  *par_mat = NULL;
   jx_ParVector     *par_rhs = NULL;
   jx_ParVector     *par_sol = NULL;

   JX_Solver solver  = NULL;
   JX_Solver precond = NULL;

   JX_Int myid, nprocs;
   MPI_Comm comm = jx_ParCSRMatrixComm(ARR_p);
         
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (TTest) starttime = jx_MPI_Wtime();
             
   jx_DataCombine4ApctlKrylov( ARR_p, AEE_p, AII_p, 
                               VRE_p, VER_p, VEI_p, VIE_p,
                               fR_p, fE_p, fI_p,
                               uR_p, uE_p, uI_p,
                               &par_mat, &par_rhs, &par_sol ); 

   if (TTest)
   {
      endtime = jx_MPI_Wtime();
      cpu_trans_in = jx_GetWallTimeMax(comm, starttime, endtime);  
   }     


   if (TTest) starttimeT = jx_MPI_Wtime();
                                          
   switch (solver_id)
   {

      case 1:  // APCTL-CG
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jx_printf("\n Solver: \033[31mAPCTL-CG\033[00m\n\n");
            else
               jx_printf("\n Solver: \033[31mPPCTL-CG\033[00m\n\n");
         }
         
         if (TTest) starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&precond);

         JX_3tAPCTLSetA(precond, par_mat);
         JX_3tAPCTLSetComm(precond, comm);
         JX_3tAPCTLSetARR(precond, ARR_p);
         JX_3tAPCTLSetAEE(precond, AEE_p);
         JX_3tAPCTLSetAII(precond, AII_p);
         JX_3tAPCTLSetVRE(precond, VRE_p);
         JX_3tAPCTLSetVER(precond, VER_p);
         JX_3tAPCTLSetVEI(precond, VEI_p);
         JX_3tAPCTLSetVIE(precond, VIE_p);        
         JX_3tAPCTLSetMaxiter(precond, 1);
         JX_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JX_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR);  /* peghoty 2011/10/30 */
         JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG);         /* peghoty 2011/10/30 */
         JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR);        /* peghoty 2011/10/30 */          
         JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JX_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JX_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JX_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JX_3tAPCTLSetACCRelaxTol(precond, tol_default);   

         JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */

         JX_ParCSRPCGCreate(comm, &solver); 
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, two_norm);  // 0: B 范数； 1：l2 范数 
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);
         
         JX_PCGSetPrecond( solver,
                           (JX_PtrToSolverFcn) JX_3tAPCTLPrecond,
                           (JX_PtrToSolverFcn) NULL,
                           precond ); 
         
         JX_3tAPCTLSetup4Jasmin( precond, (JX_ParCSRMatrix)par_mat );                   
         
         JX_PCGSetup ( solver, 
                       (JX_Matrix) par_mat, 
                       (JX_Vector) par_rhs, 
                       (JX_Vector) par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PCGSolve ( solver, 
                       (JX_Matrix) par_mat, // preOperater
                       (JX_Matrix) par_mat, 
                       (JX_Vector) par_rhs, 
                       (JX_Vector) par_sol );  

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime); 
         }

         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);

         if (print_level_apctl)
         { 
            JX_3tAPCTLIterCount(precond);
         }

         JX_ParCSRPCGDestroy(solver);
      }
      break;
      

      case 2:  // APCTL-GMRES(m)
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jx_printf("\n Solver: \033[31mAPCTL-GMRES(%d)\033[00m\n\n", k_dim);
            else
               jx_printf("\n Solver: \033[31mPPCTL-GMRES(%d)\033[00m\n\n", k_dim);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&precond);

         JX_3tAPCTLSetA(precond, par_mat);
         JX_3tAPCTLSetComm(precond, comm);
         JX_3tAPCTLSetARR(precond, ARR_p);
         JX_3tAPCTLSetAEE(precond, AEE_p);
         JX_3tAPCTLSetAII(precond, AII_p);
         JX_3tAPCTLSetVRE(precond, VRE_p);
         JX_3tAPCTLSetVER(precond, VER_p);
         JX_3tAPCTLSetVEI(precond, VEI_p);
         JX_3tAPCTLSetVIE(precond, VIE_p);        
         JX_3tAPCTLSetMaxiter(precond, 1);
         JX_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JX_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR);  /* peghoty 2011/10/30 */
         JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG);         /* peghoty 2011/10/30 */
         JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR);        /* peghoty 2011/10/30 */          
         JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JX_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JX_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JX_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JX_3tAPCTLSetACCRelaxTol(precond, tol_default);   

         JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */
                  
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);

         JX_GMRESSetPrecond( solver,
                             (JX_PtrToSolverFcn) JX_3tAPCTLPrecond,
                             (JX_PtrToSolverFcn) NULL,
                             precond );            
 
         JX_3tAPCTLSetup4Jasmin( precond, (JX_ParCSRMatrix)par_mat );
 
         JX_GMRESSetup( solver, 
                        (JX_Matrix)par_mat,
                        (JX_Vector)par_rhs,
                        (JX_Vector)par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve( solver,
                        (JX_Matrix)par_mat, // preOperator
                        (JX_Matrix)par_mat,
                        (JX_Vector)par_rhs,
                        (JX_Vector)par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime); 
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
  
         if (print_level_apctl)
         { 
            JX_3tAPCTLIterCount(precond);
         }

         JX_ParCSRGMRESDestroy(solver);
      }
      break;


      case 3:  // APCTL-BiCGSTab
      {
         if (myid == 0 && print_level) 
         {
            if (!use_ppctl)
               jx_printf("\n Solver: \033[31mAPCTL-BiCGSTab\033[00m\n\n");
            else
               jx_printf("\n Solver: \033[31mPPCTL-BiCGSTab\033[00m\n\n");
         }
         
         if (TTest) starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&precond);

         JX_3tAPCTLSetA(precond, par_mat);
         JX_3tAPCTLSetComm(precond, comm);
         JX_3tAPCTLSetARR(precond, ARR_p);
         JX_3tAPCTLSetAEE(precond, AEE_p);
         JX_3tAPCTLSetAII(precond, AII_p);
         JX_3tAPCTLSetVRE(precond, VRE_p);
         JX_3tAPCTLSetVER(precond, VER_p);
         JX_3tAPCTLSetVEI(precond, VEI_p);
         JX_3tAPCTLSetVIE(precond, VIE_p);        
         JX_3tAPCTLSetMaxiter(precond, 1);
         JX_3tAPCTLSetNumRlxPre(precond, num_relax_pre);
         JX_3tAPCTLSetNumRlxPost(precond, num_relax_post);
         JX_3tAPCTLSetPrintLevel(precond, print_level_apctl);
         JX_3tAPCTLSetBlockSmoothType(precond, blocksmooth_type); 
         JX_3tAPCTLSetARRSolverID(precond, interp_solver_ARR);  /* peghoty 2011/10/30 */
         JX_3tAPCTLSetACCSolverID(precond, SOLVER_AMG);         /* peghoty 2011/10/30 */
         JX_3tAPCTLSetARRKDim(precond, interp_kdim_ARR);        /* peghoty 2011/10/30 */          
         JX_3tAPCTLSetARRInterpMaxIt(precond, interp_maxit_ARR);
         JX_3tAPCTLSetAIIInterpMaxIt(precond, interp_maxit_AII);
         JX_3tAPCTLSetACCRelaxMaxIt(precond, 1);
         JX_3tAPCTLSetARRInterpTol(precond, interp_tol_ARR);
         JX_3tAPCTLSetAIIInterpTol(precond, interp_tol_AII);
         JX_3tAPCTLSetARRRelaxTol(precond, tol_default);
         JX_3tAPCTLSetAEERelaxTol(precond, tol_default);
         JX_3tAPCTLSetAIIRelaxTol(precond, tol_default);
         JX_3tAPCTLSetACCRelaxTol(precond, tol_default);   

         JX_3tAPCTLSetThetaWCE(precond, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(precond, threshold_wc_E);
         JX_3tAPCTLSetThetaDDR(precond, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(precond, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(precond, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(precond, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(precond, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(precond, threshold_dd_I);    

         JX_3tAPCTLSetFixItPCTLR(precond, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(precond, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(precond, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(precond, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(precond, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(precond, fixit_brlx_I);

         JX_3tAPCTLSetUseFixedModeR(precond, use_fixedmode_R); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeE(precond, use_fixedmode_E); /* peghoty  2012/02/15 */
         JX_3tAPCTLSetUseFixedModeI(precond, use_fixedmode_I); /* peghoty  2012/02/15 */
         
         JX_3tAPCTLSetUsePPCTL(precond, use_ppctl); /* peghoty  2012/03/06 */
         JX_3tAPCTLSetTestSubLSIter(precond, test_subls_iter); /* peghoty  2012/03/24 */
                  
         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);

         JX_BiCGSTABSetPrecond ( solver,
                                 (JX_PtrToSolverFcn) JX_3tAPCTLPrecond,
                                 (JX_PtrToSolverFcn) NULL,
                                 precond ); 

         JX_3tAPCTLSetup4Jasmin( precond, (JX_ParCSRMatrix)par_mat );

         JX_BiCGSTABSetup ( solver, 
                            (JX_Matrix) par_mat, 
		            (JX_Vector) par_rhs, 
		            (JX_Vector) par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_setup = jx_GetWallTimeMax(comm, starttime, endtime);  
         }               


         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_BiCGSTABSolve ( solver, 
                            (JX_Matrix) par_mat, // preOperater
                            (JX_Matrix) par_mat, 
		            (JX_Vector) par_rhs, 
		            (JX_Vector) par_sol );

         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            cpu_solve = jx_GetWallTimeMax(comm, starttime, endtime); 
         }

         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         jx_APCTLKrylovParamSetNumIterations(apctlkrylov_param, num_iterations);
         jx_APCTLKrylovParamSetLastRelNrm(apctlkrylov_param, final_res_norm);
 
         if (print_level_apctl)
         { 
            JX_3tAPCTLIterCount(precond);
         }

         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

   }  // end switch
       
   if (TTest)
   {
      endtimeT = jx_MPI_Wtime();
      cpu_total = jx_GetWallTimeMax(comm, starttimeT, endtimeT);      
   }
   
  
   //=========================================================
   //  Step 3: 获取子系统迭代次数统计信息
   //=========================================================
      
   if (jx_APCTLKrylovParamTestSubLSIter(apctlkrylov_param))
   {   
      jx_GetAPCTLNumIterOfSubLS(precond, apctlkrylov_param);
   }
   JX_3tAPCTLDestroy4Jasmin(precond); 
      
  
   //=========================================================
   //  Step 4: 将近似解向量保存到文件中
   //=========================================================
    
   if (keepsol)
   {
      jx_Vector *ser_sol = jx_ParVectorLocalVector(par_sol);
      if (myid == 0) 
      {
#if 1     
         jx_SeqVectorPrint(ser_sol, "./app");
#else       
         jx_Vector *solR = NULL;
         jx_Vector *solE = NULL;
         jx_Vector *solI = NULL;
         jx_3tGetSubVecs(ser_sol, &solR, &solE, &solI);
         jx_SeqVectorPrint(solR, "./uR-1");
         jx_SeqVectorPrint(solE, "./uE-1");
         jx_SeqVectorPrint(solI, "./uI-1");
         jx_SeqVectorDestroy(solR);
         jx_SeqVectorDestroy(solE);
         jx_SeqVectorDestroy(solI);      
#endif
      }
   }


   //=========================================================
   //  Step 5: 将近似解向量从 par_sol 还原到 uR_p, uE_p, uI_p
   //=========================================================
   
   if (TTest) starttime = jx_MPI_Wtime();
   
   jx_Vector *uR = jx_ParVectorLocalVector(uR_p);
   jx_Vector *uE = jx_ParVectorLocalVector(uE_p);
   jx_Vector *uI = jx_ParVectorLocalVector(uI_p);
   jx_Vector *sol = jx_ParVectorLocalVector(par_sol);
   JX_Int n = jx_ParCSRMatrixGlobalNumRows(ARR_p);
   
   memcpy(jx_VectorData(uR), jx_VectorData(sol), n*sizeof(JX_Real));
   memcpy(jx_VectorData(uE), jx_VectorData(sol)+n, n*sizeof(JX_Real));
   memcpy(jx_VectorData(uI), jx_VectorData(sol)+2*n, n*sizeof(JX_Real));

   if (TTest)
   {
      endtime = jx_MPI_Wtime();
      cpu_trans_out = jx_GetWallTimeMax(comm, starttime, endtime);   
   } 

   //===============================================================================
   //  Step 6: 获取数据转换，Setup 和 Solve 过程的 CPU 时间
   //===============================================================================  
   
   jx_APCTLKrylovParamSetCPUTrans(apctlkrylov_param, cpu_trans_in + cpu_trans_out);
   jx_APCTLKrylovParamSetCPUSetup(apctlkrylov_param, cpu_setup);
   jx_APCTLKrylovParamSetCPUSolve(apctlkrylov_param, cpu_solve);
   jx_APCTLKrylovParamSetCPUTotal(apctlkrylov_param, cpu_total); 
   
      
   //=========================================================
   //  Step 7: 释放内存
   //=========================================================
   
   jx_ParCSRMatrixDestroy(par_mat);
   jx_ParVectorDestroy(par_rhs);
   jx_ParVectorDestroy(par_sol);

   return (0);
}                      

/*!
 * \fn JX_Int JX_3tAPCTLSetup4Jasmin
 * \brief Setup phase of PCTL iteration or preconditioner for JASMIN interface.
 * \author peghoty
 * \date 2012/02/26
 */
JX_Int 
JX_3tAPCTLSetup4Jasmin( JX_Solver solver, JX_ParCSRMatrix A )
{   
   return( jx_3tAPCTLSetup4Jasmin( (void *) solver, (jx_ParCSRMatrix *) A ) );
}

JX_Int 
JX_3tAPCTLSetup4mgJasmin( JX_Solver solver, JX_ParCSRMatrix A )
{
   return( jx_3tAPCTLSetup4mgJasmin( (void *) solver, (jx_ParCSRMatrix *) A ) );
}

JX_Int 
JX_3tABSC1Setup4mgJasmin( JX_Solver solver, JX_ParCSRMatrix A )
{
   return( jx_3tABSC1Setup4mgJasmin( (void *) solver, (jx_ParCSRMatrix *) A ) );
}

JX_Int 
JX_3tABSC2Setup4mgJasmin( JX_Solver solver, JX_ParCSRMatrix A )
{
   return( jx_3tABSC2Setup4mgJasmin( (void *) solver, (jx_ParCSRMatrix *) A ) );
}

/*!
 * \fn JX_Int jx_3tAPCTLSetup4Jasmin
 * \brief Setup phase of PCTL Iteration or preconditioner for JASMIN interface. 
 * \author peghoty 
 * \date 2012/02/26
 */
JX_Int
jx_3tAPCTLSetup4Jasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{  
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   
   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);
      
   if (nprocs == 1)
   {
      jx_3tAPCTLSetup4Jasmin_sp(pre_3tapctl_data, A);
   }
   else if (nprocs > 1)
   { 
      jx_3tAPCTLSetup4Jasmin_mp(pre_3tapctl_data, A); 
   }
   
   return 0;
}

JX_Int
jx_3tAPCTLSetup4mgJasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
 
   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jx_3tAPCTLSetup4mgJasmin_mp(pre_3tapctl_data, A);
   }

   return 0;
}

JX_Int
jx_3tABSC1Setup4mgJasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   JX_Int nprocs;

   jx_MPI_Comm_size(comm, &nprocs);

   if ((nprocs > 1) && (nprocs % (jx_3tAPCTLDataNumGroup(pre_3tapctl_data)+2) == 0))
   {
      jx_3tABSC1Setup4mgJasmin_mp(pre_3tapctl_data, A);
   }

   return 0;
}

JX_Int
jx_3tABSC2Setup4mgJasmin( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   JX_Int nprocs;

   jx_MPI_Comm_size(comm, &nprocs);

   if ((nprocs > 1) && (nprocs % (jx_3tAPCTLDataNumGroup(pre_3tapctl_data)+2) == 0))
   {
      jx_3tABSC2Setup4mgJasmin_mp(pre_3tapctl_data, A);
   }

   return 0;
}

/*!
 * \fn JX_Int jx_3tAPCTLSetup4Jasmin_mp
 * \brief Setup phase of PCTL Iteration or preconditioner(for multi-processor case). 
 * \author peghoty 
 * \date 2012/02/25
 */
JX_Int
jx_3tAPCTLSetup4Jasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   JX_Real starttime = 0.0, endtime = 0.0;
     
   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JX_Real theta_wc_E     = jx_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JX_Real threshold_wc_E = jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JX_Real theta_dd_R     = jx_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JX_Real theta_dd_E     = jx_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JX_Real theta_dd_I     = jx_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JX_Real threshold_dd_R = jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JX_Real threshold_dd_E = jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JX_Real threshold_dd_I = jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data);  

   JX_Int ARR_interp_maxit = jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JX_Int AII_interp_maxit = jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);    
   JX_Int ACC_relax_maxit  = jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data);

   JX_Real  ARR_interp_tol = jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JX_Real  AII_interp_tol = jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JX_Real  ARR_relax_tol  = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real  AEE_relax_tol  = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real  AII_relax_tol  = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JX_Real  ACC_relax_tol  = jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   JX_Int ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);

   JX_Int ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   JX_Int ARR_kdim = jx_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JX_Int ACC_kdim = jx_3tAPCTLDataACCKDim(pre_3tapctl_data);  

   JX_Int debug_flag = jx_3tAPCTLDataDebugFlag(pre_3tapctl_data);

   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);  
   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
   JX_Int use_ppctl = jx_3tAPCTLDataUsePPCTL(pre_3tapctl_data);
  
   jx_ParAMGData   *ARR_amg_solver = NULL;
   jx_ParAMGData   *AEE_amg_solver = NULL;
   jx_ParAMGData   *AII_amg_solver = NULL;
   jx_ParAMGData   *ACC_amg_solver = NULL;

   jx_GMRESData    *ARR_gmres_solver = NULL;
   jx_GMRESData    *ACC_gmres_solver = NULL; 

   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);  
   jx_ParVector    *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector    *VER = jx_3tAPCTLDataVER(pre_3tapctl_data);
   jx_ParVector    *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector    *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);  

   jx_ParCSRMatrix *ARR_all = jx_3tAPCTLDataARRAll(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE_all = jx_3tAPCTLDataAEEAll(pre_3tapctl_data);
   jx_ParCSRMatrix *AII_all = jx_3tAPCTLDataAIIAll(pre_3tapctl_data);  
   jx_ParVector    *VRE_all = jx_3tAPCTLDataVREAll(pre_3tapctl_data);
   jx_ParVector    *VER_all = jx_3tAPCTLDataVERAll(pre_3tapctl_data);
   jx_ParVector    *VEI_all = jx_3tAPCTLDataVEIAll(pre_3tapctl_data);
   jx_ParVector    *VIE_all = jx_3tAPCTLDataVIEAll(pre_3tapctl_data);  
      
   jx_ParCSRMatrix *P   = NULL;  
   jx_ParCSRMatrix *ACC = NULL; 
     
   jx_ParVector    *WRR = NULL;
   jx_ParVector    *WEE = NULL;
   jx_ParVector    *WII = NULL;
   jx_ParVector    *WCC = NULL;

   jx_ParVector    *GCC = NULL;
 
   jx_ParVector    *RES = NULL;
   jx_ParVector    *RHS = NULL;
   jx_ParVector    *JAC = NULL;
   
   jx_ParVector    *PRR = NULL;
   jx_ParVector    *PII = NULL;

   JX_Int    IS_DD_R;
   JX_Int    IS_DD_E;    
   JX_Int    IS_DD_I;

   JX_Int ARR_relax_maxit;
   JX_Int AEE_relax_maxit;
   JX_Int AII_relax_maxit;
   
   JX_Int Need_CC;
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data); 

   JX_Real strong_threshold = jx_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JX_Int interp_type = jx_3tAPCTLDataInterpType(pre_3tapctl_data);
   JX_Int coarsen_type = jx_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JX_Int agg_num_levels = jx_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JX_Int coarse_threshold = jx_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JX_Int print_level_amg = jx_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JX_Int ARR_relax_type = RELAX_AMG;
   JX_Int AEE_relax_type = RELAX_AMG;
   JX_Int AII_relax_type = RELAX_AMG;      
   
   JX_Int *row_starts = NULL;
   JX_Int *col_starts = NULL;   

   JX_Int    maxit_default = 200;
   JX_Real tol_default   = 1.0e-6;
 
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Real temp_adrress = 0.0;

   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int np_E = jx_3tAPCTLDataNpE(pre_3tapctl_data); 
   
   JX_Int rootid_R = 0; 
   JX_Int rootid_E = np_R;
   JX_Int rootid_I = np_R + np_E; 
   
   MPI_Comm comm   = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data); 

   char MatFile[255];
   JX_Int myid, nprocs;
   JX_Int groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (!use_ppctl)
   {
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
     

      if (groupid_x == 1)
      {  
         Need_CC = jx_3tAPCTLWeakCouplingE(theta_wc_E, threshold_wc_E, AEE, VER, VEI); 
      }
      jx_MPI_Bcast(&Need_CC, 1, JX_MPI_INT, rootid_E, comm);
      jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      if (print_level && myid == 0) jx_printf(" >> Need_CC = %d\n", Need_CC);
  
      if (Need_CC)
      {
         ARR_relax_maxit = fixit_pctl_R;
         AEE_relax_maxit = fixit_pctl_E;
         AII_relax_maxit = fixit_pctl_I;
      }
      else
      {
         ARR_relax_maxit = fixit_brlx_R;
         AEE_relax_maxit = fixit_brlx_E;
         AII_relax_maxit = fixit_brlx_I;
      }
      jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

      if (groupid_x == 0)
      {  
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jx_ParVectorDestroy(VRE);
         }
         IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R)
         {      
            ARR_relax_type = RELAX_WJACOBI;        
         }
         jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;     
      }
      else if (groupid_x == 1)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jx_ParVectorDestroy(VER);
            jx_ParVectorDestroy(VEI);
         }
         IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E)
         {      
            AEE_relax_type = RELAX_WJACOBI;        
         }
         jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;        
      }
      else if (groupid_x == 2)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jx_ParVectorDestroy(VIE);
         }
         IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I)
         {      
            AII_relax_type = RELAX_WJACOBI;        
         }
         jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;  
      }
   
      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
      } 
   }
   else
   {
      Need_CC = 1;
      ARR_solver_id = SOLVER_AMG;
      ACC_solver_id = SOLVER_AMG;

      ARR_relax_type = RELAX_AMG;
      AEE_relax_type = RELAX_AMG;
      AII_relax_type = RELAX_AMG;

      ARR_relax_maxit = maxit_default;
      AEE_relax_maxit = maxit_default;
      AII_relax_maxit = maxit_default;
      ACC_relax_maxit = maxit_default;

      ARR_relax_tol = tol_default;
      AEE_relax_tol = tol_default;
      AII_relax_tol = tol_default;
      ACC_relax_tol = tol_default; 
    
      jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      jx_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id); 
      jx_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);     
      jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
      jx_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, ACC_relax_maxit);
      jx_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);
      
      jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = 0;
      jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = 0;
      jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = 0;            
   }    


   if (Need_CC == 1)
   {

      if (groupid_x == 0)
      { 
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         WRR->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetDataOwner(WRR, 0);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);
    
         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0); 
         jx_ParVectorSetConstantValues(WII, -1.0);
  
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (ARR_solver_id == SOLVER_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
            jx_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
  
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, 1); 
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
         
            ARR_gmres_solver = jx_ParCSRGMRESCreate(comm_x);
            jx_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
            jx_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
            jx_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
            jx_GMRESSetPrintLevel(ARR_gmres_solver, 0);
            jx_GMRESSetIsCheckRestarted(ARR_gmres_solver, 0); 
         
            jx_GMRESSetPrecond( ARR_gmres_solver,
                                jx_PAMGPrecond,
                                jx_PAMGSetup,
                                ARR_amg_solver );
                             
            jx_PAMGSetup(ARR_amg_solver, ARR);
         
            jx_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
         
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
            } 
         } 
 
         jx_ParVectorSetConstantValues(WEE, 0.0); 
         jx_ParVecMul(VRE, WII, RHS);
         
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WEE);  
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            { 
               jx_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, WEE);
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            {
               jx_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
            }  
         }
         
         if (test_subls_iter)
         {
            if (ARR_solver_id == SOLVER_AMG)
            {
               jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
            }
            else if (ARR_solver_id == SOLVER_AMGGMRES)
            {
               jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
            }
         }         

         if (ARR_relax_type == RELAX_AMG)
         {
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jx_PAMGDestroy(ARR_amg_solver);
            ARR_amg_solver = NULL;
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);            
         }      
      } 
      else if (groupid_x == 1)
      {       
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);
       
         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);  
         
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE); 
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }          
      }
      else if (groupid_x == 2)
      {       
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetConstantValues(WRR, -1.0);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);
     
         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         WII->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetDataOwner(WII, 0);
         jx_ParVectorSetPartitioningOwner(WII, 0); 
               
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }   

         AII_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit); 
         jx_PAMGSetTol(AII_amg_solver, AII_interp_tol); 
         jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
         jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
         jx_PAMGSetInterpType(AII_amg_solver, interp_type);
         jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
         jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
         jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
         jx_PAMGSetup(AII_amg_solver, AII);
  
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }

         jx_ParVectorSetConstantValues(WEE, 0.0);
        
         jx_ParVecMul(VIE, WRR, RHS);
         
         jx_PAMGSolve(AII_amg_solver, AII, RHS, WEE);  
           
         if ((print_level == 2 || print_level == 3) && myid == rootid_I) 
         {
            jx_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);   
         }
 
         if (test_subls_iter)
         {
            jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
         }

         if (AII_relax_type == RELAX_AMG)
         {
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jx_PAMGDestroy(AII_amg_solver);
            AII_amg_solver = NULL;  
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC); 
            jx_ParVectorSetPartitioningOwner(JAC, 0);  
         }
      }           
   }
   else
   {

      if (groupid_x == 0)
      {         
            RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(RHS);
            jx_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            WRR->local_vector->data = &temp_adrress;
            jx_ParVectorInitialize(WRR);
            jx_ParVectorSetDataOwner(WRR, 0); 
            jx_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(WEE);
            jx_ParVectorSetPartitioningOwner(WEE, 0);
      
            WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(WII); 
            jx_ParVectorSetPartitioningOwner(WII, 0);
            jx_ParVectorSetConstantValues(WII, -1.0); 
              
            if (print_level == 1 || print_level == 3)
            {
               starttime = jx_MPI_Wtime();
            }

            if (ARR_relax_type == RELAX_AMG)
            {
               ARR_amg_solver = jx_PAMGCreate();
               jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
               jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
               jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
               jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
               jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
               jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
               jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
               jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
               jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
               jx_PAMGSetup(ARR_amg_solver, ARR);
            } 
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
               jx_ParVectorInitialize(JAC);
               jx_ParVectorSetPartitioningOwner(JAC, 0);     
            }
  
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            } 
      }
      else if (groupid_x == 1)
      {
            RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(RHS);
            jx_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(WRR);
            jx_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            WEE->local_vector->data = &temp_adrress;
            jx_ParVectorInitialize(WEE);
            jx_ParVectorSetDataOwner(WEE, 0);
            jx_ParVectorSetPartitioningOwner(WEE, 0);
     
            WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(WII);
            jx_ParVectorSetPartitioningOwner(WII, 0); 
            
            if (print_level == 1 || print_level == 3)
            {      
               starttime = jx_MPI_Wtime();
            }

            if (AEE_relax_type == RELAX_AMG)
            {
               AEE_amg_solver = jx_PAMGCreate();
               jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
               jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
               jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
               jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
               jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
               jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
               jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
               jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
               jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
               jx_PAMGSetup(AEE_amg_solver, AEE); 
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
               jx_ParVectorInitialize(JAC);
               jx_ParVectorSetPartitioningOwner(JAC, 0);         
            }
         
            if (print_level == 1 || print_level == 3)
            {                      
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
            }
      }
      else if (groupid_x == 2)
      {
            RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(RHS);
            jx_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(WRR);
            jx_ParVectorSetPartitioningOwner(WRR, 0);
            jx_ParVectorSetConstantValues(WRR, -1.0);

            WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(WEE);
            jx_ParVectorSetPartitioningOwner(WEE, 0);
   
            WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            WII->local_vector->data = &temp_adrress;
            jx_ParVectorInitialize(WII);
            jx_ParVectorSetDataOwner(WII, 0);
            jx_ParVectorSetPartitioningOwner(WII, 0);
             
            if (print_level == 1 || print_level == 3)
            {
               starttime = jx_MPI_Wtime();
            }   

            if (AII_relax_type == RELAX_AMG)
            {         
               AII_amg_solver = jx_PAMGCreate();
               jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
               jx_PAMGSetTol(AII_amg_solver, AII_relax_tol); 
               jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
               jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
               jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
               jx_PAMGSetInterpType(AII_amg_solver, interp_type);
               jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
               jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
               jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
               jx_PAMGSetup(AII_amg_solver, AII); 
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
               jx_ParVectorInitialize(JAC);
               jx_ParVectorSetPartitioningOwner(JAC, 0);           
            }         
         
            if (print_level == 1 || print_level == 3)
            {                       
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
            }
      }

   } 
    
   jx_MPI_Barrier(comm);
    
  
   if (Need_CC == 1)
   {
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
  
      row_starts = jx_ParCSRMatrixRowStarts(ARR_all);
      jx_3tAPCTLParaInterpVec(comm, comm_x, groupid_x, row_starts, WEE, &PRR, &PII);

      ACC = jx_3tApctlCoarseOperator_mp(comm, ARR_all, AEE_all, AII_all, 
                                        VRE_all, VER_all, VEI_all, VIE_all, PRR, PII);
                                 
      jx_ParVectorDestroy(PRR);
      jx_ParVectorDestroy(PII);                                  

      if ((debug_flag == 1) || (debug_flag == 3))
      {
          jx_sprintf(MatFile,"%s.%02d", "./3t.acc", jx_total_iter_index);
          jx_ParCSRMatrixPrint(ACC, MatFile);
          jx_total_iter_index ++;
      }

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      } 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
  
      row_starts = jx_ParCSRMatrixRowStarts(A);
      col_starts = jx_ParCSRMatrixRowStarts(ARR_all); 
      P = jx_3tApctlInterpolation(comm, comm_x, N, groupid_x, row_starts, col_starts, WEE);

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Interp Operator", starttime, endtime, 0, 3);
      }

      GCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(GCC);
      jx_ParVectorSetPartitioningOwner(GCC, 0);  

      WCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(WCC);
      jx_ParVectorSetPartitioningOwner(WCC, 0);        

      RES = jx_ParVectorCreate(comm, N, jx_ParCSRMatrixRowStarts(A));
      jx_ParVectorInitialize(RES);
      jx_ParVectorSetPartitioningOwner(RES, 0); 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }

      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jx_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jx_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jx_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jx_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jx_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jx_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jx_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         jx_PAMGSetup(ACC_amg_solver, ACC); 
         
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }   
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, 1); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jx_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jx_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jx_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jx_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jx_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         
         ACC_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jx_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jx_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jx_GMRESSetPrintLevel(ACC_gmres_solver, 0); 
         
         jx_GMRESSetPrecond( ACC_gmres_solver,
                             jx_PAMGPrecond,
                             jx_PAMGSetup,
                             ACC_amg_solver );
                             
         jx_PAMGSetup(ACC_amg_solver, ACC);
         
         jx_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);
      
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }   
      }      
   
   }

   jx_3tAPCTLDataP(pre_3tapctl_data)              = P;
   jx_3tAPCTLDataACC(pre_3tapctl_data)            = ACC;

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = ARR_amg_solver;  
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = AII_amg_solver;
   jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = ACC_amg_solver;
   
   jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;
   
   jx_3tAPCTLDataWRR(pre_3tapctl_data)            = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data)            = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data)            = WII;
   jx_3tAPCTLDataWCC(pre_3tapctl_data)            = WCC;
   jx_3tAPCTLDataGCC(pre_3tapctl_data)            = GCC;

   jx_3tAPCTLDataRES(pre_3tapctl_data)            = RES;
   jx_3tAPCTLDataRHS(pre_3tapctl_data)            = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data)            = JAC;
                  
   return (0); 
}

JX_Int
jx_3tAPCTLSetup4mgJasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   JX_Real starttime = 0.0, endtime = 0.0;

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JX_Real theta_wc_E = jx_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JX_Real threshold_wc_E = jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JX_Real theta_dd_R = jx_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JX_Real theta_dd_E = jx_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JX_Real theta_dd_I = jx_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JX_Real threshold_dd_R = jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JX_Real threshold_dd_E = jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JX_Real threshold_dd_I = jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data);

   JX_Int ARR_interp_maxit = jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);
   JX_Int AII_interp_maxit = jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);
   JX_Int ACC_relax_maxit  = jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data);

   JX_Real ARR_interp_tol = jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data);
   JX_Real AII_interp_tol = jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JX_Real ARR_relax_tol = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real AEE_relax_tol = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real AII_relax_tol = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JX_Real ACC_relax_tol = jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data);

   JX_Int ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   JX_Int ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   JX_Int ARR_kdim = jx_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JX_Int ACC_kdim = jx_3tAPCTLDataACCKDim(pre_3tapctl_data);

   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);
   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
   JX_Int use_ppctl = jx_3tAPCTLDataUsePPCTL(pre_3tapctl_data);
   JX_Int debug_flag = jx_3tAPCTLDataDebugFlag(pre_3tapctl_data);

   JX_Real strong_threshold = jx_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JX_Int interp_type = jx_3tAPCTLDataInterpType(pre_3tapctl_data);
   JX_Int coarsen_type = jx_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JX_Int agg_num_levels = jx_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JX_Int coarse_threshold = jx_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JX_Int print_level_amg = jx_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JX_Int maxit_default = jx_3tAPCTLDataMaxItDefault(pre_3tapctl_data);
   JX_Real tol_default = jx_3tAPCTLDataTolDefault(pre_3tapctl_data);

   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   jx_ParAMGData *ARR_amg_solver = NULL;
   jx_ParAMGData *AEE_amg_solver = NULL;
   jx_ParAMGData *AII_amg_solver = NULL;
   jx_ParAMGData *ACC_amg_solver = NULL;

   jx_GMRESData *ARR_gmres_solver = NULL;
   jx_GMRESData *ACC_gmres_solver = NULL;

   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector  *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector **VER = jx_3tAPCTLDataVER2(pre_3tapctl_data);
   jx_ParVector  *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector  *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParCSRMatrix **ARR_all = jx_3tAPCTLDataARRAll2(pre_3tapctl_data);
   jx_ParCSRMatrix  *AEE_all = jx_3tAPCTLDataAEEAll(pre_3tapctl_data);
   jx_ParCSRMatrix  *AII_all = jx_3tAPCTLDataAIIAll(pre_3tapctl_data);

   jx_ParVector **VRE_all = jx_3tAPCTLDataVREAll2(pre_3tapctl_data);
   jx_ParVector **VER_all = jx_3tAPCTLDataVERAll2(pre_3tapctl_data);
   jx_ParVector  *VEI_all = jx_3tAPCTLDataVEIAll(pre_3tapctl_data);
   jx_ParVector  *VIE_all = jx_3tAPCTLDataVIEAll(pre_3tapctl_data);

   jx_ParCSRMatrix *P   = NULL;
   jx_ParCSRMatrix *ACC = NULL;

   jx_ParVector *WRR = NULL;
   jx_ParVector *WEE = NULL;
   jx_ParVector *WII = NULL;
   jx_ParVector *WCC = NULL;

   jx_ParVector *GCC = NULL;

   jx_ParVector *RES = NULL;
   jx_ParVector *RHS = NULL;
   jx_ParVector *JAC = NULL;

   jx_ParVector **PRR = NULL;
   jx_ParVector  *PII = NULL;

   JX_Int IS_DD_R;
   JX_Int IS_DD_E;
   JX_Int IS_DD_I;

   JX_Int ARR_relax_maxit;
   JX_Int AEE_relax_maxit;
   JX_Int AII_relax_maxit;

   JX_Int Need_CC;
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   JX_Int *row_starts = NULL;
   JX_Int *col_starts = NULL;

   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / (ng + 2);
   JX_Real temp_adrress = 0.0;

   MPI_Comm comm   = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);

   char MatFile[255];
   JX_Int myid, nprocs, gidx;
   JX_Int groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int rootid_R = groupid_x * np_R;
   JX_Int rootid_E = np_R * ng;
   JX_Int rootid_I = np_R + rootid_E;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (!use_ppctl)
   {
      if (print_level == 1 || print_level == 3)
      {
         starttime = jx_MPI_Wtime();
      }

      if (groupid_x == ng)
      {
         Need_CC = jx_3tAPCTLmgWeakCouplingE(theta_wc_E, threshold_wc_E, ng, AEE, VER, VEI);
      }
      jx_MPI_Bcast(&Need_CC, 1, JX_MPI_INT, rootid_E, comm);
      jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      if (print_level && myid == 0) jx_printf(" >> Need_CC = %d\n", Need_CC);

      if (Need_CC)
      {
         ARR_relax_maxit = fixit_pctl_R;
         AEE_relax_maxit = fixit_pctl_E;
         AII_relax_maxit = fixit_pctl_I;
      }
      else
      {
         ARR_relax_maxit = fixit_brlx_R;
         AEE_relax_maxit = fixit_brlx_E;
         AII_relax_maxit = fixit_brlx_I;
      }
      jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

      if (groupid_x < ng)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jx_ParVectorDestroy(VRE);
         }
         IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      }
      else if (groupid_x == ng)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            for (gidx = 0; gidx < ng; gidx ++) jx_ParVectorDestroy(VER[gidx]);
            jx_TFree(VER);
            jx_ParVectorDestroy(VEI);
         }
         IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      }
      else if (groupid_x == ng+1)
      {
         if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
         {
            jx_ParVectorDestroy(VIE);
         }
         IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      }

      if (print_level == 1 || print_level == 3)
      {
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
      }
   }
   else
   {
      Need_CC = 1;
      ARR_solver_id = SOLVER_AMG;
      ACC_solver_id = SOLVER_AMG;

      ARR_relax_type = RELAX_AMG;
      AEE_relax_type = RELAX_AMG;
      AII_relax_type = RELAX_AMG;

      ARR_relax_maxit = maxit_default;
      AEE_relax_maxit = maxit_default;
      AII_relax_maxit = maxit_default;
      ACC_relax_maxit = 1; //maxit_default;

      ARR_relax_tol = tol_default;
      AEE_relax_tol = tol_default;
      AII_relax_tol = tol_default;
      ACC_relax_tol = tol_default;

      jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      jx_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id);
      jx_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);
      jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
      jx_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, ACC_relax_maxit);
      jx_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);

      jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = 0;
      jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = 0;
      jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = 0;
   }

   if (Need_CC == 1)
   {
      if (groupid_x < ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);
         jx_ParVectorSetConstantValues(WII, -1.0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (ARR_solver_id == SOLVER_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_interp_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
            if (print_level == 1 || print_level == 3)
            {
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, 1);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            ARR_gmres_solver = jx_ParCSRGMRESCreate(comm_x);
            jx_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
            jx_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
            jx_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
            jx_GMRESSetPrintLevel(ARR_gmres_solver, 0);
            jx_GMRESSetIsCheckRestarted(ARR_gmres_solver, 0);
            jx_GMRESSetPrecond( ARR_gmres_solver, jx_PAMGPrecond, jx_PAMGSetup, ARR_amg_solver );
            jx_PAMGSetup(ARR_amg_solver, ARR);
            jx_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
            if (print_level == 1 || print_level == 3)
            {
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
            }
         }

         jx_ParVectorSetConstantValues(WEE, 0.0);
         jx_ParVecMul(VRE, WII, RHS);
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WEE);
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            {
               jx_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, WEE);
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            {
               jx_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
            }
         }

         jx_ParVectorDestroy(WII);

         if (test_subls_iter)
         {
            if (ARR_solver_id == SOLVER_AMG)
            {
               jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
            }
            else if (ARR_solver_id == SOLVER_AMGGMRES)
            {
               jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
            }
         }

         if (ARR_relax_type == RELAX_AMG)
         {
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jx_PAMGDestroy(ARR_amg_solver);
            ARR_amg_solver = NULL;
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }
      }
      else if (groupid_x == ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetConstantValues(WRR, -1.0);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         AII_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit);
         jx_PAMGSetTol(AII_amg_solver, AII_interp_tol);
         jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
         jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
         jx_PAMGSetInterpType(AII_amg_solver, interp_type);
         jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
         jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
         jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
         jx_PAMGSetup(AII_amg_solver, AII);

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }

         jx_ParVectorSetConstantValues(WEE, 0.0);
         jx_ParVecMul(VIE, WRR, RHS);
         jx_PAMGSolve(AII_amg_solver, AII, RHS, WEE);

         if ((print_level == 2 || print_level == 3) && myid == rootid_I)
         {
            jx_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
         }

         jx_ParVectorDestroy(WRR);

         if (test_subls_iter)
         {
            jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
         }

         if (AII_relax_type == RELAX_AMG)
         {
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jx_PAMGDestroy(AII_amg_solver);
            AII_amg_solver = NULL;
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }
      }
   }
   else
   {
      if (groupid_x < ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AII_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
            jx_PAMGSetup(AII_amg_solver, AII);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
   }

   jx_MPI_Barrier(comm); //  各进程(组)同步

   if (Need_CC == 1)
   {
      if (print_level == 1 || print_level == 3)
      {
         starttime = jx_MPI_Wtime();
      }

      row_starts = jx_ParCSRMatrixRowStarts(ARR_all[0]);
      jx_3tAPCTLmgParaInterpVec(comm, comm_x, groupid_x, row_starts, ng, WEE, &PRR, &PII);
      ACC = jx_3tApctlmgCoarseOperator_mp(comm, ng, ARR_all, AEE_all, AII_all, VRE_all, VER_all, VEI_all, VIE_all, PRR, PII);
      for (gidx = 0; gidx < ng; gidx ++) jx_ParVectorDestroy(PRR[gidx]);
      jx_TFree(PRR);
      jx_ParVectorDestroy(PII);

      if ((debug_flag == 1) || (debug_flag == 3))
      {
          jx_sprintf(MatFile,"%s.%02d", "./mg.acc", jx_total_iter_index);
          jx_ParCSRMatrixPrint(ACC, MatFile);
          jx_total_iter_index ++;
      }

      if (print_level == 1 || print_level == 3)
      {
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      }

      if (print_level == 1 || print_level == 3)
      {
         starttime = jx_MPI_Wtime();
      }

      row_starts = jx_ParCSRMatrixRowStarts(A);
      col_starts = jx_ParCSRMatrixRowStarts(ARR_all[0]);
      P = jx_3tApctlmgInterpolation(comm, comm_x, N, groupid_x, ng, row_starts, col_starts, WEE);

      if (print_level == 1 || print_level == 3)
      {
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Interp Operator", starttime, endtime, 0, 3);
      }

      GCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(GCC);
      jx_ParVectorSetPartitioningOwner(GCC, 0);

      WCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(WCC);
      jx_ParVectorSetPartitioningOwner(WCC, 0);

      RES = jx_ParVectorCreate(comm, N, jx_ParCSRMatrixRowStarts(A));
      jx_ParVectorInitialize(RES);
      jx_ParVectorSetPartitioningOwner(RES, 0);

      if (print_level == 1 || print_level == 3)
      {
         starttime = jx_MPI_Wtime();
      }

      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit);
         jx_PAMGSetTol(ACC_amg_solver, ACC_relax_tol);
         jx_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         jx_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jx_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jx_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jx_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jx_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jx_PAMGSetup(ACC_amg_solver, ACC);

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, 1);
         jx_PAMGSetStrongThreshold(ACC_amg_solver, strong_threshold);
         jx_PAMGSetCoarsenType(ACC_amg_solver, coarsen_type);
         jx_PAMGSetInterpType(ACC_amg_solver, interp_type);
         jx_PAMGSetAggNumLevels(ACC_amg_solver, agg_num_levels);
         jx_PAMGSetCoarseThreshold(ACC_amg_solver, coarse_threshold);
         jx_PAMGSetPrintLevel(ACC_amg_solver, print_level_amg);
         ACC_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jx_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jx_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jx_GMRESSetPrintLevel(ACC_gmres_solver, 0);
         jx_GMRESSetPrecond( ACC_gmres_solver, jx_PAMGPrecond, jx_PAMGSetup, ACC_amg_solver );
         jx_PAMGSetup(ACC_amg_solver, ACC);
         jx_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }
      }
   }

   jx_3tAPCTLDataP(pre_3tapctl_data)              = P;
   jx_3tAPCTLDataACC(pre_3tapctl_data)            = ACC;

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = ARR_amg_solver;
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = AII_amg_solver;
   jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = ACC_amg_solver;

   jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;

   jx_3tAPCTLDataWRR(pre_3tapctl_data)            = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data)            = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data)            = WII;
   jx_3tAPCTLDataWCC(pre_3tapctl_data)            = WCC;
   jx_3tAPCTLDataGCC(pre_3tapctl_data)            = GCC;

   jx_3tAPCTLDataRES(pre_3tapctl_data)            = RES;
   jx_3tAPCTLDataRHS(pre_3tapctl_data)            = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data)            = JAC;

   return (0);
}

JX_Int
jx_3tABSC1Setup4mgJasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   JX_Real starttime = 0.0, endtime = 0.0;

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JX_Real theta_wc_E = jx_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JX_Real threshold_wc_E = jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JX_Real theta_dd_R = jx_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JX_Real theta_dd_E = jx_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JX_Real theta_dd_I = jx_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JX_Real threshold_dd_R = jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JX_Real threshold_dd_E = jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JX_Real threshold_dd_I = jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data);

   JX_Real ARR_relax_tol = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real AEE_relax_tol = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real AII_relax_tol = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);

   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JX_Real strong_threshold = jx_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JX_Int interp_type = jx_3tAPCTLDataInterpType(pre_3tapctl_data);
   JX_Int coarsen_type = jx_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JX_Int agg_num_levels = jx_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JX_Int coarse_threshold = jx_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JX_Int print_level_amg = jx_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   jx_ParAMGData *ARR_amg_solver = NULL;
   jx_ParAMGData *AEE_amg_solver = NULL;
   jx_ParAMGData *AII_amg_solver = NULL;

   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector  *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector **VER = jx_3tAPCTLDataVER2(pre_3tapctl_data);
   jx_ParVector  *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector  *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParVector *WRR = NULL;
   jx_ParVector *WEE = NULL;
   jx_ParVector *WII = NULL;
   jx_ParVector *RHS = NULL;
   jx_ParVector *JAC = NULL;

   JX_Int Need_CC, sr_hsize = 0;
   JX_Int IS_DD_R, IS_DD_E, IS_DD_I;
   JX_Int ARR_relax_maxit, AEE_relax_maxit, AII_relax_maxit;

   MPI_Status status;

   JX_Real *rTEMP = NULL;
   JX_Real *sTEMP = NULL;
   JX_Real *vTEMP = NULL;
   JX_Real *tTEMP = NULL;

   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / (ng + 2);
   JX_Real temp_adrress = 0.0;

   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jx_3tAPCTLDataCommY(pre_3tapctl_data);

   JX_Int myid, nprocs, gidx, send_cnt, j;
   JX_Int groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int rootid_R = groupid_x * np_R;
   JX_Int rootid_E = np_R * ng;
   JX_Int rootid_I = np_R + rootid_E;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (groupid_x < ng)
   {
      sr_hsize = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(ARR));
   }
   else if (groupid_x == ng)
   {
      sr_hsize = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(AEE));
   }
   else if (groupid_x == ng+1)
   {
      sr_hsize = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(AII));
   }

   rTEMP = jx_CTAlloc(JX_Real, 2*sr_hsize);
   sTEMP = jx_CTAlloc(JX_Real, sr_hsize);
   vTEMP = jx_CTAlloc(JX_Real, sr_hsize);

   if (print_level == 1 || print_level == 3)
   {
      starttime = jx_MPI_Wtime();
   }

   if (groupid_x == ng)
   {
      Need_CC = jx_3tAPCTLmgWeakCouplingE(theta_wc_E, threshold_wc_E, ng, AEE, VER, VEI);
   }
   jx_MPI_Bcast(&Need_CC, 1, JX_MPI_INT, rootid_E, comm);
   jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
   if (print_level && myid == 0) jx_printf(" >> Need_CC = %d\n", Need_CC);

   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      if (groupid_x < ng)
      {
         jx_ParVectorDestroy(VRE);
      }
      else if (groupid_x == ng)
      {
         for (gidx = 0; gidx < ng; gidx ++) jx_ParVectorDestroy(VER[gidx]);
         jx_TFree(VER);
         jx_ParVectorDestroy(VEI);
      }
      else if (groupid_x == ng+1)
      {
         jx_ParVectorDestroy(VIE);
      }

      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

   if (print_level == 1 || print_level == 3)
   {
      endtime = jx_MPI_Wtime();
      jx_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
   }

   if (Need_CC)
   {
      if (groupid_x == ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         jx_MPI_Recv(rTEMP, 2*sr_hsize, JX_MPI_REAL, myid+np_R, 123, comm, &status);

         jx_BSCModifySubMat(AEE, VEI, rTEMP, sr_hsize); // update AEE

         IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

#if 0
         dest = myid - rootid_E;
         for (j = 0; j < ng; j ++)
         {
            dTEMP[j] = jx_CTAlloc(JX_Real, 2*sr_hsize);
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               dTEMP[j][gidx] = jx_CSRMatrixData(jx_ParCSRMatrixDiag(AEE))[jx_CSRMatrixI(jx_ParCSRMatrixDiag(AEE))[gidx]];
            }
            send_cnt = sr_hsize;
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               dTEMP[j][send_cnt++] = jx_VectorData(jx_ParVectorLocalVector(VER[j]))[gidx];
            }
            jx_MPI_Send(dTEMP[j], 2*sr_hsize, JX_MPI_REAL, dest, dest*321, comm);
            dest += np_R;
         }
#else
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            sTEMP[gidx] = jx_CSRMatrixData(jx_ParCSRMatrixDiag(AEE))[jx_CSRMatrixI(jx_ParCSRMatrixDiag(AEE))[gidx]];
         }
         tTEMP = jx_CTAlloc(JX_Real, (ng+2)*sr_hsize);
         send_cnt = 0;
         for (j = 0; j < ng; j ++)
         {
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               tTEMP[send_cnt++] = jx_VectorData(jx_ParVectorLocalVector(VER[j]))[gidx];
            }
         }
#endif

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         send_cnt = 0;
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            rTEMP[send_cnt++] = jx_CSRMatrixData(jx_ParCSRMatrixDiag(AII))[jx_CSRMatrixI(jx_ParCSRMatrixDiag(AII))[gidx]];
         }
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            rTEMP[send_cnt++] = jx_VectorData(jx_ParVectorLocalVector(VIE))[gidx];
         }
         jx_MPI_Send(rTEMP, 2*sr_hsize, JX_MPI_REAL, myid-np_R, 123, comm);

         IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AII_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
            jx_PAMGSetup(AII_amg_solver, AII);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }

   jx_MPI_Barrier(comm); //  各进程(组)同步

      jx_MPI_Bcast(sTEMP, sr_hsize, JX_MPI_REAL, ng, comm_y); // yue: invalid for comm_I
      jx_MPI_Scatter(tTEMP, sr_hsize, JX_MPI_REAL, vTEMP, sr_hsize, JX_MPI_REAL, ng, comm_y); // yue: invalid for comm_E and comm_I

      if (groupid_x < ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         //jx_MPI_Recv(rTEMP, 2*sr_hsize, JX_MPI_REAL, myid, myid*321, comm, &status);

         jx_BSCModify2SubMat(ARR, VRE, sTEMP, vTEMP, sr_hsize); // update ARR

         IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      }
   }
   else // BD
   {
      if (groupid_x < ng)
      {
         IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "BD == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng)
      {
         IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "BD == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AII_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
            jx_PAMGSetup(AII_amg_solver, AII);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "BD == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
   }

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) = ARR_amg_solver;
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) = AII_amg_solver;

   jx_3tAPCTLDataWRR(pre_3tapctl_data) = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data) = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data) = WII;

   jx_3tAPCTLDataRHS(pre_3tapctl_data) = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data) = JAC;

   jx_3tAPCTLDatatTEMP(pre_3tapctl_data) = tTEMP;

   jx_TFree(rTEMP);
   jx_TFree(sTEMP);
   jx_TFree(vTEMP);
#if 0
   for (j = 0; j < ng; j ++)
   {
      if (dTEMP[j]) jx_TFree(dTEMP[j]);
   }
   jx_TFree(dTEMP);
#endif

   return (0);
}

JX_Int
jx_3tABSC2Setup4mgJasmin_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   JX_Real starttime = 0.0, endtime = 0.0;

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JX_Real theta_wc_E = jx_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JX_Real threshold_wc_E = jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JX_Real theta_dd_R = jx_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JX_Real theta_dd_E = jx_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JX_Real theta_dd_I = jx_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JX_Real threshold_dd_R = jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JX_Real threshold_dd_E = jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JX_Real threshold_dd_I = jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data);

   JX_Real ARR_relax_tol = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real AEE_relax_tol = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real AII_relax_tol = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);

   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JX_Real strong_threshold = jx_3tAPCTLDataStrongThreshold(pre_3tapctl_data);
   JX_Int interp_type = jx_3tAPCTLDataInterpType(pre_3tapctl_data);
   JX_Int coarsen_type = jx_3tAPCTLDataCoarsenType(pre_3tapctl_data);
   JX_Int agg_num_levels = jx_3tAPCTLDataAggNumLevels(pre_3tapctl_data);
   JX_Int coarse_threshold = jx_3tAPCTLDataCoarseThreshold(pre_3tapctl_data);
   JX_Int print_level_amg = jx_3tAPCTLDataPrintLevelAMG(pre_3tapctl_data);

   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   jx_ParAMGData *ARR_amg_solver = NULL;
   jx_ParAMGData *AEE_amg_solver = NULL;
   jx_ParAMGData *AII_amg_solver = NULL;

   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector  *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector **VER = jx_3tAPCTLDataVER2(pre_3tapctl_data);
   jx_ParVector  *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector  *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParVector *WRR = NULL;
   jx_ParVector *WEE = NULL;
   jx_ParVector *WII = NULL;
   jx_ParVector *RHS = NULL;
   jx_ParVector *JAC = NULL;

   JX_Int Need_CC, sr_hsize = 0;
   JX_Int IS_DD_R, IS_DD_E, IS_DD_I;
   JX_Int ARR_relax_maxit, AEE_relax_maxit, AII_relax_maxit;

   JX_Real *sTEMP = NULL;
   JX_Real *vTEMP = NULL;
   JX_Real *tTEMP = NULL;

   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / (ng + 2);
   JX_Real temp_adrress = 0.0;

   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jx_3tAPCTLDataCommY(pre_3tapctl_data);

   JX_Int myid, nprocs, gidx, send_cnt, j;
   JX_Int groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int rootid_R = groupid_x * np_R;
   JX_Int rootid_E = np_R * ng;
   JX_Int rootid_I = np_R + rootid_E;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (groupid_x < ng)
   {
      sr_hsize = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(ARR));
   }
   else if (groupid_x == ng)
   {
      sr_hsize = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(AEE));
   }
   else if (groupid_x == ng+1)
   {
      sr_hsize = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(AII));
   }

   sTEMP = jx_CTAlloc(JX_Real, sr_hsize);
   vTEMP = jx_CTAlloc(JX_Real, sr_hsize);

   if (print_level == 1 || print_level == 3)
   {
      starttime = jx_MPI_Wtime();
   }

   if (groupid_x == ng)
   {
      Need_CC = jx_3tAPCTLmgWeakCouplingE(theta_wc_E, threshold_wc_E, ng, AEE, VER, VEI);
   }
   jx_MPI_Bcast(&Need_CC, 1, JX_MPI_INT, rootid_E, comm);
   jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
   if (print_level && myid == 0) jx_printf(" >> Need_CC = %d\n", Need_CC);

   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      if (groupid_x < ng)
      {
         jx_ParVectorDestroy(VRE);
      }
      else if (groupid_x == ng)
      {
         for (gidx = 0; gidx < ng; gidx ++) jx_ParVectorDestroy(VER[gidx]);
         jx_TFree(VER);
         jx_ParVectorDestroy(VEI);
      }
      else if (groupid_x == ng+1)
      {
         jx_ParVectorDestroy(VIE);
      }

      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

   if (print_level == 1 || print_level == 3)
   {
      endtime = jx_MPI_Wtime();
      jx_GetWallTime(comm, "APCTL == Check WC-DD cond", starttime, endtime, 0, 3);
   }

   if (Need_CC)
   {
      if (groupid_x == ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            sTEMP[gidx] = jx_CSRMatrixData(jx_ParCSRMatrixDiag(AEE))[jx_CSRMatrixI(jx_ParCSRMatrixDiag(AEE))[gidx]];
         }
         tTEMP = jx_CTAlloc(JX_Real, (ng+2)*sr_hsize);
         send_cnt = 0;
         for (j = 0; j < ng; j ++)
         {
            for (gidx = 0; gidx < sr_hsize; gidx ++)
            {
               tTEMP[send_cnt++] = jx_VectorData(jx_ParVectorLocalVector(VER[j]))[gidx];
            }
         }
         send_cnt += sr_hsize;
         for (gidx = 0; gidx < sr_hsize; gidx ++)
         {
            tTEMP[send_cnt++] = jx_VectorData(jx_ParVectorLocalVector(VEI))[gidx];
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }

   jx_MPI_Barrier(comm); //  各进程(组)同步

      jx_MPI_Bcast(sTEMP, sr_hsize, JX_MPI_REAL, ng, comm_y);
      jx_MPI_Scatter(tTEMP, sr_hsize, JX_MPI_REAL, vTEMP, sr_hsize, JX_MPI_REAL, ng, comm_y); // yue: invalid for comm_E

      if (groupid_x < ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         jx_BSCModify2SubMat(ARR, VRE, sTEMP, vTEMP, sr_hsize); // update ARR

         IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         jx_BSCModify2SubMat(AII, VIE, sTEMP, vTEMP, sr_hsize); // update AII

         IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AII_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
            jx_PAMGSetup(AII_amg_solver, AII);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
   }
   else
   {
      if (groupid_x < ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
         jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
         if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
         if (IS_DD_R) ARR_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;

         if (ARR_relax_type == RELAX_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit);
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(ARR_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(ARR_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(ARR_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(ARR_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(ARR_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(ARR_amg_solver, print_level_amg);
            jx_PAMGSetup(ARR_amg_solver, ARR);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
         jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
         if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
         if (IS_DD_E) AEE_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit);
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AEE_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AEE_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AEE_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AEE_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AEE_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AEE_amg_solver, print_level_amg);
            jx_PAMGSetup(AEE_amg_solver, AEE);
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
      else if (groupid_x == ng+1)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);

         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
         jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
         if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
         if (IS_DD_I) AII_relax_type = RELAX_WJACOBI;
         jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;

         if (AII_relax_type == RELAX_AMG)
         {
            AII_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit);
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
            jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
            jx_PAMGSetStrongThreshold(AII_amg_solver, strong_threshold);
            jx_PAMGSetCoarsenType(AII_amg_solver, coarsen_type);
            jx_PAMGSetInterpType(AII_amg_solver, interp_type);
            jx_PAMGSetAggNumLevels(AII_amg_solver, agg_num_levels);
            jx_PAMGSetCoarseThreshold(AII_amg_solver, coarse_threshold);
            jx_PAMGSetPrintLevel(AII_amg_solver, print_level_amg);
            jx_PAMGSetup(AII_amg_solver, AII);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
   }

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) = ARR_amg_solver;
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) = AII_amg_solver;

   jx_3tAPCTLDataWRR(pre_3tapctl_data) = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data) = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data) = WII;

   jx_3tAPCTLDataRHS(pre_3tapctl_data) = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data) = JAC;

   jx_3tAPCTLDatatTEMP(pre_3tapctl_data) = tTEMP;

   jx_TFree(sTEMP);
   jx_TFree(vTEMP);

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLSetup4Jasmin_sp
 * \brief Setup phase of PCTL Iteration or preconditioner(for single-processor case). 
 * \author peghoty 
 * \date 2012/02/25
 */
JX_Int
jx_3tAPCTLSetup4Jasmin_sp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   JX_Real starttime = 0.0, endtime = 0.0;

   MPI_Comm comm = jx_ParCSRMatrixComm(A);

   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JX_Int ARR_interp_maxit = jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JX_Int AII_interp_maxit = jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);     
   JX_Int ACC_relax_maxit  = jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data); 

   JX_Int ARR_relax_maxit;
   JX_Int AEE_relax_maxit;
   JX_Int AII_relax_maxit;

   JX_Real  ARR_interp_tol = jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JX_Real  AII_interp_tol = jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JX_Real  ARR_relax_tol  = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real  AEE_relax_tol  = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real  AII_relax_tol  = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JX_Real  ACC_relax_tol  = jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   JX_Int ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   JX_Int ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   JX_Int ARR_kdim = jx_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JX_Int ACC_kdim = jx_3tAPCTLDataACCKDim(pre_3tapctl_data); 
   
   JX_Int use_ppctl = jx_3tAPCTLDataUsePPCTL(pre_3tapctl_data);
    
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);
  
   jx_ParAMGData   *ARR_amg_solver = NULL;
   jx_ParAMGData   *AEE_amg_solver = NULL;
   jx_ParAMGData   *AII_amg_solver = NULL;
   jx_ParAMGData   *ACC_amg_solver = NULL;

   jx_GMRESData    *ARR_gmres_solver = NULL;
   jx_GMRESData    *ACC_gmres_solver = NULL; 
   
   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);  
   jx_ParVector    *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector    *VER = jx_3tAPCTLDataVER(pre_3tapctl_data);
   jx_ParVector    *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector    *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);    

   jx_ParVector    *PRR = NULL;
   jx_ParVector    *PII = NULL;
   jx_ParCSRMatrix *ACC = NULL; 
     
   jx_ParVector    *WRR = NULL;
   jx_ParVector    *WEE = NULL;
   jx_ParVector    *WII = NULL;
   jx_ParVector    *WCC = NULL;
 
   jx_ParVector    *RES = NULL;
   jx_ParVector    *RHS = NULL;
   jx_ParVector    *JAC = NULL;

   JX_Int Need_CC;
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data); 

   JX_Int ARR_relax_type = RELAX_AMG;
   JX_Int AEE_relax_type = RELAX_AMG;
   JX_Int AII_relax_type = RELAX_AMG;         

   jx_CSRMatrix  *ARR_s = jx_ParCSRMatrixDiag(ARR); 
   jx_CSRMatrix  *AEE_s = jx_ParCSRMatrixDiag(AEE);
   jx_CSRMatrix  *AII_s = jx_ParCSRMatrixDiag(AII);
   jx_Vector     *VRE_s = jx_ParVectorLocalVector(VRE);
   jx_Vector     *VER_s = jx_ParVectorLocalVector(VER);
   jx_Vector     *VEI_s = jx_ParVectorLocalVector(VEI);
   jx_Vector     *VIE_s = jx_ParVectorLocalVector(VIE);
   
   JX_Int    maxit_default = 200;
   JX_Real tol_default   = 1.0e-6;

   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Int i;
   JX_Real temp_adrress = 0.0;

   if (!use_ppctl)
   {
      jx_3tAPCTLWCDD( pre_3tapctl_data, comm, 
                      ARR_s, AEE_s, AII_s, VRE_s, VER_s, VEI_s, VIE_s );

      ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
      AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
      AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
      Need_CC        = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);

      if (Need_CC)
      {
         ARR_relax_maxit = fixit_pctl_R;
         AEE_relax_maxit = fixit_pctl_E;
         AII_relax_maxit = fixit_pctl_I;
      }
      else
      {
         ARR_relax_maxit = fixit_brlx_R;
         AEE_relax_maxit = fixit_brlx_E;
         AII_relax_maxit = fixit_brlx_I;
      }
      jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
   }
   else
   {
      Need_CC = 1;
      ARR_solver_id = SOLVER_AMG;
      ACC_solver_id = SOLVER_AMG;

      ARR_relax_type = RELAX_AMG;
      AEE_relax_type = RELAX_AMG;
      AII_relax_type = RELAX_AMG;

      ARR_relax_maxit = maxit_default;
      AEE_relax_maxit = maxit_default;
      AII_relax_maxit = maxit_default;
      ACC_relax_maxit = maxit_default;

      ARR_relax_tol = tol_default;
      AEE_relax_tol = tol_default;
      AII_relax_tol = tol_default;
      ACC_relax_tol = tol_default; 
    
      jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
      jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;
      jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;
      jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;
      jx_3tAPCTLSetACCSolverID(pre_3tapctl_data, ACC_solver_id); 
      jx_3tAPCTLSetARRSolverID(pre_3tapctl_data, ARR_solver_id);     
      jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
      jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
      jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);
      jx_3tAPCTLSetACCRelaxMaxIt(pre_3tapctl_data, ACC_relax_maxit);
      jx_3tAPCTLSetARRRelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetAEERelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetAIIRelaxTol(pre_3tapctl_data, tol_default);
      jx_3tAPCTLSetACCRelaxTol(pre_3tapctl_data, tol_default);
      
      jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data) = 0;
      jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data) = 0;
      jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data) = 0;            
   }

   RHS = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
   jx_ParVectorInitialize(RHS);
   jx_ParVectorSetPartitioningOwner(RHS, 0);

   WRR = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
   WRR->local_vector->data = &temp_adrress;
   jx_ParVectorInitialize(WRR);
   jx_ParVectorSetDataOwner(WRR, 0);
   jx_ParVectorSetPartitioningOwner(WRR, 0);   

   WEE = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AEE));
   WEE->local_vector->data = &temp_adrress;
   jx_ParVectorInitialize(WEE);
   jx_ParVectorSetDataOwner(WEE, 0);
   jx_ParVectorSetPartitioningOwner(WEE, 0);   

   WII = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
   WII->local_vector->data = &temp_adrress;
   jx_ParVectorInitialize(WII);   
   jx_ParVectorSetDataOwner(WII, 0);
   jx_ParVectorSetPartitioningOwner(WII, 0);  

   if (Need_CC == 1)
   {      
      PRR = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
      jx_ParVectorInitialize(PRR); 
      jx_ParVectorSetPartitioningOwner(PRR, 0);

      PII = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
      jx_ParVectorInitialize(PII);
      jx_ParVectorSetPartitioningOwner(PII, 0); 

      if (ARR_solver_id == SOLVER_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         } 

         ARR_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
         jx_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
         jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout      
         jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         jx_PAMGSetup(ARR_amg_solver, ARR); 

         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         } 
      }
      else if (ARR_solver_id == SOLVER_AMGGMRES)
      {  
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         } 

         ARR_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ARR_amg_solver, 1);
         jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         
         ARR_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
         jx_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
         jx_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
         jx_GMRESSetPrintLevel(ARR_gmres_solver, 0);
         jx_GMRESSetIsCheckRestarted(ARR_gmres_solver, 0); 

         jx_GMRESSetPrecond( ARR_gmres_solver,
                             jx_PAMGPrecond,
                             jx_PAMGSetup,
                             ARR_amg_solver );
                   
         jx_PAMGSetup(ARR_amg_solver, ARR);
         
         jx_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
         
         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
         } 
      } 

      for (i = 0; i < n; i ++)
      {
         RHS->local_vector->data[i] = - VRE->local_vector->data[i];   
      }
      
      jx_ParVectorSetConstantValues(PRR, 0.0);
      
      if (ARR_solver_id == SOLVER_AMG)
      { 
         jx_PAMGSolve(ARR_amg_solver, ARR, RHS, PRR);  
         if (print_level == 2 || print_level == 3)
         { 
            jx_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
         }   
      } 
      else if (ARR_solver_id == SOLVER_AMGGMRES)
      {
         jx_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, PRR);
         if (print_level == 2 || print_level == 3)
         {
            jx_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
         }  
      }
           
      if (test_subls_iter)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
         }
      }         

      if (ARR_relax_type == RELAX_AMG)
      {
         jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
         jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
         jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jx_PAMGDestroy(ARR_amg_solver);
         ARR_amg_solver = NULL;
         if (!JAC)
         {  
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }           
      }

      if (AEE_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         }      
      
         AEE_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
         jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
         jx_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AEE_amg_solver, 0);
         jx_PAMGSetup(AEE_amg_solver, AEE); 

         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }         
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }
      }
            
      if (print_level == 1 || print_level == 3)
      {
         starttime = jx_MPI_Wtime();
      }   

      AII_amg_solver = jx_PAMGCreate();
      jx_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit); 
      jx_PAMGSetTol(AII_amg_solver, AII_interp_tol); 
      jx_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
      jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
      jx_PAMGSetPrintLevel(AII_amg_solver, 0);
      jx_PAMGSetup(AII_amg_solver, AII); 

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
      }

      for (i = 0; i < n; i ++)
      {
         RHS->local_vector->data[i] = - VIE->local_vector->data[i];   
      }
      jx_ParVectorSetConstantValues(PII, 0.0);
      jx_PAMGSolve(AII_amg_solver, AII, RHS, PII);  
      if (print_level == 2 || print_level == 3) 
      {
         jx_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);   
      }

      if (test_subls_iter)
      {
         jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
      }

      if (AII_relax_type == RELAX_AMG)
      {
         jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
         jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jx_PAMGDestroy(AII_amg_solver);
         AII_amg_solver = NULL;  
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC); 
            jx_ParVectorSetPartitioningOwner(JAC, 0); 
         }           
      }  
  
   }  
   else
   {
      if (ARR_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }
         
         ARR_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
         jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
         jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         jx_PAMGSetup(ARR_amg_solver, ARR);

         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      } 
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);           
         }          
      }

      if (AEE_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         }
                  
         AEE_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
         jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
         jx_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AEE_amg_solver, 0);
         jx_PAMGSetup(AEE_amg_solver, AEE); 
         
         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }            
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }          
      }

      if (AII_relax_type == RELAX_AMG)
      {  
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }   
             
         AII_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
         jx_PAMGSetTol(AII_amg_solver, AII_relax_tol); 
         jx_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AII_amg_solver, 0);
         jx_PAMGSetup(AII_amg_solver, AII); 
            
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC); 
            jx_ParVectorSetPartitioningOwner(JAC, 0); 
         }        
      }         
          
   } 
   
   if (Need_CC == 1)
   {
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
         
      ACC = jx_3tApctlCoarseOperator_sp( comm, ARR_s, AEE_s, AII_s, VRE_s, 
                                         VER_s, VEI_s, VIE_s, PRR, PII );

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      } 

      WCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(WCC);
      jx_ParVectorSetPartitioningOwner(WCC, 0);

      RES = jx_ParVectorCreate(comm, N, jx_ParCSRMatrixRowStarts(A));
      jx_ParVectorInitialize(RES);
      jx_ParVectorSetPartitioningOwner(RES, 0);

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
     
      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jx_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0);
         jx_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout      
         jx_PAMGSetPrintLevel(ACC_amg_solver, 0);
         jx_PAMGSetup(ACC_amg_solver, ACC); 
         
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }  
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, 1); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jx_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jx_PAMGSetPrintLevel(ACC_amg_solver, 0);
         
         ACC_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jx_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jx_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jx_GMRESSetPrintLevel(ACC_gmres_solver, 0); 
         
         jx_GMRESSetPrecond( ACC_gmres_solver,
                             jx_PAMGPrecond,
                             jx_PAMGSetup,
                             ACC_amg_solver );
                             
         jx_PAMGSetup(ACC_amg_solver, ACC);
         
         jx_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);
      
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }   
      }      

   }
      
   jx_3tAPCTLDataPRR(pre_3tapctl_data) = PRR;
   jx_3tAPCTLDataPII(pre_3tapctl_data) = PII;
   jx_3tAPCTLDataACC(pre_3tapctl_data) = ACC;

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) = ARR_amg_solver;  
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) = AII_amg_solver;
   jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) = ACC_amg_solver;

   jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;

   jx_3tAPCTLDataWRR(pre_3tapctl_data) = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data) = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data) = WII;
   jx_3tAPCTLDataWCC(pre_3tapctl_data) = WCC;

   jx_3tAPCTLDataRES(pre_3tapctl_data) = RES;
   jx_3tAPCTLDataRHS(pre_3tapctl_data) = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data) = JAC;
             
   return (0); 
}

/*!
 * \fn JX_Int JX_3tAPCTLDestroy4Jasmin
 * \brief Destroy a 3tAPCTLData object.
 * \author peghoty
 * \date 2012/02/27
 */
JX_Int 
JX_3tAPCTLDestroy4Jasmin( JX_Solver solver )
{
   return( jx_3tAPCTLDestroy4Jasmin( (void *) solver ) );
}

JX_Int 
JX_3tAPCTLDestroy4mgJasmin( JX_Solver solver )
{
   return( jx_3tAPCTLDestroy4mgJasmin( (void *) solver ) );
}

/*!
 * \fn JX_Int jx_3tAPCTLDestroy4Jasmin
 * \brief Destroy the jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2012/02/27
 */
JX_Int
jx_3tAPCTLDestroy4Jasmin( void *vdata )
{
   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jx_3tAPCTLDestroy4Jasmin_sp(pre_3tapctl_data);
   }
   else if (nprocs > 1)
   {
      jx_3tAPCTLDestroy4Jasmin_mp(pre_3tapctl_data);
   }
   
   return 0;
}

JX_Int
jx_3tAPCTLDestroy4mgJasmin( void *vdata )
{
   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jx_3tAPCTLDestroy4mgJasmin_mp(pre_3tapctl_data);
   }

   return 0;
}

/*!
 * \fn JX_Int jx_3tAPCTLDestroy4Jasmin_sp
 * \brief Destroy the jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2012/02/27
 */
JX_Int
jx_3tAPCTLDestroy4Jasmin_sp( void *vdata )
{
   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   
   if (pre_3tapctl_data)
   {  
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
         
      // AEE_amg_solver
      if ( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
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
 * \fn JX_Int jx_3tAPCTLDestroy4Jasmin_mp
 * \brief Destroy the jx_3tAPCTLData type object. 
 * \author peghoty 
 * \date 2012/03/01
 */
JX_Int
jx_3tAPCTLDestroy4Jasmin_mp( void *vdata )
{

   jx_3tAPCTLData  *pre_3tapctl_data = vdata;
   JX_Int blocksmooth_type;
   JX_Int  Need_CC;      
   
   JX_Int groupid_x;
   
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
         
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            // VRE
            if ( jx_3tAPCTLDataVRE(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVRE(pre_3tapctl_data) );
            }
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

         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            // VIE
            if ( jx_3tAPCTLDataVIE(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVIE(pre_3tapctl_data) );
            }
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

JX_Int
jx_3tAPCTLDestroy4mgJasmin_mp( void *vdata )
{
   jx_3tAPCTLData *pre_3tapctl_data = vdata;
   JX_Int blocksmooth_type, Need_CC, groupid_x, ng;

   if (pre_3tapctl_data)
   {
      groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
      blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
      Need_CC = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
      ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);

      if (groupid_x < ng)
      {
         if ( jx_3tAPCTLDataARR(pre_3tapctl_data) )
         {
            jx_ParCSRMatrixDestroy( jx_3tAPCTLDataARR(pre_3tapctl_data) );
         }
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            if ( jx_3tAPCTLDataVRE(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVRE(pre_3tapctl_data) );
            }
         }
         if ( jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) )
         {
            jx_PAMGDestroy( jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) )
         {
            jx_GMRESDestroy(jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data));
         }
         if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataWRR(pre_3tapctl_data) )
         {
            jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWRR(pre_3tapctl_data)) );
            jx_ParVectorDestroy( jx_3tAPCTLDataWRR(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
      }
      else if (groupid_x == ng)
      {
         if ( jx_3tAPCTLDataAEE(pre_3tapctl_data) )
         {
            jx_ParCSRMatrixDestroy( jx_3tAPCTLDataAEE(pre_3tapctl_data) );
         }
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            if ( jx_3tAPCTLDataVER(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVER(pre_3tapctl_data) );
            }
            if ( jx_3tAPCTLDataVEI(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVEI(pre_3tapctl_data) );
            }
         }
         if ( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) )
         {
            jx_PAMGDestroy( jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWEE(pre_3tapctl_data)) );
            jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDatatTEMP(pre_3tapctl_data) ) jx_TFree( jx_3tAPCTLDatatTEMP(pre_3tapctl_data) );
      }
      else if (groupid_x == ng+1)
      {
         if ( jx_3tAPCTLDataAII(pre_3tapctl_data) )
         {
            jx_ParCSRMatrixDestroy( jx_3tAPCTLDataAII(pre_3tapctl_data) );
         }
         if ( Need_CC == 1 || blocksmooth_type == BLOCKSMOOTH_GS )
         {
            if ( jx_3tAPCTLDataVIE(pre_3tapctl_data) )
            {
               jx_ParVectorDestroy( jx_3tAPCTLDataVIE(pre_3tapctl_data) );
            }
         }
         if ( jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) )
         {
            jx_PAMGDestroy( jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataRHS(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataRHS(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataWEE(pre_3tapctl_data) )
         {
            jx_ParVectorDestroy( jx_3tAPCTLDataWEE(pre_3tapctl_data) );
         }
         if ( jx_3tAPCTLDataWII(pre_3tapctl_data) )
         {
            jx_TFree( jx_ParVectorLocalVector(jx_3tAPCTLDataWII(pre_3tapctl_data)) );
            jx_ParVectorDestroy( jx_3tAPCTLDataWII(pre_3tapctl_data) );
         }
      }

      if ( jx_3tAPCTLDataACC(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataACC(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataP(pre_3tapctl_data) )
      {
         jx_ParCSRMatrixDestroy( jx_3tAPCTLDataP(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) )
      {
         jx_PAMGDestroy( jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) )
      {
         jx_GMRESDestroy(jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data));
      }
      if ( jx_3tAPCTLDataWCC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataWCC(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataGCC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataGCC(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataRES(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataRES(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataPRR(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataPRR(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataPII(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataPII(pre_3tapctl_data) );
      }
      if ( jx_3tAPCTLDataJAC(pre_3tapctl_data) )
      {
         jx_ParVectorDestroy( jx_3tAPCTLDataJAC(pre_3tapctl_data) );
      }
      jx_MPI_Comm_free( &jx_3tAPCTLDataCommX(pre_3tapctl_data) ); 
      jx_MPI_Comm_free( &jx_3tAPCTLDataCommY(pre_3tapctl_data) );
      jx_TFree(pre_3tapctl_data);
   }
 
   return jx_error_flag;
}
