//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  apctl_mg.c -- 适合于JASMIN 应用程序接口的 APCTL-Krylov 解法器调用示例程序.
 *
 * Created by Yue Xiaoqiang 2019/10/03
 *
 * Xiangtan University
 * yuexq1111@163.com
 *
 */

#include "jx_pamg.h"
#include "jx_krylov.h"
#include "jx_apctl.h"

jx_CSRMatrix *
jx_CSRMatrixTDMGReorderByVariablesT( jx_CSRMatrix *A, JX_Int num_groups );
jx_Vector *
jx_SeqVectorTDMGReorderByVariablesT( jx_Vector *x, JX_Int num_groups );

//===========================================================================================//
//====================================== 主  函  数 ==========================================//
//===========================================================================================//

int
main( int argc, char *argv[] )
{
   MPI_Comm  comm = MPI_COMM_WORLD;
   JX_Int       myid, nprocs;

   JX_Int arg_index   = 0;
   JX_Int print_usage = 0;

   char *MatFile = NULL;
   char *RhsFile = NULL;

   /* local variables */
   jx_CSRMatrix    **ARR_s = NULL;
   jx_CSRMatrix     *AEE_s = NULL;
   jx_CSRMatrix     *AII_s = NULL;
   jx_Vector       **VRE_s = NULL;
   jx_Vector       **VER_s = NULL;
   jx_Vector        *VEI_s = NULL;
   jx_Vector        *VIE_s = NULL;

   jx_ParCSRMatrix  **ARR_p = NULL;
   jx_ParCSRMatrix   *AEE_p = NULL;
   jx_ParCSRMatrix   *AII_p = NULL;
   jx_ParVector     **VRE_p = NULL;
   jx_ParVector     **VER_p = NULL;
   jx_ParVector      *VEI_p = NULL;
   jx_ParVector      *VIE_p = NULL;

   jx_CSRMatrix     *A_s  = NULL;
   jx_CSRMatrix     *B_s  = NULL;

   jx_Vector        *f_s  = NULL;
   jx_Vector        *b_s  = NULL;
   jx_Vector        *u_s  = NULL;

   jx_Vector       **fR_s = NULL;
   jx_Vector        *fE_s = NULL;
   jx_Vector        *fI_s = NULL;

   jx_ParVector    **fR_p = NULL;
   jx_ParVector     *fE_p = NULL;
   jx_ParVector     *fI_p = NULL;

   jx_Vector       **uR_s = NULL;
   jx_Vector        *uE_s = NULL;
   jx_Vector        *uI_s = NULL;

   jx_ParVector    **uR_p = NULL;
   jx_ParVector     *uE_p = NULL;
   jx_ParVector     *uI_p = NULL;

   JX_Int ng, idx, file_sss, problem_id, file_base;

   JX_Int *mypartition = NULL;

   jx_APCTLKrylovParam *apctlkrylov_param = NULL; // apctl-gmres

   JX_Int     solver_id;             // 1: PCG; 2: PGMRES; 3: PBiCGSTAB
   JX_Int     precond_id;            // 1: APCTL; 2: Schur1; 3: Schur2
   JX_Real  tol;                   // tolerance of the APCTL-Krylov method
   JX_Int     max_iter;              // maximal number of iteration
   JX_Int     k_dim;                 // number of restart
   JX_Int     is_check_restarted;    // peghoty, 2011/11/08
   JX_Int     print_level;           // how many info to be output?
   JX_Int     TTest;                 // whether timing the program?
   JX_Int     keepsol;               // whether save the solution?

   JX_Real    strong_threshold;

   JX_Int     interp_type;
   JX_Int     coarsen_type;
   JX_Int     agg_num_levels;
   JX_Int     coarse_threshold;

   JX_Int     print_level_amg;
   JX_Int     print_level_apctl;     // how much info to be output in apctl?
                                  // 1: CPU information
                                  // 2: inner iteration information
                                  // 3: both CPU and inner iteration information
   JX_Int     blocksmooth_type;      // BD or GS type preconditioner when Coarse Correction is not needed?
                                  // BLOCKSMOOTH_BD:
                                  // BLOCKSMOOTH_GS:

   /* solver type and restart number for interpolation-building of PRR */
   JX_Int     interp_solver_ARR;     // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   JX_Int     interp_kdim_ARR;       // restart parameters for GMRES solver

   JX_Int     interp_maxit_ARR;      // maximal number of iteration for ARR to build PRR
   JX_Int     interp_maxit_AII;      // maximal number of iteration for AII to build PII
   JX_Real  interp_tol_ARR;        // tolerance for ARR to build PRR
   JX_Real  interp_tol_AII;        // tolerance for AII to build PII

   JX_Real  ARR_relax_tol;
   JX_Real  AEE_relax_tol;
   JX_Real  AII_relax_tol;
   JX_Real  ACC_relax_tol;

   JX_Int     ARR_relax_type;
   JX_Int     AEE_relax_type;
   JX_Int     AII_relax_type;

   JX_Int     fixit_pctl_R;          // fixed number of iterations for ARR in PCTL
   JX_Int     fixit_pctl_E;          // fixed number of iterations for AEE in PCTL
   JX_Int     fixit_pctl_I;          // fixed number of iterations for AII in PCTL
   JX_Int     fixit_brlx_R;          // fixed number of iterations for ARR in Block Relaxation
   JX_Int     fixit_brlx_E;          // fixed number of iterations for AEE in Block Relaxation
   JX_Int     fixit_brlx_I;          // fixed number of iterations for AII in Block Relaxation

   /* whether employ the fixed-number-of-iterations mode? peghoty, 2012/02/15 */
   JX_Int     use_fixedmode_R;
   JX_Int     use_fixedmode_E;
   JX_Int     use_fixedmode_I;

   /* parameters to describe the weaking coupling between AEE and VER, VEI */
   JX_Real  theta_wc_E;
   JX_Real  threshold_wc_E;

   /* parameters to describe the diagonal dominance */
   JX_Real  theta_dd_R;
   JX_Real  theta_dd_E;
   JX_Real  theta_dd_I;
   JX_Real  threshold_dd_R;
   JX_Real  threshold_dd_E;
   JX_Real  threshold_dd_I;

   /* flag to indicate whether diagonal elements of the three DiagonalBlock matrices
      are firstly stored in each row for the CSR format. peghoty, 2012/03/06 */
   JX_Int    is_diagelm_first;
 
   /* Whether use the pure PCTL? */
   JX_Int    use_ppctl; 

   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JX_Int    test_subls_iter;

   JX_Int    debug_flag;

   JX_Int    reset_zero;

   //--------------------------
   //  启动 MPI
   //--------------------------
   jx_MPI_Init(&argc, &argv);
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   ng = 1;
   file_sss = 0;
   file_base = 1;
   precond_id = 1;      // 1: APCTL; 2: Schur1; 3: Schur2
   problem_id = 13;

   //-----------------------
   //  命令行修改参数
   //-----------------------
   while (arg_index < argc)
   {
       if ( strcmp(argv[arg_index], "-pid") == 0 )
       {
           arg_index ++;
           problem_id = atoi(argv[arg_index++]);
       }
       else if ( strcmp(argv[arg_index], "-pcd") == 0 )
       {
           arg_index ++;
           precond_id = atoi(argv[arg_index++]);
       } 
       else if ( strcmp(argv[arg_index], "-fb") == 0 )
       {
           arg_index ++;
           file_base = atoi(argv[arg_index++]);
       } 
       else if ( strcmp(argv[arg_index], "-ng") == 0 )
       {
           arg_index ++;
           ng = atoi(argv[arg_index++]);
       }
       else if ( strcmp(argv[arg_index], "-help") == 0 )
       {
           print_usage = 1;
           break;
       }
       else
       {
           arg_index ++;
       }
   }

   if (print_usage)
   {
      jx_printf("\n");
      jx_printf("  Usage: %s [<options>]\n", argv[0]);
      jx_printf("\n");
      jx_printf("  -pid     <val> : problem id\n");
      jx_printf("  -pcd     <val> : preconditioner id\n");
      jx_printf("  -fb      <val> : file base\n");
      jx_printf("  -ng      <val> : number of groups\n");
      jx_printf("  -help          : using help message\n\n");
      exit(1);
   }

   //-------------------------------------------------------------
   //  提供线性系统文件
   //-------------------------------------------------------------
   switch (problem_id)
   {
      case 13:
      MatFile = "/home/yuexq/data/3d3t/128/A.000061.01";
      RhsFile = "/home/yuexq/data/3d3t/128/b.000061.01";
      break;
      case 14:
      MatFile = "/home/yuexq/data/3d3t/128/A.000095.01";
      RhsFile = "/home/yuexq/data/3d3t/128/b.000095.01";
      break;
      case 15:
      MatFile = "/home/yuexq/data/3d3t/128/A.000106.01";
      RhsFile = "/home/yuexq/data/3d3t/128/b.000106.01";
      break;
   }

   if (myid == 0)
   {
      jx_printf("\n\n ++++++++++++++ problem %d ng = %d np = %d ++++++++++++\n\n", problem_id, ng, nprocs);
   }

   ARR_s = jx_CTAlloc(jx_CSRMatrix *, ng);
   VRE_s = jx_CTAlloc(jx_Vector *, ng);
   VER_s = jx_CTAlloc(jx_Vector *, ng);
   ARR_p = jx_CTAlloc(jx_ParCSRMatrix *, ng);
   VRE_p = jx_CTAlloc(jx_ParVector *, ng);
   VER_p = jx_CTAlloc(jx_ParVector *, ng);
   fR_s  = jx_CTAlloc(jx_Vector *, ng);
   fR_p = jx_CTAlloc(jx_ParVector *, ng);
   uR_s = jx_CTAlloc(jx_Vector *, ng);
   uR_p = jx_CTAlloc(jx_ParVector *, ng);

   //----------------------------------------------------------------------------------------
   //  在 0 号进程，利用数据文件生成串行矩阵 A_s, 串行右端 f_s,
   //  串行解向量 u_s, 并抽取子矩阵和子向量
   //----------------------------------------------------------------------------------------
   if (myid == 0)
   {
      if (file_sss)
      {
         A_s = jx_CSRMatrixRead(MatFile, file_base);
      }
      else
      {
         B_s = jx_CSRMatrixRead(MatFile, file_base);
         A_s = jx_CSRMatrixTDMGReorderByVariablesT(B_s, ng);
         jx_CSRMatrixDestroy(B_s);
      }
      if (file_sss)
      {
         f_s = jx_SeqVectorRead(RhsFile);
      }
      else
      {
         b_s = jx_SeqVectorRead(RhsFile);
         f_s = jx_SeqVectorTDMGReorderByVariablesT(b_s, ng);
         jx_SeqVectorDestroy(b_s);
      }
      u_s = jx_SeqVectorCreate(jx_VectorSize(f_s));
      jx_SeqVectorInitialize(u_s);
      jx_SeqVectorSetConstantValues(u_s, 0.0);
      jx_mgGetSubBlocks_REIV(A_s, ARR_s, &AEE_s, &AII_s, VRE_s, VER_s, &VEI_s, &VIE_s, ng);
      jx_mgGetSubVecs(f_s, fR_s, &fE_s, &fI_s, ng);
      jx_mgGetSubVecs(u_s, uR_s, &uE_s, &uI_s, ng);
   }

   //-------------------------------------------------------------------------
   //  生成并行矩阵和并行向量
   //-------------------------------------------------------------------------

   for (idx = 0; idx < ng; idx ++) ARR_p[idx] = jx_CSRMatrixToParCSRMatrix(comm, ARR_s[idx], NULL, NULL);
   AEE_p = jx_CSRMatrixToParCSRMatrix(comm, AEE_s, NULL, NULL);
   AII_p = jx_CSRMatrixToParCSRMatrix(comm, AII_s, NULL, NULL);
   for (idx = 0; idx < ng; idx ++) VRE_p[idx] = jx_VectorToParVector(comm, VRE_s[idx], NULL);
   for (idx = 0; idx < ng; idx ++) VER_p[idx] = jx_VectorToParVector(comm, VER_s[idx], NULL);
   VEI_p = jx_VectorToParVector(comm, VEI_s, NULL);
   VIE_p = jx_VectorToParVector(comm, VIE_s, NULL);

   for (idx = 0; idx < ng; idx ++) fR_p[idx] = jx_VectorToParVector(comm, fR_s[idx], NULL);
   fE_p = jx_VectorToParVector(comm, fE_s, NULL);
   fI_p = jx_VectorToParVector(comm, fI_s, NULL);
   for (idx = 0; idx < ng; idx ++) uR_p[idx] = jx_VectorToParVector(comm, uR_s[idx], NULL);
   uE_p = jx_VectorToParVector(comm, uE_s, NULL);
   uI_p = jx_VectorToParVector(comm, uI_s, NULL);

   //------------------------------------------------------------------------------
   //  设置解法器 APCTL-GMRES 的参数
   //------------------------------------------------------------------------------
   solver_id          = 2;      // 2: PGMRES
   tol                = 1.0e-8;
   max_iter           = 200;
   k_dim              = 30;
   is_check_restarted = 0;
   print_level        = 3;
   TTest              = 1;
   keepsol            = 0;

   print_level_apctl  = 0;
   blocksmooth_type   = BLOCKSMOOTH_BD; // BlockDiag

   interp_solver_ARR  = SOLVER_AMG; // 0: SOLVER_AMG; 1: SOLVER_AMGGMRES;
   interp_kdim_ARR    = 20;

   ARR_relax_type     = RELAX_AMG;
   AEE_relax_type     = RELAX_AMG;
   AII_relax_type     = RELAX_AMG;

   interp_maxit_ARR   = 200;
   interp_maxit_AII   = 200;
   interp_tol_ARR     = 1.0e-6;
   interp_tol_AII     = 1.0e-6;

   fixit_pctl_R       = 1;
   fixit_pctl_E       = 1;
   fixit_pctl_I       = 1;
   fixit_brlx_R       = 3;
   fixit_brlx_E       = 1;
   fixit_brlx_I       = 1;

   ARR_relax_tol      = 1.0e-6;
   AEE_relax_tol      = 1.0e-6;
   AII_relax_tol      = 1.0e-6;
   ACC_relax_tol      = 1.0e-6;

   use_fixedmode_R    = 1;
   use_fixedmode_E    = 1;
   use_fixedmode_I    = 1;

   theta_wc_E         = 1.0e-4;
   threshold_wc_E     = 0.5;
   theta_dd_R         = 0.9;
   theta_dd_E         = 0.9;
   theta_dd_I         = 0.9;
   threshold_dd_R     = 1.0;
   threshold_dd_E     = 1.0;
   threshold_dd_I     = 1.0;

   strong_threshold = 0.25;

   interp_type = 0;
   coarsen_type = 6;
   agg_num_levels = 0;
   coarse_threshold = 100;
   print_level_amg = 3;

   is_diagelm_first   = 0;
   use_ppctl          = 0;  /* Whether use the pure PCTL? */

   test_subls_iter    = 1;

   debug_flag         = 0; /* 0:no; 1:setup; 2:solve; 3:setup+solve */

   reset_zero         = 1;

   apctlkrylov_param = jx_APCTLKrylovParamInitialize();
   jx_APCTLKrylovParamSetNumGroup(apctlkrylov_param, ng);
   jx_APCTLKrylovParamSetSolverID(apctlkrylov_param, solver_id);
   jx_APCTLKrylovParamSetPrecondID(apctlkrylov_param, precond_id);
   jx_APCTLKrylovParamSetTol(apctlkrylov_param, tol);
   jx_APCTLKrylovParamSetMaxIter(apctlkrylov_param, max_iter);
   jx_APCTLKrylovParamSetKDim(apctlkrylov_param, k_dim);
   jx_APCTLKrylovParamSetISCheckRestarted(apctlkrylov_param, is_check_restarted);
   jx_APCTLKrylovParamSetPrintLevel(apctlkrylov_param, print_level);
   jx_APCTLKrylovParamSetTTest(apctlkrylov_param, TTest);
   jx_APCTLKrylovParamSetKeepSol(apctlkrylov_param, keepsol);

   jx_APCTLKrylovParamSetPrintLevelAPCTL(apctlkrylov_param, print_level_apctl);
   jx_APCTLKrylovParamSetBlockSmoothType(apctlkrylov_param, blocksmooth_type);
   jx_APCTLKrylovParamSetInterpSolverARR(apctlkrylov_param, interp_solver_ARR);
   jx_APCTLKrylovParamSetInterpKdimARR(apctlkrylov_param, interp_kdim_ARR);
   jx_APCTLKrylovParamSetInterpMaxitARR(apctlkrylov_param, interp_maxit_ARR);
   jx_APCTLKrylovParamSetInterpMaxitAII(apctlkrylov_param, interp_maxit_AII);
   jx_APCTLKrylovParamSetInterpTolARR(apctlkrylov_param, interp_tol_ARR);
   jx_APCTLKrylovParamSetInterpTolAII(apctlkrylov_param, interp_tol_AII);
   jx_APCTLKrylovParamSetARRRelaxTol(apctlkrylov_param, ARR_relax_tol);
   jx_APCTLKrylovParamSetAEERelaxTol(apctlkrylov_param, AEE_relax_tol);
   jx_APCTLKrylovParamSetAIIRelaxTol(apctlkrylov_param, AII_relax_tol);
   jx_APCTLKrylovParamSetACCRelaxTol(apctlkrylov_param, ACC_relax_tol);
   jx_APCTLKrylovParamSetFixitPCTLR(apctlkrylov_param, fixit_pctl_R);
   jx_APCTLKrylovParamSetFixitPCTLE(apctlkrylov_param, fixit_pctl_E);
   jx_APCTLKrylovParamSetFixitPCTLI(apctlkrylov_param, fixit_pctl_I);
   jx_APCTLKrylovParamSetFixitBrlxR(apctlkrylov_param, fixit_brlx_R);
   jx_APCTLKrylovParamSetFixitBrlxE(apctlkrylov_param, fixit_brlx_E);
   jx_APCTLKrylovParamSetFixitBrlxI(apctlkrylov_param, fixit_brlx_I);
   jx_APCTLKrylovParamSetARRRelaxType(apctlkrylov_param, ARR_relax_type);
   jx_APCTLKrylovParamSetAEERelaxType(apctlkrylov_param, AEE_relax_type);
   jx_APCTLKrylovParamSetAIIRelaxType(apctlkrylov_param, AII_relax_type);
   jx_APCTLKrylovParamSetUseFixedModeR(apctlkrylov_param, use_fixedmode_R);
   jx_APCTLKrylovParamSetUseFixedModeE(apctlkrylov_param, use_fixedmode_E);
   jx_APCTLKrylovParamSetUseFixedModeI(apctlkrylov_param, use_fixedmode_I);
   jx_APCTLKrylovParamSetThetaWCE(apctlkrylov_param, theta_wc_E);
   jx_APCTLKrylovParamSetParamThresholdWCE(apctlkrylov_param, threshold_wc_E);
   jx_APCTLKrylovParamSetThetaDDR(apctlkrylov_param, theta_dd_R);
   jx_APCTLKrylovParamSetThetaDDE(apctlkrylov_param, theta_dd_E);
   jx_APCTLKrylovParamSetThetaDDI(apctlkrylov_param, theta_dd_I);
   jx_APCTLKrylovParamSetThresholdDDR(apctlkrylov_param, threshold_dd_R);
   jx_APCTLKrylovParamSetThresholdDDE(apctlkrylov_param, threshold_dd_E);
   jx_APCTLKrylovParamSetThresholdDDI(apctlkrylov_param, threshold_dd_I);
   jx_APCTLKrylovParamSetISDiagElmFirst(apctlkrylov_param, is_diagelm_first);
   jx_APCTLKrylovParamSetUsePPCTL(apctlkrylov_param, use_ppctl);
   jx_APCTLKrylovParamSetTestSubLSIter(apctlkrylov_param, test_subls_iter);
   jx_APCTLKrylovParamSetDebugFlag(apctlkrylov_param, debug_flag);
   jx_APCTLKrylovParamSetResetZero(apctlkrylov_param, reset_zero);
   jx_APCTLKrylovParamSetStrongThreshold(apctlkrylov_param, strong_threshold);
   jx_APCTLKrylovParamSetInterpType(apctlkrylov_param, interp_type);
   jx_APCTLKrylovParamSetCoarsenType(apctlkrylov_param, coarsen_type);
   jx_APCTLKrylovParamSetAggNumLevels(apctlkrylov_param, agg_num_levels);
   jx_APCTLKrylovParamSetCoarseThreshold(apctlkrylov_param, coarse_threshold);
   jx_APCTLKrylovParamSetPrintLevelAMG(apctlkrylov_param, print_level_amg);

   //-----------------------------------------------------------
   //  调用 APCTL-GMRES 法进行求解
   //-----------------------------------------------------------
   jx_ApctlKrylov_mgJASMIN(ARR_p, AEE_p, AII_p, VRE_p, VER_p, VEI_p, VIE_p,
                      fR_p, fE_p, fI_p, uR_p, uE_p, uI_p, apctlkrylov_param);

   if (myid == 0)
   {
      // 打印迭代次数和最终相对残量范数
      jx_printf(" >> Number of Iterations  : %d\n", apctlkrylov_param->num_iterations);
      jx_printf(" >> Last Relative Res Norm: %le\n\n", apctlkrylov_param->last_rel_nrm);

      // 打印 apctl 预条件子中各子系统的迭代次数
      if (test_subls_iter)
      {
         if (jx_APCTLKrylovParamNeedCC(apctlkrylov_param))
         {
            jx_printf(" >> num_iter_Ai_pctl_setup: %d\n", apctlkrylov_param->num_iter_Ai_pctl_setup);
            jx_printf(" >> num_iter_Ar_pctl_setup: %d\n", apctlkrylov_param->num_iter_Ar_pctl_setup);
            jx_printf(" >> num_iter_Ar_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ar_pctl_precond);
            jx_printf(" >> num_iter_Ae_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ae_pctl_precond);
            jx_printf(" >> num_iter_Ai_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ai_pctl_precond);
            jx_printf(" >> num_iter_Ac_pctl_precond: %d\n\n", apctlkrylov_param->num_iter_Ac_pctl_precond);
         }
         else
         {
            jx_printf(" >> num_iter_Ar_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ar_pctl_precond);
            jx_printf(" >> num_iter_Ae_pctl_precond: %d\n", apctlkrylov_param->num_iter_Ae_pctl_precond);
            jx_printf(" >> num_iter_Ai_pctl_precond: %d\n\n", apctlkrylov_param->num_iter_Ai_pctl_precond);
         }
      }

      // 打印各个阶段的 CPU 时间信息
      if (TTest)
      {
         jx_printf(" >> cpu_time_datatransf: %.3f seconds\n", apctlkrylov_param->cpu_trans);
         jx_printf(" >> cpu_time_setupphase: %.3f seconds\n", apctlkrylov_param->cpu_setup);
         jx_printf(" >> cpu_time_solvephase: %.3f seconds\n\n", apctlkrylov_param->cpu_solve);
         jx_printf(" >> cpu_time_setupsolve: %.3f seconds\n\n", apctlkrylov_param->cpu_total);
      }
   }

   //-----------------------------------------------------------
   //  释放内存
   //-----------------------------------------------------------

   if (myid == 0)
   {
      jx_CSRMatrixDestroy(A_s);
      jx_SeqVectorDestroy(f_s);
      jx_SeqVectorDestroy(u_s);

      for (idx = 0; idx < ng; idx ++) jx_CSRMatrixDestroy(ARR_s[idx]);
      jx_TFree(ARR_s);
      jx_CSRMatrixDestroy(AEE_s);
      jx_CSRMatrixDestroy(AII_s);

      for (idx = 0; idx < ng; idx ++) jx_SeqVectorDestroy(VRE_s[idx]);
      jx_TFree(VRE_s);
      for (idx = 0; idx < ng; idx ++) jx_SeqVectorDestroy(VER_s[idx]);
      jx_TFree(VER_s);
      jx_SeqVectorDestroy(VEI_s);
      jx_SeqVectorDestroy(VIE_s);

      for (idx = 0; idx < ng; idx ++) jx_SeqVectorDestroy(uR_s[idx]);
      jx_TFree(uR_s);
      jx_SeqVectorDestroy(uE_s);
      jx_SeqVectorDestroy(uI_s);

      for (idx = 0; idx < ng; idx ++) jx_SeqVectorDestroy(fR_s[idx]);
      jx_TFree(fR_s);
      jx_SeqVectorDestroy(fE_s);
      jx_SeqVectorDestroy(fI_s);
   }

   for (idx = 0; idx < ng; idx ++) jx_ParCSRMatrixDestroy(ARR_p[idx]);
   jx_TFree(ARR_p);
   jx_ParCSRMatrixDestroy(AEE_p);
   jx_ParCSRMatrixDestroy(AII_p);
   for (idx = 0; idx < ng; idx ++) jx_ParVectorDestroy(VRE_p[idx]);
   jx_TFree(VRE_p);
   for (idx = 0; idx < ng; idx ++) jx_ParVectorDestroy(VER_p[idx]);
   jx_TFree(VER_p);
   jx_ParVectorDestroy(VEI_p);
   jx_ParVectorDestroy(VIE_p);
   for (idx = 0; idx < ng; idx ++) jx_ParVectorDestroy(fR_p[idx]);
   jx_TFree(fR_p);
   jx_ParVectorDestroy(fE_p);
   jx_ParVectorDestroy(fI_p);
   for (idx = 0; idx < ng; idx ++) jx_ParVectorDestroy(uR_p[idx]);
   jx_TFree(uR_p);
   jx_ParVectorDestroy(uE_p);
   jx_ParVectorDestroy(uI_p);
   jx_TFree(mypartition);
   jx_TFree(apctlkrylov_param);

   //-----------------------------------------------------------
   //  终止 MPI
   //-----------------------------------------------------------
   MPI_Finalize();

   return 0;
}

jx_CSRMatrix *
jx_CSRMatrixTDMGReorderByVariablesT( jx_CSRMatrix *A, JX_Int num_groups )
{
   jx_CSRMatrix *C = NULL;
   JX_Int num_rows = jx_CSRMatrixNumRows(A);
   JX_Int *IA = jx_CSRMatrixI(A);
   JX_Int *JA = jx_CSRMatrixJ(A);
   JX_Real *AA = jx_CSRMatrixData(A);
   JX_Int *IC = NULL;
   JX_Int *JC = NULL;
   JX_Real *CC = NULL;
   JX_Int RowA = 0, RowC = 0;
   JX_Int ng_p_two = num_groups + 2;
   JX_Int sub_num_rows = num_rows / ng_p_two;
   JX_Int num_nonzerosC, j, mdo, col, row_end;

   C = jx_CSRMatrixCreate(num_rows, num_rows, jx_CSRMatrixNumNonzeros(A));
   jx_CSRMatrixInitialize(C);
   IC = jx_CSRMatrixI(C);
   JC = jx_CSRMatrixJ(C);
   CC = jx_CSRMatrixData(C);
   IC[0] = 0;
   num_nonzerosC = 0;
   for (RowC = 0; RowC < num_rows; RowC ++)
   {
      mdo = RowC / sub_num_rows;
      RowA = ng_p_two * (RowC - mdo * sub_num_rows) + mdo;
      row_end = IA[RowA+1];
      for (j = IA[RowA]; j < row_end; j ++)
      {
         col = JA[j];
         mdo = col % ng_p_two;
         JC[num_nonzerosC] = (col - mdo) / ng_p_two + mdo * sub_num_rows;
         CC[num_nonzerosC] = AA[j];
         num_nonzerosC ++;
      }
      IC[RowC+1] = num_nonzerosC;
   }

   return C;
}

jx_Vector *
jx_SeqVectorTDMGReorderByVariablesT( jx_Vector *x, JX_Int num_groups )
{
   JX_Int size = jx_VectorSize(x);
   JX_Real *x_data = jx_VectorData(x);
   jx_Vector *y = jx_SeqVectorCreate(size);
   JX_Real *y_data = NULL;
   JX_Int ng_p_two = num_groups + 2;
   JX_Int sub_size = size / ng_p_two;
   JX_Int Rowx = 0, Rowy, mdo;

   jx_SeqVectorInitialize(y);
   y_data = jx_VectorData(y);
   for (Rowy = 0; Rowy < size; Rowy ++)
   {
      mdo = Rowy / sub_size;
      Rowx = ng_p_two * (Rowy - mdo * sub_size) + mdo;
      y_data[Rowy] = x_data[Rowx];
   }

   return y;
}
