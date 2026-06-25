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
JX_Int
jx_ParVectorSetRandomValues( jx_ParVector *v, JX_Int seed );
JX_Int
jx_SeqVectorSetRandomValues( jx_Vector *x, JX_Int seed );

//===========================================================================================//
//====================================== 主  函  数 ==========================================//
//===========================================================================================//

int
main( int argc, char *argv[] )
{
   MPI_Comm comm = MPI_COMM_WORLD;
   JX_Int   myid, nprocs;

   JX_Int arg_index   = 0;
   JX_Int print_usage = 0;

   char *MatFile = NULL;
   char *RhsFile = NULL;

   /* local variables */
   jx_CSRMatrix *A_s = NULL;
   jx_CSRMatrix *B_s = NULL;

   jx_ParCSRMatrix *par_mat = NULL;

   jx_Vector *f_s = NULL;
   jx_Vector *b_s = NULL;

   jx_ParVector *par_rhs = NULL;

   jx_ParVector *par_sol = NULL;

   JX_Int ng, file_sss, problem_id;

   JX_Int initguess = 0;
   JX_Int num_iterations;

   JX_Int *mypartition = NULL;

   JX_Real norm, final_res_norm;
   JX_Real starttime, endtime;
   JX_Real starttimeT, endtimeT;

   JX_Int     solver_id;
   JX_Solver  schur_solver;
   JX_Real    tol;                   // tolerance
   JX_Int     max_iter;              // maximal number of iteration
   JX_Int     k_dim;                 // number of restart
   JX_Int     is_check_restarted;    // peghoty, 2011/11/08
   JX_Int     print_level;           // how many info to be output?
   JX_Int     keepsol;               // whether save the solution?

   JX_Real    strong_threshold;
   JX_Int     interp_type;
   JX_Int     coarsen_type;
   JX_Int     agg_num_levels;
   JX_Int     coarse_threshold;
   JX_Int     print_level_amg;
   JX_Int     print_level_schur;  // how much info to be output?
                                  // 1: CPU information
                                  // 2: inner iteration information
                                  // 3: both CPU and inner iteration information

   JX_Real  ARR_relax_tol;
   JX_Real  AEE_relax_tol;
   JX_Real  AII_relax_tol;
   JX_Int   ARR_relax_type;
   JX_Int   AEE_relax_type;
   JX_Int   AII_relax_type;
   JX_Int   fixit_pctl_R;          // fixed number of iterations for ARR in SCHUR
   JX_Int   fixit_pctl_E;          // fixed number of iterations for AEE in SCHUR
   JX_Int   fixit_pctl_I;          // fixed number of iterations for AII in SCHUR
   JX_Int   fixit_brlx_R;          // fixed number of iterations for ARR in Block Relaxation
   JX_Int   fixit_brlx_E;          // fixed number of iterations for AEE in Block Relaxation
   JX_Int   fixit_brlx_I;          // fixed number of iterations for AII in Block Relaxation
   JX_Int   use_fixedmode_R;
   JX_Int   use_fixedmode_E;
   JX_Int   use_fixedmode_I;

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
   JX_Int   is_diagelm_first;
 
   /* Whether test the number of iterations for each sub-linearsystem solution? */
   JX_Int    test_subls_iter;
   JX_Int    debug_flag;
   JX_Int    reset_zero;

   JX_Solver solver;

   //--------------------------
   //  启动 MPI
   //--------------------------
   jx_MPI_Init(&argc, &argv);
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   ng         = 1;
   solver_id  = 22;  // 22: SCHUR1-GMRES, 23: SCHUR2-GMRES
   problem_id = 13;

   //--------------------------------
   // 设置解法器 SCHUR-GMRES 的参数
   //--------------------------------
   tol                = 1.0e-8;
   max_iter           = 200;
   k_dim              = 30;
   is_check_restarted = 0;
   print_level        = 3;
   keepsol            = 0;
   print_level_schur  = 3;
   ARR_relax_type     = RELAX_AMG;
   AEE_relax_type     = RELAX_AMG;
   AII_relax_type     = RELAX_AMG;
   fixit_pctl_R       = 1;
   fixit_pctl_E       = 1;
   fixit_pctl_I       = 1;
   fixit_brlx_R       = 3;
   fixit_brlx_E       = 1;
   fixit_brlx_I       = 1;
   ARR_relax_tol      = 1.0e-6;
   AEE_relax_tol      = 1.0e-6;
   AII_relax_tol      = 1.0e-6;
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
   strong_threshold   = 0.25;
   interp_type        = 0;
   coarsen_type       = 6;
   agg_num_levels     = 0;
   coarse_threshold   = 100;
   print_level_amg    = 3;
   is_diagelm_first   = 0;
   test_subls_iter    = 1;
   debug_flag         = 0; /* 0:no; 1:setup; 2:solve; 3:setup+solve */
   reset_zero         = 1;

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
       else if ( strcmp(argv[arg_index], "-sid") == 0 )
       {
           arg_index ++;
           solver_id = atoi(argv[arg_index++]);
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
      jx_printf("  -sid     <val> : solver id\n");
      jx_printf("  -fb      <val> : file base\n");
      jx_printf("  -ng      <val> : number of groups\n");
      jx_printf("  -help          : using help message\n\n");
      exit(1);
   }

   //-------------------------------------------------------------
   //  提供线性系统文件
   //-------------------------------------------------------------

   file_sss = 0;
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

   //----------------------------------------------------------------------------------------
   // 在 0 号进程，利用数据文件生成串行矩阵 A_s, 串行右端 f_s,
   //----------------------------------------------------------------------------------------

   starttime = jx_MPI_Wtime();

   if (myid == 0)
   {
      if (file_sss)
      {
         A_s = jx_CSRMatrixRead(MatFile, 1);
      }
      else
      {
         B_s = jx_CSRMatrixRead(MatFile, 1);
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
   }

   //-------------------------------------------------------------------------
   //  生成并行矩阵和并行向量
   //-------------------------------------------------------------------------

   par_mat = jx_CSRMatrixToParCSRMatrix(comm, A_s, NULL, NULL);
   jx_ParCSRMatrixGetRowPartitioning(par_mat, &mypartition);
   par_rhs = jx_VectorToParVector(comm, f_s, mypartition);
   par_sol = jx_ParVectorCreate(comm, jx_ParVectorGlobalSize(par_rhs), mypartition);
   jx_ParVectorSetPartitioningOwner(par_sol, 0);
   jx_ParVectorInitialize(par_sol);
   if (initguess == 0)
   {
      jx_ParVectorSetConstantValues(par_sol, 0.0);
   }
   else
   {
      jx_ParVectorSetRandomValues(par_sol, 22775);
      norm = jx_ParVectorInnerProd(par_sol, par_sol);
      norm = 1.0 / sqrt(norm);
      jx_ParVectorScale(norm, par_sol);
   }

   endtime = jx_MPI_Wtime();
   jx_GetWallTime(comm, "BuildParLinearSystem", starttime, endtime, 0, 2);

   //----------------------------------------------------------------------------------------
   // 求解线性代数系统
   //----------------------------------------------------------------------------------------

   starttimeT = jx_MPI_Wtime();

   switch (solver_id)
   {
      case 22:  /* SCHUR1-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: SCHUR1-GMRES(%d) \n\n", k_dim);

         starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&schur_solver);
         JX_3tAPCTLSetNumGroup(schur_solver, ng);
         JX_3tAPCTLSetA(schur_solver, par_mat);
         JX_3tAPCTLSetIsDiagElmFirst(schur_solver, is_diagelm_first);
         JX_3tAPCTLSetSubBlocks(schur_solver);
         JX_3tAPCTLSetMaxiter(schur_solver, 1);
         JX_3tAPCTLSetPrintLevel(schur_solver, print_level_schur);
         JX_3tAPCTLSetARRRelaxTol(schur_solver, ARR_relax_tol);
         JX_3tAPCTLSetAEERelaxTol(schur_solver, AEE_relax_tol);
         JX_3tAPCTLSetAIIRelaxTol(schur_solver, AII_relax_tol);
         JX_3tAPCTLSetARRRelaxType(schur_solver, ARR_relax_type);
         JX_3tAPCTLSetAEERelaxType(schur_solver, AEE_relax_type);
         JX_3tAPCTLSetAIIRelaxType(schur_solver, AII_relax_type);
         JX_3tAPCTLSetThetaWCE(schur_solver, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(schur_solver, threshold_wc_E);
         JX_3tAPCTLSetThetaDDR(schur_solver, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(schur_solver, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(schur_solver, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(schur_solver, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(schur_solver, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(schur_solver, threshold_dd_I);
         JX_3tAPCTLSetFixItPCTLR(schur_solver, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(schur_solver, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(schur_solver, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(schur_solver, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(schur_solver, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(schur_solver, fixit_brlx_I);
         JX_3tAPCTLSetUseFixedModeR(schur_solver, use_fixedmode_R);
         JX_3tAPCTLSetUseFixedModeE(schur_solver, use_fixedmode_E);
         JX_3tAPCTLSetUseFixedModeI(schur_solver, use_fixedmode_I);
         JX_3tAPCTLSetTestSubLSIter(schur_solver, test_subls_iter);
         JX_3tAPCTLSetDebugFlag(schur_solver, debug_flag);
         JX_3tAPCTLSetResetZero(schur_solver, reset_zero);
         JX_3tAPCTLSetStrongThreshold(schur_solver, strong_threshold);
         JX_3tAPCTLSetInterpType(schur_solver, interp_type);
         JX_3tAPCTLSetCoarsenType(schur_solver, coarsen_type);
         JX_3tAPCTLSetAggNumLevels(schur_solver, agg_num_levels);
         JX_3tAPCTLSetCoarseThreshold(schur_solver, coarse_threshold);
         JX_3tAPCTLSetPrintLevelAMG(schur_solver, print_level_amg);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);

         JX_GMRESSetPrecond( solver, (JX_PtrToSolverFcn) JX_3tABSC1mgPrecond, (JX_PtrToSolverFcn) NULL, schur_solver );

         JX_3tABSC1Setup4mgJasmin(schur_solver, (JX_ParCSRMatrix)par_mat);

         JX_GMRESSetup( solver, (JX_Matrix)par_mat, (JX_Vector)par_rhs, (JX_Vector)par_sol );

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "SCHUR1 Setup", starttime, endtime, 0, 2);

         starttime = jx_MPI_Wtime();

         JX_GMRESSolve( solver,
                        (JX_Matrix)par_mat, // preOperator
                        (JX_Matrix)par_mat,
                        (JX_Vector)par_rhs,
                        (JX_Vector)par_sol );

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "SCHUR1 Solve", starttime, endtime, 0, 2);

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
 
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
  
         if (print_level_schur)
         {
            JX_3tAPCTLIterCount(schur_solver);
         }

         JX_3tAPCTLDestroy4mgJasmin(schur_solver);
         JX_ParCSRGMRESDestroy(solver);

      }
      break;

      case 23:  /* SCHUR2-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: SCHUR2-GMRES(%d) \n\n", k_dim);

         starttime = jx_MPI_Wtime();

         JX_3tAPCTLDataInitialize(&schur_solver);
         JX_3tAPCTLSetNumGroup(schur_solver, ng);
         JX_3tAPCTLSetA(schur_solver, par_mat);
         JX_3tAPCTLSetIsDiagElmFirst(schur_solver, is_diagelm_first);
         JX_3tAPCTLSetSubBlocks(schur_solver);
         JX_3tAPCTLSetMaxiter(schur_solver, 1);
         JX_3tAPCTLSetPrintLevel(schur_solver, print_level_schur);
         JX_3tAPCTLSetARRRelaxTol(schur_solver, ARR_relax_tol);
         JX_3tAPCTLSetAEERelaxTol(schur_solver, AEE_relax_tol);
         JX_3tAPCTLSetAIIRelaxTol(schur_solver, AII_relax_tol);
         JX_3tAPCTLSetARRRelaxType(schur_solver, ARR_relax_type);
         JX_3tAPCTLSetAEERelaxType(schur_solver, AEE_relax_type);
         JX_3tAPCTLSetAIIRelaxType(schur_solver, AII_relax_type);
         JX_3tAPCTLSetThetaWCE(schur_solver, theta_wc_E);
         JX_3tAPCTLSetThresholdWCE(schur_solver, threshold_wc_E);
         JX_3tAPCTLSetThetaDDR(schur_solver, theta_dd_R);
         JX_3tAPCTLSetThetaDDE(schur_solver, theta_dd_E);
         JX_3tAPCTLSetThetaDDI(schur_solver, theta_dd_I);
         JX_3tAPCTLSetThresholdDDR(schur_solver, threshold_dd_R);
         JX_3tAPCTLSetThresholdDDE(schur_solver, threshold_dd_E);
         JX_3tAPCTLSetThresholdDDI(schur_solver, threshold_dd_I);
         JX_3tAPCTLSetFixItPCTLR(schur_solver, fixit_pctl_R);
         JX_3tAPCTLSetFixItPCTLE(schur_solver, fixit_pctl_E);
         JX_3tAPCTLSetFixItPCTLI(schur_solver, fixit_pctl_I);
         JX_3tAPCTLSetFixItBRLXR(schur_solver, fixit_brlx_R);
         JX_3tAPCTLSetFixItBRLXE(schur_solver, fixit_brlx_E);
         JX_3tAPCTLSetFixItBRLXI(schur_solver, fixit_brlx_I);
         JX_3tAPCTLSetUseFixedModeR(schur_solver, use_fixedmode_R);
         JX_3tAPCTLSetUseFixedModeE(schur_solver, use_fixedmode_E);
         JX_3tAPCTLSetUseFixedModeI(schur_solver, use_fixedmode_I);
         JX_3tAPCTLSetTestSubLSIter(schur_solver, test_subls_iter);
         JX_3tAPCTLSetDebugFlag(schur_solver, debug_flag);
         JX_3tAPCTLSetResetZero(schur_solver, reset_zero);
         JX_3tAPCTLSetStrongThreshold(schur_solver, strong_threshold);
         JX_3tAPCTLSetInterpType(schur_solver, interp_type);
         JX_3tAPCTLSetCoarsenType(schur_solver, coarsen_type);
         JX_3tAPCTLSetAggNumLevels(schur_solver, agg_num_levels);
         JX_3tAPCTLSetCoarseThreshold(schur_solver, coarse_threshold);
         JX_3tAPCTLSetPrintLevelAMG(schur_solver, print_level_amg);

         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level);

         JX_GMRESSetPrecond( solver, (JX_PtrToSolverFcn) JX_3tABSC2mgPrecond, (JX_PtrToSolverFcn) NULL, schur_solver );

         JX_3tABSC2Setup4mgJasmin(schur_solver, (JX_ParCSRMatrix)par_mat);

         JX_GMRESSetup( solver, (JX_Matrix)par_mat, (JX_Vector)par_rhs, (JX_Vector)par_sol );

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "SCHUR2 Setup", starttime, endtime, 0, 2);

         starttime = jx_MPI_Wtime();

         JX_GMRESSolve( solver,
                        (JX_Matrix)par_mat, // preOperator
                        (JX_Matrix)par_mat,
                        (JX_Vector)par_rhs,
                        (JX_Vector)par_sol );

         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "SCHUR2 Solve", starttime, endtime, 0, 2);

         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
 
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
  
         if (print_level_schur)
         {
            JX_3tAPCTLIterCount(schur_solver);
         }

         JX_3tAPCTLDestroy4mgJasmin(schur_solver);
         JX_ParCSRGMRESDestroy(solver);

      }
      break;
   }

   endtimeT = jx_MPI_Wtime();
   jx_GetWallTime(comm, "Total Sove Time", starttimeT, endtimeT, 0, 2);

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

   //-----------------------------------------------------------
   //  释放内存
   //-----------------------------------------------------------

   if (myid == 0)
   {
      jx_CSRMatrixDestroy(A_s);
      jx_SeqVectorDestroy(f_s);
   }
   jx_ParCSRMatrixDestroy(par_mat);
   jx_ParVectorDestroy(par_rhs);
   jx_ParVectorDestroy(par_sol);

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

JX_Int
jx_ParVectorSetRandomValues( jx_ParVector *v, JX_Int seed )
{
   JX_Int my_id;
   jx_Vector *v_local = jx_ParVectorLocalVector(v);

   MPI_Comm 	comm = jx_ParVectorComm(v);
   jx_MPI_Comm_rank(comm, &my_id); 

   seed *= (my_id + 1);
           
   return jx_SeqVectorSetRandomValues(v_local,seed);
}

JX_Int
jx_SeqVectorSetRandomValues( jx_Vector *x, JX_Int seed )
{
   JX_Real  *vector_data = jx_VectorData(x);
   JX_Int      size        = jx_VectorSize(x);
   JX_Int      i, ierr = 0;

   jx_SeedRand(seed);

   size *= jx_VectorNumVectors(x);

   for (i = 0; i < size; i ++)
   {
      vector_data[i] = 2.0 * jx_Rand() - 1.0;
   }
   
   return ierr;
}
