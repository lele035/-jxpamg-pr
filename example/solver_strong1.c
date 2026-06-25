//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 * solver_strong.c -- This is an example demonstrating how to solve a given
 * linear system in the form of files by calling JXPAMG solver and
 * the JXPAMG-based Krylov iterative methods.
 *
 * One of the following solvers can be used by
 * assigning parameter 'solver_id'
 *
 *  solver_id = 0:  PAMG
 *  solver_id = 11: CG
 *  solver_id = 12: PAMG-CG
 *  solver_id = 13: DS-CG
 *  solver_id = 14: Euclid-CG
 *  solver_id = 21: GMRES
 *  solver_id = 22: PAMG-GMRES
 *  solver_id = 23: DS-GMRES
 *  solver_id = 24: Euclid-GMRES
 *  solver_id = 25: ILU-GMRES
 *  solver_id = 26: ILU-AdaptiveGMRES
 *  solver_id = 28: Euclid2-AdaptiveGMRES
 *  solver_id = 29: Euclid3-AdaptiveGMRES
 *  solver_id = 31: BiCGSTAB
 *  solver_id = 32: PAMG-BiCGSTAB
 *  solver_id = 33: DS-BiCGSTAB
 *  solver_id = 34: Euclid-BiCGSTAB
 *  solver_id = 42: PAMG-FlexGMRES
 *  solver_id = 62: PAMG-COGMRES
 *
 * combined preconditioners
 *
 *  solver_id = 51: PAMG-Euclid-PAMG-GMRES
 *  solver_id = 52: Euclid-PAMG-GMRES
 *  solver_id = 53: PAMG-Euclid-GMRES
 *  solver_id = 54: Euclid-PAMG-Euclid-GMRES
 *  solver_id = 56: ILU-PAMG-GMRES
 *  solver_id = 57: ILU-PAMG-ILU-GMRES
 *
 * The input data for the linear system to be solved should be
 * provided in the form as follows (the indices are of Fortran
 * style, i.e, starting from 1 not 0):
 *  1. matrix file
 *----------------------------------------
 *     n
 *     ia(i), i = 1(1)n+1
 *     ja(i), i = 1(1)nz
 *      a(i), i = 1(1)nz
 *----------------------------------------
 *  where nz = ia(n+1)-1.
 *  2. rhs file
 *----------------------------------------
 *     n
 *     f(i),i = 1(1)n
 *----------------------------------------
 *
 *  Created by peghoty 2011/05/13
 *
 *  Xiangtan University
 *  peghoty@163.com
 *
 *  Modified by Yue Xiaoqiang 2012/10/22
 *
 *  Xiangtan University
 *  yuexq1111@163.com
 *
 */

#include "jx_pamg.h"
#include "jx_ilu.h"
#include "jx_krylov.h"
#include "jx_diagscale.h"
#include "jx_euclid.h"
#include "jx_combined.h"

JX_Int
jx_ParVectorSetRandomValues( jx_ParVector *v, JX_Int seed );
JX_Int
jx_SeqVectorSetRandomValues( jx_Vector *x, JX_Int seed );

int
main( int argc, char *argv[] )
{
   MPI_Comm  comm = MPI_COMM_WORLD;
   JX_Int       myid, nprocs;
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
   JX_Int       nthreads;
#endif
   
   JX_Int arg_index   = 0;
   JX_Int print_usage = 0;
   
   JX_Real starttime, endtime;
   JX_Real starttimeT, endtimeT;
   
   char *MatFile = NULL;
   char *RhsFile = NULL;
   char *GusFile = NULL;
   char  AppFile[120];
   
   jx_ParCSRMatrix *par_matrix = NULL;
   jx_ParVector    *par_rhs    = NULL;
   jx_ParVector    *par_sol    = NULL;
   jx_CSRMatrix    *ser_matrix = NULL;
   
   JX_Int *partitioning = NULL;

   JX_Int **grid_relax_points = NULL;
   
   /* DiagScale Precond */
   JX_Solver ds_solver;
   
   /* ILU Precond */
   JX_Solver ilu_solver;
   JX_Real drop_tol;
   
   /* Euclid solver */
   JX_Solver euclid_solver;
   JX_Int euclid_level;
   JX_Int euclid_bj;
   
   /* Combined */
   JX_Solver combined_solver;
   JX_Int theta_psi;
   JX_Int theta_rho;
   JX_Int theta_phi;
   JX_Real theta_dis;
   
   /* JXPAMG solver */
   JX_Solver amg_solver;
   JX_Int       max_levels;
   JX_Int       cycle_type;
   JX_Int       relax_type;
   JX_Int       measure_type;
   JX_Int       rap2;
   JX_Int       num_functions;
   JX_Int       ns_down;
   JX_Int       ns_up;
   JX_Int       ns_coarse;
   JX_Int       restri_type;
   JX_Int       keepTranspose;
   JX_Int       coarsen_type;
   JX_Int       interp_type;
   JX_Int       P_max_elmts;
   JX_Int       agg_num_levels;
   JX_Int       ai_measure_type;
   JX_Int       ai_relax_type;
   JX_Real    strong_threshold;
   JX_Real    max_row_sum;
   
   JX_Real    relax_wt;
   JX_Real    outer_wt;
   
   JX_Real S_commpkg_switch;
   JX_Real AIR_strong_th;

   JX_Int       coarse_threshold;
   JX_Real    coarse_ratio;
   JX_Int       coarsestsolverid;
   JX_Int       conv_criteria;
   JX_Int       amg_print_level;
   
   /* iterative method */
   JX_Solver solver;
   JX_Real    tol;
   JX_Real    resdown_0_threshold;
   JX_Real    convfac_threshold_2;
   JX_Int       max_iter;
   JX_Int       k_dim;
   JX_Int       is_check_restarted;  /* peghoty, 2011/11/08 */
   JX_Int       twonorm;
   JX_Int       solver_id;
   JX_Int       problem_id;
   JX_Int       file_base;
   JX_Int       print_level;
   JX_Int       keepsol;
   JX_Int       TTest;
   JX_Int       cgs;
   JX_Int       unroll;
   
   /* other variables */
   JX_Int       lu_length;
   JX_Int       glosize, i;
   JX_Int       last_precond_type;
   JX_Int       initguess = 0;
   JX_Int       num_iterations;
   JX_Real    final_res_norm;
   JX_Real    norm;
   
   //--------------------------
   //  启动 MPI
   //--------------------------
   jx_MPI_Init(&argc, &argv);
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);
   
   //-----------------------
   //  参数设置
   //-----------------------
   max_levels       = 25;       /* 最大网格层数 */
   cycle_type       = 1;        /* Cycle 类型  1: V_Cycle; 2：W_Cycle */
   relax_type       = 3;        /* Relax 类型  3: hGS; 6：hSGS */
   measure_type     = 0;        /* 影响值的计算方式 0：局部；1：全局 */
   rap2             = 0;        /* RAP计算方式  0：RAP；1：先算Q=AP，再算RQ */
   num_functions    = 1;
   ns_down          = 1;
   ns_up            = 1;
   ns_coarse        = 1;
   restri_type      = 0;        /* 限制算子类型 0: P^T, 1: AIR, 2: AIR-2 */
   keepTranspose    = 0;        /* 存放限制算子  0：no；1：yes */
   coarsen_type     = 6;        /* 粗化策略 */
   interp_type      = 0;        /* 插值策略 */
   P_max_elmts      = 0;        /* 插值算子每行最大非零元个数 */
   agg_num_levels   = 0;        /* Aggressive粗化的层数 */
   ai_measure_type  = 0;        /* AI-策略  0: no; 1: yes */
   ai_relax_type    = 0;        /* AI-磨光  0: no; 1: yes */
   amg_print_level  = 0;        /* work only when AMG as preconditioner */
   strong_threshold = 0.25;     /* 强弱连通参数, 0.25 for 2D, 0.5 for 3D is recommended */
   max_row_sum      = 0.9;      /* 行和参数 */
   relax_wt         = 1.0;
   outer_wt         = 1.0;
   S_commpkg_switch = 1.0;
   AIR_strong_th    = 0.25;
   
   coarse_threshold = 100;      /* 最粗网格层上网格节点个数的最大值 */
   coarse_ratio     = 0.75;     /* 相邻两个网格层的粗点个数超过细点个数的 coarse_ratio, 则换成 CLJP 粗化 */
   coarsestsolverid = 9;        /* 最粗网格层解法器 */
   conv_criteria    = 0;        /* 收敛准则类型 */
   
   theta_psi = 4;               /* 控制多尺度强度的度量 */
   theta_rho = 3;               /* 控制多尺度分布的第一个度量 */
   theta_phi = 3;               /* 控制多尺度分布的第二个度量 */
   theta_dis = 1.0e-3;          /* 控制多尺度分布的个数 */
   
   drop_tol         = 0.0;      /* Drop-tolerance for ILU(0) factorization */
   
   euclid_level     = 1;        /* level of fill-in */
   euclid_bj        = 0;        /* Select PILU (0) or Block Jacobi ILU (1) */
   
   tol                 = 1.0e-7; /* 控制精度 */
   resdown_0_threshold = 1.0e-4; /* 第一次残量相对初始残差的下降阈值 */
   convfac_threshold_2 = 0.1;    /* 相继两次残量下降阈值 */
   max_iter            = 5;    /* 迭代法最大迭代次数 */
   k_dim               = 30;     /* 回头数 */
   is_check_restarted  = 1;      /* peghoty, 2011/11/08 */
   twonorm             = 0;      /* PCG 法中的范数控制类型，0: B 范数; 1: l2 范数 */
   print_level         = 3;      /* 0: 关闭；1：Setup参数；2：Solve参数；3：Setup+Solve参数 */
   keepsol             = 0;      /* 是否保存解向量 */
   TTest               = 1;      /* 是否测试时间 */
   cgs                 = 1;      /* COGMRES: if 2 performs reorthogonalization */
   unroll              = 0;      /* COGMRES: Set number of unrolling in mass funcyions, can be 4 or 8. Default: no unrolling */
   solver_id           = 22;
   problem_id          = 1;
   file_base           = 1;
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
   nthreads            = 1;      /* 线程数 */
#endif
   
   //-----------------------
   //  命令行修改参数
   //-----------------------
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-sid") == 0 )
      {
         arg_index ++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pid") == 0 )
      {
         arg_index ++;
         problem_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cgs") == 0 )
      {
         arg_index ++;
         cgs = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-unroll") == 0 )
      {
         arg_index ++;
         unroll = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fb") == 0 )
      {
         arg_index ++;
         file_base = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index ++;
         num_functions = atoi(argv[arg_index++]);
      }
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
      else if ( strcmp(argv[arg_index], "-nts") == 0 )
      {
         arg_index ++;
         nthreads = atoi(argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-rap2") == 0 )
      {
         arg_index ++;
         rap2 = 1;
      }
      else if ( strcmp(argv[arg_index], "-air") == 0 )
      {
         arg_index ++;
         restri_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-kt") == 0 )
      {
         arg_index ++;
         keepTranspose = 1;
      }
      else if (strcmp(argv[arg_index], "-Pmx") == 0)
      {
         arg_index ++;
         P_max_elmts = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_nl") == 0 )
      {
         arg_index ++;
         agg_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tnrm") == 0 )
      {
         arg_index ++;
         twonorm = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-kdim") == 0 )
      {
         arg_index ++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dtol") == 0 )
      {
         arg_index ++;
         drop_tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-euc_lvl") == 0 )
      {
         arg_index ++;
         euclid_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-euc_bj") == 0 )
      {
         arg_index ++;
         euclid_bj = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index ++;
         relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ai_rlx") == 0 )
      {
         arg_index ++;
         ai_relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ai_mt") == 0 )
      {
         arg_index ++;
         ai_measure_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ct") == 0 )
      {
         arg_index ++;
         coarsen_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ipt") == 0 )
      {
         arg_index ++;
         interp_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxct") == 0 )
      {
         arg_index ++;
         coarse_threshold = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-str") == 0 )
      {
         arg_index ++;
         strong_threshold = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index ++;
         max_row_sum = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-scs") == 0 )
      {
         arg_index ++;
         S_commpkg_switch = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-airst") == 0 )
      {
         arg_index ++;
         AIR_strong_th = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amg_ptlv") == 0 )
      {
         arg_index ++;
         amg_print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-theta_ms") == 0 )
      {
         arg_index ++;
         theta_psi = atoi(argv[arg_index++]);
         theta_rho = atoi(argv[arg_index++]);
         theta_phi = atoi(argv[arg_index++]);
         theta_dis = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ptlv") == 0 )
      {
         arg_index ++;
         print_level = atoi(argv[arg_index++]);
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
      jx_printf("    -sid <val>   : solver id\n");
      jx_printf("    -pid <val>   : problem id\n");
      jx_printf("    -nf  <val>   : number of functions\n");
      jx_printf("    -nts <val>   : threads number\n");
      jx_printf("    -rap2        : 2nd implementation of RAP\n");
      jx_printf("    -kt          : keep transpose\n");
      jx_printf("    -Pmx <val>   : maximal number of elements per row for interpolation\n");
      jx_printf(" -agg_nl <val>   : num_levels for aggressive coarsening\n");
      jx_printf("   -tnrm <val>   : two_norm in PCG\n");
      jx_printf("   -kdim <val>   : krylov dimension\n");
      jx_printf("   -dtol <val>   : Drop-tolerance for ILU(0) factorization\n");
      jx_printf("-euc_lvl <val>   : level of fill-in for Euclid\n");
      jx_printf(" -euc_bj <val>   : PILU or Block Jacobi ILU\n");
      jx_printf("  -rlx <val>     : relaxation type\n");
      jx_printf("  -ai_rlx <val>  : AI relaxation type\n");
      jx_printf("  -ai_mt  <val>  : AI measure type\n");
      jx_printf("     -ct <val>   : coarsening type\n");
      jx_printf("    -ipt <val>   : interpolation type\n");
      jx_printf("   -mxct <val>   : max. size on coarsest grid\n");
      jx_printf("    -str <val>   : AMG strength threshold\n");
      jx_printf("   -mxrs <val>   : maximum row sum threshold for dependency weakening\n");
      jx_printf("-amg_ptlv <val>  : print_level of AMG when AMG as preconditioner\n");
      jx_printf("-theta_ms <iiiv> : threshold for multi-scale judgement\n");
      jx_printf("   -ptlv <val>   : print_level\n");
      jx_printf("   -help         : using help message\n\n");
      exit(1);
   }
   
   //-----------------------
   //  提供线性系统文件
   //-----------------------
   switch (problem_id)
   {
#if 0
#if 0
      case 1:
      MatFile = "/vol6/home/xtu_yxq/data/3dmg/A.0006.01";
      RhsFile = "/vol6/home/xtu_yxq/data/3dmg/b.0006.01";
      break;
      case 2:
      MatFile = "/vol6/home/xtu_yxq/data/3dmg/A.0009.01";
      RhsFile = "/vol6/home/xtu_yxq/data/3dmg/b.0009.01";
      break;
      case 3:
      MatFile = "/vol6/home/xtu_yxq/data/3dmg/A.0013.01";
      RhsFile = "/vol6/home/xtu_yxq/data/3dmg/b.0013.01";
      break;
      case 4:
      MatFile = "/vol6/home/xtu_yxq/data/3dmg/A.0038.01";
      RhsFile = "/vol6/home/xtu_yxq/data/3dmg/b.0038.01";
      break;
      case 5:
      MatFile = "/vol6/home/xtu_yxq/data/3dmg/A.0041.01";
      RhsFile = "/vol6/home/xtu_yxq/data/3dmg/b.0041.01";
      break;
      case 6:
      MatFile = "/vol6/home/xtu_yxq/data/3dmg/A.0059.01";
      RhsFile = "/vol6/home/xtu_yxq/data/3dmg/b.0059.01";
      break;
#else
      case 1:
      MatFile = "/vol6/home/xtu_yxq/data/2dmg/A.0006.01";
      RhsFile = "/vol6/home/xtu_yxq/data/2dmg/b.0006.01";
      break;
      case 2:
      MatFile = "/vol6/home/xtu_yxq/data/2dmg/A.0009.01";
      RhsFile = "/vol6/home/xtu_yxq/data/2dmg/b.0009.01";
      break;
      case 3:
      MatFile = "/vol6/home/xtu_yxq/data/2dmg/A.0013.01";
      RhsFile = "/vol6/home/xtu_yxq/data/2dmg/b.0013.01";
      break;
      case 4:
      MatFile = "/vol6/home/xtu_yxq/data/2dmg/A.0038.01";
      RhsFile = "/vol6/home/xtu_yxq/data/2dmg/b.0038.01";
      break;
      case 5:
      MatFile = "/vol6/home/xtu_yxq/data/2dmg/A.0041.01";
      RhsFile = "/vol6/home/xtu_yxq/data/2dmg/b.0041.01";
      break;
      case 6:
      MatFile = "/vol6/home/xtu_yxq/data/2dmg/A.0059.01";
      RhsFile = "/vol6/home/xtu_yxq/data/2dmg/b.0059.01";
      break;
#endif
#else
      case 1:
      MatFile = "/vol6/home/xtu_yxq/data/1dmg/mat.6.01.4";
      RhsFile = "/vol6/home/xtu_yxq/data/1dmg/rhs.6.01.4";
      break;
      case 2:
      MatFile = "/vol6/home/xtu_yxq/data/1dmg/mat.9.01.4";
      RhsFile = "/vol6/home/xtu_yxq/data/1dmg/rhs.9.01.4";
      break;
      case 3:
      MatFile = "/vol6/home/xtu_yxq/data/1dmg/mat.13.01.4";
      RhsFile = "/vol6/home/xtu_yxq/data/1dmg/rhs.13.01.4";
      break;
      case 4:
      MatFile = "/vol6/home/xtu_yxq/data/1dmg/mat.38.01.4";
      RhsFile = "/vol6/home/xtu_yxq/data/1dmg/rhs.38.01.4";
      break;
      case 5:
      MatFile = "/vol6/home/xtu_yxq/data/1dmg/mat.41.01.4";
      RhsFile = "/vol6/home/xtu_yxq/data/1dmg/rhs.41.01.4";
      break;
      case 6:
      MatFile = "/vol6/home/xtu_yxq/data/1dmg/mat.59.01.4";
      RhsFile = "/vol6/home/xtu_yxq/data/1dmg/rhs.59.01.4";
      break;
#endif
   }
   
   if (myid == 0)
   {
      jx_printf("\n\n+++++++++++++++++++++ Problem %d,", problem_id);
#if JX_USING_BIG_INT
      jx_printf(" Using BIG_INT,");
#endif
#if JX_USING_BIG_DOUBLE
      jx_printf(" BIG_DOUBLE,");
#endif
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
      jx_printf(" With OpenMP using %d threads,", nthreads);
#endif
      jx_printf(" MPI using %d processors +++++++++++++++++++++\n\n", nprocs);
   }
   
   //----------------------------------------------------------------
   // 设定线程数
   //----------------------------------------------------------------
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
   omp_set_num_threads(nthreads);
#endif
   
   if (restri_type) /* Set Restriction to be AIR */
   {
      interp_type = 100; /* 1-pt Interp */
      relax_type = 0;
      ns_down = 0;
      ns_up = 3;
      grid_relax_points = jx_CTAlloc(JX_Int *, 4);
      grid_relax_points[0] = NULL;
      grid_relax_points[1] = jx_CTAlloc(JX_Int, ns_down);
      grid_relax_points[2] = jx_CTAlloc(JX_Int, ns_up);
      grid_relax_points[3] = jx_CTAlloc(JX_Int, ns_coarse);
      for (i = 0; i < ns_down; i ++) grid_relax_points[1][i] = 0; /* down cycle */
      if (ns_up == 3) /* up cycle */
      {
         grid_relax_points[2][0] = -1; // F
         grid_relax_points[2][1] = -1; // F
         grid_relax_points[2][2] = 1;  // C
      }
      else if (ns_up == 2)
      {
         grid_relax_points[2][0] = -1;
         grid_relax_points[2][1] = -1;
      }
      for (i = 0; i < ns_coarse; i ++) grid_relax_points[3][i] = 0; /* coarse: all */
      coarse_threshold = 100;
      agg_num_levels = 0; /* does not support aggressive coarsening */
   }

   //--------------------------------------------------------------------
   //  利用文件创建并行矩阵和并行右端, 以及并行初始迭代向量(零向量或随机向量)
   //--------------------------------------------------------------------
   if (TTest) starttime = jx_MPI_Wtime();
   
   par_matrix = jx_BuildMatParFromOneFile(MatFile, 1, file_base);
   par_rhs    = jx_BuildRhsParFromOneFile(RhsFile, par_matrix);
   
   if (!GusFile)
   {
      glosize      = jx_ParVectorGlobalSize(par_rhs);
      partitioning = jx_ParVectorPartitioning(par_rhs);
      par_sol      = jx_ParVectorCreate(comm, glosize, partitioning);
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
         norm = 1./sqrt(norm);
         jx_ParVectorScale(norm, par_sol);
      }
   }
   else
   {
      par_sol = jx_BuildRhsParFromOneFile(GusFile, par_matrix);
   }
   
   endtime = jx_MPI_Wtime();
   jx_GetWallTime(comm, "BuildParLinearSystem", starttime, endtime, 0, 2);
   
   //-----------------------------------------------------------
   //  求解线性代数系统
   //-----------------------------------------------------------
   starttimeT = jx_MPI_Wtime();
   
   switch (solver_id)
   {
      case 0:  /* PAMG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG \n\n");
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, max_iter);
         JX_PAMGSetNumFunctions(amg_solver, num_functions);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetTol(amg_solver, tol);
         JX_PAMGSetConvCriteria(amg_solver, conv_criteria);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, print_level);
         JX_PAMGSetCoarsestSolverID(amg_solver, coarsestsolverid);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetCoarseRatio(amg_solver, coarse_ratio);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);           /* relax_type for "coarsest" */
         
         //------------------------------------------------------------
         //    JX_PAMG Setup
         //------------------------------------------------------------
         if (max_levels != 1)
         {
            JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix) par_matrix);
         }
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG Setup", starttime, endtime, 0, 2);
         }
         
         //------------------------------------------------------------
         //    JX_PAMG Solve
         //------------------------------------------------------------
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PAMGSolve(amg_solver, (JX_ParCSRMatrix)par_matrix, (JX_ParVector)par_rhs, (JX_ParVector)par_sol);
         
         JX_PAMGGetNumIterations(amg_solver, &num_iterations);
         JX_PAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG Solve", starttime, endtime, 0, 2);
         }
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_PAMGDestroy(amg_solver);
      }
      break;
      
      case 11:  /* CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: CG \n\n");
         
         JX_ParCSRPCGCreate(comm, &solver);
         
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);
         
         JX_PCGSetup(solver, (JX_Matrix) par_matrix, (JX_Vector) par_rhs, (JX_Vector) par_sol);
         
         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         JX_ParCSRPCGDestroy(solver);
      }
      break;
      
      case 12:  /* PAMG-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-CG \n\n");
         
         starttime = jx_MPI_Wtime();
         
         JX_PAMGCreate(&amg_solver);
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);           /* relax_type for "coarsest" */
         
         JX_ParCSRPCGCreate(comm, &solver);
         
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);
         
         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond, (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);
         
         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "PAMG-CG Setup", starttime, endtime, 0, 2);
         
         starttime = jx_MPI_Wtime();
         
         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "PAMG-CG Solve", starttime, endtime, 0, 2);
         
         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         JX_PAMGDestroy(amg_solver);
         JX_ParCSRPCGDestroy(solver);
      }
      break;
      
      case 13:  /* DS-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: DS-CG \n\n");
         
         starttime = jx_MPI_Wtime();
         
         ds_solver = NULL;
         
         JX_ParCSRPCGCreate(comm, &solver);
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);
         
         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_DiagScalePrecond,
                                  (JX_PtrToSolverFcn)JX_DiagScaleSetup, ds_solver);
         
         JX_DiagScaleSetup(ds_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "DS-CG Setup", starttime, endtime, 0, 2);
         
         starttime = jx_MPI_Wtime();
         
         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "DS-CG Solve", starttime, endtime, 0, 2);
         
         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         JX_ParCSRPCGDestroy(solver);
      }
      break;
      
      case 14:  /* Euclid-CG */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-CG \n\n");
         
         starttime = jx_MPI_Wtime();
         
         JX_EuclidCreate(comm, &euclid_solver);
         //JX_EuclidSetParams(euclid_solver, argc, argv);
         JX_EuclidSetLevel(euclid_solver, euclid_level);
         JX_EuclidSetBJ(euclid_solver, euclid_bj);
         
         JX_ParCSRPCGCreate(comm, &solver);
         JX_PCGSetMaxIter(solver, max_iter);
         JX_PCGSetTol(solver, tol);
         JX_PCGSetTwoNorm(solver, twonorm);  // 0: B 范数； 1：l2 范数
         JX_PCGSetLogging(solver, 1);
         JX_PCGSetPrintLevel(solver, print_level);
         
         JX_PCGSetPrecond(solver, (JX_PtrToSolverFcn)JX_EuclidSolve,
                                  (JX_PtrToSolverFcn)JX_EuclidSetup, euclid_solver);
         
         JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_PCGSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "Euclid-CG Setup", starttime, endtime, 0, 2);
         
         starttime = jx_MPI_Wtime();
         
         JX_PCGSolve(solver, (JX_Matrix)par_matrix, // preOperater
                             (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "Euclid-CG Solve", starttime, endtime, 0, 2);
         
         JX_PCGGetNumIterations(solver, &num_iterations);
         JX_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         JX_EuclidDestroy(euclid_solver);
         JX_ParCSRPCGDestroy(solver);
      }
      break;
      
      case 21:  /* GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: GMRES(%d) \n\n", k_dim);
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 22:  /* PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         JX_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JX_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);           /* relax_type for "coarsest" */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                    (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);
         
         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_PAMGDestroy(amg_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 23:  /* DS-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: DS-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         ds_solver = NULL;
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_DiagScalePrecond,
                                    (JX_PtrToSolverFcn)JX_DiagScaleSetup, ds_solver);
         
         JX_DiagScaleSetup(ds_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 24:  /* Euclid-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_EuclidCreate(comm, &euclid_solver);
         //JX_EuclidSetParams(euclid_solver, argc, argv);
         JX_EuclidSetLevel(euclid_solver, euclid_level);
         JX_EuclidSetBJ(euclid_solver, euclid_bj);
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_EuclidSolve,
                                    (JX_PtrToSolverFcn)JX_EuclidSetup, euclid_solver);
         
         JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_EuclidDestroy(euclid_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 25:  /* ILU-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_ILUZeroFactorDataCreate(&ilu_solver, comm);
         JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
         JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_ILUZeroFactorDataPrecond,
                                    (JX_PtrToSolverFcn)JX_ILUZeroFactorDataSetup, ilu_solver);
         
         JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU(0)-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU(0)-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_ILUZeroFactorDataGetLULength(ilu_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_ILUZeroFactorDataDestroy(ilu_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 26:  /* ILU-AdaptiveGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-AdaptiveGMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_ILUZeroFactorDataCreate(&ilu_solver, comm);
         JX_ILUZeroFactorDataSetMaxIter(ilu_solver, 1);
         JX_ILUZeroFactorDataSetDropTol(ilu_solver, drop_tol);
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JX_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JX_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_ILUZeroFactorDataPrecond,
                                    (JX_PtrToSolverFcn)JX_ILUZeroFactorDataSetup, ilu_solver);
         
         JX_ILUZeroFactorDataSetup(ilu_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU(0)-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESAdaptiveSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                       (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU(0)-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_ILUZeroFactorDataGetLULength(ilu_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_ILUZeroFactorDataDestroy(ilu_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 28:  /* Euclid2-AdaptiveGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid2-AdaptiveGMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 1);
         JX_CombinedPrecondDataSetEuclidLevel(combined_solver, euclid_level);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataSetAMGParameters(combined_solver, max_levels, relax_type, amg_print_level,
                                                interp_type, P_max_elmts, measure_type, coarsen_type,
                                                agg_num_levels, coarse_threshold, strong_threshold);
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JX_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JX_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSetup2, combined_solver);
         
         JX_CombinedPrecondDataAdaptiveSetup2(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid2-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESAdaptiveSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                       (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid2-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_GMRESGetLastPrecondType(solver, &last_precond_type);
         if (myid == 0)
         {
            if (last_precond_type == 1)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d) with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
            else if (last_precond_type == 21)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d)-PAMG with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 29:  /* Euclid3-AdaptiveGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid3-AdaptiveGMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 1);
         JX_CombinedPrecondDataSetEuclidLevel(combined_solver, euclid_level);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataSetThetaPsiRhoPhi(combined_solver, theta_psi, theta_rho, theta_phi, theta_dis);
         JX_CombinedPrecondDataSetAMGParameters(combined_solver, max_levels, relax_type, amg_print_level,
                                                interp_type, P_max_elmts, measure_type, coarsen_type,
                                                agg_num_levels, coarse_threshold, strong_threshold);
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         JX_GMRESSetResDownZeroThreshold(solver, resdown_0_threshold);
         JX_GMRESSetConvFacThresholdTwo(solver, convfac_threshold_2);
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataAdaptiveSetup3, combined_solver);
         
         JX_CombinedPrecondDataAdaptiveSetup3(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid3-AdaptiveGMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESAdaptiveSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                       (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid3-AdaptiveGMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_GMRESGetLastPrecondType(solver, &last_precond_type);
         if (myid == 0)
         {
            if (last_precond_type == 1)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d) with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
            else if (last_precond_type == 3)
            {
               jx_printf("\n >>> Last Precond --- PAMG \n\n");
            }
            else if (last_precond_type == 21)
            {
               jx_printf("\n >>> Last Precond --- Euclid(%d)-PAMG with Drop-Tol = %.6lf \n\n", euclid_level, drop_tol);
            }
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 51:  /* PAMG-Euclid-PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-Euclid-PAMG-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 11);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);
         
         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 52:  /* Euclid-PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-PAMG-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 12);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);
         
         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 53:  /* PAMG-Euclid-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-Euclid-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 13);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);
         
         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 54:  /* Euclid-PAMG-Euclid-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-PAMG-Euclid-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 14);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);
         
         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-Euclid-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-PAMG-Euclid-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 56:  /* ILU-PAMG-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-PAMG-GMRES(%d) \n\n", k_dim);
         
         if ((nprocs > 1) && (myid == 0))
         {
            ser_matrix = jx_CSRMatrixRead(MatFile, file_base);
            jx_CSRMatrixReorder(ser_matrix);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 16);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         JX_CombinedPrecondDataSetILUMatA(combined_solver, ser_matrix); /* 0号进程上的矩阵 */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);
         
         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         if ((nprocs > 1) && (myid == 0))
         {
            jx_CSRMatrixDestroy(ser_matrix);
         }
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 57:  /* ILU-PAMG-ILU-GMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: ILU(0)-PAMG-ILU(0)-GMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_CombinedPrecondDataCreate(&combined_solver, comm);
         JX_CombinedPrecondDataSetPreID(combined_solver, 17);
         JX_CombinedPrecondDataSetDropTol(combined_solver, drop_tol);
         JX_CombinedPrecondDataInitialize(combined_solver, euclid_level); /* Yue Xiaoqiang 2013/12/11 */
         //JX_CombinedPrecondDataInitializeP(combined_solver, argc, argv); /* Yue Xiaoqiang 2014/05/24 */
         
         JX_ParCSRGMRESCreate(comm, &solver);
         JX_GMRESSetKDim(solver, k_dim);
         JX_GMRESSetIsCheckRestarted(solver, is_check_restarted); /* peghoty 2011/11/08 */
         JX_GMRESSetMaxIter(solver, max_iter);
         JX_GMRESSetTol(solver, tol);
         JX_GMRESSetLogging(solver, 1);
         JX_GMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_GMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_CombinedPrecondDataSolve,
                                    (JX_PtrToSolverFcn)JX_CombinedPrecondDataSetup, combined_solver);
         
         JX_CombinedPrecondDataSetup(combined_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_GMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-ILU(0)-GMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_GMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "ILU-PAMG-ILU(0)-GMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_CombinedPrecondDataGetLULength(combined_solver, &lu_length);
         if (myid == 0)
         {
            jx_printf(" >>> Memory Complexity of ILU(0) = %d\n", lu_length);
         }
         
         JX_GMRESGetNumIterations(solver, &num_iterations);
         JX_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_CombinedPrecondDataDestroy(combined_solver);
         JX_ParCSRGMRESDestroy(solver);
      }
      break;
      
      case 31:  /* BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: BiCGSTAB \n\n");
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);
         
         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "BiCGSTAB Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "BiCGSTAB Solve", starttime, endtime, 0, 2);
         }
         
         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;
      
      case 32:  /* PAMG-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-BiCGSTAB \n\n");
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PAMGCreate(&amg_solver);
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);           /* relax_type for "coarsest" */
         
         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);
         
         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                       (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);
         
         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }
         
         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_PAMGDestroy(amg_solver);
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;
      
      case 33:  /* DS-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: DS-BiCGSTAB \n\n");
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         ds_solver = NULL;
         
         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);
         
         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_DiagScalePrecond,
                                       (JX_PtrToSolverFcn)JX_DiagScaleSetup, ds_solver);
         
         JX_DiagScaleSetup(ds_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "DS-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }
         
         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;
      
      case 34:  /* Euclid-BiCGSTAB */
      {
         if (myid == 0) jx_printf("\n >>> Solver: Euclid-BiCGSTAB \n\n");
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_EuclidCreate(comm, &euclid_solver);
         //JX_EuclidSetParams(euclid_solver, argc, argv);
         JX_EuclidSetLevel(euclid_solver, euclid_level);
         JX_EuclidSetBJ(euclid_solver, euclid_bj);
         
         JX_ParCSRBiCGSTABCreate(comm, &solver);
         JX_BiCGSTABSetMaxIter(solver, max_iter);
         JX_BiCGSTABSetTol(solver, tol);
         JX_BiCGSTABSetAbsoluteTol(solver, 0.0);
         JX_BiCGSTABSetConvCriteria(solver, 0);
         JX_BiCGSTABSetLogging(solver, 1);
         JX_BiCGSTABSetPrintLevel(solver, print_level);
         
         JX_BiCGSTABSetPrecond(solver, (JX_PtrToSolverFcn)JX_EuclidSolve,
                                       (JX_PtrToSolverFcn)JX_EuclidSetup, euclid_solver);
         
         JX_EuclidSetup(euclid_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_BiCGSTABSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-BiCGSTAB Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_BiCGSTABSolve(solver, (JX_Matrix)par_matrix, // preOperater
                                  (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "Euclid-BiCGSTAB Solve", starttime, endtime, 0, 2);
         }
         
         JX_BiCGSTABGetNumIterations(solver, &num_iterations);
         JX_BiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_EuclidDestroy(euclid_solver);
         JX_ParCSRBiCGSTABDestroy(solver);
      }
      break;

      case 42:  /* PAMG-FlexGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-FlexGMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         JX_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JX_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);           /* relax_type for "coarsest" */
         
         JX_ParCSRFlexGMRESCreate(comm, &solver);
         JX_FlexGMRESSetKDim(solver, k_dim);
         JX_FlexGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_FlexGMRESSetMaxIter(solver, max_iter);
         JX_FlexGMRESSetTol(solver, tol);
         JX_FlexGMRESSetLogging(solver, 1);
         JX_FlexGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_FlexGMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                    (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);
         
         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
         
         /* this is optional - could be a user defined one instead */
         JX_FlexGMRESSetModifyPC(solver, (JX_PtrToModifyPCFcn)jx_FlexGMRESModifyPCDefault);
         
         JX_FlexGMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_FlexGMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-FlexGMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_FlexGMRESGetNumIterations(solver, &num_iterations);
         JX_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_PAMGDestroy(amg_solver);
         JX_ParCSRFlexGMRESDestroy(solver);
      }
      break;

      case 62:  /* PAMG-COGMRES */
      {
         if (myid == 0) jx_printf("\n >>> Solver: PAMG-COGMRES(%d) \n\n", k_dim);
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_PAMGCreate(&amg_solver);
         if (restri_type)
         {
            jx_assert(restri_type >= 0);
            JX_PAMGSetRestriction(amg_solver, restri_type);
            JX_PAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         }
         JX_PAMGSetMaxLevels(amg_solver, max_levels);
         JX_PAMGSetMaxIter(amg_solver, 1);
         JX_PAMGSetCycleType(amg_solver, cycle_type);
         JX_PAMGSetMeasureType(amg_solver, measure_type);
         JX_PAMGSetRAP2(amg_solver, rap2);
         JX_PAMGSetKeepTranspose(amg_solver, keepTranspose);
         JX_PAMGSetCoarsenType(amg_solver, coarsen_type);
         JX_PAMGSetInterpType(amg_solver, interp_type);
         JX_PAMGSetPMaxElmts(amg_solver, P_max_elmts);
         JX_PAMGSetAggNumLevels(amg_solver, agg_num_levels);
         JX_PAMGSetAIMeasureType(amg_solver, ai_measure_type);
         JX_PAMGSetAIRelaxType(amg_solver, ai_relax_type);
         JX_PAMGSetStrongThreshold(amg_solver, strong_threshold);
         JX_PAMGSetMaxRowSum(amg_solver, max_row_sum);
         JX_PAMGSetPrintLevel(amg_solver, amg_print_level);
         JX_PAMGSetCoarseThreshold(amg_solver, coarse_threshold);
         JX_PAMGSetRelaxWt(amg_solver, relax_wt);
         JX_PAMGSetOuterWt(amg_solver, outer_wt);
         JX_PAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
         JX_PAMGSetAIRStrongTh(amg_solver, AIR_strong_th);
         if (ns_down > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_down, 1);           /* sweep for "down" */
         if (ns_up > -1) JX_PAMGSetCycleNumSweeps(amg_solver, ns_up, 2);           /* sweep for "up" */
         JX_PAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);           /* sweep for "coarsest" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 1);  /* relax_type for "down" */
         JX_PAMGSetCycleRelaxType(amg_solver, relax_type, 2);  /* relax_type for "up" */
         JX_PAMGSetCycleRelaxType(amg_solver, 9, 3);           /* relax_type for "coarsest" */
         
         JX_ParCSRCOGMRESCreate(comm, &solver);
         JX_COGMRESSetKDim(solver, k_dim);
         JX_COGMRESSetUnroll(solver, unroll);
         JX_COGMRESSetCGS(solver, cgs);
         JX_COGMRESSetIsCheckRestarted(solver, is_check_restarted);
         JX_COGMRESSetMaxIter(solver, max_iter);
         JX_COGMRESSetTol(solver, tol);
         JX_COGMRESSetLogging(solver, 1);
         JX_COGMRESSetPrintLevel(solver, print_level); /* 是否在屏幕上打印残量等信息 */
         
         JX_COGMRESSetPrecond(solver, (JX_PtrToSolverFcn)JX_PAMGPrecond,
                                    (JX_PtrToSolverFcn)JX_PAMGSetup, amg_solver);
         
         JX_PAMGSetup(amg_solver, (JX_ParCSRMatrix)par_matrix);
         
         JX_COGMRESSetup(solver, (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-COGMRES Setup", starttime, endtime, 0, 2);
         }
         
         if (TTest) starttime = jx_MPI_Wtime();
         
         JX_COGMRESSolve(solver, (JX_Matrix)par_matrix, // preOperater
                               (JX_Matrix)par_matrix, (JX_Vector)par_rhs, (JX_Vector)par_sol);
         
         if (TTest)
         {
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PAMG-COGMRES Solve", starttime, endtime, 0, 2);
         }
         
         JX_COGMRESGetNumIterations(solver, &num_iterations);
         JX_COGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
         
         if (print_level == 0 && myid == 0)
         {
            jx_printf(" >>> num_iterations = %d\n", num_iterations);
            jx_printf(" >>> final_res_norm = %.4le\n", final_res_norm);
         }
         
         JX_PAMGDestroy(amg_solver);
         JX_ParCSRCOGMRESDestroy(solver);
      }
      break;
   }
   
   if (TTest)
   {
      endtimeT = jx_MPI_Wtime();
      jx_GetWallTime(comm, "Total Sove Time", starttimeT, endtimeT, 0, 2);
   }
   
   //-----------------------------------------------------------
   //  将近似解向量保存到文件中
   //-----------------------------------------------------------
   if (keepsol)
   {
      jx_Vector *ser_sol = NULL;
      ser_sol = jx_ParVectorToVectorAll(par_sol);
      if (myid == 0)
      {
         jx_sprintf(AppFile, "%s%06d%s%06d%s%03d%s%03d",
            "app_", nprocs, "_", problem_id, "_", solver_id, "_", relax_type);
         jx_SeqVectorPrint(ser_sol, AppFile);
      }
      jx_SeqVectorDestroy(ser_sol);
   }
   
   //-----------------------------------------------------------
   //  销毁并行矩阵和并行向量
   //-----------------------------------------------------------
   jx_ParCSRMatrixDestroy(par_matrix);
   jx_ParVectorDestroy(par_rhs);
   jx_ParVectorDestroy(par_sol);
   
   //-----------------------------------------------------------
   //  终止 MPI
   //-----------------------------------------------------------
   jx_MPI_Finalize();
   
   return 0;
}

/*!
 * \fn JX_Int jx_ParVectorSetRandomValues
 */
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

/*!
 * \fn JX_Int jx_SeqVectorSetRandomValues
 */
JX_Int
jx_SeqVectorSetRandomValues( jx_Vector *x, JX_Int seed )
{
   JX_Real  *vector_data = jx_VectorData(x);
   JX_Int      size        = jx_VectorSize(x);
   JX_Int      i, ierr = 0;

   jx_SeedRand(seed);

   size *= jx_VectorNumVectors(x);

   /* RDF: threading this loop may cause problems because of jx_Rand() */
   for (i = 0; i < size; i ++)
   {
      vector_data[i] = 2.0 * jx_Rand() - 1.0;
   }
   
   return ierr;
}
