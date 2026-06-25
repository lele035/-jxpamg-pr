/*
This is the reference code of SolverChallenge to optimize by the participants
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang, Haocheng Lian

Added some parallel packages: jxpamg
@Update date: 2021/5/28
@Author: Li Zhao
*/

typedef long double Real;
typedef int Int;
// typedef long long int Int;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h> 
#include <sys/resource.h>

#include "jx_pamg.h"
#include "mmio_highlevel.h"
#include "my_solver.h"
#include "mpi.h"


Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *y, Real *b, Real *Res);
void store_x(Int n, Real *x, char *filename);
void load_b(Int n, Real *b, char *filename);
void load_x(int n, Real *x, char *filename);

Int main(Int argc, char **argv)
{
    Int     m, n, nnzA, isSymmetricA;
    Int     *row_ptr; // the csr row pointer array of matrix A
    Int     *col_idx; // the csr column index array of matrix A
    Real  *val;  // the csr value array of matrix A
    Real  *x, *b; 
    Real  *y; 
    Real    norm_b;

    // 在文件开头定义两个新变量，用于存储第二个文件的名称
    char *filename_matrix2 = NULL;
    char *filename_b2      = NULL;
    char *filename_x2      = NULL;

    char    *filename_matrix = argv[1];  // the filename of matrix A
    char    *filename_b = argv[2];       // the filename of right-hand side vector b
    // char    *filename_x = argv[3];       // the filename of solution vector x
    char    *matrix_tolerance = argv[3]; // the tolerance of input matrix
    // char    *sid = argv[5];              // solver id
    Int     iter = 0;                    // the number of iterations
    Real  residule = 0.0, RRN, RES;
    Real  tolerance = atof(matrix_tolerance);
    Real  coff = 1;

    int     rank;
    int     size;
    Int     i;
    // Int     solver_id   = atoi(sid);
    Int     solver_id   = 12;
    Int     precond_id  = 3;
    Int     num_functions = 1;
    Int     max_iter = 500;
    
    // system("free -h");
    MPI_Init(NULL,NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    
    //-----------------------
    //  命令行修改参数
    //-----------------------
    Int arg_index = 1;
    while (arg_index < argc)
    {
        if ( strcmp(argv[arg_index], "-sid") == 0 )
        {
            arg_index ++;
            solver_id = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-maxiter") == 0 )
        {
            arg_index ++;
            max_iter = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nf") == 0 )
        {
            arg_index ++;
            num_functions = atoi(argv[arg_index++]);
        }
            /* 新增：第二个矩阵、右端项、初值文件名 */
        else if ( strcmp(argv[arg_index], "-mat2") == 0 )
        {
            arg_index ++;
            filename_matrix2 = argv[arg_index++];
        }
        else if ( strcmp(argv[arg_index], "-rhs2") == 0 )
        {
            arg_index ++;
            filename_b2 = argv[arg_index++];
        }
        else if ( strcmp(argv[arg_index], "-x2") == 0 )
        {
            arg_index ++;
            filename_x2 = argv[arg_index++];
        }
        else{
            arg_index ++;
        }
    }

    if (rank == 0)
    {
        printf("===================================================================\n");
        printf("A  : %s\n", filename_matrix);
        printf("b  : %s\n", filename_b);
        // printf("x  : %s\n", filename_x);
        if (filename_matrix2)
            printf("A2 : %s\n", filename_matrix2);
        if (filename_b2)
            printf("b2 : %s\n", filename_b2);
        // if (filename_x2)
        //     printf("x2 : %s\n", filename_x2);
        printf("tol: %s\n", matrix_tolerance);
        printf("sid: %d\n", solver_id);
        printf("nf : %d\n", num_functions);
        printf("===================================================================\n");

        printf("\n\n+++++++++++++++++++++ MPI using %d processors +++++++++++++++++++++\n\n", size);

        //load matrix
        // printf("load matrix:\n");
        mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
        //mmio_allinone_test(&m, &n, &nnzA,  &row_ptr, &col_idx, &val, filename_matrix);
        // printf("matrix info: m: %lld, n: %lld, nnz: %lld, isSymmertirc: %lld\n", m, n, nnzA, isSymmetricA);
        
        if (m != n)
        {
            printf("Invalid matrix size.\n");
            return 0;
        }

        x = (Real *)malloc(sizeof(Real) * n);
        y = (Real *)malloc(sizeof(Real) * m);
        b = (Real *)malloc(sizeof(Real) * m);

        // load right-hand side vector b
        // printf("load right-hand side vector:\n");
        load_b(n, b, filename_b);

        norm_b = vec2norm(b, n);
        // printf("the norm of b is %Lf\n", norm_b);
        // load inital guess vector x
        // printf("load inital guess vector x:\n");
        // load_x(n, x, filename_x);
        
        for (i = 0; i < n; i++) x[i] = 0; // 初始化为 0 

        spmv(n, row_ptr, col_idx, val, x, y);  // y=A*x_0  
        Int i;
        Real *r0 = (Real *)malloc(sizeof(Real) * n);
        for (i = 0; i < m; i++)
        {
            r0[i] = y[i] - b[i];
        }
        Real r0_L2 = vec2norm(r0, n);    // r0_L2
        // printf("the norm of r0 is %Lf\n", r0_L2);
    }

static inline void mem_usage() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    long long int mymem = r_usage.ru_maxrss;
    long long int total;
    MPI_Allreduce(&mymem, &total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0) {
        fprintf(stdout, "Total memory usage: %.2f MB\n", (double)((double)(total) / (double)(1024)));
        fflush(stdout);
    }
}
        
    JX_Solver shared_amg = NULL;   // 共享的 AMG 数据结构
    Int ret = 0;
    
    // 求解 u 方程（第一次调用，shared_amg == NULL，会新建 AMG）
    mem_usage();
    // solve x and record wall-time
    struct timeval t_start, t_stop;
    gettimeofday(&t_start, NULL);
    // if (rank == 0) printf("\nJXPAMG_Solver_Interface(int, long double -- version):\n");    
    if (JXPAMG_Solver_Interface(argc, argv, n, row_ptr, col_idx, val, x, b, tolerance, solver_id, num_functions, &iter, &shared_amg) != 0) {
        if (rank == 0) printf("First solve FAILED\n");
        ret = 1;
        goto cleanup;
    }
    gettimeofday(&t_stop, NULL);
    Real total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
    mem_usage();

    if (rank == 0)
    {
        FILE *fp1 = fopen("x_org.txt", "w");

        if (fp1 == NULL) {
            printf("文件打开失败！\n");
            return 1;
        }
        
        for (int i = 0; i < n; i++) {
            fprintf(fp1,"%.15Le\n", x[i]);
        }
        
        // 关闭文件
        fclose(fp1);
        //store x to a file
        // store_x(n, x, filename_x);
        // store_x(n, x, "x.txt");

        //check the correctness
        //print the #iteration, residual and total_time
        RRN = check_correctness(n, row_ptr, col_idx, val, x, y, b, &residule);
        printf("\n\n===============================================================================\n");
        // printf("iteration = %d, residual = %Le, RRN = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
        printf("iteration = %d, residual = %Le, RES = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
        
        //printf("iteration = %d, residual = %Le, RRN = %Le, time_CPU=%f sec\n", iter, residule, RRN, time_CPU);
        printf("===============================================================================\n");
            // 释放第一个矩阵的数据（但保留 AMG 数据结构 shared_amg）
        free(row_ptr);
        free(col_idx);
        free(val);
        free(x);
        free(y);
        free(b);
    }

    // ----------------- 第二次求解（第二个变量）-----------------
    if (filename_matrix2 != NULL && filename_b2 != NULL )
    {
        if (rank == 0)
        {

            printf("===================================================================\n");
            if (filename_matrix2)
                printf("A2 : %s\n", filename_matrix2);
            if (filename_b2)
                printf("b2 : %s\n", filename_b2);
            printf("tol: %s\n", matrix_tolerance);
            printf("sid: %d\n", solver_id);
            printf("===================================================================\n");

            // 读取第二个矩阵、右端项、初值
            mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix2);
            if (m != n) { /* 错误处理 */ }
            x = (Real *)malloc(sizeof(Real) * n);
            y = (Real *)malloc(sizeof(Real) * m);
            b = (Real *)malloc(sizeof(Real) * m);
            load_b(n, b, filename_b2);
            // load_x(n, x, filename_x2);
            for (i = 0; i < n; i++) x[i] = 0; // 初始化为 0 
        }

    
        // // 求解 v 方程（第二次调用，shared_amg != NULL，会复用）
        mem_usage();
        gettimeofday(&t_start, NULL);
        if (rank == 0) printf("\nJXPAMG_Solver_Interface for another system with last amg_setup:\n");    
            // 再次调用求解器接口，传入同一个 shared_amg 地址（此时非 NULL，内部将复用已有 AMG setup）
        if (JXPAMG_Solver_Interface(argc, argv, n, row_ptr, col_idx, val, x, b,
                                tolerance, solver_id, num_functions, &iter,
                                &shared_amg) != 0) {
            if (rank == 0) printf("Second solve FAILED\n");
            ret = 1;
            goto cleanup;
         }
        gettimeofday(&t_stop, NULL);
        Real total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
        mem_usage();

        if (rank == 0)
        {
            //store x to a file
            // store_x(n, x, filename_x);
            // store_x(n, x, "x.txt");

            //check the correctness
            //print the #iteration, residual and total_time
            RRN = check_correctness(n, row_ptr, col_idx, val, x, y, b, &residule);
            printf("\n\n===============================================================================\n");
            // printf("iteration = %d, residual = %Le, RRN = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
            printf("iteration = %d, residual = %Le, RES = %Le, total_time = %.5Lf sec\n", iter, residule, RRN, total_time / 1000.0);
            //printf("iteration = %d, residual = %Le, RRN = %Le, time_CPU=%f sec\n", iter, residule, RRN, time_CPU);
            printf("===============================================================================\n");
                // 释放第一个矩阵的数据（但保留 AMG 数据结构 shared_amg）
            free(row_ptr);
            free(col_idx);
            free(val);
            free(x);
            free(y);
            free(b);
        }
    }
    ///
cleanup:
    if (shared_amg) JX_PAMGDestroy(shared_amg);
    MPI_Finalize();
    return ret;
}

//validate the x by b-A*x
Real check_correctness(Int n, Int *row_ptr, Int *col_idx, Real *val, Real *x, Real *y, Real *b, Real *Res)
{
    Real *b_new = (Real *)malloc(sizeof(Real) * n);
    Real *check_b = (Real *)malloc(sizeof(Real) * n);
    Real *r0 = (Real *)malloc(sizeof(Real) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);  // b_new=A*x_out
    Int i;
    for (i = 0; i < n; i++)
    {
        check_b[i] = b_new[i] - b[i];       // check_b = r
        r0[i] = y[i] - b[i];
    }   
    *Res =  vec2norm(check_b, n);   // Res = rnorm
    Real r0_L2 = vec2norm(r0, n);    // r0_L2
    Real b_L2 = vec2norm(b, n);     // b_L2 = bnorm
    // printf("\nZL: ||b-Ax||_2: %Le, ||b||_2: %Le\n", vec2norm(check_b, n), b_L2);
    
    return *Res / b_L2;
    // return *Res / r0_L2;
}

//store x to a file
void store_x(Int n, Real *x, char *filename)
{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%lld\n", n);
    Int i;
    for (i = 0; i < n; i++)
        fprintf(p, "%Lf\n", x[i]);
    fclose(p);
}

//load right-hand side vector b
void load_b(Int n, Real *b, char *filename)
{
    FILE *p = fopen(filename, "r");
    Int n_right;
    Int r = fscanf(p, "%lld", &n_right);
    // printf("n_right: %lld, n: %lld.\n", n_right, n);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    Int i;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, "%Lf", &b[i]);
    fclose(p);
}


// load x 开发原子大赛
void load_x(int n, Real *x, char *filename)
{
    FILE *p = fopen(filename, "r");
    int n_right;
    int r = fscanf(p, "%d", &n_right);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of x.\n");
        return;
    }
    int i;
    int number;
    double ix_tmp, iy_tmp, iz_tmp;
    for (i = 0; i < n_right; i++)
        r = fscanf(p, " %d %Lf %lf %lf %lf", &number, &x[i], &ix_tmp, &iy_tmp, &iz_tmp);
    fclose(p);
}