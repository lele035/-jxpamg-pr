//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  euclid_dh.c
 *  Date: 2013/01/21
 */

#include "jx_euclid.h"

static void jx_get_runtime_params_private( jx_Euclid_dh ctx );
static void jx_invert_diagonals_private( jx_Euclid_dh ctx );
static void jx_compute_rho_private( jx_Euclid_dh ctx );
static void jx_factor_private( jx_Euclid_dh ctx );
static void jx_reduce_timings_private( jx_Euclid_dh ctx );

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhCreate"
void jx_Euclid_dhCreate( jx_Euclid_dh *ctxOUT )
{
    JX_START_FUNC_DH
    struct _jx_mpi_interface_dh *ctx =
         (struct _jx_mpi_interface_dh *)JX_MALLOC_DH(sizeof(struct _jx_mpi_interface_dh)); JX_CHECK_V_ERROR;
    
   *ctxOUT = ctx;
    ctx->isSetup = jx_false;
    ctx->rho_init = 2.0;
    ctx->rho_final = 0.0;
    ctx->m = 0;
    ctx->n = 0;
    ctx->rhs = NULL;
    ctx->A = NULL;
    ctx->F = NULL;
    ctx->sg = NULL;
    ctx->scale = NULL;
    ctx->isScaled = jx_false;
    ctx->work = NULL;
    ctx->work2 = NULL;
    ctx->from = 0;
    ctx->to = 0;
    strcpy(ctx->algo_par, "pilu");
    strcpy(ctx->algo_ilu, "iluk");
    ctx->level = 1;
    ctx->droptol = JX_DEFAULT_DROP_TOL;
    ctx->sparseTolA = 0.0;
    ctx->sparseTolF = 0.0;
    ctx->pivotMin = 0.0;
    ctx->pivotFix = JX_PIVOT_FIX_DEFAULT;
    ctx->maxVal = 0.0;
    ctx->slist = NULL;
    ctx->extRows = NULL;
    strcpy(ctx->krylovMethod, "bicgstab");
    ctx->maxIts = 200;
    ctx->rtol = 1e-5;
    ctx->atol = 1e-50;
    ctx->its = 0;
    ctx->itsTotal = 0;
    ctx->setupCount = 0;
    ctx->logging = 0;
    ctx->printStats = (jx_Parser_dhHasSwitch(jx_parser_dh, "-printStats"));
    JX_Int i;
    for (i = 0; i < JX_TIMING_BINS; ++ i) ctx->timing[i] = 0.0;
    for (i = 0; i < JX_STATS_BINS; ++ i) ctx->stats[i] = 0.0;
    ctx->timingsWereReduced = jx_false;
    ++ jx_ref_counter;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhDestroy"
void jx_Euclid_dhDestroy( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-eu_stats") || ctx->logging)
    {
        jx_Parser_dhInsert(jx_parser_dh, "-eu_mem", "1"); JX_CHECK_V_ERROR;
        jx_Euclid_dhPrintJxpamgReport(ctx, stdout); JX_CHECK_V_ERROR;
    }
    if (ctx->setupCount > 1 && ctx->printStats)
    {
        jx_Euclid_dhPrintStatsShorter(ctx, stdout); JX_CHECK_V_ERROR;
    }
    if (ctx->F != NULL)
    {
        jx_Factor_dhDestroy(ctx->F); JX_CHECK_V_ERROR;
    }
    if (ctx->sg != NULL)
    {
        jx_SubdomainGraph_dhDestroy(ctx->sg); JX_CHECK_V_ERROR;
    }
    if (ctx->scale != NULL)
    {
        JX_FREE_DH(ctx->scale); JX_CHECK_V_ERROR;
    }
    if (ctx->work != NULL)
    {
        JX_FREE_DH(ctx->work); JX_CHECK_V_ERROR;
    }
    if (ctx->work2 != NULL)
    {
        JX_FREE_DH(ctx->work2); JX_CHECK_V_ERROR;
    }
    if (ctx->slist != NULL)
    {
        jx_SortedList_dhDestroy(ctx->slist); JX_CHECK_V_ERROR;
    }
    if (ctx->extRows != NULL)
    {
        jx_ExternalRows_dhDestroy(ctx->extRows); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(ctx); JX_CHECK_V_ERROR;
    -- jx_ref_counter;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhSetup"
void jx_Euclid_dhSetup( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int m, n, beg_row;
    JX_Real t1;
    jx_bool isSetup = ctx->isSetup;
    jx_bool bj = jx_false;
    
    if (ctx->setupCount && ctx->printStats)
    {
        jx_Euclid_dhPrintStatsShorter(ctx, stdout); JX_CHECK_V_ERROR;
        ctx->its = 0;
    }
    JX_Int i;
    for (i = 0; i < JX_STATS_BINS; ++ i) ctx->stats[i] = 0.0;
    ctx->timing[JX_SOLVE_START_T] = jx_MPI_Wtime();
    ctx->timing[JX_TOTAL_SOLVE_T] += ctx->timing[JX_TOTAL_SOLVE_TEMP_T];
    ctx->timing[JX_TOTAL_SOLVE_TEMP_T] = 0.0;
    if (ctx->F != NULL)
    {
        jx_Factor_dhDestroy(ctx->F); JX_CHECK_V_ERROR;
        ctx->F = NULL;
    }
    if (ctx->A == NULL)
    {
        JX_SET_V_ERROR("must set ctx->A before calling init");
    }
    jx_EuclidGetDimensions(ctx->A, &beg_row, &m, &n); JX_CHECK_V_ERROR;
    ctx->m = m;
    ctx->n = n;
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-print_size"))
    {
        jx_printf_dh("setting up linear system; global rows: %i  local rows: %i (on P_0)\n", n,m);
    }
    jx_sprintf(jx_msgBuf_dh, "localRow= %i;  globalRows= %i;  beg_row= %i", m, n, beg_row);
    JX_SET_INFO(jx_msgBuf_dh);
    bj = jx_Parser_dhHasSwitch(jx_parser_dh, "-bj");
    if (ctx->sg == NULL)
    {
        JX_Int blocks = jx_np_dh;
        t1 = jx_MPI_Wtime();
        if (jx_np_dh == 1)
        {
            jx_Parser_dhReadInt(jx_parser_dh, "-blocks", &blocks); JX_CHECK_V_ERROR;
            jx_SubdomainGraph_dhCreate(&(ctx->sg)); JX_CHECK_V_ERROR;
            jx_SubdomainGraph_dhInit(ctx->sg, blocks, bj, ctx->A); JX_CHECK_V_ERROR;
        }
        else
        {
            jx_SubdomainGraph_dhCreate(&(ctx->sg)); JX_CHECK_V_ERROR;
            jx_SubdomainGraph_dhInit(ctx->sg, -1, bj, ctx->A); JX_CHECK_V_ERROR;
        }
        ctx->timing[JX_SUB_GRAPH_T] += (jx_MPI_Wtime() - t1);
    }
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-doNotFactor"))
    {
        goto END_OF_FUNCTION;
    }
    if (!isSetup)
    {
        jx_get_runtime_params_private(ctx); JX_CHECK_V_ERROR;
    }
    if (!strcmp(ctx->algo_par, "bj")) bj = jx_false;
    if (ctx->scale == NULL)
    {
        ctx->scale = (JX_REAL_DH*)JX_MALLOC_DH(m*sizeof(JX_REAL_DH)); JX_CHECK_V_ERROR;
    }
    for (i = 0; i < m; ++ i) ctx->scale[i] = 1.0;
    if (ctx->work == NULL)
    {
        ctx->work = (JX_REAL_DH*)JX_MALLOC_DH(m*sizeof(JX_REAL_DH)); JX_CHECK_V_ERROR;
    }
    if (ctx->work2 == NULL)
    {
        ctx->work2 = (JX_REAL_DH*)JX_MALLOC_DH(m*sizeof(JX_REAL_DH)); JX_CHECK_V_ERROR;
    }
    t1 = jx_MPI_Wtime();
    jx_factor_private(ctx); JX_CHECK_V_ERROR;
    ctx->timing[JX_FACTOR_T] += (jx_MPI_Wtime() - t1);
    if (strcmp(ctx->algo_par, "none"))
    {
        jx_invert_diagonals_private(ctx); JX_CHECK_V_ERROR;
    }
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-computeRho") || jx_np_dh == 1)
    {
        if (strcmp(ctx->algo_par, "none"))
        {
            t1 = jx_MPI_Wtime();
            jx_compute_rho_private(ctx); JX_CHECK_V_ERROR;
            ctx->timing[JX_COMPUTE_RHO_T] += (jx_MPI_Wtime() - t1);
        }
    }
    if (!strcmp(ctx->algo_par, "pilu") && jx_np_dh > 1)
    {
        t1 = jx_MPI_Wtime();
        jx_Factor_dhSolveSetup(ctx->F, ctx->sg); JX_CHECK_V_ERROR;
        ctx->timing[JX_SOLVE_SETUP_T] += (jx_MPI_Wtime() - t1);
    }
    
END_OF_FUNCTION: ;
    
    ctx->timing[JX_SETUP_T] += (jx_MPI_Wtime() - ctx->timing[JX_SOLVE_START_T]);
    ctx->setupCount += 1;
    ctx->isSetup = jx_true;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_get_runtime_params_private"
void jx_get_runtime_params_private( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    char *tmp;
    
    jx_Parser_dhReadInt(jx_parser_dh, "-maxIts",&(ctx->maxIts));
    jx_Parser_dhReadDouble(jx_parser_dh, "-rtol", &(ctx->rtol));
    jx_Parser_dhReadDouble(jx_parser_dh, "-atol", &(ctx->atol));
    tmp = NULL;
    jx_Parser_dhReadString(jx_parser_dh, "-par", &tmp);
    if (tmp != NULL)
    {
        strcpy(ctx->algo_par, tmp);
    }
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-bj"))
    {
        strcpy(ctx->algo_par, "bj");
    }
    jx_Parser_dhReadDouble(jx_parser_dh, "-rho", &(ctx->rho_init));
    jx_Parser_dhReadInt(jx_parser_dh, "-level", &ctx->level);
    jx_Parser_dhReadInt(jx_parser_dh, "-pc_ilu_levels", &ctx->level);
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-ilut"))
    {
        jx_Parser_dhReadDouble(jx_parser_dh, "-ilut", &ctx->droptol);
        ctx->isScaled = jx_true;
        strcpy(ctx->algo_ilu, "ilut");
    }
    if (!strcmp(ctx->algo_par, "none"))
    {
        strcpy(ctx->algo_ilu, "none");
    }
    else if (!strcmp(ctx->algo_ilu, "none"))
    {
        strcpy(ctx->algo_par, "none");
    }
    jx_Parser_dhReadDouble(jx_parser_dh, "-sparseA",&(ctx->sparseTolA));
    jx_Parser_dhReadDouble(jx_parser_dh, "-sparseF",&(ctx->sparseTolF));
    jx_Parser_dhReadDouble(jx_parser_dh, "-pivotMin", &(ctx->pivotMin));
    jx_Parser_dhReadDouble(jx_parser_dh, "-pivotFix", &(ctx->pivotFix));
    if (ctx->sparseTolA || ! strcmp(ctx->algo_ilu, "ilut"))
    {
        ctx->isScaled = jx_true;
    }
    tmp = NULL;
    jx_Parser_dhReadString(jx_parser_dh, "-ksp_type", &tmp);
    if (tmp != NULL)
    {
        strcpy(ctx->krylovMethod, tmp);
        if (!strcmp(ctx->krylovMethod, "bcgs"))
        {
            strcpy(ctx->krylovMethod, "bicgstab");
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_invert_diagonals_private"
void jx_invert_diagonals_private( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_REAL_DH *aval = ctx->F->aval;
    JX_Int *diag = ctx->F->diag;
    
    if (aval == NULL || diag == NULL)
    {
        JX_SET_INFO("can't invert diags; either F->aval or F->diag is NULL");
    }
    else
    {
        JX_Int i, m = ctx->F->m;
        for (i = 0; i < m; ++ i)
        {
            aval[diag[i]] = 1.0 / aval[diag[i]];
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_compute_rho_private"
void jx_compute_rho_private( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    if (ctx->F != NULL)
    {
        JX_Real bufLocal[3], bufGlobal[3];
        JX_Int m = ctx->m;
        
        ctx->stats[JX_NZF_STATS] = (JX_Real)ctx->F->rp[m];
        bufLocal[0] = ctx->stats[JX_NZA_STATS];
        bufLocal[1] = ctx->stats[JX_NZF_STATS];
        bufLocal[2] = ctx->stats[JX_NZA_USED_STATS];
        if (jx_np_dh == 1)
        {
            bufGlobal[0] = bufLocal[0];
            bufGlobal[1] = bufLocal[1];
            bufGlobal[2] = bufLocal[2];
        }
        else
        {
            jx_MPI_Reduce(bufLocal, bufGlobal, 3, JX_MPI_REAL, MPI_SUM, 0, jx_comm_dh);
        }
        if (jx_myid_dh == 0)
        {
            if (bufGlobal[0] && bufGlobal[1])
            {
                ctx->rho_final = bufGlobal[1] / bufGlobal[0];
            }
            else
            {
                ctx->rho_final = -1;
            }
            if (bufGlobal[0] && bufGlobal[2])
            {
                ctx->stats[JX_NZA_RATIO_STATS] = 100.0 * bufGlobal[2] / bufGlobal[0];
            }
            else
            {
                ctx->stats[JX_NZA_RATIO_STATS] = 100.0;
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_factor_private"
void jx_factor_private( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    if (!strcmp(ctx->algo_par, "none"))
    {
        goto DO_NOTHING;
    }
    JX_Int br = 0;
    JX_Int id = jx_np_dh;
    if (ctx->sg != NULL)
    {
        br = ctx->sg->beg_rowP[jx_myid_dh];
        id = ctx->sg->o2n_sub[jx_myid_dh];
    }
    jx_Factor_dhInit(ctx->A, jx_true, jx_true, ctx->rho_init, id, br, &(ctx->F)); JX_CHECK_V_ERROR;
    ctx->F->bdry_count = ctx->sg->bdry_count[jx_myid_dh];
    ctx->F->first_bdry = ctx->F->m - ctx->F->bdry_count;
    if (!strcmp(ctx->algo_par, "bj")) ctx->F->blockJacobi = jx_true;
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-bj")) ctx->F->blockJacobi = jx_true;
    if (jx_np_dh == 1)
    {
        if (!strcmp(ctx->algo_ilu, "iluk"))
        {
            ctx->from = 0;
            ctx->to = ctx->m;
            if (jx_Parser_dhHasSwitch(jx_parser_dh, "-mpi"))
            {
                if (ctx->sg != NULL && ctx->sg->blocks > 1)
                {
                    JX_SET_V_ERROR("only use -mpi, which invokes jx_ilu_mpi_pilu(), for np = 1 and -blocks 1");
                }
                jx_iluk_mpi_pilu(ctx); JX_CHECK_V_ERROR;
            }
            else
            {
                jx_iluk_seq_block(ctx); JX_CHECK_V_ERROR;
            }
        }
        else if (!strcmp(ctx->algo_ilu, "ilut"))
        {
            ctx->from = 0;
            ctx->to = ctx->m;
            jx_ilut_seq(ctx); JX_CHECK_V_ERROR;
        }
        else
        {
            jx_sprintf(jx_msgBuf_dh, "factorization method: %s is not implemented", ctx->algo_ilu);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    else
    {
        if (!strcmp(ctx->algo_par, "bj"))
        {
            ctx->from = 0;
            ctx->to = ctx->m;
            jx_iluk_mpi_bj(ctx); JX_CHECK_V_ERROR;
        }
        else if (!strcmp(ctx->algo_ilu, "iluk"))
        {
            jx_bool bj = ctx->F->blockJacobi;
            jx_SortedList_dhCreate(&(ctx->slist)); JX_CHECK_V_ERROR;
            jx_SortedList_dhInit(ctx->slist, ctx->sg); JX_CHECK_V_ERROR;
            jx_ExternalRows_dhCreate(&(ctx->extRows)); JX_CHECK_V_ERROR;
            jx_ExternalRows_dhInit(ctx->extRows, ctx); JX_CHECK_V_ERROR;
            ctx->from = 0;
            ctx->to = ctx->F->first_bdry;
            jx_iluk_seq(ctx); JX_CHECK_V_ERROR;
            if (!bj)
            {
                jx_ExternalRows_dhRecvRows(ctx->extRows); JX_CHECK_V_ERROR;
            }
            ctx->from = ctx->F->first_bdry;
            ctx->to = ctx->F->m;
            jx_iluk_mpi_pilu(ctx); JX_CHECK_V_ERROR;
            if (!bj)
            {
                jx_ExternalRows_dhSendRows(ctx->extRows); JX_CHECK_V_ERROR;
            }
            jx_SortedList_dhDestroy(ctx->slist); JX_CHECK_V_ERROR;
            ctx->slist = NULL;
            jx_ExternalRows_dhDestroy(ctx->extRows); JX_CHECK_V_ERROR;
            ctx->extRows = NULL;
        }
        else
        {
            jx_sprintf(jx_msgBuf_dh, "factorization method: %s is not implemented", ctx->algo_ilu);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    
DO_NOTHING: ;
    
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhSolve"
void jx_Euclid_dhSolve( jx_Euclid_dh ctx, jx_Vec_dh x, jx_Vec_dh b, JX_Int *its )
{
    JX_START_FUNC_DH
    JX_Int itsOUT;
    jx_Mat_dh A = (jx_Mat_dh)ctx->A;
    
    if (!strcmp(ctx->krylovMethod, "cg"))
    {
        jx_cg_euclid(A, ctx, x->vals, b->vals, &itsOUT); JX_ERRCHKA;
    }
    else if (!strcmp(ctx->krylovMethod, "bicgstab"))
    {
        jx_bicgstab_euclid(A, ctx, x->vals, b->vals, &itsOUT); JX_ERRCHKA;
    }
    else
    {
        jx_sprintf(jx_msgBuf_dh, "unknown krylov solver: %s", ctx->krylovMethod);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
   *its = itsOUT;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhPrintStats"
void jx_Euclid_dhPrintStats( jx_Euclid_dh ctx, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Real *timing;
    JX_Int nz;
    
    nz = jx_Factor_dhReadNz(ctx->F); JX_CHECK_V_ERROR;
    timing = ctx->timing;
    ctx->timing[JX_TOTAL_SOLVE_T] += ctx->timing[JX_TOTAL_SOLVE_TEMP_T];
    ctx->timing[JX_TOTAL_SOLVE_TEMP_T] = 0.0;
    jx_reduce_timings_private(ctx); JX_CHECK_V_ERROR;
    jx_fprintf_dh(fp, "\n==================== Euclid report (start) ====================\n");
    jx_fprintf_dh(fp, "\nruntime parameters\n");
    jx_fprintf_dh(fp, "------------------\n");
    jx_fprintf_dh(fp, "   setups:                 %i\n", ctx->setupCount);
    jx_fprintf_dh(fp, "   tri solves:             %i\n", ctx->itsTotal);
    jx_fprintf_dh(fp, "   parallelization method: %s\n", ctx->algo_par);
    jx_fprintf_dh(fp, "   factorization method:   %s\n", ctx->algo_ilu);
    jx_fprintf_dh(fp, "   matrix was row scaled:  %i\n", ctx->isScaled);
    jx_fprintf_dh(fp, "   matrix row count:       %i\n", ctx->n);
    jx_fprintf_dh(fp, "   nzF:                    %i\n", nz);
    jx_fprintf_dh(fp, "   rho:                    %g\n", ctx->rho_final);
    jx_fprintf_dh(fp, "   level:                  %i\n", ctx->level);
    jx_fprintf_dh(fp, "   sparseA:                %g\n", ctx->sparseTolA);
    jx_fprintf_dh(fp, "\nEuclid timing report\n");
    jx_fprintf_dh(fp, "--------------------\n");
    jx_fprintf_dh(fp, "   solves total:  %0.2f (see docs)\n", timing[JX_TOTAL_SOLVE_T]);
    jx_fprintf_dh(fp, "   tri solves:    %0.2f\n", timing[JX_TRI_SOLVE_T]);
    jx_fprintf_dh(fp, "   setups:        %0.2f\n", timing[JX_SETUP_T]);
    jx_fprintf_dh(fp, "      subdomain graph setup:  %0.2f\n", timing[JX_SUB_GRAPH_T]);
    jx_fprintf_dh(fp, "      factorization:          %0.2f\n", timing[JX_FACTOR_T]);
    jx_fprintf_dh(fp, "      solve setup:            %0.2f\n", timing[JX_SOLVE_SETUP_T]);
    jx_fprintf_dh(fp, "      rho:                    %0.2f\n", ctx->timing[JX_COMPUTE_RHO_T]);
    jx_fprintf_dh(fp, "      misc (should be small): %0.2f\n", 
           timing[JX_SETUP_T]-(timing[JX_SUB_GRAPH_T]+timing[JX_FACTOR_T]+timing[JX_SOLVE_SETUP_T]+timing[JX_COMPUTE_RHO_T]));
    if (ctx->sg != NULL)
    {
        jx_SubdomainGraph_dhPrintStats(ctx->sg, fp); JX_CHECK_V_ERROR;
        jx_SubdomainGraph_dhPrintRatios(ctx->sg, fp); JX_CHECK_V_ERROR;
    }
    jx_fprintf_dh(fp, "\nApplicable if Euclid's internal solvers were used:\n");
    jx_fprintf_dh(fp, "---------------------------------------------------\n");
    jx_fprintf_dh(fp, "   solve method: %s\n", ctx->krylovMethod);
    jx_fprintf_dh(fp, "   maxIts:       %i\n", ctx->maxIts);
    jx_fprintf_dh(fp, "   rtol:         %g\n", ctx->rtol);
    jx_fprintf_dh(fp, "   atol:         %g\n", ctx->atol);
    jx_fprintf_dh(fp, "\n==================== Euclid report (end) ======================\n");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhPrintStatsShort"
void jx_Euclid_dhPrintStatsShort( jx_Euclid_dh ctx, JX_Real setup, JX_Real solve, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Real *timing = ctx->timing;
    JX_Real apply_total;
    JX_Real apply_per_it;
    JX_Real perIt;
    JX_Int blocks = jx_np_dh;
    
    if (jx_np_dh == 1) blocks = ctx->sg->blocks;
    jx_reduce_timings_private(ctx); JX_CHECK_V_ERROR;
    apply_total = timing[JX_TRI_SOLVE_T];
    apply_per_it = apply_total/(JX_Real)ctx->its;
    perIt = solve / (JX_Real)ctx->its;
    jx_fprintf_dh(fp, "\n");
    jx_fprintf_dh(fp, "%6s %6s %6s %6s %6s %6s %6s %6s %6s %6s XX\n",
                   "method", "subdms", "level", "its", "setup", "solve", "total", "perIt", "perIt", "rows");
    jx_fprintf_dh(fp, "------  -----  -----  -----  -----  -----  -----  -----  -----  -----  XX\n");
    jx_fprintf_dh(fp, "%6s %6i %6i %6i %6.2f %6.2f %6.2f %6.4f %6.5f %6g  XXX\n",
      ctx->algo_par, blocks, ctx->level, ctx->its, setup, solve, setup+solve, perIt, apply_per_it, (JX_Real)ctx->n);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhPrintStatsShorter"
void jx_Euclid_dhPrintStatsShorter( jx_Euclid_dh ctx, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Real *stats = ctx->stats;
    JX_Int its = ctx->its;
    JX_Real rho = ctx->rho_final;
    JX_Real nzUsedRatio = stats[JX_NZA_RATIO_STATS];
    
    jx_fprintf_dh(fp, "\nStats from last linear solve: YY\n");
    jx_fprintf_dh(fp, "%6s %6s %6s     YY\n", "its", "rho","A_%");
    jx_fprintf_dh(fp, " -----  -----  -----     YY\n");
    jx_fprintf_dh(fp, "%6i %6.2f %6.2f     YYY\n", its, rho, nzUsedRatio);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhPrintScaling"
void jx_Euclid_dhPrintScaling( jx_Euclid_dh ctx, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i, m = ctx->m;
    
    if (m > 10) m = 10;
    if (ctx->scale == NULL)
    {
        JX_SET_V_ERROR("ctx->scale is NULL; was jx_Euclid_dhSetup() called?");
    }
    jx_fprintf(fp, "\n---------- 1st %i row scaling values:\n", m);
    for (i = 0; i < m; ++ i)
    {
        jx_fprintf(fp, "   %i  %g  \n", i+1, ctx->scale[i]);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_reduce_timings_private"
void jx_reduce_timings_private( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    if (jx_np_dh > 1)
    {
        JX_Real bufOUT[JX_TIMING_BINS];
        
        memcpy(bufOUT, ctx->timing, JX_TIMING_BINS*sizeof(JX_Real));
        jx_MPI_Reduce(bufOUT, ctx->timing, JX_TIMING_BINS, JX_MPI_REAL, MPI_MAX, 0, jx_comm_dh);
    }
    ctx->timingsWereReduced = jx_true;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhPrintJxpamgReport"
void jx_Euclid_dhPrintJxpamgReport( jx_Euclid_dh ctx, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Real *timing;
    JX_Int nz;
    
    nz = jx_Factor_dhReadNz(ctx->F); JX_CHECK_V_ERROR;
    timing = ctx->timing;
    ctx->timing[JX_TOTAL_SOLVE_T] += ctx->timing[JX_TOTAL_SOLVE_TEMP_T];
    ctx->timing[JX_TOTAL_SOLVE_TEMP_T] = 0.0;
    jx_reduce_timings_private(ctx); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        jx_fprintf(fp, "@@@@@@@@@@@@@@@@@@@@@@ Euclid statistical report (start)\n");
        jx_fprintf_dh(fp, "\nruntime parameters\n");
        jx_fprintf_dh(fp, "------------------\n");
        jx_fprintf_dh(fp, "   setups:                 %i\n", ctx->setupCount);
        jx_fprintf_dh(fp, "   tri solves:             %i\n", ctx->itsTotal);
        jx_fprintf_dh(fp, "   parallelization method: %s\n", ctx->algo_par);
        jx_fprintf_dh(fp, "   factorization method:   %s\n", ctx->algo_ilu);
        if (!strcmp(ctx->algo_ilu, "iluk"))
        {
            jx_fprintf_dh(fp, "      level:               %i\n", ctx->level);
        }
        if (ctx->isScaled)
        {
            jx_fprintf_dh(fp, "   matrix was row scaled\n");
        }
        jx_fprintf_dh(fp, "   global matrix row count: %i\n", ctx->n);
        jx_fprintf_dh(fp, "   nzF:                     %i\n", nz);
        jx_fprintf_dh(fp, "   rho:                     %g\n", ctx->rho_final);
        jx_fprintf_dh(fp, "   sparseA:                 %g\n", ctx->sparseTolA);
        jx_fprintf_dh(fp, "\nEuclid timing report\n");
        jx_fprintf_dh(fp, "--------------------\n");
        jx_fprintf_dh(fp, "   solves total:  %0.2f (see docs)\n", timing[JX_TOTAL_SOLVE_T]);
        jx_fprintf_dh(fp, "   tri solves:    %0.2f\n", timing[JX_TRI_SOLVE_T]);
        jx_fprintf_dh(fp, "   setups:        %0.2f\n", timing[JX_SETUP_T]);
        jx_fprintf_dh(fp, "      subdomain graph setup:  %0.2f\n", timing[JX_SUB_GRAPH_T]);
        jx_fprintf_dh(fp, "      factorization:          %0.2f\n", timing[JX_FACTOR_T]);
        jx_fprintf_dh(fp, "      solve setup:            %0.2f\n", timing[JX_SOLVE_SETUP_T]);
        jx_fprintf_dh(fp, "      rho:                    %0.2f\n", ctx->timing[JX_COMPUTE_RHO_T]);
        jx_fprintf_dh(fp, "      misc (should be small): %0.2f\n", 
           timing[JX_SETUP_T]-(timing[JX_SUB_GRAPH_T]+timing[JX_FACTOR_T]+timing[JX_SOLVE_SETUP_T]+timing[JX_COMPUTE_RHO_T]));
        if (ctx->sg != NULL)
        {
            jx_SubdomainGraph_dhPrintStats(ctx->sg, fp); JX_CHECK_V_ERROR;
            jx_SubdomainGraph_dhPrintRatios(ctx->sg, fp); JX_CHECK_V_ERROR;
        }
        jx_fprintf(fp, "@@@@@@@@@@@@@@@@@@@@@@ Euclid statistical report (end)\n");
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhPrintTestData"
void jx_Euclid_dhPrintTestData( jx_Euclid_dh ctx, FILE *fp )
{
    JX_START_FUNC_DH
    if (jx_myid_dh == 0)
    {
        jx_fprintf(fp, "   setups:                 %i\n", ctx->setupCount);
        jx_fprintf(fp, "   tri solves:             %i\n", ctx->its);
        jx_fprintf(fp, "   parallelization method: %s\n", ctx->algo_par);
        jx_fprintf(fp, "   factorization method:   %s\n", ctx->algo_ilu);
        jx_fprintf(fp, "   level:                  %i\n", ctx->level);
        jx_fprintf(fp, "   row scaling:            %i\n", ctx->isScaled);
    }
    jx_SubdomainGraph_dhPrintRatios(ctx->sg, fp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}
