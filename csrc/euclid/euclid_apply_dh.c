//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  euclid_apply_dh.c
 *  Date: 2013/01/21
 */

#include "jx_euclid.h"

static void jx_scale_rhs_private( jx_Euclid_dh ctx, JX_Real *rhs );
static void jx_permute_vec_n2o_private( jx_Euclid_dh ctx, JX_Real *xIN, JX_Real *xOUT );
static void jx_permute_vec_o2n_private( jx_Euclid_dh ctx, JX_Real *xIN, JX_Real *xOUT );

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhApply"
void jx_Euclid_dhApply( jx_Euclid_dh ctx, JX_Real *rhs, JX_Real *lhs )
{
    JX_START_FUNC_DH
    JX_Real *rhs_, *lhs_;
    JX_Real t1, t2;
    
    t1 = jx_MPI_Wtime();
    ctx->from = 0;
    ctx->to = ctx->m;
    if (!strcmp(ctx->algo_ilu, "none") || !strcmp(ctx->algo_par, "none"))
    {
        JX_Int i, m = ctx->m;
        for (i = 0; i < m; ++ i) lhs[i] = rhs[i];
        goto END_OF_FUNCTION;
    }
    if (ctx->sg != NULL)
    {
        jx_permute_vec_n2o_private(ctx, rhs, lhs); JX_CHECK_V_ERROR;
        rhs_ = lhs;
        lhs_ = ctx->work2;
    }
    else
    {
        rhs_ = rhs;
        lhs_ = lhs;
    }
    if (ctx->isScaled)
    {
        jx_scale_rhs_private(ctx, rhs_); JX_CHECK_V_ERROR;
    }
    if (jx_np_dh == 1 || !strcmp(ctx->algo_par, "bj"))
    {
        jx_Factor_dhSolveSeq(rhs_, lhs_, ctx); JX_CHECK_V_ERROR;
    }
    else
    {
        jx_Factor_dhSolve(rhs_, lhs_, ctx); JX_CHECK_V_ERROR;
    }
    if (ctx->sg != NULL)
    {
        jx_permute_vec_o2n_private(ctx, lhs_, lhs); JX_CHECK_V_ERROR;
    }
    
END_OF_FUNCTION: ;
    
    t2 = jx_MPI_Wtime();
    ctx->timing[JX_TRI_SOLVE_T] += (t2 - t1);
    ctx->timing[JX_TOTAL_SOLVE_TEMP_T] = t2 - ctx->timing[JX_SOLVE_START_T];
    ctx->its += 1;
    ctx->itsTotal += 1;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_scale_rhs_private"
void jx_scale_rhs_private( jx_Euclid_dh ctx, JX_Real *rhs )
{
    JX_START_FUNC_DH
    JX_Int i, m = ctx->m;
    JX_REAL_DH *scale = ctx->scale;
    
    if (scale != NULL)
    {
#ifdef USING_OPENMP_DH
#pragma omp for schedule(static)
#endif
        for (i = 0; i < m; ++ i)
        {
            rhs[i] *= scale[i];
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_permute_vec_o2n_private"
void jx_permute_vec_o2n_private( jx_Euclid_dh ctx, JX_Real *xIN, JX_Real *xOUT )
{
    JX_START_FUNC_DH
    JX_Int i, m = ctx->m;
    JX_Int *o2n = ctx->sg->o2n_col;
    
    for (i = 0; i < m; ++ i) xOUT[i] = xIN[o2n[i]];
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_permute_vec_n2o_private"
void jx_permute_vec_n2o_private( jx_Euclid_dh ctx, JX_Real *xIN, JX_Real *xOUT )
{
    JX_START_FUNC_DH
    JX_Int i, m = ctx->m;
    JX_Int *n2o = ctx->sg->n2o_row;
    
    for (i = 0; i < m; ++ i) xOUT[i] = xIN[n2o[i]];
    JX_END_FUNC_DH
}
