//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  krylov_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_bicgstab_euclid"
void jx_bicgstab_euclid( jx_Mat_dh A, jx_Euclid_dh ctx, JX_Real *x, JX_Real *b, JX_Int *itsOUT )
{
    JX_START_FUNC_DH
    JX_Int its, m = ctx->m;
    jx_bool monitor;
    JX_Int maxIts = ctx->maxIts;
    JX_Real atol = ctx->atol, rtol = ctx->rtol;
    JX_Real alpha, alpha_1 = 0., beta_1, widget, widget_1 = 0.;
    JX_Real rho_1, rho_2 = 0., s_norm, eps, exit_a, b_iprod, r_iprod;
    JX_Real *t, *s, *s_hat, *v, *p, *p_hat, *r, *r_hat;
    
    monitor = jx_Parser_dhHasSwitch(jx_parser_dh, "-monitor");
    t = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    s = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    s_hat = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    v = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    p = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    p_hat = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    r = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    r_hat = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    jx_Mat_dhMatVec(A, x, s);
    jx_CopyVec(m, b, r);
    jx_Axpy(m, -1.0, s, r);
    jx_CopyVec(m, r, r_hat);
    b_iprod = jx_InnerProd(m, b, b); JX_CHECK_V_ERROR;
    exit_a = atol*atol*b_iprod;  JX_CHECK_V_ERROR;
    eps = rtol * rtol * b_iprod;
    its = 0;
    while (1)
    {
        ++ its;
        rho_1 = jx_InnerProd(m, r_hat, r);
        if (rho_1 == 0)
        {
            JX_SET_V_ERROR("(r_hat . r) = 0; method fails");
        }
        if (its == 1)
        {
            jx_CopyVec(m, r, p); JX_CHECK_V_ERROR;
        }
        else
        {
            beta_1 = (rho_1 / rho_2) * (alpha_1 / widget_1);
            jx_Axpy(m, -widget_1, v, p); JX_CHECK_V_ERROR;
            jx_ScaleVec(m, beta_1, p); JX_CHECK_V_ERROR;
            jx_Axpy(m, 1.0, r, p); JX_CHECK_V_ERROR;
        }
        jx_Euclid_dhApply(ctx, p, p_hat); JX_CHECK_V_ERROR;
        jx_Mat_dhMatVec(A, p_hat, v); JX_CHECK_V_ERROR;
        JX_Real tmp = jx_InnerProd(m, r_hat, v); JX_CHECK_V_ERROR;
        alpha = rho_1 / tmp;
        jx_CopyVec(m, r, s); JX_CHECK_V_ERROR;
        jx_Axpy(m, -alpha, v, s); JX_CHECK_V_ERROR;
        s_norm = jx_InnerProd(m, s, s);
        if (s_norm < exit_a)
        {
            JX_SET_INFO("reached absolute stopping criteria");
            break;
        }
        jx_Euclid_dhApply(ctx, s, s_hat); JX_CHECK_V_ERROR;
        jx_Mat_dhMatVec(A, s_hat, t); JX_CHECK_V_ERROR;
        JX_Real tmp1, tmp2;
        tmp1 = jx_InnerProd(m, t, s); JX_CHECK_V_ERROR;
        tmp2 = jx_InnerProd(m, t, t); JX_CHECK_V_ERROR;
        widget = tmp1 / tmp2;
        jx_Axpy(m, alpha, p_hat, x); JX_CHECK_V_ERROR;
        jx_Axpy(m, widget, s_hat, x); JX_CHECK_V_ERROR;
        jx_CopyVec(m, s, r); JX_CHECK_V_ERROR;
        jx_Axpy(m, -widget, t, r); JX_CHECK_V_ERROR;
        r_iprod = jx_InnerProd(m, r, r); JX_CHECK_V_ERROR;
        if (r_iprod < eps)
        {
            JX_SET_INFO("stipulated residual reduction achieved");
            break;
        }
        if (monitor && jx_myid_dh == 0)
        {
            jx_fprintf(stderr, "[it = %i] %e\n", its, sqrt(r_iprod/b_iprod));
        }
        rho_2 = rho_1;
        widget_1 = widget;
        alpha_1 = alpha;
        if (its >= maxIts)
        {
            its = -its;
            break;
        }
    }
   *itsOUT = its;
    JX_FREE_DH(t);
    JX_FREE_DH(s);
    JX_FREE_DH(s_hat);
    JX_FREE_DH(v);
    JX_FREE_DH(p);
    JX_FREE_DH(p_hat);
    JX_FREE_DH(r);
    JX_FREE_DH(r_hat);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_cg_euclid"
void jx_cg_euclid( jx_Mat_dh A, jx_Euclid_dh ctx, JX_Real *x, JX_Real *b, JX_Int *itsOUT )
{
    JX_START_FUNC_DH
    JX_Int its, m = A->m;
    JX_Real *p, *r, *s;
    JX_Real alpha, beta, gamma, gamma_old, eps, bi_prod, i_prod;
    jx_bool monitor;
    JX_Int maxIts = ctx->maxIts;
    JX_Real rtol = ctx->rtol;
    
    monitor = jx_Parser_dhHasSwitch(jx_parser_dh, "-monitor");
    bi_prod = jx_InnerProd(m, b, b); JX_CHECK_V_ERROR;
    eps = (rtol * rtol) * bi_prod;
    p = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    s = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    r = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real));
    jx_Mat_dhMatVec(A, x, r); JX_CHECK_V_ERROR;
    jx_ScaleVec(m, -1.0, r); JX_CHECK_V_ERROR;
    jx_Axpy(m, 1.0, b, r); JX_CHECK_V_ERROR;
    jx_Euclid_dhApply(ctx, r, p); JX_CHECK_V_ERROR;
    gamma = jx_InnerProd(m, r, p); JX_CHECK_V_ERROR;
    its = 0;
    while (1)
    {
        ++ its;
        jx_Mat_dhMatVec(A, p, s);  JX_CHECK_V_ERROR;
        JX_Real tmp = jx_InnerProd(m, s, p); JX_CHECK_V_ERROR;
        alpha = gamma / tmp;
        gamma_old = gamma;
        jx_Axpy(m, alpha, p, x); JX_CHECK_V_ERROR;
        jx_Axpy(m, -alpha, s, r); JX_CHECK_V_ERROR;
        jx_Euclid_dhApply(ctx, r, s); JX_CHECK_V_ERROR;
        gamma = jx_InnerProd(m, r, s); JX_CHECK_V_ERROR;
        i_prod = jx_InnerProd(m, r, r); JX_CHECK_V_ERROR;
        if (monitor && jx_myid_dh == 0)
        {
            jx_fprintf(stderr, "iter = %i  rel. resid. norm: %e\n", its, sqrt(i_prod/bi_prod));
        }
        if (i_prod < eps) break;
        beta = gamma / gamma_old;
        jx_ScaleVec(m, beta, p); JX_CHECK_V_ERROR;
        jx_Axpy(m, 1.0, s, p); JX_CHECK_V_ERROR;
        if (its >= maxIts)
        {
            its = -its;
            break;
        }
    }
   *itsOUT = its;
    JX_FREE_DH(p);
    JX_FREE_DH(s);
    JX_FREE_DH(r);
    JX_END_FUNC_DH
}
