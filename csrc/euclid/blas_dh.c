//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  blas_dh.c
 *  Date: 2013/01/21
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_matvec_euclid_seq"
void jx_matvec_euclid_seq( JX_Int n, JX_Int *rp, JX_Int *cval, JX_Real *aval, JX_Real *x, JX_Real *y )
{
    JX_START_FUNC_DH
    JX_Int i, j;
    JX_Int from, to, col;
    JX_Real sum;
    
    if (jx_np_dh > 1) JX_SET_V_ERROR("only for sequential case!\n");
#ifdef USING_OPENMP_DH
#pragma omp parallel private(j, col, sum, from, to) default(shared) firstprivate(n, rp, cval, aval, x, y)
#endif
    {
#ifdef USING_OPENMP_DH
#pragma omp for schedule(static)
#endif
        for (i = 0; i < n; ++ i)
        {
            sum = 0.0;
            from = rp[i];
            to = rp[i+1];
            for (j = from; j < to; ++ j)
            {
                col = cval[j];
                sum += (aval[j] * x[col]);
            }
            y[i] = sum;
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Axpy"
void jx_Axpy( JX_Int n, JX_Real alpha, JX_Real *x, JX_Real *y )
{
    JX_START_FUNC_DH
    JX_Int i;
    
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(alpha, x, y) private(i)
#endif
    for (i = 0; i < n; ++ i)
    {
        y[i] = alpha * x[i] + y[i];
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_CopyVec"
void jx_CopyVec( JX_Int n, JX_Real *xIN, JX_Real *yOUT )
{
    JX_START_FUNC_DH
    JX_Int i;
    
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(yOUT, xIN) private(i)
#endif
    for (i = 0; i < n; ++ i)
    {
        yOUT[i] = xIN[i];
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_ScaleVec"
void jx_ScaleVec( JX_Int n, JX_Real alpha, JX_Real *x )
{
    JX_START_FUNC_DH
    JX_Int i;
    
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(alpha, x) private(i)
#endif
    for (i = 0; i < n; ++ i)
    {
        x[i] *= alpha;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_InnerProd"
JX_Real jx_InnerProd( JX_Int n, JX_Real *x, JX_Real *y )
{
    JX_START_FUNC_DH
    JX_Real result, local_result = 0.0;
    JX_Int i;
    
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(x, y) private(i) reduction(+:local_result)
#endif
    for (i = 0; i < n; ++ i)
    {
        local_result += x[i] * y[i];
    }
    if (jx_np_dh > 1)
    {
        jx_MPI_Allreduce(&local_result, &result, 1, JX_MPI_REAL, MPI_SUM, jx_comm_dh);
    }
    else
    {
        result = local_result;
    }
    JX_END_FUNC_VAL(result)
}

#undef __FUNC__
#define __FUNC__ "jx_Norm2"
JX_Real jx_Norm2( JX_Int n, JX_Real *x )
{
    JX_START_FUNC_DH
    JX_Real result, local_result = 0.0;
    JX_Int i;
    
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(static) firstprivate(x) private(i) reduction(+:local_result)
#endif
    for (i = 0; i < n; ++ i)
    {
        local_result += (x[i] * x[i]);
    }
    if (jx_np_dh > 1)
    {
        jx_MPI_Allreduce(&local_result, &result, 1, JX_MPI_REAL, MPI_SUM, jx_comm_dh);
    }
    else
    {
        result = local_result;
    }
    result = sqrt(result);
    JX_END_FUNC_VAL(result)
}
