//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  ilu_mpi_bj_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

JX_Int jx_symbolic_row_private( JX_Int localRow, JX_Int beg_row, JX_Int end_row,
      JX_Int *list, JX_Int *marker, JX_Int *tmpFill, JX_Int len, JX_Int *CVAL, JX_Real *AVAL, JX_Int *o2n_col, jx_Euclid_dh ctx );
static JX_Int jx_numeric_row_private( JX_Int localRow, JX_Int beg_row, JX_Int end_row,
                JX_Int len, JX_Int *CVAL, JX_Real *AVAL, JX_REAL_DH *work, JX_Int *o2n_col, jx_Euclid_dh ctx );

#undef __FUNC__
#define __FUNC__ "jx_iluk_mpi_bj"
void jx_iluk_mpi_bj( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int *rp, *cval, *diag;
    JX_Int *CVAL;
    JX_Int i, j, len, count, col, idx = 0;
    JX_Int *list, *marker, *fill, *tmpFill;
    JX_Int temp, m, from = ctx->from, to = ctx->to;
    JX_Int *n2o_row, *o2n_col;
    JX_Int first_row, last_row;
    JX_Real *AVAL;
    JX_REAL_DH *work, *aval;
    jx_Factor_dh F = ctx->F;
    jx_SubdomainGraph_dh sg = ctx->sg;
    
    if (ctx->F == NULL)
    {
        JX_SET_V_ERROR("ctx->F is NULL");
    }
    if (ctx->F->rp == NULL)
    {
        JX_SET_V_ERROR("ctx->F->rp is NULL");
    }
    m = F->m;
    rp = F->rp;
    cval = F->cval;
    fill = F->fill;
    diag = F->diag;
    aval = F->aval;
    work = ctx->work;
    n2o_row = sg->n2o_row;
    o2n_col = sg->o2n_col;
    list = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    marker = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    tmpFill = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i)
    {
        marker[i] = -1;
        work[i] = 0.0;
    }
    first_row = sg->beg_row[jx_myid_dh];
    last_row  = first_row + sg->row_count[jx_myid_dh];
    for (i = from; i < to; ++ i)
    {
        JX_Int row = n2o_row[i];
        JX_Int globalRow = row + first_row;
        
        jx_EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        if (ctx->isScaled)
        {
            jx_compute_scaling_private(i, len, AVAL, ctx); JX_CHECK_V_ERROR;
        }
        count = jx_symbolic_row_private(i, first_row, last_row,
             list, marker, tmpFill, len, CVAL, AVAL, o2n_col, ctx); JX_CHECK_V_ERROR;
        if (idx + count > F->alloc)
        {
            jx_Factor_dhReallocate(F, idx, count); JX_CHECK_V_ERROR;
            JX_SET_INFO("REALLOCATED from lu_mpi_bj");
            cval = F->cval;
            fill = F->fill;
            aval = F->aval;
        }
        col = list[m];
        while (count --)
        {
            cval[idx] = col;
            fill[idx] = tmpFill[col];
            ++ idx;
            col = list[col];
        }
        rp[i+1] = idx;
        temp = rp[i];
        while (cval[temp] != i) ++ temp;
        diag[i] = temp;
        jx_numeric_row_private(i, first_row, last_row, len, CVAL, AVAL, work, o2n_col, ctx); JX_CHECK_V_ERROR
        jx_EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            col = cval[j];
            aval[j] = work[col];
            work[col] = 0.0;
        }
        if (!aval[diag[i]])
        {
            jx_sprintf(jx_msgBuf_dh, "zero diagonal in local row %i", i+1);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    JX_FREE_DH(list); JX_CHECK_V_ERROR;
    JX_FREE_DH(tmpFill); JX_CHECK_V_ERROR;
    JX_FREE_DH(marker); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_symbolic_row_private"
JX_Int jx_symbolic_row_private( JX_Int localRow,
                          JX_Int beg_row,
                          JX_Int end_row,
                          JX_Int *list,
                          JX_Int *marker,
                          JX_Int *tmpFill,
                          JX_Int len,
                          JX_Int *CVAL,
                          JX_Real *AVAL,
                          JX_Int *o2n_col,
                          jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int level = ctx->level, m = ctx->F->m;
    JX_Int *cval = ctx->F->cval, *diag = ctx->F->diag, *rp = ctx->F->rp;
    JX_Int *fill = ctx->F->fill;
    JX_Int count = 0;
    JX_Int j, node, tmp, col, head;
    JX_Int fill1, fill2;
    float val;
    JX_Real thresh = ctx->sparseTolA;
    JX_REAL_DH scale;
    
    scale = ctx->scale[localRow];
    ctx->stats[JX_NZA_STATS] += (JX_Real)len;
    list[m] = m;
    for (j = 0; j < len; ++ j)
    {
        tmp = m;
        col = *CVAL ++;
        val = *AVAL ++;
        if (col >= beg_row && col < end_row)
        {
            col -= beg_row;
            col = o2n_col[col];
            if (fabs(scale*val) > thresh || col == localRow)
            {
                ++ count;
                while (col > list[tmp]) tmp = list[tmp];
                list[col] = list[tmp];
                list[tmp] = col;
                tmpFill[col] = 0;
                marker[col] = localRow;
            }
        }
    }
    if (marker[localRow] != localRow)
    {
        tmp = m;
        while (localRow > list[tmp]) tmp = list[tmp];
        list[localRow] = list[tmp];
        list[tmp] = localRow;
        tmpFill[localRow] = 0;
        marker[localRow] = localRow;
        ++ count;
    }
    ctx->stats[JX_NZA_USED_STATS] += (JX_Real)count;
    head = m;
    if (level > 0)
    {
        while (list[head] < localRow)
        {
            node = list[head];
            fill1 = tmpFill[node];
            if (fill1 < level)
            {
                for (j = diag[node]+1; j < rp[node+1]; ++ j)
                {
                    col = cval[j];
                    fill2 = fill1 + fill[j] + 1;
                    if (fill2 <= level)
                    {
                        if (marker[col] < localRow)
                        {
                            tmp = head;
                            marker[col] = localRow;
                            tmpFill[col] = fill2;
                            while (col > list[tmp]) tmp = list[tmp];
                            list[col] = list[tmp];
                            list[tmp] = col;
                            ++ count;
                        }
                        else
                        {
                            tmpFill[col] = (fill2 < tmpFill[col]) ? fill2 : tmpFill[col];
                        }
                    }
                }
            }
            head = list[head];
        }
    }
    JX_END_FUNC_VAL(count)
}

#undef __FUNC__
#define __FUNC__ "jx_numeric_row_private"
JX_Int jx_numeric_row_private( JX_Int localRow,
                         JX_Int beg_row,
                         JX_Int end_row,
                         JX_Int len,
                         JX_Int *CVAL,
                         JX_Real *AVAL,
                         JX_REAL_DH *work,
                         JX_Int *o2n_col,
                         jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Real pc, pv, multiplier;
    JX_Int j, k, col, row;
    JX_Int *rp = ctx->F->rp, *cval = ctx->F->cval;
    JX_Int *diag = ctx->F->diag;
    JX_Real val;
    JX_REAL_DH *aval = ctx->F->aval, scale;
    
    scale = ctx->scale[localRow];
    for (j = rp[localRow]; j < rp[localRow+1]; ++ j)
    {
        col = cval[j];
        work[col] = 0.0;
    }
    for (j = 0; j < len; ++ j)
    {
        col = *CVAL ++;
        val = *AVAL ++;
        if (col >= beg_row && col < end_row)
        {
            col -= beg_row;
            col = o2n_col[col];
            work[col] = val * scale;
        }
    }
    for (j = rp[localRow]; j < diag[localRow]; ++ j)
    {
        row = cval[j];
        pc = work[row];
        if (pc != 0.0)
        {
            pv = aval[diag[row]];
            multiplier = pc / pv;
            work[row] = multiplier;
            for (k = diag[row]+1; k < rp[row+1]; ++ k)
            {
                col = cval[k];
                work[col] -= (multiplier * aval[k]);
            }
        }
    }
    JX_END_FUNC_VAL(0)
}
