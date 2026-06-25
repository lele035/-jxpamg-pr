//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  ilu_seq_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

static jx_bool jx_check_constraint_private( jx_Euclid_dh ctx, JX_Int b, JX_Int j );
static JX_Int jx_symbolic_row_private( JX_Int localRow, JX_Int *list, JX_Int *marker,
             JX_Int *tmpFill, JX_Int len, JX_Int *CVAL, JX_Real *AVAL, JX_Int *o2n_col, jx_Euclid_dh ctx, jx_bool debug );
static JX_Int jx_numeric_row_private( JX_Int localRow, JX_Int len,
                JX_Int *CVAL, JX_Real *AVAL, JX_REAL_DH *work, JX_Int *o2n_col, jx_Euclid_dh ctx, jx_bool debug );

#undef __FUNC__
#define __FUNC__ "jx_compute_scaling_private"
void jx_compute_scaling_private( JX_Int row, JX_Int len, JX_Real *AVAL, jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Real tmp = 0.0;
    JX_Int j;
    
    for (j = 0; j < len; ++ j) tmp = JX_MAX(tmp, fabs(AVAL[j]));
    if (tmp)
    {
        ctx->scale[row] = 1.0 / tmp;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_iluk_seq"
void jx_iluk_seq( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int *rp, *cval, *diag;
    JX_Int *CVAL;
    JX_Int i, j, len, count, col, idx = 0;
    JX_Int *list, *marker, *fill, *tmpFill;
    JX_Int temp, m, from = ctx->from, to = ctx->to;
    JX_Int *n2o_row, *o2n_col, beg_row, beg_rowP;
    JX_Real *AVAL;
    JX_REAL_DH *work, *aval;
    jx_Factor_dh F = ctx->F;
    jx_SubdomainGraph_dh sg = ctx->sg;
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_ilu")) debug = jx_true;
    m = F->m;
    rp = F->rp;
    cval = F->cval;
    fill = F->fill;
    diag = F->diag;
    aval = F->aval;
    work = ctx->work;
    count = rp[from];
    if (sg == NULL)
    {
        JX_SET_V_ERROR("subdomain graph is NULL");
    }
    n2o_row = ctx->sg->n2o_row;
    o2n_col = ctx->sg->o2n_col;
    beg_row = ctx->sg->beg_row[jx_myid_dh];
    beg_rowP = ctx->sg->beg_rowP[jx_myid_dh];
    list = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    marker = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    tmpFill = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) marker[i] = -1;
    for (i = 0; i < m; ++ i) work[i] = 0.0;
    for (i = from; i < to; ++ i)
    {
        JX_Int row = n2o_row[i];
        JX_Int globalRow = row + beg_row;
        
        if (debug)
        {
            jx_fprintf(jx_logFile,
              "ILU_seq ================================= starting local row: %i, (global= %i) level= %i\n",
                                              i+1, i+1+sg->beg_rowP[jx_myid_dh], ctx->level);
        }
        jx_EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        if (ctx->isScaled)
        {
            jx_compute_scaling_private(i, len, AVAL, ctx); JX_CHECK_V_ERROR;
        }
        count = jx_symbolic_row_private(i, list, marker, tmpFill, len, CVAL, AVAL, o2n_col, ctx, debug); JX_CHECK_V_ERROR;
        if (idx + count > F->alloc)
        {
            jx_Factor_dhReallocate(F, idx, count); JX_CHECK_V_ERROR;
            JX_SET_INFO("REALLOCATED from ilu_seq");
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
        jx_numeric_row_private(i, len, CVAL, AVAL, work, o2n_col, ctx, debug); JX_CHECK_V_ERROR
        jx_EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        if (debug)
        {
            jx_fprintf(jx_logFile, "ILU_seq:  ");
            for (j = rp[i]; j < rp[i+1]; ++ j)
            {
                col = cval[j];
                aval[j] = work[col];
                work[col] = 0.0;
                jx_fprintf(jx_logFile, "%i,%i,%g ; ", 1+cval[j], fill[j], aval[j]);
                fflush(jx_logFile);
            }
            jx_fprintf(jx_logFile, "\n");
        }
        else
        {
            for (j = rp[i]; j < rp[i+1]; ++ j)
            {
                col = cval[j];
                aval[j] = work[col];
                work[col] = 0.0;
            }
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
    if (beg_rowP)
    {
        JX_Int start = rp[from];
        JX_Int stop = rp[to];
        
        for (i = start; i < stop; ++ i) cval[i] += beg_rowP;
    }
    for (i = to+1; i < m; ++ i) rp[i] = 0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_iluk_seq_block"
void jx_iluk_seq_block( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int *rp, *cval, *diag;
    JX_Int *CVAL;
    JX_Int h, i, j, len, count, col, idx = 0;
    JX_Int *list, *marker, *fill, *tmpFill;
    JX_Int temp, m;
    JX_Int *n2o_row, *o2n_col, *beg_rowP, *n2o_sub, blocks;
    JX_Int *row_count, *dummy = NULL, dummy2[1];
    JX_Real *AVAL;
    JX_REAL_DH *work, *aval;
    jx_Factor_dh F = ctx->F;
    jx_SubdomainGraph_dh sg = ctx->sg;
    jx_bool bj = jx_false, constrained = jx_false;
    JX_Int discard = 0;
    JX_Int gr = -1;
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_ilu")) debug = jx_true;
    if (!strcmp(ctx->algo_par, "bj")) bj = jx_true;
    constrained = ! jx_Parser_dhHasSwitch(jx_parser_dh, "-unconstrained");
    m = F->m;
    rp = F->rp;
    cval = F->cval;
    fill = F->fill;
    diag = F->diag;
    aval = F->aval;
    work = ctx->work;
    if (sg != NULL)
    {
        n2o_row = sg->n2o_row;
        o2n_col = sg->o2n_col;
        row_count = sg->row_count;
        beg_rowP = sg->beg_rowP;
        n2o_sub = sg->n2o_sub;
        blocks = sg->blocks;
    }
    else
    {
        dummy = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        for (i = 0; i < m; ++ i) dummy[i] = i;
        n2o_row = dummy;
        o2n_col = dummy;
        dummy2[0] = m; row_count = dummy2;
        beg_rowP = dummy;
        n2o_sub = dummy;
        blocks = 1;
    }
    list = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    marker = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    tmpFill = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) marker[i] = -1;
    for (i = 0; i < m; ++ i) work[i] = 0.0;
    for (h = 0; h < blocks; ++ h)
    {
        JX_Int curBlock = n2o_sub[h];
        JX_Int first_row = beg_rowP[curBlock];
        JX_Int end_row = first_row + row_count[curBlock];
        
        if (debug)
        {
            jx_fprintf(jx_logFile, "\n\nILU_seq BLOCK: %i @@@@@@@@@@@@@@@ \n", curBlock);
        }
        for (i = first_row; i < end_row; ++ i)
        {
            JX_Int row = n2o_row[i];
            
            ++ gr;
            if (debug)
            {
                jx_fprintf(jx_logFile, "ILU_seq  global: %i  local: %i =================================\n",
                                            1+gr, 1+i-first_row);
            }
            jx_EuclidGetRow(ctx->A, row, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
            if (ctx->isScaled)
            {
                jx_compute_scaling_private(i, len, AVAL, ctx); JX_CHECK_V_ERROR;
            }
            count = jx_symbolic_row_private(i, list, marker,
                          tmpFill, len, CVAL, AVAL, o2n_col, ctx, debug); JX_CHECK_V_ERROR;
            if (idx + count > F->alloc)
            {
                jx_Factor_dhReallocate(F, idx, count); JX_CHECK_V_ERROR;
                JX_SET_INFO("REALLOCATED from ilu_seq");
                cval = F->cval;
                fill = F->fill;
                aval = F->aval;
            }
            col = list[m];
            while (count --)
            {
                if (constrained && !bj)
                {
                    if (col >= first_row && col < end_row)
                    {
                        cval[idx] = col;
                        fill[idx] = tmpFill[col];
                        ++ idx;
                    }
                    else
                    {
                        if (jx_check_constraint_private(ctx, curBlock, col))
                        {
                            cval[idx] = col;
                            fill[idx] = tmpFill[col];
                            ++ idx;
                        }
                        else
                        {
                            ++ discard;
                        }
                    }
                    col = list[col];
                }
                else if (bj)
                {
                    if (col >= first_row && col < end_row)
                    {
                        cval[idx] = col;
                        fill[idx] = tmpFill[col];
                        ++ idx;
                    }
                    else
                    {
                        ++ discard;
                    }
                    col = list[col];
                }
                else
                {
                    cval[idx] = col;
                    fill[idx] = tmpFill[col];
                    ++ idx;
                    col = list[col];
                }
            }
            rp[i+1] = idx;
            temp = rp[i];
            while (cval[temp] != i) ++ temp;
            diag[i] = temp;
            jx_numeric_row_private(i, len, CVAL, AVAL, work, o2n_col, ctx, debug); JX_CHECK_V_ERROR
            jx_EuclidRestoreRow(ctx->A, row, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
            if (debug)
            {
                jx_fprintf(jx_logFile, "ILU_seq: ");
                for (j = rp[i]; j < rp[i+1]; ++ j)
                {
                    col = cval[j];
                    aval[j] = work[col];
                    work[col] = 0.0;
                    jx_fprintf(jx_logFile, "%i,%i,%g ; ", 1+cval[j], fill[j], aval[j]);
                }
                jx_fprintf(jx_logFile, "\n");
            }
            else
            {
                for (j = rp[i]; j < rp[i+1]; ++ j)
                {
                    col = cval[j];
                    aval[j] = work[col];
                    work[col] = 0.0;
                }
            }
            if (!aval[diag[i]])
            {
                jx_sprintf(jx_msgBuf_dh, "zero diagonal in local row %i", i+1);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
        }
    }
    if (dummy != NULL)
    {
        JX_FREE_DH(dummy); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(list); JX_CHECK_V_ERROR;
    JX_FREE_DH(tmpFill); JX_CHECK_V_ERROR;
    JX_FREE_DH(marker); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_symbolic_row_private"
JX_Int jx_symbolic_row_private( JX_Int localRow,
                          JX_Int *list,
                          JX_Int *marker,
                          JX_Int *tmpFill,
                          JX_Int len,
                          JX_Int *CVAL,
                          JX_Real *AVAL,
                          JX_Int *o2n_col,
                          jx_Euclid_dh ctx,
                          jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int level = ctx->level, m = ctx->F->m;
    JX_Int *cval = ctx->F->cval, *diag = ctx->F->diag, *rp = ctx->F->rp;
    JX_Int *fill = ctx->F->fill;
    JX_Int count = 0;
    JX_Int j, node, tmp, col, head;
    JX_Int fill1, fill2, beg_row;
    JX_Real val;
    JX_Real thresh = ctx->sparseTolA;
    JX_REAL_DH scale;
    
    scale = ctx->scale[localRow];
    ctx->stats[JX_NZA_STATS] += (JX_Real)len;
    beg_row = ctx->sg->beg_row[jx_myid_dh];
    list[m] = m;
    for (j = 0; j < len; ++ j)
    {
        tmp = m;
        col = *CVAL ++;
        col -= beg_row;
        col = o2n_col[col];
        val = *AVAL ++;
        val *= scale;
        if (fabs(val) > thresh || col == localRow)
        {
            ++ count;
            while (col > list[tmp]) tmp = list[tmp];
            list[col] = list[tmp];
            list[tmp] = col;
            tmpFill[col] = 0;
            marker[col] = localRow;
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
            if (debug)
            {
                jx_fprintf(jx_logFile, "ILU_seq   sf updating from row: %i\n", 1+node);
            }
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
                         JX_Int len,
                         JX_Int *CVAL,
                         JX_Real *AVAL,
                         JX_REAL_DH *work,
                         JX_Int *o2n_col,
                         jx_Euclid_dh ctx,
                         jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Real  pc, pv, multiplier;
    JX_Int j, k, col, row;
    JX_Int *rp = ctx->F->rp, *cval = ctx->F->cval;
    JX_Int *diag = ctx->F->diag;
    JX_Int beg_row;
    JX_Real val;
    JX_REAL_DH *aval = ctx->F->aval, scale;
    
    scale = ctx->scale[localRow];
    beg_row  = ctx->sg->beg_row[jx_myid_dh];
    for (j = rp[localRow]; j < rp[localRow+1]; ++ j)
    {
        col = cval[j];
        work[col] = 0.0;
    }
    for (j = 0; j < len; ++ j)
    {
        col = *CVAL ++;
        col -= beg_row;
        val = *AVAL ++;
        col = o2n_col[col];
        work[col] = val * scale;
    }
    for (j = rp[localRow]; j < diag[localRow]; ++ j)
    {
        row = cval[j];
        pc = work[row];
        pv = aval[diag[row]];
        if (pc != 0.0 && pv != 0.0)
        {
            multiplier = pc / pv;
            work[row] = multiplier;
            if (debug)
            {
                jx_fprintf(jx_logFile, "ILU_seq   nf updating from row: %i; multiplier= %g\n", 1+row, multiplier);
            }
            for (k = diag[row]+1; k < rp[row+1]; ++ k)
            {
                col = cval[k];
                work[col] -= (multiplier * aval[k]);
            }
        }
        else
        {
            if (debug)
            {
                jx_fprintf(jx_logFile, "ILU_seq   nf NO UPDATE from row %i; pc = %g; pv = %g\n", 1+row, pc, pv);
            }
        }
    }
    JX_END_FUNC_VAL(0)
}

JX_Int jx_ilut_row_private( JX_Int localRow, JX_Int *list, JX_Int *o2n_col, JX_Int *marker,
                JX_Int len, JX_Int *CVAL, JX_Real *AVAL, JX_REAL_DH *work, jx_Euclid_dh ctx, jx_bool debug );

#undef __FUNC__
#define __FUNC__ "jx_ilut_seq"
void jx_ilut_seq( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int *rp, *cval, *diag, *CVAL;
    JX_Int i, len, count, col, idx = 0;
    JX_Int *list, *marker;
    JX_Int temp, m, from, to;
    JX_Int *n2o_row, *o2n_col, beg_row, beg_rowP;
    JX_Real *AVAL, droptol; 
    JX_REAL_DH *work, *aval, val;
    jx_Factor_dh F = ctx->F;
    jx_SubdomainGraph_dh sg = ctx->sg;
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_ilu")) debug = jx_true;
    m = F->m;
    rp = F->rp;
    cval = F->cval;
    diag = F->diag;
    aval = F->aval;
    work = ctx->work;
    from = ctx->from;
    to = ctx->to;
    count = rp[from];
    droptol = ctx->droptol;
    if (sg == NULL)
    {
        JX_SET_V_ERROR("subdomain graph is NULL");
    }
    n2o_row = ctx->sg->n2o_row;
    o2n_col = ctx->sg->o2n_col;
    beg_row = ctx->sg->beg_row[jx_myid_dh];
    beg_rowP = ctx->sg->beg_rowP[jx_myid_dh];
    list = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    marker = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) marker[i] = -1;
    rp[0] = 0;
    for (i = 0; i < m; ++ i) work[i] = 0.0;
    for (i = from; i < to; ++ i)
    {
        JX_Int row = n2o_row[i];
        JX_Int globalRow = row + beg_row;
        
        jx_EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        jx_compute_scaling_private(i, len, AVAL, ctx); JX_CHECK_V_ERROR; 
        count = jx_ilut_row_private(i, list, o2n_col, marker, len, CVAL, AVAL, work, ctx, debug); JX_CHECK_V_ERROR;
        jx_EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        if (idx + count > F->alloc)
        {
            jx_Factor_dhReallocate(F, idx, count); JX_CHECK_V_ERROR;
            JX_SET_INFO("REALLOCATED from ilu_seq");
            cval = F->cval;
            aval = F->aval;
        }
        col = list[m];
        while (count --)
        {
            val = work[col];
            if (col == i || fabs(val) > droptol)
            {
                cval[idx] = col;
                aval[idx++] = val;
                work[col] = 0.0;
            }
            col = list[col];
        }
        rp[i+1] = idx;
        temp = rp[i];
        while (cval[temp] != i) ++ temp;
        diag[i] = temp;
        if (!aval[diag[i]])
        {
            jx_sprintf(jx_msgBuf_dh, "zero diagonal in local row %i", i+1);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    if (beg_rowP)
    {
        JX_Int start = rp[from];
        JX_Int stop = rp[to];
        
        for (i = start; i < stop; ++ i) cval[i] += beg_rowP;
    }
    JX_FREE_DH(list);
    JX_FREE_DH(marker);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_ilut_row_private"
JX_Int jx_ilut_row_private( JX_Int localRow,
                      JX_Int *list,
                      JX_Int *o2n_col,
                      JX_Int *marker,
                      JX_Int len,
                      JX_Int *CVAL,
                      JX_Real *AVAL,
                      JX_REAL_DH *work,
                      jx_Euclid_dh ctx,
                      jx_bool debug )
{
    JX_START_FUNC_DH
    jx_Factor_dh F = ctx->F;
    JX_Int j, col, m = ctx->m, *rp = F->rp, *cval = F->cval;
    JX_Int tmp, *diag = F->diag;
    JX_Int head;
    JX_Int count = 0, beg_row;
    JX_Real val;
    JX_Real mult, *aval = F->aval;
    JX_Real scale, pv, pc;
    JX_Real droptol = ctx->droptol;
    JX_Real thresh = ctx->sparseTolA;
    
    scale = ctx->scale[localRow];
    ctx->stats[JX_NZA_STATS] += (JX_Real)len;
    beg_row = ctx->sg->beg_row[jx_myid_dh];
    list[m] = m;
    for (j = 0; j < len; ++ j)
    {
        tmp = m;
        col = *CVAL ++;
        col -= beg_row;
        col = o2n_col[col];
        val = *AVAL ++;
        val *= scale;
        if (fabs(val) > thresh || col == localRow)
        {
            ++ count;
            while (col > list[tmp]) tmp = list[tmp];
            list[col] = list[tmp];
            list[tmp] = col;
            work[col] = val;
            marker[col] = localRow;
        }
    }
    if (marker[localRow] != localRow)
    {
        tmp = m;
        while (localRow > list[tmp]) tmp = list[tmp];
        list[localRow] = list[tmp];
        list[tmp] = localRow;
        marker[localRow] = localRow;
        ++ count;
    }
    head = m;
    while (list[head] < localRow)
    {
        JX_Int row = list[head];
        
        pc = work[row];
        if (pc != 0.0)
        {
            pv = aval[diag[row]];
            mult = pc / pv;
            if (fabs(mult) > droptol)
            {
                work[row] = mult;
                for (j = diag[row]+1; j < rp[row+1]; ++ j)
                {
                    col = cval[j];
                    work[col] -= (mult * aval[j]);
                    if (marker[col] < localRow)
                    {
                        marker[col] = localRow;
                        tmp = head;
                        while (col > list[tmp]) tmp = list[tmp];
                        list[col] = list[tmp];
                        list[tmp] = col;
                        ++ count;
                    }
                }
            }
        }
        head = list[head];
    }
    JX_END_FUNC_VAL(count)
}

#undef __FUNC__
#define __FUNC__ "jx_check_constraint_private"
jx_bool jx_check_constraint_private( jx_Euclid_dh ctx, JX_Int p1, JX_Int j )
{
    JX_START_FUNC_DH
    jx_bool retval = jx_false;
    JX_Int i, p2;
    JX_Int *nabors, count;
    jx_SubdomainGraph_dh sg = ctx->sg;
    
    if (sg == NULL)
    {
        JX_SET_ERROR(-1, "ctx->sg == NULL");
    }
    p2 = jx_SubdomainGraph_dhFindOwner(ctx->sg, j, jx_true);
    nabors = sg->adj + sg->ptrs[p1];
    count = sg->ptrs[p1+1] - sg->ptrs[p1];
    for (i = 0; i < count; ++ i)
    {
        if (nabors[i] == p2)
        {
            retval = jx_true;
            break;
        }
    }
    JX_END_FUNC_VAL(retval)
}
