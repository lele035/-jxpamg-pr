//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_ilu_mpi_pilu_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

static void jx_iluk_symbolic_row_private( JX_Int localRow, JX_Int len, JX_Int *CVAL,
             JX_Real *AVAL, jx_ExternalRows_dh extRows, jx_SortedList_dh sList, jx_Euclid_dh ctx, jx_bool debug );
static void jx_iluk_numeric_row_private( JX_Int new_row,
                  jx_ExternalRows_dh extRows, jx_SortedList_dh slist, jx_Euclid_dh ctx, jx_bool debug );

#undef __FUNC__
#define __FUNC__ "jx_iluk_mpi_pilu"
void jx_iluk_mpi_pilu( jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Int from = ctx->from, to = ctx->to;
    JX_Int i, m;
    JX_Int *n2o_row;
    JX_Int *rp, *cval, *diag, *fill;
    JX_Int beg_row, beg_rowP, end_rowP;
    jx_SubdomainGraph_dh sg = ctx->sg;
    JX_Int *CVAL, len, idx = 0, count;
    JX_Real *AVAL;
    JX_REAL_DH *aval;
    jx_Factor_dh F = ctx->F;
    jx_SortedList_dh slist = ctx->slist;
    jx_ExternalRows_dh extRows = ctx->extRows;
    jx_bool bj, noValues, debug = jx_false;
    
    if (jx_logFile != NULL && jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_ilu")) debug = jx_true;
    noValues = jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues");
    bj = ctx->F->blockJacobi;
    m = F->m;
    rp = F->rp;
    cval = F->cval;
    fill = F->fill;
    diag = F->diag;
    aval = F->aval;
    n2o_row = sg->n2o_row;
    if (from != 0) idx = rp[from];
    beg_row = sg->beg_row[jx_myid_dh];
    beg_rowP = sg->beg_rowP[jx_myid_dh];
    end_rowP = beg_rowP + sg->row_count[jx_myid_dh];
    for (i = from; i < to; ++ i)
    {
        JX_Int row = n2o_row[i];
        JX_Int globalRow = row + beg_row;
        
        if (debug)
        {
            jx_fprintf(jx_logFile,
              "\nILU_pilu global: %i  old_Local: %i =========================================================\n",
                              i+1+beg_rowP, row+1);
        }
        jx_EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        if (debug)
        {
            JX_Int h;
            
            jx_fprintf(jx_logFile, "ILU_pilu  jx_EuclidGetRow:\n");
            for (h = 0; h < len; ++ h) jx_fprintf(jx_logFile, "    %i   %g\n", 1+CVAL[h], AVAL[h]);
        }
        if (ctx->isScaled)
        {
            jx_compute_scaling_private(i, len, AVAL, ctx); JX_CHECK_V_ERROR;
        }
        jx_SortedList_dhReset(slist, i); JX_CHECK_V_ERROR;
        jx_iluk_symbolic_row_private(i, len, CVAL, AVAL, extRows, slist, ctx, debug); JX_CHECK_V_ERROR;
        jx_SortedList_dhEnforceConstraint(slist, sg);
        if (!noValues)
        {
            jx_iluk_numeric_row_private(i, extRows, slist, ctx, debug); JX_CHECK_V_ERROR;
        }
        jx_EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); JX_CHECK_V_ERROR;
        count = jx_SortedList_dhReadCount(slist); JX_CHECK_V_ERROR;
        if (idx + count > F->alloc)
        {
            jx_Factor_dhReallocate(F, idx, count); JX_CHECK_V_ERROR;
            JX_SET_INFO("REALLOCATED from jx_ilu_mpi_pilu");
            cval = F->cval;
            fill = F->fill;
            aval = F->aval;
        }
        if (bj)
        {
            JX_Int col;
            
            while (count --)
            {
                jx_SRecord *sr = jx_SortedList_dhGetSmallest(slist); JX_CHECK_V_ERROR;
                col = sr->col;
                if (col >= beg_rowP && col < end_rowP)
                {
                    cval[idx] = col;
                    if (noValues)
                    {
                        aval[idx] = 0.0;
                    }
                    else
                    {
                        aval[idx] = sr->val;
                    }
                    fill[idx] = sr->level;
                    ++ idx;
                }
            }
        }
        if (debug)
        {
            jx_fprintf(jx_logFile, "ILU_pilu  ");
            while (count --)
            {
                jx_SRecord *sr = jx_SortedList_dhGetSmallest(slist); JX_CHECK_V_ERROR;
                cval[idx] = sr->col;
                aval[idx] = sr->val;
                fill[idx] = sr->level;
                jx_fprintf(jx_logFile, "%i,%i,%g ; ", 1+cval[idx], fill[idx], aval[idx]);
                ++ idx;
            }
            jx_fprintf(jx_logFile, "\n");
        }
        else
        {
            while (count --)
            {
                jx_SRecord *sr = jx_SortedList_dhGetSmallest(slist); JX_CHECK_V_ERROR;
                cval[idx] = sr->col;
                aval[idx] = sr->val;
                fill[idx] = sr->level;
                ++ idx;
            }
        }
        rp[i+1] = idx;
        JX_Int temp = rp[i];
        jx_bool flag = jx_true;
        while (temp < idx)
        {
            if (cval[temp] == i+beg_rowP)
            {
                diag[i] = temp;
                flag = jx_false;
                break;
            }
            ++ temp;
        }
        if (flag)
        {
            if (jx_logFile != NULL)
            {
                JX_Int k;
                jx_fprintf(jx_logFile, "Failed to jx_find diag in localRow %i (globalRow %i; ct= %i)\n   ",
                                                                   1+i, i+1+beg_rowP, rp[i+1] - rp[i]);
                for (k = rp[i]; k < rp[i+1]; ++ k)
                {
                    jx_fprintf(jx_logFile, "%i ", cval[i]+1);
                }
                jx_fprintf(jx_logFile, "\n\n");
            }
            jx_sprintf(jx_msgBuf_dh, "failed to jx_find diagonal for localRow: %i", 1+i);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
        if (!aval[diag[i]])
        {
            jx_sprintf(jx_msgBuf_dh, "zero diagonal in local row %i", i+1);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    if (bj)
    {
        JX_Int nz = rp[m];
        for (i = 0; i < nz; ++ i) cval[i] -= beg_rowP;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_iluk_symbolic_row_private"
void jx_iluk_symbolic_row_private( JX_Int localRow,
                                JX_Int len,
                                JX_Int *CVAL,
                                JX_Real *AVAL,
                                jx_ExternalRows_dh extRows,
                                jx_SortedList_dh slist,
                                jx_Euclid_dh ctx,
                                jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int level = ctx->level, m = ctx->m;
    JX_Int beg_row = ctx->sg->beg_row[jx_myid_dh];
    JX_Int beg_rowP = ctx->sg->beg_rowP[jx_myid_dh];
    JX_Int *cval = ctx->F->cval, *diag = ctx->F->diag; 
    JX_Int *rp = ctx->F->rp, *fill = ctx->F->fill;
    JX_Int j, node, col;
    JX_Int end_rowP = beg_rowP + m;
    JX_Int level_1, level_2;
    JX_Int *cvalPtr, *fillPtr;
    jx_SRecord sr, *srPtr;
    JX_REAL_DH scale, *avalPtr;
    JX_Real thresh = ctx->sparseTolA;
    jx_bool wasInserted;
    JX_Int count = 0;
    
    scale = ctx->scale[localRow];
    ctx->stats[JX_NZA_STATS] += (JX_Real)len;
    sr.level = 0;
    for (j = 0; j < len; ++ j)
    {
        sr.col = CVAL[j];
        sr.val = scale * AVAL[j];
        wasInserted = jx_SortedList_dhPermuteAndInsert(slist, &sr, thresh); JX_CHECK_V_ERROR;
        if (wasInserted) ++ count;
        if (debug)
        {
            jx_fprintf(jx_logFile, "ILU_pilu   inserted from A: col= %i  val= %g\n", 1+CVAL[j], sr.val);
        }
    }
    sr.val = 0.0;
    sr.col = localRow + beg_rowP;
    srPtr = jx_SortedList_dhFind(slist, &sr); JX_CHECK_V_ERROR;
    if (srPtr == NULL)
    {
        jx_SortedList_dhInsert(slist, &sr); JX_CHECK_V_ERROR;
        ++ count;
        if (debug)
        {
            jx_fprintf(jx_logFile, "ILU_pilu   inserted missing diagonal: %i\n", 1+localRow+beg_row);
        }
    }
    ctx->stats[JX_NZA_USED_STATS] += (JX_Real)count;
    sr.val = 0.0;
    if (level > 0)
    {
        while(1)
        {
            srPtr = jx_SortedList_dhGetSmallestLowerTri(slist); JX_CHECK_V_ERROR;
            if (srPtr == NULL) break;
            node = srPtr->col;
            if (debug)
            {
                jx_fprintf(jx_logFile, "ILU_pilu   sf updating from row: %i\n", 1+srPtr->col);
            }
            level_1 = srPtr->level;
            if (level_1 < level)
            {
                if (node >= beg_rowP && node < end_rowP)
                {
                    node -= beg_rowP;
                    len = rp[node+1] - diag[node] - 1;
                    cvalPtr = cval + diag[node] + 1;
                    fillPtr = fill + diag[node] + 1;
                }
                else
                {
                    len = 0;
                    jx_ExternalRows_dhGetRow(extRows, node, &len, &cvalPtr, &fillPtr, &avalPtr); JX_CHECK_V_ERROR;
                    if (debug && len == 0)
                    {
                        jx_fprintf(stderr, "ILU_pilu  sf failed to get extern row: %i\n", 1+node);
                    }
                }
                for (j = 0; j < len; ++ j)
                {
                    col = *cvalPtr ++;
                    level_2 = 1+ level_1 + *fillPtr ++;
                    if (level_2 <= level)
                    {
                        sr.col = col;
                        sr.level = level_2;
                        sr.val = 0.0;
                        jx_SortedList_dhInsertOrUpdate(slist, &sr); JX_CHECK_V_ERROR;
                    }
                }
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_iluk_numeric_row_private"
void jx_iluk_numeric_row_private( JX_Int new_row, jx_ExternalRows_dh extRows, jx_SortedList_dh slist, jx_Euclid_dh ctx, jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int m = ctx->m;
    JX_Int beg_rowP = ctx->sg->beg_rowP[jx_myid_dh];
    JX_Int end_rowP = beg_rowP + m;
    JX_Int len, row;
    JX_Int *rp = ctx->F->rp, *cval = ctx->F->cval, *diag = ctx->F->diag;
    JX_REAL_DH *avalPtr, *aval = ctx->F->aval;
    JX_Int *cvalPtr;
    JX_Real multiplier, pc, pv;
    jx_SRecord sr, *srPtr;
    
    jx_SortedList_dhResetGetSmallest(slist); JX_CHECK_V_ERROR;
    while (1)
    {
        srPtr = jx_SortedList_dhGetSmallestLowerTri(slist); JX_CHECK_V_ERROR;
        if (srPtr == NULL) break;
        row = srPtr->col;
        if (row >= beg_rowP && row < end_rowP)
        {
            JX_Int local_row = row - beg_rowP;
            
            len = rp[local_row+1] - diag[local_row];
            cvalPtr = cval + diag[local_row];
            avalPtr = aval + diag[local_row];
        }
        else
        {
            len = 0;
            jx_ExternalRows_dhGetRow(extRows, row, &len, &cvalPtr, NULL, &avalPtr); JX_CHECK_V_ERROR;
            if (debug && len == 0)
            {
                jx_fprintf(stderr, "ILU_pilu  failed to get extern row: %i\n", 1+row);
            }
        }
        if (len)
        {
            sr.col = row;
            srPtr = jx_SortedList_dhFind(slist, &sr); JX_CHECK_V_ERROR;
            if (srPtr == NULL)
            {
                jx_sprintf(jx_msgBuf_dh, "jx_find failed for sr.col = %i while factoring local row= %i \n",
                                                                                    1+sr.col, new_row+1);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
            pc = srPtr->val;
            if (pc != 0.0)
            {
                pv = *avalPtr ++;
                -- len;
                ++ cvalPtr;
                multiplier = pc / pv;
                srPtr->val = multiplier;
                if (debug)
                {
                    jx_fprintf(jx_logFile, "ILU_pilu   nf updating from row: %i; multiplier = %g\n",
                                                                           1+srPtr->col, multiplier);
                }
                while (len --)
                {
                    sr.col = *cvalPtr ++;
                    sr.val = *avalPtr ++;
                    srPtr = jx_SortedList_dhFind(slist, &sr); JX_CHECK_V_ERROR;
                    if (srPtr != NULL)
                    {
                        srPtr->val -= (multiplier * sr.val);
                    }
                }
            }
            else
            {
                if (debug)
                {
                    jx_fprintf(jx_logFile, "ILU_pilu   NO UPDATE from row: %i; srPtr->val = 0.0\n", 1+srPtr->col);
                }
            }
        }
    }
    JX_END_FUNC_DH
}
