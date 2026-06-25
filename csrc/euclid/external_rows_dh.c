//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  external_rows_dh.c
 *  Date: 2013/01/21
 */

#include "jx_euclid.h"

enum{ JX_ROW_CT_TAG, JX_NZ_CT_TAG, JX_ROW_LENGTH_TAG, JX_ROW_NUMBER_TAG, JX_CVAL_TAG, JX_FILL_TAG, JX_AVAL_TAG };

#undef __FUNC__
#define __FUNC__ "jx_ExternalRows_dhCreate"
void jx_ExternalRows_dhCreate( jx_ExternalRows_dh *er )
{
    JX_START_FUNC_DH
    struct _jx_extrows_dh *tmp = (struct _jx_extrows_dh *)JX_MALLOC_DH(sizeof(struct _jx_extrows_dh)); JX_CHECK_V_ERROR;
    
   *er = tmp;
    if (JX_MAX_MPI_TASKS < jx_np_dh)
    {
        JX_SET_V_ERROR("JX_MAX_MPI_TASKS is too small; change, then recompile!");
    }
    JX_Int i;
    for (i = 0; i < JX_MAX_MPI_TASKS; ++ i)
    {
        tmp->rcv_row_lengths[i] = NULL;
        tmp->rcv_row_numbers[i] = NULL;
    }
    tmp->cvalExt = NULL;
    tmp->fillExt = NULL;
    tmp->avalExt = NULL;
    tmp->my_row_counts = NULL;
    tmp->my_row_numbers = NULL;
    tmp->cvalSend = NULL;
    tmp->fillSend = NULL;
    tmp->avalSend = NULL;
    tmp->rowLookup = NULL;
    tmp->sg = NULL;
    tmp->F = NULL;
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_ExtRows");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_ExternalRows_dhDestroy"
void jx_ExternalRows_dhDestroy( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int i;
    
    for (i = 0; i < JX_MAX_MPI_TASKS; ++ i)
    {
        if (er->rcv_row_lengths[i] != NULL)
        {
            JX_FREE_DH(er->rcv_row_lengths[i]); JX_CHECK_V_ERROR;
        }
        if (er->rcv_row_numbers[i] != NULL)
        {
            JX_FREE_DH(er->rcv_row_numbers[i]); JX_CHECK_V_ERROR;
        }
    }
    if (er->cvalExt != NULL)
    {
        JX_FREE_DH(er->cvalExt); JX_CHECK_V_ERROR;
    }
    if (er->fillExt != NULL)
    {
        JX_FREE_DH(er->fillExt); JX_CHECK_V_ERROR;
    }
    if (er->avalExt != NULL)
    {
        JX_FREE_DH(er->avalExt); JX_CHECK_V_ERROR;
    }
    if (er->my_row_counts != NULL)
    {
        JX_FREE_DH(er->my_row_counts); JX_CHECK_V_ERROR;
    }
    if (er->my_row_numbers != NULL)
    {
        JX_FREE_DH(er->my_row_numbers); JX_CHECK_V_ERROR;
    }
    if (er->cvalSend != NULL)
    {
        JX_FREE_DH(er->cvalSend); JX_CHECK_V_ERROR;
    }
    if (er->fillSend != NULL)
    {
        JX_FREE_DH(er->fillSend); JX_CHECK_V_ERROR;
    }
    if (er->avalSend != NULL)
    {
        JX_FREE_DH(er->avalSend); JX_CHECK_V_ERROR;
    }
    if (er->rowLookup != NULL)
    {
        jx_Hash_dhDestroy(er->rowLookup); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(er); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_ExternalRows_dhInit"
void jx_ExternalRows_dhInit( jx_ExternalRows_dh er, jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    er->sg = ctx->sg;
    er->F = ctx->F;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_ExternalRows_dhGetRow"
void jx_ExternalRows_dhGetRow( jx_ExternalRows_dh er, JX_Int globalRow, JX_Int *len, JX_Int **cval, JX_Int **fill, JX_REAL_DH **aval )
{
    JX_START_FUNC_DH
    if (er->rowLookup == NULL)
    {
       *len = 0;
    }
    else
    {
        jx_HashData *r = NULL;
        r = jx_Hash_dhLookup(er->rowLookup, globalRow); JX_CHECK_V_ERROR;
        if (r != NULL)
        {
           *len = r->iData;
            if (cval != NULL) *cval = r->iDataPtr;
            if (fill != NULL) *fill = r->iDataPtr2;
            if (aval != NULL) *aval = r->fDataPtr;
        }
        else
        {
           *len = 0;
        }
    }
    JX_END_FUNC_DH
}

static void jx_rcv_ext_storage_private( jx_ExternalRows_dh extRows );
static void jx_build_hash_table_private( jx_ExternalRows_dh er );
static void jx_rcv_external_rows_private( jx_ExternalRows_dh er );
static void jx_allocate_ext_row_storage_private( jx_ExternalRows_dh er );
static void jx_print_received_rows_private( jx_ExternalRows_dh er );

#undef __FUNC__
#define __FUNC__ "jx_ExternalRows_dhRecvRows"
void jx_ExternalRows_dhRecvRows( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && er->debug) debug = jx_true;
    if (er->sg->loCount > 0)
    {
        jx_rcv_ext_storage_private(er); JX_CHECK_V_ERROR;
        jx_allocate_ext_row_storage_private(er); JX_CHECK_V_ERROR;
        jx_build_hash_table_private(er); JX_CHECK_V_ERROR;
        jx_rcv_external_rows_private(er); JX_CHECK_V_ERROR;
        if (debug)
        {
            jx_print_received_rows_private(er); JX_CHECK_V_ERROR;
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_rcv_ext_storage_private"
void jx_rcv_ext_storage_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int loCount = er->sg->loCount, *loNabors = er->sg->loNabors;
    JX_Int *rcv_row_counts = er->rcv_row_counts;
    JX_Int *rcv_nz_counts = er->rcv_nz_counts;
    JX_Int **lengths = er->rcv_row_lengths, **numbers = er->rcv_row_numbers;
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && er->debug) debug = jx_true;
    for (i = 0; i < loCount; ++ i)
    {
        JX_Int nabor = loNabors[i];
        
        jx_MPI_Irecv(rcv_row_counts+i, 1, JX_MPI_INT, nabor, JX_ROW_CT_TAG, jx_comm_dh, er->req1+i);
        jx_MPI_Irecv(rcv_nz_counts+i, 1, JX_MPI_INT, nabor, JX_NZ_CT_TAG, jx_comm_dh, er->req2+i);
    }
    jx_MPI_Waitall(loCount, er->req1, er->status);
    jx_MPI_Waitall(loCount, er->req2, er->status);
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nEXR jx_rcv_ext_storage_private:: <nabor,rowCount,nzCount>\nEXR ");
        for (i = 0; i < loCount; ++ i)
        {
            jx_fprintf(jx_logFile, "<%i,%i,%i> ", loNabors[i], rcv_row_counts[i], rcv_nz_counts[i]);
        }
    }
    for (i = 0; i < loCount; ++ i)
    {
        JX_Int nz = rcv_nz_counts[i];
        JX_Int nabor = loNabors[i];
        
        lengths[i] = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        numbers[i] = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        jx_MPI_Irecv(lengths[i], nz, JX_MPI_INT, nabor, JX_ROW_LENGTH_TAG, jx_comm_dh, er->req1+i);
        jx_MPI_Irecv(numbers[i], nz, JX_MPI_INT, nabor, JX_ROW_NUMBER_TAG, jx_comm_dh, er->req2+i);
    }
    jx_MPI_Waitall(loCount, er->req1, er->status);
    jx_MPI_Waitall(loCount, er->req2, er->status);
    if (debug)
    {
        JX_Int j, nz;
        
        for (i = 0; i < loCount; ++ i)
        {
            jx_fprintf(jx_logFile, "\nEXR rows <number,length> to be received from P_%i\nEXR ", loNabors[i]);
            nz = rcv_row_counts[i];
            for (j = 0; j < nz; ++ j) jx_fprintf(jx_logFile, "<%i,%i> ", numbers[i][j], lengths[i][j]);
            jx_fprintf(jx_logFile, "\n");
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_allocate_ext_row_storage_private"
void jx_allocate_ext_row_storage_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int i, nz = 0;
    JX_Int loCount = er->sg->loCount;
    JX_Int *rcv_nz_counts = er->rcv_nz_counts;
    
    for (i = 0; i < loCount; ++ i) nz += rcv_nz_counts[i];
    er->cvalExt = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    er->fillExt = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    er->avalExt = (JX_REAL_DH *)JX_MALLOC_DH(nz*sizeof(JX_REAL_DH)); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_build_hash_table_private"
void jx_build_hash_table_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int loCount = er->sg->loCount;
    JX_Int i, j, offset, rowCt = 0;
    jx_Hash_dh table;
    jx_HashData record;
    JX_Int *extRowCval = er->cvalExt, *extRowFill = er->fillExt;
    JX_REAL_DH *extRowAval = er->avalExt;
    JX_Int *rcv_row_counts = er->rcv_row_counts;
    JX_Int **rcv_row_numbers = er->rcv_row_numbers;
    JX_Int **rcv_row_lengths = er->rcv_row_lengths;
    
    for (i = 0; i < loCount; ++ i) rowCt += rcv_row_counts[i];
    jx_Hash_dhCreate(&table, rowCt); JX_CHECK_V_ERROR;
    er->rowLookup = table;
    offset = 0;
    for (i = 0; i < loCount; ++ i)
    {
        JX_Int rowCount = rcv_row_counts[i];
        
        for (j = 0; j < rowCount; ++ j)
        {
            JX_Int row = rcv_row_numbers[i][j];
            JX_Int rowLength = rcv_row_lengths[i][j];
            
            record.iData = rowLength;
            record.iDataPtr = extRowCval + offset;
            record.iDataPtr2 = extRowFill + offset;
            record.fDataPtr = extRowAval + offset;
            jx_Hash_dhInsert(table, row, &record); JX_CHECK_V_ERROR;
            offset += rowLength;
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_rcv_external_rows_private"
void jx_rcv_external_rows_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int *rcv_nz_counts = er->rcv_nz_counts;
    JX_Int i, loCount = er->sg->loCount, *loNabors = er->sg->loNabors;
    JX_Int nabor, nz = 0, offset = 0;
    JX_Int *extRowCval = er->cvalExt, *extRowFill = er->fillExt;
    JX_Real *extRowAval = er->avalExt;
    
    nz = 0;
    for (i = 0; i < loCount; ++ i)
    {
        nabor = loNabors[i];
        nz = rcv_nz_counts[i];
        jx_MPI_Irecv(extRowCval+offset, nz, JX_MPI_INT, nabor, JX_CVAL_TAG, jx_comm_dh, er->req1+i);
        jx_MPI_Irecv(extRowFill+offset, nz, JX_MPI_INT, nabor, JX_FILL_TAG, jx_comm_dh, er->req2+i);
        jx_MPI_Irecv(extRowAval+offset, nz, JX_MPI_REAL, nabor, JX_AVAL_TAG, jx_comm_dh, er->req3+i);
        offset += nz;
    }
    jx_MPI_Waitall(loCount, er->req1, er->status);
    jx_MPI_Waitall(loCount, er->req2, er->status);
    jx_MPI_Waitall(loCount, er->req3, er->status);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_print_received_rows_private"
void jx_print_received_rows_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    jx_bool noValues = (jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues"));
    JX_Int i, j, k, rwCt, idx = 0, nabor;
    JX_Int loCount = er->sg->loCount, *loNabors = er->sg->loNabors;
    JX_Int n = er->F->n;
    
    jx_fprintf(jx_logFile, "\nEXR ================= received rows, printed from buffers ==============\n");
    for (i = 0; i < loCount; ++ i)
    {
        rwCt = er->rcv_row_counts[i];
        nabor = loNabors[i];
        jx_fprintf(jx_logFile, "\nEXR Rows received from P_%i:\n", nabor);
        for (j = 0; j < rwCt; ++ j)
        {
            JX_Int rowNum = er->rcv_row_numbers[i][j];
            JX_Int rowLen  = er->rcv_row_lengths[i][j];
            
            jx_fprintf(jx_logFile, "EXR %i :: ", 1+rowNum);
            for (k = 0; k < rowLen; ++ k)
            {
                if (noValues)
                {
                    jx_fprintf(jx_logFile, "%i,%i ; ", er->cvalExt[idx], er->fillExt[idx]);
                }
                else
                {
                    jx_fprintf(jx_logFile, "%i,%i,%g ; ", er->cvalExt[idx], er->fillExt[idx], er->avalExt[idx]);
                }
                ++ idx;
            }
            jx_fprintf(jx_logFile, "\n");
        }
    }
    jx_fprintf(jx_logFile, "\nEXR =============== received rows, printed from hash table =============\n");
    for (i = 0; i < n; ++ i)
    {
        JX_Int len, *cval, *fill;
        JX_REAL_DH *aval;
        
        jx_ExternalRows_dhGetRow(er, i, &len, &cval, &fill, &aval); JX_CHECK_V_ERROR;
        if (len > 0)
        {
            jx_fprintf(jx_logFile, "EXR %i :: ", i+1);
            for (j = 0; j < len; ++ j)
            {
                if (noValues)
                {
                    jx_fprintf(jx_logFile, "%i,%i ; ", cval[j], fill[j]);
                }
                else
                {
                    jx_fprintf(jx_logFile, "%i,%i,%g ; ", cval[j], fill[j], aval[j]);
                }
            }
            jx_fprintf(jx_logFile, "\n");
        }
    }
    JX_END_FUNC_DH
}

static void jx_send_ext_storage_private( jx_ExternalRows_dh er );
static void jx_send_external_rows_private( jx_ExternalRows_dh er );
static void jx_waitfor_sends_private( jx_ExternalRows_dh er );

#undef __FUNC__
#define __FUNC__ "jx_ExternalRows_dhSendRows"
void jx_ExternalRows_dhSendRows( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    if (er->sg->hiCount > 0)
    {
        jx_send_ext_storage_private(er); JX_CHECK_V_ERROR;
        jx_send_external_rows_private(er); JX_CHECK_V_ERROR;
        jx_waitfor_sends_private(er); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_send_ext_storage_private"
void jx_send_ext_storage_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int nz, i, j;
    JX_Int *nzCounts, *nzNumbers;
    JX_Int hiCount = er->sg->hiCount, *hiNabors = er->sg->hiNabors;
    JX_Int *rp = er->F->rp, *diag = er->F->diag;
    JX_Int m = er->F->m;
    JX_Int beg_row = er->F->beg_row;
    JX_Int rowCount = er->F->bdry_count;
    JX_Int first_bdry = er->F->first_bdry;
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && er->debug) debug = jx_true;
    nzCounts = er->my_row_counts = (JX_Int *)JX_MALLOC_DH(rowCount*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    nzNumbers = er->my_row_numbers = (JX_Int *)JX_MALLOC_DH(rowCount*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    nz = 0;
    for (i = first_bdry, j = 0; i < m; ++ i, ++ j)
    {
        JX_Int tmp = (rp[i+1] - diag[i]);
        
        nz += tmp;
        nzCounts[j] = tmp;
    }
    er->nzSend = nz;
    if (debug)
    {
        jx_fprintf(jx_logFile, "EXR jx_send_ext_storage_private:: rowCount = %i\n", rowCount);
        jx_fprintf(jx_logFile, "EXR jx_send_ext_storage_private:: nz Count = %i\n", nz);
    }
    for (i = 0; i < hiCount; ++ i)
    {
        JX_Int nabor = hiNabors[i];
        
        jx_MPI_Isend(&rowCount, 1, JX_MPI_INT, nabor, JX_ROW_CT_TAG, jx_comm_dh, er->req1+i);
        jx_MPI_Isend(&nz, 1, JX_MPI_INT, nabor, JX_NZ_CT_TAG, jx_comm_dh, er->req2+i);
    }
    for (i = 0, j = first_bdry; j < m; ++ i, ++ j)
    {
        nzNumbers[i] = j + beg_row;
    }
    for (i = 0; i < hiCount; ++ i)
    {
        JX_Int nabor = hiNabors[i];
        
        jx_MPI_Isend(nzNumbers, rowCount, JX_MPI_INT, nabor, JX_ROW_NUMBER_TAG, jx_comm_dh, er->req3+i);
        jx_MPI_Isend(nzCounts, rowCount, JX_MPI_INT, nabor, JX_ROW_LENGTH_TAG, jx_comm_dh, er->req4+i);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_send_external_rows_private"
void jx_send_external_rows_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    JX_Int i, j, hiCount = er->sg->hiCount, *hiNabors = er->sg->hiNabors;
    JX_Int offset, nz = er->nzSend;
    JX_Int *cvalSend, *fillSend;
    JX_REAL_DH *avalSend;
    JX_Int *cval = er->F->cval, *fill = er->F->fill;
    JX_Int m = er->F->m;
    JX_Int *rp = er->F->rp, *diag = er->F->diag;
    JX_Int first_bdry = er->F->first_bdry;
    JX_REAL_DH *aval = er->F->aval;
    jx_bool debug = jx_false;
    
    if (jx_logFile != NULL && er->debug) debug = jx_true;
    cvalSend = er->cvalSend = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    fillSend = er->fillSend = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    avalSend = er->avalSend = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    offset = 0;
    for (i = first_bdry, j = 0; i < m; ++ i, ++ j)
    {
        JX_Int tmp = (rp[i+1] - diag[i]);
        
        memcpy(cvalSend+offset, cval+diag[i], tmp*sizeof(JX_Int));
        memcpy(fillSend+offset, fill+diag[i], tmp*sizeof(JX_Int));
        memcpy(avalSend+offset, aval+diag[i], tmp*sizeof(JX_Real));
        offset += tmp;
    }
    if (debug)
    {
        JX_Int beg_row = er->F->beg_row;
        JX_Int idx = 0;
        jx_bool noValues = (jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues"));
        
        jx_fprintf(jx_logFile, "\nEXR ======================= send buffers ======================\n");
        for (i = first_bdry, j = 0; i < m; ++ i, ++ j)
        {
            JX_Int tmp = (rp[i+1] - diag[i]);
            
            jx_fprintf(jx_logFile, "EXR %i :: ", i+beg_row);
            for (j = 0; j < tmp; ++ j)
            {
                if (noValues)
                {
                    jx_fprintf(jx_logFile, "%i,%i ; ", cvalSend[idx], fillSend[idx]);
                }
                else
                {
                    jx_fprintf(jx_logFile, "%i,%i,%g ; ", cvalSend[idx], fillSend[idx], avalSend[idx]);
                }
                ++ idx;
            }
            jx_fprintf(jx_logFile, "\n");
        }
    }
    for (i = 0; i < hiCount; ++ i)
    {
        JX_Int nabor = hiNabors[i];
        
        jx_MPI_Isend(cvalSend, nz, JX_MPI_INT, nabor, JX_CVAL_TAG, jx_comm_dh, er->cval_req+i);
        jx_MPI_Isend(fillSend, nz, JX_MPI_INT, nabor, JX_FILL_TAG, jx_comm_dh, er->fill_req+i);
        jx_MPI_Isend(avalSend, nz, JX_MPI_REAL, nabor, JX_AVAL_TAG, jx_comm_dh, er->aval_req+i);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_waitfor_sends_private"
void jx_waitfor_sends_private( jx_ExternalRows_dh er )
{
    JX_START_FUNC_DH
    MPI_Status *status = er->status;
    JX_Int hiCount = er->sg->hiCount;
    
    if (hiCount)
    {
        jx_MPI_Waitall(hiCount, er->req1, status);
        jx_MPI_Waitall(hiCount, er->req2, status);
        jx_MPI_Waitall(hiCount, er->req3, status);
        jx_MPI_Waitall(hiCount, er->req4, status);
        jx_MPI_Waitall(hiCount, er->cval_req, status);
        jx_MPI_Waitall(hiCount, er->fill_req, status);
        jx_MPI_Waitall(hiCount, er->aval_req, status);
    }
    JX_END_FUNC_DH
}
