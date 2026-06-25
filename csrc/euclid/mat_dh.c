//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  mat_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

static void jx_setup_matvec_sends_private( jx_Mat_dh mat, JX_Int *inlist );
static void jx_setup_matvec_receives_private( jx_Mat_dh mat, JX_Int *beg_rows,
                       JX_Int *end_rows, JX_Int reqlen, JX_Int *reqind, JX_Int *outlist );

static jx_bool jx_commsOnly = jx_false;

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhCreate"
void jx_Mat_dhCreate( jx_Mat_dh *mat )
{
    JX_START_FUNC_DH
    struct _jx_mat_dh *tmp = (struct _jx_mat_dh *)JX_MALLOC_DH(sizeof(struct _jx_mat_dh)); JX_CHECK_V_ERROR;
    
   *mat = tmp;
    jx_commsOnly = jx_Parser_dhHasSwitch(jx_parser_dh, "-jx_commsOnly");
    if (jx_myid_dh == 0 && jx_commsOnly == jx_true)
    {
        fflush(stdout);
    }
    tmp->m = 0;
    tmp->n = 0;
    tmp->beg_row = 0;
    tmp->bs = 1;
    tmp->rp = NULL;
    tmp->len = NULL;
    tmp->cval = NULL;
    tmp->aval = NULL;
    tmp->diag = NULL;
    tmp->fill = NULL;
    tmp->owner = jx_true;
    tmp->len_private = 0;
    tmp->rowCheckedOut = -1;
    tmp->cval_private = NULL;
    tmp->aval_private = NULL;
    tmp->row_perm = NULL;
    tmp->num_recv = 0;
    tmp->num_send = 0;
    tmp->recv_req = NULL;
    tmp->send_req = NULL;
    tmp->status = NULL;
    tmp->recvbuf = NULL;
    tmp->sendbuf = NULL;
    tmp->sendind = NULL;
    tmp->sendlen = 0;
    tmp->recvlen = 0;
    tmp->numb = NULL;
    tmp->matvecIsSetup = jx_false;
    jx_Mat_dhZeroTiming(tmp); JX_CHECK_V_ERROR;
    tmp->matvec_timing = jx_true;
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_Mat");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhDestroy"
void jx_Mat_dhDestroy( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    JX_Int i;
    
    if (mat->owner)
    {
        if (mat->rp != NULL)
        {
            JX_FREE_DH(mat->rp); JX_CHECK_V_ERROR;
        }
        if (mat->len != NULL)
        {
            JX_FREE_DH(mat->len); JX_CHECK_V_ERROR;
        }
        if (mat->cval != NULL)
        {
            JX_FREE_DH(mat->cval); JX_CHECK_V_ERROR;
        }
        if (mat->aval != NULL)
        {
            JX_FREE_DH(mat->aval); JX_CHECK_V_ERROR;
        }
        if (mat->diag != NULL)
        {
            JX_FREE_DH(mat->diag); JX_CHECK_V_ERROR;
        }
        if (mat->fill != NULL)
        {
            JX_FREE_DH(mat->fill); JX_CHECK_V_ERROR;
        }
        if (mat->cval_private != NULL)
        {
            JX_FREE_DH(mat->cval_private); JX_CHECK_V_ERROR;
        }
        if (mat->aval_private != NULL)
        {
            JX_FREE_DH(mat->aval_private); JX_CHECK_V_ERROR;
        }
        if (mat->row_perm != NULL)
        {
            JX_FREE_DH(mat->row_perm); JX_CHECK_V_ERROR;
        }
    }
    for (i = 0; i < mat->num_recv; i ++) jx_MPI_Request_free(&mat->recv_req[i]);
    for (i = 0; i < mat->num_send; i ++) jx_MPI_Request_free(&mat->send_req[i]);
    if (mat->recv_req != NULL)
    {
        JX_FREE_DH(mat->recv_req); JX_CHECK_V_ERROR;
    }
    if (mat->send_req != NULL)
    {
        JX_FREE_DH(mat->send_req); JX_CHECK_V_ERROR;
    }
    if (mat->status != NULL)
    {
        JX_FREE_DH(mat->status); JX_CHECK_V_ERROR;
    }
    if (mat->recvbuf != NULL)
    {
        JX_FREE_DH(mat->recvbuf); JX_CHECK_V_ERROR;
    }
    if (mat->sendbuf != NULL)
    {
        JX_FREE_DH(mat->sendbuf); JX_CHECK_V_ERROR;
    }
    if (mat->sendind != NULL)
    {
        JX_FREE_DH(mat->sendind); JX_CHECK_V_ERROR;
    }
    if (mat->matvecIsSetup)
    {
        jx_Mat_dhMatVecSetdown(mat); JX_CHECK_V_ERROR;
    }
    if (mat->numb != NULL)
    {
        jx_Numbering_dhDestroy(mat->numb); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(mat); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMatVecSetDown"
void jx_Mat_dhMatVecSetdown( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    if (jx_ignoreMe) JX_SET_V_ERROR("not implemented");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMatVecSetup"
void jx_Mat_dhMatVecSetup( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    if (jx_np_dh == 1)
    {
        goto DO_NOTHING;
    }
    else
    {
        JX_Int *outlist, *inlist;
        JX_Int ierr, i, row, *rp = mat->rp, *cval = mat->cval;
        jx_Numbering_dh numb;
        JX_Int m = mat->m;
        JX_Int firstLocal = mat->beg_row;
        JX_Int lastLocal = firstLocal+m;
        JX_Int *beg_rows, *end_rows;
        
        mat->recv_req = (MPI_Request *)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
        mat->send_req = (MPI_Request *)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
        mat->status = (MPI_Status *)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
        beg_rows = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        end_rows = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        if (jx_np_dh == 1)
        {
            beg_rows[0] = 0;
            end_rows[0] = m;
        }
        else
        {
            ierr = jx_MPI_Allgather(&firstLocal, 1, JX_MPI_INT, beg_rows, 1, JX_MPI_INT, jx_comm_dh); JX_CHECK_MPI_V_ERROR(ierr);
            ierr = jx_MPI_Allgather(&lastLocal, 1, JX_MPI_INT, end_rows, 1, JX_MPI_INT, jx_comm_dh); JX_CHECK_MPI_V_ERROR(ierr);
        }
        outlist = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        inlist = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        for (i = 0; i < jx_np_dh; ++ i)
        {
            outlist[i] = 0;
            inlist[i] = 0;
        }
        jx_Numbering_dhCreate(&(mat->numb)); JX_CHECK_V_ERROR;
        numb = mat->numb;
        jx_Numbering_dhSetup(numb, mat); JX_CHECK_V_ERROR;
        jx_setup_matvec_receives_private(mat, beg_rows, end_rows,
                          numb->num_ext, numb->idx_ext, outlist); JX_CHECK_V_ERROR;
        if (jx_np_dh == 1)
        {
            inlist[0] = outlist[0];
        }
        else
        {
            ierr = jx_MPI_Alltoall(outlist, 1, JX_MPI_INT, inlist, 1, JX_MPI_INT, jx_comm_dh); JX_CHECK_MPI_V_ERROR(ierr);
        }
        jx_setup_matvec_sends_private(mat, inlist); JX_CHECK_V_ERROR;
        for (row = 0; row < m; row ++)
        {
            JX_Int len = rp[row+1] - rp[row];
            JX_Int *ind = cval+rp[row];
            
            jx_Numbering_dhGlobalToLocal(numb, len, ind, ind); JX_CHECK_V_ERROR;
        }
        JX_FREE_DH(outlist); JX_CHECK_V_ERROR;
        JX_FREE_DH(inlist); JX_CHECK_V_ERROR;
        JX_FREE_DH(beg_rows); JX_CHECK_V_ERROR;
        JX_FREE_DH(end_rows); JX_CHECK_V_ERROR;
    }
    
DO_NOTHING: ;
    
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_setup_matvec_receives_private"
void jx_setup_matvec_receives_private( jx_Mat_dh mat,
                                    JX_Int *beg_rows,
                                    JX_Int *end_rows,
                                    JX_Int reqlen,
                                    JX_Int *reqind,
                                    JX_Int *outlist )
{
    JX_START_FUNC_DH
    JX_Int ierr, i, j, this_pe;
    MPI_Request request;
    JX_Int m = mat->m;
    
    mat->num_recv = 0;
    mat->recvbuf = (JX_Real *)JX_MALLOC_DH((reqlen+m)*sizeof(JX_Real));
    for (i = 0; i < reqlen; i = j)
    {
        this_pe = jx_mat_jx_find_owner(beg_rows, end_rows, reqind[i]); JX_CHECK_V_ERROR;
        for (j = i+1; j < reqlen; j ++)
        {
            if (reqind[j] < beg_rows[this_pe] || reqind[j] > end_rows[this_pe]) break;
        }
        ierr = jx_MPI_Isend(&reqind[i], j-i, JX_MPI_INT, this_pe, 444, jx_comm_dh, &request); JX_CHECK_MPI_V_ERROR(ierr);
        ierr = jx_MPI_Request_free(&request); JX_CHECK_MPI_V_ERROR(ierr);
        outlist[this_pe] = j - i;
        ierr = jx_MPI_Recv_init(&mat->recvbuf[i+m], j-i,
                 JX_MPI_REAL, this_pe, 555, jx_comm_dh, &mat->recv_req[mat->num_recv]); JX_CHECK_MPI_V_ERROR(ierr);
        mat->num_recv ++;
        mat->recvlen += j - i;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_setup_matvec_sends_private"
void jx_setup_matvec_sends_private( jx_Mat_dh mat, JX_Int *inlist )
{
    JX_START_FUNC_DH
    JX_Int ierr, i, j, sendlen, first = mat->beg_row;
    MPI_Request *requests;
    MPI_Status  *statuses;
    
    requests = (MPI_Request *)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    statuses = (MPI_Status *)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
    sendlen = 0;
    for (i = 0; i < jx_np_dh; i ++) sendlen += inlist[i];
    mat->sendlen = sendlen;
    mat->sendbuf = (JX_Real *)JX_MALLOC_DH(sendlen*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    mat->sendind = (JX_Int *)JX_MALLOC_DH(sendlen*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    j = 0;
    mat->num_send = 0;
    for (i = 0; i < jx_np_dh; i ++)
    {
        if (inlist[i] != 0)
        {
            ierr = jx_MPI_Irecv(&mat->sendind[j], inlist[i],
                      JX_MPI_INT, i, 444, jx_comm_dh, &requests[mat->num_send]); JX_CHECK_MPI_V_ERROR(ierr);
            ierr = jx_MPI_Send_init(&mat->sendbuf[j], inlist[i],
                    JX_MPI_REAL, i, 555, jx_comm_dh, &mat->send_req[mat->num_send]); JX_CHECK_MPI_V_ERROR(ierr);
            mat->num_send ++;
            j += inlist[i];
        }
    }
    mat->time[JX_MATVEC_WORDS] = j;
    ierr = jx_MPI_Waitall(mat->num_send, requests, statuses); JX_CHECK_MPI_V_ERROR(ierr);
    for (i = 0; i < mat->sendlen; i ++) mat->sendind[i] -= first;
    JX_FREE_DH(requests);
    JX_FREE_DH(statuses);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMatVec"
void jx_Mat_dhMatVec( jx_Mat_dh mat, JX_Real *x, JX_Real *b )
{
    JX_START_FUNC_DH
    if (jx_np_dh == 1)
    {
        jx_Mat_dhMatVec_uni(mat, x, b); JX_CHECK_V_ERROR;
    }
    else
    {
        JX_Int ierr, i, row, m = mat->m;
        JX_Int *rp = mat->rp, *cval = mat->cval;
        JX_Real *aval = mat->aval;
        JX_Int *sendind = mat->sendind;
        JX_Int sendlen = mat->sendlen;
        JX_Real *sendbuf = mat->sendbuf;
        JX_Real *recvbuf = mat->recvbuf;
        JX_Real t1 = 0, t2 = 0, t3 = 0, t4 = 0;
        jx_bool timeFlag = mat->matvec_timing;
        
        if (timeFlag) t1 = jx_MPI_Wtime();
        if (!jx_commsOnly)
        {
            for (i = 0; i < sendlen; i ++) sendbuf[i] = x[sendind[i]];
        }
        if (timeFlag)
        {
            t2 = jx_MPI_Wtime();
            mat->time[JX_MATVEC_TIME] += (t2 - t1);
        }
        ierr = jx_MPI_Startall(mat->num_recv, mat->recv_req); JX_CHECK_MPI_V_ERROR(ierr);
        ierr = jx_MPI_Startall(mat->num_send, mat->send_req); JX_CHECK_MPI_V_ERROR(ierr);
        ierr = jx_MPI_Waitall(mat->num_recv, mat->recv_req, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
        ierr = jx_MPI_Waitall(mat->num_send, mat->send_req, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
        if (timeFlag)
        {
            t3 = jx_MPI_Wtime();
            mat->time[JX_MATVEC_MPI_TIME] += (t3 - t2);
        }
        if (!jx_commsOnly)
        {
            for (i = 0; i < m; i ++) recvbuf[i] = x[i];
            for (row = 0; row < m; row ++)
            {
                JX_Int len = rp[row+1] - rp[row];
                JX_Int *ind = cval + rp[row];
                JX_Real *val = aval + rp[row];
                JX_Real temp = 0.0;
                
                for (i = 0; i < len; i ++)
                {
                    temp += (val[i] * recvbuf[ind[i]]);
                }
                b[row] = temp;
            }
        }
        if (timeFlag)
        {
            t4 = jx_MPI_Wtime();
            mat->time[JX_MATVEC_TOTAL_TIME] += (t4 - t1);
            mat->time[JX_MATVEC_TIME] += (t4 - t3);
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMatVec_omp"
void jx_Mat_dhMatVec_omp( jx_Mat_dh mat, JX_Real *x, JX_Real *b )
{
    JX_START_FUNC_DH
    JX_Int ierr, i, row, m = mat->m;
    JX_Int *rp = mat->rp, *cval = mat->cval;
    JX_Real *aval = mat->aval;
    JX_Int *sendind = mat->sendind;
    JX_Int sendlen = mat->sendlen;
    JX_Real *sendbuf = mat->sendbuf;
    JX_Real *recvbuf = mat->recvbuf;
    JX_Real t1 = 0, t2 = 0, t3 = 0, t4 = 0, tx = 0;
    JX_Real *val, temp;
    JX_Int len, *ind;
    jx_bool timeFlag = mat->matvec_timing;
    
    if (timeFlag) t1 = jx_MPI_Wtime();
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(runtime) private(i)
#endif
    for (i = 0; i < sendlen; i ++) sendbuf[i] = x[sendind[i]];
    if (timeFlag)
    {
        t2 = jx_MPI_Wtime();
        mat->time[JX_MATVEC_TIME] += (t2 - t1);
    }
    ierr = jx_MPI_Startall(mat->num_recv, mat->recv_req); JX_CHECK_MPI_V_ERROR(ierr);
    ierr = jx_MPI_Startall(mat->num_send, mat->send_req); JX_CHECK_MPI_V_ERROR(ierr);
    ierr = jx_MPI_Waitall(mat->num_recv, mat->recv_req, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
    ierr = jx_MPI_Waitall(mat->num_send, mat->send_req, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
    if (timeFlag)
    {
        t3 = jx_MPI_Wtime();
        mat->time[JX_MATVEC_MPI_TIME] += (t3 - t2);
    }
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(runtime) private(i)
#endif
    for (i = 0; i < m; i ++) recvbuf[i] = x[i];
    if (timeFlag)
    {
        tx = jx_MPI_Wtime();
        mat->time[JX_MATVEC_MPI_TIME2] += (tx - t1);
    }
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(runtime) private(row,i,len,ind,val,temp)
#endif
    for (row = 0; row < m; row ++)
    {
        len = rp[row+1] - rp[row];
        ind = cval + rp[row];
        val = aval + rp[row];
        temp = 0.0;
        for (i = 0; i < len; i ++)
        {
            temp += (val[i] * recvbuf[ind[i]]);
        }
        b[row] = temp;
    }
    if (timeFlag)
    {
        t4 = jx_MPI_Wtime();
        mat->time[JX_MATVEC_TOTAL_TIME] += (t4 - t1);
        mat->time[JX_MATVEC_TIME] += (t4 - t3);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMatVec_uni_omp"
void jx_Mat_dhMatVec_uni_omp( jx_Mat_dh mat, JX_Real *x, JX_Real *b )
{
    JX_START_FUNC_DH
    JX_Int i, row, m = mat->m;
    JX_Int *rp = mat->rp, *cval = mat->cval;
    JX_Real *aval = mat->aval;
    JX_Real t1 = 0, t2 = 0;
    jx_bool timeFlag = mat->matvec_timing;
    
    if (timeFlag)
    {
        t1 = jx_MPI_Wtime();
    }
#ifdef USING_OPENMP_DH
#pragma omp parallel for schedule(runtime) private(row,i)
#endif
    for (row = 0; row < m; row ++)
    {
        JX_Int len = rp[row+1] - rp[row];
        JX_Int *ind = cval+rp[row];
        JX_Real *val = aval+rp[row];
        JX_Real temp = 0.0;
        
        for (i = 0; i < len; i ++)
        {
            temp += (val[i] * x[ind[i]]);
        }
        b[row] = temp;
    }
    if (timeFlag)
    {
        t2 = jx_MPI_Wtime();
        mat->time[JX_MATVEC_TIME] += (t2 - t1);
        mat->time[JX_MATVEC_TOTAL_TIME] += (t2 - t1);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMatVec_uni"
void jx_Mat_dhMatVec_uni( jx_Mat_dh mat, JX_Real *x, JX_Real *b )
{
    JX_START_FUNC_DH
    JX_Int i, row, m = mat->m;
    JX_Int *rp = mat->rp, *cval = mat->cval;
    JX_Real *aval = mat->aval;
    JX_Real t1 = 0, t2 = 0;
    jx_bool timeFlag = mat->matvec_timing;
    
    if (timeFlag) t1 = jx_MPI_Wtime();
    for (row = 0; row < m; row ++)
    {
        JX_Int len = rp[row+1] - rp[row];
        JX_Int *ind = cval + rp[row];
        JX_Real *val = aval + rp[row];
        JX_Real temp = 0.0;
        
        for (i = 0; i < len; i ++)
        {
            temp += (val[i] * x[ind[i]]);
        }
        b[row] = temp;
    }
    if (timeFlag)
    {
        t2 = jx_MPI_Wtime();
        mat->time[JX_MATVEC_TIME] += (t2 - t1);
        mat->time[JX_MATVEC_TOTAL_TIME] += (t2 - t1);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhReadNz"
JX_Int jx_Mat_dhReadNz( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    JX_Int ierr, retval = mat->rp[mat->m];
    JX_Int nz = retval;
    
    ierr = jx_MPI_Allreduce(&nz, &retval, 1, JX_MPI_INT, MPI_SUM, jx_comm_dh); JX_CHECK_MPI_ERROR(ierr);
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhZeroTiming"
void jx_Mat_dhZeroTiming( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    JX_Int i;
    
    for (i = 0; i < JX_MAT_DH_BINS; ++ i)
    {
        mat->time[i] = 0;
        mat->time_max[i] = 0;
        mat->time_min[i] = 0;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhReduceTiming"
void jx_Mat_dhReduceTiming( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    if (mat->time[JX_MATVEC_MPI_TIME])
    {
        mat->time[JX_MATVEC_RATIO] = mat->time[JX_MATVEC_TIME] / mat->time[JX_MATVEC_MPI_TIME];
    }
    jx_MPI_Allreduce(mat->time, mat->time_min, JX_MAT_DH_BINS, JX_MPI_REAL, MPI_MIN, jx_comm_dh);
    jx_MPI_Allreduce(mat->time, mat->time_max, JX_MAT_DH_BINS, JX_MPI_REAL, MPI_MAX, jx_comm_dh);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPermute"
void jx_Mat_dhPermute( jx_Mat_dh A, JX_Int *n2o, jx_Mat_dh *Bout )
{
    JX_START_FUNC_DH
    jx_Mat_dh B;
    JX_Int  i, j, *RP = A->rp, *CVAL = A->cval;
    JX_Int  *o2n, *rp, *cval, m = A->m, nz = RP[m];
    JX_Real *aval, *AVAL = A->aval;
    
    jx_Mat_dhCreate(&B); JX_CHECK_V_ERROR;
    B->m = B->n = m;
   *Bout = B;
    o2n = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) o2n[n2o[i]] = i;
    rp = B->rp = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    cval = B->cval = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    aval = B->aval = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    rp[0] = 0;
    for (i = 0; i < m; ++ i)
    {
        JX_Int oldRow = n2o[i];
        rp[i+1] = RP[oldRow+1] - RP[oldRow];
    }
    for (i = 1; i <= m; ++ i) rp[i] = rp[i] + rp[i-1];
    for (i = 0; i < m; ++ i)
    {
        JX_Int oldRow = n2o[i];
        JX_Int idx = rp[i];
        
        for (j = RP[oldRow]; j < RP[oldRow+1]; ++ j)
        {
            cval[idx] = o2n[CVAL[j]];
            aval[idx] = AVAL[j];
            ++ idx;
        }
    }
    JX_FREE_DH(o2n); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPrintGraph"
void jx_Mat_dhPrintGraph( jx_Mat_dh A, jx_SubdomainGraph_dh sg, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int pe, id = jx_myid_dh;
    JX_Int ierr;
    
    if (sg != NULL)
    {
        id = sg->o2n_sub[id];
    }
    for (pe = 0; pe < jx_np_dh; ++ pe)
    {
        ierr = jx_MPI_Barrier(jx_comm_dh); JX_CHECK_MPI_V_ERROR(ierr);
        if (id == pe)
        {
            if (sg == NULL)
            {
                jx_mat_dh_print_graph_private(A->m, A->beg_row,
                          A->rp, A->cval, A->aval, NULL, NULL, NULL, fp); JX_CHECK_V_ERROR;
            }
            else
            {
                JX_Int beg_row = sg->beg_rowP[jx_myid_dh];
                
                jx_mat_dh_print_graph_private(A->m, beg_row,
                      A->rp, A->cval, A->aval, sg->n2o_row, sg->o2n_col, sg->o2n_ext, fp); JX_CHECK_V_ERROR;
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPrintRows"
void jx_Mat_dhPrintRows( jx_Mat_dh A, jx_SubdomainGraph_dh sg, FILE *fp )
{
    JX_START_FUNC_DH
    jx_bool noValues;
    JX_Int m = A->m, *rp = A->rp, *cval = A->cval;
    JX_Real *aval = A->aval;
    
    noValues = (jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues"));
    if (noValues) aval = NULL;
    if (sg == NULL)
    {
        JX_Int i, j;
        JX_Int beg_row = A->beg_row;
        
        jx_fprintf(fp, "\n----- A, unpermuted ------------------------------------\n");
        for (i = 0; i < m; ++ i)
        {
            jx_fprintf(fp, "%i :: ", 1+i+beg_row);
            for (j = rp[i]; j < rp[i+1]; ++ j)
            {
                if (noValues)
                {
                    jx_fprintf(fp, "%i ", 1+cval[j]);
                }
                else
                {
                    jx_fprintf(fp, "%i,%g ; ", 1+cval[j], aval[j]);
                }
            }
            jx_fprintf(fp, "\n");
        }
    }
    else if (jx_np_dh == 1)
    {
        JX_Int i, k, idx = 1;
        JX_Int oldRow;
        
        for (i = 0; i < sg->blocks; ++ i)
        {
            JX_Int oldBlock = sg->n2o_sub[i];
            JX_Int beg_row = sg->beg_row[oldBlock];
            JX_Int end_row = beg_row + sg->row_count[oldBlock];
            
            jx_fprintf(fp, "\n");
            jx_fprintf(fp, "\n----- A, permuted, single mpi task  ------------------\n");
            jx_fprintf(fp, "---- new subdomain: %i;  old subdomain: %i\n", i, oldBlock);
            jx_fprintf(fp, "     old beg_row:   %i;  new beg_row:   %i\n",
                                       sg->beg_row[oldBlock], sg->beg_rowP[oldBlock]);
            jx_fprintf(fp, "     local rows in this block: %i\n", sg->row_count[oldBlock]);
            jx_fprintf(fp, "     bdry rows in this block:  %i\n", sg->bdry_count[oldBlock]);
            jx_fprintf(fp, "     1st bdry row= %i \n", 1+end_row-sg->bdry_count[oldBlock]);
            for (oldRow = beg_row; oldRow < end_row; ++ oldRow)
            {
                JX_Int len = 0, *cval;
                JX_Real *aval;
                
                jx_fprintf(fp, "%3i (old= %3i) :: ", idx, 1+oldRow);
                ++ idx;
                jx_Mat_dhGetRow(A, oldRow, &len, &cval, &aval); JX_CHECK_V_ERROR;
                for (k = 0; k < len; ++ k)
                {
                    if (noValues)
                    {
                        jx_fprintf(fp, "%i ", 1+sg->o2n_col[cval[k]]);
                    }
                    else
                    {
                        jx_fprintf(fp, "%i,%g ; ", 1+sg->o2n_col[cval[k]], aval[k]);
                    }
                }
                jx_fprintf(fp, "\n");
                jx_Mat_dhRestoreRow(A, oldRow, &len, &cval, &aval); JX_CHECK_V_ERROR;
            }
        }
    }
    else
    {
        jx_Hash_i_dh hash = sg->o2n_ext;
        JX_Int *o2n_col = sg->o2n_col, *n2o_row = sg->n2o_row;
        JX_Int beg_row = sg->beg_row[jx_myid_dh];
        JX_Int beg_rowP = sg->beg_rowP[jx_myid_dh];
        JX_Int i, j;
        
        for (i = 0; i < m; ++ i)
        {
            JX_Int row = n2o_row[i];
            
            jx_fprintf(fp, "%3i (old= %3i) :: ", 1+i+beg_rowP, 1+row+beg_row);
            for (j = rp[row]; j < rp[row+1]; ++ j)
            {
                JX_Int col = cval[j];
                
                if (col >= beg_row && col < beg_row+m)
                {
                    col = o2n_col[col-beg_row] + beg_rowP;
                }
                else
                {
                    JX_Int tmp = col;
                    
                    tmp = jx_Hash_i_dhLookup(hash, col); JX_CHECK_V_ERROR;
                    if (tmp == -1)
                    {
                        jx_sprintf(jx_msgBuf_dh, "nonlocal column= %i not in hash table", 1+col);
                        JX_SET_V_ERROR(jx_msgBuf_dh);
                    }
                    else
                    {
                        col = tmp;
                    }
                }
                if (noValues)
                {
                    jx_fprintf(fp, "%i ", 1+col);
                }
                else
                {
                    jx_fprintf(fp, "%i,%g ; ", 1+col, aval[j]);
                }
            }
            jx_fprintf(fp, "\n");
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPrintTriples"
void jx_Mat_dhPrintTriples( jx_Mat_dh A, jx_SubdomainGraph_dh sg, char *filename )
{
    JX_START_FUNC_DH
    JX_Int m = A->m, *rp = A->rp, *cval = A->cval;
    JX_Real *aval = A->aval;
    jx_bool noValues;
    jx_bool matlab;
    FILE *fp;
    
    noValues = (jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues"));
    if (noValues) aval = NULL;
    matlab = (jx_Parser_dhHasSwitch(jx_parser_dh, "-matlab"));
    if (sg == NULL)
    {
        JX_Int i, j, pe;
        JX_Int beg_row = A->beg_row;
        JX_Real val;
        
        for (pe = 0; pe < jx_np_dh; ++ pe)
        {
            jx_MPI_Barrier(jx_comm_dh);
            if (pe == jx_myid_dh)
            {
                if (pe == 0)
                {
                    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
                }
                else
                {
                    fp = jx_openFile_dh(filename, "a"); JX_CHECK_V_ERROR;
                }
                for (i = 0; i < m; ++ i)
                {
                    for (j = rp[i]; j < rp[i+1]; ++ j)
                    {
                        if (noValues)
                        {
                            jx_fprintf(fp, "%i %i\n", 1+i+beg_row, 1+cval[j]);
                        }
                        else
                        {
                            val = aval[j];
                            if (val == 0.0 && matlab) val = _JX_MATLAB_ZERO_;
                            jx_fprintf(fp, JX_TRIPLES_FORMAT, 1+i+beg_row, 1+cval[j], val);
                        }
                    }
                }
                jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
            }
        }
    }
    else if (jx_np_dh == 1)
    {
        JX_Int i, j, k, idx = 1;
        
        fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
        for (i = 0; i < sg->blocks; ++ i)
        {
            JX_Int oldBlock = sg->n2o_sub[i];
            JX_Int beg_row = sg->beg_rowP[oldBlock];
            JX_Int end_row = beg_row + sg->row_count[oldBlock];
            
            for (j = beg_row; j < end_row; ++ j)
            {
                JX_Int len = 0, *cval;
                JX_Real *aval;
                JX_Int oldRow = sg->n2o_row[j];
                
                jx_Mat_dhGetRow(A, oldRow, &len, &cval, &aval); JX_CHECK_V_ERROR;
                if (noValues)
                {
                    for (k = 0; k < len; ++ k)
                    {
                        jx_fprintf(fp, "%i %i\n", idx, 1+sg->o2n_col[cval[k]]);
                    }
                    ++ idx;
                }
                else
                {
                    for (k = 0; k < len; ++ k)
                    {
                        JX_Real val = aval[k];
                        
                        if (val == 0.0 && matlab) val = _JX_MATLAB_ZERO_;
                        jx_fprintf(fp, JX_TRIPLES_FORMAT, idx, 1+sg->o2n_col[cval[k]], val);
                    }
                    ++ idx;
                }
                jx_Mat_dhRestoreRow(A, oldRow, &len, &cval, &aval); JX_CHECK_V_ERROR;
            }
        }
    }
    else
    {
        jx_Hash_i_dh hash = sg->o2n_ext;
        JX_Int *o2n_col = sg->o2n_col, *n2o_row = sg->n2o_row;
        JX_Int beg_row = sg->beg_row[jx_myid_dh];
        JX_Int beg_rowP = sg->beg_rowP[jx_myid_dh];
        JX_Int i, j, pe;
        JX_Int id = sg->o2n_sub[jx_myid_dh];
        
        for (pe = 0; pe < jx_np_dh; ++ pe)
        {
            jx_MPI_Barrier(jx_comm_dh);
            if (id == pe)
            {
                if (pe == 0)
                {
                    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
                }
                else
                {
                    fp = jx_openFile_dh(filename, "a"); JX_CHECK_V_ERROR;
                }
                for (i = 0; i < m; ++ i)
                {
                    JX_Int row = n2o_row[i];
                    
                    for (j = rp[row]; j < rp[row+1]; ++ j)
                    {
                        JX_Int col = cval[j];
                        JX_Real val = 0.0;
                        
                        if (aval != NULL) val = aval[j];
                        if (val == 0.0 && matlab) val = _JX_MATLAB_ZERO_;
                        if (col >= beg_row && col < beg_row+m)
                        {
                            col = o2n_col[col-beg_row] + beg_rowP;
                        }
                        else
                        {
                            JX_Int tmp = col;
                            tmp = jx_Hash_i_dhLookup(hash, col); JX_CHECK_V_ERROR;
                            if (tmp == -1)
                            {
                                jx_sprintf(jx_msgBuf_dh, "nonlocal column= %i not in hash table", 1+col);
                                JX_SET_V_ERROR(jx_msgBuf_dh);
                            }
                            else
                            {
                                col = tmp;
                            }
                        }
                        if (noValues)
                        {
                            jx_fprintf(fp, "%i %i\n", 1+i+beg_rowP, 1+col);
                        }
                        else
                        {
                            jx_fprintf(fp, JX_TRIPLES_FORMAT, 1+i+beg_rowP, 1+col, val);
                        }
                    }
                }
                jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPrintCSR"
void jx_Mat_dhPrintCSR( jx_Mat_dh A, jx_SubdomainGraph_dh sg, char *filename )
{
    JX_START_FUNC_DH
    FILE *fp;
    
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single mpi task");
    }
    if (sg != NULL)
    {
        JX_SET_V_ERROR("not implemented for reordered matrix (jx_SubdomainGraph_dh should be NULL)");
    }
    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
    if (sg == NULL)
    {
        jx_mat_dh_print_csr_private(A->m, A->rp, A->cval, A->aval, fp); JX_CHECK_V_ERROR;
    }
    else
    {
        jx_mat_dh_print_csr_private(A->m, A->rp, A->cval, A->aval, fp); JX_CHECK_V_ERROR;
    }
    jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPrintBIN"
void jx_Mat_dhPrintBIN( jx_Mat_dh A, jx_SubdomainGraph_dh sg, char *filename )
{
    JX_START_FUNC_DH
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single jx_MPI task");
    }
    if (sg != NULL)
    {
        JX_SET_V_ERROR("not implemented for reordering; ensure sg=NULL");
    }
    jx_io_dh_print_ebin_mat_private(A->m, A->beg_row,
             A->rp, A->cval, A->aval, NULL, NULL, NULL, filename); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhReadCSR"
void jx_Mat_dhReadCSR( jx_Mat_dh *mat, char *filename )
{
    JX_START_FUNC_DH
    jx_Mat_dh A;
    FILE *fp;
    
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single jx_MPI task");
    }
    fp = jx_openFile_dh(filename, "r"); JX_CHECK_V_ERROR;
    jx_Mat_dhCreate(&A); JX_CHECK_V_ERROR;
    jx_mat_dh_read_csr_private(&A->m, &A->rp, &A->cval, &A->aval, fp); JX_CHECK_V_ERROR;
    A->n = A->m;
   *mat = A;
    jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhReadTriples"
void jx_Mat_dhReadTriples( jx_Mat_dh *mat, JX_Int ignore, char *filename )
{
    JX_START_FUNC_DH
    FILE *fp = NULL;
    jx_Mat_dh A = NULL;
    
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single jx_MPI task");
    }
    fp = jx_openFile_dh(filename, "r"); JX_CHECK_V_ERROR;
    jx_Mat_dhCreate(&A); JX_CHECK_V_ERROR;
    jx_mat_dh_read_triples_private(ignore, &A->m, &A->rp, &A->cval, &A->aval, fp); JX_CHECK_V_ERROR;
    A->n = A->m;
   *mat = A;
    jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhReadBIN"
void jx_Mat_dhReadBIN( jx_Mat_dh *mat, char *filename )
{
    JX_START_FUNC_DH
    jx_Mat_dh A;
    
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single jx_MPI task");
    }
    jx_Mat_dhCreate(&A); JX_CHECK_V_ERROR;
    jx_io_dh_read_ebin_mat_private(&A->m, &A->rp, &A->cval, &A->aval, filename); JX_CHECK_V_ERROR;
    A->n = A->m;
   *mat = A;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhTranspose"
void jx_Mat_dhTranspose( jx_Mat_dh A, jx_Mat_dh *Bout )
{
    JX_START_FUNC_DH
    jx_Mat_dh B;
    
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only for sequential");
    }
    jx_Mat_dhCreate(&B); JX_CHECK_V_ERROR;
   *Bout = B;
    B->m = B->n = A->m;
    jx_mat_dh_transpose_private(A->m, A->rp, &B->rp, A->cval, &B->cval, A->aval, &B->aval); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhMakeStructurallySymmetric"
void jx_Mat_dhMakeStructurallySymmetric( jx_Mat_dh A )
{
    JX_START_FUNC_DH
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only for sequential");
    }
    jx_make_symmetric_private(A->m, &A->rp, &A->cval, &A->aval); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

void jx_insert_diags_private( jx_Mat_dh A, JX_Int ct );

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhFixDiags"
void jx_Mat_dhFixDiags( jx_Mat_dh A )
{
    JX_START_FUNC_DH
    JX_Int i, j;
    JX_Int *rp = A->rp, *cval = A->cval, m = A->m;
    jx_bool ct = 0;
    JX_Real *aval = A->aval;
    
    for (i = 0; i < m; ++ i)
    {
        jx_bool flag = jx_true;
        
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            JX_Int col = cval[j];
            
            if (col == i)
            {
                flag = jx_false;
                break;
            }
        }
        if (flag) ++ ct;
    }
    if (ct)
    {
        jx_printf("\njx_Mat_dhFixDiags:: %i diags not explicitly present; inserting!\n", ct);
        jx_insert_diags_private(A, ct); JX_CHECK_V_ERROR;
        rp = A->rp;
        cval = A->cval;
        aval = A->aval;
    }
    for (i = 0; i < m; ++ i)
    {
        JX_Real sum = 0.0;
        
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            sum += fabs(aval[j]);
        }
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            if (cval[j] == i)
            {
                aval[j] = sum;
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_insert_diags_private"
void jx_insert_diags_private( jx_Mat_dh A, JX_Int ct )
{
    JX_START_FUNC_DH
    JX_Int *RP = A->rp, *CVAL = A->cval;
    JX_Int *rp, *cval, m = A->m;
    JX_Real *aval, *AVAL = A->aval;
    JX_Int nz = RP[m] + ct;
    JX_Int i, j, idx = 0;
    
    rp = A->rp = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    cval = A->cval = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    aval = A->aval = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    rp[0] = 0;
    for (i = 0; i < m; ++ i)
    {
        jx_bool flag = jx_true;
        
        for (j = RP[i]; j < RP[i+1]; ++ j)
        {
            cval[idx] = CVAL[j];
            aval[idx] = AVAL[j];
            ++ idx;
            if (CVAL[j] == i) flag = jx_false;
        }
        if (flag)
        {
            cval[idx] = i;
            aval[idx] = 0.0;
            ++ idx;
        }
        rp[i+1] = idx;
    }
    JX_FREE_DH(RP); JX_CHECK_V_ERROR;
    JX_FREE_DH(CVAL); JX_CHECK_V_ERROR;
    JX_FREE_DH(AVAL); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPrintDiags"
void jx_Mat_dhPrintDiags( jx_Mat_dh A, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i, j, m = A->m;
    JX_Int *rp = A->rp, *cval = A->cval;
    JX_Real *aval = A->aval;
    
    jx_fprintf(fp, "=================== diagonal elements ====================\n");
    for (i = 0; i < m; ++ i)
    {
        jx_bool flag = jx_true;
        
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            if (cval[j] == i)
            {
                jx_fprintf(fp, "%i  %g\n", i+1, aval[j]);
                flag = jx_false;
                break;
            }
        }
        if (flag)
        {
            jx_fprintf(fp, "%i  ---------- missing\n", i+1);
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhGetRow"
void jx_Mat_dhGetRow( jx_Mat_dh B, JX_Int globalRow, JX_Int *len, JX_Int **ind, JX_Real **val )
{
    JX_START_FUNC_DH
    JX_Int row = globalRow - B->beg_row;
    
    if (row > B->m)
    {
        jx_sprintf(jx_msgBuf_dh, "requested globalRow= %i, which is local row= %i, but only have %i rows!",
                                                         globalRow, row, B->m);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
   *len = B->rp[row+1] - B->rp[row];
    if (ind != NULL) *ind = B->cval + B->rp[row];
    if (val != NULL) *val = B->aval + B->rp[row];
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhRestoreRow"
void jx_Mat_dhRestoreRow( jx_Mat_dh B, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val )
{
    JX_START_FUNC_DH
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhRowPermute"
void jx_Mat_dhRowPermute( jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    if (jx_ignoreMe) JX_SET_V_ERROR("turned off; compilation problem on blue");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPartition"
void jx_build_adj_lists_private( jx_Mat_dh mat, JX_Int **rpOUT, JX_Int **cvalOUT )
{
    JX_START_FUNC_DH
    JX_Int m = mat->m;
    JX_Int *RP = mat->rp, *CVAL = mat->cval;
    JX_Int nz = RP[m];
    JX_Int i, j, *rp, *cval, idx = 0;
    
    rp = *rpOUT = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    cval = *cvalOUT = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    rp[0] = 0;
    for (i = 0; i < m; ++ i)
    {
        for (j = RP[i]; j < RP[i+1]; ++ j)
        {
            JX_Int col = CVAL[j];
            
            if (col != i)
            {
                cval[idx++] = col;
            }
        }
        rp[i+1] = idx;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mat_dhPartition"
void jx_Mat_dhPartition( jx_Mat_dh mat, JX_Int blocks, JX_Int **beg_rowOUT, JX_Int **row_countOUT,  JX_Int **n2oOUT, JX_Int **o2nOUT )
{
    JX_START_FUNC_DH
#ifndef HAVE_METIS_DH
    if (jx_ignoreMe) JX_SET_V_ERROR("not compiled for metis!");
#else
    JX_Int *beg_row, *row_count, *n2o, *o2n, bk, new, *part;
    JX_Int m = mat->m;
    JX_Int i, cutEdgeCount;
    JX_Real zero = 0.0;
    JX_Int metisOpts[5] = {0, 0, 0, 0, 0};
    JX_Int *rp, *cval;
    
    beg_row = *beg_rowOUT = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    row_count = *row_countOUT = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
   *n2oOUT = n2o = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
   *o2nOUT = o2n = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    jx_build_adj_lists_private(mat, &rp, &cval); JX_CHECK_V_ERROR;
    part = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    METIS_PartGraphKway(&m, rp, cval, NULL, NULL, &zero, &zero, &blocks, metisOpts, &cutEdgeCount, part);
    JX_FREE_DH(rp); JX_CHECK_V_ERROR;
    JX_FREE_DH(cval); JX_CHECK_V_ERROR;
    if (mat->debug)
    {
        jx_printf_dh("\nmetis partitioning vector; blocks= %i\n", blocks);
        for (i = 0; i < m; ++ i) jx_printf_dh("  %i %i\n", i+1, part[i]);
    }
    for (i = 0; i < blocks; ++ i) row_count[i] = 0;
    for (i = 0; i < m; ++ i)
    {
        bk = part[i];
        row_count[bk] += 1;
    }
    beg_row[0] = 0;
    for (i = 1; i < blocks; ++ i) beg_row[i] = beg_row[i-1] + row_count[i-1];
    if (mat->debug)
    {
        jx_printf_dh("\nrow_counts: ");
        for (i = 0; i < blocks; ++ i) jx_printf_dh(" %i", row_count[i]);
        jx_printf_dh("\nbeg_row: ");
        for (i = 0; i < blocks; ++ i) jx_printf_dh(" %i", beg_row[i]+1);
        jx_printf_dh("\n");
    }
    JX_Int *tmp = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    memcpy(tmp, beg_row, blocks*sizeof(JX_Int));
    for (i = 0; i < m; ++ i)
    {
        bk = part[i];
        new = tmp[bk];
        tmp[bk] += 1;
        o2n[i] = new;
        n2o[new] = i;
    }
    JX_FREE_DH(tmp);
    JX_FREE_DH(part); JX_CHECK_V_ERROR;
#endif
    JX_END_FUNC_DH
}
