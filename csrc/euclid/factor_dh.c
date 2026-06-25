//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  factor_dh.c
 *  Date: 2013/01/21
 */

#include "jx_euclid.h"

void jx_Factor_dh_junk()
{
}

static void jx_adjust_bj_private( jx_Factor_dh mat );
static void jx_unadjust_bj_private( jx_Factor_dh mat );

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhCreate"
void jx_Factor_dhCreate( jx_Factor_dh *mat )
{
    JX_START_FUNC_DH
    struct _jx_factor_dh *tmp;
    if (jx_np_dh > JX_MAX_MPI_TASKS)
    {
        JX_SET_V_ERROR("you must change JX_MAX_MPI_TASKS and recompile!");
    }
    tmp = (struct _jx_factor_dh *)JX_MALLOC_DH(sizeof(struct _jx_factor_dh)); JX_CHECK_V_ERROR;
   *mat = tmp;
    tmp->m = 0;
    tmp->n = 0;
    tmp->id = jx_myid_dh;
    tmp->beg_row = 0;
    tmp->first_bdry = 0;
    tmp->bdry_count = 0;
    tmp->blockJacobi = jx_false;
    tmp->rp = NULL;
    tmp->cval = NULL;
    tmp->aval = NULL;
    tmp->fill = NULL;
    tmp->diag = NULL;
    tmp->alloc = 0;
    tmp->work_y_lo = tmp->work_x_hi = NULL;
    tmp->sendbufLo = tmp->sendbufHi = NULL;
    tmp->sendindLo = tmp->sendindHi = NULL;
    tmp->num_recvLo = tmp->num_recvHi = 0;
    tmp->num_sendLo = tmp->num_sendHi = 0;
    tmp->sendlenLo = tmp->sendlenHi = 0;
    tmp->solveIsSetup = jx_false;
    tmp->numbSolve = NULL;
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_Factor");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhDestroy"
void jx_Factor_dhDestroy( jx_Factor_dh mat )
{
    JX_START_FUNC_DH
    if (mat->rp != NULL)
    {
        JX_FREE_DH(mat->rp); JX_CHECK_V_ERROR;
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
    if (mat->work_y_lo != NULL)
    {
        JX_FREE_DH(mat->work_y_lo); JX_CHECK_V_ERROR;
    }
    if (mat->work_x_hi != NULL)
    {
        JX_FREE_DH(mat->work_x_hi); JX_CHECK_V_ERROR;
    }
    if (mat->sendbufLo != NULL)
    {
        JX_FREE_DH(mat->sendbufLo); JX_CHECK_V_ERROR;
    }
    if (mat->sendbufHi != NULL)
    {
        JX_FREE_DH(mat->sendbufHi); JX_CHECK_V_ERROR;
    }
    if (mat->sendindLo != NULL)
    {
        JX_FREE_DH(mat->sendindLo); JX_CHECK_V_ERROR;
    }
    if (mat->sendindHi != NULL)
    {
        JX_FREE_DH(mat->sendindHi); JX_CHECK_V_ERROR;
    }
    if (mat->numbSolve != NULL)
    {
        jx_Numbering_dhDestroy(mat->numbSolve); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(mat); JX_CHECK_V_ERROR; 
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_create_fake_mat_private"
static void jx_create_fake_mat_private( jx_Factor_dh mat, jx_Mat_dh *matFakeIN )
{
    JX_START_FUNC_DH
    jx_Mat_dh matFake;
    
    jx_Mat_dhCreate(matFakeIN); JX_CHECK_V_ERROR;
    matFake = *matFakeIN;
    matFake->m = mat->m;
    matFake->n = mat->n;
    matFake->rp = mat->rp;
    matFake->cval = mat->cval;
    matFake->aval = mat->aval;
    matFake->beg_row = mat->beg_row;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_destroy_fake_mat_private"
static void jx_destroy_fake_mat_private( jx_Mat_dh matFake )
{
    JX_START_FUNC_DH
    matFake->rp = NULL;
    matFake->cval = NULL;
    matFake->aval = NULL;
    jx_Mat_dhDestroy(matFake); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhReadNz"
JX_Int jx_Factor_dhReadNz( jx_Factor_dh mat )
{
    JX_START_FUNC_DH
    JX_Int ierr, retval = mat->rp[mat->m];
    JX_Int nz = retval;
    
    ierr = jx_MPI_Allreduce(&nz, &retval, 1, JX_MPI_INT, MPI_SUM, jx_comm_dh); JX_CHECK_MPI_ERROR(ierr);
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhPrintRows"
void jx_Factor_dhPrintRows( jx_Factor_dh mat, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int beg_row = mat->beg_row;
    JX_Int m = mat->m, i, j;
    jx_bool noValues;
    
    noValues = (jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues"));
    if (mat->aval == NULL) noValues = jx_true;
    if (mat->blockJacobi)
    {
        jx_adjust_bj_private(mat); JX_CHECK_V_ERROR;
    }
    jx_fprintf(fp, "\n----------------------- jx_Factor_dhPrintRows ------------------\n");
    if (mat->blockJacobi)
    {
        jx_fprintf(fp, "@@@ Block Jacobi ILU; adjusted values from zero-based @@@\n");
    }
    for (i = 0; i < m; ++ i)
    {
        jx_fprintf(fp, "%i :: ", 1+i+beg_row);
        for (j = mat->rp[i]; j < mat->rp[i+1]; ++ j)
        {
            if (noValues)
            {
                jx_fprintf(fp, "%i ", 1+mat->cval[j]);
            }
            else
            {
                jx_fprintf(fp, "%i,%g ; ", 1+mat->cval[j], mat->aval[j]);
            }
        }
        jx_fprintf(fp, "\n");
    }
    if (mat->blockJacobi)
    {
        jx_unadjust_bj_private(mat); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhPrintDiags"
void jx_Factor_dhPrintDiags( jx_Factor_dh mat, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int beg_row = mat->beg_row;
    JX_Int m = mat->m, i, pe, *diag = mat->diag;
    JX_REAL_DH *aval = mat->aval;
    
    jx_fprintf_dh(fp, "\n----------------------- jx_Factor_dhPrintDiags ------------------\n");
    jx_fprintf_dh(fp, "(grep for 'ZERO')\n");
    for (pe = 0; pe < jx_np_dh; ++ pe)
    {
        jx_MPI_Barrier(jx_comm_dh);
        if (mat->id == pe)
        {
            jx_fprintf(fp, "----- subdomain: %i  processor: %i\n", pe, jx_myid_dh);
            for (i = 0; i < m; ++ i)
            {
                JX_REAL_DH val = aval[diag[i]];
                if (val)
                {
                    jx_fprintf(fp, "%i %g\n", i+1+beg_row, aval[diag[i]]);
                }
                else
                {
                    jx_fprintf(fp, "%i %g ZERO\n", i+1+beg_row, aval[diag[i]]);
                }
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhPrintGraph"
void jx_Factor_dhPrintGraph( jx_Factor_dh mat, char *filename )
{
    JX_START_FUNC_DH
    FILE *fp;
    JX_Int i, j, m = mat->m, *work, *rp = mat->rp, *cval = mat->cval;
    
    if (jx_np_dh > 1) JX_SET_V_ERROR("only implemented for single mpi task");
    work = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i)
    {
        for (j = 0; j < m; ++ j) work[j] = 0;
        for (j = rp[i]; j < rp[i]; ++ j) work[cval[j]] = 1;
        for (j = 0; j < m; ++ j)
        {
            if (work[j])
            {
                jx_fprintf(fp, " x ");
            }
            else
            {
                jx_fprintf(fp, "   ");
            }
        }
        jx_fprintf(fp, "\n");
    }
    jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    JX_FREE_DH(work);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhPrintTriples"
void jx_Factor_dhPrintTriples( jx_Factor_dh mat, char *filename )
{
    JX_START_FUNC_DH
    JX_Int pe, i, j;
    JX_Int m = mat->m, *rp = mat->rp;
    JX_Int beg_row = mat->beg_row;
    JX_REAL_DH *aval = mat->aval;
    jx_bool noValues;
    FILE *fp;
    
    if (mat->blockJacobi)
    {
        jx_adjust_bj_private(mat); JX_CHECK_V_ERROR;
    }
    noValues = (jx_Parser_dhHasSwitch(jx_parser_dh, "-noValues"));
    if (noValues) aval = NULL;
    for (pe = 0; pe < jx_np_dh; ++ pe)
    {
        jx_MPI_Barrier(jx_comm_dh);
        if (mat->id == pe)
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
                        jx_fprintf(fp, "%i %i\n", 1+i+beg_row, 1+mat->cval[j]);
                    }
                    else
                    {
                        jx_fprintf(fp, JX_TRIPLES_FORMAT, 1+i+beg_row, 1+mat->cval[j], aval[j]);
                    }
                }
            }
            jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
        }
    }
    if (mat->blockJacobi)
    {
        jx_unadjust_bj_private(mat); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_setup_receives_private"
static JX_Int jx_setup_receives_private( jx_Factor_dh mat,
                                   JX_Int *beg_rows,
                                   JX_Int *end_rows,
                                   JX_Real *recvBuf,
                                   MPI_Request *req,
                                   JX_Int *reqind,
                                   JX_Int reqlen,
                                   JX_Int *outlist,
                                   jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int i, j, this_pe, num_recv = 0;
    MPI_Request request;
    
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nFACT ========================================================\n");
        jx_fprintf(jx_logFile, "FACT STARTING: jx_setup_receives_private\n");
    }
    for (i = 0; i < reqlen; i = j)
    {
        this_pe = jx_mat_jx_find_owner(beg_rows, end_rows, reqind[i]); JX_CHECK_ERROR(-1);
        for (j = i+1; j < reqlen; j ++)
        {
            JX_Int idx = reqind[j];
            
            if (idx < beg_rows[this_pe] || idx >= end_rows[this_pe])
            {
                break;
            }
        }
        if (debug)
        {
            JX_Int k;
            
            jx_fprintf(jx_logFile, "FACT need nodes from P_%i: ", this_pe);
            for (k = i; k < j; ++ k) jx_fprintf(jx_logFile, "%i ", 1+reqind[k]);
            jx_fprintf(jx_logFile,"\n");
        }
        outlist[this_pe] = j - i;
        jx_MPI_Isend(reqind+i, j-i, JX_MPI_INT, this_pe, 444, jx_comm_dh, &request);
        jx_MPI_Request_free(&request);
        jx_MPI_Recv_init(recvBuf+i, j-i, JX_MPI_REAL, this_pe, 555, jx_comm_dh, req+num_recv);
        ++ num_recv;
    }
    JX_END_FUNC_VAL(num_recv);
}

#undef __FUNC__
#define __FUNC__ "jx_setup_sends_private"
static void jx_setup_sends_private( jx_Factor_dh mat, JX_Int *inlist, JX_Int *o2n_subdomain, jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int i, jLo, jHi, sendlenLo, sendlenHi, first = mat->beg_row;
    MPI_Request *requests = mat->requests, *sendReq;
    MPI_Status *statuses = mat->status;
    jx_bool isHigher;
    JX_Int *rcvBuf;
    JX_Real *sendBuf;
    JX_Int myidNEW = o2n_subdomain[jx_myid_dh];
    JX_Int count;
    
    if (debug)
    {
        jx_fprintf(jx_logFile, "FACT \nSTARTING: jx_setup_sends_private\n");
    }
    sendlenLo = sendlenHi = 0;
    for (i = 0; i < jx_np_dh; i ++)
    {
        if (inlist[i])
        {
            if (o2n_subdomain[i] < myidNEW)
            {
                sendlenLo += inlist[i];
            }
            else
            {
                sendlenHi += inlist[i];
            }
        }
    }
    mat->sendlenLo = sendlenLo;
    mat->sendlenHi = sendlenHi;
    mat->sendbufLo = (JX_Real *)JX_MALLOC_DH(sendlenLo * sizeof(JX_Real)); JX_CHECK_V_ERROR;
    mat->sendbufHi = (JX_Real *)JX_MALLOC_DH(sendlenHi * sizeof(JX_Real)); JX_CHECK_V_ERROR;
    mat->sendindLo = (JX_Int *)JX_MALLOC_DH(sendlenLo * sizeof(JX_Int)); JX_CHECK_V_ERROR;
    mat->sendindHi = (JX_Int *)JX_MALLOC_DH(sendlenHi * sizeof(JX_Int)); JX_CHECK_V_ERROR;
    count = 0;
    jLo = jHi = 0;
    mat->num_sendLo = 0;
    mat->num_sendHi = 0;
    for (i = 0; i < jx_np_dh; i ++)
    {
        if (inlist[i])
        {
            isHigher = (o2n_subdomain[i] < myidNEW) ? jx_false : jx_true;
            if (isHigher)
            {
                rcvBuf = &mat->sendindHi[jHi];
                sendBuf = &mat->sendbufHi[jHi];
                sendReq = &mat->send_reqHi[mat->num_sendHi];
                mat->num_sendHi ++;
                jHi += inlist[i];
            }
            else
            {
                rcvBuf = &mat->sendindLo[jLo];
                sendBuf = &mat->sendbufLo[jLo];
                sendReq = &mat->send_reqLo[mat->num_sendLo];
                mat->num_sendLo ++;
                jLo += inlist[i];
            }
            jx_MPI_Irecv(rcvBuf, inlist[i], JX_MPI_INT, i, 444, jx_comm_dh, requests+count);
            ++ count;
            jx_MPI_Send_init(sendBuf, inlist[i], JX_MPI_REAL, i, 555, jx_comm_dh, sendReq);
        }
    }
    jx_MPI_Waitall(count, requests, statuses);
    if (debug)
    {
        JX_Int j;
        
        jLo = jHi = 0;
        jx_fprintf(jx_logFile, "\nFACT columns that I must send to other subdomains:\n");
        for (i = 0; i < jx_np_dh; i ++)
        {
            if (inlist[i])
            {
                isHigher = (o2n_subdomain[i] < myidNEW) ? jx_false : jx_true;
                if (isHigher)
                {
                    rcvBuf = &mat->sendindHi[jHi];
                    jHi += inlist[i];
                }
                else
                {
                    rcvBuf = &mat->sendindLo[jLo];
                    jLo += inlist[i];
                }
                jx_fprintf(jx_logFile, "FACT  send to P_%i: ", i);
                for (j = 0; j < inlist[i]; ++ j) jx_fprintf(jx_logFile, "%i ", rcvBuf[j]+1);
                jx_fprintf(jx_logFile, "\n");
            }
        }
    }
    for (i = 0; i < mat->sendlenLo; i ++) mat->sendindLo[i] -= first;
    for (i = 0; i < mat->sendlenHi; i ++) mat->sendindHi[i] -= first;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhSolveSetup"
void jx_Factor_dhSolveSetup( jx_Factor_dh mat, jx_SubdomainGraph_dh sg )
{
    JX_START_FUNC_DH
    JX_Int *outlist, *inlist;
    JX_Int i, row, *rp = mat->rp, *cval = mat->cval;
    jx_Numbering_dh numb;
    JX_Int m = mat->m;
    JX_Int *beg_rows = sg->beg_rowP, *row_count = sg->row_count, *end_rows;
    jx_Mat_dh matFake;
    jx_bool debug = jx_false;
    JX_Real *recvBuf;
    
    if (mat->debug && jx_logFile != NULL) debug = jx_true;
    end_rows = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    outlist = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    inlist  = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < jx_np_dh; ++ i)
    {
        inlist[i] = 0;
        outlist[i] = 0;
        end_rows[i] = beg_rows[i] + row_count[i];
    }
    jx_create_fake_mat_private(mat, &matFake); JX_CHECK_V_ERROR;
    jx_Numbering_dhCreate(&(mat->numbSolve)); JX_CHECK_V_ERROR;
    numb = mat->numbSolve;
    jx_Numbering_dhSetup(numb, matFake); JX_CHECK_V_ERROR;
    jx_destroy_fake_mat_private(matFake); JX_CHECK_V_ERROR;
    if (debug)
    {
        jx_fprintf(stderr, "jx_Numbering_dhSetup completed\n");
    }
    i = m + numb->num_ext;
    mat->work_y_lo = (JX_Real *)JX_MALLOC_DH(i*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    mat->work_x_hi = (JX_Real *)JX_MALLOC_DH(i*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    if (debug)
    {
        jx_fprintf(jx_logFile, "FACT num_extLo= %i  num_extHi= %i\n", numb->num_extLo, numb->num_extHi);
    }
    mat->num_recvLo = 0;
    mat->num_recvHi = 0;
    if (numb->num_extLo)
    {
        recvBuf = mat->work_y_lo + m;
        mat->num_recvLo = jx_setup_receives_private(mat, beg_rows, end_rows,
            recvBuf, mat->recv_reqLo, numb->idx_extLo, numb->num_extLo, outlist, debug); JX_CHECK_V_ERROR;
    }
    if (numb->num_extHi)
    {
        recvBuf = mat->work_x_hi + m + numb->num_extLo;
        mat->num_recvHi = jx_setup_receives_private(mat, beg_rows, end_rows,
                    recvBuf, mat->recv_reqHi, numb->idx_extHi, numb->num_extHi, outlist, debug); JX_CHECK_V_ERROR;
    }
    jx_MPI_Alltoall(outlist, 1, JX_MPI_INT, inlist, 1, JX_MPI_INT, jx_comm_dh);
    jx_setup_sends_private(mat, inlist, sg->o2n_sub, debug); JX_CHECK_V_ERROR;
    for (row = 0; row < m; row ++)
    {
        JX_Int len = rp[row+1] - rp[row];
        JX_Int *ind = cval + rp[row];
        
        jx_Numbering_dhGlobalToLocal(numb, len, ind, ind); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(outlist); JX_CHECK_V_ERROR;
    JX_FREE_DH(inlist); JX_CHECK_V_ERROR;
    JX_FREE_DH(end_rows); JX_CHECK_V_ERROR;
    if (debug)
    {
        JX_Int ii, jj;
        
        jx_fprintf(jx_logFile, "\n--------- row/col structure, after global to local renumbering\n");
        for (ii = 0; ii < mat->m; ++ ii)
        {
            jx_fprintf(jx_logFile, "local row %i :: ", ii+1);
            for (jj = mat->rp[ii]; jj < mat->rp[ii+1]; ++ jj)
            {
                jx_fprintf(jx_logFile, "%i ", 1+mat->cval[jj]);
            }
            jx_fprintf(jx_logFile, "\n");
        }
        jx_fprintf(jx_logFile, "\n");
        fflush(jx_logFile);
    }
    JX_END_FUNC_DH
}

static void jx_forward_solve_private( JX_Int m, JX_Int from, JX_Int to,
          JX_Int *rp, JX_Int *cval, JX_Int *diag, JX_Real *aval, JX_Real *rhs, JX_Real *work_y, jx_bool debug );
static void jx_backward_solve_private( JX_Int m, JX_Int from, JX_Int to,
          JX_Int *rp, JX_Int *cval, JX_Int *diag, JX_Real *aval, JX_Real *work_y, JX_Real *work_x, jx_bool debug );
static JX_Int jx_beg_rowG;

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhSolve"
void jx_Factor_dhSolve( JX_Real *rhs, JX_Real *lhs, jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    jx_Factor_dh mat = ctx->F;
    JX_Int from, to;
    JX_Int ierr, i, m = mat->m, first_bdry = mat->first_bdry;
    JX_Int offsetLo = mat->numbSolve->num_extLo;
    JX_Int offsetHi = mat->numbSolve->num_extHi;
    JX_Int *rp = mat->rp, *cval = mat->cval, *diag = mat->diag;
    JX_Real *aval = mat->aval;
    JX_Int *sendindLo = mat->sendindLo, *sendindHi = mat->sendindHi;
    JX_Int sendlenLo = mat->sendlenLo, sendlenHi = mat->sendlenHi;
    JX_Real *sendbufLo = mat->sendbufLo, *sendbufHi = mat->sendbufHi; 
    JX_Real *work_y = mat->work_y_lo;
    JX_Real *work_x = mat->work_x_hi;
    jx_bool debug = jx_false;
    
    if (mat->debug && jx_logFile != NULL) debug = jx_true;
    if (debug) jx_beg_rowG = ctx->F->beg_row;
    if (debug)
    {
        jx_fprintf(jx_logFile, "\n=====================================================\n");
        jx_fprintf(jx_logFile, "FACT jx_Factor_dhSolve: num_recvLo= %i num_recvHi = %i\n", mat->num_recvLo, mat->num_recvHi);
    }
    if (mat->num_recvLo)
    {
        jx_MPI_Startall(mat->num_recvLo, mat->recv_reqLo);
    }
    if (mat->num_recvHi)
    {
        jx_MPI_Startall(mat->num_recvHi, mat->recv_reqHi);
    }
    from = 0;
    to = first_bdry;
    if (from != to)
    {
        jx_forward_solve_private(m, from, to, rp, cval, diag, aval, rhs, work_y, debug); JX_CHECK_V_ERROR;
    }
    if (mat->num_recvLo)
    {
        jx_MPI_Waitall(mat->num_recvLo, mat->recv_reqLo, mat->status);
        if (debug)
        {
            jx_fprintf(jx_logFile, "FACT got 'y' values from lower neighbors; work buffer:\n  ");
            for (i = 0; i < offsetLo; ++ i)
            {
                jx_fprintf(jx_logFile, "%g ", work_y[m+i]);
            }
        }
    }
    from = first_bdry;
    to = m;
    if (from != to)
    {
        jx_forward_solve_private(m, from, to, rp, cval, diag, aval, rhs, work_y, debug); JX_CHECK_V_ERROR;
    }
    if (mat->num_sendHi)
    {
        for (i = 0; i < sendlenHi; i ++)
        {
            sendbufHi[i] = work_y[sendindHi[i]];
        }
        jx_MPI_Startall(mat->num_sendHi, mat->send_reqHi);
        if (debug)
        {
            jx_fprintf(jx_logFile, "\nFACT sending 'y' values to higher neighbor:\nFACT   ");
            for (i = 0; i < sendlenHi; i ++)
            {
                jx_fprintf(jx_logFile, "%g ", sendbufHi[i]);
            }
            jx_fprintf(jx_logFile, "\n");
        }
    }
    if (mat->num_recvHi)
    {
        ierr = jx_MPI_Waitall(mat->num_recvHi, mat->recv_reqHi, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
        if (debug)
        {
            jx_fprintf(jx_logFile, "FACT got 'x' values from higher neighbors:\n  ");
            for (i = m+offsetLo; i < m+offsetLo+offsetHi; ++ i)
            {
                jx_fprintf(jx_logFile, "%g ", work_x[i]);
            }
            jx_fprintf(jx_logFile, "\n");
        }
    }
    from = m;
    to = first_bdry;
    if (from != to)
    {
        jx_backward_solve_private(m, from, to, rp, cval, diag, aval, work_y, work_x, debug); JX_CHECK_V_ERROR;
    }
    if (mat->num_sendLo)
    {
        for (i = 0; i < sendlenLo; i ++)
        {
            sendbufLo[i] = work_x[sendindLo[i]];
        }
        ierr = jx_MPI_Startall(mat->num_sendLo, mat->send_reqLo); JX_CHECK_MPI_V_ERROR(ierr);
        if (debug)
        {
            jx_fprintf(jx_logFile, "\nFACT sending 'x' values to lower neighbor:\nFACT   ");
            for (i = 0; i < sendlenLo; i ++)
            {
                jx_fprintf(jx_logFile, "%g ", sendbufLo[i]);
            }
            jx_fprintf(jx_logFile, "\n");
        }
    }
    from = first_bdry;
    to = 0;
    if (from != to)
    {
        jx_backward_solve_private(m, from, to, rp, cval, diag, aval, work_y, work_x, debug); JX_CHECK_V_ERROR;
    }
    memcpy(lhs, work_x, m*sizeof(JX_Real));
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nFACT solution: ");
        for (i = 0; i < m; ++ i)
        {
            jx_fprintf(jx_logFile, "%g ", lhs[i]);
        }
        jx_fprintf(jx_logFile, "\n");
    }
    if (mat->num_sendLo)
    {
        ierr = jx_MPI_Waitall(mat->num_sendLo, mat->send_reqLo, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
    }
    if (mat->num_sendHi)
    {
        ierr = jx_MPI_Waitall(mat->num_sendHi, mat->send_reqHi, mat->status); JX_CHECK_MPI_V_ERROR(ierr);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_forward_solve_private"
void jx_forward_solve_private( JX_Int m,
                            JX_Int from,
                            JX_Int to,
                            JX_Int *rp,
                            JX_Int *cval,
                            JX_Int *diag,
                            JX_Real *aval,
                            JX_Real *rhs,
                            JX_Real *work_y,
                            jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int i, j, idx;
    
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nFACT starting jx_forward_solve_private; from= %i; to= %i, m= %i\n", 1+from, 1+to, m);
    }
    if (debug)
    {
        for (i = from; i < to; ++ i)
        {
            JX_Int len = diag[i] - rp[i];
            JX_Int *col = cval + rp[i];
            JX_Real *val  = aval + rp[i];
            JX_Real sum = rhs[i];
            
            jx_fprintf(jx_logFile, "FACT   solving for work_y[%i] (global)\n", i+1+jx_beg_rowG);
            jx_fprintf(jx_logFile, "FACT        sum = %g\n", sum);
            for (j = 0; j < len; ++ j)
            {
                idx = col[j];
                sum -= (val[j] * work_y[idx]);
                jx_fprintf(jx_logFile, "FACT        sum(%g) -= val[j] (%g) * work_y[%i] (%g)\n",
                                                                sum, val[j], 1+idx, work_y[idx]);
            }
            work_y[i] = sum;
            jx_fprintf(jx_logFile, "FACT  work_y[%i] = %g\n", 1+i+jx_beg_rowG, work_y[i]);
            jx_fprintf(jx_logFile, "-----------\n");
        }
        jx_fprintf(jx_logFile, "\nFACT   work vector at end of forward solve:\n");
        for (i = 0; i < to; i ++) jx_fprintf(jx_logFile, "    %i %g\n", i+1+jx_beg_rowG, work_y[i]);
    }
    else
    {
        for (i = from; i < to; ++ i)
        {
            JX_Int len = diag[i] - rp[i];
            JX_Int *col = cval + rp[i];
            JX_Real *val = aval + rp[i];
            JX_Real sum = rhs[i];
            
            for (j = 0; j < len; ++ j)
            {
                idx = col[j];
                sum -= (val[j] * work_y[idx]);
            }
            work_y[i] = sum;
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_backward_solve_private"
void jx_backward_solve_private( JX_Int m,
                             JX_Int from,
                             JX_Int to,
                             JX_Int *rp,
                             JX_Int *cval,
                             JX_Int *diag,
                             JX_Real *aval,
                             JX_Real *work_y,
                             JX_Real *work_x,
                             jx_bool debug )
{
    JX_START_FUNC_DH
    JX_Int i, j, idx;
    
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nFACT starting jx_backward_solve_private; from= %i; to= %i, m= %i\n", 1+from, 1+to, m);
        for (i = from-1; i >= to; -- i)
        {
            JX_Int len = rp[i+1] - diag[i] - 1;
            JX_Int *col = cval + diag[i] + 1;
            JX_Real *val = aval + diag[i] + 1;
            JX_Real sum = work_y[i];
            
            jx_fprintf(jx_logFile, "FACT   solving for work_x[%i]\n", i+1+jx_beg_rowG);
            for (j = 0; j < len; ++ j)
            {
                idx = col[j];
                sum -= (val[j] * work_x[idx]);
                jx_fprintf(jx_logFile, "FACT        sum(%g) -= val[j] (%g) * work_x[idx] (%g)\n",
                                                                        sum, val[j], work_x[idx]);
            }
            work_x[i] = sum * aval[diag[i]];
            jx_fprintf(jx_logFile, "FACT   work_x[%i] = %g\n", 1+i, work_x[i]);
            jx_fprintf(jx_logFile, "----------\n");
        }
    }
    else
    {
        for (i = from-1; i >= to; -- i)
        {
            JX_Int len = rp[i+1] - diag[i] - 1;
            JX_Int *col = cval + diag[i] + 1;
            JX_Real *val  = aval + diag[i] + 1;
            JX_Real sum = work_y[i];
            
            for (j = 0; j < len; ++ j)
            {
                idx = col[j];
                sum -= (val[j] * work_x[idx]);
            }
            work_x[i] = sum * aval[diag[i]];
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhInit"
void jx_Factor_dhInit( void *A, jx_bool fillFlag, jx_bool avalFlag, JX_Real rho, JX_Int id, JX_Int beg_rowP, jx_Factor_dh *Fout )
{
    JX_START_FUNC_DH
    JX_Int m, n, beg_row, alloc;
    jx_Factor_dh F;
    
    jx_EuclidGetDimensions(A, &beg_row, &m, &n); JX_CHECK_V_ERROR;
    alloc = rho * m;
    jx_Factor_dhCreate(&F); JX_CHECK_V_ERROR;
   *Fout = F;
    F->m = m;
    F->n = n;
    F->beg_row = beg_rowP;
    F->id = id;
    F->alloc = alloc;
    F->rp = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    F->rp[0] = 0;
    F->cval = (JX_Int *)JX_MALLOC_DH(alloc*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    F->diag = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (fillFlag)
    {
        F->fill = (JX_Int *)JX_MALLOC_DH(alloc*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    }
    if (avalFlag)
    {
        F->aval = (JX_REAL_DH *)JX_MALLOC_DH(alloc*sizeof(JX_REAL_DH)); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhReallocate"
void jx_Factor_dhReallocate( jx_Factor_dh F, JX_Int used, JX_Int additional )
{
    JX_START_FUNC_DH
    JX_Int alloc = F->alloc;
    
    if (used+additional > F->alloc)
    {
        JX_Int *tmpI;
        
        while (alloc < used+additional) alloc *= 2.0;
        F->alloc = alloc;
        tmpI = F->cval;
        F->cval = (JX_Int *)JX_MALLOC_DH(alloc*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        memcpy(F->cval, tmpI, used*sizeof(JX_Int));
        JX_FREE_DH(tmpI); JX_CHECK_V_ERROR;
        if (F->fill != NULL)
        {
            tmpI = F->fill;
            F->fill = (JX_Int *)JX_MALLOC_DH(alloc*sizeof(JX_Int)); JX_CHECK_V_ERROR;
            memcpy(F->fill, tmpI, used*sizeof(JX_Int));
            JX_FREE_DH(tmpI); JX_CHECK_V_ERROR;
        }
        if (F->aval != NULL)
        {
            JX_REAL_DH *tmpF = F->aval;
            F->aval = (JX_REAL_DH *)JX_MALLOC_DH(alloc*sizeof(JX_REAL_DH)); JX_CHECK_V_ERROR;
            memcpy(F->aval, tmpF, used*sizeof(JX_REAL_DH));
            JX_FREE_DH(tmpF); JX_CHECK_V_ERROR;
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhTranspose"
void jx_Factor_dhTranspose( jx_Factor_dh A, jx_Factor_dh *Bout )
{
    JX_START_FUNC_DH
    jx_Factor_dh B;
    
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only for sequential");
    }
    jx_Factor_dhCreate(&B); JX_CHECK_V_ERROR;
   *Bout = B;
    B->m = B->n = A->m;
    if (B->aval == NULL)
    {
        jx_mat_dh_transpose_private(A->m, A->rp, &B->rp, A->cval, &B->cval, A->aval, NULL); JX_CHECK_V_ERROR;
    }
    else
    {
        jx_mat_dh_transpose_private(A->m, A->rp, &B->rp, A->cval, &B->cval, A->aval, &B->aval); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhSolveSeq"
void jx_Factor_dhSolveSeq( JX_Real *rhs, JX_Real *lhs, jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    jx_Factor_dh F = ctx->F;
    JX_Int *rp, *cval, *diag;
    JX_Int i, j, *vi, nz, m = F->m;
    JX_REAL_DH *aval, *work;
    JX_REAL_DH *v, sum;
    jx_bool debug = jx_false;
    
    if (ctx->F->debug && jx_logFile != NULL) debug = jx_true;
    rp = F->rp;
    cval = F->cval;
    aval = F->aval;
    diag = F->diag;
    work = ctx->work;
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nFACT ============================================================\n");
        jx_fprintf(jx_logFile, "FACT starting jx_Factor_dhSolveSeq\n");
        jx_fprintf(jx_logFile, "\nFACT   STARTING FORWARD SOLVE\n------------\n");
        work[0] = rhs[0];
        jx_fprintf(jx_logFile, "FACT   work[0] = %g\n------------\n", work[0]);
        for (i = 1; i < m; i ++)
        {
            v = aval + rp[i];
            vi = cval + rp[i];
            nz = diag[i] - rp[i];
            jx_fprintf(jx_logFile, "FACT   solving for work[%i]\n", i+1);
            sum = rhs[i];
            for (j = 0; j < nz; ++ j)
            {
                sum -= (v[j] * work[vi[j]]);
                jx_fprintf(jx_logFile, "FACT         sum (%g) -= v[j] (%g) * work[vi[j]] (%g)\n", sum, v[j], work[vi[j]]);
            }
            work[i] = sum;
            jx_fprintf(jx_logFile, "FACT   work[%i] = %g\n------------\n", 1+i, work[i]);
        }
        jx_fprintf(jx_logFile, "\nFACT   work vector at end of forward solve:\n");
        for (i = 0; i < m; i ++) jx_fprintf(jx_logFile, "    %i %g\n", i+1, work[i]);
        jx_fprintf(jx_logFile, "\nFACT   STARTING JX_BACKWARD SOLVE\n--------------\n");
        for (i = m-1; i >= 0; i --)
        {
            v = aval + diag[i] + 1;
            vi = cval + diag[i] + 1;
            nz = rp[i+1] - diag[i] - 1;
            jx_fprintf(jx_logFile, "FACT   solving for lhs[%i]\n", i+1);
            sum = work[i];
            for (j = 0; j < nz; ++ j)
            {
                sum -= (v[j] * work[vi[j]]);
                jx_fprintf(jx_logFile, "FACT         sum (%g) -= v[j] (%g) * work[vi[j]] (%g)\n", sum, v[j], work[vi[j]]);
            }
            lhs[i] = work[i] = sum * aval[diag[i]];
            jx_fprintf(jx_logFile, "FACT   lhs[%i] = %g\n------------\n", 1+i, lhs[i]);
            jx_fprintf(jx_logFile, "FACT   solving for lhs[%i]\n", i+1);
        }
        jx_fprintf(jx_logFile, "\nFACT solution: ");
        for (i = 0; i < m; ++ i) jx_fprintf(jx_logFile, "%g ", lhs[i]);
        jx_fprintf(jx_logFile, "\n");
    }
    else
    {
        work[0] = rhs[0];
        for (i = 1; i < m; i ++)
        {
            v = aval + rp[i];
            vi = cval + rp[i];
            nz = diag[i] - rp[i];
            sum = rhs[i];
            while (nz --) sum -= (*v++ * work[*vi++]);
            work[i] = sum;
        }
        for (i = m-1; i >= 0; i --)
        {
            v = aval + diag[i] + 1;
            vi = cval + diag[i] + 1;
            nz = rp[i+1] - diag[i] - 1;
            sum = work[i];
            while (nz --) sum -= (*v++ * work[*vi++]);
            lhs[i] = work[i] = sum*aval[diag[i]];
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_adjust_bj_private"
void jx_adjust_bj_private( jx_Factor_dh mat )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int nz = mat->rp[mat->m];
    JX_Int beg_row = mat->beg_row;
    
    for (i = 0; i < nz; ++ i) mat->cval[i] += beg_row;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_unadjust_bj_private"
void jx_unadjust_bj_private( jx_Factor_dh mat )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int nz = mat->rp[mat->m];
    JX_Int beg_row = mat->beg_row;
    
    for (i = 0; i < nz; ++ i) mat->cval[i] -= beg_row;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhMaxPivotInverse"
JX_Real jx_Factor_dhMaxPivotInverse( jx_Factor_dh mat )
{
    JX_START_FUNC_DH
    JX_Int i, m = mat->m, *diags = mat->diag;
    JX_REAL_DH *aval = mat->aval;
    JX_Real minGlobal = 0.0, min = aval[diags[0]];
    JX_Real retval;
    
    for (i = 0; i < m; ++ i) min = JX_MIN(min, fabs(aval[diags[i]]));
    if (jx_np_dh == 1)
    {
        minGlobal = min;
    }
    else
    {
        jx_MPI_Reduce(&min, &minGlobal, 1, JX_MPI_REAL, MPI_MIN, 0, jx_comm_dh);
    }
    if (minGlobal == 0)
    {
        retval = 0;
    }
    else
    {
        retval = 1.0 / minGlobal;
    }
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhMaxValue"
JX_Real jx_Factor_dhMaxValue( jx_Factor_dh mat )
{
    JX_START_FUNC_DH
    JX_Real maxGlobal = 0.0, max = 0.0;
    JX_Int i, nz = mat->rp[mat->m];
    JX_REAL_DH *aval = mat->aval;
    
    for (i = 0; i < nz; ++ i)
    {
        max = JX_MAX(max, fabs(aval[i]));
    }
    if (jx_np_dh == 1)
    {
        maxGlobal = max;
    }
    else
    {
        jx_MPI_Reduce(&max, &maxGlobal, 1, JX_MPI_REAL, MPI_MAX, 0, jx_comm_dh);
    }
    JX_END_FUNC_VAL(maxGlobal)
}

#undef __FUNC__
#define __FUNC__ "jx_Factor_dhCondEst"
JX_Real jx_Factor_dhCondEst( jx_Factor_dh mat, jx_Euclid_dh ctx )
{
    JX_START_FUNC_DH
    JX_Real max = 0.0, maxGlobal = 0.0;
    JX_Real *x;
    JX_Int i, m = mat->m;
    jx_Vec_dh lhs, rhs;
    
    jx_Vec_dhCreate(&lhs); JX_CHECK_ERROR(-1);
    jx_Vec_dhInit(lhs, m); JX_CHECK_ERROR(-1);
    jx_Vec_dhDuplicate(lhs,&rhs); JX_CHECK_ERROR(-1);
    jx_Vec_dhSet(rhs, 1.0); JX_CHECK_ERROR(-1);
    jx_Euclid_dhApply(ctx, rhs->vals, lhs->vals); JX_CHECK_ERROR(-1);
    x = lhs->vals;
    for (i = 0; i < m; ++ i)
    {
        max = JX_MAX(max, fabs(x[i]));
    }
    if (jx_np_dh == 1)
    {
        maxGlobal = max;
    }
    else
    {
        jx_MPI_Reduce(&max, &maxGlobal, 1, JX_MPI_REAL, MPI_MAX, 0, jx_comm_dh);
    }
    JX_END_FUNC_VAL(maxGlobal)
}
