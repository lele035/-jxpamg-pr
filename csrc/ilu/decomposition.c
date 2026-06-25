//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 * decomposition.c
 *
 * Created by Yue Xiaoqiang 2014/03/24
 * Xiangtan University
 *
 */

#include "jx_ilu.h"

/*!
 * \fn JX_Int jx_ILUZeroDecompositionA
 * \brief ILU(0) decomposition
 * \author Yue Xiaoqiang
 * \date 2014/03/15
 */
JX_Int
jx_ILUZeroDecompositionA( jx_CSRMatrix *A,
                          JX_Int **indexDP_ptr,
                          JX_Int **indexLU_ptr,
                          JX_Real **valueLU_ptr )
{
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Int *JA = jx_CSRMatrixJ(A);
    JX_Real *AA = jx_CSRMatrixData(A);
    JX_Int *indexDP = NULL;
    JX_Int *indexLU = NULL;
    JX_Int *placeRC = NULL;
    JX_Real *valueLU = NULL;
    JX_Int i, j, k, l, m, n, s, rend, send, tend, cnt = 0;
    JX_Real b;
    
    indexDP = jx_CTAlloc(JX_Int, num_rows);
   *indexDP_ptr = indexDP;
    indexLU = jx_CTAlloc(JX_Int, num_nonzeros);
   *indexLU_ptr = indexLU;
    placeRC = jx_CTAlloc(JX_Int, num_rows);
    valueLU = jx_CTAlloc(JX_Real, num_nonzeros);
   *valueLU_ptr = valueLU;
    jx_IntegerArraySetConstantValues(num_rows, placeRC, -1);
    indexDP[0] = 0;
    n = IA[1];
    jx_IntegerArrayCopy(n, JA, indexLU);
    jx_DoubleArrayCopy(n, AA, valueLU);
    for (i = 1; i < num_rows; i ++)
    {
        send = IA[i+1];
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m < i)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexDP[i] = n;
        indexLU[n] = i;
        valueLU[n] = AA[IA[i]];
        placeRC[i] = n;
        n ++;
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m > i)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        rend = indexDP[i];
        k = IA[i];
        while (k < rend)
        {
            s = k;
            l = indexLU[k];
            for (j = k+1; j < rend; j ++)
            {
                if (indexLU[j] < l)
                {
                    s = j;
                    l = indexLU[j];
                }
            }
            if (s != k)
            {
                m = indexLU[k];
                indexLU[k] = indexLU[s];
                indexLU[s] = m;
                placeRC[l] = k;
                placeRC[m] = s;
                b = valueLU[k];
                valueLU[k] = valueLU[s];
                valueLU[s] = b;
            }
            valueLU[k] /= valueLU[indexDP[l]];
            tend = IA[l+1];
            for (j = indexDP[l]+1; j < tend; j ++)
            {
                m = indexLU[j];
                if (placeRC[m] < IA[i])
                {
                    cnt ++; // Need to count as it's dropped
                    continue;
                }
                valueLU[placeRC[m]] -= valueLU[k] * valueLU[j];
            }
            k ++;
        }
    }
    jx_DoubleArrayReciprocalMap(valueLU, num_rows, indexDP);
    jx_TFree(placeRC);
    
    return cnt;
}

/*!
 * \fn JX_Int jx_ILUZeroDecompositionB
 * \brief ILU(0) decomposition
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
jx_ILUZeroDecompositionB( jx_CSRMatrix *A,
                          JX_Real drop_tol,
                          JX_Int **indexAP_ptr,
                          JX_Int **indexDP_ptr,
                          JX_Int **indexLU_ptr,
                          JX_Real **valueLU_ptr )
{
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int num_nonzeros = jx_CSRMatrixNumNonzeros(A);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Int *JA = jx_CSRMatrixJ(A);
    JX_Real *AA = jx_CSRMatrixData(A);
    JX_Int *indexAP = NULL;
    JX_Int *indexDP = NULL;
    JX_Int *indexLU = NULL;
    JX_Int *placeRC = NULL;
    JX_Real *valueLU = NULL;
    JX_Int i, j, k, l, m, n, s, rend, send, tend, cnt = 0;
    JX_Real tmpval, b;
    
    indexAP = jx_CTAlloc(JX_Int, num_rows+1);
   *indexAP_ptr = indexAP;
    indexDP = jx_CTAlloc(JX_Int, num_rows);
   *indexDP_ptr = indexDP;
    indexLU = jx_CTAlloc(JX_Int, num_nonzeros);
   *indexLU_ptr = indexLU;
    placeRC = jx_CTAlloc(JX_Int, num_rows);
    valueLU = jx_CTAlloc(JX_Real, num_nonzeros);
   *valueLU_ptr = valueLU;
    jx_IntegerArraySetConstantValues(num_rows, placeRC, -1);
    n = 0;
    indexAP[0] = n;
    indexDP[0] = 0;
    indexLU[0] = JA[0];
    valueLU[0] = AA[0];
    n ++;
    tmpval = drop_tol * fabs(AA[0]);
    send = IA[1];
    for (j = 1; j < send; j ++)
    {
        if (fabs(AA[j]) > tmpval)
        {
            indexLU[n] = JA[j];
            valueLU[n] = AA[j];
            n ++;
        }
    }
    indexAP[1] = n;
    for (i = 1; i < num_rows; i ++)
    {
        send = IA[i+1];
        tmpval = drop_tol * fabs(AA[IA[i]]);
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if ((m < i) && (fabs(AA[j]) > tmpval))
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexDP[i] = n;
        indexLU[n] = i;
        valueLU[n] = AA[IA[i]];
        placeRC[i] = n;
        n ++;
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if ((m > i) && (fabs(AA[j]) > tmpval))
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexAP[i+1] = n;
        rend = indexDP[i];
        k = indexAP[i];
        while (k < rend)
        {
            s = k;
            l = indexLU[k];
            for (j = k+1; j < rend; j ++)
            {
                if (indexLU[j] < l)
                {
                    s = j;
                    l = indexLU[j];
                }
            }
            if (s != k)
            {
                m = indexLU[k];
                indexLU[k] = indexLU[s];
                indexLU[s] = m;
                placeRC[l] = k;
                placeRC[m] = s;
                b = valueLU[k];
                valueLU[k] = valueLU[s];
                valueLU[s] = b;
            }
            valueLU[k] /= valueLU[indexDP[l]];
            tend = indexAP[l+1];
            for (j = indexDP[l]+1; j < tend; j ++)
            {
                m = indexLU[j];
                if (placeRC[m] < indexAP[i])
                {
                    cnt ++; // Need to count as it's dropped
                    continue;
                }
                valueLU[placeRC[m]] -= valueLU[k] * valueLU[j];
            }
            k ++;
        }
    }
    jx_DoubleArrayReciprocalMap(valueLU, num_rows, indexDP);
    jx_TFree(placeRC);
    
    return cnt;
}

/*!
 * \fn JX_Int *jx_ILUZeroParallelDecompositionA
 * \brief Parallel ILU(0) decomposition used in the case of 'x-y & node-first'
 * \author Yue Xiaoqiang
 * \date 2015/08/25
 */
JX_Int *
jx_ILUZeroParallelDecompositionA( jx_ParCSRMatrix *par_A,
                                  JX_Int **indexDP_ptr,
                                  JX_Int **indexLU_ptr,
                                  JX_Real **valueLU_ptr,
                                  JX_Int *num_nonzeros,
                                  JX_Int *fill_in_drop,
                                  jx_ILUZeroFactorData *ilu_data )
{
    MPI_Comm comm = jx_ParCSRMatrixComm(par_A);
    JX_Int num_rows = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(par_A));
    JX_Int *row_starts = jx_ParCSRMatrixRowStarts(par_A);
    jx_CSRMatrix *ser_B = jx_MergeDiagAndOffd(par_A); // Form the locally rectangular matrix
    
    jx_GridPartitionData *grid_data = ilu_data->par_grid;
    JX_Int x_part_len = grid_data->x_part_len;
    JX_Int y_part_len = grid_data->y_part_len;
    JX_Int x_lower_idx = grid_data->x_lower_idx;
    JX_Int y_lower_idx = grid_data->y_lower_idx;
    JX_Int num_smallside = grid_data->num_smallside;
    JX_Int num_largeside = grid_data->num_largeside; // YUE: Needn't to consider the cross ones in 5-point stencil
    JX_Int *smallprocs = grid_data->sideprocs;
    JX_Int *largeprocs = smallprocs + num_smallside;
    JX_Int *smallprcpos = grid_data->sideprcpos;
    JX_Int *largeprcpos = smallprcpos + num_smallside;
    JX_Int *largeprcxsrt = grid_data->sideprcxsrt + num_smallside;
    JX_Int *largeprcysrt = grid_data->sideprcysrt + num_smallside;
    JX_Int *smallprclength = grid_data->sideprclength;
    JX_Int *largeprclength = smallprclength + num_smallside;
    
    jx_CSRMatrix *ser_A = NULL;
    JX_Int *ser_IA = NULL;
    JX_Int *ret_IA = NULL;
    
    JX_Int *idxDP = NULL;
    JX_Int *indexLU = NULL;
    JX_Int *indexDP = NULL;
    JX_Int *placeRC = NULL;
    JX_Int *tmpindex = NULL;
    JX_Real *valueLU = NULL;
    JX_Real *tmpvalue = NULL;
    
    MPI_Status *status = NULL;
    
    JX_Int ng_pt = ilu_data->num_equns;
    JX_Int ng_pt_mx = ng_pt * x_part_len;
    JX_Int ng_pt_my = ng_pt * y_part_len;
    JX_Int num_procs, my_id, pid, j, k, l, cnt, dnt, ocnt, llj, tstt = 0, ex_len, jnt;
    JX_Int recv_downsrt, recv_leftsrt, recv_nonzeros, recv_downlsrt, recv_leftlsrt;
    JX_Int rel_x, rel_y, rel_row, jend, jjend, int_uprgt_pnt, ey_len, lx_len, ly_len;
    JX_Int dx_len, kx_len, dy_len, postn_a, postn_b, postn_c, dwn_cnum_rows, lft_cnum_rows;
    
    JX_Int *recvleft = NULL;
    JX_Int *recvdown = NULL;
    JX_Int *sendleft = NULL;
    JX_Int *senddown = NULL;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    /* Allocate storage for communication */
    status = jx_CTAlloc(MPI_Status, grid_data->num_nocrossside);
    ilu_data->status = status;
    recvleft = jx_CTAlloc(JX_Int, ng_pt_my);
    recvdown = jx_CTAlloc(JX_Int, ng_pt_mx);
    sendleft = jx_CTAlloc(JX_Int, ng_pt_my);
    senddown = jx_CTAlloc(JX_Int, ng_pt_mx);
    ilu_data->permute = jx_CTAlloc(JX_Int, num_rows);
    ilu_data->senddown = jx_CTAlloc(JX_Real, ng_pt_mx); // YUE: used in CYCLE phase
    recv_nonzeros = 0;
    /* jx_MPI_Irecv and jx_MPI_Isend shouldn't be used here as summations of each offset */
    /* Receive data from DOWN processors, if any. Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            jx_MPI_Recv(recvdown, 1, JX_MPI_INT, smallprocs[pid], my_id*321, comm, &status[pid]);
            recv_nonzeros += recvdown[0];
        }
    }
    /* Receive data from LEFT processors, if any. Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            jend = smallprclength[pid];
            jx_MPI_Recv(recvleft, jend, JX_MPI_INT, smallprocs[pid], my_id*123, comm, &status[pid]);
            for (j = 0; j < jend; j ++)
            {
                recv_nonzeros += recvleft[j];
            }
        }
    }
    /* Fill data, Send data to UP processors, if any. Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            rel_x = largeprcxsrt[pid] - x_lower_idx; // globally -> locally
            rel_y = largeprcysrt[pid] - y_lower_idx;
            rel_row = ng_pt * (rel_y * x_part_len + rel_x); // x-direction firstly
            jend = largeprclength[pid];
            senddown[0] = jx_CSRMatrixI(ser_B)[rel_row+ng_pt*jend] - jx_CSRMatrixI(ser_B)[rel_row];
            jx_MPI_Send(senddown, 1, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*321, comm);
        }
    }
    /* Fill data, Send data to RIGHT processors, if any. Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 3) /* RIGHT */
        {
            rel_x = largeprcxsrt[pid] - x_lower_idx; // globally -> locally
            rel_y = largeprcysrt[pid] - y_lower_idx;
            rel_row = ng_pt * (rel_y * x_part_len + rel_x); // x-direction firstly
            jend = largeprclength[pid];
            sendleft[0] = jx_CSRMatrixI(ser_B)[rel_row+ng_pt] - jx_CSRMatrixI(ser_B)[rel_row];
            for (j = 1; j < jend; j ++)
            {
                rel_row += ng_pt_mx;
                sendleft[j] = jx_CSRMatrixI(ser_B)[rel_row+ng_pt] - jx_CSRMatrixI(ser_B)[rel_row];
            }
            jx_MPI_Send(sendleft, jend, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*123, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    //jx_printf("\n >>> NNZ = %d in Proc %d of %d\n", recv_nonzeros, my_id, num_procs);
    ser_A = jx_CSRMatrixMergeReorderIntUpRtDnLtQuasiBdy(comm, row_starts, ilu_data->permute,
                                                        ser_B, recv_nonzeros, ng_pt, grid_data,
                                                        &ex_len, &ey_len, &dx_len, &kx_len, &dy_len,
                                                        &lx_len, &ly_len, &postn_a, &postn_b, &postn_c);
    ilu_data->ex_len = ex_len; // YUE: offset of ser_A->I
    ilu_data->ey_len = ey_len; // YUE: number of quasi-boundary points
    ilu_data->dx_len = dx_len;
    ilu_data->kx_len = kx_len;
    ilu_data->dy_len = dy_len;
    ilu_data->lx_len = lx_len;
    ilu_data->ly_len = ly_len;
    ilu_data->pos_int_end = postn_a; // YUE: end of TYPE 1 elements
    ilu_data->pos_dwn_end = postn_b; // YUE: end of TYPE 1 and 2 elements
    ilu_data->pos_lft_end = postn_c; // YUE: end of TYPE 1, 2 and 3 elements
   *num_nonzeros = jx_CSRMatrixNumNonzeros(ser_A);
    jx_CSRMatrixDestroy(ser_B);
    /* Here needn't to check the diagonal-first property of ser_A,
      as it is guaranteed by the algorithm. Yue Xiaoqiang, 2014/08/26 */
    //jx_MPI_Barrier(comm);
    //exit(0);
    indexDP = jx_CTAlloc(JX_Int, jx_CSRMatrixNumRows(ser_A));
   *indexDP_ptr = indexDP;
    indexLU = jx_CTAlloc(JX_Int, jx_CSRMatrixNumNonzeros(ser_A));
   *indexLU_ptr = indexLU;
    tmpindex = jx_CTAlloc(JX_Int, jx_CSRMatrixNumNonzeros(ser_A));
    placeRC = jx_CTAlloc(JX_Int, jx_CSRMatrixNumCols(ser_A));
    jx_IntegerArraySetConstantValues(jx_CSRMatrixNumCols(ser_A), placeRC, -1);
    valueLU = jx_CTAlloc(JX_Real, jx_CSRMatrixNumNonzeros(ser_A));
   *valueLU_ptr = valueLU;
    tmpvalue = jx_CTAlloc(JX_Real, jx_CSRMatrixNumNonzeros(ser_A));
    ser_IA = jx_CSRMatrixI(ser_A) + ex_len;
    idxDP = indexDP + ex_len;
    int_uprgt_pnt = num_rows - ey_len;
    /* Count the number of dropped fill-in in ILU decomposition */
    //jx_assert(postn_a == int_uprgt_pnt);
    /* TYPE 1. ILU factorization of interior, up and right quasi-boundary points. Yue Xiaoqiang, 2014/08/26 */
   *fill_in_drop = jx_ILUZeroLocalDecompositionIntURPntsA(ser_IA, jx_CSRMatrixJ(ser_A),
                                                          jx_CSRMatrixData(ser_A), indexLU, idxDP,
                                                          placeRC, valueLU, int_uprgt_pnt, row_starts[my_id]);
    /* Send LU parts of up and right quasi-boundary points. Yue Xiaoqiang, 2014/08/26 */
    jx_CSRMatrixI(ser_A)[0] = dnt = cnt = 0;
    recv_downsrt = recv_leftsrt = recv_downlsrt = 0;
    /* Receive data from down processors */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            jend = (smallprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            ocnt = cnt;
            jx_MPI_Recv(&recvdown[recv_downsrt], jend, JX_MPI_INT, smallprocs[pid], my_id*331, comm, &status[pid]);
            for (j = 0; j < jend; j ++)
            {
                jx_CSRMatrixI(ser_A)[cnt+1] = jx_CSRMatrixI(ser_A)[cnt] + recvdown[recv_downsrt+j];
                cnt ++;
            }
            tstt = jx_CSRMatrixI(ser_A)[cnt] - jx_CSRMatrixI(ser_A)[ocnt];
            jx_MPI_Recv(&recvdown[recv_downsrt], jend, JX_MPI_INT, smallprocs[pid], my_id*341, comm, &status[pid]);
            llj = jx_CSRMatrixI(ser_A)[ocnt];
            for (j = 0; j < jend; j ++)
            {
                indexDP[dnt++] = recvdown[recv_downsrt+j] + llj;
            }
            recv_downsrt += jend; // YUE: aims to loop for pid
            jx_MPI_Recv(&indexLU[recv_downlsrt], tstt, JX_MPI_INT, smallprocs[pid], my_id*131, comm, &status[pid]);
            jx_MPI_Recv(&valueLU[recv_downlsrt], tstt, JX_MPI_REAL, smallprocs[pid], my_id*141, comm, &status[pid]);
            recv_downlsrt += tstt;
        }
    }
    dwn_cnum_rows = cnt;
    recv_leftlsrt = recv_downlsrt;
    /* Receive data from left processors */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            jjend = smallprclength[pid] - 1; // YUE: exclude the first block
            jend = jjend * ng_pt;
            ocnt = cnt;
            jx_MPI_Recv(&recvleft[recv_leftsrt], jend, JX_MPI_INT, smallprocs[pid], my_id*351, comm, &status[pid]);
            for (j = 0; j < jend; j ++)
            {
                jx_CSRMatrixI(ser_A)[cnt+1] = jx_CSRMatrixI(ser_A)[cnt] + recvleft[recv_leftsrt+j];
                cnt ++;
            }
            tstt = jx_CSRMatrixI(ser_A)[cnt] - jx_CSRMatrixI(ser_A)[ocnt];
            jx_MPI_Recv(&recvleft[recv_leftsrt], jend, JX_MPI_INT, smallprocs[pid], my_id*361, comm, &status[pid]);
            for (j = 0; j < jjend; j ++)
            {
                llj = jx_CSRMatrixI(ser_A)[ocnt];
                ocnt += ng_pt;
                for (k = 0; k < ng_pt; k ++)
                {
                    indexDP[dnt++] = recvleft[recv_leftsrt] + llj;
                    recv_leftsrt ++; // YUE: aims to loop for pid
                }
            }
            jx_MPI_Recv(&indexLU[recv_leftlsrt], tstt, JX_MPI_INT, smallprocs[pid], my_id*151, comm, &status[pid]);
            jx_MPI_Recv(&valueLU[recv_leftlsrt], tstt, JX_MPI_REAL, smallprocs[pid], my_id*161, comm, &status[pid]);
            recv_leftlsrt += tstt;
        }
    }
    lft_cnum_rows = cnt;
    /* Fill data, Send data to up processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            jend = (largeprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            cnt = 0;
            for (j = 0; j < jend; j ++)
            {
                senddown[cnt++] = ser_IA[dy_len+j+1] - ser_IA[dy_len+j];
            }
            jx_MPI_Send(senddown, jend, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*331, comm);
            cnt = 0;
            dnt = ser_IA[dy_len];
            for (j = 0; j < jend; j ++)
            {
                senddown[cnt++] = idxDP[dy_len+j] - dnt;
            }
            jx_MPI_Send(senddown, jend, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*341, comm);
            jx_MPI_Send(&indexLU[ser_IA[dy_len]], ser_IA[dy_len+jend]-ser_IA[dy_len],
                                     JX_MPI_INT, largeprocs[pid], largeprocs[pid]*131, comm);
            jx_MPI_Send(&valueLU[ser_IA[dy_len]], ser_IA[dy_len+jend]-ser_IA[dy_len],
                                  JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*141, comm);
        }
    }
    /* Fill data, Send data to right processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 3) /* RIGHT */
        {
            jjend = largeprclength[pid] - 1; // YUE: exclude the first block
            jend = jjend * ng_pt;
            cnt = 0;
            dnt = dx_len;
            for (j = 0; j < jjend; j ++)
            {
                for (k = 0; k < ng_pt; k ++)
                {
                    sendleft[cnt++] = ser_IA[dnt+k+1] - ser_IA[dnt+k];
                }
                dnt += kx_len;
            }
            jx_MPI_Send(sendleft, jend, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*351, comm);
            jnt = cnt = 0;
            for (j = 0; j < jjend; j ++)
            {
                dnt = ser_IA[dx_len];
                for (k = 0; k < ng_pt; k ++)
                {
                    sendleft[cnt++] = idxDP[dx_len+k] - dnt;
                }
                ocnt = ser_IA[dx_len+ng_pt];
                for (l = ser_IA[dx_len]; l < ocnt; l ++)
                {
                    tmpindex[jnt] = indexLU[l];
                    tmpvalue[jnt++] = valueLU[l];
                }
                dx_len += kx_len;
            }
            jx_MPI_Send(sendleft, jend, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*361, comm);
            jx_MPI_Send(tmpindex, jnt, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*151, comm);
            jx_MPI_Send(tmpvalue, jnt, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*161, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    /* TYPE 2. ILU factorization of down quasi-boundary points. Yue Xiaoqiang, 2014/08/26 */
    jx_ILUZeroLocalDecompositionDPntsA(ser_IA, jx_CSRMatrixJ(ser_A),
                                       jx_CSRMatrixData(ser_A), indexLU,
                                       idxDP, placeRC, valueLU, int_uprgt_pnt,
                                       row_starts[my_id], postn_b, ex_len, fill_in_drop);
    /* TYPE 3. ILU factorization of left quasi-boundary points. Yue Xiaoqiang, 2014/08/26 */
    jx_ILUZeroLocalDecompositionLPntsA(ser_IA, jx_CSRMatrixJ(ser_A),
                                       jx_CSRMatrixData(ser_A), indexLU,
                                       idxDP, placeRC, valueLU, postn_b, row_starts[my_id],
                                       postn_c, dwn_cnum_rows, ex_len, fill_in_drop);
    dnt = cnt = lft_cnum_rows; // YUE: dnt = cnt
    recv_downlsrt = recv_leftlsrt;
    /* Receive data from down processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            ocnt = cnt;
            jx_MPI_Recv(&recvdown[recv_downsrt], ng_pt, JX_MPI_INT, smallprocs[pid], my_id*371, comm, &status[pid]);
            for (j = 0; j < ng_pt; j ++)
            {
                jx_CSRMatrixI(ser_A)[cnt+1] = jx_CSRMatrixI(ser_A)[cnt] + recvdown[recv_downsrt+j];
                cnt ++;
            }
            tstt = jx_CSRMatrixI(ser_A)[cnt] - jx_CSRMatrixI(ser_A)[ocnt];
            jx_MPI_Recv(&recvdown[recv_downsrt], ng_pt, JX_MPI_INT, smallprocs[pid], my_id*381, comm, &status[pid]);
            llj = jx_CSRMatrixI(ser_A)[ocnt];
            for (j = 0; j < ng_pt; j ++)
            {
                indexDP[dnt++] = recvdown[recv_downsrt+j] + llj;
            }
            recv_downsrt += ng_pt; // YUE: aims to loop for pid
            jx_MPI_Recv(&indexLU[recv_downlsrt], tstt, JX_MPI_INT, smallprocs[pid], my_id*171, comm, &status[pid]);
            jx_MPI_Recv(&valueLU[recv_downlsrt], tstt, JX_MPI_REAL, smallprocs[pid], my_id*181, comm, &status[pid]);
            recv_downlsrt += tstt;
        }
    }
    dwn_cnum_rows = cnt;
    recv_leftlsrt = recv_downlsrt;
    /* Receive data from left processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            ocnt = cnt;
            jx_MPI_Recv(&recvleft[recv_leftsrt], ng_pt, JX_MPI_INT, smallprocs[pid], my_id*311, comm, &status[pid]);
            for (j = 0; j < ng_pt; j ++)
            {
                jx_CSRMatrixI(ser_A)[cnt+1] = jx_CSRMatrixI(ser_A)[cnt] + recvleft[recv_leftsrt+j];
                cnt ++;
            }
            tstt = jx_CSRMatrixI(ser_A)[cnt] - jx_CSRMatrixI(ser_A)[ocnt];
            jx_MPI_Recv(&recvleft[recv_leftsrt], ng_pt, JX_MPI_INT, smallprocs[pid], my_id*391, comm, &status[pid]);
            llj = jx_CSRMatrixI(ser_A)[ocnt];
            for (j = 0; j < ng_pt; j ++)
            {
                indexDP[dnt++] = recvleft[recv_leftsrt+j] + llj;
            }
            recv_leftsrt += ng_pt; // YUE: aims to loop for pid
            jx_MPI_Recv(&indexLU[recv_leftlsrt], tstt, JX_MPI_INT, smallprocs[pid], my_id*111, comm, &status[pid]);
            jx_MPI_Recv(&valueLU[recv_leftlsrt], tstt, JX_MPI_REAL, smallprocs[pid], my_id*191, comm, &status[pid]);
            recv_leftlsrt += tstt;
        }
    }
    /* Fill data, Send data to up processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            cnt = 0;
            for (j = 0; j < ng_pt; j ++)
            {
                senddown[cnt++] = ser_IA[ly_len+j+1] - ser_IA[ly_len+j];
            }
            jx_MPI_Send(senddown, ng_pt, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*371, comm);
            cnt = 0;
            dnt = ser_IA[ly_len];
            for (j = 0; j < ng_pt; j ++)
            {
                senddown[cnt++] = idxDP[ly_len+j] - dnt;
            }
            jx_MPI_Send(senddown, ng_pt, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*381, comm);
            jx_MPI_Send(&indexLU[ser_IA[ly_len]], ser_IA[ly_len+ng_pt]-ser_IA[ly_len],
                                      JX_MPI_INT, largeprocs[pid], largeprocs[pid]*171, comm);
            jx_MPI_Send(&valueLU[ser_IA[ly_len]], ser_IA[ly_len+ng_pt]-ser_IA[ly_len],
                                   JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*181, comm);
        }
    }
    /* Fill data, Send data to right processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 3) /* RIGHT */
        {
            cnt = 0;
            for (j = 0; j < ng_pt; j ++)
            {
                sendleft[cnt++] = ser_IA[lx_len+j+1] - ser_IA[lx_len+j];
            }
            jx_MPI_Send(sendleft, ng_pt, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*311, comm);
            cnt = 0;
            dnt = ser_IA[lx_len];
            for (j = 0; j < ng_pt; j ++)
            {
                sendleft[cnt++] = idxDP[lx_len+j] - dnt;
            }
            jx_MPI_Send(sendleft, ng_pt, JX_MPI_INT, largeprocs[pid], largeprocs[pid]*391, comm);
            jx_MPI_Send(&indexLU[ser_IA[lx_len]], ser_IA[lx_len+ng_pt]-ser_IA[lx_len],
                                      JX_MPI_INT, largeprocs[pid], largeprocs[pid]*111, comm);
            jx_MPI_Send(&valueLU[ser_IA[lx_len]], ser_IA[lx_len+ng_pt]-ser_IA[lx_len],
                                   JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*191, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    /* TYPE 4. ILU factorization of LD-Corner quasi-boundary points. Yue Xiaoqiang, 2014/08/26 */
    if (my_id != 0)
    {
        jx_ILUZeroLocalDecompositionLDPntsA(ser_IA, jx_CSRMatrixJ(ser_A), jx_CSRMatrixData(ser_A),
                                            indexLU, idxDP, placeRC, valueLU, postn_c,
                                            row_starts[my_id], row_starts[my_id-1], num_rows,
                                            lft_cnum_rows, dwn_cnum_rows, ex_len, fill_in_drop);
    }
    jx_DoubleArrayReciprocalMap(valueLU, num_rows, idxDP);
    jx_TFree(recvleft);
    jx_TFree(recvdown);
    jx_TFree(sendleft);
    jx_TFree(senddown);
    jx_TFree(tmpindex);
    jx_TFree(placeRC);
    jx_TFree(tmpvalue);
    ret_IA = jx_CSRMatrixI(ser_A);
    /* Destroy ser_A, but IA still persists. Yue Xiaoqiang, 2014/04/27 */
    if (jx_CSRMatrixRownnz(ser_A))
    {
        jx_TFree(jx_CSRMatrixRownnz(ser_A));
    }
    if (jx_CSRMatrixOwnsData(ser_A))
    {
        jx_TFree(jx_CSRMatrixData(ser_A));
        jx_TFree(jx_CSRMatrixJ(ser_A));
    }
    jx_TFree(ser_A);
    
    return ret_IA;
}

/*!
 * \fn JX_Int jx_ILUZeroLocalDecompositionIntURPntsA
 * \brief ILU(0) decomposition for Int-U-R points
 * \author Yue Xiaoqiang
 * \date 2014/08/25
 */
JX_Int
jx_ILUZeroLocalDecompositionIntURPntsA( JX_Int *IA,
                                        JX_Int *JA,
                                        JX_Real *AA,
                                        JX_Int *indexLU,
                                        JX_Int *indexDP,
                                        JX_Int *placeRC,
                                        JX_Real *valueLU,
                                        JX_Int int_uprgt_pnt,
                                        JX_Int first_row_idx )
{
    JX_Int current_row_idx = first_row_idx;
    JX_Int i, j, k, l, m, n, s, rend, send, tend, cnt = 0;
    JX_Real b;
    
    /* Attention: IA[0] != 0, and only IA, indexDP have their offsets. Xiaoqiang Yue 2014/04/27 */
    indexDP[0] = IA[0];
    jx_IntegerArrayCopy(IA[1]-IA[0], &JA[IA[0]], &indexLU[IA[0]]);
    jx_DoubleArrayCopy(IA[1]-IA[0], &AA[IA[0]], &valueLU[IA[0]]);
    n = IA[1];
    for (i = 1; i < int_uprgt_pnt; i ++)
    {
        current_row_idx ++; // global row index
        send = IA[i+1];
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m < current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexDP[i] = n;
        indexLU[n] = current_row_idx;
        valueLU[n] = AA[IA[i]];
        placeRC[current_row_idx] = n;
        n ++;
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m > current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        rend = indexDP[i];
        k = IA[i];
        while (k < rend)
        {
            s = k;
            l = indexLU[k];
            for (j = k+1; j < rend; j ++)
            {
                if (indexLU[j] < l)
                {
                    s = j;
                    l = indexLU[j];
                }
            }
            if (s != k)
            {
                m = indexLU[k];
                indexLU[k] = indexLU[s];
                indexLU[s] = m;
                placeRC[l] = k;
                placeRC[m] = s;
                b = valueLU[k];
                valueLU[k] = valueLU[s];
                valueLU[s] = b;
            }
            l -= first_row_idx;
            valueLU[k] /= valueLU[indexDP[l]];
            tend = IA[l+1];
            for (j = indexDP[l]+1; j < tend; j ++)
            {
                m = indexLU[j];
                if (placeRC[m] < IA[i])
                {
                    cnt ++; // Need to count as it's dropped
                    continue;
                }
                valueLU[placeRC[m]] -= valueLU[k] * valueLU[j];
            }
            k ++;
        }
    }
    
    return cnt;
}

/*!
 * \fn void jx_ILUZeroLocalDecompositionDPntsA
 * \brief ILU(0) decomposition for D points
 * \author Yue Xiaoqiang
 * \date 2014/08/15
 */
void
jx_ILUZeroLocalDecompositionDPntsA( JX_Int *IA,
                                    JX_Int *JA,
                                    JX_Real *AA,
                                    JX_Int *indexLU,
                                    JX_Int *indexDP,
                                    JX_Int *placeRC,
                                    JX_Real *valueLU,
                                    JX_Int int_uprgt_pnt,
                                    JX_Int first_row_idx,
                                    JX_Int num_rows,
                                    JX_Int ex_len,
                                    JX_Int *fill_in_drop )
{
    JX_Int current_row_idx = first_row_idx + int_uprgt_pnt - 1;
    JX_Int i, j, k, l, m, n, s, rend, send, tend, sstmp = -ex_len;
    JX_Real b;
    
    /* Attention: Only IA, indexDP have their offsets. Xiaoqiang Yue 2014/04/27 */
    n = IA[int_uprgt_pnt];
    for (i = int_uprgt_pnt; i < num_rows; i ++)
    {
        current_row_idx ++;
        send = IA[i+1];
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m < current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexDP[i] = n;
        indexLU[n] = current_row_idx;
        valueLU[n] = AA[IA[i]];
        placeRC[current_row_idx] = n;
        n ++;
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m > current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        rend = indexDP[i];
        k = IA[i];
        while (k < rend)
        {
            s = k;
            l = indexLU[k];
            for (j = k+1; j < rend; j ++)
            {
                if (indexLU[j] < l)
                {
                    s = j;
                    l = indexLU[j];
                }
            }
            if (s != k)
            {
                m = indexLU[k];
                indexLU[k] = indexLU[s];
                indexLU[s] = m;
                placeRC[l] = k;
                placeRC[m] = s;
                b = valueLU[k];
                valueLU[k] = valueLU[s];
                valueLU[s] = b;
            }
            l -= first_row_idx;
            if (l < 0) /* More consideration needed here, Yue Xiaoqiang, 2014/08/15 */
            {
                l = sstmp; // YUE: only one non-zero off-diagonal column
                sstmp ++;
            }
            valueLU[k] /= valueLU[indexDP[l]];
            tend = IA[l+1];
            for (j = indexDP[l]+1; j < tend; j ++)
            {
                m = indexLU[j];
                if (placeRC[m] < IA[i])
                {
                  (*fill_in_drop) ++; // Need to count as it's dropped
                    continue;
                }
                valueLU[placeRC[m]] -= valueLU[k] * valueLU[j];
            }
            k ++;
        }
    }
}

/*!
 * \fn void jx_ILUZeroLocalDecompositionLPntsA
 * \brief ILU(0) decomposition for L points
 * \author Yue Xiaoqiang
 * \date 2014/08/16
 */
void
jx_ILUZeroLocalDecompositionLPntsA( JX_Int *IA,
                                    JX_Int *JA,
                                    JX_Real *AA,
                                    JX_Int *indexLU,
                                    JX_Int *indexDP,
                                    JX_Int *placeRC,
                                    JX_Real *valueLU,
                                    JX_Int int_uprgtdwn_pnt,
                                    JX_Int first_row_idx,
                                    JX_Int num_rows,
                                    JX_Int dwn_num_rows,
                                    JX_Int ex_len,
                                    JX_Int *fill_in_drop )
{
    JX_Int current_row_idx = first_row_idx + int_uprgtdwn_pnt - 1;
    JX_Int sstmp = dwn_num_rows - ex_len;
    JX_Int i, j, k, l, m, n, s, rend, send, tend;
    JX_Real b;
    
    /* Attention: Only IA, indexDP have their offsets. Xiaoqiang Yue 2014/04/27 */
    n = IA[int_uprgtdwn_pnt];
    for (i = int_uprgtdwn_pnt; i < num_rows; i ++)
    {
        current_row_idx ++;
        send = IA[i+1];
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m < current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexDP[i] = n;
        indexLU[n] = current_row_idx;
        valueLU[n] = AA[IA[i]];
        placeRC[current_row_idx] = n;
        n ++;
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m > current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        rend = indexDP[i];
        k = IA[i];
        while (k < rend)
        {
            s = k;
            l = indexLU[k];
            for (j = k+1; j < rend; j ++)
            {
                if (indexLU[j] < l)
                {
                    s = j;
                    l = indexLU[j];
                }
            }
            if (s != k)
            {
                m = indexLU[k];
                indexLU[k] = indexLU[s];
                indexLU[s] = m;
                placeRC[l] = k;
                placeRC[m] = s;
                b = valueLU[k];
                valueLU[k] = valueLU[s];
                valueLU[s] = b;
            }
            l -= first_row_idx;
            if (l < 0) /* More consideration needed here, Yue Xiaoqiang, 2014/08/15 */
            {
                l = sstmp; // YUE: only one non-zero off-diagonal column
                sstmp ++;
            }
            valueLU[k] /= valueLU[indexDP[l]];
            tend = IA[l+1];
            for (j = indexDP[l]+1; j < tend; j ++)
            {
                m = indexLU[j];
                if (placeRC[m] < IA[i])
                {
                  (*fill_in_drop) ++; // Need to count as it's dropped
                    continue;
                }
                valueLU[placeRC[m]] -= valueLU[k] * valueLU[j];
            }
            k ++;
        }
    }
}

/*!
 * \fn void jx_ILUZeroLocalDecompositionLDPntsA
 * \brief ILU(0) decomposition for LD-Corner points
 * \author Yue Xiaoqiang
 * \date 2014/08/17
 */
void
jx_ILUZeroLocalDecompositionLDPntsA( JX_Int *IA,
                                     JX_Int *JA,
                                     JX_Real *AA,
                                     JX_Int *indexLU,
                                     JX_Int *indexDP,
                                     JX_Int *placeRC,
                                     JX_Real *valueLU,
                                     JX_Int int_urdl_pnt,
                                     JX_Int first_row_idx,
                                     JX_Int fst_row_idx,
                                     JX_Int num_rows,
                                     JX_Int lft_cnum_rows,
                                     JX_Int dwn_cnum_rows,
                                     JX_Int ex_len,
                                     JX_Int *fill_in_drop )
{
    JX_Int current_row_idx = first_row_idx + int_urdl_pnt - 1;
    JX_Int sstmp = lft_cnum_rows - ex_len;
    JX_Int kktmp = dwn_cnum_rows - ex_len;
    JX_Int i, j, k, l, m, n, s, rend, send, tend;
    JX_Real b;
    
    /* Attention: Only IA, indexDP have their offsets. Xiaoqiang Yue 2014/04/27 */
    n = IA[int_urdl_pnt];
    for (i = int_urdl_pnt; i < num_rows; i ++)
    {
        current_row_idx ++;
        send = IA[i+1];
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m < current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        indexDP[i] = n;
        indexLU[n] = current_row_idx;
        valueLU[n] = AA[IA[i]];
        placeRC[current_row_idx] = n;
        n ++;
        for (j = IA[i]+1; j < send; j ++)
        {
            m = JA[j];
            if (m > current_row_idx)
            {
                indexLU[n] = m;
                valueLU[n] = AA[j];
                placeRC[m] = n;
                n ++;
            }
        }
        rend = indexDP[i];
        k = IA[i];
        while (k < rend)
        {
            s = k;
            l = indexLU[k];
            for (j = k+1; j < rend; j ++)
            {
                if (indexLU[j] < l)
                {
                    s = j;
                    l = indexLU[j];
                }
            }
            if (s != k)
            {
                m = indexLU[k];
                indexLU[k] = indexLU[s];
                indexLU[s] = m;
                placeRC[l] = k;
                placeRC[m] = s;
                b = valueLU[k];
                valueLU[k] = valueLU[s];
                valueLU[s] = b;
            }
            l -= first_row_idx;
            if (l < 0) /* More consideration needed here, Yue Xiaoqiang, 2014/08/15 */
            {
                // YUE: two non-zero off-diagonal column
                if (l+first_row_idx < fst_row_idx) // DOWN
                {
                    l = sstmp;
                    sstmp ++;
                }
                else // LEFT
                {
                    l = kktmp;
                    kktmp ++;
                }
            }
            valueLU[k] /= valueLU[indexDP[l]];
            tend = IA[l+1];
            for (j = indexDP[l]+1; j < tend; j ++)
            {
                m = indexLU[j];
                if (placeRC[m] < IA[i])
                {
                  (*fill_in_drop) ++; // Need to count as it's dropped
                    continue;
                }
                valueLU[placeRC[m]] -= valueLU[k] * valueLU[j];
            }
            k ++;
        }
    }
}

/*------------------------------------------------------------*/
/*------------------------ End of File -----------------------*/
/*------------------------------------------------------------*/
