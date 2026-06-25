//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  ilucycle.c
 *
 * Created by Yue Xiaoqiang 2014/03/24
 * Xiangtan University
 *
 */

#include "jx_ilu.h"

/*!
 * \fn jx_ILUZeroFactorDataPrecond
 * \brief Precond
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
jx_ILUZeroFactorDataPrecond( void            *ilu_vdata,
                             jx_ParCSRMatrix *par_A,
                             jx_ParVector    *par_b,
                             jx_ParVector    *par_x )
{
    MPI_Comm comm = jx_ParCSRMatrixComm(par_A);
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    jx_CSRMatrix *A = NULL;
    jx_Vector *f = NULL;
    jx_Vector *u = NULL;
    JX_Int num_procs, my_id;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    
    if (num_procs == 1)
    {
        A = jx_ParCSRMatrixDiag(par_A);
        f = jx_ParVectorLocalVector(par_b);
        u = jx_ParVectorLocalVector(par_x);
        //jx_printf(" >>> Precond: Drop-Tol = %le on ID = %d of %d\n", ilu_data->drop_tol, my_id, num_procs);
        if (ilu_data->drop_tol == 0.0)
        {
            jx_ILUZeroFactorDataCycleA(ilu_data, A, f, u);
        }
        else
        {
            jx_ILUZeroFactorDataCycleB(ilu_data, A, f, u);
        }
    }
    else if (ilu_data->matA != NULL) /* Still Serial ILU(0), Yue Xiaoqiang, 2014/04/18 */
    {
        f = jx_ParVectorToVectorAll(par_b);
        if (my_id == 0)
        {
            A = ilu_data->matA;
            //jx_printf(" >>> Precond: Drop-Tol = %le on ID = %d of %d\n", ilu_data->drop_tol, my_id, num_procs);
            if (ilu_data->drop_tol == 0.0)
            {
                jx_ILUZeroFactorDataCycleA(ilu_data, A, f, ilu_data->tmp_vec);
            }
            else
            {
                jx_ILUZeroFactorDataCycleB(ilu_data, A, f, ilu_data->tmp_vec);
            }
        }
        jx_VectorToParVector_Allocated2(comm, ilu_data->tmp_vec, jx_ParVectorPartitioning(par_x), par_x);
        jx_SeqVectorDestroy(f);
    }
    else if (ilu_data->drop_tol == 0.0)
    {
        jx_ILUZeroFactorDataParallelCycleA(ilu_data, par_A, par_b, par_x);
    }
    else
    {
        jx_printf("\n >>> Wrong parameters in ILU Cycle\n");
        exit(0);
    }
    
    return 0;
}

/*!
 * \fn void jx_ILUZeroFactorDataCycleA
 * \brief Cycle
 * \author Yue Xiaoqiang
 * \date 2014/03/16
 */
void
jx_ILUZeroFactorDataCycleA( void *ilu_vdata, jx_CSRMatrix *A, jx_Vector *f, jx_Vector *u )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    JX_Int num_rows = jx_VectorSize(u);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Real *f_data = jx_VectorData(f);
    JX_Real *u_data = jx_VectorData(u);
    JX_Int *indexDP = ilu_data->indexD;
    JX_Int *indexLU = ilu_data->index;
    JX_Real *work = jx_VectorData(ilu_data->aux_vec);
    JX_Real *valueLU = ilu_data->value;
    JX_Int i, j, row_end = 0;
    JX_Real tmp;
    
    work[0] = f_data[0];
    for (i = 1; i < num_rows; i ++)
    {
        tmp = f_data[i];
        row_end = indexDP[i];
        for (j = IA[i]; j < row_end; j ++)
        {
            tmp -= valueLU[j] * work[indexLU[j]];
        }
        work[i] = tmp;
    }
    u_data[num_rows-1] = work[num_rows-1] * valueLU[row_end];
    for (i = num_rows-2; i > -1; i --)
    {
        tmp = work[i];
        row_end = IA[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            tmp -= valueLU[j] * u_data[indexLU[j]];
        }
        u_data[i] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataCycleB
 * \brief Cycle
 * \author Yue Xiaoqiang
 * \date 2014/03/16
 */
void
jx_ILUZeroFactorDataCycleB( void *ilu_vdata, jx_CSRMatrix *A, jx_Vector *f, jx_Vector *u )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    JX_Int num_rows = jx_VectorSize(u);
    JX_Int *indexAP = ilu_data->indexA;
    JX_Real *f_data = jx_VectorData(f);
    JX_Real *u_data = jx_VectorData(u);
    JX_Int *indexDP = ilu_data->indexD;
    JX_Int *indexLU = ilu_data->index;
    JX_Real *work = jx_VectorData(ilu_data->aux_vec);
    JX_Real *valueLU = ilu_data->value;
    JX_Int i, j, row_end = 0;
    JX_Real tmp;
    
    work[0] = f_data[0];
    for (i = 1; i < num_rows; i ++)
    {
        tmp = f_data[i];
        row_end = indexDP[i];
        for (j = indexAP[i]; j < row_end; j ++)
        {
            tmp -= valueLU[j] * work[indexLU[j]];
        }
        work[i] = tmp;
    }
    u_data[num_rows-1] = work[num_rows-1] * valueLU[row_end];
    for (i = num_rows-2; i > -1; i --)
    {
        tmp = work[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            tmp -= valueLU[j] * u_data[indexLU[j]];
        }
        u_data[i] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelCycleA
 * \brief Parallel Cycle
 * \author Yue Xiaoqiang
 * \date 2014/10/28
 */
void
jx_ILUZeroFactorDataParallelCycleA( jx_ILUZeroFactorData *ilu_data,
                                    jx_ParCSRMatrix *par_A,
                                    jx_ParVector *par_b,
                                    jx_ParVector *par_x )
{
    MPI_Comm comm = jx_ParCSRMatrixComm(par_A);
    JX_Int num_rows = jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(par_A));
    JX_Int *row_starts = jx_ParCSRMatrixRowStarts(par_A);
    jx_Vector *rhs_vec = jx_ParVectorLocalVector(par_b);
    jx_Vector *app_vec = jx_ParVectorLocalVector(par_x);
    JX_Int ex_len = ilu_data->ex_len;
    JX_Int dx_len = ilu_data->dx_len;
    JX_Int kx_len = ilu_data->kx_len;
    JX_Int dy_len = ilu_data->dy_len;
    JX_Int lx_len = ilu_data->lx_len;
    JX_Int ly_len = ilu_data->ly_len;
    JX_Int postn_b = ilu_data->pos_dwn_end;
    JX_Int postn_c = ilu_data->pos_lft_end;
    JX_Int ng_pt = ilu_data->num_equns;
    JX_Int *indexLU = ilu_data->index;
    JX_Int *indexAP = ilu_data->indexA + ex_len;
    JX_Int *indexDP = ilu_data->indexD + ex_len;
    JX_Int *permute = ilu_data->permute;
    JX_Real *valueLU = ilu_data->value;
    JX_Real *senddown = ilu_data->senddown;
    MPI_Status *status = ilu_data->status;
    jx_Vector *aux_vec = jx_ParVectorLocalVector(ilu_data->par_aux_vec);
    jx_Vector *res_vec = jx_ParVectorLocalVector(ilu_data->par_res_vec);
    jx_GridPartitionData *grid_data = ilu_data->par_grid;
    JX_Int num_smallside = grid_data->num_smallside;
    JX_Int num_largeside = grid_data->num_largeside; // YUE: Needn't to consider the cross ones in 5-point stencil
    JX_Int *smallprocs = grid_data->sideprocs;
    JX_Int *largeprocs = smallprocs + num_smallside;
    JX_Int *smallprcpos = grid_data->sideprcpos;
    JX_Int *largeprcpos = smallprcpos + num_smallside;
    JX_Int *smallprclength = grid_data->sideprclength;
    JX_Int *largeprclength = smallprclength + num_smallside;
    JX_Int ng_pt_mx = ng_pt * grid_data->x_part_len;
    //JX_Int ng_pt_my = ng_pt * grid_data->y_part_len;
    JX_Int num_procs, my_id, j, k, pid, jend, jjend, cnt, dnt;
    JX_Int int_uprgt_pnt, recv_downsrt, recv_leftsrt, recv_dnftsrt;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    int_uprgt_pnt = num_rows - ilu_data->ey_len;
    /* TYPE 1. Back substitution of L parts for interior, up and right quasi-boundary points */
    jx_ILUZeroFactorDataParallelLPartIntURPntsA(indexLU, indexAP, indexDP, valueLU,
                                                permute, int_uprgt_pnt, jx_VectorData(aux_vec),
                                                jx_VectorData(rhs_vec), row_starts[my_id]);
    /* Send back substitution of L parts for UP and RIGHT quasi-boundary points. Xiaoqiang Yue, 2014/04/28 */
    recv_downsrt = 0;
    /* jx_MPI_Irecv and jx_MPI_Isend shouldn't be used here as summations of each offset */
    /* Receive data from down processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            jend = (smallprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_downsrt,
                jend, JX_MPI_REAL, smallprocs[pid], my_id*233, comm, &status[pid]);
            recv_downsrt += jend; // YUE: aims to loop for pid
        }
    }
    recv_leftsrt = recv_downsrt;
    /* Receive data from left processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            jend = (smallprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_leftsrt,
                jend, JX_MPI_REAL, smallprocs[pid], my_id*235, comm, &status[pid]);
            recv_leftsrt += jend; // YUE: aims to loop for pid
        }
    }
    /* Fill data, Send data to up processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            jend = (largeprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            jx_MPI_Send(jx_VectorData(aux_vec)+dy_len, jend, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*233, comm);
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
                    senddown[cnt++] = jx_VectorData(aux_vec)[dnt+k];
                }
                dnt += kx_len;
            }
            jx_MPI_Send(senddown, jend, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*235, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    //jx_assert(postn_a == int_uprgt_pnt);
    /* TYPE 2. Back substitution of L parts for down quasi-boundary points */
    jx_ILUZeroFactorDataParallelLPartDPntsA(indexLU, indexAP, indexDP, valueLU,
                                            int_uprgt_pnt, postn_b, permute,
                                            jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                            jx_VectorData(rhs_vec), row_starts[my_id]);
    /* TYPE 3. Back substitution of L parts for left quasi-boundary points */
    jx_ILUZeroFactorDataParallelLPartLPntsA(indexLU, indexAP, indexDP, valueLU,
                                            postn_b, postn_c, recv_downsrt, permute,
                                            jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                            jx_VectorData(rhs_vec), row_starts[my_id]);
    recv_dnftsrt = recv_downsrt = recv_leftsrt;
    /* Receive data from down processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_downsrt,
                ng_pt, JX_MPI_REAL, smallprocs[pid], my_id*237, comm, &status[pid]);
            recv_downsrt += ng_pt; // YUE: aims to loop for pid
        }
    }
    recv_leftsrt = recv_downsrt;
    /* Receive data from left processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_leftsrt,
                ng_pt, JX_MPI_REAL, smallprocs[pid], my_id*231, comm, &status[pid]);
            recv_leftsrt += ng_pt; // YUE: aims to loop for pid
        }
    }
    /* Fill data, Send data to up processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            jx_MPI_Send(jx_VectorData(aux_vec)+ly_len, ng_pt, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*237, comm);
        }
    }
    /* Fill data, Send data to right processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 3) /* RIGHT */
        {
            jx_MPI_Send(jx_VectorData(aux_vec)+lx_len, ng_pt, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*231, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    /* TYPE 4. Back substitution of L parts for LD-Corner quasi-boundary points */
    if (my_id != 0)
    {
        jx_ILUZeroFactorDataParallelLPartLDPntsA(indexLU, indexAP, indexDP, valueLU,
                                                 postn_c, num_rows, row_starts[my_id-1],
                                                 permute, recv_dnftsrt, recv_downsrt,
                                                 jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                                 jx_VectorData(rhs_vec), row_starts[my_id]);
    }
    /* TYPE 4. Back substitution of U parts for LD-Corner quasi-boundary points */
    if (my_id != 0)
    {
        jx_ILUZeroFactorDataParallelUPartLDPntsA(indexLU, indexAP, indexDP,
                                                 valueLU, postn_c, num_rows,
                                                 permute, jx_VectorData(aux_vec),
                                                 jx_VectorData(app_vec), row_starts[my_id]);
    }
    recv_downsrt = 0;
    /* Receive data from UP processor, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_downsrt,
                ng_pt, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*234, comm, &status[pid]);
            recv_downsrt += ng_pt; // YUE: aims to loop for pid
        }
    }
    recv_leftsrt = ng_pt_mx; // YUE: pay attention to the offset
    /* Receive data from RIGHT processor, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 3) /* RIGHT */
        {
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_leftsrt,
                ng_pt, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*236, comm, &status[pid]);
            recv_leftsrt += ng_pt; // YUE: aims to loop for pid
        }
    }
    /* Fill data, Send data to down processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            jx_MPI_Send(jx_VectorData(app_vec), ng_pt, JX_MPI_REAL, smallprocs[pid], my_id*234, comm);
        }
    }
    /* Fill data, Send data to left processors, 1 block, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            jx_MPI_Send(jx_VectorData(app_vec), ng_pt, JX_MPI_REAL, smallprocs[pid], my_id*236, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    /* TYPE 3. Back substitution of U parts for left quasi-boundary points */
    jx_ILUZeroFactorDataParallelUPartLPntsA(indexLU, indexAP, indexDP, valueLU,
                                            postn_b, postn_c, recv_downsrt, permute,
                                            jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                            jx_VectorData(app_vec), row_starts[my_id], row_starts[my_id+1]);
    /* TYPE 2. Back substitution of U parts for down quasi-boundary points */
    jx_ILUZeroFactorDataParallelUPartDPntsA(indexLU, indexAP, indexDP, valueLU,
                                            int_uprgt_pnt, postn_b, recv_leftsrt,
                                            permute, jx_VectorData(aux_vec),
                                            jx_VectorData(res_vec), jx_VectorData(app_vec),
                                            row_starts[my_id], row_starts[my_id+1]);
    //jx_MPI_Barrier(comm);
    //exit(0);
    /* Receive data from UP processor, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 1) /* UP */
        {
            jend = (largeprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_downsrt,
                jend, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*232, comm, &status[pid]);
            recv_downsrt += jend; // YUE: aims to loop for pid
        }
    }
    /* Receive data from RIGHT processor, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_largeside; pid ++)
    {
        if (largeprcpos[pid] == 3) /* RIGHT */
        {
            jend = (largeprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            jx_MPI_Recv(jx_VectorData(res_vec)+recv_leftsrt,
                jend, JX_MPI_REAL, largeprocs[pid], largeprocs[pid]*238, comm, &status[pid]);
            recv_leftsrt += jend; // YUE: aims to loop for pid
        }
    }
    /* Fill data, Send data to down processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 2) /* DOWN */
        {
            jend = (smallprclength[pid] - 1) * ng_pt; // YUE: exclude the first block
            jx_MPI_Send(jx_VectorData(app_vec)+ng_pt, jend, JX_MPI_REAL, smallprocs[pid], my_id*232, comm);
        }
    }
    /* Fill data, Send data to left processors, Yue Xiaoqiang, 2014/08/06 */
    for (pid = 0; pid < num_smallside; pid ++)
    {
        if (smallprcpos[pid] == 4) /* LEFT */
        {
            jjend = smallprclength[pid] - 1; // YUE: exclude the first block
            jend = jjend * ng_pt;
            dnt = cnt = 0;
            for (j = 0; j < jjend; j ++)
            {
                dnt += ng_pt_mx;
                for (k = 0; k < ng_pt; k ++)
                {
                    senddown[cnt++] = jx_VectorData(app_vec)[dnt+k];
                }
            }
            jx_MPI_Send(senddown, jend, JX_MPI_REAL, smallprocs[pid], my_id*238, comm);
        }
    }
    //jx_MPI_Barrier(comm);
    //exit(0);
    /* TYPE 1: Back substitution of U parts for interior, up and right quasi-boundary points */
    if (grid_data->part_type == 1) /* UP and RIGHT both exist */
    {
        //jx_printf("\n >>> 1 Proc %d of %d\n\n", my_id, num_procs);
        jx_ILUZeroFactorDataParallelUPartIntURPntsA(indexLU, indexAP, indexDP, valueLU,
                                                    int_uprgt_pnt, recv_downsrt, recv_leftsrt,
                                                    permute, jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                                    jx_VectorData(app_vec), row_starts[my_id],
                                                    row_starts[my_id+1], row_starts[my_id+2]);
    }
    else if (grid_data->part_type == 2) /* Only UP exists */
    {
        //jx_printf("\n >>> 2 Proc %d of %d\n\n", my_id, num_procs);
        jx_ILUZeroFactorDataParallelUPartIntURPntsB(indexLU, indexAP, indexDP, valueLU,
                                                    int_uprgt_pnt, recv_downsrt, permute,
                                                    jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                                    jx_VectorData(app_vec), row_starts[my_id], row_starts[my_id+1]);
    }
    else if (grid_data->part_type == 3) /* Only RIGHT exists */
    {
        //jx_printf("\n >>> 3 Proc %d of %d\n\n", my_id, num_procs);
        jx_ILUZeroFactorDataParallelUPartIntURPntsC(indexLU, indexAP, indexDP, valueLU,
                                                    int_uprgt_pnt, recv_leftsrt, permute,
                                                    jx_VectorData(aux_vec), jx_VectorData(res_vec),
                                                    jx_VectorData(app_vec), row_starts[my_id], row_starts[my_id+1]);
    }
    else if (grid_data->part_type == 4) /* None of UP or RIGHT */
    {
        //jx_printf("\n >>> 4 Proc %d of %d\n\n", my_id, num_procs);
        jx_ILUZeroFactorDataParallelUPartIntURPntsD(indexLU, indexAP, indexDP, valueLU,
                                                    int_uprgt_pnt, permute, jx_VectorData(aux_vec),
                                                    jx_VectorData(res_vec), jx_VectorData(app_vec),
                                                    row_starts[my_id]);
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelLPartIntURPntsA
 * \brief Parallel Cycle of L parts for Int-U-R points
 * \author Yue Xiaoqiang
 * \date 2014/08/26
 */
void
jx_ILUZeroFactorDataParallelLPartIntURPntsA( JX_Int *indexLU,
                                             JX_Int *indexAP,
                                             JX_Int *indexDP,
                                             JX_Real *valueLU,
                                             JX_Int *permute,
                                             JX_Int int_uprgt_pnt,
                                             JX_Real *aux_data,
                                             JX_Real *rhs_data,
                                             JX_Int first_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    aux_data[0] = rhs_data[permute[0]];
    for (i = 1; i < int_uprgt_pnt; i ++)
    {
        tmp = rhs_data[permute[i]];
        row_end = indexDP[i];
        for (j = indexAP[i]; j < row_end; j ++)
        {
            tmp -= valueLU[j] * aux_data[indexLU[j]-first_row_idx];
        }
        aux_data[i] = tmp;
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelLPartDPntsA
 * \brief Parallel Cycle of L parts for D points
 * \author Yue Xiaoqiang
 * \date 2014/08/28
 */
void
jx_ILUZeroFactorDataParallelLPartDPntsA( JX_Int *indexLU,
                                         JX_Int *indexAP,
                                         JX_Int *indexDP,
                                         JX_Real *valueLU,
                                         JX_Int int_uprgt_pnt,
                                         JX_Int num_rows,
                                         JX_Int *permute,
                                         JX_Real *aux_data,
                                         JX_Real *res_data,
                                         JX_Real *rhs_data,
                                         JX_Int first_row_idx )
{
    JX_Int i, j, l, row_end, sstmp = 0;
    JX_Real tmp;
    
    for (i = int_uprgt_pnt; i < num_rows; i ++)
    {
        tmp = rhs_data[permute[i]];
        row_end = indexDP[i];
        for (j = indexAP[i]; j < row_end; j ++)
        {
            l = indexLU[j] - first_row_idx;
            if (l < 0)
            {
                tmp -= valueLU[j] * res_data[sstmp];
                sstmp ++;
            }
            else
            {
                tmp -= valueLU[j] * aux_data[l];
            }
        }
        aux_data[i] = tmp;
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelLPartLPntsA
 * \brief Parallel Cycle of L parts for L points
 * \author Yue Xiaoqiang
 * \date 2014/08/28
 */
void
jx_ILUZeroFactorDataParallelLPartLPntsA( JX_Int *indexLU,
                                         JX_Int *indexAP,
                                         JX_Int *indexDP,
                                         JX_Real *valueLU,
                                         JX_Int int_uprgtdwn_pnt,
                                         JX_Int num_rows,
                                         JX_Int dwn_num_rows,
                                         JX_Int *permute,
                                         JX_Real *aux_data,
                                         JX_Real *res_data,
                                         JX_Real *rhs_data,
                                         JX_Int first_row_idx )
{
    JX_Int i, j, l, row_end, sstmp = dwn_num_rows;
    JX_Real tmp;
    
    for (i = int_uprgtdwn_pnt; i < num_rows; i ++)
    {
        tmp = rhs_data[permute[i]];
        row_end = indexDP[i];
        for (j = indexAP[i]; j < row_end; j ++)
        {
            l = indexLU[j] - first_row_idx;
            if (l < 0)
            {
                tmp -= valueLU[j] * res_data[sstmp];
                sstmp ++;
            }
            else
            {
                tmp -= valueLU[j] * aux_data[l];
            }
        }
        aux_data[i] = tmp;
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelLPartLDPntsA
 * \brief Parallel Cycle of L parts for LD-Corner points
 * \author Yue Xiaoqiang
 * \date 2014/08/28
 */
void
jx_ILUZeroFactorDataParallelLPartLDPntsA( JX_Int *indexLU,
                                          JX_Int *indexAP,
                                          JX_Int *indexDP,
                                          JX_Real *valueLU,
                                          JX_Int int_urdl_pnt,
                                          JX_Int num_rows,
                                          JX_Int fst_row_idx,
                                          JX_Int *permute,
                                          JX_Int lft_cnum_rows,
                                          JX_Int dwn_dnum_rows,
                                          JX_Real *aux_data,
                                          JX_Real *res_data,
                                          JX_Real *rhs_data,
                                          JX_Int first_row_idx )
{
    JX_Int sstmp = lft_cnum_rows, kktmp = dwn_dnum_rows;
    JX_Int i, j, l, row_end;
    JX_Real tmp;
    
    for (i = int_urdl_pnt; i < num_rows; i ++)
    {
        tmp = rhs_data[permute[i]];
        row_end = indexDP[i];
        for (j = indexAP[i]; j < row_end; j ++)
        {
            l = indexLU[j] - first_row_idx;
            if (l < 0)
            {
                if (l+first_row_idx < fst_row_idx) // DOWN
                {
                    tmp -= valueLU[j] * res_data[sstmp];
                    sstmp ++;
                }
                else // LEFT
                {
                    tmp -= valueLU[j] * res_data[kktmp];
                    kktmp ++;
                }
            }
            else
            {
                tmp -= valueLU[j] * aux_data[l];
            }
        }
        aux_data[i] = tmp;
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartLDPntsA
 * \brief Parallel Cycle of U parts for LD-Corner points
 * \author Yue Xiaoqiang
 * \date 2014/08/30
 */
void
jx_ILUZeroFactorDataParallelUPartLDPntsA( JX_Int *indexLU,
                                          JX_Int *indexAP,
                                          JX_Int *indexDP,
                                          JX_Real *valueLU,
                                          JX_Int int_urdl_pnt,
                                          JX_Int num_rows,
                                          JX_Int *permute,
                                          JX_Real *aux_data,
                                          JX_Real *app_data,
                                          JX_Int first_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    app_data[permute[num_rows-1]] = aux_data[num_rows-1] * valueLU[indexDP[num_rows-1]];
    for (i = num_rows-2; i >= int_urdl_pnt; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartLPntsA
 * \brief Parallel Cycle of U parts for L points
 * \author Yue Xiaoqiang
 * \date 2014/08/30
 */
void
jx_ILUZeroFactorDataParallelUPartLPntsA( JX_Int *indexLU,
                                         JX_Int *indexAP,
                                         JX_Int *indexDP,
                                         JX_Real *valueLU,
                                         JX_Int int_urdl_pnt,
                                         JX_Int num_rows,
                                         JX_Int recv_downsrt,
                                         JX_Int *permute,
                                         JX_Real *aux_data,
                                         JX_Real *res_data,
                                         JX_Real *app_data,
                                         JX_Int first_row_idx,
                                         JX_Int next_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    for (i = num_rows-1; i >= int_urdl_pnt; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            if (indexLU[j] < next_row_idx)
            {
                tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
            }
            else
            {
                recv_downsrt --;
                tmp -= valueLU[j] * res_data[recv_downsrt];
            }
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartDPntsA
 * \brief Parallel Cycle of U parts for D points
 * \author Yue Xiaoqiang
 * \date 2014/08/30
 */
void
jx_ILUZeroFactorDataParallelUPartDPntsA( JX_Int *indexLU,
                                         JX_Int *indexAP,
                                         JX_Int *indexDP,
                                         JX_Real *valueLU,
                                         JX_Int int_uprgt_pnt,
                                         JX_Int num_rows,
                                         JX_Int recv_leftsrt,
                                         JX_Int *permute,
                                         JX_Real *aux_data,
                                         JX_Real *res_data,
                                         JX_Real *app_data,
                                         JX_Int first_row_idx,
                                         JX_Int next_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    for (i = num_rows-1; i >= int_uprgt_pnt; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            if (indexLU[j] < next_row_idx)
            {
                tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
            }
            else
            {
                recv_leftsrt --;
                tmp -= valueLU[j] * res_data[recv_leftsrt];
            }
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartIntURPntsA
 * \brief Parallel Cycle of U parts for Int-U-R points
 * \author Yue Xiaoqiang
 * \date 2014/09/03
 */
void
jx_ILUZeroFactorDataParallelUPartIntURPntsA( JX_Int *indexLU,
                                             JX_Int *indexAP,
                                             JX_Int *indexDP,
                                             JX_Real *valueLU,
                                             JX_Int int_uprgt_pnt,
                                             JX_Int recv_downsrt,
                                             JX_Int recv_leftsrt,
                                             JX_Int *permute,
                                             JX_Real *aux_data,
                                             JX_Real *res_data,
                                             JX_Real *app_data,
                                             JX_Int first_row_idx,
                                             JX_Int next_row_idx,
                                             JX_Int nnxt_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    for (i = int_uprgt_pnt-1; i > -1; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            if (indexLU[j] < next_row_idx)
            {
                tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
            }
            else if (indexLU[j] < nnxt_row_idx)
            {
                recv_leftsrt --;
                tmp -= valueLU[j] * res_data[recv_leftsrt];
            }
            else
            {
                recv_downsrt --;
                tmp -= valueLU[j] * res_data[recv_downsrt];
            }
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartIntURPntsB
 * \brief Parallel Cycle of U parts for Int-U-R points
 * \author Yue Xiaoqiang
 * \date 2014/09/11
 */
void
jx_ILUZeroFactorDataParallelUPartIntURPntsB( JX_Int *indexLU,
                                             JX_Int *indexAP,
                                             JX_Int *indexDP,
                                             JX_Real *valueLU,
                                             JX_Int int_uprgt_pnt,
                                             JX_Int recv_downsrt,
                                             JX_Int *permute,
                                             JX_Real *aux_data,
                                             JX_Real *res_data,
                                             JX_Real *app_data,
                                             JX_Int first_row_idx,
                                             JX_Int next_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    for (i = int_uprgt_pnt-1; i > -1; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            if (indexLU[j] < next_row_idx)
            {
                tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
            }
            else
            {
                recv_downsrt --;
                tmp -= valueLU[j] * res_data[recv_downsrt];
            }
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartIntURPntsC
 * \brief Parallel Cycle of U parts for Int-U-R points
 * \author Yue Xiaoqiang
 * \date 2014/09/17
 */
void
jx_ILUZeroFactorDataParallelUPartIntURPntsC( JX_Int *indexLU,
                                             JX_Int *indexAP,
                                             JX_Int *indexDP,
                                             JX_Real *valueLU,
                                             JX_Int int_uprgt_pnt,
                                             JX_Int recv_leftsrt,
                                             JX_Int *permute,
                                             JX_Real *aux_data,
                                             JX_Real *res_data,
                                             JX_Real *app_data,
                                             JX_Int first_row_idx,
                                             JX_Int next_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    for (i = int_uprgt_pnt-1; i > -1; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            if (indexLU[j] < next_row_idx)
            {
                tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
            }
            else
            {
                recv_leftsrt --;
                tmp -= valueLU[j] * res_data[recv_leftsrt];
            }
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*!
 * \fn void jx_ILUZeroFactorDataParallelUPartIntURPntsD
 * \brief Parallel Cycle of U parts for Int-U-R points
 * \author Yue Xiaoqiang
 * \date 2014/09/23
 */
void
jx_ILUZeroFactorDataParallelUPartIntURPntsD( JX_Int *indexLU,
                                             JX_Int *indexAP,
                                             JX_Int *indexDP,
                                             JX_Real *valueLU,
                                             JX_Int int_uprgt_pnt,
                                             JX_Int *permute,
                                             JX_Real *aux_data,
                                             JX_Real *res_data,
                                             JX_Real *app_data,
                                             JX_Int first_row_idx )
{
    JX_Int i, j, row_end;
    JX_Real tmp;
    
    for (i = int_uprgt_pnt-1; i > -1; i --)
    {
        tmp = aux_data[i];
        row_end = indexAP[i+1];
        for (j = indexDP[i]+1; j < row_end; j ++)
        {
            tmp -= valueLU[j] * app_data[permute[indexLU[j]-first_row_idx]];
        }
        app_data[permute[i]] = tmp * valueLU[indexDP[i]];
    }
}

/*------------------------------------------------------------*/
/*------------------------ End of File -----------------------*/
/*------------------------------------------------------------*/
