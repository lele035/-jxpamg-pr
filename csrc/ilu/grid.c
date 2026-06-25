//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  grid.c
 *
 * Created by Yue Xiaoqiang 2014/08/03
 * Xiangtan University
 *
 */

#include "jx_ilu.h"

/*!
 * \fn JX_Int JX_GridPartitionDataCreate
 * \author Yue Xiaoqiang
 * \date 2014/10/20
 */
JX_Int
JX_GridPartitionDataCreate( JX_Solver *solver, MPI_Comm comm, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy )
{
  *solver = (JX_Solver)jx_GridPartitionDataInitialize(comm, nx, ny, npx, npy);
   if (!solver)
   {
      jx_error_in_arg(1);
   }
   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_GridPartitionDataSetEachSides4Comm
 * \author Yue Xiaoqiang
 * \date 2014/08/03
 */
JX_Int
JX_GridPartitionDataSetEachSides4Comm( JX_Solver solver )
{
   return( jx_GridPartitionDataSetEachSides4Comm( (void *) solver ) );
}

/*!
 * \fn JX_Int JX_GridPartitionDataDestroy
 * \author Yue Xiaoqiang
 * \date 2014/08/03
 */
JX_Int
JX_GridPartitionDataDestroy( JX_Solver solver )
{
   return( jx_GridPartitionDataFinalize( (void *) solver ) );
}

/*!
 * \fn void *jx_GridPartitionDataInitialize
 * \brief Initialize
 * \author Yue Xiaoqiang
 * \date 2014/10/20
 */
void *
jx_GridPartitionDataInitialize( MPI_Comm comm, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy )
{
    jx_GridPartitionData *grid_data = jx_CTAlloc(jx_GridPartitionData, 1);
    JX_Int *nx_part = NULL;
    JX_Int *ny_part = NULL;
    JX_Int num_procs, my_id, iy, ix, cnt = 0;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    grid_data->comm = comm;
    grid_data->xlo_array = jx_CTAlloc(JX_Int, num_procs);
    grid_data->xup_array = jx_CTAlloc(JX_Int, num_procs);
    grid_data->ylo_array = jx_CTAlloc(JX_Int, num_procs);
    grid_data->yup_array = jx_CTAlloc(JX_Int, num_procs);
    jx_GeneratePartitioning(nx, npx, &nx_part);
    jx_GeneratePartitioning(ny, npy, &ny_part);
    if (0) // Processors in x-direction first
    {
        for (iy = 0; iy < npy; iy ++)
        {
            for (ix = 0; ix < npx; ix ++)
            {
                grid_data->xlo_array[cnt] = nx_part[ix];
                grid_data->xup_array[cnt] = nx_part[ix+1];
                grid_data->ylo_array[cnt] = ny_part[iy];
                grid_data->yup_array[cnt] = ny_part[iy+1];
                cnt ++;
            }
        }
    }
    else // Processors in y-direction first
    {
        for (ix = 0; ix < npx; ix ++)
        {
            for (iy = 0; iy < npy; iy ++)
            {
                grid_data->xlo_array[cnt] = nx_part[ix];
                grid_data->xup_array[cnt] = nx_part[ix+1];
                grid_data->ylo_array[cnt] = ny_part[iy];
                grid_data->yup_array[cnt] = ny_part[iy+1];
                cnt ++;
            }
        }
    }
    jx_TFree(nx_part);
    jx_TFree(ny_part);
    grid_data->x_lower_idx = grid_data->xlo_array[my_id];
    grid_data->x_upper_idx = grid_data->xup_array[my_id];
    grid_data->y_lower_idx = grid_data->ylo_array[my_id];
    grid_data->y_upper_idx = grid_data->yup_array[my_id];
    grid_data->x_part_len = grid_data->xup_array[my_id] - grid_data->xlo_array[my_id];
    grid_data->y_part_len = grid_data->yup_array[my_id] - grid_data->ylo_array[my_id];
    
    return (void *)grid_data;
}

/*!
 * \fn JX_Int jx_GridPartitionDataSetEachSides4Comm
 * \brief Set sideprocs and sideprcfgstln
 * \note not for the case of a street intersection
 * \author Yue Xiaoqiang
 * \date 2014/08/03
 */
JX_Int
jx_GridPartitionDataSetEachSides4Comm( void *grid_vdata )
{
    jx_GridPartitionData *grid_data = grid_vdata;
    JX_Int x_lower_idx = grid_data->x_lower_idx;
    JX_Int x_upper_idx = grid_data->x_upper_idx;
    JX_Int y_lower_idx = grid_data->y_lower_idx;
    JX_Int y_upper_idx = grid_data->y_upper_idx;
    JX_Int *tmpsidepro = NULL;
    JX_Int myid, nprocs, i, x_cnt, y_cnt, pid;
    JX_Int num_sideproc, num_smallside, num_largeside;
    JX_Int num_small_dlr, num_large_ulr, num_upside, num_rightside;
    
    jx_MPI_Comm_size(grid_data->comm, &nprocs);
    jx_MPI_Comm_rank(grid_data->comm, &myid);
    tmpsidepro = jx_CTAlloc(JX_Int, nprocs);
    // store all the side processors, Yue Xiaoqiang, 2014/08/03
    num_sideproc = 0;
    for (i = 0; i < nprocs; i ++)
    {
        if (i == myid)
        {
            continue;
        }
        x_cnt = 0;
        if ((grid_data->xlo_array[i] >= x_lower_idx) && (grid_data->xlo_array[i] <= x_upper_idx))
        {
            x_cnt ++;
        }
        if ((grid_data->xup_array[i] >= x_lower_idx) && (grid_data->xup_array[i] <= x_upper_idx))
        {
            x_cnt ++;
        }
        if (x_cnt < 1) // x_cnt = 0
        {
            // symmetric case, Yue Xiaoqiang, 2014/08/03
            if ((x_lower_idx >= grid_data->xlo_array[i]) && (x_lower_idx <= grid_data->xup_array[i]))
            {
                x_cnt ++;
            }
            if ((x_upper_idx >= grid_data->xlo_array[i]) && (x_upper_idx <= grid_data->xup_array[i]))
            {
                x_cnt ++;
            }
            if (x_cnt < 1)
            {
                continue;
            }
        }
        y_cnt = 0;
        if ((grid_data->ylo_array[i] >= y_lower_idx) && (grid_data->ylo_array[i] <= y_upper_idx))
        {
            y_cnt ++;
        }
        if ((grid_data->yup_array[i] >= y_lower_idx) && (grid_data->yup_array[i] <= y_upper_idx))
        {
            y_cnt ++;
        }
        if (y_cnt > 0)
        {
            tmpsidepro[num_sideproc++] = i;
            continue;
        }
        // symmetric case, Yue Xiaoqiang, 2014/08/03
        // y_cnt = 0
        if ((y_lower_idx >= grid_data->ylo_array[i]) && (y_lower_idx <= grid_data->yup_array[i]))
        {
            y_cnt ++;
        }
        if ((y_upper_idx >= grid_data->ylo_array[i]) && (y_upper_idx <= grid_data->yup_array[i]))
        {
            y_cnt ++;
        }
        if (y_cnt > 0)
        {
            tmpsidepro[num_sideproc++] = i;
        }
    }
    grid_data->num_sideproc = num_sideproc;
    num_smallside = 0;
    num_largeside = 0;
    num_small_dlr = 0;
    num_large_ulr = 0;
    num_upside = 0;
    num_rightside = 0;
    for (i = 0; i < num_sideproc; i ++)
    {
        pid = tmpsidepro[i];
        // determinations to deal with possible cases, Yue Xiaoqiang, 2014/08/03
        if ((grid_data->xlo_array[pid] == x_lower_idx) && (grid_data->xup_array[pid] == x_upper_idx))
        {
            if (y_lower_idx < grid_data->ylo_array[pid])
            {
                num_largeside ++; // UP
                num_upside ++;
            }
            else // won't be equal
            {
                num_smallside ++; // DOWN
            }
        }
        else if ((grid_data->ylo_array[pid] == y_lower_idx) && (grid_data->yup_array[pid] == y_upper_idx))
        {
            if (x_lower_idx < grid_data->xlo_array[pid])
            {
                num_largeside ++; // RIGHT
                num_rightside ++;
            }
            else // won't be equal
            {
                num_smallside ++; // LEFT
            }
        }
        else if (grid_data->xlo_array[pid] == x_upper_idx) // case 3
        {
            if (grid_data->ylo_array[pid] == y_upper_idx)
            {
                num_large_ulr ++; // UP - RIGHT
            }
            else if (grid_data->yup_array[pid] == y_lower_idx)
            {
                num_small_dlr ++; // DOWN - RIGHT
            }
            else
            {
                num_largeside ++; // RIGHT
                num_rightside ++;
            }
        }
        else if (grid_data->xup_array[pid] == x_lower_idx) // case 4
        {
            if (grid_data->ylo_array[pid] == y_upper_idx)
            {
                num_large_ulr ++; // UP - LEFT
            }
            else if (grid_data->yup_array[pid] == y_lower_idx)
            {
                num_small_dlr ++; // DOWN - LEFT
            }
            else
            {
                num_smallside ++; // LEFT
            }
        }
        else if (grid_data->ylo_array[pid] == y_upper_idx) // case 5
        {
            if (grid_data->xlo_array[pid] == x_upper_idx)
            {
                num_large_ulr ++; // UP - RIGHT
            }
            else if (grid_data->xup_array[pid] == x_lower_idx)
            {
                num_large_ulr ++; // UP - LEFT
            }
            else
            {
                num_largeside ++; // UP
                num_upside ++;
            }
        }
        else if (grid_data->yup_array[pid] == y_lower_idx) // case 6
        {
            if (grid_data->xlo_array[pid] == x_upper_idx)
            {
                num_small_dlr ++; // DOWN - RIGHT
            }
            else if (grid_data->xup_array[pid] == x_lower_idx)
            {
                num_small_dlr ++; // DOWN - LEFT
            }
            else
            {
                num_smallside ++; // DOWN
            }
        }
    }
    if (num_upside > 0)
    {
        if (num_rightside > 0)
        {
            grid_data->part_type = 1; // UP and RIGHT both exist
        }
        else
        {
            grid_data->part_type = 2; // Only UP exists
        }
    }
    else
    {
        if (num_rightside > 0)
        {
            grid_data->part_type = 3; // Only RIGHT exists
        }
        else
        {
            grid_data->part_type = 4; // None of UP or RIGHT
        }
    }
    grid_data->num_smallside = num_smallside;
    grid_data->num_largeside = num_largeside;
    grid_data->num_smallcross = num_small_dlr;
    grid_data->num_largecross = num_large_ulr;
    grid_data->num_nocrossside = num_smallside + num_largeside;
    //jx_assert(num_smallside+num_largeside+num_small_dlr+num_large_ulr == num_sideproc);
    num_large_ulr = num_smallside + num_largeside + num_small_dlr;
    num_small_dlr = num_smallside + num_largeside;
    num_largeside = num_smallside;
    num_smallside = 0;
    grid_data->sideprocs = jx_CTAlloc(JX_Int, num_sideproc);
    grid_data->sideprcpos = jx_CTAlloc(JX_Int, num_sideproc);
    grid_data->sideprcxsrt = jx_CTAlloc(JX_Int, num_sideproc);
    grid_data->sideprcysrt = jx_CTAlloc(JX_Int, num_sideproc);
    grid_data->sideprclength = jx_CTAlloc(JX_Int, num_sideproc);
    for (i = 0; i < num_sideproc; i ++)
    {
        pid = tmpsidepro[i];
        // determinations to deal with possible cases, Yue Xiaoqiang, 2014/08/03
        if ((grid_data->xlo_array[pid] == x_lower_idx) && (grid_data->xup_array[pid] == x_upper_idx))
        {
            if (y_lower_idx < grid_data->ylo_array[pid])
            {
                grid_data->sideprocs[num_largeside] = pid;
                grid_data->sideprcpos[num_largeside] = 1;
                grid_data->sideprcxsrt[num_largeside] = x_lower_idx;
                grid_data->sideprcysrt[num_largeside] = y_upper_idx - 1;
                grid_data->sideprclength[num_largeside] = x_upper_idx - x_lower_idx;
                num_largeside ++;
            }
            else // won't be equal
            {
                grid_data->sideprocs[num_smallside] = pid;
                grid_data->sideprcpos[num_smallside] = 2;
                grid_data->sideprcxsrt[num_smallside] = x_lower_idx;
                grid_data->sideprcysrt[num_smallside] = y_lower_idx;
                grid_data->sideprclength[num_smallside] = x_upper_idx - x_lower_idx;
                num_smallside ++;
            }
        }
        else if ((grid_data->ylo_array[pid] == y_lower_idx) && (grid_data->yup_array[pid] == y_upper_idx))
        {
            if (x_lower_idx < grid_data->xlo_array[pid])
            {
                grid_data->sideprocs[num_largeside] = pid;
                grid_data->sideprcpos[num_largeside] = 3;
                grid_data->sideprcxsrt[num_largeside] = x_upper_idx - 1;
                grid_data->sideprcysrt[num_largeside] = y_lower_idx;
                grid_data->sideprclength[num_largeside] = y_upper_idx - y_lower_idx;
                num_largeside ++;
            }
            else // won't be equal
            {
                grid_data->sideprocs[num_smallside] = pid;
                grid_data->sideprcpos[num_smallside] = 4;
                grid_data->sideprcxsrt[num_smallside] = x_lower_idx;
                grid_data->sideprcysrt[num_smallside] = y_lower_idx;
                grid_data->sideprclength[num_smallside] = y_upper_idx - y_lower_idx;
                num_smallside ++;
            }
        }
        else if (grid_data->xlo_array[pid] == x_upper_idx) // case 3
        {
            if (grid_data->ylo_array[pid] == y_upper_idx)
            {
                grid_data->sideprocs[num_large_ulr] = pid;
                grid_data->sideprcpos[num_large_ulr] = 6;
                grid_data->sideprcxsrt[num_large_ulr] = x_upper_idx - 1;
                grid_data->sideprcysrt[num_large_ulr] = y_upper_idx - 1;
                grid_data->sideprclength[num_large_ulr] = 1;
                num_large_ulr ++;
            }
            else if (grid_data->yup_array[pid] == y_lower_idx)
            {
                grid_data->sideprocs[num_small_dlr] = pid;
                grid_data->sideprcpos[num_small_dlr] = 8;
                grid_data->sideprcxsrt[num_small_dlr] = x_upper_idx - 1;
                grid_data->sideprcysrt[num_small_dlr] = y_lower_idx;
                grid_data->sideprclength[num_small_dlr] = 1;
                num_small_dlr ++;
            }
            else
            {
                grid_data->sideprocs[num_largeside] = pid;
                grid_data->sideprcpos[num_largeside] = 3;
                grid_data->sideprcxsrt[num_largeside] = x_upper_idx - 1;
                grid_data->sideprcysrt[num_largeside] = jx_max(y_lower_idx, grid_data->ylo_array[pid]);
                grid_data->sideprclength[num_largeside] = jx_min(y_upper_idx,
                                   grid_data->yup_array[pid]) - jx_max(y_lower_idx, grid_data->ylo_array[pid]);
                num_largeside ++;
            }
        }
        else if (grid_data->xup_array[pid] == x_lower_idx) // case 4
        {
            if (grid_data->ylo_array[pid] == y_upper_idx)
            {
                grid_data->sideprocs[num_large_ulr] = pid;
                grid_data->sideprcpos[num_large_ulr] = 5;
                grid_data->sideprcxsrt[num_large_ulr] = x_lower_idx;
                grid_data->sideprcysrt[num_large_ulr] = y_upper_idx - 1;
                grid_data->sideprclength[num_large_ulr] = 1;
                num_large_ulr ++;
            }
            else if (grid_data->yup_array[pid] == y_lower_idx)
            {
                grid_data->sideprocs[num_small_dlr] = pid;
                grid_data->sideprcpos[num_small_dlr] = 7;
                grid_data->sideprcxsrt[num_small_dlr] = x_lower_idx;
                grid_data->sideprcysrt[num_small_dlr] = y_lower_idx;
                grid_data->sideprclength[num_small_dlr] = 1;
                num_small_dlr ++;
            }
            else
            {
                grid_data->sideprocs[num_smallside] = pid;
                grid_data->sideprcpos[num_smallside] = 4;
                grid_data->sideprcxsrt[num_smallside] = x_lower_idx;
                grid_data->sideprcysrt[num_smallside] = jx_max(y_lower_idx, grid_data->ylo_array[pid]);
                grid_data->sideprclength[num_smallside] = jx_min(y_upper_idx,
                                   grid_data->yup_array[pid]) - jx_max(y_lower_idx, grid_data->ylo_array[pid]);
                num_smallside ++;
            }
        }
        else if (grid_data->ylo_array[pid] == y_upper_idx) // case 5
        {
            if (grid_data->xlo_array[pid] == x_upper_idx)
            {
                grid_data->sideprocs[num_large_ulr] = pid;
                grid_data->sideprcpos[num_large_ulr] = 6;
                grid_data->sideprcxsrt[num_large_ulr] = x_upper_idx - 1;
                grid_data->sideprcysrt[num_large_ulr] = y_upper_idx - 1;
                grid_data->sideprclength[num_large_ulr] = 1;
                num_large_ulr ++;
            }
            else if (grid_data->xup_array[pid] == x_lower_idx)
            {
                grid_data->sideprocs[num_large_ulr] = pid;
                grid_data->sideprcpos[num_large_ulr] = 5;
                grid_data->sideprcxsrt[num_large_ulr] = x_lower_idx;
                grid_data->sideprcysrt[num_large_ulr] = y_upper_idx - 1;
                grid_data->sideprclength[num_large_ulr] = 1;
                num_large_ulr ++;
            }
            else
            {
                grid_data->sideprocs[num_largeside] = pid;
                grid_data->sideprcpos[num_largeside] = 1;
                grid_data->sideprcxsrt[num_largeside] = jx_max(x_lower_idx, grid_data->xlo_array[pid]);
                grid_data->sideprcysrt[num_largeside] = y_upper_idx - 1;
                grid_data->sideprclength[num_largeside] = jx_min(x_upper_idx,
                                   grid_data->xup_array[pid]) - jx_max(x_lower_idx, grid_data->xlo_array[pid]);
                num_largeside ++;
            }
        }
        else if (grid_data->yup_array[pid] == y_lower_idx) // case 6
        {
            if (grid_data->xlo_array[pid] == x_upper_idx)
            {
                grid_data->sideprocs[num_small_dlr] = pid;
                grid_data->sideprcpos[num_small_dlr] = 8;
                grid_data->sideprcxsrt[num_small_dlr] = x_upper_idx - 1;
                grid_data->sideprcysrt[num_small_dlr] = y_lower_idx;
                grid_data->sideprclength[num_small_dlr] = 1;
                num_small_dlr ++;
            }
            else if (grid_data->xup_array[pid] == x_lower_idx)
            {
                grid_data->sideprocs[num_small_dlr] = pid;
                grid_data->sideprcpos[num_small_dlr] = 7;
                grid_data->sideprcxsrt[num_small_dlr] = x_lower_idx;
                grid_data->sideprcysrt[num_small_dlr] = y_lower_idx;
                grid_data->sideprclength[num_small_dlr] = 1;
                num_small_dlr ++;
            }
            else
            {
                grid_data->sideprocs[num_smallside] = pid;
                grid_data->sideprcpos[num_smallside] = 2;
                grid_data->sideprcxsrt[num_smallside] = jx_max(x_lower_idx, grid_data->xlo_array[pid]);
                grid_data->sideprcysrt[num_smallside] = y_lower_idx;
                grid_data->sideprclength[num_smallside] = jx_min(x_upper_idx,
                                   grid_data->xup_array[pid]) - jx_max(x_lower_idx, grid_data->xlo_array[pid]);
                num_smallside ++;
            }
        }
    }
    //jx_assert(num_large_ulr == num_sideproc);
    jx_TFree(tmpsidepro);
    
    return 0;
}

/*!
 * \fn JX_Int jx_GridPartitionDataFinalize
 * \brief Finalize
 * \author Yue Xiaoqiang
 * \date 2014/08/03
 */
JX_Int
jx_GridPartitionDataFinalize( void *grid_vdata )
{
    jx_GridPartitionData *grid_data = grid_vdata;
    
    if (grid_data)
    {
        if (grid_data->xlo_array)
        {
           jx_TFree(grid_data->xlo_array);
        }
        if (grid_data->xup_array)
        {
           jx_TFree(grid_data->xup_array);
        }
        if (grid_data->ylo_array)
        {
           jx_TFree(grid_data->ylo_array);
        }
        if (grid_data->yup_array)
        {
           jx_TFree(grid_data->yup_array);
        }
        if (grid_data->sideprocs)
        {
            jx_TFree(grid_data->sideprocs);
        }
        if (grid_data->sideprcpos)
        {
           jx_TFree(grid_data->sideprcpos);
        }
        if (grid_data->sideprcxsrt)
        {
           jx_TFree(grid_data->sideprcxsrt);
        }
        if (grid_data->sideprcysrt)
        {
           jx_TFree(grid_data->sideprcysrt);
        }
        if (grid_data->sideprclength)
        {
           jx_TFree(grid_data->sideprclength);
        }
        jx_TFree(grid_data);
    }
    
    return 0;
}
