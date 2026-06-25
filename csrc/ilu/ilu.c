//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  ilu.c
 *
 * Created by Yue Xiaoqiang 2014/03/24
 * Xiangtan University
 *
 */

#include "jx_ilu.h"

/*!
 * \fn JX_Int JX_ILUZeroFactorDataCreate
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
JX_ILUZeroFactorDataCreate( JX_Solver *solver, MPI_Comm comm )
{
  *solver = (JX_Solver)jx_ILUZeroFactorDataInitialize(comm);
   if (!solver)
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetMaxIter
 * \brief Set max_iter
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
JX_ILUZeroFactorDataSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_ILUZeroFactorDataSetMaxIter( (void *) solver, max_iter ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetNxy
 * \brief Set nx and ny
 * \author Yue Xiaoqiang
 * \date 2014/04/21
 */
JX_Int
JX_ILUZeroFactorDataSetNxy( JX_Solver solver, JX_Int nx, JX_Int ny )
{
   return( jx_ILUZeroFactorDataSetNxy( (void *) solver, nx, ny ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetNpxy
 * \brief Set npx and npy
 * \author Yue Xiaoqiang
 * \date 2014/04/21
 */
JX_Int
JX_ILUZeroFactorDataSetNpxy( JX_Solver solver, JX_Int npx, JX_Int npy )
{
   return( jx_ILUZeroFactorDataSetNpxy( (void *) solver, npx, npy ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetNumEquns
 * \brief Set num_equns
 * \author Yue Xiaoqiang
 * \date 2014/04/21
 */
JX_Int
JX_ILUZeroFactorDataSetNumEquns( JX_Solver solver, JX_Int num_equns )
{
   return( jx_ILUZeroFactorDataSetNumEquns( (void *) solver, num_equns ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetDropTol
 * \brief Set drop_tol
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
JX_ILUZeroFactorDataSetDropTol( JX_Solver solver, JX_Real drop_tol )
{
   return( jx_ILUZeroFactorDataSetDropTol( (void *) solver, drop_tol ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetMatA
 * \brief Set matA
 * \author Yue Xiaoqiang
 * \date 2014/04/03
 */
JX_Int
JX_ILUZeroFactorDataSetMatA( JX_Solver solver, jx_CSRMatrix *matA )
{
   return( jx_ILUZeroFactorDataSetMatA( (void *) solver, matA ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataGetLULength
 * \brief Get lu_length
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
JX_ILUZeroFactorDataGetLULength( JX_Solver solver, JX_Int *lu_length )
{
   return( jx_ILUZeroFactorDataGetLULength( (void *) solver, lu_length ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataDestroy
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
JX_ILUZeroFactorDataDestroy( JX_Solver solver )
{
   return( jx_ILUZeroFactorDataFinalize( (void *) solver ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataGenerateParGrid
 * \brief Generate par_grid
 * \author Yue Xiaoqiang
 * \date 2014/10/20
 */
JX_Int
JX_ILUZeroFactorDataGenerateParGrid( JX_Solver solver )
{
   return( jx_ILUZeroFactorDataGenerateParGrid( (void *) solver ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataSetup
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
JX_ILUZeroFactorDataSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix )
{
   return( jx_ILUZeroFactorDataSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int JX_ILUZeroFactorDataPrecond
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
JX_ILUZeroFactorDataPrecond( JX_Solver       solver,
                             JX_ParCSRMatrix par_matrix,
                             JX_ParVector    par_rhs,
                             JX_ParVector    par_app  )
{
   return( jx_ILUZeroFactorDataPrecond( (void *) solver,
                                        (jx_ParCSRMatrix *) par_matrix,
                                        (jx_ParVector *) par_rhs,
                                        (jx_ParVector *) par_app ) );
}

/*!
 * \fn void *jx_ILUZeroFactorDataInitialize
 * \brief Initialize
 * \author Zhiyang Zhou, Yue Xiaoqiang
 * \date 2010/12/11, 2014/03/24
 */
void *
jx_ILUZeroFactorDataInitialize( MPI_Comm comm )
{
    jx_ILUZeroFactorData *ilu_data = jx_CTAlloc(jx_ILUZeroFactorData, 1);
    JX_Int max_iter = 200;
    JX_Real drop_tol = 0.0;
    
    ilu_data->comm = comm;
    ilu_data->max_iter = max_iter;
    ilu_data->drop_tol = drop_tol;
    
    return (void *)ilu_data;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetMaxIter
 * \brief Set max_iter
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
jx_ILUZeroFactorDataSetMaxIter( void *ilu_vdata, JX_Int max_iter )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->max_iter = max_iter;
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetNxy
 * \brief Set nx and ny
 * \author Yue Xiaoqiang
 * \date 2014/04/21
 */
JX_Int
jx_ILUZeroFactorDataSetNxy( void *ilu_vdata, JX_Int nx, JX_Int ny )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->nx = nx;
    ilu_data->ny = ny;
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetNpxy
 * \brief Set npx and npy
 * \author Yue Xiaoqiang
 * \date 2014/04/21
 */
JX_Int
jx_ILUZeroFactorDataSetNpxy( void *ilu_vdata, JX_Int npx, JX_Int npy )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->npx = npx;
    ilu_data->npy = npy;
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetNumEquns
 * \brief Set num_equns
 * \author Yue Xiaoqiang
 * \date 2014/04/21
 */
JX_Int
jx_ILUZeroFactorDataSetNumEquns( void *ilu_vdata, JX_Int num_equns )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->num_equns = num_equns;
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetDropTol
 * \brief Set drop_tol
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
jx_ILUZeroFactorDataSetDropTol( void *ilu_vdata, JX_Real drop_tol )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->drop_tol = drop_tol;
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetMatA
 * \brief Set matA
 * \author Yue Xiaoqiang
 * \date 2014/04/03
 */
JX_Int
jx_ILUZeroFactorDataSetMatA( void *ilu_vdata, jx_CSRMatrix *matA )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->matA = matA;
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataGetLULength
 * \brief Get lu_length
 * \author Yue Xiaoqiang
 * \date 2014/03/17
 */
JX_Int
jx_ILUZeroFactorDataGetLULength( void *ilu_vdata, JX_Int *lu_length )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    if (!ilu_data)
    {
        jx_printf(" Warning: ILUZeroFactor object empty!\n");
        jx_error_in_arg(1);
        return jx_error_flag;
    }
   *lu_length = ilu_data->length;
    return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataFinalize
 * \brief Finalize
 * \author Zhiyang Zhou, Yue Xiaoqiang
 * \date 2010/12/11, 2014/03/24
 */
JX_Int
jx_ILUZeroFactorDataFinalize( void *ilu_vdata )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    
    if (ilu_data)
    {
        if (ilu_data->index)
        {
            jx_TFree(ilu_data->index);
        }
        if (ilu_data->indexA)
        {
            jx_TFree(ilu_data->indexA);
        }
        if (ilu_data->indexD)
        {
            jx_TFree(ilu_data->indexD);
        }
        if (ilu_data->permute)
        {
            jx_TFree(ilu_data->permute);
        }
        if (ilu_data->value)
        {
            jx_TFree(ilu_data->value);
        }
        if (ilu_data->senddown)
        {
            jx_TFree(ilu_data->senddown);
        }
        if (ilu_data->status)
        {
            jx_TFree(ilu_data->status);
        }
        if (ilu_data->aux_vec)
        {
            jx_SeqVectorDestroy(ilu_data->aux_vec);
        }
        if (ilu_data->res_vec)
        {
            jx_SeqVectorDestroy(ilu_data->res_vec);
        }
        if (ilu_data->tmp_vec)
        {
            jx_SeqVectorDestroy(ilu_data->tmp_vec);
        }
        if (ilu_data->par_aux_vec)
        {
            jx_ParVectorDestroy(ilu_data->par_aux_vec);
        }
        if (ilu_data->par_res_vec)
        {
            jx_ParVectorDestroy(ilu_data->par_res_vec);
        }
        if (ilu_data->par_grid)
        {
            jx_GridPartitionDataFinalize(ilu_data->par_grid);
        }
        jx_TFree(ilu_data);
    }
    
    return 0;
}

/*!
 * \fn JX_Int jx_ILUZeroFactorDataGenerateParGrid
 * \brief Generate par_grid
 * \author Yue Xiaoqiang
 * \date 2014/10/20
 */
JX_Int
jx_ILUZeroFactorDataGenerateParGrid( void *ilu_vdata )
{
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    ilu_data->par_grid = jx_GridPartitionDataInitialize(ilu_data->comm,
                     ilu_data->nx, ilu_data->ny, ilu_data->npx, ilu_data->npy);
    jx_GridPartitionDataSetEachSides4Comm(ilu_data->par_grid);
    return 0;
}
