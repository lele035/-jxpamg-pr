//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 * ilusetup.c
 *
 * Created by Yue Xiaoqiang 2014/03/24
 * Xiangtan University
 *
 */

#include "jx_ilu.h"

/*!
 * \fn JX_Int jx_ILUZeroFactorDataSetup
 * \brief SETUP phase
 * \author peghoty, Yue Xiaoqiang
 * \date 2010/12/11, 2014/03/24
 */
JX_Int
jx_ILUZeroFactorDataSetup( void *ilu_vdata, jx_ParCSRMatrix *par_A )
{
    MPI_Comm comm = jx_ParCSRMatrixComm(par_A);
    jx_ILUZeroFactorData *ilu_data = ilu_vdata;
    JX_Real drop_tol = ilu_data->drop_tol;
    JX_Int *index = NULL;
    JX_Int *indexA = NULL;
    JX_Int *indexD = NULL;
    JX_Real *value = NULL;
    jx_CSRMatrix *A = NULL;
    JX_Int *part_aux = NULL;
    JX_Int *part_res = NULL;
    JX_Int num_procs, my_id, num_rows;
    JX_Int length, dfill_in, maxlength, maxdfill_in;
    
    jx_MPI_Comm_size(comm, &num_procs);
    jx_MPI_Comm_rank(comm, &my_id);
    
    if (num_procs == 1)
    {
        A = jx_ParCSRMatrixDiag(par_A);
        num_rows = jx_CSRMatrixNumRows(A);
        //jx_printf(" >>> SETUP: Drop-Tol = %le on ID = %d of %d\n", drop_tol, my_id, num_procs);
        if (drop_tol == 0.0)
        {
            ilu_data->num_fill_in_drop = jx_ILUZeroDecompositionA(A, &indexD, &index, &value);
            ilu_data->length = jx_CSRMatrixNumNonzeros(A);
        }
        else
        {
            ilu_data->num_fill_in_drop = jx_ILUZeroDecompositionB(A, drop_tol, &indexA, &indexD, &index, &value);
            ilu_data->length = indexA[num_rows];
        }
        ilu_data->index = index;
        ilu_data->indexA = indexA;
        ilu_data->indexD = indexD;
        ilu_data->value = value;
        
        ilu_data->aux_vec = jx_SeqVectorCreate(num_rows);
        jx_SeqVectorInitialize(ilu_data->aux_vec);
        ilu_data->res_vec = jx_SeqVectorCreate(num_rows);
        jx_SeqVectorInitialize(ilu_data->res_vec);
        ilu_data->tmp_vec = jx_SeqVectorCreate(num_rows);
        jx_SeqVectorInitialize(ilu_data->tmp_vec);
    }
    else if ((ilu_data->matA != NULL) && (my_id == 0))
    {
        A = ilu_data->matA;
        num_rows = jx_CSRMatrixNumRows(A);
        //jx_printf(" >>> SETUP: Drop-Tol = %le on ID = %d of %d\n", drop_tol, my_id, num_procs);
        if (drop_tol == 0.0)
        {
            ilu_data->num_fill_in_drop = jx_ILUZeroDecompositionA(A, &indexD, &index, &value);
            ilu_data->length = jx_CSRMatrixNumNonzeros(A);
        }
        else
        {
            ilu_data->num_fill_in_drop = jx_ILUZeroDecompositionB(A, drop_tol, &indexA, &indexD, &index, &value);
            ilu_data->length = indexA[num_rows];
        }
        ilu_data->index = index;
        ilu_data->indexA = indexA;
        ilu_data->indexD = indexD;
        ilu_data->value = value;
        
        ilu_data->aux_vec = jx_SeqVectorCreate(num_rows);
        jx_SeqVectorInitialize(ilu_data->aux_vec);
        ilu_data->res_vec = jx_SeqVectorCreate(num_rows);
        jx_SeqVectorInitialize(ilu_data->res_vec);
        ilu_data->tmp_vec = jx_SeqVectorCreate(num_rows);
        jx_SeqVectorInitialize(ilu_data->tmp_vec);
    }
    else if (drop_tol == 0.0)
    {
        //jx_printf(" >>> SETUP: Drop-Tol = %le on ID = %d of %d\n", drop_tol, my_id, num_procs);
        indexA = jx_ILUZeroParallelDecompositionA(par_A, &indexD, &index, &value, &length, &dfill_in, ilu_data);
        jx_MPI_Reduce(&length, &maxlength, 1, JX_MPI_INT, MPI_MAX, 0, comm);
        jx_MPI_Reduce(&dfill_in, &maxdfill_in, 1, JX_MPI_INT, MPI_MAX, 0, comm);
        ilu_data->index = index;
        ilu_data->indexA = indexA;
        ilu_data->indexD = indexD;
        ilu_data->value = value;
        ilu_data->length = maxlength;
        ilu_data->num_fill_in_drop = maxdfill_in;
        
        // Allocate memory for auxiliary vector
        jx_ParCSRMatrixGetRowPartitioning(par_A, &part_aux);
        ilu_data->par_aux_vec = jx_ParVectorCreate(comm, jx_ParCSRMatrixGlobalNumRows(par_A), part_aux);
        jx_ParVectorInitialize(ilu_data->par_aux_vec);
        // Allocate memory for residual vector
        jx_ParCSRMatrixGetRowPartitioning(par_A, &part_res);
        ilu_data->par_res_vec = jx_ParVectorCreate(comm, jx_ParCSRMatrixGlobalNumRows(par_A), part_res);
        jx_ParVectorInitialize(ilu_data->par_res_vec);
    }
    else
    {
        jx_printf("\n >>> Wrong parameters in ILU Setup\n");
        exit(0);
    }
    
    return 0;
}

/*------------------------------------------------------------*/
/*------------------------ End of File -----------------------*/
/*------------------------------------------------------------*/
