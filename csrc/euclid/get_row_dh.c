//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  get_row_dh.c
 *  Date: 2013/01/21
 */

#include "jx_euclid.h"

#if defined(JXPAMG_GET_ROW)

#undef __FUNC__
#define __FUNC__ "jx_EuclidGetRow (JXPAMG_GET_ROW)"
void jx_EuclidGetRow( void *A, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val )
{
    JX_START_FUNC_DH
    JX_Int ierr;
    jx_ParCSRMatrix *mat = (jx_ParCSRMatrix *)A;
    
    ierr = jx_ParCSRMatrixGetRow(mat, row, len, ind, val);
    if (ierr)
    {
        jx_sprintf(jx_msgBuf_dh, "jx_ParCSRMatrixGetRow(row= %i) returned %i", row+1, ierr);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidRestoreRow (JXPAMG_GET_ROW)"
void jx_EuclidRestoreRow( void *A, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val )
{
    JX_START_FUNC_DH
    JX_Int ierr;
    jx_ParCSRMatrix *mat = (jx_ParCSRMatrix *)A;
    
    ierr = jx_ParCSRMatrixRestoreRow(mat, row, len, ind, val);
    if (ierr)
    {
        jx_sprintf(jx_msgBuf_dh, "jx_ParCSRMatrixRestoreRow(row= %i) returned %i", row+1, ierr);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidGetDimensions (JXPAMG_GET_ROW)"
void jx_EuclidGetDimensions( void *A, JX_Int *beg_row, JX_Int *rowsLocal, JX_Int *rowsGlobal )
{
    JX_START_FUNC_DH
    JX_Int ierr, m, n;
    JX_Int row_start, row_end, col_start, col_end;
    jx_ParCSRMatrix *mat = (jx_ParCSRMatrix *)A;
    
    ierr = jx_ParCSRMatrixGetDims(mat, &m, &n);
    if (ierr)
    {
        jx_sprintf(jx_msgBuf_dh, "jx_ParCSRMatrixGetDims() returned %i", ierr);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    ierr = jx_ParCSRMatrixGetLocalRange(mat, &row_start, &row_end, &col_start, &col_end);
    if (ierr)
    {
        jx_sprintf(jx_msgBuf_dh, "jx_ParCSRMatrixGetLocalRange() returned %i", ierr);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
   *beg_row = row_start;
   *rowsLocal = (row_end - row_start + 1);
   *rowsGlobal = n;
    JX_END_FUNC_DH
}

#else

#undef __FUNC__
#define __FUNC__ "jx_EuclidGetRow (ERROR)"
void jx_EuclidGetRow( void *A, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val )
{
    JX_START_FUNC_DH
    JX_SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidRestoreRow (ERROR)"
void jx_EuclidRestoreRow( void *A, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val )
{
    JX_START_FUNC_DH
    JX_SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidGetDimensions (ERROR)"
void jx_EuclidGetDimensions( void *A, JX_Int *beg_row, JX_Int *rowsLocal, JX_Int *rowsGlobal )
{
    JX_START_FUNC_DH
    JX_SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidReadLocalNz (ERROR)"
JX_Int jx_EuclidReadLocalNz( void *A )
{
    JX_START_FUNC_DH
    JX_SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
    JX_END_FUNC_DH
}

#endif

#undef __FUNC__
#define __FUNC__ "jx_PrintMatUsingGetRow"
void jx_PrintMatUsingGetRow( void* A, JX_Int beg_row, JX_Int m, JX_Int *n2o_row, JX_Int *n2o_col, char *filename )
{
    JX_START_FUNC_DH
    FILE *fp;
    JX_Int *o2n_col = NULL, pe, i, j, *cval, len;
    JX_Int newCol, newRow;
    JX_Real *aval;
    
    if (n2o_col != NULL)
    {
        o2n_col = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        for (i = 0; i < m; ++ i) o2n_col[n2o_col[i]] = i;
    }
    for (pe = 0; pe < jx_np_dh; ++ pe)
    {
        jx_MPI_Barrier(jx_comm_dh);
        if (jx_myid_dh == pe)
        {
            if (pe == 0)
            {
                fp = fopen(filename, "w");
            }
            else
            {
                fp = fopen(filename, "a");
            }
            if (fp == NULL)
            {
                jx_sprintf(jx_msgBuf_dh, "can't open %s for writing\n", filename);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
            for (i = 0; i < m; ++ i)
            {
                if (n2o_row == NULL)
                {
                    jx_EuclidGetRow(A, i+beg_row, &len, &cval, &aval); JX_CHECK_V_ERROR;
                    for (j = 0; j < len; ++ j)
                    {
                        jx_fprintf(fp, "%i %i %g\n", i+1, cval[j], aval[j]);
                    }
                    jx_EuclidRestoreRow(A, i, &len, &cval, &aval); JX_CHECK_V_ERROR;
                }
                else
                {
                    newRow = n2o_row[i] + beg_row;
                    jx_EuclidGetRow(A, newRow, &len, &cval, &aval); JX_CHECK_V_ERROR;
                    for (j = 0; j < len; ++ j)
                    {
                        newCol = o2n_col[cval[j]-beg_row] + beg_row;
                        jx_fprintf(fp, "%i %i %g\n", i+1, newCol, aval[j]);
                    }
                    jx_EuclidRestoreRow(A, i, &len, &cval, &aval); JX_CHECK_V_ERROR;
                }
            }
            fclose(fp);
        }
    }
    if (n2o_col != NULL)
    {
        JX_FREE_DH(o2n_col); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Euclid_dhInputJXpamgMat"
void jx_Euclid_dhInputJXpamgMat( jx_Euclid_dh ctx, JX_ParCSRMatrix A )
{
    JX_START_FUNC_DH
    JX_Int M, N;
    JX_Int beg_row, end_row, junk;
    
    jx_ParCSRMatrixGetDims((jx_ParCSRMatrix *)A, &M, &N);
    if (M != N)
    {
        jx_sprintf(jx_msgBuf_dh, "Global matrix is not square: M= %i, N= %i", M, N);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    jx_ParCSRMatrixGetLocalRange((jx_ParCSRMatrix *)A, &beg_row, &end_row, &junk, &junk);
    ctx->m = end_row - beg_row + 1;
    ctx->n = M;
    ctx->A = (void *)A;
    JX_END_FUNC_DH
}
