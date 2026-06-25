//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  csr_matrix.c -- operations for CSR matrices.
 *  Date: 2014/07/05
 */

#include "jx_mv.h"

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixTDMGReorderByNodes
 * \brief Reorder a matrix ordered by variables by nodes
 * \author Yue Xiaoqiang
 * \date 2014/04/09
 */
jx_CSRMatrix *
jx_CSRMatrixTDMGReorderByNodes( jx_CSRMatrix *A, JX_Int num_equns )
{
    jx_CSRMatrix *C = NULL;
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Int *JA = jx_CSRMatrixJ(A);
    JX_Real *AA = jx_CSRMatrixData(A);
    JX_Int *IC = NULL;
    JX_Int *JC = NULL;
    JX_Real *CC = NULL;
    JX_Int RowA = 0, RowC = 0;
    JX_Int sub_num_rows = num_rows / num_equns;
    JX_Int num_nonzerosC, i, j, k, mdo, col, row_end;
    
    C = jx_CSRMatrixCreate(num_rows, num_rows, jx_CSRMatrixNumNonzeros(A));
    jx_CSRMatrixInitialize(C);
    IC = jx_CSRMatrixI(C);
    JC = jx_CSRMatrixJ(C);
    CC = jx_CSRMatrixData(C);
    IC[0] = 0;
    num_nonzerosC = 0;
    for (i = 0; i < sub_num_rows; i ++)
    {
        for (k = 0; k < num_equns; k ++)
        {
            RowC = i * num_equns + k;
            RowA = k * sub_num_rows + i;
            row_end = IA[RowA+1];
            for (j = IA[RowA]; j < row_end; j ++)
            {
                col = JA[j];
                mdo = col / sub_num_rows;
                JC[num_nonzerosC] = num_equns * (col - mdo * sub_num_rows) + mdo;
                CC[num_nonzerosC] = AA[j];
                num_nonzerosC ++;
            }
            IC[RowC+1] = num_nonzerosC;
        }
    }
    
    return C;
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixTDMGReorderByVariables
 * \brief Reorder a matrix ordered by nodes by variables
 * \author Yue Xiaoqiang
 * \date 2014/04/09
 */
jx_CSRMatrix *
jx_CSRMatrixTDMGReorderByVariables( jx_CSRMatrix *A, JX_Int num_equns )
{
    jx_CSRMatrix *C = NULL;
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Int *JA = jx_CSRMatrixJ(A);
    JX_Real *AA = jx_CSRMatrixData(A);
    JX_Int *IC = NULL;
    JX_Int *JC = NULL;
    JX_Real *CC = NULL;
    JX_Int RowA = 0, RowC = 0;
    JX_Int sub_num_rows = num_rows / num_equns;
    JX_Int num_nonzerosC, j, mdo, col, row_end;
    
    C = jx_CSRMatrixCreate(num_rows, num_rows, jx_CSRMatrixNumNonzeros(A));
    jx_CSRMatrixInitialize(C);
    IC = jx_CSRMatrixI(C);
    JC = jx_CSRMatrixJ(C);
    CC = jx_CSRMatrixData(C);
    IC[0] = 0;
    num_nonzerosC = 0;
    for (RowC = 0; RowC < num_rows; RowC ++)
    {
        mdo = RowC / sub_num_rows;
        RowA = num_equns * (RowC - mdo * sub_num_rows) + mdo;
        row_end = IA[RowA+1];
        for (j = IA[RowA]; j < row_end; j ++)
        {
            col = JA[j];
            mdo = col % num_equns;
            JC[num_nonzerosC] = (col - mdo) / num_equns + mdo * sub_num_rows;
            CC[num_nonzerosC] = AA[j];
            num_nonzerosC ++;
        }
        IC[RowC+1] = num_nonzerosC;
    }
    
    return C;
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixTDMGReorderAlongX
 * \brief Reorder a matrix ordered along x-axis direction
 * \author Yue Xiaoqiang
 * \date 04/09/2014
 */
jx_CSRMatrix *
jx_CSRMatrixTDMGReorderAlongX( jx_CSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
    jx_CSRMatrix *C = NULL;
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Int *JA = jx_CSRMatrixJ(A);
    JX_Real *AA = jx_CSRMatrixData(A);
    JX_Int *IC = NULL;
    JX_Int *JC = NULL;
    JX_Real *CC = NULL;
    JX_Int num_nonzerosC = 0, t_i = 1;
    JX_Int i, j, k, fxp, col, row_end;
    JX_Int ncn, mdo, ndn, bnx, bny;
    
    C = jx_CSRMatrixCreate(num_rows, num_rows, jx_CSRMatrixNumNonzeros(A));
    jx_CSRMatrixInitialize(C);
    IC = jx_CSRMatrixI(C);
    JC = jx_CSRMatrixJ(C);
    CC = jx_CSRMatrixData(C);
    IC[0] = 0;
    for (i = 0; i < ny; i ++)
    {
        for (j = 0; j < nx; j ++)
        {
            fxp = num_equns * (j * ny + i);
            for (k = 0; k < num_equns; k ++)
            {
                row_end = IA[fxp+1];
                for (col = IA[fxp++]; col < row_end; col ++)
                {
                    ncn = JA[col];
                    mdo = ncn / num_equns;
                    ndn = ncn - num_equns * mdo;
                    bny = mdo / ny;
                    bnx = mdo - bny * ny;
                    JC[num_nonzerosC] = num_equns * (bnx * nx + bny) + ndn;
                    CC[num_nonzerosC++] = AA[col];
                }
                IC[t_i++] = num_nonzerosC;
            }
        }
    }
    
    return C;
}

/*!
 * \fn jx_CSRMatrix *jx_CSRMatrixTDMGReorderAlongY
 * \brief Reorder a matrix ordered along y-axis direction
 * \author Yue Xiaoqiang
 * \date 04/09/2014
 */
jx_CSRMatrix *
jx_CSRMatrixTDMGReorderAlongY( jx_CSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
    jx_CSRMatrix *C = NULL;
    JX_Int num_rows = jx_CSRMatrixNumRows(A);
    JX_Int *IA = jx_CSRMatrixI(A);
    JX_Int *JA = jx_CSRMatrixJ(A);
    JX_Real *AA = jx_CSRMatrixData(A);
    JX_Int *IC = NULL;
    JX_Int *JC = NULL;
    JX_Real *CC = NULL;
    JX_Int num_nonzerosC = 0, t_i = 1;
    JX_Int i, j, k, fxp, col, row_end;
    JX_Int ncn, mdo, ndn, bnx, bny;
    
    C = jx_CSRMatrixCreate(num_rows, num_rows, jx_CSRMatrixNumNonzeros(A));
    jx_CSRMatrixInitialize(C);
    IC = jx_CSRMatrixI(C);
    JC = jx_CSRMatrixJ(C);
    CC = jx_CSRMatrixData(C);
    IC[0] = 0;
    for (i = 0; i < nx; i ++)
    {
        for (j = 0; j < ny; j ++)
        {
            fxp = num_equns * (j * nx + i);
            for (k = 0; k < num_equns; k ++)
            {
                row_end = IA[fxp+1];
                for (col = IA[fxp++]; col < row_end; col ++)
                {
                    ncn = JA[col];
                    mdo = ncn / num_equns;
                    ndn = ncn - num_equns * mdo;
                    bny = mdo / nx;
                    bnx = mdo - bny * nx;
                    JC[num_nonzerosC] = num_equns * (bnx * ny + bny) + ndn;
                    CC[num_nonzerosC++] = AA[col];
                }
                IC[t_i++] = num_nonzerosC;
            }
        }
    }
    
    return C;
}
