//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  vector.c -- operations for sequential vectors.
 *  Date: 2014/07/05
 */

#include "jx_mv.h"

/*!
 * \fn jx_Vector *jx_VectorTDMGReorderByNodes
 * \brief Reorder a vector ordered by variables by nodes
 * \author Yue Xiaoqiang
 * \date 2014/04/09
 */
jx_Vector *
jx_VectorTDMGReorderByNodes( jx_Vector *x, JX_Int num_equns )
{
    JX_Int size = jx_VectorSize(x);
    JX_Real *x_data = jx_VectorData(x);
    jx_Vector *y = jx_SeqVectorCreate(size);
    JX_Real *y_data = NULL;
    JX_Int sub_size = size / num_equns;
    JX_Int i, k, t_i = 0;
    
    jx_SeqVectorInitialize(y);
    y_data = jx_VectorData(y);
    for (i = 0; i < sub_size; i ++)
    {
        for (k = 0; k < num_equns-2; k ++)
        {
            y_data[t_i++] = x_data[k*sub_size+i];
        }
        y_data[t_i++] = x_data[k*sub_size+i];
        k ++;
        y_data[t_i++] = x_data[k*sub_size+i];
    }
    
    return y;
}

/*!
 * \fn jx_Vector *jx_VectorTDMGReorderByVariables
 * \brief Reorder a vector ordered by nodes by variables
 * \author Yue Xiaoqiang
 * \date 2014/04/09
 */
jx_Vector *
jx_VectorTDMGReorderByVariables( jx_Vector *x, JX_Int num_equns )
{
    JX_Int size = jx_VectorSize(x);
    JX_Real *x_data = jx_VectorData(x);
    jx_Vector *y = jx_SeqVectorCreate(size);
    JX_Real *y_data = NULL;
    JX_Int sub_size = size / num_equns;
    JX_Int Rowx = 0, Rowy, mdo;
    
    jx_SeqVectorInitialize(y);
    y_data = jx_VectorData(y);
    for (Rowy = 0; Rowy < size; Rowy ++)
    {
        mdo = Rowy / sub_size;
        Rowx = num_equns * (Rowy - mdo * sub_size) + mdo;
        y_data[Rowy] = x_data[Rowx];
    }
    
    return y;
}

/*!
 * \fn jx_Vector *jx_VectorTDMGReorderAlongX
 * \brief Reorder a vector ordered by x-axis direction
 * \author Yue Xiaoqiang
 * \date 04/09/2014
 */
jx_Vector *
jx_VectorTDMGReorderAlongX( jx_Vector *x, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
    JX_Int size = jx_VectorSize(x);
    JX_Real *x_data = jx_VectorData(x);
    jx_Vector *y = jx_SeqVectorCreate(size);
    JX_Real *y_data = NULL;
    JX_Int i, j, k, fxp, t_i = 0;
    
    jx_SeqVectorInitialize(y);
    y_data = jx_VectorData(y);
    for (i = 0; i < ny; i ++)
    {
        for (j = 0; j < nx; j ++)
        {
            fxp = num_equns * (j * ny + i);
            for (k = 0; k < num_equns; k ++)
            {
                y_data[t_i++] = x_data[fxp+k];
            }
        }
    }
    
    return y;
}

/*!
 * \fn jx_Vector *jx_VectorTDMGReorderAlongY
 * \brief Reorder a vector ordered by y-axis direction
 * \author Yue Xiaoqiang
 * \date 04/09/2014
 */
jx_Vector *
jx_VectorTDMGReorderAlongY( jx_Vector *x, JX_Int num_equns, JX_Int nx, JX_Int ny )
{
    JX_Int size = jx_VectorSize(x);
    JX_Real *x_data = jx_VectorData(x);
    jx_Vector *y = jx_SeqVectorCreate(size);
    JX_Real *y_data = NULL;
    JX_Int i, j, k, fxp, t_i = 0;
    
    jx_SeqVectorInitialize(y);
    y_data = jx_VectorData(y);
    for (i = 0; i < nx; i ++)
    {
        for (j = 0; j < ny; j ++)
        {
            fxp = num_equns * (j * nx + i);
            for (k = 0; k < num_equns; k ++)
            {
                y_data[t_i++] = x_data[fxp++];
            }
        }
    }
    
    return y;
}
