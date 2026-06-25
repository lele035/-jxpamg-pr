//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  shell_sort_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_shellSort_int"
void jx_shellSort_int( const JX_Int n, JX_Int *x )
{
    JX_START_FUNC_DH
    JX_Int m, max, j, k, itemp;
    
    m = n/2;
    while (m > 0)
    {
        max = n - m;
        for (j = 0; j < max; j ++)
        {
            for (k = j; k >= 0; k -= m)
            {
                if (x[k+m] >= x[k]) break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }
        m = m / 2;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_shellSort_float"
void jx_shellSort_float( const JX_Int n, JX_Real *x )
{
    JX_START_FUNC_DH
    JX_Int m, max, j, k;
    JX_Real itemp;
    
    m = n / 2;
    while (m > 0)
    {
        max = n - m;
        for (j = 0; j < max; j ++)
        {
            for (k = j; k >= 0; k -= m)
            {
                if (x[k+m] >= x[k]) break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }
        m = m / 2;
    }
    JX_END_FUNC_DH
}
