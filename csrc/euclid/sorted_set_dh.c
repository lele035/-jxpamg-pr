//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  sorted_set_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_SortedSet_dhCreate"
void jx_SortedSet_dhCreate( jx_SortedSet_dh *ss, JX_Int size )
{
    JX_START_FUNC_DH
    struct _jx_sortedset_dh *tmp = (struct _jx_sortedset_dh *)JX_MALLOC_DH(sizeof(struct _jx_sortedset_dh)); JX_CHECK_V_ERROR;
    
   *ss= tmp;
    tmp->n = size;
    tmp->list = (JX_Int *)JX_MALLOC_DH(size*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    tmp->count = 0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedSet_dhDestroy"
void jx_SortedSet_dhDestroy( jx_SortedSet_dh ss )
{
    JX_START_FUNC_DH
    if (ss->list != NULL)
    {
        JX_FREE_DH(ss->list); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(ss); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedSet_dhInsert"
void jx_SortedSet_dhInsert( jx_SortedSet_dh ss, JX_Int idx )
{
    JX_START_FUNC_DH
    jx_bool isInserted = jx_false;
    JX_Int ct = ss->count;
    JX_Int *list = ss->list;
    JX_Int i, n = ss->n;
    
    for (i = 0; i < ct; ++ i)
    {
        if (list[i] == idx)
        {
            isInserted = jx_true;
            break;
        }
    }
    if (!isInserted)
    {
        if (ct == n)
        {
            JX_Int *tmp = (JX_Int *)JX_MALLOC_DH(n*2*sizeof(JX_Int)); JX_CHECK_V_ERROR;
            
            memcpy(tmp, list, n*sizeof(JX_Int));
            JX_FREE_DH(list); JX_CHECK_V_ERROR;
            list = ss->list = tmp;
            ss->n *= 2;
        }
        list[ct] = idx;
        ss->count += 1;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedSet_dhGetList"
void jx_SortedSet_dhGetList( jx_SortedSet_dh ss, JX_Int **list, JX_Int *count )
{
    JX_START_FUNC_DH
    jx_shellSort_int(ss->count, ss->list);
   *list = ss->list;
   *count = ss->count;
    JX_END_FUNC_DH
}
