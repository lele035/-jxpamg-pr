//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  numbering_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_Numbering_dhCreate"
void jx_Numbering_dhCreate( jx_Numbering_dh *numb )
{
    JX_START_FUNC_DH
    struct _jx_numbering_dh *tmp = (struct _jx_numbering_dh *)JX_MALLOC_DH(sizeof(struct _jx_numbering_dh)); JX_CHECK_V_ERROR;
   *numb = tmp;
    tmp->size = 0;
    tmp->first = 0;
    tmp->m = 0;
    tmp->num_ext = 0;
    tmp->num_extLo = 0;
    tmp->num_extHi = 0;
    tmp->idx_ext = NULL;
    tmp->idx_extLo = NULL;
    tmp->idx_extHi = NULL;
    tmp->idx_ext = NULL;
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_Numbering");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Numbering_dhDestroy"
void jx_Numbering_dhDestroy( jx_Numbering_dh numb )
{
    JX_START_FUNC_DH
    if (numb->global_to_local != NULL)
    {
        jx_Hash_i_dhDestroy(numb->global_to_local); JX_CHECK_V_ERROR;
    }
    if (numb->idx_ext != NULL)
    {
        JX_FREE_DH(numb->idx_ext); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(numb); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Numbering_dhSetup"
void jx_Numbering_dhSetup( jx_Numbering_dh numb, jx_Mat_dh mat )
{
    JX_START_FUNC_DH
    JX_Int i, len, *cval = mat->cval;
    JX_Int num_ext, num_extLo, num_extHi;
    JX_Int m = mat->m, size;
    jx_Hash_i_dh global_to_local_hash;
    JX_Int first = mat->beg_row, last  = first + m;
    JX_Int *idx_ext;
    JX_Int data;
    
    numb->first = first;
    numb->m = m;
    numb->size = size = m;
    jx_Hash_i_dhCreate(&(numb->global_to_local), m); JX_CHECK_V_ERROR;
    global_to_local_hash = numb->global_to_local;
    idx_ext = numb->idx_ext = (JX_Int*)JX_MALLOC_DH(size*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    len = mat->rp[m];
    num_ext = num_extLo = num_extHi = 0;
    for (i = 0; i < len; i ++)
    {
        JX_Int index = cval[i];
        
        if (index < first || index >= last)
        {
            data = jx_Hash_i_dhLookup(global_to_local_hash, cval[i]); JX_CHECK_V_ERROR;
            if (data == -1)
            {
                if (m+num_ext >= size)
                {
                    JX_Int newSize = size * 1.5;
                    JX_Int *tmp = (JX_Int *)JX_MALLOC_DH(newSize*sizeof(JX_Int)); JX_CHECK_V_ERROR;
                    
                    memcpy(tmp, idx_ext, size*sizeof(size));
                    JX_FREE_DH(idx_ext); JX_CHECK_V_ERROR;
                    size = numb->size = newSize;
                    numb->idx_ext = idx_ext = tmp;
                    JX_SET_INFO("reallocated ext_idx[]");
                }
                jx_Hash_i_dhInsert(global_to_local_hash, index, num_ext); JX_CHECK_V_ERROR;
                idx_ext[num_ext] = index;
                num_ext ++;
                if (index < first)
                {
                    num_extLo ++;
                }
                else
                {
                    num_extHi ++;
                }
            }
        }
    }
    numb->num_ext = num_ext;
    numb->num_extLo = num_extLo;
    numb->num_extHi = num_extHi;
    numb->idx_extLo = idx_ext;
    numb->idx_extHi = idx_ext + num_extLo;
    jx_shellSort_int(num_ext, idx_ext);
    jx_Hash_i_dhReset(global_to_local_hash); JX_CHECK_V_ERROR;
    for (i = 0; i < num_ext; i ++)
    {
        jx_Hash_i_dhInsert(global_to_local_hash, idx_ext[i], i+m); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Numbering_dhGlobalToLocal"
void jx_Numbering_dhGlobalToLocal( jx_Numbering_dh numb, JX_Int len, JX_Int *global, JX_Int *local )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int first = numb->first;
    JX_Int last = first + numb->m;
    JX_Int data;
    jx_Hash_i_dh global_to_local = numb->global_to_local;
    
    for (i = 0; i < len; i ++)
    {
        JX_Int idxGlobal = global[i];
        
        if (idxGlobal >= first && idxGlobal < last)
        {
            local[i] = idxGlobal - first;
        }
        else
        {
            data = jx_Hash_i_dhLookup(global_to_local, idxGlobal); JX_CHECK_V_ERROR;
            if (data == -1)
            {
                jx_sprintf(jx_msgBuf_dh, "global index %i not found in map\n", idxGlobal);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
            else
            {
                local[i] = data;
            }
        }
    }
    JX_END_FUNC_DH
}
