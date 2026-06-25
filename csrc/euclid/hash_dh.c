//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  hash_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

static void jx_Hash_dhInit_private( jx_Hash_dh h, JX_Int s );

#define JX_CUR_MARK_INIT -1

struct _jx_hash_node_private
{
    JX_Int key;
    JX_Int mark;
    jx_HashData data;
};

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhCreate"
void jx_Hash_dhCreate( jx_Hash_dh *h, JX_Int size )
{
    JX_START_FUNC_DH
    struct _jx_hash_dh *tmp = (struct _jx_hash_dh *)JX_MALLOC_DH(sizeof(struct _jx_hash_dh)); JX_CHECK_V_ERROR;
   *h = tmp;
    tmp->size = 0;
    tmp->count = 0;
    tmp->curMark = JX_CUR_MARK_INIT + 1;
    tmp->data = NULL;
    jx_Hash_dhInit_private(*h,size); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhDestroy"
void jx_Hash_dhDestroy( jx_Hash_dh h )
{
    JX_START_FUNC_DH
    if (h->data != NULL)
    {
        JX_FREE_DH(h->data); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(h); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhReset"
void jx_Hash_dhReset( jx_Hash_dh h )
{
    JX_START_FUNC_DH
    h->count = 0;
    h->curMark += 1;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhInit_private"
void jx_Hash_dhInit_private( jx_Hash_dh h, JX_Int s )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int size = 16;
    jx_HashRecord *data;
    
    while (size < s) size *= 2;
    if ((size-s) < (.1*size))
    {
        size *= 2.0;
    }
    h->size = size;
    data = h->data = (jx_HashRecord *)JX_MALLOC_DH(size*sizeof(jx_HashRecord)); JX_CHECK_V_ERROR;
    for (i = 0; i < size; ++ i)
    {
        data[i].key = -1;
        data[i].mark = -1;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhLookup"
jx_HashData *jx_Hash_dhLookup( jx_Hash_dh h, JX_Int key )
{
    JX_START_FUNC_DH
    JX_Int i, start;
    JX_Int curMark = h->curMark;
    JX_Int size = h->size;
    jx_HashData *retval = NULL;
    jx_HashRecord *data = h->data;
    
    JX_HASH_1(key, size, &start)
    for (i = 0; i < size; ++ i)
    {
        JX_Int tmp, idx;
        
        JX_HASH_2(key, size, &tmp)
        idx = (start + i * tmp) % size;
        if (data[idx].mark != curMark)
        {
            break;
        }
        else
        {
            if (data[idx].key == key)
            {
                retval = &(data[idx].data);
                break;
            }
        }
    }
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhInsert"
void jx_Hash_dhInsert( jx_Hash_dh h, JX_Int key, jx_HashData *dataIN )
{
    JX_START_FUNC_DH
    JX_Int i, start, size = h->size;
    JX_Int curMark = h->curMark;
    jx_HashRecord *data;
    
    data = h->data;
    h->count += 1;
    if (h->count == h->size)
    {
        JX_SET_V_ERROR("hash table overflow; rehash need implementing!");
    }
    JX_HASH_1(key, size, &start)
    for (i = 0; i < size; ++ i)
    {
        JX_Int tmp, idx;
        
        JX_HASH_2(key, size, &tmp)
        idx = (start + i * tmp) % size;
        if (data[idx].mark < curMark)
        {
            data[idx].key = key;
            data[idx].mark = curMark;
            memcpy(&(data[idx].data), dataIN, sizeof(jx_HashData));
            break;
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_dhPrint"
void jx_Hash_dhPrint( jx_Hash_dh h, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i, size = h->size;
    JX_Int curMark = h->curMark;
    jx_HashRecord *data = h->data;
    
    jx_fprintf(fp, "\n--------------------------- hash table \n");
    for (i = 0; i < size; ++ i)
    {
        if (data[i].mark == curMark)
        {
            jx_fprintf(fp, "key = %2i;  iData = %3i;  fData = %g\n",
                               data[i].key, data[i].data.iData, data[i].data.fData);
        }
    }
    jx_fprintf(fp, "\n");
    JX_END_FUNC_DH
}
