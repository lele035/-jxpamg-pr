//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  hash_i_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#define JX_DEFAULT_TABLE_SIZE 16

static void jx_rehash_private( jx_Hash_i_dh h );

#define JX_HASH_1(k,size,idxOut) \
      { \
       *idxOut = k % size; \
      }

#define JX_HASH_2(k,size,idxOut) \
      { \
        JX_Int r = k % (size-13); \
        r = (r % 2) ? r : r+1; \
       *idxOut = r; \
      }

typedef struct _jx_hash_i_node_private jx_Hash_i_Record;

struct _jx_hash_i_node_private
{
    JX_Int key;
    JX_Int mark;
    JX_Int data;
};

struct _jx_hash_i_dh
{
    JX_Int size;
    JX_Int count;
    JX_Int curMark;
    jx_Hash_i_Record *data;
};

#undef __FUNC__
#define __FUNC__ "jx_Hash_i_dhCreate"
void jx_Hash_i_dhCreate( jx_Hash_i_dh *h, JX_Int sizeIN )
{
    JX_START_FUNC_DH
    JX_Int i, size;
    jx_Hash_i_Record *tmp2;
    struct _jx_hash_i_dh *tmp;
    
    size = JX_DEFAULT_TABLE_SIZE;
    if (sizeIN == -1)
    {
        sizeIN = size = JX_DEFAULT_TABLE_SIZE;
    }
    tmp = (struct _jx_hash_i_dh *)JX_MALLOC_DH( sizeof(struct _jx_hash_i_dh)); JX_CHECK_V_ERROR;
   *h = tmp;
    tmp->size = 0;
    tmp->count = 0;
    tmp->curMark = 0;
    tmp->data = NULL;
    while (size < sizeIN) size *= 2;
    if ((size-sizeIN) < (.1*size))
    {
        size *= 2.0;
    }
    tmp->size = size;
    tmp2 = tmp->data = (jx_Hash_i_Record *)JX_MALLOC_DH(size*sizeof(jx_Hash_i_Record)); JX_CHECK_V_ERROR;
    for (i = 0; i < size; ++ i)
    {
        tmp2[i].key = -1;
        tmp2[i].mark = -1;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_i_dhDestroy"
void jx_Hash_i_dhDestroy( jx_Hash_i_dh h )
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
#define __FUNC__ "jx_Hash_i_dhReset"
void jx_Hash_i_dhReset( jx_Hash_i_dh h )
{
    JX_START_FUNC_DH
    h->count = 0;
    h->curMark += 1;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_i_dhLookup"
JX_Int jx_Hash_i_dhLookup( jx_Hash_i_dh h, JX_Int key )
{
    JX_START_FUNC_DH
    JX_Int idx, inc, i, start;
    JX_Int curMark = h->curMark;
    JX_Int size = h->size;
    JX_Int retval = -1;
    jx_Hash_i_Record *data = h->data;
    
    JX_HASH_1(key, size, &start)
    JX_HASH_2(key, size, &inc)
    for (i = 0; i < size; ++ i)
    {
        idx = (start + i * inc) % size;
        if (data[idx].mark != curMark)
        {
            break;
        }
        else
        {
            if (data[idx].key == key)
            {
                retval = data[idx].data;
                break;
            }
        }
    }
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Hash_i_dhInsert"
void jx_Hash_i_dhInsert( jx_Hash_i_dh h, JX_Int key, JX_Int dataIN )
{
    JX_START_FUNC_DH
    JX_Int i, idx, inc, start, size;
    JX_Int curMark = h->curMark;
    jx_Hash_i_Record *data;
    jx_bool success = jx_false;
    
    if (dataIN < 0)
    {
        jx_sprintf(jx_msgBuf_dh, "data = %i must be >= 0", dataIN);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    if (h->count >= 0.9 * h->size)
    {
        jx_rehash_private(h); JX_CHECK_V_ERROR;
    }
    size = h->size;
    data = h->data;
    h->count += 1;
    JX_HASH_1(key, size, &start)
    JX_HASH_2(key, size, &inc)
    for (i = 0; i < size; ++ i)
    {
        idx = (start + i * inc) % size;
        if (data[idx].mark == curMark && data[idx].key == key)
        {
            jx_sprintf(jx_msgBuf_dh, "key,data= <%i, %i> already inserted", key, dataIN);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
        if (data[idx].mark < curMark)
        {
            data[idx].key = key;
            data[idx].mark = curMark;
            data[idx].data = dataIN;
            success = jx_true;
            break;
        }
    }
    if (!success)
    {
        jx_sprintf(jx_msgBuf_dh, "Failed to insert key= %i, data= %i", key, dataIN);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_rehash_private"
void jx_rehash_private( jx_Hash_i_dh h )
{
    JX_START_FUNC_DH
    JX_Int i, old_size = h->size, new_size = old_size * 2, oldCurMark = h->curMark;
    jx_Hash_i_Record *oldData = h->data, *newData;
    
    jx_sprintf(jx_msgBuf_dh, "rehashing; old_size= %i, new_size= %i", old_size, new_size);
    JX_SET_INFO(jx_msgBuf_dh);
    newData = (jx_Hash_i_Record *)JX_MALLOC_DH(new_size*sizeof(jx_Hash_i_Record)); JX_CHECK_V_ERROR;
    for (i = 0; i < new_size; ++ i)
    {
        newData[i].key = -1;
        newData[i].mark = -1;
    }
    h->size = new_size;
    h->data = newData;
    h->count = 0;
    h->curMark = 0;
    for (i = h->count; i < new_size; ++ i)
    {
        newData[i].key = -1;
        newData[i].mark = -1;
    }
    for (i = 0; i < old_size; ++ i)
    {
        if (oldData[i].mark == oldCurMark)
        {
            jx_Hash_i_dhInsert(h, oldData[i].key, oldData[i].data); JX_CHECK_V_ERROR;
        }
    }
    JX_FREE_DH(oldData); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}
