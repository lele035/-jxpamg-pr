//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_mem_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

typedef struct
{
    JX_Real size;
    JX_Real cookie;
} jx_memRecord_dh;

struct _jx_mem_dh
{
    JX_Real maxMem;
    JX_Real curMem;
    JX_Real totalMem;
    JX_Real mallocCount;
    JX_Real freeCount;
};

#undef __FUNC__
#define __FUNC__ "jx_Mem_dhCreate"
void jx_Mem_dhCreate( jx_Mem_dh *m )
{
    JX_START_FUNC_DH
    struct _jx_mem_dh *tmp = (struct _jx_mem_dh *)JX_PRIVATE_MALLOC(sizeof(struct _jx_mem_dh)); JX_CHECK_V_ERROR;
   *m = tmp;
    tmp->maxMem = 0.0;
    tmp->curMem = 0.0;
    tmp->totalMem = 0.0;
    tmp->mallocCount = 0.0;
    tmp->freeCount = 0.0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mem_dhDestroy"
void jx_Mem_dhDestroy( jx_Mem_dh m )
{
    JX_START_FUNC_DH
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-eu_mem"))
    {
        jx_Mem_dhPrint(m, stdout, jx_false); JX_CHECK_V_ERROR;
    }
    JX_PRIVATE_FREE(m);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Mem_dhMalloc"
void *jx_Mem_dhMalloc( jx_Mem_dh m, size_t size )
{
    JX_START_FUNC_DH_2
    void *retval;
    jx_memRecord_dh *tmp;
    size_t s = size + 2 * sizeof(jx_memRecord_dh);
    void *address;
    
    address = JX_PRIVATE_MALLOC(s);
    if (address == NULL)
    {
        jx_sprintf(jx_msgBuf_dh, "JX_PRIVATE_MALLOC failed; totalMem = %g; requested additional = %i", m->totalMem, (JX_Int)s);
        JX_SET_ERROR(NULL, jx_msgBuf_dh);
    }
    retval = (char *)address + sizeof(jx_memRecord_dh);
    tmp = (jx_memRecord_dh *)address;
    tmp->size = (JX_Real)s;
    m->mallocCount += 1;
    m->totalMem += (JX_Real)s;
    m->curMem += (JX_Real)s;
    m->maxMem = JX_MAX(m->maxMem, m->curMem);
    JX_END_FUNC_VAL_2(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Mem_dhFree"
void jx_Mem_dhFree( jx_Mem_dh m, void *ptr )
{
    JX_START_FUNC_DH_2
    JX_Real size;
    char *tmp = (char *)ptr;
    jx_memRecord_dh *rec;
    tmp -= sizeof(jx_memRecord_dh);
    rec = (jx_memRecord_dh *)tmp;
    size = rec->size;
    jx_mem_dh->curMem -= size;
    jx_mem_dh->freeCount += 1;
    JX_PRIVATE_FREE(tmp);
    JX_END_FUNC_DH_2
}

#undef __FUNC__
#define __FUNC__ "jx_Mem_dhPrint"
void jx_Mem_dhPrint( jx_Mem_dh m, FILE *fp, jx_bool allPrint )
{
    JX_START_FUNC_DH_2
    if (fp == NULL) JX_SET_V_ERROR("fp == NULL");
    if (jx_myid_dh == 0 || allPrint)
    {
        JX_Real tmp;
        
        jx_fprintf(fp, "---------------------- Euclid memory report (start)\n");
        jx_fprintf(fp, "malloc calls = %g\n", m->mallocCount);
        jx_fprintf(fp, "free   calls = %g\n", m->freeCount);
        jx_fprintf(fp, "curMem          = %g Mbytes (should be zero)\n", m->curMem/1000000);
        tmp = m->totalMem / 1000000;
        jx_fprintf(fp, "total allocated = %g Mbytes\n", tmp);
        jx_fprintf(fp, "max malloc      = %g Mbytes (max allocated at any point in time)\n", m->maxMem/1000000);
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "---------------------- Euclid memory report (end)\n");
    }
    JX_END_FUNC_DH_2
}
