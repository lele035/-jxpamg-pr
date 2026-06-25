//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  sorted_list_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

struct _jx_sortedList_dh
{
    JX_Int m;
    JX_Int row;
    JX_Int beg_row;
    JX_Int beg_rowP;
    JX_Int count;
    JX_Int countMax;
    JX_Int *o2n_local;
    jx_Hash_i_dh o2n_external;
    
    jx_SRecord *list;
    JX_Int alloc;
    JX_Int getLower;
    JX_Int get;
    
    jx_bool debug;
};

static void jx_lengthen_list_private( jx_SortedList_dh sList );

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhCreate"
void jx_SortedList_dhCreate( jx_SortedList_dh *sList )
{
    JX_START_FUNC_DH
    struct _jx_sortedList_dh *tmp = (struct _jx_sortedList_dh *)JX_MALLOC_DH(sizeof(struct _jx_sortedList_dh)); JX_CHECK_V_ERROR;
    
   *sList = tmp;
    tmp->m = 0;
    tmp->row = -1;
    tmp->beg_row = 0;
    tmp->count = 1;
    tmp->countMax = 1;
    tmp->o2n_external = NULL;
    tmp->o2n_local = NULL;
    tmp->get = 0;
    tmp->getLower = 0;
    tmp->alloc = 0;
    tmp->list = NULL;
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_SortedList");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhDestroy"
void jx_SortedList_dhDestroy( jx_SortedList_dh sList )
{
    JX_START_FUNC_DH
    if (sList->list != NULL)
    {
        JX_FREE_DH(sList->list); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(sList); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhInit"
void jx_SortedList_dhInit( jx_SortedList_dh sList, jx_SubdomainGraph_dh sg )
{
    JX_START_FUNC_DH
    sList->o2n_local = sg->o2n_col;
    sList->m = sg->m;
    sList->beg_row = sg->beg_row[jx_myid_dh];
    sList->beg_rowP = sg->beg_rowP[jx_myid_dh];
    sList->count = 1;
    sList->countMax = 1;
    sList->o2n_external = sg->o2n_ext;
    sList->alloc = sList->m + 5;
    sList->list = (jx_SRecord *)JX_MALLOC_DH(sList->alloc*sizeof(jx_SRecord));
    sList->list[0].col = INT_MAX;
    sList->list[0].next = 0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhReset"
void jx_SortedList_dhReset( jx_SortedList_dh sList, JX_Int row )
{
    JX_START_FUNC_DH
    sList->row = row;
    sList->count = 1;
    sList->countMax = 1;
    sList->get = 0;
    sList->getLower = 0;
    sList->list[0].next = 0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhReadCount"
JX_Int jx_SortedList_dhReadCount( jx_SortedList_dh sList )
{
    JX_START_FUNC_DH
    JX_END_FUNC_VAL(sList->count-1)
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhResetGetSmallest"
void jx_SortedList_dhResetGetSmallest( jx_SortedList_dh sList )
{
    JX_START_FUNC_DH
    sList->getLower = 0;
    sList->get = 0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhGetSmallest"
jx_SRecord *jx_SortedList_dhGetSmallest( jx_SortedList_dh sList )
{
    JX_START_FUNC_DH
    jx_SRecord *node = NULL;
    jx_SRecord *list = sList->list;
    JX_Int get = sList->get;
    
    get = list[get].next;
    if (list[get].col < INT_MAX)
    {
        node = &(list[get]);
        sList->get = get;
    }
    JX_END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhGetSmallestLowerTri"
jx_SRecord *jx_SortedList_dhGetSmallestLowerTri( jx_SortedList_dh sList )
{
    JX_START_FUNC_DH
    jx_SRecord *node = NULL;
    jx_SRecord *list = sList->list;
    JX_Int getLower = sList->getLower;
    JX_Int globalRow = sList->row + sList->beg_rowP;
    
    getLower = list[getLower].next;
    if (list[getLower].col < globalRow)
    {
        node = &(list[getLower]);
        sList->getLower = getLower;
    }
    JX_END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhPermuteAndInsert"
jx_bool jx_SortedList_dhPermuteAndInsert( jx_SortedList_dh sList, jx_SRecord *sr, JX_Real thresh )
{
    JX_START_FUNC_DH
    jx_bool wasInserted = jx_false;
    JX_Int col = sr->col;
    JX_Real testVal = fabs(sr->val);
    JX_Int beg_row = sList->beg_row, end_row = beg_row + sList->m;
    JX_Int beg_rowP = sList->beg_rowP;
    
    if (col >= beg_row && col < end_row)
    {
        col -= beg_row;
        col = sList->o2n_local[col];
        if (testVal > thresh || col == sList->row)
        {
            col += beg_rowP;
        }
        else
        {
            col = -1;
        }
    }
    else
    {
        if (testVal < thresh) goto END_OF_FUNCTION;
        if (sList->o2n_external == NULL)
        {
            col = -1;
        }
        else
        {
            JX_Int tmp = jx_Hash_i_dhLookup(sList->o2n_external, col); JX_CHECK_ERROR(-1);
            
            if (tmp == -1)
            {
                col = -1;
            }
            else
            {
                col = tmp;
            }
        }
    }
    if (col != -1)
    {
        sr->col = col;
        jx_SortedList_dhInsert(sList, sr); JX_CHECK_ERROR(-1);
        wasInserted = jx_true;
    }
    
END_OF_FUNCTION: ;
    
    JX_END_FUNC_VAL(wasInserted)
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhInsertOrUpdate"
void jx_SortedList_dhInsertOrUpdate( jx_SortedList_dh sList, jx_SRecord *sr )
{
    JX_START_FUNC_DH
    jx_SRecord *node = jx_SortedList_dhFind(sList, sr); JX_CHECK_V_ERROR;
    
    if (node == NULL)
    {
        jx_SortedList_dhInsert(sList, sr); JX_CHECK_V_ERROR;
    }
    else
    {
        node->level = JX_MIN(sr->level, node->level);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhInsert"
void jx_SortedList_dhInsert( jx_SortedList_dh sList, jx_SRecord *sr )
{
    JX_START_FUNC_DH
    JX_Int prev, next;
    JX_Int ct, col = sr->col;
    jx_SRecord *list = sList->list;
    
    if (sList->countMax == sList->alloc)
    {
        jx_lengthen_list_private(sList); JX_CHECK_V_ERROR;
        list = sList->list;
    }
    ct = sList->countMax;
    sList->countMax += 1;
    sList->count += 1;
    list[ct].col = col;
    list[ct].level = sr->level;
    list[ct].val = sr->val;
    prev = 0;
    next = list[0].next;
    while (col > list[next].col)
    {
        prev = next;
        next = list[next].next;
    }
    list[prev].next = ct;
    list[ct].next = next;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhFind"
jx_SRecord *jx_SortedList_dhFind( jx_SortedList_dh sList, jx_SRecord *sr )
{
    JX_START_FUNC_DH
    JX_Int i, count = sList->countMax;
    JX_Int c = sr->col;
    jx_SRecord *s = sList->list;
    jx_SRecord *node = NULL;
    
    for (i = 1; i < count; ++ i)
    {
        if (s[i].col == c)
        {
            node = &(s[i]);
            break;
        }
    }
    JX_END_FUNC_VAL(node)
}

#undef __FUNC__
#define __FUNC__ "jx_lengthen_list_private"
void jx_lengthen_list_private( jx_SortedList_dh sList )
{
    JX_START_FUNC_DH
    jx_SRecord *tmp = sList->list;
    JX_Int size = sList->alloc = 2 * sList->alloc;
    
    JX_SET_INFO("lengthening list");
    sList->list = (jx_SRecord *)JX_MALLOC_DH(size*sizeof(jx_SRecord));
    memcpy(sList->list, tmp, sList->countMax*sizeof(jx_SRecord));
    JX_SET_INFO("doubling size of sList->list");
    JX_FREE_DH(tmp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

static jx_bool jx_check_constraint_private( jx_SubdomainGraph_dh sg, JX_Int thisSubdomain, JX_Int col );
void jx_delete_private( jx_SortedList_dh sList, JX_Int col );

#undef __FUNC__
#define __FUNC__ "jx_SortedList_dhEnforceConstraint"
void jx_SortedList_dhEnforceConstraint( jx_SortedList_dh sList, jx_SubdomainGraph_dh sg )
{
    JX_START_FUNC_DH
    JX_Int thisSubdomain = jx_myid_dh;
    JX_Int col, count;
    JX_Int beg_rowP = sList->beg_rowP;
    JX_Int end_rowP = beg_rowP + sList->m;
    jx_bool debug = jx_false;
    
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_SortedList")) debug = jx_true;
    if (debug)
    {
        jx_fprintf(jx_logFile, "SLIST ======= enforcing constraint for row= %i\n", 1+sList->row);
        jx_fprintf(jx_logFile, "\nSLIST ---- before checking: ");
        count = jx_SortedList_dhReadCount(sList); JX_CHECK_V_ERROR;
        while (count --)
        {
            jx_SRecord *sr = jx_SortedList_dhGetSmallest(sList); JX_CHECK_V_ERROR;
            jx_fprintf(jx_logFile, "%i ", sr->col+1);
        }
        jx_fprintf(jx_logFile, "\n");
        sList->get = 0;
    }
    count = jx_SortedList_dhReadCount(sList); JX_CHECK_V_ERROR;
    while (count --)
    {
        jx_SRecord *sr = jx_SortedList_dhGetSmallest(sList); JX_CHECK_V_ERROR;
        col = sr->col;
        if (debug)
        {
            jx_fprintf(jx_logFile, "SLIST  next col= %i\n", col+1);
        }
        if (col < beg_rowP || col >= end_rowP)
        {
            if (debug)
            {
                jx_fprintf(jx_logFile, "SLIST     external col: %i ; ", 1+col);
            }
            if (jx_check_constraint_private(sg, thisSubdomain, col))
            {
                jx_delete_private(sList, col); JX_CHECK_V_ERROR;
                sList->count -= 1;
                if (debug)
                {
                    jx_fprintf(jx_logFile, " deleted\n");
                }
            }
            else
            {
                if (debug)
                {
                    jx_fprintf(jx_logFile, " kept\n");
                }
            }
        }
    }
    sList->get = 0;
    if (debug)
    {
        jx_fprintf(jx_logFile, "SLIST---- after checking: ");
        count = jx_SortedList_dhReadCount(sList); JX_CHECK_V_ERROR;
        while (count --)
        {
            jx_SRecord *sr = jx_SortedList_dhGetSmallest(sList); JX_CHECK_V_ERROR;
            jx_fprintf(jx_logFile, "%i ", sr->col+1);
        }
        jx_fprintf(jx_logFile, "\n");
        fflush(jx_logFile);
        sList->get = 0;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_check_constraint_private"
jx_bool jx_check_constraint_private( jx_SubdomainGraph_dh sg, JX_Int p1, JX_Int j )
{
    JX_START_FUNC_DH
    jx_bool retval = jx_false;
    JX_Int i, p2;
    JX_Int *nabors, count;
    
    p2 = jx_SubdomainGraph_dhFindOwner(sg, j, jx_true);
    nabors = sg->adj + sg->ptrs[p1];
    count = sg->ptrs[p1+1]  - sg->ptrs[p1];
    for (i = 0; i < count; ++ i)
    {
        if (nabors[i] == p2)
        {
            retval = jx_true;
            break;
        }
    }
    JX_END_FUNC_VAL(! retval)
}

#undef __FUNC__
#define __FUNC__ "jx_delete_private"
void jx_delete_private( jx_SortedList_dh sList, JX_Int col )
{
    JX_START_FUNC_DH
    JX_Int curNode = 0;
    jx_SRecord *list = sList->list;
    JX_Int next;
    
    while (list[list[curNode].next].col != col)
    {
        curNode = list[curNode].next;
    }
    next = list[curNode].next;
    list[next].col = -1;
    next = list[next].next;
    list[curNode].next = next;
    JX_END_FUNC_DH
}
