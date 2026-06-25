//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  time_log_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#define JX_MAX_TIME_MARKS 100
#define JX_MAX_DESC_LENGTH 60

struct _jx_timeLog_dh
{
    JX_Int first;
    JX_Int last;
    JX_Real time[JX_MAX_TIME_MARKS];
    char desc[JX_MAX_TIME_MARKS][JX_MAX_DESC_LENGTH];
    jx_Timer_dh timer;
};

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhCreate"
void jx_TimeLog_dhCreate( jx_TimeLog_dh *t )
{
    JX_START_FUNC_DH
    JX_Int i;
    struct _jx_timeLog_dh *tmp = (struct _jx_timeLog_dh *)JX_MALLOC_DH(sizeof(struct _jx_timeLog_dh)); JX_CHECK_V_ERROR;
    
   *t = tmp;
    tmp->first = tmp->last = 0;
    jx_Timer_dhCreate(&tmp->timer);
    for (i = 0; i < JX_MAX_TIME_MARKS; ++ i) strcpy(tmp->desc[i], "X");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhDestroy"
void jx_TimeLog_dhDestroy( jx_TimeLog_dh t )
{
    JX_START_FUNC_DH
    jx_Timer_dhDestroy(t->timer);
    JX_FREE_DH(t);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhStart"
void jx_TimeLog_dhStart( jx_TimeLog_dh t )
{
    JX_START_FUNC_DH
    jx_Timer_dhStart(t->timer);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhStop"
void jx_TimeLog_dhStop( jx_TimeLog_dh t )
{
    JX_START_FUNC_DH
    jx_Timer_dhStop(t->timer);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhMark"
void jx_TimeLog_dhMark( jx_TimeLog_dh t, const char *desc )
{
    JX_START_FUNC_DH
    if (t->last < JX_MAX_TIME_MARKS - 3)
    {
        jx_Timer_dhStop(t->timer);
        t->time[t->last] = jx_Timer_dhReadWall(t->timer);
        jx_Timer_dhStart(t->timer);
        jx_sprintf(t->desc[t->last], "%s", desc);
        t->last += 1;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhReset"
void jx_TimeLog_dhReset( jx_TimeLog_dh t )
{
    JX_START_FUNC_DH
    if (t->last < JX_MAX_TIME_MARKS - 2)
    {
        JX_Real total = 0.0;
        JX_Int i, first = t->first, last = t->last;
        
        for (i = first; i < last; ++ i) total += t->time[i];
        t->time[last] = total;
        jx_sprintf(t->desc[last], "========== totals, and reset ==========\n");
        t->last += 1;
        t->first = t->last;
        jx_Timer_dhStart(t->timer);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_TimeLog_dhPrint"
void jx_TimeLog_dhPrint( jx_TimeLog_dh t, FILE *fp, jx_bool allPrint )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Real total = 0.0;
    JX_Real timeMax[JX_MAX_TIME_MARKS];
    JX_Real timeMin[JX_MAX_TIME_MARKS];
    static jx_bool wasSummed = jx_false;
    
    if (!wasSummed)
    {
        for (i = t->first; i < t->last; ++ i) total += t->time[i];
        t->time[t->last] = total;
        jx_sprintf(t->desc[t->last], "========== totals, and reset ==========\n");
        t->last += 1;
        jx_MPI_Allreduce(t->time, timeMax, t->last, JX_MPI_REAL, MPI_MAX, jx_comm_dh);
        jx_MPI_Allreduce(t->time, timeMin, t->last, JX_MPI_REAL, MPI_MIN, jx_comm_dh);
        wasSummed = jx_true;
    }
    if (fp != NULL)
    {
        if (jx_myid_dh == 0 || allPrint)
        {
            jx_fprintf(fp,"\n----------------------------------------- timing report\n");
            jx_fprintf(fp, "\n   self     max     min\n");
            for (i = 0; i < t->last; ++ i)
            {
                jx_fprintf(fp, "%7.3f %7.3f %7.3f   #%s\n", t->time[i], timeMax[i], timeMin[i], t->desc[i]);
            }
            fflush(fp);
        }
    }
    JX_END_FUNC_DH
}
