//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  timer_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhCreate"
void jx_Timer_dhCreate( jx_Timer_dh *t )
{
    JX_START_FUNC_DH
    struct _jx_timer_dh *tmp = (struct _jx_timer_dh *)JX_MALLOC_DH(sizeof(struct _jx_timer_dh)); JX_CHECK_V_ERROR;
    
   *t = tmp;
    tmp->isRunning = jx_false;
    tmp->begin_wall = 0.0;
    tmp->end_wall = 0.0;
#ifdef EUCLID_TIMING
    tmp->sc_clk_tck = sysconf(_SC_CLK_TCK);
#else
    tmp->sc_clk_tck = CLOCKS_PER_SEC;
#endif
#if defined(EUCLID_TIMING)
    jx_sprintf(jx_msgBuf_dh, "using EUCLID_TIMING; _SC_CLK_TCK = %i", (JX_Int)tmp->sc_clk_tck);
    JX_SET_INFO(jx_msgBuf_dh);
#elif defined(jx_MPI_TIMING)
    JX_SET_INFO("using jx_MPI timing")
#else
    JX_SET_INFO("using JUNK timing")
#endif
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhDestroy"
void jx_Timer_dhDestroy( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_FREE_DH(t);
    JX_END_FUNC_DH
}

#ifdef EUCLID_TIMING

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhStart"
void jx_Timer_dhStart( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    t->begin_wall = times(&(t->begin_cpu));
    t->isRunning = jx_true;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhStop"
void jx_Timer_dhStop( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    t->end_wall = times(&(t->end_cpu));
    t->isRunning = jx_false;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadWall"
JX_Real jx_Timer_dhReadWall( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_Real retval = 0.0;
    long JX_Int sc_clk_tck = t->sc_clk_tck;
    
    if (t->isRunning) t->end_wall = times(&(t->end_cpu));
    retval = (JX_Real)(t->end_wall - t->begin_wall) / (JX_Real)sc_clk_tck;
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadCPU"
JX_Real jx_Timer_dhReadCPU( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_Real retval;
    long JX_Int sc_clk_tck = t->sc_clk_tck;
    
    if (t->isRunning) t->end_wall = times(&(t->end_cpu));
    retval = (JX_Real)(t->end_cpu.tms_utime - t->begin_cpu.tms_utime
                    + t->end_cpu.tms_stime -  t->begin_cpu.tms_stime
                    + t->end_cpu.tms_cutime - t->begin_cpu.tms_cutime
                    + t->end_cpu.tms_cstime -  t->begin_cpu.tms_cstime) / (JX_Real)sc_clk_tck;
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadUsage"
JX_Real jx_Timer_dhReadUsage( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_Real cpu = jx_Timer_dhReadCPU(t);
    JX_Real wall = jx_Timer_dhReadWall(t);
    JX_Real retval = 100.0 * cpu / wall;
    JX_END_FUNC_VAL(retval);
}

#elif defined(jx_MPI_TIMING)

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhStart"
void jx_Timer_dhStart( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    t->begin_wall = jx_MPI_Wtime();
    t->isRunning = jx_true;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhStop"
void jx_Timer_dhStop( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    t->end_wall = jx_MPI_Wtime();
    t->isRunning = jx_false;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadWall"
JX_Real jx_Timer_dhReadWall( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_Real retval;
    
    if (t->isRunning) t->end_wall = jx_MPI_Wtime();
    retval = t->end_wall - t->begin_wall;
    JX_END_FUNC_VAL(retval)
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadCPU"
JX_Real jx_Timer_dhReadCPU( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_VAL(-1.0)
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadUsage"
JX_Real jx_Timer_dhReadUsage( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_VAL(-1.0);
}

#else

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhStart"
void jx_Timer_dhStart( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhStop"
void jx_Timer_dhStop( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadWall"
JX_Real jx_Timer_dhReadWall( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_VAL(-1.0)
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadCPU"
JX_Real jx_Timer_dhReadCPU( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_VAL(-1.0)
}

#undef __FUNC__
#define __FUNC__ "jx_Timer_dhReadUsage"
JX_Real jx_Timer_dhReadUsage( jx_Timer_dh t )
{
    JX_START_FUNC_DH
    JX_END_FUNC_VAL(-1.0);
}

#endif
