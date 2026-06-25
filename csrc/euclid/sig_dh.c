//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  sig_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"
#include <signal.h>

extern void jx_sigRegister_dh();
extern void jx_sigHandler_dh( int sig );

#ifdef WIN32
int jx_euclid_signals_len = 2;
int jx_euclid_signals[] = {SIGSEGV, SIGFPE};
#else
int jx_euclid_signals_len = 3;
int jx_euclid_signals[] = {SIGSEGV, SIGFPE, SIGBUS};
#endif

static char *JX_SIGNAME[] = {
    "Unknown signal",
    "HUP (Hangup detected on controlling terminal or death of controlling process)",
    "INT: Interrupt from keyboard",
    "QUIT: Quit from keyboard",
    "ILL: Illegal Instruction",
    "TRAP",
    "ABRT: Abort signal",
    "EMT",
    "FPE (Floating Point Exception)",
    "KILL: Kill signal",
    "BUS (Bus Error, possibly illegal memory access)",
    "SEGV (Segmentation Violation (memory access out of range?))",
    "SYS",
    "PIPE: Broken pipe: write to pipe with no readers",
    "ALRM: Timer signal",
    "TERM: Termination signal",
    "URG",
    "STOP",
    "TSTP",
    "CONT",
    "CHLD"
};

#undef __FUNC__
#define __FUNC__ "jx_sigHandler_dh"
void jx_sigHandler_dh( int sig )
{
    jx_fprintf(stderr, "\n[%i] Euclid Signal Handler got: %s\n", jx_myid_dh, JX_SIGNAME[sig]);
    jx_fprintf(stderr, "[%i] ========================================================\n", jx_myid_dh);
    jx_fprintf(stderr, "[%i] function calling sequence that led to the exception:\n", jx_myid_dh);
    jx_fprintf(stderr, "[%i] ========================================================\n", jx_myid_dh);
    jx_printFunctionStack(stderr);
    jx_fprintf(stderr, "\n\n");
    if (jx_logFile != NULL)
    {
        jx_fprintf(jx_logFile, "\n[%i] Euclid Signal Handler got: %s\n", jx_myid_dh, JX_SIGNAME[sig]);
        jx_fprintf(jx_logFile, "[%i] ========================================================\n", jx_myid_dh);
        jx_fprintf(jx_logFile, "[%i] function calling sequence that led to the exception:\n", jx_myid_dh);
        jx_fprintf(jx_logFile, "[%i] ========================================================\n", jx_myid_dh);
        jx_printFunctionStack(jx_logFile);
        jx_fprintf(jx_logFile, "\n\n");
    }
    JX_EUCLID_EXIT;
}

#undef __FUNC__
#define __FUNC__ "jx_sigRegister_dh"
void jx_sigRegister_dh()
{
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-sig_dh"))
    {
        int i;
        
        for (i = 0; i < jx_euclid_signals_len; ++ i)
        {
            signal(jx_euclid_signals[i], jx_sigHandler_dh);
        }
    }
}
