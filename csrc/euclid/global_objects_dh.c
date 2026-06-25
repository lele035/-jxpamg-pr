//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  global_objects_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

extern void jx_sigRegister_dh();

jx_bool jx_errFlag_dh = jx_false;
jx_Parser_dh jx_parser_dh = NULL;
jx_TimeLog_dh jx_tlog_dh = NULL;
jx_Mem_dh jx_mem_dh = NULL;
FILE *jx_logFile = NULL;
char jx_msgBuf_dh[JX_MSG_BUF_SIZE_DH];
JX_Int jx_np_dh = 1;
JX_Int jx_myid_dh = 0;
MPI_Comm jx_comm_dh = 0;
FILE *jx_logFile;
void jx_openLogfile_dh( JX_Int argc, char *argv[] );
void jx_closeLogfile_dh();
jx_bool jx_logInfoToStderr = jx_false;
jx_bool jx_logInfoToFile = jx_true;
jx_bool jx_logFuncsToStderr = jx_false;
jx_bool jx_logFuncsToFile = jx_false;
jx_bool jx_ignoreMe = jx_true;
JX_Int jx_ref_counter = 0;

#define JX_MAX_MSG_SIZE 1024
#define JX_MAX_STACK_SIZE 20

static char jx_errMsg_private[JX_MAX_STACK_SIZE][JX_MAX_MSG_SIZE];
static JX_Int jx_errCount_private = 0;
static char jx_calling_stack[JX_MAX_STACK_SIZE][JX_MAX_MSG_SIZE];
static JX_Int jx_jx_calling_stack_count = 0;

void jx_openLogfile_dh( JX_Int argc, char *argv[] )
{
    char buf[1024];
    
    if (jx_logFile != NULL) return;
    jx_sprintf(buf, "jx_logFile");
    if (argc && argv != NULL)
    {
        JX_Int j;
        
        for (j = 1; j < argc; ++ j)
        {
            if (strcmp(argv[j],"-jx_logFile") == 0)
            {
                if (j+1 < argc)
                {
                    jx_sprintf(buf, "%s", argv[j+1]);
                    break;
                }
            }
        }
    }
    if (strcmp(buf, "none"))
    {
        char a[5];
        
        jx_sprintf(a, ".%i", jx_myid_dh);
        strcat(buf, a);
        if ((jx_logFile = fopen(buf, "w")) == NULL )
        {
            jx_fprintf(stderr, "can't open >%s< for writing; continuing anyway\n", buf);
        }
    }
}

void jx_closeLogfile_dh()
{
    if (jx_logFile != NULL)
    {
        if (fclose(jx_logFile))
        {
            jx_fprintf(stderr, "Error closing jx_logFile\n");
        }
        jx_logFile = NULL;
    }
}

void jx_setInfo_dh( char *msg, char *function, char *file, JX_Int line )
{
    if (jx_logInfoToFile && jx_logFile != NULL)
    {
        jx_fprintf(jx_logFile, "INFO: %s;\n       function= %s  file=%s  line=%i\n", msg, function, file, line);
        fflush(jx_logFile);
    }
    if (jx_logInfoToStderr)
    {
        jx_fprintf(stderr, "INFO: %s;\n       function= %s  file=%s  line=%i\n", msg, function, file, line);
    }
}

void jx_dh_StartFunc( char *function, char *file, JX_Int line, JX_Int priority )
{
    if (priority == 1)
    {
        jx_sprintf(jx_calling_stack[jx_jx_calling_stack_count], "[%i]   %s  file= %s  line= %i", jx_myid_dh, function, file, line);
        ++ jx_jx_calling_stack_count;
        if (jx_jx_calling_stack_count == JX_MAX_STACK_SIZE)
        {
            jx_fprintf(stderr, "_____________ jx_dh_StartFunc: OVERFLOW _____________________\n");
            if (jx_logFile != NULL)
            {
                jx_fprintf(jx_logFile, "_____________ jx_dh_StartFunc: OVERFLOW _____________________\n");
            }
            -- jx_jx_calling_stack_count;
        }
    }
}

void jx_dh_EndFunc( char *function, JX_Int priority )
{
    if (priority == 1)
    {
        -- jx_jx_calling_stack_count;
        if (jx_jx_calling_stack_count < 0)
        {
            jx_jx_calling_stack_count = 0;
            jx_fprintf(stderr, "_____________ jx_dh_EndFunc: UNDERFLOW _____________________\n");
            if (jx_logFile != NULL)
            {
                jx_fprintf(jx_logFile, "_____________ jx_dh_EndFunc: UNDERFLOW _____________________\n");
            }
        }
    }
}

void jx_setError_dh( char *msg, char *function, char *file, JX_Int line )
{
    jx_errFlag_dh = jx_true;
    if (!strcmp(msg, ""))
    {
        jx_sprintf(jx_errMsg_private[jx_errCount_private],
            "[%i] called from: %s  file= %s  line= %i", jx_myid_dh, function, file, line);
    }
    else
    {
        jx_sprintf(jx_errMsg_private[jx_errCount_private],
            "[%i] ERROR: %s\n       %s  file= %s  line= %i\n", jx_myid_dh, msg, function, file, line);
    }
    ++ jx_errCount_private;
    if (jx_errCount_private == JX_MAX_STACK_SIZE) -- jx_errCount_private;
}

void jx_printErrorMsg( FILE *fp )
{
    if (!jx_errFlag_dh)
    {
        jx_fprintf(fp, "jx_errFlag_dh is not set; nothing to print!\n");
        fflush(fp);
    }
    else
    {
        JX_Int i;
        
        jx_fprintf(fp, "\n============= error stack trace ====================\n");
        for (i = 0; i < jx_errCount_private; ++ i)
        {
            jx_fprintf(fp, "%s\n", jx_errMsg_private[i]);
        }
        jx_fprintf(fp, "\n");
        fflush(fp);
    }
}

void jx_printFunctionStack( FILE *fp )
{
    JX_Int i;
    
    for (i = 0; i < jx_jx_calling_stack_count; ++ i)
    {
        jx_fprintf(fp, "%s\n", jx_calling_stack[i]);
    }
    jx_fprintf(fp, "\n");
    fflush(fp);
}

#define JX_MAX_ERROR_SPACES 200
static char jx_spaces[JX_MAX_ERROR_SPACES];
static JX_Int jx_nesting = 0;
static jx_bool jx_initSpaces = jx_true;
#define JX_INDENT_DH 3

void jx_Error_dhStartFunc( char *function, char *file, JX_Int line )
{
    if (jx_initSpaces)
    {
        memset(jx_spaces, ' ', JX_MAX_ERROR_SPACES*sizeof(char));
        jx_initSpaces = jx_false;
    }
    jx_spaces[JX_INDENT_DH*jx_nesting] = ' ';
    ++ jx_nesting;
    if (jx_nesting > JX_MAX_ERROR_SPACES-1) jx_nesting = JX_MAX_ERROR_SPACES - 1;
    jx_spaces[JX_INDENT_DH*jx_nesting] = '\0';
    if (jx_logFuncsToStderr)
    {
        jx_fprintf(stderr, "%s(%i) %s  [file= %s  line= %i]\n", jx_spaces, jx_nesting, function, file, line);
    }
    if (jx_logFuncsToFile && jx_logFile != NULL)
    {
        jx_fprintf(jx_logFile, "%s(%i) %s  [file= %s  line= %i]\n", jx_spaces, jx_nesting, function, file, line);
        fflush(jx_logFile);
    }
}

void jx_Error_dhEndFunc( char *function )
{
    jx_nesting -= 1;
    if (jx_nesting < 0) jx_nesting = 0;
    jx_spaces[JX_INDENT_DH*jx_nesting] = '\0';
}

static jx_bool jx_EuclidIsActive = jx_false;

#undef __FUNC__
#define __FUNC__ "jx_EuclidIsInitialized"
jx_bool jx_EuclidIsInitialized()
{
    return jx_EuclidIsActive;
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidInitialize"
void jx_EuclidInitialize( JX_Int argc, char *argv[], char *help )
{
    if (!jx_EuclidIsActive)
    {
        jx_MPI_Comm_size(jx_comm_dh, &jx_np_dh);
        jx_MPI_Comm_rank(jx_comm_dh, &jx_myid_dh);
        jx_openLogfile_dh(argc, argv);
        if (jx_mem_dh == NULL)
        {
            jx_Mem_dhCreate(&jx_mem_dh); JX_CHECK_V_ERROR;
        }
        if (jx_tlog_dh == NULL)
        {
            jx_TimeLog_dhCreate(&jx_tlog_dh); JX_CHECK_V_ERROR;
        }
        if (jx_parser_dh == NULL)
        {
            jx_Parser_dhCreate(&jx_parser_dh); JX_CHECK_V_ERROR;
        }
        jx_Parser_dhInit(jx_parser_dh, argc, argv); JX_CHECK_V_ERROR;
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-sig_dh"))
        {
            jx_sigRegister_dh(); JX_CHECK_V_ERROR;
        }
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-help"))
        {
            if (jx_myid_dh == 0) jx_printf("%s\n\n", help);
            JX_EUCLID_EXIT;
        }
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-jx_logFuncsToFile"))
        {
            jx_logFuncsToFile = jx_true;
        }
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-jx_logFuncsToStderr"))
        {
            jx_logFuncsToStderr = jx_true;
        }
        jx_EuclidIsActive = jx_true;
    }
}

#undef __FUNC__
#define __FUNC__ "jx_EuclidFinalize"
void jx_EuclidFinalize()
{
    if (jx_ref_counter) return;
    if (jx_EuclidIsActive)
    {
        if (jx_parser_dh != NULL)
        {
            jx_Parser_dhDestroy(jx_parser_dh); JX_CHECK_V_ERROR;
        }
        if (jx_tlog_dh != NULL)
        {
            jx_TimeLog_dhDestroy(jx_tlog_dh); JX_CHECK_V_ERROR;
        }
        if (jx_logFile != NULL)
        {
            jx_Mem_dhPrint(jx_mem_dh, jx_logFile, jx_true); JX_CHECK_V_ERROR;
        }
        if (jx_mem_dh != NULL)
        {
            jx_Mem_dhDestroy(jx_mem_dh); JX_CHECK_V_ERROR;
        }
        if (jx_logFile != NULL)
        {
            jx_closeLogfile_dh(); JX_CHECK_V_ERROR;
        }
        jx_EuclidIsActive = jx_false;
    }
}

#undef __FUNC__
#define __FUNC__ "jx_printf_dh"
void jx_printf_dh( char *fmt, ... )
{
    JX_START_FUNC_DH
    va_list args;
    char *buf = jx_msgBuf_dh;
    
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    if (jx_myid_dh == 0)
    {
        jx_fprintf(stdout, "%s", buf);
    }
    va_end(args);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_fprintf_dh"
void jx_fprintf_dh( FILE *fp, char *fmt, ... )
{
    JX_START_FUNC_DH
    va_list args;
    char *buf = jx_msgBuf_dh;
    
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    if (jx_myid_dh == 0)
    {
        jx_fprintf(fp, "%s", buf);
    }
    va_end(args);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_echoInvocation_dh"
void jx_echoInvocation_dh( MPI_Comm comm, char *prefix, JX_Int argc, char *argv[] )
{
    JX_START_FUNC_DH
    JX_Int i, id;
    
    jx_MPI_Comm_rank(comm, &id);
    if (prefix != NULL)
    {
        jx_printf_dh("\n%s ", prefix);
    }
    else
    {
        jx_printf_dh("\n");
    }
    jx_printf_dh("program invocation: ");
    for (i = 0; i < argc; ++ i)
    {
        jx_printf_dh("%s ", argv[i]);
    }
    jx_printf_dh("\n");
    JX_END_FUNC_DH
}
