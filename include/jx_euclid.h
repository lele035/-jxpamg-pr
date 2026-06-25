//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_euclid.h -- head files for Parallel ILU Solver
 *  Date: 2013/01/20
 *
 *  Created by Yue Xiaoqiang
 */

#ifndef JX_EUCLID_HEADER
#define JX_EUCLID_HEADER

#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif

#define JXPAMG_GET_ROW

#ifndef JX_EUCLID_CONF_DH
#define JX_EUCLID_CONF_DH

#define JX_MAX_MPI_TASKS 50000
#define JX_TRIPLES_FORMAT "%i %i %1.8e\n"

#define JX_EUCLID_EXIT MPI_Abort(jx_comm_dh, -1)

#define JX_ERRCHKA \
      if (jx_errFlag_dh) { \
        jx_setError_dh("", __FUNC__, __FILE__, __LINE__); \
        if (jx_logFile != NULL) { \
          jx_printErrorMsg(jx_logFile); \
          jx_closeLogfile_dh(); \
        } \
        jx_printErrorMsg(stderr); \
        if (jx_myid_dh == 0) { \
          jx_Mem_dhPrint(jx_mem_dh, stderr, jx_false); \
        } \
        JX_EUCLID_EXIT; \
      }

#define JX_PIVOT_FIX_DEFAULT 1e-3

#define JX_MALLOC_DH(s) jx_Mem_dhMalloc(jx_mem_dh, (s))
#define JX_FREE_DH(p) jx_Mem_dhFree(jx_mem_dh, p)

#define JX_PRIVATE_MALLOC malloc
#define JX_PRIVATE_FREE free

#endif

#ifndef JX_MACROS_DH
#define JX_MACROS_DH

#ifndef JX_MAX
#define JX_MAX(a,b) ((a)>(b)?(a):(b))
#endif

#ifndef JX_MIN
#define JX_MIN(a,b) ((a)<(b)?(a):(b))
#endif

#define _JX_MATLAB_ZERO_  1e-100

#define JX_CHECK_MPI_V_ERROR(errCode) \
      { \
        if (errCode) { \
          jx_setError_dh("MPI error!", __FUNC__, __FILE__, __LINE__); \
          return; \
        } \
      }

#define JX_CHECK_MPI_ERROR(errCode) \
      { \
        if (errCode) { \
          jx_setError_dh("MPI error!", __FUNC__, __FILE__, __LINE__); \
          return(errCode); \
        } \
      }

#define JX_SET_V_ERROR(msg) \
      { jx_setError_dh(msg, __FUNC__, __FILE__, __LINE__); \
        return; \
      }

#define JX_SET_ERROR(retval, msg) \
      { jx_setError_dh(msg, __FUNC__, __FILE__, __LINE__); \
        return (retval); \
      }

#define JX_CHECK_V_ERROR \
      if (jx_errFlag_dh) { \
        jx_setError_dh("",  __FUNC__, __FILE__, __LINE__); \
        return; \
      }

#define JX_CHECK_ERROR(retval) \
      if (jx_errFlag_dh) { \
        jx_setError_dh("",  __FUNC__, __FILE__, __LINE__); \
        return (retval); \
      }

#define JX_SET_INFO(msg) jx_setInfo_dh(msg, __FUNC__, __FILE__, __LINE__);

#define JX_START_FUNC_DH \
      jx_dh_StartFunc(__FUNC__, __FILE__, __LINE__, 1); \
      {

#define JX_END_FUNC_DH \
      } \
      jx_dh_EndFunc(__FUNC__, 1);

#define JX_END_FUNC_VAL(a) \
      jx_dh_EndFunc(__FUNC__, 1); \
      return a ; \
      }

#define JX_START_FUNC_DH_2
#define JX_END_FUNC_DH_2
#define JX_END_FUNC_VAL_2(a) return a;

#endif

#ifndef JX_COMMON_DH
#define JX_COMMON_DH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>

#define JX_REAL_DH JX_Real

#include "jx_mv.h"

typedef struct _jx_matgenfd * jx_MatGenFD;
typedef struct _jx_subdomain_dh * jx_SubdomainGraph_dh;
typedef struct _jx_timer_dh * jx_Timer_dh;
typedef struct _jx_parser_dh * jx_Parser_dh;
typedef struct _jx_timeLog_dh * jx_TimeLog_dh;
typedef struct _jx_mem_dh * jx_Mem_dh;
typedef struct _jx_mat_dh * jx_Mat_dh;
typedef struct _jx_factor_dh * jx_Factor_dh;
typedef struct _jx_vec_dh * jx_Vec_dh;
typedef struct _jx_numbering_dh * jx_Numbering_dh;
typedef struct _jx_hash_dh * jx_Hash_dh;
typedef struct _jx_hash_i_dh * jx_Hash_i_dh;
typedef struct _jx_mpi_interface_dh * jx_Euclid_dh;
typedef struct _jx_sortedList_dh * jx_SortedList_dh;
typedef struct _jx_extrows_dh * jx_ExternalRows_dh;
typedef struct _jx_stack_dh * jx_Stack_dh;
typedef struct _jx_queue_dh * jx_Queue_dh;
typedef struct _jx_sortedset_dh * jx_SortedSet_dh;

typedef JX_Int jx_bool;
#define jx_true 1
#define jx_false 0

extern jx_Parser_dh jx_parser_dh;
extern jx_TimeLog_dh jx_tlog_dh;
extern jx_Mem_dh jx_mem_dh;
extern FILE *jx_logFile;
extern JX_Int jx_np_dh;
extern JX_Int jx_myid_dh;
extern MPI_Comm jx_comm_dh;
extern jx_bool jx_ignoreMe;
extern JX_Int jx_ref_counter;
extern jx_bool jx_errFlag_dh;
extern void jx_setInfo_dh( char *msg, char *function, char *file, JX_Int line );
extern void jx_setError_dh( char *msg, char *function, char *file, JX_Int line );
extern void jx_printErrorMsg( FILE *fp );

#ifndef JX_MPI_MAX_ERROR_STRING
#define JX_MPI_MAX_ERROR_STRING 256
#endif

#define JX_MSG_BUF_SIZE_DH JX_MAX(1024, JX_MPI_MAX_ERROR_STRING)
extern char jx_msgBuf_dh[JX_MSG_BUF_SIZE_DH];

extern void jx_openLogfile_dh( JX_Int argc, char *argv[] );
extern void jx_closeLogfile_dh();

extern jx_bool jx_logInfoToStderr;
extern jx_bool jx_logInfoToFile;
extern jx_bool jx_logFuncsToStderr;
extern jx_bool jx_logFuncsToFile;

extern void jx_Error_dhStartFunc( char *function, char *file, JX_Int line );
extern void jx_Error_dhEndFunc( char *function );
extern void jx_dh_StartFunc( char *function, char *file, JX_Int line, JX_Int priority );
extern void jx_dh_EndFunc( char *function, JX_Int priority );
extern void jx_printFunctionStack( FILE *fp );
extern void jx_EuclidInitialize( JX_Int argc, char *argv[], char *help );
extern void jx_EuclidFinalize();
extern jx_bool jx_EuclidIsInitialized();
extern void jx_printf_dh( char *fmt, ... );
extern void jx_fprintf_dh( FILE *fp, char *fmt, ... );
extern void jx_echoInvocation_dh( MPI_Comm comm, char *prefix, JX_Int argc, char *argv[] );

#endif

#ifndef JX_EXTERNAL_ROWS_DH_H
#define JX_EXTERNAL_ROWS_DH_H

extern void jx_ExternalRows_dhCreate( jx_ExternalRows_dh *er );
extern void jx_ExternalRows_dhDestroy( jx_ExternalRows_dh er );
extern void jx_ExternalRows_dhInit( jx_ExternalRows_dh er, jx_Euclid_dh ctx );
extern void jx_ExternalRows_dhRecvRows( jx_ExternalRows_dh extRows );
extern void jx_ExternalRows_dhSendRows( jx_ExternalRows_dh extRows );
extern void jx_ExternalRows_dhGetRow( jx_ExternalRows_dh er,
               JX_Int globalRow, JX_Int *len, JX_Int **cval, JX_Int **fill, JX_REAL_DH **aval );

struct _jx_extrows_dh
{
    jx_SubdomainGraph_dh sg;
    jx_Factor_dh F;
    
    MPI_Status status[JX_MAX_MPI_TASKS];
    MPI_Request req1[JX_MAX_MPI_TASKS]; 
    MPI_Request req2[JX_MAX_MPI_TASKS];
    MPI_Request req3[JX_MAX_MPI_TASKS]; 
    MPI_Request req4[JX_MAX_MPI_TASKS];
    MPI_Request cval_req[JX_MAX_MPI_TASKS];
    MPI_Request fill_req[JX_MAX_MPI_TASKS];
    MPI_Request aval_req[JX_MAX_MPI_TASKS];
    
    JX_Int rcv_row_counts[JX_MAX_MPI_TASKS];
    JX_Int rcv_nz_counts[JX_MAX_MPI_TASKS];
    JX_Int *rcv_row_lengths[JX_MAX_MPI_TASKS];
    JX_Int *rcv_row_numbers[JX_MAX_MPI_TASKS];
    
    JX_Int *cvalExt;
    JX_Int *fillExt;
    JX_REAL_DH *avalExt;
    
    jx_Hash_dh rowLookup;
    
    JX_Int *my_row_counts;
    JX_Int *my_row_numbers;
    
    JX_Int nzSend;
    JX_Int *cvalSend;
    JX_Int *fillSend;
    JX_REAL_DH *avalSend;
    
    jx_bool debug;
};

#endif

#ifndef JX_FACTOR_DH
#define JX_FACTOR_DH

struct _jx_factor_dh
{
    JX_Int m, n;
    
    JX_Int id;
    JX_Int beg_row;
    JX_Int first_bdry;
    JX_Int bdry_count;
    
    jx_bool blockJacobi;
    
    JX_Int *rp;
    JX_Int *cval;
    JX_REAL_DH *aval;
    JX_Int *fill;
    JX_Int *diag;
    JX_Int alloc;
    
    JX_Int num_recvLo, num_recvHi;
    JX_Int num_sendLo, num_sendHi;
    JX_Real *work_y_lo;
    JX_Real *work_x_hi;
    JX_Real *sendbufLo, *sendbufHi;
    JX_Int *sendindLo, *sendindHi;
    JX_Int sendlenLo, sendlenHi;
    jx_bool solveIsSetup;
    jx_Numbering_dh numbSolve;
    
    MPI_Request recv_reqLo[JX_MAX_MPI_TASKS], recv_reqHi[JX_MAX_MPI_TASKS];
    MPI_Request send_reqLo[JX_MAX_MPI_TASKS], send_reqHi[JX_MAX_MPI_TASKS];
    MPI_Request requests[JX_MAX_MPI_TASKS];
    MPI_Status status[JX_MAX_MPI_TASKS];
    
    jx_bool debug;
};

extern void jx_Factor_dhCreate( jx_Factor_dh *mat );
extern void jx_Factor_dhDestroy( jx_Factor_dh mat );
extern void jx_Factor_dhTranspose( jx_Factor_dh matIN, jx_Factor_dh *matOUT );
extern void jx_Factor_dhInit( void *A, jx_bool fillFlag, jx_bool avalFlag,
                              JX_Real rho, JX_Int id, JX_Int beg_rowP, jx_Factor_dh *F );
extern void jx_Factor_dhReallocate( jx_Factor_dh F, JX_Int used, JX_Int additional );
extern void jx_Factor_dhSolveSetup( jx_Factor_dh mat, jx_SubdomainGraph_dh sg );
extern void jx_Factor_dhSolve( JX_Real *rhs, JX_Real *lhs, jx_Euclid_dh ctx );
extern void jx_Factor_dhSolveSeq( JX_Real *rhs, JX_Real *lhs, jx_Euclid_dh ctx );
extern JX_Real jx_Factor_dhCondEst( jx_Factor_dh mat, jx_Euclid_dh ctx );
extern JX_Real jx_Factor_dhMaxValue( jx_Factor_dh mat );
extern JX_Real jx_Factor_dhMaxPivotInverse( jx_Factor_dh mat );
extern JX_Int jx_Factor_dhReadNz( jx_Factor_dh mat );
extern void jx_Factor_dhPrintTriples( jx_Factor_dh mat, char *filename );
extern void jx_Factor_dhPrintGraph( jx_Factor_dh mat, char *filename );
extern void jx_Factor_dhPrintDiags( jx_Factor_dh mat, FILE *fp );
extern void jx_Factor_dhPrintRows( jx_Factor_dh mat, FILE *fp );

#endif

#ifndef JX_VEC_DH_H
#define JX_VEC_DH_H

struct _jx_vec_dh
{
    JX_Int n;
    JX_Real *vals;
};

extern void jx_Vec_dhCreate( jx_Vec_dh *v );
extern void jx_Vec_dhDestroy( jx_Vec_dh v );
extern void jx_Vec_dhInit( jx_Vec_dh v, JX_Int size );
extern void jx_Vec_dhDuplicate( jx_Vec_dh v, jx_Vec_dh *out );
extern void jx_Vec_dhCopy( jx_Vec_dh x, jx_Vec_dh y );
extern void jx_Vec_dhSet( jx_Vec_dh v, JX_Real value );
extern void jx_Vec_dhSetRand( jx_Vec_dh v );
extern void jx_Vec_dhRead( jx_Vec_dh *v, JX_Int ignore, char *filename );
extern void jx_Vec_dhReadBIN( jx_Vec_dh *v, char *filename );
extern void jx_Vec_dhPrint( jx_Vec_dh v, jx_SubdomainGraph_dh sg, char *filename );
extern void jx_Vec_dhPrintBIN( jx_Vec_dh v, jx_SubdomainGraph_dh sg, char *filename );

#endif

#ifndef JX_MATGENFD_DH_DH
#define JX_MATGENFD_DH_DH

struct _jx_matgenfd
{
    jx_bool allocateMem;
    JX_Int px, py, pz;
    jx_bool threeD;
    JX_Int m;
    JX_Int cc;
    JX_Real hh;
    JX_Int id;
    JX_Int np;
    JX_Real stencil[8];
    
    JX_Real a, b, c, d, e, f, g, h;
    
    JX_Int first;
    jx_bool debug;
    
    JX_Real bcX1, bcX2;
    JX_Real bcY1, bcY2;
    JX_Real bcZ1, bcZ2;
    
    JX_Real (*A)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*B)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*C)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*D)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*E)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*F)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*G)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
    JX_Real (*H)( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
};

extern void jx_MatGenFD_Create( jx_MatGenFD *mg );
extern void jx_MatGenFD_Destroy( jx_MatGenFD mg );
extern void jx_MatGenFD_Run( jx_MatGenFD mg, JX_Int id, JX_Int np, jx_Mat_dh *A, jx_Vec_dh *rhs );
extern JX_Real jx_konstant( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
extern JX_Real jx_e2_xy( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );

#define JX_BOX1_X1 0.1
#define JX_BOX1_X2 0.4
#define JX_BOX1_Y1 0.1
#define JX_BOX1_Y2 0.4
#define JX_BOX2_X1 0.6
#define JX_BOX2_X2 0.9
#define JX_BOX2_Y1 0.1
#define JX_BOX2_Y2 0.4
#define JX_BOX3_X1 0.2
#define JX_BOX3_X2 0.8
#define JX_BOX3_Y1 0.6
#define JX_BOX3_Y2 0.8
#define JX_BOX1_DD 10
#define JX_BOX2_DD 100
#define JX_BOX3_DD 50

extern JX_Real jx_box_1( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );
extern JX_Real jx_box_2( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );

#endif

#ifndef JX_MAT_DH_DH
#define JX_MAT_DH_DH

#define JX_MAT_DH_BINS 10
#define JX_MATVEC_TIME 0
#define JX_MATVEC_MPI_TIME 1
#define JX_MATVEC_MPI_TIME2 5
#define JX_MATVEC_TOTAL_TIME 2
#define JX_MATVEC_RATIO 3
#define JX_MATVEC_WORDS 4

struct _jx_mat_dh
{
    JX_Int m, n;
    JX_Int beg_row;
    JX_Int bs;
    
    JX_Int *rp;
    JX_Int *len;
    JX_Int *cval;
    JX_Int *fill;
    JX_Int *diag;
    JX_Real *aval;
    jx_bool owner;
    
    JX_Int len_private;
    JX_Int rowCheckedOut;
    JX_Int *cval_private;
    JX_Real *aval_private;
    
    JX_Int *row_perm;
    
    JX_Real time[JX_MAT_DH_BINS];
    JX_Real time_max[JX_MAT_DH_BINS];
    JX_Real time_min[JX_MAT_DH_BINS];
    jx_bool matvec_timing;
    
    JX_Int num_recv;
    JX_Int num_send;
    MPI_Request *recv_req;
    MPI_Request *send_req;
    JX_Real *recvbuf, *sendbuf;
    JX_Int *sendind;
    JX_Int sendlen;
    JX_Int recvlen;
    jx_bool matvecIsSetup;
    jx_Numbering_dh numb;
    MPI_Status *status;
    
    jx_bool debug;
};

extern void jx_Mat_dhCreate( jx_Mat_dh *mat );
extern void jx_Mat_dhDestroy( jx_Mat_dh mat );
extern void jx_Mat_dhTranspose( jx_Mat_dh matIN, jx_Mat_dh *matOUT );
extern void jx_Mat_dhMakeStructurallySymmetric( jx_Mat_dh A );
extern void jx_Mat_dhMatVecSetup( jx_Mat_dh mat );
extern void jx_Mat_dhMatVecSetdown( jx_Mat_dh mat );
extern void jx_Mat_dhMatVec( jx_Mat_dh mat, JX_Real *lhs, JX_Real *rhs );
extern void jx_Mat_dhMatVec_omp( jx_Mat_dh mat, JX_Real *lhs, JX_Real *rhs );
extern void jx_Mat_dhMatVec_uni( jx_Mat_dh mat, JX_Real *lhs, JX_Real *rhs );
extern void jx_Mat_dhMatVec_uni_omp( jx_Mat_dh mat, JX_Real *lhs, JX_Real *rhs );
extern JX_Int jx_Mat_dhReadNz( jx_Mat_dh mat );
extern void jx_Mat_dhPrintGraph( jx_Mat_dh mat, jx_SubdomainGraph_dh sg, FILE *fp );
extern void jx_Mat_dhPrintRows( jx_Mat_dh mat, jx_SubdomainGraph_dh sg, FILE *fp );
extern void jx_Mat_dhPrintCSR( jx_Mat_dh mat, jx_SubdomainGraph_dh sg, char *filename );
extern void jx_Mat_dhPrintTriples( jx_Mat_dh mat, jx_SubdomainGraph_dh sg, char *filename );
extern void jx_Mat_dhPrintBIN( jx_Mat_dh mat, jx_SubdomainGraph_dh sg, char *filename );
extern void jx_Mat_dhReadCSR( jx_Mat_dh *mat, char *filename );
extern void jx_Mat_dhReadTriples( jx_Mat_dh *mat, JX_Int ignore, char *filename );
extern void jx_Mat_dhReadBIN( jx_Mat_dh *mat, char *filename );
extern void jx_Mat_dhPermute( jx_Mat_dh Ain, JX_Int *pIN, jx_Mat_dh *Bout );
extern void jx_Mat_dhFixDiags( jx_Mat_dh A );
extern void jx_Mat_dhPrintDiags( jx_Mat_dh A, FILE *fp );
extern void jx_Mat_dhGetRow( jx_Mat_dh B, JX_Int globalRow, JX_Int *len, JX_Int **ind, JX_Real **val );
extern void jx_Mat_dhRestoreRow( jx_Mat_dh B, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val );
extern void jx_Mat_dhPartition( jx_Mat_dh mat, JX_Int k, JX_Int **beg_rowOUT,
                                JX_Int **row_countOUT, JX_Int **n2oOUT, JX_Int **o2nOUT );
extern void jx_Mat_dhZeroTiming( jx_Mat_dh mat );
extern void jx_Mat_dhReduceTiming( jx_Mat_dh mat );
extern void jx_Mat_dhRowPermute( jx_Mat_dh );
extern void jx_dldperm( JX_Int job, JX_Int n, JX_Int nnz, JX_Int colptr[],
              JX_Int adjncy[], JX_Real nzval[], JX_Int *perm, JX_Real u[], JX_Real v[] );

#endif

#ifndef JX_SUBDOMAIN_GRAPH_DH
#define JX_SUBDOMAIN_GRAPH_DH

#define JX_JX_TIMING_BINS_SG 10

enum{ JX_TOTAL_SGT, JX_FIND_NABORS_SGT, JX_ORDER_BDRY_SGT, JX_FORM_GRAPH_SGT, JX_EXCHANGE_PERMS_SGT };

struct _jx_subdomain_dh
{
    JX_Int blocks;
    JX_Int *ptrs, *adj;
    JX_Int *o2n_sub;
    JX_Int *n2o_sub;
    JX_Int colors;
    jx_bool doNotColor;
    JX_Int *colorVec;
    
    JX_Int *beg_row;
    JX_Int *beg_rowP;
    JX_Int *row_count;
    JX_Int *bdry_count;
    
    JX_Int *loNabors, loCount;
    JX_Int *hiNabors, hiCount;
    JX_Int *allNabors, allCount;
    
    JX_Int m;
    JX_Int *n2o_row;
    JX_Int *o2n_col;
    
    jx_Hash_i_dh o2n_ext;
    jx_Hash_i_dh n2o_ext;
    
    JX_Real timing[JX_JX_TIMING_BINS_SG];
    jx_bool debug;
};

extern void jx_SubdomainGraph_dhCreate( jx_SubdomainGraph_dh *s );
extern void jx_SubdomainGraph_dhDestroy( jx_SubdomainGraph_dh s );
extern void jx_SubdomainGraph_dhInit( jx_SubdomainGraph_dh s, JX_Int blocks, jx_bool bj, void *A );
extern void jx_SubdomainGraph_dhColor( jx_SubdomainGraph_dh s );
extern JX_Int jx_SubdomainGraph_dhFindOwner( jx_SubdomainGraph_dh s, JX_Int idx, jx_bool permuted );
extern void jx_SubdomainGraph_dhExchangePerms( jx_SubdomainGraph_dh s );
extern void jx_SubdomainGraph_dhPrintSubdomainGraph( jx_SubdomainGraph_dh s, FILE *fp );
extern void jx_SubdomainGraph_dhPrintStatsLong( jx_SubdomainGraph_dh s, FILE *fp );
extern void jx_SubdomainGraph_dhDump( jx_SubdomainGraph_dh s, char *filename );
extern void jx_SubdomainGraph_dhPrintRatios( jx_SubdomainGraph_dh s, FILE *fp );
extern void jx_SubdomainGraph_dhPrintStats( jx_SubdomainGraph_dh sg, FILE *fp );

#endif

#ifndef JX_TIMELOG_DH_DH
#define JX_TIMELOG_DH_DH

extern void jx_TimeLog_dhCreate( jx_TimeLog_dh *t );
extern void jx_TimeLog_dhDestroy( jx_TimeLog_dh t );
extern void jx_TimeLog_dhStart( jx_TimeLog_dh t );
extern void jx_TimeLog_dhStop( jx_TimeLog_dh t );
extern void jx_TimeLog_dhReset( jx_TimeLog_dh t );
extern void jx_TimeLog_dhMark( jx_TimeLog_dh t, const char *description );
extern void jx_TimeLog_dhPrint( jx_TimeLog_dh t, FILE *fp, jx_bool allPrint );

#endif

#ifndef JX_SORTED_SET_DH
#define JX_SORTED_SET_DH

struct _jx_sortedset_dh
{
    JX_Int n;
    JX_Int *list;
    JX_Int count;
};

extern void jx_SortedSet_dhCreate( jx_SortedSet_dh *ss, JX_Int initialSize );
extern void jx_SortedSet_dhDestroy( jx_SortedSet_dh ss );
extern void jx_SortedSet_dhInsert( jx_SortedSet_dh ss, JX_Int idx );
extern void jx_SortedSet_dhGetList( jx_SortedSet_dh ss, JX_Int **list, JX_Int *count );

#endif

#ifndef JX_MEM_DH_DH
#define JX_MEM_DH_DH

extern void jx_Mem_dhCreate( jx_Mem_dh *m );
extern void jx_Mem_dhDestroy( jx_Mem_dh m );
extern void *jx_Mem_dhMalloc( jx_Mem_dh m, size_t size );
extern void jx_Mem_dhFree( jx_Mem_dh m, void *ptr );
extern void jx_Mem_dhPrint( jx_Mem_dh m, FILE* fp, jx_bool allPrint );

#endif

#ifndef JX_SUPPORT_DH
#define JX_SUPPORT_DH

extern void jx_shellSort_int( const JX_Int n, JX_Int *x );
extern void jx_shellSort_float( JX_Int n, JX_Real *v );

#endif

#ifndef JX_NUMBERING_DH_H
#define JX_NUMBERING_DH_H

struct _jx_numbering_dh
{
    JX_Int size;
    JX_Int first;
    JX_Int m;
    JX_Int *idx_ext;
    JX_Int *idx_extLo;
    JX_Int *idx_extHi;
    JX_Int num_ext;
    JX_Int num_extLo;
    JX_Int num_extHi;
    jx_Hash_i_dh global_to_local;
    
    jx_bool debug;
};

extern void jx_Numbering_dhCreate( jx_Numbering_dh *numb );
extern void jx_Numbering_dhDestroy( jx_Numbering_dh numb );
extern void jx_Numbering_dhSetup( jx_Numbering_dh numb, jx_Mat_dh mat );
extern void jx_Numbering_dhGlobalToLocal( jx_Numbering_dh numb, JX_Int len, JX_Int *global_in, JX_Int *local_out );

#endif

#ifndef JX_HASH_I_DH
#define JX_HASH_I_DH

extern void jx_Hash_i_dhCreate( jx_Hash_i_dh *h, JX_Int size );
extern void jx_Hash_i_dhDestroy( jx_Hash_i_dh h );
extern void jx_Hash_i_dhReset( jx_Hash_i_dh h );
extern void jx_Hash_i_dhInsert( jx_Hash_i_dh h, JX_Int key, JX_Int data );
extern JX_Int jx_Hash_i_dhLookup( jx_Hash_i_dh h, JX_Int key );

#endif

#ifndef JX_TIMER_DH_H
#define JX_TIMER_DH_H

#include <time.h>
#include <unistd.h>

struct _jx_timer_dh
{
    jx_bool isRunning;
    jx_longint sc_clk_tck;
    JX_Real begin_wall;
    JX_Real end_wall;
};

extern void jx_Timer_dhCreate( jx_Timer_dh *t );
extern void jx_Timer_dhDestroy( jx_Timer_dh t );
extern void jx_Timer_dhStart( jx_Timer_dh t );
extern void jx_Timer_dhStop( jx_Timer_dh t );
extern JX_Real jx_Timer_dhReadCPU( jx_Timer_dh t );
extern JX_Real jx_Timer_dhReadWall( jx_Timer_dh t );
extern JX_Real jx_Timer_dhReadUsage( jx_Timer_dh t );

#endif

#ifndef JX_PARSER_DH_DH
#define JX_PARSER_DH_DH

extern void jx_Parser_dhCreate( jx_Parser_dh *p );
extern void jx_Parser_dhDestroy( jx_Parser_dh p );
extern jx_bool jx_Parser_dhHasSwitch( jx_Parser_dh p, char *in );
extern jx_bool jx_Parser_dhReadString( jx_Parser_dh p, char *in, char **out );
extern jx_bool jx_Parser_dhReadInt( jx_Parser_dh p, char *in, JX_Int *out );
extern jx_bool jx_Parser_dhReadDouble( jx_Parser_dh p, char *in, JX_Real *out );
extern void jx_Parser_dhPrint( jx_Parser_dh p, FILE *fp, jx_bool allPrint );
extern void jx_Parser_dhInsert( jx_Parser_dh p, char *name, char *value );
extern void jx_Parser_dhUpdateFromFile( jx_Parser_dh p, char *name );
extern void jx_Parser_dhInit( jx_Parser_dh p, JX_Int argc, char *argv[] );

#endif

#ifndef JX_SORTEDLIST_DH_H
#define JX_SORTEDLIST_DH_H

typedef struct _jx_srecord
{
    JX_Int col;
    JX_Int level;
    JX_Real val;
    JX_Int next;
} jx_SRecord;

extern void jx_SortedList_dhCreate( jx_SortedList_dh *sList );
extern void jx_SortedList_dhDestroy( jx_SortedList_dh sList );
extern void jx_SortedList_dhInit( jx_SortedList_dh sList, jx_SubdomainGraph_dh sg );
extern void jx_SortedList_dhEnforceConstraint( jx_SortedList_dh sList, jx_SubdomainGraph_dh sg );
extern void jx_SortedList_dhReset( jx_SortedList_dh sList, JX_Int row );
extern JX_Int jx_SortedList_dhReadCount( jx_SortedList_dh sList );
extern void jx_SortedList_dhResetGetSmallest( jx_SortedList_dh sList );
extern jx_SRecord *jx_SortedList_dhGetSmallest( jx_SortedList_dh sList );
extern jx_SRecord *jx_SortedList_dhGetSmallestLowerTri( jx_SortedList_dh sList );
extern void jx_SortedList_dhInsert( jx_SortedList_dh sList, jx_SRecord *sr );
extern void jx_SortedList_dhInsertOrUpdateVal( jx_SortedList_dh sList, jx_SRecord *sr );
extern jx_bool jx_SortedList_dhPermuteAndInsert( jx_SortedList_dh sList, jx_SRecord *sr, JX_Real thresh );
extern void jx_SortedList_dhInsertOrUpdate( jx_SortedList_dh sList, jx_SRecord *sr );
extern jx_SRecord *jx_SortedList_dhFind( jx_SortedList_dh sList, jx_SRecord *sr );
extern void jx_SortedList_dhUpdateVal( jx_SortedList_dh sList, jx_SRecord *sr );

#endif

#ifndef JX_HASH_D_DH
#define JX_HASH_D_DH

typedef struct _jx_hash_node
{
    JX_Int iData;
    JX_Real fData;
    JX_Int *iDataPtr;
    JX_Int *iDataPtr2;
    JX_Real *fDataPtr;
} jx_HashData;

typedef struct _jx_hash_node_private jx_HashRecord;

struct _jx_hash_dh
{
    JX_Int size;
    JX_Int count;
    JX_Int curMark;
    jx_HashRecord *data;
};

extern void jx_Hash_dhCreate( jx_Hash_dh *h, JX_Int size );
extern void jx_Hash_dhDestroy( jx_Hash_dh h );
extern void jx_Hash_dhInsert( jx_Hash_dh h, JX_Int key, jx_HashData *data );
extern jx_HashData *jx_Hash_dhLookup(jx_Hash_dh h, JX_Int key );
extern void jx_Hash_dhReset( jx_Hash_dh h );
extern void jx_Hash_dhPrint( jx_Hash_dh h, FILE *fp );

#define JX_HASH_1(k,size,idxOut) \
      {  *idxOut = k % size;  }

#define JX_HASH_2(k,size,idxOut) \
      { \
        JX_Int r = k % (size-13); \
        r = (r % 2) ? r : r+1; \
       *idxOut = r; \
      }

#endif

#ifndef JX_MAT_DH_PRIVATE
#define JX_MAT_DH_PRIVATE

extern JX_Int jx_mat_jx_find_owner( JX_Int *beg_rows, JX_Int *end_rows, JX_Int index );
extern void jx_mat_dh_transpose_private( JX_Int m, JX_Int *rpIN, JX_Int **rpOUT,
                    JX_Int *cvalIN, JX_Int **cvalOUT, JX_Real *avalIN, JX_Real **avalOUT );
extern void jx_mat_dh_transpose_reuse_private( JX_Int m, JX_Int *rpIN,
               JX_Int *cvalIN, JX_Real *avalIN, JX_Int *rpOUT, JX_Int *cvalOUT, JX_Real *avalOUT );
extern void jx_readMat( jx_Mat_dh *Aout, char *fileType, char *fileName, JX_Int ignore );
extern void jx_readVec( jx_Vec_dh *bout, char *fileType, char *fileName, JX_Int ignore );
extern void jx_writeMat( jx_Mat_dh Ain, char *fileType, char *fileName );
extern void jx_writeVec( jx_Vec_dh b, char *fileType, char *fileName );
extern void jx_readMat_par( jx_Mat_dh *Aout, char *fileType, char *fileName, JX_Int ignore );
extern void jx_profileMat( jx_Mat_dh A );
extern void jx_mat_dh_print_graph_private( JX_Int m, JX_Int beg_row,
              JX_Int *rp, JX_Int *cval, JX_Real *aval, JX_Int *n2o, JX_Int *o2n, jx_Hash_i_dh hash, FILE* fp );
extern void jx_mat_dh_print_csr_private( JX_Int m, JX_Int *rp, JX_Int *cval, JX_Real *aval, FILE* fp );
extern void jx_mat_dh_read_csr_private( JX_Int *m, JX_Int **rp, JX_Int **cval, JX_Real **aval, FILE* fp );
extern void jx_mat_dh_read_triples_private( JX_Int ignore, JX_Int *m, JX_Int **rp, JX_Int **cval, JX_Real **aval, FILE* fp );
extern void jx_create_nat_ordering_private( JX_Int m, JX_Int **p );
extern void jx_destroy_nat_ordering_private( JX_Int *p );
extern void jx_invert_perm( JX_Int m, JX_Int *pIN, JX_Int *pOUT );
extern void jx_make_full_private( JX_Int m, JX_Int **rp, JX_Int **cval, JX_Real **aval );
extern void jx_make_symmetric_private( JX_Int m, JX_Int **rp, JX_Int **cval, JX_Real **aval );

#endif

#ifndef JX_GET_ROW_DH
#define JX_GET_ROW_DH

extern void jx_EuclidGetDimensions( void *A, JX_Int *beg_row, JX_Int *rowsLocal, JX_Int *rowsGlobal );
extern void jx_EuclidGetRow( void *A, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val );
extern void jx_EuclidRestoreRow( void *A, JX_Int row, JX_Int *len, JX_Int **ind, JX_Real **val );
extern JX_Int jx_EuclidReadLocalNz( void *A );
extern void jx_PrintMatUsingGetRow( void *A, JX_Int beg_row, JX_Int m, JX_Int *n2o_row, JX_Int *n2o_col, char *filename );

#endif

#ifndef JX_ILU_MPI_DH
#define JX_ILU_MPI_DH

void jx_reallocate_private( JX_Int row, JX_Int newEntries,
           JX_Int *nzHave, JX_Int **rp, JX_Int **cval, float **aval, JX_Real **avalD, JX_Int **fill );
extern void jx_ilu_mpi_pilu( jx_Euclid_dh ctx );
extern void jx_iluk_mpi_pilu( jx_Euclid_dh ctx );
extern void jx_compute_scaling_private( JX_Int row, JX_Int len, JX_Real *AVAL, jx_Euclid_dh ctx );
extern void jx_iluk_mpi_bj( jx_Euclid_dh ctx );
extern void jx_iluk_seq( jx_Euclid_dh ctx );
extern void jx_iluk_seq_block( jx_Euclid_dh ctx );
extern void jx_ilut_seq( jx_Euclid_dh ctx );

#endif

#ifndef JX_EUCLID_MPI_INTERFACE_DH
#define JX_EUCLID_MPI_INTERFACE_DH

#define JX_DEFAULT_DROP_TOL 0.01

extern void jx_Euclid_dhCreate( jx_Euclid_dh *ctxOUT );
extern void jx_Euclid_dhDestroy( jx_Euclid_dh ctx );
extern void jx_Euclid_dhSetup( jx_Euclid_dh ctx );
extern void jx_Euclid_dhSolve( jx_Euclid_dh ctx, jx_Vec_dh lhs, jx_Vec_dh rhs, JX_Int *its );
extern void jx_Euclid_dhApply( jx_Euclid_dh ctx, JX_Real *lhs, JX_Real *rhs );
extern void jx_Euclid_dhPrintTestData( jx_Euclid_dh ctx, FILE *fp );
extern void jx_Euclid_dhPrintScaling( jx_Euclid_dh ctx, FILE *fp );
extern void jx_Euclid_dhPrintStatsShort( jx_Euclid_dh ctx, JX_Real setup, JX_Real solve, FILE *fp );
extern void jx_Euclid_dhPrintStatsShorter( jx_Euclid_dh ctx, FILE *fp );
extern void jx_Euclid_dhPrintJxpamgReport( jx_Euclid_dh ctx, FILE *fp );
extern void jx_Euclid_dhPrintStats( jx_Euclid_dh ctx, FILE *fp );
extern void jx_Euclid_dhInputJXpamgMat( jx_Euclid_dh ctx, JX_ParCSRMatrix A );

#define JX_MAX_OPT_LEN 20
#define JX_TIMING_BINS 10

enum{ JX_SOLVE_START_T, JX_TRI_SOLVE_T, JX_SETUP_T,
      JX_SUB_GRAPH_T, JX_FACTOR_T, JX_SOLVE_SETUP_T,
      JX_COMPUTE_RHO_T, JX_TOTAL_SOLVE_TEMP_T, JX_TOTAL_SOLVE_T };

#define JX_STATS_BINS 10

enum{ JX_NZA_STATS, JX_NZF_STATS, JX_NZA_USED_STATS, JX_NZA_RATIO_STATS };

struct _jx_mpi_interface_dh
{
    jx_bool isSetup;
    
    JX_Real rho_init;
    JX_Real rho_final;
    
    JX_Int m;
    JX_Int n;
    JX_Real *rhs;
    void *A;
    jx_Factor_dh F;
    jx_SubdomainGraph_dh sg;
    
    JX_REAL_DH *scale;
    jx_bool isScaled;
    
    JX_Real *work;
    JX_Real *work2;
    JX_Int from, to;
    
    char algo_par[JX_MAX_OPT_LEN];
    char algo_ilu[JX_MAX_OPT_LEN];
    JX_Int level;
    JX_Real droptol;
    JX_Real sparseTolA;
    JX_Real sparseTolF;
    JX_Real pivotMin;
    JX_Real pivotFix;
    JX_Real maxVal;
    
    jx_SortedList_dh slist;
    jx_ExternalRows_dh extRows;
    
    char krylovMethod[JX_MAX_OPT_LEN];
    JX_Int maxIts;
    JX_Real rtol;
    JX_Real atol;
    JX_Int its;
    JX_Int itsTotal;
    
    JX_Int setupCount;
    JX_Int logging;
    JX_Real timing[JX_TIMING_BINS];
    JX_Real stats[JX_STATS_BINS];
    jx_bool timingsWereReduced;
    jx_bool printStats;
};

#endif

#ifndef JX_THREADED_KRYLOV_H
#define JX_THREADED_KRYLOV_H

extern void jx_bicgstab_euclid( jx_Mat_dh A, jx_Euclid_dh ctx, JX_Real *x, JX_Real *b, JX_Int *itsOUT );
extern void jx_cg_euclid( jx_Mat_dh A, jx_Euclid_dh ctx, JX_Real *x, JX_Real *b, JX_Int *itsOUT );

#endif

#ifndef JX_IO_DH
#define JX_IO_DH

extern FILE *jx_openFile_dh( const char *filenameIN, const char *modeIN );
extern void jx_closeFile_dh( FILE *fpIN );
jx_bool jx_isSmallEndian();
extern void jx_io_dh_print_ebin_mat_private( JX_Int m, JX_Int beg_row, JX_Int *rp,
             JX_Int *cval, JX_Real *aval, JX_Int *n2o, JX_Int *o2n, jx_Hash_i_dh hash, char *filename );
extern void jx_io_dh_read_ebin_mat_private( JX_Int *m, JX_Int **rp, JX_Int **cval, JX_Real **aval, char *filename );
extern void jx_io_dh_print_ebin_vec_private( JX_Int n, JX_Int beg_row, JX_Real *vals,
                                JX_Int *n2o, JX_Int *o2n, jx_Hash_i_dh hash, char *filename );
extern void jx_io_dh_read_ebin_vec_private( JX_Int *n, JX_Real **vals, char *filename );

#endif

#ifndef JX_THREADED_BLAS_DH
#define JX_THREADED_BLAS_DH

extern void jx_matvec_euclid_seq( JX_Int n, JX_Int *rp, JX_Int *cval, JX_Real *aval, JX_Real *x, JX_Real *y );
extern JX_Real jx_InnerProd( JX_Int local_n, JX_Real *x, JX_Real *y );
extern void jx_Axpy( JX_Int n, JX_Real alpha, JX_Real *x, JX_Real *y );
extern JX_Real jx_Norm2( JX_Int n, JX_Real *x );
extern void jx_CopyVec( JX_Int n, JX_Real *xIN, JX_Real *yOUT );
extern void jx_ScaleVec( JX_Int n, JX_Real alpha, JX_Real *x );

#endif

#endif
