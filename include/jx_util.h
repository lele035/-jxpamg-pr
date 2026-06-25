//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_util.h -- head files for basic tools
 *  Date: 2011/09/08
 *  Modified Date: 2012/10/22
 *
 *  Created by peghoty
 *  Modified by Yue Xiaoqiang
 */ 

#ifndef JX_UTIL_HEADER
#define JX_UTIL_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/times.h>
#include <float.h>
#ifndef JX_SEQUENTIAL
#include <mpi.h>
#endif
#if JX_USING_OPENMP
#include <omp.h>
#endif

/*----------------------------------------------------------------*
 *                      Macro Definition                          *
 *----------------------------------------------------------------*/ 
// #define PCG_Modified_zl // zhaoli 

/* BigInt */
typedef long int jx_longint;
#if JX_USING_BIG_INT
typedef long long int JX_Int;
typedef unsigned long long int JX_UInt;
#define JX_MPI_INT MPI_LONG_LONG_INT
#else
typedef int JX_Int;
typedef unsigned int JX_UInt;
#define JX_MPI_INT MPI_INT
#endif

/* BigDouble */
#if JX_USING_BIG_DOUBLE
typedef long double JX_Real;
#define JX_MPI_REAL MPI_LONG_DOUBLE
#else
typedef double JX_Real;
#define JX_MPI_REAL MPI_DOUBLE
#endif

/* Sequential MPI stuff */
#ifdef JX_SEQUENTIAL
typedef JX_Int MPI_Comm;
typedef JX_Int MPI_Group;
typedef JX_Int MPI_Request;
typedef JX_Int MPI_Datatype;
typedef void (MPI_User_function) ();
typedef struct
{
   JX_Int MPI_SOURCE;
   JX_Int MPI_TAG;
} MPI_Status;
typedef JX_Int MPI_Op;
typedef JX_Int MPI_Aint;
#endif

/* small real number */
#define EPS 1.0e-15
#define JX_PARCSR 5555
#define JX_UNITIALIZED -999
#define JX_REAL_MIN_EXP DBL_MIN_EXP

/* max and min operation */
#define jx_max(a,b)  (((a)<(b)) ? (b) : (a))
#define jx_min(a,b)  (((a)<(b)) ? (a) : (b))

/* Global variable used in jx-error checking */
extern JX_Int jx__kdim_memory_error;
extern JX_Int jx__global_mvcpu_flag;
extern JX_Int jx__global_error;
#define jx_error_flag     jx__global_error
#define JX_ERROR_ARG      4                /* argument error */
#define JX_ERROR_GENERIC  1
#define JX_ERROR_MEMORY   2
#define JX_ERROR_CONV     256

/* Thread */
#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)
JX_Int jx_NumThreads( void );
JX_Int jx_NumActiveThreads( void );
JX_Int jx_GetThreadNum( void );
#else
#define jx_NumThreads() 1
#define jx_NumActiveThreads() 1
#define jx_GetThreadNum() 0
#endif

/* Memory */
#define jx_TAlloc(type, count)        ( (type *)jx_MAlloc((size_t)(sizeof(type) * (count))) )
#define jx_CTAlloc(type, count)       ( (type *)jx_CAlloc((size_t)(count), (size_t)sizeof(type)) )
#define jx_TReAlloc(ptr, type, count) ( (type *)jx_ReAlloc((char *)ptr, (size_t)(sizeof(type) * (count))) )
#define jx_TFree(ptr)                 ( jx_Free((char *)ptr), ptr = NULL )

/* Fortran, BLAS, LAPACK */
#define jx_F90_NAME(name,NAME) name##_
#define jx_F90_NAME_BLAS(name,NAME) name##_
#define jx_F90_NAME_LAPACK(name,NAME) name##_

/* Partition */
#define JX_OMP_GET_START_END(procid,nprocs,n,start,end) \
        if((procid)<(n)%(nprocs))                       \
        {                                               \
            (end)=(n)/(nprocs)+1;                       \
            (start)=(end)*(procid);                     \
        }                                               \
        else                                            \
        {                                               \
            (end)=(n)/(nprocs);                         \
            (start)=(end)*(procid)+(n)%(nprocs);        \
        }                                               \
        (end)=(end)+(start);

/* common struct */
struct jx_Solver_struct;
typedef struct jx_Solver_struct *JX_Solver;

struct jx_Matrix_struct;
typedef struct jx_Matrix_struct *JX_Matrix;

struct jx_Vector_struct;
typedef struct jx_Vector_struct *JX_Vector;

struct jx_ParCSRMatrix_struct;
typedef struct jx_ParCSRMatrix_struct *JX_ParCSRMatrix;

struct jx_ParVector_struct;
typedef struct jx_ParVector_struct *JX_ParVector;

struct jx_IJMatrix_struct;
typedef struct jx_IJMatrix_struct *JX_IJMatrix;

struct jx_IJVector_struct;
typedef struct jx_IJVector_struct *JX_IJVector;

/*------------------------------------------------------------*/
/*--------------------- Extern Definition --------------------*/
/*------------------------------------------------------------*/

extern JX_UInt jx_long_size_length_rap;
extern JX_UInt jx_long_size_length_interp;
extern JX_UInt jx_total_calloc_global_rap;
extern JX_UInt jx_total_calloc_global_interp;
extern JX_UInt jx_real_long_size_length_rap;
extern JX_UInt jx_real_long_size_length_interp;
extern JX_Real jx_total_elapsed_time_matvec;
extern JX_Int  jx_total_iter_index;

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jx_DataExchangeResponse
 * \brief In the fill_response() function the user needs to set the recv__buf
 *        and the response_message_size.  Memory of size send_response_storage has been
 *        alllocated for the send_buf (in exchange_data) - if more is needed, then
 *        realloc and adjust the send_response_storage. 
 *        The realloc amount should be storage+overhead. 
 *        If the response is an empty "confirmation" message, then set
 *        response_message_size = 0 (and do not modify the send_buf).
 */
typedef struct
{
   JX_Int  (*fill_response)(void* recv_buf, JX_Int contact_size, 
                            JX_Int contact_proc, void* response_obj, 
                            MPI_Comm comm, void** response_buf, 
                            JX_Int* response_message_size);
   JX_Int  send_response_overhead; /*set by exchange data */
   JX_Int  send_response_storage;  /*storage allocated for send_response_buf*/
   void   *data1;                  /*data fields user may want to access in fill_response */
   void   *data2;
   
} jx_DataExchangeResponse;


/*!
 * \struct jx_BinaryTree
 */
typedef struct
{
   JX_Int  parent_id;
   JX_Int  num_child;
   JX_Int *child_id;
   
} jx_BinaryTree;

#define jx_BinaryTreeParentId(tree)    (tree->parent_id)
#define jx_BinaryTreeNumChild(tree)    (tree->num_child)
#define jx_BinaryTreeChildIds(tree)    (tree->child_id)
#define jx_BinaryTreeChildId(tree, i)  (tree->child_id[i])


/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/utilities/exchange.c */
JX_Int 
jx_DataExchangeList( JX_Int                       num_contacts, 
		     JX_Int                      *contact_proc_list, 
		     void                     *contact_send_buf, 
		     JX_Int                      *contact_send_buf_starts, 
		     JX_Int                       contact_obj_size, 
                     JX_Int                       response_obj_size,
		     jx_DataExchangeResponse  *response_obj, 
		     JX_Int                       max_response_size, 
                     JX_Int                       rnum, 
                     MPI_Comm                  comm,  
                     void                    **p_response_recv_buf, 
                     JX_Int                     **p_response_recv_buf_starts );
JX_Int jx_CreateBinaryTree( JX_Int myid, JX_Int num_procs, jx_BinaryTree *tree );
JX_Int jx_DestroyBinaryTree( jx_BinaryTree *tree );

/* csrc/utilities/memory.c */
JX_Int    jx_OutOfMemory( size_t size );
char * jx_MAlloc( size_t size );
char * jx_CAlloc( size_t count, size_t elt_size );
char * jx_ReAlloc( char *ptr, size_t size );
void   jx_Free( char *ptr );

/* csrc/utilities/error.c */
void jx_error_handler( char *filename, JX_Int line, JX_Int ierr, const char *msg );
#define jx_error(IERR)  jx_error_handler(__FILE__, __LINE__, IERR, NULL)
#define jx_error_w_msg(IERR, msg) jx_error_handler(__FILE__, __LINE__, IERR, msg)
#define jx_error_in_arg(IARG)  jx_error(JX_ERROR_ARG | IARG<<3)
#ifdef NDEBUG
#define jx_assert(EX)
#else
#define jx_assert(EX) \
 if(!(EX)) {jx_fprintf(stderr,"jx_assert failed: %s\n", #EX); jx_error(1);}
#endif

/* /csrc/utilities/sort.c */
void jx_swap( JX_Int *v, JX_Int i, JX_Int j );
void jx_swap1( JX_Real *v, JX_Int i, JX_Int j );
void jx_swap2( JX_Int *v, JX_Real *w, JX_Int i, JX_Int j );
void jx_swap2i( JX_Int *v, JX_Int *w, JX_Int i, JX_Int j );
void jx_swap3i( JX_Int *v, JX_Int *w, JX_Int *z, JX_Int i, JX_Int j );
void jx_qsortd( JX_Real *v, JX_Int left, JX_Int right );
void jx_qsort0( JX_Int *v, JX_Int left, JX_Int right );
void jx_qsort1( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right );
void jx_qsort2( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right );
void jx_qsort2i( JX_Int *v, JX_Int *w, JX_Int left, JX_Int right );
void jx_qsort2abs( JX_Int *v, JX_Real *w, JX_Int left, JX_Int right );
void jx_qsort3i( JX_Int *v, JX_Int *w, JX_Int *z, JX_Int left, JX_Int right );
void jx_diQuickSort12( JX_Real *v, JX_Int *w, JX_Int left, JX_Int right );
void jx_diQuickSort21( JX_Real *v, JX_Int *w, JX_Int left, JX_Int right );

/* /csrc/utilities/search.c */
JX_Int jx_BinarySearch( JX_Int *list, JX_Int value, JX_Int list_length );

/* /csrc/utilities/array.c */
JX_Int jx_DoubleArrayPrint( JX_Real *x, JX_Int ndof, char *filename );
JX_Int jx_IntegerArrayGetInterp( JX_Int *icor_interp, JX_Int n_fine, JX_Int nbl, JX_Int nbr );
JX_Int jx_IntegerArraySetConstantValues( JX_Int n, JX_Int *x, JX_Int val );
JX_Int jx_IntegerArrayModFine2Coarse( JX_Int num_rows_A, JX_Int *fine_to_coarse );
JX_Int jx_IntegerArrayGenerateIcor( JX_Int num_rows_A,
                                    JX_Int num_cols_P,
                                    JX_Int *fine_to_coarse,
                                    JX_Int nbl,
                                    JX_Int nbr,
                                    JX_Int *CF_marker,
                                    JX_Int *icor );
JX_Int jx_IntegerArrayCopy( JX_Int size, JX_Int *x, JX_Int *y );
JX_Int jx_DoubleArrayCopy( JX_Int size, JX_Real *x, JX_Real *y );
JX_Int jx_DoubleArrayReciprocalMap( JX_Real *u, JX_Int n, JX_Int *map );
JX_Real **jx_DyadicDoubleArrayCTAlloc( JX_Int first, JX_Int second );
JX_Real ***jx_TeracidicDoubleArrayCTAlloc( JX_Int first, JX_Int second, JX_Int third );
JX_Real ****jx_TetravalentDoubleArrayCTAlloc( JX_Int first, JX_Int second, JX_Int third, JX_Int fourth );
void jx_DyadicDoubleArrayFree( JX_Real **ptr, JX_Int first );
void jx_TeracidicDoubleArrayFree( JX_Real ***ptr, JX_Int first, JX_Int second );
void jx_TetravalentDoubleArrayFree( JX_Real ****ptr, JX_Int first, JX_Int second, JX_Int third );
JX_Real jx_DoubleArrayAbsMaxElement( JX_Real *x, JX_Int n );
JX_Real jx_DoubleArrayAbsMinElement( JX_Real *x, JX_Int n );
JX_Real jx_DoubleArrayMaxElement( JX_Real *x, JX_Int n );

/* /csrc/utilities/intersection.c */
JX_Int jx_MultiSetsIntersection( JX_Int numset, JX_Int **set, JX_Int *size, JX_Int **c_ptr, JX_Int *nc_ptr, JX_Int *empty_ptr );
JX_Int jx_TwoSetsIntersection( JX_Int *a, JX_Int na, JX_Int *b, JX_Int nb, JX_Int **c_ptr, JX_Int *nc_ptr, JX_Int *empty_ptr );

/* csrc/utilities/random.c */
void jx_SeedRand( JX_Int seed );
JX_Real jx_Rand();                   
                                         
/* csrc/utilities/timer.c */
JX_Real 
jx_time_getWallclockSeconds(void);
void
jx_GetWallTime( MPI_Comm  comm, 
                char     *fctname, 
                JX_Real    starttime, 
                JX_Real    endtime, 
                JX_Int       allproc,
                JX_Int       dp );
JX_Real
jx_GetWallTimeMax( MPI_Comm  comm, 
                   JX_Real    starttime, 
                   JX_Real    endtime );

/* csrc/utilities/printf.c */
JX_Int
jx_new_format( const char *format, char **newformat_ptr );
JX_Int
jx_free_format( char *newformat );
JX_Int
jx_printf( const char *format, ... );
JX_Int
jx_fprintf( FILE *stream, const char *format, ... );
JX_Int
jx_sprintf( char *s, const char *format, ... );
JX_Int
jx_scanf( const char *format, ... );
JX_Int
jx_fscanf( FILE *stream, const char *format, ... );
JX_Int
jx_sscanf( char *s, const char *format, ... );

/* csrc/utilities/set.c */
void jx_set_mvcpu_handler( JX_Int new_mvcpu_flag );

/* csrc/utilities/mpistubs.c */
JX_Int
jx_MPI_Init( int *argc, char ***argv );
JX_Int
jx_MPI_Finalize();
JX_Int
jx_MPI_Abort( MPI_Comm comm, JX_Int errorcode );
JX_Real
jx_MPI_Wtime();
JX_Real
jx_MPI_Wtick();
JX_Int
jx_MPI_Barrier( MPI_Comm comm );
JX_Int
jx_MPI_Comm_create( MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm );
JX_Int
jx_MPI_Comm_dup( MPI_Comm comm, MPI_Comm *newcomm );
JX_Int
jx_MPI_Comm_size( MPI_Comm comm, JX_Int *size );
JX_Int
jx_MPI_Comm_rank( MPI_Comm comm, JX_Int *rank );
JX_Int
jx_MPI_Comm_free( MPI_Comm *comm );
JX_Int
jx_MPI_Comm_group( MPI_Comm comm, MPI_Group *group );
JX_Int
jx_MPI_Comm_split( MPI_Comm comm, JX_Int n, JX_Int m, MPI_Comm *comms );
JX_Int
jx_MPI_Group_incl( MPI_Group group, JX_Int n, JX_Int *ranks, MPI_Group *newgroup );
JX_Int
jx_MPI_Group_free( MPI_Group *group );
JX_Int
jx_MPI_Address( void *location, MPI_Aint *address );
JX_Int
jx_MPI_Get_count( MPI_Status *status, MPI_Datatype datatype, JX_Int *count );
JX_Int
jx_MPI_Alltoall( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, MPI_Comm comm );
JX_Int
jx_MPI_Allgather( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype, void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, MPI_Comm comm );
JX_Int
jx_MPI_Allgatherv( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, JX_Int *recvcounts, JX_Int *displs, MPI_Datatype recvtype, MPI_Comm comm );
JX_Int
jx_MPI_Gather( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  JX_Int recvcount, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm );
JX_Int
jx_MPI_Gatherv( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  JX_Int *recvcounts, JX_Int *displs, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm );
JX_Int
jx_MPI_Scatter( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm );
JX_Int
jx_MPI_Scatterv( void *sendbuf, JX_Int *sendcounts, JX_Int *displs, MPI_Datatype sendtype,
                   void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm );
JX_Int
jx_MPI_Bcast( void *buffer, JX_Int count, MPI_Datatype datatype, JX_Int root, MPI_Comm comm );
JX_Int
jx_MPI_Send( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest, JX_Int tag, MPI_Comm comm );
JX_Int
jx_MPI_Recv( void *buf, JX_Int count, MPI_Datatype  datatype, JX_Int source, JX_Int tag, MPI_Comm comm, MPI_Status *status );
JX_Int
jx_MPI_Isend( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest, JX_Int tag, MPI_Comm comm, MPI_Request *request );
JX_Int
jx_MPI_Irecv( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int source, JX_Int tag, MPI_Comm comm, MPI_Request *request );
JX_Int
jx_MPI_Send_init( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest, JX_Int tag, MPI_Comm comm, MPI_Request *request );
JX_Int
jx_MPI_Recv_init( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest,
                     JX_Int tag, MPI_Comm comm, MPI_Request *request );
JX_Int
jx_MPI_Irsend( void *buf, JX_Int count, MPI_Datatype datatype,
                  JX_Int dest, JX_Int tag, MPI_Comm comm, MPI_Request *request );
JX_Int
jx_MPI_Startall( JX_Int count, MPI_Request *array_of_requests );
JX_Int
jx_MPI_Probe( JX_Int source, JX_Int tag, MPI_Comm comm, MPI_Status *status );
JX_Int
jx_MPI_Iprobe( JX_Int source, JX_Int tag, MPI_Comm comm, JX_Int *flag, MPI_Status *status );
JX_Int
jx_MPI_Test( MPI_Request *request, JX_Int *flag, MPI_Status *status );
JX_Int
jx_MPI_Testall( JX_Int count, MPI_Request *array_of_requests, JX_Int *flag, MPI_Status *array_of_statuses );
JX_Int
jx_MPI_Wait( MPI_Request *request, MPI_Status *status );
JX_Int
jx_MPI_Waitall( JX_Int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses );
JX_Int
jx_MPI_Waitany( JX_Int count, MPI_Request *array_of_requests, JX_Int *index, MPI_Status *status );
JX_Int
jx_MPI_Allreduce( void *sendbuf, void *recvbuf, JX_Int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm );
JX_Int
jx_MPI_Reduce( void *sendbuf, void *recvbuf, JX_Int count, MPI_Datatype datatype, MPI_Op op, JX_Int root, MPI_Comm comm );
JX_Int
jx_MPI_Scan( void *sendbuf, void *recvbuf, JX_Int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm );
JX_Int
jx_MPI_Request_free( MPI_Request *request );
JX_Int
jx_MPI_Type_contiguous( JX_Int count, MPI_Datatype oldtype, MPI_Datatype *newtype );
JX_Int
jx_MPI_Type_vector( JX_Int count, JX_Int blocklength, JX_Int stride, MPI_Datatype oldtype, MPI_Datatype *newtype );
JX_Int
jx_MPI_Type_hvector( JX_Int count, JX_Int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype );
JX_Int
jx_MPI_Type_struct( JX_Int count, JX_Int *array_of_blocklengths, MPI_Aint *array_of_displacements,
                       MPI_Datatype *array_of_types, MPI_Datatype *newtype );
JX_Int
jx_MPI_Type_commit( MPI_Datatype *datatype );
JX_Int
jx_MPI_Type_free( MPI_Datatype *datatype );
JX_Int
jx_MPI_Op_free( MPI_Op *op );
JX_Int
jx_MPI_Op_create( MPI_User_function *function, int commute, MPI_Op *op );

/* csrc/euclid/parcsr_euclid.c */
JX_Int 
JX_EuclidCreate( MPI_Comm comm, JX_Solver *solver );
JX_Int 
JX_EuclidDestroy( JX_Solver solver );
JX_Int 
JX_EuclidSetup( JX_Solver solver,
                JX_ParCSRMatrix A );
JX_Int 
JX_EuclidSolve( JX_Solver solver,
                JX_ParCSRMatrix A,
                JX_ParVector bb,
                JX_ParVector xx );
JX_Int 
JX_EuclidSetParams( JX_Solver solver, JX_Int argc, char *argv[] );
JX_Int 
JX_EuclidSetParamsFromFile( JX_Solver solver, char *filename );
JX_Int 
JX_EuclidSetLevel( JX_Solver solver, JX_Int level );
JX_Int 
JX_EuclidSetBJ( JX_Solver solver, JX_Int bj );
JX_Int 
JX_EuclidSetStats( JX_Solver solver, JX_Int eu_stats );
JX_Int 
JX_EuclidSetMem( JX_Solver solver, JX_Int eu_mem );
JX_Int 
JX_EuclidSetILUT( JX_Solver solver, JX_Real ilut );
JX_Int 
JX_EuclidSetSparseA( JX_Solver solver, JX_Real sparse_A );
JX_Int 
JX_EuclidSetRowScale( JX_Solver solver, JX_Int row_scale );

#endif
