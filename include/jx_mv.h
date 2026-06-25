//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_mv.h -- head files for matrix-vector operation
 *  Date: 2011/09/08
 *  Modified Date: 2012/10/22
 *
 *  Created by peghoty
 *  Modified by Yue Xiaoqiang
 */ 

#ifndef JX_MV_HEADER
#define JX_MV_HEADER

#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/
 
/*!
 * \struct jx_Vector
 */  
typedef struct
{
   JX_Real  *data;
   JX_Int    size;
   JX_Int    owns_data;
   JX_Int    num_vectors;  
   JX_Int    multivec_storage_method;
   JX_Int    vecstride, idxstride;

} jx_Vector;

#define jx_VectorData(vector)                   ((vector) -> data)
#define jx_VectorSize(vector)                   ((vector) -> size)
#define jx_VectorOwnsData(vector)               ((vector) -> owns_data)
#define jx_VectorNumVectors(vector)             ((vector) -> num_vectors)
#define jx_VectorMultiVecStorageMethod(vector)  ((vector) -> multivec_storage_method)
#define jx_VectorVectorStride(vector)           ((vector) -> vecstride )
#define jx_VectorIndexStride(vector)            ((vector) -> idxstride )

/*!
 * \struct jx_CSRMatrix
 */    
typedef struct
{
   JX_Real *data;
   JX_Int  *i;
   JX_Int  *j;
   JX_Int   num_rows;
   JX_Int   num_cols;
   JX_Int   num_nonzeros;
   JX_Int  *rownnz;
   JX_Int   num_rownnz;
   JX_Int   owns_data;

} jx_CSRMatrix;

#define jx_CSRMatrixData(matrix)         ((matrix) -> data)
#define jx_CSRMatrixI(matrix)            ((matrix) -> i)
#define jx_CSRMatrixJ(matrix)            ((matrix) -> j)
#define jx_CSRMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define jx_CSRMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define jx_CSRMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define jx_CSRMatrixRownnz(matrix)       ((matrix) -> rownnz)
#define jx_CSRMatrixNumRownnz(matrix)    ((matrix) -> num_rownnz)
#define jx_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)

#define JX_USING_PERSISTENT_COMM
#ifdef JX_USING_PERSISTENT_COMM
typedef enum CommPkgJobType
{
   JX_COMM_PKG_JOB_COMPLEX = 0,
   JX_COMM_PKG_JOB_COMPLEX_TRANSPOSE,
   JX_COMM_PKG_JOB_INT,
   JX_COMM_PKG_JOB_INT_TRANSPOSE,
   NUM_OF_COMM_PKG_JOB_TYPE
} CommPkgJobType;

typedef struct
{
   void     *send_data;
   void     *recv_data;

   JX_Int          num_requests;
   MPI_Request    *requests;

   JX_Int          own_send_data, own_recv_data;

} jx_ParCSRPersistentCommHandle;
#endif

/*!
 * \struct jx_ParCSRCommPkg
 * \brief Structure containing information for doing communications
 */    
typedef struct
{
   MPI_Comm     comm;
   JX_Int       num_sends;
   JX_Int      *send_procs;
   JX_Int      *send_map_starts;
   JX_Int      *send_map_elmts;

   JX_Int       num_recvs;
   JX_Int      *recv_procs;
   JX_Int      *recv_vec_starts;

   /* remote communication information */
   MPI_Datatype  *send_mpi_types;
   MPI_Datatype  *recv_mpi_types;

#ifdef JX_USING_PERSISTENT_COMM
   jx_ParCSRPersistentCommHandle *persistent_comm_handles[NUM_OF_COMM_PKG_JOB_TYPE];
#endif

} jx_ParCSRCommPkg;

#define jx_ParCSRCommPkgComm(comm_pkg)           (comm_pkg -> comm)                                        
#define jx_ParCSRCommPkgNumSends(comm_pkg)       (comm_pkg -> num_sends)
#define jx_ParCSRCommPkgSendProcs(comm_pkg)      (comm_pkg -> send_procs)
#define jx_ParCSRCommPkgSendProc(comm_pkg, i)    (comm_pkg -> send_procs[i])
#define jx_ParCSRCommPkgSendMapStarts(comm_pkg)  (comm_pkg -> send_map_starts)
#define jx_ParCSRCommPkgSendMapStart(comm_pkg,i) (comm_pkg -> send_map_starts[i])
#define jx_ParCSRCommPkgSendMapElmts(comm_pkg)   (comm_pkg -> send_map_elmts)
#define jx_ParCSRCommPkgSendMapElmt(comm_pkg,i)  (comm_pkg -> send_map_elmts[i])
#define jx_ParCSRCommPkgNumRecvs(comm_pkg)       (comm_pkg -> num_recvs)
#define jx_ParCSRCommPkgRecvProcs(comm_pkg)      (comm_pkg -> recv_procs)
#define jx_ParCSRCommPkgRecvProc(comm_pkg, i)    (comm_pkg -> recv_procs[i])
#define jx_ParCSRCommPkgRecvVecStarts(comm_pkg)  (comm_pkg -> recv_vec_starts)
#define jx_ParCSRCommPkgRecvVecStart(comm_pkg,i) (comm_pkg -> recv_vec_starts[i])
#define jx_ParCSRCommPkgSendMPITypes(comm_pkg)   (comm_pkg -> send_mpi_types)
#define jx_ParCSRCommPkgSendMPIType(comm_pkg,i)  (comm_pkg -> send_mpi_types[i])
#define jx_ParCSRCommPkgRecvMPITypes(comm_pkg)   (comm_pkg -> recv_mpi_types)
#define jx_ParCSRCommPkgRecvMPIType(comm_pkg,i)  (comm_pkg -> recv_mpi_types[i])

/*!
 * \struct jx_ParCSRCommHandle
 */
typedef struct
{
   jx_ParCSRCommPkg  *comm_pkg;
   void 	     *send_data;
   void 	     *recv_data;

   JX_Int                num_requests;
   MPI_Request       *requests;

} jx_ParCSRCommHandle;

#define jx_ParCSRCommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define jx_ParCSRCommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define jx_ParCSRCommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define jx_ParCSRCommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define jx_ParCSRCommHandleRequests(comm_handle)    (comm_handle -> requests)
#define jx_ParCSRCommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

/*!
 * \struct jx_AuxParCSRMatrix
 */
typedef struct
{
   JX_Int      local_num_rows;   /* defines number of rows on this processors */
   JX_Int      local_num_cols;   /* defines number of cols of diag */

   JX_Int      need_aux; /* if need_aux = 1, aux_j, aux_data are used to
			generate the parcsr matrix (default),
			for need_aux = 0, data is put directly into
			parcsr structure (requires the knowledge of
			offd_i and diag_i ) */

   JX_Int     *row_length; /* row_length_diag[i] contains number of stored
				elements in i-th row */
   JX_Int     *row_space; /* row_space_diag[i] contains space allocated to
				i-th row */
   JX_Int    **aux_j;	/* contains collected column indices */
   JX_Real   **aux_data; /* contains collected data */

   JX_Int     *indx_diag; /* indx_diag[i] points to first empty space of portion
			 in diag_j , diag_data assigned to row i */
   JX_Int     *indx_offd; /* indx_offd[i] points to first empty space of portion
			 in offd_j , offd_data assigned to row i */
   JX_Int	    max_off_proc_elmts; /* length of off processor stash set for
					SetValues and AddTOValues */
   JX_Int	    current_num_elmts; /* current no. of elements stored in stash */
   JX_Int	    off_proc_i_indx; /* pointer to first empty space in
				set_off_proc_i_set */
   JX_Int     *off_proc_i; /* length 2*num_off_procs_elmts, contains info pairs
			(code, no. of elmts) where code contains global
			row no. if  SetValues, and (-global row no. -1)
			if  AddToValues*/
   JX_Int     *off_proc_j; /* contains column indices */
   JX_Real  *off_proc_data; /* contains corresponding data */
   JX_Int	    cancel_indx; /* number of elements that have to be deleted due
			   to setting values from another processor */
} jx_AuxParCSRMatrix;

#define jx_AuxParCSRMatrixLocalNumRows(matrix)     ((matrix) -> local_num_rows)
#define jx_AuxParCSRMatrixLocalNumCols(matrix)     ((matrix) -> local_num_cols)
#define jx_AuxParCSRMatrixNeedAux(matrix)          ((matrix) -> need_aux)
#define jx_AuxParCSRMatrixRowLength(matrix)        ((matrix) -> row_length)
#define jx_AuxParCSRMatrixRowSpace(matrix)         ((matrix) -> row_space)
#define jx_AuxParCSRMatrixAuxJ(matrix)             ((matrix) -> aux_j)
#define jx_AuxParCSRMatrixAuxData(matrix)          ((matrix) -> aux_data)
#define jx_AuxParCSRMatrixIndxDiag(matrix)         ((matrix) -> indx_diag)
#define jx_AuxParCSRMatrixIndxOffd(matrix)         ((matrix) -> indx_offd)
#define jx_AuxParCSRMatrixMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define jx_AuxParCSRMatrixCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define jx_AuxParCSRMatrixOffProcIIndx(matrix)     ((matrix) -> off_proc_i_indx)
#define jx_AuxParCSRMatrixOffProcI(matrix)         ((matrix) -> off_proc_i)
#define jx_AuxParCSRMatrixOffProcJ(matrix)         ((matrix) -> off_proc_j)
#define jx_AuxParCSRMatrixOffProcData(matrix)      ((matrix) -> off_proc_data)
#define jx_AuxParCSRMatrixCancelIndx(matrix)       ((matrix) -> cancel_indx)

/*!
 * \struct jx_AuxParVector
 */
typedef struct
{
   JX_Int	    max_off_proc_elmts; /* length of off processor stash for
                                           SetValues and AddToValues*/
   JX_Int	    current_num_elmts;  /* current no. of elements stored in stash */
   JX_Int      *off_proc_i;         /* contains column indices */
   JX_Real     *off_proc_data;      /* contains corresponding data */
   JX_Int	    cancel_indx;        /* number of elements that have to be deleted due
                                    to setting values from another processor */
} jx_AuxParVector;

#define jx_AuxParVectorMaxOffProcElmts(matrix)  ((matrix) -> max_off_proc_elmts)
#define jx_AuxParVectorCurrentNumElmts(matrix)  ((matrix) -> current_num_elmts)
#define jx_AuxParVectorOffProcI(matrix)         ((matrix) -> off_proc_i)
#define jx_AuxParVectorOffProcData(matrix)      ((matrix) -> off_proc_data)
#define jx_AuxParVectorCancelIndx(matrix)       ((matrix) -> cancel_indx)

/*!
 * \struct jx_IJMatrix
 */
typedef struct jx_IJMatrix_struct
{
   MPI_Comm    comm;

   JX_Int        *row_partitioning;    /* distribution of rows across processors */
   JX_Int        *col_partitioning;    /* distribution of columns */

   JX_Int         object_type;         /* Indicates the type of "object" */
   void       *object;              /* Structure for storing local portion */
   void       *translator;          /* optional storage_type specfic structure
                                       for holding additional local info */
   void       *assumed_part;	    /* IJMatrix assumed partition */
   JX_Int         assemble_flag;       /* indicates whether matrix has been assembled */

   JX_Int         global_first_row;    /* these for data items are necessary */
   JX_Int         global_first_col;    /* to be able to avoind using the global */
   JX_Int         global_num_rows;     /* global partition */ 
   JX_Int         global_num_cols;
   JX_Int         omp_flag;
   JX_Int         print_level;

} jx_IJMatrix;

#define jx_IJMatrixComm(matrix)              ((matrix) -> comm)
#define jx_IJMatrixRowPartitioning(matrix)   ((matrix) -> row_partitioning)
#define jx_IJMatrixColPartitioning(matrix)   ((matrix) -> col_partitioning)
#define jx_IJMatrixObjectType(matrix)        ((matrix) -> object_type)
#define jx_IJMatrixObject(matrix)            ((matrix) -> object)
#define jx_IJMatrixTranslator(matrix)        ((matrix) -> translator)
#define jx_IJMatrixAssumedPart(matrix)       ((matrix) -> assumed_part)
#define jx_IJMatrixAssembleFlag(matrix)      ((matrix) -> assemble_flag)
#define jx_IJMatrixGlobalFirstRow(matrix)    ((matrix) -> global_first_row)
#define jx_IJMatrixGlobalFirstCol(matrix)    ((matrix) -> global_first_col)
#define jx_IJMatrixGlobalNumRows(matrix)     ((matrix) -> global_num_rows)
#define jx_IJMatrixGlobalNumCols(matrix)     ((matrix) -> global_num_cols)
#define jx_IJMatrixOMPFlag(matrix)           ((matrix) -> omp_flag)
#define jx_IJMatrixPrintLevel(matrix)        ((matrix) -> print_level)

/*!
 * \struct jx_IJVector
 */ 
typedef struct jx_IJVector_struct
{
   MPI_Comm      comm;
   JX_Int 		*partitioning;      /* Indicates partitioning over tasks */
   JX_Int           object_type;       /* Indicates the type of "local storage" */
   void         *object;            /* Structure for storing local portion */
   void         *translator;        /* Structure for storing off processor information */
   void         *assumed_part;        /* IJ Vector assumed partition */
   JX_Int           global_first_row;  /* these for data items are necessary */
   JX_Int           global_num_rows;   /* to be able to avoind using the global */
                                    /* global partition */
   JX_Int	         print_level;
} jx_IJVector;

#define jx_IJVectorComm(vector)           ((vector) -> comm)
#define jx_IJVectorPartitioning(vector)   ((vector) -> partitioning)
#define jx_IJVectorObjectType(vector)     ((vector) -> object_type)
#define jx_IJVectorObject(vector)         ((vector) -> object)
#define jx_IJVectorTranslator(vector)     ((vector) -> translator)
#define jx_IJVectorAssumedPart(vector)    ((vector) -> assumed_part)
#define jx_IJVectorGlobalFirstRow(vector) ((vector) -> global_first_row)
#define jx_IJVectorGlobalNumRows(vector)  ((vector) -> global_num_rows)
#define jx_IJVectorPrintLevel(vector)     ((vector) -> print_level)

/*!
 * \struct jx_IJAssumedPart
 */  
typedef struct
{
   JX_Int  length;
   JX_Int  row_start;
   JX_Int  row_end;
   JX_Int  storage_length;
   JX_Int *proc_list;
   JX_Int *row_start_list;
   JX_Int *row_end_list;  
   JX_Int *sort_index;

} jx_IJAssumedPart;


/*!
 * \struct jx_ParCSRMatrix
 */
typedef struct
{
   MPI_Comm          comm;
   JX_Int               global_num_rows;
   JX_Int               global_num_cols;
   JX_Int               first_row_index;
   JX_Int               first_col_diag;
   JX_Int               last_row_index;
   JX_Int               last_col_diag;
   jx_CSRMatrix	    *diag;
   jx_CSRMatrix	    *offd;
   jx_CSRMatrix     *diagT;
   jx_CSRMatrix     *offdT;
   JX_Int              *col_map_offd;
   JX_Int              *row_starts;
   JX_Int              *col_starts;
   jx_ParCSRCommPkg *comm_pkg;
   jx_ParCSRCommPkg *comm_pkgT;
   JX_Int               owns_data;
   JX_Int               owns_row_starts;
   JX_Int               owns_col_starts;
   JX_Int               num_nonzeros;
   JX_Real            d_num_nonzeros;
   JX_Int              *rowindices;
   JX_Real             *rowvalues;
   JX_Int               getrowactive;
   jx_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                           is used (compile-time option) */
} jx_ParCSRMatrix;

#define jx_ParCSRMatrixComm(matrix)	        ((matrix) -> comm)
#define jx_ParCSRMatrixGlobalNumRows(matrix)    ((matrix) -> global_num_rows)
#define jx_ParCSRMatrixGlobalNumCols(matrix)    ((matrix) -> global_num_cols)
#define jx_ParCSRMatrixFirstRowIndex(matrix)    ((matrix) -> first_row_index)
#define jx_ParCSRMatrixFirstColDiag(matrix)     ((matrix) -> first_col_diag)
#define jx_ParCSRMatrixLastRowIndex(matrix)     ((matrix) -> last_row_index)
#define jx_ParCSRMatrixLastColDiag(matrix)      ((matrix) -> last_col_diag)
#define jx_ParCSRMatrixDiag(matrix)  	        ((matrix) -> diag)
#define jx_ParCSRMatrixOffd(matrix)  	        ((matrix) -> offd)
#define jx_ParCSRMatrixDiagT(matrix)  	        ((matrix) -> diagT)
#define jx_ParCSRMatrixOffdT(matrix)  	        ((matrix) -> offdT)
#define jx_ParCSRMatrixColMapOffd(matrix)       ((matrix) -> col_map_offd)
#define jx_ParCSRMatrixRowStarts(matrix)        ((matrix) -> row_starts)
#define jx_ParCSRMatrixColStarts(matrix)        ((matrix) -> col_starts)
#define jx_ParCSRMatrixCommPkg(matrix)          ((matrix) -> comm_pkg)
#define jx_ParCSRMatrixCommPkgT(matrix)         ((matrix) -> comm_pkgT)
#define jx_ParCSRMatrixOwnsData(matrix)         ((matrix) -> owns_data)
#define jx_ParCSRMatrixOwnsRowStarts(matrix)    ((matrix) -> owns_row_starts)
#define jx_ParCSRMatrixOwnsColStarts(matrix)    ((matrix) -> owns_col_starts)
#define jx_ParCSRMatrixNumRows(matrix)          jx_CSRMatrixNumRows(jx_ParCSRMatrixDiag(matrix))
#define jx_ParCSRMatrixNumCols(matrix)          jx_CSRMatrixNumCols(jx_ParCSRMatrixDiag(matrix))
#define jx_ParCSRMatrixNumNonzeros(matrix)      ((matrix) -> num_nonzeros)
#define jx_ParCSRMatrixDNumNonzeros(matrix)     ((matrix) -> d_num_nonzeros)
#define jx_ParCSRMatrixRowindices(matrix)       ((matrix) -> rowindices)
#define jx_ParCSRMatrixRowvalues(matrix)        ((matrix) -> rowvalues)
#define jx_ParCSRMatrixGetrowactive(matrix)     ((matrix) -> getrowactive)
#define jx_ParCSRMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)

/*!
 * \struct jx_ParVector
 */
typedef struct
{
   MPI_Comm	     comm;

   JX_Int      	     global_size;    // global size of the vector
   JX_Int      	     first_index;    // first index(in global sense) of the local vector
   JX_Int               last_index;     // last  index(in global sense) of the local vector
   JX_Int      	    *partitioning;
   jx_Vector        *local_vector; 
   JX_Int      	     owns_data;
   JX_Int      	     owns_partitioning;

   jx_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                           is used (compile-time option) AND this partition
                                           needed(for setting off-proc elements, for example) */
} jx_ParVector;

#define jx_ParVectorComm(vector)             ((vector) -> comm)
#define jx_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define jx_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define jx_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define jx_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define jx_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define jx_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define jx_ParVectorOwnsPartitioning(vector) ((vector) -> owns_partitioning)
#define jx_ParVectorNumVectors(vector)       (jx_VectorNumVectors( jx_ParVectorLocalVector(vector) ))
#define jx_ParVectorAssumedPartition(vector) ((vector) -> assumed_partition)

/*!
 * \struct jx_ParVecCommPkg
 * \brief Communication Package for the swith between two
 *        parallel vectors with different partitionings.
 * \author peghoty
 * \date 2011/09/07
 */
typedef struct
{
   MPI_Comm  comm;
   
   JX_Int       num_sends;
   JX_Int      *send_procs;
   JX_Int      *send_starts;

   JX_Int       num_recvs;
   JX_Int      *recv_procs;
   JX_Int      *recv_starts;

} jx_ParVecCommPkg;

#define jx_ParVecCommPkgComm(comm_pkg)          (comm_pkg -> comm)                               
#define jx_ParVecCommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define jx_ParVecCommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define jx_ParVecCommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define jx_ParVecCommPkgSendStarts(comm_pkg)    (comm_pkg -> send_starts)
#define jx_ParVecCommPkgSendStart(comm_pkg,i)   (comm_pkg -> send_starts[i])
#define jx_ParVecCommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define jx_ParVecCommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define jx_ParVecCommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define jx_ParVecCommPkgRecvStarts(comm_pkg)    (comm_pkg -> recv_starts)
#define jx_ParVecCommPkgRecvStart(comm_pkg,i)   (comm_pkg -> recv_starts[i])

/*!
 * \struct jx_ParVecCommHandle
 * \brief Communication handle for the swith between two
 *        parallel vectors with different partitionings.
 * \author peghoty
 * \date 2011/09/07
 */
typedef struct
{
   jx_ParVecCommPkg     *comm_pkg;
   void 	        *send_data;
   void 	        *recv_data;

   JX_Int                   num_requests;
   MPI_Request          *requests;

} jx_ParVecCommHandle;

#define jx_ParVecCommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define jx_ParVecCommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define jx_ParVecCommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define jx_ParVecCommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define jx_ParVecCommHandleRequests(comm_handle)    (comm_handle -> requests)
#define jx_ParVecCommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

/*!
 * \struct jx_ProcListElements
 */
typedef struct
{
   JX_Int      length;
   JX_Int      storage_length; 
   JX_Int     *id;
   JX_Int     *vec_starts;
   JX_Int      element_storage_length; 
   JX_Int     *elements;
   JX_Real  *d_elements; // ?
   void    *v_elements;
   
}  jx_ProcListElements;

/*!
 * \struct jx_CSRBlockMatrix
 */
typedef struct
{
  JX_Real *data;
  JX_Int  *i;
  JX_Int  *j;
  JX_Int   block_size;
  JX_Int   num_rows;
  JX_Int   num_cols;
  JX_Int   num_nonzeros;
  JX_Int   owns_data;
  
} jx_CSRBlockMatrix;

#define jx_CSRBlockMatrixData(matrix)         ((matrix) -> data)
#define jx_CSRBlockMatrixI(matrix)            ((matrix) -> i)
#define jx_CSRBlockMatrixJ(matrix)            ((matrix) -> j)
#define jx_CSRBlockMatrixBlockSize(matrix)    ((matrix) -> block_size)
#define jx_CSRBlockMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define jx_CSRBlockMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define jx_CSRBlockMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define jx_CSRBlockMatrixOwnsData(matrix)     ((matrix) -> owns_data)

/*!
 * \struct jx_ParCSRBlockMatrix
 */
typedef struct
{
   MPI_Comm	      comm;

   JX_Int     	      global_num_rows;
   JX_Int     	      global_num_cols;
   JX_Int		      first_row_index;
   JX_Int		      first_col_diag;
   JX_Int                last_row_index;
   JX_Int                last_col_diag;

   jx_CSRBlockMatrix *diag;
   jx_CSRBlockMatrix *offd;

   JX_Int               *col_map_offd; 
   JX_Int               *row_starts; 
   JX_Int               *col_starts;

   jx_ParCSRCommPkg  *comm_pkg;
   jx_ParCSRCommPkg  *comm_pkgT;

   JX_Int                owns_data;
   JX_Int                owns_row_starts;
   JX_Int                owns_col_starts;
   JX_Int                num_nonzeros;
   JX_Real             d_num_nonzeros;
   JX_Int               *rowindices;
   JX_Real              *rowvalues;
   JX_Int                getrowactive;

   jx_IJAssumedPart  *assumed_partition; /* only populated if no_global_partition option
                                            is used (compile-time option) */
} jx_ParCSRBlockMatrix;

#define jx_ParCSRBlockMatrixComm(matrix)	     ((matrix)->comm)
#define jx_ParCSRBlockMatrixGlobalNumRows(matrix)    ((matrix)->global_num_rows)
#define jx_ParCSRBlockMatrixGlobalNumCols(matrix)    ((matrix)->global_num_cols)
#define jx_ParCSRBlockMatrixFirstRowIndex(matrix)    ((matrix)->first_row_index)
#define jx_ParCSRBlockMatrixFirstColDiag(matrix)     ((matrix)->first_col_diag)
#define jx_ParCSRBlockMatrixLastRowIndex(matrix)     ((matrix) -> last_row_index)
#define jx_ParCSRBlockMatrixLastColDiag(matrix)      ((matrix) -> last_col_diag)
#define jx_ParCSRBlockMatrixBlockSize(matrix)        ((matrix)->diag->block_size)
#define jx_ParCSRBlockMatrixDiag(matrix)  	     ((matrix) -> diag)
#define jx_ParCSRBlockMatrixOffd(matrix)  	     ((matrix) -> offd)
#define jx_ParCSRBlockMatrixColMapOffd(matrix)       ((matrix) -> col_map_offd)
#define jx_ParCSRBlockMatrixRowStarts(matrix)        ((matrix) -> row_starts)
#define jx_ParCSRBlockMatrixColStarts(matrix)        ((matrix) -> col_starts)
#define jx_ParCSRBlockMatrixCommPkg(matrix)	     ((matrix) -> comm_pkg)
#define jx_ParCSRBlockMatrixCommPkgT(matrix)	     ((matrix) -> comm_pkgT)
#define jx_ParCSRBlockMatrixOwnsData(matrix)         ((matrix) -> owns_data)
#define jx_ParCSRBlockMatrixOwnsRowStarts(matrix)    ((matrix) -> owns_row_starts)
#define jx_ParCSRBlockMatrixOwnsColStarts(matrix)    ((matrix) -> owns_col_starts)
#define jx_ParCSRBlockMatrixNumRows(matrix)          jx_CSRBlockMatrixNumRows(jx_ParCSRBlockMatrixDiag(matrix))
#define jx_ParCSRBlockMatrixNumCols(matrix)          jx_CSRBlockMatrixNumCols(jx_ParCSRBlockMatrixDiag(matrix))
#define jx_ParCSRBlockMatrixNumNonzeros(matrix)      ((matrix) -> num_nonzeros)
#define jx_ParCSRBlockMatrixDNumNonzeros(matrix)     ((matrix) -> d_num_nonzeros)
#define jx_ParCSRBlockMatrixRowindices(matrix)       ((matrix) -> rowindices)
#define jx_ParCSRBlockMatrixRowvalues(matrix)        ((matrix) -> rowvalues)
#define jx_ParCSRBlockMatrixGetrowactive(matrix)     ((matrix) -> getrowactive)
#define jx_ParCSRBlockMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)


/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/
 
/* csrc/mat_vec_data/seq_csr_matrix.c */  
jx_CSRMatrix *jx_CSRMatrixCreate( JX_Int num_rows, JX_Int num_cols, JX_Int num_nonzeros );
JX_Int jx_CSRMatrixInitialize( jx_CSRMatrix *matrix );
JX_Int jx_CSRMatrixDestroy( jx_CSRMatrix *matrix );
JX_Int jx_CSRMatrixPrint( jx_CSRMatrix *matrix, char *file_name );
jx_CSRMatrix *jx_CSRMatrixRead( char *file_name, JX_Int file_base );
jx_CSRMatrix *jx_CSRMatrixRead2( char *file_name, JX_Int file_base );
JX_Int jx_CSRMatrixCopy( jx_CSRMatrix *A, jx_CSRMatrix *B, JX_Int copy_data );
JX_Int jx_CSRMatrixGetBandWidth( jx_CSRMatrix *A, JX_Int *nbl_ptr, JX_Int *nbr_ptr );
JX_Int jx_CSRMatrixSetRownnz( jx_CSRMatrix *matrix );

/* csrc/mat_vec_data/seq_csr_matop.c */
JX_Int jx_CSRMatrixReorder(jx_CSRMatrix *A);
JX_Int jx_CSRMatrixTranspose( jx_CSRMatrix *A, jx_CSRMatrix **AT, JX_Int data );
void jx_CSRMatrixReorderColumnNumber12( jx_CSRMatrix *A );
void jx_CSRMatrixReorderColumnNumberAll( jx_CSRMatrix *A );
jx_CSRMatrix *jx_CSRMatrixAdd( jx_CSRMatrix *A, jx_CSRMatrix *B );
jx_CSRMatrix *jx_CSRMatrixMultiply( jx_CSRMatrix *A, jx_CSRMatrix *B );
void jx_CSRMatrixScale( jx_CSRMatrix *A, JX_Real alpha );
jx_CSRMatrix *jx_CSRMatrixSymmetrization( jx_CSRMatrix *A );
jx_CSRMatrix *jx_CSRMatrixMGReorderByVariables( jx_CSRMatrix *A, JX_Int num_groups );

/* csrc/mat_vec_data/seq_vector.c */
jx_Vector *jx_SeqVectorCreate( JX_Int size );
JX_Int jx_SeqVectorInitialize( jx_Vector *vector );
JX_Int jx_SeqVectorSetDataOwner( jx_Vector *vector, JX_Int owns_data );
JX_Int jx_SeqVectorDestroy( jx_Vector *vector );
JX_Int jx_SeqVectorCopy( jx_Vector *x, jx_Vector *y );
JX_Int jx_SeqVectorSetConstantValues( jx_Vector *x, JX_Real value );
JX_Real jx_SeqVectorInnerProd( jx_Vector *x, jx_Vector *y );
JX_Int jx_SeqVectorPrint( jx_Vector *vector, char *file_name );
JX_Int jx_SeqVectorAxpy( JX_Real alpha, jx_Vector *x, jx_Vector *y );
JX_Int jx_SeqVectorMassDotpTwo( jx_Vector *x, jx_Vector *y, jx_Vector **z, JX_Int k, JX_Int unroll, JX_Real *result_x, JX_Real *result_y );
JX_Int jx_SeqVectorMassDotpTwo8( jx_Vector *x, jx_Vector *y, jx_Vector **z, JX_Int k, JX_Real *result_x, JX_Real *result_y );
JX_Int jx_SeqVectorMassDotpTwo4( jx_Vector *x, jx_Vector *y, jx_Vector **z, JX_Int k, JX_Real *result_x, JX_Real *result_y );
JX_Int jx_SeqVectorMassAxpy( JX_Real *alpha, jx_Vector **x, jx_Vector *y, JX_Int k, JX_Int unroll );
JX_Int jx_SeqVectorMassAxpy8( JX_Real *alpha, jx_Vector **x, jx_Vector *y, JX_Int k );
JX_Int jx_SeqVectorMassAxpy4( JX_Real *alpha, jx_Vector **x, jx_Vector *y, JX_Int k );
JX_Int jx_SeqVectorMassInnerProd( jx_Vector *x, jx_Vector **y, JX_Int k, JX_Int unroll, JX_Real *result );
JX_Int jx_SeqVectorMassInnerProd8( jx_Vector *x, jx_Vector **y, JX_Int k, JX_Real *result );
JX_Int jx_SeqVectorMassInnerProd4( jx_Vector *x, jx_Vector **y, JX_Int k, JX_Real *result );
jx_Vector *jx_SeqVectorMGReorderByVariables( jx_Vector *x, JX_Int num_groups );
jx_Vector *jx_SeqVectorRead( char *file_name );
jx_Vector *jx_SeqVectorRead2( char *file_name );
jx_Vector *jx_SeqMultiVectorCreate( JX_Int size, JX_Int num_vectors );
JX_Int jx_SeqVectorScale( JX_Real alpha, jx_Vector *y );
JX_Real jx_SeqVectorNorm1( jx_Vector *x );
JX_Real jx_SeqVectorPointWiseRelNorm1( jx_Vector *x, jx_Vector *y );
JX_Int jx_SeqVecZXY( jx_Vector *z, JX_Real alpha, jx_Vector *x, jx_Vector *y );
JX_Int jx_SeqVecMul( jx_Vector *x, jx_Vector *y, jx_Vector *z );

/* csrc/mat_vec_data/seq_csr_matvec.c */
JX_Int
jx_CSRMatrixMatvec( JX_Real          alpha,
                    jx_CSRMatrix   *A,
                    jx_Vector      *x,
                    JX_Real          beta,
                    jx_Vector      *y    );              
JX_Int
jx_CSRMatrixMatvecT( JX_Real        alpha,
                     jx_CSRMatrix *A,
                     jx_Vector    *x,
                     JX_Real        beta,
                     jx_Vector    *y     );
                     
/* csrc/mat_vec_data/seq_csr_ls.c */
void
jx_ReadLinearSystemData( char    *MatFile, 
                         char    *RhsFile,
                         JX_Int    **IA_ptr,
                         JX_Int    **JA_ptr,
                         JX_Real **AA_ptr,
                         JX_Real **F_ptr,
                         JX_Int     *n_ptr  );
void
jx_InitializeLinearSystemStruct( jx_CSRMatrix **matrix_ptr, 
                                 jx_Vector    **vectorF_ptr, 
                                 jx_Vector    **vectorX_ptr );
void
jx_FillData2Struct( JX_Int              n, 
                    JX_Int             *ia, 
                    JX_Int             *ja, 
                    JX_Real          *aa, 
                    JX_Real          *f, 
                    JX_Real          *x, 
                    jx_CSRMatrix    *matrix, 
                    jx_Vector       *vectorF, 
                    jx_Vector       *vectorX  );

/* csrc/mat_vec_data/par_csr_matop.c */
JX_Int jx_ParCSRMatrixTranspose( jx_ParCSRMatrix *A, jx_ParCSRMatrix **AT_ptr, JX_Int data );
jx_CSRMatrix *jx_ParCSRMatrixExtractBExt( jx_ParCSRMatrix *B, jx_ParCSRMatrix *A, JX_Int data );
void 
jx_ParCSRMatrixExtractBExt_Arrays( JX_Int             **pB_ext_i, 
                                   JX_Int             **pB_ext_j, 
                                   JX_Real          **pB_ext_data, 
                                   JX_Int             **pB_ext_row_map,
                                   JX_Int              *num_nonzeros,
                                   JX_Int               data, 
                                   JX_Int               find_row_map, 
                                   MPI_Comm          comm, 
                                   jx_ParCSRCommPkg *comm_pkg,
                                   JX_Int               num_cols_B, 
                                   JX_Int               num_recvs, 
                                   JX_Int               num_sends,
                                   JX_Int               first_col_diag, 
                                   JX_Int               first_row_index,
                                   JX_Int              *recv_vec_starts, 
                                   JX_Int              *send_map_starts, 
                                   JX_Int              *send_map_elmts,
                                   JX_Int              *diag_i, 
                                   JX_Int              *diag_j, 
                                   JX_Int              *offd_i, 
                                   JX_Int              *offd_j, 
                                   JX_Int              *col_map_offd,
                                   JX_Real           *diag_data, 
                                   JX_Real           *offd_data );
jx_ParCSRMatrix *
jx_ParMatmul( jx_ParCSRMatrix *A, jx_ParCSRMatrix *B );
jx_ParCSRMatrix *
jx_ParTMatmul( jx_ParCSRMatrix *A, jx_ParCSRMatrix *B );
void
jx_ParMatmul_RowSizes( JX_Int **C_diag_i,
                       JX_Int **C_offd_i,
                       JX_Int  *A_diag_i,
                       JX_Int  *A_diag_j,
                       JX_Int  *A_offd_i,
                       JX_Int  *A_offd_j,
                       JX_Int  *B_diag_i,
                       JX_Int  *B_diag_j,
                       JX_Int  *B_offd_i,
                       JX_Int  *B_offd_j,
                       JX_Int  *B_ext_diag_i,
                       JX_Int  *B_ext_diag_j, 
                       JX_Int  *B_ext_offd_i,
                       JX_Int  *B_ext_offd_j,
                       JX_Int  *map_B_to_C,
                       JX_Int  *C_diag_size,
                       JX_Int  *C_offd_size,
                       JX_Int   num_rows_diag_A,
                       JX_Int   num_cols_offd_A,
                       JX_Int   allsquare,
                       JX_Int   num_cols_diag_B,
                       JX_Int   num_cols_offd_B,
                       JX_Int   num_cols_offd_C );

/* csrc/mat_vec_data/par_csr_matrix.c */
jx_ParCSRMatrix *jx_BuildMatParFromOneFile( char *filename, JX_Int num_functions, JX_Int file_base );
jx_ParCSRMatrix *jx_BuildMatParFromOneFile5( char *filename, JX_Int num_functions, JX_Int file_base );
jx_ParCSRMatrix *
jx_ParCSRMatrixCreate( MPI_Comm comm,
                       JX_Int global_num_rows,
                       JX_Int global_num_cols,
                       JX_Int *row_starts,
                       JX_Int *col_starts,
                       JX_Int num_cols_offd,
                       JX_Int num_nonzeros_diag,
                       JX_Int num_nonzeros_offd );
JX_Int jx_ParCSRMatrixInitialize( jx_ParCSRMatrix *matrix );
jx_CSRMatrix *jx_ParCSRMatrixToCSRMatrixAll(jx_ParCSRMatrix *par_matrix);
jx_ParCSRMatrix *jx_CSRMatrixToParCSRMatrix( MPI_Comm comm, jx_CSRMatrix *A, JX_Int *row_starts, JX_Int *col_starts );
jx_ParCSRMatrix *
jx_CSRMatrixToParCSRMatrix_FromGivenPro( MPI_Comm       comm,
                                         JX_Int            srcid,
                                         jx_CSRMatrix  *A,
                                         JX_Int           *row_starts,
                                         JX_Int           *col_starts );
jx_ParCSRMatrix *jx_CSRMatrixToParCSRMatrix_sp( MPI_Comm comm, jx_CSRMatrix *A );
JX_Int jx_GenerateDiagAndOffd( jx_CSRMatrix *A, jx_ParCSRMatrix *matrix, JX_Int first_col_diag, JX_Int last_col_diag );
jx_CSRMatrix *jx_MergeDiagAndOffd( jx_ParCSRMatrix *par_matrix );
jx_ParCSRMatrix *jx_ParCSRMatrixRead( MPI_Comm comm, const char *file_name, JX_Int file_base );
JX_Int jx_ParCSRMatrixPrint( jx_ParCSRMatrix *matrix, const char *file_name );
JX_Int jx_ParCSRMatrixCopy( jx_ParCSRMatrix *A, jx_ParCSRMatrix *B, JX_Int copy_data );

JX_Int jx_ParCSRMatrixDestroy( jx_ParCSRMatrix *matrix );
JX_Int 
jx_ParCSRMatrixGetLocalRange( jx_ParCSRMatrix *matrix, 
                              JX_Int             *row_start, 
                              JX_Int             *row_end, 
                              JX_Int             *col_start, 
                              JX_Int             *col_end );
JX_Int
JX_ParCSRMatrixGetRow( JX_ParCSRMatrix  matrix,
                       JX_Int              row,
                       JX_Int             *size,
                       JX_Int            **col_ind,
                       JX_Real         **values );
JX_Int 
jx_ParCSRMatrixGetRow( jx_ParCSRMatrix *matrix,
                       JX_Int row,
                       JX_Int *size,
                       JX_Int **col_ind,
                       JX_Real **values );
JX_Int
JX_ParCSRMatrixRestoreRow( JX_ParCSRMatrix  matrix,
                           JX_Int              row,
                           JX_Int             *size,
                           JX_Int            **col_ind,
                           JX_Real         **values );
JX_Int 
jx_ParCSRMatrixRestoreRow( jx_ParCSRMatrix *matrix,
                           JX_Int row,
                           JX_Int *size,
                           JX_Int **col_ind,
                           JX_Real **values );
JX_Int jx_ParCSRMatrixCreateAssumedPartition( jx_ParCSRMatrix *matrix );
JX_Int 
jx_LocateAssummedPartition( MPI_Comm          comm,
                            JX_Int               row_start,
                            JX_Int               row_end,
                            JX_Int               global_first_row,
                            JX_Int               global_num_rows,
                            jx_IJAssumedPart *part,
                            JX_Int               myid );
JX_Int
jx_GetAssumedPartitionProcFromRow( MPI_Comm comm,
                                   JX_Int      row,
                                   JX_Int      global_first_row,
                                   JX_Int      global_num_rows,
                                   JX_Int     *proc_id );
JX_Int
jx_FillResponseParToCSRMatrix( void      *p_recv_contact_buf, 
                               JX_Int        contact_size, 
                               JX_Int        contact_proc, 
                               void      *ro, 
                               MPI_Comm   comm, 
                               void     **p_send_response_buf, 
                               JX_Int       *response_message_size );
JX_Int jx_ParCSRMatrixDestroyAssumedPartition( jx_ParCSRMatrix *matrix );
JX_Int jx_ParCSRMatrixGetRowPartitioning( jx_ParCSRMatrix *matrix, JX_Int **row_partitioning_ptr );
JX_Int jx_ParCSRMatrixGetColPartitioning( jx_ParCSRMatrix *matrix, JX_Int **col_partitioning_ptr );
JX_Int jx_ParCSRMatrixSetNumNonzeros( jx_ParCSRMatrix *matrix );
JX_Int jx_ParCSRMatrixSetDNumNonzeros( jx_ParCSRMatrix *matrix );
JX_Int jx_ParCSRMatrixSetDataOwner( jx_ParCSRMatrix *matrix, JX_Int owns_data );
JX_Int jx_ParCSRMatrixSetRowStartsOwner( jx_ParCSRMatrix *matrix, JX_Int owns_row_starts );
JX_Int jx_ParCSRMatrixSetColStartsOwner( jx_ParCSRMatrix *matrix, JX_Int owns_col_starts );
JX_Int
jx_BuildCSRMatrixMPIDataType( JX_Int           num_nonzeros, 
                              JX_Int           num_rows,
                              JX_Real       *a_data, 
                              JX_Int          *a_i, 
                              JX_Int          *a_j, 
                              MPI_Datatype *csr_matrix_datatype );
JX_Int      
jx_Build_Par_LinearSystem( MPI_Comm          comm, 
                           char             *filenamemat, 
                           char             *filenamerhs, 
                           jx_ParCSRMatrix **par_matrix_ptr,
                           jx_ParVector    **par_rhs_ptr );                        
JX_Int jx_PutDiagFirst( JX_Int *ia, JX_Int *ja, JX_Real *a, JX_Int n );
JX_Int jx_ParCSRMatrixGetDims( jx_ParCSRMatrix *A, JX_Int *M, JX_Int *N );
JX_Int jx_ParCSRMatrixGetComm( jx_ParCSRMatrix *matrix, MPI_Comm *comm );

/* csrc/mat_vec_data/par_vector.c */
jx_ParVector *jx_BuildRhsParFromOneFile( char *filename, jx_ParCSRMatrix *A );
jx_ParVector *jx_BuildRhsParFromOneFile3( char *filename, jx_ParCSRMatrix *A );
jx_ParVector *jx_ParVectorCreate( MPI_Comm comm, JX_Int global_size, JX_Int *partitioning );
JX_Int jx_ParVectorInitialize( jx_ParVector *vector );
JX_Int jx_ParVectorCopy( jx_ParVector *x, jx_ParVector *y );
JX_Int jx_ParVectorDestroy( jx_ParVector *vector );
JX_Int jx_ParVectorDestroyAssumedPartition( jx_ParVector *vector );
JX_Int jx_ParVectorSetConstantValues( jx_ParVector *v, JX_Real value );
JX_Real jx_ParVectorInnerProd( jx_ParVector *x, jx_ParVector *y );
JX_Int jx_ParVectorMassInnerProd( jx_ParVector *x, jx_ParVector **y, JX_Int k, JX_Int unroll, JX_Real *result );
JX_Real jx_ParVectorNorm1( jx_ParVector *x );
JX_Real jx_ParVectorNorm2( jx_ParVector *x );
JX_Real jx_ParVectorPointWiseRelNorm1( jx_ParVector *x, jx_ParVector *y );
JX_Int jx_ParVectorAxpy( JX_Real alpha, jx_ParVector *x, jx_ParVector *y  );
JX_Int jx_ParVectorMassDotpTwo( jx_ParVector *x, jx_ParVector *y, jx_ParVector **z, JX_Int k, JX_Int unroll, JX_Real *result_x, JX_Real *result_y );
JX_Int jx_ParVectorMassAxpy( JX_Real *alpha, jx_ParVector **x, jx_ParVector *y, JX_Int k, JX_Int unroll );
JX_Int jx_ParVectorScale( JX_Real alpha, jx_ParVector *y );
jx_Vector *jx_ParVectorToVectorAll (jx_ParVector *par_v);
JX_Int jx_ParVectorToVector_Alloctaed( jx_ParVector *par_v, jx_Vector *vector );
JX_Int jx_ParVectorPrint( jx_ParVector *vector, const char *file_name );
jx_ParVector *jx_ParVectorRead( MPI_Comm comm, const char *file_name );
JX_Int jx_GeneratePartitioning( JX_Int length, JX_Int num_procs, JX_Int **part_ptr );
JX_Int jx_GenerateLocalPartitioning( JX_Int length, JX_Int num_procs, JX_Int myid, JX_Int **part_ptr );
JX_Int
jx_FillResponseParToVectorAll( void      *p_recv_contact_buf, 
                               JX_Int        contact_size, 
                               JX_Int        contact_proc, 
                               void      *ro, 
                               MPI_Comm   comm, 
                               void     **p_send_response_buf, 
                               JX_Int       *response_message_size );
jx_ParVector *jx_VectorToParVector( MPI_Comm comm, jx_Vector *v, JX_Int *vec_starts );
jx_ParVector *jx_VectorToParVector_FromGivenPro( MPI_Comm comm, JX_Int srcid, jx_Vector *v, JX_Int *vec_starts );
jx_ParVector *jx_VectorToParVector_sp( MPI_Comm comm, jx_Vector *vector );
JX_Int
jx_VectorToParVector_Allocated( MPI_Comm       comm, 
                                jx_Vector     *v, 
                                JX_Int           *vec_starts, 
                                jx_ParVector  *par_vector );
JX_Int
jx_VectorToParVector_Allocated2( MPI_Comm       comm, 
                                 jx_Vector     *v, 
                                 JX_Int           *vec_starts, 
                                 jx_ParVector  *par_vector );
JX_Int
jx_VectorToParVector_Allocated_FromGivenPro( MPI_Comm       comm, 
                                             JX_Int            id,
                                             jx_Vector     *v, 
                                             JX_Int           *vec_starts, 
                                             jx_ParVector  *par_vector );                               
jx_ParVector *jx_ParMultiVectorCreate( MPI_Comm  comm, JX_Int global_size, JX_Int *partitioning, JX_Int num_vectors );
JX_Int jx_ParVectorSetDataOwner( jx_ParVector *vector, JX_Int owns_data );
JX_Int jx_ParVectorSetPartitioningOwner( jx_ParVector *vector, JX_Int owns_partitioning );
jx_ParVecCommPkg *jx_ParVecCommPkgCreate( MPI_Comm comm, JX_Int *px, JX_Int *py );
jx_ParVecCommHandle *
jx_ParVecCommHandleCreate( JX_Int                job, 
                           jx_ParVecCommPkg  *comm_pkg, 
                           void              *send_data, 
                           void              *recv_data );
JX_Int jx_ParVecCommHandleDestroy( jx_ParVecCommHandle *comm_handle );
JX_Int jx_ParVecCommPkgDestroy( jx_ParVecCommPkg *comm_pkg );
JX_Int 
jx_ParVecSwitch( jx_ParVecCommPkg  *comm_pkg,
                 JX_Int                job,
                 jx_ParVector      *x,
                 jx_ParVector      *y );
JX_Int jx_ParVecMul( jx_ParVector *x, jx_ParVector *y, jx_ParVector *z );
JX_Int jx_ParVecZXY( jx_ParVector *z, JX_Real alpha, jx_ParVector *x, jx_ParVector *y );
                         
/* csrc/mat_vec_data/par_csr_matvec.c */
JX_Int
jx_ParCSRMatrixMatvec( JX_Real           alpha,
              	       jx_ParCSRMatrix *A,
                       jx_ParVector    *x,
                       JX_Real           beta,
                       jx_ParVector    *y  );
JX_Int
jx_ParCSRMatrixMatvecT( JX_Real           alpha,
                        jx_ParCSRMatrix *A,
                        jx_ParVector    *x,
                        JX_Real           beta,
                        jx_ParVector    *y  );

/* csrc/mat_vec_data/par_csr_com.c */ 
JX_Int jx_MatvecCommPkgCreate ( jx_ParCSRMatrix *par_A );
jx_ParCSRCommHandle *
jx_ParCSRCommHandleCreate ( JX_Int 	      job,
			    jx_ParCSRCommPkg *comm_pkg,
                            void             *send_data, 
                            void             *recv_data );
JX_Int jx_ParCSRCommHandleDestroy( jx_ParCSRCommHandle *comm_handle );
void
jx_MatvecCommPkgCreate_core ( MPI_Comm  comm, 
                              JX_Int      *col_map_offd, 
                              JX_Int       first_col_diag, 
                              JX_Int      *col_starts,
                              JX_Int       num_cols_diag, 
                              JX_Int       num_cols_offd,
                              JX_Int       firstColDiag, 
                              JX_Int      *colMapOffd,
                              JX_Int       data,  
                              JX_Int      *p_num_recvs, 
                              JX_Int     **p_recv_procs, 
                              JX_Int     **p_recv_vec_starts,
                              JX_Int      *p_num_sends, 
                              JX_Int     **p_send_procs, 
                              JX_Int     **p_send_map_starts,
                              JX_Int     **p_send_map_elmts );
JX_Int jx_MatvecCommPkgDestroy( jx_ParCSRCommPkg *comm_pkg );

/* csrc/mat_vec_data/new_commpkg.c */ 
JX_Int 
jx_NewCommPkgCreate_core( MPI_Comm           comm, 
                          JX_Int               *col_map_off_d, 
                          JX_Int                first_col_diag,
                          JX_Int                col_start, 
                          JX_Int                col_end, 
                          JX_Int                num_cols_off_d, 
                          JX_Int                global_num_cols,
                          JX_Int               *p_num_recvs, 
                          JX_Int              **p_recv_procs, 
                          JX_Int              **p_recv_vec_starts,
                          JX_Int               *p_num_sends, 
                          JX_Int              **p_send_procs, 
                          JX_Int              **p_send_map_starts,
                          JX_Int              **p_send_map_elements, 
                          jx_IJAssumedPart  *apart );
JX_Int
jx_RangeFillResponseIJDetermineRecvProcs( void      *p_recv_contact_buf,
                                          JX_Int        contact_size, 
                                          JX_Int        contact_proc, 
                                          void      *ro, 
                                          MPI_Comm   comm, 
                                          void     **p_send_response_buf, 
                                          JX_Int       *response_message_size );
JX_Int
jx_FillResponseIJDetermineSendProcs( void      *p_recv_contact_buf, 
                                     JX_Int        contact_size, 
                                     JX_Int        contact_proc, 
                                     void      *ro, 
                                     MPI_Comm   comm, 
                                     void     **p_send_response_buf, 
                                     JX_Int       *response_message_size );
JX_Int
jx_GetAssumedPartitionRowRange( MPI_Comm comm,
                                JX_Int  proc_id,
                                JX_Int  global_first_row,
                                JX_Int  global_num_rows,
                                JX_Int *row_start,
                                JX_Int *row_end );

/* csrc/mat_vec_data/ij_matrix.c */
JX_Int
JX_IJMatrixCreate( MPI_Comm     comm,
                   JX_Int          ilower,
                   JX_Int          iupper,
                   JX_Int          jlower,
                   JX_Int          jupper,
                   JX_IJMatrix *matrix );
JX_Int
jx_IJMatrixCreateParCSR( jx_IJMatrix *matrix );
JX_Int
jx_AuxParCSRMatrixCreate( jx_AuxParCSRMatrix **aux_matrix,
			  JX_Int  local_num_rows,
                       	  JX_Int  local_num_cols,
			  JX_Int *sizes );
JX_Int
JX_IJMatrixInitialize( JX_IJMatrix matrix );
JX_Int
jx_IJMatrixInitializeParCSR( jx_IJMatrix *matrix );
JX_Int
jx_AuxParCSRMatrixInitialize( jx_AuxParCSRMatrix *matrix );
JX_Int
JX_IJMatrixDestroy( JX_IJMatrix matrix );
JX_Int
jx_IJMatrixDestroyParCSR( jx_IJMatrix *matrix );
JX_Int
jx_AuxParCSRMatrixDestroy( jx_AuxParCSRMatrix *matrix );
JX_Int
jx_AssumedPartitionDestroy( jx_IJAssumedPart *apart );
JX_Int
JX_IJMatrixSetObjectType( JX_IJMatrix matrix, JX_Int type );
JX_Int
JX_IJMatrixSetValues( JX_IJMatrix    matrix,
                      JX_Int            nrows,
                      JX_Int           *ncols,
                      const JX_Int     *rows,
                      const JX_Int     *cols,
                      const JX_Real  *values );
JX_Int
jx_IJMatrixSetValuesOMPParCSR( jx_IJMatrix  *matrix,
                               JX_Int           nrows,
                               JX_Int          *ncols,
                               const JX_Int    *rows,
                               const JX_Int    *cols,
                               const JX_Real *values );
JX_Int
jx_IJMatrixSetValuesParCSR( jx_IJMatrix  *matrix,
                            JX_Int           nrows,
                            JX_Int          *ncols,
                            const JX_Int    *rows,
                            const JX_Int    *cols,
                            const JX_Real *values );
JX_Int
JX_IJMatrixAssemble( JX_IJMatrix matrix );
JX_Int
jx_IJMatrixAssembleParCSR( jx_IJMatrix *matrix );
JX_Int
jx_IJMatrixAssembleOffProcValsParCSR( jx_IJMatrix *matrix,
                                      JX_Int          off_proc_i_indx,
                                      JX_Int          max_off_proc_elmts,
                                      JX_Int          current_num_elmts,
                                      JX_Int         *off_proc_i,
                                      JX_Int         *off_proc_j,
                                      JX_Real      *off_proc_data );
JX_Int
jx_IJMatrixCreateAssumedPartition( jx_IJMatrix *matrix );
JX_Int
jx_FillResponseIJOffProcVals( void      *p_recv_contact_buf, 
                              JX_Int        contact_size,
                              JX_Int        contact_proc,
                              void      *ro,
                              MPI_Comm   comm,
                              void     **p_send_response_buf,
                              JX_Int       *response_message_size );
JX_Int
jx_IJMatrixAddToValuesParCSR( jx_IJMatrix   *matrix,
                              JX_Int            nrows,
                              JX_Int           *ncols,
                              const JX_Int     *rows,
                              const JX_Int     *cols,
                              const JX_Real  *values );
JX_Int
JX_IJMatrixAddToValues( JX_IJMatrix   matrix,
                        JX_Int           nrows,
                        JX_Int          *ncols,
                        const JX_Int    *rows,
                        const JX_Int    *cols,
                        const JX_Real *values );
JX_Int
JX_IJMatrixRead( const char  *filename,
                 MPI_Comm     comm,
                 JX_Int          type,
		 JX_IJMatrix *matrix_ptr );
JX_Int
JX_IJMatrixPrint( JX_IJMatrix matrix, const char *filename );
JX_Int
jx_IJMatrixAddToValuesOMPParCSR( jx_IJMatrix  *matrix,
                                 JX_Int           nrows,
                                 JX_Int          *ncols,
                                 const JX_Int    *rows,
                                 const JX_Int    *cols,
                                 const JX_Real *values );
JX_Int
JX_IJMatrixGetObject( JX_IJMatrix matrix, void **object );

/* csrc/mat_vec_data/ij_vector.c */
JX_Int
JX_IJVectorCreate( MPI_Comm     comm,
                   JX_Int          jlower,
                   JX_Int          jupper,
                   JX_IJVector *vector );
JX_Int
jx_IJVectorCreatePar( jx_IJVector *vector, JX_Int *IJpartitioning );
JX_Int
jx_AuxParVectorCreate( jx_AuxParVector **aux_vector );
JX_Int
JX_IJVectorInitialize( JX_IJVector vector );
JX_Int
jx_IJVectorInitializePar( jx_IJVector *vector );
JX_Int
jx_AuxParVectorInitialize( jx_AuxParVector *vector );
JX_Int
JX_IJVectorDestroy( JX_IJVector vector );
JX_Int
jx_IJVectorDestroyPar( jx_IJVector *vector );
JX_Int
jx_AuxParVectorDestroy( jx_AuxParVector *vector );
JX_Int
JX_IJVectorSetObjectType( JX_IJVector vector, JX_Int type );
JX_Int
JX_IJVectorSetValues( JX_IJVector   vector,
                      JX_Int           nvalues,
                      const JX_Int    *indices,
                      const JX_Real *values );
JX_Int
jx_IJVectorSetValuesPar( jx_IJVector  *vector,
                         JX_Int           num_values,
                         const JX_Int    *indices,
                         const JX_Real *values );
JX_Int
JX_IJVectorAssemble( JX_IJVector vector );
JX_Int
jx_IJVectorAssemblePar( jx_IJVector *vector );
JX_Int
jx_IJVectorAssembleOffProcValsPar( jx_IJVector *vector, 
   				   JX_Int          max_off_proc_elmts,
   				   JX_Int          current_num_elmts,
   				   JX_Int         *off_proc_i,
   			     	   JX_Real      *off_proc_data );
JX_Int
jx_IJVectorCreateAssumedPartition( jx_IJVector *vector );
JX_Int
jx_FindProc( JX_Int *list, JX_Int value, JX_Int list_length );
JX_Int
JX_IJVectorGetValues( JX_IJVector  vector,
                      JX_Int          nvalues,
                      const JX_Int   *indices,
                      JX_Real      *values );
JX_Int
jx_IJVectorGetValuesPar( jx_IJVector *vector,
                         JX_Int          num_values,
                         const JX_Int   *indices,
                         JX_Real      *values );
JX_Int
JX_IJVectorAddToValues( JX_IJVector    vector,
                        JX_Int            nvalues,
                        const JX_Int     *indices,
                        const JX_Real  *values );
JX_Int
JX_IJVectorRead( const char  *filename,
                 MPI_Comm     comm,
                 JX_Int          type,
                 JX_IJVector *vector_ptr );
JX_Int
JX_IJVectorPrint( JX_IJVector vector, const char *filename );
JX_Int
jx_IJVectorAddToValuesPar( jx_IJVector   *vector,
                           JX_Int            num_values,
                           const JX_Int     *indices,
                           const JX_Real  *values );

/* csrc/operation/matops/par_gs.c */
JX_Int
jx_gselim(JX_Real *A_matrix, JX_Real *x_vector, JX_Int size);

/* csrc/operation/matops/par_gs_piv.c */
JX_Int
jx_gselim_piv(JX_Real *A_matrix, JX_Real *x_vector, JX_Int size);

/* csrc/operation/matops/par_rap.c */
jx_CSRMatrix *
jx_ExchangeRAPData( jx_CSRMatrix *RAP_int, jx_ParCSRCommPkg *comm_pkg_RT );
JX_Int
jx_PAMGBuildCoarseOperator( jx_ParCSRMatrix  *RT,
                            jx_ParCSRMatrix  *par_A,
                            jx_ParCSRMatrix  *par_P,
                            jx_ParCSRMatrix **RAP_ptr );
JX_Int
jx_PAMGBuildCoarseOperatorKT( jx_ParCSRMatrix  *RT,
                              jx_ParCSRMatrix  *par_A,
                              jx_ParCSRMatrix  *par_P,
                              JX_Int            keepTranspose,
                              jx_ParCSRMatrix **RAP_ptr );

/* csrc/operation/matops/par_rap_omp.c */
JX_Int
jx_PAMGBuildCoarseOperatorOMP( jx_ParCSRMatrix  *RT,
                               jx_ParCSRMatrix  *par_A,
                               jx_ParCSRMatrix  *par_P,
                               jx_ParCSRMatrix **RAP_ptr,
                               JX_Int              *icor_yoo );

/* csrc/operation/relaxations/par_relax_0.c */
JX_Int
jx_PAMGRelax0( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int
jx_PAMGRelaxAI0( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_1.c */
JX_Int
jx_PAMGRelax1( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI1( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_2.c */
JX_Int
jx_PAMGRelax2( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI2( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_3.c */
JX_Int
jx_PAMGRelax3( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI3( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_4.c */
JX_Int
jx_PAMGRelax4( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI4( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_5.c */
JX_Int
jx_PAMGRelax5( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI5( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_6.c */
JX_Int
jx_PAMGRelax6( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI6( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_7.c */
JX_Int
jx_PAMGRelax7( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );
JX_Int  
jx_PAMGRelaxAI7( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp );

/* csrc/operation/relaxations/par_relax_9.c */
JX_Int
jx_PAMGRelax9( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp );

#endif
