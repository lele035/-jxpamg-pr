//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_ilu.h -- head files for ILU Preconditioner
 *  Date: 2014/03/24
 *
 *  Created by Yue Xiaoqiang
 */

#ifndef JX_ILU_HEADER
#define JX_ILU_HEADER
 
#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif

#ifndef JX_MV_HEADER 
#include "jx_mv.h"
#endif

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jx_GridPartitionData
 */
typedef struct
{
    MPI_Comm comm;
    
    JX_Int part_type;
    JX_Int x_part_len;
    JX_Int y_part_len;
    JX_Int x_lower_idx;
    JX_Int x_upper_idx;
    JX_Int y_lower_idx;
    JX_Int y_upper_idx;
    JX_Int num_sideproc;
    JX_Int num_smallside;
    JX_Int num_largeside;
    JX_Int num_smallcross;
    JX_Int num_largecross;
    JX_Int num_nocrossside;
    
    JX_Int *xlo_array;
    JX_Int *xup_array;
    JX_Int *ylo_array;
    JX_Int *yup_array;
    JX_Int *sideprocs;
    JX_Int *sideprcpos;
    JX_Int *sideprcxsrt;
    JX_Int *sideprcysrt;
    JX_Int *sideprclength;
    
} jx_GridPartitionData;

/*!
 * \struct jx_ILUZeroFactorData
 */
typedef struct
{
    MPI_Comm comm;
    
    JX_Int nx;
    JX_Int ny;
    JX_Int npx;
    JX_Int npy;
    JX_Int num_equns;
    
    JX_Int ex_len;
    JX_Int ey_len;
    JX_Int dx_len;
    JX_Int kx_len;
    JX_Int dy_len;
    JX_Int lx_len;
    JX_Int ly_len;
    JX_Int length;
    JX_Int max_iter;
    JX_Int pos_int_end;
    JX_Int pos_dwn_end;
    JX_Int pos_lft_end;
    JX_Int num_fill_in_drop;
    
    JX_Real drop_tol;
    
    JX_Int *index;
    JX_Int *indexA;
    JX_Int *indexD;
    JX_Int *permute;
    
    JX_Real *value;
    JX_Real *senddown;
    
    MPI_Status *status;
    
    jx_CSRMatrix *matA;
    
    jx_Vector *aux_vec;
    jx_Vector *res_vec;
    jx_Vector *tmp_vec;
    
    jx_ParVector *par_aux_vec;
    jx_ParVector *par_res_vec;
    
    jx_GridPartitionData *par_grid;
    
} jx_ILUZeroFactorData;


/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/ilu/csr_matrix.c */
jx_CSRMatrix *jx_CSRMatrixTDMGReorderByNodes( jx_CSRMatrix *A, JX_Int num_equns );
jx_CSRMatrix *jx_CSRMatrixTDMGReorderByVariables( jx_CSRMatrix *A, JX_Int num_equns );
jx_CSRMatrix *jx_CSRMatrixTDMGReorderAlongX( jx_CSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny );
jx_CSRMatrix *jx_CSRMatrixTDMGReorderAlongY( jx_CSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny );

/* csrc/ilu/decomposition.c */
JX_Int jx_ILUZeroDecompositionA( jx_CSRMatrix *A,
                              JX_Int **indexDP_ptr,
                              JX_Int **indexLU_ptr,
                              JX_Real **valueLU_ptr );
JX_Int jx_ILUZeroDecompositionB( jx_CSRMatrix *A,
                              JX_Real drop_tol,
                              JX_Int **indexAP_ptr,
                              JX_Int **indexDP_ptr,
                              JX_Int **indexLU_ptr,
                              JX_Real **valueLU_ptr );
JX_Int *jx_ILUZeroParallelDecompositionA( jx_ParCSRMatrix *par_A,
                                       JX_Int **indexDP_ptr,
                                       JX_Int **indexLU_ptr,
                                       JX_Real **valueLU_ptr,
                                       JX_Int *num_nonzeros,
                                       JX_Int *fill_in_drop,
                                       jx_ILUZeroFactorData *ilu_data );
JX_Int
jx_ILUZeroLocalDecompositionIntURPntsA( JX_Int *IA,
                                        JX_Int *JA,
                                        JX_Real *AA,
                                        JX_Int *indexLU,
                                        JX_Int *indexDP,
                                        JX_Int *placeRC,
                                        JX_Real *valueLU,
                                        JX_Int int_uprgt_pnt,
                                        JX_Int first_row_idx );
void
jx_ILUZeroLocalDecompositionDPntsA( JX_Int *IA,
                                    JX_Int *JA,
                                    JX_Real *AA,
                                    JX_Int *indexLU,
                                    JX_Int *indexDP,
                                    JX_Int *placeRC,
                                    JX_Real *valueLU,
                                    JX_Int int_uprgt_pnt,
                                    JX_Int first_row_idx,
                                    JX_Int num_rows,
                                    JX_Int ex_len,
                                    JX_Int *fill_in_drop );
void
jx_ILUZeroLocalDecompositionLPntsA( JX_Int *IA,
                                    JX_Int *JA,
                                    JX_Real *AA,
                                    JX_Int *indexLU,
                                    JX_Int *indexDP,
                                    JX_Int *placeRC,
                                    JX_Real *valueLU,
                                    JX_Int int_uprgtdwn_pnt,
                                    JX_Int first_row_idx,
                                    JX_Int num_rows,
                                    JX_Int dwn_num_rows,
                                    JX_Int ex_len,
                                    JX_Int *fill_in_drop );
void
jx_ILUZeroLocalDecompositionLDPntsA( JX_Int *IA,
                                     JX_Int *JA,
                                     JX_Real *AA,
                                     JX_Int *indexLU,
                                     JX_Int *indexDP,
                                     JX_Int *placeRC,
                                     JX_Real *valueLU,
                                     JX_Int int_urdl_pnt,
                                     JX_Int first_row_idx,
                                     JX_Int fst_row_idx,
                                     JX_Int num_rows,
                                     JX_Int lft_cnum_rows,
                                     JX_Int dwn_cnum_rows,
                                     JX_Int ex_len,
                                     JX_Int *fill_in_drop );
void jx_ILUZeroFactorDataParallelUPartIntURPntsA( JX_Int *indexLU,
                                                  JX_Int *indexAP,
                                                  JX_Int *indexDP,
                                                  JX_Real *valueLU,
                                                  JX_Int int_uprgt_pnt,
                                                  JX_Int recv_downsrt,
                                                  JX_Int recv_leftsrt,
                                                  JX_Int *permute,
                                                  JX_Real *aux_data,
                                                  JX_Real *res_data,
                                                  JX_Real *app_data,
                                                  JX_Int first_row_idx,
                                                  JX_Int next_row_idx,
                                                  JX_Int nnxt_row_idx );
void jx_ILUZeroFactorDataParallelUPartIntURPntsB( JX_Int *indexLU,
                                                  JX_Int *indexAP,
                                                  JX_Int *indexDP,
                                                  JX_Real *valueLU,
                                                  JX_Int int_uprgt_pnt,
                                                  JX_Int recv_downsrt,
                                                  JX_Int *permute,
                                                  JX_Real *aux_data,
                                                  JX_Real *res_data,
                                                  JX_Real *app_data,
                                                  JX_Int first_row_idx,
                                                  JX_Int next_row_idx );
void jx_ILUZeroFactorDataParallelUPartIntURPntsC( JX_Int *indexLU,
                                                  JX_Int *indexAP,
                                                  JX_Int *indexDP,
                                                  JX_Real *valueLU,
                                                  JX_Int int_uprgt_pnt,
                                                  JX_Int recv_leftsrt,
                                                  JX_Int *permute,
                                                  JX_Real *aux_data,
                                                  JX_Real *res_data,
                                                  JX_Real *app_data,
                                                  JX_Int first_row_idx,
                                                  JX_Int next_row_idx );
void jx_ILUZeroFactorDataParallelUPartIntURPntsD( JX_Int *indexLU,
                                                  JX_Int *indexAP,
                                                  JX_Int *indexDP,
                                                  JX_Real *valueLU,
                                                  JX_Int int_uprgt_pnt,
                                                  JX_Int *permute,
                                                  JX_Real *aux_data,
                                                  JX_Real *res_data,
                                                  JX_Real *app_data,
                                                  JX_Int first_row_idx );

/* csrc/ilu/grid.c */
JX_Int JX_GridPartitionDataCreate( JX_Solver *solver, MPI_Comm comm, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy );
JX_Int JX_GridPartitionDataSetEachSides4Comm( JX_Solver solver );
JX_Int JX_GridPartitionDataDestroy( JX_Solver solver );
void *jx_GridPartitionDataInitialize( MPI_Comm comm, JX_Int nx, JX_Int ny, JX_Int npx, JX_Int npy );
JX_Int jx_GridPartitionDataSetEachSides4Comm( void *grid_vdata );
JX_Int jx_GridPartitionDataFinalize( void *grid_vdata );

/* csrc/ilu/ilu.c */
JX_Int JX_ILUZeroFactorDataCreate( JX_Solver *solver, MPI_Comm comm );
JX_Int JX_ILUZeroFactorDataSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_ILUZeroFactorDataSetNxy( JX_Solver solver, JX_Int nx, JX_Int ny );
JX_Int JX_ILUZeroFactorDataSetNpxy( JX_Solver solver, JX_Int npx, JX_Int npy );
JX_Int JX_ILUZeroFactorDataSetNumEquns( JX_Solver solver, JX_Int num_equns );
JX_Int JX_ILUZeroFactorDataSetDropTol( JX_Solver solver, JX_Real drop_tol );
JX_Int JX_ILUZeroFactorDataSetMatA( JX_Solver solver, jx_CSRMatrix *matA );
JX_Int JX_ILUZeroFactorDataGetLULength( JX_Solver solver, JX_Int *lu_length );
JX_Int JX_ILUZeroFactorDataDestroy( JX_Solver solver );
JX_Int JX_ILUZeroFactorDataGenerateParGrid( JX_Solver solver );
JX_Int JX_ILUZeroFactorDataSetup( JX_Solver solver, JX_ParCSRMatrix par_matrix );
JX_Int JX_ILUZeroFactorDataPrecond( JX_Solver       solver,
                                 JX_ParCSRMatrix par_matrix,
                                 JX_ParVector    par_rhs,
                                 JX_ParVector    par_app  );
void *jx_ILUZeroFactorDataInitialize( MPI_Comm comm );
JX_Int jx_ILUZeroFactorDataSetMaxIter( void *ilu_vdata, JX_Int max_iter );
JX_Int jx_ILUZeroFactorDataSetNxy( void *ilu_vdata, JX_Int nx, JX_Int ny );
JX_Int jx_ILUZeroFactorDataSetNpxy( void *ilu_vdata, JX_Int npx, JX_Int npy );
JX_Int jx_ILUZeroFactorDataSetNumEquns( void *ilu_vdata, JX_Int num_equns );
JX_Int jx_ILUZeroFactorDataSetDropTol( void *ilu_vdata, JX_Real drop_tol );
JX_Int jx_ILUZeroFactorDataSetMatA( void *ilu_vdata, jx_CSRMatrix *matA );
JX_Int jx_ILUZeroFactorDataGetLULength( void *ilu_vdata, JX_Int *lu_length );
JX_Int jx_ILUZeroFactorDataFinalize( void *ilu_vdata );
JX_Int jx_ILUZeroFactorDataGenerateParGrid( void *ilu_vdata );

/* csrc/ilu/ilucycle.c */
JX_Int jx_ILUZeroFactorDataPrecond( void            *ilu_vdata,
                                 jx_ParCSRMatrix *par_A,
                                 jx_ParVector    *par_b,
                                 jx_ParVector    *par_x );
void jx_ILUZeroFactorDataCycleA( void *ilu_vdata, jx_CSRMatrix *A, jx_Vector *f, jx_Vector *u );
void jx_ILUZeroFactorDataCycleB( void *ilu_vdata, jx_CSRMatrix *A, jx_Vector *f, jx_Vector *u );
void jx_ILUZeroFactorDataParallelCycleA( jx_ILUZeroFactorData *ilu_data,
                                         jx_ParCSRMatrix *par_A,
                                         jx_ParVector *par_b,
                                         jx_ParVector *par_x );
void jx_ILUZeroFactorDataParallelLPartIntURPntsA( JX_Int *indexLU,
                                                  JX_Int *indexAP,
                                                  JX_Int *indexDP,
                                                  JX_Real *valueLU,
                                                  JX_Int *permute,
                                                  JX_Int int_uprgt_pnt,
                                                  JX_Real *aux_data,
                                                  JX_Real *rhs_data,
                                                  JX_Int first_row_idx );
void jx_ILUZeroFactorDataParallelLPartDPntsA( JX_Int *indexLU,
                                              JX_Int *indexAP,
                                              JX_Int *indexDP,
                                              JX_Real *valueLU,
                                              JX_Int int_uprgt_pnt,
                                              JX_Int num_rows,
                                              JX_Int *permute,
                                              JX_Real *aux_data,
                                              JX_Real *res_data,
                                              JX_Real *rhs_data,
                                              JX_Int first_row_idx );
void jx_ILUZeroFactorDataParallelLPartLPntsA( JX_Int *indexLU,
                                              JX_Int *indexAP,
                                              JX_Int *indexDP,
                                              JX_Real *valueLU,
                                              JX_Int int_uprgtdwn_pnt,
                                              JX_Int num_rows,
                                              JX_Int dwn_num_rows,
                                              JX_Int *permute,
                                              JX_Real *aux_data,
                                              JX_Real *res_data,
                                              JX_Real *rhs_data,
                                              JX_Int first_row_idx );
void jx_ILUZeroFactorDataParallelLPartLDPntsA( JX_Int *indexLU,
                                               JX_Int *indexAP,
                                               JX_Int *indexDP,
                                               JX_Real *valueLU,
                                               JX_Int int_urdl_pnt,
                                               JX_Int num_rows,
                                               JX_Int fst_row_idx,
                                               JX_Int *permute,
                                               JX_Int lft_cnum_rows,
                                               JX_Int dwn_dnum_rows,
                                               JX_Real *aux_data,
                                               JX_Real *res_data,
                                               JX_Real *rhs_data,
                                               JX_Int first_row_idx );
void jx_ILUZeroFactorDataParallelUPartLDPntsA( JX_Int *indexLU,
                                               JX_Int *indexAP,
                                               JX_Int *indexDP,
                                               JX_Real *valueLU,
                                               JX_Int int_urdl_pnt,
                                               JX_Int num_rows,
                                               JX_Int *permute,
                                               JX_Real *aux_data,
                                               JX_Real *app_data,
                                               JX_Int first_row_idx );
void jx_ILUZeroFactorDataParallelUPartLPntsA( JX_Int *indexLU,
                                              JX_Int *indexAP,
                                              JX_Int *indexDP,
                                              JX_Real *valueLU,
                                              JX_Int int_urdl_pnt,
                                              JX_Int num_rows,
                                              JX_Int recv_downsrt,
                                              JX_Int *permute,
                                              JX_Real *aux_data,
                                              JX_Real *res_data,
                                              JX_Real *app_data,
                                              JX_Int first_row_idx,
                                              JX_Int next_row_idx );
void jx_ILUZeroFactorDataParallelUPartDPntsA( JX_Int *indexLU,
                                              JX_Int *indexAP,
                                              JX_Int *indexDP,
                                              JX_Real *valueLU,
                                              JX_Int int_uprgt_pnt,
                                              JX_Int num_rows,
                                              JX_Int recv_leftsrt,
                                              JX_Int *permute,
                                              JX_Real *aux_data,
                                              JX_Real *res_data,
                                              JX_Real *app_data,
                                              JX_Int first_row_idx,
                                              JX_Int next_row_idx );

/* csrc/ilu/ilusetup.c */
JX_Int jx_ILUZeroFactorDataSetup( void *ilu_vdata, jx_ParCSRMatrix *par_A );

/* csrc/ilu/par_csr_matrix.c */
jx_ParCSRMatrix *jx_BuildMatParFromOneFile2( char *filename, 
                                             JX_Int   file_base,
                                             JX_Int  *row_part, 
                                             JX_Int  *col_part,
                                             JX_Int   num_equns,
                                             JX_Int   nx,
                                             JX_Int   ny );
jx_ParCSRMatrix *jx_BuildMatParFromOneFile3( char *filename, JX_Int file_base, JX_Int num_equns, JX_Int nx, JX_Int ny );
jx_ParCSRMatrix *
jx_ParCSRMatrixCreate2( MPI_Comm   comm,
                        JX_Int        global_num_rows,
                        JX_Int        global_num_cols,
                        JX_Int       *row_starts,
                        JX_Int       *col_starts,
                        JX_Int        num_cols_offd,
                        JX_Int        num_nonzeros_diag,
                        JX_Int        num_nonzeros_offd,
                        JX_Int        num_equns,
                        JX_Int        nx,
                        JX_Int        ny );
jx_ParCSRMatrix *jx_CSRMatrixToParCSRMatrix2( MPI_Comm comm, jx_CSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny );
jx_CSRMatrix *jx_MergeDiagAndOffdDropSmall( jx_ParCSRMatrix *par_matrix, JX_Real drop_tol );
jx_CSRMatrix *
jx_CSRMatrixMergeReorderIntUpRtDnLtQuasiBdy( MPI_Comm              comm,
                                             JX_Int                  *row_starts,
                                             JX_Int                  *permute,
                                             jx_CSRMatrix         *ser_B,
                                             JX_Int                   nz_srt,
                                             JX_Int                   ng_pt,
                                             jx_GridPartitionData *grid_data,
                                             JX_Int                  *ex_len,
                                             JX_Int                  *ey_len,
                                             JX_Int                  *dx_len,
                                             JX_Int                  *kx_len,
                                             JX_Int                  *dy_len,
                                             JX_Int                  *lx_len,
                                             JX_Int                  *ly_len,
                                             JX_Int                  *postn_a,
                                             JX_Int                  *postn_b,
                                             JX_Int                  *postn_c );

/* csrc/ilu/par_vector.c */
jx_ParVector *jx_BuildRhsParFromOneFile2( char *filename, jx_ParCSRMatrix *A, JX_Int num_equns, JX_Int nx, JX_Int ny );
JX_Int jx_GeneratePartitioning2(JX_Int num_procs, JX_Int **part_ptr, JX_Int num_equns, JX_Int length, JX_Int ny);

/* csrc/ilu/vector.c */
jx_Vector *jx_VectorTDMGReorderByNodes( jx_Vector *x, JX_Int num_equns );
jx_Vector *jx_VectorTDMGReorderByVariables( jx_Vector *x, JX_Int num_equns );
jx_Vector *jx_VectorTDMGReorderAlongX( jx_Vector *x, JX_Int num_equns, JX_Int nx, JX_Int ny );
jx_Vector *jx_VectorTDMGReorderAlongY( jx_Vector *x, JX_Int num_equns, JX_Int nx, JX_Int ny );

#endif
