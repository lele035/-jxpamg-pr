//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_pamg.h -- head files for parallel amg solver
 *  Date: 2011/09/08
 *
 *  Created by peghoty
 */ 
 
#ifndef JX_PAMG_HEADER
#define JX_PAMG_HEADER
 
#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif

#ifndef JX_MV_HEADER 
#include "jx_mv.h"
#endif
 
// pardiso added by zhaoli, 2021.06.30
#define WITH_PARDISO 1
#if WITH_PARDISO
#include "mkl_pardiso.h"
#include "mkl_types.h"
#endif	

#define AMG_CG_ZL 0
/*----------------------------------------------------------------*
 *                      Macro Definition                          *
 *----------------------------------------------------------------*/ 
 
#define LIST_HEAD -1
#define LIST_TAIL -2


/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jx_double_linked_list
 */
struct jx_double_linked_list
{
   JX_Int                           data;
   struct jx_double_linked_list *next_elt;
   struct jx_double_linked_list *prev_elt;
   JX_Int                           head;
   JX_Int                           tail;
};

typedef struct jx_double_linked_list jx_ListElement;
typedef jx_ListElement  *jx_LinkList; 



#define PARDISO_SOLVER 10
#if WITH_PARDISO
/**
 * \struct Pardiso_data
 * \brief  Data for Intel MKL PARDISO interface
 *
 * Added on 11/28/2015
 */
typedef struct {
    
    //! Internal solver memory pointer
    void *pt[64];
    
    //! Pardiso control parameters
    MKL_INT iparm[64];
    
    //! Type of the matrix
    MKL_INT mtype;
    
    //! Maximum number of numerical factorizations
    MKL_INT maxfct;
    
    //! Indicate the actual matrix for the solution phase, 1 <= mnum <= maxfct
    MKL_INT mnum; 

} Pardiso_data; /**< Data for PARDISO */
#endif



/*!
 * \struct jx_ParAMGData
 */ 
#define CUMNUMIT
typedef struct
{
   /* setup params */
   JX_Int      max_levels;
   JX_Real   strong_threshold;
   JX_Real   max_row_sum;
   JX_Real   trunc_factor;
   JX_Real   jacobi_trunc_threshold;
   JX_Real   S_commpkg_switch;
   JX_Real   CR_rate;
   JX_Real   CR_strong_th;
   JX_Int      measure_type;
   JX_Int      setup_type;
   JX_Int      coarsen_type;
   JX_Int      P_max_elmts;
   JX_Int      interp_type;
   JX_Int      restr_par;
   JX_Real   AIR_strong_th;
   JX_Int      agg_num_levels;
   JX_Int      agg_interp_type;
   JX_Int      agg_P_max_elmts;
   JX_Int      agg_P12_max_elmts;
   JX_Real   agg_trunc_factor;
   JX_Real   agg_P12_trunc_factor;
   JX_Int      num_paths;
   JX_Int      post_interp_type;
   JX_Int      num_CR_relax_steps;
   JX_Int      IS_type;
   JX_Int      CR_use_CG;
   JX_Int      cgc_its;
   JX_Int      spmt_rap_type; /* RAP Type for single processor with multi-threads Yue Xiaoqiang 2012/10/12 */
   JX_Int      wall_time_option;       // run-time of AMG component, Yue Xiaoqiang 2015/09/30
   JX_Int      ai_measure_type;        // Algebraic interface, Yue Xiaoqiang 2014/02/26
   JX_Int      ai_relax_type;          // AI-smoothing, Yue Xiaoqiang 2014/07/06
   JX_Int      measure_type_rlx;       // peghoty 2010/05/29
   JX_Int      number_syn_rlx;         // peghoty 2010/05/29
   JX_Real   measure_threshold_rlx;  // peghoty 2010/05/29

   /* solve params */
   JX_Int      max_iter;
   JX_Int      min_iter;
   JX_Int      cycle_type;    
   JX_Int     *num_grid_sweeps;  
   JX_Int     *grid_relax_type;   
   JX_Int    **grid_relax_points;
   JX_Int      relax_order;
   JX_Int      user_relax_type;
   JX_Int      user_coarse_relax_type;   
   JX_Real  *relax_weight; 
   JX_Real  *omega;
   JX_Real   tol;
   JX_Real   rhsnrm_threshold; // for the V-cycle control, the following strategy is used 
                              // when rhsnrm_threshold != 0:
                              // 1. if ||rhs|| > rhsnrm_threshold, use relative norm control
                              // 2. otherwise, use absolute norm control
                              // default of rhsnrm_threshold: 0.0   
   /* problem data */
   jx_ParCSRMatrix  *A;
   JX_Int      num_variables;
   JX_Int      num_functions;
   JX_Int      nodal;
   JX_Int      nodal_diag;
   JX_Int      num_points;
   JX_Int     *dof_func;
   JX_Int     *dof_point;           
   JX_Int     *point_dof_map;

   // data for Intel MKL PARDISO, 粗空间解法器
#if WITH_PARDISO
   Pardiso_data pdata; 
   jx_CSRMatrix *Pardiso_AH;
#endif

   /* data generated in the setup phase */
   jx_ParCSRMatrix **A_array;
   jx_ParVector    **F_array;
   jx_ParVector    **U_array;
   jx_ParCSRMatrix **P_array;
   jx_ParCSRMatrix **R_array;
   JX_Real          **AI_measure_array;
   JX_Int             **CF_marker_array;
   JX_Int             **dof_func_array;
   JX_Int             **dof_point_array;
   JX_Int             **point_dof_map_array;
   JX_Int               num_levels;
   JX_Real          **l1_norms;
   JX_Int            *AIR_maxsize_ls;

   /* Block data */
   jx_ParCSRBlockMatrix **A_block_array;
   jx_ParCSRBlockMatrix **P_block_array;
   jx_ParCSRBlockMatrix **R_block_array;

   JX_Int block_mode;

   /* data for more complex smoothers */
   JX_Int               smooth_num_levels;
   JX_Int               smooth_type;
   JX_Solver        *smoother;
   JX_Int               smooth_num_sweeps;
   JX_Int               schw_variant;
   JX_Int               schw_overlap;
   JX_Int               schw_domain_type;
   JX_Real            schwarz_rlx_weight;
   JX_Int               schwarz_use_nonsymm;
   JX_Int               ps_sym;
   JX_Int               ps_level;
   JX_Int               pi_max_nz_per_row;
   JX_Int               eu_level;
   JX_Int               eu_bj;
   JX_Real            ps_threshold;
   JX_Real            ps_filter;
   JX_Real            pi_drop_tol;
   JX_Real            eu_sparse_A;
   char             *euclidfile;

   /* data generated in the solve phase */
   jx_ParVector   *Vtemp;
   jx_Vector      *Vtemp_local;
   JX_Real         *Vtemp_local_data;
   JX_Real          cycle_op_count;
   jx_ParVector   *Rtemp;
   jx_ParVector   *Ptemp;
   jx_ParVector   *Ztemp;

   /* fields used by GSMG and LS interpolation */
   JX_Int             gsmg;        /* nonzero indicates use of GSMG */
   JX_Int             num_samples; /* number of sample vectors */

   /* log info */
   JX_Int      logging;
   JX_Int      num_iterations;
#ifdef CUMNUMIT
   JX_Int      cum_num_iterations;
#endif
   JX_Real   rel_resid_norm;
   jx_ParVector *residual; /* available if logging > 1 */

   /* output params */
   JX_Int      print_level;
   JX_Int      print_coarse_matrix;
   char     log_file_name[256];
   JX_Int      debug_flag;

   /* whether to print the constructed coarse grids BM Oct 22, 2006 */
   JX_Int      plot_grids;
   char     plot_filename[251];

   /* coordinate data BM Oct 17, 2006 */
   JX_Int      coorddim;
   float   *coordinates;

   /* peghoty 2009/07/27 */
   JX_Int      conv_criteria;
   JX_Int      coarsestsolverid; 
   
   /* peghoty 2010/04/14 */
   JX_Int      coarse_threshold;   // the allowed number of gridpoints in the coarsest level
   JX_Real   coarse_ratio;       // when coarse_size >= fine_size*coarse_ratio,
                                // use CLJP instead of the current coarsening strategy.
   JX_Real   convfac_threshold;  // if the current convergence factor > convfac_threshold, 
                                // stop the iteration

   /* Use 2 mat-mat-muls instead of triple product*/
   JX_Int rap2;

   JX_Int keepTranspose;
   
} jx_ParAMGData;
/* Predictive reuse API */
JX_Int JXPAMG_PR_SetupFromLevelExperimental(jx_ParAMGData *, JX_Int, JX_Int);
JX_Int JXPAMG_PR_AdaptiveReuseAlgorithm(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real, JX_Int, JX_Real);
JX_Int JXPAMG_PR_AdaptiveReuse3B2(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real, JX_Int, JX_Int);
JX_Int JXPAMG_PR_AdaptiveReuse3B2_Residual(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real *, JX_Real, JX_Int, JX_Int);
JX_Int JXPAMG_PR_PredictCutReadOnly(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real, JX_Int, JX_Real, JX_Int *);
JX_Int JXPAMG_PR_PredictCutReadOnlyWithCandidate(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real, JX_Int, JX_Real, JX_Int *, jx_ParCSRMatrix **);
JX_Int JXPAMG_PR_FullSetupOnNewMatrix(jx_ParAMGData *, jx_ParCSRMatrix *);
JX_Int JXPAMG_PR_SetupFromLevelWithACut(jx_ParAMGData *, JX_Int, jx_ParCSRMatrix *);
JX_Int JXPAMG_PR_NoOpSetup(JX_Solver, JX_ParCSRMatrix);
JX_Int JXPAMG_PR_CanFullReuse(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Int, JX_Int);

/* setup params */	
#if WITH_PARDISO	  		      
#define jx_ParAMGDataPardisoData(amg_data)          ((amg_data)->pdata)
#endif
#define jx_ParAMGDataRestriction(amg_data)          ((amg_data)->restr_par)
#define jx_ParAMGDataAIRStrongTh(amg_data)          ((amg_data)->AIR_strong_th)
#define jx_ParAMGDataMaxLevels(amg_data)            ((amg_data)->max_levels)
#define jx_ParAMGDataStrongThreshold(amg_data)      ((amg_data)->strong_threshold)
#define jx_ParAMGDataMaxRowSum(amg_data)            ((amg_data)->max_row_sum)
#define jx_ParAMGDataTruncFactor(amg_data)          ((amg_data)->trunc_factor)
#define jx_ParAMGDataJacobiTruncThreshold(amg_data) ((amg_data)->jacobi_trunc_threshold)
#define jx_ParAMGDataSCommPkgSwitch(amg_data)       ((amg_data)->S_commpkg_switch)
#define jx_ParAMGDataInterpType(amg_data)           ((amg_data)->interp_type)
#define jx_ParAMGDataCoarsenType(amg_data)          ((amg_data)->coarsen_type)
#define jx_ParAMGDataMeasureType(amg_data)          ((amg_data)->measure_type)
#define jx_ParAMGDataSetupType(amg_data)            ((amg_data)->setup_type)
#define jx_ParAMGDataPMaxElmts(amg_data)            ((amg_data)->P_max_elmts)
#define jx_ParAMGDataNumPaths(amg_data)             ((amg_data)->num_paths)
#define jx_ParAMGDataAggNumLevels(amg_data)         ((amg_data)->agg_num_levels)
#define jx_ParAMGDataAggInterpType(amg_data)        ((amg_data)->agg_interp_type)
#define jx_ParAMGDataAggPMaxElmts(amg_data)         ((amg_data)->agg_P_max_elmts)
#define jx_ParAMGDataAggP12MaxElmts(amg_data)       ((amg_data)->agg_P12_max_elmts)
#define jx_ParAMGDataAggTruncFactor(amg_data)       ((amg_data)->agg_trunc_factor)
#define jx_ParAMGDataAggP12TruncFactor(amg_data)    ((amg_data)->agg_P12_trunc_factor)
#define jx_ParAMGDataPostInterpType(amg_data)       ((amg_data)->post_interp_type)
#define jx_ParAMGDataNumCRRelaxSteps(amg_data)      ((amg_data)->num_CR_relax_steps)
#define jx_ParAMGDataCRRate(amg_data)               ((amg_data)->CR_rate)
#define jx_ParAMGDataCRStrongTh(amg_data)           ((amg_data)->CR_strong_th)
#define jx_ParAMGDataISType(amg_data)               ((amg_data)->IS_type)
#define jx_ParAMGDataCRUseCG(amg_data)              ((amg_data)->CR_use_CG)
#define jx_ParAMGDataL1Norms(amg_data)              ((amg_data)->l1_norms)
#define jx_ParAMGDataCGCIts(amg_data)               ((amg_data)->cgc_its)
#define jx_ParAMGDataSpMtRapType(amg_data)          ((amg_data)->spmt_rap_type)         // Yue Xiaoqiang 2012/10/13
#define jx_ParAMGDataWallTimeOption(amg_data)       ((amg_data)->wall_time_option)      // Yue Xiaoqiang 2015/09/30
#define jx_ParAMGDataAIMeasureType(amg_data)        ((amg_data)->ai_measure_type)       // Yue Xiaoqiang 2014/02/26
#define jx_ParAMGDataAIRelaxType(amg_data)          ((amg_data)->ai_relax_type)         // Yue Xiaoqiang 2014/07/06
#define jx_ParAMGDataMeasureTypeRlx(amg_data)       ((amg_data)->measure_type_rlx)      // peghoty 2010/05/29
#define jx_ParAMGDataNumberSynRlx(amg_data)         ((amg_data)->number_syn_rlx)        // peghoty 2010/05/29
#define jx_ParAMGDataMeasureThresholdRlx(amg_data)  ((amg_data)->measure_threshold_rlx) // peghoty 2010/05/29
#define jx_ParAMGDataAIRMaxSizeLS(amg_data)         ((amg_data)->AIR_maxsize_ls)        // Yue Xiaoqiang 2018/11/25
/* solve params */
#define jx_ParAMGDataMinIter(amg_data)             ((amg_data)->min_iter)
#define jx_ParAMGDataMaxIter(amg_data)             ((amg_data)->max_iter)
#define jx_ParAMGDataCycleType(amg_data)           ((amg_data)->cycle_type)
#define jx_ParAMGDataNumGridSweeps(amg_data)       ((amg_data)->num_grid_sweeps)
#define jx_ParAMGDataUserCoarseRelaxType(amg_data) ((amg_data)->user_coarse_relax_type)
#define jx_ParAMGDataGridRelaxType(amg_data)       ((amg_data)->grid_relax_type)
#define jx_ParAMGDataGridRelaxPoints(amg_data)     ((amg_data)->grid_relax_points)
#define jx_ParAMGDataRelaxOrder(amg_data)          ((amg_data)->relax_order)
#define jx_ParAMGDataUserRelaxType(amg_data)       ((amg_data)->user_relax_type)
#define jx_ParAMGDataRelaxWeight(amg_data)         ((amg_data)->relax_weight)
#define jx_ParAMGDataOmega(amg_data)               ((amg_data)->omega)
#define jx_ParAMGDataTol(amg_data)                 ((amg_data)->tol)
#define jx_ParAMGDataRhsNrmThreshold(amg_data)     ((amg_data)->rhsnrm_threshold)
/* problem data parameters */
#define jx_ParAMGDataNumVariables(amg_data)        ((amg_data)->num_variables)
#define jx_ParAMGDataNumFunctions(amg_data)        ((amg_data)->num_functions)
#define jx_ParAMGDataNodal(amg_data)               ((amg_data)->nodal)
#define jx_ParAMGDataNodalDiag(amg_data)           ((amg_data)->nodal_diag)
#define jx_ParAMGDataNumPoints(amg_data)           ((amg_data)->num_points)
#define jx_ParAMGDataDofFunc(amg_data)             ((amg_data)->dof_func)
#define jx_ParAMGDataDofPoint(amg_data)            ((amg_data)->dof_point)
#define jx_ParAMGDataPointDofMap(amg_data)         ((amg_data)->point_dof_map)
/* data generated by the setup phase */
#define jx_ParAMGDataAIMeasureArray(amg_data)       ((amg_data)->AI_measure_array)
#define jx_ParAMGDataCFMarkerArray(amg_data)       ((amg_data)->CF_marker_array)
#define jx_ParAMGDataAArray(amg_data)              ((amg_data)->A_array)
#define jx_ParAMGDataFArray(amg_data)              ((amg_data)->F_array)
#define jx_ParAMGDataUArray(amg_data)              ((amg_data)->U_array)
#define jx_ParAMGDataPArray(amg_data)              ((amg_data)->P_array)
#define jx_ParAMGDataRArray(amg_data)              ((amg_data)->R_array)
#define jx_ParAMGDataDofFuncArray(amg_data)        ((amg_data)->dof_func_array)
#define jx_ParAMGDataDofPointArray(amg_data)       ((amg_data)->dof_point_array)
#define jx_ParAMGDataPointDofMapArray(amg_data)    ((amg_data)->point_dof_map_array) 
#define jx_ParAMGDataNumLevels(amg_data)           ((amg_data)->num_levels)	
#define jx_ParAMGDataSmoothType(amg_data)          ((amg_data)->smooth_type)
#define jx_ParAMGDataSmoothNumLevels(amg_data)     ((amg_data)->smooth_num_levels)
#define jx_ParAMGDataSmoothNumSweeps(amg_data)     ((amg_data)->smooth_num_sweeps)	
#define jx_ParAMGDataSmoother(amg_data)            ((amg_data)->smoother)	
#define jx_ParAMGDataVariant(amg_data)             ((amg_data)->schw_variant)	
#define jx_ParAMGDataOverlap(amg_data)             ((amg_data)->schw_overlap)	
#define jx_ParAMGDataDomainType(amg_data)          ((amg_data)->schw_domain_type)	
#define jx_ParAMGDataSchwarzRlxWeight(amg_data)    ((amg_data)->schwarz_rlx_weight)
#define jx_ParAMGDataSchwarzUseNonSymm(amg_data)   ((amg_data)->schwarz_use_nonsymm)
#define jx_ParAMGDataSym(amg_data)                 ((amg_data)->ps_sym)	
#define jx_ParAMGDataLevel(amg_data)               ((amg_data)->ps_level)	
#define jx_ParAMGDataMaxNzPerRow(amg_data)         ((amg_data)->pi_max_nz_per_row)
#define jx_ParAMGDataThreshold(amg_data)           ((amg_data)->ps_threshold)	
#define jx_ParAMGDataFilter(amg_data)              ((amg_data)->ps_filter)	
#define jx_ParAMGDataDropTol(amg_data)             ((amg_data)->pi_drop_tol)	
#define jx_ParAMGDataEuclidFile(amg_data)          ((amg_data)->euclidfile)	
#define jx_ParAMGDataEuLevel(amg_data)             ((amg_data)->eu_level)	
#define jx_ParAMGDataEuSparseA(amg_data)           ((amg_data)->eu_sparse_A)
#define jx_ParAMGDataEuBJ(amg_data)                ((amg_data)->eu_bj)
/* block */
#define jx_ParAMGDataABlockArray(amg_data)         ((amg_data)->A_block_array)
#define jx_ParAMGDataPBlockArray(amg_data)         ((amg_data)->P_block_array)
#define jx_ParAMGDataRBlockArray(amg_data)         ((amg_data)->R_block_array)
#define jx_ParAMGDataBlockMode(amg_data)           ((amg_data)->block_mode)
/* data generated in the solve phase */
#define jx_ParAMGDataVtemp(amg_data)               ((amg_data)->Vtemp)
#define jx_ParAMGDataVtempLocal(amg_data)          ((amg_data)->Vtemp_local)
#define jx_ParAMGDataVtemplocalData(amg_data)      ((amg_data)->Vtemp_local_data)
#define jx_ParAMGDataCycleOpCount(amg_data)        ((amg_data)->cycle_op_count)
#define jx_ParAMGDataRtemp(amg_data)               ((amg_data)->Rtemp)
#define jx_ParAMGDataPtemp(amg_data)               ((amg_data)->Ptemp)
#define jx_ParAMGDataZtemp(amg_data)               ((amg_data)->Ztemp)
/* fields used by GSMG */
#define jx_ParAMGDataGSMG(amg_data)                ((amg_data)->gsmg)
#define jx_ParAMGDataNumSamples(amg_data)          ((amg_data)->num_samples)
/* log info data */
#define jx_ParAMGDataLogging(amg_data)             ((amg_data)->logging)
#define jx_ParAMGDataNumIterations(amg_data)       ((amg_data)->num_iterations)
#ifdef CUMNUMIT
#define jx_ParAMGDataCumNumIterations(amg_data)    ((amg_data)->cum_num_iterations)
#endif
#define jx_ParAMGDataRelativeResidualNorm(amg_data)((amg_data)->rel_resid_norm)
#define jx_ParAMGDataResidual(amg_data)            ((amg_data)->residual)
/* output parameters */
#define jx_ParAMGDataPrintLevel(amg_data)          ((amg_data)->print_level)
#define jx_ParAMGDataPrintCoarseSystem(amg_data)   ((amg_data)->print_coarse_matrix)
#define jx_ParAMGDataLogFileName(amg_data)         ((amg_data)->log_file_name)
#define jx_ParAMGDataDebugFlag(amg_data)           ((amg_data)->debug_flag)
/* BM Oct 22, 2006 */
#define jx_ParAMGDataPlotGrids(amg_data)           ((amg_data)->plot_grids)
#define jx_ParAMGDataPlotFileName(amg_data)        ((amg_data)->plot_filename)
/* coordinates BM Oct 17, 2006 */
#define jx_ParAMGDataCoordDim(amg_data)            ((amg_data)->coorddim)
#define jx_ParAMGDataCoordinates(amg_data)         ((amg_data)->coordinates)
/* peghoty 2009/07/27 */
#define jx_ParAMGDataConvCriteria(amg_data)        ((amg_data)->conv_criteria)
#define jx_ParAMGDataCoarsestSolverID(amg_data)    ((amg_data)->coarsestsolverid)
/* peghoty 2010/04/14 */
#define jx_ParAMGDataCoarseThreshold(amg_data)     ((amg_data)->coarse_threshold)
#define jx_ParAMGDataCoarseRatio(amg_data)         ((amg_data)->coarse_ratio)
 
#define jx_ParAMGDataConvFacThreshold(amg_data) ((amg_data)->convfac_threshold)

#define jx_ParAMGDataRAP2(amg_data) ((amg_data)->rap2)
#define jx_ParAMGDataKeepTranspose(amg_data) ((amg_data)->keepTranspose)

/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/amg/amg_linklist.c */
void jx_dispose_elt ( jx_LinkList element_ptr );
void 
jx_remove_point( jx_LinkList   *LoL_head_ptr, 
                 jx_LinkList   *LoL_tail_ptr, 
                 JX_Int            measure,
                 JX_Int            index, 
                 JX_Int           *lists, 
                 JX_Int           *where );
jx_LinkList jx_create_elt( JX_Int Item );
void 
jx_enter_on_lists( jx_LinkList   *LoL_head_ptr, 
                   jx_LinkList   *LoL_tail_ptr, 
                   JX_Int            measure,
                   JX_Int            index, 
                   JX_Int           *lists, 
                   JX_Int           *where );



/* csrc/amg/par_amg.c */

/*!
 * Create a solver or preconditioner object. 
 */
JX_Int JX_PAMGCreate( JX_Solver *solver);

/*!
 * Destroy a solver or preconditioner object. 
 */
JX_Int JX_PAMGDestroy( JX_Solver solver );

/*!
 * Set up the PAMG solver or preconditioner. 
 * If used as a preconditioner, this function should be passed
 * to the iterative solver SetPrecond function.
 * This page has been automatically generated with DOC++
 * Parameters: 
 *    solver — [IN] object to be set up.
 *    par_matrix — [IN] ParCSR matrix used to construct the solver/preconditioner.
 */
JX_Int JX_PAMGSetup( JX_Solver solver, JX_ParCSRMatrix  par_matrix );

/*!
 * Solve the system or apply PAMG as a preconditioner. 
 * If used as a preconditioner, this function should be
 * passed to the iterative solver SetPrecond function.
 * Parameters:                     
 *  solver — [IN] solver or preconditioner object to be applied.
 *  par_matrix — [IN] ParCSR matrix, matrix of the linear system to be solved
 *  par_rhs — [IN] right hand side of the linear system to be solved
 *  par_app — [OUT] approximated solution of the linear system to be solved
 */
JX_Int JX_PAMGSolve( JX_Solver       solver, 
                  JX_ParCSRMatrix par_matrix, 
                  JX_ParVector    par_rhs, 
                  JX_ParVector    par_app );
                  
/* peghoty 2011/09/04 */               
JX_Int JX_PAMGPrecond( JX_Solver       solver, 
                    JX_ParCSRMatrix par_matrix,
                    JX_ParVector    par_rhs, 
                    JX_ParVector    par_app );

JX_Int JX_PAMGSetRestriction( JX_Solver solver, JX_Int restr_par );
JX_Int JX_PAMGSetAIRStrongTh( JX_Solver solver, JX_Real AIR_strong_th );

/*!
 * Set maximum number of multigrid levels. The default is 25.
 */
JX_Int JX_PAMGSetMaxLevels( JX_Solver solver, JX_Int max_levels );
JX_Int JX_PAMGGetMaxLevels( JX_Solver solver, JX_Int *max_levels );

/*!
 * Set AMG strength threshold. The default is 0.25. For 2d Laplace operators, 
 * 0.25 is a good value, for 3d Laplace operators, 0.5 or 0.6 is a better value. 
 * For elasticity problems, a large strength threshold,
 * such as 0.9, is often better.
 */
JX_Int JX_PAMGSetStrongThreshold( JX_Solver solver, JX_Real strong_threshold );

JX_Int JX_PAMGGetStrongThreshold( JX_Solver solver, JX_Real *strong_threshold );

/*!
 * Set a parameter to modify the definition of strength for diagonal dominant 
 * portions of the matrix. The default is 0.9. If max row sum is 1,
 * no checking for diagonally dominant rows is performed.
 */
JX_Int JX_PAMGSetMaxRowSum( JX_Solver solver, JX_Real max_row_sum );
JX_Int JX_PAMGGetMaxRowSum( JX_Solver solver, JX_Real *max_row_sum );

/*!
 * Define a truncation factor for the interpolation. 
 * The default is 0.
 */
JX_Int JX_PAMGSetTruncFactor( JX_Solver solver, JX_Real trunc_factor );
JX_Int JX_PAMGGetTruncFactor( JX_Solver solver, JX_Real *trunc_factor );

/*!
 * Define the maximal number of elements per row for the interpolation. 
 * The default is 0.
 */
JX_Int JX_PAMGSetPMaxElmts( JX_Solver solver, JX_Int P_max_elmts );
JX_Int JX_PAMGGetPMaxElmts( JX_Solver solver, JX_Int *P_max_elmts );

JX_Int JX_PAMGSetJacobiTruncThreshold( JX_Solver solver, JX_Real jacobi_trunc_threshold );
JX_Int JX_PAMGGetJacobiTruncThreshold( JX_Solver solver, JX_Real *jacobi_trunc_threshold );
JX_Int JX_PAMGSetPostInterpType( JX_Solver solver, JX_Int post_interp_type );
JX_Int JX_PAMGGetPostInterpType( JX_Solver solver, JX_Int *post_interp_type );

/*!
 * Define the largest strength threshold for which the strength 
 * matrix S uses the communication package of the operator A. 
 * If the strength threshold is larger than this values, a communication 
 * package is generated for S. This can save memory and decrease the amount of data 
 * that needs to be communicated, if S is substantially sparser than A.
 * The default is 1.0.
 */
JX_Int JX_PAMGSetSCommPkgSwitch( JX_Solver solver, JX_Real S_commpkg_switch );

/*!
 * Define which parallel interpolation operator is used. 
 * There are the following options for interp type:
 * 0    classical modified interpolation
 * 3    direct interpolation (with separation of weights)
 * 4    multipass interpolation
 * 5    multipass interpolation (with separation of weights)
 * 6    extended classical modified interpolation
 * 7    extended (if no common C neighbor) classical modified interpolation
 * 8    standard interpolation
 * 9    standard interpolation (with separation of weights)
 * The default is 0.
 */
JX_Int JX_PAMGSetInterpType( JX_Solver solver, JX_Int interp_type );

/*!
 * Set minimum number of iterations.
 * The default is 0.
 */
JX_Int JX_PAMGSetMinIter( JX_Solver solver, JX_Int min_iter );

/*!
 * Set maximum number of iterations, if PAMG is used as a solver. If it is used as a
 * preconditioner, this function has no effect. The default is 20.
 */
JX_Int JX_PAMGSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_PAMGGetMaxIter( JX_Solver solver, JX_Int *max_iter );

/*! 
 * Define which parallel coarsening algorithm is used. There are the following options for
 * coarsen type:
 * 0  CLJP-coarsening (a parallel coarsening algorithm using independent sets.
 * 1  classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
 * 11 one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
 * 2  Ruge2B
 * 3  classical Ruge-Stueben coarsening on each processor, followed by a third pass, which adds coarse
 *    points on the boundaries
 * 4  Ruge3c third pass on boundary,keep C points
 * 5  Ruge relax special points
 * 6  Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points
 *    generated by 1 as its first independent set)
 * 8  PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating
 *    lower complexities than CLJP, might also lead to slower convergence)
 * 10 HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed
 *    by PMIS using the interior C-points generated as its first independent set)
 * 91 RRS0 relaxed RS0
 * 90 RCLJP relaxed CLJP
 *  The default is 6.
 */
JX_Int JX_PAMGSetCoarsenType( JX_Solver solver, JX_Int coarsen_type );
JX_Int JX_PAMGGetCoarsenType( JX_Solver solver, JX_Int *coarsen_type );

/*! 
 * Set the measure type  
 * 0: measures are determined locally
 * 1: measures are determined globally
 * The default is 0.
 */
JX_Int JX_PAMGSetMeasureType( JX_Solver solver, JX_Int measure_type );
JX_Int JX_PAMGGetMeasureType( JX_Solver solver, JX_Int *measure_type );

/*! 
 * Set the setup type  
 * 0: don't do the AMG setup
 * 1: do the AMG setup
 * The default is 1.
 * peghoty  2009/12/11
 */
JX_Int JX_PAMGSetSetupType( JX_Solver solver, JX_Int setup_type );

/*! 
 * Define the type of cycle. For a V-cycle, set cycle type to 1, 
 * for a W-cycle set cycle type to 2.
 * The default is 1.
 */
JX_Int JX_PAMGSetCycleType( JX_Solver solver, JX_Int cycle_type );
JX_Int JX_PAMGGetCycleType( JX_Solver solver, JX_Int *cycle_type );

/*!
 * Set the convergence tolerance, if PAMG is used as a solver. 
 * If it is used as a preconditioner,
 * this function has no effect. 
 * The default is 1.e-7.
 */
JX_Int JX_PAMGSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_PAMGGetTol( JX_Solver solver, JX_Real *tol );

/* peghoty, 2011/09/28 */
JX_Int JX_PAMGSetRhsNrmThreshold( JX_Solver solver, JX_Real rhsnrm_threshold );

/*!
 * Define the number of sweeps for the fine and coarse grid, the up and down cycle.
 * Note:  This routine will be phased out!!!!           
 * Use JX_PAMGSetNumSweeps or JX_PAMGSetCycleNumSweeps instead.
 */
JX_Int JX_PAMGSetNumGridSweeps( JX_Solver solver, JX_Int *num_grid_sweeps );

/*!
 * Set the number of sweeps. On the finest level, the up and 
 * the down cycle the number of sweeps are set to
 * num sweeps and on the coarsest level to 1. The default is 1.
 */
JX_Int JX_PAMGSetNumSweeps( JX_Solver solver, JX_Int num_sweeps );

/*!
 * Set the number of sweeps at a specified cycle. 
 * There are the following options for k:
 * the finest level    if k=0
 * the down cycle      if k=1
 * the up cycle        if k=2
 * the coarsest level  if k=3.
 */
JX_Int JX_PAMGSetCycleNumSweeps( JX_Solver solver, JX_Int num_sweeps, JX_Int k );
JX_Int JX_PAMGGetCycleNumSweeps( JX_Solver solver, JX_Int *num_sweeps, JX_Int k );

/*!
 * Define which smoother is used on the fine and coarse grid, the up and down cycle.
 * Note: This routine will be phased out!!!!
 * Use JX_PAMGSetRelaxType or JX_PAMGSetCycleRelaxType instead.
 */
JX_Int JX_PAMGSetGridRelaxType( JX_Solver  solver, JX_Int *grid_relax_type );

/*!
 * Define the smoother to be used. It uses the given smoother on the fine grid, 
 * the up and the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is Gauss-  Seidel(3).
 * There are the following options for relax type:
 * 0   Jacobi
 * 1   Gauss-Seidel, sequential (very slow!)
 * 2   Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 * 3   hybrid Gauss-Seidel or SOR, forward solve
 * 4   hybrid Gauss-Seidel or SOR, backward solve
 * 5   hybrid chaotic Gauss-Seidel (works only with OpenMP)
 * 6   hybrid symmetric Gauss-Seidel or SSOR
 * 7   Jacobi (uses Matvec), only needed in CGNR
 * 9   Gaussian elimination (only on coarsest level)
 * The default is 3.
 */
JX_Int JX_PAMGSetRelaxType( JX_Solver solver, JX_Int relax_type );

/*!
 * Define the smoother at a given cycle. For options of relax type see description of
 * JX_PAMGSetRelaxType. Options for k are
 * the finest level    if k=0
 * the down cycle      if k=1
 * the up cycle        if k=2
 * the coarsest level  if k=3.
  */
JX_Int JX_PAMGSetCycleRelaxType( JX_Solver solver, JX_Int relax_type, JX_Int k );
JX_Int JX_PAMGGetCycleRelaxType( JX_Solver solver, JX_Int * relax_type, JX_Int k );

/*!
 * Define in which order the points are relaxed. 
 * There are the following options for relax order:
 * 0  the points are relaxed in natural or lexicographic order on each processor
 * 1  CF-relaxation is used, i.e on the fine grid and the down cycle the 
 *    coarse points are relaxed first, followed by the fine points;
 *    on the up cycle the F-points are relaxed first, followed by the C-points.
 *    On the coarsest level, if an iterative scheme is used,
 *    the points are relaxed in lexicographic order.
 * The default is 1 (CF-relaxation).
 */
JX_Int JX_PAMGSetRelaxOrder( JX_Solver solver, JX_Int relax_order );

/*!
 * Define in which order the points are relaxed.
 * Note: This routine will be phased out!!!! 
 * Use JX_PAMGSetRelaxOrder instead.
 */
JX_Int JX_PAMGSetGridRelaxPoints( JX_Solver solver, JX_Int **grid_relax_points );

/*!
 * Define the relaxation weight for smoothed Jacobi and hybrid SOR.
 * Note:  This routine will be phased out!!!!
 * Use JX_PAMGSetRelaxWt or JX_PAMGSetLevelRelaxWt instead.
 */
JX_Int JX_PAMGSetRelaxWeight( JX_Solver solver, JX_Real *relax_weight );

/*!
 * Define the relaxation weight for smoothed Jacobi and hybrid SOR on all levels.
 * relax weight > 0  this assigns the given relaxation weight on all levels                    
 * relax weight = 0  the weight is determined on each level with the estimate 
 *                   3 / (4*||D^{-1/2}AD^{-1/2}||), where D is the diagonal matrix
 *                   of A (this should only be used with Jacobi)
 * relax weight = -k the relaxation weight is determined with at most k CG steps on each level
 *                   this should only be used for symmetric positive definite problems
 * The default is 1.
 */
JX_Int JX_PAMGSetRelaxWt( JX_Solver solver, JX_Real relax_wt );

/*!
 * Define the relaxation weight for smoothed Jacobi and hybrid SOR on the user defined level. 
 * Note that the finest level is denoted 0, the next coarser level 1, etc. 
 ＊ For nonpositive relax weight, the parameter is determined on
 * the given level as described for JX_PAMGSetRelaxWt. 
 ＊ The default is 1.
 */
JX_Int JX_PAMGSetLevelRelaxWt( JX_Solver solver, JX_Real relax_wt, JX_Int level );

/*!  
 * Define the outer relaxation weight for hybrid SOR. 
 * Note: This routine will be phased out!!!! 
 * Use JX_PAMGSetOuterWt or JX_PAMGSetLevelOuterWt instead.
*/
JX_Int JX_PAMGSetOmega( JX_Solver solver, JX_Real *omega );

/*!
 * Define the outer relaxation weight for hybrid SOR and SSOR on all levels.
 * omega > 0   this assigns the same outer relaxation weight omega on each level
 * omega = -k  an outer relaxation weight is determined with at most k CG steps on each level
               (this only makes sense for symmetric positive definite problems and smoothers, e.g. SSOR)
 * The default is 1.
 */
JX_Int JX_PAMGSetOuterWt( JX_Solver solver, JX_Real outer_wt );

/*!
 * Define the outer relaxation weight for hybrid SOR or SSOR on the user defined level. 
 * Note that the finest level is denoted 0, the next coarser level 1, etc. 
 * For nonpositive omega, the parameter is determined
 * on the given level as described for JX_PAMGSetOuterWt. 
 * The default is 1.
 */
JX_Int JX_PAMGSetLevelOuterWt( JX_Solver solver, JX_Real outer_wt, JX_Int level );

/*!
 * Enables the use of more complex smoothers. 
 * The following options exist for smooth type:
 *
 * value   smoother            routines needed to set smoother parameters
 *
 *  6     Schwarz smoothers     JX_PAMGSetDomainType, JX_PAMGSetOverlap,
 *                              JX_PAMGSetVariant,    JX_PAMGAMGSetSchwarzRlxWeight
 *  7     Pilut                 JX_PAMGAMGSetDropTol, JX_PAMGAMGSetMaxNzPerRow
 *  8     ParaSails             JX_PAMGAMGSetSym,     JX_PAMGAMGSetLevel,
 *                              JX_PAMGAMGSetFilter,  JX_PAMGAMGSetThreshold
 *  9     Euclid                JX_PAMGAMGSetEuclidFile
 *
 * The default is 6.
 * Also, if no smoother parameters are set via the routines mentioned in the table above,
 * default values are used.
 */
JX_Int JX_PAMGSetSmoothType( JX_Solver solver, JX_Int smooth_type );
JX_Int JX_PAMGGetSmoothType( JX_Solver solver, JX_Int * smooth_type );

/*!
 * Set the number of levels for more complex smoothers.            
 * The smoothers, as defined
 * by JX_PAMGSetSmoothType, will be used on level 0 (the finest level) through level
 * smooth num levels-1. 
 * The default is 0, i.e. no complex smoothers are used.
 */
JX_Int JX_PAMGSetSmoothNumLevels( JX_Solver solver, JX_Int smooth_num_levels );
JX_Int JX_PAMGGetSmoothNumLevels( JX_Solver solver, JX_Int *smooth_num_levels );

/*!
 * Set the number of sweeps for more complex smoothers. 
 * The default is 1.
 */
JX_Int JX_PAMGSetSmoothNumSweeps( JX_Solver solver, JX_Int smooth_num_sweeps );
JX_Int JX_PAMGGetSmoothNumSweeps( JX_Solver solver, JX_Int *smooth_num_sweeps );

/*!
 * Requests additional computations for diagnostic and similar data to be logged by the user.
 * Default to 0 for do nothing. The latest residual will be available if logging > 1.
 */
JX_Int JX_PAMGSetLogging( JX_Solver solver, JX_Int logging );
JX_Int JX_PAMGGetLogging( JX_Solver solver, JX_Int *logging );

/*!
 * Requests automatic printing of setup and solve information.
 * 0   no printout (default)
 * 1   print setup information
 * 2   print solve information
 * 3   print both setup and solve information
 * Note, that if one desires to print information and uses PAMG as a preconditioner, 
 * suggested print level is 1 to avoid excessive output,
 * and use print level of solver for solve phase information.
 */
JX_Int JX_PAMGSetPrintLevel( JX_Solver solver, JX_Int print_level );
JX_Int JX_PAMGGetPrintLevel( JX_Solver solver, JX_Int *print_level );

JX_Int JX_PAMGSetPrintCoarseMatrix( JX_Solver solver, JX_Int print_coarse_matrix );

JX_Int JX_PAMGSetPrintFileName( JX_Solver solver, const char *print_file_name );
JX_Int JX_PAMGSetDebugFlag( JX_Solver solver, JX_Int debug_flag );
JX_Int JX_PAMGGetDebugFlag( JX_Solver solver, JX_Int *debug_flag );
JX_Int JX_PAMGGetNumIterations( JX_Solver solver, JX_Int *num_iterations );
JX_Int JX_PAMGGetCumNumIterations( JX_Solver solver, JX_Int *cum_num_iterations );
JX_Int JX_PAMGGetResidual( JX_Solver solver, JX_ParVector *residual );                
JX_Int JX_PAMGGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *rel_resid_norm );
JX_Int JX_PAMGSetVariant( JX_Solver solver, JX_Int variant );
JX_Int JX_PAMGGetVariant( JX_Solver solver, JX_Int *variant );
JX_Int JX_PAMGSetOverlap( JX_Solver solver, JX_Int overlap );
JX_Int JX_PAMGGetOverlap( JX_Solver solver, JX_Int *overlap );
JX_Int JX_PAMGSetDomainType( JX_Solver solver, JX_Int domain_type );
JX_Int JX_PAMGGetDomainType( JX_Solver solver, JX_Int *domain_type );
JX_Int JX_PAMGSetSchwarzRlxWeight( JX_Solver solver, JX_Real schwarz_rlx_weight );
JX_Int JX_PAMGGetSchwarzRlxWeight( JX_Solver solver, JX_Real *schwarz_rlx_weight );
JX_Int JX_PAMGSetSchwarzUseNonSymm( JX_Solver solver, JX_Int use_nonsymm );
JX_Int JX_PAMGSetSym( JX_Solver solver, JX_Int sym );
JX_Int JX_PAMGSetLevel( JX_Solver solver, JX_Int level );
JX_Int JX_PAMGSetThreshold( JX_Solver solver, JX_Real threshold );
JX_Int JX_PAMGSetFilter( JX_Solver solver, JX_Real filter );
JX_Int JX_PAMGSetDropTol( JX_Solver solver, JX_Real drop_tol );
JX_Int JX_PAMGSetMaxNzPerRow( JX_Solver solver, JX_Int max_nz_per_row );
JX_Int JX_PAMGSetEuclidFile( JX_Solver solver, char *euclidfile );
JX_Int JX_PAMGSetEuLevel( JX_Solver solver, JX_Int eu_level );
JX_Int JX_PAMGSetEuSparseA( JX_Solver solver, JX_Real eu_sparse_A );
JX_Int JX_PAMGSetEuBJ( JX_Solver solver, JX_Int eu_bj );
JX_Int JX_PAMGSetNumFunctions( JX_Solver solver, JX_Int num_functions );
JX_Int JX_PAMGGetNumFunctions( JX_Solver solver, JX_Int *num_functions );
JX_Int JX_PAMGSetNodal( JX_Solver solver, JX_Int nodal );
JX_Int JX_PAMGSetNodalDiag( JX_Solver solver, JX_Int nodal );
JX_Int JX_PAMGSetDofFunc( JX_Solver solver, JX_Int  *dof_func );
JX_Int JX_PAMGSetNumPaths( JX_Solver solver, JX_Int num_paths );
JX_Int JX_PAMGSetAggNumLevels( JX_Solver solver, JX_Int agg_num_levels );
JX_Int JX_PAMGSetAggInterpType( JX_Solver solver, JX_Int agg_interp_type );
JX_Int JX_PAMGSetAggPMaxElmts( JX_Solver solver, JX_Int agg_P_max_elmts );
JX_Int JX_PAMGSetAggP12MaxElmts( JX_Solver solver, JX_Int agg_P12_max_elmts );
JX_Int JX_PAMGSetAggTruncFactor( JX_Solver solver, JX_Real agg_trunc_factor );
JX_Int JX_PAMGSetAggP12TruncFactor( JX_Solver solver, JX_Real agg_P12_trunc_factor );
JX_Int JX_PAMGSetNumCRRelaxSteps( JX_Solver solver, JX_Int num_CR_relax_steps );
JX_Int JX_PAMGSetCRRate( JX_Solver solver, JX_Real CR_rate );
JX_Int JX_PAMGSetCRStrongTh( JX_Solver solver, JX_Real CR_strong_th );
JX_Int JX_PAMGSetISType( JX_Solver solver, JX_Int IS_type );
JX_Int JX_PAMGSetCRUseCG( JX_Solver solver, JX_Int CR_use_CG );
JX_Int JX_PAMGSetGSMG( JX_Solver solver, JX_Int gsmg );
JX_Int JX_PAMGSetNumSamples( JX_Solver solver, JX_Int gsmg );
JX_Int JX_PAMGSetCGCIts ( JX_Solver solver, JX_Int its );
// Yue Xiaoqiang 2012/10/22
JX_Int JX_PAMGSetSpMtRapType( JX_Solver solver, JX_Int spmt_rap_type );
// Yue Xiaoqiang 2015/09/30
JX_Int JX_PAMGSetWallTimeOption( JX_Solver solver, JX_Int wall_time_option );
// Yue Xiaoqiang 2014/02/26
JX_Int JX_PAMGSetAIMeasureType( JX_Solver solver, JX_Int ai_measure_type );
// Yue Xiaoqiang 2014/07/06
JX_Int JX_PAMGSetAIRelaxType( JX_Solver solver, JX_Int ai_relax_type );
// peghoty 2010/05/29
JX_Int JX_PAMGSetRelaxedCoarsenMeasureType( JX_Solver solver, JX_Int measure_type_rlx );
// peghoty 2010/05/29
JX_Int JX_PAMGSetRelaxedCoarsenNumberSyn( JX_Solver solver, JX_Int number_syn_rlx );
// peghoty 2010/05/29
JX_Int JX_PAMGSetRelaxedCoarsenMeasureThreshold( JX_Solver solver, JX_Real measure_threshold_rlx );
JX_Int JX_PAMGSetPlotGrids ( JX_Solver solver, JX_Int plotgrids );
JX_Int JX_PAMGSetPlotFileName ( JX_Solver solver, const char *plotfilename );
JX_Int JX_PAMGSetCoordDim ( JX_Solver solver, JX_Int coorddim );
JX_Int JX_PAMGSetCoordinates ( JX_Solver solver, float *coordinates );

/*!
 * Set the stopping criteria for convergence
 * There are the following options for conv_criteria:
 * ( u^{k} is the k-th approxiated solution, r^{k}=b-A*u^{k} )
 *     0:  ||r^{k}||_2 / ||r^{0}||_2                 (Relative Residual Norm-2 (default))
 *     1:  ||u^{k} - u^{k-1}||_1 / ||u^{k}||_1       (Relative Error Norm-1)
 *    11:  max { |u_i^{k} - u_i^{k-1}| / |u_i^{k}| } (Relative Point-wise Error Norm-1)
 *     2:  ||u^{k} - u^{k-1}||_2 / ||u^{k}||_2       (Relative Error Norm-2)
 *   where 
 *    ||u||_1 = max{|u_i|}， ||u||_2 = sqrt((u_1)^2 + ... + (u_n)^2)
 * The default is 0.
 * peghoty 2009/12/11
 */
JX_Int JX_PAMGSetConvCriteria( JX_Solver solver, JX_Int conv_criteria );

/*!
 * Set the solver on the coarsest level(including the case when max_levels=1)
 * There are the following options for coarsestsolverid:
 *   0 -> Jacobi or CF-Jacobi
 *   1 -> Gauss-Seidel <--- very slow, sequential
 *   2 -> Gauss_Seidel: interior points in parallel ,
 *        boundary sequential 
 *   3 -> hybrid: SOR-J mix off-processor, SOR on-processor
 *                with outer relaxation parameters (forward solve)
 *   4 -> hybrid: SOR-J mix off-processor, SOR on-processor
 *                with outer relaxation parameters (backward solve)
 *   5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
 *   6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
 *                with outer relaxation parameters 
 *   9 -> Gaussian Elimination (default)
 *  10 -> Gaussian Elimination - with pivoting
 *  The default is 9.
 *  peghoty 2010/02/27
 */
JX_Int JX_PAMGSetCoarsestSolverID( JX_Solver solver, JX_Int coarsestsolverid );

/*!
 * Set the allowed number of gridpoints in the coarsest level.
 * The default is 100.
 * peghoty  2010/04/14
 */
JX_Int JX_PAMGSetCoarseThreshold( JX_Solver solver, JX_Int coarse_threshold );


/*!
 * Set the parameter which indicates when to modify the 
 * current coarsening strategy.
 *   If coarse_size >= fine_size*coarse_ratio, we use CLJP 
 * instead of the current coarsening strategy. 
 * The default is 0.75.
 * peghoty  2010/04/14
 */
JX_Int JX_PAMGSetCoarseRatio( JX_Solver solver, JX_Real coarse_ratio );

/*
 *  * Set the parameter convfac_threshold.
 *   *   If the current convegence factor > convfac_threshold, the iteration
 *    * should be terminated at once.
 *     * 
 *      * The default is 1.0.
 *       *
 *        * peghoty  2010/06/22
 *         */
JX_Int JX_PAMGSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold );
JX_Int JX_PAMGSetRAP2( JX_Solver solver, JX_Int rap2 );
JX_Int JX_PAMGSetKeepTranspose( JX_Solver solver, JX_Int keepTranspose );

void *jx_PAMGCreate();
JX_Int jx_PAMGDestroy( void *data );
JX_Int jx_PAMGSetRestriction( void *data,JX_Int restr_par );
JX_Int jx_PAMGSetAIRStrongTh( void *data, JX_Real AIR_strong_th );
JX_Int jx_PAMGSetMaxLevels( void *data, JX_Int max_levels );
JX_Int jx_PAMGGetMaxLevels( void *data, JX_Int *max_levels );
JX_Int jx_PAMGSetStrongThreshold( void *data, JX_Real strong_threshold );
JX_Int jx_PAMGGetStrongThreshold( void *data, JX_Real *strong_threshold );
JX_Int jx_PAMGSetMaxRowSum( void *data, JX_Real max_row_sum );
JX_Int jx_PAMGGetMaxRowSum( void *data, JX_Real *max_row_sum );
JX_Int jx_PAMGSetTruncFactor( void *data, JX_Real trunc_factor );
JX_Int jx_PAMGGetTruncFactor( void *data, JX_Real *trunc_factor );
JX_Int jx_PAMGSetPMaxElmts( void *data, JX_Int P_max_elmts );
JX_Int jx_PAMGGetPMaxElmts( void *data, JX_Int *P_max_elmts );
JX_Int jx_PAMGSetJacobiTruncThreshold( void *data, JX_Real jacobi_trunc_threshold );
JX_Int jx_PAMGGetJacobiTruncThreshold( void *data, JX_Real *jacobi_trunc_threshold );
JX_Int jx_PAMGSetPostInterpType( void *data, JX_Int post_interp_type );
JX_Int jx_PAMGGetPostInterpType( void *data, JX_Int *post_interp_type );
JX_Int jx_PAMGSetSCommPkgSwitch( void *data, JX_Real S_commpkg_switch );
JX_Int jx_PAMGGetSCommPkgSwitch( void *data, JX_Real *S_commpkg_switch );
JX_Int jx_PAMGSetInterpType( void *data, JX_Int interp_type );
JX_Int jx_PAMGGetInterpType( void *data, JX_Int *interp_type );
JX_Int jx_PAMGSetMinIter( void *data, JX_Int min_iter );
JX_Int jx_PAMGGetMinIter( void *data, JX_Int *min_iter );
JX_Int jx_PAMGSetMaxIter( void *data, JX_Int max_iter );
JX_Int jx_PAMGGetMaxIter( void *data, JX_Int *max_iter );
JX_Int jx_PAMGSetCoarsenType( void *data, JX_Int coarsen_type );
JX_Int jx_PAMGGetCoarsenType( void *data, JX_Int *coarsen_type );
JX_Int jx_PAMGSetMeasureType( void *data, JX_Int measure_type );
JX_Int jx_PAMGGetMeasureType( void *data, JX_Int *measure_type );
JX_Int jx_PAMGSetSetupType( void *data, JX_Int setup_type );
JX_Int jx_PAMGGetSetupType( void *data, JX_Int *setup_type );
JX_Int jx_PAMGSetCycleType( void *data, JX_Int cycle_type );
JX_Int jx_PAMGGetCycleType( void *data, JX_Int *cycle_type );
JX_Int jx_PAMGSetTol( void *data, JX_Real tol );
JX_Int jx_PAMGSetRhsNrmThreshold( void *data, JX_Real rhsnrm_threshold );
JX_Int jx_PAMGGetTol( void *data, JX_Real *tol );
JX_Int jx_PAMGSetNumSweeps( void *data, JX_Int num_sweeps ); 
JX_Int jx_PAMGSetCycleNumSweeps( void *data, JX_Int num_sweeps, JX_Int k );
JX_Int jx_PAMGGetCycleNumSweeps( void *data, JX_Int *num_sweeps, JX_Int k );
JX_Int jx_PAMGSetNumGridSweeps( void *data, JX_Int *num_grid_sweeps );
JX_Int jx_PAMGGetNumGridSweeps( void *data, JX_Int **num_grid_sweeps );
JX_Int jx_PAMGSetRelaxType( void *data, JX_Int relax_type );
JX_Int jx_PAMGSetCycleRelaxType( void *data, JX_Int relax_type, JX_Int k );
JX_Int jx_PAMGGetCycleRelaxType( void *data, JX_Int *relax_type, JX_Int k );
JX_Int jx_PAMGSetRelaxOrder( void *data, JX_Int relax_order );
JX_Int jx_PAMGGetRelaxOrder( void *data, JX_Int *relax_order );
JX_Int jx_PAMGSetGridRelaxType( void *data, JX_Int  *grid_relax_type );
JX_Int jx_PAMGGetGridRelaxType( void *data, JX_Int **grid_relax_type );
JX_Int jx_PAMGSetGridRelaxPoints( void *data, JX_Int **grid_relax_points );
JX_Int jx_PAMGGetGridRelaxPoints( void *data, JX_Int ***grid_relax_points );
JX_Int jx_PAMGSetRelaxWeight( void *data, JX_Real *relax_weight );
JX_Int jx_PAMGGetRelaxWeight( void *data, JX_Real **relax_weight );
JX_Int jx_PAMGSetRelaxWt( void *data, JX_Real relax_weight );
JX_Int jx_PAMGSetLevelRelaxWt( void *data, JX_Real relax_weight, JX_Int level );
JX_Int jx_PAMGGetLevelRelaxWt( void *data, JX_Real *relax_weight, JX_Int level );
JX_Int jx_PAMGSetOmega( void *data, JX_Real *omega );
JX_Int jx_PAMGGetOmega( void *data, JX_Real **omega );
JX_Int jx_PAMGSetOuterWt( void *data, JX_Real omega );
JX_Int jx_PAMGSetLevelOuterWt( void *data, JX_Real omega, JX_Int level );
JX_Int jx_PAMGGetLevelOuterWt( void *data, JX_Real *omega, JX_Int level );
JX_Int jx_PAMGSetSmoothType( void *data, JX_Int smooth_type );
JX_Int jx_PAMGGetSmoothType( void *data, JX_Int *smooth_type );
JX_Int jx_PAMGSetSmoothNumLevels( void *data, JX_Int smooth_num_levels );
JX_Int jx_PAMGGetSmoothNumLevels( void *data, JX_Int *smooth_num_levels );
JX_Int jx_PAMGSetSmoothNumSweeps( void *data, JX_Int smooth_num_sweeps );
JX_Int jx_PAMGGetSmoothNumSweeps( void *data, JX_Int *smooth_num_sweeps );
JX_Int jx_PAMGSetLogging( void *data, JX_Int logging );
JX_Int jx_PAMGGetLogging( void *data, JX_Int *logging );
JX_Int jx_PAMGSetPrintLevel( void *data, JX_Int print_level );
JX_Int jx_PAMGGetPrintLevel( void *data, JX_Int *print_level );
JX_Int jx_PAMGSetPrintCoarseMatrix( void *data, JX_Int print_coarse_matrix );
JX_Int jx_PAMGSetPrintFileName( void *data, const char *print_file_name );
JX_Int jx_PAMGGetPrintFileName( void *data, char **print_file_name );
JX_Int jx_PAMGSetNumIterations( void *data, JX_Int num_iterations );
JX_Int jx_PAMGSetDebugFlag( void *data, JX_Int debug_flag );
JX_Int jx_PAMGGetDebugFlag( void *data, JX_Int  *debug_flag );
JX_Int jx_PAMGSetGSMG( void *data, JX_Int par );
JX_Int jx_PAMGSetNumSamples( void *data, JX_Int par );
JX_Int jx_PAMGSetCGCIts( void *data, JX_Int its );
// Yue Xiaoqiang 2012/10/22
JX_Int jx_PAMGSetSpMtRapType( void *data, JX_Int spmt_rap_type );
// Yue Xiaoqiang 2015/09/30
JX_Int jx_PAMGSetWallTimeOption( void *data, JX_Int wall_time_option );
// Yue Xiaoqiang 2014/02/26
JX_Int jx_PAMGSetAIMeasureType( void *amg_data, JX_Int ai_measure_type );
// Yue Xiaoqiang 2014/07/06
JX_Int jx_PAMGSetAIRelaxType( void *data, JX_Int ai_relax_type );
// peghoty 2010/05/29
JX_Int jx_PAMGSetRelaxedCoarsenMeasureType( void *data, JX_Int measure_type_rlx );
// peghoty 2010/05/29
JX_Int jx_PAMGSetRelaxedCoarsenNumberSyn( void *data, JX_Int number_syn_rlx );
// peghoty 2010/05/29
JX_Int jx_PAMGSetRelaxedCoarsenMeasureThreshold( void *data, JX_Real measure_threshold_rlx );
JX_Int jx_PAMGSetPlotGrids( void *data, JX_Int plotgrids );
JX_Int jx_PAMGSetPlotFileName( void *data, const char *plot_file_name );
JX_Int jx_PAMGSetCoordDim( void *data, JX_Int coorddim );
JX_Int jx_PAMGSetCoordinates( void *data, float *coordinates );
JX_Int jx_PAMGSetNumFunctions( void *data, JX_Int num_functions );
JX_Int jx_PAMGGetNumFunctions( void *data, JX_Int *num_functions );
JX_Int jx_PAMGSetNodal( void *data, JX_Int nodal );
JX_Int jx_PAMGSetNodalDiag( void *data, JX_Int nodal );
JX_Int jx_PAMGSetNumPaths( void *data, JX_Int num_paths );
JX_Int jx_PAMGSetAggNumLevels( void *data, JX_Int agg_num_levels );
JX_Int jx_PAMGSetAggInterpType( void *data, JX_Int agg_interp_type );
JX_Int jx_PAMGSetAggPMaxElmts( void *data, JX_Int agg_P_max_elmts );
JX_Int jx_PAMGSetAggP12MaxElmts( void *data, JX_Int agg_P12_max_elmts );
JX_Int jx_PAMGSetAggTruncFactor( void *data, JX_Real agg_trunc_factor );
JX_Int jx_PAMGSetAggP12TruncFactor( void *data, JX_Real agg_P12_trunc_factor );
JX_Int jx_PAMGSetNumCRRelaxSteps( void *data, JX_Int num_CR_relax_steps );
JX_Int jx_PAMGSetCRRate( void *data, JX_Real CR_rate );
JX_Int jx_PAMGSetCRStrongTh( void *data, JX_Real CR_strong_th );
JX_Int jx_PAMGSetISType( void *data, JX_Int IS_type );
JX_Int jx_PAMGSetCRUseCG( void *data, JX_Int CR_use_CG );
JX_Int jx_PAMGSetNumPoints( void *data, JX_Int num_points );
JX_Int jx_PAMGSetDofFunc( void *data, JX_Int *dof_func );
JX_Int jx_PAMGSetPointDofMap( void *data, JX_Int *point_dof_map );
JX_Int jx_PAMGSetDofPoint( void *data, JX_Int *dof_point );
JX_Int jx_PAMGGetNumIterations( void *data, JX_Int *num_iterations );
JX_Int jx_PAMGGetCumNumIterations( void *data, JX_Int *cum_num_iterations );
JX_Int jx_PAMGGetResidual( void *data, jx_ParVector **resid );                     
JX_Int jx_PAMGGetRelResidualNorm( void *data, JX_Real *rel_resid_norm );
JX_Int jx_PAMGSetVariant( void *data, JX_Int variant );
JX_Int jx_PAMGGetVariant( void *data, JX_Int *variant );
JX_Int jx_PAMGSetOverlap( void *data, JX_Int overlap );
JX_Int jx_PAMGGetOverlap( void *data, JX_Int *overlap );
JX_Int jx_PAMGSetDomainType( void *data, JX_Int domain_type );
JX_Int jx_PAMGGetDomainType( void *data, JX_Int *domain_type );
JX_Int jx_PAMGSetSchwarzRlxWeight( void *data, JX_Real schwarz_rlx_weight );
JX_Int jx_PAMGGetSchwarzRlxWeight( void *data, JX_Real *schwarz_rlx_weight );
JX_Int jx_PAMGSetSchwarzUseNonSymm( void *data, JX_Int use_nonsymm );
JX_Int jx_PAMGSetSym( void *data, JX_Int sym );
JX_Int jx_PAMGSetLevel( void *data, JX_Int level );
JX_Int jx_PAMGSetThreshold( void *data, JX_Real thresh );
JX_Int jx_PAMGSetFilter( void *data, JX_Real filter );
JX_Int jx_PAMGSetDropTol( void *data, JX_Real drop_tol );
JX_Int jx_PAMGSetMaxNzPerRow( void *data, JX_Int max_nz_per_row );
JX_Int jx_PAMGSetEuclidFile( void *data, char *euclidfile );
JX_Int jx_PAMGSetEuLevel( void *data, JX_Int eu_level );
JX_Int jx_PAMGSetEuSparseA( void *data, JX_Real eu_sparse_A );
JX_Int jx_PAMGSetEuBJ( void *data, JX_Int eu_bj );
/* peghoty 2009/07/27 */
JX_Int jx_PAMGSetConvCriteria( void *data, JX_Int conv_criteria );
JX_Int jx_PAMGSetCoarsestSolverID( void *data, JX_Int coarsestsolverid );
/* peghoty 2010/04/14 */
JX_Int jx_PAMGSetCoarseThreshold( void *data, JX_Int coarse_threshold );
/* peghoty 2010/04/14 */
JX_Int jx_PAMGSetCoarseRatio( void *data, JX_Real coarse_ratio );

JX_Int jx_PAMGSetConvFacThreshold( void *data, JX_Real convfac_threshold );

JX_Int jx_PAMGSetRAP2( void *data, JX_Int rap2 );
JX_Int jx_PAMGSetKeepTranspose( void *data, JX_Int keepTranspose );


/* csrc/amg/par_cparam.c */
JX_Int
jx_PAMGCoarseParms( MPI_Comm comm,
                    JX_Int      local_num_variables,
                    JX_Int      num_functions,
                    JX_Int     *dof_func,
                    JX_Int     *CF_marker, 
                    JX_Int    **coarse_dof_func_ptr, 
                    JX_Int    **coarse_pnts_global_ptr );


/* csrc/amg/par_amg_setup.c */
JX_Int
jx_PAMGSetup( void *amg_vdata, jx_ParCSRMatrix *par_matrix );


/* csrc/amg/par_amg_solve.c */
JX_Int jx_PAMGSolve( void *amg_vdata, jx_ParCSRMatrix *par_matrix, jx_ParVector *par_rhs, jx_ParVector *par_app );
// peghoty 2011/09/04
JX_Int jx_PAMGPrecond( void *amg_vdata, jx_ParCSRMatrix *par_matrix, jx_ParVector *par_rhs, jx_ParVector *par_app );


/* csrc/amg/coarsen_cljp.c */
JX_Int
jx_PAMGCoarsen( jx_ParCSRMatrix  *par_S,
                jx_ParCSRMatrix  *par_matrix,
                JX_Int               CF_init,
                JX_Int               debug_flag,
                JX_Int             **CF_marker_ptr);

/* csrc/amg/coarsen_cljp_ai.c */
JX_Int
jx_PAMGCoarsenAI( jx_ParCSRMatrix  *par_S,
                jx_ParCSRMatrix  *par_matrix,
                JX_Real           *AI_measure,
                JX_Int               CF_init,
                JX_Int               debug_flag,
                JX_Int             **CF_marker_ptr);


/* csrc/amg/coarsen_falgout.c */
JX_Int
jx_PAMGCoarsenFalgout( jx_ParCSRMatrix  *par_S,
                       jx_ParCSRMatrix  *par_matrix,
                       JX_Int               measure_type,
                       JX_Int               debug_flag,
                       JX_Int             **CF_marker_ptr );

/* csrc/amg/coarsen_falgout_ai.c */
JX_Int
jx_PAMGCoarsenFalgoutAI( jx_ParCSRMatrix  *par_S,
                       jx_ParCSRMatrix  *par_matrix,
                       JX_Real           *AI_measure,
                       JX_Int               measure_type,
                       JX_Int               debug_flag,
                       JX_Int             **CF_marker_ptr );


/* csrc/amg/coarsen_hmis.c */
JX_Int
jx_PAMGCoarsenHMIS( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Int               measure_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );
                    

/* csrc/amg/coarsen_hmis_ai.c */
JX_Int
jx_PAMGCoarsenHMISAI( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Real           *AI_measure,
                    JX_Int               CF_init,
                    JX_Int               measure_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );
                    
/* csrc/amg/coarsen_pmis.c */
JX_Int
jx_PAMGCoarsenPMIS( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Int               CF_init,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );

/* csrc/amg/coarsen_pmis_ai.c */
JX_Int
jx_PAMGCoarsenPMISAI( jx_ParCSRMatrix  *par_S,
                  jx_ParCSRMatrix  *par_matrix,
                  JX_Real           *AI_measure,
                  JX_Int               CF_init,
                  JX_Int               debug_flag,
                  JX_Int             **CF_marker_ptr );

/* csrc/amg/splitting_ai.c */
JX_Int
jx_PAMGMeasureAI( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Int               debug_flag,
                    JX_Real          **AI_measure_ptr );

/* csrc/amg/coarsen_rcljp.c */
JX_Int
jx_PAMGCoarsenRCLJP( jx_ParCSRMatrix    *S,
                     jx_ParCSRMatrix    *A,
                     JX_Int                 measure_type_rlx,
                     JX_Int                 number_syn_rlx,
                     JX_Real              measure_threshold_rlx,
                     JX_Int                 CF_init,
                     JX_Int                 debug_flag,
                     JX_Int               **CF_marker_ptr );     


/* csrc/amg/coarsen_rrs0.c */
JX_Int
jx_PAMGCoarsenRRS0( jx_ParCSRMatrix  *S,
                    jx_ParCSRMatrix  *A,
                    JX_Int               measure_type,
                    JX_Int               measure_type_rlx,
                    JX_Int               number_syn_rlx,
                    JX_Real            measure_threshold_rlx,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );                                          


/* csrc/amg/coarsen_ruge.c */
JX_Int
jx_PAMGCoarsenRuge( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Int               measure_type,
                    JX_Int               coarsen_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );

/* csrc/amg/coarsen_ruge_ai.c */
JX_Int
jx_PAMGCoarsenRugeAI( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Real           *AI_measure,
                    JX_Int                 CF_init,
                    JX_Int               measure_type,
                    JX_Int               coarsen_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );

/* csrc/amg/coarsen_xml.c */
JX_Int
jx_PAMGCoarsenXML( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Real           *AI_measure,
                    JX_Int               measure_type,
                    JX_Int               coarsen_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr );
                    

/* csrc/amg/par_csolver.c */            
JX_Int
jx_CoarsestSolver( void            *amg_vdata, 
                   jx_ParCSRMatrix *par_matrix, 
                   jx_ParVector    *par_rhs, 
                   jx_ParVector    *par_app );
JX_Int
jx_check_convergence( JX_Int             *relax_marker,
                      jx_ParCSRMatrix *par_matrix,
                      jx_ParVector    *par_rhs,
                      JX_Real           alpha,
                      JX_Real           beta,
                      JX_Int              iter,
                      jx_ParVector    *par_app,
                      jx_ParVector    *Vtemp,
                      jx_ParVector    *Utemp,
                      JX_Int              my_id,
                      JX_Int              print_level,
                      JX_Int              conv_criteria,
                      JX_Real           rhs_norm,
                      JX_Real           resid_nrm_init,
                      JX_Real           *resid_nrm,
                      JX_Real           *relative,
                      JX_Real           *conv_factor );
JX_Int jx_CoarsestSolverInfo( MPI_Comm comm, void *amg_vdata );


/* csrc/amg/par_cycle.c */   
JX_Int jx_PAMGCycle( void *amg_vdata, jx_ParVector **F_array, jx_ParVector **U_array );


/* csrc/amg/par_gesolver.c */ 
JX_Int 
jx_GaussElimination( JX_Int              solverid, 
                     jx_ParCSRMatrix *par_matrix, 
                     jx_ParVector    *par_rhs, 
                     jx_ParVector    *par_app );
                     
                     
/* csrc/amg/indepset.c */
JX_Int
jx_PAMGIndepSetInit( jx_ParCSRMatrix *par_S, JX_Real *measure_array, JX_Int seq_rand );

JX_Int
jx_PAMGIndepSet( jx_ParCSRMatrix *par_S,
                 JX_Real          *measure_array,
                 JX_Int             *graph_array,
                 JX_Int              graph_array_size,
                 JX_Int             *graph_array_offd,
                 JX_Int              graph_array_offd_size,
                 JX_Int             *IS_marker,
                 JX_Int             *IS_marker_offd );
                     
                     
/* csrc/amg/par_indepsetrlx.c */
JX_Int jx_PAMGIndepSetRelaxInit( jx_ParCSRMatrix *S, JX_Real *measure_array );
JX_Int
jx_PAMGIndepSetRelax( jx_ParCSRMatrix *S,
		      jx_CSRMatrix    *S_ext,
                      JX_Real          *measure_array,
                      JX_Int             *graph_array,
                      JX_Int              graph_array_size,
                      JX_Int             *graph_array_offd,
                      JX_Int              graph_array_offd_size,
                      JX_Int             *IS_marker,
                      JX_Int             *IS_marker_offd,
                      JX_Int              nc_per,
                      JX_Real           measure_th );
                      
                      
/* csrc/amg/par_interp_0.c */
JX_Int
jx_PAMGBuildInterp( jx_ParCSRMatrix   *par_A,
                    JX_Int               *CF_marker,
                    jx_ParCSRMatrix   *par_S,
                    JX_Int               *num_cpts_global,
                    JX_Int                num_functions,
                    JX_Int               *dof_func,
                    JX_Int                debug_flag,
                    JX_Real             trunc_factor,
                    JX_Int                max_elmts,
                    JX_Int               *col_offd_S_to_A,
                    jx_ParCSRMatrix  **P_ptr );
JX_Int jx_PAMGInterpTruncation( jx_ParCSRMatrix *par_P, JX_Real trunc_factor, JX_Int max_elmts );
JX_Int jx_GetCommPkgRTFromCommPkgA( jx_ParCSRMatrix *RT, jx_ParCSRMatrix *par_A, JX_Int *fine_to_coarse_offd );
JX_Int jx_PAMGBuildInterp1( jx_ParCSRMatrix   *par_A,
                         JX_Int               *CF_marker,
                         jx_ParCSRMatrix   *par_S,
                         JX_Int               *num_cpts_global,
                         JX_Int                num_functions,
                         JX_Int               *dof_func,
                         JX_Int                debug_flag,
                         JX_Real             trunc_factor,
                         JX_Int                max_elmts,
                         JX_Int               *col_offd_S_to_A,
                         jx_ParCSRMatrix  **P_ptr,
                         JX_Int               *opt_icor );

/* csrc/amg/par_interp_100.c */
JX_Int
jx_PAMGBuildInterpOnePnt( jx_ParCSRMatrix  *A,
                          JX_Int           *CF_marker,
                          jx_ParCSRMatrix  *S,
                          JX_Int           *num_cpts_global,
                          JX_Int            num_functions,
                          JX_Int           *dof_func,
                          JX_Int            debug_flag,
                          JX_Int           *col_offd_S_to_A,
                          jx_ParCSRMatrix **P_ptr );

/* csrc/amg/par_interp_3.c */
JX_Int
jx_PAMGBuildDirInterp( jx_ParCSRMatrix   *par_A,
                       JX_Int               *CF_marker,
                       jx_ParCSRMatrix   *par_S,
                       JX_Int               *num_cpts_global,
                       JX_Int                num_functions,
                       JX_Int               *dof_func,
                       JX_Int                debug_flag,
                       JX_Real             trunc_factor,
                       JX_Int                max_elmts,
                       JX_Int               *col_offd_S_to_A,
                       jx_ParCSRMatrix  **P_ptr );

                 
/* csrc/amg/par_interp_4.c */                                   
JX_Int
jx_PAMGBuildMultipass( jx_ParCSRMatrix  *par_A,
                       JX_Int              *CF_marker,
                       jx_ParCSRMatrix  *par_S,
                       JX_Int              *num_cpts_global,
                       JX_Int               num_functions,
                       JX_Int              *dof_func,
                       JX_Int               debug_flag,
                       JX_Real            trunc_factor,
                       JX_Int               P_max_elmts,
                       JX_Int               weight_option,
                       JX_Int              *col_offd_S_to_A,
                       jx_ParCSRMatrix **P_ptr );

                           
/* csrc/amg/par_interp_6.c */                                   
JX_Int
jx_PAMGBuildExtPIInterp( jx_ParCSRMatrix  *par_A, 
                         JX_Int              *CF_marker,
			 jx_ParCSRMatrix  *par_S, 
			 JX_Int              *num_cpts_global,
			 JX_Int               num_functions, 
			 JX_Int              *dof_func, 
			 JX_Int               debug_flag,
			 JX_Real            trunc_factor, 
			 JX_Int               max_elmts, 
			 JX_Int              *col_offd_S_to_A,
			 jx_ParCSRMatrix **P_ptr );
JX_Int 
jx_new_offd_nodes( JX_Int             **found, 
                   JX_Int               num_cols_A_offd, 
                   JX_Int              *A_ext_i, 
                   JX_Int              *A_ext_j, 
                   JX_Int               num_cols_S_offd,
		   JX_Int              *col_map_offd, 
		   JX_Int               col_1, 
		   JX_Int               col_n,
		   JX_Int              *Sop_i, 
		   JX_Int              *Sop_j, 
		   JX_Int              *CF_marker,
		   jx_ParCSRCommPkg *comm_pkg );
void 
jx_initialize_vecs( JX_Int  diag_n, 
                    JX_Int  offd_n, 
                    JX_Int *diag_ftc, 
                    JX_Int *offd_ftc, 
		    JX_Int *diag_pm, 
		    JX_Int *offd_pm, 
		    JX_Int *tmp_CF );
JX_Int jx_ssort(JX_Int *data, JX_Int size);
JX_Int jx_index_of_minimum(JX_Int *data, JX_Int size);
void jx_swap_int(JX_Int *data, JX_Int va, JX_Int vb);
JX_Int 
jx_alt_insert_new_nodes( jx_ParCSRCommPkg *comm_pkg, 
                         jx_ParCSRCommPkg *extend_comm_pkg,
                         JX_Int              *IN_marker, 
                         JX_Int               full_off_procNodes,
                         JX_Int              *OUT_marker );
JX_Int
jx_ParCSRFindExtendCommPkg( jx_ParCSRMatrix   *par_A, 
                            JX_Int                newoff, 
                            JX_Int               *found, 
                            jx_ParCSRCommPkg **extend_comm_pkg );

	    
/* csrc/amg/par_interp_7.c */ 		    
JX_Int
jx_PAMGBuildExtPICCInterp( jx_ParCSRMatrix   *par_A, 
                           JX_Int               *CF_marker,
			   jx_ParCSRMatrix   *par_S, 
			   JX_Int               *num_cpts_global,
			   JX_Int                num_functions, 
			   JX_Int               *dof_func, 
			   JX_Int                debug_flag,
			   JX_Real             trunc_factor, 
			   JX_Int                max_elmts, 
			   JX_Int               *col_offd_S_to_A,
			   jx_ParCSRMatrix  **P_ptr );


/* csrc/amg/par_interp_8.c */			               
JX_Int
jx_PAMGBuildStdInterp( jx_ParCSRMatrix   *par_A, 
                       JX_Int               *CF_marker,
                       jx_ParCSRMatrix   *par_S, 
                       JX_Int               *num_cpts_global,
                       JX_Int                num_functions, 
                       JX_Int               *dof_func, 
                       JX_Int                debug_flag,
                       JX_Real             trunc_factor, 
                       JX_Int                max_elmts, 
                       JX_Int                sep_weight, 
                       JX_Int               *col_offd_S_to_A, 
                       jx_ParCSRMatrix  **P_ptr );
                    

/* csrc/amg/par_itersolver.c */
JX_Int  
jx_CoarsestIterativeMethod( jx_ParCSRMatrix *par_matrix,
                            jx_ParVector    *par_rhs,
                            JX_Int              solverid,
                            JX_Real           weight,
                            JX_Real           omega,
                            jx_ParVector    *par_app,
                            jx_ParVector    *Vtemp );

/* csrc/amg/par_itersolver_ai.c */
JX_Int  
jx_CoarsestIterativeMethodAI( jx_ParCSRMatrix *par_matrix,
                            jx_ParVector    *par_rhs,
                            JX_Int             *relax_marker,
                            JX_Int              solverid,
                            JX_Real           weight,
                            JX_Real           omega,
                            jx_ParVector    *par_app,
                            jx_ParVector    *Vtemp );

/* csrc/amg/par_relax.c */
JX_Int  
jx_PAMGRelax( jx_ParCSRMatrix *par_matrix,
              jx_ParVector    *par_rhs,
              JX_Int             *cf_marker,
              JX_Int              relax_type,
              JX_Int              relax_points,
              JX_Real           relax_weight,
              JX_Real           omega,
              jx_ParVector    *par_app,
              jx_ParVector    *Vtemp );

/* csrc/amg/par_relax_if.c */
JX_Int  
jx_PAMGRelaxIF( jx_ParCSRMatrix *par_matrix,
                jx_ParVector    *par_rhs,
                JX_Int             *cf_marker,
                JX_Int              relax_type,
                JX_Int              relax_order,
                JX_Int              cycle_type,
                JX_Real           relax_weight,
                JX_Real           omega,
                jx_ParVector    *par_app,
                jx_ParVector    *Vtemp );

/* csrc/amg/par_relax_ai.c */
JX_Int  
jx_PAMGRelaxAI( jx_ParCSRMatrix *par_matrix,
                jx_ParVector    *par_rhs,
                JX_Int             *cf_marker,
                JX_Int             *relax_marker,
              /*JX_Real             *ai_measure,
              JX_Real           ai_measure_th, 
              JX_Real           ai_measure_th_h, */
                JX_Int              relax_type,
                JX_Int              relax_points,
                JX_Real           relax_weight,
                JX_Real           omega,
                jx_ParVector    *par_app,
                jx_ParVector    *Vtemp );  
JX_Int  
jx_PAMGRelaxIFAI( jx_ParCSRMatrix *par_matrix,
                  jx_ParVector    *par_rhs,
                  JX_Int             *cf_marker,
                  JX_Int             *relax_marker,
              /*JX_Real             *ai_measure,
              JX_Real           ai_measure_th, 
              JX_Real           ai_measure_th_h, */
                  JX_Int              relax_type,
                  JX_Int              relax_order,
                  JX_Int              cycle_type,
                  JX_Real           relax_weight,
                  JX_Real           omega,
                  jx_ParVector    *par_app,
                  jx_ParVector    *Vtemp );

/* csrc/amg/par_status.c */
JX_Int jx_PAMGSetupStatus( void *amg_vdata, jx_ParCSRMatrix *par_matrix );
JX_Int jx_PAMGSolveStatus( void *data );

/* csrc/amg/par_restr.c */
JX_Int
jx_PAMGBuildRestrAIR( jx_ParCSRMatrix   *A,
                      JX_Int            *CF_marker,
                      jx_ParCSRMatrix   *S,
                      JX_Int            *num_cpts_global,
                      JX_Int             num_functions,
                      JX_Int            *dof_func,
                      JX_Int             debug_flag,
                      JX_Real            trunc_factor,
                      JX_Int             max_elmts,
                      JX_Int            *col_offd_S_to_A,
                      jx_ParCSRMatrix  **R_ptr,
                      JX_Int            *R_max_size_ptr );
JX_Int
jx_PAMGBuildRestrDist2AIR( jx_ParCSRMatrix   *A,
                           JX_Int            *CF_marker,
                           jx_ParCSRMatrix   *S,
                           JX_Int            *num_cpts_global,
                           JX_Int             num_functions,
                           JX_Int            *dof_func,
                           JX_Int             debug_flag,
                           JX_Real            trunc_factor,
                           JX_Int             max_elmts,
                           JX_Int            *col_offd_S_to_A,
                           jx_ParCSRMatrix  **R_ptr,
                           JX_Int            *R_max_size_ptr );

/* csrc/amg/par_sthength.c */
JX_Int
jx_PAMGCreateS( jx_ParCSRMatrix    *par_A,
                JX_Real              strength_threshold,
                JX_Real              max_row_sum,
                JX_Int                 num_functions,
                JX_Int                *dof_func,
                jx_ParCSRMatrix   **S_ptr );
JX_Int
jx_PAMGCreateSabs( jx_ParCSRMatrix  *A,
                   JX_Real           strength_threshold,
                   JX_Real           max_row_sum,
                   JX_Int            num_functions,
                   JX_Int           *dof_func,
                   jx_ParCSRMatrix **S_ptr );
JX_Int jx_PAMGCreateSCommPkg( jx_ParCSRMatrix *par_A, jx_ParCSRMatrix *par_S, JX_Int **col_offd_S_to_A_ptr );
JX_Int
jx_PAMGCreate2ndS( jx_ParCSRMatrix  *S,
                   JX_Int              *CF_marker,
                   JX_Int               num_paths,
                   JX_Int              *coarse_row_starts,
                   jx_ParCSRMatrix **C_ptr );
JX_Int jx_PAMGCorrectCFMarker( JX_Int *CF_marker, JX_Int num_var, JX_Int *new_CF_marker );



#endif
