//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_amg.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"


/*!
 * \fn JX_Int JX_PAMGCreate
 * \date 2011/09/03
 */
JX_Int
JX_PAMGCreate( JX_Solver *solver)
{
   *solver = (JX_Solver) jx_PAMGCreate( ) ;
   if (!solver)
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_PAMGDestroy
 * \date 2011/09/03
 */
JX_Int 
JX_PAMGDestroy( JX_Solver solver )
{
   return( jx_PAMGDestroy( (void *) solver ) );
}

/*!
 * \fn JX_Int JX_PAMGSetup
 * \date 2011/09/03
 */
JX_Int 
JX_PAMGSetup( JX_Solver        solver, 
              JX_ParCSRMatrix  par_matrix )
{
   return( jx_PAMGSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int JX_PAMGSolve
 * \date 2011/09/03
 */
JX_Int 
JX_PAMGSolve( JX_Solver       solver,
              JX_ParCSRMatrix par_matrix,
              JX_ParVector    par_rhs,
              JX_ParVector    par_app  )
{
   return( jx_PAMGSolve( (void *) solver,
                         (jx_ParCSRMatrix *) par_matrix,
                         (jx_ParVector *) par_rhs,
                         (jx_ParVector *) par_app ) );
}

/*!
 * \fn JX_Int JX_PAMGPrecond
 * \author peghoty
 * \date 2011/09/04
 */
JX_Int 
JX_PAMGPrecond( JX_Solver       solver,
                JX_ParCSRMatrix par_matrix,
                JX_ParVector    par_rhs,
                JX_ParVector    par_app  )
{
   return( jx_PAMGPrecond( (void *) solver,
                           (jx_ParCSRMatrix *) par_matrix,
                           (jx_ParVector *) par_rhs,
                           (jx_ParVector *) par_app ) );
}

/*!
 * \fn void *jx_PAMGCreate
 * \date 2011/09/03
 */
void *
jx_PAMGCreate()
{
   jx_ParAMGData  *amg_data;

   /* setup params */
   JX_Int      max_levels;
   JX_Real   strong_threshold;
   JX_Real   max_row_sum;
   JX_Real   trunc_factor;
   JX_Real   jacobi_trunc_threshold;
   JX_Real   S_commpkg_switch;
   JX_Real   CR_rate;
   JX_Real   CR_strong_th;
   JX_Real   AIR_strong_th;
   JX_Int      interp_type;
   JX_Int      coarsen_type;
   JX_Int      measure_type;
   JX_Int      setup_type;
   JX_Int      P_max_elmts;
   JX_Int      num_functions;
   JX_Int      nodal, nodal_diag;
   JX_Int      num_paths;
   JX_Int      agg_num_levels;
   JX_Int      agg_interp_type;
   JX_Int      agg_P_max_elmts;
   JX_Int      agg_P12_max_elmts;
   JX_Real   agg_trunc_factor;
   JX_Real   agg_P12_trunc_factor;
   JX_Int      post_interp_type;
   JX_Int      num_CR_relax_steps;
   JX_Int      IS_type;
   JX_Int      CR_use_CG;
   JX_Int      cgc_its;
   JX_Int      spmt_rap_type;          // Yue Xiaoqiang 2012/10/22
   JX_Int      wall_time_option;       // Yue Xiaoqiang 2015/09/30
   JX_Int      ai_measure_type;        // Yue Xiaoqiang 2014/02/26
   JX_Int      ai_relax_type;          // Yue Xiaoqiang 2014/07/06
   JX_Int      measure_type_rlx;       // peghoty 2010/05/29
   JX_Int      number_syn_rlx;         // peghoty 2010/05/29
   JX_Real   measure_threshold_rlx;  // peghoty 2010/05/29   

   /* solve params */
   JX_Int      min_iter;
   JX_Int      max_iter;
   JX_Int      cycle_type;    
 
   JX_Real   tol;
   JX_Real   rhsnrm_threshold; /* peghoty, 2011/09/28 */

   JX_Int      num_sweeps;  
   JX_Int      relax_type;   
   JX_Int      relax_order;   
   JX_Real   relax_wt;
   JX_Real   outer_wt;
   JX_Int      smooth_type;
   JX_Int      smooth_num_levels;
   JX_Int      smooth_num_sweeps;

   JX_Int      variant, overlap, domain_type, schwarz_use_nonsymm;
   JX_Real   schwarz_rlx_weight;
   JX_Int      level, sym;
   JX_Int      eu_level, eu_bj;
   JX_Int      max_nz_per_row;
   JX_Real   thresh, filter;
   JX_Real   drop_tol;
   JX_Real   eu_sparse_A;
   char    *euclidfile;

   JX_Int block_mode;
   
   /* log info */
   JX_Int      num_iterations;
   JX_Int      cum_num_iterations;

   /* output params */
   JX_Int      print_level;
   JX_Int      print_coarse_matrix;
   JX_Int      logging;
   char     log_file_name[256];
   char     plot_file_name[251];
   JX_Int      debug_flag;
   
   JX_Int      conv_criteria;     /* peghoty  2009/07/25 */
   JX_Int      coarsestsolverid;  /* peghoty  2009/07/27 */
   JX_Int      coarse_threshold;  /* peghoty  2010/04/14 */
   JX_Real   coarse_ratio;      /* peghoty  2010/04/14 */
   JX_Real   convfac_threshold;      /* peghoty  2010/04/14 */

  /*-----------------------------------------------------------------------
   * Setup default values for parameters
   *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   strong_threshold = 0.25;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   jacobi_trunc_threshold = 0.01;
   S_commpkg_switch = 1.0;
   AIR_strong_th = 0.1;
   interp_type = 0;
   coarsen_type = 6;
   measure_type = 0;
   setup_type = 1;
   P_max_elmts = 0;
   num_functions = 1;
   nodal = 0;
   nodal_diag = 0;
   num_paths = 1;
   agg_num_levels = 0;
   agg_interp_type = 4;
   agg_P_max_elmts = 0;
   agg_P12_max_elmts = 0;
   agg_trunc_factor = 0.0;
   agg_P12_trunc_factor = 0.0;
   post_interp_type = 0;
   num_CR_relax_steps = 2;
   CR_rate = 0.7;
   CR_strong_th = 0;
   IS_type = 1;
   CR_use_CG = 0;
   cgc_its = 1;
   spmt_rap_type = 1;            // Yue Xiaoqiang 2012/10/22
   wall_time_option = 0;         // Yue Xiaoqiang 2015/09/30
   ai_measure_type = 0;          // Yue Xiaoqiang 2014/02/26
   ai_relax_type = 0;            // Yue Xiaoqiang 2014/07/06
   measure_type_rlx = 1;         // peghoty 2010/05/29
   number_syn_rlx = 2;           // peghoty 2010/05/29
   measure_threshold_rlx = 0.8;  // peghoty 2010/05/29

   variant = 0;
   overlap = 1;
   domain_type = 2;
   schwarz_rlx_weight = 1.0;
   smooth_num_sweeps = 1;
   smooth_num_levels = 0;
   smooth_type = 6;
   schwarz_use_nonsymm = 0;
   
   level = 1;
   sym = 0;
   thresh = 0.1;
   filter = 0.05;
   drop_tol = 0.0001;
   max_nz_per_row = 20;
   euclidfile = NULL;
   eu_level = 0;
   eu_sparse_A = 0.0;
   eu_bj = 0;

   /* solve params */
   min_iter  = 0;
   max_iter  = 20;
   cycle_type = 1;
   tol = 1.0e-7;
   rhsnrm_threshold = 0.0;

   num_sweeps = 1;
   relax_type = 3;
   relax_order = 1;
   relax_wt = 1.0;
   outer_wt = 1.0;

   block_mode = 0;

   /* log info */
   num_iterations = 0;
   cum_num_iterations = 0;

   /* output params */
   print_level = 0;
   print_coarse_matrix = 0;
   logging = 0;
   jx_sprintf(log_file_name, "%s", "amg.out.log");
   jx_sprintf(plot_file_name, "%s", "amg.plot.log"); /* peghoty 2009/08/25 */
   debug_flag = 0;

   
   conv_criteria    = 0;    /* peghoty 2009/07/25 */
   coarsestsolverid = 9;    /* peghoty 2009/07/27 */
   coarse_threshold = 9;    /* peghoty 2010/04/14 */
   coarse_ratio     = 0.75; /* peghoty 2010/04/14 */    

   convfac_threshold = 1.0e4;

  /*-----------------------------------------------------------------------
   * Create the jx_ParAMGData structure and return
   *-----------------------------------------------------------------------*/

   amg_data = jx_CTAlloc(jx_ParAMGData, 1);

   jx_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   jx_ParAMGDataUserRelaxType(amg_data) = -1;
   jx_PAMGSetMaxLevels(amg_data, max_levels);
   jx_PAMGSetStrongThreshold(amg_data, strong_threshold);
   jx_PAMGSetMaxRowSum(amg_data, max_row_sum);
   jx_PAMGSetTruncFactor(amg_data, trunc_factor);
   jx_PAMGSetJacobiTruncThreshold(amg_data, jacobi_trunc_threshold);
   jx_PAMGSetSCommPkgSwitch(amg_data, S_commpkg_switch);
   jx_PAMGSetInterpType(amg_data, interp_type);
   jx_PAMGSetMeasureType(amg_data, measure_type);
   jx_PAMGSetCoarsenType(amg_data, coarsen_type);
   jx_PAMGSetSetupType(amg_data, setup_type);
   jx_PAMGSetPMaxElmts(amg_data, P_max_elmts);
   jx_PAMGSetNumFunctions(amg_data, num_functions);
   jx_PAMGSetNodal(amg_data, nodal);
   jx_PAMGSetNodal(amg_data, nodal_diag);
   jx_PAMGSetNumPaths(amg_data, num_paths);
   jx_PAMGSetAggNumLevels(amg_data, agg_num_levels);
   jx_PAMGSetAggInterpType(amg_data, agg_interp_type);
   jx_PAMGSetAggPMaxElmts(amg_data, agg_P_max_elmts);
   jx_PAMGSetAggP12MaxElmts(amg_data, agg_P12_max_elmts);
   jx_PAMGSetAggTruncFactor(amg_data, agg_trunc_factor);
   jx_PAMGSetAggP12TruncFactor(amg_data, agg_P12_trunc_factor);
   jx_PAMGSetPostInterpType(amg_data, post_interp_type);
   jx_PAMGSetNumCRRelaxSteps(amg_data, num_CR_relax_steps);
   jx_PAMGSetCRRate(amg_data, CR_rate);
   jx_PAMGSetCRStrongTh(amg_data, CR_strong_th);
   jx_PAMGSetISType(amg_data, IS_type);
   jx_PAMGSetCRUseCG(amg_data, CR_use_CG);
   jx_PAMGSetCGCIts(amg_data, cgc_its);
   
   jx_PAMGSetSpMtRapType(amg_data, spmt_rap_type); // Yue Xiaoqiang 2012/10/13
   jx_PAMGSetWallTimeOption(amg_data, wall_time_option); // Yue Xiaoqiang 2015/09/30
   jx_PAMGSetAIMeasureType(amg_data, ai_measure_type); // Yue Xiaoqiang 2014/02/26
   jx_PAMGSetAIRelaxType(amg_data, ai_relax_type); // Yue Xiaoqiang 2014/07/06
   
   jx_PAMGSetRelaxedCoarsenMeasureType(amg_data,measure_type_rlx);           // peghoty 2010/05/29
   jx_PAMGSetRelaxedCoarsenNumberSyn(amg_data,number_syn_rlx);               // peghoty 2010/05/29
   jx_PAMGSetRelaxedCoarsenMeasureThreshold(amg_data,measure_threshold_rlx); // peghoty 2010/05/29

   jx_PAMGSetVariant(amg_data, variant);
   jx_PAMGSetOverlap(amg_data, overlap);
   jx_PAMGSetSchwarzRlxWeight(amg_data, schwarz_rlx_weight);
   jx_PAMGSetSchwarzUseNonSymm(amg_data, schwarz_use_nonsymm);
   jx_PAMGSetDomainType(amg_data, domain_type);
   jx_PAMGSetSym(amg_data, sym);
   jx_PAMGSetLevel(amg_data, level);
   jx_PAMGSetThreshold(amg_data, thresh);
   jx_PAMGSetFilter(amg_data, filter);
   jx_PAMGSetDropTol(amg_data, drop_tol);
   jx_PAMGSetMaxNzPerRow(amg_data, max_nz_per_row);
   jx_PAMGSetEuclidFile(amg_data, euclidfile);
   jx_PAMGSetEuLevel(amg_data, eu_level);
   jx_PAMGSetEuSparseA(amg_data, eu_sparse_A);
   jx_PAMGSetEuBJ(amg_data, eu_bj);

   jx_PAMGSetMinIter(amg_data, min_iter);
   jx_PAMGSetMaxIter(amg_data, max_iter);
   jx_PAMGSetCycleType(amg_data, cycle_type);
   jx_PAMGSetTol(amg_data, tol); 
   jx_PAMGSetRhsNrmThreshold(amg_data, rhsnrm_threshold);
   jx_PAMGSetNumSweeps(amg_data, num_sweeps);
   jx_PAMGSetRelaxType(amg_data, relax_type);
   jx_PAMGSetRelaxOrder(amg_data, relax_order);
   jx_PAMGSetRelaxWt(amg_data, relax_wt);
   jx_PAMGSetOuterWt(amg_data, outer_wt);
   jx_PAMGSetSmoothType(amg_data, smooth_type);
   jx_PAMGSetSmoothNumLevels(amg_data, smooth_num_levels);
   jx_PAMGSetSmoothNumSweeps(amg_data, smooth_num_sweeps);

   jx_PAMGSetNumIterations(amg_data, num_iterations);
#ifdef CUMNUMIT
   jx_ParAMGDataCumNumIterations(amg_data) = cum_num_iterations;
#endif
   jx_PAMGSetPrintLevel(amg_data, print_level);
   jx_PAMGSetPrintCoarseMatrix(amg_data, print_coarse_matrix);
   jx_PAMGSetLogging(amg_data, logging);
   jx_PAMGSetPrintFileName(amg_data, log_file_name); 
   jx_PAMGSetDebugFlag(amg_data, debug_flag);
   
   jx_PAMGSetConvCriteria(amg_data, conv_criteria);        /* peghoty 2009/07/25 */
   jx_PAMGSetCoarsestSolverID(amg_data, coarsestsolverid); /* peghoty 2009/07/27 */
   jx_PAMGSetCoarseThreshold(amg_data, coarse_threshold ); /* peghoty 2010/04/14 */
   jx_PAMGSetCoarseRatio(amg_data, coarse_ratio );         /* peghoty 2010/04/14 */

   jx_PAMGSetConvFacThreshold(amg_data, convfac_threshold);

   jx_PAMGSetRestriction(amg_data, 0);
   jx_PAMGSetAIRStrongTh(amg_data, AIR_strong_th);


   jx_PAMGSetGSMG(amg_data, 0);
   jx_PAMGSetNumSamples(amg_data, 0);
   
   jx_ParAMGDataAArray(amg_data) = NULL;
   jx_ParAMGDataPArray(amg_data) = NULL;
   jx_ParAMGDataRArray(amg_data) = NULL;
   jx_ParAMGDataAIMeasureArray(amg_data) = NULL;
   jx_ParAMGDataCFMarkerArray(amg_data) = NULL;
   jx_ParAMGDataVtemp(amg_data)  = NULL;
   jx_ParAMGDataRtemp(amg_data)  = NULL;
   jx_ParAMGDataPtemp(amg_data)  = NULL;
   jx_ParAMGDataZtemp(amg_data)  = NULL;
   jx_ParAMGDataFArray(amg_data) = NULL;
   jx_ParAMGDataUArray(amg_data) = NULL;
   jx_ParAMGDataDofFunc(amg_data) = NULL;
   jx_ParAMGDataDofFuncArray(amg_data) = NULL;
   jx_ParAMGDataDofPointArray(amg_data) = NULL;
   jx_ParAMGDataDofPointArray(amg_data) = NULL;
   jx_ParAMGDataPointDofMapArray(amg_data) = NULL;
   jx_ParAMGDataSmoother(amg_data) = NULL;
   jx_ParAMGDataL1Norms(amg_data) = NULL;
  
   jx_ParAMGDataABlockArray(amg_data) = NULL;
   jx_ParAMGDataPBlockArray(amg_data) = NULL;
   jx_ParAMGDataRBlockArray(amg_data) = NULL;

   /* this can not be set by the user currently */
   jx_ParAMGDataBlockMode(amg_data) = block_mode;

   /* BM Oct 22, 2006 */
   jx_ParAMGDataPlotGrids(amg_data) = 0;
   jx_PAMGSetPlotFileName(amg_data, plot_file_name);

   /* BM Oct 17, 2006 */
   jx_ParAMGDataCoordDim(amg_data) = 0;
   jx_ParAMGDataCoordinates(amg_data) = NULL;

   jx_ParAMGDataRAP2(amg_data) = 0;
   jx_ParAMGDataKeepTranspose(amg_data) = 0;

   return (void *) amg_data;
}

/*!
 * \fn JX_Int jx_PAMGDestroy
 * \date 2011/09/03
 */
JX_Int
jx_PAMGDestroy( void *data )
{
   jx_ParAMGData  *amg_data = data;
   JX_Int num_levels = jx_ParAMGDataNumLevels(amg_data);
   JX_Int i;

   if (jx_ParAMGDataNumGridSweeps(amg_data))
   {
      jx_TFree (jx_ParAMGDataNumGridSweeps(amg_data));
      jx_ParAMGDataNumGridSweeps(amg_data) = NULL; 
   }
   if (jx_ParAMGDataGridRelaxType(amg_data))
   {
      jx_TFree (jx_ParAMGDataGridRelaxType(amg_data));
      jx_ParAMGDataGridRelaxType(amg_data) = NULL; 
   }
   if (jx_ParAMGDataRelaxWeight(amg_data))
   {
      jx_TFree (jx_ParAMGDataRelaxWeight(amg_data));
      jx_ParAMGDataRelaxWeight(amg_data) = NULL; 
   }
   if (jx_ParAMGDataOmega(amg_data))
   {
      jx_TFree (jx_ParAMGDataOmega(amg_data));
      jx_ParAMGDataOmega(amg_data) = NULL; 
   }
   if (jx_ParAMGDataDofFunc(amg_data))
   {
      jx_TFree (jx_ParAMGDataDofFunc(amg_data));
      jx_ParAMGDataDofFunc(amg_data) = NULL; 
   }
   if (jx_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i = 0; i < 4; i ++)
      {
   	 jx_TFree (jx_ParAMGDataGridRelaxPoints(amg_data)[i]);
      }
      jx_TFree (jx_ParAMGDataGridRelaxPoints(amg_data));
      jx_ParAMGDataGridRelaxPoints(amg_data) = NULL; 
   }
   for (i = 1; i < num_levels; i ++)
   {
	jx_ParVectorDestroy(jx_ParAMGDataFArray(amg_data)[i]);
	jx_ParVectorDestroy(jx_ParAMGDataUArray(amg_data)[i]);

        if (jx_ParAMGDataAArray(amg_data)[i])
        {
           jx_ParCSRMatrixDestroy(jx_ParAMGDataAArray(amg_data)[i]);
        }

        if (jx_ParAMGDataPArray(amg_data)[i-1])
        {
            jx_ParCSRMatrixDestroy(jx_ParAMGDataPArray(amg_data)[i-1]);
        }

        if (jx_ParAMGDataRestriction(amg_data))
        {
           if (jx_ParAMGDataRArray(amg_data)[i-1])
           {
              jx_ParCSRMatrixDestroy(jx_ParAMGDataRArray(amg_data)[i-1]);
           }
           jx_TFree(jx_ParAMGDataAIRMaxSizeLS(amg_data));
        }

        if (jx_ParAMGDataAIMeasureArray(amg_data)[i-1])
        {
            jx_TFree(jx_ParAMGDataAIMeasureArray(amg_data)[i-1]);
        }

	jx_TFree(jx_ParAMGDataCFMarkerArray(amg_data)[i-1]);
   }

   if (jx_ParAMGDataL1Norms(amg_data))
   {
      for (i = 0; i < num_levels; i ++)
      {
         if (jx_ParAMGDataL1Norms(amg_data)[i])
         {
            jx_TFree(jx_ParAMGDataL1Norms(amg_data)[i]);
         }
      }
      jx_TFree(jx_ParAMGDataL1Norms(amg_data));
   }

   /* see comments in par_coarsen.c regarding special case for CF_marker */
   if (num_levels == 1)
   {
      jx_TFree(jx_ParAMGDataAIMeasureArray(amg_data)[0]);
      jx_TFree(jx_ParAMGDataCFMarkerArray(amg_data)[0]);
   }
   jx_ParVectorDestroy(jx_ParAMGDataVtemp(amg_data));
   jx_TFree(jx_ParAMGDataFArray(amg_data));
   jx_TFree(jx_ParAMGDataUArray(amg_data));
   jx_TFree(jx_ParAMGDataAArray(amg_data));
   jx_TFree(jx_ParAMGDataPArray(amg_data));
   if (jx_ParAMGDataRestriction(amg_data)) jx_TFree(jx_ParAMGDataRArray(amg_data));
   jx_TFree(jx_ParAMGDataAIMeasureArray(amg_data));
   jx_TFree(jx_ParAMGDataCFMarkerArray(amg_data));
   if (jx_ParAMGDataRtemp(amg_data))
   {
      jx_ParVectorDestroy(jx_ParAMGDataRtemp(amg_data));
   }
   if (jx_ParAMGDataPtemp(amg_data))
   {
      jx_ParVectorDestroy(jx_ParAMGDataPtemp(amg_data));
   }
   if (jx_ParAMGDataZtemp(amg_data))
   {
      jx_ParVectorDestroy(jx_ParAMGDataZtemp(amg_data));
   }

   if (jx_ParAMGDataDofFuncArray(amg_data))
   {
      for (i = 1; i < num_levels; i ++)
      {
         jx_TFree(jx_ParAMGDataDofFuncArray(amg_data)[i]);
      }
      jx_TFree(jx_ParAMGDataDofFuncArray(amg_data));
      jx_ParAMGDataDofFuncArray(amg_data) = NULL;
   }
   if (jx_ParAMGDataRestriction(amg_data))
   {
      jx_TFree(jx_ParAMGDataRArray(amg_data));
      jx_ParAMGDataRArray(amg_data) = NULL;
   }
   if (jx_ParAMGDataDofPointArray(amg_data))
   {
      for (i = 0; i < num_levels; i ++)
      {
         jx_TFree(jx_ParAMGDataDofPointArray(amg_data)[i]);
      }
      jx_TFree(jx_ParAMGDataDofPointArray(amg_data));
      jx_ParAMGDataDofPointArray(amg_data) = NULL;
   }
   if (jx_ParAMGDataPointDofMapArray(amg_data))
   {
      for (i = 0; i < num_levels; i ++)
      {
         jx_TFree(jx_ParAMGDataPointDofMapArray(amg_data)[i]);
      }
      jx_TFree(jx_ParAMGDataPointDofMapArray(amg_data));
      jx_ParAMGDataPointDofMapArray(amg_data) = NULL;
   }
   if ( jx_ParAMGDataResidual(amg_data) ) 
   {
      jx_ParVectorDestroy( jx_ParAMGDataResidual(amg_data) );
      jx_ParAMGDataResidual(amg_data) = NULL;
   }

   jx_TFree(amg_data);
   return jx_error_flag;
}



/*!
 * \fn JX_Int JX_PAMGSetXXXX
 * \brief  Routines to set the setup phase parameters begin with "JX_"
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetRestriction( JX_Solver solver, JX_Int restr_par )
{
   return( jx_PAMGSetRestriction( (void *) solver, restr_par ) );
}


JX_Int
JX_PAMGSetAIRStrongTh( JX_Solver solver, JX_Real AIR_strong_th )
{
   return( jx_PAMGSetAIRStrongTh( (void *) solver, AIR_strong_th ) );
}


JX_Int
JX_PAMGSetMaxLevels( JX_Solver solver, JX_Int  max_levels )
{
   return( jx_PAMGSetMaxLevels( (void *) solver, max_levels ) );
}

JX_Int
JX_PAMGGetMaxLevels( JX_Solver solver, JX_Int *max_levels )
{
   return( jx_PAMGGetMaxLevels( (void *) solver, max_levels ) );
}

JX_Int
JX_PAMGSetStrongThreshold( JX_Solver solver, JX_Real strong_threshold  )
{
   return( jx_PAMGSetStrongThreshold( (void *) solver, strong_threshold ) );
}

JX_Int
JX_PAMGGetStrongThreshold( JX_Solver solver, JX_Real *strong_threshold )
{
   return( jx_PAMGGetStrongThreshold( (void *) solver, strong_threshold ) );
}

JX_Int
JX_PAMGSetMaxRowSum( JX_Solver solver, JX_Real max_row_sum )
{
   return( jx_PAMGSetMaxRowSum( (void *) solver, max_row_sum ) );
}

JX_Int
JX_PAMGGetMaxRowSum( JX_Solver solver, JX_Real *max_row_sum )
{
   return( jx_PAMGGetMaxRowSum( (void *) solver, max_row_sum ) );
}

JX_Int
JX_PAMGSetTruncFactor( JX_Solver solver, JX_Real trunc_factor )
{
   return( jx_PAMGSetTruncFactor( (void *) solver, trunc_factor ) );
}

JX_Int
JX_PAMGGetTruncFactor( JX_Solver solver, JX_Real *trunc_factor )
{
   return( jx_PAMGGetTruncFactor( (void *) solver, trunc_factor ) );
}

JX_Int
JX_PAMGSetPMaxElmts( JX_Solver solver, JX_Int P_max_elmts )
{
   return( jx_PAMGSetPMaxElmts( (void *) solver, P_max_elmts ) );
}

JX_Int
JX_PAMGGetPMaxElmts( JX_Solver solver, JX_Int *P_max_elmts )
{
   return( jx_PAMGGetPMaxElmts( (void *) solver, P_max_elmts ) );
}

JX_Int
JX_PAMGSetJacobiTruncThreshold( JX_Solver solver, JX_Real jacobi_trunc_threshold )
{
   return( jx_PAMGSetJacobiTruncThreshold( (void *) solver, jacobi_trunc_threshold ) );
}

JX_Int
JX_PAMGGetJacobiTruncThreshold( JX_Solver solver, JX_Real *jacobi_trunc_threshold )
{
   return( jx_PAMGGetJacobiTruncThreshold( (void *) solver, jacobi_trunc_threshold ) );
}

/*!
 * \fn JX_Int JX_PAMGSetPostInterpType
 * \brief If > 0, specifies something to do to improve a computed 
 *        interpolation matrix. defaults to 0, for nothing.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetPostInterpType( JX_Solver solver, JX_Int post_interp_type )
{
   return( jx_PAMGSetPostInterpType( (void *) solver, post_interp_type ) );
}

JX_Int
JX_PAMGGetPostInterpType( JX_Solver solver, JX_Int *post_interp_type )
{
   return( jx_PAMGGetPostInterpType( (void *) solver, post_interp_type ) );
}

JX_Int
JX_PAMGSetSCommPkgSwitch( JX_Solver solver, JX_Real S_commpkg_switch )
{
   return( jx_PAMGSetSCommPkgSwitch( (void *) solver, S_commpkg_switch ) );
}

JX_Int
JX_PAMGSetInterpType( JX_Solver solver, JX_Int interp_type )
{
   return( jx_PAMGSetInterpType( (void *) solver, interp_type ) );
}

JX_Int
JX_PAMGSetMinIter( JX_Solver solver, JX_Int min_iter )
{
   return( jx_PAMGSetMinIter( (void *) solver, min_iter ) );
}

JX_Int
JX_PAMGSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_PAMGSetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_PAMGGetMaxIter( JX_Solver solver, JX_Int *max_iter )
{
   return( jx_PAMGGetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_PAMGSetCoarsenType( JX_Solver solver, JX_Int coarsen_type )
{
   return( jx_PAMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

JX_Int
JX_PAMGGetCoarsenType( JX_Solver solver, JX_Int *coarsen_type )
{
   return( jx_PAMGGetCoarsenType( (void *) solver, coarsen_type ) );
}

JX_Int
JX_PAMGSetMeasureType( JX_Solver solver, JX_Int measure_type )
{
   return( jx_PAMGSetMeasureType( (void *) solver, measure_type ) );
}

JX_Int
JX_PAMGGetMeasureType( JX_Solver solver, JX_Int *measure_type )
{
   return( jx_PAMGGetMeasureType( (void *) solver, measure_type ) );
}

JX_Int
JX_PAMGSetSetupType( JX_Solver solver, JX_Int setup_type )
{
   return( jx_PAMGSetSetupType( (void *) solver, setup_type ) );
}

JX_Int
JX_PAMGSetCycleType( JX_Solver solver,JX_Int cycle_type )
{
   return( jx_PAMGSetCycleType( (void *) solver, cycle_type ) );
}

JX_Int
JX_PAMGGetCycleType( JX_Solver solver, JX_Int *cycle_type )
{
   return( jx_PAMGGetCycleType( (void *) solver, cycle_type ) );
}

JX_Int
JX_PAMGSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_PAMGSetTol( (void *) solver, tol ) );
}

JX_Int
JX_PAMGSetRhsNrmThreshold( JX_Solver solver, JX_Real rhsnrm_threshold )
{
   return( jx_PAMGSetRhsNrmThreshold( (void *) solver, rhsnrm_threshold ) );
}

JX_Int
JX_PAMGGetTol( JX_Solver solver, JX_Real *tol )
{
   return( jx_PAMGGetTol( (void *) solver, tol ) );
}

/*!
 * \fn JX_Int JX_PAMGSetNumGridSweeps
 * \note DEPRECATED.  There are memory management problems associated with the
 *       use of a user-supplied array (who releases it?).
 *       Use SetNumSweeps and SetCycleNumSweeps instead.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetNumGridSweeps( JX_Solver  solver, JX_Int *num_grid_sweeps )
{
   return( jx_PAMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*!
 * \fn JX_Int JX_PAMGSetNumSweeps
 * \note There is no corresponding Get function.  Use GetCycleNumSweeps.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetNumSweeps( JX_Solver  solver, JX_Int num_sweeps )
{
   return( jx_PAMGSetNumSweeps( (void *) solver, num_sweeps ) );
}

JX_Int
JX_PAMGSetCycleNumSweeps( JX_Solver solver, JX_Int num_sweeps, JX_Int k )
{
   return( jx_PAMGSetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

JX_Int
JX_PAMGGetCycleNumSweeps( JX_Solver solver, JX_Int *num_sweeps, JX_Int k )
{
   return( jx_PAMGGetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

/*!
 * \fn JX_Int JX_PAMGSetGridRelaxType
 * \note DEPRECATED.  There are memory management problems associated with the
 *       use of a user-supplied array (who releases it?).
 *       Use SetRelaxType and SetCycleRelaxType instead.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetGridRelaxType( JX_Solver  solver, JX_Int *grid_relax_type )
{
   return( jx_PAMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

JX_Int
JX_PAMGSetRelaxType( JX_Solver solver, JX_Int relax_type  )
{
   return( jx_PAMGSetRelaxType( (void *) solver, relax_type ) );
}

JX_Int
JX_PAMGSetCycleRelaxType( JX_Solver solver, JX_Int relax_type, JX_Int k )
{
   return( jx_PAMGSetCycleRelaxType( (void *) solver, relax_type, k ) );
}

JX_Int
JX_PAMGGetCycleRelaxType( JX_Solver solver, JX_Int * relax_type, JX_Int k )
{
   return( jx_PAMGGetCycleRelaxType( (void *) solver, relax_type, k ) );
}

JX_Int
JX_PAMGSetRelaxOrder( JX_Solver solver, JX_Int relax_order )
{
   return( jx_PAMGSetRelaxOrder( (void *) solver, relax_order ) );
}

/*!
 * \fn JX_Int JX_PAMGSetGridRelaxPoints
 * \note DEPRECATED.  There are memory management problems associated with the
 *       use of a user-supplied array (who releases it?).
 *       Ulrike Yang suspects that nobody uses this function.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetGridRelaxPoints( JX_Solver solver, JX_Int **grid_relax_points )
{
   return( jx_PAMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*!
 * \fn JX_Int JX_PAMGSetRelaxWeight
 * \note DEPRECATED.  There are memory management problems associated with the
 *       use of a user-supplied array (who releases it?).
 *       Use SetRelaxWt and SetLevelRelaxWt instead.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetRelaxWeight( JX_Solver solver, JX_Real *relax_weight )
{
   return( jx_PAMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

JX_Int
JX_PAMGSetRelaxWt( JX_Solver solver, JX_Real relax_wt )
{
   return( jx_PAMGSetRelaxWt( (void *) solver, relax_wt ) );
}

JX_Int
JX_PAMGSetLevelRelaxWt( JX_Solver solver, JX_Real relax_wt, JX_Int level )
{
   return( jx_PAMGSetLevelRelaxWt( (void *) solver, relax_wt, level ) );
}

JX_Int
JX_PAMGSetOmega( JX_Solver solver, JX_Real *omega )
{
   return( jx_PAMGSetOmega( (void *) solver, omega ) );
}

JX_Int
JX_PAMGSetOuterWt( JX_Solver solver, JX_Real outer_wt )
{
   return( jx_PAMGSetOuterWt( (void *) solver, outer_wt ) );
}

JX_Int
JX_PAMGSetLevelOuterWt( JX_Solver solver, JX_Real outer_wt, JX_Int level )
{
   return( jx_PAMGSetLevelOuterWt( (void *) solver, outer_wt, level ) );
}

JX_Int
JX_PAMGSetSmoothType( JX_Solver solver, JX_Int smooth_type )
{
   return( jx_PAMGSetSmoothType( (void *) solver, smooth_type ) );
}

JX_Int
JX_PAMGGetSmoothType( JX_Solver solver, JX_Int * smooth_type )
{
   return( jx_PAMGGetSmoothType( (void *) solver, smooth_type ) );
}

JX_Int
JX_PAMGSetSmoothNumLevels( JX_Solver solver, JX_Int smooth_num_levels )
{
   return( jx_PAMGSetSmoothNumLevels((void *)solver,smooth_num_levels ));
}

JX_Int
JX_PAMGGetSmoothNumLevels( JX_Solver solver, JX_Int *smooth_num_levels )
{
   return( jx_PAMGGetSmoothNumLevels((void *)solver,smooth_num_levels ));
}

JX_Int
JX_PAMGSetSmoothNumSweeps( JX_Solver solver, JX_Int smooth_num_sweeps )
{
   return( jx_PAMGSetSmoothNumSweeps((void *)solver,smooth_num_sweeps ));
}

JX_Int
JX_PAMGGetSmoothNumSweeps( JX_Solver solver, JX_Int *smooth_num_sweeps )
{
   return( jx_PAMGGetSmoothNumSweeps((void *)solver,smooth_num_sweeps ));
}

JX_Int
JX_PAMGSetLogging( JX_Solver solver, JX_Int logging )
{
  /*-------------------------------------------------------------------------
   * This function should be called before Setup.  Logging changes
   * may require allocation or freeing of arrays, which is presently
   * only done there.
   * It may be possible to support logging changes at other times,
   * but there is little need.
   *------------------------------------------------------------------------*/
   return( jx_PAMGSetLogging( (void *) solver, logging ) );
}

JX_Int
JX_PAMGGetLogging( JX_Solver solver, JX_Int * logging )
{
   return( jx_PAMGGetLogging( (void *) solver, logging ) );
}

JX_Int
JX_PAMGSetPrintLevel( JX_Solver solver, JX_Int print_level )
{
   return( jx_PAMGSetPrintLevel( (void *) solver, print_level ) );
}

JX_Int
JX_PAMGSetPrintCoarseMatrix( JX_Solver solver, JX_Int print_coarse_matrix )
{
   return( jx_PAMGSetPrintCoarseMatrix( (void *) solver, print_coarse_matrix ) );
}

JX_Int
JX_PAMGGetPrintLevel( JX_Solver solver, JX_Int *print_level )
{
   return( jx_PAMGGetPrintLevel( (void *) solver, print_level ) );
}

JX_Int
JX_PAMGSetPrintFileName( JX_Solver solver, const char *print_file_name )
{
   return( jx_PAMGSetPrintFileName( (void *) solver, print_file_name ) );
}

JX_Int
JX_PAMGSetDebugFlag( JX_Solver solver, JX_Int debug_flag )
{
   return( jx_PAMGSetDebugFlag( (void *) solver, debug_flag ) );
}

JX_Int
JX_PAMGGetDebugFlag( JX_Solver solver, JX_Int *debug_flag )
{
   return( jx_PAMGGetDebugFlag( (void *) solver, debug_flag ) );
}

JX_Int
JX_PAMGGetNumIterations( JX_Solver solver, JX_Int *num_iterations )
{
   return( jx_PAMGGetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int
JX_PAMGGetCumNumIterations( JX_Solver solver, JX_Int *cum_num_iterations )
{
   return( jx_PAMGGetCumNumIterations( (void *) solver, cum_num_iterations ) );
}

JX_Int
JX_PAMGGetResidual( JX_Solver solver, JX_ParVector *residual )
{
   return jx_PAMGGetResidual( (void *) solver, (jx_ParVector **)residual );
}
                            
JX_Int
JX_PAMGGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *rel_resid_norm )
{
   return( jx_PAMGGetRelResidualNorm( (void *) solver, rel_resid_norm ) );
}

JX_Int
JX_PAMGSetVariant( JX_Solver solver,  JX_Int variant )
{
   return( jx_PAMGSetVariant( (void *) solver, variant ) );
}

JX_Int
JX_PAMGGetVariant( JX_Solver solver, JX_Int *variant )
{
   return( jx_PAMGGetVariant( (void *) solver, variant ) );
}

JX_Int
JX_PAMGSetOverlap( JX_Solver solver, JX_Int overlap )
{
   return( jx_PAMGSetOverlap( (void *) solver, overlap ) );
}

JX_Int
JX_PAMGGetOverlap( JX_Solver solver, JX_Int *overlap )
{
   return( jx_PAMGGetOverlap( (void *) solver, overlap ) );
}

JX_Int
JX_PAMGSetDomainType( JX_Solver solver, JX_Int domain_type )
{
   return( jx_PAMGSetDomainType( (void *) solver, domain_type ) );
}

JX_Int
JX_PAMGGetDomainType( JX_Solver solver, JX_Int *domain_type )
{
   return( jx_PAMGGetDomainType( (void *) solver, domain_type ) );
}

JX_Int
JX_PAMGSetSchwarzRlxWeight( JX_Solver solver, JX_Real schwarz_rlx_weight )
{
   return( jx_PAMGSetSchwarzRlxWeight( (void *)solver, schwarz_rlx_weight ) );
}

JX_Int
JX_PAMGGetSchwarzRlxWeight( JX_Solver solver, JX_Real *schwarz_rlx_weight )
{
   return( jx_PAMGGetSchwarzRlxWeight( (void *)solver, schwarz_rlx_weight ) );
}

JX_Int
JX_PAMGSetSchwarzUseNonSymm( JX_Solver solver, JX_Int use_nonsymm )
{
   return( jx_PAMGSetSchwarzUseNonSymm( (void *) solver, use_nonsymm ) );
}

JX_Int
JX_PAMGSetSym( JX_Solver solver, JX_Int sym )
{
   return( jx_PAMGSetSym( (void *) solver, sym ) );
}

JX_Int
JX_PAMGSetLevel( JX_Solver solver, JX_Int level )
{
   return( jx_PAMGSetLevel( (void *) solver, level ) );
}

JX_Int
JX_PAMGSetThreshold( JX_Solver solver, JX_Real threshold )
{
   return( jx_PAMGSetThreshold( (void *) solver, threshold ) );
}

JX_Int
JX_PAMGSetFilter( JX_Solver solver, JX_Real filter )
{
   return( jx_PAMGSetFilter( (void *) solver, filter ) );
}

JX_Int
JX_PAMGSetDropTol( JX_Solver solver, JX_Real drop_tol )
{
   return( jx_PAMGSetDropTol( (void *) solver, drop_tol ) );
}

JX_Int
JX_PAMGSetMaxNzPerRow( JX_Solver solver, JX_Int max_nz_per_row )
{
   return( jx_PAMGSetMaxNzPerRow( (void *) solver, max_nz_per_row ) );
}

JX_Int
JX_PAMGSetEuclidFile( JX_Solver solver, char *euclidfile )
{
   return( jx_PAMGSetEuclidFile( (void *) solver, euclidfile ) );
}

JX_Int
JX_PAMGSetEuLevel( JX_Solver solver, JX_Int eu_level )
{
   return( jx_PAMGSetEuLevel( (void *) solver, eu_level ) );
}

JX_Int
JX_PAMGSetEuSparseA( JX_Solver solver, JX_Real eu_sparse_A )
{
   return( jx_PAMGSetEuSparseA( (void *) solver, eu_sparse_A ) );
}

JX_Int
JX_PAMGSetEuBJ( JX_Solver solver, JX_Int eu_bj )
{
   return( jx_PAMGSetEuBJ( (void *) solver, eu_bj ) );
}

JX_Int
JX_PAMGSetNumFunctions( JX_Solver solver, JX_Int num_functions )
{
   return( jx_PAMGSetNumFunctions( (void *) solver, num_functions ) );
}

JX_Int
JX_PAMGGetNumFunctions( JX_Solver solver, JX_Int *num_functions )
{
   return( jx_PAMGGetNumFunctions( (void *) solver, num_functions ) );
}

JX_Int
JX_PAMGSetNodal( JX_Solver solver, JX_Int nodal )
{
   return( jx_PAMGSetNodal( (void *) solver, nodal ) );
}

JX_Int
JX_PAMGSetNodalDiag( JX_Solver solver, JX_Int nodal )
{
   return( jx_PAMGSetNodalDiag( (void *) solver, nodal ) );
}

/*!
 * \fn JX_Int JX_PAMGSetDofFunc
 * \note Warning about a possible memory problem: When the PAMG object is 
 *       destroyed in jx_PAMGDestroy, dof_func aka DofFunc will be destroyed
 *       Normally this is what we want.  But if the user provided dof_func
 *       by calling JX_PAMGSetDofFunc, this could be an unwanted surprise.
 *       As jx is currently commonly used, this situation is likely to be rare.
 * \date 2011/09/03
 */
JX_Int
JX_PAMGSetDofFunc( JX_Solver solver, JX_Int  *dof_func )
{
   return( jx_PAMGSetDofFunc( (void *) solver, dof_func ) );
}

JX_Int
JX_PAMGSetNumPaths( JX_Solver solver, JX_Int num_paths )
{
   return( jx_PAMGSetNumPaths( (void *) solver, num_paths ) );
}

JX_Int
JX_PAMGSetAggNumLevels( JX_Solver solver, JX_Int agg_num_levels )
{
   return( jx_PAMGSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

JX_Int
JX_PAMGSetAggInterpType( JX_Solver solver, JX_Int agg_interp_type )
{
   return( jx_PAMGSetAggInterpType( (void *) solver, agg_interp_type ) );
}

JX_Int
JX_PAMGSetAggPMaxElmts( JX_Solver solver, JX_Int agg_P_max_elmts )
{
   return( jx_PAMGSetAggPMaxElmts( (void *) solver, agg_P_max_elmts ) );
}

JX_Int
JX_PAMGSetAggP12MaxElmts( JX_Solver solver, JX_Int agg_P12_max_elmts )
{
   return( jx_PAMGSetAggP12MaxElmts( (void *) solver, agg_P12_max_elmts ) );
}

JX_Int
JX_PAMGSetAggTruncFactor( JX_Solver solver, JX_Real agg_trunc_factor )
{
   return( jx_PAMGSetAggTruncFactor( (void *) solver, agg_trunc_factor ) );
}

JX_Int
JX_PAMGSetAggP12TruncFactor( JX_Solver solver, JX_Real agg_P12_trunc_factor )
{
   return( jx_PAMGSetAggP12TruncFactor( (void *) solver, agg_P12_trunc_factor ) );
}

JX_Int
JX_PAMGSetNumCRRelaxSteps( JX_Solver solver, JX_Int num_CR_relax_steps )
{
   return( jx_PAMGSetNumCRRelaxSteps( (void *) solver, num_CR_relax_steps ) );
}

JX_Int
JX_PAMGSetCRRate( JX_Solver solver, JX_Real CR_rate )
{
   return( jx_PAMGSetCRRate( (void *) solver, CR_rate ) );
}

JX_Int
JX_PAMGSetCRStrongTh( JX_Solver solver, JX_Real CR_strong_th )
{
   return( jx_PAMGSetCRStrongTh( (void *) solver, CR_strong_th ) );
}

JX_Int
JX_PAMGSetISType( JX_Solver solver, JX_Int IS_type )
{
   return( jx_PAMGSetISType( (void *) solver, IS_type ) );
}

JX_Int
JX_PAMGSetCRUseCG( JX_Solver solver, JX_Int CR_use_CG )
{
   return( jx_PAMGSetCRUseCG( (void *) solver, CR_use_CG ) );
}

JX_Int
JX_PAMGSetGSMG( JX_Solver solver, JX_Int gsmg )
{
   return( jx_PAMGSetGSMG( (void *) solver, gsmg ) );
}

JX_Int
JX_PAMGSetNumSamples( JX_Solver solver, JX_Int gsmg )
{
   return( jx_PAMGSetNumSamples( (void *) solver, gsmg ) );
}
                                                                                                                                                                                                            
JX_Int
JX_PAMGSetCGCIts ( JX_Solver solver, JX_Int its )
{
  return (jx_PAMGSetCGCIts ( (void *) solver, its ) );
}

/*!
 * \fn JX_Int JX_PAMGSetSpMtRapType
 * \author Yue Xiaoqiang
 * \date 2012/10/13
 */
JX_Int
JX_PAMGSetSpMtRapType( JX_Solver solver, JX_Int spmt_rap_type )
{
   return (jx_PAMGSetSpMtRapType ( (void *) solver, spmt_rap_type ) );
}

/*!
 * \fn JX_Int JX_PAMGSetWallTimeOption
 * \author Yue Xiaoqiang
 * \date 2015/09/30
 */
JX_Int
JX_PAMGSetWallTimeOption( JX_Solver solver, JX_Int wall_time_option )
{
   return (jx_PAMGSetWallTimeOption ( (void *) solver, wall_time_option ) );
}

/*!
 * \fn JX_Int JX_PAMGSetAIMeasureType
 * \author Yue Xiaoqiang
 * \date 2014/02/26
 */
JX_Int
JX_PAMGSetAIMeasureType( JX_Solver solver, JX_Int ai_measure_type )
{
   return (jx_PAMGSetAIMeasureType ( (void *) solver, ai_measure_type ) );
}

/*!
 * \fn JX_Int JX_PAMGSetAIRelaxType
 * \author Yue Xiaoqiang
 * \date 2014/07/06
 */
JX_Int
JX_PAMGSetAIRelaxType( JX_Solver solver, JX_Int ai_relax_type )
{
   return (jx_PAMGSetAIRelaxType ( (void *) solver, ai_relax_type ) );
}

/*!
 * \fn JX_Int JX_PAMGSetRelaxedCoarsenMeasureType
 * \author peghoty
 * \date 2010/05/29
 */
JX_Int
JX_PAMGSetRelaxedCoarsenMeasureType( JX_Solver solver, JX_Int measure_type_rlx )
{
   return (jx_PAMGSetRelaxedCoarsenMeasureType ( (void *) solver, measure_type_rlx ) );
}

/*!
 * \fn JX_Int JX_PAMGSetRelaxedCoarsenNumberSyn
 * \author peghoty
 * \date 2010/05/29
 */
JX_Int
JX_PAMGSetRelaxedCoarsenNumberSyn( JX_Solver solver, JX_Int number_syn_rlx )
{
   return (jx_PAMGSetRelaxedCoarsenNumberSyn ( (void *) solver, number_syn_rlx ) );
}

/*!
 * \fn JX_Int JX_PAMGSetRelaxedCoarsenMeasureThreshold
 * \author peghoty
 * \date 2010/05/29
 */
JX_Int
JX_PAMGSetRelaxedCoarsenMeasureThreshold( JX_Solver solver, JX_Real measure_threshold_rlx )
{
   return (jx_PAMGSetRelaxedCoarsenMeasureThreshold ( (void *) solver, measure_threshold_rlx ) );
}
                                                                                                       
JX_Int
JX_PAMGSetPlotGrids ( JX_Solver solver, JX_Int plotgrids )
{
   return (jx_PAMGSetPlotGrids ( (void *) solver, plotgrids ) );
}
                                                                                                                                                                                                            
JX_Int
JX_PAMGSetPlotFileName ( JX_Solver solver, const char *plotfilename )
{
   return (jx_PAMGSetPlotFileName ( (void *) solver, plotfilename ) );
}
                                                                                                                                                                                                              
JX_Int
JX_PAMGSetCoordDim ( JX_Solver solver, JX_Int coorddim )
{
   return (jx_PAMGSetCoordDim ( (void *) solver, coorddim ) );
}
                                                                                                                                                                                                           
JX_Int
JX_PAMGSetCoordinates ( JX_Solver solver, float *coordinates )
{
   return (jx_PAMGSetCoordinates ( (void *) solver, coordinates ) );
}

/*!
 * \fn JX_Int JX_PAMGSetConvCriteria
 * \author peghoty
 * \date 2009/07/25
 */
JX_Int
JX_PAMGSetConvCriteria( JX_Solver solver, JX_Int conv_criteria )
{
   return (jx_PAMGSetConvCriteria ( (void *) solver, conv_criteria ) );
}

JX_Int
JX_PAMGSetCoarsestSolverID( JX_Solver solver, JX_Int coarsestsolverid )
{
   return (jx_PAMGSetCoarsestSolverID ( (void *) solver, coarsestsolverid ) );
}

/*!
 * \fn JX_Int JX_PAMGSetCoarseThreshold
 * \author peghoty
 * \date 2010/04/14
 */
JX_Int
JX_PAMGSetCoarseThreshold( JX_Solver solver, JX_Int coarse_threshold )
{
   return (jx_PAMGSetCoarseThreshold ( (void *) solver, coarse_threshold ) );
}

/*!
 * \fn JX_Int JX_PAMGSetCoarseRatio
 * \author peghoty
 * \date 2010/04/14
 */
JX_Int
JX_PAMGSetCoarseRatio( JX_Solver solver, JX_Real coarse_ratio )
{
   return (jx_PAMGSetCoarseRatio ( (void *) solver, coarse_ratio ) );
}

JX_Int
JX_PAMGSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold )
{
   return (jx_PAMGSetConvFacThreshold ( (void *) solver, convfac_threshold ) );
}

JX_Int
JX_PAMGSetRAP2( JX_Solver solver, JX_Int rap2 )
{
   return (jx_PAMGSetRAP2 ( (void *) solver, rap2 ) );
}

JX_Int
JX_PAMGSetKeepTranspose( JX_Solver solver, JX_Int keepTranspose )
{
   return (jx_PAMGSetKeepTranspose ( (void *) solver, keepTranspose ) );
}

/*!
 * \fn JX_Int jx_PAMGSet*
 * \brief  Routines to set the setup phase parameters begin with "jx_"
 * \date 2010/04/14
 */
JX_Int
jx_PAMGSetRestriction( void *data, JX_Int restr_par )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   /* RL: currently, 0: R = P^T, 1: AIR, >1: AIR-2 */
   if (restr_par < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 

   jx_ParAMGDataRestriction(amg_data) = restr_par;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetAIRStrongTh( void *data, JX_Real AIR_strong_th )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_ParAMGDataAIRStrongTh(amg_data) = AIR_strong_th;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetMaxLevels( void *data, JX_Int max_levels )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (max_levels < 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataMaxLevels(amg_data) = max_levels;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetMaxLevels( void *data, JX_Int *max_levels )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *max_levels = jx_ParAMGDataMaxLevels(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetStrongThreshold( void *data, JX_Real strong_threshold )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataStrongThreshold(amg_data) = strong_threshold;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetStrongThreshold( void *data, JX_Real *strong_threshold )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *strong_threshold = jx_ParAMGDataStrongThreshold(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetMaxRowSum( void *data, JX_Real max_row_sum )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (max_row_sum <= 0 || max_row_sum > 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataMaxRowSum(amg_data) = max_row_sum;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetMaxRowSum( void *data, JX_Real *max_row_sum )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *max_row_sum = jx_ParAMGDataMaxRowSum(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetTruncFactor( void *data, JX_Real trunc_factor )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (trunc_factor < 0 || trunc_factor >= 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataTruncFactor(amg_data) = trunc_factor;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetTruncFactor( void *data, JX_Real *trunc_factor )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *trunc_factor = jx_ParAMGDataTruncFactor(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetPMaxElmts( void *data, JX_Int P_max_elmts )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (P_max_elmts < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataPMaxElmts(amg_data) = P_max_elmts;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetPMaxElmts( void *data, JX_Int *P_max_elmts )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *P_max_elmts = jx_ParAMGDataPMaxElmts(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetJacobiTruncThreshold( void *data, JX_Real jacobi_trunc_threshold )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (jacobi_trunc_threshold < 0 || jacobi_trunc_threshold >= 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataJacobiTruncThreshold(amg_data) = jacobi_trunc_threshold;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetJacobiTruncThreshold( void *data, JX_Real *jacobi_trunc_threshold )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *jacobi_trunc_threshold = jx_ParAMGDataJacobiTruncThreshold(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetPostInterpType( void *data, JX_Int post_interp_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (post_interp_type < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataPostInterpType(amg_data) = post_interp_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetPostInterpType( void *data, JX_Int *post_interp_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *post_interp_type = jx_ParAMGDataPostInterpType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetSCommPkgSwitch( void *data, JX_Real S_commpkg_switch )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_ParAMGDataSCommPkgSwitch(amg_data) = S_commpkg_switch;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetSCommPkgSwitch( void *data, JX_Real *S_commpkg_switch )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *S_commpkg_switch = jx_ParAMGDataSCommPkgSwitch(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetInterpType( void *data, JX_Int interp_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

  /*------------------------------------------------------------------------
   *  “interp_type > 25” has been modified to “interp_type > 100” since
   *  we may add other interpolations. peghoty  2009/08/08  
   *------------------------------------------------------------------------*/
   if (interp_type < 0 || interp_type > 100)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataInterpType(amg_data) = interp_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetInterpType( void *data, JX_Int *interp_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *interp_type = jx_ParAMGDataInterpType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetMinIter( void *data, JX_Int min_iter )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_ParAMGDataMinIter(amg_data) = min_iter;

   return jx_error_flag;
} 

JX_Int
jx_PAMGGetMinIter( void *data, JX_Int *min_iter )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *min_iter = jx_ParAMGDataMinIter(amg_data);

   return jx_error_flag;
} 

JX_Int
jx_PAMGSetMaxIter( void *data, JX_Int max_iter )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (max_iter < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataMaxIter(amg_data) = max_iter;

   return jx_error_flag;
} 

JX_Int
jx_PAMGGetMaxIter( void *data, JX_Int *max_iter )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *max_iter = jx_ParAMGDataMaxIter(amg_data);

   return jx_error_flag;
} 

JX_Int
jx_PAMGSetCoarsenType( void *data, JX_Int coarsen_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetCoarsenType( void *data, JX_Int *coarsen_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *coarsen_type = jx_ParAMGDataCoarsenType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetMeasureType( void *data, JX_Int measure_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_ParAMGDataMeasureType(amg_data) = measure_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetMeasureType( void *data, JX_Int *measure_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *measure_type = jx_ParAMGDataMeasureType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetSetupType( void *data, JX_Int setup_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   jx_ParAMGDataSetupType(amg_data) = setup_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetSetupType( void *data, JX_Int *setup_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *setup_type = jx_ParAMGDataSetupType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetCycleType( void *data, JX_Int cycle_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (cycle_type < 0 || cycle_type > 2)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataCycleType(amg_data) = cycle_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetCycleType( void *data, JX_Int *cycle_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *cycle_type = jx_ParAMGDataCycleType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetTol( void *data, JX_Real tol  )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (tol < 0 || tol > 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataTol(amg_data) = tol;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetRhsNrmThreshold( void *data, JX_Real rhsnrm_threshold )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (rhsnrm_threshold < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   jx_ParAMGDataRhsNrmThreshold(amg_data) = rhsnrm_threshold;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetTol( void *data, JX_Real *tol )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   *tol = jx_ParAMGDataTol(amg_data);

   return jx_error_flag;
}

/* The "Get" function for SetNumSweeps is GetCycleNumSweeps. */
JX_Int
jx_PAMGSetNumSweeps( void *data, JX_Int num_sweeps )  /* default: num_sweeps = 1 */
{
   JX_Int i;
   JX_Int *num_grid_sweeps;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (num_sweeps < 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      jx_ParAMGDataNumGridSweeps(amg_data) = jx_CTAlloc(JX_Int, 4);
   }
       
   num_grid_sweeps = jx_ParAMGDataNumGridSweeps(amg_data);

   for (i = 0; i < 3; i ++)
   {
      num_grid_sweeps[i] = num_sweeps;
   }
   num_grid_sweeps[3] = 1;

   return jx_error_flag;
}
 
JX_Int
jx_PAMGSetCycleNumSweeps( void *data, JX_Int num_sweeps, JX_Int k )
{
   JX_Int i;
   JX_Int *num_grid_sweeps;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 

   if (num_sweeps < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (k < 1 || k > 3)
   {
      jx_printf (" Warning! Invalid cycle! num_sweeps not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }

   if (jx_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
       num_grid_sweeps = jx_CTAlloc(JX_Int, 4);
       for (i = 0; i < 4; i ++)
       {
	  num_grid_sweeps[i] = 1;
       }
       jx_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
   }
       
   jx_ParAMGDataNumGridSweeps(amg_data)[k] = num_sweeps;

   return jx_error_flag;
}
 
JX_Int
jx_PAMGGetCycleNumSweeps( void *data, JX_Int *num_sweeps, JX_Int k )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      jx_printf (" Warning! Invalid cycle! No num_sweeps to get!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }

   if (jx_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
       
   *num_sweeps = jx_ParAMGDataNumGridSweeps(amg_data)[k];

   return jx_error_flag;
}
 
JX_Int
jx_PAMGSetNumGridSweeps( void *data, JX_Int *num_grid_sweeps )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (!num_grid_sweeps)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataNumGridSweeps(amg_data))
   {
      jx_TFree(jx_ParAMGDataNumGridSweeps(amg_data));
   }
   jx_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return jx_error_flag;
}
 
JX_Int
jx_PAMGGetNumGridSweeps( void *data, JX_Int **num_grid_sweeps )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *num_grid_sweeps = jx_ParAMGDataNumGridSweeps(amg_data);

   return jx_error_flag;
}

/*!
 * \fn void jx_dispose_elt
 * \note The "Get" function for SetRelaxType is GetCycleRelaxType.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGSetRelaxType( void *data, JX_Int relax_type ) /* default: relax_type = 3 */
{
   JX_Int i;
   JX_Int *grid_relax_type;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (relax_type < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      jx_ParAMGDataGridRelaxType(amg_data) = jx_CTAlloc(JX_Int, 4);
   }
   grid_relax_type = jx_ParAMGDataGridRelaxType(amg_data);

   for (i = 0; i < 3; i ++)
   {
      grid_relax_type[i] = relax_type;
   }
   grid_relax_type[3] = 9;
   jx_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   jx_ParAMGDataUserRelaxType(amg_data) = relax_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetCycleRelaxType( void *data, JX_Int relax_type, JX_Int k )
{
   JX_Int i;
   JX_Int *grid_relax_type;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      jx_printf (" Warning! Invalid cycle! relax_type not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }
   if (relax_type < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      grid_relax_type = jx_CTAlloc(JX_Int, 4);
      for (i = 0; i < 3; i ++)
      {
         grid_relax_type[i] = 3;
      }
      grid_relax_type[3] = 9;
      jx_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   }
      
   jx_ParAMGDataGridRelaxType(amg_data)[k] = relax_type;
   if (k == 3)
   {
      jx_ParAMGDataUserCoarseRelaxType(amg_data) = relax_type;
   }
   return jx_error_flag;
}

JX_Int
jx_PAMGGetCycleRelaxType( void *data, JX_Int *relax_type, JX_Int k )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (k < 1 || k > 3)
   {
      jx_printf (" Warning! Invalid cycle! relax_type not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }

   if (jx_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
      
   *relax_type = jx_ParAMGDataGridRelaxType(amg_data)[k];

   return jx_error_flag;
}

JX_Int
jx_PAMGSetRelaxOrder( void *data, JX_Int relax_order )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetRelaxOrder( void *data, JX_Int *relax_order )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *relax_order = jx_ParAMGDataRelaxOrder(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetGridRelaxType( void *data, JX_Int  *grid_relax_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (!grid_relax_type)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataGridRelaxType(amg_data))
   {
      jx_TFree(jx_ParAMGDataGridRelaxType(amg_data));
   }
   jx_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   jx_ParAMGDataUserCoarseRelaxType(amg_data) = grid_relax_type[3];

   return jx_error_flag;
}

JX_Int
jx_PAMGGetGridRelaxType( void *data, JX_Int **grid_relax_type )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *grid_relax_type = jx_ParAMGDataGridRelaxType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetGridRelaxPoints( void *data, JX_Int **grid_relax_points )
{
   JX_Int i;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (!grid_relax_points)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i = 0; i < 4; i ++)
      {
   	 jx_TFree (jx_ParAMGDataGridRelaxPoints(amg_data)[i]);
      }
      jx_TFree(jx_ParAMGDataGridRelaxPoints(amg_data));
   }
   jx_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points; 

   return jx_error_flag;
}

JX_Int
jx_PAMGGetGridRelaxPoints( void *data, JX_Int ***grid_relax_points )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *grid_relax_points = jx_ParAMGDataGridRelaxPoints(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetRelaxWeight( void *data, JX_Real *relax_weight )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (!relax_weight)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   if (jx_ParAMGDataRelaxWeight(amg_data))
   {
      jx_TFree(jx_ParAMGDataRelaxWeight(amg_data));
   }
   jx_ParAMGDataRelaxWeight(amg_data) = relax_weight;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetRelaxWeight( void *data, JX_Real **relax_weight )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *relax_weight = jx_ParAMGDataRelaxWeight(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetRelaxWt( void *data, JX_Real relax_weight )
{
   JX_Int i, num_levels;
   JX_Real *relax_weight_array;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   num_levels = jx_ParAMGDataMaxLevels(amg_data);
   if (jx_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      jx_ParAMGDataRelaxWeight(amg_data) = jx_CTAlloc(JX_Real, num_levels);
   }
                     
   relax_weight_array = jx_ParAMGDataRelaxWeight(amg_data);
   for (i = 0; i < num_levels; i ++)
   {
      relax_weight_array[i] = relax_weight;
   }
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetLevelRelaxWt( void *data, JX_Real relax_weight, JX_Int level )
{
   JX_Int i, num_levels;
   jx_ParAMGData  *amg_data = data;
   num_levels = jx_ParAMGDataMaxLevels(amg_data);
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (level > num_levels-1 || level < 0) 
   {
      jx_printf (" Warning! Invalid level! Relax weight not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }
   if (jx_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      jx_ParAMGDataRelaxWeight(amg_data) = jx_CTAlloc(JX_Real, num_levels);
      for (i = 0; i < num_levels; i ++)
      {
         jx_ParAMGDataRelaxWeight(amg_data)[i] = 1.0;
      }
   }
               
   jx_ParAMGDataRelaxWeight(amg_data)[level] = relax_weight;
   
   return jx_error_flag;
}

JX_Int
jx_PAMGGetLevelRelaxWt( void *data, JX_Real *relax_weight, JX_Int level )
{
   JX_Int num_levels;
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   num_levels = jx_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1 || level < 0) 
   {
      jx_printf (" Warning! Invalid level! Relax weight not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }
   if (jx_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
               
   *relax_weight = jx_ParAMGDataRelaxWeight(amg_data)[level];
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetOmega( void *data, JX_Real *omega )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (!omega)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   if (jx_ParAMGDataOmega(amg_data))
   {
      jx_TFree(jx_ParAMGDataOmega(amg_data));
   }
   jx_ParAMGDataOmega(amg_data) = omega;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetOmega( void *data, JX_Real **omega )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *omega = jx_ParAMGDataOmega(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetOuterWt( void *data, JX_Real omega )
{
   JX_Int i, num_levels;
   JX_Real *omega_array;
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   num_levels = jx_ParAMGDataMaxLevels(amg_data);
   if (jx_ParAMGDataOmega(amg_data) == NULL)
   {
      jx_ParAMGDataOmega(amg_data) = jx_CTAlloc(JX_Real, num_levels);
   }
                     
   omega_array = jx_ParAMGDataOmega(amg_data);
   for (i = 0; i < num_levels; i ++)
   {      
      omega_array[i] = omega;
   }
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetLevelOuterWt( void *data, JX_Real omega, JX_Int level )
{
   JX_Int i, num_levels;
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   num_levels = jx_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      jx_printf (" Warning! Invalid level! Outer weight not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }
   if (jx_ParAMGDataOmega(amg_data) == NULL)
   {
      jx_ParAMGDataOmega(amg_data) = jx_CTAlloc(JX_Real, num_levels);
      for (i = 0; i < num_levels; i ++)
      {
         jx_ParAMGDataOmega(amg_data)[i] = 1.0;
      }
   }
               
   jx_ParAMGDataOmega(amg_data)[level] = omega;
   
   return jx_error_flag;
}

JX_Int
jx_PAMGGetLevelOuterWt( void *data, JX_Real  *omega, JX_Int level )
{
   JX_Int num_levels;
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   num_levels = jx_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels-1) 
   {
      jx_printf (" Warning! Invalid level! Outer weight not set!\n");
      jx_error_in_arg(3);
      return jx_error_flag;
   }
   if (jx_ParAMGDataOmega(amg_data) == NULL)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
               
   *omega = jx_ParAMGDataOmega(amg_data)[level];
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetSmoothType( void *data, JX_Int smooth_type )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
               
   jx_ParAMGDataSmoothType(amg_data) = smooth_type;
   
   return jx_error_flag;
}

JX_Int
jx_PAMGGetSmoothType( void *data, JX_Int *smooth_type )
{
   jx_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *smooth_type = jx_ParAMGDataSmoothType(amg_data);
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetSmoothNumLevels( void *data, JX_Int smooth_num_levels )
{
   jx_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (smooth_num_levels < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataSmoothNumLevels(amg_data) = smooth_num_levels;
   
   return jx_error_flag;
}

JX_Int
jx_PAMGGetSmoothNumLevels( void *data, JX_Int *smooth_num_levels )
{
   jx_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *smooth_num_levels = jx_ParAMGDataSmoothNumLevels(amg_data);
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetSmoothNumSweeps( void *data, JX_Int smooth_num_sweeps )
{
   jx_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (smooth_num_sweeps < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataSmoothNumSweeps(amg_data) = smooth_num_sweeps;
   
   return jx_error_flag;
}

JX_Int
jx_PAMGGetSmoothNumSweeps( void *data, JX_Int *smooth_num_sweeps )
{
   jx_ParAMGData  *amg_data = data;
               
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *smooth_num_sweeps = jx_ParAMGDataSmoothNumSweeps(amg_data);
   
   return jx_error_flag;
}

JX_Int
jx_PAMGSetLogging( void *data, JX_Int logging )
{
  /*-------------------------------------------------------------------------
   * This function should be called before Setup.  Logging changes
   * may require allocation or freeing of arrays, which is presently
   * only done there.
   * It may be possible to support logging changes at other times,
   * but there is little need.
   *------------------------------------------------------------------------*/      
      
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataLogging(amg_data) = logging;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetLogging( void *data, JX_Int *logging )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *logging = jx_ParAMGDataLogging(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetPrintLevel( void *data, JX_Int print_level )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataPrintLevel(amg_data) = print_level;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetPrintCoarseMatrix( void *data, JX_Int print_coarse_matrix )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataPrintCoarseSystem(amg_data) = print_coarse_matrix;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetPrintLevel( void *data, JX_Int *print_level )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *print_level = jx_ParAMGDataPrintLevel(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetPrintFileName( void *data, const char *print_file_name )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if( strlen(print_file_name) > 256 )
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 

   jx_sprintf(jx_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return jx_error_flag;
}

JX_Int
jx_PAMGGetPrintFileName( void *data, char **print_file_name )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_sprintf( *print_file_name, "%s", jx_ParAMGDataLogFileName(amg_data) );

   return jx_error_flag;
}

JX_Int
jx_PAMGSetNumIterations( void *data, JX_Int num_iterations )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataNumIterations(amg_data) = num_iterations;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetDebugFlag( void *data, JX_Int debug_flag )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetDebugFlag( void *data, JX_Int  *debug_flag )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *debug_flag = jx_ParAMGDataDebugFlag(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetGSMG( void *data, JX_Int par )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   amg_data->gsmg = par;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetNumSamples( void *data, JX_Int par )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   amg_data->num_samples = par;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetCGCIts( void *data, JX_Int its )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataCGCIts(amg_data) = its;
  return (ierr);
}

/*!
 * \fn JX_Int jx_PAMGSetSpMtRapType
 * \author Yue Xiaoqiang
 * \date 2012/10/13
 */
JX_Int
jx_PAMGSetSpMtRapType( void *data, JX_Int spmt_rap_type )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataSpMtRapType(amg_data) = spmt_rap_type;
  return (ierr);
}


/*!
 * \fn JX_Int jx_PAMGSetWallTimeOption
 * \author Yue Xiaoqiang
 * \date 2015/09/30
 */
JX_Int
jx_PAMGSetWallTimeOption( void *data, JX_Int wall_time_option )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataWallTimeOption(amg_data) = wall_time_option;
  return (ierr);
}

/*!
 * \fn JX_Int jx_PAMGSetAIMeasureType
 * \author Yue Xiaoqiang
 * \date 2014/02/26
 */
JX_Int
jx_PAMGSetAIMeasureType( void *data, JX_Int ai_measure_type )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataAIMeasureType(amg_data) = ai_measure_type;
  return (ierr);
}

/*!
 * \fn JX_Int jx_PAMGSetAIRelaxType
 * \author Yue Xiaoqiang
 * \date 2014/07/06
 */
JX_Int
jx_PAMGSetAIRelaxType( void *data, JX_Int ai_relax_type )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataAIRelaxType(amg_data) = ai_relax_type;
  return (ierr);
}

/*!
 * \fn JX_Int jx_PAMGSetRelaxedCoarsenMeasureType
 * \author peghoty
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetRelaxedCoarsenMeasureType( void *data, JX_Int measure_type_rlx )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataMeasureTypeRlx(amg_data) = measure_type_rlx;
  return (ierr);
}

/*!
 * \fn JX_Int jx_PAMGSetRelaxedCoarsenNumberSyn
 * \author peghoty
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetRelaxedCoarsenNumberSyn( void *data, JX_Int number_syn_rlx )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataNumberSynRlx(amg_data) = number_syn_rlx;
  return (ierr);
}

/*!
 * \fn JX_Int jx_PAMGSetRelaxedCoarsenMeasureThreshold
 * \author peghoty
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetRelaxedCoarsenMeasureThreshold( void *data, JX_Real measure_threshold_rlx )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataMeasureThresholdRlx(amg_data) = measure_threshold_rlx;
  return (ierr);
}

JX_Int
jx_PAMGSetPlotGrids( void *data, JX_Int plotgrids )
{
  JX_Int ierr = 0;
  jx_ParAMGData *amg_data = data;

  jx_ParAMGDataPlotGrids(amg_data) = plotgrids;
  return (ierr);
}

JX_Int
jx_PAMGSetPlotFileName( void *data, const char *plot_file_name )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if( strlen(plot_file_name) > 251 )
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   if (strlen(plot_file_name) == 0 )
   {
      jx_sprintf(jx_ParAMGDataPlotFileName(amg_data), "%s", "AMGgrids.CF.dat");
   }
   else
   {
      jx_sprintf(jx_ParAMGDataPlotFileName(amg_data), "%s", plot_file_name);
   }

   return jx_error_flag;
}

JX_Int
jx_PAMGSetCoordDim( void *data, JX_Int coorddim )
{
   JX_Int ierr = 0;
   jx_ParAMGData *amg_data = data;

   jx_ParAMGDataCoordDim(amg_data) = coorddim;
   return (ierr);
}

JX_Int
jx_PAMGSetCoordinates( void *data, float *coordinates )
{
   JX_Int ierr = 0;
   jx_ParAMGData *amg_data = data;

   jx_ParAMGDataCoordinates(amg_data) = coordinates;
   return (ierr);
}



/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

JX_Int
jx_PAMGSetNumFunctions( void *data, JX_Int num_functions )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (num_functions < 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataNumFunctions(amg_data) = num_functions;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetNumFunctions( void *data, JX_Int *num_functions )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *num_functions = jx_ParAMGDataNumFunctions(amg_data);

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetNodal
 * \note Indicate whether to use nodal systems function.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetNodal( void *data, JX_Int nodal )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataNodal(amg_data) = nodal;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetNodalDiag
 * \note Indicate how to treat diag for primary matrix 
 *       with nodal systems function.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetNodalDiag( void *data, JX_Int nodal )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataNodalDiag(amg_data) = nodal;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetNumPaths
 * \note Indicate the degree of aggressive coarsening.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetNumPaths( void *data, JX_Int num_paths )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (num_paths < 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataNumPaths(amg_data) = num_paths;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetAggNumLevels
 * \note Indicates the number of levels of aggressive coarsening.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetAggNumLevels( void *data, JX_Int agg_num_levels )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (agg_num_levels < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataAggNumLevels(amg_data) = agg_num_levels;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetAggInterpType
 * \note Indicates the interpolation type of aggressive coarsening.
 * \date 2015/08/14
 */
JX_Int
jx_PAMGSetAggInterpType( void *data, JX_Int agg_interp_type )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParAMGDataAggInterpType(amg_data) = agg_interp_type;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetAggPMaxElmts
 * \note Indicates the max. number of entries in interpolation of aggressive coarsening.
 * \date 2015/08/14
 */
JX_Int
jx_PAMGSetAggPMaxElmts( void *data, JX_Int agg_P_max_elmts )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (agg_P_max_elmts < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   jx_ParAMGDataAggPMaxElmts(amg_data) = agg_P_max_elmts;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetAggP12MaxElmts
 * \note Indicates the max. number of entries in 12 interpolation of aggressive coarsening.
 * \date 2015/08/14
 */
JX_Int
jx_PAMGSetAggP12MaxElmts( void *data, JX_Int agg_P12_max_elmts )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (agg_P12_max_elmts < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   jx_ParAMGDataAggP12MaxElmts(amg_data) = agg_P12_max_elmts;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetAggTruncFactor
 * \note Indicates the truncated factor in interpolation of aggressive coarsening.
 * \date 2015/08/14
 */
JX_Int
jx_PAMGSetAggTruncFactor( void *data, JX_Real agg_trunc_factor )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParAMGDataAggTruncFactor(amg_data) = agg_trunc_factor;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetAggP12TruncFactor
 * \note Indicates the truncated factor in 12 interpolation of aggressive coarsening.
 * \date 2015/08/14
 */
JX_Int
jx_PAMGSetAggP12TruncFactor( void *data, JX_Real agg_P12_trunc_factor )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParAMGDataAggP12TruncFactor(amg_data) = agg_P12_trunc_factor;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetNumCRRelaxSteps
 * \note Indicates the number of relaxation steps for Compatible relaxation.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetNumCRRelaxSteps( void *data, JX_Int num_CR_relax_steps )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (num_CR_relax_steps < 1)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataNumCRRelaxSteps(amg_data) = num_CR_relax_steps;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetCRRate
 * \note Indicates the desired convergence rate for Compatible relaxation.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetCRRate( void *data, JX_Real CR_rate )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataCRRate(amg_data) = CR_rate;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetCRStrongTh
 * \note Indicates the desired convergence rate for Compatible relaxation.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetCRStrongTh( void *data, JX_Real CR_strong_th )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataCRStrongTh(amg_data) = CR_strong_th;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetISType
 * \note Indicates which independent set algorithm is used for CR.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetISType( void *data, JX_Int IS_type )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (IS_type < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataISType(amg_data) = IS_type;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetCRUseCG 
 * \note Indicates whether to use CG for compatible relaxation.
 * \date 2010/05/29
 */
JX_Int
jx_PAMGSetCRUseCG( void *data, JX_Int CR_use_CG )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataCRUseCG(amg_data) = CR_use_CG;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetNumPoints( void *data, JX_Int num_points )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataNumPoints(amg_data) = num_points;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetDofFunc( void *data, JX_Int *dof_func )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_TFree(jx_ParAMGDataDofFunc(amg_data));
   jx_ParAMGDataDofFunc(amg_data) = dof_func;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetPointDofMap( void *data, JX_Int *point_dof_map )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_TFree(jx_ParAMGDataPointDofMap(amg_data));
   jx_ParAMGDataPointDofMap(amg_data) = point_dof_map;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetDofPoint( void *data, JX_Int *dof_point )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_TFree(jx_ParAMGDataDofPoint(amg_data));
   jx_ParAMGDataDofPoint(amg_data) = dof_point;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetNumIterations( void *data, JX_Int *num_iterations )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *num_iterations = jx_ParAMGDataNumIterations(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGGetCumNumIterations( void *data, JX_Int *cum_num_iterations )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
#ifdef CUMNUMIT
   *cum_num_iterations = jx_ParAMGDataCumNumIterations(amg_data);
#endif

   return jx_error_flag;
}

JX_Int
jx_PAMGGetResidual( void *data, jx_ParVector **resid )
{
   jx_ParAMGData  *amg_data = data;
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *resid = jx_ParAMGDataResidual( amg_data );
   return jx_error_flag;
}

JX_Int
jx_PAMGGetRelResidualNorm( void *data, JX_Real *rel_resid_norm )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *rel_resid_norm = jx_ParAMGDataRelativeResidualNorm(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetVariant( void *data, JX_Int variant )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (variant < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataVariant(amg_data) = variant;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetVariant( void *data, JX_Int *variant )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *variant = jx_ParAMGDataVariant(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetOverlap( void *data, JX_Int overlap )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (overlap < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataOverlap(amg_data) = overlap;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetOverlap( void *data, JX_Int *overlap )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *overlap = jx_ParAMGDataOverlap(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetDomainType( void *data, JX_Int domain_type )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (domain_type < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataDomainType(amg_data) = domain_type;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetDomainType( void *data, JX_Int *domain_type )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *domain_type = jx_ParAMGDataDomainType(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetSchwarzRlxWeight( void *data, JX_Real schwarz_rlx_weight )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataSchwarzRlxWeight(amg_data) = schwarz_rlx_weight;

   return jx_error_flag;
}

JX_Int
jx_PAMGGetSchwarzRlxWeight( void *data, JX_Real *schwarz_rlx_weight )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   *schwarz_rlx_weight = jx_ParAMGDataSchwarzRlxWeight(amg_data);

   return jx_error_flag;
}

JX_Int
jx_PAMGSetSchwarzUseNonSymm( void *data, JX_Int use_nonsymm )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataSchwarzUseNonSymm(amg_data) = use_nonsymm;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetSym( void *data, JX_Int sym )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataSym(amg_data) = sym;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetLevel( void *data, JX_Int level )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataLevel(amg_data) = level;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetThreshold( void *data, JX_Real thresh )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataThreshold(amg_data) = thresh;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetFilter( void *data, JX_Real filter )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataFilter(amg_data) = filter;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetDropTol( void *data, JX_Real drop_tol )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataDropTol(amg_data) = drop_tol;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetMaxNzPerRow( void *data, JX_Int max_nz_per_row )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   if (max_nz_per_row < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   } 
   jx_ParAMGDataMaxNzPerRow(amg_data) = max_nz_per_row;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetEuclidFile( void *data, char *euclidfile )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataEuclidFile(amg_data) = euclidfile;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetEuLevel( void *data, JX_Int eu_level )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataEuLevel(amg_data) = eu_level;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetEuSparseA( void *data, JX_Real eu_sparse_A )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataEuSparseA(amg_data) = eu_sparse_A;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetEuBJ( void *data, JX_Int eu_bj )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataEuBJ(amg_data) = eu_bj;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetConvCriteria
 * \author peghoty
 * \date 2009/07/25
 */
JX_Int
jx_PAMGSetConvCriteria( void *data, JX_Int conv_criteria )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataConvCriteria(amg_data) = conv_criteria;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetConvCriteria
 * \note The option of "coarsestsolverid" is similar to "relax_type", 
 *       except that coarsestsolverid could be set to 10 (gselim_piv)
 * \author peghoty
 * \date 2009/07/27
 */
JX_Int
jx_PAMGSetCoarsestSolverID( void *data, JX_Int coarsestsolverid )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataCoarsestSolverID(amg_data) = coarsestsolverid;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetCoarseThreshold
 * \author peghoty
 * \date 2010/04/14
 */
JX_Int
jx_PAMGSetCoarseThreshold( void *data, JX_Int coarse_threshold )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataCoarseThreshold(amg_data) = coarse_threshold;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PAMGSetCoarseRatio
 * \author peghoty
 * \date 2010/04/14
 */ 
JX_Int
jx_PAMGSetCoarseRatio( void *data, JX_Real coarse_ratio )
{
   jx_ParAMGData  *amg_data = data;
 
   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   jx_ParAMGDataCoarseRatio(amg_data) = coarse_ratio;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetConvFacThreshold( void *data, JX_Real convfac_threshold )
{
   jx_ParAMGData  *amg_data = data;

   if (!amg_data)
   {
      jx_printf("Warning! PAMG object empty!\n");
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParAMGDataConvFacThreshold(amg_data) = convfac_threshold;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetRAP2( void *data, JX_Int rap2 )
{
   jx_ParAMGData *amg_data = data;

   jx_ParAMGDataRAP2(amg_data) = rap2;

   return jx_error_flag;
}

JX_Int
jx_PAMGSetKeepTranspose( void *data, JX_Int keepTranspose )
{
   jx_ParAMGData *amg_data = data;

   jx_ParAMGDataKeepTranspose(amg_data) = keepTranspose;

   return jx_error_flag;
}
