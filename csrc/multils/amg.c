/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 * amg.c
 *
 * Created by peghoty 2010/08/29
 * Xiangtan University
 * peghoty@163.com
 *  
 */

#include "jx_multils.h"

/*!
 * \fn void *fsls_AMGInitialize
 * \brief Initialize the amg_solver for AMG
 * \author peghoty
 * \date 2010/08/29 
 */
void *
fsls_AMGInitialize()
{
   fsls_AMGData *amg_data = NULL;

   /* setup params */
   JX_Int      max_levels;
   JX_Real   strong_threshold;
   JX_Real   max_row_sum;
   JX_Real   A_trunc_factor;
   JX_Real   P_trunc_factor;
   JX_Int      A_max_elmts;
   JX_Int      P_max_elmts;
   JX_Int      coarsen_type;
   JX_Int      interp_type;
   JX_Int      coarse_threshold;
   JX_Int      S_mode;     

   /* solve params */
   JX_Real   tol;               // tolerance    
   JX_Int      max_iter;          // maximum number of iteration
   JX_Int      min_iter;          // minimum number of iteration
   JX_Int      cycle_type;        // cycle type
   JX_Int      corrective_type;   // corrective type or iterative type when cycling   
   JX_Real   relax_weight;      // relaxation weight
   JX_Int      relax_type;        // relaxation type
   JX_Int      relax_order;       // relaxation order
   JX_Int      relax_sweeps;      // relaxation sweeps 
   JX_Int      relax_sym;         // symmetricly relax or not?
   JX_Int      poly_degree;       // degree of polynomial for Polynomial relaxation
   JX_Int      coarsest_solver;   // solver type on the coarsest level
   JX_Int      level_of_fill_in;  // the threshold 'p' in ILU(p)
   JX_Int      ilu_smooth_level;  // number of levels use that employ ILU as smoother   
   JX_Real   rhsnorm_threshold; // for the iteration control, the following strategy is used 
                               // when rhsnorm_threshold != 0:
                               // 1. if ||rhs|| > rhsnorm_threshold, use relative norm control
                               // 2. otherwise, use absolute norm control
                               // default of rhsnorm_threshold: 0.0

   /* output params */
   JX_Int      print_level;

  /*-----------------------------------------------------------------------
   * Setup default values for parameters
   *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels       = 25;
   strong_threshold = 0.25;
   max_row_sum      = 0.9;
   A_trunc_factor   = 0.0;
   P_trunc_factor   = 0.0;
   A_max_elmts      = 0;
   P_max_elmts      = 0;
   coarsen_type     = 0;
   interp_type      = 0;
   coarse_threshold = 9; 
   S_mode           = 0;

   /* solve params */
   tol               = 1.0e-8;
   max_iter          = 100;
   min_iter          = 0;
   cycle_type        = 1;
   corrective_type   = 0;
   relax_weight      = 1.0;
   relax_type        = Smooth_GS;
   relax_order       = CPFIRST; 
   relax_sweeps      = 1;  
   relax_sym         = 1;
   poly_degree       = 3;
   coarsest_solver   = Smooth_GE;   
   level_of_fill_in  = 0;
   ilu_smooth_level  = 0;   
   rhsnorm_threshold = 0.0; 

   /* output params */
   print_level       = 3;

  /*----------------------------------------------------------
   * Create the fsls_AMGData structure and return
   *--------------------------------------------------------*/
   amg_data = fsls_CTAlloc(fsls_AMGData, 1);
   fsls_AMGSetMaxLevels(amg_data, max_levels);
   fsls_AMGSetStrongThreshold(amg_data, strong_threshold); 
   fsls_AMGSetMaxRowSum(amg_data, max_row_sum);
   fsls_AMGSetATruncFactor(amg_data, A_trunc_factor);
   fsls_AMGSetPTruncFactor(amg_data, P_trunc_factor);
   fsls_AMGSetAMaxElmts(amg_data, A_max_elmts);
   fsls_AMGSetPMaxElmts(amg_data, P_max_elmts);
   fsls_AMGSetCoarsenType(amg_data, coarsen_type);
   fsls_AMGSetInterpType(amg_data, interp_type);
   fsls_AMGSetCoarseThreshold(amg_data, coarse_threshold);
   fsls_AMGSetSMode(amg_data, S_mode);
   fsls_AMGSetTol(amg_data, tol);    
   fsls_AMGSetMaxIter(amg_data, max_iter);
   fsls_AMGSetMinIter(amg_data, min_iter);
   fsls_AMGSetCycleType(amg_data, cycle_type);
   fsls_AMGSetCorrectiveType(amg_data, corrective_type);
   fsls_AMGSetRelaxType(amg_data, relax_type);
   fsls_AMGSetRelaxOrder(amg_data, relax_order);
   fsls_AMGSetRelaxWeight(amg_data, relax_weight);
   fsls_AMGSetRelaxSweeps(amg_data, relax_sweeps);
   fsls_AMGSetRelaxSym(amg_data, relax_sym);
   fsls_AMGSetPolyDegree(amg_data, poly_degree);
   fsls_AMGSetCoarsestSolver(amg_data, coarsest_solver);
   fsls_AMGSetLevelOfFillIn(amg_data, level_of_fill_in);
   fsls_AMGSetILUSmoothLevel(amg_data, ilu_smooth_level);
   fsls_AMGSetRhsNrmThreshold(amg_data, rhsnorm_threshold);
   fsls_AMGSetPrintLevel(amg_data, print_level);
   
   return (void *) amg_data;
}

/*!
 * \fn fsls_AMGSet**
 * \brief Set parameters for the amg_solver
 * \author peghoty
 * \date 2010/08/29 
 */
JX_Int
fsls_AMGSetMaxLevels( void *data, JX_Int max_levels )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataMaxLevels(amg_data) = max_levels;
   return (ierr);
}

JX_Int
fsls_AMGSetStrongThreshold( void *data, JX_Real strong_threshold )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataStrongThreshold(amg_data) = strong_threshold;
   return (ierr);
}

JX_Int
fsls_AMGSetMaxRowSum( void *data, JX_Real max_row_sum )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataMaxRowSum(amg_data) = max_row_sum;
   return (ierr);
}

JX_Int
fsls_AMGSetATruncFactor( void *data, JX_Real A_trunc_factor )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataATruncFactor(amg_data) = A_trunc_factor;
   return (ierr);
}

JX_Int
fsls_AMGSetPTruncFactor( void *data, JX_Real P_trunc_factor )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataPTruncFactor(amg_data) = P_trunc_factor;
   return (ierr);
}

JX_Int
fsls_AMGSetAMaxElmts( void *data, JX_Int A_max_elmts )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataAMaxElmts(amg_data) = A_max_elmts;
   return (ierr);
}

JX_Int
fsls_AMGSetPMaxElmts( void *data, JX_Int P_max_elmts )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataPMaxElmts(amg_data) = P_max_elmts;
   return (ierr);
}

JX_Int
fsls_AMGSetCoarsenType( void *data, JX_Int coarsen_type )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataCoarsenType(amg_data) = coarsen_type;
   return (ierr);
}

JX_Int
fsls_AMGSetInterpType( void *data, JX_Int interp_type )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataInterpType(amg_data) = interp_type;
   return (ierr);
}

JX_Int
fsls_AMGSetCoarseThreshold( void *data, JX_Int coarse_threshold )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataCoarseThreshold(amg_data) = coarse_threshold;
   return (ierr);
}

JX_Int
fsls_AMGSetSMode( void *data, JX_Int S_mode )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataSMode(amg_data) = S_mode;
   return (ierr);
}

JX_Int
fsls_AMGSetTol( void *data, JX_Real tol )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataTol(amg_data) = tol;
   return (ierr);
}

JX_Int
fsls_AMGSetMaxIter( void *data, JX_Int max_iter )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataMaxIter(amg_data) = max_iter;
   return (ierr);
} 

JX_Int
fsls_AMGSetMinIter( void *data, JX_Int min_iter )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataMinIter(amg_data) = min_iter;
   return (ierr);
} 

JX_Int
fsls_AMGSetCycleType( void *data, JX_Int cycle_type )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataCycleType(amg_data) = cycle_type;
   return (ierr);
}

JX_Int
fsls_AMGSetCorrectiveType( void *data, JX_Int corrective_type )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataCorrectiveType(amg_data) = corrective_type;
   return (ierr);
}

JX_Int
fsls_AMGSetRelaxWeight( void  *data, JX_Real relax_weight )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataRelaxWeight(amg_data) = relax_weight;
   return (ierr);
}

JX_Int
fsls_AMGSetRelaxType( void  *data, JX_Int relax_type )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataRelaxType(amg_data) = relax_type;
   return (ierr);
}

JX_Int 
fsls_AMGSetRelaxOrder( void  *data, JX_Int relax_order )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataRelaxOrder(amg_data) = relax_order;
   return (ierr);
}

JX_Int
fsls_AMGSetRelaxSweeps( void  *data, JX_Int relax_sweeps )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataRelaxSweeps(amg_data) = relax_sweeps;
   return (ierr);
}

JX_Int
fsls_AMGSetRelaxSym( void  *data, JX_Int relax_sym )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataRelaxSym(amg_data) = relax_sym;
   return (ierr);
}

JX_Int
fsls_AMGSetPolyDegree( void  *data, JX_Int poly_degree )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataPolyDegree(amg_data) = poly_degree;
   return (ierr);
}

JX_Int 
fsls_AMGSetCoarsestSolver( void  *data, JX_Int coarsest_solver )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataCoarsestSolver(amg_data) = coarsest_solver;
   return (ierr);
}

JX_Int
fsls_AMGSetLevelOfFillIn( void  *data, JX_Int level_of_fill_in )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataLevelOfFillIn(amg_data) = level_of_fill_in;
   return (ierr);
}

JX_Int
fsls_AMGSetILUSmoothLevel( void  *data, JX_Int ilu_smooth_level )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataILUSmoothLevel(amg_data) = ilu_smooth_level;
   return (ierr);
}

JX_Int
fsls_AMGSetRhsNrmThreshold( void  *data, JX_Real rhsnorm_threshold )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataRhsNrmThreshold(amg_data) = rhsnorm_threshold;
   return (ierr);
}

JX_Int
fsls_AMGSetPrintLevel( void *data, JX_Int print_level )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataPrintLevel(amg_data) = print_level;
   return (ierr);
}

JX_Int
fsls_AMGSetCycleCount( void *data, JX_Int cycle_count )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataCycleCount(amg_data) = cycle_count;
   return (ierr);
}

JX_Int
fsls_AMGGetCycleCount( void *data, JX_Int *cycle_count )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   *cycle_count = fsls_AMGDataCycleCount(amg_data);
   return (ierr);
}

JX_Int
fsls_AMGSetLastRelNrm( void *data, JX_Real last_rel_nrm )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataLastRelNrm(amg_data) = last_rel_nrm;
   return (ierr);
}

JX_Int
fsls_AMGGetLastRelNrm( void *data, JX_Real *last_rel_nrm )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   *last_rel_nrm = fsls_AMGDataLastRelNrm(amg_data);
   return (ierr);
}

JX_Int 
fsls_AMGSetAveConvFactor( void *data, JX_Real ave_conv_factor )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   fsls_AMGDataAveConvFactor(amg_data) = ave_conv_factor;
   return (ierr);
}

JX_Int 
fsls_AMGGetAveConvFactor( void *data, JX_Real *ave_conv_factor )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   *ave_conv_factor = fsls_AMGDataAveConvFactor(amg_data);
   return (ierr);
}

JX_Int
fsls_AMGSetGridRelaxSweeps( void *data, JX_Int *grid_relax_sweeps )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   if (fsls_AMGDataGridRelaxSweeps(amg_data))
   {
      fsls_TFree(fsls_AMGDataGridRelaxSweeps(amg_data));
   }
   fsls_AMGDataGridRelaxSweeps(amg_data) = grid_relax_sweeps;
   return (ierr);
}

JX_Int
fsls_AMGSetGridRelaxType( void *data, JX_Int *grid_relax_type )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   if (fsls_AMGDataGridRelaxType(amg_data)) 
   {  
      fsls_TFree(fsls_AMGDataGridRelaxType(amg_data));
   }
   fsls_AMGDataGridRelaxType(amg_data) = grid_relax_type;
   return (ierr);
}

JX_Int
fsls_AMGSetGridRelaxOrder( void *data, JX_Int *grid_relax_order )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   if (fsls_AMGDataGridRelaxOrder(amg_data))
   {  
      fsls_TFree(fsls_AMGDataGridRelaxOrder(amg_data));
   }
   fsls_AMGDataGridRelaxOrder(amg_data) = grid_relax_order;
   return (ierr);
}

JX_Int
fsls_AMGSetGridRelaxWeight( void *data, JX_Real *grid_relax_weight )
{
   JX_Int ierr = 0;
   fsls_AMGData *amg_data = data;
   if (fsls_AMGDataGridRelaxWeight(amg_data))
   {    
      fsls_TFree(fsls_AMGDataGridRelaxWeight(amg_data));
   }
   fsls_AMGDataGridRelaxWeight(amg_data) = grid_relax_weight; 
   return (ierr);
}

/*!
 * \fn void fsls_AMGFinalize
 * \brief Finalize the amg_solver for AMG
 * \author peghoty
 * \date 2010/08/30 
 */
void
fsls_AMGFinalize( void *data )
{
   fsls_AMGData       *amg_data         = data;
   JX_Int                 num_levels       = fsls_AMGDataNumLevels(amg_data);
   JX_Int                 ilu_smooth_level = fsls_AMGDataILUSmoothLevel(amg_data);
   fsls_RelaxILUData  *relax_ilu_data   = fsls_AMGDataRelaxILUData(amg_data);   
   JX_Int i;   

  /*-------------------------------------------------------------------------------------
   *  free grid_relax_sweeps, grid_relax_type, grid_relax_order and grid_relax_weight
   *-----------------------------------------------------------------------------------*/
   if (fsls_AMGDataGridRelaxSweeps(amg_data))
      fsls_TFree(fsls_AMGDataGridRelaxSweeps(amg_data));
   if (fsls_AMGDataGridRelaxType(amg_data))
      fsls_TFree(fsls_AMGDataGridRelaxType(amg_data));   
   if (fsls_AMGDataGridRelaxOrder(amg_data))
      fsls_TFree(fsls_AMGDataGridRelaxOrder(amg_data));
   if (fsls_AMGDataGridRelaxWeight(amg_data))
      fsls_TFree(fsls_AMGDataGridRelaxWeight(amg_data)); 

  /*------------------------------------------------------------
   *  free A_array
   *-----------------------------------------------------------*/
   if (fsls_AMGDataAArray(amg_data))
   {
      for (i = 1; i < num_levels; i ++)
      {
         if (fsls_AMGDataAArray(amg_data)[i])
           fsls_CSRMatrixDestroy(fsls_AMGDataAArray(amg_data)[i]);
      }
      fsls_TFree(fsls_AMGDataAArray(amg_data));
   }

  /*------------------------------------------------------------
   *  free P_array
   *-----------------------------------------------------------*/
   if (fsls_AMGDataPArray(amg_data))
   {
      for (i = 0; i < num_levels-1; i ++)
      {
         if (fsls_AMGDataPArray(amg_data)[i])
            fsls_CSRMatrixDestroy(fsls_AMGDataPArray(amg_data)[i]);
      }
      fsls_TFree(fsls_AMGDataPArray(amg_data));
   }   

  /*------------------------------------------------------------
   *  F_array
   *-----------------------------------------------------------*/
   if (fsls_AMGDataFArray(amg_data))
   {
      for (i = 1; i < num_levels; i ++)
      {
         if (fsls_AMGDataFArray(amg_data)[i])
            fsls_SeqVectorDestroy(fsls_AMGDataFArray(amg_data)[i]);
      }
      fsls_TFree(fsls_AMGDataFArray(amg_data));
   }     
   
  /*------------------------------------------------------------
   *  free U_array
   *-----------------------------------------------------------*/
   if (fsls_AMGDataUArray(amg_data))
   {
      for (i = 1; i < num_levels; i ++)
      {
         if (fsls_AMGDataUArray(amg_data)[i])
            fsls_SeqVectorDestroy(fsls_AMGDataUArray(amg_data)[i]);
      }
      fsls_TFree(fsls_AMGDataUArray(amg_data));
   }

  /*------------------------------------------------------------
   *  free CF_marker_array
   *-----------------------------------------------------------*/
   if (fsls_AMGDataCFMarkerArray(amg_data))
   {
      for (i = 0; i < num_levels-1; i ++)
      {
         if (fsls_AMGDataCFMarkerArray(amg_data)[i])
            fsls_TFree(fsls_AMGDataCFMarkerArray(amg_data)[i]);
      }
      fsls_TFree(fsls_AMGDataCFMarkerArray(amg_data));
   }
   
  /*------------------------------------------------------------
   *  free RNrm_array
   *-----------------------------------------------------------*/      
   if (fsls_AMGDataRNrmArray(amg_data))
   {
      for (i = 0; i < num_levels-1; i ++)
      {
         if (fsls_AMGDataRNrmArray(amg_data)[i])
            fsls_SeqVectorDestroy(fsls_AMGDataRNrmArray(amg_data)[i]);
      }
      fsls_TFree(fsls_AMGDataRNrmArray(amg_data));
   }
   
  /*------------------------------------------------------------
   *  free Vtemp and full
   *-----------------------------------------------------------*/
   if (fsls_AMGDataVtemp(amg_data))
      fsls_SeqVectorDestroy(fsls_AMGDataVtemp(amg_data));
   if (fsls_AMGDataFull(amg_data))   
   fsls_TFree(fsls_AMGDataFull(amg_data));
   
  /*------------------------------------------------------------
   *  free temp1,temp2,temp3
   *-----------------------------------------------------------*/   
   if (fsls_AMGDataTemp1(amg_data))
      fsls_SeqVectorDestroy(fsls_AMGDataTemp1(amg_data));
   if (fsls_AMGDataTemp2(amg_data))
      fsls_SeqVectorDestroy(fsls_AMGDataTemp2(amg_data));
   if (fsls_AMGDataTemp3(amg_data))
      fsls_SeqVectorDestroy(fsls_AMGDataTemp3(amg_data));
      
  /*------------------------------------------------------------
   *  free relax_ilu_data     2010/12/08
   *-----------------------------------------------------------*/  
   if (ilu_smooth_level > 0)
   {
      if (relax_ilu_data)
      {
         for (i = 0; i < ilu_smooth_level; i ++)
         {
            fsls_TFree(fsls_RelaxILUDataIndexArray(relax_ilu_data)[i]);
            fsls_TFree(fsls_RelaxILUDataValueArray(relax_ilu_data)[i]);
         }
         fsls_TFree(fsls_RelaxILUDataIndexArray(relax_ilu_data));
         fsls_TFree(fsls_RelaxILUDataValueArray(relax_ilu_data));
         if (fsls_RelaxILUDataWork(relax_ilu_data))
         {
            fsls_TFree(fsls_RelaxILUDataWork(relax_ilu_data));
         }
         fsls_TFree(relax_ilu_data);
      }
   }      
      
   fsls_TFree(amg_data);
}

/*!
 * \fn void fsls_AMGComplx
 * \brief Compute the grid and operater complexity.
 * \author peghoty
 * \date 2010/08/30 
 */
void
fsls_AMGComplx( void *data, JX_Real *grid, JX_Real *operater )
{
   fsls_AMGData    *amg_data   = data;
   JX_Int              num_levels = fsls_AMGDataNumLevels(amg_data);
   fsls_CSRMatrix **A_array    = fsls_AMGDataAArray(amg_data);
    
   JX_Int      total_coeffs    = 0;
   JX_Int      total_variables = 0;
   JX_Real   operat_cmplxty  = 0.0;
   JX_Real   grid_cmplxty    = 0.0;
   JX_Int     *num_coeffs      = fsls_CTAlloc(JX_Int, num_levels);
   JX_Int     *num_variables   = fsls_CTAlloc(JX_Int, num_levels);
   JX_Int      j; 
   
   for (j = 0; j < num_levels; j ++)
   {
      num_coeffs[j]    = fsls_CSRMatrixNumNonzeros(A_array[j]);
      num_variables[j] = fsls_CSRMatrixNumRows(A_array[j]);
   }
  
   /* Grid complexity and Operater complexity */
   for (j = 0;j < fsls_AMGDataNumLevels(amg_data); j ++)
   {
      total_coeffs += num_coeffs[j];
      total_variables += num_variables[j];
   }

   grid_cmplxty   = ((JX_Real) total_variables) / ((JX_Real) num_variables[0]);
   operat_cmplxty = ((JX_Real) total_coeffs) / ((JX_Real) num_coeffs[0]);
  
   fsls_TFree(num_coeffs);
   fsls_TFree(num_variables);
   
   *grid = grid_cmplxty;
   *operater = operat_cmplxty;
}

/*!
 * \fn JX_Int fsls_AMGSetup
 * \brief SETUP phase of AMG.
 * \param *amg_vdata pointer to the amg_solver. 
 * \param *A the pointer to the matrix of a given linear system.
 * \param *f pointer to the right hand side vector for the linear system.
 * \param *u pointer to the initial guess vector for the linear system.
 * \note some remarks
 *       (1) The 'f' and 'u' are not necessary for fsls_AMGSetup, they are kept 
 *           just in case that 'f' is needed when coarsening strategies are used.     
 *       (2) The parameters-setting for relaxation is encapsulated in a function
 *           in order to save some memory since the real num_levels has been obtained
 *           after the Setup process.
 * \note Modified the special treatment for 'num_levels == 1'. 2011/04/28
 * \author peghoty
 * \date 2010/08/29 
 */
JX_Int
fsls_AMGSetup( void *amg_vdata, fsls_CSRMatrix *A, fsls_Vector *f, fsls_Vector *u )
{ 
   fsls_AMGData *amg_data = amg_vdata;

   /* variables in amg_data */
   JX_Int      max_levels        = fsls_AMGDataMaxLevels(amg_data);       // 25 by default
   JX_Real   strong_threshold  = fsls_AMGDataStrongThreshold(amg_data); // 0.25 by default
   JX_Real   max_row_sum       = fsls_AMGDataMaxRowSum(amg_data);       // 0.9 by default
   JX_Real   A_trunc_factor    = fsls_AMGDataATruncFactor(amg_data);    // 0.0 by default
   JX_Real   P_trunc_factor    = fsls_AMGDataPTruncFactor(amg_data);    // 0.0 by default
   JX_Int      A_max_elmts       = fsls_AMGDataAMaxElmts(amg_data);       // 0 by default
   JX_Int      P_max_elmts       = fsls_AMGDataPMaxElmts(amg_data);       // 0 by default
   JX_Int      interp_type       = fsls_AMGDataInterpType(amg_data);      // 0 by default
   JX_Int	    coarsen_type      = fsls_AMGDataCoarsenType(amg_data);     // 0 by default
   JX_Int      coarse_threshold  = fsls_AMGDataCoarseThreshold(amg_data); // 100 by default  
   JX_Int      S_mode            = fsls_AMGDataSMode(amg_data);           // 0 by default
   JX_Int      print_level       = fsls_AMGDataPrintLevel(amg_data);      // 3 by default   

   /* Data Structure variables */
   fsls_CSRMatrix  **A_array          = NULL;
   fsls_CSRMatrix  **P_array          = NULL;   
   fsls_Vector     **F_array          = NULL;
   fsls_Vector     **U_array          = NULL;
   JX_Int             **CF_marker_array  = NULL;    

   /* local variables or arrays */
   JX_Int  use_Jacobi_on_coarsest = 0;
   JX_Int  Setup_err_flag = 0;
   JX_Int  not_finished_coarsening = 1;
   JX_Int  level = 0; 
   JX_Int  coarse_size;
   JX_Int  fine_size;
   JX_Int  j;
   JX_Int  num_levels;     
   JX_Int             *CF_marker = NULL;  
   fsls_ICSRMatrix *S         = NULL;
   fsls_CSRMatrix  *P         = NULL;
   fsls_CSRMatrix  *A_H       = NULL; 

   if (max_levels == 1)
   {
      return 0;
   }

   A_array = fsls_CTAlloc(fsls_CSRMatrix*, max_levels);
   P_array = fsls_CTAlloc(fsls_CSRMatrix*, max_levels-1); 
   CF_marker_array = fsls_CTAlloc(JX_Int*, max_levels);
   A_array[0] = A;

  /*------------------------------------------------------
   *  Enter Coarsening Loop
   *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {
      fine_size = fsls_CSRMatrixNumRows(A_array[level]);

     /*---------------------------------------------------------
      * Step 1: Coarsen (C/F spliting) on the current level
      *--------------------------------------------------------*/
     
      if (coarsen_type == RS_ORIGINAL) /* original RS from hypre */
      {
          fsls_AMGCreateS(A_array[level], strong_threshold, S_mode, &S);
          fsls_AMGCoarsenRugeLoL(A_array[level], S, &CF_marker, &coarse_size);
      }
      else if (coarsen_type == RS_MODIFIED) /* modified RS by peghoty */
      {
          fsls_AMGCreateSNew(A_array[level], strong_threshold, max_row_sum, S_mode, &S);
          fsls_AMGCoarsenRSNew(A_array[level], S, &CF_marker, &coarse_size);
      }
      else if (coarsen_type == CLJP) /* CLJP */
      {
          fsls_AMGCreateSNew(A_array[level], strong_threshold, max_row_sum, S_mode, &S);
          fsls_AMGCoarsenCLJPNew(A_array[level], S, &CF_marker, &coarse_size);           
      }     

      /* if no coarse-grid, stop coarsening. 2011/04/28 */
      if (coarse_size == 0 || coarse_size == fine_size) 
      {
         use_Jacobi_on_coarsest = 1;
         if (S)
         {
            fsls_ICSRMatrixDestroy(S);
         }
         if (CF_marker)
         {
            fsls_TFree(CF_marker);
         }
         break;
      }

      CF_marker_array[level] = CF_marker;
      
     /*-------------------------------------------------------------
      * Step 2: Construct interpolation on the current level
      *-----------------------------------------------------------*/

      if (interp_type == CLASSIC)  /* modified classic interpolation */
      {
         fsls_AMGClassicalInterp(A_array[level], CF_marker_array[level], S, &P);
      }
      else if (interp_type == DIRECT) /* direct interpolation */
      {
         fsls_AMGDirectInterp(A_array[level], CF_marker_array[level], S, &P);
      }
   
      if (P_trunc_factor > 0 || P_max_elmts > 0)
      {
         fsls_AMGPTruncation(P, P_trunc_factor, P_max_elmts);
      }   

      P_array[level] = P; 
      fsls_ICSRMatrixDestroy(S);
      
     /*------------------------------------------------------------------------------------
      * Step 3: Build coarse-grid operator, A_array[level+1] by R*A*P
      *-----------------------------------------------------------------------------------*/
    
      fsls_AMGBuildCoarseOperator(P_array[level], A_array[level], P_array[level], &A_H);     
    
      if (A_trunc_factor > 0 || A_max_elmts > 0)
      {
         fsls_AMGAHTruncation(A_H, A_trunc_factor, A_max_elmts);
      }

      ++ level;
      A_array[level] = A_H;
      
     /*----------------------------------------------------------------------------------------------
      * Step 4: How to terminate the coarsening ?
      *--------------------------------------------------------------------------------------------*/  
          
      if ( level+1 >= max_levels || coarse_size <= coarse_threshold )
      {
         not_finished_coarsening = 0;
      }
   }
   
   /* real number of levels */
   num_levels = level + 1; 
   fsls_AMGDataNumLevels(amg_data) = num_levels;

  /*------------------------------------------------------------------------------
   * fill 'A_array', 'P_array', 'CF_marker_array' into amg_data data structure
   *----------------------------------------------------------------------------*/
   fsls_AMGDataCFMarkerArray(amg_data) = CF_marker_array;
   fsls_AMGDataAArray(amg_data)        = A_array;
   fsls_AMGDataPArray(amg_data)        = P_array;

  /*-----------------------------------------------------------------------
   * Setup F and U arrays
   *-----------------------------------------------------------------------*/
   F_array = fsls_CTAlloc(fsls_Vector*, num_levels);
   U_array = fsls_CTAlloc(fsls_Vector*, num_levels);
   for (j = 1; j < num_levels; j ++)
   {
      F_array[j] = fsls_SeqVectorCreate(fsls_CSRMatrixNumRows(A_array[j]));
      fsls_SeqVectorInitialize(F_array[j]);

      U_array[j] = fsls_SeqVectorCreate(fsls_CSRMatrixNumRows(A_array[j]));
      fsls_SeqVectorInitialize(U_array[j]);
   }
   fsls_AMGDataFArray(amg_data)    = F_array;
   fsls_AMGDataUArray(amg_data)    = U_array;
   fsls_AMGDataPreMatrix(amg_data) = A;
  
  /*-------------------------------------------------------------
   * Relaxation setting for the Solve phase of AMG 
   * add 'use_Jacobi_on_coarsest'   2011/04/28
   *-----------------------------------------------------------*/
   fsls_AMGSetRelaxParams(amg_data, use_Jacobi_on_coarsest);

  /*-----------------------------------------------------
   * Print some staff
   *---------------------------------------------------*/
   if (print_level == 1 || print_level == 3) 
   {
      fsls_AMGSetupParams(amg_data);
   } 

   return (Setup_err_flag);
} 

/*!
 * \fn fsls_AMGSetRelaxParams
 * \brief Set parameters and allocate some memories 
 *        for relaxation in the solusion phase of AMG.
 * \author peghoty
 * \date 2010/08/29 
 */
JX_Int
fsls_AMGSetRelaxParams( void *data, JX_Int use_Jacobi_on_coarsest )
{
   fsls_AMGData       *amg_data = data;
   JX_Int                 num_levels;
   JX_Int                 coarsest_solver;   // solver type on the coarsest level
   JX_Int                *grid_relax_sweeps; // number of relaxation sweeps on each level
   JX_Int                *grid_relax_type;   // type of relaxation on each level  
   JX_Int                *grid_relax_order;  // order of relaxation on each level
   JX_Real             *grid_relax_weight; // weight of relaxation on each level
   JX_Real              relax_weight;      // relaxation weight
   JX_Int                 relax_type;        // relaxation type
   JX_Int                 relax_order;       // relaxation order
   JX_Int                 relax_sweeps;      // relaxation sweeps 
   JX_Int                 relax_sym;         // symmetricly relax or not?  
   fsls_Vector       **RNrm_array;        // row-norm data for Kaczmarz relaxation
   JX_Int                 level_of_fill_in;  // level of fill in for ILU(p) decomposition
   JX_Int                 ilu_smooth_level;  // number of levels use that employ ILU as smoother  
   fsls_RelaxILUData  *relax_ilu_data; 
   JX_Int               **index_array;       // index part of the L/U in MSR format 
   JX_Real            **value_array;       // value part of the L/U in MSR format 
   JX_Real             *work;
   fsls_CSRMatrix    **A_array;  
   
   fsls_Vector        *temp1 = NULL;
   fsls_Vector        *temp2 = NULL;
   fsls_Vector        *temp3 = NULL; 
   
   fsls_Vector        *Vtemp = NULL;
   JX_Real             *full  = NULL;     
   
   JX_Int j, cs, num_levels_minus_one;
   JX_Int level = 0, ierr = 0;
   JX_Int finest_size;
   JX_Int coarsest_size;
      
   num_levels       = fsls_AMGDataNumLevels(amg_data);
   coarsest_solver  = fsls_AMGDataCoarsestSolver(amg_data);
   A_array          = fsls_AMGDataAArray(amg_data);
   relax_weight     = fsls_AMGDataRelaxWeight(amg_data);
   relax_type       = fsls_AMGDataRelaxType(amg_data);
   relax_order      = fsls_AMGDataRelaxOrder(amg_data);
   relax_sweeps     = fsls_AMGDataRelaxSweeps(amg_data);
   relax_sym        = fsls_AMGDataRelaxSym(amg_data);
   level_of_fill_in = fsls_AMGDataLevelOfFillIn(amg_data); 
   ilu_smooth_level = fsls_AMGDataILUSmoothLevel(amg_data);
   finest_size      = fsls_CSRMatrixNumRows(A_array[0]);
   
   num_levels_minus_one = num_levels - 1;
   
   /* allocate memory */
   grid_relax_sweeps = fsls_CTAlloc(JX_Int, num_levels);
   grid_relax_type   = fsls_CTAlloc(JX_Int, num_levels);
   grid_relax_order  = fsls_CTAlloc(JX_Int, 5);
   grid_relax_weight = fsls_CTAlloc(JX_Real, num_levels); 
   RNrm_array        = fsls_CTAlloc(fsls_Vector *, num_levels_minus_one); 
   
  /*----------------------------------------------------------
   * Set 'grid_relax_sweeps', 'grid_relax_type'
   *---------------------------------------------------------*/ 

   /* finest level and down(up) cycle */
   for (j = 0; j < num_levels_minus_one; j ++)
   {
      grid_relax_sweeps[j] = relax_sweeps;
      grid_relax_type[j] = relax_type;   
   }

   /* coarsest level */
   grid_relax_sweeps[num_levels_minus_one] = 1;
   grid_relax_type[num_levels_minus_one] = coarsest_solver;

  /*-----------------------------------------------------------------------
   * Set 'grid_relax_order'
   * NOTE: Option to have numerous relaxation sweeps per level.  
   *       here we would use C/F relaxation going down, F/C going 
   *       up, for all sweeps by default.
   *---------------------------------------------------------------------*/ 
   
   grid_relax_order[FINEST_LEFT]  = relax_order;
   grid_relax_order[DOWN_CYCLE]   = relax_order;
   grid_relax_order[COARSEST]     = relax_order;
   grid_relax_order[UP_CYCLE]     = relax_order;
   grid_relax_order[FINEST_RIGHT] = relax_order;
   if (relax_sym)
   {
      grid_relax_order[UP_CYCLE]     = -relax_order;
      grid_relax_order[FINEST_RIGHT] = -relax_order;
   }
 
  /*-------------------------------------------------------
   * Set 'grid_relax_weight'
   *------------------------------------------------------*/
   
   for (j = 0; j < num_levels; j ++)
   {
      grid_relax_weight[j] = relax_weight;
   }
   
   fsls_AMGSetGridRelaxSweeps(amg_data, grid_relax_sweeps);
   fsls_AMGSetGridRelaxType(amg_data, grid_relax_type);
   fsls_AMGSetGridRelaxOrder(amg_data, grid_relax_order);
   fsls_AMGSetGridRelaxWeight(amg_data, grid_relax_weight);
   
   /* row norm data for Kaczmarz relaxation */
   for (j = 0; j < num_levels_minus_one; j ++)
   {
      if (grid_relax_type[j] == Smooth_KACZMARZ)
      {
         RNrm_array[j] = fsls_SeqVectorCreate(fsls_CSRMatrixNumRows(A_array[j]));
         fsls_SeqVectorInitialize(RNrm_array[j]);
         fsls_RowNormCompute(A_array[j], RNrm_array[j]);       
      }
   }
   fsls_AMGDataRNrmArray(amg_data) = RNrm_array;
   
   /* auxiliary vector for Polynomial relaxation */
   if (relax_type == Smooth_POLY)
   {
      temp1 = fsls_SeqVectorCreate(finest_size);
      temp2 = fsls_SeqVectorCreate(finest_size);
      temp3 = fsls_SeqVectorCreate(finest_size);
      fsls_SeqVectorInitialize(temp1);
      fsls_SeqVectorInitialize(temp2);
      fsls_SeqVectorInitialize(temp3);
      fsls_AMGDataTemp1(amg_data) = temp1;
      fsls_AMGDataTemp2(amg_data) = temp2;
      fsls_AMGDataTemp3(amg_data) = temp3;    
   }

  /*---------------------------------------------------------
   * If 'ilu_smooth_level' > 0, prepare the ILU Data.
   *--------------------------------------------------------*/
   
   if (ilu_smooth_level > num_levels_minus_one)
   {  
      /* in case that the coarsest level be included */
      ilu_smooth_level = num_levels_minus_one;  
      fsls_AMGSetILUSmoothLevel(amg_data, ilu_smooth_level);
   }
   
   if (ilu_smooth_level > 0)
   {
      relax_ilu_data = fsls_CTAlloc(fsls_RelaxILUData, 1);
      
      work        = fsls_CTAlloc(JX_Real, 2*finest_size);
      index_array = fsls_CTAlloc(JX_Int *, ilu_smooth_level);
      value_array = fsls_CTAlloc(JX_Real *, ilu_smooth_level);

      for (level = 0; level < ilu_smooth_level; level ++)
      {
         fsls_ILUp_Decomp( A_array[level], level_of_fill_in, 
                           &index_array[level], &value_array[level] );
         
         /* change the relaxation type into ILUp */
         grid_relax_type[level] = Smooth_ILUP;
      }
      fsls_RelaxILUDataIndexArray(relax_ilu_data) = index_array;
      fsls_RelaxILUDataValueArray(relax_ilu_data) = value_array;
      fsls_RelaxILUDataWork(relax_ilu_data)       = work;
      
      fsls_AMGDataRelaxILUData(amg_data) = relax_ilu_data;
   }  


  /*-----------------------------------------------
   * fill 'Vtemp'
   *----------------------------------------------*/
   Vtemp = fsls_SeqVectorCreate(finest_size);
   fsls_SeqVectorInitialize(Vtemp);
   fsls_AMGDataVtemp(amg_data) = Vtemp;
   

  /*------------------------------------------------------------------------------------------
   * Decide what should be done for the coarsest level according to 'use_Jacobi_on_coarsest'
   * (1) use_Jacobi_on_coarsest = 1
   *     A simple Jacobi Iteration will be performed on the coarsest level since the matrix
   *  is strongly diagonal dominant. No extra memory is need!
   * (2) use_Jacobi_on_coarsest = 0
   *     The size of the coarse matrix will <= coarse_threshold, which is suitable for a 
   *  direct solver.
   *     (a) If coarsest_solver = Smooth_GE or Smooth_GEP, a full matrix 
   *         should be allocated here.
   *     (b) If coarsest_solver = Smooth_MUMPS, index-displacing(from C to Fortran style)
   *         work should be done here.             
   *                                         peghoty   2011/09/17
   *-----------------------------------------------------------------------------------------*/      
         
   if (use_Jacobi_on_coarsest == 1)
   {
      grid_relax_sweeps[num_levels_minus_one] = 1;
      grid_relax_type[num_levels_minus_one]   = Smooth_WJACOBI;
      grid_relax_weight[num_levels_minus_one] = 1.0;
      if (num_levels == 1)
      {
         grid_relax_order[FINEST_LEFT] = ASCEND;
      }
      else
      {
         grid_relax_order[COARSEST] = ASCEND;
      }
   } 
   else
   {   
      cs = grid_relax_type[num_levels_minus_one];
      if (cs == Smooth_GE || cs == Smooth_GEP)
      { 
         coarsest_size = fsls_CSRMatrixNumRows(A_array[num_levels_minus_one]);
         full = fsls_CTAlloc(JX_Real, coarsest_size*coarsest_size);
         fsls_AMGDataFull(amg_data) = full;
      }   
   }   

   return (ierr);
}


/*!
 * \fn JX_Int fsls_AMGSetupParams
 * \brief Routine for getting matrix statistics from setup
 * \author peghoty
 * \date 2010/08/29 
 */
JX_Int
fsls_AMGSetupParams( void *amg_vdata )
{
   fsls_AMGData *amg_data = amg_vdata;

   /* Data Structure variables */
   fsls_CSRMatrix **A_array = NULL;
   fsls_CSRMatrix **P_array = NULL;
   
   JX_Int interp_type  = fsls_AMGDataInterpType(amg_data);
   JX_Int coarsen_type = fsls_AMGDataCoarsenType(amg_data);  

   JX_Int num_levels; 
   JX_Int num_nonzeros;

   /* Local variables */
   JX_Int    *A_i    = NULL;
   JX_Real *A_data = NULL;

   JX_Int    *P_i    = NULL;
   JX_Real *P_data = NULL;

   JX_Int     level;
   JX_Int     i,j;
   JX_Int     fine_size;
   JX_Int     coarse_size;
   JX_Int     entries;
   JX_Int     total_entries;
   JX_Int     min_entries;
   JX_Int     max_entries;
   JX_Real  avg_entries;
   JX_Real  rowsum;
   JX_Real  min_rowsum;
   JX_Real  max_rowsum;
   JX_Real  sparse;
   JX_Real  min_weight;
   JX_Real  max_weight;
   JX_Real  op_complxty   = 0.0;
   JX_Real  grid_complxty = 0.0;
   JX_Real  num_nz0;
   JX_Real  num_var0;

   A_array = fsls_AMGDataAArray(amg_data);
   P_array = fsls_AMGDataPArray(amg_data);
   num_levels = fsls_AMGDataNumLevels(amg_data);
 
   jx_printf("\n \033[36mAMG SETUP PARAMETERS:\033[00m\n\n");
   jx_printf(" Strength threshold = %.2f\n", fsls_AMGDataStrongThreshold(amg_data));
   jx_printf(" Max_Row_Sum        = %.2f\n", fsls_AMGDataMaxRowSum(amg_data));
   jx_printf(" Max levels         = %d\n", fsls_AMGDataMaxLevels(amg_data));
   jx_printf(" Num levels         = %d\n\n", num_levels);
   
   if (num_levels >= 1) /* when num_levels = 1, we also output the information of coarsening and interpolation */
   {
   
      jx_printf(" Coarsening    Type : ");
      switch (coarsen_type)
      {
         case RS_ORIGINAL:
            jx_printf(" original RS from hypre\n");
         break;
         case RS_MODIFIED:
            jx_printf(" modified RS by peghoty\n");
         break;
         case CLJP:
            jx_printf(" CLJP\n");
         break;
         default:
            jx_printf(" UNKNOWN\n");
         break;            
      }

      jx_printf(" Interpolation Type : ");
      switch (interp_type)
      {
         case CLASSIC:
            jx_printf(" modified classic interpolation\n");
         break;
         case DIRECT:
            jx_printf(" direct interpolation\n");
         break;
         case ENERGYMIN:
            jx_printf(" Energy-Minimizing based interpolation\n");
         break;      
         default:
            jx_printf(" UNKNOWN\n");
         break;            
      }
      
   } // end if (num_levels > 1)
   
   jx_printf("\n >>> \033[34mOperator Matrix Information\033[00m:\n\n");
   jx_printf(" -----------------------------------------------------------------------\n");
   jx_printf("               nonzero          entries per row          row sums       \n");
   jx_printf(" Lev   Rows    entries  sparse   min  max  avg        min        max    \n");
   jx_printf(" -----------------------------------------------------------------------\n");

  /*---------------------------------------------------------
   *  Enter Statistics Loop (1st)
   *-------------------------------------------------------*/
   num_var0 = (JX_Real) fsls_CSRMatrixNumRows(A_array[0]);
   num_nz0  = (JX_Real) fsls_CSRMatrixNumNonzeros(A_array[0]);
 
   for (level = 0; level < num_levels; level ++)
   {
       A_i    = fsls_CSRMatrixI(A_array[level]);
       A_data = fsls_CSRMatrixData(A_array[level]);

       fine_size = fsls_CSRMatrixNumRows(A_array[level]);
       num_nonzeros = fsls_CSRMatrixNumNonzeros(A_array[level]);
       sparse = num_nonzeros /((JX_Real) fine_size * (JX_Real) fine_size);
       
       /* modified by peghoty 2009/12/29 */
       op_complxty   += (JX_Real)num_nonzeros;
       grid_complxty += (JX_Real)fine_size;
       
       min_entries = A_i[1]-A_i[0];  // the nonzeros of the first row 
       max_entries = 0;
       total_entries = 0;
       min_rowsum = 0.0;
       max_rowsum = 0.0;
       
       /* the row sum of the first row in A[level] */
       for (j = A_i[0]; j < A_i[1]; j ++)
       {
          min_rowsum += A_data[j];
       }

       max_rowsum = min_rowsum;

       for (j = 0; j < fine_size; j ++)
       {
           entries = A_i[j+1] - A_i[j];
           min_entries = fsls_min(entries, min_entries);
           max_entries = fsls_max(entries, max_entries);
           total_entries += entries;

           /* the row sum of the j-th row */
           rowsum = 0.0;
           for (i = A_i[j]; i < A_i[j+1]; i ++)
           {
             rowsum += A_data[i];
           }
           
           min_rowsum = fsls_min(rowsum, min_rowsum);
           max_rowsum = fsls_max(rowsum, max_rowsum);
       }
  
       /* the average nonzeros per row */
       avg_entries = ((JX_Real) total_entries) / ((JX_Real) fine_size);

       jx_printf(" %2d %7d %9d    %0.3f  %3d  %3d",
               level, fine_size, num_nonzeros, sparse, min_entries, max_entries);
       jx_printf("  %5.1f    %9.2e  %9.2e\n", avg_entries, min_rowsum, max_rowsum);
   }
   /* modified by peghoty 2009/12/29 */
   op_complxty   /= num_nz0;
   grid_complxty /= num_var0;
   
   
   if (num_levels > 1)
   {   
       
      jx_printf( "\n\n >>> \033[34mInterpolation Matrix Information\033[00m:\n\n");
      jx_printf(" -------------------------------------------------------------------------\n");
      jx_printf("                     Entries/Row    min        max          row sums      \n");
      jx_printf(" Lev   Rows x Cols     min max     weight     weight      min       max   \n");
      jx_printf(" -------------------------------------------------------------------------\n");

     /*-----------------------------------------------------
      *  Enter Statistics Loop (2nd)
      *-----------------------------------------------------*/

      for (level = 0; level < num_levels-1; level ++)
      {
          P_i    = fsls_CSRMatrixI(P_array[level]);
          P_data = fsls_CSRMatrixData(P_array[level]);

          fine_size    = fsls_CSRMatrixNumRows(P_array[level]);
          coarse_size  = fsls_CSRMatrixNumCols(P_array[level]);
          num_nonzeros = fsls_CSRMatrixNumNonzeros(P_array[level]);

          min_entries   = P_i[1]-P_i[0];
          max_entries   = 0;
          total_entries = 0;
          min_rowsum    = 0.0;
          max_rowsum    = 0.0;
          min_weight    = P_data[0];
          max_weight    = 0.0;

          /* the row sum of the first row in P[level] */
          for (j = P_i[0]; j < P_i[1]; j ++)
          {
             min_rowsum += P_data[j];
          }
       
          max_rowsum = min_rowsum;

          for (j = 0; j < num_nonzeros; j ++)
          {
             if (P_data[j] != 1.0)
             {
                min_weight = fsls_min(min_weight,P_data[j]);
                max_weight = fsls_max(max_weight,P_data[j]);
             }
          }

          for (j = 0; j < fine_size; j ++)
          {
              entries = P_i[j+1] - P_i[j];
              min_entries = fsls_min(entries, min_entries);
              max_entries = fsls_max(entries, max_entries);
              total_entries += entries;

              rowsum = 0.0;
              for (i = P_i[j]; i < P_i[j+1]; i ++)
              {
                rowsum += P_data[i];
              }

              min_rowsum = fsls_min(rowsum, min_rowsum);
              max_rowsum = fsls_max(rowsum, max_rowsum);
          }

          jx_printf(" %2d %7d x %-7d %3d %3d",level,fine_size,coarse_size,min_entries,max_entries);
          jx_printf("    %9.2e  %9.2e %9.2e  %9.2e\n",min_weight,max_weight,min_rowsum,max_rowsum);
      }
   
   } // end if (num_levels > 1)

   jx_printf("\n >>> \033[34mGrid Complexity\033[00m    : %.3f\n", grid_complxty);   
   jx_printf(" >>> \033[34mOperator Complexity\033[00m: %.3f\n", op_complxty); 

   return(0);
}

/*!
 * \fn JX_Int fsls_AMGSolve
 * \brief SOLVE phase of AMG.
 * \param *amg_vdata pointer to the object for AMG solving
 * \param *A pointer to the coefficient matrix for the linear system
 * \param *f pointer to the right hand side vector for the linear system
 * \param *u pointer to the initial guess vector for the linear system
 *           as an input parameter, and pointer to the approximate  
 *           solution vector as an output parameter.
 * \return Solve_err_flag
 * \author peghoty
 * \date 2010/09/12 
 */
JX_Int
fsls_AMGSolve( void           *amg_vdata,
               fsls_CSRMatrix *A,
               fsls_Vector    *f,
               fsls_Vector    *u  )
{
   fsls_AMGData *amg_data = amg_vdata;
   
   JX_Int corrective_type = fsls_AMGDataCorrectiveType(amg_data);  
   JX_Int Solve_err_flag  = 0;

   JX_Int max_levels    = fsls_AMGDataMaxLevels(amg_data);  
   JX_Int num_variables = fsls_CSRMatrixNumRows(A);
      
   //----------------------------------------------------------------//
   //           Special treatment for max_levels = 1 cases           //
   //----------------------------------------------------------------//
   
   if (max_levels == 1)
   {
      jx_printf("\n >>> \033[31mmax_levels = 1, Direct Solver are employed!\033[00m\n");
      if (num_variables < GEMAXDOF)
      {
         JX_Real *full = fsls_CTAlloc(JX_Real, num_variables*num_variables);
         fsls_AMGRelaxGE(full, A, f, u);
         fsls_TFree(full);
      }
      else
      {
         jx_printf("\n >>> \033[31mThe system of size %d can't be solved by direct solver!\033[00m\n", num_variables);
      } 
      return 0;
   }    
    
   if (corrective_type) /* corrective type */
   {
      Solve_err_flag = fsls_AMGSolve_CorrectiveType(amg_data, A, f, u);
   }
   else  /* iterative type (by default) */
   {
      Solve_err_flag = fsls_AMGSolve_IterativeType(amg_data, A, f, u);
   }
   
   return Solve_err_flag; 
}

/*!
 * \fn JX_Int fsls_AMGSolve_IterativeType
 * \brief SOLVE phase of AMG (iterative type).
 * \param *amg_vdata pointer to the object for AMG solving
 * \param *A pointer to the coefficient matrix for the linear system
 * \param *f pointer to the right hand side vector for the linear system
 * \param *u pointer to the initial guess vector for the linear system
 *           as an input parameter, and pointer to the approximate  
 *           solution vector as an output parameter.
 * \return Solve_err_flag
 * \author peghoty
 * \date 2010/08/29 
 */
JX_Int
fsls_AMGSolve_IterativeType( void           *amg_vdata,
                             fsls_CSRMatrix *A,
                             fsls_Vector    *f,
                             fsls_Vector    *u  )
{
   fsls_AMGData  *amg_data = amg_vdata;
   
   /* variables in amg_data */   
   JX_Real tol               = fsls_AMGDataTol(amg_data);
   JX_Int    max_iter          = fsls_AMGDataMaxIter(amg_data);
   JX_Int    min_iter          = fsls_AMGDataMinIter(amg_data);
   JX_Int    print_level       = fsls_AMGDataPrintLevel(amg_data);
   JX_Real rhsnorm_threshold = fsls_AMGDataRhsNrmThreshold(amg_data);
     
   /* Data Structure variables */
   fsls_CSRMatrix **A_array = fsls_AMGDataAArray(amg_data);
   fsls_Vector    **F_array = fsls_AMGDataFArray(amg_data);
   fsls_Vector    **U_array = fsls_AMGDataUArray(amg_data);
   fsls_Vector     *Vtemp   = fsls_AMGDataVtemp(amg_data);
        
   /* local variables */
   JX_Int      Solve_err_flag   = 0;
   JX_Int      cycle_count      = 0;
   JX_Real   conv_factor      = 0.0;
   JX_Real   res_nrm          = 0.0;
   JX_Real   res_nrm_init     = 0.0;
   JX_Real   relative_res_nrm = 0.0;
   JX_Real   rhs_nrm          = 0.0; 
   JX_Real   res_nrm_old      = 0.0;

   A_array[0] = A; // this has been done in the setup phase 
   F_array[0] = f;
   U_array[0] = u;
   
  /*-----------------------------------------------------------------------
   *  Print some initial info
   *----------------------------------------------------------------------*/

   if (print_level == 2 || print_level == 3)
   { 
      fsls_AMGSolveParams(amg_data);
      jx_printf("\n\n \033[36mAMG SOLUTION INFO:\033[00m\n\n");
   }

  /*-----------------------------------------------------------------------
   *    Compute rhs vector and initial fine-grid residual
   *----------------------------------------------------------------------*/
   rhs_nrm = fsls_SeqVectorL2Norm(f);       // rhs_nrm = ||f||_2
   fsls_SeqVectorCopy(F_array[0], Vtemp);
   fsls_CSRMatrixMatvec(-1.0, A_array[0], U_array[0], 1.0, Vtemp);
   res_nrm = fsls_SeqVectorL2Norm(Vtemp);   // res_nrm = ||f - A*u||_2
   res_nrm_init = res_nrm; 

   /* modified by peghoty.  2011/05/21 */
   if (rhsnorm_threshold)
   {
      if (rhs_nrm > rhsnorm_threshold)
         relative_res_nrm = res_nrm_init / rhs_nrm;
      else
         relative_res_nrm = res_nrm_init;
   }
   else // if (rhsnorm_threshold == 0)
   {
      if (rhs_nrm)
         relative_res_nrm = res_nrm_init / rhs_nrm;
      else
         relative_res_nrm = res_nrm_init;
   } 

   if (print_level == 2 || print_level == 3)
   {   
      jx_printf(" ----------------------------------------------------\n"); 
      if (rhs_nrm > rhsnorm_threshold) 
      {   
         jx_printf("    cycle    residual      factor      rel_res_nrm   \n");
      }
      else
      {
         jx_printf("    cycle    residual      factor      abs_res_nrm   \n");
      }
      jx_printf(" ----------------------------------------------------\n");
      jx_printf("      0    %e                %e\n", res_nrm_init, relative_res_nrm);
   }

  /*-----------------------------------------------------------------------
   *    Main V-cycle loop
   *----------------------------------------------------------------------*/

   while ( (relative_res_nrm >= tol || cycle_count < min_iter) && 
           (cycle_count < max_iter) && Solve_err_flag == 0 )
   {
      /* one AMG Cycle */
      Solve_err_flag = fsls_AMGCycle(amg_data, F_array, U_array); 

      res_nrm_old = res_nrm;

      /* Compute fine-grid residual and residual norm */
      fsls_SeqVectorCopy(F_array[0], Vtemp);
      fsls_CSRMatrixMatvec(-1.0, A_array[0], U_array[0], 1.0, Vtemp);
      res_nrm = fsls_SeqVectorL2Norm(Vtemp);  // res_nrm = ||f - A*u||_2

      /* Compute convergence factor */
      if (res_nrm_old)
         conv_factor = res_nrm / res_nrm_old;
      else
         conv_factor = res_nrm;   
      
      /* Compute relative_res_nrm, modified by peghoty. 2011/05/21 */
      if (rhsnorm_threshold)
      {
         if (rhs_nrm > rhsnorm_threshold)
            relative_res_nrm = res_nrm / rhs_nrm;
         else
            relative_res_nrm = res_nrm;
      }
      else // if (rhsnorm_threshold == 0)
      {
         if (rhs_nrm)
            relative_res_nrm = res_nrm / rhs_nrm;
         else
            relative_res_nrm = res_nrm;
      }       
      
      ++ cycle_count;

      if (print_level == 2 || print_level == 3)
      { 
         jx_printf("    %3d    %e   %f     %e \n",cycle_count,res_nrm,conv_factor,relative_res_nrm);
      }
   }

   if (cycle_count == max_iter && relative_res_nrm >= tol) Solve_err_flag = 1;

   fsls_AMGSetCycleCount(amg_data, cycle_count);
   fsls_AMGSetLastRelNrm(amg_data, relative_res_nrm);

   /* Compute average convergence factor */
   if (res_nrm_init)
   {
      conv_factor = pow( (res_nrm / res_nrm_init), (1.0 / ((JX_Real) cycle_count)) );
   }
   fsls_AMGSetAveConvFactor(amg_data, conv_factor);

   if (print_level == 2 || print_level == 3)
   {
      if (Solve_err_flag == 1)
      {
         jx_printf("\n\n ==============================================");
         jx_printf("\n  NOTE: Convergence tolerance was not achieved\n");
         jx_printf("        within the allowed %d V-cycles\n",max_iter);
         jx_printf(" ==============================================\n");
      }
      jx_printf("\n >>> \033[34mAverage Convergence Factor\033[00m = %f\n\n", conv_factor);
   }

   return (Solve_err_flag);
}

/*!
 * \fn JX_Int fsls_AMGSolve_CorrectiveType
 * \brief SOLVE phase of AMG (corrective type).
 * \param *amg_vdata pointer to the object for AMG solving
 * \param *A pointer to the coefficient matrix for the linear system
 * \param *f pointer to the right hand side vector for the linear system
 * \param *u pointer to the initial guess vector for the linear system
 *           as an input parameter, and pointer to the approximate  
 *           solution vector as an output parameter.
 * \return Solve_err_flag
 * \author peghoty
 * \date 2010/09/13 
 */
JX_Int
fsls_AMGSolve_CorrectiveType( void           *amg_vdata,
                              fsls_CSRMatrix *A,
                              fsls_Vector    *f,
                              fsls_Vector    *u  )
{
   fsls_AMGData  *amg_data = amg_vdata;
   
   /* variables in amg_data */   
   JX_Real tol               = fsls_AMGDataTol(amg_data);
   JX_Int    max_iter          = fsls_AMGDataMaxIter(amg_data);
   JX_Int    print_level       = fsls_AMGDataPrintLevel(amg_data);
   JX_Real rhsnorm_threshold = fsls_AMGDataRhsNrmThreshold(amg_data);
     
   /* Data Structure variables */
   fsls_CSRMatrix **A_array = fsls_AMGDataAArray(amg_data);
   fsls_Vector    **F_array = fsls_AMGDataFArray(amg_data);
   fsls_Vector    **U_array = fsls_AMGDataUArray(amg_data);
   fsls_Vector     *Vtemp   = fsls_AMGDataVtemp(amg_data);
        
   /* local variables */
   JX_Int      Solve_err_flag   = 0;
   JX_Int      cycle_count      = 0;
   JX_Real   conv_factor      = 0.0;
   JX_Real   res_nrm          = 0.0;
   JX_Real   res_nrm_init     = 0.0;
   JX_Real   relative_res_nrm = 0.0;
   JX_Real   rhs_nrm          = 0.0; 
   JX_Real   res_nrm_old      = 0.0;
   JX_Int      num_variables    = fsls_CSRMatrixNumRows(A);

   A_array[0] = A;
   
   F_array[0] = fsls_SeqVectorCreate(num_variables);
   fsls_SeqVectorInitialize(F_array[0]);
   U_array[0] = fsls_SeqVectorCreate(num_variables);
   fsls_SeqVectorInitialize(U_array[0]);      
   
  /*-----------------------------------------------------------------------
   *  Print some initial info
   *----------------------------------------------------------------------*/

   if (print_level == 2 || print_level == 3)
   { 
      fsls_AMGSolveParams(amg_data);
      jx_printf("\n\n \033[36mAMG SOLUTION INFO:\033[00m\n\n");
   }

  /*-----------------------------------------------------------------------
   *    Compute rhs vector and initial fine-grid residual
   *----------------------------------------------------------------------*/
   rhs_nrm = fsls_SeqVectorL2Norm(f);       // rhs_nrm = ||f||_2
   fsls_SeqVectorCopy(f, Vtemp);
   fsls_CSRMatrixMatvec(-1.0, A_array[0], u, 1.0, Vtemp);
   res_nrm = fsls_SeqVectorL2Norm(Vtemp);   // res_nrm = ||f - A*u||_2
   res_nrm_init = res_nrm; 

   /* modified by peghoty.  2011/05/21 */
   if (rhsnorm_threshold)
   {
      if (rhs_nrm > rhsnorm_threshold)
         relative_res_nrm = res_nrm_init / rhs_nrm;
      else
         relative_res_nrm = res_nrm_init;
   }
   else // if (rhsnorm_threshold == 0)
   {
      if (rhs_nrm)
         relative_res_nrm = res_nrm_init / rhs_nrm;
      else
         relative_res_nrm = res_nrm_init;
   }    

   if (print_level == 2 || print_level == 3)
   {   
      jx_printf(" ----------------------------------------------------\n");     
      jx_printf("    cycle    residual      factor      rel_res_nrm   \n");
      jx_printf(" ----------------------------------------------------\n");
      jx_printf("      0    %e                %e\n", res_nrm_init,relative_res_nrm);
   }


  /*-----------------------------------------------------------------------
   *    Main V-cycle loop
   *----------------------------------------------------------------------*/

   while ( relative_res_nrm >= tol && cycle_count < max_iter && Solve_err_flag == 0 )
   {
      /* copy the residual as the right hand side of the Residual Equation */
      fsls_SeqVectorCopy(Vtemp, F_array[0]);
      
      /* reset the initial guess for the Residual Equation */
      fsls_SeqVectorSetConstantValues(U_array[0], 0.0);
      
      /* one AMG Cycle */
      Solve_err_flag = fsls_AMGCycle(amg_data, F_array, U_array); 
      
      /* get the new iterative approximation by adding the correction */
      fsls_SeqVectorAxpy(1.0, U_array[0], u); // u = u + U_array[0]

      res_nrm_old = res_nrm;

      /* Compute fine-grid residual and residual norm */
      fsls_SeqVectorCopy(f, Vtemp);
      fsls_CSRMatrixMatvec(-1.0, A_array[0], u, 1.0, Vtemp);
      res_nrm = fsls_SeqVectorL2Norm(Vtemp);  // res_nrm = ||f - A*u||_2

      /* Compute convergence factor */
      if (res_nrm_old)
         conv_factor = res_nrm / res_nrm_old;
      else
         conv_factor = res_nrm;   

      /* Compute relative_res_nrm, modified by peghoty. 2011/05/21 */
      if (rhsnorm_threshold)
      {
         if (rhs_nrm > rhsnorm_threshold)
            relative_res_nrm = res_nrm / rhs_nrm;
         else
            relative_res_nrm = res_nrm;
      }
      else // if (rhsnorm_threshold == 0)
      {
         if (rhs_nrm)
            relative_res_nrm = res_nrm / rhs_nrm;
         else
            relative_res_nrm = res_nrm;
      }

      ++ cycle_count;

      if (print_level == 2 || print_level == 3)
      { 
         jx_printf("    %3d    %e   %f     %e \n",cycle_count,res_nrm,conv_factor,relative_res_nrm);
      }
   }
   fsls_SeqVectorDestroy(F_array[0]);
   fsls_SeqVectorDestroy(U_array[0]);

   if (cycle_count == max_iter && relative_res_nrm >= tol) Solve_err_flag = 1;

   fsls_AMGSetCycleCount(amg_data, cycle_count);
   fsls_AMGSetLastRelNrm(amg_data, relative_res_nrm);

   /* Compute average convergence factor */
   if (res_nrm_init)
   {
      conv_factor = pow( (res_nrm / res_nrm_init), (1.0 / ((JX_Real) cycle_count)) );
   }
   fsls_AMGSetAveConvFactor(amg_data, conv_factor);

   if (print_level == 2 || print_level == 3)
   {
      if (Solve_err_flag == 1)
      {
         jx_printf("\n\n ==============================================");
         jx_printf("\n  NOTE: Convergence tolerance was not achieved\n");
         jx_printf("        within the allowed %d V-cycles\n",max_iter);
         jx_printf(" ==============================================\n");
      }
      jx_printf("\n >>> \033[34mAverage Convergence Factor\033[00m = %f\n\n", conv_factor);
   }

   return (Solve_err_flag);
}

/*!
 * \fn fsls_AMGCycle
 * \brief AMG cycle
 * \param *amg_vdata pointer to AMGData object
 * \param **F_array  rhs vector of each level
 * \param **U_array  app vector of each level
 * \author peghoty
 * \date 2010/08/30
 */
JX_Int
fsls_AMGCycle( void          *amg_vdata, 
               fsls_Vector  **F_array,
               fsls_Vector  **U_array )
{
   fsls_AMGData *amg_data = amg_vdata;

   /* members in amg_data */
   JX_Int                cycle_type        = fsls_AMGDataCycleType(amg_data);
   JX_Int                num_levels        = fsls_AMGDataNumLevels(amg_data);
   JX_Int               *grid_relax_sweeps = fsls_AMGDataGridRelaxSweeps(amg_data);
   JX_Int               *grid_relax_type   = fsls_AMGDataGridRelaxType(amg_data);
   JX_Int               *grid_relax_order  = fsls_AMGDataGridRelaxOrder(amg_data);
   JX_Real            *grid_relax_weight = fsls_AMGDataGridRelaxWeight(amg_data);
   fsls_CSRMatrix   **A_array           = fsls_AMGDataAArray(amg_data);
   fsls_CSRMatrix   **P_array           = fsls_AMGDataPArray(amg_data);
   fsls_Vector       *Vtemp             = fsls_AMGDataVtemp(amg_data);
   fsls_Vector      **RNrm_array        = fsls_AMGDataRNrmArray(amg_data);   
   JX_Int              **CF_marker_array   = fsls_AMGDataCFMarkerArray(amg_data);
   JX_Real            *full              = fsls_AMGDataFull(amg_data);
   fsls_RelaxILUData *relax_ilu_data    = fsls_AMGDataRelaxILUData(amg_data);
   
   fsls_Vector     *temp1 = NULL;
   fsls_Vector     *temp2 = NULL;
   fsls_Vector     *temp3 = NULL;

   /* Local variables  */
   JX_Int *lev_counter = NULL;
   JX_Int  Solve_err_flag = 0;
   JX_Int  j,k,level;
   JX_Int  Not_Finished;
   JX_Int  cycle_param; /* 0: finesest(L) 1: down cycle 2: coarsest 3: up cycle 4: finesest(R) */
   JX_Int  coarse_grid, fine_grid;
   JX_Int  relax_sweeps, relax_type, relax_order;
   JX_Int  poly_degree = fsls_AMGDataPolyDegree(amg_data);
   JX_Real relax_weight;
   JX_Int  smooter = fsls_AMGDataRelaxType(amg_data);
   
   if (smooter == Smooth_POLY)
   {
      temp1 = fsls_AMGDataTemp1(amg_data);
      temp2 = fsls_AMGDataTemp2(amg_data);
      temp3 = fsls_AMGDataTemp3(amg_data); 
   }  

  /*---------------------------------------------------------------------
   * Initialize cycling control counter
   *
   *  Cycling is controlled using a level counter: lev_counter[k]
   *     
   *  Each time relaxation is performed on level k, the
   *  counter is decremented by 1. If the counter is then
   *  negative, we go to the next finer level. If non-
   *  negative, we go to the next coarser level. The
   *  following actions control cycling:
   *     
   *  a. lev_counter[0] is initialized to 1.
   *  b. lev_counter[k] is initialized to cycle_type for k > 0.
   *     
   *  c. During cycling, when going down to level k, lev_counter[k]
   *     is set to the max of (lev_counter[k],cycle_type)
   *---------------------------------------------------------------------*/
   
   lev_counter = fsls_CTAlloc(JX_Int, num_levels);
   lev_counter[0] = 1;
   for (k = 1; k < num_levels; k ++) 
   {
      lev_counter[k] = cycle_type;
   }
   Not_Finished = 1;
   level        = 0;
   cycle_param  = FINEST_LEFT;
   
  /*----------------------------------------------------------------*
   *                  Main loop of cycling                          *
   *----------------------------------------------------------------*/
  
   while (Not_Finished)
   {
      relax_sweeps = grid_relax_sweeps[level];
      relax_type   = grid_relax_type[level];
      relax_weight = grid_relax_weight[level];
      relax_order  = grid_relax_order[cycle_param];

     /*--------------------------------------------------------
      *   Do the relaxation relax_sweeps times
      *-------------------------------------------------------*/
       
      for (j = 0; j < relax_sweeps; j ++)
      {
         switch (relax_type)
         {
            case Smooth_WJACOBI: /* w-Jacobi */
            {
               Solve_err_flag = fsls_AMGRelaxJacobi( A_array[level], 
                                                     F_array[level],
                                                     CF_marker_array[level],
                                                     relax_order,
                                                     relax_weight,
                                                     U_array[level],
                                                     Vtemp );
            }
            break;
            
            case Smooth_GS: /* Gauss-Seidel */
            {
               Solve_err_flag = fsls_AMGRelaxGS( A_array[level], 
                                                 F_array[level],
                                                 CF_marker_array[level],
                                                 relax_order,
                                                 U_array[level] );
            }
            break;
            
            case Smooth_SGS: /* symmetric Gauss-Seidel */
            {
               Solve_err_flag = fsls_AMGRelaxGS( A_array[level], F_array[level], NULL, ASCEND,  U_array[level] );
               Solve_err_flag = fsls_AMGRelaxGS( A_array[level], F_array[level], NULL, DESCEND, U_array[level] );
            }
            break;
            
            case Smooth_SOR: /* Successive Over Relaxation */
            {
               Solve_err_flag = fsls_AMGRelaxSOR( A_array[level], 
                                                  F_array[level],
                                                  CF_marker_array[level],
                                                  relax_order,
                                                  relax_weight,
                                                  U_array[level] );
            }
            break;

            case Smooth_KACZMARZ: /* Kaczmarz relaxation */
            {
               Solve_err_flag = fsls_AMGRelaxKaczmarz( A_array[level], 
                                                       F_array[level],
                                                       CF_marker_array[level],
                                                       relax_order,
                                                       relax_weight,
                                                       U_array[level],
                                                       RNrm_array[level] );
            }
            break;
            
            case Smooth_POLY: /* polynomial relaxation */
            {
               Solve_err_flag = fsls_AMGRelaxPolynomial( poly_degree,
                                                         A_array[level],
                                                         F_array[level],
                                                         U_array[level],
                                                         Vtemp, temp1, temp2, temp3 );
            }
            break;
 
            case Smooth_ILUP: /* ILU(p) relaxation */
            {
               Solve_err_flag = fsls_AMGRelaxILUp( level,
                                                   relax_ilu_data,
                                                   A_array[level],
                                                   F_array[level],
                                                   U_array[level],
                                                   Vtemp );                         
            }
            break;
            
            case Smooth_GE: /* Gaussian Elimination */
            {       
               Solve_err_flag = fsls_AMGRelaxGE( full,
                                                  A_array[level], 
                                                  F_array[level],
                                                  U_array[level] );
            }
            break;
            
            case Smooth_GEP: /* Gaussian Elimination with pivoting */
            {
               Solve_err_flag = fsls_AMGRelaxGEP( full,
                                                  A_array[level], 
                                                  F_array[level],
                                                  U_array[level] );
            }
            break;
         }

         if (Solve_err_flag != 0) return (Solve_err_flag);
       
      }

     /*------------------------------------------------------------------------
      * Decrement the control counter and determine which grid to visit next
      *----------------------------------------------------------------------*/

      -- lev_counter[level];
       
      if (lev_counter[level] >= 0 && level != num_levels-1)
      {
                               
        /*---------------------------------------------------------------
         * Visit coarser level next.  
         * Compute residual using fsls_CSRMatrixMatvec01.
         * Perform restriction using fsls_CSRMatrixMatvecT.
         * Reset counters and cycling parameters for coarse level
         *--------------------------------------------------------------*/

         fine_grid   = level;
         coarse_grid = level + 1;

         fsls_SeqVectorSetConstantValues(U_array[coarse_grid], 0.0);
          
         fsls_SeqVectorCopy(F_array[fine_grid],Vtemp);

         /* Vtemp = F_array[fine_grid] - A_array[fine_grid] * U_array[fine_grid] */
         fsls_CSRMatrixMatvec01(-1.0, A_array[fine_grid], U_array[fine_grid], 1.0, Vtemp); 

         /* F_array[coarse_grid] = {P_array[fine_grid]}^T * Vtemp */
         fsls_CSRMatrixMatvecT(1.0, P_array[fine_grid], Vtemp, 0.0, F_array[coarse_grid]);

         ++ level;
         
         lev_counter[level] = fsls_max(lev_counter[level],cycle_type);
         
         cycle_param = DOWN_CYCLE;
         
         if (level == num_levels-1) cycle_param = COARSEST;
      }

      else if (level != 0)
      {
                            
        /*---------------------------------------------------------------
         * Visit finer level next.
         * Interpolate and add correction using fsls_CSRMatrixMatvec.
         * Reset counters and cycling parameters for finer level.
         *--------------------------------------------------------------*/

         fine_grid   = level - 1;
         coarse_grid = level;

         /* U_array[fine_grid] = U_array[fine_grid] + P_array[fine_grid] * U_array[coarse_grid] */
         fsls_CSRMatrixMatvec01(1.0, P_array[fine_grid], U_array[coarse_grid], 1.0, U_array[fine_grid]);            
 
         -- level;
         
         cycle_param = UP_CYCLE;
         
         if (level == 0) cycle_param = FINEST_RIGHT;
      }
      else
      {
         Not_Finished = 0;
      }
   }

   fsls_TFree(lev_counter);

   return(Solve_err_flag);
}

/*!
 * \fn JX_Int fsls_AMGPrecond
 * \brief Iterative Process of AMG as preconditioner(z = Br).
 * \param *amg_vdata pointer to the object for AMG solving
 * \param *r pointer to the right hand side vector for the linear system
 * \param *z pointer to the initial guess vector for the linear system
 *           as an input parameter, and pointer to the new approximate  
 *           solution vector as an output parameter.
 * \author peghoty
 * \date 2010/12/24 
 */
JX_Int
fsls_AMGPrecond( void             *amg_vdata,
                 fsls_Vector      *r,
                 fsls_Vector      *z  )
{
   fsls_AMGData    *amg_data = amg_vdata;
   fsls_Vector    **F_array  = fsls_AMGDataFArray(amg_data);
   fsls_Vector    **U_array  = fsls_AMGDataUArray(amg_data);

   JX_Int max_iter    = fsls_AMGDataMaxIter(amg_data);        
   JX_Int cycle_count = 0;

   F_array[0] = r;
   U_array[0] = z;

   /* Generally, max_iter should be set to 1 when AMG is used as a 
      preconditioner, here, we keep the parameter max_iter for convenience 
      if max_iter should be larger than 1 */
   while (cycle_count < max_iter)
   {
      fsls_AMGCycle(amg_data, F_array, U_array); 
      cycle_count ++;
   }
 
   return 0;
}

/*!
 * \fn void fsls_AMGSolveParams
 * \brief Output some information of AMG Solve phase
 * \author peghoty
 * \date 2010/08/30 
 */
void  
fsls_AMGSolveParams( void *data )
{
   fsls_AMGData *amg_data = data;

   /* amg solve params */
   JX_Int    num_levels      = fsls_AMGDataNumLevels(amg_data);   
   JX_Int    max_iter        = fsls_AMGDataMaxIter(amg_data);
   JX_Int    cycle_type      = fsls_AMGDataCycleType(amg_data); 
   JX_Int    corrective_type = fsls_AMGDataCorrectiveType(amg_data);
   JX_Int    print_level     = fsls_AMGDataPrintLevel(amg_data);   
   JX_Real tol             = fsls_AMGDataTol(amg_data); 
    
   JX_Int    *grid_relax_sweeps = fsls_AMGDataGridRelaxSweeps(amg_data);
   JX_Int    *grid_relax_type   = fsls_AMGDataGridRelaxType(amg_data);
   JX_Int    *grid_relax_order  = fsls_AMGDataGridRelaxOrder(amg_data);

   JX_Int     cycle_param;   

   jx_printf("\n\n \033[36mAMG SOLVE PARAMETERS:\033[00m\n\n");
   jx_printf( " Real    Number of Levels           :  %d\n", num_levels);   
   jx_printf( " Maximum Number of Cycles           :  %d\n", max_iter);
   jx_printf( " Stopping Tolerance                 :  %.1e\n", tol); 
   jx_printf( " Cycle Type(1 = V, 2 = W, etc.)     :  %d\n", cycle_type);
   jx_printf( " Solve type(iterative or corrective): ");
   if (corrective_type)
   {
      jx_printf( " Corrective Type\n\n");
   }
   else
   {
      jx_printf( " Iterative Type\n\n");
   }
   
   jx_printf( " Relaxation Parameters:\n\n");

   if (num_levels == 1)
   {
      jx_printf( "   Visiting Grid         :  Finest/Coarsest\n");
      jx_printf( "   Number of Sweeps      :  %4d\n", grid_relax_sweeps[0]);
      jx_printf( "   Type of Relaxation    :  %4d\n", grid_relax_type[0]);
   }
   else if (num_levels == 2)
   {
      jx_printf( "   Visiting Grid         :  Finest  Coarsest\n");
      jx_printf( "   Number of Sweeps      :  %4d   %4d \n",
                 grid_relax_sweeps[0], grid_relax_sweeps[num_levels-1]);
      jx_printf( "   Type of Relaxation    :  %4d   %4d \n",
                 grid_relax_type[0], grid_relax_type[num_levels-1]);
   }
   else
   {
      jx_printf( "   Visiting Grid         :  Finest Down  Up  Coarsest\n");
      jx_printf( "   Number of Sweeps      :  %4d  %4d   %2d  %4d \n",
                 grid_relax_sweeps[0],grid_relax_sweeps[1],
                 grid_relax_sweeps[1],grid_relax_sweeps[num_levels-1]);
      jx_printf( "   Type of Relaxation    :  %4d  %4d   %2d  %4d \n",
                 grid_relax_type[0],grid_relax_type[1],
                 grid_relax_type[1],grid_relax_type[num_levels-1]);
   }
   jx_printf( "   ( 0|Jacobi    1|GS      2|SGS     3|SOR   4|Kaczmarz  \n");
   jx_printf( "     5|Poly      6|MUMPS   8|ILU(p)  9|GE   10|GEP )\n");  
    
   jx_printf( "   Order of Relaxation   :\n");

   if (num_levels == 1)
   {
      jx_printf( "      Finest Grid  --> ");
      switch (grid_relax_order[0])
      {
            case FPFIRST:
            jx_printf(" F points First\n"); break;
            case CPFIRST:
            jx_printf(" C points First\n"); break;
            case FP_ONLY:
            jx_printf(" F points Only\n"); break;
            case CP_ONLY:
            jx_printf(" C points Only\n"); break; 
            case ASCEND:
            jx_printf(" Ascendingly Order\n"); break;
            case DESCEND:
            jx_printf(" Descendingly Order\n"); break;                     
      }
   }
   else if (num_levels == 2)
   {
      jx_printf( "      Finest Grid     --> ");
      switch (grid_relax_order[0])
      {
            case FPFIRST:
            jx_printf(" F points First\n"); break;
            case CPFIRST:
            jx_printf(" C points First\n"); break;
            case FP_ONLY:
            jx_printf(" F points Only\n"); break;
            case CP_ONLY:
            jx_printf(" C points Only\n"); break; 
            case ASCEND:
            jx_printf(" Ascendingly Order\n"); break;
            case DESCEND:
            jx_printf(" Descendingly Order\n"); break;                     
      }
      jx_printf( "      Coarsest Grid   --> ");
      switch (grid_relax_order[2])
      {
            case FPFIRST:
            jx_printf(" F points First\n"); break;
            case CPFIRST:
            jx_printf(" C points First\n"); break;
            case FP_ONLY:
            jx_printf(" F points Only\n"); break;
            case CP_ONLY:
            jx_printf(" C points Only\n"); break; 
            case ASCEND:
            jx_printf(" Ascendingly Order\n"); break;
            case DESCEND:
            jx_printf(" Descendingly Order\n"); break;                     
      }      
   }
   else 
   {
      for (cycle_param = 0; cycle_param < 5; cycle_param ++)
      {
         //if (cycle_param != 2)
         {
            switch (cycle_param)
            {
               case 0:
               jx_printf( "      Finest Grid(L)  --> "); break;
               case 1:
               jx_printf( "          Down Cycle  --> "); break;
               case 2:
               jx_printf( "       Coarsest Grid  --> "); break;
               case 3:
               jx_printf( "            Up Cycle  --> "); break;
               case 4:
               jx_printf( "      Finest Grid(R)  --> "); break;    
            }
      
            switch (grid_relax_order[cycle_param])
            {
               case FPFIRST:
               jx_printf(" F points First\n"); break;
               case CPFIRST:
               jx_printf(" C points First\n"); break;
               case FP_ONLY:
               jx_printf(" F points Only\n"); break;
               case CP_ONLY:
               jx_printf(" C points Only\n"); break; 
               case ASCEND:
               jx_printf(" Ascendingly Order\n"); break;
               case DESCEND:
               jx_printf(" Descendingly Order\n"); break;                     
            }
         }
      }
   }

   jx_printf( "\n   Output Flag (print_level) :  %d \n", print_level);

   return;
}

/*!
 * \fn JX_Int fsls_AMGCoarsenCLJP
 * \brief CLJP coarsening algorithm
 * \param *A pointer to the matrix
 * \param *S pointer to the strength matrix of A
 * \param **CF_marker_ptr pointer to the pointer to the CF marker
 * \param *coarse_size_ptr pointer to the number of C points
 * \note the S are initialized 
 *                   / -1, i->j is a strong edge;
 *    S = (s_{ij}) = 
 *                   \ 0, otherwise.
 * \author peghoty
 * \date 2010/01/20
 */
JX_Int
fsls_AMGCoarsenCLJP( fsls_CSRMatrix   *A,
                     fsls_ICSRMatrix  *S,
                     JX_Int             **CF_marker_ptr,
                     JX_Int              *coarse_size_ptr )
{
   JX_Int     num_variables = fsls_CSRMatrixNumRows(A);
               
   JX_Int    *S_i    = NULL;
   JX_Int    *S_j    = NULL;
   JX_Int    *S_data = NULL;
                 
   JX_Int    *CF_marker = NULL;
   JX_Int     coarse_size;

   JX_Real *measure_array = NULL;
   JX_Int    *graph_array   = NULL;
   JX_Int     graph_size;

   JX_Int     i, j, k, jS, kS, ig;
   JX_Int     ierr = 0;

   S_i    = fsls_ICSRMatrixI(S);
   S_j    = fsls_ICSRMatrixJ(S);
   S_data = fsls_ICSRMatrixData(S);


  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 1: Compute the measures
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
  /*-------------------------------------------------------------
   * 1.1 Compute the true measures of each variable.
   * Remark: The measures are currently given by the column  
   *         sums of S. Hence, measure_array[i] is the number 
   *         of influences of variable i.
   *------------------------------------------------------------*/

   measure_array = fsls_CTAlloc(JX_Real, num_variables);

   for (i = 0; i < num_variables; i ++)
   {
      for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
      {
         j = S_j[jS];
         measure_array[j] ++;
      }
   }

  /*-------------------------------------------------------------
   * 1.2  Modify the measure_array so that each component
   *      are augmented by a random number between 0 and 1.
   *-----------------------------------------------------------*/
   
   fsls_InitAMGIndepSet(S, measure_array, 1.0);
   

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 2:  Initialize the graph array and the C/F marker array
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   graph_array = fsls_CTAlloc(JX_Int, num_variables);
   for (i = 0; i < num_variables; i ++)
   {
      graph_array[i] = i;
   }

   CF_marker = fsls_CTAlloc(JX_Int, num_variables);


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 3:  Loop until all points are either fine or coarse.
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   coarse_size = 0;             // record the currently number of C-points
   graph_size  = num_variables; // record the nodes left in the current working graph

   while (1)
   {
   
     /*------------------------------------------------
      * Set F-pts and update subgraph
      *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++)
      {
         i = graph_array[ig];  // ig: local; i: global

         if ( (CF_marker[i] != CPOINT) && (measure_array[i] < 1) )
         {
            /* set to be an F-pt */
            CF_marker[i] = FPOINT;

            /* make sure all dependencies have been accounted for */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               if (S_data[jS] < 0)  /* i.e. S_data[jS] = -1 */
               {
                  CF_marker[i] = UNDECIDED;
               }
            }
         }

         if (CF_marker[i])
         {
            measure_array[i] = 0;

            /* take point out of the subgraph */
            graph_size --;
            graph_array[ig] = graph_array[graph_size];
            graph_array[graph_size] = i;
            ig --; /* to make sure the variable swapped just now will be accessed next */
         }
      }
      

     /*------------------------------------------------
      * Test for convergence
      *------------------------------------------------*/

      if (graph_size == 0) break;

     /*------------------------------------------------------
      * Pick an independent set of points with
      * local maximal measure.
      *----------------------------------------------------*/

      fsls_AMGIndepSet( S, measure_array, 1.0, graph_array,
                        graph_size, CF_marker );


     /*-------------------------------------------------
      * Set C-pts and apply heuristics.
      *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++)
      {
         i = graph_array[ig];

        /*---------------------------------------------
         * Heuristic: C-pts don't interpolate from
         * neighbors that influence them.
         *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = CPOINT;
            coarse_size ++;

            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               if (S_data[jS] < 0)
               {
                  j = S_j[jS];
               
                  /* "remove" edge from S */
                  S_data[jS] = -S_data[jS];
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j] --;
                  }
               }
            }
         }


        /*---------------------------------------------
         * Heuristic: points that depend on a common
         * C-pt are less dependent on each other.
         *
         * NOTE: CF_marker is used to help check for
         * common C-pt's in the heuristic.
         *---------------------------------------------*/

         else
         {
            /* marked dependences */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               j = S_j[jS];

               if (CF_marker[j] > 0)
               {
                  if (S_data[jS] < 0)
                  {
                     /* "remove" edge from S */
                     S_data[jS] = -S_data[jS];
                  }

                  /* IMPORTANT: consider all dependencies */
                  if (S_data[jS])
                  {
                     /* temporarily modify CF_marker */
                     CF_marker[j] = COMMON_CPOINT;
                  }
               }
            }

            /* unmarked dependences */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               if (S_data[jS] < 0)
               {
                  j = S_j[jS]; /* j must be an undecided point */

                  /* check for common C-pt */
                  for (kS = S_i[j]; kS < S_i[j+1]; kS ++)
                  {
                     k = S_j[kS];

                     /* IMPORTANT: consider all dependencies */
                     if (S_data[kS])
                     {
                        if (CF_marker[k] == COMMON_CPOINT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_data[jS] = -S_data[jS];
                           measure_array[j] --;
                           break;
                        }
                     }
                  }
               }
            }

            /* reset CF_marker */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               j = S_j[jS];

               if (CF_marker[j] == COMMON_CPOINT)
               {
                  CF_marker[j] = CPOINT;
               }
            }
         }
      }
   }


  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   fsls_TFree(measure_array);
   fsls_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}


/*!
 * \fn JX_Int fsls_AMGCoarsenCLJPNew
 * \brief CLJP coarsening algorithm
 * \param *A pointer to the matrix
 * \param *S pointer to the strength matrix of A
 * \param **CF_marker_ptr pointer to the pointer to the CF marker
 * \param *coarse_size_ptr pointer to the number of C points
 * \note the S are initialized 
 *                   / -1, i->j is a strong edge;
 *    S = (s_{ij}) = 
 *                   \ 0, otherwise.
 * \remark Add the treatment to special F-points.
 * \author peghoty
 * \date 2010/01/20
 */
JX_Int
fsls_AMGCoarsenCLJPNew( fsls_CSRMatrix   *A,
                        fsls_ICSRMatrix  *S,
                        JX_Int             **CF_marker_ptr,
                        JX_Int              *coarse_size_ptr )
{
   JX_Int     num_variables = fsls_CSRMatrixNumRows(A);
               
   JX_Int    *S_i    = NULL;
   JX_Int    *S_j    = NULL;
   JX_Int    *S_data = NULL;
                 
   JX_Int    *CF_marker = NULL;
   JX_Int     coarse_size;

   JX_Real *measure_array = NULL;
   JX_Int    *graph_array   = NULL;
   JX_Int     graph_size;

   JX_Int     i, j, k, jS, kS, ig;
   JX_Int     ierr = 0;

   S_i    = fsls_ICSRMatrixI(S);
   S_j    = fsls_ICSRMatrixJ(S);
   S_data = fsls_ICSRMatrixData(S);

   measure_array = fsls_CTAlloc(JX_Real, num_variables);
   graph_array   = fsls_CTAlloc(JX_Int, num_variables);
   CF_marker     = fsls_CTAlloc(JX_Int, num_variables);

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 1: Compute the measures
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
  /*-------------------------------------------------------------
   * 1.1 Compute the true measures of each variable.
   * Remark: The measures are currently given by the column  
   *         sums of S. Hence, measure_array[i] is the number 
   *         of influences of variable i.
   *------------------------------------------------------------*/

   for (i = 0; i < num_variables; i ++)
   {
      for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
      {
         j = S_j[jS];
         measure_array[j] ++;
      }
   }

  /*-------------------------------------------------------------
   * 1.2  Modify the measure_array so that each component
   *      are augmented by a random number between 0 and 1.
   *-----------------------------------------------------------*/
   
   fsls_InitAMGIndepSet(S, measure_array, 1.0);
   

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 2:  Initialize the graph array and the C/F marker array
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
   graph_size = 0; // record the nodes left in the current working graph
   for (i = 0; i < num_variables; i ++)
   {
      if (S_i[i+1] == S_i[i])
      {
         CF_marker[i] = SFPOINT; // SFPOINT = -3 special F-points 
         measure_array[i] = 0;  
      }
      else
      {
         graph_array[graph_size++] = i;
      }
   }

  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 3:  Loop until all points are either fine or coarse.
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   coarse_size = 0; // record the currently number of C-points

   while (1)
   {
   
     /*------------------------------------------------
      * Set F-pts and update subgraph
      *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++)
      {
         i = graph_array[ig];  // ig: local; i: global

         if ( (CF_marker[i] != CPOINT) && (measure_array[i] < 1) )
         {
            /* set to be an F-pt */
            CF_marker[i] = FPOINT;

            /* make sure all dependencies have been accounted for */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               if (S_data[jS] < 0)  /* i.e. S_data[jS] = -1 */
               {
                  CF_marker[i] = UNDECIDED;
               }
            }
         }

         if (CF_marker[i])
         {
            measure_array[i] = 0;

            /* take point out of the subgraph */
            graph_size --;
            graph_array[ig] = graph_array[graph_size];
            graph_array[graph_size] = i;
            ig --; /* to make sure the variable swapped just now will be accessed next */
         }
      }
      

     /*------------------------------------------------
      * Test for convergence
      *------------------------------------------------*/

      if (graph_size == 0) break;

     /*------------------------------------------------------
      * Pick an independent set of points with
      * local maximal measure.
      *----------------------------------------------------*/

      fsls_AMGIndepSet( S, measure_array, 1.0, graph_array,
                        graph_size, CF_marker );


     /*-------------------------------------------------
      * Set C-pts and apply heuristics.
      *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig ++)
      {
         i = graph_array[ig];

        /*---------------------------------------------
         * Heuristic: C-pts don't interpolate from
         * neighbors that influence them.
         *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = CPOINT;
            coarse_size ++;

            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               if (S_data[jS] < 0)
               {
                  j = S_j[jS];
               
                  /* "remove" edge from S */
                  S_data[jS] = -S_data[jS];
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j] --;
                  }
               }
            }
         }


        /*---------------------------------------------
         * Heuristic: points that depend on a common
         * C-pt are less dependent on each other.
         *
         * NOTE: CF_marker is used to help check for
         * common C-pt's in the heuristic.
         *---------------------------------------------*/

         else
         {
            /* marked dependences */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               j = S_j[jS];

               if (CF_marker[j] > 0 || CF_marker[j] == SFPOINT)
               {
                  if (S_data[jS] < 0)
                  {
                     /* "remove" edge from S */
                     S_data[jS] = -S_data[jS];
                  }

                  /* IMPORTANT: consider all dependencies */
                  if (S_data[jS])
                  {
                     /* temporarily modify CF_marker */
                     CF_marker[j] = COMMON_CPOINT;
                  }
               }
            }

            /* unmarked dependences */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               if (S_data[jS] < 0)
               {
                  j = S_j[jS]; /* j must be an undecided point */

                  /* check for common C-pt */
                  for (kS = S_i[j]; kS < S_i[j+1]; kS ++)
                  {
                     k = S_j[kS];

                     /* IMPORTANT: consider all dependencies */
                     if (S_data[kS])
                     {
                        if (CF_marker[k] == COMMON_CPOINT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_data[jS] = -S_data[jS];
                           measure_array[j] --;
                           break;
                        }
                     }
                  }
               }
            }

            /* reset CF_marker */
            for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
            {
               j = S_j[jS];

               if (CF_marker[j] == COMMON_CPOINT)
               {
                  CF_marker[j] = CPOINT;
               }
            }
         }
      }
   }


  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   fsls_TFree(measure_array);
   fsls_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*!
 * \fn JX_Int fsls_InitAMGIndepSet
 * \brief Modify the measure_array so that each component
 *        are augmented by a random number between 0 and 1.
 * \param *S pointer to a ICSR matrix(actually, only its size are needed)
 * \param *measure_array the array to be modified
 * \param ccnst a constant
 * \author peghoty
 * \date 2010/01/20
 */
JX_Int
fsls_InitAMGIndepSet( fsls_ICSRMatrix *S, JX_Real *measure_array, JX_Real cconst )
{
   JX_Int  S_num_nodes = fsls_ICSRMatrixNumRows(S);
   JX_Int  i, ierr = 0;

   fsls_SeedRand(2747);
   
   for (i = 0; i < S_num_nodes; i ++)
   {
      measure_array[i] += fsls_Rand()*cconst;
   }

   return (ierr);
}



/*!
 * \fn JX_Int fsls_AMGIndepSet
 * \brief Select an independent set from a graph.
 *
 * \param *S parent graph matrix in ICSR format
 * \param *measure_array measures assigned to each node of the parent graph
 * \param cconst parameter to decide an initial independent set
 * \param *graph_array node numbers in the subgraph to be partitioned 
 * \param graph_array_size number of nodes in the subgraph to be partitioned
 * \param *IS_marker marker array for independent set
 *
 * \note
 * 
 * (1) This graph is actually a subgraph of some parent graph.
 *     The parent graph is described as a matrix in compressed
 *     sparse row format, where edges in the graph are represented
 *     by nonzero matrix coefficients (zero coefficients are ignored).
 *     A positive measure is given for each node in the subgraph,
 *     and this is used to pick the independent set. A measure 
 *     of zero must be given for all other nodes in the parent graph.
 *     The subgraph is a collection of nodes in the parent graph.

 *     Positive entries in the `IS\_marker' array indicate nodes 
 *     in the independent set. All other entries are zero.

 *     The algorithm proceeds by first setting all nodes in 'graph\_array'
 *     to be in the independent set.  Nodes are then removed from the
 *     independent set by simply comparing the measures of adjacent nodes.
 *
 * (2) For a fixed variable 'i', we only take j\in S_i into consideration.
 *     Maybe you'd worry about that we don't compare (i,k) for 'k'\in S_i^T,
 *     actually, this pair will be processed when it comes to k.  
 *
 * \author peghoty
 * \date 2010/08/29
 */
JX_Int
fsls_AMGIndepSet( fsls_ICSRMatrix *S,
                  JX_Real          *measure_array,
                  JX_Real           cconst,
                  JX_Int             *graph_array,
                  JX_Int              graph_array_size,
                  JX_Int             *IS_marker )
{
   JX_Int    *S_i    = fsls_ICSRMatrixI(S);
   JX_Int    *S_j    = fsls_ICSRMatrixJ(S);
   JX_Int    *S_data = fsls_ICSRMatrixData(S);
         
   JX_Int     i, j, ig, jS;
   JX_Int     ierr = 0;

  /*-------------------------------------------------------
   * Initialize IS_marker by putting all nodes 
   * (which have at least one strong-influence point)
   * in the independent set.
   *------------------------------------------------------*/

   for (ig = 0; ig < graph_array_size; ig ++)
   {
      i = graph_array[ig];
      if (measure_array[i] > cconst)  /* cconst is usually set 1.0  */
      {
         IS_marker[i] = 1;
      }
   }

  /*-------------------------------------------------------
   * Remove nodes from the initial independent set
   *-------------------------------------------------------*/

   for (ig = 0; ig < graph_array_size; ig ++)
   {
      i = graph_array[ig];

      if (measure_array[i] > cconst)
      {
         for (jS = S_i[i]; jS < S_i[i+1]; jS ++)
         {
            j = S_j[jS];
            
            /* only consider valid graph edges */
            /* maybe '&& (S_data[jS])' can be moved away! peghoty 2010/04/07 */
            if ( (measure_array[j] > cconst) && (S_data[jS]) ) 
            {
               if (measure_array[i] > measure_array[j])
               {
                  IS_marker[j] = 0;
               }
               else if (measure_array[j] > measure_array[i])
               {
                  IS_marker[i] = 0;
               }
            }
         }
      }
   }
            
   return (ierr);
}

/*!
 * \fn JX_Int fsls_AMGClassicalInterp
 * \brief Construct Modified Classical interpolation.
 * \param *A pointer to the matrix.
 * \param *CF_marker pointer to the C/F marker.
 * \param *S pointer to the strength matrix of A.
 * \param **P_ptr pointer to the pointer to the interpolation matrix.
 * \note Refer to 
 *  V.E.Henson, U.M.Yang, BoomerAMG: a parallel algebraic multigrid solver and
 *  preconditioner, Applied Numerical Mathematics, 41:155-177, 2002.
 * \remark Add the treatment to special F-points. 2010/11/12
 * \note In order to save memory, the auxiliary array 'fine_to_coarse' is replaced
 *       by 'CF_marker' by modifying its components.
 *       For present, 'CF_marker' is pulled back after 'P' has been generated.
 *       Actually, this is not necessary if we do some modification in 
 *       the smoothing phase when C/F order is selected.  2011/04/14
 * \author peghoty
 * \date 2010/08/29, modified on 2011/04/14
 */
JX_Int
fsls_AMGClassicalInterp( fsls_CSRMatrix    *A,
                         JX_Int               *CF_marker,
                         fsls_ICSRMatrix   *S,
                         fsls_CSRMatrix   **P_ptr )
{ 
   JX_Real          *A_data;
   JX_Int             *A_i;
   JX_Int             *A_j;

   JX_Int             *S_i;
   JX_Int             *S_j;

   fsls_CSRMatrix  *P; 

   JX_Real          *P_data;
   JX_Int             *P_i;
   JX_Int             *P_j;

   JX_Int              P_size; 
   JX_Int             *P_marker;

   JX_Int              jj_counter;
   JX_Int              jj_begin_row;
   JX_Int              jj_end_row;
   JX_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   JX_Int              n_fine;
   JX_Int              n_coarse;
   JX_Int              strong_f_marker;
   JX_Int              coarse_counter;
   JX_Int              i,i1,i2;
   JX_Int              jj,jj1;
   JX_Int              sgn;
   
   JX_Real           diagonal;
   JX_Real           sum;
   JX_Real           distribute;          
   
   JX_Real           zero = 0.0;
   JX_Real           one  = 1.0;
   
  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for A and S. Also get size of fine grid.
   *----------------------------------------------------------------------*/

   A_data = fsls_CSRMatrixData(A);
   A_i    = fsls_CSRMatrixI(A);
   A_j    = fsls_CSRMatrixJ(A);

   S_i    = fsls_ICSRMatrixI(S);
   S_j    = fsls_ICSRMatrixJ(S);

   n_fine = fsls_CSRMatrixNumRows(A);


  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   *  First Pass: Determine size of P.
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

  /*-----------------------------------------------------------------------
   *  Intialize counters
   *-----------------------------------------------------------------------*/

   coarse_counter = 0;
   jj_counter = start_indexing; // counting number of nonzeros in P
      
  /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i ++)
   {
      
     /*--------------------------------------------------------------------
      *  If i is a C-point, interpolation is the identity. Also set up
      *  mapping vector.
      *--------------------------------------------------------------------*/

      if (CF_marker[i] > 0)
      {
         jj_counter ++;
         CF_marker[i] = coarse_counter;
         coarse_counter ++;
      }
      
     /*--------------------------------------------------------------------
      *  If i is a F-point, interpolation is from the C-points that
      *  strongly influence i.
      *--------------------------------------------------------------------*/

      else
      {
         for (jj = S_i[i]; jj < S_i[i+1]; jj ++)
         {
            i1 = S_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_counter ++;
            }
         }
      }
   }
   
  /*-----------------------------------------------------------------------
   *  Allocate arrays.
   *-----------------------------------------------------------------------*/

   n_coarse = coarse_counter;

   P_size = jj_counter; // number of nonzeros in P

   P_i    = fsls_CTAlloc(JX_Int, n_fine+1);
   P_j    = fsls_CTAlloc(JX_Int, P_size);
   P_data = fsls_CTAlloc(JX_Real, P_size);

   P_marker = fsls_CTAlloc(JX_Int, n_fine);


  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   *  Second Pass: Define interpolation and fill in P_data, P_i, and P_j.
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

  /*-----------------------------------------------------------------------
   *  Initialize some stuff.
   *----------------------------------------------------------------------*/

   for (i = 0; i < n_fine; i ++)
   {      
      P_marker[i] = -1;
   }
   
   strong_f_marker = -2;

   jj_counter = start_indexing;
   
  /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i ++)
   {     
     /*--------------------------------------------------------------------
      *  If i is a C-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
      
      if (CF_marker[i] >= 0)
      {
         P_i[i] = jj_counter;
         P_j[jj_counter]    = CF_marker[i];
         P_data[jj_counter] = one;
         jj_counter ++;
      }
      
     /*--------------------------------------------------------------------
      *  If i is a F-point, build interpolation.
      *--------------------------------------------------------------------*/

      else if (CF_marker[i] == FPOINT)
      {
         P_i[i] = jj_counter;
         jj_begin_row = jj_counter;

         for (jj = S_i[i]; jj < S_i[i+1]; jj ++)
         {
            i1 = S_j[jj];   

           /*---------------------------------------------------------------
            * If neighbor i1 is a C-point, set column number in P_j and
            * initialize interpolation weight to zero.
            *--------------------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1]       = jj_counter;
               P_j[jj_counter]    = CF_marker[i1]; // set column number in P_j
               P_data[jj_counter] = zero;          // initialize interpolation weight to zero.
               jj_counter ++;
            }

           /*---------------------------------------------------------------
            * If neighbor i1 is a F-point, mark it as a strong F-point
            * whose connection needs to be distributed.
            *--------------------------------------------------------------*/

            else
            {
               P_marker[i1] = strong_f_marker;
            }            
         }

         jj_end_row = jj_counter; // starting position of the next row peghoty 2010/06/08
         
         diagonal = A_data[A_i[i]];
         
         for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
         {
            i1 = A_j[jj];

           /*---------------------------------------------------------------
            * Case 1: neighbor i1 is a C-point and strongly influences i,
            * accumulate a_{i,i1} into the interpolation weight.
            *--------------------------------------------------------------*/

            if (P_marker[i1] >= jj_begin_row) // i (F) strongly depends on i1 (C)
            {
               P_data[P_marker[i1]] += A_data[jj];
            }
 
           /*----------------------------------------------------------------
            * Case 2: neighbor i1 is a F-point and strongly influences i,
            * distribute a_{i,i1} to C-points that strongly infuence i.
            * Note: currently no distribution to the diagonal in this case.
            *--------------------------------------------------------------*/
            
            else if (P_marker[i1] == strong_f_marker)
            {
               sum = zero;
               
              /*------------------------------------------------------------
               * Loop over row of A for point i1 and calculate the sum
               * of the connections to C-points that strongly influence i.
               *-----------------------------------------------------------*/

               sgn = 1;
               if (A_data[A_i[i1]] < 0) sgn = -1;
               
               for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1 ++)
               {
                  i2 = A_j[jj1];
                  if (P_marker[i2] >= jj_begin_row && (sgn*A_data[jj1]) < 0)
                  {
                     sum += A_data[jj1];
                  }
               }
               
               if (sum != 0.0)
               {
                   distribute = A_data[jj] / sum;
               
                  /*------------------------------------------------------------
                   * Loop over row of A for point i1 and do the distribution.
                   *-----------------------------------------------------------*/

                   for (jj1 = A_i[i1]; jj1 < A_i[i1+1]; jj1 ++)
                   {
                      i2 = A_j[jj1];
                      if (P_marker[i2] >= jj_begin_row && (sgn*A_data[jj1]) < 0)
                      {
                         P_data[P_marker[i2]] += distribute * A_data[jj1];
                      }
                   }
               }
               else
               {
                  diagonal += A_data[jj];
               }
            }
   
           /*---------------------------------------------------------------
            * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
            * into the diagonal.
            *--------------------------------------------------------------*/

            else
            {
                diagonal += A_data[jj];
            }            
         }

        /*-----------------------------------------------------------------
         * Set interpolation weight by dividing by the diagonal.
         *---------------------------------------------------------------*/

         for (jj = jj_begin_row; jj < jj_end_row; jj ++)
         {
            P_data[jj] /= -diagonal;
         }
      }
      
      else if (CF_marker[i] == SFPOINT)  // special F point, do nothing
      {
         P_i[i] = jj_counter;
      }
   
     /*--------------------------------------------------------------------
      * Interpolation formula for i is done, update marker for strong
      * f connections for next i.
      *-------------------------------------------------------------------*/
   
      strong_f_marker --;
   }
  
   P_i[n_fine] = jj_counter;

   P = fsls_CSRMatrixCreate(n_fine, n_coarse, P_size);
   fsls_CSRMatrixData(P) = P_data; 
   fsls_CSRMatrixI(P) = P_i; 
   fsls_CSRMatrixJ(P) = P_j; 

   *P_ptr = P; 
   
  /*-------------------------------------------------
   *  Give back the real CF_marker.
   *------------------------------------------------*/
   /* We don't need this any more, because the C- and F- point
      can be still distinguished by CF_marker if we treat 'CF_marker'
      very careful when carrying out C/F order relaxation.   peghoty  2011/04/14  */
   /*for (i = 0; i < n_fine; i ++)
   {
      if (CF_marker[i] >= 0) CF_marker[i] = 1;
   }*/

  /*-------------------------------------------------
   *  Free marker array.
   *------------------------------------------------*/

   fsls_TFree(P_marker); 
 
   return(0);  
}

/*!
 * \fn JX_Int fsls_AMGDirectInterp
 * \brief Construct Stuben's Direct interpolation.
 * \param *A pointer to the matrix.
 * \param *CF_marker pointer to the C/F marker.
 * \param *S pointer to the strength matrix of A.
 * \param **P_ptr pointer to the pointer to the interpolation matrix.
 * \note Refer to Page 479 in
 *       U. Trottenberg, C. W. Oosterlee, and A. Sch¨uller. Multigrid. 
 *	 Academic Press Inc., San Diego, CA, 2001.
 * \note the direct interpolation can be described as follows: for any i\in F
 *         e_i = \sum_{k\in C_i}w_{i,k}e_k
 *  where
 *                   / -alpha_i*a_{i,k}/a_{i,i}, if k\in C_i^{-}
 *         w_{i,k} = 
 *                   \ -beta_i*a_{i,k}/a_{i,i},  if k\in C_i^{+}
 *  and
 *     alpha_i = \sum_{k\in N_i^{-}}a_{i,k} / \sum_{k\in C_i^{-}}a_{i,k}
 *     beta_i  = \sum_{k\in N_i^{+}}a_{i,k} / \sum_{k\in C_i^{+}}a_{i,k}
 * \remark
 *  (1) If C_i^{+} = \emptyset, set beta_i = 0,  and add all positive entries(if there are any) to the diagonal;
 *  (2) If C_i^{-} = \emptyset, set alpha_i = 0, and add all negative entries(if there are any) to the diagonal;
 *  (3) If C_i = \emptyset, then i doesn't require interpolation.
 * \remark Add the treatment to special F-points. 2010/11/12 
 * \note In order to save memory, the auxiliary array 'fine_to_coarse' is replaced
 *       by 'CF_marker' by modifying its components.
 *       For present, 'CF_marker' is pulled back after 'P' has been generated.
 *       Actually, this is not necessary if we do some modification in 
 *       the smoothing phase when C/F order is selected.  2011/04/14 
 * \author peghoty
 * \date 2010/09/11, modified on 2011/04/14
 */
JX_Int
fsls_AMGDirectInterp( fsls_CSRMatrix    *A,
                      JX_Int               *CF_marker,
                      fsls_ICSRMatrix   *S,
                      fsls_CSRMatrix   **P_ptr )
{ 
   JX_Real          *A_data;
   JX_Int             *A_i;
   JX_Int             *A_j;

   JX_Int             *S_i;
   JX_Int             *S_j;

   fsls_CSRMatrix  *P; 

   JX_Real          *P_data;
   JX_Int             *P_i;
   JX_Int             *P_j;

   JX_Int              P_size; 
   JX_Int             *P_marker;

   JX_Int              jj_counter;
   JX_Int              jj_begin_row;
   JX_Int              jj_end_row;
   JX_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   JX_Int              n_fine;
   JX_Int              n_coarse;
   JX_Int              coarse_counter;
   JX_Int              i,i1,jj;
   
   JX_Real           diagonal;       
   
   JX_Real           zero = 0.0;
   JX_Real           one  = 1.0;
   
   JX_Real           alpha_up_sum    = 0.0;
   JX_Real           alpha_down_sum  = 0.0;
   JX_Real           beta_up_sum     = 0.0;
   JX_Real           beta_down_sum   = 0.0;
   JX_Real           alpha           = 0.0;
   JX_Real           beta            = 0.0;
 
  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for A and S. Also get size of fine grid.
   *----------------------------------------------------------------------*/

   A_data = fsls_CSRMatrixData(A);
   A_i    = fsls_CSRMatrixI(A);
   A_j    = fsls_CSRMatrixJ(A);

   S_i    = fsls_ICSRMatrixI(S);
   S_j    = fsls_ICSRMatrixJ(S);

   n_fine = fsls_CSRMatrixNumRows(A);


  /*-----------------------------------------------*
   *  Step 1: First Pass -- Determine size of P    *
   *-----------------------------------------------*/

  /*--------------------------------------------------------
   *  (1) Intialize counters.
   *-------------------------------------------------------*/

   coarse_counter = 0;
   jj_counter     = start_indexing; // counting number of nonzeros in P

 
  /*-----------------------------------------------------------------------
   *  (2) Loop over fine grid.
   *----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i ++)
   {
      
     /*----------------------------------------------
      *  If i is a C-point, interpolation is the 
      *  identity, also set up mapping vector.
      *---------------------------------------------*/

      if (CF_marker[i] > 0)
      {
         jj_counter ++;
         CF_marker[i] = coarse_counter;
         coarse_counter ++;
      }
      
     /*----------------------------------------------
      *  If i is a F-point, interpolation is from 
      *  the C-points that strongly influence i.
      *---------------------------------------------*/
      
      else
      {
         for (jj = S_i[i]; jj < S_i[i+1]; jj ++)
         {
            i1 = S_j[jj];           
            if (CF_marker[i1] >= 0)
            {
               jj_counter ++;
            }
         }
      }
   }
   
  /*---------------------------------------------
   *  Allocate arrays.
   *--------------------------------------------*/

   n_coarse = coarse_counter;
   P_size   = jj_counter; // nonzeros of P

   P_i      = fsls_CTAlloc(JX_Int, n_fine+1);
   P_j      = fsls_CTAlloc(JX_Int, P_size);
   P_data   = fsls_CTAlloc(JX_Real, P_size);
   P_marker = fsls_CTAlloc(JX_Int, n_fine);


  /*---------------------------------------------------------------------------------*
   *  Step 2: Second Pass -- Define interpolation and fill in P_data, P_i, and P_j   *
   *---------------------------------------------------------------------------------*/

  /*---------------------------------
   *  (1) Intialize some stuff.
   *--------------------------------*/
 
   for (i = 0; i < n_fine; i ++)
   {      
      P_marker[i] = -1;
   }
   
   jj_counter = start_indexing;
   
  /*-----------------------------------------------------------------------
   *  (2) Loop over fine grid points.
   *----------------------------------------------------------------------*/
    
   for (i = 0; i < n_fine; i ++)
   {     
     /*------------------------------------------------------
      *  If i is a C-point, interpolation is the identity.
      *-----------------------------------------------------*/
      
      if (CF_marker[i] >= 0)
      {
         P_i[i] = jj_counter;
         P_j[jj_counter]    = CF_marker[i];
         P_data[jj_counter] = one;
         jj_counter ++;
      }
      
     /*-----------------------------------------------------------
      *  If i is a F-point, build interpolation.
      *----------------------------------------------------------*/

      else if (CF_marker[i] == FPOINT)
      {
         P_i[i] = jj_counter;
         jj_begin_row = jj_counter;

         for (jj = S_i[i]; jj < S_i[i+1]; jj ++)
         {
            i1 = S_j[jj];   

           /*------------------------------------------------------
            * If neighbor i1 is a C-point, set column number in 
            * P_j and initialize interpolation weight to zero.
            *-----------------------------------------------------*/

            if (CF_marker[i1] >= 0)
            {
               P_marker[i1]       = jj_counter;
               P_j[jj_counter]    = CF_marker[i1]; // set column number in P_j
               P_data[jj_counter] = zero;               // initialize interpolation weight to zero.
               jj_counter ++;
            }         
         }

         jj_end_row = jj_counter;   // starting position of the next row
         
        if (jj_end_row > jj_begin_row)
        {
            diagonal = A_data[A_i[i]]; // get the diagonal entry in the current row
         
            alpha_up_sum   = 0.0;
            alpha_down_sum = 0.0;
            beta_up_sum    = 0.0;
            beta_down_sum  = 0.0;
            alpha          = 0.0;
            beta           = 0.0;
         
            for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
            {
               i1 = A_j[jj];
            
               if (A_data[jj] < 0)
               {
                  alpha_up_sum += A_data[jj];
                  if (P_marker[i1] >= jj_begin_row) // i (F-point) strongly depends on i1 (C-point)
                  {
                     alpha_down_sum += A_data[jj];
                  }
               }
               else
               {
                  beta_up_sum += A_data[jj];
                  if (P_marker[i1] >= jj_begin_row) // i (F-point) strongly depends on i1 (C-point)
                  {
                     beta_down_sum += A_data[jj];
                  }
               }
            }
         
            if (alpha_down_sum == 0.0)
            {
               alpha = 0.0;
               diagonal += alpha_up_sum;
            }
            else
            {
               alpha = alpha_up_sum / alpha_down_sum;
            }

            if (beta_down_sum == 0.0)
            {
               beta = 0.0;
               diagonal += beta_up_sum;
            }
            else
            {
               beta = beta_up_sum / beta_down_sum;
            }
         
            for (jj = A_i[i]+1; jj < A_i[i+1]; jj ++)
            {
               i1 = A_j[jj];
               if (P_marker[i1] >= jj_begin_row) // i (F-point) strongly depends on i1 (C-point)
               {
                  if (A_data[jj] < 0)
                  {
                     P_data[P_marker[i1]] = - alpha*A_data[jj] / diagonal;
                  }
                  else
                  {
                     P_data[P_marker[i1]] = - beta*A_data[jj] / diagonal;
                  } 
               }
            }
         } // end if (jj_end_row > jj_begin_row) 
      } 
      
      else if (CF_marker[i] == SFPOINT)
      {
         P_i[i] = jj_counter;
      }
      
   } // end for i
  
   P_i[n_fine] = jj_counter;

   P = fsls_CSRMatrixCreate(n_fine, n_coarse, P_size);
   fsls_CSRMatrixData(P) = P_data; 
   fsls_CSRMatrixI(P) = P_i; 
   fsls_CSRMatrixJ(P) = P_j; 

   *P_ptr = P; 

  /*-------------------------------------------------
   *  Free marker array.
   *------------------------------------------------*/

   fsls_TFree(P_marker);   
 
   return(0);  
}

/*!
 * \fn JX_Int fsls_AMGPTruncation
 * \brief Truncate some entries in A, the interpolation operater.
 * \note the truncation rules are as follows:
 *  1. when trunc_factor > 0.0, we do the following for the i-th row:
 *     (1) a_{i,j} := 0.0, if |a_{i,j}| < trunc_factor*diag;
 *     (2) a_{i,i} += \sum a_{i,j}, for all |a_{i,j}| < trunc_factor*diag,
 *  where diag = max{a_{i,i}}.
 *  2. when max_elmts > 0, we do the following for the i-th row if nonzeros[i] > max_elmts :
 *     (1) reorder all the entries decendingly according to their absolute values;
 *     (2) only keep the first max_elmts entries;
 *     (3) re-scale the remaining entries to so that the total sum remains unchanged.
 * \author peghoty
 * \date 2010/08/29
 */
JX_Int
fsls_AMGPTruncation( fsls_CSRMatrix *A, JX_Real trunc_factor, JX_Int max_elmts )
{

   JX_Int    *A_i    = fsls_CSRMatrixI(A);
   JX_Int    *A_j    = fsls_CSRMatrixJ(A);
   JX_Real *A_data = fsls_CSRMatrixData(A);
   JX_Int     num_variables = fsls_CSRMatrixNumRows(A);
      
   JX_Real max_coef, row_sum, scale;
   JX_Int    i, j, start;
   JX_Int    now_checking, num_lost, next_open;
   JX_Int    ierr = 0;
   
   if (trunc_factor > 0.0)
   {
      num_lost = 0;
      next_open = 0;
      now_checking = 0;
      
      for (i = 0; i < num_variables; i ++)
      { 
         /* max_coef = trunc_factor*max(|a_{ij}|) */
         max_coef = 0.0;
         for (j = A_i[i]; j < A_i[i+1]; j ++)
         {
	    if (max_coef < fabs(A_data[j])) 
	    {
	       max_coef = fabs(A_data[j]);
	    }
	 }
         max_coef *= trunc_factor;
         
         row_sum = 0.0;
         scale = 0.0;
         start = A_i[i];
         A_i[i] -= num_lost;
         for (j = start; j < A_i[i+1]; j ++)
         {
 	    row_sum += A_data[now_checking];
	    if (fabs(A_data[now_checking]) < max_coef)
	    {
	       num_lost ++;
	       now_checking ++;
	    }
	    else
	    {
	       /* scale += A_data[now_checking]; */
	       A_data[next_open] = A_data[now_checking];
	       A_j[next_open] = A_j[now_checking];
	       now_checking ++;
	       next_open ++;
	    }
         }
 	 if (max_elmts > 0 && (next_open-A_i[i]) > max_elmts)
	 {
	    fsls_idFabsQuickSort21(A_j, A_data, A_i[i], next_open-1);
	    num_lost += next_open-A_i[i]-max_elmts;
	    next_open = A_i[i] + max_elmts;
	 }
	 
	 for (j = A_i[i]; j < next_open; j ++)
	 {
	    scale += A_data[j];
         }
         
         if (scale != 0 && scale != row_sum)
         {
	    scale = row_sum / scale;
            for (j = A_i[i]; j < A_i[i+1]-num_lost; j ++)
            {
	       A_data[j] *= scale;
	    }
         }
      }
      A_i[num_variables] -= num_lost;
      fsls_CSRMatrixNumNonzeros(A) = A_i[num_variables];
   }
   else if (max_elmts > 0)
   {
      next_open = 0;
      start = 0;
      for (i = 0; i < num_variables; i ++)
      {
         row_sum = 0;
         for (j = A_i[i]; j < A_i[i+1]; j ++)
         {
	    row_sum += A_data[j];
	 }
	 
	 start = next_open;
	 if (A_i[i] > next_open)
	 {
	    for (j = A_i[i]; j < A_i[i+1]; j ++)
	    {
	       A_data[next_open] = A_data[j];
	       A_j[next_open++] = A_j[j];
	    }
	 }
 	 if ((A_i[i+1]-A_i[i]) > max_elmts)
	 {
	    /* modified by peghoty 2010/09/09 */
	    if (i == 0)
	    {
	       fsls_idFabsQuickSort21(A_j, A_data, 0, max_elmts-1);
	    }
	    else
	    {
	       fsls_idFabsQuickSort21(A_j, A_data, start, next_open-1);
	    }
	    next_open = start+max_elmts;
	 }
	 else
	 {
	    next_open = start + (A_i[i+1]-A_i[i]);
	 }

         scale = 0;
	 for (j = start; j < next_open; j ++)
 	 {
	    scale += A_data[j];
 	 }
         
         if (scale != 0 && scale != row_sum)
         {
	    scale = row_sum / scale;
            for (j = start; j < next_open; j ++)
	       A_data[j] *= scale;
         }
	 A_i[i] = start;
      }
      A_i[num_variables] = next_open;
      fsls_CSRMatrixNumNonzeros(A) = next_open;
   }

   return ierr;
}

/*!
 * \fn void fsls_dispose_elt
 * \brief dispose of memory space used by the element
 *        pointed to by element_ptr.  Use the 'free()'
 *        system call to return it to the free memory pool.
 * \author peghoty
 * \date 2010/08/29 
 */ 
void 
fsls_dispose_elt( fsls_LinkList element_ptr )
{
   free( element_ptr );
}

/*!
 * \fn void fsls_remove_point
 * \brief removes a point from the lists.
 * \author peghoty
 * \date 2010/08/29 
 */  
void 
fsls_remove_point( fsls_LinkList   *LoL_head_ptr, 
                   fsls_LinkList   *LoL_tail_ptr, 
                   JX_Int              measure,
                   JX_Int              index, 
                   JX_Int             *lists, 
                   JX_Int             *where )

{
   fsls_LinkList   LoL_head = *LoL_head_ptr;
   fsls_LinkList   LoL_tail = *LoL_tail_ptr;
   fsls_LinkList   list_ptr;

   list_ptr =  LoL_head;

   do
   {
      if (measure == list_ptr->data)
      {
         /* point to be removed is only point on list, which must be destroyed */
         if (list_ptr->head == index && list_ptr->tail == index)
         {
            /* removing only list, so num_left better be 0! */
            if (list_ptr == LoL_head && list_ptr == LoL_tail)
            {
               LoL_head = NULL;
               LoL_tail = NULL;
               fsls_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_head == list_ptr) /*removing 1st (max_measure) list */
            {
               list_ptr -> next_elt -> prev_elt = NULL;
               LoL_head = list_ptr->next_elt;
               fsls_dispose_elt(list_ptr);
               
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else if (LoL_tail == list_ptr) /* removing last list */
            {
               list_ptr -> prev_elt -> next_elt = NULL;
               LoL_tail = list_ptr->prev_elt;
               fsls_dispose_elt(list_ptr);

               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
            else
            {
               list_ptr -> next_elt -> prev_elt = list_ptr -> prev_elt;
               list_ptr -> prev_elt -> next_elt = list_ptr -> next_elt;
               fsls_dispose_elt(list_ptr);
               
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail;
               return;
            }
         }
         else if (list_ptr->head == index)  /* index is head of list */
         {
            list_ptr->head = lists[index];
            where[lists[index]] = FSLS_LIST_HEAD;
            return;
         }
         else if (list_ptr->tail == index)  /* index is tail of list */
         {
            list_ptr->tail = where[index];
            lists[where[index]] = FSLS_LIST_TAIL;
            return;
         }
         else                              /* index is in middle of list */
         {
            lists[where[index]] = lists[index];
            where[lists[index]] = where[index];
            return;
         }
      }
      list_ptr = list_ptr -> next_elt;
   } while (list_ptr != NULL);
   
   jx_printf("No such list!\n");
   return;
}

/*!
 * \fn fsls_LinkList fsls_create_elt
 * \brief Create an element using Item for its data field.
 * \author peghoty
 * \date 2010/08/29 
 */ 
fsls_LinkList 
fsls_create_elt( JX_Int Item )
{
    fsls_LinkList   new_elt_ptr;
 
    /* Allocate memory space for the new node. 
       return with error if no space available */
    if ( (new_elt_ptr = (fsls_LinkList) malloc (sizeof(fsls_ListElement))) == NULL)
    {
       jx_printf("\n fsls_create_elt: malloc failed \n\n");
    }
    else 
    {
       new_elt_ptr -> data = Item;
       new_elt_ptr -> next_elt = NULL;
       new_elt_ptr -> prev_elt = NULL;
       new_elt_ptr -> head = FSLS_LIST_TAIL;
       new_elt_ptr -> tail = FSLS_LIST_HEAD;
    }

    return (new_elt_ptr);
}

/*!
 * \fn void fsls_enter_on_lists
 * \brief places point in new list.
 * \author peghoty
 * \date 2010/08/29 
 */ 
void 
fsls_enter_on_lists( fsls_LinkList   *LoL_head_ptr, 
                     fsls_LinkList   *LoL_tail_ptr, 
                     JX_Int              measure,
                     JX_Int              index, 
                     JX_Int             *lists, 
                     JX_Int             *where )
{
    fsls_LinkList   LoL_head = *LoL_head_ptr;
    fsls_LinkList   LoL_tail = *LoL_tail_ptr;

    fsls_LinkList   list_ptr;
    fsls_LinkList   new_ptr;

    JX_Int old_tail;

    list_ptr =  LoL_head;

    if (LoL_head == NULL)   /* no lists exist yet */
    {
        new_ptr = fsls_create_elt(measure);
        new_ptr->head = index;
        new_ptr->tail = index;
        lists[index] = FSLS_LIST_TAIL;
        where[index] = FSLS_LIST_HEAD; 
        LoL_head = new_ptr;
        LoL_tail = new_ptr;

        *LoL_head_ptr = LoL_head;
        *LoL_tail_ptr = LoL_tail;
        return;
    }
    else  
    {
        do
        {
            if (measure > list_ptr->data)
            {
               new_ptr = fsls_create_elt(measure);
               new_ptr->head = index;
               new_ptr->tail = index;
               lists[index] = FSLS_LIST_TAIL;
               where[index] = FSLS_LIST_HEAD;

               if ( list_ptr->prev_elt != NULL)
               { 
                  new_ptr->prev_elt            = list_ptr->prev_elt;
                  list_ptr->prev_elt->next_elt = new_ptr;   
                  list_ptr->prev_elt           = new_ptr;
                  new_ptr->next_elt            = list_ptr;
               }
               else
               {
                  new_ptr->next_elt  = list_ptr;
                  list_ptr->prev_elt = new_ptr;
                  new_ptr->prev_elt  = NULL;
                  LoL_head = new_ptr;
               }
               *LoL_head_ptr = LoL_head;
               *LoL_tail_ptr = LoL_tail; 
               return;
            }
            else if (measure == list_ptr->data)
            {
               old_tail = list_ptr->tail;
               lists[old_tail] = index;
               where[index] = old_tail;
               lists[index] = FSLS_LIST_TAIL;
               list_ptr->tail = index;
               return;
            }
      
            list_ptr = list_ptr->next_elt;
            
        } while (list_ptr != NULL);

        new_ptr = fsls_create_elt(measure);   
        new_ptr->head = index;
        new_ptr->tail = index;
        lists[index] = FSLS_LIST_TAIL;
        where[index] = FSLS_LIST_HEAD;
        LoL_tail->next_elt = new_ptr;
        new_ptr->prev_elt = LoL_tail;
        new_ptr->next_elt = NULL;
        LoL_tail = new_ptr;

        *LoL_head_ptr = LoL_head;
        *LoL_tail_ptr = LoL_tail;
        return;
    }   
}

/*!
 * \fn JX_Int fsls_AMGBuildCoarseOperator
 * \brief Build the coarse operater.
 * \note This is basicly the same as 'fsls_CSRMatrixRAP'.
 * \date 2010/08/29
 */
JX_Int 
fsls_AMGBuildCoarseOperator( fsls_CSRMatrix   *RT,
                             fsls_CSRMatrix   *A,
                             fsls_CSRMatrix   *P,
                             fsls_CSRMatrix  **RAP_ptr )
{
   fsls_CSRMatrix  *RAP;
   
   JX_Real          *A_data;
   JX_Int             *A_i;
   JX_Int             *A_j;

   JX_Real          *P_data;
   JX_Int             *P_i;
   JX_Int             *P_j;

   JX_Real          *RAP_data;
   JX_Int             *RAP_i;
   JX_Int             *RAP_j;

   JX_Int              RAP_size;
   
   fsls_CSRMatrix  *R;
   
   JX_Real          *R_data;
   JX_Int             *R_i;
   JX_Int             *R_j;

   JX_Int             *P_marker;
   JX_Int             *A_marker;

   JX_Int              n_coarse;
   JX_Int              n_fine;
   
   JX_Int              ic, i;
   JX_Int              i1, i2, i3;
   JX_Int              jj1, jj2, jj3;
   
   JX_Int              jj_counter;
   JX_Int              jj_row_begining;
   JX_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */

   JX_Real           r_entry;
   JX_Real           r_a_product;
   JX_Real           r_a_p_product;
   
   JX_Real           zero = 0.0;
   
  /*-----------------------------------------------------------------------
   *  Copy RT into R so that we have row-wise access to restriction.
   *----------------------------------------------------------------------*/

   fsls_CSRMatrixTranspose(RT, &R, 1);   /* could call PETSc MatTranspose */

  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for R, A, P. Also get sizes of fine and
   *  coarse grids.
   *----------------------------------------------------------------------*/

   R_data = fsls_CSRMatrixData(R);
   R_i    = fsls_CSRMatrixI(R);
   R_j    = fsls_CSRMatrixJ(R);

   A_data = fsls_CSRMatrixData(A);
   A_i    = fsls_CSRMatrixI(A);
   A_j    = fsls_CSRMatrixJ(A);

   P_data = fsls_CSRMatrixData(P);
   P_i    = fsls_CSRMatrixI(P);
   P_j    = fsls_CSRMatrixJ(P);

   n_fine   = fsls_CSRMatrixNumRows(A);
   n_coarse = fsls_CSRMatrixNumRows(R);

  /*-----------------------------------------------------------------------
   *  Allocate RAP_i and marker arrays.
   *----------------------------------------------------------------------*/

   RAP_i    = fsls_CTAlloc(JX_Int, n_coarse+1);
   P_marker = fsls_CTAlloc(JX_Int, n_coarse);
   A_marker = fsls_CTAlloc(JX_Int, n_fine);

  /*-----------------------------------------------------------------------
   *  First Pass: Determine size of RAP and set up RAP_i
   *----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < n_coarse; ic ++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < n_fine; i ++)
   {      
      A_marker[i] = -1;
   }   

  /*-----------------------------------------------------------------------
   *  Loop over c-points.
   *----------------------------------------------------------------------*/
    
   for (ic = 0; ic < n_coarse; ic ++)
   {
      
     /*--------------------------------------------------------------------
      *  Set marker for diagonal entry, RAP_{ic,ic}.
      *-------------------------------------------------------------------*/

      P_marker[ic] = jj_counter;
      jj_row_begining = jj_counter;
      jj_counter ++;

     /*--------------------------------------------------------------------
      *  Loop over entries in row ic of R.
      *-------------------------------------------------------------------*/
   
      for (jj1 = R_i[ic]; jj1 < R_i[ic+1]; jj1 ++)
      {
         i1 = R_j[jj1];

        /*-----------------------------------------------------------------
         *  Loop over entries in row i1 of A.
         *----------------------------------------------------------------*/
         
         for (jj2 = A_i[i1]; jj2 < A_i[i1+1]; jj2 ++)
         {
            i2 = A_j[jj2];

           /*----------------------------------------------------------------
            *  Check A_marker to see if point i2 has been previously
            *  visited. New entries in RAP only occur from unmarked points.
            *---------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

              /*-----------------------------------------------------------
               *  Mark i2 as visited.
               *----------------------------------------------------------*/

               A_marker[i2] = ic;
               
              /*-----------------------------------------------------------
               *  Loop over entries in row i2 of P.
               *----------------------------------------------------------*/

               for (jj3 = P_i[i2]; jj3 < P_i[i2+1]; jj3++)
               {
                  i3 = P_j[jj3];
                  
                 /*------------------------------------------------------------
                  *  Check P_marker to see that RAP_{ic,i3} has not already
                  *  been accounted for. If it has not, mark it and increment
                  *  counter.
                  *-----------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter ++;
                  }
               }
            }
         }
      }
            
     /*--------------------------------------------------------------------
      * Set RAP_i for this row.
      *--------------------------------------------------------------------*/

      RAP_i[ic] = jj_row_begining;
      
   }
  
   RAP_i[n_coarse] = jj_counter;
 
  /*-----------------------------------------------------------------------
   *  Allocate RAP_data and RAP_j arrays.
   *----------------------------------------------------------------------*/

   RAP_size = jj_counter;
   RAP_data = fsls_CTAlloc(JX_Real, RAP_size);
   RAP_j    = fsls_CTAlloc(JX_Int, RAP_size);

  /*-----------------------------------------------------------------------
   *  Second Pass: Fill in RAP_data and RAP_j.
   *----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < n_coarse; ic ++)
   {      
      P_marker[ic] = -1;
   }
   for (i = 0; i < n_fine; i ++)
   {      
      A_marker[i] = -1;
   }   
   
  /*-----------------------------------------------------------------------
   *  Loop over c-points.
   *----------------------------------------------------------------------*/
    
   for (ic = 0; ic < n_coarse; ic ++)
   {
      
     /*--------------------------------------------------------------------
      *  Create diagonal entry, RAP_{ic,ic}.
      *-------------------------------------------------------------------*/

      P_marker[ic] = jj_counter;
      jj_row_begining = jj_counter;
      RAP_data[jj_counter] = zero;
      RAP_j[jj_counter] = ic;
      jj_counter ++;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R.
       *-------------------------------------------------------------------*/
   
      for (jj1 = R_i[ic]; jj1 < R_i[ic+1]; jj1 ++)
      {
         i1 = R_j[jj1];
         r_entry = R_data[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_i[i1]; jj2 < A_i[i1+1]; jj2 ++)
         {
            i2 = A_j[jj2];
            r_a_product = r_entry*A_data[jj2];
            
           /*----------------------------------------------------------------
            *  Check A_marker to see if point i2 has been previously
            *  visited. New entries in RAP only occur from unmarked points.
            *---------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

              /*-----------------------------------------------------------
               *  Mark i2 as visited.
               *----------------------------------------------------------*/

               A_marker[i2] = ic;
               
              /*-----------------------------------------------------------
               *  Loop over entries in row i2 of P.
               *----------------------------------------------------------*/

               for (jj3 = P_i[i2]; jj3 < P_i[i2+1]; jj3++)
               {
                  i3 = P_j[jj3];
                  r_a_p_product = r_a_product * P_data[jj3];
                  
                 /*-----------------------------------------------------------
                  *  Check P_marker to see that RAP_{ic,i3} has not already
                  *  been accounted for. If it has not, create a new entry.
                  *  If it has, add new contribution.
                  *----------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_data[jj_counter] = r_a_p_product;
                     RAP_j[jj_counter] = i3;
                     jj_counter ++;
                  }
                  else
                  {
                     RAP_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

           /*--------------------------------------------------------------
            *  If i2 is previously visted ( A_marker[12]=ic ) it yields
            *  no new entries in RAP and can just add new contributions.
            *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_i[i2]; jj3 < P_i[i2+1]; jj3 ++)
               {
                  i3 = P_j[jj3];
                  r_a_p_product = r_a_product*P_data[jj3];
                  RAP_data[P_marker[i3]] += r_a_p_product;
               }
            }
         }
      }
   }

   RAP = fsls_CSRMatrixCreate(n_coarse, n_coarse, RAP_size);
   fsls_CSRMatrixData(RAP) = RAP_data; 
   fsls_CSRMatrixI(RAP) = RAP_i; 
   fsls_CSRMatrixJ(RAP) = RAP_j; 
   
#if 0 // Test whether Reorder has been done. peghoty 2010/03/07    
   fsls_CSRMatrixReorder(RAP);
#endif
   
   *RAP_ptr = RAP;

  /*--------------------------------------------
   *  Free R and marker arrays.
   *-------------------------------------------*/

   fsls_CSRMatrixDestroy(R);
   fsls_TFree(P_marker);   
   fsls_TFree(A_marker);

   return(0);
   
}            
      
/*--------------------------------------------------------------------------
 * OLD NOTES:
 * Sketch of John's code to build RAP
 *
 * Uses two integer arrays icg and ifg as marker arrays
 *
 *  icg needs to be of size n_fine; size of ia.
 *     A negative value of icg(i) indicates i is a f-point, otherwise
 *     icg(i) is the converts from fine to coarse grid orderings. 
 *     Note that I belive the code assumes that if i<j and both are
 *     c-points, then icg(i) < icg(j).
 *  ifg needs to be of size n_coarse; size of irap
 *     I don't think it has meaning as either input or output.
 *
 * In the code, both the interpolation and restriction operator
 * are stored row-wise in the array b. If i is a f-point,
 * ib(i) points the row of the interpolation operator for point
 * i. If i is a c-point, ib(i) points the row of the restriction
 * operator for point i.
 *
 * In the CSR storage for rap, its guaranteed that the rows will
 * be ordered ( i.e. ic<jc -> irap(ic) < irap(jc)) but I don't
 * think there is a guarantee that the entries within a row will
 * be ordered in any way except that the diagonal entry comes first.
 *
 * As structured now, the code requires that the size of rap be
 * predicted up front. To avoid this, one could execute the code
 * twice, the first time would only keep track of icg ,ifg and ka.
 * Then you would know how much memory to allocate for rap and jrap.
 * The second time would fill in these arrays. Actually you might
 * be able to include the filling in of jrap into the first pass;
 * just overestimate its size (its an integer array) and cut it
 * back before the second time through. This would avoid some if tests
 * in the second pass.
 *
 * Questions
 *            1) parallel (PetSc) version?
 *            2) what if we don't store R row-wise and don't
 *               even want to store a copy of it in this form
 *               temporarily? 
 *--------------------------------------------------------------------------*/

/*!
 * \fn JX_Int fsls_AMGAHTruncation
 * \brief Truncate some entries in A, the coarse grid operater.
 * \note the truncation rules are as follows:
 *  1. when trunc_factor > 0.0, we do the following for the i-th row:
 *     (1) a_{i,j} := 0.0, if |a_{i,j}| < trunc_factor*diag;
 *     (2) a_{i,i} += \sum a_{i,j}, for all |a_{i,j}| < trunc_factor*diag,
 *  where diag = max{a_{i,i}}.
 *  2. when max_elmts > 0, we do the following for the i-th row if nonzeros[i] > max_elmts :
 *     (1) reorder all the entries decendingly according to their absolute values;
 *     (2) only keep the first max_elmts entries;
 *     (3) re-scale the remaining entries to so that the total sum remains unchanged. 
 * \remark Attention, if (trunc_factor > 0.0) and ( max_elmts > 0) are satisfied at the same
 *         time, the former would be treated first.
 * \author peghoty
 * \date 2010/08/29
 */
JX_Int
fsls_AMGAHTruncation( fsls_CSRMatrix *A, JX_Real trunc_factor, JX_Int max_elmts )
{
   JX_Int     *A_i    = fsls_CSRMatrixI(A);
   JX_Int     *A_j    = fsls_CSRMatrixJ(A);
   JX_Real  *A_data = fsls_CSRMatrixData(A);
   JX_Int      num_variables = fsls_CSRMatrixNumRows(A);
   
   JX_Int      now_checking, num_lost, next_open;
   JX_Real   max_coef = 0.0, row_sum, scale;
   JX_Int      i, j, start, diag; 
   JX_Int      ierr = 0;     

   if (trunc_factor > 0.0)
   {
      num_lost = 0;
      next_open = 0;
      now_checking = 0;
      /* max_coef = trunc_factor*max(|a_{ii}|) */
      for (i = 0; i < num_variables; i ++)
      {
         max_coef = 0.0;
	 if (max_coef < fabs(A_data[A_i[i]])) 
	 {
	    max_coef = fabs(A_data[A_i[i]]);
	 }
         max_coef *= trunc_factor;
      }
      for (i = 0; i < num_variables; i ++)
      {
         start = A_i[i];
         A_i[i] -= num_lost;
	 diag = next_open;
	 A_data[next_open] = A_data[now_checking];
	 A_j[next_open] = A_j[now_checking];
	 now_checking ++;
	 next_open ++;
         for (j = start+1; j < A_i[i+1]; j ++) /* off-diagonal entries */
         {
	    if (fabs(A_data[now_checking]) < max_coef)
	    {
	       /* the truncated entries are added to the diagonal entry, modified by peghoty 2010/09/09 */
	       A_data[diag] += A_data[now_checking];	    
	       num_lost ++;
	       now_checking ++;
	    }
	    else
	    {
	       A_data[next_open] = A_data[now_checking];
	       A_j[next_open] = A_j[now_checking];
	       now_checking ++;
	       next_open ++;
	    }
         }
      }
      A_i[num_variables] -= num_lost;
      fsls_CSRMatrixNumNonzeros(A) = A_i[num_variables];
   }
   else if (max_elmts > 0)
   {
      next_open = 0;
      start = 0;
      for (i = 0; i < num_variables; i ++)
      {
         /* get the sum of the i-th row */
         row_sum = 0.0;
         for (j = A_i[i]; j < A_i[i+1]; j ++)
         {
	    row_sum += A_data[j];
	 }
	 start = next_open;
	 if (A_i[i] > next_open)
	 {
	    for (j = A_i[i]; j < A_i[i+1]; j ++)
	    {
	       A_data[next_open] = A_data[j];
	       A_j[next_open++]  = A_j[j];
	    }
	 }
	 /* reorder 'A_j' and 'A_data' in the same way */
 	 if ((A_i[i+1] - A_i[i]) > max_elmts)
	 {
	    /* modified by peghoty 2010/09/09 */
	    if (i == 0)
	    {
	       fsls_idFabsQuickSort21(A_j, A_data, 0, max_elmts-1); // 21: descendingly
	    }
	    else
	    {
	       fsls_idFabsQuickSort21(A_j, A_data, start, next_open-1); // 21: descendingly
	    }
	    next_open = start + max_elmts;
	 }
	 else
	 {
	    next_open = start + (A_i[i+1] - A_i[i]);
	 }

         scale = 0.0;
	 for (j = start; j < next_open; j ++)
 	 {
	    scale += A_data[j];
 	 }
         
         if (scale != 0 && scale != row_sum)
         {
	    scale = row_sum / scale;
	    
	    /* re-scale to keep the original row sum, peghoty */
            for (j = start; j < next_open; j ++)
            {
	       A_data[j] *= scale;
	    }
         }
	 A_i[i] = start;
      }
      A_i[num_variables] = next_open;
      fsls_CSRMatrixNumNonzeros(A) = next_open;
   }

   return ierr;
}

/*!
 * \fn fsls_AMGCoarsenRSNew
 * \brief Ruge's coarsening algorithm using double_linked_list 
 * \param *A pointer to the matrix
 * \param *S pointer to the strength matrix of A
 * \param **CF_marker_ptr pointer to the pointer to the CF marker
 * \param *coarse_size_ptr pointer to the number of C points
 * \note Compared with hypre's corresponding function, two main differences
 *       can be described as follows:
 *       (1) the measures of the unassigned points which strongly
 *           inference the previously selected F points are
 *           updated(+1) before they enter the JX_Real linked
 *           list. It's supposed to be more effecient.  
 *       (2) I modified the rules to make sure that only one F
 *           point is changed to be C point every time a pair of F points
 *           that are strongly connected voilate the C1 condition.
 * \remark Add the treatment to special F-points.
 * \author peghoty
 * \date 2010/11/12
 */
JX_Int
fsls_AMGCoarsenRSNew( fsls_CSRMatrix    *A,
                      fsls_ICSRMatrix   *S,
                      JX_Int              **CF_marker_ptr,
                      JX_Int               *coarse_size_ptr )
{
   JX_Int              num_variables = fsls_CSRMatrixNumRows(A);
              
   JX_Int             *S_i;
   JX_Int             *S_j;
                 
   fsls_ICSRMatrix *ST;
   JX_Int             *ST_i;
   JX_Int             *ST_j;
                 
   JX_Int             *CF_marker;
   JX_Int              coarse_size;

   JX_Int             *measure_array;
   JX_Int             *graph_array;

   JX_Int              measure;
   JX_Int              i, j, k;
   JX_Int		    ji, jj, index;
   JX_Int		    set_empty = 1;
   JX_Int		    num_strong;
   JX_Int              cnt;
   JX_Int              jkeep = 0;

   JX_Int              ierr = 0;

   fsls_LinkList  LoL_head;
   fsls_LinkList  LoL_tail;

   JX_Int *lists;
   JX_Int *where;
   JX_Int  new_meas;
   JX_Int  num_left;
   JX_Int  nabor, nabor_two;


  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 1: Initialize the C/F marker, LoL_head, LoL_tail arrays
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = fsls_CTAlloc(JX_Int, num_variables);
   where = fsls_CTAlloc(JX_Int, num_variables);

   CF_marker = fsls_CTAlloc(JX_Int, num_variables);

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 2: Compute the measures, here the measures are given 
   *         by the row sums of ST. Hence, measure_array[i] is
   *         the number of influences of variable i.
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   S_i = fsls_ICSRMatrixI(S);
   S_j = fsls_ICSRMatrixJ(S);

   num_strong = S_i[num_variables];  // number of nonzeros in S
   ST = fsls_ICSRMatrixCreate(num_variables, num_variables, num_strong);

   ST_i = fsls_CTAlloc(JX_Int, num_variables+1);
   fsls_ICSRMatrixI(ST) = ST_i;

   ST_j = fsls_CTAlloc(JX_Int, num_strong);
   fsls_ICSRMatrixJ(ST) = ST_j;

   fsls_ICSRMatrixNumNonzeros(ST) = num_strong;

  /*----------------------------------------------------------
   * generate transpose of S, ST
   * note: Compared to the CLJP, you may find that: the 
   *       transposing operation in CLJP is not implemented 
   *       since ST is not needed.  peghoty
   *----------------------------------------------------------*/
 
   for (i = 0; i < num_strong; i ++)
   {
      ST_i[S_j[i]+1] ++;
   }
  
   /* get the true 'IA' array for ST */
   for (i = 0; i < num_variables; i ++)
   {
      ST_i[i+1] += ST_i[i];
   }
 
   /* get the true 'JA' array for ST, but 'IA' has been modified */
   for (i = 0; i < num_variables; i ++)
   {
      for (j = S_i[i]; j < S_i[i+1]; j ++)
      {
           index = S_j[j];    // column index of S, i.e., row index of ST
           ST_j[ST_i[index]] = i;
           ST_i[index] ++;
      }
   }
 
   /* recover the true 'IA' array */      
   for (i = num_variables; i > 0; i --)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;


   /* Compute the measures */
   measure_array = fsls_CTAlloc(JX_Int, num_variables);
   for (j = 0; j < num_variables; j ++) 
   {    
      measure_array[j] = ST_i[j+1] - ST_i[j];
   }    

 
  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 3: Initialize the lists
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/   

   num_left = num_variables;
   coarse_size = 0;
   
   /* Update measures of j\in S_i, where 'i' s.t. measure[i]=0. peghoty */
   for (j = 0; j < num_variables; j ++) 
   {  
      if (S_i[j+1] == S_i[j])
      {
         CF_marker[j] = SFPOINT; /* SFPOINT = -3 */
         measure_array[j] = 0;
         num_left --;
      }
      else if (measure_array[j] == 0)
      {
         CF_marker[j] = FPOINT; /* FPOINT = -1 */
         num_left --;
         for (k = S_i[j]; k < S_i[j+1]; k ++)
         {
            nabor = S_j[k];
            if (CF_marker[nabor] != SFPOINT) /* this 'if' is indispensible. peghoty 2011/01/24 */
            {
               measure_array[nabor] ++; 
            } 
         }
      }
   }   

   /* Put the points with positive measures into the list. peghoty */
   for (j = 0; j < num_variables; j ++) 
   {  
      measure = measure_array[j];
      if (measure > 0)
      {
         fsls_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
      }
   }   


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 4: The First Coloring Pass of RS 
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 
   
   
  /****************************************************************
   *
   *  Main loop of Ruge-Stueben first coloring pass.
   *
   *  WHILE there are still points to classify DO:
   *        1) find first point, i, on list with max_measure
   *           make i a C-point, remove it from the lists
   *        2) For each point, j, in S_i^T,
   *           a) Set j to be an F-point
   *           b) For each point, k, in S_j
   *                move k to the list in LoL with measure one
   *                greater than it occupies (creating new LoL
   *                entry if necessary)
   *
   ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;  /* variable 'index' has the maximal measures, 
                                    and especially it firstly enters the list. */

      CF_marker[index] = CPOINT;
      ++coarse_size;
      measure = measure_array[index];
      measure_array[index] = 0;  /* clear up the measure of variable 'index' */
      --num_left;
      
      fsls_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
      
      /* deal with the UNDECIDED variables in S^{T}_{index} */
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = FPOINT;
            measure = measure_array[nabor];

            fsls_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;
            
            /* deal with the UNDECIDED variables in S_{nabor} */
            for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  
                  fsls_remove_point(&LoL_head, &LoL_tail, measure,nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
               }
            }
         }
      }
      
      /* deal with the UNDECIDED variables in S_{index} */
      for (j = S_i[index]; j < S_i[index+1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            fsls_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            if (measure > 0)
            {
               fsls_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            }
            else
            {
               CF_marker[nabor] = FPOINT;
               --num_left;
             
               /* deal with the UNDECIDED variables in S_{nabor} */
               for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     
                     fsls_remove_point(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

   fsls_TFree(lists);
   fsls_TFree(where);
   fsls_TFree(LoL_head);
   fsls_TFree(LoL_tail);
   fsls_TFree(measure_array);
   fsls_ICSRMatrixDestroy(ST);
   

  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 5: Second pass, check fine points for coarse neighbors
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
   /* Initialize the graph array */
   graph_array = fsls_CTAlloc(JX_Int, num_variables);
   for (i = 0; i < num_variables; i ++)
   {
      graph_array[i] = -1;
   }
 

   for (i = 0; i < num_variables; i ++)
   {      
      if (CF_marker[i] == FPOINT)
      {
         /* if 'i' is an F-point, mark all the C-points in S_{i}
            by graph_array, 'graph_array[j] = i' means 'j\in S_{i}\cap C' */
	 for (ji = S_i[i]; ji < S_i[i+1]; ji++)
	 {
	    j = S_j[ji];
	    if (CF_marker[j] == CPOINT)
	    {
	       graph_array[j] = i;  /* i(F)->j(C) */
	    }
 	 }
 	 
 	 cnt = 0;
 	 
	 for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
	 {
	      j = S_j[ji];
	    
	      if (CF_marker[j] == FPOINT)
	      {
	          /* (i,j) is an F-pair */
	       
	          set_empty = 1;
	       
	          for (jj = S_i[j]; jj < S_i[j+1]; jj ++)
	          {
		     index = S_j[jj];
		     if (graph_array[index] == i)
		     {
		        set_empty = 0;
		        break;
	             }
	          }
	       
	          if (set_empty)
	          {
	               if (cnt == 0)
	               {
                          CF_marker[j] = CPOINT;
                          coarse_size ++;
                          graph_array[j] = i;
                          jkeep = j; 
                          cnt = 1;
	               }
	               else
	               {
	                  CF_marker[i] = CPOINT;
	                  CF_marker[jkeep] = FPOINT;
	                  break;
	               }
	          }
	      }
	    
	 } // end for 'ji'
	 	 
      } // end if
      
   } // end for 'i'

  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   fsls_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*!
 * \fn fsls_AMGCoarsenRS
 * \brief Ruge's coarsening algorithm using double_linked_list 
 * \param *A pointer to the matrix
 * \param *S pointer to the strength matrix of A
 * \param **CF_marker_ptr pointer to the pointer to the CF marker
 * \param *coarse_size_ptr pointer to the number of C points
 * \note Compared with hypre's corresponding function, two main differences
 *       can be described as follows:
 *       (1) the measures of the unassigned points which strongly
 *           inference the previously selected F points are
 *           updated(+1) before they enter the JX_Real linked
 *           list. It's supposed to be more effecient.  
 *       (2) I modified the rules to make sure that only one F
 *           point is changed C point every time a pair of F points
 *           that are strongly connected voilate the C1 condition.
 * \author peghoty
 * \date 2010/08/29
 */
JX_Int
fsls_AMGCoarsenRS( fsls_CSRMatrix    *A,
                   fsls_ICSRMatrix   *S,
                   JX_Int              **CF_marker_ptr,
                   JX_Int               *coarse_size_ptr )
{
   JX_Int              num_variables = fsls_CSRMatrixNumRows(A);
              
   JX_Int             *S_i;
   JX_Int             *S_j;
                 
   fsls_ICSRMatrix *ST;
   JX_Int             *ST_i;
   JX_Int             *ST_j;
                 
   JX_Int             *CF_marker;
   JX_Int              coarse_size;

   JX_Int             *measure_array;
   JX_Int             *graph_array;

   JX_Int              measure;
   JX_Int              i, j, k;
   JX_Int		    ji, jj, index;
   JX_Int		    set_empty = 1;
   JX_Int		    num_strong;
   JX_Int              cnt;
   JX_Int              jkeep = 0;

   JX_Int              ierr = 0;

   fsls_LinkList  LoL_head;
   fsls_LinkList  LoL_tail;

   JX_Int *lists;
   JX_Int *where;
   JX_Int  new_meas;
   JX_Int  num_left;
   JX_Int  nabor, nabor_two;


  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 1: Initialize the C/F marker, LoL_head, LoL_tail arrays
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = fsls_CTAlloc(JX_Int, num_variables);
   where = fsls_CTAlloc(JX_Int, num_variables);

   CF_marker = fsls_CTAlloc(JX_Int, num_variables);

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 2: Compute the measures, here the measures are given 
   *         by the row sums of ST. Hence, measure_array[i] is
   *         the number of influences of variable i.
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   S_i = fsls_ICSRMatrixI(S);
   S_j = fsls_ICSRMatrixJ(S);

   num_strong = S_i[num_variables];  // number of nonzeros in S
   ST = fsls_ICSRMatrixCreate(num_variables, num_variables, num_strong);

   ST_i = fsls_CTAlloc(JX_Int, num_variables+1);
   fsls_ICSRMatrixI(ST) = ST_i;

   ST_j = fsls_CTAlloc(JX_Int, num_strong);
   fsls_ICSRMatrixJ(ST) = ST_j;

   fsls_ICSRMatrixNumNonzeros(ST) = num_strong;

  /*----------------------------------------------------------
   * generate transpose of S, ST
   * note: Compared to the CLJP, you may find that: the 
   *       transposing operation in CLJP is not implemented 
   *       since ST is not needed.  peghoty
   *----------------------------------------------------------*/
 
   for (i = 0; i < num_strong; i ++)
   {
      ST_i[S_j[i]+1] ++;
   }
  
   /* get the true 'IA' array for ST */
   for (i = 0; i < num_variables; i ++)
   {
      ST_i[i+1] += ST_i[i];
   }
 
   /* get the true 'JA' array for ST, but 'IA' has been modified */
   for (i = 0; i < num_variables; i ++)
   {
      for (j = S_i[i]; j < S_i[i+1]; j ++)
      {
           index = S_j[j];    // column index of S, i.e., row index of ST
           ST_j[ST_i[index]] = i;
           ST_i[index] ++;
      }
   }
 
   /* recover the true 'IA' array */      
   for (i = num_variables; i > 0; i --)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;


   /* Compute the measures */
   measure_array = fsls_CTAlloc(JX_Int, num_variables);
   for (j = 0; j < num_variables; j ++) 
   {    
      measure_array[j] = ST_i[j+1] - ST_i[j];
   }    

 
  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 3: Initialize the lists
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/   

   num_left = num_variables;
   coarse_size = 0;
   
   /* Update measures of j\in S_i, where 'i' s.t. measure[i]=0. peghoty */
   for (j = 0; j < num_variables; j ++) 
   {  
      if (measure_array[j] == 0)
      {
         CF_marker[j] = FPOINT; /* FPOINT = -1 */
         num_left --;
         for (k = S_i[j]; k < S_i[j+1]; k ++)
         {
            nabor = S_j[k];
            measure_array[nabor] ++;  
         }
      }
   }   

   /* Put the points with positive measures into the list. peghoty */
   for (j = 0; j < num_variables; j ++) 
   {  
      measure = measure_array[j];
      if (measure > 0)
      {
         fsls_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
      }
   }   


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 4: The First Coloring Pass of RS 
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 
   
   
  /****************************************************************
   *
   *  Main loop of Ruge-Stueben first coloring pass.
   *
   *  WHILE there are still points to classify DO:
   *        1) find first point, i, on list with max_measure
   *           make i a C-point, remove it from the lists
   *        2) For each point, j, in S_i^T,
   *           a) Set j to be an F-point
   *           b) For each point, k, in S_j
   *                move k to the list in LoL with measure one
   *                greater than it occupies (creating new LoL
   *                entry if necessary)
   *
   ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;  /* variable 'index' has the maximal measures, 
                                    and especially it firstly enters the list. */

      CF_marker[index] = CPOINT;
      ++coarse_size;
      measure = measure_array[index];
      measure_array[index] = 0;  /* clear up the measure of variable 'index' */
      --num_left;
      
      fsls_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
      
      /* deal with the UNDECIDED variables in S^{T}_{index} */
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = FPOINT;
            measure = measure_array[nabor];

            fsls_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;
            
            /* deal with the UNDECIDED variables in S_{nabor} */
            for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  
                  fsls_remove_point(&LoL_head, &LoL_tail, measure,nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
               }
            }
         }
      }
      
      /* deal with the UNDECIDED variables in S_{index} */
      for (j = S_i[index]; j < S_i[index+1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            fsls_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            if (measure > 0)
            {
               fsls_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            }
            else
            {
               CF_marker[nabor] = FPOINT;
               --num_left;
             
               /* deal with the UNDECIDED variables in S_{nabor} */
               for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     
                     fsls_remove_point(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

   fsls_TFree(lists);
   fsls_TFree(where);
   fsls_TFree(LoL_head);
   fsls_TFree(LoL_tail);
   fsls_TFree(measure_array);
   fsls_ICSRMatrixDestroy(ST);
   

  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 5: Second pass, check fine points for coarse neighbors
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
   /* Initialize the graph array */
   graph_array = fsls_CTAlloc(JX_Int, num_variables);
   for (i = 0; i < num_variables; i ++)
   {
      graph_array[i] = -1;
   }
 

   for (i = 0; i < num_variables; i ++)
   {      
      if (CF_marker[i] == FPOINT)
      {
         /* if 'i' is an F-point, mark all the C-points in S_{i}
            by graph_array, 'graph_array[j] = i' means 'j\in S_{i}\cap C' */
	 for (ji = S_i[i]; ji < S_i[i+1]; ji++)
	 {
	    j = S_j[ji];
	    if (CF_marker[j] == CPOINT)
	    {
	       graph_array[j] = i;  /* i(F)->j(C) */
	    }
 	 }
 	 
 	 cnt = 0;
 	 
	 for (ji = S_i[i]; ji < S_i[i+1]; ji ++)
	 {
	      j = S_j[ji];
	    
	      if (CF_marker[j] == FPOINT)
	      {
	          /* (i,j) is an F-pair */
	       
	          set_empty = 1;
	       
	          for (jj = S_i[j]; jj < S_i[j+1]; jj ++)
	          {
		     index = S_j[jj];
		     if (graph_array[index] == i)
		     {
		        set_empty = 0;
		        break;
	             }
	          }
	       
	          if (set_empty)
	          {
	               if (cnt == 0)
	               {
                          CF_marker[j] = CPOINT;
                          coarse_size ++;
                          graph_array[j] = i;
                          jkeep = j; 
                          cnt = 1;
	               }
	               else
	               {
	                  CF_marker[i] = CPOINT;
	                  CF_marker[jkeep] = FPOINT;
	                  break;
	               }
	          }
	      }
	    
	 } // end for 'ji'
	 	 
      } // end if
      
   } // end for 'i'

  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   fsls_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}


/*!
 * \fn fsls_AMGCoarsenRugeLoL
 * \brief Ruge's coarsening algorithm using double_linked_list
 * \param *A pointer to the matrix
 * \param *S pointer to the strength matrix of A
 * \param **CF_marker_ptr pointer to the pointer to the CF marker
 * \param *coarse_size_ptr pointer to the number of C points
 * \note this is the original code from hypre2.6.0b
 * \author peghoty
 * \date 2010/01/18
 */
JX_Int
fsls_AMGCoarsenRugeLoL( fsls_CSRMatrix   *A,
                        fsls_ICSRMatrix  *S,
                        JX_Int             **CF_marker_ptr,
                        JX_Int              *coarse_size_ptr )
{
   JX_Int              num_variables = fsls_CSRMatrixNumRows(A);
              
   JX_Int             *S_i;
   JX_Int             *S_j;
                 
   fsls_ICSRMatrix *ST;
   JX_Int             *ST_i;
   JX_Int             *ST_j;
                 
   JX_Int             *CF_marker;
   JX_Int              coarse_size;

   JX_Int             *measure_array;
   JX_Int             *graph_array;

   JX_Int              measure;
   JX_Int              i, j, k;
   JX_Int		    ji, jj, index;
   JX_Int		    ci_tilde = -1;
   JX_Int		    ci_tilde_mark = -1;
   JX_Int		    set_empty = 1;
   JX_Int		    C_i_nonempty = 0;
   JX_Int		    num_strong;

   JX_Int              ierr = 0;

   fsls_LinkList LoL_head;
   fsls_LinkList LoL_tail;

   JX_Int *lists, *where;
   JX_Int  new_meas;
   JX_Int  num_left;
   JX_Int  nabor, nabor_two;


  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 1: Initialize the C/F marker, LoL_head, LoL_tail arrays
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = fsls_CTAlloc(JX_Int, num_variables);
   where = fsls_CTAlloc(JX_Int, num_variables);

   CF_marker = fsls_CTAlloc(JX_Int, num_variables);

#if 0   
   for (j = 0; j < num_variables; j ++) /* it's unnecessary peghoty */
   {
      CF_marker[j] = UNDECIDED;
   }
#endif   

  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 2: Compute the measures, here the measures are given 
   *         by the row sums of ST. Hence, measure_array[i] is
   *         the number of influences of variable i.
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

   S_i = fsls_ICSRMatrixI(S);
   S_j = fsls_ICSRMatrixJ(S);

   num_strong = S_i[num_variables];  // number of nonzeros in S
   ST = fsls_ICSRMatrixCreate(num_variables, num_variables, num_strong);

   ST_i = fsls_CTAlloc(JX_Int, num_variables+1);
   fsls_ICSRMatrixI(ST) = ST_i;

   ST_j = fsls_CTAlloc(JX_Int, num_strong);
   fsls_ICSRMatrixJ(ST) = ST_j;

   fsls_ICSRMatrixNumNonzeros(ST) = num_strong;

  /*----------------------------------------------------------
   * generate transpose of S, ST
   * note: Compared to the CLJP, you may find that: the 
   *       transposing operation in CLJP is not implemented 
   *       since ST is not needed.  peghoty
   *----------------------------------------------------------*/

#if 0
   for (i = 0; i <= num_variables; i ++) /* it's unnecessary peghoty */
   {
      ST_i[i] = 0;
   }
#endif
 
   for (i = 0; i < num_strong; i ++)
   {
      ST_i[S_j[i]+1]++;
   }
   
   /* get the true 'IA' array for ST */
   for (i = 0; i < num_variables; i ++)
   {
      ST_i[i+1] += ST_i[i];
   }
   
   /* get the true 'JA' array for ST, but 'IA' has been modified */
   for (i = 0; i < num_variables; i ++)
   {
      for (j = S_i[i]; j < S_i[i+1]; j ++)
      {
           index = S_j[j];    // column index of S, i.e., row index of ST
           ST_j[ST_i[index]] = i;
           ST_i[index]++;
      }
   }

   /* recover the true 'IA' array */      
   for (i = num_variables; i > 0; i --)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;


   /* Compute the measures */
   measure_array = fsls_CTAlloc(JX_Int, num_variables);
   for (j = 0; j < num_variables; j ++) 
   {    
      measure_array[j] = ST_i[j+1] - ST_i[j];
   }    
   
 
  /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 3: Initialize the lists
   *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/   

   num_left = num_variables;
   coarse_size = 0;
 
   for (j = 0; j < num_variables; j ++) 
   {    
      measure = measure_array[j];
      if (measure > 0) 
      {
         fsls_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
      }
      else
      {
         if (measure < 0) 
         {
            /* It will never take place! Just for robustness peghoty */
            jx_printf("negative measure!\n"); 
         }
         
         CF_marker[j] = FPOINT; /* FPOINT = -1 */
         
	 for (k = S_i[j]; k < S_i[j+1]; k ++)
	 {
	    nabor = S_j[k];
	    if (nabor < j)  /* 'j' has been delt with */
	    {
	       new_meas = measure_array[nabor];
	       if (new_meas > 0)
	       {  
	          /* This holds without any condition */
		  fsls_remove_point(&LoL_head, &LoL_tail, new_meas,
				     nabor, lists, where);
               }
	       new_meas = ++(measure_array[nabor]);
	       fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
				    nabor, lists, where);
	    }
	    else  /* 'j' hasn't been delt with */
	    {
	       new_meas = ++(measure_array[nabor]);
	    }
  	 }
         --num_left;
      }
   }
   

  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 4: The First Coloring Pass of RS 
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 
   
   
  /****************************************************************
   *
   *  Main loop of Ruge-Stueben first coloring pass.
   *
   *  WHILE there are still points to classify DO:
   *        1) find first point, i, on list with max_measure
   *           make i a C-point, remove it from the lists
   *        2) For each point, j, in S_i^T,
   *           a) Set j to be an F-point
   *           b) For each point, k, in S_j
   *                move k to the list in LoL with measure one
   *                greater than it occupies (creating new LoL
   *                entry if necessary)
   *
   ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;  /* variable 'index' has the maximal measures, 
                                    and especially it firstly enters the list. */

      CF_marker[index] = CPOINT;
      ++coarse_size;
      measure = measure_array[index];
      measure_array[index] = 0;  /* clear up the measure of variable 'index' */
      --num_left;
      
      fsls_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
      
      /* deal with the UNDECIDED variables in S^{T}_{index} */
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = FPOINT;
            measure = measure_array[nabor];

            fsls_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;
            
            /* deal with the UNDECIDED variables in S_{nabor} */
            for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  
                  fsls_remove_point(&LoL_head, &LoL_tail, measure,nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
               }
            }
         }
      }
      
      /* deal with the UNDECIDED variables in S_{index} */
      for (j = S_i[index]; j < S_i[index+1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            fsls_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            if (measure > 0)
            {
               fsls_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            }
            else
            {
               CF_marker[nabor] = FPOINT;
               --num_left;
             
               /* deal with the UNDECIDED variables in S_{nabor} */
               for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     
                     fsls_remove_point(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     fsls_enter_on_lists(&LoL_head, &LoL_tail, new_meas, nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

   fsls_TFree(lists);
   fsls_TFree(where);
   fsls_TFree(LoL_head);
   fsls_TFree(LoL_tail);
   fsls_TFree(measure_array);
   fsls_ICSRMatrixDestroy(ST);
   

  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   * Step 5: Second pass, check fine points for coarse neighbors
   *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   
   /* Initialize the graph array */
   graph_array = fsls_CTAlloc(JX_Int, num_variables);
   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }
   
  /*--------------------------------------------------
   * Initial Value:
   * ci_tilde      = -1;
   * ci_tilde_mark = -1;
   * set_empty     =  1;
   * C_i_nonempty  =  0;
   *
   * remark: 'a |= b' means 'a = (a | b)'
   *         (a |= b) is false only when a=b=0,
   *         otherwise it's true.
   *
   *--------------------------------------------------*/

   for (i = 0; i < num_variables; i++)
   {
      if (ci_tilde_mark |= i) 
        ci_tilde = -1;        // change "|=" into "!=" will be more reasonable!! peghoty 2010/04/03
      
      if (CF_marker[i] == FPOINT)
      {
         /* if 'i' is an F-point, mark all the C-points in S_{i}
            by graph_array, 'graph_array[j] = i' means 'j\in S_{i}\cap C' */
	 for (ji = S_i[i]; ji < S_i[i+1]; ji++)
	 {
	    j = S_j[ji];
	    if (CF_marker[j] > 0)
	    {
	       graph_array[j] = i;  /* i(F)->j(C) */
	    }
 	 }
 	 
 	 
	 for (ji = S_i[i]; ji < S_i[i+1]; ji++)
	 {
	    j = S_j[ji];
	    
	    if (CF_marker[j] == -1)
	    {
	    
	       set_empty = 1;
	       
	       for (jj = S_i[j]; jj < S_i[j+1]; jj++)
	       {
		  index = S_j[jj];
		  if (graph_array[index] == i)
		  {
		     set_empty = 0;
		     break;
	          }
	       }
	       
	       if (set_empty)
	       {
		    if (C_i_nonempty)
		    {
		       CF_marker[i] = 1;
		       coarse_size++;
		       if (ci_tilde > -1)
		       {
			  CF_marker[ci_tilde] = -1;
		          coarse_size--;
		          ci_tilde = -1;
		       }
		       C_i_nonempty = 0;
		       break;
		    }
		    else
		    {
		       ci_tilde = j;
		       ci_tilde_mark = i;
		       CF_marker[j] = 1;
		       coarse_size++;
		       C_i_nonempty = 1;
		       i--;
		       break;
		    }
	       }
	       
	    }
	 }
	
      }
   }

  /*---------------------------------------------------
   * Clean up and return
   *---------------------------------------------------*/

   fsls_TFree(graph_array);

   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;

   return (ierr);
}

/*!
 * \fn JX_Int fsls_AMGCreateS
 * \brief Creates strength matrix for A.
 * \param *A pointer to the matrix
 * \param strength_threshold parameter to define 'strength connection'
 * \param mode type of defining a strength matrix
 * \param **S_ptr pointer to the pointer of the resulting matrix
 * \note the "strength" of dependence / influence is defined in
 *       the following way: 
 *    (1) when mode = 0
 *       'i depends on j' if
 *        aij >= strength_threshold*fsls_max (k != i) aik, aii < 0
 *        or
 *        aij <= strength_threshold*fsls_min (k != i) aik, aii >= 0
 *    (2) when mode != 0
 *       'i depends on j' if 'aij >= strength_threshold*fsls_max (k != i) |aik|
 *       Then S_ij = -1, else S_ij = 0.
 * \author peghoty
 * \date 2010/08/29 
 */
JX_Int
fsls_AMGCreateS( fsls_CSRMatrix *A, JX_Real strength_threshold, JX_Int mode, fsls_ICSRMatrix **S_ptr )
{
   JX_Int             *A_i           = fsls_CSRMatrixI(A);
   JX_Int             *A_j           = fsls_CSRMatrixJ(A);
   JX_Real          *A_data        = fsls_CSRMatrixData(A);
   JX_Int              num_variables = fsls_CSRMatrixNumRows(A);
                  
   fsls_ICSRMatrix *S;
   JX_Int             *S_i;
   JX_Int             *S_j;
   JX_Int             *S_data;
                 
   JX_Real           diag, row_scale;
   JX_Int              i, jA, jS;
   JX_Int              ierr = 0;
   
   S = fsls_ICSRMatrixCreate(num_variables, num_variables, A_i[num_variables]);
   fsls_ICSRMatrixInitialize(S);

   S_i    = fsls_ICSRMatrixI(S);
   S_j    = fsls_ICSRMatrixJ(S);
   S_data = fsls_ICSRMatrixData(S);

   /* give S same nonzero structure as A */
   for (i = 0; i < num_variables; i ++)
   {
      S_i[i] = A_i[i];
      for (jA = A_i[i]; jA < A_i[i+1]; jA ++)
      {
         S_j[jA] = A_j[jA];
      }
   }
   S_i[num_variables] = A_i[num_variables];

   if (mode) /* consider the absolute value */
   {
       for (i = 0; i < num_variables; i ++)
       {
          diag = A_data[A_i[i]];

          /* compute scaling factor */
          row_scale = 0.0;
          for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
          {
             row_scale = fsls_max(row_scale, fabs(A_data[jA]));
          }

          /* compute row entries of S */
          S_data[A_i[i]] = 0;
          for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
          {
             S_data[jA] = 0;
             if ( fabs(A_data[jA]) >= strength_threshold*row_scale )
             {            
                S_data[jA] = -1;
             }
          }
       }
   }
   else  /* mode = 0 (default) */
   {  
       for (i = 0; i < num_variables; i ++)
       {
           diag = A_data[A_i[i]];

           /* compute scaling factor */
           row_scale = 0.0;
           if (diag < 0)
           {
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 row_scale = fsls_max(row_scale, A_data[jA]);
              }
           }
           else /* diag >= 0 */
           {
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 row_scale = fsls_min(row_scale, A_data[jA]);
              }
           }

           /* compute row entries of S */
           S_data[A_i[i]] = 0;
           if (diag < 0) 
           { 
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 S_data[jA] = 0;
                 if ( A_data[jA] >= strength_threshold*row_scale )
                 {
                    S_data[jA] = -1;
                 }
              }
           }
           else /* diag >= 0 */
           {
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 S_data[jA] = 0;
                 if ( A_data[jA] <= strength_threshold*row_scale )
                 {
                    S_data[jA] = -1;
                 }
              }
           }
       }
   }

  /*-----------------------------------------------------------------
   * "Compress" the strength matrix.
   * NOTE: S has *NO DIAGONAL ELEMENT* on any row.
   * NOTE: This "compression" section of code may be removed, and
   * coarsening will still be done correctly.  However, the routine
   * that builds interpolation would have to be modified first.
   *----------------------------------------------------------------*/
   jS = 0;
   for (i = 0; i < num_variables; i ++)
   {
      S_i[i] = jS;
      for (jA = A_i[i]; jA < A_i[i+1]; jA ++)
      {
         if (S_data[jA])
         {
            S_j[jS]    = S_j[jA];
            S_data[jS] = S_data[jA];
            jS ++;
         }
      }
   }
   S_i[num_variables] = jS;
   fsls_ICSRMatrixNumNonzeros(S) = jS;

   *S_ptr = S;

   return (ierr);
}


/*!
 * \fn JX_Int fsls_AMGCreateSNew
 * \brief Creates strength matrix for A.
 * \param *A pointer to the matrix
 * \param strength_threshold parameter to define 'strength connection'
 * \param max_row_sum parameter to modify the definition of 'strength connection'
 * \param mode type of defining a strength matrix
 * \param **S_ptr pointer to the pointer of the resulting matrix
 * \note the "strength" of dependence / influence is defined in
 *       the following way: 
 *    (1) when mode = 0
 *       'i depends on j' if
 *        aij >= strength_threshold*fsls_max (k != i) aik, aii < 0
 *        or
 *        aij <= strength_threshold*fsls_min (k != i) aik, aii >= 0
 *    (2) when mode != 0
 *       'i depends on j' if 'aij >= strength_threshold*fsls_max (k != i) |aik|
 *       Then S_ij = -1, else S_ij = 0.
 * \note Attention to the parameter 'max_row_sum' and how it works.
 * \author peghoty
 * \date 2010/11/11 
 */
JX_Int
fsls_AMGCreateSNew( fsls_CSRMatrix    *A, 
                    JX_Real             strength_threshold,
                    JX_Real             max_row_sum, 
                    JX_Int                mode, 
                    fsls_ICSRMatrix  **S_ptr )
{
   JX_Int             *A_i           = fsls_CSRMatrixI(A);
   JX_Int             *A_j           = fsls_CSRMatrixJ(A);
   JX_Real          *A_data        = fsls_CSRMatrixData(A);
   JX_Int              num_variables = fsls_CSRMatrixNumRows(A);
                  
   fsls_ICSRMatrix *S;
   JX_Int             *S_i;
   JX_Int             *S_j;
   JX_Int             *S_data;
                 
   JX_Real           diag, row_scale, row_sum;
   JX_Int              i, jA, jS;
   JX_Int              ierr = 0;
   
   S = fsls_ICSRMatrixCreate(num_variables, num_variables, A_i[num_variables]);
   fsls_ICSRMatrixInitialize(S);

   S_i    = fsls_ICSRMatrixI(S);
   S_j    = fsls_ICSRMatrixJ(S);
   S_data = fsls_ICSRMatrixData(S);

   /* give S same nonzero structure as A */
   for (i = 0; i < num_variables; i ++)
   {
      S_i[i] = A_i[i];
      for (jA = A_i[i]; jA < A_i[i+1]; jA ++)
      {
         S_j[jA] = A_j[jA];
      }
   }
   S_i[num_variables] = A_i[num_variables];

   if (mode) /* consider the absolute value */
   {
       for (i = 0; i < num_variables; i ++)
       {
          diag = A_data[A_i[i]];

          /* compute scaling factor */
          row_scale = 0.0;
          row_sum   = diag;
          for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
          {
             row_scale = fsls_max(row_scale, fabs(A_data[jA]));
             row_sum += fabs(A_data[jA]);
          }
          
          row_sum = fabs( row_sum / diag );
          
          /* compute row entries of S */
          S_data[A_i[i]] = 0;
          if ( (row_sum > max_row_sum) && (max_row_sum < 1.0) )
          {
             for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
             {
                S_data[jA] = 0;
             }
          }
          else
          {
             for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
             {
                S_data[jA] = 0;
                if ( fabs(A_data[jA]) >= strength_threshold*row_scale )
                {            
                   S_data[jA] = -1;
                }
             }
          }
       }
   }
   else  /* mode = 0 (default) */
   {  
       for (i = 0; i < num_variables; i ++)
       {
           diag = A_data[A_i[i]];

           /* compute scaling factor */
           row_scale = 0.0;
           row_sum   = diag;
           if (diag < 0)
           {
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 row_scale = fsls_max(row_scale, A_data[jA]);
                 row_sum += A_data[jA];
              }
           }
           else /* diag >= 0 */
           {
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 row_scale = fsls_min(row_scale, A_data[jA]);
                 row_sum += A_data[jA];
              }
           }
           
           row_sum = fabs( row_sum / diag );

           /* compute row entries of S */
           S_data[A_i[i]] = 0;
           if ( (row_sum > max_row_sum) && (max_row_sum < 1.0) )
           {
              for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
              {
                 S_data[jA] = 0;
              }
           }
           else
           {
              if (diag < 0) 
              { 
                 for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
                 {
                    S_data[jA] = 0;
                    if ( A_data[jA] >= strength_threshold*row_scale )
                    {
                       S_data[jA] = -1;
                    }
                 }
              }
              else /* diag >= 0 */
              {
                 for (jA = A_i[i]+1; jA < A_i[i+1]; jA ++)
                 {
                    S_data[jA] = 0;
                    if ( A_data[jA] <= strength_threshold*row_scale )
                    {
                       S_data[jA] = -1;
                    }
                 }
              }
           }
       }
   }

  /*-----------------------------------------------------------------
   * "Compress" the strength matrix.
   * NOTE: S has *NO DIAGONAL ELEMENT* on any row.
   * NOTE: This "compression" section of code may be removed, and
   * coarsening will still be done correctly.  However, the routine
   * that builds interpolation would have to be modified first.
   *----------------------------------------------------------------*/
   jS = 0;
   for (i = 0; i < num_variables; i ++)
   {
      S_i[i] = jS;
      for (jA = A_i[i]; jA < A_i[i+1]; jA ++)
      {
         if (S_data[jA])
         {
            S_j[jS]    = S_j[jA];
            S_data[jS] = S_data[jA];
            jS ++;
         }
      }
   }
   S_i[num_variables] = jS;
   fsls_ICSRMatrixNumNonzeros(S) = jS;

   *S_ptr = S;

   return (ierr);
}
