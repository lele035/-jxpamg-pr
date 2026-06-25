//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_coarsen_hmis.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGCoarsenHMIS
 * \brief HMIS (RS + PMIS) coarsening routine.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGCoarsenXML( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Real           *AI_measure,
                    JX_Int               measure_type,
                    JX_Int               coarsen_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr )
{
   JX_Int ierr = 0;

  /*----------------------------------------------------------------------------
   * Perform AI-proir coarsening: 
   * 1st phase: coarsening on AI-parts
   *            using high operator complexity strategy, CLJP, RS0/3, FALGOUT.
   * 2nd phase: coarsening on ESS-parts
   *            using low operator complexity strategy, PMIS, HMIS. 
   *----------------------------------------------------------------------------*/

   if (coarsen_type == 908) {
      ierr += jx_PAMGCoarsenAI (par_S, par_matrix, AI_measure, 0, debug_flag, CF_marker_ptr);
   } else if (coarsen_type == 918) {
      ierr += jx_PAMGCoarsenRugeAI (par_S, par_matrix, AI_measure, 0, measure_type, 11, debug_flag, CF_marker_ptr);
   } else if (coarsen_type == 928) {
      ierr += jx_PAMGCoarsenRugeAI (par_S, par_matrix, AI_measure, 0, measure_type, 1, debug_flag, CF_marker_ptr);
   } else if (coarsen_type == 938) {
      ierr += jx_PAMGCoarsenRugeAI (par_S, par_matrix, AI_measure, 0, measure_type, 3, debug_flag, CF_marker_ptr);
   } else if (coarsen_type == 968) {
      ierr += jx_PAMGCoarsenFalgoutAI (par_S, par_matrix, AI_measure, measure_type, debug_flag, CF_marker_ptr);
   }
  
   jx_printf("=========== jx_PAMGCoarsenPMISAI ....\n"); 
   ierr += jx_PAMGCoarsenPMISAI (par_S, par_matrix, AI_measure, 9, debug_flag, CF_marker_ptr);
   //ierr += jx_PAMGCoarsenHMISAI(par_S, par_matrix, AI_measure, 9, measure_type, debug_flag, CF_marker_ptr);

   return (ierr);
}
