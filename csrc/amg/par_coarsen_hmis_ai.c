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
jx_PAMGCoarsenHMISAI( jx_ParCSRMatrix  *par_S,
                    jx_ParCSRMatrix  *par_matrix,
                    JX_Real           *AI_measure,
                    JX_Int               CF_init,
                    JX_Int               measure_type,
                    JX_Int               debug_flag,
                    JX_Int             **CF_marker_ptr )
{
   JX_Int ierr = 0;

  /*--------------------------------------------------------
   * Perform Ruge coarsening followed by PMIS coarsening
   *--------------------------------------------------------*/

   ierr += jx_PAMGCoarsenRugeAI (par_S, par_matrix, AI_measure, CF_init, measure_type, 10, debug_flag, CF_marker_ptr);

   ierr += jx_PAMGCoarsenPMISAI (par_S, par_matrix, AI_measure, 1, debug_flag, CF_marker_ptr);

   return (ierr);
}
