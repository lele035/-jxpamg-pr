//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGRelax
 * \brief Relaxation phase of PAMG.
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelaxAI( jx_ParCSRMatrix *par_matrix,
                jx_ParVector    *par_rhs,
                JX_Int             *cf_marker,
                JX_Int             *relax_marker,
                JX_Int              relax_type,
                JX_Int              relax_points,
                JX_Real           relax_weight,
                JX_Real           omega,
                jx_ParVector    *par_app,
                jx_ParVector    *Vtemp )
{
   JX_Int relax_error = 0;

  /*-----------------------------------------------------------------------------
   * Switch statement to direct control based on relax_type:
   *     relax_type = 0 -> Jacobi or CF-Jacobi
   *     relax_type = 1 -> Gauss-Seidel <--- very slow, sequential
   *     relax_type = 2 -> Gauss_Seidel: interior points in parallel ,
   *			 	   	  boundary sequential 
   *     relax_type = 3 -> hybrid: SOR-J mix off-processor, SOR on-processor
   *     		    with outer relaxation parameters (forward solve)
   *     relax_type = 4 -> hybrid: SOR-J mix off-processor, SOR on-processor
   *     		    with outer relaxation parameters (backward solve)
   *     relax_type = 5 -> hybrid: GS-J mix off-processor, chaotic GS on-node
   *     relax_type = 6 -> hybrid: SSOR-J mix off-processor, SSOR on-processor
   *     		    with outer relaxation parameters 
   *     relax_type = 7 -> Jacobi (uses Matvec), only needed in CGNR
   *     relax_type = 9 -> Direct Solve
   *-----------------------------------------------------------------------------*/
  
   if (relax_type == 0)
   {
      relax_error = jx_PAMGRelaxAI0(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 1)
   {
      relax_error = jx_PAMGRelaxAI1(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 2)
   {
      relax_error = jx_PAMGRelaxAI2(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 3)
   {
      relax_error = jx_PAMGRelaxAI3(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 4)
   {
      relax_error = jx_PAMGRelaxAI4(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 5)
   {
      relax_error = jx_PAMGRelaxAI5(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 6)
   {
      relax_error = jx_PAMGRelaxAI6(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }
   else if (relax_type == 7)
   {
      relax_error = jx_PAMGRelaxAI7(par_matrix, par_rhs, cf_marker, relax_marker, relax_points,
                                  relax_weight, omega, par_app, Vtemp);
   }

   return(relax_error); 
}
