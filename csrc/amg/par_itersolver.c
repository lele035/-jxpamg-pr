//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_itersolver.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_CoarsestIterativeMethod
 * \brief One iteration for some iterative method.
 * \date 2011/09/03
 */
JX_Int  
jx_CoarsestIterativeMethod( jx_ParCSRMatrix *par_matrix,
                            jx_ParVector    *par_rhs,
                            JX_Int              solverid,
                            JX_Real           weight,
                            JX_Real           omega,
                            jx_ParVector    *par_app,
                            jx_ParVector    *Vtemp )
{
   JX_Int relax_error = 0;
   
  /*-----------------------------------------------------------------------------
   * Switch statement to direct control based on relax_type:
   *     relax_type = 0 -> Jacobi
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
   *-----------------------------------------------------------------------------*/
   
   switch (solverid)
   {
      case 0:
      {
         relax_error = jx_PAMGRelax0(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;

      case 1:
      {
         relax_error = jx_PAMGRelax1(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;

      case 2:
      {
         relax_error = jx_PAMGRelax2(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;

      case 3:
      {
         relax_error = jx_PAMGRelax3(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;

      case 4:
      {
         relax_error = jx_PAMGRelax4(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;

      case 5:
      {
         relax_error = jx_PAMGRelax5(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;

      case 6:
      {
         relax_error = jx_PAMGRelax6(par_matrix, par_rhs, NULL, 0, weight, omega, par_app, Vtemp);
      }
      break;
   }

   return(relax_error); 
}
