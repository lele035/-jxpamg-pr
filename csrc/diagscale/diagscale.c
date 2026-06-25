//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  diagscale.c
 *  Date: 2013/10/27
 */ 

#include "jx_diagscale.h"


/*!
 * \fn JX_Int JX_DiagScaleSetup
 * \date 2013/10/27
 */
JX_Int 
JX_DiagScaleSetup( JX_Solver        solver, 
                   JX_ParCSRMatrix  par_matrix )
{
   return( jx_DiagScaleSetup( (void *) solver, (jx_ParCSRMatrix *) par_matrix ) );
}

/*!
 * \fn JX_Int jx_DiagScaleSetup
 * \brief Setup phase of DiagScale preconditioner.
 * \date 2013/10/27
 */
JX_Int
jx_DiagScaleSetup( void             *solver, 
                   jx_ParCSRMatrix  *par_matrix )
{
   return 0;
}

/*!
 * \fn JX_Int JX_DiagScalePrecond
 * \brief Solve phase of DiagScale preconditioner.
 * \date 2013/10/27
 */
JX_Int 
JX_DiagScalePrecond( JX_Solver       solver,
                     JX_ParCSRMatrix par_matrix,
                     JX_ParVector    par_rhs,
                     JX_ParVector    par_app  )
{
   return( jx_DiagScalePrecond( (void *) solver,
                                (jx_ParCSRMatrix *) par_matrix,
                                (jx_ParVector *) par_rhs,
                                (jx_ParVector *) par_app ) );
}

/*!
 * \fn JX_Int jx_DiagScalePrecond
 * \brief DiagScale preconditioner.
 * \param solver pointer to NULL
 * \param par_matrix pointer to the coefficient matrix 
 * \param par_rhs pointer to the right hand side vector
 * \param par_app pointer to the approxamation vector
 * \date 2013/10/27
 */
JX_Int
jx_DiagScalePrecond( void            *solver,
                     jx_ParCSRMatrix *par_matrix,
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app  )
{
   JX_Int    *A_i     = jx_CSRMatrixI(jx_ParCSRMatrixDiag(par_matrix));
   JX_Real *A_data  = jx_CSRMatrixData(jx_ParCSRMatrixDiag(par_matrix));
   JX_Real *x_data  = jx_VectorData(jx_ParVectorLocalVector(par_app));
   JX_Real *y_data  = jx_VectorData(jx_ParVectorLocalVector(par_rhs));
   JX_Int  local_size = jx_VectorSize(jx_ParVectorLocalVector(par_app));
   JX_Int  i, ierr = 0;
   
   for (i = 0; i < local_size; i ++)
   {
      x_data[i] = y_data[i] / A_data[A_i[i]];
   }
   
   return ierr;
}
