//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_diagscale.h -- head files for diagscale preconditioner
 *  Date: 2013/10/27
 */ 

#ifndef JX_DIAGSCALE_HEADER
#define JX_DIAGSCALE_HEADER
 
#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif

#ifndef JX_MV_HEADER 
#include "jx_mv.h"
#endif

/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/

/* csrc/diagscale/diagscale.c */
JX_Int 
JX_DiagScaleSetup( JX_Solver        solver, 
                   JX_ParCSRMatrix  par_matrix );
JX_Int
jx_DiagScaleSetup( void             *solver, 
                   jx_ParCSRMatrix  *par_matrix );
JX_Int 
JX_DiagScalePrecond( JX_Solver       solver,
                     JX_ParCSRMatrix par_matrix,
                     JX_ParVector    par_rhs,
                     JX_ParVector    par_app  );
JX_Int
jx_DiagScalePrecond( void            *solver,
                     jx_ParCSRMatrix *par_matrix,
                     jx_ParVector    *par_rhs,
                     jx_ParVector    *par_app  );

#endif
