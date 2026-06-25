//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  pbicgstab.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"
#include "jx_krylov.h"

/*!
 * \fn JX_Int JX_ParCSRBiCGSTABCreate
 */ 
JX_Int
JX_ParCSRBiCGSTABCreate( MPI_Comm comm, JX_Solver *solver )
{
   jx_BiCGSTABFunctions *bicgstab_functions =
      jx_BiCGSTABFunctionsCreate(
         jx_ParKrylovCreateVector,
         jx_ParKrylovDestroyVector, 
         jx_ParKrylovMatvecCreate,
         jx_ParKrylovMatvec, 
         jx_ParKrylovMatvecDestroy,
         jx_ParKrylovInnerProd, 
         jx_ParKrylovCopyVector,
         jx_ParKrylovClearVector,
         jx_ParKrylovScaleVector, 
         jx_ParKrylovAxpy,
         jx_ParKrylovCommInfo,
         jx_ParKrylovIdentitySetup, 
         jx_ParKrylovIdentity );

   *solver = ( (JX_Solver) jx_BiCGSTABCreate( bicgstab_functions) );
   
   if (!solver) 
   {
      jx_error_in_arg(2);
   }
    
   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ParCSRBiCGSTABDestroy
 */ 
JX_Int 
JX_ParCSRBiCGSTABDestroy( JX_Solver solver )
{
   return( jx_BiCGSTABDestroy( (void *) solver ) );
}

/*-------------------------------------------------------------------------*
 * JX_BiCGSTABCreate does not exist.  Call the appropriate function which  *
 * also specifies the vector type, e.g. JX_ParCSRBiCGSTABCreate            *
 *-------------------------------------------------------------------------*/


/*!
 * \fn JX_Int JX_BiCGSTABDestroy
 */ 
JX_Int 
JX_BiCGSTABDestroy( JX_Solver solver )
{
   return( jx_BiCGSTABDestroy( (void *) solver ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetup
 */ 
JX_Int 
JX_BiCGSTABSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX )
{
   return( jx_BiCGSTABSetup( (void *)solver, (void *)matA, (void *)vecB, (void *)vecX ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSolve
 */ 
JX_Int 
JX_BiCGSTABSolve( JX_Solver solver,
                  JX_Matrix preOperator,
                  JX_Matrix matA,
                  JX_Vector vecB,
                  JX_Vector vecX  )
{
   return( jx_BiCGSTABSolve( (void *)solver, (void *)preOperator, (void *)matA, (void *)vecB, (void *)vecX ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetPreOperator
 * \brief Set the preconditioner for BiCGSTAB
 * \author peghoty
 * \date 2011/09/04
 */ 
JX_Int
JX_BiCGSTABSetPreOperator( JX_Solver solver, JX_ParCSRMatrix preOperator )
{
   return( jx_BiCGSTABSetPreOperator( (void *) solver, (void *) preOperator ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetTol
 */
JX_Int
JX_BiCGSTABSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_BiCGSTABSetTol( (void *) solver, tol ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetAbsoluteTol
 */
JX_Int
JX_BiCGSTABSetAbsoluteTol( JX_Solver solver, JX_Real a_tol )
{
   return( jx_BiCGSTABSetAbsoluteTol( (void *) solver, a_tol ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetConvergenceFactorTol
 */
JX_Int
JX_BiCGSTABSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol )
{
   return( jx_BiCGSTABSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetMinIter
 */
JX_Int
JX_BiCGSTABSetMinIter( JX_Solver solver, JX_Int min_iter )
{
   return( jx_BiCGSTABSetMinIter( (void *) solver, min_iter ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetMaxIter
 */
JX_Int
JX_BiCGSTABSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_BiCGSTABSetMaxIter( (void *) solver, max_iter ) );
}


/*!
 * \fn JX_Int JX_BiCGSTABSetConvCriteria
 * \brief Set the parameter 'conv_criteria'.
 * \author peghoty
 * \date 2010/03/20
 */ 
JX_Int
JX_BiCGSTABSetConvCriteria( JX_Solver solver, JX_Int conv_criteria )
{
   return( jx_BiCGSTABSetConvCriteria( (void *) solver, conv_criteria ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetConvFacThreshold
 * \brief Set the parameter 'convfac_threshold'.
 * \author peghoty
 * \date 2010/06/22
 */ 
JX_Int
JX_BiCGSTABSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold )
{
   return( jx_BiCGSTABSetConvFacThreshold( (void *) solver, convfac_threshold) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetPrecond
 */ 
JX_Int
JX_BiCGSTABSetPrecond( JX_Solver         solver,
                       JX_PtrToSolverFcn precond,
                       JX_PtrToSolverFcn precond_setup,
                       JX_Solver         precond_solver )
{
   return( jx_BiCGSTABSetPrecond( (void *) solver,
                                  precond, 
                                  precond_setup,
                                  (void *) precond_solver ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABGetPrecond
 */ 
JX_Int
JX_BiCGSTABGetPrecond( JX_Solver solver, JX_Solver *precond_data_ptr )
{
   return( jx_BiCGSTABGetPrecond( (void *) solver, (JX_Solver *) precond_data_ptr ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetLogging
 */ 
JX_Int
JX_BiCGSTABSetLogging( JX_Solver solver, JX_Int logging )
{
   return( jx_BiCGSTABSetLogging( (void *) solver, logging ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABSetPrintLevel
 */ 
JX_Int
JX_BiCGSTABSetPrintLevel( JX_Solver solver, JX_Int print_level )
{
   return( jx_BiCGSTABSetPrintLevel( (void *) solver, print_level ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABGetNumIterations
 */
JX_Int
JX_BiCGSTABGetNumIterations( JX_Solver  solver, JX_Int *num_iterations )
{
   return( jx_BiCGSTABGetNumIterations( (void *) solver, num_iterations ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABGetFinalRelativeResidualNorm
 */
JX_Int
JX_BiCGSTABGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm )
{
   return( jx_BiCGSTABGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*!
 * \fn JX_Int JX_BiCGSTABGetResidual
 */
JX_Int
JX_BiCGSTABGetResidual( JX_Solver solver, void **residual  )
{
   return( jx_BiCGSTABGetResidual( (void *) solver, residual ) );
}

/*!
 * \fn jx_BiCGSTABFunctions *jx_BiCGSTABFunctionsCreate
 */
jx_BiCGSTABFunctions *
jx_BiCGSTABFunctionsCreate(
   void * (*CreateVector)  ( void *vvector ),
   JX_Int    (*DestroyVector) ( void *vvector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y ),
   JX_Int    (*MatvecDestroy) ( void *matvec_data ),
   JX_Real (*InnerProd)     ( void *x, void *y ),
   JX_Int    (*CopyVector)    ( void *x, void *y ),
   JX_Int    (*ClearVector)   ( void *x ),
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x ),
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y ),
   JX_Int    (*CommInfo)      ( void *A, JX_Int *my_id, JX_Int *num_procs ),
   JX_Int    (*PrecondSetup)  ( void *vdata, void *A ),
   JX_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   jx_BiCGSTABFunctions *bicgstab_functions;
   bicgstab_functions = (jx_BiCGSTABFunctions *)jx_CTAlloc(jx_BiCGSTABFunctions, 1);

   bicgstab_functions->CreateVector = CreateVector;
   bicgstab_functions->DestroyVector = DestroyVector;
   bicgstab_functions->MatvecCreate = MatvecCreate;
   bicgstab_functions->Matvec = Matvec;
   bicgstab_functions->MatvecDestroy = MatvecDestroy;
   bicgstab_functions->InnerProd = InnerProd;
   bicgstab_functions->CopyVector = CopyVector;
   bicgstab_functions->ClearVector = ClearVector;
   bicgstab_functions->ScaleVector = ScaleVector;
   bicgstab_functions->Axpy = Axpy;
   bicgstab_functions->CommInfo = CommInfo;
   bicgstab_functions->precond_setup = PrecondSetup;
   bicgstab_functions->precond = Precond;

   return bicgstab_functions;
}

/*!
 * \fn void *jx_BiCGSTABCreate
 */
void *
jx_BiCGSTABCreate( jx_BiCGSTABFunctions * bicgstab_functions )
{
   jx_BiCGSTABData *bicgstab_data;
 
   bicgstab_data = jx_CTAlloc( jx_BiCGSTABData, 1);
 
   bicgstab_data->functions = bicgstab_functions;

   /* set defaults */
   (bicgstab_data -> tol)                = 1.0e-06;
   (bicgstab_data -> min_iter)           = 0;
   (bicgstab_data -> max_iter)           = 1000;
   (bicgstab_data -> conv_criteria)      = 0;    // peghoty 2010/03/20
   (bicgstab_data -> convfac_threshold)  = 1.0e4;  // peghoty 2010/06/22
   (bicgstab_data -> a_tol)              = 0.0;  
   (bicgstab_data -> cf_tol)             = 0.0;  // peghoty 2010/03/20
   (bicgstab_data -> precond_data)       = NULL;
   (bicgstab_data -> logging)            = 0;
   (bicgstab_data -> print_level)        = 0;
   (bicgstab_data -> B)                  = NULL; // peghoty 2011/10/08
   (bicgstab_data -> p)                  = NULL;
   (bicgstab_data -> q)                  = NULL;
   (bicgstab_data -> r)                  = NULL;
   (bicgstab_data -> r0)                 = NULL;
   (bicgstab_data -> s)                  = NULL;    
   (bicgstab_data -> v)                  = NULL;
   (bicgstab_data -> matvec_data)        = NULL;
   (bicgstab_data -> norms)              = NULL;
   (bicgstab_data -> log_file_name)      = NULL;
 
   return (void *) bicgstab_data;
}

/*!
 * \fn JX_Int jx_BiCGSTABDestroy
 */
JX_Int
jx_BiCGSTABDestroy( void *bicgstab_vdata )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   jx_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   if (bicgstab_data)
   {
      if ( (bicgstab_data -> norms) != NULL )
      {
         jx_TFree(bicgstab_data -> norms);
      }
 
      (*(bicgstab_functions->MatvecDestroy))(bicgstab_data -> matvec_data);
 
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> r);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> r0);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> s);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> v);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> p);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> q);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> Xtmp); // peghoty 2010/03/20
 
      jx_TFree(bicgstab_data);
      jx_TFree(bicgstab_functions);
   }
 
   return(jx_error_flag);
}

/*!
 * \fn JX_Int jx_BiCGSTABSetup
 */
JX_Int
jx_BiCGSTABSetup( void *bicgstab_vdata, void *matA, void *vecB, void *vecX )
{
   jx_BiCGSTABData      *bicgstab_data      = bicgstab_vdata;
   jx_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   JX_Int max_iter = (bicgstab_data -> max_iter);   
   
   (bicgstab_data -> A) = matA;

  /*---------------------------------------------------
   * The arguments for NewVector are important to
   * maintain consistency between the setup and
   * compute phases of matvec and the preconditioner.
   *--------------------------------------------------*/
 
   if ((bicgstab_data -> p) == NULL)
   {
      (bicgstab_data -> p) = (*(bicgstab_functions->CreateVector))(vecB);
   }
   if ((bicgstab_data -> q) == NULL)
   {
      (bicgstab_data -> q) = (*(bicgstab_functions->CreateVector))(vecB);
   }
   if ((bicgstab_data -> r) == NULL)
   {
      (bicgstab_data -> r) = (*(bicgstab_functions->CreateVector))(vecB);
   }
   if ((bicgstab_data -> r0) == NULL)
   {
      (bicgstab_data -> r0) = (*(bicgstab_functions->CreateVector))(vecB);
   }
   if ((bicgstab_data -> s) == NULL)
   {
      (bicgstab_data -> s) = (*(bicgstab_functions->CreateVector))(vecB);
   }
   if ((bicgstab_data -> v) == NULL)
   {
      (bicgstab_data -> v) = (*(bicgstab_functions->CreateVector))(vecB);
   }

   /* peghoty 2010/03/20 */
   if ((bicgstab_data -> Xtmp) == NULL)
   {
      (bicgstab_data -> Xtmp) = (*(bicgstab_functions->CreateVector))(vecB); 
   }     
 
   if ((bicgstab_data -> matvec_data) == NULL)
   { 
      (bicgstab_data -> matvec_data) = (*(bicgstab_functions->MatvecCreate))(matA, vecX);
   }

  /*------------------------------------------------------------------------
   * Allocate space for log info
   *-----------------------------------------------------------------------*/
   if ( (bicgstab_data->logging) > 0 || (bicgstab_data->print_level) > 0 )
   {
      if ((bicgstab_data -> norms) == NULL)
      {
         (bicgstab_data -> norms) = jx_CTAlloc(JX_Real, max_iter + 1);
      }
   }
   if ((bicgstab_data -> print_level) > 0)
   {
      if ((bicgstab_data -> log_file_name) == NULL)
      {
         (bicgstab_data -> log_file_name) = "bicgstab.out.log";
      }
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetPreOperator
 * \brief Set the preconditioner operator for BiCGSTAB.
 * \author peghoty
 * \date 2011/09/04
 */ 
JX_Int
jx_BiCGSTABSetPreOperator( void *bicgstab_vdata, void *preOperator )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> B) = preOperator;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetTol
 */  
JX_Int
jx_BiCGSTABSetTol( void *bicgstab_vdata, JX_Real tol )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> tol) = tol;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetAbsoluteTol
 */ 
JX_Int
jx_BiCGSTABSetAbsoluteTol( void *bicgstab_vdata, JX_Real a_tol )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> a_tol) = a_tol;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetConvergenceFactorTol
 */ 
JX_Int
jx_BiCGSTABSetConvergenceFactorTol( void *bicgstab_vdata, JX_Real cf_tol )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> cf_tol) = cf_tol;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetMinIter
 */ 
JX_Int
jx_BiCGSTABSetMinIter( void *bicgstab_vdata, JX_Int min_iter )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> min_iter) = min_iter;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetMaxIter
 */ 
JX_Int
jx_BiCGSTABSetMaxIter( void *bicgstab_vdata, JX_Int max_iter )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> max_iter) = max_iter;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetConvCriteria
 * \brief Set the parameter 'conv_criteria'.
 * \author peghoty
 * \date 2010/03/20
 */  
JX_Int
jx_BiCGSTABSetConvCriteria( void *bicgstab_vdata, JX_Int conv_criteria )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> conv_criteria) = conv_criteria;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetConvFacThreshold
 * \brief Set the parameter 'convfac_threshold'.
 * \author peghoty
 * \date 2010/06/22
 */
JX_Int
jx_BiCGSTABSetConvFacThreshold( void *bicgstab_vdata, JX_Real convfac_threshold )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> convfac_threshold) = convfac_threshold;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetPrecond
 */ 
JX_Int
jx_BiCGSTABSetPrecond( void  *bicgstab_vdata,
                       JX_Int  (*precond)(),
                       JX_Int  (*precond_setup)(),
                       void  *precond_data  )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   jx_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   (bicgstab_functions -> precond)       = precond;
   (bicgstab_functions -> precond_setup) = precond_setup;
   (bicgstab_data -> precond_data)       = precond_data;
 
   return jx_error_flag;
}
 
/*!
 * \fn JX_Int jx_BiCGSTABGetPrecond
 */ 
JX_Int
jx_BiCGSTABGetPrecond( void *bicgstab_vdata, JX_Solver *precond_data_ptr )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   *precond_data_ptr = (JX_Solver)(bicgstab_data -> precond_data);
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetLogging
 */ 
JX_Int
jx_BiCGSTABSetLogging( void *bicgstab_vdata, JX_Int logging )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> logging) = logging;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABSetPrintLevel
 */ 
JX_Int
jx_BiCGSTABSetPrintLevel( void *bicgstab_vdata, JX_Int print_level )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   (bicgstab_data -> print_level) = print_level;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_BiCGSTABGetConverged
 */ 
JX_Int
jx_BiCGSTABGetConverged( void *bicgstab_vdata, JX_Int *converged )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   *converged = (bicgstab_data -> converged);
   return jx_error_flag;
}
 
/*!
 * \fn JX_Int jx_BiCGSTABGetNumIterations
 */ 
JX_Int
jx_BiCGSTABGetNumIterations( void *bicgstab_vdata, JX_Int *num_iterations )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   *num_iterations = (bicgstab_data -> num_iterations);
   return jx_error_flag;
}
 
/*!
 * \fn JX_Int jx_BiCGSTABGetFinalRelativeResidualNorm
 */ 
JX_Int
jx_BiCGSTABGetFinalRelativeResidualNorm( void *bicgstab_vdata, JX_Real *relative_residual_norm )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   return jx_error_flag;
} 

/*!
 * \fn JX_Int jx_BiCGSTABGetResidual
 */ 
JX_Int
jx_BiCGSTABGetResidual( void *bicgstab_vdata, void **residual )
{
   jx_BiCGSTABData *bicgstab_data = bicgstab_vdata;
   *residual = (bicgstab_data -> r);
   return jx_error_flag;
} 

/*!
 * \fn JX_Int jx_BiCGSTABSolve
 */ 
JX_Int
jx_BiCGSTABSolve( void *bicgstab_vdata,
                  void *preOperator, 
                  void *matA, 
                  void *vecB, 
                  void *vecX  )
{
   jx_BiCGSTABData      *bicgstab_data      = bicgstab_vdata;
   jx_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   JX_Int         min_iter      = (bicgstab_data -> min_iter);
   JX_Int         max_iter      = (bicgstab_data -> max_iter);
   JX_Int         conv_criteria = (bicgstab_data -> conv_criteria); // peghoty 2010/03/20
   JX_Real      convfac_threshold = (bicgstab_data -> convfac_threshold); // peghoty 2010/06/22
   JX_Real      r_tol         = (bicgstab_data -> tol);
   JX_Real      cf_tol        = (bicgstab_data -> cf_tol);
   void       *matvec_data   = (bicgstab_data -> matvec_data);
   JX_Real      a_tol         = (bicgstab_data -> a_tol);
  
   void       *vecR          = (bicgstab_data -> r);
   void       *vecR0         = (bicgstab_data -> r0);
   void       *vecS          = (bicgstab_data -> s);
   void       *vecV          = (bicgstab_data -> v);
   void       *vecP          = (bicgstab_data -> p);
   void       *vecQ          = (bicgstab_data -> q);
   void       *Xtmp          = (bicgstab_data -> Xtmp); // peghoty 2010/03/20

   JX_Int 	      (*precond)()   = (bicgstab_functions -> precond);
   JX_Int 	      *precond_data  = (bicgstab_data -> precond_data);

   /* logging variables */
   JX_Int        logging        = (bicgstab_data -> logging);
   JX_Int        print_level    = (bicgstab_data -> print_level);
   JX_Real    *norms          = (bicgstab_data -> norms);

//   JX_Int        ierr = 0;
   JX_Int        iter; 
   JX_Int        my_id, num_procs;
   JX_Real     alpha, beta, gamma, temp, res, r_norm, b_norm;
   JX_Real     epsilon = 0.0;
   JX_Real     epsmac = 1.e-128; 
   JX_Real     ieee_check = 0.;
   JX_Real     cf_ave_0 = 0.0;
   JX_Real     cf_ave_1 = 0.0;
   JX_Real     weight;
   JX_Real     r_norm_0;
   JX_Real     den_norm;
   
   /* peghoty 2010/03/20 */
   JX_Real     old_relative = 0.0;
   JX_Real     relative     = 0.0;
   JX_Real     normup       = 0.0;
   JX_Real     normdown     = 0.0;
   JX_Real     conv_factor  = 0.0;
     
   (bicgstab_data -> converged) = 0;

   (*(bicgstab_functions->CommInfo))(matA, &my_id, &num_procs);
   
   if (logging > 0 || print_level > 0)
   {
      norms = (bicgstab_data -> norms);
   }


   /* initialize work arrays */
   (*(bicgstab_functions->CopyVector))(vecB, vecR0);


   /* compute initial residual */
   (*(bicgstab_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR0);
   (*(bicgstab_functions->CopyVector))(vecR0, vecR);
   (*(bicgstab_functions->CopyVector))(vecR0, vecP);

   b_norm = sqrt((*(bicgstab_functions->InnerProd))(vecB, vecB));


   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm / b_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         jx_printf("\n\nERROR detected by JXPAMG ...  BEGIN\n");
         jx_printf("ERROR -- jx_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ...  END\n\n\n");
//         break;
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }


   res = (*(bicgstab_functions->InnerProd))(vecR0, vecR0);
   r_norm = sqrt(res);
   r_norm_0 = r_norm;
 
 
   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm / r_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         jx_printf("\n\nERROR detected by JXPAMG ...  BEGIN\n");
         jx_printf("ERROR -- jx_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ...  END\n\n\n");
//         break;
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   normdown = r_norm;

   if (logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if (print_level > 0 && my_id == 0)
      {
  	 jx_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            jx_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         jx_printf("Initial L2 norm of residual: %e\n\n", r_norm);
      }
   }
   
   
   iter = 0;


   /* peghoty 2010/03/20 */
   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= r_tol*|b| if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= r_tol*|r0| if |b| = 0 */
      den_norm = r_norm;
   }
   
   epsilon = jx_max(a_tol, r_tol*den_norm);
   
   if (conv_criteria) relative = 2*r_tol;  
     
   if (print_level > 0 && my_id == 0)
   {
      if (conv_criteria == 0)
      {
         if (b_norm > 0.0)
         {
             jx_printf("=============================================\n\n");
             jx_printf("Iters     resid.norm     conv.rate  rel.res.norm(wrt.b_norm)\n");
             jx_printf("-----    ------------    ---------- ------------------------\n");
         }
         else
         {
             jx_printf("=============================================\n\n");
             jx_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
             jx_printf("-----    ------------    ---------- ------------\n");
         }
      }
      else if (conv_criteria == 1 || conv_criteria == 11 || conv_criteria == 2)
      {
         if (conv_criteria == 1)
         {
            jx_printf("                                      Relative   \n");
            jx_printf(" Iters     Error Norm-1    Factor    Error Norm-1\n");
            jx_printf("-------    ------------   --------  -------------\n");
         }
         else if (conv_criteria == 11)
         {
            jx_printf("                  Relative Point-wise \n");
            jx_printf(" Iters   Factor      Error Norm-1     \n");
            jx_printf("-------  -------    --------------    \n");
         }
         else if (conv_criteria == 2)
         {
            jx_printf("                                      Relative   \n");
            jx_printf(" Iters     Error Norm-2    Factor    Error Norm-2\n");
            jx_printf("-------    ------------   --------  -------------\n");          
         }
      }      
   }

   (bicgstab_data -> num_iterations) = iter;
   
   if (b_norm > 0.0)
   {
      (bicgstab_data -> rel_residual_norm) = r_norm / b_norm;
   }   
      
   
  /*-----------------------------------------------------------------
   * The main 'while' loop 
   *---------------------------------------------------------------*/  
    
   while (iter < max_iter)
   {

        if (r_norm == 0.0)
        {
//	   ierr = 0;
	   return jx_error_flag;
	}
	
	if (conv_criteria == 1 || conv_criteria == 11 || conv_criteria == 2)
        {
           (*(bicgstab_functions->CopyVector))(vecX, Xtmp);
        }
	
	/* check for convergence, evaluate actual residual */
	if (conv_criteria == 0)
	{ 
	   if (r_norm <= epsilon && iter >= min_iter) 
           {
	      (*(bicgstab_functions->CopyVector))(vecB,vecR);
              (*(bicgstab_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
	      r_norm = sqrt((*(bicgstab_functions->InnerProd))(vecR, vecR));
	      if (r_norm <= epsilon)
              {
                 if (print_level > 0 && my_id == 0)
                 {
                    jx_printf("\n\n");
                    jx_printf(" Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (bicgstab_data -> converged) = 1;
                 break;
              }
	      else
	      {
	         (*(bicgstab_functions->CopyVector))(vecR, vecP);
	      }
	   }
	}
	else if (conv_criteria == 1 || conv_criteria == 11 || conv_criteria == 2)
        {           
	   if (relative <= r_tol && iter >= min_iter) 
           {
              /* Is this necessary? peghoty 2010/03/20 */
	      (*(bicgstab_functions->CopyVector))(vecB, vecR);
              (*(bicgstab_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
	      r_norm = sqrt((*(bicgstab_functions->InnerProd))(vecR, vecR));
	      if (r_norm <= epsilon) 
              {
                 if (print_level > 0 && my_id == 0)
                 {
                    jx_printf("\n\n");
                    jx_printf(" Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (bicgstab_data -> converged) = 1;
                 break;
              }
	      else
	      {
	         (*(bicgstab_functions->CopyVector))(vecR, vecP);
	      }
	   }
        }

	
       /*--------------------------------------------------------------------
        * Optional test to see if adequate progress is being made.
        * The average convergence factor is recorded and compared
        * against the tolerance 'cf_tol'. The weighting factor is
        * intended to pay more attention to the test when an accurate
        * estimate for average convergence factor is available.
        *--------------------------------------------------------------------*/
        if (cf_tol > 0.0)
        {
           cf_ave_0 = cf_ave_1;
           cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

           weight   = fabs(cf_ave_1 - cf_ave_0);
           weight   = weight / jx_max(cf_ave_1, cf_ave_0);
           weight   = 1.0 - weight;
           if (weight * cf_ave_1 > cf_tol) 
           {
              break;
           }
        }


        iter ++;
	
	(*(bicgstab_functions->ClearVector))(vecV);
		
        precond(precond_data, preOperator, vecP, vecV);
	        
        (*(bicgstab_functions->Matvec))(matvec_data, 1.0, matA, vecV, 0.0, vecQ);
	        
      	temp = (*(bicgstab_functions->InnerProd))(vecR0, vecQ);
	      	
      	if (fabs(temp) >= epsmac)
      	{
	   alpha = res / temp;
	}
	else
	{
	   jx_printf("BiCGSTAB broke down!! divide by near zero\n");
	   return (1);
	}
		
	(*(bicgstab_functions->Axpy))(alpha, vecV, vecX);
	(*(bicgstab_functions->Axpy))(-alpha, vecQ, vecR);
	
	(*(bicgstab_functions->ClearVector))(vecV);
	
        precond(precond_data, preOperator, vecR, vecV);
       
        (*(bicgstab_functions->Matvec))(matvec_data, 1.0, matA, vecV, 0.0, vecS);
        
      	gamma = (*(bicgstab_functions->InnerProd))(vecR, vecS) /
                (*(bicgstab_functions->InnerProd))(vecS, vecS);
          
	(*(bicgstab_functions->Axpy))(gamma, vecV, vecX);

	
	if (conv_criteria == 1)
	{
           old_relative = relative;
           (*(bicgstab_functions->Axpy))(-1.0, vecX, Xtmp);  // Xtmp = Xtmp - vecX
           normup   = jx_ParVectorNorm1(Xtmp);
           normdown = jx_ParVectorNorm1(vecX);
           if (normdown < EPS)
           {
              relative = normup;
           }
           else
           {
              relative = normup / normdown;
           }  
           conv_factor = relative / old_relative;
	}
	else if (conv_criteria == 11)
	{
            old_relative = relative;
            (*(bicgstab_functions->Axpy))(-1.0, vecX, Xtmp);  // Xtmp = Xtmp - vecX
            relative = jx_ParVectorPointWiseRelNorm1(Xtmp, vecX);
            conv_factor = relative / old_relative;
	}
	else if (conv_criteria == 2)
	{
            old_relative = relative;
            (*(bicgstab_functions->Axpy))(-1.0, vecX, Xtmp);  // Xtmp = Xtmp - vecX
            normup = jx_ParVectorNorm2(Xtmp);
            normdown = jx_ParVectorNorm2(vecX);
            if (normdown < EPS)
            {
               relative = normup;
            }
            else
            {
               relative = normup / normdown;
            }
            conv_factor = relative / old_relative;
	}
	

	(*(bicgstab_functions->Axpy))(-gamma, vecS, vecR);
      	if (fabs(res) >= epsmac)
        {
           beta = 1.0 / res;
        }
	else
	{
	   jx_printf("BiCGSTAB broke down!! res = 0 \n");
	   return(2);
	}
        res = (*(bicgstab_functions->InnerProd))(vecR0, vecR);
        beta *= res;    
	(*(bicgstab_functions->Axpy))(-gamma, vecQ, vecP);
      	if (fabs(gamma) >= epsmac)
      	{
           (*(bicgstab_functions->ScaleVector))((beta*alpha / gamma), vecP);
        }
	else
	{
	   jx_printf("BiCGSTAB broke down!! gamma = 0 \n");
	   return (3);
	}
	(*(bicgstab_functions->Axpy))(1.0, vecR, vecP);

	r_norm = sqrt((*(bicgstab_functions->InnerProd))(vecR, vecR));

        normup = r_norm;
        conv_factor = normup / normdown;
        normdown = normup;
	
	if (logging > 0 || print_level > 0)
	{
	   norms[iter] = r_norm;
	}


        if (print_level > 0 && my_id == 0)
	{
           if (conv_criteria == 0)
           {
              if (b_norm > 0.0)
              {
                 jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
			 norms[iter]/norms[iter-1], norms[iter]/b_norm);
              }
              else
              {
                 jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
		         norms[iter]/norms[iter-1], norms[iter]/r_norm_0);
	      }
	   }
           if (conv_criteria == 1 || conv_criteria == 2)
           {
              if (iter == 1)
              {
                 jx_printf("   %3d     %e             %e \n", iter, normup, relative);
              }
              else
              {
                 jx_printf("   %3d     %e   %6f  %e \n", iter, normup, conv_factor, relative);
              }
           }
           if (conv_criteria == 11)
           {
              if (iter == 1)
              {
                 jx_printf("   %3d               %e \n", iter, relative);
              }
              else
              {
                 jx_printf("   %3d   %6f    %e \n", iter, conv_factor, relative);
              }
           }
	}

        if (iter > 1 && conv_factor > convfac_threshold)
        {
           if (my_id == 0 && print_level > 0)
           {
              jx_printf("\n  Warning: iteration has terminated because the\n");
              jx_printf("           current conv_factor > convfac_threshold(%lf) !!\n\n",convfac_threshold);
           }
           break;
        }

   }

   (bicgstab_data -> num_iterations) = iter;
   
   if (conv_criteria == 0)
   {
      if (b_norm > 0.0)
      {
         (bicgstab_data -> rel_residual_norm) = r_norm / b_norm;
      }
      if (b_norm == 0.0)
      {
         (bicgstab_data -> rel_residual_norm) = r_norm / r_norm_0;
      }
   }
   else
   {
      (bicgstab_data -> rel_residual_norm) = relative;
   }
   
#if 0
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf(" Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
      jx_error(JX_ERROR_CONV);
   }
#endif

   return jx_error_flag;
}
