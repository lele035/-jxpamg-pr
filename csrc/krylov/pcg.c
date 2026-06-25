//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  pcg.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"
#include "jx_krylov.h"

/*!
 * \fn JX_Int JX_ParCSRPCGCreate
 */ 
JX_Int
JX_ParCSRPCGCreate( MPI_Comm comm, JX_Solver *solver )
{
   /* The function names with a PCG in them are in
      parcsr_ls/pcg_par.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., jx_struct_Free, to reduce the
      chance of name conflicts. */
   jx_PCGFunctions * pcg_functions =
      jx_PCGFunctionsCreate(
         jx_CAlloc, 
         jx_ParKrylovFree, 
         jx_ParKrylovCommInfo,
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
         jx_ParKrylovIdentitySetup, 
         jx_ParKrylovIdentity );

   *solver = ( (JX_Solver) jx_PCGCreate( pcg_functions ) );
   
   if (!solver) 
   {
      jx_error_in_arg(2);
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PCGFunctionsCreate
 */
jx_PCGFunctions *
jx_PCGFunctionsCreate(
   char * (*CAlloc)        ( size_t count, size_t elt_size ),
   JX_Int    (*Free)          ( char *ptr ),
   JX_Int    (*CommInfo)      ( void *A, JX_Int *my_id, JX_Int *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   JX_Int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y ),
   JX_Int    (*MatvecDestroy) ( void *matvec_data ),
   JX_Real (*InnerProd)     ( void *x, void *y ),
   JX_Int    (*CopyVector)    ( void *x, void *y ),
   JX_Int    (*ClearVector)   ( void *x ),
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x ),
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y ),
   JX_Int    (*PrecondSetup)  ( void *vdata, void *A ),
   JX_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   jx_PCGFunctions *pcg_functions;
   pcg_functions = (jx_PCGFunctions *)CAlloc(1, sizeof(jx_PCGFunctions));

   pcg_functions->CAlloc = CAlloc;
   pcg_functions->Free = Free;
   pcg_functions->CommInfo = CommInfo;
   pcg_functions->CreateVector = CreateVector;
   pcg_functions->DestroyVector = DestroyVector;
   pcg_functions->MatvecCreate = MatvecCreate;
   pcg_functions->Matvec = Matvec;
   pcg_functions->MatvecDestroy = MatvecDestroy;
   pcg_functions->InnerProd = InnerProd;
   pcg_functions->CopyVector = CopyVector;
   pcg_functions->ClearVector = ClearVector;
   pcg_functions->ScaleVector = ScaleVector;
   pcg_functions->Axpy = Axpy;
   /* default preconditioner must be set here but can be changed later... */
   pcg_functions->precond_setup = PrecondSetup;
   pcg_functions->precond       = Precond;

   return pcg_functions;
}

/*!
 * \fn JX_Int jx_PCGCreate
 */
void *
jx_PCGCreate( jx_PCGFunctions *pcg_functions )
{
   jx_PCGData *pcg_data;

   pcg_data = jx_CTAllocF(jx_PCGData, 1, pcg_functions);

   pcg_data -> functions = pcg_functions;

   /* set defaults */
   (pcg_data -> tol)                = 1.0e-06;
   (pcg_data -> atolf)              = 0.0;
   (pcg_data -> cf_tol)             = 0.0;
   (pcg_data -> a_tol)              = 0.0;
   (pcg_data -> max_iter)           = 1000;
   (pcg_data -> two_norm)           = 0;
   (pcg_data -> rel_change)         = 0;
   (pcg_data -> recompute_residual) = 0;
   (pcg_data -> stop_crit)          = 0;
   (pcg_data -> conv_criteria)      = 0;    // peghoty 2010/06/23
   (pcg_data -> convfac_threshold)  = 1.0e4;  // peghoty 2010/06/23
   (pcg_data -> converged)          = 0;
   (pcg_data -> owns_matvec_data )  = 1;
   (pcg_data -> matvec_data)        = NULL;
   (pcg_data -> precond_data)       = NULL;
   (pcg_data -> print_level)        = 0;
   (pcg_data -> logging)            = 0;
   (pcg_data -> norms)              = NULL;
   (pcg_data -> rel_norms)          = NULL;
   (pcg_data -> B)                  = NULL; // peghoty 2011/10/08
   (pcg_data -> p)                  = NULL;
   (pcg_data -> s)                  = NULL;
   (pcg_data -> r)                  = NULL;

   return (void *) pcg_data;
}

/*!
 * \fn JX_Int JX_PCGSetup
 */
JX_Int 
JX_PCGSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX )
{
   return( jx_PCGSetup( solver, matA, vecB, vecX ) );
}

/*!
 * \fn JX_Int JX_PCGSolve
 */
JX_Int 
JX_PCGSolve( JX_Solver  solver, 
             JX_Matrix  preOperator,
             JX_Matrix  matA, 
             JX_Vector  vecB, 
             JX_Vector  vecX )
{
   return( jx_PCGSolve( (void *) solver, 
                        (void *) preOperator,
                        (void *) matA, 
                        (void *) vecB, 
                        (void *) vecX ) );
}

/*!
 * \fn JX_Int JX_PCGSetPreOperator
 * \brief Set Preconditioner operator.
 * \author peghoty
 * \date  2011/09/04 
 */
JX_Int
JX_PCGSetPreOperator( JX_Solver solver, JX_ParCSRMatrix preOperator )
{
   return( jx_PCGSetPreOperator( (void *) solver, (void *) preOperator ) );
}

/*!
 * \fn JX_Int JX_PCGSetXXX
 */

JX_Int
JX_PCGSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_PCGSetTol( (void *) solver, tol ) );
}

JX_Int
JX_PCGGetTol( JX_Solver solver, JX_Real *tol )
{
   return( jx_PCGGetTol( (void *) solver, tol ) );
}

JX_Int
JX_PCGSetAbsoluteTol( JX_Solver solver, JX_Real a_tol )
{
   return( jx_PCGSetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_PCGGetAbsoluteTol( JX_Solver solver, JX_Real *a_tol )
{
   return( jx_PCGGetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_PCGSetAbsoluteTolFactor( JX_Solver solver, JX_Real abstolf )
{
   return( jx_PCGSetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

JX_Int
JX_PCGGetAbsoluteTolFactor( JX_Solver solver, JX_Real *abstolf )
{
   return( jx_PCGGetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

JX_Int
JX_PCGSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol )
{
   return jx_PCGSetConvergenceFactorTol( (void *) solver, cf_tol );
}

JX_Int
JX_PCGGetConvergenceFactorTol( JX_Solver solver, JX_Real * cf_tol )
{
   return jx_PCGGetConvergenceFactorTol( (void *) solver, cf_tol );
}

JX_Int
JX_PCGSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_PCGSetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_PCGGetMaxIter( JX_Solver solver, JX_Int *max_iter )
{
   return( jx_PCGGetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_PCGSetStopCrit( JX_Solver solver, JX_Int stop_crit )
{
   return( jx_PCGSetStopCrit( (void *) solver, stop_crit ) );
}

JX_Int
JX_PCGGetStopCrit( JX_Solver solver, JX_Int *stop_crit )
{
   return( jx_PCGGetStopCrit( (void *) solver, stop_crit ) );
}

/*!
 * \fn JX_Int JX_PCGSetConvCriteria
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
JX_PCGSetConvCriteria( JX_Solver solver, JX_Int conv_criteria )
{
   return( jx_PCGSetConvCriteria( (void *) solver, conv_criteria ) );
}

/*!
 * \fn JX_Int JX_PCGSetConvFacThreshold
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
JX_PCGSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold )
{
   return( jx_PCGSetConvFacThreshold( (void *) solver, convfac_threshold ) );
}

JX_Int
JX_PCGSetTwoNorm( JX_Solver solver, JX_Int two_norm )
{
   return( jx_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

JX_Int
JX_PCGGetTwoNorm( JX_Solver solver, JX_Int * two_norm )
{
   return( jx_PCGGetTwoNorm( (void *) solver, two_norm ) );
}

JX_Int
JX_PCGSetRelChange( JX_Solver solver, JX_Int rel_change )
{
   return( jx_PCGSetRelChange( (void *) solver, rel_change ) );
}

JX_Int
JX_PCGGetRelChange( JX_Solver solver, JX_Int *rel_change )
{
   return( jx_PCGGetRelChange( (void *) solver, rel_change ) );
}

JX_Int
JX_PCGSetRecomputeResidual( JX_Solver solver, JX_Int recompute_residual )
{
   return( jx_PCGSetRecomputeResidual( (void *) solver, recompute_residual ) );
}

JX_Int
JX_PCGGetRecomputeResidual( JX_Solver solver, JX_Int *recompute_residual )
{
   return( jx_PCGGetRecomputeResidual( (void *) solver, recompute_residual ) );
}

/*!
 * \fn JX_Int jx_PCGSetConvCriteria
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
jx_PCGSetConvCriteria( void *pcg_vdata, JX_Int conv_criteria  )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> conv_criteria) = conv_criteria;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PCGSetConvFacThreshold
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
jx_PCGSetConvFacThreshold( void *pcg_vdata, JX_Real convfac_threshold )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> convfac_threshold) = convfac_threshold;
   return jx_error_flag;
}

JX_Int
JX_PCGSetPrecond( JX_Solver         solver,
                  JX_PtrToSolverFcn precond,
                  JX_PtrToSolverFcn precond_setup,
                  JX_Solver         precond_solver )
{
   return( jx_PCGSetPrecond( (void *) solver, precond, precond_setup, (void *) precond_solver ) );
}

JX_Int
JX_PCGGetPrecond( JX_Solver  solver, JX_Solver *precond_data_ptr )
{
   return( jx_PCGGetPrecond( (void *) solver, (JX_Solver *) precond_data_ptr ) );
}

/*-----------------------------------------------------------------------------
 * JX_PCGSetLogging, JX_PCGGetLogging
 * SetLogging sets both the print and log level, for backwards compatibility.
 * Soon the SetPrintLevel call should be deleted.
 *-----------------------------------------------------------------------------*/

JX_Int
JX_PCGSetLogging( JX_Solver solver, JX_Int level )
{
   return ( jx_PCGSetLogging( (void *) solver, level ) );
}

JX_Int
JX_PCGGetLogging( JX_Solver solver, JX_Int *level )
{
   return ( jx_PCGGetLogging( (void *) solver, level ) );
}

JX_Int
JX_PCGSetPrintLevel( JX_Solver solver, JX_Int level )
{
   return( jx_PCGSetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_PCGGetPrintLevel( JX_Solver solver, JX_Int *level )
{
   return( jx_PCGGetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_PCGGetNumIterations( JX_Solver solver, JX_Int *num_iterations )
{
   return( jx_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int
JX_PCGGetConverged( JX_Solver solver, JX_Int *converged )
{
   return( jx_PCGGetConverged( (void *) solver, converged ) );
}

JX_Int
JX_PCGGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm )
{
   return( jx_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

JX_Int 
JX_PCGGetResidual( JX_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return jx_PCGGetResidual( (void *) solver, residual );
}

JX_Int 
JX_ParCSRPCGDestroy( JX_Solver solver )
{
   return( jx_PCGDestroy( (void *) solver ) );
}

/*!
 * \fn JX_Int jx_PCGGetResidual
 */
JX_Int 
jx_PCGGetResidual( void *pcg_vdata, void **residual )
{
   /* returns a pointer to the residual vector */
   jx_PCGData  *pcg_data = pcg_vdata;
   *residual = pcg_data->r;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PCGDestroy
 */
JX_Int
jx_PCGDestroy( void *pcg_vdata )
{
   jx_PCGData      *pcg_data      = pcg_vdata;
   jx_PCGFunctions *pcg_functions = pcg_data->functions;

   if (pcg_data)
   {
      if ( (pcg_data -> norms) != NULL )
      {
         jx_TFreeF( pcg_data -> norms, pcg_functions );
         pcg_data -> norms = NULL;
      } 
      if ( (pcg_data -> rel_norms) != NULL )
      {
         jx_TFreeF( pcg_data -> rel_norms, pcg_functions );
         pcg_data -> rel_norms = NULL;
      }
      if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
      {
         (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
         pcg_data -> matvec_data = NULL;
      }
      if ( pcg_data -> p != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> p);
         pcg_data -> p = NULL;
      }
      if ( pcg_data -> s != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> s);
         pcg_data -> s = NULL;
      }
      if ( pcg_data -> r != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> r);
         pcg_data -> r = NULL;
      }
#ifdef PCG_Modified_zl 
      if ( pcg_data -> vec_tmp != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> vec_tmp);
         pcg_data -> vec_tmp = NULL;
      }
#endif
      jx_TFreeF( pcg_data, pcg_functions );
      jx_TFreeF( pcg_functions, pcg_functions );
   }

   return(jx_error_flag);
}

/*!
 * \fn JX_Int jx_PCGSetup
 */
JX_Int
jx_PCGSetup( void *pcg_vdata, void *matA, void *vecB, void *vecX )
{
   jx_PCGData      *pcg_data          = pcg_vdata;
   jx_PCGFunctions *pcg_functions     = pcg_data->functions;
   JX_Int              max_iter          = (pcg_data -> max_iter);

   (pcg_data -> A) = matA;

  /*---------------------------------------------------
   * The arguments for CreateVector are important to
   * maintain consistency between the setup and
   * compute phases of matvec and the preconditioner.
   *--------------------------------------------------*/

   if ( pcg_data -> p != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> p);
   }   
   (pcg_data -> p) = (*(pcg_functions->CreateVector))(vecX);

   if ( pcg_data -> s != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> s);
   }
   (pcg_data -> s) = (*(pcg_functions->CreateVector))(vecX);

   if ( pcg_data -> r != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> r);
   }
   (pcg_data -> r) = (*(pcg_functions->CreateVector))(vecB);

#ifdef PCG_Modified_zl                  
   if ( pcg_data -> vec_tmp != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> vec_tmp);
   }
   (pcg_data -> vec_tmp) = (*(pcg_functions->CreateVector))(vecB);             
#endif

   if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
   {
      (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
   }
   (pcg_data -> matvec_data) = (*(pcg_functions->MatvecCreate))(matA, vecX);

  /*-------------------------------------------------------------------
   * Allocate space for log info
   *------------------------------------------------------------------*/
   
   if ( (pcg_data->logging) > 0 || (pcg_data->print_level) > 0 ) 
   {
      if ( (pcg_data -> norms) != NULL )
      {
         jx_TFreeF( pcg_data -> norms, pcg_functions );
      }
      (pcg_data -> norms) = jx_CTAllocF( JX_Real, max_iter + 1, pcg_functions);

      if ( (pcg_data -> rel_norms) != NULL )
      {
         jx_TFreeF( pcg_data -> rel_norms, pcg_functions );
      }
      (pcg_data -> rel_norms) = jx_CTAllocF( JX_Real, max_iter + 1, pcg_functions );
   }

   return jx_error_flag;
}

JX_Int
jx_PCGSetTol( void *pcg_vdata, JX_Real tol )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> tol) = tol;
   return jx_error_flag;
}

JX_Int
jx_PCGGetTol( void *pcg_vdata, JX_Real *tol )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *tol = (pcg_data -> tol);
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PCGSetPreOperator
 * \author peghoty
 * \date 2011/09/04 
 */
JX_Int
jx_PCGSetPreOperator( void *pcg_vdata, void *preOperator )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> B) = preOperator;
   return jx_error_flag;
}

JX_Int
jx_PCGSetAbsoluteTol( void *pcg_vdata, JX_Real a_tol )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> a_tol) = a_tol;
   return jx_error_flag;
}

JX_Int
jx_PCGGetAbsoluteTol( void *pcg_vdata, JX_Real *a_tol )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *a_tol = (pcg_data -> a_tol);
   return jx_error_flag;
}

JX_Int
jx_PCGSetAbsoluteTolFactor( void *pcg_vdata, JX_Real atolf )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> atolf) = atolf;
   return jx_error_flag;
}

JX_Int
jx_PCGGetAbsoluteTolFactor( void *pcg_vdata, JX_Real *atolf )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *atolf = (pcg_data -> atolf);
   return jx_error_flag;
}

JX_Int
jx_PCGSetConvergenceFactorTol( void *pcg_vdata, JX_Real cf_tol )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> cf_tol) = cf_tol;
   return jx_error_flag;
}

JX_Int
jx_PCGGetConvergenceFactorTol( void *pcg_vdata, JX_Real *cf_tol )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *cf_tol = (pcg_data -> cf_tol);
   return jx_error_flag;
}

JX_Int
jx_PCGSetMaxIter( void *pcg_vdata, JX_Int max_iter )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> max_iter) = max_iter;
   return jx_error_flag;
}

JX_Int
jx_PCGGetMaxIter( void *pcg_vdata, JX_Int *max_iter )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *max_iter = (pcg_data -> max_iter);
   return jx_error_flag;
}

JX_Int
jx_PCGSetTwoNorm( void *pcg_vdata, JX_Int two_norm )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> two_norm) = two_norm;
   return jx_error_flag;
}

JX_Int
jx_PCGGetTwoNorm( void *pcg_vdata, JX_Int *two_norm )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *two_norm = (pcg_data -> two_norm);
   return jx_error_flag;
}

JX_Int
jx_PCGSetRelChange( void *pcg_vdata, JX_Int rel_change )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> rel_change) = rel_change;
   return jx_error_flag;
}

JX_Int
jx_PCGGetRelChange( void *pcg_vdata, JX_Int *rel_change )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *rel_change = (pcg_data -> rel_change);
   return jx_error_flag;
}

JX_Int
jx_PCGSetRecomputeResidual( void *pcg_vdata, JX_Int recompute_residual )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> recompute_residual) = recompute_residual;
   return jx_error_flag;
}

JX_Int
jx_PCGGetRecomputeResidual( void *pcg_vdata, JX_Int *recompute_residual )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *recompute_residual = (pcg_data -> recompute_residual);
   return jx_error_flag;
}

JX_Int
jx_PCGSetStopCrit( void *pcg_vdata, JX_Int stop_crit )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> stop_crit) = stop_crit;
   return jx_error_flag;
}

JX_Int
jx_PCGGetStopCrit( void *pcg_vdata, JX_Int *stop_crit )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *stop_crit = (pcg_data -> stop_crit);
   return jx_error_flag;
}

JX_Int
jx_PCGGetPrecond( void *pcg_vdata, JX_Solver *precond_data_ptr )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *precond_data_ptr = (JX_Solver)(pcg_data -> precond_data);
   return jx_error_flag;
}

JX_Int
jx_PCGSetPrecond( void  *pcg_vdata,
                  JX_Int  (*precond)(),
                  JX_Int  (*precond_setup)(),
                  void  *precond_data )
{
   jx_PCGData *pcg_data = pcg_vdata;
   jx_PCGFunctions *pcg_functions = pcg_data->functions;

   (pcg_functions -> precond)       = precond;
   (pcg_functions -> precond_setup) = precond_setup;
   (pcg_data -> precond_data)       = precond_data;
 
   return jx_error_flag;
}

JX_Int
jx_PCGSetPrintLevel( void *pcg_vdata, JX_Int level )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> print_level) = level;
   return jx_error_flag;
}

JX_Int
jx_PCGGetPrintLevel( void *pcg_vdata, JX_Int *level )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *level = (pcg_data -> print_level);
   return jx_error_flag;
}

JX_Int
jx_PCGSetLogging( void *pcg_vdata, JX_Int level )
{
   jx_PCGData *pcg_data = pcg_vdata;
   (pcg_data -> logging) = level;
   return jx_error_flag;
}

JX_Int
jx_PCGGetLogging( void *pcg_vdata, JX_Int *level )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *level = (pcg_data -> logging);
   return jx_error_flag;
}

JX_Int
jx_PCGGetNumIterations( void *pcg_vdata, JX_Int *num_iterations )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *num_iterations = (pcg_data -> num_iterations);
   return jx_error_flag;
}

JX_Int
jx_PCGGetConverged( void *pcg_vdata, JX_Int *converged )
{
   jx_PCGData *pcg_data = pcg_vdata;
   *converged = (pcg_data -> converged);
   return jx_error_flag;
}

JX_Int
jx_PCGPrintLogging( void *pcg_vdata, JX_Int myid )
{
   jx_PCGData *pcg_data       = pcg_vdata;

   JX_Int         num_iterations = (pcg_data -> num_iterations);
   JX_Int         print_level    = (pcg_data -> print_level);
   JX_Real     *norms          = (pcg_data -> norms);
   JX_Real     *rel_norms      = (pcg_data -> rel_norms);

   JX_Int         i;

   if (myid == 0)
   {
      if (print_level > 0)
      {
         for (i = 0; i < num_iterations; i ++)
         {
            jx_printf("Residual norm[%d] = %e   ", i, norms[i]);
            jx_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
         }
      }
   }
  
   return jx_error_flag;
}

JX_Int
jx_PCGGetFinalRelativeResidualNorm( void *pcg_vdata, JX_Real *relative_residual_norm )
{
   jx_PCGData *pcg_data          = pcg_vdata;
   JX_Real      rel_residual_norm = (pcg_data -> rel_residual_norm);
  *relative_residual_norm = rel_residual_norm;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_PCGSolve
 * \note
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps 
 */
JX_Int
jx_PCGSolve( void *pcg_vdata, void *preOperator, void *matA, void *vecB, void *vecX )
{
   jx_PCGData      *pcg_data      = pcg_vdata;
   jx_PCGFunctions *pcg_functions = pcg_data -> functions;

   JX_Real          r_tol              = (pcg_data -> tol);
   JX_Real          a_tol              = (pcg_data -> a_tol);
   JX_Real          atolf              = (pcg_data -> atolf);
   JX_Real          cf_tol             = (pcg_data -> cf_tol);
   JX_Int             max_iter           = (pcg_data -> max_iter);
   JX_Int             two_norm           = (pcg_data -> two_norm);
   JX_Int             rel_change         = (pcg_data -> rel_change);
   JX_Int             recompute_residual = (pcg_data -> recompute_residual);
   JX_Int             stop_crit          = (pcg_data -> stop_crit);
   JX_Real          convfac_threshold = (pcg_data -> convfac_threshold); // peghoty 2010/06/23
   void           *vecP               = (pcg_data -> p);
   void           *vecS               = (pcg_data -> s);
   void           *vecR               = (pcg_data -> r);
#ifdef PCG_Modified_zl
   void           *vec_tmp            = (pcg_data -> vec_tmp);
#endif
   void           *matvec_data        = (pcg_data -> matvec_data);
   JX_Int           (*precond)()         = (pcg_functions -> precond);
   void           *precond_data       = (pcg_data -> precond_data);
   JX_Int             print_level        = (pcg_data -> print_level);
   JX_Int             logging            = (pcg_data -> logging);
   JX_Real         *norms              = (pcg_data -> norms);
   JX_Real         *rel_norms          = (pcg_data -> rel_norms);
                
   JX_Real          alpha, beta;
   JX_Real          gamma, gamma_old;
   JX_Real          i_prod = 0.0;
   JX_Real          bi_prod, eps;
   JX_Real          pi_prod, xi_prod;
   JX_Real          ieee_check = 0.;
                
   JX_Real          i_prod_0 = 0.0;
   JX_Real          cf_ave_0 = 0.0;
   JX_Real          cf_ave_1 = 0.0;
   JX_Real          weight;
   JX_Real          ratio;

   JX_Real          guard_zero_residual, sdotp;
   JX_Int             tentatively_converged = 0; 

   JX_Int             i = 0;
   JX_Int             my_id, num_procs;

   JX_Real  normup       = 0.0;
   JX_Real  normdown     = 0.0;
   JX_Real  conv_factor  = 0.0;


   (pcg_data -> converged) = 0;

   (*(pcg_functions->CommInfo))(matA, &my_id, &num_procs);

  /*-----------------------------------------------------------------------
   * With relative change convergence test on, it is possible to attempt
   * another iteration with a zero residual. This causes the parameter
   * alpha to go NaN. The guard_zero_residual parameter is to circumvent
   * this. Perhaps it should be set to something non-zero (but small).
   *-----------------------------------------------------------------------*/

   guard_zero_residual = 0.0;

  /*-----------------------------------------------------------------------
   * Start pcg solve
   *----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = (*(pcg_functions->InnerProd))(vecB, vecB);
      if (print_level > 1 && my_id == 0) 
      {
         jx_printf(" <b,b>: %e\n",bi_prod);
      }
   }
   else
   {
      /* bi_prod = <C*b,b> */
      (*(pcg_functions->ClearVector))(vecP);
      precond(precond_data, preOperator, vecB, vecP);
      bi_prod = (*(pcg_functions->InnerProd))(vecP, vecB);
      if (print_level > 1 && my_id == 0)
      {
         jx_printf(" <C*b,b>: %e\n",bi_prod);
      }
   }

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (bi_prod != 0.) ieee_check = bi_prod / bi_prod; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
         jx_printf("\n\nERROR detected by JXPAMG ...  BEGIN\n");
         jx_printf("ERROR -- jx_PCGSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ...  END\n\n\n");
//         break;
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   eps = r_tol*r_tol; /* note: this may be re-assigned below */
   if ( bi_prod > 0.0 ) 
   {
      if ( stop_crit && !rel_change && atolf <= 0 ) 
      {  
         /* pure absolute tolerance */
         eps = eps / bi_prod;
         /* Note: this section is obsolete.  Aside from backwards comatability
            concerns, we could delete the stop_crit parameter and related code,
            using tol & atolf instead. */
      }
      else if ( atolf > 0 )  /* mixed relative and absolute tolerance */
      {
         bi_prod += atolf;
      }
      else /* DEFAULT (stop_crit and atolf exist for backwards compatibilty and are not in the reference manual) */
      {
        /* convergence criteria:  <C*r,r>  <= max( a_tol^2, r_tol^2 * <C*b,b> )
           note: default for a_tol is 0.0, so relative residual criteria is used unless
           user specifies a_tol, or sets r_tol = 0.0, which means absolute
           tol only is checked  */
         eps = jx_max(r_tol*r_tol, a_tol*a_tol/bi_prod);
      }
   }
   else /* bi_prod == 0.0: the rhs vector b is zero */
   {
      /* Set x equal to zero and return */
      (*(pcg_functions->CopyVector))(vecB, vecX);
      if (logging > 0 || print_level > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }

      return jx_error_flag;
      /* In this case, for the original parcsr pcg, the code would take special
         action to force iterations even though the exact value was known. */
   }

   /* r = b - Ax */
   (*(pcg_functions->CopyVector))(vecB, vecR);
   (*(pcg_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
 
   /* p = C*r */
   (*(pcg_functions->ClearVector))(vecP);
   precond(precond_data, preOperator, vecR, vecP);

   /* gamma = <r,p> */
   gamma = (*(pcg_functions->InnerProd))(vecR,vecP);

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (gamma != 0.) ieee_check = gamma / gamma; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
         jx_printf("\n\nERROR detected by JXPAMG ...  BEGIN\n");
         jx_printf("ERROR -- jx_PCGSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ...  END\n\n\n");
//         break;
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   if (two_norm)
   {
      normdown = (*(pcg_functions->InnerProd))(vecR,vecR);
   }
   else
   {
      normdown = gamma;
   }


   /* Set initial residual norm */
   if ( logging > 0 || print_level > 0 || cf_tol > 0.0 )
   {
      if (two_norm)
      {
         i_prod_0 = (*(pcg_functions->InnerProd))(vecR, vecR);
      }
      else
      {
         i_prod_0 = gamma;
      }

      if ( logging > 0 || print_level > 0 ) 
      {
         norms[0] = sqrt(i_prod_0);
      }
   }
   
   if ( print_level > 0 && my_id == 0 )
   {
      jx_printf("\n\n");
      if (two_norm)
      {
         if ( stop_crit && !rel_change && atolf == 0 ) 
         {  
            /* pure absolute tolerance */
            jx_printf("Iters       ||r||_2     conv.rate\n");
            jx_printf("-----    ------------   ---------\n");
         }
         else 
         {
            jx_printf("Iters       ||r||_2     conv.rate  ||r||_2/||b||_2\n");
            jx_printf("-----    ------------   ---------  ------------ \n");
         }
      }
      else  /* !two_norm */
      {
         jx_printf("Iters       ||r||_C     conv.rate  ||r||_C/||b||_C\n");
         jx_printf("-----    ------------    ---------  ------------ \n");
      }
   }

   while ((i+1) <= max_iter)
   {
     /*--------------------------------------------------------------------
      * the core CG calculations...
      *--------------------------------------------------------------------*/
      i ++;

      /* s = A*p */
      (*(pcg_functions->Matvec))(matvec_data, 1.0, matA, vecP, 0.0, vecS);

      /* alpha = gamma / <s,p> */
      sdotp = (*(pcg_functions->InnerProd))(vecS, vecP);
      if (sdotp == 0.0)
      {
         if (i == 1) i_prod = i_prod_0;
         break;
      }
      alpha = gamma / sdotp;

      gamma_old = gamma;

      /* x = x + alpha*p */
      (*(pcg_functions->Axpy))(alpha, vecP, vecX);

      /* r = r - alpha*s */
      (*(pcg_functions->Axpy))(-alpha, vecS, vecR);
         
      /* s = C*r */
      (*(pcg_functions->ClearVector))(vecS);
      precond(precond_data, preOperator, vecR, vecS);

      /* gamma = <r,s> */
      gamma = (*(pcg_functions->InnerProd))(vecR, vecS);

      if (two_norm)
      {
#ifdef PCG_Modified_zl  
         // normup = (*(pcg_functions->InnerProd))(vecR,vecR);
         /* r = b - Ax */
         (*(pcg_functions->CopyVector))(vecB, vec_tmp);
         (*(pcg_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vec_tmp);
         normup = (*(pcg_functions->InnerProd))(vec_tmp, vec_tmp);

#if 0
         double normup1 = (*(pcg_functions->InnerProd))(vecR,vecR);
         printf("||r||_2 = ||b-Ax||_2 = %e, ||r||_2 = ||r - alpha*A*p||_2 = %e\n", sqrt(normup), sqrt(normup1));
#endif         
#else
         normup = (*(pcg_functions->InnerProd))(vecR,vecR);
#endif
         i_prod = normup;
      }
      else
      {
         normup = gamma;
         i_prod = gamma;
      }
      conv_factor = sqrt(normup / normdown);
      normdown = normup;

      // /* set i_prod for convergence test */
      // if (two_norm)
      // {
      //    // i_prod = (*(pcg_functions->InnerProd))(vecR,vecR);
      //    (*(pcg_functions->CopyVector))(vecB, (void *)vecR_tmp);
      //    (*(pcg_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, (void *)vecR_tmp);
      //    i_prod = (*(pcg_functions->InnerProd))((void *)vecR_tmp,(void *)vecR_tmp);
      // }
      // else
      // {
      //    i_prod = gamma;
      // }
 
      /* print norm info */
      if ( logging > 0 || print_level > 0 )
      {
         norms[i]     = sqrt(i_prod);
         rel_norms[i] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }
      if ( print_level > 0 && my_id == 0 )
      {
         if (two_norm)
         {
            if ( stop_crit && !rel_change && atolf==0 ) 
            {  
               /* pure absolute tolerance */
               jx_printf("% 5d    %e    %f\n", i, norms[i], norms[i]/norms[i-1]);
            }
            else 
            {
               jx_printf("% 5d    %e    %f    %e\n", i, norms[i], norms[i]/norms[i-1], rel_norms[i]);
            }
         }
         else 
         {
            jx_printf("% 5d    %e    %f    %e\n", i, norms[i], norms[i]/norms[i-1], rel_norms[i]);
         }
      }

      if (i > 1 && conv_factor > convfac_threshold)
      {
         if (my_id == 0 && print_level > 0)
         {
            jx_printf("\n  Warning: iteration has terminated because the\n");
            jx_printf("           current conv_factor > convfac_threshold(%lf) !!\n\n",convfac_threshold);
         }
         break;
      }

      if (i > 1 && conv_factor > convfac_threshold)
      {
         if (my_id == 0 && print_level > 0)
         {
            jx_printf("\n  Warning: iteration has terminated because the\n");
            jx_printf("           current conv_factor > convfac_threshold(%lf) !!\n\n",convfac_threshold);
         }
         break;
      }

     /*--------------------------------------------------------------------
      * check for convergence
      *--------------------------------------------------------------------*/
      if (i_prod / bi_prod < eps)  /* the basic convergence test */
      {
         tentatively_converged = 1;
      }

     /* At user request, don't trust the convergence test until we've recomputed
        the residual from scratch.  This is expensive in the usual case where an
        the norm is the energy norm.
        This calculation is coded on the assumption that r's accuracy is only a
        concern for problems where CG takes many iterations. */
      if ( tentatively_converged && recompute_residual )
      {
         /* r = b - Ax */
         (*(pcg_functions->CopyVector))(vecB, vecR);
         (*(pcg_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);

         /* set i_prod for convergence test */
         if (two_norm)
         {
            i_prod = (*(pcg_functions->InnerProd))(vecR, vecR);
         }
         else
         {
            /* s = C*r */
            (*(pcg_functions->ClearVector))(vecS);
            precond(precond_data, preOperator, vecR, vecS);
            /* iprod = gamma = <r,s> */
            i_prod = (*(pcg_functions->InnerProd))(vecR, vecS);
         }
         
         if (i_prod / bi_prod >= eps) 
         {
            tentatively_converged = 0;
         }
      }

     /* At user request, don't treat this as converged unless x didn't change
        much in the last iteration. */ 
      if ( tentatively_converged && rel_change && (i_prod > guard_zero_residual) )
      {
         pi_prod = (*(pcg_functions->InnerProd))(vecP, vecP); 
         xi_prod = (*(pcg_functions->InnerProd))(vecX, vecX);
         ratio = alpha*alpha*pi_prod / xi_prod;
         if (ratio >= eps) 
         {
            tentatively_converged = 0;
         }
      }
      
      /* we've passed all the convergence tests, it's for real */
      if ( tentatively_converged )  
      {
         (pcg_data -> converged) = 1;
         break;
      }

      if ( (gamma<1.0e-292) && ((-gamma)<1.0e-292) ) 
      {
         jx_error(JX_ERROR_CONV);  
         break;
      }
      
      /* ... gamma should be >=0.  IEEE subnormal numbers are < 2**(-1022)=2.2e-308
         (and >= 2**(-1074)=4.9e-324).  So a gamma this small means we're getting
         dangerously close to subnormal or zero numbers (usually if gamma is small,
         so will be other variables).  Thus further calculations risk a crash.
         Such small gamma generally means no hope of progress anyway. */

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

        /* i_prod_0 is zero, or (almost) subnormal, yet i_prod wasn't small
           enough to pass the convergence test.  Therefore initial guess was good,
           and we're just calculating garbage - time to bail out before the
           next step, which will be a divide by zero (or close to it). */         
         if ( i_prod_0 < 1.0e-292 ) 
         {
            jx_error(JX_ERROR_CONV);  
            break;
         }
	      cf_ave_1 = pow( i_prod / i_prod_0, 1.0/(2.0*i)); 

         weight   = fabs(cf_ave_1 - cf_ave_0);
         weight   = weight / jx_max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
         
         if (weight * cf_ave_1 > cf_tol) break;
      }

     /*--------------------------------------------------------------------
      * back to the core CG calculations
      *--------------------------------------------------------------------*/

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      (*(pcg_functions->ScaleVector))(beta, vecP);   
      (*(pcg_functions->Axpy))(1.0, vecS, vecP);
   }


   /*--------------------------------------------------------------------
    * Finish up with some outputs.
    *--------------------------------------------------------------------*/

   if ( print_level > 1 && my_id == 0 )
   {
      jx_printf("\n\n");
   }

   (pcg_data -> num_iterations) = i;
   
   if (bi_prod > 0.0)
   {
      (pcg_data -> rel_residual_norm) = sqrt(i_prod / bi_prod);
   }
   else /* actually, we'll never get here... */
   {
      (pcg_data -> rel_residual_norm) = 0.0;
   }

   return jx_error_flag;
}
