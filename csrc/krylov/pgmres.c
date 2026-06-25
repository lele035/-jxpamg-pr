//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  pgmres.c
 *  Date: 2011/09/03
 */ 

#include "jx_combined.h"
#include "jx_krylov.h"

JX_Int jx__kdim_memory_error = -1; // Feng Chunsheng & Yue Xiaoqiang 2012/10/26

/*!
 * \fn JX_Int JX_ParCSRGMRESCreate
 */ 
JX_Int
JX_ParCSRGMRESCreate( MPI_Comm comm, JX_Solver *solver )
{
   jx_GMRESFunctions * gmres_functions =
      jx_GMRESFunctionsCreate(
         jx_CAlloc, 
         jx_ParKrylovFree, 
         jx_ParKrylovCommInfo,
         jx_ParKrylovCreateVector,
         jx_ParKrylovCreateVectorArray,
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

   *solver = ( (JX_Solver) jx_GMRESCreate( gmres_functions ) );
   
   if (!solver) 
   {
      jx_error_in_arg(2);
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int JX_ParCSRGMRESDestroy
 */ 
JX_Int 
JX_ParCSRGMRESDestroy( JX_Solver solver )
{
   return( jx_GMRESDestroy( (void *) solver ) );
}

/*!
 * \fn JX_Int JX_GMRESSetup
 */ 
JX_Int 
JX_GMRESSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX )
{
   return( jx_GMRESSetup( solver, matA, vecB, vecX ) );
}

/*!
 * \fn JX_Int JX_GMRESSolve
 */
JX_Int 
JX_GMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX )
{
   return( jx_GMRESSolve( solver, preOperator, matA, vecB, vecX ) );
}

/*!
 * \fn JX_Int JX_GMRESAdaptiveSolve
 */
JX_Int
JX_GMRESAdaptiveSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX )
{
   return( jx_GMRESAdaptiveSolve( solver, preOperator, matA, vecB, vecX ) );
}

/*!
 * \fn JX_Int JX_GMRESSetPreOperator
 * \brief Set preconditioner operater.
 * \author peghoty
 * \date 2011/09/04 
 */
JX_Int
JX_GMRESSetPreOperator( JX_Solver solver, JX_ParCSRMatrix preOperator )
{
   return( jx_GMRESSetPreOperator( (void *) solver, (void *) preOperator ) );
}

JX_Int
JX_GMRESSetKDim( JX_Solver solver, JX_Int k_dim )
{
   return( jx_GMRESSetKDim( (void *) solver, k_dim ) );
}

JX_Int
JX_GMRESGetKDim( JX_Solver solver, JX_Int *k_dim )
{
   return( jx_GMRESGetKDim( (void *) solver, k_dim ) );
}

JX_Int
JX_GMRESSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_GMRESSetTol( (void *) solver, tol ) );
}

JX_Int
JX_GMRESGetTol( JX_Solver solver, JX_Real *tol )
{
   return( jx_GMRESGetTol( (void *) solver, tol ) );
}

JX_Int
JX_GMRESSetAbsoluteTol( JX_Solver solver, JX_Real a_tol )
{
   return( jx_GMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_GMRESGetAbsoluteTol( JX_Solver solver, JX_Real *a_tol )
{
   return( jx_GMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_GMRESSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol )
{
   return( jx_GMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

JX_Int
JX_GMRESGetConvergenceFactorTol( JX_Solver solver, JX_Real *cf_tol )
{
   return( jx_GMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

JX_Int
JX_GMRESSetMinIter( JX_Solver solver, JX_Int min_iter )
{
   return( jx_GMRESSetMinIter( (void *) solver, min_iter ) );
}

JX_Int
JX_GMRESGetMinIter( JX_Solver solver, JX_Int *min_iter )
{
   return( jx_GMRESGetMinIter( (void *) solver, min_iter ) );
}

JX_Int
JX_GMRESSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_GMRESGetMaxIter( JX_Solver solver, JX_Int *max_iter )
{
   return( jx_GMRESGetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_GMRESSetStopCrit( JX_Solver solver, JX_Int stop_crit )
{
   return( jx_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

JX_Int
JX_GMRESGetStopCrit( JX_Solver solver, JX_Int *stop_crit )
{
   return( jx_GMRESGetStopCrit( (void *) solver, stop_crit ) );
}

/*!
 * \fn JX_Int JX_GMRESSetConvCriteria
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
JX_GMRESSetConvCriteria( JX_Solver solver, JX_Int conv_criteria )
{
   return( jx_GMRESSetConvCriteria( (void *) solver, conv_criteria ) );
}

/*!
 * \fn JX_Int JX_GMRESSetConvFacThreshold
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
JX_GMRESSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold )
{
   return( jx_GMRESSetConvFacThreshold( (void *) solver, convfac_threshold ) );
}

/*!
 * \fn JX_Int JX_GMRESSetResDownZeroThreshold
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
JX_GMRESSetResDownZeroThreshold( JX_Solver solver, JX_Real resdown_0_threshold )
{
   return( jx_GMRESSetResDownZeroThreshold( (void *) solver, resdown_0_threshold ) );
}

/*!
 * \fn JX_Int JX_GMRESSetConvFacThresholdTwo
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
JX_GMRESSetConvFacThresholdTwo( JX_Solver solver, JX_Real convfac_threshold_2 )
{
   return( jx_GMRESSetConvFacThresholdTwo( (void *) solver, convfac_threshold_2 ) );
}

/*!
 * \fn JX_Int JX_GMRESSetIsCheckRestarted
 * \author peghoty
 * \date 2011/11/08 
 */
JX_Int
JX_GMRESSetIsCheckRestarted( JX_Solver solver, JX_Int is_check_restarted )
{
   return( jx_GMRESSetIsCheckRestarted( (void *) solver, is_check_restarted ) );
}

JX_Int
JX_GMRESSetRelChange( JX_Solver solver, JX_Int rel_change )
{
   return( jx_GMRESSetRelChange( (void *) solver, rel_change ) );
}

JX_Int
JX_GMRESGetRelChange( JX_Solver solver, JX_Int *rel_change )
{
   return( jx_GMRESGetRelChange( (void *) solver, rel_change ) );
}

/*!
 * \fn JX_Int JX_GMRESGetLastPrecondType
 * \brief Get last_precond_type
 * \author Yue Xiaoqiang
 * \date 2014/10/24
 */
JX_Int
JX_GMRESGetLastPrecondType( JX_Solver solver, JX_Int *last_precond_type )
{
   return( jx_GMRESGetLastPrecondType( (void *) solver, last_precond_type ) );
}

/*!
 * \fn JX_Int JX_GMRESSetPrecond
 */
JX_Int
JX_GMRESSetPrecond( JX_Solver          solver,
                    JX_PtrToSolverFcn  precond,
                    JX_PtrToSolverFcn  precond_setup,
                    JX_Solver          precond_solver )
{
   return( jx_GMRESSetPrecond( (void *) solver,
                               precond, 
                               precond_setup,
                               (void *) precond_solver ) );
}

/*!
 * \fn JX_Int JX_GMRESGetPrecond
 */
JX_Int
JX_GMRESGetPrecond( JX_Solver solver, JX_Solver *precond_data_ptr )
{
   return( jx_GMRESGetPrecond( (void *) solver, (JX_Solver *) precond_data_ptr ) );
}

JX_Int
JX_GMRESSetPrintLevel( JX_Solver solver, JX_Int level )
{
   return( jx_GMRESSetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_GMRESGetPrintLevel( JX_Solver solver, JX_Int *level )
{
   return( jx_GMRESGetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_GMRESGetCompAMGILUBcoIter( JX_Solver solver, JX_Int *amg_iter, JX_Int *ilu_iter, JX_Int *bco_iter )
{
   return( jx_GMRESGetCompAMGILUBcoIter( (void *) solver, amg_iter, ilu_iter, bco_iter ) );
}

JX_Int
JX_GMRESSetLogging( JX_Solver solver, JX_Int level )
{
   return( jx_GMRESSetLogging( (void *) solver, level ) );
}

JX_Int
JX_GMRESGetLogging( JX_Solver solver, JX_Int * level )
{
   return( jx_GMRESGetLogging( (void *) solver, level ) );
}

JX_Int
JX_GMRESGetNumIterations( JX_Solver solver, JX_Int *num_iterations )
{
   return( jx_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int
JX_GMRESGetConverged( JX_Solver solver, JX_Int *converged )
{
   return( jx_GMRESGetConverged( (void *) solver, converged ) );
}

JX_Int
JX_GMRESGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm )
{
   return( jx_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

JX_Int JX_GMRESGetResidual( JX_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return jx_GMRESGetResidual( (void *) solver, residual );
}

/*!
 * \fn void *jx_ParCSRGMRESCreate
 * \author peghoty
 * \date 2011/09/03
 */
void *
jx_ParCSRGMRESCreate( MPI_Comm comm )
{
   jx_GMRESData *solver = NULL;
   
   jx_GMRESFunctions * gmres_functions =
      jx_GMRESFunctionsCreate(
         jx_CAlloc, 
         jx_ParKrylovFree, 
         jx_ParKrylovCommInfo,
         jx_ParKrylovCreateVector,
         jx_ParKrylovCreateVectorArray,
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

   solver = ( (jx_GMRESData *)jx_GMRESCreate( gmres_functions ) );
   
   if (!solver) 
   {
      jx_error_in_arg(2);
   }

   return solver;
}

/*!
 * \fn jx_GMRESFunctions *jx_GMRESFunctionsCreate
 */
jx_GMRESFunctions *
jx_GMRESFunctionsCreate(
   char * (*CAlloc)             ( size_t count, size_t elt_size ),
   JX_Int    (*Free)               ( char *ptr ),
   JX_Int    (*CommInfo)           ( void *A, JX_Int *my_id, JX_Int *num_procs ),
   void * (*CreateVector)       ( void *vector ),
   void * (*CreateVectorArray)  ( JX_Int size, void *vectors ),
   JX_Int    (*DestroyVector)      ( void *vector ),
   void * (*MatvecCreate)       ( void *A, void *x ),
   JX_Int    (*Matvec)             ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y ),
   JX_Int    (*MatvecDestroy)      ( void *matvec_data ),
   JX_Real (*InnerProd)          ( void *x, void *y ),
   JX_Int    (*CopyVector)         ( void *x, void *y ),
   JX_Int    (*ClearVector)        ( void *x ),
   JX_Int    (*ScaleVector)        ( JX_Real alpha, void *x ),
   JX_Int    (*Axpy)               ( JX_Real alpha, void *x, void *y ),
   JX_Int    (*PrecondSetup)       ( void *vdata, void *A ),
   JX_Int    (*Precond)            ( void *vdata, void *A, void *b, void *x )
)
{
   jx_GMRESFunctions *gmres_functions;
   gmres_functions = (jx_GMRESFunctions *)CAlloc( 1, sizeof(jx_GMRESFunctions) );

   gmres_functions->CAlloc = CAlloc;
   gmres_functions->Free = Free;
   gmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   gmres_functions->CreateVector = CreateVector;
   gmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   gmres_functions->DestroyVector = DestroyVector;
   gmres_functions->MatvecCreate = MatvecCreate;
   gmres_functions->Matvec = Matvec;
   gmres_functions->MatvecDestroy = MatvecDestroy;
   gmres_functions->InnerProd = InnerProd;
   gmres_functions->CopyVector = CopyVector;
   gmres_functions->ClearVector = ClearVector;
   gmres_functions->ScaleVector = ScaleVector;
   gmres_functions->Axpy = Axpy;
   /* default preconditioner must be set here but can be changed later... */
   gmres_functions->precond_setup = PrecondSetup;
   gmres_functions->precond       = Precond;

   return gmres_functions;
}

/*!
 * \fn void *jx_GMRESCreate
 */
void *
jx_GMRESCreate( jx_GMRESFunctions *gmres_functions )
{
   jx_GMRESData *gmres_data;
 
   gmres_data = jx_CTAllocF(jx_GMRESData, 1, gmres_functions);

   gmres_data->functions = gmres_functions;
 
   /* set defaults */
   (gmres_data -> k_dim)               = 5;
   (gmres_data -> tol)                 = 1.0e-06; /* relative residual tol */
   (gmres_data -> cf_tol)              = 0.0;
   (gmres_data -> a_tol)               = 0.0;     /* abs. residual tol */
   (gmres_data -> min_iter)            = 0;
   (gmres_data -> max_iter)            = 1000;
   (gmres_data -> rel_change)          = 0;
   (gmres_data -> stop_crit)           = 0;       /* rel. residual norm  - this is obsolete! */
   (gmres_data -> conv_criteria)       = 0;       /* peghoty 2010/06/23 */
   (gmres_data -> convfac_threshold)   = 1.0e4;   /* peghoty 2010/06/23 */
   (gmres_data -> resdown_0_threshold) = 0.1;     /* Yue Xiaoqiang, 2014/03/24 */
   (gmres_data -> convfac_threshold_2) = 0.1;     /* Yue Xiaoqiang, 2014/03/24 */
   (gmres_data -> comp_amg_iter)       = 0;       /* Yue Xiaoqiang, 2016/02/20 */
   (gmres_data -> comp_ilu_iter)       = 0;       /* Yue Xiaoqiang, 2016/02/20 */
   (gmres_data -> comp_bco_iter)       = 0;       /* Yue Xiaoqiang, 2016/02/20 */
   (gmres_data -> is_check_restarted)  = 0;       /* peghoty 2011/11/08 */
   (gmres_data -> converged)           = 0;
   (gmres_data -> precond_data)        = NULL;
   (gmres_data -> print_level)         = 0;
   (gmres_data -> logging)             = 0;
   (gmres_data -> B)                   = NULL;    /* peghoty 2011/10/08 */
   (gmres_data -> p)                   = NULL;
   (gmres_data -> r)                   = NULL;
   (gmres_data -> w)                   = NULL;
   (gmres_data -> w_2)                 = NULL;
   (gmres_data -> matvec_data)         = NULL;
   (gmres_data -> norms)               = NULL;
   (gmres_data -> log_file_name)       = NULL;
 
   return (void *) gmres_data;
}

/*!
 * \fn JX_Int jx_GMRESDestroy
 */
JX_Int
jx_GMRESDestroy( void *gmres_vdata )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   jx_GMRESFunctions *gmres_functions = gmres_data->functions;
   JX_Int i;
 
   if (gmres_data)
   {
      if ( (gmres_data->logging>0) || (gmres_data->print_level) > 0 )
      {
         if ( (gmres_data -> norms) != NULL )
         {
            jx_TFreeF( gmres_data -> norms, gmres_functions );
         }
      }
 
      if ( (gmres_data -> matvec_data) != NULL )
      {
         (*(gmres_functions->MatvecDestroy))(gmres_data -> matvec_data);
      }
 
      if ( (gmres_data -> r) != NULL )
      {
         (*(gmres_functions->DestroyVector))(gmres_data -> r);
      }
      
      if ( (gmres_data -> w) != NULL )
      {
         (*(gmres_functions->DestroyVector))(gmres_data -> w);
      }
      
      if ( (gmres_data -> w_2) != NULL )
      {
         (*(gmres_functions->DestroyVector))(gmres_data -> w_2);
      }

      if ( (gmres_data -> p) != NULL )
      {
         for (i = 0; i < (gmres_data -> k_dim+1); i ++)
         {
            if ( (gmres_data -> p)[i] != NULL )
            {
               (*(gmres_functions->DestroyVector))( (gmres_data -> p)[i] );
            }
         }
         jx_TFreeF( gmres_data->p, gmres_functions );
      }
      jx_TFreeF( gmres_data, gmres_functions );
      jx_TFreeF( gmres_functions, gmres_functions );
   }
 
   return jx_error_flag;
}

JX_Int 
jx_GMRESGetResidual( void *gmres_vdata, void **residual )
{
   /* returns a pointer to the residual vector */
   jx_GMRESData  *gmres_data = gmres_vdata;
   *residual = gmres_data->r;
   return jx_error_flag;
   
}

/*!
 * \fn JX_Int jx_GMRESSetup
 */
JX_Int
jx_GMRESSetup( void *gmres_vdata, void *matA, void *vecB, void *vecX )
{
   jx_GMRESData      *gmres_data      = gmres_vdata;
   jx_GMRESFunctions *gmres_functions = gmres_data->functions;

   JX_Int            k_dim            = (gmres_data -> k_dim);
   JX_Int            max_iter         = (gmres_data -> max_iter);
   JX_Int            rel_change       = (gmres_data -> rel_change);
    
   (gmres_data -> A) = matA;
 
  /*--------------------------------------------------
   * The arguments for NewVector are important to
   * maintain consistency between the setup and
   * compute phases of matvec and the preconditioner.
   *--------------------------------------------------*/
 
 
   if ((gmres_data -> r) == NULL)
   {
      (gmres_data -> r) = (*(gmres_functions->CreateVector))(vecB);
   }
   
   if ((gmres_data -> w) == NULL)
   {
      (gmres_data -> w) = (*(gmres_functions->CreateVector))(vecB);
   }
 
   if (rel_change)
   {  
      if ((gmres_data -> w_2) == NULL)
      {
         (gmres_data -> w_2) = (*(gmres_functions->CreateVector))(vecB);
      }
   }
   
   if ((gmres_data -> matvec_data) == NULL)
   {
      (gmres_data -> matvec_data) = (*(gmres_functions->MatvecCreate))(matA, vecX);
   }

  /*-----------------------------------------------------
   * Allocate space for log info
   *-----------------------------------------------------*/
 
   if ( (gmres_data->logging) > 0 || (gmres_data->print_level) > 0 )
   {
      if ((gmres_data -> norms) == NULL)
      {
         (gmres_data -> norms) = jx_CTAllocF(JX_Real, max_iter + 1, gmres_functions);
      }
   }
   if ( (gmres_data->print_level) > 0 ) 
   {
      if ((gmres_data -> log_file_name) == NULL)
      {
         (gmres_data -> log_file_name) = "gmres.out.log";
      }
   }
 
   if ((gmres_data -> p) == NULL)
   {
      (gmres_data -> p) = (*(gmres_functions->CreateVectorArray))(k_dim + 1, vecX);
      if (jx__kdim_memory_error > 1) // Feng Chunsheng & Yue Xiaoqiang 2012/10/26
      {
         if ( (gmres_data->print_level) > 0 )
         {
            jx_printf("\n\nWARNING: K_DIM = %d IS TOO BIG, ", k_dim);
            jx_printf("MODIFIED TO BE THE SUGGESTED VALUE %d\n\n", jx__kdim_memory_error-1);
         }
         k_dim = jx__kdim_memory_error - 1;
         gmres_data -> k_dim = k_dim;
         jx__kdim_memory_error = -1;
         (gmres_data -> p) = (*(gmres_functions->CreateVectorArray))(k_dim + 1, vecX);
      }
   }
 
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetPreOperator
 * \brief Set preconditioner operater.
 * \author peghoty
 * \date 2011/09/04 
 */
JX_Int
jx_GMRESSetPreOperator( void *gmres_vdata, void *preOperator )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> B) = preOperator;
   return jx_error_flag;
}

JX_Int
jx_GMRESSetKDim( void *gmres_vdata, JX_Int k_dim )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> k_dim) = k_dim;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetKDim( void *gmres_vdata, JX_Int *k_dim )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *k_dim = (gmres_data -> k_dim);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetTol( void *gmres_vdata, JX_Real tol )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> tol) = tol;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetTol( void *gmres_vdata, JX_Real *tol )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *tol = (gmres_data -> tol);
   return jx_error_flag;
}
 
JX_Int
jx_GMRESSetAbsoluteTol( void *gmres_vdata, JX_Real a_tol )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> a_tol) = a_tol;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetAbsoluteTol( void *gmres_vdata, JX_Real *a_tol )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *a_tol = (gmres_data -> a_tol);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetConvergenceFactorTol( void *gmres_vdata, JX_Real cf_tol )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> cf_tol) = cf_tol;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetConvergenceFactorTol( void *gmres_vdata, JX_Real *cf_tol )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *cf_tol = (gmres_data -> cf_tol);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetMinIter( void *gmres_vdata, JX_Int min_iter )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> min_iter) = min_iter;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetMinIter( void *gmres_vdata, JX_Int *min_iter )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *min_iter = (gmres_data -> min_iter);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetMaxIter( void *gmres_vdata, JX_Int max_iter )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> max_iter) = max_iter;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetMaxIter( void *gmres_vdata, JX_Int *max_iter )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *max_iter = (gmres_data -> max_iter);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetRelChange( void *gmres_vdata, JX_Int rel_change )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> rel_change) = rel_change;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetRelChange( void *gmres_vdata, JX_Int *rel_change )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *rel_change = (gmres_data -> rel_change);
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESGetLastPrecondType
 * \brief Get last_precond_type
 * \author Yue Xiaoqiang
 * \date 2014/10/24
 */
JX_Int
jx_GMRESGetLastPrecondType( void *gmres_vdata, JX_Int *last_precond_type )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   JX_Int break_adaptive = (gmres_data -> break_adaptive);
   JX_Int *precond_data = (gmres_data -> precond_data);
   *last_precond_type = break_adaptive * 10 + jx_CombinedPrecondDataPreID((jx_CombinedPrecondData *)precond_data);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetStopCrit( void *gmres_vdata, JX_Int stop_crit )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> stop_crit) = stop_crit;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetStopCrit( void *gmres_vdata, JX_Int *stop_crit )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *stop_crit = (gmres_data -> stop_crit);
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetConvCriteria
 * \brief Set preconditioner operater.
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
jx_GMRESSetConvCriteria( void *gmres_vdata, JX_Int conv_criteria )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> conv_criteria ) = conv_criteria;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetConvFacThreshold
 * \brief Set preconditioner operater.
 * \author peghoty
 * \date 2010/06/23 
 */
JX_Int
jx_GMRESSetConvFacThreshold( void *gmres_vdata, JX_Real convfac_threshold )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> convfac_threshold ) = convfac_threshold;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetResDownZeroThreshold
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
jx_GMRESSetResDownZeroThreshold( void *gmres_vdata, JX_Real resdown_0_threshold )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> resdown_0_threshold ) = resdown_0_threshold;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetConvFacThresholdTwo
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
jx_GMRESSetConvFacThresholdTwo( void *gmres_vdata, JX_Real convfac_threshold_2 )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> convfac_threshold_2 ) = convfac_threshold_2;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetIsCheckRestarted
 * \brief Set is_check_restarted.
 * \author peghoty
 * \date 2011/11/08 
 */
JX_Int
jx_GMRESSetIsCheckRestarted( void *gmres_vdata, JX_Int is_check_restarted )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> is_check_restarted ) = is_check_restarted;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSetPrecond
 */
JX_Int
jx_GMRESSetPrecond( void  *gmres_vdata,
                    JX_Int  (*precond)(),
                    JX_Int  (*precond_setup)(),
                    void  *precond_data )
{
   jx_GMRESData      *gmres_data      = gmres_vdata;
   jx_GMRESFunctions *gmres_functions = gmres_data->functions;
   
   (gmres_functions -> precond)       = precond;
   (gmres_functions -> precond_setup) = precond_setup;
   (gmres_data -> precond_data)       = precond_data;
 
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESGetPrecond
 */
JX_Int
jx_GMRESGetPrecond( void *gmres_vdata, JX_Solver *precond_data_ptr )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *precond_data_ptr = (JX_Solver)(gmres_data -> precond_data);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetPrintLevel( void *gmres_vdata, JX_Int level )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> print_level) = level;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetPrintLevel( void *gmres_vdata, JX_Int *level )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *level = (gmres_data -> print_level);
   return jx_error_flag;
}

JX_Int
jx_GMRESGetCompAMGILUBcoIter( void *gmres_vdata, JX_Int *amg_iter, JX_Int *ilu_iter, JX_Int *bco_iter )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *amg_iter = (gmres_data -> comp_amg_iter);
   *ilu_iter = (gmres_data -> comp_ilu_iter);
   *bco_iter = (gmres_data -> comp_bco_iter);
   return jx_error_flag;
}

JX_Int
jx_GMRESSetLogging( void *gmres_vdata, JX_Int level )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   (gmres_data -> logging) = level;
   return jx_error_flag;
}

JX_Int
jx_GMRESGetLogging( void *gmres_vdata, JX_Int *level )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *level = (gmres_data -> logging);
   return jx_error_flag;
}

JX_Int
jx_GMRESGetNumIterations( void *gmres_vdata, JX_Int *num_iterations )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *num_iterations = (gmres_data -> num_iterations);
   return jx_error_flag;
}

JX_Int
jx_GMRESGetConverged( void *gmres_vdata, JX_Int *converged )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *converged = (gmres_data -> converged);
   return jx_error_flag;
}

JX_Int
jx_GMRESGetFinalRelativeResidualNorm( void *gmres_vdata, JX_Real *relative_residual_norm )
{
   jx_GMRESData *gmres_data = gmres_vdata;
   *relative_residual_norm = (gmres_data -> rel_residual_norm);
   return jx_error_flag;
} 

/*!
 * \fn JX_Int jx_GMRESSolve
 */
JX_Int
jx_GMRESSolve( void *gmres_vdata, void *preOperator, void *matA, void *vecB, void *vecX )
{
   jx_GMRESData      *gmres_data      = gmres_vdata;
   jx_GMRESFunctions *gmres_functions = (gmres_data -> functions);
   JX_Int 		     k_dim            = (gmres_data -> k_dim);
   JX_Int               min_iter         = (gmres_data -> min_iter);
   JX_Int 		     max_iter         = (gmres_data -> max_iter);
   JX_Int               rel_change       = (gmres_data -> rel_change);
   JX_Real 	     r_tol            = (gmres_data -> tol);
   JX_Real 	     cf_tol           = (gmres_data -> cf_tol);
   JX_Real            a_tol            = (gmres_data -> a_tol);
   void             *matvec_data      = (gmres_data -> matvec_data);

   JX_Real            convfac_threshold = (gmres_data -> convfac_threshold); // peghoty 2010/06/23

   void             *vecR             = (gmres_data -> r);
   void             *vecW             = (gmres_data -> w);
   void             *vecW2            = (gmres_data -> w_2); /* vecW2 is only allocated if rel_change = 1 */
   void            **vecP             = (gmres_data -> p);
   JX_Int 	           (*precond)()       = (gmres_functions -> precond);
   JX_Int 	            *precond_data     = (gmres_data -> precond_data);
   JX_Int             print_level        = (gmres_data -> print_level);
   JX_Int             logging            = (gmres_data -> logging);
   JX_Real         *norms              = (gmres_data -> norms);
   JX_Int             is_check_restarted = (gmres_data -> is_check_restarted); /* Xu Xiaowen, 2011/11/08 */
   
   JX_Int        break_value = 0;
   JX_Int	      i, j, k;
   JX_Real     *rs, **hh, *c, *s, *rs_2 = NULL; 
   JX_Int        iter; 
   JX_Int        my_id, num_procs;
   JX_Real     epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   JX_Real     w_norm;
   
   JX_Real     epsmac = 1.e-16; 
   JX_Real     ieee_check = 0.0;

   JX_Real     guard_zero_residual; 
   JX_Real     cf_ave_0 = 0.0;
   JX_Real     cf_ave_1 = 0.0;
   JX_Real     weight;
   JX_Real     r_norm_0;
   JX_Real     relative_error = 1.0;

   JX_Int        rel_change_passed = 0, num_rel_change_check = 0;

   /* peghoty 2010/06/23 */
   JX_Real  normup       = 0.0;
   JX_Real  normdown     = 0.0;
   JX_Real  conv_factor  = 0.0;

#if 0
   /* 保存离散系统和初始解向量 */
   jx_CSRMatrix *ser_matrix = NULL;
   ser_matrix = jx_ParCSRMatrixToCSRMatrixAll((jx_ParCSRMatrix *)matA);
   jx_CSRMatrixPrint(ser_matrix,"./A_ser");
   
   jx_Vector *ser_rhs = NULL;
   ser_rhs = jx_ParVectorToVectorAll ((jx_ParVector *)vecB);
   jx_SeqVectorPrint(ser_rhs, "./f_ser");
   
   jx_Vector *ser_sol = NULL;
   ser_sol = jx_ParVectorToVectorAll ((jx_ParVector *)vecX);
   jx_SeqVectorPrint(ser_sol, "./u0_ser"); 

   jx_CSRMatrixDestroy(ser_matrix);
   jx_SeqVectorDestroy(ser_rhs);
   jx_SeqVectorDestroy(ser_sol);
   
   jx_printf("\n\n zzy kdim = %d  data saved!!\n\n", k_dim);
   jx_printf(" Continue to solve? Type 'enter' to go on!!\n\n"); getchar();
#endif 

   (gmres_data -> converged) = 0;
   
  /*------------------------------------------------------------------------
   * With relative change convergence test on, it is possible to attempt
   * another iteration with a zero residual. This causes the parameter
   * alpha to go NaN. The guard_zero_residual parameter is to circumvent
   * this. Perhaps it should be set to something non-zero (but small).
   *-----------------------------------------------------------------------*/
   
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(matA, &my_id, &num_procs);
   
   if ( logging > 0 || print_level > 0 )
   {
      norms = (gmres_data -> norms);
   }

   /* initialize work arrays */
   rs = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions); 
   c  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   s  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   if (rel_change) 
   {
      rs_2 = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions);
   }
   

   hh = jx_CTAllocF(JX_Real*, k_dim + 1, gmres_functions); 
   for (i = 0; i < k_dim + 1; i ++)
   {	
      hh[i] = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(vecB, vecP[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecP[0]);

   b_norm = sqrt((*(gmres_functions->InnerProd))(vecB, vecB));

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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   r_norm = sqrt((*(gmres_functions->InnerProd))(vecP[0], vecP[0]));
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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   /* peghoty  2010/06/23 */
   normdown = r_norm;

   if ( logging > 0 || print_level > 0 )
   {
      norms[0] = r_norm;
      if ( print_level > 0 && my_id == 0 )
      {
  	 jx_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            jx_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         jx_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   }


   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
   epsilon = jx_max(a_tol, r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level > 0 && my_id == 0 )
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


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;


   /* outer iteration cycle */
   
   while (iter < max_iter)
   {
        /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           jx_TFreeF(c, gmres_functions); 
           jx_TFreeF(s, gmres_functions); 
           jx_TFreeF(rs, gmres_functions);
           if (rel_change)  
           {
              jx_TFreeF(rs_2,gmres_functions);
           }
           for (i = 0; i < k_dim + 1; i ++) 
           {
              jx_TFreeF(hh[i], gmres_functions);
           }
           jx_TFreeF(hh, gmres_functions); 
	   return jx_error_flag;
	}

        /* see if we are already converged and should print the final norm and exit */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           if (!rel_change) /* shouldn't exit after no iterations if relative change is on */
           {
              (*(gmres_functions->CopyVector))(vecB, vecR);
              (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
              r_norm = sqrt((*(gmres_functions->InnerProd))(vecR, vecR));
              if (r_norm <= epsilon)
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf("\n\n");
                    jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (gmres_data -> converged) = 1; /* added by peghoty, 2011/11/08 */
                 break;
              }
              else
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf(" false convergence 1:");
                    if (b_norm > 0.0)
                    {
                       jx_printf(" ||r|| = %.2le  ||r||/||b|| = %.2le\n", r_norm, r_norm / b_norm);
                    }
                    else
                    {
                       jx_printf(" ||r|| = %.2le  ||r||/||r_0|| = %.2le\n", r_norm, r_norm / r_norm_0);
                    }
                 }
                 
                 /* added by Xu Xiaowen, 2011/11/07 */
                 if (is_check_restarted == 0) 
                 {
                    break;
                 }
              }
           }
	}


      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t, vecP[0]);
	i = 0;

        /*** RESTART CYCLE (right-preconditioning) ***/
        
        while (i < k_dim && iter < max_iter)
	{
           i ++;
           iter ++;
           (*(gmres_functions->ClearVector))(vecR);
           //jx_printf("precond: 1\n");
           precond(precond_data, preOperator, vecP[i-1], vecR);
           (*(gmres_functions->Matvec))(matvec_data, 1.0, matA, vecR, 0.0, vecP[i]);
           
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = (*(gmres_functions->InnerProd))(vecP[j], vecP[i]);
              (*(gmres_functions->Axpy))(-hh[j][i-1],vecP[j], vecP[i]);
           }
           t = sqrt((*(gmres_functions->InnerProd))(vecP[i], vecP[i]));
           hh[i][i-1] = t;
           if (t != 0.0)
           {
              t = 1.0 / t;
              (*(gmres_functions->ScaleVector))(t, vecP[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step. update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1] = hh[i-1][i-1]/gamma;
           s[i-1] = hh[i][i-1]/gamma;
           rs[i] = -hh[i][i-1]*rs[i-1];
           rs[i] /=  gamma;
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* peghoty 2010/06/23 */
           normup = r_norm;
           conv_factor = normup / normdown;
           //jx_printf("\n conv_factor = %lf\n\n",conv_factor);
           normdown = normup;

           /* print ? */
           if ( print_level > 0 )
           {
              norms[iter] = r_norm;
              if ( print_level > 0 && my_id == 0 )
              {
                 if (b_norm > 0.0)
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                               norms[iter]/norms[iter-1],norms[iter]/b_norm);
                 }
                 else
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                               norms[iter]/norms[iter-1], norms[iter]/r_norm_0);
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
 
           /* convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0 / (2.0*iter));
              
              weight   = fabs(cf_ave_1 - cf_ave_0);
              weight   = weight / jx_max(cf_ave_1, cf_ave_0);
              weight   = 1.0 - weight;
              
              if (weight * cf_ave_1 > cf_tol) 
              {
                 break_value = 1;
                 break;
              }
           }
           
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              if (rel_change && !rel_change_passed)
              {
                 
                 /* To decide whether to break here: to actually
                  determine the relative change requires the approx
                  solution (so a triangular solve) and a
                  precond. solve - so if we have to do this many
                  times, it will be expensive...(unlike cg where is
                  is relatively straightforward)

                  previously, the intent (there was a bug), was to
                  exit the restart cycle based on the residual norm
                  and check the relative change outside the cycle.
                  Here we will check the relative here as we don't
                  want to exit the restart cycle prematurely */
                 
                 for (k = 0; k < i; k ++) /* extra copy of rs so we don't need to change the later solve */
                 {
                    rs_2[k] = rs[k];
                 }

                 /* solve tri. system */
                 rs_2[i-1] = rs_2[i-1] / hh[i-1][i-1];
                 for (k = i - 2; k >= 0; k --)
                 {
                    t = 0.0;
                    for (j = k + 1; j < i; j ++)
                    {
                       t -= hh[k][j]*rs_2[j];
                    }
                    t += rs_2[k];
                    rs_2[k] = t / hh[k][k];
                 }
                 
                 (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
                 (*(gmres_functions->ScaleVector))(rs_2[i-1], vecW);
                 for (j = i - 2; j >= 0; j --)
                 {
                    (*(gmres_functions->Axpy))(rs_2[j], vecP[j], vecW);
                 }
                    
                 (*(gmres_functions->ClearVector))(vecR);
                 
                 /* find correction (in r) */
                 //jx_printf("precond: 2\n");
                 precond(precond_data, preOperator, vecW, vecR);
                 
                 /* copy current solution (x) to w (don't want to over-write x) */
                 (*(gmres_functions->CopyVector))(vecX, vecW);

                 /* add the correction */
                 (*(gmres_functions->Axpy))(1.0, vecR, vecW);

                 /* now w is the approx solution  - get the norm*/
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(vecW, vecW) );

                 if ( !(x_norm <= guard_zero_residual) ) /* don't divide by zero */
                 {  
                    /* now get x_i - x_i-1 */
                    if (num_rel_change_check)
                    {
                       /* have already checked once so we can avoid another precond. solve */
                       (*(gmres_functions->CopyVector))(vecW, vecR);
                       (*(gmres_functions->Axpy))(-1.0, vecW2, vecR);
                       
                       /* now r contains x_i - x_i-1*/

                       /* save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(vecW, vecW2);
                    }
                    else
                    {
                       /* first time to check rel change*/

                       /* first save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(vecW, vecW2);

                       /* for relative change take x_(i-1) to be 
                          x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                          Now
                          x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                          - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                          = M^{-1} rs_{i-1}{p_{i-1}} */
                       
                       (*(gmres_functions->ClearVector))(vecW);
                       (*(gmres_functions->Axpy))(rs_2[i-1], vecP[i-1], vecW);
                       (*(gmres_functions->ClearVector))(vecR);
                       /* apply the preconditioner */
                       //jx_printf("precond: 3\n");
                       precond(precond_data, preOperator, vecW, vecR);
                       /* now r contains x_i - x_i-1 */          
                    }
                    
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );
                    relative_error = w_norm / x_norm;
                    if (relative_error <= r_tol)
                    {
                       rel_change_passed = 1;
                       break;
                    }
                 }
                 else
                 {
                    rel_change_passed = 1;
                    break;

                 }
                 num_rel_change_check ++;
              }
              else /* no relative change */
              {
                 break;
              }
           }
           
	} /*** end of restart cycle ***/

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i - 2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k + 1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
        (*(gmres_functions->ScaleVector))(rs[i-1], vecW);
        for (j = i - 2; j >= 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j], vecP[j], vecW);
        }

	(*(gmres_functions->ClearVector))(vecR);
	
	/* find correction (in r) */
        //jx_printf("precond: 4\n");
        precond(precond_data, preOperator, vecW, vecR);

        /* update current solution x (in x) */
	(*(gmres_functions->Axpy))(1.0, vecR, vecX);
         

        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm */
           (*(gmres_functions->CopyVector))(vecB, vecR);
           (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
           r_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );

           if (r_norm <= epsilon)
           {
              if (rel_change && !rel_change_passed) /* calculate the relative change */
              {

                 /* calculate the norm of the solution */
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(vecX,vecX) );
               
                 if ( !(x_norm <= guard_zero_residual )) /* don't divide by zero */
                 {
                    /* for relative change take x_(i-1) to be 
                       x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                       Now
                       x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                       - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                       = M^{-1} rs_{i-1}{p_{i-1}} */
                    (*(gmres_functions->ClearVector))(vecW);
                    (*(gmres_functions->Axpy))(rs[i-1], vecP[i-1], vecW);
                    (*(gmres_functions->ClearVector))(vecR);
                    
                    /* apply the preconditioner */
                    //jx_printf("precond: 5\n");
                    precond(precond_data, preOperator, vecW, vecR);
                    
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );
                    relative_error= w_norm / x_norm;
                    if ( relative_error < r_tol )
                    {
                       (gmres_data -> converged) = 1;
                       if ( print_level > 0 && my_id == 0 )
                       {
                          jx_printf("\n");
                          jx_printf("Final L2 norm of residual: %e\n", r_norm);
                       }
                       break;
                    }
                 }
                 else
                 {
                    (gmres_data -> converged) = 1;
                    if ( print_level > 0 && my_id == 0 )
                    {
                       jx_printf("\n");
                       jx_printf("Final L2 norm of residual: %e\n", r_norm);
                    }
                    break;
                 }

              }
              else /* don't need to check rel. change */
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf("\n");
                    jx_printf("Final L2 norm of residual: %e\n", r_norm);
                 }
                 (gmres_data -> converged) = 1;
                 break;
              }
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if ( print_level > 0 && my_id == 0)
              {
                 jx_printf(" false convergence 2:");
                 if (b_norm > 0.0)
                 {
                    jx_printf(" ||r|| = %.2le  ||r||/||b|| = %.2le\n", r_norm, r_norm / b_norm);
                 }
                 else
                 {
                    jx_printf(" ||r|| = %.2le  ||r||/||r_0|| = %.2le\n", r_norm, r_norm / r_norm_0);
                 }
              }
              
              /* modified by Xu Xiaowen, 2011/11/07 */
              if (is_check_restarted == 1) 
              {
                 (*(gmres_functions->CopyVector))(vecR, vecP[0]);
                 i = 0;
              } 
              else 
              {
                 break;  
              }
           }
	} /* end of convergence check */

        if (iter > 1 && conv_factor > convfac_threshold)
        {
           break;
        }

        /* compute residual vector and continue loop */
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j] = c[j-1]*rs[j];
	}
        
        if (i) 
        {
           (*(gmres_functions->Axpy))(rs[i]-1.0, vecP[i], vecP[i]);
        }
        
        for (j = i - 1 ; j > 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j],vecP[j],vecP[i]);
        }
        
        if (i)
        {
           (*(gmres_functions->Axpy))(rs[0]-1.0, vecP[0], vecP[0]);
           (*(gmres_functions->Axpy))(1.0, vecP[i], vecP[0]);
        }
        
   } /* END of iteration while loop */


   if ( print_level > 1 && my_id == 0 ) jx_printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   
   if (b_norm > 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   
   if (b_norm == 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / r_norm_0;
   }

#if 0
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf(" Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
      jx_error(JX_ERROR_CONV);
   }
#endif 

#if 0
   /* 保存解向量 */   
   jx_Vector *ser_app = NULL;
   ser_app = jx_ParVectorToVectorAll ((jx_ParVector *)vecX);
   jx_SeqVectorPrint(ser_app, "./u_ser"); 
   jx_SeqVectorDestroy(ser_app);
   jx_printf("\n\n zzy solution has been saved!!\n\n");
#endif

#if 0
   /* 修改解向量 */ 
   JX_Int change_sol;
   jx_printf("\n");
   jx_printf(" Do you want the modify the solution with the file './app'?\n");
   jx_printf(" >>> 0: NO\n");
   jx_printf(" >>> 1: YES\n");
   jx_scanf("%d", &change_sol); 
   if (change_sol == 1) 
   {
      FILE *fp = NULL;
      jx_Vector *tmp_app = NULL;
      JX_Int ii, num_rows;
      tmp_app = jx_ParVectorLocalVector((jx_ParVector *)vecX);
   
      fp = fopen("./app", "r");
      jx_fscanf(fp, "%d", &num_rows);
      for (ii = 0; ii < num_rows; ii ++)
      {
         jx_fscanf(fp, "%le", &jx_VectorData(tmp_app)[ii]);
      }
      jx_printf("\n\n zzy solution has been changed for test!!\n\n");
   }
#endif
  
   jx_TFreeF(c,gmres_functions); 
   jx_TFreeF(s,gmres_functions); 
   jx_TFreeF(rs,gmres_functions);
   if (rel_change)  
   {
      jx_TFreeF(rs_2,gmres_functions);
   }

   for (i = 0; i < k_dim + 1; i ++)
   {	
      jx_TFreeF(hh[i], gmres_functions);
   }
   jx_TFreeF(hh, gmres_functions); 

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESAdaptiveSolve
 * \brief Adaptive between AMG, ILU(0) and combined preconditioner
 * \author Yue Xiaoqiang
 * \date 2014/03/24
 */
JX_Int
jx_GMRESAdaptiveSolve( void *gmres_vdata, void *preOperator, void *matA, void *vecB, void *vecX )
{
   jx_GMRESData      *gmres_data      = gmres_vdata;
   jx_GMRESFunctions *gmres_functions = (gmres_data -> functions);
   JX_Int 		     k_dim            = (gmres_data -> k_dim);
   JX_Int               min_iter         = (gmres_data -> min_iter);
   JX_Int 		     max_iter         = (gmres_data -> max_iter);
   JX_Int               rel_change       = (gmres_data -> rel_change);
   JX_Real 	     r_tol            = (gmres_data -> tol);
   JX_Real 	     cf_tol           = (gmres_data -> cf_tol);
   JX_Real            a_tol            = (gmres_data -> a_tol);
   void             *matvec_data      = (gmres_data -> matvec_data);

   JX_Real            convfac_threshold   = (gmres_data -> convfac_threshold); // peghoty 2010/06/23
   JX_Real            resdown_0_threshold = (gmres_data -> resdown_0_threshold); /* Yue Xiaoqiang, 2014/03/24 */
   JX_Real            convfac_threshold_2 = (gmres_data -> convfac_threshold_2); /* Yue Xiaoqiang, 2014/03/24 */

   void             *vecR             = (gmres_data -> r);
   void             *vecW             = (gmres_data -> w);
   void             *vecW2            = (gmres_data -> w_2); /* vecW2 is only allocated if rel_change = 1 */
   void            **vecP             = (gmres_data -> p);
   JX_Int 	           (*precond)()       = (gmres_functions -> precond);
   JX_Int 	            *precond_data     = (gmres_data -> precond_data);
   JX_Int             print_level        = (gmres_data -> print_level);
   JX_Int             logging            = (gmres_data -> logging);
   JX_Real         *norms              = (gmres_data -> norms);
   JX_Int             is_check_restarted = (gmres_data -> is_check_restarted); /* Xu Xiaowen, 2011/11/08 */
   
   JX_Int          *part_aux  = NULL;
   JX_Int          *part_res  = NULL;
   jx_ParVector *aux_vec   = NULL;
   jx_ParVector *res_vec   = NULL;
   JX_Solver     amg_data  = NULL;
   JX_Int amg_max_levels, amg_relax_type, amg_print_level;
   JX_Int amg_P_max_elmts, amg_measure_type, amg_interp_type;
   JX_Int amg_coarsen_type, amg_agg_num_levels, amg_coarse_threshold;
   JX_Real amg_strong_threshold;
   
   JX_Int        break_value = 0;
   JX_Int        break_adaptive = 0;
   JX_Real     r_current_norm;
   JX_Int	      i, j, k;
   JX_Real     *rs, **hh, *c, *s, *rs_2 = NULL; 
   JX_Int        iter; 
   JX_Int        my_id, num_procs;
   JX_Real     epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   JX_Real     w_norm;
   
   JX_Real     epsmac = 1.e-16; 
   JX_Real     ieee_check = 0.0;

   JX_Real     guard_zero_residual; 
   JX_Real     cf_ave_0 = 0.0;
   JX_Real     cf_ave_1 = 0.0;
   JX_Real     weight;
   JX_Real     r_norm_0;
   JX_Real     relative_error = 1.0;

   JX_Int        rel_change_passed = 0, num_rel_change_check = 0;

   /* peghoty 2010/06/23 */
   JX_Real  normup       = 0.0;
   JX_Real  normdown     = 0.0;
   JX_Real  conv_factor  = 0.0;

#if 0
   /* 保存离散系统和初始解向量 */
   jx_CSRMatrix *ser_matrix = NULL;
   ser_matrix = jx_ParCSRMatrixToCSRMatrixAll((jx_ParCSRMatrix *)matA);
   jx_CSRMatrixPrint(ser_matrix,"./A_ser");
   
   jx_Vector *ser_rhs = NULL;
   ser_rhs = jx_ParVectorToVectorAll ((jx_ParVector *)vecB);
   jx_SeqVectorPrint(ser_rhs, "./f_ser");
   
   jx_Vector *ser_sol = NULL;
   ser_sol = jx_ParVectorToVectorAll ((jx_ParVector *)vecX);
   jx_SeqVectorPrint(ser_sol, "./u0_ser"); 

   jx_CSRMatrixDestroy(ser_matrix);
   jx_SeqVectorDestroy(ser_rhs);
   jx_SeqVectorDestroy(ser_sol);
   
   jx_printf("\n\n zzy kdim = %d  data saved!!\n\n", k_dim);
   jx_printf(" Continue to solve? Type 'enter' to go on!!\n\n"); getchar();
#endif

   if (jx_CombinedPrecondDataPreID((jx_CombinedPrecondData *)precond_data) == 3) // regular PGMRES
   {

   (gmres_data -> converged) = 0;
   
  /*------------------------------------------------------------------------
   * With relative change convergence test on, it is possible to attempt
   * another iteration with a zero residual. This causes the parameter
   * alpha to go NaN. The guard_zero_residual parameter is to circumvent
   * this. Perhaps it should be set to something non-zero (but small).
   *-----------------------------------------------------------------------*/
   
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(matA, &my_id, &num_procs);
   
   if ( logging > 0 || print_level > 0 )
   {
      norms = (gmres_data -> norms);
   }

   /* initialize work arrays */
   rs = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions); 
   c  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   s  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   if (rel_change) 
   {
      rs_2 = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions);
   }
   

   hh = jx_CTAllocF(JX_Real*, k_dim + 1, gmres_functions); 
   for (i = 0; i < k_dim + 1; i ++)
   {	
      hh[i] = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(vecB, vecP[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecP[0]);

   b_norm = sqrt((*(gmres_functions->InnerProd))(vecB, vecB));

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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   r_norm = sqrt((*(gmres_functions->InnerProd))(vecP[0], vecP[0]));
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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   /* peghoty  2010/06/23 */
   normdown = r_norm;

   if ( logging > 0 || print_level > 0 )
   {
      norms[0] = r_norm;
      if ( print_level > 0 && my_id == 0 )
      {
  	 jx_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            jx_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         jx_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   }


   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
   epsilon = jx_max(a_tol, r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level > 0 && my_id == 0 )
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


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;


   /* outer iteration cycle */
   
   while (iter < max_iter)
   {
        /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           jx_TFreeF(c, gmres_functions); 
           jx_TFreeF(s, gmres_functions); 
           jx_TFreeF(rs, gmres_functions);
           if (rel_change)  
           {
              jx_TFreeF(rs_2,gmres_functions);
           }
           for (i = 0; i < k_dim + 1; i ++) 
           {
              jx_TFreeF(hh[i], gmres_functions);
           }
           jx_TFreeF(hh, gmres_functions); 
	   return jx_error_flag;
	}

        /* see if we are already converged and should print the final norm and exit */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           if (!rel_change) /* shouldn't exit after no iterations if relative change is on */
           {
              (*(gmres_functions->CopyVector))(vecB, vecR);
              (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
              r_norm = sqrt((*(gmres_functions->InnerProd))(vecR, vecR));
              if (r_norm <= epsilon)
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf("\n\n");
                    jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (gmres_data -> converged) = 1; /* added by peghoty, 2011/11/08 */
                 break;
              }
              else
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf(" false convergence 1:");
                    if (b_norm > 0.0)
                    {
                       jx_printf(" ||r|| = %.2le  ||r||/||b|| = %.2le\n", r_norm, r_norm / b_norm);
                    }
                    else
                    {
                       jx_printf(" ||r|| = %.2le  ||r||/||r_0|| = %.2le\n", r_norm, r_norm / r_norm_0);
                    }
                 }
                 
                 /* added by Xu Xiaowen, 2011/11/07 */
                 if (is_check_restarted == 0) 
                 {
                    break;
                 }
              }
           }
	}


      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t, vecP[0]);
	i = 0;

        /*** RESTART CYCLE (right-preconditioning) ***/
        
        while (i < k_dim && iter < max_iter)
	{
           i ++;
           iter ++;
           (*(gmres_functions->ClearVector))(vecR);
           //jx_printf("precond: 1\n");
           precond(precond_data, preOperator, vecP[i-1], vecR);
           (*(gmres_functions->Matvec))(matvec_data, 1.0, matA, vecR, 0.0, vecP[i]);
           
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = (*(gmres_functions->InnerProd))(vecP[j], vecP[i]);
              (*(gmres_functions->Axpy))(-hh[j][i-1],vecP[j], vecP[i]);
           }
           t = sqrt((*(gmres_functions->InnerProd))(vecP[i], vecP[i]));
           hh[i][i-1] = t;
           if (t != 0.0)
           {
              t = 1.0 / t;
              (*(gmres_functions->ScaleVector))(t, vecP[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step. update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1] = hh[i-1][i-1]/gamma;
           s[i-1] = hh[i][i-1]/gamma;
           rs[i] = -hh[i][i-1]*rs[i-1];
           rs[i] /=  gamma;
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* peghoty 2010/06/23 */
           normup = r_norm;
           conv_factor = normup / normdown;
           //jx_printf("\n conv_factor = %lf\n\n",conv_factor);
           normdown = normup;

           /* print ? */
           if ( print_level > 0 )
           {
              norms[iter] = r_norm;
              if ( print_level > 0 && my_id == 0 )
              {
                 if (b_norm > 0.0)
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                               norms[iter]/norms[iter-1],norms[iter]/b_norm);
                 }
                 else
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                               norms[iter]/norms[iter-1], norms[iter]/r_norm_0);
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
 
           /* convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0 / (2.0*iter));
              
              weight   = fabs(cf_ave_1 - cf_ave_0);
              weight   = weight / jx_max(cf_ave_1, cf_ave_0);
              weight   = 1.0 - weight;
              
              if (weight * cf_ave_1 > cf_tol) 
              {
                 break_value = 1;
                 break;
              }
           }
           
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              if (rel_change && !rel_change_passed)
              {
                 
                 /* To decide whether to break here: to actually
                  determine the relative change requires the approx
                  solution (so a triangular solve) and a
                  precond. solve - so if we have to do this many
                  times, it will be expensive...(unlike cg where is
                  is relatively straightforward)

                  previously, the intent (there was a bug), was to
                  exit the restart cycle based on the residual norm
                  and check the relative change outside the cycle.
                  Here we will check the relative here as we don't
                  want to exit the restart cycle prematurely */
                 
                 for (k = 0; k < i; k ++) /* extra copy of rs so we don't need to change the later solve */
                 {
                    rs_2[k] = rs[k];
                 }

                 /* solve tri. system */
                 rs_2[i-1] = rs_2[i-1] / hh[i-1][i-1];
                 for (k = i - 2; k >= 0; k --)
                 {
                    t = 0.0;
                    for (j = k + 1; j < i; j ++)
                    {
                       t -= hh[k][j]*rs_2[j];
                    }
                    t += rs_2[k];
                    rs_2[k] = t / hh[k][k];
                 }
                 
                 (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
                 (*(gmres_functions->ScaleVector))(rs_2[i-1], vecW);
                 for (j = i - 2; j >= 0; j --)
                 {
                    (*(gmres_functions->Axpy))(rs_2[j], vecP[j], vecW);
                 }
                    
                 (*(gmres_functions->ClearVector))(vecR);
                 
                 /* find correction (in r) */
                 //jx_printf("precond: 2\n");
                 precond(precond_data, preOperator, vecW, vecR);
                 
                 /* copy current solution (x) to w (don't want to over-write x) */
                 (*(gmres_functions->CopyVector))(vecX, vecW);

                 /* add the correction */
                 (*(gmres_functions->Axpy))(1.0, vecR, vecW);

                 /* now w is the approx solution  - get the norm*/
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(vecW, vecW) );

                 if ( !(x_norm <= guard_zero_residual) ) /* don't divide by zero */
                 {  
                    /* now get x_i - x_i-1 */
                    if (num_rel_change_check)
                    {
                       /* have already checked once so we can avoid another precond. solve */
                       (*(gmres_functions->CopyVector))(vecW, vecR);
                       (*(gmres_functions->Axpy))(-1.0, vecW2, vecR);
                       
                       /* now r contains x_i - x_i-1*/

                       /* save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(vecW, vecW2);
                    }
                    else
                    {
                       /* first time to check rel change*/

                       /* first save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(vecW, vecW2);

                       /* for relative change take x_(i-1) to be 
                          x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                          Now
                          x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                          - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                          = M^{-1} rs_{i-1}{p_{i-1}} */
                       
                       (*(gmres_functions->ClearVector))(vecW);
                       (*(gmres_functions->Axpy))(rs_2[i-1], vecP[i-1], vecW);
                       (*(gmres_functions->ClearVector))(vecR);
                       /* apply the preconditioner */
                       //jx_printf("precond: 3\n");
                       precond(precond_data, preOperator, vecW, vecR);
                       /* now r contains x_i - x_i-1 */          
                    }
                    
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );
                    relative_error = w_norm / x_norm;
                    if (relative_error <= r_tol)
                    {
                       rel_change_passed = 1;
                       break;
                    }
                 }
                 else
                 {
                    rel_change_passed = 1;
                    break;

                 }
                 num_rel_change_check ++;
              }
              else /* no relative change */
              {
                 break;
              }
           }
           
	} /*** end of restart cycle ***/

        if (iter > 1 && conv_factor > convfac_threshold)
        {
           break;
        }

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i - 2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k + 1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
        (*(gmres_functions->ScaleVector))(rs[i-1], vecW);
        for (j = i - 2; j >= 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j], vecP[j], vecW);
        }

	(*(gmres_functions->ClearVector))(vecR);
	
	/* find correction (in r) */
        //jx_printf("precond: 4\n");
        precond(precond_data, preOperator, vecW, vecR);

        /* update current solution x (in x) */
	(*(gmres_functions->Axpy))(1.0, vecR, vecX);
         

        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm */
           (*(gmres_functions->CopyVector))(vecB, vecR);
           (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
           r_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );

           if (r_norm <= epsilon)
           {
              if (rel_change && !rel_change_passed) /* calculate the relative change */
              {

                 /* calculate the norm of the solution */
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(vecX,vecX) );
               
                 if ( !(x_norm <= guard_zero_residual )) /* don't divide by zero */
                 {
                    /* for relative change take x_(i-1) to be 
                       x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                       Now
                       x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                       - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                       = M^{-1} rs_{i-1}{p_{i-1}} */
                    (*(gmres_functions->ClearVector))(vecW);
                    (*(gmres_functions->Axpy))(rs[i-1], vecP[i-1], vecW);
                    (*(gmres_functions->ClearVector))(vecR);
                    
                    /* apply the preconditioner */
                    //jx_printf("precond: 5\n");
                    precond(precond_data, preOperator, vecW, vecR);
                    
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );
                    relative_error= w_norm / x_norm;
                    if ( relative_error < r_tol )
                    {
                       (gmres_data -> converged) = 1;
                       if ( print_level > 0 && my_id == 0 )
                       {
                          jx_printf("\n");
                          jx_printf("Final L2 norm of residual: %e\n", r_norm);
                       }
                       break;
                    }
                 }
                 else
                 {
                    (gmres_data -> converged) = 1;
                    if ( print_level > 0 && my_id == 0 )
                    {
                       jx_printf("\n");
                       jx_printf("Final L2 norm of residual: %e\n", r_norm);
                    }
                    break;
                 }

              }
              else /* don't need to check rel. change */
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf("\n");
                    jx_printf("Final L2 norm of residual: %e\n", r_norm);
                 }
                 (gmres_data -> converged) = 1;
                 break;
              }
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if ( print_level > 0 && my_id == 0)
              {
                 jx_printf(" false convergence 2:");
                 if (b_norm > 0.0)
                 {
                    jx_printf(" ||r|| = %.2le  ||r||/||b|| = %.2le\n", r_norm, r_norm / b_norm);
                 }
                 else
                 {
                    jx_printf(" ||r|| = %.2le  ||r||/||r_0|| = %.2le\n", r_norm, r_norm / r_norm_0);
                 }
              }
              
              /* modified by Xu Xiaowen, 2011/11/07 */
              if (is_check_restarted == 1) 
              {
                 (*(gmres_functions->CopyVector))(vecR, vecP[0]);
                 i = 0;
              } 
              else 
              {
                 break;  
              }
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j] = c[j-1]*rs[j];
	}
        
        if (i) 
        {
           (*(gmres_functions->Axpy))(rs[i]-1.0, vecP[i], vecP[i]);
        }
        
        for (j = i - 1 ; j > 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j],vecP[j],vecP[i]);
        }
        
        if (i)
        {
           (*(gmres_functions->Axpy))(rs[0]-1.0, vecP[0], vecP[0]);
           (*(gmres_functions->Axpy))(1.0, vecP[i], vecP[0]);
        }
        
   } /* END of iteration while loop */


   if ( print_level > 1 && my_id == 0 ) jx_printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   (gmres_data -> comp_amg_iter)  = iter;
   (gmres_data -> break_adaptive) = break_adaptive;
   
   if (b_norm > 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   
   if (b_norm == 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / r_norm_0;
   }

   }
   else // adaptive PGMRES
   {

   (gmres_data -> converged) = 0;
   
   MPI_Comm comm = jx_CombinedPrecondDataComm((jx_CombinedPrecondData *)precond_data);
   jx_MPI_Comm_size(comm, &num_procs);

  /*------------------------------------------------------------------------
   * With relative change convergence test on, it is possible to attempt
   * another iteration with a zero residual. This causes the parameter
   * alpha to go NaN. The guard_zero_residual parameter is to circumvent
   * this. Perhaps it should be set to something non-zero (but small).
   *-----------------------------------------------------------------------*/
   
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(matA, &my_id, &num_procs);
   
   if ( logging > 0 || print_level > 0 )
   {
      norms = (gmres_data -> norms);
   }

   /* initialize work arrays */
   rs = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions); 
   c  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   s  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   if (rel_change) 
   {
      rs_2 = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions);
   }
   

   hh = jx_CTAllocF(JX_Real*, k_dim + 1, gmres_functions); 
   for (i = 0; i < k_dim + 1; i ++)
   {	
      hh[i] = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(vecB, vecP[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecP[0]);

   b_norm = sqrt((*(gmres_functions->InnerProd))(vecB, vecB));

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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESAdaptiveSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   r_norm = sqrt((*(gmres_functions->InnerProd))(vecP[0], vecP[0]));
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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESAdaptiveSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   /* peghoty  2010/06/23 */
   normdown = r_norm;

   if ( logging > 0 || print_level > 0 )
   {
      norms[0] = r_norm;
      if ( print_level > 0 && my_id == 0 )
      {
  	 jx_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            jx_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         jx_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   }


   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
   epsilon = jx_max(a_tol, r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level > 0 && my_id == 0 )
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


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;


   /* outer iteration cycle */
   
   while (iter < max_iter)
   {
        /* initialize first term of hessenberg system */

	rs[0] = r_norm;
	r_current_norm = r_norm;
        if (r_norm == 0.0)
        {
           jx_TFreeF(c, gmres_functions); 
           jx_TFreeF(s, gmres_functions); 
           jx_TFreeF(rs, gmres_functions);
           if (rel_change)  
           {
              jx_TFreeF(rs_2,gmres_functions);
           }
           for (i = 0; i < k_dim + 1; i ++) 
           {
              jx_TFreeF(hh[i], gmres_functions);
           }
           jx_TFreeF(hh, gmres_functions); 
	   return jx_error_flag;
	}

        /* see if we are already converged and should print the final norm and exit */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           if (!rel_change) /* shouldn't exit after no iterations if relative change is on */
           {
              (*(gmres_functions->CopyVector))(vecB, vecR);
              (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
              r_norm = sqrt((*(gmres_functions->InnerProd))(vecR, vecR));
              if (r_norm <= epsilon)
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf("\n\n");
                    jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (gmres_data -> converged) = 1; /* added by peghoty, 2011/11/08 */
                 break;
              }
              else
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf(" false convergence 1:");
                    if (b_norm > 0.0)
                    {
                       jx_printf(" ||r|| = %.2le  ||r||/||b|| = %.2le\n", r_norm, r_norm / b_norm);
                    }
                    else
                    {
                       jx_printf(" ||r|| = %.2le  ||r||/||r_0|| = %.2le\n", r_norm, r_norm / r_norm_0);
                    }
                 }
                 
                 /* added by Xu Xiaowen, 2011/11/07 */
                 if (is_check_restarted == 0) 
                 {
                    break;
                 }
              }
           }
	}

      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t, vecP[0]);
	i = 0;

        /*** RESTART CYCLE (right-preconditioning) ***/
        
        while (i < k_dim && iter < max_iter)
	{
           // here AMG SETUP phase
           if (break_adaptive == 1)
           {
           amg_max_levels = jx_CombinedPrecondDataAMGMaxLevels((jx_CombinedPrecondData *)precond_data);
           amg_relax_type = jx_CombinedPrecondDataAMGRelaxType((jx_CombinedPrecondData *)precond_data);
           amg_print_level = jx_CombinedPrecondDataAMGPrintLevel((jx_CombinedPrecondData *)precond_data);
           amg_interp_type = jx_CombinedPrecondDataAMGInterpType((jx_CombinedPrecondData *)precond_data);
           amg_P_max_elmts = jx_CombinedPrecondDataAMGPMaxElmts((jx_CombinedPrecondData *)precond_data);
           amg_measure_type = jx_CombinedPrecondDataAMGMeasureType((jx_CombinedPrecondData *)precond_data);
           amg_coarsen_type = jx_CombinedPrecondDataAMGCoarsenType((jx_CombinedPrecondData *)precond_data);
           amg_agg_num_levels = jx_CombinedPrecondDataAMGAggNumLevels((jx_CombinedPrecondData *)precond_data);
           amg_coarse_threshold = jx_CombinedPrecondDataAMGCoarseThreshold((jx_CombinedPrecondData *)precond_data);
           amg_strong_threshold = jx_CombinedPrecondDataAMGStrongThreshold((jx_CombinedPrecondData *)precond_data);
              JX_PAMGCreate(&amg_data);
              JX_PAMGSetMaxLevels(amg_data, amg_max_levels);
              JX_PAMGSetMaxIter(amg_data, 1);
              JX_PAMGSetMeasureType(amg_data, amg_measure_type);
              JX_PAMGSetCoarsenType(amg_data, amg_coarsen_type);
              JX_PAMGSetInterpType(amg_data, amg_interp_type);
              JX_PAMGSetPMaxElmts(amg_data, amg_P_max_elmts);
              JX_PAMGSetAggNumLevels(amg_data, amg_agg_num_levels);
              JX_PAMGSetStrongThreshold(amg_data, amg_strong_threshold);
              JX_PAMGSetPrintLevel(amg_data, amg_print_level);
              JX_PAMGSetCoarseThreshold(amg_data, amg_coarse_threshold);
              JX_PAMGSetCycleNumSweeps(amg_data, 1, 1);
              JX_PAMGSetCycleNumSweeps(amg_data, 1, 2);
              JX_PAMGSetCycleNumSweeps(amg_data, 1, 3);
              JX_PAMGSetCycleRelaxType(amg_data, amg_relax_type, 1);
              JX_PAMGSetCycleRelaxType(amg_data, amg_relax_type, 2);
              JX_PAMGSetCycleRelaxType(amg_data, 9, 3);
              JX_PAMGSetup(amg_data, (JX_ParCSRMatrix)preOperator);
              // Allocate memory for auxiliary and residual vectors
              jx_ParCSRMatrixGetRowPartitioning((jx_ParCSRMatrix *)preOperator, &part_aux);
              aux_vec = jx_ParVectorCreate(jx_ParCSRMatrixComm((jx_ParCSRMatrix *)preOperator),
                             jx_ParCSRMatrixGlobalNumRows((jx_ParCSRMatrix *)preOperator), part_aux);
              jx_ParVectorInitialize(aux_vec);
              jx_ParCSRMatrixGetRowPartitioning((jx_ParCSRMatrix *)preOperator, &part_res);
              res_vec = jx_ParVectorCreate(jx_ParCSRMatrixComm((jx_ParCSRMatrix *)preOperator),
                             jx_ParCSRMatrixGlobalNumRows((jx_ParCSRMatrix *)preOperator), part_res);
              jx_ParVectorInitialize(res_vec);
              break_adaptive = 2;
           }
           
           i ++;
           iter ++;
           (*(gmres_functions->ClearVector))(vecR);
           //jx_printf("precond: 1\n");
           precond(precond_data, preOperator, vecP[i-1], vecR);
           if (break_adaptive == 2)
           {
              jx_ParVectorCopy(vecP[i-1], res_vec);
              jx_ParCSRMatrixMatvec(-1.0, (jx_ParCSRMatrix *)preOperator, vecR, 1.0, res_vec);
              jx_ParVectorSetConstantValues(aux_vec, 0.0);
              JX_PAMGPrecond(amg_data, (JX_ParCSRMatrix)preOperator, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
              jx_ParVectorAxpy(1.0, aux_vec, vecR);
           }
           (*(gmres_functions->Matvec))(matvec_data, 1.0, matA, vecR, 0.0, vecP[i]);
           
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = (*(gmres_functions->InnerProd))(vecP[j], vecP[i]);
              (*(gmres_functions->Axpy))(-hh[j][i-1], vecP[j], vecP[i]);
           }
           t = sqrt((*(gmres_functions->InnerProd))(vecP[i], vecP[i]));
           hh[i][i-1] = t;
           if (t != 0.0)
           {
              t = 1.0 / t;
              (*(gmres_functions->ScaleVector))(t, vecP[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step. update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1] = hh[i-1][i-1]/gamma;
           s[i-1] = hh[i][i-1]/gamma;
           rs[i] = -hh[i][i-1]*rs[i-1];
           rs[i] /=  gamma;
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if ( print_level > 0 )
           {
              norms[iter] = r_norm;
              if ( print_level > 0 && my_id == 0 )
              {
                 if (b_norm > 0.0)
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                               norms[iter]/norms[iter-1],norms[iter]/b_norm);
                 }
                 else
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                               norms[iter]/norms[iter-1], norms[iter]/r_norm_0);
                 }
              }
           }

           /* peghoty 2010/06/23 */
           normup = r_norm;
           conv_factor = normup / normdown;
           //jx_printf("\n conv_factor = %lf\n\n",conv_factor);
           if (b_norm > 0.0)
           {
               if ((break_adaptive == 0) && ((normup/b_norm > resdown_0_threshold)
                                         || ((iter > 1) && (conv_factor > convfac_threshold_2))))
               {
                  (gmres_data -> comp_ilu_iter) = iter;
                  break_adaptive = 1;
                  i = 0;
                  iter = 0;
                  rs[0] = r_current_norm;
                  continue;
               }
           }
           else
           {
               if ((break_adaptive == 0) && ((normup/r_norm_0 > resdown_0_threshold)
                                         || ((iter > 1) && (conv_factor > convfac_threshold_2))))
               {
                  (gmres_data -> comp_ilu_iter) = iter;
                  break_adaptive = 1;
                  i = 0;
                  iter = 0;
                  rs[0] = r_current_norm;
                  continue;
               }
           }
           normdown = normup;
          
           if (iter > 1 && conv_factor > convfac_threshold)
           {
              if (my_id == 0 && print_level > 0)
              {
                 jx_printf("\n  Warning: iteration has terminated because the\n");
                 jx_printf("           current conv_factor > convfac_threshold(%lf) !!\n\n",convfac_threshold);
              }
              break;
           }
 
           /* convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0 / (2.0*iter));
              
              weight   = fabs(cf_ave_1 - cf_ave_0);
              weight   = weight / jx_max(cf_ave_1, cf_ave_0);
              weight   = 1.0 - weight;
              
              if (weight * cf_ave_1 > cf_tol) 
              {
                 break_value = 1;
                 break;
              }
           }
           
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              if (rel_change && !rel_change_passed)
              {
                 
                 /* To decide whether to break here: to actually
                  determine the relative change requires the approx
                  solution (so a triangular solve) and a
                  precond. solve - so if we have to do this many
                  times, it will be expensive...(unlike cg where is
                  is relatively straightforward)

                  previously, the intent (there was a bug), was to
                  exit the restart cycle based on the residual norm
                  and check the relative change outside the cycle.
                  Here we will check the relative here as we don't
                  want to exit the restart cycle prematurely */
                 
                 for (k = 0; k < i; k ++) /* extra copy of rs so we don't need to change the later solve */
                 {
                    rs_2[k] = rs[k];
                 }

                 /* solve tri. system */
                 rs_2[i-1] = rs_2[i-1] / hh[i-1][i-1];
                 for (k = i - 2; k >= 0; k --)
                 {
                    t = 0.0;
                    for (j = k + 1; j < i; j ++)
                    {
                       t -= hh[k][j]*rs_2[j];
                    }
                    t += rs_2[k];
                    rs_2[k] = t / hh[k][k];
                 }
                 
                 (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
                 (*(gmres_functions->ScaleVector))(rs_2[i-1], vecW);
                 for (j = i - 2; j >= 0; j --)
                 {
                    (*(gmres_functions->Axpy))(rs_2[j], vecP[j], vecW);
                 }
                    
                 (*(gmres_functions->ClearVector))(vecR);
                 
                 /* find correction (in r) */
                 //jx_printf("precond: 2\n");
                 precond(precond_data, preOperator, vecW, vecR);
                 if (break_adaptive == 2)
                 {
                    jx_ParVectorCopy(vecW, res_vec);
                    jx_ParCSRMatrixMatvec(-1.0, (jx_ParCSRMatrix *)preOperator, vecR, 1.0, res_vec);
                    jx_ParVectorSetConstantValues(aux_vec, 0.0);
                    JX_PAMGPrecond(amg_data, (JX_ParCSRMatrix)preOperator,
                                             (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
                    jx_ParVectorAxpy(1.0, aux_vec, vecR);
                 }
                 
                 /* copy current solution (x) to w (don't want to over-write x) */
                 (*(gmres_functions->CopyVector))(vecX, vecW);

                 /* add the correction */
                 (*(gmres_functions->Axpy))(1.0, vecR, vecW);

                 /* now w is the approx solution  - get the norm*/
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(vecW, vecW) );

                 if ( !(x_norm <= guard_zero_residual) ) /* don't divide by zero */
                 {  
                    /* now get x_i - x_i-1 */
                    if (num_rel_change_check)
                    {
                       /* have already checked once so we can avoid another precond. solve */
                       (*(gmres_functions->CopyVector))(vecW, vecR);
                       (*(gmres_functions->Axpy))(-1.0, vecW2, vecR);
                       
                       /* now r contains x_i - x_i-1*/

                       /* save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(vecW, vecW2);
                    }
                    else
                    {
                       /* first time to check rel change*/

                       /* first save current soln w in w_2 for next time */
                       (*(gmres_functions->CopyVector))(vecW, vecW2);

                       /* for relative change take x_(i-1) to be 
                          x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                          Now
                          x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                          - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                          = M^{-1} rs_{i-1}{p_{i-1}} */
                       
                       (*(gmres_functions->ClearVector))(vecW);
                       (*(gmres_functions->Axpy))(rs_2[i-1], vecP[i-1], vecW);
                       (*(gmres_functions->ClearVector))(vecR);
                       /* apply the preconditioner */
                       //jx_printf("precond: 3\n");
                       precond(precond_data, preOperator, vecW, vecR);
                       if (break_adaptive == 2)
                       {
                          jx_ParVectorCopy(vecW, res_vec);
                          jx_ParCSRMatrixMatvec(-1.0, (jx_ParCSRMatrix *)preOperator, vecR, 1.0, res_vec);
                          jx_ParVectorSetConstantValues(aux_vec, 0.0);
                          JX_PAMGPrecond(amg_data, (JX_ParCSRMatrix)preOperator,
                                                   (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
                          jx_ParVectorAxpy(1.0, aux_vec, vecR);
                       }
                       /* now r contains x_i - x_i-1 */          
                    }
                    
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );
                    relative_error = w_norm / x_norm;
                    if (relative_error <= r_tol)
                    {
                       rel_change_passed = 1;
                       break;
                    }
                 }
                 else
                 {
                    rel_change_passed = 1;
                    break;

                 }
                 num_rel_change_check ++;
              }
              else /* no relative change */
              {
                 break;
              }
           }
           
	} /*** end of restart cycle ***/

        if (iter > 1 && conv_factor > convfac_threshold)
        {
           break;
        }

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i - 2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k + 1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
        (*(gmres_functions->ScaleVector))(rs[i-1], vecW);
        for (j = i - 2; j >= 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j], vecP[j], vecW);
        }

	(*(gmres_functions->ClearVector))(vecR);
	
	/* find correction (in r) */
        //jx_printf("precond: 4\n");
        precond(precond_data, preOperator, vecW, vecR);
        if (break_adaptive == 2)
        {
           jx_ParVectorCopy(vecW, res_vec);
           jx_ParCSRMatrixMatvec(-1.0, (jx_ParCSRMatrix *)preOperator, vecR, 1.0, res_vec);
           jx_ParVectorSetConstantValues(aux_vec, 0.0);
           JX_PAMGPrecond(amg_data, (JX_ParCSRMatrix)preOperator, (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
           jx_ParVectorAxpy(1.0, aux_vec, vecR);
        }

        /* update current solution x (in x) */
	(*(gmres_functions->Axpy))(1.0, vecR, vecX);
         

        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm */
           (*(gmres_functions->CopyVector))(vecB, vecR);
           (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
           r_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );

           if (r_norm <= epsilon)
           {
              if (rel_change && !rel_change_passed) /* calculate the relative change */
              {

                 /* calculate the norm of the solution */
                 x_norm = sqrt( (*(gmres_functions->InnerProd))(vecX,vecX) );
               
                 if ( !(x_norm <= guard_zero_residual )) /* don't divide by zero */
                 {
                    /* for relative change take x_(i-1) to be 
                       x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                       Now
                       x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                       - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                       = M^{-1} rs_{i-1}{p_{i-1}} */
                    (*(gmres_functions->ClearVector))(vecW);
                    (*(gmres_functions->Axpy))(rs[i-1], vecP[i-1], vecW);
                    (*(gmres_functions->ClearVector))(vecR);
                    
                    /* apply the preconditioner */
                    //jx_printf("precond: 5\n");
                    precond(precond_data, preOperator, vecW, vecR);
                    if (break_adaptive == 2)
                    {
                       jx_ParVectorCopy(vecW, res_vec);
                       jx_ParCSRMatrixMatvec(-1.0, (jx_ParCSRMatrix *)preOperator, vecR, 1.0, res_vec);
                       jx_ParVectorSetConstantValues(aux_vec, 0.0);
                       JX_PAMGPrecond(amg_data, (JX_ParCSRMatrix)preOperator,
                                                (JX_ParVector)res_vec, (JX_ParVector)aux_vec);
                       jx_ParVectorAxpy(1.0, aux_vec, vecR);
                    }
                    
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );
                    relative_error= w_norm / x_norm;
                    if ( relative_error < r_tol )
                    {
                       (gmres_data -> converged) = 1;
                       if ( print_level > 0 && my_id == 0 )
                       {
                          jx_printf("\n");
                          jx_printf("Final L2 norm of residual: %e\n", r_norm);
                       }
                       break;
                    }
                 }
                 else
                 {
                    (gmres_data -> converged) = 1;
                    if ( print_level > 0 && my_id == 0 )
                    {
                       jx_printf("\n");
                       jx_printf("Final L2 norm of residual: %e\n", r_norm);
                    }
                    break;
                 }

              }
              else /* don't need to check rel. change */
              {
                 if ( print_level > 0 && my_id == 0 )
                 {
                    jx_printf("\n");
                    jx_printf("Final L2 norm of residual: %e\n", r_norm);
                 }
                 (gmres_data -> converged) = 1;
                 break;
              }
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if ( print_level > 0 && my_id == 0)
              {
                 jx_printf(" false convergence 2:");
                 if (b_norm > 0.0)
                 {
                    jx_printf(" ||r|| = %.2le  ||r||/||b|| = %.2le\n", r_norm, r_norm / b_norm);
                 }
                 else
                 {
                    jx_printf(" ||r|| = %.2le  ||r||/||r_0|| = %.2le\n", r_norm, r_norm / r_norm_0);
                 }
              }
              
              /* modified by Xu Xiaowen, 2011/11/07 */
              if (is_check_restarted == 1) 
              {
                 (*(gmres_functions->CopyVector))(vecR, vecP[0]);
                 i = 0;
              } 
              else 
              {
                 break;  
              }
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j] = c[j-1]*rs[j];
	}
        
        if (i) 
        {
           (*(gmres_functions->Axpy))(rs[i]-1.0, vecP[i], vecP[i]);
        }
        
        for (j = i - 1 ; j > 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j],vecP[j],vecP[i]);
        }
        
        if (i)
        {
           (*(gmres_functions->Axpy))(rs[0]-1.0, vecP[0], vecP[0]);
           (*(gmres_functions->Axpy))(1.0, vecP[i], vecP[0]);
        }
        
   } /* END of iteration while loop */


   if ( print_level > 1 && my_id == 0 ) jx_printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   if (break_adaptive == 0)
   {
      (gmres_data -> comp_ilu_iter) = iter;
   }
   else
   {
      (gmres_data -> comp_bco_iter) = iter;
   }
   (gmres_data -> break_adaptive) = break_adaptive;
   
   if (b_norm > 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   
   if (b_norm == 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / r_norm_0;
   }

   if ( amg_data )
   {
      JX_PAMGDestroy( amg_data );
   }

   if ( aux_vec )
   {
      jx_ParVectorDestroy( aux_vec );
   }

   if ( res_vec )
   {
      jx_ParVectorDestroy( res_vec );
   }

   }

#if 0
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf(" Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
      jx_error(JX_ERROR_CONV);
   }
#endif

#if 0
   /* 保存解向量 */   
   jx_Vector *ser_app = NULL;
   ser_app = jx_ParVectorToVectorAll ((jx_ParVector *)vecX);
   jx_SeqVectorPrint(ser_app, "./u_ser"); 
   jx_SeqVectorDestroy(ser_app);
   jx_printf("\n\n zzy solution has been saved!!\n\n");
#endif

#if 0
   /* 修改解向量 */ 
   JX_Int change_sol;
   jx_printf("\n");
   jx_printf(" Do you want the modify the solution with the file './app'?\n");
   jx_printf(" >>> 0: NO\n");
   jx_printf(" >>> 1: YES\n");
   jx_scanf("%d", &change_sol); 
   if (change_sol == 1) 
   {
      FILE *fp = NULL;
      jx_Vector *tmp_app = NULL;
      JX_Int ii, num_rows;
      tmp_app = jx_ParVectorLocalVector((jx_ParVector *)vecX);
   
      fp = fopen("./app", "r");
      jx_fscanf(fp, "%d", &num_rows);
      for (ii = 0; ii < num_rows; ii ++)
      {
         jx_fscanf(fp, "%le", &jx_VectorData(tmp_app)[ii]);
      }
      jx_printf("\n\n zzy solution has been changed for test!!\n\n");
   }
#endif

   jx_TFreeF(c,gmres_functions); 
   jx_TFreeF(s,gmres_functions); 
   jx_TFreeF(rs,gmres_functions);
   if (rel_change)  
   {
      jx_TFreeF(rs_2,gmres_functions);
   }
   for (i = 0; i < k_dim + 1; i ++)
   {	
      jx_TFreeF(hh[i], gmres_functions);
   }
   jx_TFreeF(hh, gmres_functions);

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_GMRESSolveDefault
 * \note This id a simplified version with default parameters. 
 * \author peghoty
 * \date 2011/11/08
 */
JX_Int
jx_GMRESSolveDefault( void *gmres_vdata, void *preOperator, void *matA, void *vecB, void *vecX )
{
   jx_GMRESData      *gmres_data      = gmres_vdata;
   jx_GMRESFunctions *gmres_functions = gmres_data -> functions;
   JX_Int 		     k_dim            = (gmres_data -> k_dim);
   JX_Int               min_iter         = (gmres_data -> min_iter);
   JX_Int 		     max_iter         = (gmres_data -> max_iter);
   JX_Real 	     r_tol            = (gmres_data -> tol);
   JX_Real            a_tol            = (gmres_data -> a_tol);
   void             *matvec_data      = (gmres_data -> matvec_data);
   void             *vecR             = (gmres_data -> r);
   void             *vecW             = (gmres_data -> w);
   void            **vecP             = (gmres_data -> p);
   JX_Int 	           (*precond)()       = (gmres_functions -> precond);
   JX_Int 	            *precond_data     = (gmres_data -> precond_data);
   JX_Int             print_level        = (gmres_data -> print_level);
   JX_Int             logging            = (gmres_data -> logging);
   JX_Real         *norms              = (gmres_data -> norms);

   JX_Int	      i, j, k;
   JX_Real     *rs, **hh, *c, *s; 
   JX_Int        iter; 
   JX_Int        my_id, num_procs;
   JX_Real     epsilon, gamma, t, r_norm, b_norm, den_norm;
   
   JX_Real     epsmac = 1.e-16; 
   JX_Real     ieee_check = 0.0;

   JX_Real     r_norm_0;

   (gmres_data -> converged) = 0;
   
   (*(gmres_functions->CommInfo))(matA, &my_id, &num_procs);
   
   if ( logging > 0 || print_level > 0 )
   {
      norms = (gmres_data -> norms);
   }

   /* initialize work arrays */
   rs = jx_CTAllocF(JX_Real, k_dim + 1, gmres_functions); 
   c  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   s  = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   hh = jx_CTAllocF(JX_Real*, k_dim + 1, gmres_functions); 
   for (i = 0; i < k_dim + 1; i ++)
   {	
      hh[i] = jx_CTAllocF(JX_Real, k_dim, gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(vecB, vecP[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecP[0]);

   b_norm = sqrt((*(gmres_functions->InnerProd))(vecB, vecB));

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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   r_norm = sqrt((*(gmres_functions->InnerProd))(vecP[0], vecP[0]));
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
         jx_printf("\n\nERROR detected by JXPAMG ... BEGIN\n");
         jx_printf("ERROR -- jx_GMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   if ( logging > 0 || print_level > 0 )
   {
      norms[0] = r_norm;
      if ( print_level > 0 && my_id == 0 )
      {
  	 jx_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            jx_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         jx_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   }


   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user specifies a_tol, or sets r_tol = 0.0, which means absolute
            tol only is checked  */
      
   epsilon = jx_max(a_tol, r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level > 0 && my_id == 0 )
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


   /* outer iteration cycle */
   
   while (iter < max_iter)
   {
        /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           jx_TFreeF(c, gmres_functions); 
           jx_TFreeF(s, gmres_functions); 
           jx_TFreeF(rs, gmres_functions);
           for (i = 0; i < k_dim + 1; i ++) 
           {
              jx_TFreeF(hh[i], gmres_functions);
           }
           jx_TFreeF(hh, gmres_functions); 
	   return jx_error_flag;
	}

        /* see if we are already converged and should print the final norm and exit */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           (*(gmres_functions->CopyVector))(vecB, vecR);
           (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
           r_norm = sqrt((*(gmres_functions->InnerProd))(vecR, vecR));
           if (r_norm <= epsilon)
           {
              if ( print_level > 0 && my_id == 0 )
              {
                 jx_printf("\n\n");
                 jx_printf(" Final L2 norm of residual: %e\n\n", r_norm);
              }
              (gmres_data -> converged) = 1; // peghoty, 2011/11/08
              break;
           }
           else
           {
              if ( print_level > 0 && my_id == 0)
              {
                 jx_printf(" false convergence 1\n");
              }
           }
	}

      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t, vecP[0]);
	i = 0;

        /*** RESTART CYCLE (right-preconditioning) ***/
        
        while (i < k_dim && iter < max_iter)
	{
           i ++;
           iter ++;
           (*(gmres_functions->ClearVector))(vecR);
           precond(precond_data, preOperator, vecP[i-1], vecR);
           (*(gmres_functions->Matvec))(matvec_data, 1.0, matA, vecR, 0.0, vecP[i]);
           
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = (*(gmres_functions->InnerProd))(vecP[j], vecP[i]);
              (*(gmres_functions->Axpy))(-hh[j][i-1],vecP[j], vecP[i]);
           }
           t = sqrt((*(gmres_functions->InnerProd))(vecP[i], vecP[i]));
           hh[i][i-1] = t;	
           if (t != 0.0)
           {
              t = 1.0 / t;
              (*(gmres_functions->ScaleVector))(t, vecP[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step. update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1] = hh[i-1][i-1]/gamma;
           s[i-1] = hh[i][i-1]/gamma;
           rs[i] = -hh[i][i-1]*rs[i-1];
           rs[i] /=  gamma;
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if ( print_level > 0 )
           {
              norms[iter] = r_norm;
              if ( print_level > 0 && my_id == 0 )
              {
                 if (b_norm > 0.0)
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                                              norms[iter]/norms[iter-1],norms[iter]/b_norm);
                 }
                 else
                 {
                    jx_printf("% 5d    %e    %f   %e\n", iter, norms[iter], norms[iter]/norms[iter-1],norms[iter]/r_norm_0);
                 }
              }
           }
           
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              break;
           }
           
	} /*** end of restart cycle ***/


	/* now compute solution, first solve upper triangular system */

	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i - 2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k + 1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        (*(gmres_functions->CopyVector))(vecP[i-1], vecW);
        (*(gmres_functions->ScaleVector))(rs[i-1], vecW);
        for (j = i - 2; j >= 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j], vecP[j], vecW);
        }

	(*(gmres_functions->ClearVector))(vecR);
	
	/* find correction (in r) */
        precond(precond_data, preOperator, vecW, vecR);

        /* update current solution x (in x) */
	(*(gmres_functions->Axpy))(1.0, vecR, vecX);
         

        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm */
           (*(gmres_functions->CopyVector))(vecB, vecR);
           (*(gmres_functions->Matvec))(matvec_data, -1.0, matA, vecX, 1.0, vecR);
           r_norm = sqrt( (*(gmres_functions->InnerProd))(vecR,vecR) );

           if (r_norm <= epsilon)
           {
              if ( print_level > 0 && my_id == 0 )
              {
                 jx_printf("\n");
                 jx_printf(" Final L2 norm of residual: %e\n", r_norm);
              }
              (gmres_data -> converged) = 1;
              break;
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if ( print_level > 0 && my_id == 0)
              {
                 jx_printf(" false convergence 2\n");
              }
              (*(gmres_functions->CopyVector))(vecR, vecP[0]);
              i = 0;
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j] = c[j-1]*rs[j];
	}
        
        if (i) 
        {
           (*(gmres_functions->Axpy))(rs[i]-1.0, vecP[i], vecP[i]);
        }
        
        for (j = i - 1 ; j > 0; j --)
        {
           (*(gmres_functions->Axpy))(rs[j],vecP[j],vecP[i]);
        }
        
        if (i)
        {
           (*(gmres_functions->Axpy))(rs[0]-1.0, vecP[0], vecP[0]);
           (*(gmres_functions->Axpy))(1.0, vecP[i], vecP[0]);
        }
        
   } /* END of iteration while loop */


   if ( print_level > 1 && my_id == 0 ) jx_printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   
   if (b_norm > 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   
   if (b_norm == 0.0)
   {
      (gmres_data -> rel_residual_norm) = r_norm / r_norm_0;
   }

#if 0
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf(" Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
      jx_error(JX_ERROR_CONV);
   }
#endif
  
   jx_TFreeF(c,gmres_functions); 
   jx_TFreeF(s,gmres_functions); 
   jx_TFreeF(rs,gmres_functions);

   for (i = 0; i < k_dim + 1; i ++)
   {	
      jx_TFreeF(hh[i], gmres_functions);
   }
   jx_TFreeF(hh, gmres_functions); 

   return jx_error_flag;
}
