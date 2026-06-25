//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  pcogmres.c
 *  Date: 2019/02/12
 */ 

#include "jx_combined.h"
#include "jx_krylov.h"

JX_Int
JX_ParCSRCOGMRESCreate( MPI_Comm comm, JX_Solver *solver )
{
   jx_COGMRESFunctions * cogmres_functions;

   cogmres_functions = jx_COGMRESFunctionsCreate(
         jx_CAlloc, jx_ParKrylovFree, jx_ParKrylovCommInfo,
         jx_ParKrylovCreateVector,
         jx_ParKrylovCreateVectorArray,
         jx_ParKrylovDestroyVector, jx_ParKrylovMatvecCreate,
         jx_ParKrylovMatvec, jx_ParKrylovMatvecDestroy,
         jx_ParKrylovInnerProd, jx_ParKrylovMassInnerProd, 
         jx_ParKrylovMassDotpTwo, jx_ParKrylovCopyVector,
         //jx_ParKrylovCopyVector,
         jx_ParKrylovClearVector,
         jx_ParKrylovScaleVector, jx_ParKrylovAxpy, jx_ParKrylovMassAxpy,
         jx_ParKrylovIdentitySetup, jx_ParKrylovIdentity );

   *solver = ( (JX_Solver) jx_COGMRESCreate( cogmres_functions ) );

   if (!solver)
   {
      jx_error_in_arg(2);
   }

   return jx_error_flag;
}

JX_Int
JX_ParCSRCOGMRESDestroy( JX_Solver solver )
{
   return( jx_COGMRESDestroy( (void *) solver ) );
}

JX_Int
JX_COGMRESSetup( JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x )
{
   return( jx_COGMRESSetup( solver, A, b, x ) );
}

JX_Int 
JX_COGMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix A, JX_Vector b, JX_Vector x )
{
   return( jx_COGMRESSolve( solver, preOperator, A, b, x ) );
}

JX_Int
JX_COGMRESSetKDim( JX_Solver solver, JX_Int k_dim )
{
   return( jx_COGMRESSetKDim( (void *) solver, k_dim ) );
}

JX_Int
JX_COGMRESGetKDim( JX_Solver solver, JX_Int * k_dim )
{
   return( jx_COGMRESGetKDim( (void *) solver, k_dim ) );
}

JX_Int
JX_COGMRESSetUnroll( JX_Solver solver, JX_Int unroll )
{
   return( jx_COGMRESSetUnroll( (void *) solver, unroll ) );
}

JX_Int
JX_COGMRESGetUnroll( JX_Solver solver, JX_Int * unroll )
{
   return( jx_COGMRESGetUnroll( (void *) solver, unroll ) );
}

JX_Int
JX_COGMRESSetCGS( JX_Solver solver, JX_Int cgs )
{
   return( jx_COGMRESSetCGS( (void *) solver, cgs ) );
}

JX_Int
JX_COGMRESGetCGS( JX_Solver solver, JX_Int * cgs )
{
   return( jx_COGMRESGetCGS( (void *) solver, cgs ) );
}

JX_Int
JX_COGMRESSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_COGMRESSetTol( (void *) solver, tol ) );
}

JX_Int
JX_COGMRESGetTol( JX_Solver solver, JX_Real * tol )
{
   return( jx_COGMRESGetTol( (void *) solver, tol ) );
}

JX_Int
JX_COGMRESSetAbsoluteTol( JX_Solver solver, JX_Real a_tol )
{
   return( jx_COGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_COGMRESGetAbsoluteTol( JX_Solver solver, JX_Real * a_tol )
{
   return( jx_COGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_COGMRESSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol )
{
   return( jx_COGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

JX_Int
JX_COGMRESGetConvergenceFactorTol( JX_Solver solver, JX_Real * cf_tol )
{
   return( jx_COGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

JX_Int
JX_COGMRESSetMinIter( JX_Solver solver, JX_Int min_iter )
{
   return( jx_COGMRESSetMinIter( (void *) solver, min_iter ) );
}

JX_Int
JX_COGMRESGetMinIter( JX_Solver solver, JX_Int * min_iter )
{
   return( jx_COGMRESGetMinIter( (void *) solver, min_iter ) );
}

JX_Int
JX_COGMRESSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_COGMRESSetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_COGMRESSetIsCheckRestarted( JX_Solver solver, JX_Int is_check_restarted )
{
   return( jx_COGMRESSetIsCheckRestarted( (void *) solver, is_check_restarted ) );
}

JX_Int
JX_COGMRESGetMaxIter( JX_Solver solver, JX_Int * max_iter )
{
   return( jx_COGMRESGetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_COGMRESSetPrecond( JX_Solver          solver,
                      JX_PtrToSolverFcn  precond,
                      JX_PtrToSolverFcn  precond_setup,
                      JX_Solver          precond_solver )
{
   return( jx_COGMRESSetPrecond( (void *) solver,
                                 (JX_Int (*)(void*, void*, void*, void*))precond,
                                 (JX_Int (*)(void*, void*, void*, void*))precond_setup,
                                 (void *) precond_solver ) );
}

JX_Int
JX_COGMRESGetPrecond( JX_Solver  solver, JX_Solver *precond_data_ptr )
{
   return( jx_COGMRESGetPrecond( (void *) solver, (JX_Solver *) precond_data_ptr ) );
}

JX_Int
JX_COGMRESSetPrintLevel( JX_Solver solver, JX_Int level )
{
   return( jx_COGMRESSetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_COGMRESGetPrintLevel( JX_Solver solver, JX_Int * level )
{
   return( jx_COGMRESGetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_COGMRESSetLogging( JX_Solver solver, JX_Int level )
{
   return( jx_COGMRESSetLogging( (void *) solver, level ) );
}

JX_Int
JX_COGMRESGetLogging( JX_Solver solver, JX_Int * level )
{
   return( jx_COGMRESGetLogging( (void *) solver, level ) );
}

JX_Int
JX_COGMRESGetNumIterations( JX_Solver  solver, JX_Int *num_iterations )
{
   return( jx_COGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int
JX_COGMRESGetConverged( JX_Solver  solver, JX_Int *converged )
{
   return( jx_COGMRESGetConverged( (void *) solver, converged ) );
}

JX_Int
JX_COGMRESGetFinalRelativeResidualNorm( JX_Solver  solver, JX_Real *norm )
{
   return( jx_COGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

JX_Int
JX_COGMRESGetResidual( JX_Solver solver, void *residual )
{
   return jx_COGMRESGetResidual( (void *) solver, (void **) residual );
}

JX_Int
JX_COGMRESSetModifyPC( JX_Solver  solver, JX_Int (*modify_pc)(JX_Solver, JX_Int, JX_Real) )
{
   return jx_COGMRESSetModifyPC( (void *) solver, (JX_Int(*)(void*, JX_Int, JX_Real))modify_pc );
}

jx_COGMRESFunctions *
jx_COGMRESFunctionsCreate(
   char *       (*CAlloc)        ( size_t count, size_t elt_size ),
   JX_Int    (*Free)          ( char *ptr ),
   JX_Int    (*CommInfo)      ( void  *A, JX_Int   *my_id, JX_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( JX_Int size, void *vectors ),
   JX_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y ),
   JX_Int    (*MatvecDestroy) ( void *matvec_data ),
   JX_Real   (*InnerProd)     ( void *x, void *y ),
   JX_Int    (*MassInnerProd) ( void *x, void **y, JX_Int k, JX_Int unroll, void *result ),
   JX_Int    (*MassDotpTwo)   ( void *x, void *y, void **z, JX_Int k, JX_Int unroll, void *result_x, void *result_y ),
   JX_Int    (*CopyVector)    ( void *x, void *y ),
   JX_Int    (*ClearVector)   ( void *x ),
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x ),
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y ),
   JX_Int    (*MassAxpy)      ( JX_Real *alpha, void **x, void *y, JX_Int k, JX_Int unroll ),
   JX_Int    (*PrecondSetup)  ( void *vdata, void *A ),
   JX_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   jx_COGMRESFunctions *cogmres_functions;
   cogmres_functions = (jx_COGMRESFunctions *)CAlloc( 1, sizeof(jx_COGMRESFunctions) );

   cogmres_functions->CAlloc            = CAlloc;
   cogmres_functions->Free              = Free;
   cogmres_functions->CommInfo          = CommInfo; /* not in PCGFunctionsCreate */
   cogmres_functions->CreateVector      = CreateVector;
   cogmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   cogmres_functions->DestroyVector     = DestroyVector;
   cogmres_functions->MatvecCreate      = MatvecCreate;
   cogmres_functions->Matvec            = Matvec;
   cogmres_functions->MatvecDestroy     = MatvecDestroy;
   cogmres_functions->InnerProd         = InnerProd;
   cogmres_functions->MassInnerProd     = MassInnerProd;
   cogmres_functions->MassDotpTwo       = MassDotpTwo;
   cogmres_functions->CopyVector        = CopyVector;
   cogmres_functions->ClearVector       = ClearVector;
   cogmres_functions->ScaleVector       = ScaleVector;
   cogmres_functions->Axpy              = Axpy;
   cogmres_functions->MassAxpy          = MassAxpy;
   /* default preconditioner must be set here but can be changed later... */
   cogmres_functions->precond_setup     = PrecondSetup;
   cogmres_functions->precond           = Precond;

   return cogmres_functions;
}

void *
jx_COGMRESCreate( jx_COGMRESFunctions *cogmres_functions )
{
   jx_COGMRESData *cogmres_data;

   cogmres_data = jx_CTAllocF(jx_COGMRESData, 1, cogmres_functions);

   cogmres_data->functions = cogmres_functions;

   /* set defaults */
   (cogmres_data -> k_dim)              = 5;
   (cogmres_data -> cgs)                = 1; /* if 2 performs reorthogonalization */
   (cogmres_data -> tol)                = 1.0e-06; /* relative residual tol */
   (cogmres_data -> cf_tol)             = 0.0;
   (cogmres_data -> a_tol)              = 0.0; /* abs. residual tol */
   (cogmres_data -> min_iter)           = 0;
   (cogmres_data -> max_iter)           = 1000;
   (cogmres_data -> rel_change)         = 0;
   (cogmres_data -> is_check_restarted) = 0;
   (cogmres_data -> skip_real_r_check)  = 0;
   (cogmres_data -> converged)          = 0;
   (cogmres_data -> precond_data)       = NULL;
   (cogmres_data -> print_level)        = 0;
   (cogmres_data -> logging)            = 0;
   (cogmres_data -> p)                  = NULL;
   (cogmres_data -> r)                  = NULL;
   (cogmres_data -> w)                  = NULL;
   (cogmres_data -> w_2)                = NULL;
   (cogmres_data -> matvec_data)        = NULL;
   (cogmres_data -> norms)              = NULL;
   (cogmres_data -> log_file_name)      = NULL;
   (cogmres_data -> unroll)             = 0;

   return (void *) cogmres_data;
}

JX_Int
jx_COGMRESDestroy( void *cogmres_vdata )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   JX_Int i;

   if (cogmres_data)
   {
      jx_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
      if ( (cogmres_data->logging>0) || (cogmres_data->print_level) > 0 )
      {
         if ( (cogmres_data -> norms) != NULL )
         jx_TFreeF( cogmres_data -> norms, cogmres_functions );
      }

      if ( (cogmres_data -> matvec_data) != NULL )
         (*(cogmres_functions->MatvecDestroy))(cogmres_data -> matvec_data);

      if ( (cogmres_data -> r) != NULL )
         (*(cogmres_functions->DestroyVector))(cogmres_data -> r);
      if ( (cogmres_data -> w) != NULL )
         (*(cogmres_functions->DestroyVector))(cogmres_data -> w);
      if ( (cogmres_data -> w_2) != NULL )
         (*(cogmres_functions->DestroyVector))(cogmres_data -> w_2);

      if ( (cogmres_data -> p) != NULL )
      {
         for (i = 0; i < (cogmres_data -> k_dim+1); i++)
         {
            if ( (cogmres_data -> p)[i] != NULL )
            (*(cogmres_functions->DestroyVector))( (cogmres_data -> p) [i]);
         }
         jx_TFreeF( cogmres_data->p, cogmres_functions );
      }
      jx_TFreeF( cogmres_data, cogmres_functions );
      jx_TFreeF( cogmres_functions, cogmres_functions );
   }

   return jx_error_flag;
}

JX_Int
jx_COGMRESGetResidual( void *cogmres_vdata, void **residual )
{
   jx_COGMRESData  *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *residual = cogmres_data->r;
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetup( void *cogmres_vdata, void *A, void *b, void *x )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   jx_COGMRESFunctions *cogmres_functions = cogmres_data->functions;

   JX_Int k_dim            = (cogmres_data -> k_dim);
   JX_Int max_iter         = (cogmres_data -> max_iter);
   JX_Int (*precond_setup)(void*,void*,void*,void*) = (cogmres_functions->precond_setup);
   void       *precond_data   = (cogmres_data -> precond_data);
   JX_Int rel_change       = (cogmres_data -> rel_change);

   (cogmres_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ((cogmres_data -> p) == NULL)
      (cogmres_data -> p) = (void**)(*(cogmres_functions->CreateVectorArray))(k_dim+1,x);
   if ((cogmres_data -> r) == NULL)
      (cogmres_data -> r) = (*(cogmres_functions->CreateVector))(b);
   if ((cogmres_data -> w) == NULL)
      (cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);

   if (rel_change)
   {  
      if ((cogmres_data -> w_2) == NULL)
         (cogmres_data -> w_2) = (*(cogmres_functions->CreateVector))(b);
   }

   if ((cogmres_data -> matvec_data) == NULL)
      (cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (cogmres_data->logging)>0 || (cogmres_data->print_level) > 0 )
   {
      if ((cogmres_data -> norms) == NULL)
         (cogmres_data -> norms) = jx_CTAllocF(JX_Real, max_iter + 1, cogmres_functions);
   }
   if ( (cogmres_data->print_level) > 0 ) 
   {
      if ((cogmres_data -> log_file_name) == NULL)
         (cogmres_data -> log_file_name) = (char*)"cogmres.out.log";
   }

   return jx_error_flag;
}

JX_Int
jx_COGMRESSolve( void  *cogmres_vdata, void *preOperator, void  *A, void  *b, void  *x )
{
   jx_COGMRESData      *cogmres_data      = (jx_COGMRESData *)cogmres_vdata;
   jx_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   JX_Int     k_dim              = (cogmres_data -> k_dim);
   JX_Int     unroll             = (cogmres_data -> unroll);
   JX_Int     cgs                = (cogmres_data -> cgs);
   JX_Int     min_iter           = (cogmres_data -> min_iter);
   JX_Int     max_iter           = (cogmres_data -> max_iter);
   JX_Int     rel_change         = (cogmres_data -> rel_change);
   JX_Int     is_check_restarted = cogmres_data -> is_check_restarted;
   JX_Int     skip_real_r_check  = (cogmres_data -> skip_real_r_check);
   JX_Real    r_tol              = (cogmres_data -> tol);
   JX_Real    cf_tol             = (cogmres_data -> cf_tol);
   JX_Real    a_tol              = (cogmres_data -> a_tol);
   void         *matvec_data       = (cogmres_data -> matvec_data);

   void         *r                 = (cogmres_data -> r);
   void         *w                 = (cogmres_data -> w);
   /* note: w_2 is only allocated if rel_change = 1 */
   void         *w_2               = (cogmres_data -> w_2); 

   void        **p                 = (cogmres_data -> p);

   JX_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
   JX_Int  *precond_data       = (JX_Int*)(cogmres_data -> precond_data);

   JX_Int print_level = (cogmres_data -> print_level);
   JX_Int logging     = (cogmres_data -> logging);

   JX_Real     *norms          = (cogmres_data -> norms);
  /* not used yet   char           *log_file_name  = (cogmres_data -> log_file_name);*/
  /*   FILE           *fp; */

   JX_Int  break_value = 0;
   JX_Int  i, j, k;
  /*KS: rv is the norm history */
   JX_Real *rs, *hh, *uu, *c, *s, *rs_2 = NULL, *rv;
  //, *tmp; 
   JX_Int  iter; 
   JX_Int  my_id, num_procs;
   JX_Real epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   JX_Real w_norm;

   JX_Real epsmac = 1.e-16; 
   JX_Real ieee_check = 0.;

   JX_Real guard_zero_residual; 
   JX_Real cf_ave_0 = 0.0;
   JX_Real cf_ave_1 = 0.0;
   JX_Real weight;
   JX_Real r_norm_0;
   JX_Real relative_error = 1.0;

   JX_Int        rel_change_passed = 0, num_rel_change_check = 0;
   JX_Int    itmp = 0;

   JX_Real real_r_norm_old, real_r_norm_new;

   (cogmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(cogmres_functions->CommInfo))(A,&my_id,&num_procs);
   if ( logging>0 || print_level>0 )
   {
      norms          = (cogmres_data -> norms);
   }

   /* initialize work arrays */
   rs = jx_CTAllocF(JX_Real,k_dim+1,cogmres_functions);
   c  = jx_CTAllocF(JX_Real,k_dim,cogmres_functions);
   s  = jx_CTAllocF(JX_Real,k_dim,cogmres_functions);
   if (rel_change) rs_2 = jx_CTAllocF(JX_Real, k_dim+1, cogmres_functions); 

   rv = jx_CTAllocF(JX_Real, k_dim+1, cogmres_functions);
  
   hh = jx_CTAllocF(JX_Real, (k_dim+1)*k_dim, cogmres_functions);
   uu = jx_CTAllocF(JX_Real, (k_dim+1)*k_dim, cogmres_functions);

   (*(cogmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(cogmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(cogmres_functions->InnerProd))(b,b));
   real_r_norm_old = b_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
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
         jx_printf("ERROR -- jx_COGMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   r_norm   = sqrt((*(cogmres_functions->InnerProd))(p[0],p[0]));
   r_norm_0 = r_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
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
         jx_printf("ERROR -- jx_COGMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by JXPAMG ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }
 
   if ( logging>0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level>1 && my_id == 0 )
      {
         jx_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            jx_printf("Rel_resid_norm actually contains the residual norm\n");
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
   };

   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
      user specifies a_tol, or sets r_tol = 0.0, which means absolute
      tol only is checked  */

   epsilon = jx_max(a_tol,r_tol*den_norm);

   /* so now our stop criteria is |r_i| <= epsilon */

   if ( print_level>1 && my_id == 0 )
   {
      if (b_norm > 0.0)
      {
         jx_printf("=============================================\n\n");
         jx_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
         jx_printf("-----    ------------    ---------- ------------\n");

      }
      else
      {
         jx_printf("=============================================\n\n");
         jx_printf("Iters     resid.norm     conv.rate\n");
         jx_printf("-----    ------------    ----------\n");
      };
   }


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;

   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */
      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         jx_TFreeF(c,cogmres_functions); 
         jx_TFreeF(s,cogmres_functions); 
         jx_TFreeF(rs,cogmres_functions);
         jx_TFreeF(rv,cogmres_functions);
         if (rel_change)  jx_TFreeF(rs_2,cogmres_functions);
         jx_TFreeF(hh,cogmres_functions); 
         jx_TFreeF(uu,cogmres_functions); 
         return jx_error_flag;
      }

      /* see if we are already converged and 
         should print the final norm and exit */

      if (r_norm  <= epsilon && iter >= min_iter) 
      {
         if (!rel_change) /* shouldn't exit after no iterations if
                           * relative change is on*/
         {
            (*(cogmres_functions->CopyVector))(b,r);
            (*(cogmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
            r_norm = sqrt((*(cogmres_functions->InnerProd))(r,r));
            if (r_norm  <= epsilon)
            {
               if ( print_level>1 && my_id == 0)
               {
                  jx_printf("\n\n");
                  jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               break;
            }
            else
            {
               if ( print_level>0 && my_id == 0) jx_printf("false convergence 1\n");
               if (is_check_restarted == 0) break;
            }
         }
      }

      t = 1.0 / r_norm;
      (*(cogmres_functions->ScaleVector))(t,p[0]);
      i = 0;
      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < k_dim && iter < max_iter)
      {
         i++;
         iter++;
         itmp = (i-1)*(k_dim+1);

         (*(cogmres_functions->ClearVector))(r);
        
         precond(precond_data, preOperator, p[i-1], r);
         (*(cogmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
         for (j=0; j<i; j++)
            rv[j]  = 0;

         if (cgs > 1)
         {
            (*(cogmres_functions->MassDotpTwo))((void *) p[i], p[i-1], p, i, unroll, &hh[itmp], &uu[itmp]);
            for (j=0; j<i-1; j++) uu[j*(k_dim+1)+i-1] = uu[itmp+j];
            for (j=0; j<i; j++) rv[j] = hh[itmp+j];
            for (k=0; k < i; k++)
            {
               for (j=0; j < i; j++)
               {
                  hh[itmp+j] -= (uu[k*(k_dim+1)+j]*rv[j]);
               }
            }
            for (j=0; j<i; j++)
               hh[itmp+j]  = -rv[j]-hh[itmp+j];
         }
         else
         {
            (*(cogmres_functions->MassInnerProd))((void *) p[i], p, i, unroll, &hh[itmp]);
            for (j=0; j<i; j++)
               hh[itmp+j]  = -hh[itmp+j];
         }

         (*(cogmres_functions->MassAxpy))(&hh[itmp],p,p[i], i, unroll);
         for (j=0; j<i; j++)
            hh[itmp+j]  = -hh[itmp+j];
         t = sqrt( (*(cogmres_functions->InnerProd))(p[i],p[i]) );
         hh[itmp+i] = t;

         if (hh[itmp+i] != 0.0)
         {
            t = 1.0/t;
            (*(cogmres_functions->ScaleVector))(t,p[i]);
         }
         for (j = 1; j < i; j++)
         {
            t = hh[itmp+j-1];
            hh[itmp+j-1] = s[j-1]*hh[itmp+j] + c[j-1]*t;
            hh[itmp+j] = -s[j-1]*t + c[j-1]*hh[itmp+j];
         }
         t= hh[itmp+i]*hh[itmp+i];
         t+= hh[itmp+i-1]*hh[itmp+i-1];
         gamma = sqrt(t);
         if (gamma == 0.0) gamma = epsmac;
         c[i-1] = hh[itmp+i-1]/gamma;
         s[i-1] = hh[itmp+i]/gamma;
         rs[i] = -hh[itmp+i]*rs[i-1];
         rs[i] /=  gamma;
         rs[i-1] = c[i-1]*rs[i-1];
         // determine residual norm 
         hh[itmp+i-1] = s[i-1]*hh[itmp+i] + c[i-1]*hh[itmp+i-1];
         r_norm = fabs(rs[i]);
         if ( print_level>0 )
         {
            norms[iter] = r_norm;
            if ( print_level>1 && my_id == 0 )
            {
               if (b_norm > 0.0)
                  jx_printf("% 5d    %e    %f   %e\n", iter, 
                     norms[iter],norms[iter]/norms[iter-1],
                     norms[iter]/b_norm);
               else
                  jx_printf("% 5d    %e    %f\n", iter, norms[iter],
                     norms[iter]/norms[iter-1]);
            }
         }
         /*convergence factor tolerance */
         if (cf_tol > 0.0)
         {
            cf_ave_0 = cf_ave_1;
            cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

            weight = fabs(cf_ave_1 - cf_ave_0);
            weight = weight / jx_max(cf_ave_1, cf_ave_0);

            weight = 1.0 - weight;
#if 0
           jx_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
              i, cf_ave_1, cf_ave_0, weight );
#endif
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
               for (k=0; k<i; k++) /* extra copy of rs so we don't need
                                   to change the later solve */
                  rs_2[k] = rs[k];

               /* solve tri. system*/
               rs_2[i-1] = rs_2[i-1]/hh[itmp+i-1];
               for (k = i-2; k >= 0; k--)
               {
                  t = 0.0;
                  for (j = k+1; j < i; j++)
                  {
                     t -= hh[j*(k_dim+1)+k]*rs_2[j];
                  }
                  t+= rs_2[k];
                  rs_2[k] = t/hh[k*(k_dim+1)+k];
               }
               (*(cogmres_functions->CopyVector))(p[i-1],w);
               (*(cogmres_functions->ScaleVector))(rs_2[i-1],w);
               for (j = i-2; j >=0; j--)
                  (*(cogmres_functions->Axpy))(rs_2[j], p[j], w);

               (*(cogmres_functions->ClearVector))(r);
               /* find correction (in r) */
               precond(precond_data, preOperator, w, r);
               /* copy current solution (x) to w (don't want to over-write x)*/
               (*(cogmres_functions->CopyVector))(x,w);

               /* add the correction */
               (*(cogmres_functions->Axpy))(1.0,r,w);

               /* now w is the approx solution  - get the norm*/
               x_norm = sqrt( (*(cogmres_functions->InnerProd))(w,w) );

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {  /* now get  x_i - x_i-1 */
                  if (num_rel_change_check)
                  {
                     /* have already checked once so we can avoid another precond.
                        solve */
                     (*(cogmres_functions->CopyVector))(w, r);
                     (*(cogmres_functions->Axpy))(-1.0, w_2, r);
                     /* now r contains x_i - x_i-1*/

                     /* save current soln w in w_2 for next time */
                     (*(cogmres_functions->CopyVector))(w, w_2);
                  }
                  else
                  {
                     /* first time to check rel change*/
                     /* first save current soln w in w_2 for next time */
                     (*(cogmres_functions->CopyVector))(w, w_2);

                     (*(cogmres_functions->ClearVector))(w);
                     (*(cogmres_functions->Axpy))(rs_2[i-1], p[i-1], w);
                     (*(cogmres_functions->ClearVector))(r);
                     /* apply the preconditioner */
                     precond(precond_data, preOperator, w, r);
                     /* now r contains x_i - x_i-1 */          
                  }
                  /* find the norm of x_i - x_i-1 */          
                  w_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
                  relative_error = w_norm/x_norm;
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
               num_rel_change_check++;
            }
            else /* no relative change */
            {
               break;
            }
         }
      } /*** end of restart cycle ***/

      /* now compute solution, first solve upper triangular system */
      if (break_value) break;
     
      rs[i-1] = rs[i-1]/hh[itmp+i-1];
      for (k = i-2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k+1; j < i; j++)
         {
            t -= hh[j*(k_dim+1)+k]*rs[j];
         }
         t+= rs[k];
         rs[k] = t/hh[k*(k_dim+1)+k];
      }

      (*(cogmres_functions->CopyVector))(p[i-1],w);
      (*(cogmres_functions->ScaleVector))(rs[i-1],w);
      for (j = i-2; j >=0; j--)
         (*(cogmres_functions->Axpy))(rs[j], p[j], w);

      (*(cogmres_functions->ClearVector))(r);
      /* find correction (in r) */
      precond(precond_data, preOperator, w, r);

      /* update current solution x (in x) */
      (*(cogmres_functions->Axpy))(1.0,r,x);


      /* check for convergence by evaluating the actual residual */
      if (r_norm  <= epsilon && iter >= min_iter)
      {
         if (skip_real_r_check)
         {
            (cogmres_data -> converged) = 1;
            break;
         }

         /* calculate actual residual norm*/
         (*(cogmres_functions->CopyVector))(b,r);
         (*(cogmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
         real_r_norm_new = r_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );

         if (r_norm <= epsilon)
         {
            if (rel_change && !rel_change_passed) /* calculate the relative change */
            {
               /* calculate the norm of the solution */
               x_norm = sqrt( (*(cogmres_functions->InnerProd))(x,x) );

               if ( !(x_norm <= guard_zero_residual ))
               /* don't divide by zero */
               {
                  (*(cogmres_functions->ClearVector))(w);
                  (*(cogmres_functions->Axpy))(rs[i-1], p[i-1], w);
                  (*(cogmres_functions->ClearVector))(r);
                  /* apply the preconditioner */
                  precond(precond_data, preOperator, w, r);
                  /* find the norm of x_i - x_i-1 */          
                  w_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
                  relative_error= w_norm/x_norm;
                  if ( relative_error < r_tol )
                  {
                     (cogmres_data -> converged) = 1;
                     if ( print_level>1 && my_id == 0 )
                     {
                        jx_printf("\n\n");
                        jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
                     }
                     break;
                  }
               }
               else
               {
                  (cogmres_data -> converged) = 1;
                  if ( print_level>1 && my_id == 0 )
                  {
                     jx_printf("\n\n");
                     jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
                  }
                  break;
               }
            }
            else /* don't need to check rel. change */
            {
               if ( print_level>1 && my_id == 0 )
               {
                  jx_printf("\n\n");
                  jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_data -> converged) = 1;
               break;
            }
         }
         else /* conv. has not occurred, according to true residual */
         {
            /* exit if the real residual norm has not decreased */
            if (real_r_norm_new >= real_r_norm_old)
            {
               if (print_level > 1 && my_id == 0)
               {
                  jx_printf("\n\n");
                  jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_data -> converged) = 1;
               break;
            }
            /* report discrepancy between real/COGMRES residuals and restart */
            if ( print_level>0 && my_id == 0)
               jx_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
            if (is_check_restarted == 1)
            {
               (*(cogmres_functions->CopyVector))(r,p[0]);
               i = 0;
               real_r_norm_old = real_r_norm_new;
            }
            else
            {
               break;
            }
         }
      } /* end of convergence check */

      /* compute residual vector and continue loop */
      for (j=i ; j > 0; j--)
      {
         rs[j-1] = -s[j-1]*rs[j];
         rs[j] = c[j-1]*rs[j];
      }

      if (i) (*(cogmres_functions->Axpy))(rs[i]-1.0,p[i],p[i]);
      for (j=i-1 ; j > 0; j--)
         (*(cogmres_functions->Axpy))(rs[j],p[j],p[i]);

      if (i)
      {
         (*(cogmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
         (*(cogmres_functions->Axpy))(1.0,p[i],p[0]);
      }

   } /* END of iteration while loop */


   (cogmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (cogmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (cogmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0) jx_error(JX_ERROR_CONV);

   jx_TFreeF(c,cogmres_functions); 
   jx_TFreeF(s,cogmres_functions); 
   jx_TFreeF(rs,cogmres_functions);
   jx_TFreeF(rv,cogmres_functions);
   if (rel_change)  jx_TFreeF(rs_2,cogmres_functions);

   /*for (i=0; i < k_dim+1; i++)
   {  
      jx_TFreeF(hh[i],cogmres_functions);
      jx_TFreeF(uu[i],cogmres_functions);
   }*/
   jx_TFreeF(hh,cogmres_functions); 
   jx_TFreeF(uu,cogmres_functions);

   return jx_error_flag;
}

JX_Int
jx_COGMRESSetKDim( void   *cogmres_vdata, JX_Int   k_dim )
{
   jx_COGMRESData *cogmres_data =(jx_COGMRESData *) cogmres_vdata;
   (cogmres_data -> k_dim) = k_dim;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetKDim( void   *cogmres_vdata, JX_Int * k_dim )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *k_dim = (cogmres_data -> k_dim);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetUnroll( void   *cogmres_vdata, JX_Int   unroll )
{
   jx_COGMRESData *cogmres_data =(jx_COGMRESData *) cogmres_vdata;
   (cogmres_data -> unroll) = unroll;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetUnroll( void   *cogmres_vdata, JX_Int * unroll )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *unroll = (cogmres_data -> unroll);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetCGS( void   *cogmres_vdata, JX_Int   cgs )
{
   jx_COGMRESData *cogmres_data =(jx_COGMRESData *) cogmres_vdata;
   (cogmres_data -> cgs) = cgs;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetCGS( void   *cogmres_vdata, JX_Int * cgs )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *cgs = (cogmres_data -> cgs);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetTol( void   *cogmres_vdata, JX_Real  tol )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> tol) = tol;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetTol( void   *cogmres_vdata, JX_Real  * tol )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *tol = (cogmres_data -> tol);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetAbsoluteTol( void   *cogmres_vdata, JX_Real  a_tol )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> a_tol) = a_tol;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetAbsoluteTol( void   *cogmres_vdata, JX_Real  * a_tol )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *a_tol = (cogmres_data -> a_tol);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetConvergenceFactorTol( void   *cogmres_vdata, JX_Real  cf_tol )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> cf_tol) = cf_tol;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetConvergenceFactorTol( void   *cogmres_vdata, JX_Real * cf_tol )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *cf_tol = (cogmres_data -> cf_tol);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetMinIter( void *cogmres_vdata, JX_Int   min_iter )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> min_iter) = min_iter;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetMinIter( void *cogmres_vdata, JX_Int * min_iter )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *min_iter = (cogmres_data -> min_iter);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetMaxIter( void *cogmres_vdata, JX_Int   max_iter )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> max_iter) = max_iter;
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetIsCheckRestarted( void *cogmres_vdata, JX_Int is_check_restarted )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> is_check_restarted) = is_check_restarted;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetMaxIter( void *cogmres_vdata, JX_Int * max_iter )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *max_iter = (cogmres_data -> max_iter);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetRelChange( void *cogmres_vdata, JX_Int   rel_change )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> rel_change) = rel_change;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetRelChange( void *cogmres_vdata, JX_Int * rel_change )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *rel_change = (cogmres_data -> rel_change);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetSkipRealResidualCheck( void *cogmres_vdata, JX_Int skip_real_r_check )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> skip_real_r_check) = skip_real_r_check;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetSkipRealResidualCheck( void *cogmres_vdata, JX_Int *skip_real_r_check )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *skip_real_r_check = (cogmres_data -> skip_real_r_check);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetPrecond( void  *cogmres_vdata,
        JX_Int  (*precond)(void*,void*,void*,void*),
        JX_Int  (*precond_setup)(void*,void*,void*,void*),
        void  *precond_data )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   jx_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   (cogmres_functions -> precond)        = precond;
   (cogmres_functions -> precond_setup)  = precond_setup;
   (cogmres_data -> precond_data)   = precond_data;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetPrecond( void *cogmres_vdata, JX_Solver *precond_data_ptr )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *precond_data_ptr = (JX_Solver)(cogmres_data -> precond_data);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetPrintLevel( void *cogmres_vdata, JX_Int   level )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> print_level) = level;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetPrintLevel( void *cogmres_vdata, JX_Int * level )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *level = (cogmres_data -> print_level);
   return jx_error_flag;
}

JX_Int
jx_COGMRESSetLogging( void *cogmres_vdata, JX_Int   level )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   (cogmres_data -> logging) = level;
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetLogging( void *cogmres_vdata, JX_Int * level )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *level = (cogmres_data -> logging);
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetNumIterations( void *cogmres_vdata, JX_Int  *num_iterations )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *num_iterations = (cogmres_data -> num_iterations);
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetConverged( void *cogmres_vdata, JX_Int  *converged )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *converged = (cogmres_data -> converged);
   return jx_error_flag;
}

JX_Int
jx_COGMRESGetFinalRelativeResidualNorm( void   *cogmres_vdata, JX_Real *relative_residual_norm )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   *relative_residual_norm = (cogmres_data -> rel_residual_norm);
   return jx_error_flag;
}


JX_Int
jx_COGMRESSetModifyPC( void *cogmres_vdata, JX_Int (*modify_pc)(void *precond_data, JX_Int iteration, JX_Real rel_residual_norm) )
{
   jx_COGMRESData *cogmres_data = (jx_COGMRESData *)cogmres_vdata;
   jx_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   (cogmres_functions -> modify_pc)        = modify_pc;
   return jx_error_flag;
} 
