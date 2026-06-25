//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  pflexgmres.c
 *  Date: 2019/12/05
 */ 

#include "jx_pamg.h"
#include "jx_krylov.h"

JX_Int
JX_ParCSRFlexGMRESCreate( MPI_Comm comm, JX_Solver *solver )
{
   jx_FlexGMRESFunctions *fgmres_functions;

   if (!solver)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }

   fgmres_functions =
      jx_FlexGMRESFunctionsCreate(
         jx_CAlloc, jx_ParKrylovFree, jx_ParKrylovCommInfo,
         jx_ParKrylovCreateVector,
         jx_ParKrylovCreateVectorArray,
         jx_ParKrylovDestroyVector, jx_ParKrylovMatvecCreate,
         jx_ParKrylovMatvec, jx_ParKrylovMatvecDestroy,
         jx_ParKrylovInnerProd, jx_ParKrylovCopyVector,
         jx_ParKrylovClearVector,
         jx_ParKrylovScaleVector, jx_ParKrylovAxpy,
         jx_ParKrylovIdentitySetup, jx_ParKrylovIdentity );

  *solver = ( (JX_Solver) jx_FlexGMRESCreate( fgmres_functions ) );

   return jx_error_flag;
}

JX_Int 
JX_ParCSRFlexGMRESDestroy( JX_Solver solver )
{
   return( jx_FlexGMRESDestroy( (void *) solver ) );
}

JX_Int 
JX_FlexGMRESSetup( JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x )
{
   return( jx_FlexGMRESSetup( solver, A, b, x ) );
}

JX_Int 
JX_FlexGMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix A, JX_Vector b, JX_Vector x )
{
   return( jx_FlexGMRESSolve( solver, preOperator, A, b, x ) );
}

JX_Int
JX_FlexGMRESSetKDim( JX_Solver solver, JX_Int k_dim )
{
   return( jx_FlexGMRESSetKDim( (void *) solver, k_dim ) );
}

JX_Int
JX_FlexGMRESSetTol( JX_Solver solver, JX_Real tol )
{
   return( jx_FlexGMRESSetTol( (void *) solver, tol ) );
}

JX_Int
JX_FlexGMRESSetAbsoluteTol( JX_Solver solver, JX_Real a_tol )
{
   return( jx_FlexGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

JX_Int
JX_FlexGMRESSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol )
{
   return( jx_FlexGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

JX_Int
JX_FlexGMRESSetMinIter( JX_Solver solver, JX_Int min_iter )
{
   return( jx_FlexGMRESSetMinIter( (void *) solver, min_iter ) );
}

JX_Int
JX_FlexGMRESSetMaxIter( JX_Solver solver, JX_Int max_iter )
{
   return( jx_FlexGMRESSetMaxIter( (void *) solver, max_iter ) );
}

JX_Int
JX_FlexGMRESSetPrecond( JX_Solver solver,
                        JX_PtrToSolverFcn precond,
                        JX_PtrToSolverFcn precond_setup,
                        JX_Solver precond_solver )
{
   return( jx_FlexGMRESSetPrecond( (void *) solver,
                                   (JX_Int (*)(void*, void*, void*, void*))precond,
                                   (JX_Int (*)(void*, void*))precond_setup,
                                   (void *) precond_solver ) );
}

JX_Int
JX_FlexGMRESSetPrintLevel( JX_Solver solver, JX_Int level )
{
   return( jx_FlexGMRESSetPrintLevel( (void *) solver, level ) );
}

JX_Int
JX_FlexGMRESSetIsCheckRestarted( JX_Solver solver, JX_Int is_check_restarted )
{
   return( jx_FlexGMRESSetIsCheckRestarted( (void *) solver, is_check_restarted ) );
}

JX_Int
JX_FlexGMRESSetLogging( JX_Solver solver, JX_Int level )
{
   return( jx_FlexGMRESSetLogging( (void *) solver, level ) );
}

JX_Int
JX_FlexGMRESGetNumIterations( JX_Solver solver, JX_Int *num_iterations )
{
   return( jx_FlexGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

JX_Int
JX_FlexGMRESGetConverged( JX_Solver solver, JX_Int *converged )
{
   return( jx_FlexGMRESGetConverged( (void *) solver, converged ) );
}

JX_Int
JX_FlexGMRESGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm )
{
   return( jx_FlexGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

JX_Int
JX_FlexGMRESGetResidual( JX_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return jx_FlexGMRESGetResidual( (void *) solver, (void **) residual );
}

JX_Int
JX_FlexGMRESSetModifyPC( JX_Solver  solver, JX_Int (*modify_pc)(JX_Solver, JX_Int, JX_Real) )
{
   return jx_FlexGMRESSetModifyPC( (void *) solver, (JX_Int(*)(void*, JX_Int, JX_Real))modify_pc );
}

/*!
 * \fn jx_FlexGMRESFunctions *jx_FlexGMRESFunctionsCreate
 */
jx_FlexGMRESFunctions *
jx_FlexGMRESFunctionsCreate(
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
   JX_Int    (*CopyVector)    ( void *x, void *y ),
   JX_Int    (*ClearVector)   ( void *x ),
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x ),
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y ),
   JX_Int    (*PrecondSetup)  ( void *vdata, void *A ),
   JX_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   jx_FlexGMRESFunctions *fgmres_functions;
   fgmres_functions = (jx_FlexGMRESFunctions *)CAlloc( 1, sizeof(jx_FlexGMRESFunctions) );

   fgmres_functions->CAlloc = CAlloc;
   fgmres_functions->Free = Free;
   fgmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   fgmres_functions->CreateVector = CreateVector;
   fgmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   fgmres_functions->DestroyVector = DestroyVector;
   fgmres_functions->MatvecCreate = MatvecCreate;
   fgmres_functions->Matvec = Matvec;
   fgmres_functions->MatvecDestroy = MatvecDestroy;
   fgmres_functions->InnerProd = InnerProd;
   fgmres_functions->CopyVector = CopyVector;
   fgmres_functions->ClearVector = ClearVector;
   fgmres_functions->ScaleVector = ScaleVector;
   fgmres_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   fgmres_functions->precond_setup = PrecondSetup;
   fgmres_functions->precond       = Precond;

   fgmres_functions->modify_pc     = jx_FlexGMRESModifyPCDefault;

   return fgmres_functions;
}

/*!
 * \fn void *jx_FlexGMRESCreate
 */
void *
jx_FlexGMRESCreate( jx_FlexGMRESFunctions *fgmres_functions )
{
   jx_FlexGMRESData *fgmres_data;
 
   fgmres_data = jx_CTAllocF(jx_FlexGMRESData, 1, fgmres_functions);

   fgmres_data->functions = fgmres_functions;
 
   /* set defaults */
   (fgmres_data -> k_dim)              = 20;
   (fgmres_data -> tol)                = 1.0e-06;
   (fgmres_data -> cf_tol)             = 0.0;
   (fgmres_data -> a_tol)              = 0.0; /* abs. residual tol */
   (fgmres_data -> min_iter)           = 0;
   (fgmres_data -> max_iter)           = 1000;
   (fgmres_data -> rel_change)         = 0;
   (fgmres_data -> stop_crit)          = 0; /* rel. residual norm */
   (fgmres_data -> converged)          = 0;
   (fgmres_data -> is_check_restarted) = 0;
   (fgmres_data -> precond_data)       = NULL;
   (fgmres_data -> print_level)        = 0;
   (fgmres_data -> logging)            = 0;
   (fgmres_data -> p)                  = NULL;
   (fgmres_data -> r)                  = NULL;
   (fgmres_data -> w)                  = NULL;
   (fgmres_data -> w_2)                = NULL;
   (fgmres_data -> matvec_data)        = NULL;
   (fgmres_data -> norms)              = NULL;
   (fgmres_data -> log_file_name)      = NULL;
 
   return (void *) fgmres_data;
}

/*!
 * \fn JX_Int jx_FlexGMRESDestroy
 */
 
JX_Int
jx_FlexGMRESDestroy( void *fgmres_vdata )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   JX_Int i;
 
   if (fgmres_data)
   {
      jx_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;
      if ( (fgmres_data->logging>0) || (fgmres_data->print_level) > 0 )
      {
         if ( (fgmres_data -> norms) != NULL ) jx_TFreeF( fgmres_data -> norms, fgmres_functions );
      }
 
      if ( (fgmres_data -> matvec_data) != NULL ) (*(fgmres_functions->MatvecDestroy))(fgmres_data -> matvec_data);
 
      if ( (fgmres_data -> r) != NULL ) (*(fgmres_functions->DestroyVector))(fgmres_data -> r);
      if ( (fgmres_data -> w) != NULL ) (*(fgmres_functions->DestroyVector))(fgmres_data -> w);
      if ( (fgmres_data -> w_2) != NULL ) (*(fgmres_functions->DestroyVector))(fgmres_data -> w_2);

      if ( (fgmres_data -> p) != NULL )
      {
         for (i = 0; i < (fgmres_data -> k_dim+1); i++)
         {
            if ( (fgmres_data -> p)[i] != NULL ) (*(fgmres_functions->DestroyVector))( (fgmres_data -> p) [i]);
         }
         jx_TFreeF( fgmres_data->p, fgmres_functions );
      }

      /* fgmres mod  - space for precond. vectors*/
      if ( (fgmres_data -> pre_vecs) != NULL )
      {
         for (i = 0; i < (fgmres_data -> k_dim + 1); i++)
         {
            if ( (fgmres_data -> pre_vecs)[i] != NULL ) (*(fgmres_functions->DestroyVector))( (fgmres_data -> pre_vecs) [i]);
         }
         jx_TFreeF( fgmres_data->pre_vecs, fgmres_functions );
      }
      /*---*/

      jx_TFreeF( fgmres_data, fgmres_functions );
      jx_TFreeF( fgmres_functions, fgmres_functions );
   }
 
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_FlexGMRESGetResidual
 */
JX_Int
jx_FlexGMRESGetResidual( void *fgmres_vdata, void **residual )
{
   /* returns a pointer to the residual vector */
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
  *residual = fgmres_data->r;
   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_FlexGMRESSetup
 */
JX_Int
jx_FlexGMRESSetup( void *fgmres_vdata, void *A, void *b, void *x )
{
   jx_FlexGMRESData *fgmres_data     = (jx_FlexGMRESData *)fgmres_vdata;
   jx_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;

   JX_Int            k_dim            = (fgmres_data -> k_dim);
   JX_Int            max_iter         = (fgmres_data -> max_iter);
   JX_Int          (*precond_setup)(void*,void*) = (fgmres_functions->precond_setup);
   void          *precond_data     = (fgmres_data -> precond_data);

   JX_Int            rel_change       = (fgmres_data -> rel_change);
 
   (fgmres_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((fgmres_data -> p) == NULL) (fgmres_data -> p) = (void**)(*(fgmres_functions->CreateVectorArray))(k_dim+1,x);
   if ((fgmres_data -> r) == NULL) (fgmres_data -> r) = (*(fgmres_functions->CreateVector))(b);
   if ((fgmres_data -> w) == NULL) (fgmres_data -> w) = (*(fgmres_functions->CreateVector))(b);
 
   if (rel_change)
   {  
      if ((fgmres_data -> w_2) == NULL) (fgmres_data -> w_2) = (*(fgmres_functions->CreateVector))(b);
   }
 
   /* fgmres mod */
   (fgmres_data -> pre_vecs) = (void**)(*(fgmres_functions->CreateVectorArray))(k_dim+1,x); 
   /*---*/

   if ((fgmres_data -> matvec_data) == NULL) (fgmres_data -> matvec_data) = (*(fgmres_functions->MatvecCreate))(A, x);
 
   precond_setup(precond_data, A);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ( (fgmres_data->logging)>0 || (fgmres_data->print_level) > 0 )
   {
      if ((fgmres_data -> norms) == NULL)
         (fgmres_data -> norms) = jx_CTAllocF(JX_Real, max_iter + 1,fgmres_functions);
   }

   if ( (fgmres_data->print_level) > 0 )
   {
      if ((fgmres_data -> log_file_name) == NULL) (fgmres_data -> log_file_name) = (char*)"gmres.out.log";
   }
 
   return jx_error_flag;
}
 
/*!
 * \fn JX_Int jx_FlexGMRESSolve
 */
JX_Int
jx_FlexGMRESSolve( void *fgmres_vdata, void *preOperator, void *A, void *b, void *x )
{
   jx_FlexGMRESData  *fgmres_data   = (jx_FlexGMRESData *)fgmres_vdata;
   jx_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;
   JX_Int 		     k_dim        = (fgmres_data -> k_dim);
   JX_Int               min_iter     = (fgmres_data -> min_iter);
   JX_Int 		     max_iter     = (fgmres_data -> max_iter);
   JX_Real 	     r_tol        = (fgmres_data -> tol);
   JX_Real 	     cf_tol       = (fgmres_data -> cf_tol);
   JX_Real        a_tol        = (fgmres_data -> a_tol);
   void             *matvec_data  = (fgmres_data -> matvec_data);

   void             *r            = (fgmres_data -> r);
   void             *w            = (fgmres_data -> w);

   void            **p            = (fgmres_data -> p);

   /* fgmres  mod*/
   void          **pre_vecs       = (fgmres_data ->pre_vecs);
   /*---*/

   JX_Int 	           (*precond)(void*,void*,void*,void*)   = (fgmres_functions -> precond);
   JX_Int 	            *precond_data = (JX_Int*)(fgmres_data -> precond_data);

   JX_Int             print_level    = (fgmres_data -> print_level);
   JX_Int             logging        = (fgmres_data -> logging);

   JX_Int             is_check_restarted = (fgmres_data -> is_check_restarted);

   JX_Real     *norms          = (fgmres_data -> norms);
   
   JX_Int        break_value = 0;
   JX_Int	      i, j, k;
   JX_Real *rs, **hh, *c, *s; 
   JX_Int        iter; 
   JX_Int        my_id, num_procs;
   JX_Real epsilon, gamma, t, r_norm, b_norm, den_norm;

   JX_Real epsmac = 1.e-16; 
   JX_Real ieee_check = 0.;

   JX_Real cf_ave_0 = 0.0;
   JX_Real cf_ave_1 = 0.0;
   JX_Real weight;
   JX_Real r_norm_0;

   JX_Int 	      (*modify_pc)(void*,JX_Int,JX_Real)   = (fgmres_functions -> modify_pc);

   /* We are not checking rel. change for now... */

   (fgmres_data -> converged) = 0;

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   (*(fgmres_functions->CommInfo))(A,&my_id,&num_procs);
   if ( logging>0 || print_level>0 )
   {
      norms          = (fgmres_data -> norms);
      /* not used yet      log_file_name  = (fgmres_data -> log_file_name);*/
      /* fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays  */
   rs = jx_CTAllocF(JX_Real,k_dim+1,fgmres_functions);
   c = jx_CTAllocF(JX_Real,k_dim,fgmres_functions);
   s = jx_CTAllocF(JX_Real,k_dim,fgmres_functions);

   /* fgmres mod. - need non-modified hessenberg ???? */
   hh = jx_CTAllocF(JX_Real*,k_dim+1,fgmres_functions); 
   for (i=0; i < k_dim+1; i++)
   {
      hh[i] = jx_CTAllocF(JX_Real,k_dim,fgmres_functions); 
   }

   (*(fgmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(fgmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(fgmres_functions->InnerProd))(b,b));

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
        jx_printf("\n\nERROR detected by Hypre ... BEGIN\n");
        jx_printf("ERROR -- jx_FlexGMRESSolve: INFs and/or NaNs detected in input.\n");
        jx_printf("User probably placed non-numerics in supplied b.\n");
        jx_printf("Returning error flag += 101.  Program not terminated.\n");
        jx_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      jx_error(JX_ERROR_GENERIC);
      return jx_error_flag;
   }

   r_norm = sqrt((*(fgmres_functions->InnerProd))(p[0],p[0]));
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
        jx_printf("\n\nERROR detected by Hypre ... BEGIN\n");
        jx_printf("ERROR -- jx_FlexGMRESSolve: INFs and/or NaNs detected in input.\n");
        jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
        jx_printf("Returning error flag += 101.  Program not terminated.\n");
        jx_printf("ERROR detected by Hypre ... END\n\n\n");
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
         if (b_norm == 0.0) jx_printf("Rel_resid_norm actually contains the residual norm\n");
         jx_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
/* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
     den_norm= b_norm;
   }
   else
   {
/* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
     den_norm= r_norm;
   }

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
      }
   }

   /* outer iteration cycle */
   while (iter < max_iter)
   {
   /* initialize first term of hessenberg system */

      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         jx_TFreeF(c,fgmres_functions); 
         jx_TFreeF(s,fgmres_functions); 
         jx_TFreeF(rs,fgmres_functions);

         for (i=0; i < k_dim+1; i++)
         {
            jx_TFreeF(hh[i],fgmres_functions);
         }

         jx_TFreeF(hh,fgmres_functions); 
	     return jx_error_flag;
      }

        /* see if we are already converged and should print the final norm and exit */
      if (r_norm  <= epsilon && iter >= min_iter) 
      {
         (*(fgmres_functions->CopyVector))(b,r);
         (*(fgmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
         r_norm = sqrt((*(fgmres_functions->InnerProd))(r,r));
         if (r_norm <= epsilon)
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

      t = 1.0 / r_norm;

      (*(fgmres_functions->ScaleVector))(t,p[0]);
      i = 0;

      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < k_dim  && iter < max_iter)
      {
         i++;
         iter++;

         (*(fgmres_functions->ClearVector))(pre_vecs[i-1]);

         /* allow some user function here (to change
          * prec. attributes, i.e.tolerances, etc. ? */
         modify_pc(precond_data, iter, r_norm/den_norm );

         /*apply preconditioner and store in pre_vecs */
         precond(precond_data, preOperator, p[i-1], pre_vecs[i-1]);
         /*apply operator and store in p */
         (*(fgmres_functions->Matvec))(matvec_data, 1.0, A, pre_vecs[i-1], 0.0, p[i]);

         /* modified Gram_Schmidt */
         for (j=0; j < i; j++)
         {
            hh[j][i-1] = (*(fgmres_functions->InnerProd))(p[j],p[i]);
            (*(fgmres_functions->Axpy))(-hh[j][i-1],p[j],p[i]);
         }
         t = sqrt((*(fgmres_functions->InnerProd))(p[i],p[i]));
         hh[i][i-1] = t;	
         if (t != 0.0)
         {
            t = 1.0/t;
            (*(fgmres_functions->ScaleVector))(t,p[i]);
         }

         /* done with modified Gram_schmidt and Arnoldi step.
            update factorization of hh */
         for (j = 1; j < i; j++)
         {
            t = hh[j-1][i-1];
            hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
            hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
         }
         t= hh[i][i-1]*hh[i][i-1];
         t+= hh[i-1][i-1]*hh[i-1][i-1];
         gamma = sqrt(t);
         if (gamma == 0.0) gamma = epsmac;
         c[i-1] = hh[i-1][i-1]/gamma;
         s[i-1] = hh[i][i-1]/gamma;
         rs[i] = -hh[i][i-1]*rs[i-1];
         rs[i]/=  gamma;
         rs[i-1] = c[i-1]*rs[i-1];
         /* determine residual norm */
         hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
         r_norm = fabs(rs[i]);

         /* print ? */
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
         if (r_norm  <= epsilon && iter >= min_iter)
         {
            /* no relative change */
            break;
         }
      } /*** end of restart cycle ***/

      /* now compute solution, first solve upper triangular system */

      if (break_value) break;

      rs[i-1] = rs[i-1]/hh[i-1][i-1];
      for (k = i-2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k+1; j < i; j++)
         {
            t -= hh[k][j]*rs[j];
         }
         t+= rs[k];
         rs[k] = t/hh[k][k];
      }

      /* form linear combination of pre_vecs's to get solution */

      (*(fgmres_functions->CopyVector))(pre_vecs[i-1],w);
      (*(fgmres_functions->ScaleVector))(rs[i-1],w);
      for (j = i-2; j >=0; j--) (*(fgmres_functions->Axpy))(rs[j], pre_vecs[j], w);

      /* don't need to un-wind precond... - so now the correction is
       * in w */

      /* update current solution x (in x) */
      (*(fgmres_functions->Axpy))(1.0,w,x);

      /* check for convergence by evaluating the actual residual */
      if (r_norm <= epsilon && iter >= min_iter) 
      {
         /* calculate actual residual norm*/
         (*(fgmres_functions->CopyVector))(b,r);
         (*(fgmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
         r_norm = sqrt( (*(fgmres_functions->InnerProd))(r,r) );

         if (r_norm <= epsilon)
         {
            if ( print_level>1 && my_id == 0 )
            {
               jx_printf("\n\n");
               jx_printf("Final L2 norm of residual: %e\n\n", r_norm);
            }
            (fgmres_data -> converged) = 1;
            break;
         }
         else /* conv. has not occurred, according to true residual */ 
         {
            if ( print_level>0 && my_id == 0) jx_printf("false convergence 2\n");
            if (is_check_restarted == 1)
            {
               (*(fgmres_functions->CopyVector))(r,p[0]);
               i = 0;
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

      if (i) (*(fgmres_functions->Axpy))(rs[i]-1.0,p[i],p[i]);
      for (j=i-1 ; j > 0; j--) (*(fgmres_functions->Axpy))(rs[j],p[j],p[i]);

      if (i)
      {
         (*(fgmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
         (*(fgmres_functions->Axpy))(1.0,p[i],p[0]);
      }
   } /* END of iteration while loop */

   if ( print_level>1 && my_id == 0 ) jx_printf("\n\n"); 

   (fgmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (fgmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (fgmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0) jx_error(JX_ERROR_CONV);

   jx_TFreeF(c,fgmres_functions);
   jx_TFreeF(s,fgmres_functions);
   jx_TFreeF(rs,fgmres_functions);

   for (i=0; i < k_dim+1; i++)
   {
   	  jx_TFreeF(hh[i],fgmres_functions);
   }
   jx_TFreeF(hh,fgmres_functions);

   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetKDim( void *fgmres_vdata, JX_Int k_dim )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> k_dim) = k_dim;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetTol( void *fgmres_vdata, JX_Real tol )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> tol) = tol;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetAbsoluteTol( void *fgmres_vdata, JX_Real a_tol )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> a_tol) = a_tol;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetConvergenceFactorTol( void *fgmres_vdata, JX_Real cf_tol )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> cf_tol) = cf_tol;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetMinIter( void *fgmres_vdata, JX_Int min_iter )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> min_iter) = min_iter;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetMaxIter( void *fgmres_vdata, JX_Int max_iter )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> max_iter) = max_iter;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetStopCrit( void *fgmres_vdata, JX_Int stop_crit )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> stop_crit) = stop_crit;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetPrecond( void *fgmres_vdata,
                        JX_Int (*precond)(void*,void*,void*,void*),
                        JX_Int (*precond_setup)(void*,void*),
                        void *precond_data )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   jx_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;
   (fgmres_functions -> precond) = precond;
   (fgmres_functions -> precond_setup) = precond_setup;
   (fgmres_data -> precond_data) = precond_data;
   return jx_error_flag;
}
 
JX_Int
jx_FlexGMRESSetPrintLevel( void *fgmres_vdata, JX_Int level )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> print_level) = level;
   return jx_error_flag;
}
 
JX_Int
jx_FlexGMRESSetIsCheckRestarted( void *fgmres_vdata, JX_Int is_check_restarted )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> is_check_restarted) = is_check_restarted;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESSetLogging( void *fgmres_vdata, JX_Int level )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   (fgmres_data -> logging) = level;
   return jx_error_flag;
}

JX_Int
jx_FlexGMRESGetNumIterations( void *fgmres_vdata, JX_Int *num_iterations )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
  *num_iterations = (fgmres_data -> num_iterations);
   return jx_error_flag;
}
 
JX_Int
jx_FlexGMRESGetConverged( void *fgmres_vdata, JX_Int *converged )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
  *converged = (fgmres_data -> converged);
   return jx_error_flag;
}
 
JX_Int
jx_FlexGMRESGetFinalRelativeResidualNorm( void *fgmres_vdata, JX_Real *relative_residual_norm )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
  *relative_residual_norm = (fgmres_data -> rel_residual_norm);
   return jx_error_flag;
}
 
JX_Int
jx_FlexGMRESSetModifyPC( void *fgmres_vdata, 
                         JX_Int (*modify_pc)(void *precond_data, JX_Int iteration, JX_Real rel_residual_norm) )
{
   jx_FlexGMRESData *fgmres_data = (jx_FlexGMRESData *)fgmres_vdata;
   jx_FlexGMRESFunctions *fgmres_functions = fgmres_data->functions;
   (fgmres_functions -> modify_pc) = modify_pc;
   return jx_error_flag;
}
 
JX_Int
jx_FlexGMRESModifyPCDefault( void *precond_data, JX_Int iteration, JX_Real rel_residual_norm )
{
   /* Here would could check the number of its and the current
      residual and make some changes to the preconditioner. */
   return 0;
}
