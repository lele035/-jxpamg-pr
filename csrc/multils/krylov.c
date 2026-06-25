/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 * krylov.c
 *
 * Created by peghoty 2010/12/14
 *
 * Xiangtan University
 * peghoty@163.com
 *  
 */

#include "jx_multils.h"

/*!
 * \fn JX_Int fsls_CSRPAGMRESKrylovIdentitySetup
 * \brief Identity Setup for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESKrylovIdentity
 * \brief Identity Preconditioning for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESKrylovIdentity( void *vdata, void *b, void *x )
{
   return( fsls_SeqVectorCopy( b, x ) );
}

/*!
 * \fn fsls_CSRPAGMRESFunctions *fsls_CSRPAGMRESFunctionsCreate
 * \brief Function Creating for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
fsls_CSRPAGMRESFunctions *
fsls_CSRPAGMRESFunctionsCreate( JX_Int  (*PrecondSetup)  ( void *vdata, void *A ),
                                JX_Int  (*Precond)       ( void *vdata, void *b, void *x )  )
{
   fsls_CSRPAGMRESFunctions *pagmres_functions = NULL;
   pagmres_functions = (fsls_CSRPAGMRESFunctions *)calloc( 1, sizeof(fsls_CSRPAGMRESFunctions));

   /* default preconditioner must be set here but can be changed later... */
   pagmres_functions -> precond_setup = PrecondSetup;
   pagmres_functions -> precond       = Precond;

   return pagmres_functions;
}

/*!
 * \fn fsls_CSRPAGMRESData *fsls_CSRPAGMRESCreate 
 * \brief Create a PAGMRES-Data Object
 * \author peghoty
 * \date 2010/12/14
 */
fsls_CSRPAGMRESData *fsls_CSRPAGMRESCreate()
{
   fsls_CSRPAGMRESData      *pagmres_data      = NULL;
   fsls_CSRPAGMRESFunctions *pagmres_functions = NULL;
   
   pagmres_functions = fsls_CSRPAGMRESFunctionsCreate( fsls_CSRPAGMRESKrylovIdentitySetup,
                                                       fsls_CSRPAGMRESKrylovIdentity );   
  
   pagmres_data = fsls_CTAlloc(fsls_CSRPAGMRESData, 1);

   (pagmres_data -> functions) = pagmres_functions;

   /* set defaults */
   (pagmres_data -> k_dim_max)      = 30;
   (pagmres_data -> k_dim_min)      = 3;
   (pagmres_data -> k_dim_d)        = 3;
   (pagmres_data -> cr_max)         = 0.990;
   (pagmres_data -> cr_min)         = 0.174;      
   (pagmres_data -> tol)            = 1.0e-06; // relative residual tol
   (pagmres_data -> cf_tol)         = 0.0;
   (pagmres_data -> min_iter)       = 0;
   (pagmres_data -> max_iter)       = 1000;
   (pagmres_data -> stop_crit)      = 0;       // rel. residual norm - this is obsolete!
   (pagmres_data -> converged)      = 0;
   (pagmres_data -> precond_data)   = NULL;
   (pagmres_data -> print_level)    = 0;
   (pagmres_data -> p)              = NULL;
   (pagmres_data -> r)              = NULL;
   (pagmres_data -> w)              = NULL;
   (pagmres_data -> norms)          = NULL;

   return (void *) pagmres_data;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetKDimMax
 * \brief Set parameter 'k_dim_max' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetKDimMax( void *pagmres_vdata, JX_Int k_dim_max )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> k_dim_max) = k_dim_max;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetKDimMin
 * \brief Set parameter 'k_dim_min' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetKDimMin( void *pagmres_vdata, JX_Int k_dim_min )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> k_dim_min) = k_dim_min;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetKDimD
 * \brief Set parameter 'k_dim_d' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetKDimD( void *pagmres_vdata, JX_Int k_dim_d )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> k_dim_d) = k_dim_d;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetCRMax
 * \brief Set parameter 'cr_max' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetCRMax( void *pagmres_vdata, JX_Real cr_max )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> cr_max) = cr_max;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetCRMin
 * \brief Set parameter 'cr_min' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetCRMin( void *pagmres_vdata, JX_Real cr_min )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> cr_min) = cr_min;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetTol
 * \brief Set parameter 'tol' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetTol( void *pagmres_vdata, JX_Real tol )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> tol) = tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetConvergenceFactorTol
 * \brief Set parameter 'cf_tol' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetConvergenceFactorTol( void *pagmres_vdata, JX_Real cf_tol )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> cf_tol) = cf_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetMinIter
 * \brief Set parameter 'min_iter' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetMinIter( void *pagmres_vdata, JX_Int min_iter )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> min_iter) = min_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetMaxIter
 * \brief Set parameter 'max_iter' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetMaxIter( void *pagmres_vdata, JX_Int max_iter )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> max_iter) = max_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetStopCrit
 * \brief Set parameter 'stop_crit ' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetStopCrit( void *pagmres_vdata, JX_Int stop_crit )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> stop_crit) = stop_crit;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetPrintLevel
 * \brief Set parameter 'print_level' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetPrintLevel( void *pagmres_vdata, JX_Int print_level )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   (pagmres_data -> print_level) = print_level;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESGetNumIterations
 * \brief Get parameter 'num_iterations ' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESGetNumIterations( void *pagmres_vdata, JX_Int *num_iterations )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   *num_iterations = (pagmres_data -> num_iterations);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESGetConverged
 * \brief Get parameter 'converged' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESGetConverged( void *pagmres_vdata, JX_Int  *converged )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   *converged = (pagmres_data -> converged);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESGetFinalRelativeResidualNorm
 * \brief Get parameter 'relative_residual_norm' for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */ 
JX_Int
fsls_CSRPAGMRESGetFinalRelativeResidualNorm( void *pagmres_vdata, JX_Real *relative_residual_norm )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   *relative_residual_norm = (pagmres_data -> rel_residual_norm);
   return 0;
} 

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetPrecond
 * \brief Set preconditioner for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetPrecond( void *pagmres_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   fsls_CSRPAGMRESFunctions *pagmres_functions = pagmres_data->functions;

   (pagmres_functions -> precond)       = precond;
   (pagmres_functions -> precond_setup) = precond_setup;
   (pagmres_data -> precond_data)       = precond_data;
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSetup
 * \brief Setup phase for PAGMRES
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSetup( void *pagmres_vdata, fsls_CSRMatrix *A, fsls_Vector *b, fsls_Vector *x )
{
   fsls_CSRPAGMRESData      *pagmres_data      = pagmres_vdata;
   fsls_CSRPAGMRESFunctions *pagmres_functions = (pagmres_data -> functions);

   JX_Int     k_dim              = (pagmres_data -> k_dim_max);
   JX_Int     max_iter           = (pagmres_data -> max_iter);
   JX_Int     (*precond_setup)() = (pagmres_functions -> precond_setup);
   void   *precond_data       = (pagmres_data -> precond_data);

   JX_Int     i;
   /* I change 'size = fsls_VectorSize(b)' into 'size = fsls_CSRMatrixNumRows(A)'
      since 'b' maybe NULL sometimes. peghoty, 2011/06/24 */
   JX_Int     size        = fsls_CSRMatrixNumRows(A);
   JX_Int     print_level = pagmres_data -> print_level;
   JX_Real *norms       = NULL;   
   
   fsls_Vector **p   = NULL;
   fsls_Vector  *r   = NULL;
   fsls_Vector  *w   = NULL;
 
   if ((pagmres_data -> p) == NULL)
   {
      p = fsls_CTAlloc(fsls_Vector *, k_dim+1);
      for (i = 0; i < k_dim+1; i ++)
      {
         p[i] = fsls_SeqVectorCreate(size);
         fsls_SeqVectorInitialize(p[i]);
      }
      (pagmres_data -> p) = (void **)p; 
   }  
      
   if ((pagmres_data -> r) == NULL)
   {
      r = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(r);
      (pagmres_data -> r) = r;
   }
   
   if ((pagmres_data -> w) == NULL)
   {
      w = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(w);
      (pagmres_data -> w) = w;
   }
   
   /* preconditioning Setup */
   precond_setup(precond_data, A, b, x);

   if (print_level > 0)
   {
      if ((pagmres_data -> norms) == NULL)
      {
         norms = fsls_CTAlloc(JX_Real, max_iter+1);
         (pagmres_data -> norms) = norms;
      }
   }

   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESDestroy
 * \brief Destroy a PAGMRES object
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESDestroy( void *pagmres_vdata )
{
   fsls_CSRPAGMRESData *pagmres_data = pagmres_vdata;
   JX_Int i;
   JX_Int k_dim = (pagmres_data -> k_dim_max);
 
   if (pagmres_data)
   {
      if ( (pagmres_data -> print_level) > 0 )
      {
         if ( (pagmres_data -> norms) != NULL )
         {
            fsls_TFree(pagmres_data -> norms);
            (pagmres_data -> norms) = NULL;
         }
      }
      if ( (pagmres_data -> r) != NULL )
      {
         fsls_SeqVectorDestroy(pagmres_data -> r);
         (pagmres_data -> r) = NULL;
      }      
      if ( (pagmres_data -> w) != NULL )
      {
         fsls_SeqVectorDestroy(pagmres_data -> w);
         (pagmres_data -> w) = NULL;
      }  
      if ( (pagmres_data -> p) != NULL )
      {
         for (i = 0; i < k_dim+1; i ++)
         {
            if ( (pagmres_data -> p)[i] != NULL )
            {
	       fsls_SeqVectorDestroy((pagmres_data -> p)[i]);
	       (pagmres_data -> p)[i] = NULL;
	    }
         }
         fsls_TFree(pagmres_data -> p);
         pagmres_data -> p = NULL;
      }
      
      fsls_TFree(pagmres_data -> functions);
      fsls_TFree(pagmres_data);            
   }
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPAGMRESSolve
 * \brief Solve phase for PAGMRES
 * \note This can be easily reduced to general GMRES(m)
 *       when we set k_dim_max = k_dim_min or k_dim_d = 0.
 * \author peghoty
 * \date 2010/12/14
 */
JX_Int
fsls_CSRPAGMRESSolve( void            *pagmres_vdata,
                      fsls_CSRMatrix  *A,
                      fsls_Vector     *b,
                      fsls_Vector     *x )
{
   fsls_CSRPAGMRESData       *pagmres_data      = pagmres_vdata;
   fsls_CSRPAGMRESFunctions  *pagmres_functions = (pagmres_data -> functions);
   JX_Int 	                     (*precond)()       = (pagmres_functions -> precond);
   void                      *precond_data      = (pagmres_data -> precond_data);
 
   JX_Int      k_dim_max    = (pagmres_data -> k_dim_max);   // default: 30
   JX_Int      k_dim_min    = (pagmres_data -> k_dim_min);   // default: 3
   JX_Int      d            = (pagmres_data -> k_dim_d);     // default: 3
   JX_Int      min_iter     = (pagmres_data -> min_iter);    // default: 0
   JX_Int      max_iter     = (pagmres_data -> max_iter);    // default: 1000
   JX_Real   tol          = (pagmres_data -> tol);         // default: 1.0e-06
   JX_Real   cr_max       = (pagmres_data -> cr_max);      // default: 0.99
   JX_Real   cr_min       = (pagmres_data -> cr_min);      // default: 0.174   
   JX_Int      print_level  = (pagmres_data -> print_level); // default: 0
      
   void    *r            = (pagmres_data -> r);
   void    *w            = (pagmres_data -> w);
   void   **p            = (pagmres_data -> p);
   JX_Real  *norms        = (pagmres_data -> norms); 

   JX_Int	    i,j,k;
   JX_Int      k_dim_max_1 = k_dim_max + 1;
   JX_Int      iter,k_dim = 0; 
   JX_Real   epsilon, gamma, t;
   JX_Real   r_norm, b_norm, den_norm;
   JX_Real   epsmac = 1.e-16; 
   JX_Real   r_norm_old = 0.0;
   JX_Real  *work = NULL;
   JX_Real  *rs   = NULL;
   JX_Real  *c    = NULL;
   JX_Real  *s    = NULL; 
   JX_Real **hh   = NULL;
   JX_Real  *hhs  = NULL;  
   JX_Real   cr   = 1.0;
   
   (pagmres_data -> converged) = 0;

   if (print_level > 0)
   {
      norms = (pagmres_data -> norms);
   }

   /* initialize work arrays */
   work = fsls_CTAlloc(JX_Real, (k_dim_max + 4)*k_dim_max + 1);
   hh   = fsls_CTAlloc(JX_Real*, k_dim_max_1);
   rs = work; c = rs + k_dim_max_1; s = c + k_dim_max;
   hhs = s + k_dim_max;
   for (i = 0; i < k_dim_max_1; i ++) hh[i] = hhs + i*k_dim_max;

   /* compute initial residual */
   fsls_SeqVectorCopy(b, p[0]);
   fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, p[0]);
  
   b_norm = sqrt( fsls_SeqVectorInnerProd(b,b) );
   r_norm = sqrt( fsls_SeqVectorInnerProd(p[0],p[0]) );

   if (print_level > 0)
   {
      norms[0] = r_norm;
      jx_printf("\n >>> L2 norm of b: %e\n", b_norm);
      if (b_norm == 0.0)
      {
         jx_printf(" >>> Rel_resid_norm actually contains the residual norm\n");
      }
      jx_printf(" >>> Initial L2 norm of residual: %e\n\n", r_norm);
   }
 
   iter = 0;

   if (b_norm > 0.0) 
      den_norm= b_norm;
   else
      den_norm= r_norm;  
   epsilon = tol*den_norm;
   
   /* so now our stop criteria is |r_i| <= epsilon */
   
   if (print_level > 0)
   {
      if (b_norm > 0.0)
      {
         jx_printf(" ===================================================\n");
         jx_printf("   iter   resid.norm    conv.factor  rel.res.norm   \n");
         jx_printf(" ===================================================\n");
      }
      else
      {
         jx_printf(" =====================================\n");
         jx_printf("   iter    resid.norm   conv.factor   \n");
         jx_printf(" =====================================\n");
      }
   }

   /* outer iteration cycle */
   while (iter < max_iter)
   {
        /* initialize first term of hessenberg system */
	rs[0] = r_norm;
	r_norm_old = r_norm;
        if (r_norm == 0.0)
        {
           /* free some stuff */
           fsls_TFree(work);
           fsls_TFree(hh);
	   return 11;
	}
	
	/* adjust the restart parameter */
	if (cr > cr_max || iter == 0)
	{
	   k_dim = k_dim_max;
	}
	else if (cr < cr_min)
	{
	   k_dim = k_dim;
	}
	else
	{
	   if (k_dim - d > k_dim_min)
	   {
	      k_dim = k_dim - d;
	      //jx_printf(" >>> k_dim decreases by d!! \n");
	   }
	   else
	   {
	      k_dim = k_dim_max;
	      //jx_printf(" >>> k_dim becomes k_dim_max!! \n");
	   }
	}

        /* see if we are already converged and should print the final norm and exit */
	if (r_norm  <= epsilon && iter >= min_iter) 
        {
           fsls_SeqVectorCopy(b,r);
           fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
           r_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
              
           if (r_norm <= epsilon)
           {
              if (print_level > 0)
              {
                 jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
              }
              break;
           }
           else
           {
              if (print_level > 0) jx_printf(" >>> false convergence (1)\n");
           }
	}

      	t = 1.0 / r_norm;
	fsls_SeqVectorScale(t,p[0]);
	
	
        /*--------- RESTART CYCLE (right-preconditioning) --------*/	
	i = 0;
        while (i < k_dim && iter < max_iter)
	{
           i ++;
           iter ++;
           
           fsls_SeqVectorSetConstantValues(r, 0.0);
           precond(precond_data, p[i-1], r);
           
           fsls_CSRMatrixMatvec01(1.0, A, r, 0.0, p[i]);
           
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = fsls_SeqVectorInnerProd(p[j],p[i]);
              fsls_SeqVectorAxpy(-hh[j][i-1],p[j],p[i]);
           }
           t = sqrt( fsls_SeqVectorInnerProd(p[i],p[i]) );
           hh[i][i-1] = t;	
           if (t != 0.0)
           {
              t = 1.0/t;
              fsls_SeqVectorScale(t,p[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step.
              update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t  = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1]  = hh[i-1][i-1] / gamma;
           s[i-1]  = hh[i][i-1] / gamma;
           rs[i]   = -s[i-1]*rs[i-1];
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if (print_level > 0)
           {
              norms[iter] = r_norm;

              if (b_norm > 0.0)
                 jx_printf("% 5d    %e    %f    %e\n", iter, norms[iter],
                                                    norms[iter]/norms[iter-1], norms[iter]/b_norm);
              else
                 jx_printf("% 5d    %e     %f\n", iter, norms[iter], norms[iter]/norms[iter-1]);
           }

           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              break;
           }  
	} 
	/*----------- End of RESTART CYCLE ---------------*/
	

	/* now compute solution, first solve upper triangular system */
	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i-2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k+1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        fsls_SeqVectorCopy(p[i-1],w);
        fsls_SeqVectorScale(rs[i-1],w);
        
        for (j = i-2; j >= 0; j --)
        {
           fsls_SeqVectorAxpy(rs[j], p[j], w);
        }

        /* find correction (in r) */
	fsls_SeqVectorSetConstantValues(r, 0.0);
	precond(precond_data, w, r);
	
        /* update current solution x (in x) */
	fsls_SeqVectorAxpy(1.0,r,x);
         
        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm*/
           fsls_SeqVectorCopy(b,r);
           fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
           r_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
           
           if (r_norm <= epsilon)
           {
              if (print_level > 0)
              {
                 jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
              }
              (pagmres_data -> converged) = 1;
              break;
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if (print_level > 0)
              {
                 jx_printf(" >>> false convergence (2)\n");
              }
              fsls_SeqVectorCopy(r,p[0]);
              i = 0;
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j]   =  c[j-1]*rs[j];
	}
        
        if (i) 
        {
           fsls_SeqVectorAxpy(rs[i]-1.0,p[i],p[i]);
        }
        for (j = i-1 ; j > 0; j --)
        {
           fsls_SeqVectorAxpy(rs[j],p[j],p[i]);
        }
        
        if (i)
        {
           fsls_SeqVectorAxpy(rs[0]-1.0,p[0],p[0]);
           fsls_SeqVectorAxpy(1.0,p[i],p[0]);
        }
        
        /* compute the convergence rate  newly added */
        cr = r_norm / r_norm_old;
        
   } 
   /* END of iteration while loop */
   
   (pagmres_data -> num_iterations) = iter;  // get the number of iterations

   /* get the last relative residual-norm */
   if (b_norm > 0.0)
   {
      (pagmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (pagmres_data -> rel_residual_norm) = r_norm;
   }
   
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf("\n Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
   }
  
  /*-------------------------------------------
   * Free some stuff
   *------------------------------------------*/   
   fsls_TFree(work);
   fsls_TFree(hh);
   
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABKrylovIdentitySetup
 * \brief Identity Setup for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}


/*!
 * \fn JX_Int fsls_CSRPBICGSTABKrylovIdentity
 * \brief Identity Preconditioning for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABKrylovIdentity( void *vdata, void *b, void *x )
{
   return( fsls_SeqVectorCopy( b, x ) );
}

/*!
 * \fn fsls_CSRPBICGSTABFunctions *fsls_CSRPBICGSTABFunctionsCreate
 * \brief Function Creating for PBicgstab
 * \author peghoty
 * \date 2010/08/26
 */
fsls_CSRPBICGSTABFunctions *
fsls_CSRPBICGSTABFunctionsCreate( JX_Int  (*PrecondSetup)  ( void *vdata, void *A ),
                                  JX_Int  (*Precond)       ( void *vdata, void *b, void *x ) )
{
   fsls_CSRPBICGSTABFunctions *pbicgstab_functions = NULL;
   pbicgstab_functions = (fsls_CSRPBICGSTABFunctions *)calloc( 1, sizeof(fsls_CSRPBICGSTABFunctions));

   /* default preconditioner must be set here but can be changed later... */
   (pbicgstab_functions -> precond_setup) = PrecondSetup;
   (pbicgstab_functions -> precond)       = Precond;

   return pbicgstab_functions;
}


/*!
 * \fn fsls_CSRPBICGSTABData *fsls_CSRPBICGSTABCreate 
 * \brief Create a PBicgstab-Data Object
 * \author peghoty
 * \date 2010/08/26
 */
fsls_CSRPBICGSTABData *fsls_CSRPBICGSTABCreate()
{
   fsls_CSRPBICGSTABData      *pbicgstab_data      = NULL;
   fsls_CSRPBICGSTABFunctions *pbicgstab_functions = NULL;
   
   pbicgstab_functions = fsls_CSRPBICGSTABFunctionsCreate( fsls_CSRPBICGSTABKrylovIdentitySetup, 
                                                            fsls_CSRPBICGSTABKrylovIdentity );
   
   pbicgstab_data = fsls_CTAlloc(fsls_CSRPBICGSTABData, 1);

   (pbicgstab_data -> functions) = pbicgstab_functions;

   /* set defaults */
   (pbicgstab_data -> tol)          = 1.0e-06;
   (pbicgstab_data -> min_iter)     = 0;
   (pbicgstab_data -> max_iter)     = 1000;
   (pbicgstab_data -> stop_crit)    = 0;
   (pbicgstab_data -> a_tol)        = 0.0;
   (pbicgstab_data -> cf_tol)       = 0.0;  // added by peghoty
   (pbicgstab_data -> precond_data) = NULL;
   (pbicgstab_data -> print_level)  = 0;
   (pbicgstab_data -> p)            = NULL;
   (pbicgstab_data -> q)            = NULL;
   (pbicgstab_data -> r)            = NULL;
   (pbicgstab_data -> r0)           = NULL;
   (pbicgstab_data -> s)            = NULL;
   (pbicgstab_data -> v)            = NULL;
   (pbicgstab_data -> norms)        = NULL;

   return pbicgstab_data;
}


/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetTol
 * \brief Set parameter 'tol' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetTol( void *pbicgstab_vdata, JX_Real tol )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> tol) = tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetAbsoluteTol
 * \brief Set parameter 'a_tol' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetAbsoluteTol( void *pbicgstab_vdata, JX_Real a_tol )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> a_tol) = a_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetConvergenceFactorTol
 * \brief Set parameter 'cf_tol' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetConvergenceFactorTol( void *pbicgstab_vdata, JX_Real cf_tol )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> cf_tol) = cf_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetMinIter
 * \brief Set parameter 'min_iter' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetMinIter( void *pbicgstab_vdata, JX_Int min_iter )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> min_iter) = min_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetMaxIter
 * \brief Set parameter 'max_iter' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetMaxIter( void *pbicgstab_vdata, JX_Int max_iter )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> max_iter) = max_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetStopCrit
 * \brief Set parameter 'stop_crit' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetStopCrit( void *pbicgstab_vdata, JX_Int stop_crit )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> stop_crit) = stop_crit;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetPrintLevel
 * \brief Set parameter 'print_level' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetPrintLevel( void *pbicgstab_vdata, JX_Int print_level )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   (pbicgstab_data -> print_level) = print_level;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABGetConverged
 * \brief Set parameter 'converged' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABGetConverged( void *pbicgstab_vdata, JX_Int *converged )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   *converged = (pbicgstab_data -> converged);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABGetNumIterations
 * \brief Get parameter 'num_iterations' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABGetNumIterations( void *pbicgstab_vdata, JX_Int *num_iterations )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   *num_iterations = (pbicgstab_data -> num_iterations);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABGetFinalRelativeResidualNorm
 * \brief Get parameter 'relative_residual_norm' for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABGetFinalRelativeResidualNorm( void *pbicgstab_vdata, JX_Real *relative_residual_norm )
{
   fsls_CSRPBICGSTABData *pbicgstab_data = pbicgstab_vdata;
   *relative_residual_norm = (pbicgstab_data -> rel_residual_norm);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetPrecond
 * \brief Set preconditioner for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetPrecond( void *pbicgstab_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data )
{
   fsls_CSRPBICGSTABData      *pbicgstab_data      = pbicgstab_vdata;
   fsls_CSRPBICGSTABFunctions *pbicgstab_functions = (pbicgstab_data -> functions);

   (pbicgstab_functions -> precond)       = precond;
   (pbicgstab_functions -> precond_setup) = precond_setup;
   (pbicgstab_data -> precond_data)       = precond_data;
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABSetup
 * \brief Setup phase for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSetup( void *pbicgstab_vdata, fsls_CSRMatrix *A, fsls_Vector *b, fsls_Vector *x )
{
   fsls_CSRPBICGSTABData      *pbicgstab_data      = pbicgstab_vdata;
   fsls_CSRPBICGSTABFunctions *pbicgstab_functions = pbicgstab_data -> functions;

   JX_Int   max_iter           = (pbicgstab_data -> max_iter);
   JX_Int   (*precond_setup)() = (pbicgstab_functions -> precond_setup);
   void  *precond_data      = (pbicgstab_data -> precond_data);
   /* I change 'size = fsls_VectorSize(b)' into 'size = fsls_CSRMatrixNumRows(A)'
      since 'b' maybe NULL sometimes. peghoty, 2011/06/24 */
   JX_Int size                 = fsls_CSRMatrixNumRows(A);
   
   JX_Real *norms   = NULL;
   fsls_Vector *p  = NULL;
   fsls_Vector *q  = NULL;
   fsls_Vector *r  = NULL;  
   fsls_Vector *r0 = NULL;
   fsls_Vector *s  = NULL;
   fsls_Vector *v  = NULL; 
  
   if ((pbicgstab_data -> p) == NULL)
   {
      p = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(p);
      (pbicgstab_data -> p) = p;
   }
   if ((pbicgstab_data -> q) == NULL)
   {
      q = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(q);
      (pbicgstab_data -> q) = q;
   }
   if ((pbicgstab_data -> r) == NULL)
   {
      r = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(r);
      (pbicgstab_data -> r) = r;
   }
   if ((pbicgstab_data -> r0) == NULL)
   {
      r0 = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(r0);
      (pbicgstab_data -> r0) = r0;
   }
   if ((pbicgstab_data -> s) == NULL)
   {
      s = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(s);
      (pbicgstab_data -> s) = s;
   }
   if ((pbicgstab_data -> v) == NULL)
   {
      v = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(v);
      (pbicgstab_data -> v) = v;
   }
 
   /* preconditioning Setup */
   precond_setup(precond_data, A, b, x);
 
  /*-----------------------------------------------------
   * Allocate space for log info
   *----------------------------------------------------*/
   if ((pbicgstab_data -> print_level) > 0)
   {
      if ((pbicgstab_data -> norms) == NULL)
      {
         norms = fsls_CTAlloc(JX_Real, max_iter+1);
         (pbicgstab_data -> norms) = norms;
      }
   }
 
   return 0;
}


/*!
 * \fn JX_Int fsls_CSRPBICGSTABSolve
 * \brief Solve phase for PBICGSTAB
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABSolve( void            *pbicgstab_vdata,
                        fsls_CSRMatrix  *A,
                        fsls_Vector     *b,
                        fsls_Vector     *x  )
{
   fsls_CSRPBICGSTABData       *pbicgstab_data      = pbicgstab_vdata;
   fsls_CSRPBICGSTABFunctions  *pbicgstab_functions = (pbicgstab_data -> functions);
   JX_Int                         (*precond)()         = (pbicgstab_functions -> precond);
   JX_Int 	                        *precond_data       = (pbicgstab_data -> precond_data);   

   JX_Int      min_iter  = (pbicgstab_data -> min_iter);  // default: 0
   JX_Int      max_iter  = (pbicgstab_data -> max_iter);  // default: 1000
   JX_Int      stop_crit = (pbicgstab_data -> stop_crit); // default: 0
   JX_Real   r_tol     = (pbicgstab_data -> tol);       // default: 1.0e-06
   JX_Real   cf_tol    = (pbicgstab_data -> cf_tol);    // default: 0.0
   JX_Real   a_tol     = (pbicgstab_data -> a_tol);     // default: 0.0
  
   void    *r   = (pbicgstab_data -> r);
   void    *r0  = (pbicgstab_data -> r0);
   void    *s   = (pbicgstab_data -> s);
   void    *v   = (pbicgstab_data -> v);
   void    *p   = (pbicgstab_data -> p);
   void    *q   = (pbicgstab_data -> q);

   JX_Int      print_level = (pbicgstab_data -> print_level);
   JX_Real  *norms       = (pbicgstab_data -> norms);

   JX_Int      iter; 
   JX_Real   alpha, beta, gamma, epsilon;
   JX_Real   temp, res;
   JX_Real   r_norm, b_norm;
   JX_Real   epsmac = 1.e-128; 
   JX_Real   ieee_check = 0.;
   JX_Real   cf_ave_0 = 0.0;
   JX_Real   cf_ave_1 = 0.0;
   JX_Real   weight;
   JX_Real   r_norm_0;
   JX_Real   den_norm;
   
   (pbicgstab_data -> converged) = 0; 

   if (print_level > 0)
   {
      norms = (pbicgstab_data -> norms);
   }

   /* initialize work arrays */
   fsls_SeqVectorCopy(b,r0);

   /* compute initial residual */
   fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r0);
   fsls_SeqVectorCopy(r0,r);
   fsls_SeqVectorCopy(r0,p);

   b_norm = sqrt( fsls_SeqVectorInnerProd(b,b) );

  /*----------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an error
   * flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (b_norm != 0.0) 
   {
      ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF 
      */
      if (print_level > 0)
      {
         jx_printf("\n\nERROR detected by FSLS ...  BEGIN\n");
         jx_printf("ERROR -- fsls_CSRPBICGSTAB: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by FSLS ...  END\n\n\n");
      }
      return 100;
   }

   res = fsls_SeqVectorInnerProd(r0,r0);
   r_norm = sqrt(res);
   r_norm_0 = r_norm;

  /*----------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an error
   * flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (r_norm != 0.0) 
   {
      ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF 
      */
      if (print_level > 0)
      {
         jx_printf("\n\nERROR detected by FSLS ...  BEGIN\n");
         jx_printf("ERROR -- fsls_CSRPBICGSTAB: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by FSLS ...  END\n\n\n");
      }
      return 100;
   }

   if (print_level > 0)
   {
      norms[0] = r_norm;
      if (print_level > 0)
      {
  	 jx_printf("\n >>> L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0) jx_printf(" >>> Rel_resid_norm actually contains the residual norm\n");
         jx_printf(" >>> Initial L2 norm of residual: %e\n\n", r_norm);
      }
   }
  
   iter = 0;

   if (b_norm > 0.0)
   {
      den_norm = b_norm; // convergence criterion |r_i| <= r_tol*|b| if |b| > 0
   }
   else
   {
      den_norm = r_norm; // convergence criterion |r_i| <= r_tol*|r0| if |b| = 0
   }

   /* convergence criterion |r_i| <= r_tol/a_tol, absolute residual norm */
   if (stop_crit)
   {
      if (a_tol == 0.0) 
      {
         /* this is for backwards compatibility
            (accomodating setting stop_crit to 1, but not setting a_tol) -
            eventually we will get rid of the stop_crit flag as with GMRES */
         epsilon = r_tol;
      }
      else
      {
         /* this means new interface fcn called */
         epsilon = a_tol;
      }
   }
   else /* default convergence test (stop_crit = 0) */
   {
      /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
         den_norm = |r_0| or |b|
         note: default for a_tol is 0.0, so relative residual criteria is 
               used unless user also specifies a_tol or sets r_tol = 0.0, 
               which means absolute tol only is checked.  
      */
      epsilon = fsls_max(a_tol, r_tol*den_norm);
   }
   
   
   if (print_level > 0)
   {
      if (b_norm > 0.0)
      {
          jx_printf(" ===================================================\n");
          jx_printf("   iter   resid.norm    conv.factor  rel.res.norm   \n");
          jx_printf(" ===================================================\n");
      }
      else
      {
          jx_printf(" =====================================\n");
          jx_printf("   iter   resid.norm    conv.factor   \n");
          jx_printf(" =====================================\n");
      }
   }

   if (b_norm > 0.0)
   {
      (pbicgstab_data -> rel_residual_norm) = r_norm / b_norm;
   }
 
   
   while (iter < max_iter)
   {
        if (r_norm == 0.0)
        {
	   return 2;
	}

        /* check for convergence, evaluate actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
	   fsls_SeqVectorCopy(b,r);
           fsls_CSRMatrixMatvec01(-1.0,A,x,1.0,r);
	   r_norm = sqrt(fsls_SeqVectorInnerProd(r,r));
	   if (r_norm <= epsilon)
           {
              if (print_level > 0)
              {
                 jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
              }
              (pbicgstab_data -> converged) = 1;
              break;
           }
	   else
	   {
	      fsls_SeqVectorCopy(r,p);
	   }
	}

       /*--------------------------------------------------------------
        * Optional test to see if adequate progress is being made.
        * The average convergence factor is recorded and compared
        * against the tolerance 'cf_tol'. The weighting factor is
        * intended to pay more attention to the test when an accurate
        * estimate for average convergence factor is available.
        *-------------------------------------------------------------*/

        if (cf_tol > 0.0)
        {
           cf_ave_0 = cf_ave_1;
           cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

           weight   = fabs(cf_ave_1 - cf_ave_0);
           weight   = weight / fsls_max(cf_ave_1, cf_ave_0);
           weight   = 1.0 - weight;
           
           if (weight * cf_ave_1 > cf_tol) break;
        }

        iter ++;

	fsls_SeqVectorSetConstantValues(v,0.0);
        precond(precond_data, p, v);

        fsls_CSRMatrixMatvec01(1.0, A, v, 0.0, q);
      	temp = fsls_SeqVectorInnerProd(r0,q);
      	if (fabs(temp) >= epsmac)
      	{
	   alpha = res / temp;
	}
	else
	{
	   jx_printf(" >>> BiCGSTAB broke down, divide by near zero!!\n");
	   return(1);
	}

	fsls_SeqVectorAxpy(alpha, v, x);
	fsls_SeqVectorAxpy(-alpha, q, r);
	fsls_SeqVectorSetConstantValues(v, 0.0);
	precond(precond_data, r, v);
        
        fsls_CSRMatrixMatvec01(1.0, A, v, 0.0, s);
      	gamma = fsls_SeqVectorInnerProd(r,s) / fsls_SeqVectorInnerProd(s,s);
	fsls_SeqVectorAxpy(gamma,v,x);
	fsls_SeqVectorAxpy(-gamma,s,r);
      	if (fabs(res) >= epsmac)
      	{
           beta = 1.0 / res;
        }
	else
	{
	   jx_printf(" >>> BiCGSTAB broke down, res = 0!! \n");
	   return(2);
	}
        res = fsls_SeqVectorInnerProd(r0,r);
        beta *= res;    
	fsls_SeqVectorAxpy(-gamma,q,p);
      	if (fabs(gamma) >= epsmac)
      	{
           fsls_SeqVectorScale((beta*alpha/gamma),p);
        }
	else
	{
	   jx_printf(" >>> BiCGSTAB broke down, gamma = 0!! \n");
	   return(3);
	}
	fsls_SeqVectorAxpy(1.0,r,p);

	r_norm = sqrt(fsls_SeqVectorInnerProd(r,r));
	
	if (print_level > 0)
	{
	   norms[iter] = r_norm;
	}

        if (print_level > 0)
	{
           if (b_norm > 0.0)
           {
              jx_printf("% 5d    %e    %f    %e\n", iter, norms[iter],
                      norms[iter]/norms[iter-1], norms[iter]/b_norm);
           }
           else
           {
              jx_printf("% 5d    %e    %f\n", iter, norms[iter],
                     norms[iter]/norms[iter-1]);
           }
	}
   }  
   /* End the bicgstab loop */
   
   /* get the number of iteration */
   (pbicgstab_data -> num_iterations) = iter;  

   if (b_norm > 0.0)
   {
      (pbicgstab_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (pbicgstab_data -> rel_residual_norm) = r_norm;
   }

   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf("\n Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
   }

   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPBICGSTABDestroy
 * \brief Destroy a PBICGSTAB object
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPBICGSTABDestroy( void *pbicgstab_vdata )
{
   fsls_CSRPBICGSTABData  *pbicgstab_data = pbicgstab_vdata;
 
   if (pbicgstab_data)
   {
      if ( (pbicgstab_data -> norms) != NULL )
      {
         fsls_TFree(pbicgstab_data -> norms);
         (pbicgstab_data -> norms) = NULL;
      }
      if ( (pbicgstab_data -> r) != NULL )
      {
         fsls_SeqVectorDestroy(pbicgstab_data -> r);
         (pbicgstab_data -> r) = NULL;
      }
      if ( (pbicgstab_data -> r0) != NULL )
      {
         fsls_SeqVectorDestroy(pbicgstab_data -> r0);
         (pbicgstab_data -> r0) = NULL;
      }
      if ( (pbicgstab_data -> s) != NULL )
      {
         fsls_SeqVectorDestroy(pbicgstab_data -> s);
         (pbicgstab_data -> s) = NULL;
      }   
      if ( (pbicgstab_data -> v) != NULL )
      {
         fsls_SeqVectorDestroy(pbicgstab_data -> v);
         (pbicgstab_data -> v) = NULL;
      }
      if ( (pbicgstab_data -> p) != NULL )
      {
         fsls_SeqVectorDestroy(pbicgstab_data -> p);
         (pbicgstab_data -> p) = NULL;
      }
      if ( (pbicgstab_data -> q) != NULL )
      {
         fsls_SeqVectorDestroy(pbicgstab_data -> q);
         (pbicgstab_data -> q) = NULL;
      }

      fsls_TFree(pbicgstab_data -> functions);
      fsls_TFree(pbicgstab_data);
   }
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGKrylovIdentitySetup
 * \brief Identity Setup for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGKrylovIdentity
 * \brief Identity Preconditioning for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGKrylovIdentity( void *vdata, void *b, void *x )
{
   return( fsls_SeqVectorCopy( b, x ) );
}

/*!
 * \fn fsls_CSRPCGFunctions *fsls_CSRPCGFunctionsCreate
 * \brief Function Creating for PCG
 * \author peghoty
 * \date 2010/08/26
 */
fsls_CSRPCGFunctions *
fsls_CSRPCGFunctionsCreate( JX_Int  (*PrecondSetup)  ( void *vdata, void *A ),
                            JX_Int  (*Precond)       ( void *vdata, void *b, void *x )  )
{
   fsls_CSRPCGFunctions *pcg_functions = NULL;
   pcg_functions = (fsls_CSRPCGFunctions *)calloc( 1, sizeof(fsls_CSRPCGFunctions));

   /* default preconditioner must be set here but can be changed later... */
   pcg_functions -> precond_setup = PrecondSetup;
   pcg_functions -> precond       = Precond;

   return pcg_functions;
}


/*!
 * \fn fsls_CSRPCGData *fsls_CSRPCGCreate 
 * \brief Create a PCG-Data Object
 * \author peghoty
 * \date 2010/08/26
 */
fsls_CSRPCGData *fsls_CSRPCGCreate()
{
   fsls_CSRPCGData      *pcg_data      = NULL;
   fsls_CSRPCGFunctions *pcg_functions = NULL;
   
   pcg_functions = fsls_CSRPCGFunctionsCreate( fsls_CSRPCGKrylovIdentitySetup, 
                                               fsls_CSRPCGKrylovIdentity );
   
   pcg_data = fsls_CTAlloc(fsls_CSRPCGData, 1);

   (pcg_data -> functions) = pcg_functions;

   /* set defaults */
   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> atolf)        = 0.0;
   (pcg_data -> cf_tol)       = 0.0;
   (pcg_data -> a_tol)        = 0.0;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> recompute_residual) = 0;
   (pcg_data -> stop_crit)    = 0;   
   (pcg_data -> converged)    = 0;
   (pcg_data -> precond_data) = NULL;
   (pcg_data -> print_level)  = 0;
   (pcg_data -> norms)        = NULL;
   (pcg_data -> rel_norms)    = NULL;   
   (pcg_data -> p)            = NULL;
   (pcg_data -> s)            = NULL;
   (pcg_data -> r)            = NULL;

   return pcg_data;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetTol
 * \brief Set parameter 'tol' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetTol( void *pcg_vdata, JX_Real tol )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> tol) = tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetAbsoluteTol
 * \brief Set parameter 'a_tol' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetAbsoluteTol( void *pcg_vdata, JX_Real a_tol )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> a_tol) = a_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetAbsoluteTolFactor
 * \brief Set parameter 'atolf' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetAbsoluteTolFactor( void *pcg_vdata, JX_Real atolf )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> atolf) = atolf;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetConvergenceFactorTol
 * \brief Set parameter 'cf_tol' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetConvergenceFactorTol( void *pcg_vdata, JX_Real cf_tol )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> cf_tol) = cf_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetMaxIter
 * \brief Set parameter 'max_iter' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetMaxIter( void *pcg_vdata, JX_Int max_iter )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> max_iter) = max_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetTwoNorm
 * \brief Set parameter 'two_norm' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetTwoNorm( void *pcg_vdata, JX_Int two_norm )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> two_norm) = two_norm;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetRelChange
 * \brief Set parameter 'rel_change' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetRelChange( void *pcg_vdata, JX_Int rel_change )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> rel_change) = rel_change;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetRecomputeResidual
 * \brief Set parameter 'recompute_residual' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetRecomputeResidual( void *pcg_vdata, JX_Int recompute_residual )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> recompute_residual) = recompute_residual;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetStopCrit
 * \brief Set parameter 'stop_crit' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetStopCrit( void *pcg_vdata, JX_Int stop_crit )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> stop_crit) = stop_crit;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetConverged
 * \brief Set parameter 'converged' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetConverged( void *pcg_vdata, JX_Int converged )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   (pcg_data -> converged) = converged;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetPrintLevel
 * \brief Set parameter 'print_level' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetPrintLevel( void *pcg_vdata, JX_Int print_level )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   
   (pcg_data -> print_level) = print_level;
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGGetConverged
 * \brief Get parameter 'converged' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGGetConverged( void *pcg_vdata, JX_Int *converged )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   *converged = (pcg_data -> converged);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGGetNumIterations
 * \brief Get parameter 'num_iterations' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGGetNumIterations( void *pcg_vdata, JX_Int *num_iterations )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   *num_iterations = (pcg_data -> num_iterations);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGGetFinalRelativeResidualNorm
 * \brief Get parameter 'relative_residual_norm' for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGGetFinalRelativeResidualNorm( void *pcg_vdata, JX_Real *relative_residual_norm )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;
   JX_Real rel_residual_norm = (pcg_data -> rel_residual_norm);
  *relative_residual_norm = rel_residual_norm;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetPrecond
 * \brief Set preconditioner for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetPrecond( void *pcg_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data )
{
   fsls_CSRPCGData      *pcg_data      = pcg_vdata;
   fsls_CSRPCGFunctions *pcg_functions = (pcg_data -> functions);

   (pcg_functions -> precond)       = precond;
   (pcg_functions -> precond_setup) = precond_setup;
   (pcg_data -> precond_data)       = precond_data;
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPCGSetup
 * \brief Setup phase for PCG
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGSetup( void *pcg_vdata, fsls_CSRMatrix *A, fsls_Vector *b, fsls_Vector *x )
{
   fsls_CSRPCGData      *pcg_data           = pcg_vdata;
   fsls_CSRPCGFunctions *pcg_functions      = (pcg_data -> functions);
   JX_Int                   max_iter           = (pcg_data -> max_iter);
   JX_Int                   (*precond_setup)() = (pcg_functions -> precond_setup);
   void                 *precond_data       = (pcg_data -> precond_data);
   
   fsls_Vector *p = NULL;
   fsls_Vector *s = NULL;
   fsls_Vector *r = NULL;
 
   JX_Real  *norms     = NULL;
   JX_Real  *rel_norms = NULL;      
   
   /* I change 'size = fsls_VectorSize(b)' into 'size = fsls_CSRMatrixNumRows(A)'
      since 'b' maybe NULL sometimes. peghoty, 2011/05/17 */
   JX_Int size        = fsls_CSRMatrixNumRows(A);
   JX_Int print_level = pcg_data -> print_level;

   if ((pcg_data -> p) == NULL)
   {
      p = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(p);
      (pcg_data -> p) = p;
   }
   if ((pcg_data -> s) == NULL)
   {
      s = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(s);
      (pcg_data -> s) = s;
   }
   if ((pcg_data -> r) == NULL)
   {
      r = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(r);
      (pcg_data -> r) = r;
   }
   
   /* preconditioning Setup */
   precond_setup(precond_data, A, b, x);

  /*-----------------------------------------------------
   * Allocate space for log info
   *----------------------------------------------------*/
   if ( print_level > 0 ) 
   {
      if ((pcg_data -> norms) == NULL)
      {
         norms = fsls_CTAlloc(JX_Real, max_iter+1);
         (pcg_data -> norms) = norms;
      }
      
      if ((pcg_data -> rel_norms) == NULL)
      {
         rel_norms = fsls_CTAlloc(JX_Real, max_iter+1); 
         (pcg_data -> rel_norms) = rel_norms;
      }     
   }

   return 0;
}


/*!
 * \fn fsls_CSRPCGSolve
 * \brief Solve "Ax=b" using PCG.
 * \param *pcg_vdata pointer to the PCGDATA
 * \param *A pointer to the coefficient matrix(fsls_CSRMatrix)
 * \param *b pointer to the right hand side vector(fsls_Vector)
 * \param *x pointer to first the initial guess and then the solution vector(fsls_Vector) 
 *
 * \note We use the following convergence test as the default 
 *      (see Ashby, Holst, Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 *     where we let (for the time being) kappa_A(CA) = 1.
 *     We implement the test as:
 *
 *     gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps
 *
 * \date 2010/08/22
 * \author peghoty
 *
 */
JX_Int
fsls_CSRPCGSolve( void            *pcg_vdata,
                  fsls_CSRMatrix  *A,
                  fsls_Vector     *b,
                  fsls_Vector     *x   )
{
   fsls_CSRPCGData       *pcg_data      = pcg_vdata;
   fsls_CSRPCGFunctions  *pcg_functions = (pcg_data -> functions);
   JX_Int                  (*precond)()    = (pcg_functions -> precond);
   void                  *precond_data  = (pcg_data -> precond_data); 
        
   JX_Real r_tol               = (pcg_data -> tol);                // default: 1.0e-06
   JX_Real atolf               = (pcg_data -> atolf);              // default: 0.0
   JX_Real cf_tol              = (pcg_data -> cf_tol);             // default: 0.0
   JX_Real a_tol               = (pcg_data -> a_tol);              // default: 0.0
   JX_Int    max_iter            = (pcg_data -> max_iter);           // default: 1000
   JX_Int    two_norm            = (pcg_data -> two_norm);           // default: 0
   JX_Int    rel_change          = (pcg_data -> rel_change);         // default: 0
   JX_Int    recompute_residual  = (pcg_data -> recompute_residual); // default: 0
   JX_Int    stop_crit           = (pcg_data -> stop_crit);          // default: 0   
   JX_Int    print_level         = (pcg_data -> print_level);        // default: 0 
   
   void  *p = (pcg_data -> p);
   void  *s = (pcg_data -> s);
   void  *r = (pcg_data -> r);

   JX_Real *norms     = (pcg_data -> norms);
   JX_Real *rel_norms = (pcg_data -> rel_norms);
                
   JX_Real  alpha, beta;
   JX_Real  gamma, gamma_old;
   JX_Real  eps;
   JX_Real  bi_prod, pi_prod, xi_prod;
   JX_Real  ieee_check = 0.0;
   JX_Real  i_prod     = 0.0;                
   JX_Real  i_prod_0   = 0.0;
   JX_Real  cf_ave_0   = 0.0;
   JX_Real  cf_ave_1   = 0.0;
   JX_Real  weight;
   JX_Real  ratio;

   JX_Real guard_zero_residual,sdotp;
   JX_Int    tentatively_converged = 0; 

   JX_Int    iter = 0;
   
   fsls_CSRPCGSetConverged(pcg_data, 0);   

  /*-----------------------------------------------------------------------
   * With relative change convergence test on, it is possible to attempt
   * another iteration with a zero residual. This causes the parameter
   * alpha to go NaN. The guard_zero_residual parameter is to circumvent
   * this. Perhaps it should be set to something non-zero (but small).
   *----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

  /*-----------------------------------------------------------------------
   * Start pcg solve
   *----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = fsls_SeqVectorInnerProd(b,b);
      if (print_level > 0)
      { 
         if (bi_prod >= 0.0) jx_printf("\n >>> ||b||_2 = %e\n",sqrt(bi_prod));
      }
   }  
   else
   {
      /* bi_prod = <C*b,b> */
      fsls_SeqVectorSetConstantValues(p, 0.0);
      precond(precond_data, b, p);

      bi_prod = fsls_SeqVectorInnerProd(p, b);
      if (print_level > 0)
      {
         if (bi_prod >= 0.0) jx_printf("\n >>> ||b||_C = %e\n",sqrt(bi_prod));
      }
   }


  /*--------------------------------------------------------------------------- 
   *  Since it does not diminish performance, attempt to return an 
   *  error flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (bi_prod != 0.0) 
   {
      ieee_check = bi_prod/bi_prod; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF 
      */
      if (print_level > 0)
      {
         jx_printf("\n\nERROR detected by FSLS ...  BEGIN\n");
         jx_printf("ERROR -- fsls_CSRPCGSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by FSLS ...  END\n\n\n");
      }
      return 100;
   }

   eps = r_tol*r_tol; /* note: this may be re-assigned below */
   
   if (bi_prod > 0.0) 
   {
      if (stop_crit && !rel_change && atolf <= 0) 
      { 
         /* pure absolute tolerance */
         eps = eps / bi_prod;
         /* Note: this section is obsolete.  Aside from backwards comatability
            concerns, we could delete the stop_crit parameter and related code,
            using tol & atolf instead. */
      }
      else if (atolf > 0)  
      {
         /* mixed relative and absolute tolerance */
         bi_prod += atolf;
      }
      else 
      {
        /* DEFAULT (stop_crit and atolf exist for backwards compatibilty
           and are not in the reference manual) */
        /* convergence criteria:  <C*r,r>  <= max( a_tol^2, r_tol^2 * <C*b,b> )
           note: default for a_tol is 0.0, so relative residual criteria is 
           used unless user specifies a_tol, or sets r_tol = 0.0, which means 
           absolute tol only is checked  */
         eps = fsls_max(r_tol*r_tol, a_tol*a_tol/bi_prod);
      }
   }
   else  /* bi_prod == 0.0: the rhs vector b is zero */
   {
      /* Set x equal to zero and return */
      fsls_SeqVectorCopy(b, x);
      if (print_level > 0)
      {
         norms[0]     = 0.0;
         rel_norms[0] = 0.0;
      }
      return 200;
      /* NOte: In this case, for the original parcsr pcg, the code would take special
         action to force iterations even though the exact value was known. */
   }

   /* r = b - Ax */
   fsls_SeqVectorCopy(b, r);
   fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
  
   /* p = C*r */
   fsls_SeqVectorSetConstantValues(p, 0.0);
   precond(precond_data, r, p);
         
   /* gamma = <r,p> */
   gamma = fsls_SeqVectorInnerProd(r,p);

  /*----------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an error
   * flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (gamma != 0.0) 
   {
      ieee_check = gamma / gamma; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF 
      */
      if (print_level > 0)
      {
          jx_printf("\n\nERROR detected by FSLS ...  BEGIN\n");
          jx_printf("ERROR -- fsls_PCGSolve: INFs and/or NaNs detected in input.\n");
          jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
          jx_printf("Returning error flag += 101.  Program not terminated.\n");
          jx_printf("ERROR detected by FSLS ...  END\n\n\n");
      }
      return 100;
   }

   /* Set initial residual norm */
   if (print_level > 0 || cf_tol > 0.0)
   {
      if (two_norm)
      {
         i_prod_0 = fsls_SeqVectorInnerProd(r,r);
      }
      else
      {
         i_prod_0 = gamma;
      }

      if (print_level > 0) 
      {  
         norms[0] = sqrt(i_prod_0);
      }
   }
   if (print_level > 0)
   {
      jx_printf("\n");
      if (two_norm)
      {
         if (stop_crit && !rel_change && atolf == 0) 
         {  
            /* pure absolute tolerance */
            jx_printf(" ======================================\n");
            jx_printf("   iter     ||r||_2     conv.factor    \n");
            jx_printf(" ======================================\n");
         }
         else 
         {
            jx_printf(" ======================================================\n");
            jx_printf("   iter    ||r||_2      conv.factor   ||r||_2/||b||_2 \n");
            jx_printf(" ======================================================\n");
         }
      }
      else  /* !two_norm */
      {
         jx_printf(" ======================================================\n");
         jx_printf("   iter    ||r||_C      conv.factor   ||r||_C/||b||_C \n");
         jx_printf(" ======================================================\n");
      }
   }

   while ( (iter + 1) <= max_iter )
   {
     /*-----------------------------------
      * the core CG calculations...
      *----------------------------------*/
      iter ++;

      /* s = A*p */
      fsls_CSRMatrixMatvec01(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      sdotp = fsls_SeqVectorInnerProd(s, p);
      if (sdotp == 0.0)
      {
         if (iter == 1) i_prod = i_prod_0;
         break;
      }
   
      alpha = gamma / sdotp;

      gamma_old = gamma;

      /* x = x + alpha*p */
      fsls_SeqVectorAxpy(alpha, p, x);

      /* r = r - alpha*s */
      fsls_SeqVectorAxpy(-alpha, s, r);
         
      /* s = C*r */
      fsls_SeqVectorSetConstantValues(s,0.0);
      precond(precond_data, r, s);
   
      /* gamma = <r,s> */
      gamma = fsls_SeqVectorInnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
      {
         i_prod = fsls_SeqVectorInnerProd(r,r);
      }
      else
      {
         i_prod = gamma;
      }

     /*--------------------------------------------------------------------
      * optional output
      *-------------------------------------------------------------------*/ 
      /* print norm info */
      if (print_level > 0)
      {
         norms[iter]     = sqrt(i_prod);
         rel_norms[iter] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }
      if (print_level > 0)
      {
         if (two_norm)
         {
            if (stop_crit && !rel_change && atolf == 0) 
            {  
               /* pure absolute tolerance */
               jx_printf("% 5d    %e    %f\n", iter, norms[iter], 
                                            norms[iter]/norms[iter-1] );
            }
            else 
            {
               jx_printf("% 5d    %e    %f      %e\n", iter, norms[iter], 
                                                  norms[iter]/norms[iter-1], rel_norms[iter] );
            }
         }
         else 
         {
            jx_printf("% 5d    %e    %f      %e\n", iter, norms[iter], 
                                                 norms[iter]/norms[iter-1], rel_norms[iter] );
         }
      }


     /*----------------------------------------
      * check for convergence
      *--------------------------------------*/
       
      if (i_prod / bi_prod < eps)  
      {
         /* the basic convergence test */
         tentatively_converged = 1;
      }
      if (tentatively_converged && recompute_residual)
      {
        /*------------------------------------------------------------------------------
         * At user request, don't trust the convergence test until we've recomputed
         * the residual from scratch.  This is expensive in the usual case where an
         * the norm is the energy norm.
         * This calculation is coded on the assumption that r's accuracy is only a
         * concern for problems where CG takes many iterations. 
         *-----------------------------------------------------------------------------*/
         /* r = b - Ax */
         fsls_SeqVectorCopy(b, r);
         fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);

         /* set i_prod for convergence test */
         if (two_norm)
         {
            i_prod = fsls_SeqVectorInnerProd(r,r);
         }
         else
         {
            /* s = C*r */
            fsls_SeqVectorSetConstantValues(s, 0.0);
            precond(precond_data, r, s);
            
            /* iprod = gamma = <r,s> */
            i_prod = fsls_SeqVectorInnerProd(r,s);
         }
         
         if (i_prod / bi_prod >= eps) 
         {
            tentatively_converged = 0;
         }
      }
      if (tentatively_converged && rel_change && (i_prod > guard_zero_residual))
      {
         /*--------------------------------------------------------------
          *  At user request, don't treat this as converged unless 
          *  x didn't change much in the last iteration. 
          *------------------------------------------------------------*/
	  pi_prod = fsls_SeqVectorInnerProd(p,p); 
 	  xi_prod = fsls_SeqVectorInnerProd(x,x);
          ratio = alpha*alpha*pi_prod / xi_prod;
          if (ratio >= eps) tentatively_converged = 0;
      }
      if ( tentatively_converged )
      {
         /* we've passed all the convergence tests, it's for real */
         fsls_CSRPCGSetConverged(pcg_data, 1);  
         break;
      }

      if ((gamma < 1.0e-292) && ((-gamma) < 1.0e-292)) 
      {
         jx_printf("\n >>> Warning: Convergence Error!!\n\n");
         break;
      }
      
     /*--------------------------------------------------------------------------------
      *  ... gamma should be >=0.  IEEE subnormal numbers are < 2**(-1022)=2.2e-308
      *  (and >= 2**(-1074)=4.9e-324).  So a gamma this small means we're getting
      *  dangerously close to subnormal or zero numbers (usually if gamma is small,
      *  so will be other variables).  Thus further calculations risk a crash.
      *  Such small gamma generally means no hope of progress anyway. 
      *-------------------------------------------------------------------------------*/

     /*----------------------------------------------------------------
      * Optional test to see if adequate progress is being made.
      * The average convergence factor is recorded and compared
      * against the tolerance 'cf_tol'. The weighting factor is  
      * intended to pay more attention to the test when an accurate
      * estimate for average convergence factor is available.  
      *---------------------------------------------------------------*/

      if (cf_tol > 0.0)
      {
         cf_ave_0 = cf_ave_1;
         if (i_prod_0 < 1.0e-292) 
         {
           /*-----------------------------------------------------------------------
            * i_prod_0 is zero, or (almost) subnormal, yet i_prod wasn't small
            * enough to pass the convergence test.  Therefore initial guess was 
            * good, and we're just calculating garbage - time to bail out before 
            * the next step, which will be a divide by zero (or close to it). 
            *----------------------------------------------------------------------*/
            jx_printf("\n >>> Warning: Convergence Error!!\n\n");
            break;
         }
	 cf_ave_1 = pow( i_prod / i_prod_0, 1.0/(2.0*iter)); 
         weight   = fabs(cf_ave_1 - cf_ave_0);
         weight   = weight / fsls_max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;

         if ( weight*cf_ave_1 > cf_tol ) break;
      }

     /*---------------------------------------------
      * back to the core CG calculations
      *-------------------------------------------*/

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta*p */
      fsls_SeqVectorScale(beta, p);   
      fsls_SeqVectorAxpy(1.0, s, p);
   }
   (pcg_data -> num_iterations) = iter;
  
  /*--------------------------------------------------------------------
   * Finish up with some outputs.
   *-------------------------------------------------------------------*/
   
   if (print_level > 0) jx_printf("\n");

   if (bi_prod > 0.0)
   {
      (pcg_data -> rel_residual_norm) = sqrt(i_prod/bi_prod);
   }
   else
   {
      (pcg_data -> rel_residual_norm) = 0.0; /* actually, we'll never get here... */
   }

   return(0);                         
}


/*!
 * \fn JX_Int fsls_CSRPCGDestroy
 * \brief Destroy a PCG object
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPCGDestroy( void *pcg_vdata )
{
   fsls_CSRPCGData *pcg_data = pcg_vdata;

   if (pcg_data)
   {
      if ( (pcg_data -> norms) != NULL )
      {
         fsls_TFree(pcg_data -> norms);
         (pcg_data -> norms) = NULL;
      } 
      if ( (pcg_data -> rel_norms) != NULL )
      {
         fsls_TFree(pcg_data -> rel_norms);
         (pcg_data -> rel_norms) = NULL;
      }
      if ( (pcg_data -> p) != NULL )
      {
         fsls_SeqVectorDestroy(pcg_data -> p);
         (pcg_data -> p) = NULL;
      }
      if ( (pcg_data -> s) != NULL )
      {
         fsls_SeqVectorDestroy(pcg_data -> s);
         (pcg_data -> s) = NULL;
      }
      if ( (pcg_data -> r) != NULL )
      {
         fsls_SeqVectorDestroy(pcg_data -> r);
         (pcg_data -> r) = NULL;
      }

      fsls_TFree(pcg_data -> functions);
      fsls_TFree(pcg_data);
   }

   return 0;
}

/*!
 * \fn fsls_CSRPCGSolveLanczos
 * \brief Solve "Ax=b" using PCG, and form a CSR matrix Q for Eignvalue computing.
 * \param *pcg_vdata pointer to the PCGDATA
 * \param *A pointer to the coefficient matrix(fsls_CSRMatrix)
 * \param *b pointer to the right hand side vector(fsls_Vector)
 * \param *x pointer to first the initial guess and then the solution vector(fsls_Vector) 
 * \param **Q_ptr JX_Real pointer to the resulting CSR matrix for Eignvalue computing.
 * \note We use the following convergence test as the default 
 *      (see Ashby, Holst, Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 *     where we let (for the time being) kappa_A(CA) = 1.
 *     We implement the test as:
 *
 *     gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps
 *
 * \date 2011/01/04
 * \author peghoty
 */
JX_Int
fsls_CSRPCGSolveLanczos( void             *pcg_vdata,
                         fsls_CSRMatrix   *A,
                         fsls_Vector      *b,
                         fsls_Vector      *x,
                         fsls_CSRMatrix  **Q_ptr   )
{
   fsls_CSRPCGData       *pcg_data      = pcg_vdata;
   fsls_CSRPCGFunctions  *pcg_functions = (pcg_data -> functions);
   JX_Int                  (*precond)()    = (pcg_functions -> precond);
   void                  *precond_data  = (pcg_data -> precond_data); 
        
   JX_Real r_tol       = (pcg_data -> tol);         // default: 1.0e-06
   JX_Int    max_iter    = (pcg_data -> max_iter);    // default: 1000
   JX_Int    two_norm    = (pcg_data -> two_norm);    // default: 0 
   JX_Int    print_level = (pcg_data -> print_level); // default: 0 
   
   void  *p = (pcg_data -> p);
   void  *s = (pcg_data -> s);
   void  *r = (pcg_data -> r);

   JX_Real *norms     = (pcg_data -> norms);
   JX_Real *rel_norms = (pcg_data -> rel_norms);
                
   JX_Real  alpha, beta;
   JX_Real  gamma, gamma_old;
   JX_Real  eps;
   JX_Real  bi_prod;
   JX_Real  i_prod     = 0.0;                
   JX_Real  i_prod_0   = 0.0;

   JX_Real sdotp;

   JX_Int    iter = 0;
   
   fsls_CSRMatrix  *Q = NULL;
   JX_Int    *iq = NULL;
   JX_Int    *jq = NULL;
   JX_Real *q  = NULL;
   JX_Int     i, cnt;
   
   JX_Real *work   = NULL;
   JX_Real *qalpha = NULL;
   JX_Real *qbeta  = NULL;
   
   work  = fsls_CTAlloc(JX_Real, 2*max_iter);
   qalpha = work;
   qbeta  = work + max_iter;
   
   fsls_CSRPCGSetConverged(pcg_data, 0);   

   if (two_norm)
   {
      bi_prod = fsls_SeqVectorInnerProd(b,b);
      if (print_level > 0)
      { 
         if (bi_prod >= 0.0) jx_printf("\n >>> ||b||_2 = %e\n",sqrt(bi_prod));
      }
   }  
   else
   {
      fsls_SeqVectorSetConstantValues(p, 0.0);
      precond(precond_data, b, p);

      bi_prod = fsls_SeqVectorInnerProd(p, b);
      if (print_level > 0)
      {
         if (bi_prod >= 0.0) jx_printf("\n >>> ||b||_C = %e\n",sqrt(bi_prod));
      }
   }

   eps = r_tol*r_tol;
   
   if (bi_prod == 0.0)
   {
      fsls_SeqVectorCopy(b, x);
      if (print_level > 0)
      {
         norms[0]     = 0.0;
         rel_norms[0] = 0.0;
      }
      return 200;
   }

   /* r = b - Ax */
   fsls_SeqVectorCopy(b, r);
   fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
  
   /* p = C*r */
   fsls_SeqVectorSetConstantValues(p, 0.0);
   precond(precond_data, r, p);
         
   /* gamma = <r,p> */
   gamma = fsls_SeqVectorInnerProd(r,p);

   /* Set initial residual norm */
   if (print_level > 0)
   {
      if (two_norm)
      {
         i_prod_0 = fsls_SeqVectorInnerProd(r,r);
      }
      else
      {
         i_prod_0 = gamma;
      }

      if (print_level > 0) 
      {  
         norms[0] = sqrt(i_prod_0);
      }
   }
   if (print_level > 0)
   {
      jx_printf("\n");
      if (two_norm)
      {
         jx_printf(" ======================================================\n");
         jx_printf("   iter    ||r||_2      conv.factor   ||r||_2/||b||_2 \n");
         jx_printf(" ======================================================\n");
      }
      else  /* !two_norm */
      {
         jx_printf(" ======================================================\n");
         jx_printf("   iter    ||r||_C      conv.factor   ||r||_C/||b||_C \n");
         jx_printf(" ======================================================\n");
      }
   }

   while ( (iter + 1) <= max_iter )
   {
     /*-----------------------------------
      * the core CG calculations...
      *----------------------------------*/
      iter ++;

      /* s = A*p */
      fsls_CSRMatrixMatvec01(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      sdotp = fsls_SeqVectorInnerProd(s, p);
      if (sdotp == 0.0)
      {
         if (iter == 1) i_prod = i_prod_0;
         break;
      }
      
      alpha = gamma / sdotp;
      qalpha[iter-1] = alpha;

      gamma_old = gamma;

      /* x = x + alpha*p */
      fsls_SeqVectorAxpy(alpha, p, x);

      /* r = r - alpha*s */
      fsls_SeqVectorAxpy(-alpha, s, r);
         
      /* s = C*r */
      fsls_SeqVectorSetConstantValues(s,0.0);
      precond(precond_data, r, s);
   
      /* gamma = <r,s> */
      gamma = fsls_SeqVectorInnerProd(r, s);

      /* set i_prod for convergence test */
      if (two_norm)
      {
         i_prod = fsls_SeqVectorInnerProd(r,r);
      }
      else
      {
         i_prod = gamma;
      }

     /*--------------------------------------------------------------------
      * optional output
      *-------------------------------------------------------------------*/ 
      /* print norm info */
      if (print_level > 0)
      {
         norms[iter]     = sqrt(i_prod);
         rel_norms[iter] = bi_prod ? sqrt(i_prod/bi_prod) : 0;
      }
      if (print_level > 0)
      {
         if (two_norm)
         {

            jx_printf("% 5d    %e    %f      %e\n", iter, norms[iter], 
                   norms[iter]/norms[iter-1], rel_norms[iter] );
         }
         else 
         {
            jx_printf("% 5d    %e    %f      %e\n", iter, norms[iter], 
                   norms[iter]/norms[iter-1], rel_norms[iter] );
         }
      }


      if ((gamma < 1.0e-292) && ((-gamma) < 1.0e-292)) 
      {
         jx_printf("\n >>> Warning: Convergence Error!!\n\n");
         break;
      }

     /*---------------------------------------------
      * back to the core CG calculations
      *-------------------------------------------*/

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;
      qbeta[iter-1] = beta;
      
     /*----------------------------------------
      * check for convergence
      *--------------------------------------*/
       
      if (i_prod / bi_prod < eps)  
      {
         fsls_CSRPCGSetConverged(pcg_data, 1);  
         break;
      }      

      /* p = s + beta*p */
      fsls_SeqVectorScale(beta, p);   
      fsls_SeqVectorAxpy(1.0, s, p);
   }
   
   (pcg_data -> num_iterations) = iter;
      
   //-----------------------------------------------------------
   // Construct the CSR matrix Q
   //-----------------------------------------------------------
   
   Q = fsls_CSRMatrixCreate(iter, iter, 3*iter-2);
   fsls_CSRMatrixInitialize(Q);
   q  = fsls_CSRMatrixData(Q);
   iq = fsls_CSRMatrixI(Q);
   jq = fsls_CSRMatrixJ(Q);
   
   /* Generate 'iq' */
   iq[0] = 0;
   iq[1] = 2;
   for (i = 2; i < iter; i ++)
   {
      iq[i] = iq[i-1] + 3;
   } 
   iq[iter] = iq[iter-1] + 2; 

   /* Generate 'jq' and 'q' at the same time */
#if 1  // my formula  
   jq[0] = 0;
    q[0] = (1.0 + qbeta[0]) / qalpha[0];
   jq[1] = 1;
    q[1] = - qbeta[0] / qalpha[1];
   cnt = 2;
   for (i = 1; i < iter-1; i ++)
   {
      jq[cnt] = i;
       q[cnt] = (1.0 + qbeta[i]) / qalpha[i];
      cnt ++;
      jq[cnt] = i - 1;
       q[cnt] = - 1.0 / qalpha[i-1];
      cnt ++;
      jq[cnt] = i + 1;
       q[cnt] = - qbeta[i] / qalpha[i+1];
      cnt ++;
   }
   jq[cnt] = i;
    q[cnt] = (1.0 + qbeta[i]) / qalpha[i];
   cnt ++;
   jq[cnt] = i - 1;
    q[cnt] = - 1.0 / qalpha[0];
#else // Wang Junxian's formula
   jq[0] = 0;
    q[0] = 1.0 / qalpha[0];
   jq[1] = 1;
    q[1] = sqrt(qbeta[0]) / qalpha[0];
   cnt = 2;
   for (i = 1; i < iter-1; i ++)
   {
      jq[cnt] = i;
       q[cnt] = 1.0 / qalpha[i] + qbeta[i-1] / qalpha[i-1];
      cnt ++;
      jq[cnt] = i - 1;
       q[cnt] = sqrt(qbeta[i-1]) / qalpha[i-1];
      cnt ++;
      jq[cnt] = i + 1;
       q[cnt] = sqrt(qbeta[i]) / qalpha[i];
      cnt ++;
   }
   jq[cnt] = i;
    q[cnt] = 1.0 / qalpha[i] + qbeta[i-1] / qalpha[i-1];
   cnt ++;
   jq[cnt] = i - 1;
    q[cnt] = sqrt(qbeta[i-1]) / qalpha[i-1];
#endif
   
   *Q_ptr = Q;
   
  /*--------------------------------------------------------------------
   * Finish up with some outputs.
   *-------------------------------------------------------------------*/
   
   if (print_level > 0) jx_printf("\n");

   if (bi_prod > 0.0)
   {
      (pcg_data -> rel_residual_norm) = sqrt(i_prod/bi_prod);
   }
   else
   {
      (pcg_data -> rel_residual_norm) = 0.0;
   }

   return(0);                         
}

/*!
 * \fn JX_Int fsls_CSRPGMRESKrylovIdentitySetup
 * \brief Identity Setup for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESKrylovIdentity
 * \brief Identity Preconditioning for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESKrylovIdentity( void *vdata, void *b, void *x )
{
   return( fsls_SeqVectorCopy( b, x ) );
}

/*!
 * \fn fsls_CSRPGMRESFunctions *fsls_CSRPGMRESFunctionsCreate
 * \brief Function Creating for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
fsls_CSRPGMRESFunctions *
fsls_CSRPGMRESFunctionsCreate( JX_Int  (*PrecondSetup)  ( void *vdata, void *A ),
                               JX_Int  (*Precond)       ( void *vdata, void *b, void *x )  )
{
   fsls_CSRPGMRESFunctions *pgmres_functions = NULL;
   pgmres_functions = (fsls_CSRPGMRESFunctions *)calloc( 1, sizeof(fsls_CSRPGMRESFunctions));

   /* default preconditioner must be set here but can be changed later... */
   pgmres_functions -> precond_setup = PrecondSetup;
   pgmres_functions -> precond       = Precond;

   return pgmres_functions;
}

/*!
 * \fn fsls_CSRPGMRESData *fsls_CSRPGMRESCreate 
 * \brief Create a PGMRES-Data Object
 * \author peghoty
 * \date 2010/08/26
 */
fsls_CSRPGMRESData *fsls_CSRPGMRESCreate()
{
   fsls_CSRPGMRESData      *pgmres_data      = NULL;
   fsls_CSRPGMRESFunctions *pgmres_functions = NULL;
   
   pgmres_functions = fsls_CSRPGMRESFunctionsCreate( fsls_CSRPGMRESKrylovIdentitySetup,
                                                     fsls_CSRPGMRESKrylovIdentity );   
  
   pgmres_data = fsls_CTAlloc(fsls_CSRPGMRESData, 1);

   (pgmres_data -> functions) = pgmres_functions;
 
   /* set defaults */
   (pgmres_data -> k_dim)          = 5;
   (pgmres_data -> tol)            = 1.0e-06; // relative residual tol
   (pgmres_data -> cf_tol)         = 0.0;
   (pgmres_data -> a_tol)          = 0.0;     // abs. residual tol
   (pgmres_data -> min_iter)       = 0;
   (pgmres_data -> max_iter)       = 1000;
   (pgmres_data -> rel_change)     = 0;
   (pgmres_data -> stop_crit)      = 0;       // rel. residual norm - this is obsolete!
   (pgmres_data -> converged)      = 0;
   (pgmres_data -> precond_data)   = NULL;
   (pgmres_data -> print_level)    = 0;
   (pgmres_data -> p)              = NULL;
   (pgmres_data -> r)              = NULL;
   (pgmres_data -> w)              = NULL;
   (pgmres_data -> w_2)            = NULL;
   (pgmres_data -> norms)          = NULL;

   return (void *) pgmres_data;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetKDim
 * \brief Set parameter 'k_dim' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetKDim( void *pgmres_vdata, JX_Int k_dim )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> k_dim) = k_dim;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetTol
 * \brief Set parameter 'tol' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetTol( void *pgmres_vdata, JX_Real tol )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> tol) = tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetAbsoluteTol
 * \brief Set parameter 'a_tol' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetAbsoluteTol( void *pgmres_vdata, JX_Real a_tol )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> a_tol) = a_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetConvergenceFactorTol
 * \brief Set parameter 'cf_tol' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetConvergenceFactorTol( void *pgmres_vdata, JX_Real cf_tol )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> cf_tol) = cf_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetMinIter
 * \brief Set parameter 'min_iter' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetMinIter( void *pgmres_vdata, JX_Int min_iter )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> min_iter) = min_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetMaxIter
 * \brief Set parameter 'max_iter' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetMaxIter( void *pgmres_vdata, JX_Int max_iter )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> max_iter) = max_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetRelChange
 * \brief Set parameter 'rel_change' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetRelChange( void *pgmres_vdata, JX_Int rel_change )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> rel_change) = rel_change;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetStopCrit
 * \brief Set parameter 'stop_crit ' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetStopCrit( void *pgmres_vdata, JX_Int stop_crit )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> stop_crit) = stop_crit;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetPrintLevel
 * \brief Set parameter 'print_level' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetPrintLevel( void *pgmres_vdata, JX_Int print_level )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   (pgmres_data -> print_level) = print_level;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESGetNumIterations
 * \brief Get parameter 'num_iterations ' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESGetNumIterations( void *pgmres_vdata, JX_Int *num_iterations )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   *num_iterations = (pgmres_data -> num_iterations);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESGetConverged
 * \brief Get parameter 'converged' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESGetConverged( void *pgmres_vdata, JX_Int  *converged )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   *converged = (pgmres_data -> converged);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESGetFinalRelativeResidualNorm
 * \brief Get parameter 'relative_residual_norm' for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */ 
JX_Int
fsls_CSRPGMRESGetFinalRelativeResidualNorm( void *pgmres_vdata, JX_Real *relative_residual_norm )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   *relative_residual_norm = (pgmres_data -> rel_residual_norm);
   return 0;
} 

/*!
 * \fn JX_Int fsls_CSRPGMRESSetPrecond
 * \brief Set preconditioner for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetPrecond( void *pgmres_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   fsls_CSRPGMRESFunctions *pgmres_functions = pgmres_data->functions;

   (pgmres_functions -> precond)       = precond;
   (pgmres_functions -> precond_setup) = precond_setup;
   (pgmres_data -> precond_data)       = precond_data;
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESSetup
 * \brief Setup phase for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESSetup( void *pgmres_vdata, fsls_CSRMatrix *A, fsls_Vector *b, fsls_Vector *x )
{
   fsls_CSRPGMRESData      *pgmres_data      = pgmres_vdata;
   fsls_CSRPGMRESFunctions *pgmres_functions = (pgmres_data -> functions);

   JX_Int     k_dim              = (pgmres_data -> k_dim);
   JX_Int     max_iter           = (pgmres_data -> max_iter);
   JX_Int     (*precond_setup)() = (pgmres_functions -> precond_setup);
   void   *precond_data       = (pgmres_data -> precond_data);
   JX_Int     rel_change         = (pgmres_data -> rel_change);

   JX_Int     i;
   /* I change 'size = fsls_VectorSize(b)' into 'size = fsls_CSRMatrixNumRows(A)'
      since 'b' maybe NULL sometimes. peghoty, 2011/06/24 */
   JX_Int     size        = fsls_CSRMatrixNumRows(A);
   JX_Int     print_level = pgmres_data -> print_level;
   JX_Real *norms       = NULL;   
   
   fsls_Vector **p   = NULL;
   fsls_Vector  *r   = NULL;
   fsls_Vector  *w   = NULL;
   fsls_Vector  *w_2 = NULL;
 
   if ((pgmres_data -> p) == NULL)
   {
      p = fsls_CTAlloc(fsls_Vector *, k_dim+1);
      for (i = 0; i < k_dim+1; i ++)
      {
         p[i] = fsls_SeqVectorCreate(size);
         fsls_SeqVectorInitialize(p[i]);
      }
      (pgmres_data -> p) = (void **)p; 
   }  
      
   if ((pgmres_data -> r) == NULL)
   {
      r = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(r);
      (pgmres_data -> r) = r;
   }
   
   if ((pgmres_data -> w) == NULL)
   {
      w = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(w);
      (pgmres_data -> w) = w;
   }
    
   if (rel_change)
   {  
      if ((pgmres_data -> w_2) == NULL)
      {
         w_2 = fsls_SeqVectorCreate(size);
         fsls_SeqVectorInitialize(w_2);
         (pgmres_data -> w_2) = w_2;
      }
   }
   
   /* preconditioning Setup */
   precond_setup(precond_data, A, b, x);

   if (print_level > 0)
   {
      if ((pgmres_data -> norms) == NULL)
      {
         norms = fsls_CTAlloc(JX_Real, max_iter+1);
         (pgmres_data -> norms) = norms;
      }
   }

   return 0;
}
 
/*!
 * \fn JX_Int fsls_CSRPGMRESSolve
 * \brief Solve phase for PGMRES
 * \author peghoty
 * \date 2010/08/26
 */
#define TEST_PRECOND_CPU 0
JX_Int
fsls_CSRPGMRESSolve( void            *pgmres_vdata,
                     fsls_CSRMatrix  *A,
                     fsls_Vector     *b,
                     fsls_Vector     *x )
{
#if TEST_PRECOND_CPU
   struct timeval tStart,tEnd;
   JX_Real precond_time = 0.0;
#endif

   fsls_CSRPGMRESData       *pgmres_data      = pgmres_vdata;
   fsls_CSRPGMRESFunctions  *pgmres_functions = (pgmres_data -> functions);
   JX_Int 	                     (*precond)()     = (pgmres_functions -> precond);
   void                      *precond_data    = (pgmres_data -> precond_data);
 
   JX_Int      k_dim        = (pgmres_data -> k_dim);       // default: 5
   JX_Int      min_iter     = (pgmres_data -> min_iter);    // default: 0
   JX_Int      max_iter     = (pgmres_data -> max_iter);    // default: 1000
   JX_Int      rel_change   = (pgmres_data -> rel_change);  // default: 0
   JX_Real   r_tol        = (pgmres_data -> tol);         // default: 1.0e-06
   JX_Real   cf_tol       = (pgmres_data -> cf_tol);      // default: 0.0
   JX_Real   a_tol        = (pgmres_data -> a_tol);       // default: 0.0 
   JX_Int      print_level  = (pgmres_data -> print_level); // default: 0
      
   void    *r            = (pgmres_data -> r);
   void    *w            = (pgmres_data -> w);
   void    *w_2          = (pgmres_data -> w_2);   // only allocated if 'rel_change > 0'
   void   **p            = (pgmres_data -> p);
   JX_Real  *norms        = (pgmres_data -> norms); 
   
   JX_Int      break_value = 0;
   JX_Int	    i,j,k;
   JX_Int      k_dim_plus_one = k_dim + 1;
   JX_Int      iter; 
   JX_Int      rel_change_passed = 0;
   JX_Int      num_rel_change_check = 0;
   JX_Real   epsilon, gamma, t;
   JX_Real   r_norm, b_norm, den_norm, x_norm, w_norm;
   JX_Real   epsmac = 1.e-16; 
   JX_Real   ieee_check = 0.;
   JX_Real   guard_zero_residual; 
   JX_Real   cf_ave_0 = 0.0;
   JX_Real   cf_ave_1 = 0.0;
   JX_Real   weight;
   JX_Real   r_norm_0;
   JX_Real   relative_error = 1.0;
   JX_Real  *work = NULL;
   JX_Real  *rs   = NULL;
   JX_Real  *c    = NULL;
   JX_Real  *s    = NULL; 
   JX_Real  *rs_2 = NULL;
   JX_Real **hh   = NULL;
   JX_Real  *hhs  = NULL;   

   (pgmres_data -> converged) = 0;

  /*-----------------------------------------------------------------------
   * With relative change convergence test on, it is possible to attempt
   * another iteration with a zero residual. This causes the parameter
   * alpha to go NaN. The guard_zero_residual parameter is to circumvent
   * this. Perhaps it should be set to something non-zero (but small).
   *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   if (print_level > 0)
   {
      norms = (pgmres_data -> norms);
   }

   /* initialize work arrays */
   work = fsls_CTAlloc(JX_Real, (k_dim + 4)*k_dim + 1);
   hh   = fsls_CTAlloc(JX_Real*, k_dim_plus_one);
   rs = work; c = rs + k_dim_plus_one; s = c + k_dim;
   hhs = s + k_dim;
   for (i = 0; i < k_dim_plus_one; i ++) hh[i] = hhs + i*k_dim;
   if (rel_change) rs_2 = fsls_CTAlloc(JX_Real, k_dim_plus_one);

   /* compute initial residual */
   fsls_SeqVectorCopy(b, p[0]);
   fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, p[0]);
  
   b_norm = sqrt( fsls_SeqVectorInnerProd(b,b) );
   
  /*---------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an 
   * error flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (b_norm != 0.0) 
   {
      ieee_check = b_norm / b_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0)
      {
         jx_printf("\n\nERROR detected by FSLS ... BEGIN\n");
         jx_printf("ERROR -- fsls_CSRPGMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by FSLS ... END\n\n\n");
      }
      return 100;
   }   

   r_norm   = sqrt( fsls_SeqVectorInnerProd(p[0],p[0]) );
   r_norm_0 = r_norm;
   
  /*---------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an 
   * error flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (r_norm != 0.0) 
   {
      ieee_check = r_norm / r_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0)
      {
        jx_printf("\n\nERROR detected by FSLS ... BEGIN\n");
        jx_printf("ERROR -- fsls_CSRPGMRESSolve: INFs and/or NaNs detected in input.\n");
        jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
        jx_printf("Returning error flag += 101.  Program not terminated.\n");
        jx_printf("ERROR detected by FSLS ... END\n\n\n");
      }
      return 100;
   }   
  
   if (print_level > 0)
   {
      norms[0] = r_norm;
      jx_printf("\n >>> L2 norm of b: %e\n", b_norm);
      if (b_norm == 0.0)
      {
         jx_printf(" >>> Rel_resid_norm actually contains the residual norm\n");
      }
      jx_printf(" >>> Initial L2 norm of residual: %e\n\n", r_norm);
   }
 
   iter = 0;

   if (b_norm > 0.0)
   {  
      den_norm = b_norm;  // convergence criterion |r_i|/|b| <= accuracy if |b| > 0
   }
   else
   { 
      den_norm = r_norm;  // convergence criterion |r_i|/|r0| <= accuracy if |b| = 0
   }

  /*---------------------------------------------------------------------------
   * convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
   * den_norm = |r_0| or |b|
   * note: default for a_tol is 0.0, so relative residual criteria is 
   *       used unless user specifies a_tol, or sets r_tol = 0.0, which 
   *       means absolute tol only is checked. 
   *--------------------------------------------------------------------------*/
   epsilon = fsls_max(a_tol,r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */
   
   if (print_level > 0)
   {
      if (b_norm > 0.0)
      {
         jx_printf(" ===================================================\n");
         jx_printf("   iter   resid.norm    conv.factor  rel.res.norm   \n");
         jx_printf(" ===================================================\n");
      }
      else
      {
         jx_printf(" =====================================\n");
         jx_printf("   iter    resid.norm   conv.factor   \n");
         jx_printf(" =====================================\n");
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
           /* free some stuff */
           fsls_TFree(work);
           fsls_TFree(hh);
           if (rel_change) fsls_TFree(rs_2); 
	   return 11;
	}

        /* see if we are already converged and 
           should print the final norm and exit */
	if (r_norm  <= epsilon && iter >= min_iter) 
        {
           if (!rel_change) /* shouldn't exit after no iterations if relative change is on */
           {
              fsls_SeqVectorCopy(b,r);
              fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
              r_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
              
              if (r_norm <= epsilon)
              {
                 if (print_level > 0)
                 {
                    jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 break;
              }
              else
              {
                 if (print_level > 0) jx_printf(" >>> false convergence (1)\n");
              }
           }
	}

      	t = 1.0 / r_norm;
	fsls_SeqVectorScale(t,p[0]);
	
	
        /*--------- RESTART CYCLE (right-preconditioning) --------*/	
	i = 0;
        while (i < k_dim && iter < max_iter)
	{
           i ++;
           iter ++;

#if TEST_PRECOND_CPU
           GetTime(tStart);
#endif            
           fsls_SeqVectorSetConstantValues(r, 0.0);
           precond(precond_data, p[i-1], r);
           
#if TEST_PRECOND_CPU
           GetTime(tEnd);
           precond_time += mytime(tStart,tEnd);
#endif  

           fsls_CSRMatrixMatvec01(1.0, A, r, 0.0, p[i]);
           
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = fsls_SeqVectorInnerProd(p[j],p[i]);
              fsls_SeqVectorAxpy(-hh[j][i-1],p[j],p[i]);
           }
           t = sqrt( fsls_SeqVectorInnerProd(p[i],p[i]) );
           hh[i][i-1] = t;	
           if (t != 0.0)
           {
              t = 1.0/t;
              fsls_SeqVectorScale(t,p[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step.
              update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t  = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1]  = hh[i-1][i-1] / gamma;
           s[i-1]  = hh[i][i-1] / gamma;
           //rs[i]   = -hh[i][i-1]*rs[i-1];
           //rs[i]  /= gamma;               /* Optimized by peghoty 2010/08/28 */
           rs[i]   = -s[i-1]*rs[i-1];
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if (print_level > 0)
           {
              norms[iter] = r_norm;

              if (b_norm > 0.0)
                 jx_printf("% 5d    %e    %f    %e\n", iter, norms[iter],
                                                    norms[iter]/norms[iter-1], norms[iter]/b_norm);
              else
                 jx_printf("% 5d    %e     %f\n", iter, norms[iter], norms[iter]/norms[iter-1]);
           }
           
           /* convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));
              
              weight = fabs(cf_ave_1 - cf_ave_0);
              weight = weight / fsls_max(cf_ave_1, cf_ave_0);
              weight = 1.0 - weight;

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
                 
                 /* extra copy of rs so we don't need to change the later solve */
                 for (k = 0; k < i; k ++) rs_2[k] = rs[k];

                 /* solve tri. system*/
                 rs_2[i-1] = rs_2[i-1] / hh[i-1][i-1];
                 for (k = i-2; k >= 0; k --)
                 {
                    t = 0.0;
                    for (j = k+1; j < i; j ++)
                    {
                       t -= hh[k][j]*rs_2[j];
                    }
                    t += rs_2[k];
                    rs_2[k] = t/hh[k][k];
                 }
                 
                 fsls_SeqVectorCopy(p[i-1],w);
                 fsls_SeqVectorScale(rs_2[i-1],w);
                 
                 for (j = i-2; j >= 0; j --)
                 {
                    fsls_SeqVectorAxpy(rs_2[j], p[j], w);
                 }

#if TEST_PRECOND_CPU
                 GetTime(tStart);
#endif     
                 fsls_SeqVectorSetConstantValues(r, 0.0);
                 precond(precond_data, w, r);
                 
#if TEST_PRECOND_CPU
                 GetTime(tEnd);
                 precond_time += mytime(tStart,tEnd);
#endif

                 /* copy current solution (x) to w (don't want to over-write x)*/
                 fsls_SeqVectorCopy(x,w);

                 /* add the correction */
                 fsls_SeqVectorAxpy(1.0,r,w);

                 /* now w is the approx solution  - get the norm*/
                 x_norm = sqrt( fsls_SeqVectorInnerProd(w,w) );

                 if ( !(x_norm <= guard_zero_residual )) /* don't divide by zero */
                 {  
                    /* now get  x_i - x_i-1 */
                    if (num_rel_change_check)
                    {
                       /* have already checked once so we can avoid another precond solve */
                       fsls_SeqVectorCopy(w, r);
                       fsls_SeqVectorAxpy(-1.0, w_2, r);
                       /* now r contains x_i - x_i-1*/

                       /* save current soln w in w_2 for next time */
                       fsls_SeqVectorCopy(w, w_2);
                    }
                    else
                    {
                       /* first time to check rel change*/

                       /* first save current soln w in w_2 for next time */
                       fsls_SeqVectorCopy(w, w_2);

                       /* for relative change take x_(i-1) to be 
                          x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                          Now
                          x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                          - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                          = M^{-1} rs_{i-1}{p_{i-1}} */
                       fsls_SeqVectorSetConstantValues(w, 0.0);
                       fsls_SeqVectorAxpy(rs_2[i-1], p[i-1], w);
                       
                       /* apply the preconditioner */
#if TEST_PRECOND_CPU
                       GetTime(tStart);
#endif 
                       fsls_SeqVectorSetConstantValues(r, 0.0);
                       precond(precond_data, w, r);
                       
#if TEST_PRECOND_CPU
                       GetTime(tEnd);
                       precond_time += mytime(tStart,tEnd);
#endif
                       /* now r contains x_i - x_i-1 */          
                    }
                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
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
           
	} 
	/*----------- End of RESTART CYCLE ---------------*/
	

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i-2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k+1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        fsls_SeqVectorCopy(p[i-1],w);
        fsls_SeqVectorScale(rs[i-1],w);
        
        for (j = i-2; j >= 0; j --)
        {
           fsls_SeqVectorAxpy(rs[j], p[j], w);
        }

        /* find correction (in r) */
#if TEST_PRECOND_CPU
        GetTime(tStart);
#endif           
	fsls_SeqVectorSetConstantValues(r, 0.0);
	precond(precond_data, w, r);
	
#if TEST_PRECOND_CPU
        GetTime(tEnd);
        precond_time += mytime(tStart,tEnd);
#endif
        /* update current solution x (in x) */
	fsls_SeqVectorAxpy(1.0,r,x);
         
        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm*/
           fsls_SeqVectorCopy(b,r);
           fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
           r_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
           
           if (r_norm <= epsilon)
           {
              if (rel_change && !rel_change_passed) /* calculate the relative change */
              {
                 /* calculate the norm of the solution */
                 x_norm = sqrt( fsls_SeqVectorInnerProd(x,x) );
               
                 if ( !(x_norm <= guard_zero_residual ))  /* don't divide by zero */
                 {
                    /* for relative change take x_(i-1) to be 
                       x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
                       Now
                       x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
                       - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
                       = M^{-1} rs_{i-1}{p_{i-1}} */
                    fsls_SeqVectorSetConstantValues(w, 0.0);
                    fsls_SeqVectorAxpy(rs[i-1], p[i-1], w);
                    
                    /* apply the preconditioner */
#if TEST_PRECOND_CPU
                    GetTime(tStart);
#endif
                    fsls_SeqVectorSetConstantValues(r, 0.0);
                    precond(precond_data, w, r);

#if TEST_PRECOND_CPU
                    GetTime(tEnd);
                    precond_time += mytime(tStart,tEnd);
#endif

                    /* find the norm of x_i - x_i-1 */          
                    w_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
                    
                    relative_error= w_norm / x_norm;
                    if ( relative_error < r_tol )
                    {
                       (pgmres_data -> converged) = 1;
                       if (print_level > 0)
                       {
                          jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
                       }
                       break;
                    }
                 }
                 else
                 {
                    (pgmres_data -> converged) = 1;
                    if (print_level > 0)
                    {
                       jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
                    }
                    break;
                 }

              }
              else /* don't need to check rel. change */
              {
                 if (print_level > 0)
                 {
                    jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
                 }
                 (pgmres_data -> converged) = 1;
                 break;
              }
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if (print_level > 0)
              {
                 jx_printf(" >>> false convergence (2)\n");
              }
              fsls_SeqVectorCopy(r,p[0]);
              i = 0;
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j]   =  c[j-1]*rs[j];
	}
        
        if (i) 
        {
           fsls_SeqVectorAxpy(rs[i]-1.0,p[i],p[i]);
        }
        for (j = i-1 ; j > 0; j --)
        {
           fsls_SeqVectorAxpy(rs[j],p[j],p[i]);
        }
        
        if (i)
        {
           fsls_SeqVectorAxpy(rs[0]-1.0,p[0],p[0]);
           fsls_SeqVectorAxpy(1.0,p[i],p[0]);
        }
   } 
   /* END of iteration while loop */
   
   (pgmres_data -> num_iterations) = iter;  // get the number of iterations

   /* get the last relative residual-norm */
   if (b_norm > 0.0)
   {
      (pgmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (pgmres_data -> rel_residual_norm) = r_norm;
   }
   
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf("\n Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
   }
  
  /*-------------------------------------------
   * Free some stuff
   *------------------------------------------*/   
   fsls_TFree(work);
   fsls_TFree(hh);
   if (rel_change) fsls_TFree(rs_2); 

#if TEST_PRECOND_CPU
   jx_printf("\n >>> \033[31mB-act time:\033[00m %.3lf seconds\n", precond_time);
#endif

   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPGMRESDestroy
 * \brief Destroy a PGMRES object
 * \author peghoty
 * \date 2010/08/26
 */
JX_Int
fsls_CSRPGMRESDestroy( void *pgmres_vdata )
{
   fsls_CSRPGMRESData *pgmres_data = pgmres_vdata;
   JX_Int i;
   JX_Int k_dim = (pgmres_data -> k_dim);
 
   if (pgmres_data)
   {
      if ( (pgmres_data -> print_level) > 0 )
      {
         if ( (pgmres_data -> norms) != NULL )
         {
            fsls_TFree(pgmres_data -> norms);
            (pgmres_data -> norms) = NULL;
         }
      }
      if ( (pgmres_data -> r) != NULL )
      {
         fsls_SeqVectorDestroy(pgmres_data -> r);
         (pgmres_data -> r) = NULL;
      }      
      if ( (pgmres_data -> w) != NULL )
      {
         fsls_SeqVectorDestroy(pgmres_data -> w);
         (pgmres_data -> w) = NULL;
      }  
      if ( (pgmres_data -> w_2) != NULL )
      {
         fsls_SeqVectorDestroy(pgmres_data -> w_2);
         (pgmres_data -> w_2) = NULL;
      }   
      if ( (pgmres_data -> p) != NULL )
      {
         for (i = 0; i < k_dim+1; i ++)
         {
            if ( (pgmres_data -> p)[i] != NULL )
            {
	       fsls_SeqVectorDestroy((pgmres_data -> p)[i]);
	       (pgmres_data -> p)[i] = NULL;
	    }
         }
         fsls_TFree(pgmres_data -> p);
         pgmres_data -> p = NULL;
      }
      
      fsls_TFree(pgmres_data -> functions);
      fsls_TFree(pgmres_data);            
   }
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESKrylovIdentitySetup
 * \brief Identity Setup for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESKrylovIdentitySetup( void *vdata, void *A )
{
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESKrylovIdentity
 * \brief Identity Preconditioning for a Krylov subspace iterative method.
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESKrylovIdentity( void *vdata, void *b, void *x )
{
   return( fsls_SeqVectorCopy( b, x ) );
}

/*!
 * \fn fsls_CSRPLGMRESFunctions *fsls_CSRPLGMRESFunctionsCreate
 * \brief Function Creating for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
fsls_CSRPLGMRESFunctions *
fsls_CSRPLGMRESFunctionsCreate( JX_Int  (*PrecondSetup)  ( void *vdata, void *A ),
                                JX_Int  (*Precond)       ( void *vdata, void *b, void *x )  )
{
   fsls_CSRPLGMRESFunctions *plgmres_functions = NULL;
   plgmres_functions = (fsls_CSRPLGMRESFunctions *)calloc( 1, sizeof(fsls_CSRPLGMRESFunctions));

   /* default preconditioner must be set here but can be changed later... */
   plgmres_functions -> precond_setup = PrecondSetup;
   plgmres_functions -> precond       = Precond;

   return plgmres_functions;
}

/*!
 * \fn fsls_CSRPLGMRESData *fsls_CSRPLGMRESCreate 
 * \brief Create a PLGMRES-Data Object
 * \author peghoty
 * \date 2010/12/11
 */
fsls_CSRPLGMRESData *fsls_CSRPLGMRESCreate()
{
   fsls_CSRPLGMRESData      *plgmres_data      = NULL;
   fsls_CSRPLGMRESFunctions *plgmres_functions = NULL;
   
   plgmres_functions = fsls_CSRPLGMRESFunctionsCreate( fsls_CSRPLGMRESKrylovIdentitySetup,
                                                       fsls_CSRPLGMRESKrylovIdentity );   
  
   plgmres_data = fsls_CTAlloc(fsls_CSRPLGMRESData, 1);

   (plgmres_data -> functions) = plgmres_functions;
 
   /* set defaults */
   (plgmres_data -> k_dim)           = 5;
   (plgmres_data -> tol)             = 1.0e-06; // relative residual tol
   (plgmres_data -> cf_tol)          = 0.0;
   (plgmres_data -> a_tol)           = 0.0;     // abs. residual tol
   (plgmres_data -> min_iter)        = 0;
   (plgmres_data -> max_iter)        = 1000;
   (plgmres_data -> stop_crit)       = 0;       // rel. residual norm - this is obsolete!
   (plgmres_data -> converged)       = 0;
   (plgmres_data -> precond_data)    = NULL;
   (plgmres_data -> print_level)     = 0;
   (plgmres_data -> p)               = NULL;
   (plgmres_data -> r)               = NULL;
   (plgmres_data -> w)               = NULL;
   (plgmres_data -> norms)           = NULL;
   
   /* lgmres specific */
   (plgmres_data -> aug_dim)         = 2;
   (plgmres_data -> approx_constant) = 1;   

   return (void *) plgmres_data;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetKDim
 * \brief Set parameter 'k_dim' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetKDim( void *plgmres_vdata, JX_Int k_dim )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> k_dim) = k_dim;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetAugDim
 * \brief Set parameter 'aug_dim' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetAugDim( void *plgmres_vdata, JX_Int aug_dim )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> aug_dim) = aug_dim;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetTol
 * \brief Set parameter 'tol' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetTol( void *plgmres_vdata, JX_Real tol )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> tol) = tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetAbsoluteTol
 * \brief Set parameter 'a_tol' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetAbsoluteTol( void *plgmres_vdata, JX_Real a_tol )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> a_tol) = a_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetConvergenceFactorTol
 * \brief Set parameter 'cf_tol' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetConvergenceFactorTol( void *plgmres_vdata, JX_Real cf_tol )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> cf_tol) = cf_tol;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetMinIter
 * \brief Set parameter 'min_iter' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetMinIter( void *plgmres_vdata, JX_Int min_iter )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> min_iter) = min_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetMaxIter
 * \brief Set parameter 'max_iter' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetMaxIter( void *plgmres_vdata, JX_Int max_iter )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> max_iter) = max_iter;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetStopCrit
 * \brief Set parameter 'stop_crit ' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetStopCrit( void *plgmres_vdata, JX_Int stop_crit )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> stop_crit) = stop_crit;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetPrintLevel
 * \brief Set parameter 'print_level' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetPrintLevel( void *plgmres_vdata, JX_Int print_level )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   (plgmres_data -> print_level) = print_level;
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESGetNumIterations
 * \brief Get parameter 'num_iterations ' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESGetNumIterations( void *plgmres_vdata, JX_Int *num_iterations )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   *num_iterations = (plgmres_data -> num_iterations);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESGetConverged
 * \brief Get parameter 'converged' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESGetConverged( void *plgmres_vdata, JX_Int *converged )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   *converged = (plgmres_data -> converged);
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESGetFinalRelativeResidualNorm
 * \brief Get parameter 'relative_residual_norm' for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */ 
JX_Int
fsls_CSRPLGMRESGetFinalRelativeResidualNorm( void *plgmres_vdata, JX_Real *relative_residual_norm )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   *relative_residual_norm = (plgmres_data -> rel_residual_norm);
   return 0;
} 

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetPrecond
 * \brief Set preconditioner for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetPrecond( void *plgmres_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   fsls_CSRPLGMRESFunctions *plgmres_functions = plgmres_data->functions;

   (plgmres_functions -> precond)       = precond;
   (plgmres_functions -> precond_setup) = precond_setup;
   (plgmres_data -> precond_data)       = precond_data;
 
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESSetup
 * \brief Setup phase for PLGMRES
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSetup( void *plgmres_vdata, fsls_CSRMatrix *A, fsls_Vector *b, fsls_Vector *x )
{
   fsls_CSRPLGMRESData      *plgmres_data      = plgmres_vdata;
   fsls_CSRPLGMRESFunctions *plgmres_functions = (plgmres_data -> functions);

   JX_Int     k_dim              = (plgmres_data -> k_dim);
   JX_Int     max_iter           = (plgmres_data -> max_iter);
   JX_Int     (*precond_setup)() = (plgmres_functions -> precond_setup);
   void   *precond_data       = (plgmres_data -> precond_data);
   
   //--------------------------------------------------------//
   //                   lgmres mod                           //
   //--------------------------------------------------------//
   JX_Int     aug_dim            = (plgmres_data -> aug_dim);
   
   JX_Int     i;
   /* I change 'size = fsls_VectorSize(b)' into 'size = fsls_CSRMatrixNumRows(A)'
      since 'b' maybe NULL sometimes. peghoty, 2011/06/24 */
   JX_Int     size        = fsls_CSRMatrixNumRows(A);
   JX_Int     print_level = plgmres_data -> print_level;
   JX_Real *norms       = NULL;   
   
   fsls_Vector **p   = NULL;
   fsls_Vector  *r   = NULL;
   fsls_Vector  *w   = NULL;
   
   //--------------------------------------------------------//
   //                   lgmres mod                           //
   //--------------------------------------------------------//
   fsls_Vector **aug_vecs   = NULL;
   fsls_Vector **a_aug_vecs = NULL;
 
   if ((plgmres_data -> p) == NULL)
   {
      p = fsls_CTAlloc(fsls_Vector *, k_dim+1);
      for (i = 0; i < k_dim+1; i ++)
      {
         p[i] = fsls_SeqVectorCreate(size);
         fsls_SeqVectorInitialize(p[i]);
      }
      (plgmres_data -> p) = (void **)p; 
   }  
      
   if ((plgmres_data -> r) == NULL)
   {
      r = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(r);
      (plgmres_data -> r) = r;
   }
   
   if ((plgmres_data -> w) == NULL)
   {
      w = fsls_SeqVectorCreate(size);
      fsls_SeqVectorInitialize(w);
      (plgmres_data -> w) = w;
   }

   //--------------------------------------------------------//
   //                   lgmres mod                           //
   //--------------------------------------------------------//
   if ((plgmres_data -> aug_vecs) == NULL)
   {
      aug_vecs = fsls_CTAlloc(fsls_Vector *, aug_dim+1);
      for (i = 0; i < aug_dim+1; i ++)
      {
         aug_vecs[i] = fsls_SeqVectorCreate(size);
         fsls_SeqVectorInitialize(aug_vecs[i]);
      }
      (plgmres_data -> aug_vecs) = (void **)aug_vecs; 
   }
   if ((plgmres_data -> a_aug_vecs) == NULL)
   {
      a_aug_vecs = fsls_CTAlloc(fsls_Vector *, aug_dim);
      for (i = 0; i < aug_dim; i ++)
      {
         a_aug_vecs[i] = fsls_SeqVectorCreate(size);
         fsls_SeqVectorInitialize(a_aug_vecs[i]);
      }
      (plgmres_data -> a_aug_vecs) = (void **)a_aug_vecs; 
   }
   if ((plgmres_data -> aug_order) == NULL)
   {       
      (plgmres_data -> aug_order) = fsls_CTAlloc(JX_Int, aug_dim);
   } 
   
   /* preconditioning Setup */
   precond_setup(precond_data, A, b, x);

   if (print_level > 0)
   {
      if ((plgmres_data -> norms) == NULL)
      {
         norms = fsls_CTAlloc(JX_Real, max_iter+1);
         (plgmres_data -> norms) = norms;
      }
   }

   return 0;
}
 
/*!
 * \fn JX_Int fsls_CSRPLGMRESSolve
 * \brief Solve phase for PLGMRES
 * \note We are not checking rel. change for now.
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESSolve( void            *plgmres_vdata,
                      fsls_CSRMatrix  *A,
                      fsls_Vector     *b,
                      fsls_Vector     *x )
{
   fsls_CSRPLGMRESData       *plgmres_data      = plgmres_vdata;
   fsls_CSRPLGMRESFunctions  *plgmres_functions = (plgmres_data -> functions);
   JX_Int 	                     (*precond)()       = (plgmres_functions -> precond);
   void                      *precond_data      = (plgmres_data -> precond_data);
 
   JX_Int      k_dim        = (plgmres_data -> k_dim);       // default: 5
   JX_Int      min_iter     = (plgmres_data -> min_iter);    // default: 0
   JX_Int      max_iter     = (plgmres_data -> max_iter);    // default: 1000
   JX_Real   r_tol        = (plgmres_data -> tol);         // default: 1.0e-06
   JX_Real   cf_tol       = (plgmres_data -> cf_tol);      // default: 0.0
   JX_Real   a_tol        = (plgmres_data -> a_tol);       // default: 0.0 
   JX_Int      print_level  = (plgmres_data -> print_level); // default: 0
      
   void    *r            = (plgmres_data -> r);
   void    *w            = (plgmres_data -> w);
   void   **p            = (plgmres_data -> p);
   JX_Real  *norms        = (plgmres_data -> norms); 
   
   JX_Int      break_value = 0;
   JX_Int	    i,j,k,dadim;
   JX_Int      k_dim_plus_one = k_dim + 1;
   JX_Int      iter; 
   JX_Real   epsilon, gamma, t;
   JX_Real   r_norm, b_norm, den_norm;
   JX_Real   epsmac = 1.e-16; 
   JX_Real   ieee_check = 0.;
   JX_Real   cf_ave_0 = 0.0;
   JX_Real   cf_ave_1 = 0.0;
   JX_Real   weight;
   JX_Real   r_norm_0;
   JX_Real  *work = NULL;
   JX_Real  *rs   = NULL;
   JX_Real  *c    = NULL;
   JX_Real  *s    = NULL; 
   JX_Real **hh   = NULL;
   JX_Real  *hhs  = NULL;   

   //--------------------------------------------------------//
   //                   lgmres mod                           //
   //--------------------------------------------------------//
   void   **aug_vecs        = (plgmres_data -> aug_vecs);
   void   **a_aug_vecs      = (plgmres_data -> a_aug_vecs);
   JX_Int     *aug_order       = (plgmres_data -> aug_order);
   JX_Int      aug_dim         = (plgmres_data -> aug_dim);
   JX_Int      approx_constant = (plgmres_data -> approx_constant);
   JX_Int      it_arnoldi, aug_ct, it_total, it_aug;
   JX_Int      ii, order, spot = 0;
   JX_Real   tmp_norm, r_norm_last;
   
   (plgmres_data -> converged) = 0;

   if (print_level > 0)
   {
      norms = (plgmres_data -> norms);
   }

   /* initialize work arrays */
   work = fsls_CTAlloc(JX_Real, (4 + 2*aug_dim + k_dim)*k_dim + (aug_dim + 1)*aug_dim + 1);
   hh   = fsls_CTAlloc(JX_Real*, k_dim_plus_one+aug_dim);
   rs = work; c = rs + k_dim_plus_one; s = c + k_dim;
   hhs   = s + k_dim;
   dadim = k_dim + aug_dim;
   for (i = 0; i < k_dim_plus_one+aug_dim; i ++) hh[i] = hhs + i*dadim;

   /* compute initial residual */
   fsls_SeqVectorCopy(b, p[0]);
   fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, p[0]);
  
   b_norm = sqrt( fsls_SeqVectorInnerProd(b,b) );
   
  /*---------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an 
   * error flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (b_norm != 0.0) 
   {
      ieee_check = b_norm / b_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0)
      {
         jx_printf("\n\nERROR detected by FSLS ... BEGIN\n");
         jx_printf("ERROR -- fsls_CSRPLGMRESSolve: INFs and/or NaNs detected in input.\n");
         jx_printf("User probably placed non-numerics in supplied b.\n");
         jx_printf("Returning error flag += 101.  Program not terminated.\n");
         jx_printf("ERROR detected by FSLS ... END\n\n\n");
      }
      return 100;
   }   

   r_norm   = sqrt( fsls_SeqVectorInnerProd(p[0],p[0]) );
   r_norm_0 = r_norm;
   
  /*---------------------------------------------------------------------------
   * Since it does not diminish performance, attempt to return an 
   * error flag and notify users when they supply bad input. 
   *--------------------------------------------------------------------------*/
   if (r_norm != 0.0) 
   {
      ieee_check = r_norm / r_norm; /* INF -> NaN conversion */
   }
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0)
      {
        jx_printf("\n\nERROR detected by FSLS ... BEGIN\n");
        jx_printf("ERROR -- fsls_CSRPLGMRESSolve: INFs and/or NaNs detected in input.\n");
        jx_printf("User probably placed non-numerics in supplied A or x_0.\n");
        jx_printf("Returning error flag += 101.  Program not terminated.\n");
        jx_printf("ERROR detected by FSLS ... END\n\n\n");
      }
      return 100;
   }   
  
   if (print_level > 0)
   {
      norms[0] = r_norm;
      jx_printf("\n >>> L2 norm of b: %e\n", b_norm);
      if (b_norm == 0.0)
      {
         jx_printf(" >>> Rel_resid_norm actually contains the residual norm\n");
      }
      jx_printf(" >>> Initial L2 norm of residual: %e\n\n", r_norm);
   }
 
   iter = 0;

   if (b_norm > 0.0)
   {  
      den_norm= b_norm;  // convergence criterion |r_i|/|b| <= accuracy if |b| > 0
   }
   else
   { 
      den_norm= r_norm;  // convergence criterion |r_i|/|r0| <= accuracy if |b| = 0
   }

  /*---------------------------------------------------------------------------
   * convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
   * den_norm = |r_0| or |b|
   * note: default for a_tol is 0.0, so relative residual criteria is 
   *       used unless user specifies a_tol, or sets r_tol = 0.0, which 
   *       means absolute tol only is checked. 
   *--------------------------------------------------------------------------*/
   epsilon = fsls_max(a_tol,r_tol*den_norm);
   
   /* so now our stop criteria is |r_i| <= epsilon */
   
   if (print_level > 0)
   {
      if (b_norm > 0.0)
      {
         jx_printf(" ===================================================\n");
         jx_printf("   iter   resid.norm    conv.factor  rel.res.norm   \n");
         jx_printf(" ===================================================\n");
      }
      else
      {
         jx_printf(" =====================================\n");
         jx_printf("   iter    resid.norm   conv.factor   \n");
         jx_printf(" =====================================\n");
      }
   }

   //--------------------------------------------------------//
   //                   lgmres mod                           //
   //--------------------------------------------------------//
   /* lgmres initialization */
   for (ii = 0; ii < aug_dim; ii ++) 
   {
      aug_order[ii] = 0;
   }
   aug_ct = 0; /* number of aug. vectors available */

   /* outer iteration cycle */
   while (iter < max_iter)
   {
        /* initialize first term of hessenberg system */
	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           /* free some stuff */
           fsls_TFree(work);
           fsls_TFree(hh);
	   return 11;
	}

        /* see if we are already converged and 
           should print the final norm and exit */
	if (r_norm  <= epsilon && iter >= min_iter) 
        {
           fsls_SeqVectorCopy(b,r);
           fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
           r_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
              
           if (r_norm <= epsilon)
           {
              if (print_level > 0)
              {
                 jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
              }
              break;
           }
           else
           {
              if (print_level > 0) jx_printf(" >>> false convergence (1)\n");
           }
	}

      	t = 1.0 / r_norm;
      	r_norm_last = r_norm; /* lgmres mod */
	fsls_SeqVectorScale(t,p[0]);

        //--------------------------------------------------------//
        //                   lgmres mod                           //
        //--------------------------------------------------------//
        /* lgmres mod: determine number of arnoldi steps to take */
        /* if approx_constant then we keep the space the same size
           even if we don't have the full number of aug vectors yet */
        if (approx_constant) 
        {
           it_arnoldi = k_dim - aug_ct;
        } 
        else 
        {
           it_arnoldi = k_dim - aug_dim; 
        }
        it_total =  it_arnoldi + aug_ct;
        it_aug = 0; /* keep track of augmented iterations */
        
        /*--------- RESTART CYCLE (right-preconditioning) --------*/	
	i = 0;
        while (i < it_total && iter < max_iter)
	{
           i ++;
           iter ++;
           
           fsls_SeqVectorSetConstantValues(r, 0.0);

           //-----------------------------------------------------------------------//
           // LGMRES_MOD: decide whether this is an arnoldi step or an aug step     //
           //-----------------------------------------------------------------------//
           if ( i <= it_arnoldi ) /* Arnoldi */
           { 
              precond(precond_data, p[i-1], r);
              fsls_CSRMatrixMatvec01(1.0, A, r, 0.0, p[i]);
           } 
           else /* lgmres aug step */
           { 
              it_aug ++;
              order = i - it_arnoldi - 1; /* which aug step (note i starts at 1) - aug order number at 0*/ 
              for (ii = 0; ii < aug_dim; ii ++) 
              {
                 if (aug_order[ii] == order) 
                 {
                    spot = ii;
                    break; /* must have this because there will be duplicates before aug_ct = aug_dim */ 
                 }  
              }
              /* copy a_aug_vecs[spot] to p[i] */ 
              fsls_SeqVectorCopy(a_aug_vecs[spot],p[i]);
              
              /*note: an alternate implementation choice would be to only save the
                AUGVECS and not A_AUGVEC and then apply the PC here to the augvec */
           }       
          
           /* modified Gram_Schmidt */
           for (j = 0; j < i; j ++)
           {
              hh[j][i-1] = fsls_SeqVectorInnerProd(p[j],p[i]);
              fsls_SeqVectorAxpy(-hh[j][i-1],p[j],p[i]);
           }
           t = sqrt( fsls_SeqVectorInnerProd(p[i],p[i]) );
           hh[i][i-1] = t;	
           if (t != 0.0)
           {
              t = 1.0/t;
              fsls_SeqVectorScale(t,p[i]);
           }
           
           /* done with modified Gram_schmidt and Arnoldi step.
              update factorization of hh */
           for (j = 1; j < i; j ++)
           {
              t = hh[j-1][i-1];
              hh[j-1][i-1] = s[j-1]*hh[j][i-1] + c[j-1]*t;
              hh[j][i-1]   = -s[j-1]*t + c[j-1]*hh[j][i-1];
           }
           t  = hh[i][i-1]*hh[i][i-1];
           t += hh[i-1][i-1]*hh[i-1][i-1];
           gamma = sqrt(t);
           if (gamma == 0.0) gamma = epsmac;
           c[i-1]  = hh[i-1][i-1] / gamma;
           s[i-1]  = hh[i][i-1] / gamma;
           rs[i]   = -s[i-1]*rs[i-1];
           rs[i-1] = c[i-1]*rs[i-1];
           /* determine residual norm */
           hh[i-1][i-1] = s[i-1]*hh[i][i-1] + c[i-1]*hh[i-1][i-1];
           r_norm = fabs(rs[i]);

           /* print ? */
           if (print_level > 0)
           {
              norms[iter] = r_norm;

              if (b_norm > 0.0)
                 jx_printf("% 5d    %e    %f    %e\n", iter, norms[iter],
                                                    norms[iter]/norms[iter-1], norms[iter]/b_norm);
              else
                 jx_printf("% 5d    %e     %f\n", iter, norms[iter], norms[iter]/norms[iter-1]);
           }
           
           /* convergence factor tolerance */
           if (cf_tol > 0.0)
           {
              cf_ave_0 = cf_ave_1;
              cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));
              
              weight = fabs(cf_ave_1 - cf_ave_0);
              weight = weight / fsls_max(cf_ave_1, cf_ave_0);
              weight = 1.0 - weight;

              if (weight * cf_ave_1 > cf_tol) 
              {
                 break_value = 1;
                 break;
              }
           }
           
           /* should we exit the restart cycle? (conv. check) */
           if (r_norm <= epsilon && iter >= min_iter)
           {
              break;
           }
           
	} 
	/*----------- End of RESTART CYCLE ---------------*/
	

	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1] / hh[i-1][i-1];
	for (k = i-2; k >= 0; k --)
	{
           t = 0.0;
           for (j = k+1; j < i; j ++)
           {
              t -= hh[k][j]*rs[j];
           }
           t += rs[k];
           rs[k] = t / hh[k][k];
	}

        //--------------------------------------------------------//
        //                   lgmres mod                           //
        //--------------------------------------------------------//
        /* form linear combination of p's to get solution */
        /* put the new aug_vector in aug_vecs[aug_dim] - a temp position */	
        /* i          = number of iterations */  
        /* it_aug     = number of augmented iterations */ 
        /* it_arnoldi = number of arnoldi iterations */

        /* check if exited early before all arnoldi its */
        if (it_arnoldi > i) it_arnoldi = i; 	
	
        if (!it_aug)
        {
           fsls_SeqVectorCopy(p[i-1],w);    
           fsls_SeqVectorScale(rs[i-1],w);
           
           for (j = i-2; j >= 0; j --)
           {
              fsls_SeqVectorAxpy(rs[j], p[j], w);
           }
        }
        else /* need some of the augvecs */
        {
           fsls_SeqVectorCopy(p[0],w);
           fsls_SeqVectorScale(rs[0],w);

           /* reg. arnoldi directions */  
           for (j = 1; j < it_arnoldi; j ++) /* first one already done */
           {
              fsls_SeqVectorAxpy(rs[j], p[j], w);
           }
            
           /* augment directions */
           for (ii = 0; ii < it_aug; ii ++) 
           {
              for (j = 0; j < aug_dim; j ++) 
              {
                 if (aug_order[j] == ii) 
                 {
                    spot = j;
                    break; /* must have this because there will be
                            * duplicates before aug_ct = aug_dim */ 
                 }  
              }
              fsls_SeqVectorAxpy(rs[it_arnoldi+ii], aug_vecs[spot], w);
           }
        }
	
        /* grab the new aug vector before the prec */
        fsls_SeqVectorCopy(w, aug_vecs[aug_dim]);

	/* find correction (in r) (un-wind precond.) */
	fsls_SeqVectorSetConstantValues(r, 0.0);
	precond(precond_data, w, r);        

        /* update current solution x (in x) */
	fsls_SeqVectorAxpy(1.0,r,x);
	

        /* check for convergence by evaluating the actual residual */
	if (r_norm <= epsilon && iter >= min_iter) 
        {
           /* calculate actual residual norm*/
           fsls_SeqVectorCopy(b,r);
           fsls_CSRMatrixMatvec01(-1.0, A, x, 1.0, r);
           r_norm = sqrt( fsls_SeqVectorInnerProd(r,r) );
           
           if (r_norm <= epsilon)
           {
              if (print_level > 0)
              {
                 jx_printf("\n >>> Final L2 norm of residual: %e\n\n", r_norm);
              }
              (plgmres_data -> converged) = 1;
              break;
           }
           else /* conv. has not occurred, according to true residual */ 
           {
              if (print_level > 0)
              {
                 jx_printf(" >>> false convergence (2)\n");
              }
              fsls_SeqVectorCopy(r,p[0]);
              i = 0;
           }
	} /* end of convergence check */

        /* compute residual vector and continue loop */
        
        /* copy r0 (not scaled) to w */  
	fsls_SeqVectorCopy(p[0],w);
	fsls_SeqVectorScale(r_norm_last,w);
        
	for (j = i; j > 0; j --)
	{
           rs[j-1] = -s[j-1]*rs[j];
           rs[j]   =  c[j-1]*rs[j];
	}
        
        if (i) 
        {
           fsls_SeqVectorAxpy(rs[i]-1.0,p[i],p[i]);
        }
        for (j = i-1 ; j > 0; j --)
        {
           fsls_SeqVectorAxpy(rs[j],p[j],p[i]);
        }
        
        if (i)
        {
           fsls_SeqVectorAxpy(rs[0]-1.0,p[0],p[0]);
           fsls_SeqVectorAxpy(1.0,p[i],p[0]);
        }
        
        //--------------------------------------------------------------------//
        //                           lgmres mod                               //
        //  collect aug vector and A*augvector for future restarts - only     //
        //  if we will be restarting (i.e. this cycle performed it_total      //
        //  iterations). ordering starts at 0.                                //
        //--------------------------------------------------------------------//
        if (aug_dim > 0) 
        {
           if (!aug_ct) 
           {
              spot = 0; aug_ct ++;
           } 
           else if (aug_ct < aug_dim) 
           {
              spot = aug_ct; aug_ct ++;
           } 
           else /* truncate - already have aug_dim number of vectors */
           { 
              for (ii=0; ii<aug_dim; ii++) 
              {
                 if (aug_order[ii] == (aug_dim-1)) 
                 {
                    spot = ii;
                 }  
              }
           } 
           /* aug_vecs[aug_dim] contains new aug vector */
           fsls_SeqVectorCopy(aug_vecs[aug_dim], aug_vecs[spot]);
           
           /* need to normalize */
           tmp_norm = sqrt( fsls_SeqVectorInnerProd(aug_vecs[spot], aug_vecs[spot]) );
           tmp_norm = 1.0 / tmp_norm;
           fsls_SeqVectorScale(tmp_norm ,aug_vecs[spot]);
           
           /*set new aug vector to order 0  - move all others back one */
           for (ii = 0; ii < aug_dim; ii ++) 
           {
              aug_order[ii] ++;
           } 
           aug_order[spot] = 0; 

          /*----------------------------------------------------------------
           * now add the A*aug vector to A_AUGVEC(spot) 
           * - this is independ. of preconditioning type
           * A*augvec = V*H*y = r0 - rm (r0 is in w and rm is in p[0]) 
           *--------------------------------------------------------------*/
           fsls_SeqVectorCopy(w, a_aug_vecs[spot]);
           fsls_SeqVectorScale(-1.0, a_aug_vecs[spot]);       /* -r0 */
           fsls_SeqVectorAxpy(1.0, p[0],a_aug_vecs[spot]);    /* rm - r0 */
           fsls_SeqVectorScale(-tmp_norm, a_aug_vecs[spot]);  /* r0-rm / norm */
        } // end if (aug_dim > 0)
   } 
   /* END of iteration while loop */
   
   (plgmres_data -> num_iterations) = iter;  // get the number of iterations

   /* get the last relative residual-norm */
   if (b_norm > 0.0)
   {
      (plgmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (plgmres_data -> rel_residual_norm) = r_norm;
   }
   
   if (iter >= max_iter && r_norm > epsilon) 
   {
      jx_printf("\n Warning: Not reaching the given tolerance in %d iterations!!\n\n", max_iter);
   }
  
  /*-------------------------------------------
   * Free some stuff
   *------------------------------------------*/   
   fsls_TFree(work);
   fsls_TFree(hh);
   
   return 0;
}

/*!
 * \fn JX_Int fsls_CSRPLGMRESDestroy
 * \brief Destroy a PLGMRES object
 * \author peghoty
 * \date 2010/12/11
 */
JX_Int
fsls_CSRPLGMRESDestroy( void *plgmres_vdata )
{
   fsls_CSRPLGMRESData *plgmres_data = plgmres_vdata;
   JX_Int i;
   JX_Int k_dim   = (plgmres_data -> k_dim);
   JX_Int aug_dim = (plgmres_data -> aug_dim);
 
   if (plgmres_data)
   {
      if ( (plgmres_data -> print_level) > 0 )
      {
         /* norms */
         if ( (plgmres_data -> norms) != NULL )
         {
            fsls_TFree(plgmres_data -> norms);
            (plgmres_data -> norms) = NULL;
         }
      }
      /* r */
      if ( (plgmres_data -> r) != NULL )
      {
         fsls_SeqVectorDestroy(plgmres_data -> r);
         (plgmres_data -> r) = NULL;
      }  
      /* w */    
      if ( (plgmres_data -> w) != NULL )
      {
         fsls_SeqVectorDestroy(plgmres_data -> w);
         (plgmres_data -> w) = NULL;
      }   
      /* p */ 
      if ( (plgmres_data -> p) != NULL )
      {
         for (i = 0; i < k_dim+1; i ++)
         {
            if ( (plgmres_data -> p)[i] != NULL )
            {
	       fsls_SeqVectorDestroy((plgmres_data -> p)[i]);
	       (plgmres_data -> p)[i] = NULL;
	    }
         }
         fsls_TFree(plgmres_data -> p);
         plgmres_data -> p = NULL;
      }
      //----------------------------------------------------//
      //                   lgmres mod                       //
      //----------------------------------------------------//      
      /* aug_order */
      if ( plgmres_data -> aug_order != NULL )
      {
         fsls_TFree(plgmres_data -> aug_order);
      }
      /* aug_vecs */
      if ( (plgmres_data -> aug_vecs) != NULL )
      {
         for (i = 0; i < aug_dim+1; i ++)
         {
            if ( (plgmres_data -> aug_vecs)[i] != NULL )
            {
	       fsls_SeqVectorDestroy((plgmres_data -> aug_vecs)[i]);
	       (plgmres_data -> aug_vecs)[i] = NULL;
	    }
         }
         fsls_TFree(plgmres_data -> aug_vecs);
         plgmres_data -> aug_vecs = NULL;
      }
      
      /* a_aug_vecs */
      if ( (plgmres_data -> a_aug_vecs) != NULL )
      {
         for (i = 0; i < aug_dim; i ++)
         {
            if ( (plgmres_data -> a_aug_vecs)[i] != NULL )
            {
	       fsls_SeqVectorDestroy((plgmres_data -> a_aug_vecs)[i]);
	       (plgmres_data -> a_aug_vecs)[i] = NULL;
	    }
         }
         fsls_TFree(plgmres_data -> a_aug_vecs);
         plgmres_data -> a_aug_vecs = NULL;
      }            
      
      fsls_TFree(plgmres_data -> functions);
      fsls_TFree(plgmres_data);            
   }
 
   return 0;
}
