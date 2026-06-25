//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_krylov.h -- head files for Krylov subspace iterative methods
 *  Date: 2009/10/08
 * 
 *  Created by peghoty
 */ 

#ifndef JX_KRYLOV_HEADER
#define JX_KRYLOV_HEADER 

#ifndef JX_UTIL_HEADER 
#include "jx_util.h"
#endif



/*----------------------------------------------------------------*
 *                      Macro Definition                          *
 *----------------------------------------------------------------*/ 
                                   
#define jx_CTAllocF(type, count, funcs) \
( (type *)(*(funcs->CAlloc))((JX_UInt)(count), (JX_UInt)sizeof(type)) )
#define jx_TFreeF( ptr, funcs ) \
( (*(funcs->Free))((char *)ptr), ptr = NULL )

typedef JX_Int (*JX_PtrToSolverFcn)(JX_Solver, JX_Matrix, JX_Vector, JX_Vector);
typedef JX_Int (*JX_PtrToModifyPCFcn)(JX_Solver, JX_Int, JX_Real);

/*----------------------------------------------------------------*
 *                   Struct Declaration                           *
 *----------------------------------------------------------------*/

/*!
 * \struct jx_PCGFunctions
 * \brief fuctions used in PCG.
 */ 
typedef struct
{
   char * (*CAlloc)        ( size_t count, size_t elt_size );
   JX_Int    (*Free)          ( char *ptr );
   JX_Int    (*CommInfo)      ( void *A, JX_Int *my_id, JX_Int *num_procs );
   void * (*CreateVector)  ( void *vector );
   JX_Int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y );
   JX_Int    (*MatvecDestroy) ( void *matvec_data );
   JX_Real (*InnerProd)     ( void *x, void *y );
   JX_Int    (*CopyVector)    ( void *x, void *y );
   JX_Int    (*ClearVector)   ( void *x );
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x );
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y );

   JX_Int    (*precond)();
   JX_Int    (*precond_setup)();

} jx_PCGFunctions;


/*!
 * \struct jx_PCGData
 */  
typedef struct
{
   JX_Real   tol;
   JX_Real   atolf;
   JX_Real   cf_tol;
   JX_Real   a_tol;
   JX_Int      max_iter;
   JX_Int      two_norm;
   JX_Int      rel_change;
   JX_Int      recompute_residual;
   JX_Int      stop_crit;
   JX_Int      converged;
   JX_Int      conv_criteria;      /* peghoty 2010/06/23 */
   JX_Real   convfac_threshold;  /* peghoty 2010/06/23 */

   void    *A;
   void    *B; /* precond matrix  peghoty 2011/09/04 */  
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging > 1 */

#ifdef PCG_Modified_zl                  
   void    *vec_tmp; // zhaoli               
#endif

   JX_Int      owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    *matvec_data;
   void    *precond_data;

   jx_PCGFunctions *functions;

   /* log info (always logged) */
   JX_Int      num_iterations;
   JX_Real   rel_residual_norm;

   JX_Int      print_level; /* printing when print_level > 0 */
   JX_Int      logging;     /* extra computations for logging when logging > 0 */
   JX_Real  *norms;
   JX_Real  *rel_norms;

} jx_PCGData;

/*!
 * \struct jx_GMRESFunctions
 */  
typedef struct
{
   char * (*CAlloc)            ( size_t count, size_t elt_size );
   JX_Int    (*Free)              ( char *ptr );
   JX_Int    (*CommInfo)          ( void  *A, JX_Int   *my_id, JX_Int   *num_procs );
   void * (*CreateVector)      ( void *vector );
   void * (*CreateVectorArray) ( JX_Int size, void *vectors );
   JX_Int    (*DestroyVector)     ( void *vector );
   void * (*MatvecCreate)      ( void *A, void *x );
   JX_Int    (*Matvec)            ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y );
   JX_Int    (*MatvecDestroy)     ( void *matvec_data );
   JX_Real (*InnerProd)         ( void *x, void *y );
   JX_Int    (*CopyVector)        ( void *x, void *y );
   JX_Int    (*ClearVector)       ( void *x );
   JX_Int    (*ScaleVector)       ( JX_Real alpha, void *x );
   JX_Int    (*Axpy)              ( JX_Real alpha, void *x, void *y );

   JX_Int    (*precond)();
   JX_Int    (*precond_setup)();

} jx_GMRESFunctions;

/*!
 * \struct jx_GMRESData
 */
typedef struct
{
   JX_Int      k_dim;
   JX_Int      min_iter;
   JX_Int      max_iter;
   JX_Int      rel_change;
   JX_Int      stop_crit;
   JX_Int      conv_criteria;       /* peghoty,  2010/06/23 */
   JX_Real   convfac_threshold;   /* peghoty,  2010/06/23 */
   JX_Real   resdown_0_threshold; /* Yue Xiaoqiang, 2014/03/24 */
   JX_Real   convfac_threshold_2; /* Yue Xiaoqiang, 2014/03/24 */
   JX_Int      is_check_restarted;  /* peghoty,  2011/11/08 */
   JX_Int      break_adaptive;      /* Yue Xiaoqiang, 2014/10/24 */
   JX_Int      last_precond_type;   /* Yue Xiaoqiang, 2014/10/24 */
   JX_Int      comp_amg_iter;       /* Yue Xiaoqiang, 2016/02/20 */
   JX_Int      comp_ilu_iter;       /* Yue Xiaoqiang, 2016/02/20 */
   JX_Int      comp_bco_iter;       /* Yue Xiaoqiang, 2016/02/20 */
   JX_Int      converged;
   JX_Real   tol;
   JX_Real   cf_tol;
   JX_Real   a_tol;
   JX_Real   rel_residual_norm;

   void    *A;
   void    *B; /* precond matrix  peghoty 2011/09/04 */   
   void    *r;
   void    *w;
   void    *w_2;
   void   **p;

   void    *matvec_data;
   void    *precond_data;

   jx_GMRESFunctions *functions;

   /* log info (always logged) */
   JX_Int     num_iterations;
 
   JX_Int     print_level; /* printing when print_level > 0 */
   JX_Int     logging;     /* extra computations for logging when logging > 0 */
   JX_Real *norms;
   char   *log_file_name;

} jx_GMRESData;

/*!
 * \struct jx_COGMRESFunctions
 */ 
typedef struct
{
   char *       (*CAlloc)        ( size_t count, size_t elt_size );
   JX_Int    (*Free)          ( char *ptr );
   JX_Int    (*CommInfo)      ( void  *A, JX_Int   *my_id, JX_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   void *       (*CreateVectorArray)  ( JX_Int size, void *vectors );
   JX_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y );
   JX_Int    (*MatvecDestroy) ( void *matvec_data );
   JX_Real   (*InnerProd)     ( void *x, void *y );
   JX_Int    (*MassInnerProd) ( void *x, void **p, JX_Int k, JX_Int unroll, void *result );
   JX_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, JX_Int k, JX_Int unroll, void *result_x, void *result_y );
   JX_Int    (*CopyVector)    ( void *x, void *y );
   JX_Int    (*ClearVector)   ( void *x );
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x );
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y );
   JX_Int    (*MassAxpy)      ( JX_Real *alpha, void **x, void *y, JX_Int k, JX_Int unroll );
   JX_Int    (*precond)       ();
   JX_Int    (*precond_setup) ();

   JX_Int    (*modify_pc)( void *precond_data, JX_Int iteration, JX_Real rel_residual_norm );

} jx_COGMRESFunctions;

/*!
 * \struct jx_COGMRESData
 */
typedef struct
{
   JX_Int      k_dim;
   JX_Int      unroll;
   JX_Int      cgs;
   JX_Int      min_iter;
   JX_Int      max_iter;
   JX_Int      rel_change;
   JX_Int      is_check_restarted;
   JX_Int      skip_real_r_check;
   JX_Int      converged;
   JX_Real   tol;
   JX_Real   cf_tol;
   JX_Real   a_tol;
   JX_Real   rel_residual_norm;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   jx_COGMRESFunctions * functions;

   /* log info (always logged) */
   JX_Int      num_iterations;
 
   JX_Int     print_level; /* printing when print_level>0 */
   JX_Int     logging;  /* extra computations for logging when logging>0 */
   JX_Real  *norms;
   char    *log_file_name;

} jx_COGMRESData;

/*!
 * \struct jx_FlexGMRESFunctions
 */
typedef struct
{
   char *       (*CAlloc)        ( size_t count, size_t elt_size );
   JX_Int    (*Free)          ( char *ptr );
   JX_Int    (*CommInfo)      ( void  *A, JX_Int   *my_id, JX_Int   *num_procs );
   void *       (*CreateVector)  ( void *vector );
   void *       (*CreateVectorArray)  ( JX_Int size, void *vectors );
   JX_Int    (*DestroyVector) ( void *vector );
   void *       (*MatvecCreate)  ( void *A, void *x );
   JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y );
   JX_Int    (*MatvecDestroy) ( void *matvec_data );
   JX_Real   (*InnerProd)     ( void *x, void *y );
   JX_Int    (*CopyVector)    ( void *x, void *y );
   JX_Int    (*ClearVector)   ( void *x );
   JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x );
   JX_Int    (*Axpy)          ( JX_Real alpha, void *x, void *y );

   JX_Int    (*precond)(void *vdata, void *A, void *b, void *x );
   JX_Int    (*precond_setup)(void *vdata, void *A );

   JX_Int    (*modify_pc)(void *precond_data, JX_Int iteration, JX_Real rel_residual_norm );

} jx_FlexGMRESFunctions;

/*!
 * \struct jx_FlexGMRESData
 */
typedef struct
{
   JX_Int      k_dim;
   JX_Int      min_iter;
   JX_Int      max_iter;
   JX_Int      rel_change;
   JX_Int      stop_crit;
   JX_Int      converged;
   JX_Int      is_check_restarted;
   JX_Real   tol;
   JX_Real   cf_tol;
   JX_Real   a_tol;
   JX_Real   rel_residual_norm;

   void   **pre_vecs;

   void  *A;
   void  *r;
   void  *w;
   void  *w_2;
   void  **p;

   void    *matvec_data;
   void    *precond_data;

   jx_FlexGMRESFunctions * functions;

   /* log info (always logged) */
   JX_Int      num_iterations;
 
   JX_Int     print_level; /* printing when print_level>0 */
   JX_Int     logging;  /* extra computations for logging when logging>0 */
   JX_Real  *norms;
   char    *log_file_name;

} jx_FlexGMRESData;

/*!
 * \struct jx_BiCGSTABFunctions
 */ 
typedef struct
{
  void * (*CreateVector)  ( void *vvector );
  JX_Int    (*DestroyVector) ( void *vvector );
  void * (*MatvecCreate)  ( void *A, void *x );
  JX_Int    (*Matvec)        ( void *matvec_data, JX_Real alpha, void *A, void *x, JX_Real beta, void *y );
  JX_Int    (*MatvecDestroy) ( void *matvec_data );
  JX_Real (*InnerProd)     ( void *x, void *y );
  JX_Int    (*CopyVector)    ( void *x, void *y );
  JX_Int    (*ClearVector)   ( void *x );
  JX_Int    (*ScaleVector)   ( JX_Real alpha, void *x );
  JX_Int    (*Axpy)          ( JX_Real alpha, void *x , void *y );
  JX_Int    (*CommInfo)      ( void *A, JX_Int *my_id, JX_Int *num_procs );
  
  JX_Int    (*precond_setup)();
  JX_Int    (*precond)();

} jx_BiCGSTABFunctions;


/*!
 * \struct jx_BiCGSTABData
 */ 
typedef struct
{
   JX_Int      min_iter;
   JX_Int      max_iter;
   JX_Int      conv_criteria;      /* peghoty 2010/03/20 */
   JX_Real   convfac_threshold;  /* peghoty 2010/06/23 */
   JX_Int      converged;
   JX_Real   tol;
   JX_Real   cf_tol;
   JX_Real   rel_residual_norm;
   JX_Real   a_tol;
   
   void    *A;
   void    *B; /* precond matrix  peghoty 2011/09/04 */
   void    *r;
   void    *r0;
   void    *s;
   void    *v;
   void    *p;
   void    *q;
   void    *Xtmp; /* peghoty 2010/03/20 */

   void    *matvec_data;
   void    *precond_data;

   jx_BiCGSTABFunctions *functions;

   /* log info (always logged) */
   JX_Int      num_iterations;
 
   /* additional log info (logged when 'logging' > 0) */
   JX_Int      logging;
   JX_Int      print_level;
   JX_Real  *norms;
   char    *log_file_name;

} jx_BiCGSTABData;


/*----------------------------------------------------------------*
 *                   Function Declaration                         *
 *----------------------------------------------------------------*/
 
/* /csrc/krylov/krylov.c */
char *jx_ParKrylovCAlloc( JX_Int count, JX_Int elt_size );                              
JX_Int jx_ParKrylovFree( char *ptr );                                                  
JX_Int jx_ParKrylovCommInfo( void *A, JX_Int *my_id, JX_Int *num_procs );                                                 
void *jx_ParKrylovCreateVector( void *vvector );                           
void *jx_ParKrylovCreateVectorArray( JX_Int n, void *vvector );                        
JX_Int jx_ParKrylovDestroyVector( void *vvector );                                                                        
void *jx_ParKrylovMatvecCreate( void *A, void *x );                      
JX_Int jx_ParKrylovMatvec( void *matvec_data, JX_Real alpha, void *A, void *x,JX_Real beta, void *y );
JX_Int jx_ParKrylovMatvecDestroy( void *matvec_data );
JX_Real jx_ParKrylovInnerProd( void *x, void *y );
JX_Int jx_ParKrylovMassInnerProd( void *x, void **y, JX_Int k, JX_Int unroll, void  *result );
JX_Int jx_ParKrylovMassDotpTwo( void *x, void *y, void **z, JX_Int k, JX_Int unroll, void *result_x, void *result_y );
JX_Int jx_ParKrylovCopyVector( void *x, void *y );
JX_Int jx_ParKrylovClearVector( void *x );
JX_Int jx_ParKrylovScaleVector( JX_Real alpha, void *x );
JX_Int jx_ParKrylovAxpy( JX_Real alpha, void *x, void *y );
JX_Int jx_ParKrylovMassAxpy( JX_Real *alpha, void **x, void *y, JX_Int k, JX_Int unroll );
JX_Int jx_ParKrylovIdentitySetup( void *vdata, void *A );            
JX_Int jx_ParKrylovIdentity( void *vdata, void *A, void *b, void *x );

/* /csrc/krylov/pcg.c */
JX_Int JX_ParCSRPCGCreate( MPI_Comm comm, JX_Solver *solver );                
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
);                                              
void *jx_PCGCreate( jx_PCGFunctions *pcg_functions );                                     
JX_Int JX_PCGSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );                     
JX_Int JX_PCGSolve( JX_Solver solver, JX_Matrix  preOperator, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );  

JX_Int JX_PCGSetPreOperator( JX_Solver solver, JX_ParCSRMatrix preOperator ); /* peghoty 2011/09/04 */
JX_Int JX_PCGSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_PCGGetTol( JX_Solver solver, JX_Real *tol );                       
JX_Int JX_PCGSetAbsoluteTol( JX_Solver solver, JX_Real a_tol );
JX_Int JX_PCGGetAbsoluteTol( JX_Solver solver, JX_Real *a_tol );                        
JX_Int JX_PCGSetAbsoluteTolFactor( JX_Solver solver, JX_Real abstolf );
JX_Int JX_PCGGetAbsoluteTolFactor( JX_Solver solver, JX_Real *abstolf );                                        
JX_Int JX_PCGSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol );
JX_Int JX_PCGGetConvergenceFactorTol( JX_Solver solver, JX_Real *cf_tol );                                         
JX_Int JX_PCGSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_PCGGetMaxIter( JX_Solver solver, JX_Int *max_iter );                                                            
JX_Int JX_PCGSetStopCrit( JX_Solver solver, JX_Int stop_crit );
JX_Int JX_PCGGetStopCrit( JX_Solver solver, JX_Int *stop_crit );               
JX_Int JX_PCGSetConvCriteria( JX_Solver solver, JX_Int conv_criteria ); /* peghoty 2010/06/23 */
JX_Int JX_PCGSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold );                 
JX_Int JX_PCGSetTwoNorm( JX_Solver solver, JX_Int two_norm );
JX_Int JX_PCGGetTwoNorm( JX_Solver solver, JX_Int *two_norm );               
JX_Int JX_PCGSetRelChange( JX_Solver solver, JX_Int rel_change );
JX_Int JX_PCGGetRelChange( JX_Solver solver, JX_Int *rel_change );
JX_Int JX_PCGSetRecomputeResidual( JX_Solver solver, JX_Int recompute_residual );
JX_Int JX_PCGGetRecomputeResidual( JX_Solver solver, JX_Int *recompute_residual );
JX_Int 
JX_PCGSetPrecond( JX_Solver          solver, 
                  JX_PtrToSolverFcn  precond,
                  JX_PtrToSolverFcn  precond_setup, 
                  JX_Solver          precond_solver );
JX_Int JX_PCGGetPrecond( JX_Solver solver, JX_Solver *precond_data_ptr );
JX_Int JX_PCGSetLogging( JX_Solver solver, JX_Int level );
JX_Int JX_PCGGetLogging( JX_Solver solver, JX_Int *level );
JX_Int JX_PCGSetPrintLevel( JX_Solver solver, JX_Int level );
JX_Int JX_PCGGetPrintLevel( JX_Solver solver, JX_Int *level );
JX_Int JX_PCGGetNumIterations( JX_Solver  solver, JX_Int *num_iterations );                         
JX_Int JX_PCGGetConverged( JX_Solver solver, JX_Int *converged );                                                    
JX_Int JX_PCGGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm );                           
JX_Int JX_PCGGetResidual( JX_Solver solver, void **residual );                                                   
JX_Int JX_ParCSRPCGDestroy( JX_Solver solver );                                                
JX_Int jx_PCGDestroy( void *pcg_vdata );
JX_Int jx_PCGGetResidual( void *pcg_vdata, void **residual );
JX_Int jx_PCGSetup( void *pcg_vdata, void *matA, void *vecB, void *vecX );
JX_Int jx_PCGSolve( void *pcg_vdata, void *preOperator, void *matA, void *vecB, void *vecX );
JX_Int jx_PCGSetPreOperator( void *pcg_vdata, void *preOperator ); /* peghoty 2011/09/04 */
JX_Int jx_PCGSetTol( void *pcg_vdata, JX_Real tol );
JX_Int jx_PCGGetTol( void *pcg_vdata, JX_Real *tol );
JX_Int jx_PCGSetAbsoluteTol( void *pcg_vdata, JX_Real a_tol );
JX_Int jx_PCGGetAbsoluteTol( void *pcg_vdata, JX_Real *a_tol );
JX_Int jx_PCGSetAbsoluteTolFactor( void *pcg_vdata, JX_Real atolf );
JX_Int jx_PCGGetAbsoluteTolFactor( void *pcg_vdata, JX_Real *atolf );
JX_Int jx_PCGSetConvergenceFactorTol( void *pcg_vdata, JX_Real cf_tol );
JX_Int jx_PCGGetConvergenceFactorTol( void *pcg_vdata, JX_Real *cf_tol );
JX_Int jx_PCGSetMaxIter( void *pcg_vdata, JX_Int max_iter );
JX_Int jx_PCGGetMaxIter( void *pcg_vdata, JX_Int *max_iter );
JX_Int jx_PCGSetTwoNorm( void *pcg_vdata, JX_Int two_norm );
JX_Int jx_PCGGetTwoNorm( void *pcg_vdata, JX_Int *two_norm );
JX_Int jx_PCGSetRelChange( void *pcg_vdata, JX_Int rel_change );
JX_Int jx_PCGGetRelChange( void *pcg_vdata, JX_Int *rel_change );
JX_Int jx_PCGSetRecomputeResidual( void *pcg_vdata, JX_Int recompute_residual );
JX_Int jx_PCGGetRecomputeResidual( void *pcg_vdata, JX_Int *recompute_residual );
JX_Int jx_PCGSetStopCrit( void *pcg_vdata, JX_Int stop_crit );
JX_Int jx_PCGGetStopCrit( void *pcg_vdata, JX_Int * stop_crit );
JX_Int jx_PCGSetConvCriteria( void *pcg_vdata, JX_Int conv_criteria ); /* peghoty 2010/06/23 */
JX_Int jx_PCGSetConvFacThreshold( void *pcg_vdata, JX_Real convfac_threshold ); /* peghoty 2010/06/23 */
JX_Int jx_PCGGetPrecond( void *pcg_vdata, JX_Solver *precond_data_ptr );
JX_Int jx_PCGSetPrecond( void *pcg_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data ); 
JX_Int jx_PCGSetPrintLevel( void *pcg_vdata, JX_Int level );
JX_Int jx_PCGGetPrintLevel( void *pcg_vdata, JX_Int *level );
JX_Int jx_PCGSetLogging( void *pcg_vdata, JX_Int level );
JX_Int jx_PCGGetLogging( void *pcg_vdata, JX_Int *level);
JX_Int jx_PCGGetNumIterations( void *pcg_vdata, JX_Int *num_iterations );
JX_Int jx_PCGGetConverged( void *pcg_vdata, JX_Int *converged );
JX_Int jx_PCGPrintLogging( void *pcg_vdata, JX_Int myid );
JX_Int jx_PCGGetFinalRelativeResidualNorm( void *pcg_vdata, JX_Real *relative_residual_norm );

/* /csrc/krylov/pcogmres.c */
JX_Int JX_ParCSRCOGMRESCreate( MPI_Comm comm, JX_Solver *solver );
JX_Int JX_ParCSRCOGMRESDestroy( JX_Solver solver );
JX_Int JX_COGMRESSetup( JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x );
JX_Int JX_COGMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix A, JX_Vector b, JX_Vector x );
JX_Int JX_COGMRESSetKDim( JX_Solver solver, JX_Int k_dim );
JX_Int JX_COGMRESGetKDim( JX_Solver solver, JX_Int * k_dim );
JX_Int JX_COGMRESSetUnroll( JX_Solver solver, JX_Int unroll );
JX_Int JX_COGMRESGetUnroll( JX_Solver solver, JX_Int * unroll );
JX_Int JX_COGMRESSetCGS( JX_Solver solver, JX_Int cgs );
JX_Int JX_COGMRESGetCGS( JX_Solver solver, JX_Int * cgs );
JX_Int JX_COGMRESSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_COGMRESGetTol( JX_Solver solver, JX_Real * tol );
JX_Int JX_COGMRESSetAbsoluteTol( JX_Solver solver, JX_Real a_tol );
JX_Int JX_COGMRESGetAbsoluteTol( JX_Solver solver, JX_Real * a_tol );
JX_Int JX_COGMRESSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol );
JX_Int JX_COGMRESGetConvergenceFactorTol( JX_Solver solver, JX_Real * cf_tol );
JX_Int JX_COGMRESSetMinIter( JX_Solver solver, JX_Int min_iter );
JX_Int JX_COGMRESGetMinIter( JX_Solver solver, JX_Int * min_iter );
JX_Int JX_COGMRESSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_COGMRESSetIsCheckRestarted( JX_Solver solver, JX_Int is_check_restarted );
JX_Int JX_COGMRESGetMaxIter( JX_Solver solver, JX_Int * max_iter );
JX_Int
JX_COGMRESSetPrecond( JX_Solver          solver,
                      JX_PtrToSolverFcn  precond,
                      JX_PtrToSolverFcn  precond_setup,
                      JX_Solver          precond_solver );
JX_Int JX_COGMRESGetPrecond( JX_Solver  solver, JX_Solver *precond_data_ptr );
JX_Int JX_COGMRESSetPrintLevel( JX_Solver solver, JX_Int level );
JX_Int JX_COGMRESGetPrintLevel( JX_Solver solver, JX_Int * level );
JX_Int JX_COGMRESSetLogging( JX_Solver solver, JX_Int level );
JX_Int JX_COGMRESGetLogging( JX_Solver solver, JX_Int * level );
JX_Int JX_COGMRESGetNumIterations( JX_Solver  solver, JX_Int *num_iterations );
JX_Int JX_COGMRESGetConverged( JX_Solver  solver, JX_Int *converged );
JX_Int JX_COGMRESGetFinalRelativeResidualNorm( JX_Solver  solver, JX_Real *norm );
JX_Int JX_COGMRESGetResidual( JX_Solver solver, void *residual );
JX_Int JX_COGMRESSetModifyPC( JX_Solver  solver, JX_Int (*modify_pc)(JX_Solver, JX_Int, JX_Real) );
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
);
void *jx_COGMRESCreate( jx_COGMRESFunctions *cogmres_functions );
JX_Int jx_COGMRESDestroy( void *cogmres_vdata );
JX_Int jx_COGMRESGetResidual( void *cogmres_vdata, void **residual );
JX_Int jx_COGMRESSetup( void *cogmres_vdata, void *A, void *b, void *x );
JX_Int jx_COGMRESSolve( void  *cogmres_vdata, void *preOperator, void  *A, void  *b, void  *x );
JX_Int jx_COGMRESSetKDim( void   *cogmres_vdata, JX_Int   k_dim );
JX_Int jx_COGMRESGetKDim( void   *cogmres_vdata, JX_Int * k_dim );
JX_Int jx_COGMRESSetUnroll( void   *cogmres_vdata, JX_Int   unroll );
JX_Int jx_COGMRESGetUnroll( void   *cogmres_vdata, JX_Int * unroll );
JX_Int jx_COGMRESSetCGS( void   *cogmres_vdata, JX_Int   cgs );
JX_Int jx_COGMRESGetCGS( void   *cogmres_vdata, JX_Int * cgs );
JX_Int jx_COGMRESSetTol( void   *cogmres_vdata, JX_Real  tol );
JX_Int jx_COGMRESGetTol( void   *cogmres_vdata, JX_Real  * tol );
JX_Int jx_COGMRESSetAbsoluteTol( void   *cogmres_vdata, JX_Real  a_tol );
JX_Int jx_COGMRESGetAbsoluteTol( void   *cogmres_vdata, JX_Real  * a_tol );
JX_Int jx_COGMRESSetConvergenceFactorTol( void   *cogmres_vdata, JX_Real  cf_tol );
JX_Int jx_COGMRESGetConvergenceFactorTol( void   *cogmres_vdata, JX_Real * cf_tol );
JX_Int jx_COGMRESSetMinIter( void *cogmres_vdata, JX_Int   min_iter );
JX_Int jx_COGMRESGetMinIter( void *cogmres_vdata, JX_Int * min_iter );
JX_Int jx_COGMRESSetMaxIter( void *cogmres_vdata, JX_Int   max_iter );
JX_Int jx_COGMRESSetIsCheckRestarted( void *cogmres_vdata, JX_Int is_check_restarted );
JX_Int jx_COGMRESGetMaxIter( void *cogmres_vdata, JX_Int * max_iter );
JX_Int jx_COGMRESSetRelChange( void *cogmres_vdata, JX_Int   rel_change );
JX_Int jx_COGMRESGetRelChange( void *cogmres_vdata, JX_Int * rel_change );
JX_Int jx_COGMRESSetSkipRealResidualCheck( void *cogmres_vdata, JX_Int skip_real_r_check );
JX_Int jx_COGMRESGetSkipRealResidualCheck( void *cogmres_vdata, JX_Int *skip_real_r_check );
JX_Int
jx_COGMRESSetPrecond( void  *cogmres_vdata,
        JX_Int  (*precond)(void*,void*,void*,void*),
        JX_Int  (*precond_setup)(void*,void*,void*,void*),
        void  *precond_data );
JX_Int jx_COGMRESGetPrecond( void *cogmres_vdata, JX_Solver *precond_data_ptr );
JX_Int jx_COGMRESSetPrintLevel( void *cogmres_vdata, JX_Int   level );
JX_Int jx_COGMRESGetPrintLevel( void *cogmres_vdata, JX_Int * level );
JX_Int jx_COGMRESSetLogging( void *cogmres_vdata, JX_Int   level );
JX_Int jx_COGMRESGetLogging( void *cogmres_vdata, JX_Int * level );
JX_Int jx_COGMRESGetNumIterations( void *cogmres_vdata, JX_Int  *num_iterations );
JX_Int jx_COGMRESGetConverged( void *cogmres_vdata, JX_Int  *converged );
JX_Int jx_COGMRESGetFinalRelativeResidualNorm( void   *cogmres_vdata, JX_Real *relative_residual_norm );
JX_Int jx_COGMRESSetModifyPC( void *cogmres_vdata, JX_Int (*modify_pc)(void *precond_data, JX_Int iteration, JX_Real rel_residual_norm) );

/* /csrc/krylov/pgmres.c */
JX_Int JX_ParCSRGMRESCreate( MPI_Comm comm, JX_Solver *solver );
JX_Int JX_ParCSRGMRESDestroy( JX_Solver solver );
JX_Int JX_GMRESSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );
JX_Int JX_GMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );
JX_Int JX_GMRESAdaptiveSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );
JX_Int JX_GMRESSetPreOperator( JX_Solver solver, JX_ParCSRMatrix preOperator ); /* peghoty 2011/09/04 */ 
JX_Int JX_GMRESSetKDim( JX_Solver solver, JX_Int k_dim );
JX_Int JX_GMRESGetKDim( JX_Solver solver, JX_Int *k_dim );
JX_Int JX_GMRESSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_GMRESGetTol( JX_Solver solver, JX_Real *tol );
JX_Int JX_GMRESSetAbsoluteTol( JX_Solver solver, JX_Real a_tol );
JX_Int JX_GMRESGetAbsoluteTol( JX_Solver solver, JX_Real *a_tol );
JX_Int JX_GMRESSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol );
JX_Int JX_GMRESGetConvergenceFactorTol( JX_Solver solver, JX_Real *cf_tol );
JX_Int JX_GMRESSetMinIter( JX_Solver solver, JX_Int min_iter );
JX_Int JX_GMRESGetMinIter( JX_Solver solver, JX_Int *min_iter );
JX_Int JX_GMRESSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_GMRESGetMaxIter( JX_Solver solver, JX_Int *max_iter );
JX_Int JX_GMRESSetStopCrit( JX_Solver solver, JX_Int stop_crit );
JX_Int JX_GMRESGetStopCrit( JX_Solver solver, JX_Int *stop_crit );
JX_Int JX_GMRESSetConvCriteria( JX_Solver solver, JX_Int conv_criteria ); /* peghoty 2010/06/23 */
JX_Int JX_GMRESSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold ); /* peghoty 2010/06/23 */
JX_Int JX_GMRESSetResDownZeroThreshold( JX_Solver solver, JX_Real resdown_0_threshold ); /* Yue Xiaoqiang 2014/03/24 */
JX_Int JX_GMRESSetConvFacThresholdTwo( JX_Solver solver, JX_Real convfac_threshold_2 ); /* Yue Xiaoqiang 2014/03/24 */
JX_Int JX_GMRESSetIsCheckRestarted( JX_Solver solver, JX_Int is_check_restarted ); /* peghoty 2011/11/08 */
JX_Int JX_GMRESSetRelChange( JX_Solver solver, JX_Int rel_change );
JX_Int JX_GMRESGetRelChange( JX_Solver solver, JX_Int *rel_change );
JX_Int JX_GMRESGetLastPrecondType( JX_Solver solver, JX_Int *last_precond_type ); /* Yue Xiaoqiang 2014/10/24 */
JX_Int 
JX_GMRESSetPrecond( JX_Solver          solver,
                    JX_PtrToSolverFcn  precond,
                    JX_PtrToSolverFcn  precond_setup,
                    JX_Solver          precond_solver );                         
JX_Int JX_GMRESGetPrecond( JX_Solver solver, JX_Solver *precond_data_ptr );
JX_Int JX_GMRESSetPrintLevel( JX_Solver solver, JX_Int level );
JX_Int JX_GMRESGetPrintLevel( JX_Solver solver, JX_Int *level );
JX_Int JX_GMRESGetCompAMGILUBcoIter( JX_Solver solver, JX_Int *amg_iter, JX_Int *ilu_iter, JX_Int *bco_iter );
JX_Int JX_GMRESSetLogging( JX_Solver solver, JX_Int level );
JX_Int JX_GMRESGetLogging( JX_Solver solver, JX_Int *level );
JX_Int JX_GMRESGetNumIterations( JX_Solver solver, JX_Int *num_iterations );
JX_Int JX_GMRESGetConverged( JX_Solver solver, JX_Int  *converged );
JX_Int JX_GMRESGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm );
JX_Int JX_GMRESGetResidual( JX_Solver solver, void **residual );
void *jx_ParCSRGMRESCreate( MPI_Comm comm );  /* peghoty 2011/10/26 */
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
);
void *jx_GMRESCreate( jx_GMRESFunctions *gmres_functions );
JX_Int jx_GMRESDestroy( void *gmres_vdata );
JX_Int jx_GMRESGetResidual( void *gmres_vdata, void **residual );
JX_Int jx_GMRESSetup( void *gmres_vdata, void *matA, void *vecB, void *vecX );
JX_Int jx_GMRESSetPreOperator( void *gmres_vdata, void *preOperator ); /* peghoty 2011/09/04 */
JX_Int jx_GMRESSetKDim( void *gmres_vdata, JX_Int k_dim );
JX_Int jx_GMRESGetKDim( void *gmres_vdata, JX_Int *k_dim );
JX_Int jx_GMRESSetTol( void *gmres_vdata, JX_Real tol );
JX_Int jx_GMRESGetTol( void *gmres_vdata, JX_Real *tol );
JX_Int jx_GMRESSetAbsoluteTol( void *gmres_vdata, JX_Real a_tol );
JX_Int jx_GMRESGetAbsoluteTol( void *gmres_vdata, JX_Real *a_tol );
JX_Int jx_GMRESSetConvergenceFactorTol( void *gmres_vdata, JX_Real cf_tol );
JX_Int jx_GMRESGetConvergenceFactorTol( void *gmres_vdata, JX_Real *cf_tol );
JX_Int jx_GMRESSetMinIter( void *gmres_vdata, JX_Int min_iter );
JX_Int jx_GMRESGetMinIter( void *gmres_vdata, JX_Int *min_iter );
JX_Int jx_GMRESSetMaxIter( void *gmres_vdata, JX_Int max_iter );
JX_Int jx_GMRESGetMaxIter( void *gmres_vdata, JX_Int *max_iter );
JX_Int jx_GMRESSetRelChange( void *gmres_vdata, JX_Int rel_change );
JX_Int jx_GMRESGetRelChange( void *gmres_vdata, JX_Int *rel_change );
JX_Int jx_GMRESGetLastPrecondType( void *gmres_vdata, JX_Int *last_precond_type ); /* Yue Xiaoqiang 2014/10/24 */
JX_Int jx_GMRESSetStopCrit( void *gmres_vdata, JX_Int stop_crit );
JX_Int jx_GMRESGetStopCrit( void *gmres_vdata, JX_Int *stop_crit );
JX_Int jx_GMRESSetConvCriteria( void *gmres_vdata, JX_Int conv_criteria ); /* peghoty 2010/06/23 */
JX_Int jx_GMRESSetConvFacThreshold( void *gmres_vdata, JX_Real convfac_threshold ); /* peghoty 2010/06/23 */
JX_Int jx_GMRESSetResDownZeroThreshold( void *gmres_vdata, JX_Real resdown_0_threshold ); /* Yue Xiaoqiang 2014/03/24 */
JX_Int jx_GMRESSetConvFacThresholdTwo( void *gmres_vdata, JX_Real convfac_threshold_2 ); /* Yue Xiaoqiang 2014/03/24 */
JX_Int jx_GMRESSetIsCheckRestarted( void *gmres_vdata, JX_Int is_check_restarted ); /* peghoty 2011/11/08 */
JX_Int jx_GMRESSetPrecond( void *gmres_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data );
JX_Int jx_GMRESGetPrecond( void *gmres_vdata, JX_Solver *precond_data_ptr );
JX_Int jx_GMRESSetPrintLevel( void *gmres_vdata, JX_Int level );
JX_Int jx_GMRESGetPrintLevel( void *gmres_vdata, JX_Int *level );
JX_Int jx_GMRESGetCompAMGILUBcoIter( void *gmres_vdata, JX_Int *amg_iter, JX_Int *ilu_iter, JX_Int *bco_iter );
JX_Int jx_GMRESSetLogging( void *gmres_vdata, JX_Int level );
JX_Int jx_GMRESGetLogging( void *gmres_vdata, JX_Int *level );
JX_Int jx_GMRESGetNumIterations( void *gmres_vdata, JX_Int *num_iterations );
JX_Int jx_GMRESGetConverged( void *gmres_vdata, JX_Int *converged );
JX_Int jx_GMRESGetFinalRelativeResidualNorm( void *gmres_vdata, JX_Real *relative_residual_norm );
JX_Int jx_GMRESSolve( void *gmres_vdata, void *preOperator, void *matA, void *vecB, void *vecX );
JX_Int jx_GMRESAdaptiveSolve( void *gmres_vdata, void *preOperator, void *matA, void *vecB, void *vecX );
JX_Int jx_GMRESSolveDefault( void *gmres_vdata, void *preOperator, void *matA, void *vecB, void *vecX );

/* /csrc/krylov/pflexgmres.c */
JX_Int JX_ParCSRFlexGMRESCreate( MPI_Comm comm, JX_Solver *solver );
JX_Int JX_ParCSRFlexGMRESDestroy( JX_Solver solver );
JX_Int JX_FlexGMRESSetup( JX_Solver solver, JX_Matrix A, JX_Vector b, JX_Vector x );
JX_Int JX_FlexGMRESSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix A, JX_Vector b, JX_Vector x );
JX_Int JX_FlexGMRESSetKDim( JX_Solver solver, JX_Int k_dim );
JX_Int JX_FlexGMRESSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_FlexGMRESSetAbsoluteTol( JX_Solver solver, JX_Real a_tol );
JX_Int JX_FlexGMRESSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol );
JX_Int JX_FlexGMRESSetMinIter( JX_Solver solver, JX_Int min_iter );
JX_Int JX_FlexGMRESSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_FlexGMRESSetPrecond( JX_Solver solver,
                               JX_PtrToSolverFcn precond,
                               JX_PtrToSolverFcn precond_setup,
                               JX_Solver precond_solver );
JX_Int JX_FlexGMRESSetPrintLevel( JX_Solver solver, JX_Int level );
JX_Int JX_FlexGMRESSetIsCheckRestarted( JX_Solver solver, JX_Int is_check_restarted );
JX_Int JX_FlexGMRESSetLogging( JX_Solver solver, JX_Int level );
JX_Int JX_FlexGMRESGetNumIterations( JX_Solver solver, JX_Int *num_iterations );
JX_Int JX_FlexGMRESGetConverged( JX_Solver solver, JX_Int *converged );
JX_Int JX_FlexGMRESGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm );
JX_Int JX_FlexGMRESGetResidual( JX_Solver solver, void *residual );
JX_Int JX_FlexGMRESSetModifyPC( JX_Solver  solver, JX_Int (*modify_pc)(JX_Solver, JX_Int, JX_Real) );
jx_FlexGMRESFunctions *jx_FlexGMRESFunctionsCreate(
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
   JX_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ) );
void *jx_FlexGMRESCreate( jx_FlexGMRESFunctions *fgmres_functions );
JX_Int jx_FlexGMRESDestroy( void *fgmres_vdata );
JX_Int jx_FlexGMRESGetResidual( void *fgmres_vdata, void **residual );
JX_Int jx_FlexGMRESSetup( void *fgmres_vdata, void *A, void *b, void *x );
JX_Int jx_FlexGMRESSolve( void *fgmres_vdata, void *preOperator, void *A, void *b, void *x );
JX_Int jx_FlexGMRESSetKDim( void *fgmres_vdata, JX_Int k_dim );
JX_Int jx_FlexGMRESSetTol( void *fgmres_vdata, JX_Real tol );
JX_Int jx_FlexGMRESSetAbsoluteTol( void *fgmres_vdata, JX_Real a_tol );
JX_Int jx_FlexGMRESSetConvergenceFactorTol( void *fgmres_vdata, JX_Real cf_tol );
JX_Int jx_FlexGMRESSetMinIter( void *fgmres_vdata, JX_Int min_iter );
JX_Int jx_FlexGMRESSetMaxIter( void *fgmres_vdata, JX_Int max_iter );
JX_Int jx_FlexGMRESSetStopCrit( void *fgmres_vdata, JX_Int stop_crit );
JX_Int jx_FlexGMRESSetPrecond( void *fgmres_vdata,
                               JX_Int (*precond)(void*,void*,void*,void*),
                               JX_Int (*precond_setup)(void*,void*),
                               void *precond_data );
JX_Int jx_FlexGMRESSetPrintLevel( void *fgmres_vdata, JX_Int level );
JX_Int jx_FlexGMRESSetIsCheckRestarted( void *fgmres_vdata, JX_Int is_check_restarted );
JX_Int jx_FlexGMRESSetLogging( void *fgmres_vdata, JX_Int level );
JX_Int jx_FlexGMRESGetNumIterations( void *fgmres_vdata, JX_Int *num_iterations );
JX_Int jx_FlexGMRESGetConverged( void *fgmres_vdata, JX_Int *converged );
JX_Int jx_FlexGMRESGetFinalRelativeResidualNorm( void *fgmres_vdata, JX_Real *relative_residual_norm );
JX_Int jx_FlexGMRESSetModifyPC( void *fgmres_vdata, JX_Int (*modify_pc)(void *precond_data, JX_Int iteration, JX_Real rel_residual_norm) );
JX_Int jx_FlexGMRESModifyPCDefault( void *precond_data, JX_Int iteration, JX_Real rel_residual_norm );

/* /csrc/krylov/pbicgstab.c */
JX_Int JX_ParCSRBiCGSTABCreate( MPI_Comm comm, JX_Solver *solver );
JX_Int JX_ParCSRBiCGSTABDestroy( JX_Solver solver );
JX_Int JX_BiCGSTABDestroy( JX_Solver solver );
JX_Int JX_BiCGSTABSetup( JX_Solver solver, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );
JX_Int JX_BiCGSTABSolve( JX_Solver solver, JX_Matrix preOperator, JX_Matrix matA, JX_Vector vecB, JX_Vector vecX );
JX_Int JX_BiCGSTABSetPreOperator( JX_Solver solver, JX_ParCSRMatrix preOperator ); /* peghoty 2011/09/04 */
JX_Int JX_BiCGSTABSetTol( JX_Solver solver, JX_Real tol );
JX_Int JX_BiCGSTABSetAbsoluteTol( JX_Solver solver, JX_Real a_tol );
JX_Int JX_BiCGSTABSetConvergenceFactorTol( JX_Solver solver, JX_Real cf_tol );
JX_Int JX_BiCGSTABSetMinIter( JX_Solver solver, JX_Int min_iter );
JX_Int JX_BiCGSTABSetMaxIter( JX_Solver solver, JX_Int max_iter );
JX_Int JX_BiCGSTABSetConvCriteria( JX_Solver solver, JX_Int conv_criteria ); /* peghoty 2010/03/20 */
JX_Int JX_BiCGSTABSetConvFacThreshold( JX_Solver solver, JX_Real convfac_threshold ); /* peghoty 2010/03/22 */
JX_Int
JX_BiCGSTABSetPrecond( JX_Solver         solver,
                       JX_PtrToSolverFcn precond,
                       JX_PtrToSolverFcn precond_setup,
                       JX_Solver         precond_solver );
JX_Int JX_BiCGSTABGetPrecond( JX_Solver solver, JX_Solver *precond_data_ptr );
JX_Int JX_BiCGSTABSetLogging( JX_Solver solver, JX_Int logging );
JX_Int JX_BiCGSTABSetPrintLevel( JX_Solver solver, JX_Int print_level );
JX_Int JX_BiCGSTABGetNumIterations( JX_Solver solver, JX_Int *num_iterations );
JX_Int JX_BiCGSTABGetFinalRelativeResidualNorm( JX_Solver solver, JX_Real *norm );
JX_Int JX_BiCGSTABGetResidual( JX_Solver solver, void **residual );
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
   JX_Int    (*PrecondSetup)  (  void *vdata, void *A ),
   JX_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
);
void *jx_BiCGSTABCreate( jx_BiCGSTABFunctions * bicgstab_functions );
JX_Int jx_BiCGSTABDestroy( void *bicgstab_vdata );
JX_Int jx_BiCGSTABSetup( void *bicgstab_vdata, void *matA, void *vecB, void *vecX );
JX_Int jx_BiCGSTABSetPreOperator( void *bicgstab_vdata, void *preOperator ); /* peghoty 2011/09/04 */
JX_Int jx_BiCGSTABSetTol( void *bicgstab_vdata, JX_Real tol );
JX_Int jx_BiCGSTABSetAbsoluteTol( void *bicgstab_vdata, JX_Real a_tol );
JX_Int jx_BiCGSTABSetConvergenceFactorTol( void *bicgstab_vdata, JX_Real cf_tol );
JX_Int jx_BiCGSTABSetMinIter( void *bicgstab_vdata, JX_Int min_iter );
JX_Int jx_BiCGSTABSetMaxIter( void *bicgstab_vdata, JX_Int max_iter );
JX_Int jx_BiCGSTABSetConvCriteria( void *bicgstab_vdata, JX_Int conv_criteria ); /* peghoty 2010/03/20 */
JX_Int jx_BiCGSTABSetConvFacThreshold( void *bicgstab_vdata, JX_Real convfac_threshold ); /* peghoty 2010/03/22 */
JX_Int jx_BiCGSTABSetPrecond( void *bicgstab_vdata, JX_Int (*precond)(), JX_Int (*precond_setup)(), void *precond_data );
JX_Int jx_BiCGSTABGetPrecond( void *bicgstab_vdata, JX_Solver *precond_data_ptr );
JX_Int jx_BiCGSTABSetLogging( void *bicgstab_vdata, JX_Int logging );   
JX_Int jx_BiCGSTABSetPrintLevel( void *bicgstab_vdata, JX_Int print_level );
JX_Int jx_BiCGSTABGetConverged( void *bicgstab_vdata, JX_Int *converged ); 
JX_Int jx_BiCGSTABGetNumIterations( void *bicgstab_vdata, JX_Int *num_iterations );
JX_Int jx_BiCGSTABGetFinalRelativeResidualNorm( void *bicgstab_vdata, JX_Real *relative_residual_norm );
JX_Int jx_BiCGSTABGetResidual( void *bicgstab_vdata, void **residual );
JX_Int jx_BiCGSTABSolve( void *bicgstab_vdata, void *preOperator, void *matA, void *vecB, void *vecX );

#endif
