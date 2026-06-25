/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 *  par_mllssove.c
 *  
 *  Created by peghoty  2010/11/22
 *  Xiangtan University
 *  peghoty@163.com
 *
 */

#include "jx_multils.h"

/*!
 * \fn fsls_SolverParam *fsls_SolverParamCreate
 * \brief Create and initialize a fsls_SolverParam object.
 * \author peghoty
 * \date 2010/11/23 
 */
fsls_SolverParam *
fsls_SolverParamCreate()
{
   fsls_SolverParam *solver = fsls_CTAlloc(fsls_SolverParam, 1);
   fsls_SolverParamSolverID(solver)   = 0;
   fsls_SolverParamTolerance(solver)  = 1.0e-8;
   fsls_SolverParamMaxIter(solver)    = 500;
   fsls_SolverParamMinIter(solver)    = 0;
   fsls_SolverParamTwoNorm(solver)    = 0;
   fsls_SolverParamKDim(solver)       = 10;
   fsls_SolverParamPrintLevel(solver) = 3;
   
   return solver;
}  

/*!
 * \fn JX_Int fsls_SolverParamSet**
 * \author peghoty
 * \date 2010/11/23 
 */
JX_Int
fsls_SolverParamSetSolverID( void *vdata, JX_Int solver_id )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamSolverID(solver) = solver_id;
   return ierr;
}

JX_Int
fsls_SolverParamSetTolerance( void *vdata, JX_Real tol )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamTolerance(solver) = tol;
   return ierr;
}

JX_Int
fsls_SolverParamSetMaxIter( void *vdata, JX_Int max_iter )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamMaxIter(solver) = max_iter;
   return ierr;
}

JX_Int
fsls_SolverParamSetMinIter( void *vdata, JX_Int min_iter )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamMinIter(solver) = min_iter;
   return ierr;
}

JX_Int
fsls_SolverParamSetTwoNorm( void *vdata, JX_Int two_norm )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamTwoNorm(solver) = two_norm;
   return ierr;
}

JX_Int
fsls_SolverParamSetKDim( void *vdata, JX_Int k_dim )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamKDim(solver) = k_dim;
   return ierr;
}

JX_Int
fsls_SolverParamSetPrintLevel( void *vdata, JX_Int print_level )
{
   JX_Int ierr = 0;
   fsls_SolverParam *solver = vdata;
   fsls_SolverParamPrintLevel(solver) = print_level;
   return ierr;
}

/*!
 * \fn JX_Int fsls_ParallelSolve
 * \brief Solve the multi linear systems parallelly
 * \param *ls_data pointer to a fsls_ParLSData object
 * \param *solver solver parameters
 * \note You can select one of the following solvers by 'solver_id' 
 *
 *      solver_id = 0 : AMG
 *      solver_id = 11: CG
 *      solver_id = 12: AMG-CG
 *      solver_id = 13: DS-CG
 *      solver_id = 21: GMRES
 *      solver_id = 22: AMG-GMRES
 *      solver_id = 23: DS-GMRES
 *      solver_id = 31: BICGSTAB
 *      solver_id = 32: AMG-BICGSTAB
 *      solver_id = 33: DS-BICGSTAB
 *  
 * \author peghoty
 * \date 2010/11/22 
 */  
JX_Int
fsls_ParallelSolve( fsls_ParLSData *ls_data, fsls_SolverParam *solver )
{
   MPI_Comm  comm = fsls_ParLSDataComm(ls_data);
   
   /* parameters for iterative solver */
   JX_Int    solver_id   = fsls_SolverParamSolverID(solver);    // solver id
   JX_Real tol         = fsls_SolverParamTolerance(solver);   // the tolerance   
   JX_Int    max_iter    = fsls_SolverParamMaxIter(solver);     // the maximal number of iterations
   JX_Int    min_iter    = fsls_SolverParamMinIter(solver);     // the minimal number of iterations  
   JX_Int    two_norm    = fsls_SolverParamTwoNorm(solver);     // L^2 norm(1) or preconditioner-norm(0)? used for PCG
   JX_Int    k_dim       = fsls_SolverParamKDim(solver);        // the dimension of krylov subspace, used for PGMRES 
   JX_Int    print_level = fsls_SolverParamPrintLevel(solver);  // how much information to be printed

   fsls_CSRPCGData       *pcg_solver       = NULL;
   fsls_CSRPGMRESData    *pgmres_solver    = NULL;
   fsls_CSRPBICGSTABData *pbicgstab_solver = NULL;
   fsls_AMGData          *amg_solver       = NULL;
   FSLS_Solver            precond          = NULL; 

   JX_Int i, myid;

   fsls_CSRMatrix *A = NULL;
   fsls_Vector    *b = NULL;
   fsls_Vector    *x = NULL;     
      
   /* parameters for AMG solver */
   JX_Int          max_levels;        // maximal number of grid levels
   JX_Real       strong_threshold;  // the strong threshold which defines the Strenth Matrix
   JX_Real       max_row_sum;       // parameter to modify the definition of strength for diagonal 
                                   // dominant portions of the matrix.
   JX_Int          cycle_type;        // cycle type
   JX_Int          corrective_type;   // corrective type or iterative type when cycling    
   JX_Int          interp_type;       // interpolation type 
   JX_Int          coarse_threshold;  // number of dofs allowed on the coarsest level
   JX_Int          S_mode;            // how to define the strength matrix S
   JX_Int          coarsen_type;      // coarsening type
   JX_Real       relax_weight;      // relaxation weight
   JX_Int          relax_type;        // relaxation type
   JX_Int          relax_order;       // relaxation order
   JX_Int          relax_sweeps;      // relaxation sweeps 
   JX_Int          relax_sym;         // symmetricly relax or not? 
   JX_Int          poly_degree;       // degree of polynomial for Polynomial relaxation   
   JX_Real       A_trunc_factor;    // truncation parameter for coarse operater
   JX_Real       P_trunc_factor;    // truncation parameter for interpolation operater
   JX_Int          A_max_elmts;       // maximal number of entries in each row of coarse operater
   JX_Int          P_max_elmts;       // maximal number of entries in each row of interpolation operater
   
   /* output information */
   JX_Real       operat_cmplxty  = 0.0;
   JX_Real       grid_cmplxty    = 0.0; 
   JX_Real       last_rel_nrm    = 0.0;
   JX_Real       conv_factor     = 0.0;
   JX_Int          solver_iter     = 0;  

   jx_MPI_Comm_rank(comm, &myid); 
   
  /*------------------------------------------------------
   * Set all parameters for the iterative solvers
   *-----------------------------------------------------*/

   /* parameters for AMG solver */
   max_levels       = 25;
   strong_threshold = 0.25;
   max_row_sum      = 1.0;
   cycle_type       = 1;
   corrective_type  = 0;
   coarsen_type     = RS_MODIFIED;  // RS_ORIGINAL; // RS_MODIFIED; // CLJP; 
   interp_type      = CLASSIC;      // CLASSIC; // DIRECT; // ENERGYMIN
   coarse_threshold = 100;
   S_mode           = 0;   
   relax_weight     = 1.0;
   relax_type       = Smooth_GS; // Smooth_WJACOBI; // Smooth_GS; // Smooth_SOR; // Smooth_KACZMARZ; // Smooth_POLY
   relax_order      = CPFIRST;   // DESCEND; // ASCEND; // CPFIRST; // FPFIRST;
   relax_sweeps     = 1;
   relax_sym        = 1;
   poly_degree      = 3;
   A_trunc_factor   = 0.0;
   P_trunc_factor   = 0.0;
   A_max_elmts      = 0;
   P_max_elmts      = 0;   
   
  /*------------------------------------------------------
   *  Sovle the linear system
   *-----------------------------------------------------*/
   
   if (solver_id == 0)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mAMG\033[00m \n", myid);
      
      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         /* some parameter-setting for setup and solve phase */
         amg_solver = fsls_AMGInitialize();
         fsls_AMGSetMaxIter(amg_solver, max_iter);
         fsls_AMGSetMaxLevels(amg_solver, max_levels);
         fsls_AMGSetTol(amg_solver, tol);
         fsls_AMGSetStrongThreshold(amg_solver, strong_threshold);
         fsls_AMGSetMaxRowSum(amg_solver, max_row_sum);
         fsls_AMGSetCycleType(amg_solver, cycle_type);
         fsls_AMGSetCorrectiveType(amg_solver, corrective_type);
         fsls_AMGSetInterpType(amg_solver, interp_type);
         fsls_AMGSetCoarseThreshold(amg_solver, coarse_threshold);
         fsls_AMGSetSMode(amg_solver, S_mode);   
         fsls_AMGSetCoarsenType(amg_solver, coarsen_type);
         fsls_AMGSetRelaxType(amg_solver, relax_type);
         fsls_AMGSetRelaxOrder(amg_solver, relax_order);   
         fsls_AMGSetRelaxWeight(amg_solver, relax_weight);
         fsls_AMGSetRelaxSweeps(amg_solver, relax_sweeps);
         fsls_AMGSetRelaxSym(amg_solver, relax_sym);
         fsls_AMGSetPolyDegree(amg_solver, poly_degree);
         fsls_AMGSetATruncFactor(amg_solver, A_trunc_factor);
         fsls_AMGSetPTruncFactor(amg_solver, P_trunc_factor);
         fsls_AMGSetAMaxElmts(amg_solver, A_max_elmts);
         fsls_AMGSetPMaxElmts(amg_solver, P_max_elmts);      
         fsls_AMGSetPrintLevel(amg_solver, print_level);

         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];
               
         fsls_AMGSetup(amg_solver, A, b, x);
         fsls_AMGSolve(amg_solver, A, b, x);
   
         if (print_level == 0)
         {
            fsls_AMGComplx(amg_solver, &grid_cmplxty, &operat_cmplxty);
            fsls_AMGGetCycleCount(amg_solver, &solver_iter);
            fsls_AMGGetLastRelNrm(amg_solver, &last_rel_nrm);
            fsls_AMGGetAveConvFactor(amg_solver, &conv_factor);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of AMG ITER         :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm         :\033[00m %.3le", last_rel_nrm);
            jx_printf("\n >> \033[31mAverage Convergence Factor :\033[00m %.3lf", conv_factor);
            jx_printf("\n >> \033[31mGrid Complexity            :\033[00m %.3lf", grid_cmplxty);
            jx_printf("\n >> \033[31mOperater Complexity        :\033[00m %.3lf\n\n", operat_cmplxty);
         }
         else if (print_level == 1)
         {
            fsls_AMGGetCycleCount(amg_solver, &solver_iter);
            fsls_AMGGetLastRelNrm(amg_solver, &last_rel_nrm);
            fsls_AMGGetAveConvFactor(amg_solver, &conv_factor);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of AMG ITER         :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm         :\033[00m %.3le", last_rel_nrm);
            jx_printf("\n >> \033[31mAverage Convergence Factor :\033[00m %.3lf\n\n", conv_factor);
         }
         
         /* destroy a amg solver */
         fsls_AMGFinalize(amg_solver);
      }
   }  
   else if (solver_id == 11)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mCG\033[00m \n", myid);
      
      pcg_solver = fsls_CSRPCGCreate();
      fsls_CSRPCGSetTol(pcg_solver, tol);
      fsls_CSRPCGSetMaxIter(pcg_solver, max_iter);
      fsls_CSRPCGSetTwoNorm(pcg_solver, 1);
      fsls_CSRPCGSetPrintLevel(pcg_solver, print_level);
      
      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i]; 
             
         fsls_CSRPCGSetup(pcg_solver, A, b, x);
         fsls_CSRPCGSolve(pcg_solver, A, b, x);
      
         if (print_level == 0)
         {
            fsls_CSRPCGGetNumIterations(pcg_solver, &solver_iter);       
            fsls_CSRPCGGetFinalRelativeResidualNorm(pcg_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of CG ITER  :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm :\033[00m %le\n\n", last_rel_nrm);
         }
      }      

      fsls_CSRPCGDestroy(pcg_solver);
   }   
   else if (solver_id == 12)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mAMG-CG\033[00m \n", myid);
   
      pcg_solver = fsls_CSRPCGCreate();
      fsls_CSRPCGSetTol(pcg_solver, tol);
      fsls_CSRPCGSetMaxIter(pcg_solver, max_iter);
      fsls_CSRPCGSetTwoNorm(pcg_solver, two_norm);
      fsls_CSRPCGSetPrintLevel(pcg_solver, print_level);

      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {     
         /* some parameter-setting for setup and solve phase */ 
         precond = fsls_AMGInitialize();
         fsls_AMGSetMaxIter(precond, 1);
         fsls_AMGSetMaxLevels(precond, max_levels);
         fsls_AMGSetStrongThreshold(precond, strong_threshold);
         fsls_AMGSetMaxRowSum(precond, max_row_sum);
         fsls_AMGSetCycleType(precond, cycle_type);
         fsls_AMGSetCorrectiveType(precond, corrective_type);
         fsls_AMGSetInterpType(precond, interp_type);
         fsls_AMGSetCoarseThreshold(precond, coarse_threshold);
         fsls_AMGSetSMode(precond, S_mode);      
         fsls_AMGSetCoarsenType(precond, coarsen_type);
         fsls_AMGSetRelaxType(precond, relax_type);
         fsls_AMGSetRelaxOrder(precond, relax_order);   
         fsls_AMGSetRelaxWeight(precond, relax_weight);
         fsls_AMGSetRelaxSweeps(precond, relax_sweeps);
         fsls_AMGSetRelaxSym(precond, relax_sym);
         fsls_AMGSetPolyDegree(precond, poly_degree);      
         fsls_AMGSetATruncFactor(precond, A_trunc_factor);
         fsls_AMGSetPTruncFactor(precond, P_trunc_factor);
         fsls_AMGSetAMaxElmts(precond, A_max_elmts);
         fsls_AMGSetPMaxElmts(precond, P_max_elmts);           
         fsls_AMGSetPrintLevel(precond, 0);
      
         fsls_CSRPCGSetPrecond(pcg_solver, fsls_AMGPrecond, fsls_AMGSetup, precond);

         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i]; 
         
         fsls_CSRPCGSetup(pcg_solver, A, b, x); 
         fsls_CSRPCGSolve(pcg_solver, A, b, x);   
      
         if (print_level == 0)
         {
            fsls_CSRPCGGetNumIterations(pcg_solver, &solver_iter);       
            fsls_CSRPCGGetFinalRelativeResidualNorm(pcg_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of AMG-CG ITER :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm    :\033[00m %le\n\n", last_rel_nrm);  
         }  
         
         fsls_AMGFinalize(precond);
      }        

      fsls_CSRPCGDestroy(pcg_solver);
   }    
   else if (solver_id == 13)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mDS-CG\033[00m \n", myid);
   
      pcg_solver = fsls_CSRPCGCreate();
      fsls_CSRPCGSetTol(pcg_solver, tol);
      fsls_CSRPCGSetMaxIter(pcg_solver, max_iter);
      fsls_CSRPCGSetTwoNorm(pcg_solver, two_norm);
      fsls_CSRPCGSetPrintLevel(pcg_solver, print_level);

      fsls_CSRPCGSetPrecond(pcg_solver, fsls_PreDSSolve, fsls_PreDSSetup, precond);
      
      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i]; 
               
         fsls_CSRPCGSetup(pcg_solver, A, b, x);
         fsls_CSRPCGSolve(pcg_solver, A, b, x);

         if (print_level == 0)
         {
            fsls_CSRPCGGetNumIterations(pcg_solver, &solver_iter);       
            fsls_CSRPCGGetFinalRelativeResidualNorm(pcg_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of DS-CG ITER :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm   :\033[00m %le\n\n", last_rel_nrm);
         }    
      }        

      fsls_CSRPCGDestroy(pcg_solver);
   }      
   else if (solver_id == 21)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mGMRES\033[00m \n", myid);
      
      pgmres_solver = fsls_CSRPGMRESCreate();
      fsls_CSRPGMRESSetKDim(pgmres_solver, k_dim);
      fsls_CSRPGMRESSetTol(pgmres_solver, tol);
      fsls_CSRPGMRESSetMaxIter(pgmres_solver, max_iter);
      fsls_CSRPGMRESSetMinIter(pgmres_solver, min_iter);
      fsls_CSRPGMRESSetPrintLevel(pgmres_solver, print_level);

      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];
          
         fsls_CSRPGMRESSetup(pgmres_solver, A, b, x);
         fsls_CSRPGMRESSolve(pgmres_solver, A, b, x);    
      
         if (print_level == 0)
         {
            fsls_CSRPGMRESGetNumIterations(pgmres_solver, &solver_iter);       
            fsls_CSRPGMRESGetFinalRelativeResidualNorm(pgmres_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of GMRES(%d) ITER :\033[00m %d", pgmres_solver->k_dim, solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm      :\033[00m %le\n\n", last_rel_nrm);
         }
      }   

      fsls_CSRPGMRESDestroy(pgmres_solver);
   } 
   else if (solver_id == 22)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mAMG-GMRES\033[00m \n", myid);
      
      pgmres_solver = fsls_CSRPGMRESCreate();
      fsls_CSRPGMRESSetKDim(pgmres_solver, k_dim);
      fsls_CSRPGMRESSetTol(pgmres_solver, tol);
      fsls_CSRPGMRESSetMaxIter(pgmres_solver, max_iter);
      fsls_CSRPGMRESSetMinIter(pgmres_solver, min_iter);
      fsls_CSRPGMRESSetPrintLevel(pgmres_solver, print_level);

      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         /* some parameter-setting for setup and solve phase */
         precond = fsls_AMGInitialize();
         fsls_AMGSetMaxIter(precond, 1);
         fsls_AMGSetMaxLevels(precond, max_levels);
         fsls_AMGSetStrongThreshold(precond, strong_threshold);
         fsls_AMGSetMaxRowSum(precond, max_row_sum);
         fsls_AMGSetCycleType(precond, cycle_type);
         fsls_AMGSetCorrectiveType(precond, corrective_type);
         fsls_AMGSetInterpType(precond, interp_type);
         fsls_AMGSetCoarseThreshold(precond, coarse_threshold);
         fsls_AMGSetSMode(precond, S_mode);    
         fsls_AMGSetCoarsenType(precond, coarsen_type);
         fsls_AMGSetRelaxType(precond, relax_type);
         fsls_AMGSetRelaxOrder(precond, relax_order);   
         fsls_AMGSetRelaxWeight(precond, relax_weight);
         fsls_AMGSetRelaxSweeps(precond, relax_sweeps);
         fsls_AMGSetRelaxSym(precond, relax_sym);
         fsls_AMGSetPolyDegree(precond, poly_degree);        
         fsls_AMGSetATruncFactor(precond, A_trunc_factor);
         fsls_AMGSetPTruncFactor(precond, P_trunc_factor);
         fsls_AMGSetAMaxElmts(precond, A_max_elmts);
         fsls_AMGSetPMaxElmts(precond, P_max_elmts);         
         fsls_AMGSetPrintLevel(precond, 0);
      
         fsls_CSRPGMRESSetPrecond(pgmres_solver, fsls_AMGPrecond, fsls_AMGSetup, precond);   
     
         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];
               
         fsls_CSRPGMRESSetup(pgmres_solver, A, b, x);
      
         fsls_CSRPGMRESSolve(pgmres_solver, A, b, x);
      
         if (print_level == 0)
         {
            fsls_CSRPGMRESGetNumIterations(pgmres_solver, &solver_iter);       
            fsls_CSRPGMRESGetFinalRelativeResidualNorm(pgmres_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of AMG-GMRES(%d) ITER :\033[00m %d", pgmres_solver->k_dim, solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm          :\033[00m %le\n\n", last_rel_nrm);
         } 
         
         fsls_AMGFinalize(precond); 
      }

      fsls_CSRPGMRESDestroy(pgmres_solver);
   }     
   else if (solver_id == 23)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mDS-GMRES\033[00m \n", myid);

      pgmres_solver = fsls_CSRPGMRESCreate();
      fsls_CSRPGMRESSetKDim(pgmres_solver, k_dim);
      fsls_CSRPGMRESSetTol(pgmres_solver, tol);
      fsls_CSRPGMRESSetMaxIter(pgmres_solver, max_iter);
      fsls_CSRPGMRESSetMinIter(pgmres_solver, min_iter);
      fsls_CSRPGMRESSetPrintLevel(pgmres_solver, print_level);

      fsls_CSRPGMRESSetPrecond(pgmres_solver, fsls_PreDSSolve, fsls_PreDSSetup, precond);   
     
      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];
                
         fsls_CSRPGMRESSetup(pgmres_solver, A, b, x);
      
         fsls_CSRPGMRESSolve(pgmres_solver, A, b, x);   
      
         if (print_level == 0)
         {
            fsls_CSRPGMRESGetNumIterations(pgmres_solver, &solver_iter);       
            fsls_CSRPGMRESGetFinalRelativeResidualNorm(pgmres_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of DS-GMRES(%d) ITER :\033[00m %d", pgmres_solver->k_dim, solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm          :\033[00m %le\n\n", last_rel_nrm);
         }
      }
      
      fsls_CSRPGMRESDestroy(pgmres_solver);
   }     
   else if (solver_id == 31)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mBiCGSTAB\033[00m \n", myid);

      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         pbicgstab_solver = fsls_CSRPBICGSTABCreate();
         fsls_CSRPBICGSTABSetTol(pbicgstab_solver, tol);
         fsls_CSRPBICGSTABSetMaxIter(pbicgstab_solver, max_iter);
         fsls_CSRPBICGSTABSetPrintLevel(pbicgstab_solver, print_level);

         A = fsls_ParLSDataAArray(ls_data)[i];
         //fsls_CSRMatrixReorder(A);
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];

#if 0         
         if (myid == 0)
         {
            jx_printf(" myid = %d local_num_ls = %d \n", myid, fsls_ParLSDataLocalNumLS(ls_data));
            fsls_CSRMatrixPrint(fsls_ParLSDataAArray(ls_data)[0], "./mat_0_new");
            fsls_SeqVectorPrint(fsls_ParLSDataBArray(ls_data)[0], "./rhs_0_new");
            fsls_SeqVectorPrint(fsls_ParLSDataXArray(ls_data)[0], "./app_0_new");
         }
         else if (myid == 1)
         {
            jx_printf(" myid = %d local_num_ls = %d \n", myid, fsls_ParLSDataLocalNumLS(ls_data));
            fsls_CSRMatrixPrint(fsls_ParLSDataAArray(ls_data)[0], "./mat_1_new");
            fsls_SeqVectorPrint(fsls_ParLSDataBArray(ls_data)[0], "./rhs_1_new");
            fsls_SeqVectorPrint(fsls_ParLSDataXArray(ls_data)[0], "./app_1_new");
         }
#endif
                
         fsls_CSRPBICGSTABSetup(pbicgstab_solver, A, b, x);
         fsls_CSRPBICGSTABSolve(pbicgstab_solver, A, b, x);  

         if (print_level == 0)
         {
            fsls_CSRPBICGSTABGetNumIterations(pbicgstab_solver, &solver_iter);       
            fsls_CSRPBICGSTABGetFinalRelativeResidualNorm(pbicgstab_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of BICGSTAB ITER :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm      :\033[00m %le\n\n", last_rel_nrm);
         }
         
         fsls_CSRPBICGSTABDestroy(pbicgstab_solver);
      }    
   }      
   else if (solver_id == 32)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mAMG-BiCGSTAB\033[00m \n", myid);

      pbicgstab_solver = fsls_CSRPBICGSTABCreate();
      fsls_CSRPBICGSTABSetTol(pbicgstab_solver, tol);
      fsls_CSRPBICGSTABSetMaxIter(pbicgstab_solver, max_iter);
      fsls_CSRPBICGSTABSetPrintLevel(pbicgstab_solver, print_level);

      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         /* some parameter-setting for setup and solve phase */
         precond = fsls_AMGInitialize();
         fsls_AMGSetMaxIter(precond, 1);
         fsls_AMGSetMaxLevels(precond, max_levels);
         fsls_AMGSetStrongThreshold(precond, strong_threshold);
         fsls_AMGSetMaxRowSum(precond, max_row_sum);      
         fsls_AMGSetCycleType(precond, cycle_type);
         fsls_AMGSetCorrectiveType(precond, corrective_type);
         fsls_AMGSetInterpType(precond, interp_type);
         fsls_AMGSetCoarseThreshold(precond, coarse_threshold);
         fsls_AMGSetSMode(precond, S_mode);    
         fsls_AMGSetCoarsenType(precond, coarsen_type);
         fsls_AMGSetRelaxType(precond, relax_type);
         fsls_AMGSetRelaxOrder(precond, relax_order);   
         fsls_AMGSetRelaxWeight(precond, relax_weight);
         fsls_AMGSetRelaxSweeps(precond, relax_sweeps);
         fsls_AMGSetRelaxSym(precond, relax_sym);
         fsls_AMGSetPolyDegree(precond, poly_degree);        
         fsls_AMGSetATruncFactor(precond, A_trunc_factor);
         fsls_AMGSetPTruncFactor(precond, P_trunc_factor);
         fsls_AMGSetAMaxElmts(precond, A_max_elmts);
         fsls_AMGSetPMaxElmts(precond, P_max_elmts);      
         fsls_AMGSetPrintLevel(precond, 0);
      
         fsls_CSRPBICGSTABSetPrecond(pbicgstab_solver, fsls_AMGPrecond, fsls_AMGSetup, precond);   

         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];
         
         fsls_CSRPBICGSTABSetup(pbicgstab_solver, A, b, x);
         fsls_CSRPBICGSTABSolve(pbicgstab_solver, A, b, x);  

         if (print_level == 0)
         {
            fsls_CSRPBICGSTABGetNumIterations(pbicgstab_solver, &solver_iter);       
            fsls_CSRPBICGSTABGetFinalRelativeResidualNorm(pbicgstab_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of AMG-BICGSTAB ITER :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm          :\033[00m %le\n\n", last_rel_nrm);
         } 
         
         fsls_AMGFinalize(precond);
      }  
      
      fsls_CSRPBICGSTABDestroy(pbicgstab_solver);
   }   
   else if (solver_id == 33)
   {
      if (print_level > 0)
      jx_printf(" myid = \033[31m%d\033[00m solver: \033[31mDS-BiCGSTAB\033[00m \n", myid);

      pbicgstab_solver = fsls_CSRPBICGSTABCreate();
      fsls_CSRPBICGSTABSetTol(pbicgstab_solver, tol);
      fsls_CSRPBICGSTABSetMaxIter(pbicgstab_solver, max_iter);
      fsls_CSRPBICGSTABSetPrintLevel(pbicgstab_solver, print_level);
      
      fsls_CSRPBICGSTABSetPrecond(pbicgstab_solver, fsls_PreDSSolve, fsls_PreDSSetup, precond);   

      for (i = 0; i < fsls_ParLSDataLocalNumLS(ls_data); i ++)
      {
         A = fsls_ParLSDataAArray(ls_data)[i];
         b = fsls_ParLSDataBArray(ls_data)[i];
         x = fsls_ParLSDataXArray(ls_data)[i];
         
         fsls_CSRPBICGSTABSetup(pbicgstab_solver, A, b, x);
         fsls_CSRPBICGSTABSolve(pbicgstab_solver, A, b, x);

         if (print_level == 0)
         {
            fsls_CSRPBICGSTABGetNumIterations(pbicgstab_solver, &solver_iter);       
            fsls_CSRPBICGSTABGetFinalRelativeResidualNorm(pbicgstab_solver, &last_rel_nrm);
            jx_printf("\n >> The \033[31m%d\033[00m-th linear system in proc \033[31m%d\033[00m", i, myid);
            jx_printf("\n >> \033[31mNumber of DS-BICGSTAB ITER :\033[00m %d", solver_iter);
            jx_printf("\n >> \033[31mLast Relative Norm         :\033[00m %le\n\n", last_rel_nrm);
         }  
      } 

      fsls_CSRPBICGSTABDestroy(pbicgstab_solver);
   }
      
   return 0;
}
