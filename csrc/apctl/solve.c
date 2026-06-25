//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  solve.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

JX_Int jx_total_iter_index = 0;

/*!
 * \fn JX_Int JX_3tAPCTLSolve
 * \brief Solve phase of PCTL iteration.
 * \author peghoty
 * \date 2011/09/17
 */
JX_Int 
JX_3tAPCTLSolve( JX_Solver        solver,
                 JX_ParCSRMatrix  A,
                 JX_ParVector     b,
                 JX_ParVector     x )
{
   return( jx_3tAPCTLSolve( (void *) solver,
                            (jx_ParCSRMatrix *) A,
                            (jx_ParVector *) b,
                            (jx_ParVector *) x ) );
}

/*!
 * \fn JX_Int JX_3tAPCTLPrecond
 * \brief PCTL as preconditioner.
 * \author peghoty
 * \date 2011/09/17
 */
JX_Int 
JX_3tAPCTLPrecond( JX_Solver        solver,
                   JX_ParCSRMatrix  A,
                   JX_ParVector     b,
                   JX_ParVector     x  )
{
   return( jx_3tAPCTLPrecond( (void *) solver,
                              (jx_ParCSRMatrix *) A,
                              (jx_ParVector *) b,
                              (jx_ParVector *) x ) );
} 

JX_Int 
JX_3tAPCTLmgPrecond( JX_Solver        solver,
                     JX_ParCSRMatrix  A,
                     JX_ParVector     b,
                     JX_ParVector     x  )
{
   return( jx_3tAPCTLmgPrecond( (void *) solver,
                                (jx_ParCSRMatrix *) A,
                                (jx_ParVector *) b,
                                (jx_ParVector *) x ) );
}

JX_Int 
JX_3tABSC1mgPrecond( JX_Solver        solver,
                     JX_ParCSRMatrix  A,
                     JX_ParVector     b,
                     JX_ParVector     x  )
{
   return( jx_3tABSC1mgPrecond( (void *) solver,
                                (jx_ParCSRMatrix *) A,
                                (jx_ParVector *) b,
                                (jx_ParVector *) x ) );
}

JX_Int 
JX_3tABSC2mgPrecond( JX_Solver        solver,
                     JX_ParCSRMatrix  A,
                     JX_ParVector     b,
                     JX_ParVector     x  )
{
   return( jx_3tABSC2mgPrecond( (void *) solver,
                                (jx_ParCSRMatrix *) A,
                                (jx_ParVector *) b,
                                (jx_ParVector *) x ) );
}

/*!
 * \fn JX_Int jx_3tAPCTLOneIteration
 * \brief One Iteration of PCTL. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                        jx_ParCSRMatrix   *A,
                        jx_ParVector      *g,
                        jx_ParVector      *w )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   
   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jx_3tAPCTLOneIteration_sp(pre_3tapctl_data, A, g, w);
   }
   else if (nprocs > 1)
   {
      jx_3tAPCTLOneIteration_mp(pre_3tapctl_data, A, g, w);
   }
   
   return 0;
}

JX_Int
jx_3tAPCTLmgOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                          jx_ParCSRMatrix   *A,
                          jx_ParVector      *g,
                          jx_ParVector      *w )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jx_3tAPCTLmgOneIteration_mp(pre_3tapctl_data, A, g, w);
   }

   return 0;
}

JX_Int
jx_3tABSC1mgOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                          jx_ParCSRMatrix   *A,
                          jx_ParVector      *g,
                          jx_ParVector      *w )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jx_3tABSC1mgOneIteration_mp(pre_3tapctl_data, A, g, w);
   }

   return 0;
}

JX_Int
jx_3tABSC2mgOneIteration( jx_3tAPCTLData    *pre_3tapctl_data,
                          jx_ParCSRMatrix   *A,
                          jx_ParVector      *g,
                          jx_ParVector      *w )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);

   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs > 1)
   {
      jx_3tABSC2mgOneIteration_mp(pre_3tapctl_data, A, g, w);
   }

   return 0;
}

/*!
 * \fn JX_Int jx_3tAPCTLOneIteration_mp
 * \brief One Iteration of PCTL. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLOneIteration_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w )
{
   MPI_Comm  comm    = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm  comm_x  = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   
   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);   

   JX_Int  Need_CC        = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JX_Int  debug_flag     = jx_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JX_Int  groupid_x      = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JX_Int  print_level    = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   JX_Int  num_relax_pre  = jx_3tAPCTLDataNumRlxPre(pre_3tapctl_data);
   JX_Int  num_relax_post = jx_3tAPCTLDataNumRlxPost(pre_3tapctl_data);
   
   JX_Int  ACC_solver_id  = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);
   
   jx_ParAMGData   *ACC_amg_solver   = jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data); 
   jx_GMRESData    *ACC_gmres_solver = jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data);     

   jx_ParCSRMatrix *P   = jx_3tAPCTLDataP(pre_3tapctl_data);
   jx_ParCSRMatrix *ACC = jx_3tAPCTLDataACC(pre_3tapctl_data);
   
   jx_ParVector    *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector    *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector    *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);
   jx_ParVector    *WCC = jx_3tAPCTLDataWCC(pre_3tapctl_data);

   jx_ParVector    *RES = jx_3tAPCTLDataRES(pre_3tapctl_data);
   jx_ParVector    *GCC = jx_3tAPCTLDataGCC(pre_3tapctl_data);
   
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JX_Int i, myid, myid_x;
   JX_Int np;
   char MatFile[255];

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &np);
   jx_MPI_Comm_rank(comm_x, &myid_x);

  
   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================
   
   if (Need_CC == 1)
   {
      for (i = 0; i < num_relax_pre; i ++)
      {
         jx_3tAPCTLRelax_GSType_mp(pre_3tapctl_data, A, g, w);
         if ((debug_flag == 2) || (debug_flag == 3))
         {
            jx_sprintf(MatFile,"%s.%02d", "./3t.w", jx_total_iter_index);
            jx_ParVectorPrint(w, MatFile);
            jx_total_iter_index ++;
         }
      }
   }
   else // if (Need_CC == 0)
   {  

      if (blocksmooth_type == BLOCKSMOOTH_GS)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jx_3tAPCTLRelax_GSType_mp(pre_3tapctl_data, A, g, w);
         }
      }
      else // if (blocksmooth_type == BLOCKSMOOTH_BD)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jx_3tAPCTLRelax_BDType_mp(pre_3tapctl_data, A, g, w);
         }
      }
   }
   

   if (Need_CC == 1)
   {
      //=============================================================
      //  计算细网格层上的残量，并将其限制到粗网格层
      //=============================================================  
   
      jx_ParVectorCopy(g, RES);                      // RES = g
      jx_ParCSRMatrixMatvec(-1.0, A, w, 1.0, RES);   // RES = g - A*w
      jx_ParCSRMatrixMatvecT(1.0, P, RES, 0.0, GCC); // GCC = P^T*RES
 
 
      //================================================================
      // 求解粗网格方程 ACC*WCC = GCC
      //================================================================  
   
      jx_ParVectorSetConstantValues(WCC, 0.0); 
      
      if (ACC_solver_id == SOLVER_AMG)
      {  
         jx_PAMGSolve(ACC_amg_solver, ACC, GCC, WCC);

         if ((print_level == 2 || print_level == 3) && myid == 0)
         { 
            jx_printf(" APCTL-Solve == ACC AMG-Iter: %d\n", ACC_amg_solver->num_iterations);
         } 
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         jx_GMRESSolve(ACC_gmres_solver, ACC, ACC, GCC, WCC);
         if ((print_level == 2 || print_level == 3) && myid == 0)
         { 
            jx_printf(" APCTL-Solve == ACC AMGGMRES-Iter: %d\n", ACC_gmres_solver->num_iterations);
         } 
      }  
      
      if (test_subls_iter)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_amg_solver->num_iterations;
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_gmres_solver->num_iterations;
         }
      }
 
      //================================================================
      //  提升与校正: w = w + P*WCC
      //================================================================  
       
      jx_ParCSRMatrixMatvec(1.0, P, WCC, 1.0, w);  
   
  
      //=============================================================
      //  后（块）磨光 （磨光序为先电子，后离子和光子）
      //=============================================================  

      for (i = 0; i < num_relax_post; i ++)
      {
         jx_3tAPCTLRelax_GSType_mp(pre_3tapctl_data, A, g, w);
      } 
   }

   //=============================================================
   //  将辅助数组的数据部分指向空指针，避免 JX_Real free memory. 
   //============================================================= 
   
   if (groupid_x == 0)
   {
      WRR->local_vector->data = NULL;
   }
   else if (groupid_x == 1)
   {
      WEE->local_vector->data = NULL;
   }
   else if (groupid_x == 2)
   {
      WII->local_vector->data = NULL;
   } 
  
   return (0);
}

JX_Int
jx_3tAPCTLmgOneIteration_mp( jx_3tAPCTLData  *pre_3tapctl_data,
                             jx_ParCSRMatrix *A,
                             jx_ParVector    *g,
                             jx_ParVector    *w )
{
   MPI_Comm  comm    = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm  comm_x  = jx_3tAPCTLDataCommX(pre_3tapctl_data);

   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);

   JX_Int  Need_CC     = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JX_Int  debug_flag  = jx_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JX_Int  print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JX_Int  ACC_solver_id  = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   jx_ParAMGData *ACC_amg_solver   = jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data);
   jx_GMRESData  *ACC_gmres_solver = jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data);

   jx_ParCSRMatrix *P   = jx_3tAPCTLDataP(pre_3tapctl_data);
   jx_ParCSRMatrix *ACC = jx_3tAPCTLDataACC(pre_3tapctl_data);
 
   jx_ParVector *WCC = jx_3tAPCTLDataWCC(pre_3tapctl_data);

   jx_ParVector *RES = jx_3tAPCTLDataRES(pre_3tapctl_data);
   jx_ParVector *GCC = jx_3tAPCTLDataGCC(pre_3tapctl_data);

   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JX_Int myid, myid_x;
   char MatFile[255];

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================

   if (Need_CC == 1)
   {
      jx_3tAPCTLmgRelax_GSType_mp(pre_3tapctl_data, A, g, w);
      if ((debug_flag == 2) || (debug_flag == 3))
      {
         jx_sprintf(MatFile,"%s.%02d", "./mg.w", jx_total_iter_index);
         jx_ParVectorPrint(w, MatFile);
         jx_total_iter_index ++;
      }
   }
   else // if (Need_CC == 0)
   {
      if (blocksmooth_type == BLOCKSMOOTH_GS)
      {
         jx_3tAPCTLmgRelax_GSType_mp(pre_3tapctl_data, A, g, w);
      }
      else // if (blocksmooth_type == BLOCKSMOOTH_BD)
      {
         jx_3tAPCTLmgRelax_BDType_mp(pre_3tapctl_data, A, g, w);
      }
   }

   if (Need_CC == 1)
   {
      //=============================================================
      //  计算细网格层上的残量，并将其限制到粗网格层
      //=============================================================

      jx_ParVectorCopy(g, RES);                      // RES = g
      jx_ParCSRMatrixMatvec(-1.0, A, w, 1.0, RES);   // RES = g - A*w
      jx_ParCSRMatrixMatvecT(1.0, P, RES, 0.0, GCC); // GCC = P^T*RES
 
      //================================================================
      // 求解粗网格方程 ACC*WCC = GCC
      //================================================================

      jx_ParVectorSetConstantValues(WCC, 0.0);

      if (ACC_solver_id == SOLVER_AMG)
      {
         jx_PAMGSolve(ACC_amg_solver, ACC, GCC, WCC);

         if ((print_level == 2 || print_level == 3) && myid == 0)
         {
            jx_printf(" APCTL-Solve == ACC AMG-Iter: %d\n", ACC_amg_solver->num_iterations);
         }
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         jx_GMRESSolve(ACC_gmres_solver, ACC, ACC, GCC, WCC);
         if ((print_level == 2 || print_level == 3) && myid == 0)
         {
            jx_printf(" APCTL-Solve == ACC AMGGMRES-Iter: %d\n", ACC_gmres_solver->num_iterations);
         }
      }
 
      if (test_subls_iter)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_amg_solver->num_iterations;
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_gmres_solver->num_iterations;
         }
      }
 
      //================================================================
      //  提升与校正: w = w + P*WCC
      //================================================================

      jx_ParCSRMatrixMatvec(1.0, P, WCC, 1.0, w);
   }

   return (0);
}

JX_Int
jx_3tABSC1mgOneIteration_mp( jx_3tAPCTLData  *pre_3tapctl_data,
                             jx_ParCSRMatrix *A,
                             jx_ParVector    *g,
                             jx_ParVector    *w )
{
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jx_3tAPCTLDataCommY(pre_3tapctl_data);

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);
   JX_Int groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JX_Int Need_CC = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JX_Int debug_flag = jx_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JX_Int reset_zero = jx_3tAPCTLDataResetZero(pre_3tapctl_data);
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   JX_Int ARR_relax_maxit = jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JX_Int AEE_relax_maxit = jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JX_Int AII_relax_maxit = jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data);

   jx_ParAMGData *ARR_amg_solver = jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jx_ParAMGData *AEE_amg_solver = jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jx_ParAMGData *AII_amg_solver = jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);

   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector **VER = jx_3tAPCTLDataVER2(pre_3tapctl_data);
   jx_ParVector  *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector  *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector  *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParVector *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);

   jx_ParVector *JAC = jx_3tAPCTLDataJAC(pre_3tapctl_data);
   jx_ParVector *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);

   /* newly added, peghoty, 2012/02/16 */
   JX_Int use_fixedmode_R = jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JX_Int use_fixedmode_E = jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JX_Int use_fixedmode_I = jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);

   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   //JX_Int rootid_E = np_R * ng;

   JX_Real *tTEMP = jx_3tAPCTLDatatTEMP(pre_3tapctl_data);

   MPI_Status status;

#if 0
   MPI_Comm ycomm, yycomm;
   MPI_Group world_group, select_group;
   JX_Int *yrank = NULL;
   JX_Int gidx, ymyid;
   JX_Int ygroupid = MPI_UNDEFINED;
#endif

   /* local variables */
   JX_Int myid, myid_x, sweep, dest, gidx;
   char MatFile[255];

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================

   if (Need_CC == 1)
   {
      if (groupid_x == ng+1)
      {
         WII->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WII, 0.0);

            if (use_fixedmode_I)
            {
               jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
            }
            else
            {
               jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_I)
               {
                  jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
               }
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AII_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_I)
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
               }
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
         }

         jx_MPI_Send(WII->local_vector->data, WII->local_vector->size, JX_MPI_REAL, myid-np_R, 123, comm);
      }
      else if (groupid_x == ng)
      {
         jx_MPI_Recv(WII->local_vector->data, WII->local_vector->size, JX_MPI_REAL, myid+np_R, 123, comm, &status);

         WEE->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
         jx_ParVecZXY(RHS, -1.0, VEI, WII);

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WEE, 0.0);

            if (use_fixedmode_E)
            {
               jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
            }
            else
            {
               jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AEE_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_E)
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
               }
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
         }

#if 0
         dest = myid - rootid_E;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            jx_MPI_Send(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, dest, dest*321, comm);
            dest += np_R;
         }
#endif
      }

   jx_MPI_Barrier(comm); //  各进程(组)同步
      jx_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, ng, comm_y); // yue: invalid for comm_I

#if 0
      jx_MPI_Comm_group(comm, &world_group);
      sweep = (ng + 1) * np_R;
      yrank = jx_CTAlloc(JX_Int, sweep);
      for (gidx = 0; gidx < sweep; gidx ++) yrank[gidx] = gidx;
      jx_MPI_Group_incl(world_group, sweep, yrank, &select_group);
      jx_MPI_Comm_create(comm, select_group, &ycomm);

      jx_MPI_Comm_rank(ycomm, &ymyid);
      ygroupid = ymyid % np_R;
      jx_MPI_Comm_split(ycomm, ygroupid, ymyid, &yycomm);
      jx_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, ng, yycomm);

      if (select_group != MPI_GROUP_NULL) jx_MPI_Group_free(&select_group);
      if (world_group != MPI_GROUP_NULL) jx_MPI_Group_free(&world_group);
      if (ycomm != MPI_COMM_NULL) jx_MPI_Comm_free(&ycomm);
      if (yycomm != MPI_COMM_NULL) jx_MPI_Comm_free(&yycomm);
      jx_TFree(yrank);
#endif

      if (groupid_x < ng)
      {
         //jx_MPI_Recv(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, myid, myid*321, comm, &status);

         WRR->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
         jx_ParVecZXY(RHS, -1.0, VRE, WEE);

         if (ARR_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_R)
            {
               jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_R)
               {
                  jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
               }
               else
               {
                  jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
               }
            }
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (ARR_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_R)
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
               }
            }
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            }
         }

         //jx_MPI_Send(WRR->local_vector->data, WRR->local_vector->size, JX_MPI_REAL, myid, myid*321, comm);
      }

   jx_MPI_Barrier(comm); //  各进程(组)同步
      jx_MPI_Gather(WRR->local_vector->data, WRR->local_vector->size, JX_MPI_REAL,
                                     tTEMP, WRR->local_vector->size, JX_MPI_REAL, ng, comm_y); // yue: invalid from comm_E and comm_I

      if (groupid_x == ng)
      {
         jx_ParVectorSetConstantValues(RHS, 0.0);
         dest = 0;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
            {
               RHS->local_vector->data[sweep] += (VER[gidx]->local_vector->data[sweep] * tTEMP[dest++]);
            }
         }
#if 0
         dest = myid - rootid_E;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            jx_MPI_Recv(WRR->local_vector->data, WRR->local_vector->size, JX_MPI_REAL, dest, dest*321, comm, &status);
            for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
            {
               RHS->local_vector->data[sweep] += (VER[gidx]->local_vector->data[sweep] * WRR->local_vector->data[sweep]);
            }
            dest += np_R;
         }
#endif

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_E)
            {
               jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jx_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jx_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == 2AEE JAC-Iter: %d\n", AEE_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AEE_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_E)
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
               }
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
         }

         jx_ParVectorAxpy(-1.0, WRR, WEE);

         jx_MPI_Send(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, myid+np_R, 113, comm);
      }
      else if (groupid_x == ng+1)
      {
         jx_MPI_Recv(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, myid-np_R, 113, comm, &status);

         jx_ParVecMul(WEE, VIE, RHS);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_I)
            {
               jx_PAMGPrecond(AII_amg_solver, AII, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(AII_amg_solver, AII, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_I)
               {
                  jx_printf( " APCTL-Solve == 2AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jx_printf( " APCTL-Solve == 2AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
               }
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == 2AII JAC-Iter: %d\n", AII_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AII_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_I)
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
               }
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
         }

         jx_ParVectorAxpy(-1.0, WRR, WII);
      }

      if ((debug_flag == 2) || (debug_flag == 3))
      {
         jx_sprintf(MatFile,"%s.%02d", "./mg.w", jx_total_iter_index);
         jx_ParVectorPrint(w, MatFile);
         jx_total_iter_index ++;
      }
   }
   else
   {
      if (groupid_x == ng)
      {
         WEE->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WEE, 0.0);

            if (use_fixedmode_E)
            {
               jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
            }
            else
            {
               jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jx_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jx_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            { 
               jx_printf(" BD-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AEE_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_E)
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
               }
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
         }
      }
      else if (groupid_x < ng)
      {
         WRR->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (ARR_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_R)
            {
               jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_R)
               {
                  jx_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
               }
               else
               {
                  jx_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
               }
            }
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" BD-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (ARR_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_R)
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
               }
            }
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            }
         }
      }
      else if (groupid_x == ng+1)
      {
         WII->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WII, 0.0);

            if (use_fixedmode_I)
            {
               jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
            }
            else
            {
               jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_I)
               {
                  jx_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jx_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
               }
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC);
            }
            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" BD-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AII_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_I)
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
               }
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
         }
      }
   }

   return (0);
}

JX_Int
jx_3tABSC2mgOneIteration_mp( jx_3tAPCTLData  *pre_3tapctl_data,
                             jx_ParCSRMatrix *A,
                             jx_ParVector    *g,
                             jx_ParVector    *w )
{
   MPI_Comm comm = jx_3tAPCTLDataComm(pre_3tapctl_data);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jx_3tAPCTLDataCommY(pre_3tapctl_data);

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);
   JX_Int groupid_x = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);

   JX_Int Need_CC = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JX_Int debug_flag = jx_3tAPCTLDataDebugFlag(pre_3tapctl_data);
   JX_Int reset_zero = jx_3tAPCTLDataResetZero(pre_3tapctl_data);
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);

   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);

   JX_Int ARR_relax_maxit = jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JX_Int AEE_relax_maxit = jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JX_Int AII_relax_maxit = jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data);

   jx_ParAMGData *ARR_amg_solver = jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jx_ParAMGData *AEE_amg_solver = jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jx_ParAMGData *AII_amg_solver = jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);

   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data);
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector **VER = jx_3tAPCTLDataVER2(pre_3tapctl_data);
   jx_ParVector  *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector  *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector  *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParVector *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);

   jx_ParVector *JAC = jx_3tAPCTLDataJAC(pre_3tapctl_data);
   jx_ParVector *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);

   /* newly added, peghoty, 2012/02/16 */
   JX_Int use_fixedmode_R = jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JX_Int use_fixedmode_E = jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JX_Int use_fixedmode_I = jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);

   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   JX_Real *tTEMP = jx_3tAPCTLDatatTEMP(pre_3tapctl_data);

   /* local variables */
   JX_Int myid, myid_x, sweep, dest, gidx;
   char MatFile[255];

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================

   if (Need_CC == 1)
   {
      if (groupid_x == ng)
      {
         WEE->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WEE, 0.0);

            if (use_fixedmode_E)
            {
               jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
            }
            else
            {
               jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AEE_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_E)
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
               }
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
         }
      }

   jx_MPI_Barrier(comm); //  各进程(组)同步
      jx_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, ng, comm_y);

      if (groupid_x < ng)
      {
         WRR->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
         jx_ParVecZXY(RHS, -1.0, VRE, WEE);

         if (ARR_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_R)
            {
               jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_R)
               {
                  jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
               }
               else
               {
                  jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
               }
            }
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (ARR_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_R)
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
               }
            }
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            }
         }
      }
      else if (groupid_x == ng+1)
      {
         WRR->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
         jx_ParVecZXY(RHS, -1.0, VIE, WEE);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_I)
            {
               jx_PAMGPrecond(AII_amg_solver, AII, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(AII_amg_solver, AII, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_I)
               {
                  jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
               }
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AII_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_I)
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
               }
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
         }
      }

   jx_MPI_Barrier(comm); //  各进程(组)同步
      jx_MPI_Gather(WRR->local_vector->data, WRR->local_vector->size, JX_MPI_REAL,
                                     tTEMP, WRR->local_vector->size, JX_MPI_REAL, ng, comm_y); // yue: invalid from comm_E

      if (groupid_x == ng)
      {
         jx_ParVectorSetConstantValues(RHS, 0.0);
         dest = 0;
         for (gidx = 0; gidx < ng; gidx ++)
         {
            for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
            {
               RHS->local_vector->data[sweep] += (VER[gidx]->local_vector->data[sweep] * tTEMP[dest++]);
            }
         }
         dest += WRR->local_vector->size; // skip comm_E
         for (sweep = 0; sweep < WRR->local_vector->size; sweep ++)
         {
            RHS->local_vector->data[sweep] += (VEI->local_vector->data[sweep] * tTEMP[dest++]);
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_E)
            {
               jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jx_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jx_printf(" APCTL-Solve == 2AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" APCTL-Solve == 2AEE JAC-Iter: %d\n", AEE_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AEE_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_E)
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
               }
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
         }

         jx_ParVectorAxpy(-1.0, WRR, WEE);
      }

      if ((debug_flag == 2) || (debug_flag == 3))
      {
         jx_sprintf(MatFile,"%s.%02d", "./mg.w", jx_total_iter_index);
         jx_ParVectorPrint(w, MatFile);
         jx_total_iter_index ++;
      }
   }
   else
   {
      if (groupid_x == ng)
      {
         WEE->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (AEE_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WEE, 0.0);

            if (use_fixedmode_E)
            {
               jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
            }
            else
            {
               jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_E)
               {
                  jx_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
               }
               else
               {
                  jx_printf(" BD-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
               }
            }
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            { 
               jx_printf(" BD-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AEE_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_E)
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
               }
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            }
         }
      }
      else if (groupid_x < ng)
      {
         WRR->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (ARR_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WRR, 0.0);

            if (use_fixedmode_R)
            {
               jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
            }
            else
            {
               jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_R)
               {
                  jx_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
               }
               else
               {
                  jx_printf(" BD-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
               }
            }
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" BD-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (ARR_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_R)
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
               }
            }
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            }
         }
      }
      else if (groupid_x == ng+1)
      {
         WII->local_vector->data = w->local_vector->data;
         jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

         if (AII_relax_type == RELAX_AMG)
         {
            if (reset_zero) jx_ParVectorSetConstantValues(WII, 0.0);

            if (use_fixedmode_I)
            {
               jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
            }
            else
            {
               jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
            }

            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               if (use_fixedmode_I)
               {
                  jx_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
               }
               else
               {
                  jx_printf(" BD-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
               }
            }
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
            {
               jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC);
            }
            if ((print_level == 2 || print_level == 3) && myid_x == 0)
            {
               jx_printf(" BD-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
            }
         }

         if (test_subls_iter)
         {
            if (AII_relax_type == RELAX_AMG)
            {
               if (use_fixedmode_I)
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
               }
               else
               {
                  jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
               }
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            }
         }
      }
   }

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLOneIteration_sp
 * \brief One Iteration of PCTL(single-processor case). 
 * \author peghoty 
 * \date 2011/09/27
 */
JX_Int
jx_3tAPCTLOneIteration_sp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w )
{
   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);   

   JX_Int  Need_CC        = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);
   JX_Int  print_level    = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   JX_Int  num_relax_pre  = jx_3tAPCTLDataNumRlxPre(pre_3tapctl_data);
   JX_Int  num_relax_post = jx_3tAPCTLDataNumRlxPost(pre_3tapctl_data);
   
   JX_Int  ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);

   jx_ParAMGData   *ACC_amg_solver   = jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data); 
   jx_GMRESData    *ACC_gmres_solver = jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data);    

   jx_ParCSRMatrix *ACC = jx_3tAPCTLDataACC(pre_3tapctl_data);
   jx_ParVector    *WCC = jx_3tAPCTLDataWCC(pre_3tapctl_data);
   jx_ParVector    *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);
   
   jx_ParVector    *RES = jx_3tAPCTLDataRES(pre_3tapctl_data);
   
   jx_ParVector    *PRR = jx_3tAPCTLDataPRR(pre_3tapctl_data);
   jx_ParVector    *PII = jx_3tAPCTLDataPII(pre_3tapctl_data);
    
   JX_Real *rhs_data = NULL;   
   JX_Real *prr_data = NULL;
   JX_Real *pii_data = NULL;
   JX_Real *res_data = NULL;
   JX_Real *wcc_data = NULL;
   JX_Real *w_data   = NULL;

   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
     
   /* local variables */
   JX_Int i;
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Int nplusn = 2*n;   


   //=============================================================
   //  前（块）磨光 （磨光序为先电子，后离子和光子）
   //=============================================================
   
   if (Need_CC == 1)
   {
      for (i = 0; i < num_relax_pre; i ++)
      {
         jx_3tAPCTLRelax_GSType_sp(pre_3tapctl_data, A, g, w);
      }
   }
   else // if (Need_CC == 0)
   {  
      if (blocksmooth_type == BLOCKSMOOTH_GS)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jx_3tAPCTLRelax_GSType_sp(pre_3tapctl_data, A, g, w);
         }
      }
      else // if (blocksmooth_type == BLOCKSMOOTH_BD)
      {
         for (i = 0; i < num_relax_pre; i ++)
         {
            jx_3tAPCTLRelax_BDType_sp(pre_3tapctl_data, A, g, w);
         }
      }
   }
   
   if (Need_CC == 1)
   {
      //=============================================================
      //  计算细网格层上的残量
      //=============================================================  
   
      jx_ParVectorCopy(g, RES);                      // RES = g
      jx_ParCSRMatrixMatvec(-1.0, A, w, 1.0, RES);   // RES = g - A*w
      

      //=============================================================
      //  将残量限制到粗网格层
      //=============================================================  

      rhs_data = jx_VectorData(jx_ParVectorLocalVector(RHS));   
      prr_data = jx_VectorData(jx_ParVectorLocalVector(PRR));
      pii_data = jx_VectorData(jx_ParVectorLocalVector(PII)); 
      res_data = jx_VectorData(jx_ParVectorLocalVector(RES));     
      for (i = 0; i < n; i ++)
      {
         rhs_data[i] = res_data[i]*prr_data[i] + res_data[i+n] + res_data[i+nplusn]*pii_data[i];
      }      
 

      //================================================================
      // 求解粗网格方程 ACC*WCC = GCC
      //================================================================  
   
      jx_ParVectorSetConstantValues(WCC, 0.0);
      
      if (ACC_solver_id == SOLVER_AMG)
      {   
         jx_PAMGSolve(ACC_amg_solver, ACC, RHS, WCC);

         if (print_level == 2 || print_level == 3)
         { 
            jx_printf(" APCTL-Solve == ACC AMG-Iter: %d\n", ACC_amg_solver->num_iterations);
         }
      } 
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         jx_GMRESSolve(ACC_gmres_solver, ACC, ACC, RHS, WCC);
         if (print_level == 2 || print_level == 3)
         { 
            jx_printf(" APCTL-Solve == ACC AMGGMRES-Iter: %d\n", ACC_gmres_solver->num_iterations);
         } 
      }
               
      if (test_subls_iter)
      {
         if (ACC_solver_id == SOLVER_AMG)
         {
            jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_amg_solver->num_iterations;
         }
         else if (ACC_solver_id == SOLVER_AMGGMRES)
         {
            jx_3tAPCTLDataNumIterAcPrecond(pre_3tapctl_data) += ACC_gmres_solver->num_iterations;
         }
      }


      //================================================================
      // 提升与校正: w = w + P*WCC
      //================================================================  
 
      wcc_data = jx_VectorData(jx_ParVectorLocalVector(WCC));
      w_data   = jx_VectorData(jx_ParVectorLocalVector(w));
      for (i = 0; i < n; i ++) 
      {
         w_data[i] += prr_data[i]*wcc_data[i];
         w_data[i+n] += wcc_data[i];
         w_data[i+nplusn] += pii_data[i]*wcc_data[i]; 
      } 
      

      //=============================================================
      //  后（块）磨光 （磨光序为先电子，后离子和光子）
      //=============================================================  

      for (i = 0; i < num_relax_post; i ++)
      {
         jx_3tAPCTLRelax_GSType_sp(pre_3tapctl_data, A, g, w);
      }
   }
   
   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLPrecond
 * \brief Solution phase of PCTL as preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                   jx_ParCSRMatrix   *A,
                   jx_ParVector      *f,
                   jx_ParVector      *u )
{   
   JX_Int  max_iter = jx_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JX_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jx_3tAPCTLOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

JX_Int
jx_3tAPCTLmgPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                     jx_ParCSRMatrix   *A,
                     jx_ParVector      *f,
                     jx_ParVector      *u )
{   
   JX_Int  max_iter = jx_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JX_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jx_3tAPCTLmgOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

JX_Int
jx_3tABSC1mgPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                     jx_ParCSRMatrix   *A,
                     jx_ParVector      *f,
                     jx_ParVector      *u )
{   
   JX_Int  max_iter = jx_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JX_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jx_3tABSC1mgOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

JX_Int
jx_3tABSC2mgPrecond( jx_3tAPCTLData    *pre_3tapctl_data,
                     jx_ParCSRMatrix   *A,
                     jx_ParVector      *f,
                     jx_ParVector      *u )
{   
   JX_Int  max_iter = jx_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JX_Int  num_iter = 0;

   while (num_iter < max_iter)
   {
      /* One iteration of APCTL as preconditioner */
      jx_3tABSC2mgOneIteration(pre_3tapctl_data, A, f, u);
      num_iter ++;
   }

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLSolve
 * \brief SOLVE phase of PCTL iteration.
 * \author peghoty 
 * \date 2011/09/17
 */ 
JX_Int
jx_3tAPCTLSolve( jx_3tAPCTLData    *pre_3tapctl_data,
                 jx_ParCSRMatrix   *A,
                 jx_ParVector      *f,
                 jx_ParVector      *u )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
     
   /* members of pre_3tapctl_data */ 
   JX_Real        tol         = jx_3tAPCTLDataTol(pre_3tapctl_data);
   JX_Int           max_iter    = jx_3tAPCTLDataMaxIter(pre_3tapctl_data);
   JX_Int           print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);
   jx_ParVector *R           = jx_3tAPCTLDataRES(pre_3tapctl_data);

   /* local variables */
   JX_Int      num_iterations   = 0;
   JX_Real   conv_factor      = 0.0;
   JX_Real   res_nrm          = 0.0;
   JX_Real   res_nrm_init     = 0.0;
   JX_Real   relative_res_nrm = 0.0;
   JX_Real   rhs_nrm          = 0.0; 
   JX_Real   res_nrm_old      = 0.0;
   
   JX_Int Solve_err_flag = 0;
   JX_Int myid; 

   jx_MPI_Comm_rank(comm, &myid);

  /*-----------------------------------------------------------------------
   *  Compute rhs vector and initial fine-grid residual
   *----------------------------------------------------------------------*/
   rhs_nrm = jx_ParVectorNorm2(f);            // rhs_nrm = ||f||_2
   jx_ParVectorCopy(f, R);                    // R = f
   jx_ParCSRMatrixMatvec(-1.0, A, u, 1.0, R); // R = f - Au
   res_nrm = jx_ParVectorNorm2(R);            // res_nrm = ||f - A*u||_2
   res_nrm_init = res_nrm; 

   if (rhs_nrm)
      relative_res_nrm = res_nrm_init / rhs_nrm;
   else
      relative_res_nrm = res_nrm_init;

   if (print_level && myid == 0)
   {  
      jx_printf("\n");   
      jx_printf(" ----------------------------------------------------\n");     
      jx_printf("    iters    residual      factor      rel_res_nrm   \n");
      jx_printf(" ----------------------------------------------------\n");
      jx_printf("      0    %e                %e\n", res_nrm_init,relative_res_nrm);
   }

  /*---------------------------------------------------------------*
   *                 Main Iterative loop                           *
   *---------------------------------------------------------------*/

   while ( relative_res_nrm >= tol && num_iterations < max_iter )
   {
     /*-----------------------------------------------------------
      * One iteration of PCTL
      *---------------------------------------------------------*/
      jx_3tAPCTLOneIteration(pre_3tapctl_data, A, f, u);

     /*-----------------------------------------------------------
      * Save the res_nrm
      *---------------------------------------------------------*/
      res_nrm_old = res_nrm;

     /*--------------------------------------------------------------------------
      * Compute fine-grid residual and residual norm
      *------------------------------------------------------------------------*/
      jx_ParVectorCopy(f, R);                      // R:= f           
      jx_ParCSRMatrixMatvec(-1.0, A, u, 1.0, R);   // R:= R - Au ( = f - Au )
      res_nrm = jx_ParVectorNorm2(R);              // res_nrm = ||R||_2  

     /*-----------------------------------------------------------
      * Compute convergence factor
      *---------------------------------------------------------*/
      if (res_nrm_old)
         conv_factor = res_nrm / res_nrm_old;
      else
         conv_factor = res_nrm;   

     /*---------------------------------------------
      * Compute relative_res_nrm
      *-------------------------------------------*/ 
      if (rhs_nrm)
         relative_res_nrm = res_nrm / rhs_nrm;
      else
         relative_res_nrm = res_nrm;

      num_iterations ++;

      if (print_level && myid == 0)
      { 
         jx_printf("    %3d    %e   %f     %e \n",num_iterations,res_nrm,conv_factor,relative_res_nrm);
      }
   } // end while

   if (num_iterations == max_iter && relative_res_nrm >= tol) Solve_err_flag = 1;

   jx_3tAPCTLSetNumIterations(pre_3tapctl_data, num_iterations);
   jx_3tAPCTLSetLastRelNrm(pre_3tapctl_data, relative_res_nrm);

   /* Compute average convergence factor */
   if (res_nrm_init)
   {
      conv_factor = pow( (res_nrm / res_nrm_init), (1.0 / ((JX_Real) num_iterations)) );
   }
   jx_3tAPCTLSetAveConvFactor(pre_3tapctl_data, conv_factor);

   if (print_level && myid == 0)
   {
      if (Solve_err_flag == 1)
      {
         jx_printf("\n\n==============================================");
         jx_printf("\n NOTE: Convergence tolerance was not achieved\n");
         jx_printf("       within the allowed %d iterations\n",max_iter);
         jx_printf("==============================================\n");
      }
      jx_printf("\n >>> \033[34mAverage Convergence Factor\033[00m = %f\n\n", conv_factor);
   }

   return (Solve_err_flag);
}
