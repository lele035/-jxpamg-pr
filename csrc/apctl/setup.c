//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  setup.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn JX_Int JX_3tAPCTLSetup
 * \brief Setup phase of PCTL iteration or preconditioner.
 * \author peghoty
 * \date 2011/09/17
 */
JX_Int 
JX_3tAPCTLSetup( JX_Solver solver, JX_ParCSRMatrix A )
{
   return( jx_3tAPCTLSetup( (void *) solver, (jx_ParCSRMatrix *) A ) );
}

/*!
 * \fn JX_Int jx_3tAPCTLSetup
 * \brief Setup phase of PCTL Iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLSetup( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(A);
   
   JX_Int nprocs;
   jx_MPI_Comm_size(comm, &nprocs);
   
   if (nprocs == 1)
   {
      jx_3tAPCTLSetup_sp(pre_3tapctl_data, A);
   }
   else if (nprocs > 1)
   { 
      jx_3tAPCTLSetup_mp(pre_3tapctl_data, A); 
   }
   
   return 0;
}

/*!
 * \fn JX_Int jx_3tAPCTLSetup_mp
 * \brief Setup phase of PCTL Iteration or preconditioner(for multi-processor case). 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLSetup_mp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{   
   JX_Real starttime = 0.0, endtime = 0.0;

   MPI_Comm comm = jx_ParCSRMatrixComm(A);
        
   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);

   JX_Real theta_wc_E     = jx_3tAPCTLDataThetaWCE(pre_3tapctl_data);
   JX_Real threshold_wc_E = jx_3tAPCTLDataThresholdWCE(pre_3tapctl_data);
   JX_Real theta_dd_R     = jx_3tAPCTLDataThetaDDR(pre_3tapctl_data);
   JX_Real theta_dd_E     = jx_3tAPCTLDataThetaDDE(pre_3tapctl_data);
   JX_Real theta_dd_I     = jx_3tAPCTLDataThetaDDI(pre_3tapctl_data);
   JX_Real threshold_dd_R = jx_3tAPCTLDataThresholdDDR(pre_3tapctl_data);
   JX_Real threshold_dd_E = jx_3tAPCTLDataThresholdDDE(pre_3tapctl_data);
   JX_Real threshold_dd_I = jx_3tAPCTLDataThresholdDDI(pre_3tapctl_data);  

   JX_Int IS_DD_R;
   JX_Int IS_DD_E;    
   JX_Int IS_DD_I;
   
   /* max_iter for each subblock */
   JX_Int ARR_interp_maxit = jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JX_Int AII_interp_maxit = jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);    
   JX_Int ACC_relax_maxit  = jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data);
   
   /* 需要根据参数 Need_CC 来待定 */ 
   JX_Int ARR_relax_maxit;
   JX_Int AEE_relax_maxit;
   JX_Int AII_relax_maxit;
      
   /* tolerace for each subblock */
   JX_Real  ARR_interp_tol = jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JX_Real  AII_interp_tol = jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JX_Real  ARR_relax_tol  = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real  AEE_relax_tol  = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real  AII_relax_tol  = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JX_Real  ACC_relax_tol  = jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   /* solver type for interpolation-building of PRR.  peghoty,2011/10/29 */
   JX_Int ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   
   /* solver type for PCTL iteration.  peghoty,2011/10/29 */
   JX_Int ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);
   
   /* restart parameters for GMRES solver. peghoty,2011/10/29 */
   JX_Int ARR_kdim = jx_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JX_Int ACC_kdim = jx_3tAPCTLDataACCKDim(pre_3tapctl_data);  

   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);  
   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
  
   jx_ParAMGData   *ARR_amg_solver = NULL;
   jx_ParAMGData   *AEE_amg_solver = NULL;
   jx_ParAMGData   *AII_amg_solver = NULL;
   jx_ParAMGData   *ACC_amg_solver = NULL;

   /* gmres solver data for ARR and ACC. peghoty,2011/10/29 */
   jx_GMRESData    *ARR_gmres_solver = NULL;
   jx_GMRESData    *ACC_gmres_solver = NULL; 
   
   jx_ParCSRMatrix *ARR = NULL;
   jx_ParCSRMatrix *AEE = NULL;
   jx_ParCSRMatrix *AII = NULL;
   
   jx_ParVector    *VRE = NULL;
   jx_ParVector    *VER = NULL;
   jx_ParVector    *VEI = NULL;
   jx_ParVector    *VIE = NULL;      

   jx_ParCSRMatrix *P   = NULL;  
   jx_ParCSRMatrix *ACC = NULL; 
     
   jx_ParVector    *WRR = NULL;
   jx_ParVector    *WEE = NULL;
   jx_ParVector    *WII = NULL;
   jx_ParVector    *WCC = NULL;

   jx_ParVector    *GCC = NULL;
 
   jx_ParVector    *RES = NULL;
   jx_ParVector    *RHS = NULL;
   jx_ParVector    *JAC = NULL;
   
   jx_ParCSRMatrix *ARR_all = NULL;
   jx_ParCSRMatrix *AEE_all = NULL;
   jx_ParCSRMatrix *AII_all = NULL;  
   jx_ParVector    *VRE_all = NULL;
   jx_ParVector    *VER_all = NULL;
   jx_ParVector    *VEI_all = NULL;
   jx_ParVector    *VIE_all = NULL;   

   JX_Int Need_CC; 
   
   JX_Int ARR_relax_type = RELAX_AMG;
   JX_Int AEE_relax_type = RELAX_AMG;
   JX_Int AII_relax_type = RELAX_AMG;      

   jx_ParVector *PRR = NULL;
   jx_ParVector *PII = NULL; 

   JX_Int *row_starts = NULL;
   JX_Int *col_starts = NULL; 
   
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
      
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Int rootid_R, rootid_E, rootid_I;
   JX_Real temp_adrress = 0.0;

   JX_Int myid, nprocs;
   JX_Int np_R = jx_3tAPCTLDataNpR(pre_3tapctl_data);
   JX_Int np_E = jx_3tAPCTLDataNpE(pre_3tapctl_data); 
   JX_Int np_I = jx_3tAPCTLDataNpI(pre_3tapctl_data);

   JX_Int group_num_x = 3;
   JX_Int group_num_y;
   JX_Int groupid_x = MPI_UNDEFINED;
   JX_Int groupid_y = MPI_UNDEFINED; 
   MPI_Comm comm_x, comm_y;    

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (myid < np_R)
   {
      groupid_x = 0;
   }
   else if (myid < np_R + np_E)
   {
      groupid_x = 1;
   }
   else if (myid < np_R + np_E + np_I)
   {
      groupid_x = 2;
   }
   rootid_R = 0; 
   rootid_E = np_R;
   rootid_I = np_R + np_E;    
   jx_MPI_Comm_split(comm, groupid_x, myid, &comm_x); 
   
   group_num_y = nprocs / group_num_x; 
   groupid_y   = myid % group_num_y;
   jx_MPI_Comm_split(comm, groupid_y, myid, &comm_y);

   if (print_level == 1 || print_level == 3)
   {       
      starttime = jx_MPI_Wtime();
   }
      
   if (groupid_x == 1)
   {  
      jx_MatVecGroupE(comm_x, A, &AEE, &VER, &VEI); 
      Need_CC = jx_3tAPCTLWeakCouplingE(theta_wc_E, threshold_wc_E, AEE, VER, VEI); 
   }
   jx_MPI_Bcast(&Need_CC, 1, JX_MPI_INT, rootid_E, comm); /* 将标志变量广播给所有进程 */
   jx_3tAPCTLSetNeedCC(pre_3tapctl_data, Need_CC);
   if (print_level && myid == 0) jx_printf(" >> Need_CC = %d\n", Need_CC);
      
  
   if (groupid_x == 0)
   {  
      if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
      {
         jx_MatVecGroupR(comm_x, A, &ARR, &VRE);
      }
      else
      {
         jx_MatGroupR(comm_x, A, &ARR);
      }  
   }
   else if (groupid_x == 1)
   {
      if ( Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD )
      {
         jx_ParVectorDestroy(VER);
         jx_ParVectorDestroy(VEI);
      }     
   }
   else if (groupid_x == 2)
   {
      if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
      {
         jx_MatVecGroupI(comm_x, A, &AII, &VIE);
      }
      else
      {
         jx_MatGroupI(comm_x, A, &AII);
      } 
   }
   
   if (Need_CC == 1)
   {
      jx_MatVecGroup2All( comm, groupid_x,  
                          ARR, AEE, AII, VRE, VER, VEI, VIE,
                          &ARR_all, &AEE_all, &AII_all, 
                          &VRE_all, &VER_all, &VEI_all, &VIE_all );    
   }
   
   if (print_level == 1 || print_level == 3)
   {                       
      endtime = jx_MPI_Wtime();
      jx_GetWallTime(comm, "APCTL == Get Seven Blocks", starttime, endtime, 0, 3);
   }  

   
   if (groupid_x == 0)
   {  
      IS_DD_R = jx_3tAPCTLDDCheck(theta_dd_R, threshold_dd_R, ARR);
      jx_3tAPCTLSetISDDR(pre_3tapctl_data, IS_DD_R);
      if (print_level && myid == rootid_R) jx_printf(" >> IS_DD_R = %d\n", IS_DD_R);
      if (IS_DD_R)
      {      
         ARR_relax_type = RELAX_WJACOBI;        
      }
      jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data) = ARR_relax_type;     
   }
   else if (groupid_x == 1)
   {
      IS_DD_E = jx_3tAPCTLDDCheck(theta_dd_E, threshold_dd_E, AEE);
      jx_3tAPCTLSetISDDE(pre_3tapctl_data, IS_DD_E);
      if (print_level && myid == rootid_E) jx_printf(" >> IS_DD_E = %d\n", IS_DD_E);
      if (IS_DD_E)
      {      
         AEE_relax_type = RELAX_WJACOBI;        
      }
      jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data) = AEE_relax_type;        
   }
   else if (groupid_x == 2)
   { 
      IS_DD_I = jx_3tAPCTLDDCheck(theta_dd_I, threshold_dd_I, AII);
      jx_3tAPCTLSetISDDI(pre_3tapctl_data, IS_DD_I);
      if (print_level && myid == rootid_I) jx_printf(" >> IS_DD_I = %d\n", IS_DD_I);
      if (IS_DD_I)
      {      
         AII_relax_type = RELAX_WJACOBI;        
      }
      jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data) = AII_relax_type;  
   }


   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);



   if (Need_CC == 1)
   {

      //======================================================
      //  0 号进程组处理光子系统
      //====================================================== 
   
      if (groupid_x == 0)
      {  
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         WRR->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetDataOwner(WRR, 0);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
         jx_ParVectorInitialize(WII); 
         jx_ParVectorSetConstantValues(WII, -1.0);
         jx_ParVectorSetPartitioningOwner(WII, 0);        
                
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }

         if (ARR_solver_id == SOLVER_AMG)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
            jx_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
            jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
            jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout            
            jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
            jx_PAMGSetup(ARR_amg_solver, ARR);
  
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            ARR_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(ARR_amg_solver, 1); 
            jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25); 
            jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout   
            jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         
            ARR_gmres_solver = jx_ParCSRGMRESCreate(comm_x);
            jx_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
            jx_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
            jx_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
            jx_GMRESSetPrintLevel(ARR_gmres_solver, 0); 
         
            jx_GMRESSetPrecond( ARR_gmres_solver,
                                jx_PAMGPrecond,
                                jx_PAMGSetup,
                                ARR_amg_solver );
                             
            jx_PAMGSetup(ARR_amg_solver, ARR);
         
            jx_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
         
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
            } 
         } 

         jx_ParVectorSetConstantValues(WEE, 0.0); 
         jx_ParVecMul(VRE, WII, RHS);
         
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WEE);  
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            { 
               jx_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
            }
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, WEE);
            if ((print_level == 2 || print_level == 3) && myid == rootid_R)
            {
               jx_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
            }  
         }

         if (test_subls_iter)
         {
            if (ARR_solver_id == SOLVER_AMG)
            {
               jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
            }
            else if (ARR_solver_id == SOLVER_AMGGMRES)
            {
               jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
            }
         }     

         if (ARR_relax_type == RELAX_AMG)
         {
            /* modify parameters for Relaxation */
            jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
            jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
            jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
            jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout
            jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jx_PAMGDestroy(ARR_amg_solver);
            ARR_amg_solver = NULL;
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorSetPartitioningOwner(JAC, 0);            
         }
      }
        
      //======================================================
      //  1 号进程组处理电子系统
      //====================================================== 
     
      else if (groupid_x == 1)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         WEE->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetDataOwner(WEE, 0);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
         jx_ParVectorInitialize(WII);
         jx_ParVectorSetPartitioningOwner(WII, 0);         
   
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         }

         if (AEE_relax_type == RELAX_AMG)
         {
            AEE_amg_solver = jx_PAMGCreate();
            jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
            jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
            jx_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
            jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
            jx_PAMGSetPrintLevel(AEE_amg_solver, 0);
            jx_PAMGSetup(AEE_amg_solver, AEE); 
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);
         }

         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }
      }
 
      //======================================================
      //  2 号进程组处理离子系统
      //======================================================  
  
      else if (groupid_x == 2)
      {
         RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(RHS);
         jx_ParVectorSetPartitioningOwner(RHS, 0);

         WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WRR);
         jx_ParVectorSetConstantValues(WRR, -1.0);
         jx_ParVectorSetPartitioningOwner(WRR, 0);

         WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         jx_ParVectorInitialize(WEE);
         jx_ParVectorSetPartitioningOwner(WEE, 0);

         WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
         WII->local_vector->data = &temp_adrress;
         jx_ParVectorInitialize(WII);   
         jx_ParVectorSetDataOwner(WII, 0);
         jx_ParVectorSetPartitioningOwner(WII, 0);      
                 
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }   

         AII_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit); 
         jx_PAMGSetTol(AII_amg_solver, AII_interp_tol); 
         jx_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AII_amg_solver, 0);
         jx_PAMGSetup(AII_amg_solver, AII); 
  
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }

         jx_ParVectorSetConstantValues(WEE, 0.0);
        
         jx_ParVecMul(VIE, WRR, RHS);
         
         jx_PAMGSolve(AII_amg_solver, AII, RHS, WEE);  
           
         if ((print_level == 2 || print_level == 3) && myid == rootid_I) 
         {
            jx_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);   
         }
 
         if (test_subls_iter)
         {
            jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
         }

         if (AII_relax_type == RELAX_AMG)
         {
            jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
            jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jx_PAMGDestroy(AII_amg_solver);
            AII_amg_solver = NULL;  

            JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC); 
            jx_ParVectorSetPartitioningOwner(JAC, 0);           
         }
      }   
 
   }
   else // if (Need_CC == 0)
   {
   
      //======================================================
      //  0 号进程组处理光子系统
      //====================================================== 
   
      if (groupid_x == 0)
      {         
            RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(RHS);
            jx_ParVectorSetPartitioningOwner(RHS, 0); 

            WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            WRR->local_vector->data = &temp_adrress;
            jx_ParVectorInitialize(WRR);
            jx_ParVectorSetDataOwner(WRR, 0);
            jx_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(WEE);
            jx_ParVectorSetPartitioningOwner(WEE, 0);

            WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(WII); 
            jx_ParVectorSetConstantValues(WII, -1.0); 
            jx_ParVectorSetPartitioningOwner(WII, 0);       
                
            if (print_level == 1 || print_level == 3)
            {
               starttime = jx_MPI_Wtime();
            }

            if (ARR_relax_type == RELAX_AMG)
            {
               ARR_amg_solver = jx_PAMGCreate();
               jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
               jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
               jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
               jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
               jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
               jx_PAMGSetup(ARR_amg_solver, ARR);
            } 
            else if (ARR_relax_type == RELAX_WJACOBI)
            {
               JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(ARR));
               jx_ParVectorInitialize(JAC); 
               jx_ParVectorSetPartitioningOwner(JAC, 0);         
            }
  
            if (print_level == 1 || print_level == 3)
            {                    
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
            } 
      }
   
      //======================================================
      //  1 号进程组处理电子系统
      //====================================================== 
     
      else if (groupid_x == 1)
      {
            RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(RHS);
            jx_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(WRR);
            jx_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            WEE->local_vector->data = &temp_adrress;
            jx_ParVectorInitialize(WEE);
            jx_ParVectorSetDataOwner(WEE, 0);
            jx_ParVectorSetPartitioningOwner(WEE, 0);

            WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(WII);
            jx_ParVectorSetPartitioningOwner(WII, 0);         
   
            if (print_level == 1 || print_level == 3)
            {      
               starttime = jx_MPI_Wtime();
            }

            if (AEE_relax_type == RELAX_AMG)
            {
               AEE_amg_solver = jx_PAMGCreate();
               jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
               jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
               jx_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
               jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
               jx_PAMGSetPrintLevel(AEE_amg_solver, 0);
               jx_PAMGSetup(AEE_amg_solver, AEE); 
            }
            else if (AEE_relax_type == RELAX_WJACOBI)
            {
               JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AEE));
               jx_ParVectorInitialize(JAC);
               jx_ParVectorSetPartitioningOwner(JAC, 0);          
            }
         
            if (print_level == 1 || print_level == 3)
            {                      
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
            }
      }
   
      //======================================================
      //  2 号进程组处理离子系统
      //======================================================  
    
      else if (groupid_x == 2)
      {
            RHS = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(RHS);
            jx_ParVectorSetPartitioningOwner(RHS, 0);

            WRR = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(WRR);
            jx_ParVectorSetConstantValues(WRR, -1.0);
            jx_ParVectorSetPartitioningOwner(WRR, 0);

            WEE = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(WEE);
            jx_ParVectorSetPartitioningOwner(WEE, 0);

            WII = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
            WII->local_vector->data = &temp_adrress;
            jx_ParVectorInitialize(WII);   
            jx_ParVectorSetDataOwner(WII, 0);
            jx_ParVectorSetPartitioningOwner(WII, 0);      
                  
            if (print_level == 1 || print_level == 3)
            {
               starttime = jx_MPI_Wtime();
            }   

            if (AII_relax_type == RELAX_AMG)
            {         
               AII_amg_solver = jx_PAMGCreate();
               jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
               jx_PAMGSetTol(AII_amg_solver, AII_relax_tol); 
               jx_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
               jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
               jx_PAMGSetPrintLevel(AII_amg_solver, 0);
               jx_PAMGSetup(AII_amg_solver, AII); 
            }
            else if (AII_relax_type == RELAX_WJACOBI)
            {
               JAC = jx_ParVectorCreate(comm_x, n, jx_ParCSRMatrixRowStarts(AII));
               jx_ParVectorInitialize(JAC);
               jx_ParVectorSetPartitioningOwner(JAC, 0);          
            }         
         
            if (print_level == 1 || print_level == 3)
            {                       
               endtime = jx_MPI_Wtime();
               jx_GetWallTime(comm_x, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
            }
      }

   } // end if Need_CC  
    
   jx_MPI_Barrier(comm); //  各进程(组)同步
    

   if (Need_CC == 1)
   {           
      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
  
      jx_3tAPCTLParaInterpVec(comm, comm_x, groupid_x, NULL, WEE, &PRR, &PII);

      ACC = jx_3tApctlCoarseOperator_mp(comm, ARR_all, AEE_all, AII_all, 
                                        VRE_all, VER_all, VEI_all, VIE_all, PRR, PII);

      /* 释放 PRR 和 PII */                                   
      jx_ParVectorDestroy(PRR);
      jx_ParVectorDestroy(PII); 
                                        
      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      } 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
  
      row_starts = jx_ParCSRMatrixRowStarts(A);
      col_starts = jx_ParCSRMatrixRowStarts(ACC); 
      P = jx_3tApctlInterpolation(comm, comm_x, N, groupid_x, row_starts, col_starts, WEE);                   

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Interp Operator", starttime, endtime, 0, 3);
      }

      /* 为生成 ACC 开设的辅助矩阵和向量释放内存 */
      jx_ParCSRMatrixDestroy(ARR_all);
      jx_ParCSRMatrixDestroy(AEE_all);
      jx_ParCSRMatrixDestroy(AII_all);
      jx_ParVectorDestroy(VRE_all);
      jx_ParVectorDestroy(VER_all);
      jx_ParVectorDestroy(VEI_all);
      jx_ParVectorDestroy(VIE_all);            

      //=============================================================
      //  生成表示粗网格系统右端和解向量的两个并行向量
      //=============================================================

      GCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(GCC); 
      jx_ParVectorSetPartitioningOwner(GCC, 0);    

      WCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(WCC);
      jx_ParVectorSetPartitioningOwner(WCC, 0);

      RES = jx_ParVectorCreate(comm, N, jx_ParCSRMatrixRowStarts(A));
      jx_ParVectorInitialize(RES); 
      jx_ParVectorSetPartitioningOwner(RES, 0); 

      //================================================================
      // 对粗网格矩阵 ACC 进行 AMG 或 AMGGMRES 的 Setup 过程
      //================================================================ 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }

      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jx_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0); 
         jx_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jx_PAMGSetPrintLevel(ACC_amg_solver, 0);
         jx_PAMGSetup(ACC_amg_solver, ACC); 
         
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }   
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, 1); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jx_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jx_PAMGSetPrintLevel(ACC_amg_solver, 0);
         
         ACC_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jx_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jx_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jx_GMRESSetPrintLevel(ACC_gmres_solver, 0); 
         
         jx_GMRESSetPrecond( ACC_gmres_solver,
                             jx_PAMGPrecond,
                             jx_PAMGSetup,
                             ACC_amg_solver );
                             
         jx_PAMGSetup(ACC_amg_solver, ACC);
         
         jx_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);
      
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }   
      }      
   
   } // end if Need_CC

   //=============================================================
   //  填充结构体 pre_3tapctl_data 的部分成员
   //=============================================================
     
   jx_3tAPCTLDataComm(pre_3tapctl_data)           = comm;  
   jx_3tAPCTLDataCommX(pre_3tapctl_data)          = comm_x; 
   jx_3tAPCTLDataCommY(pre_3tapctl_data)          = comm_y;     
   jx_3tAPCTLDataGroupIdX(pre_3tapctl_data)       = groupid_x;
   jx_3tAPCTLDataGroupIdY(pre_3tapctl_data)       = groupid_y;

   jx_3tAPCTLDataA(pre_3tapctl_data)              = A;
   jx_3tAPCTLDataARR(pre_3tapctl_data)            = ARR;
   jx_3tAPCTLDataAEE(pre_3tapctl_data)            = AEE;
   jx_3tAPCTLDataAII(pre_3tapctl_data)            = AII;

   jx_3tAPCTLDataVRE(pre_3tapctl_data)            = VRE;
   jx_3tAPCTLDataVER(pre_3tapctl_data)            = VER;
   jx_3tAPCTLDataVEI(pre_3tapctl_data)            = VEI;
   jx_3tAPCTLDataVIE(pre_3tapctl_data)            = VIE;
   
   jx_3tAPCTLDataP(pre_3tapctl_data)              = P;
   jx_3tAPCTLDataACC(pre_3tapctl_data)            = ACC;

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = ARR_amg_solver;  
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = AII_amg_solver;
   jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = ACC_amg_solver;
   
   jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;
   
   jx_3tAPCTLDataWRR(pre_3tapctl_data)            = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data)            = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data)            = WII;
   jx_3tAPCTLDataWCC(pre_3tapctl_data)            = WCC;
   jx_3tAPCTLDataGCC(pre_3tapctl_data)            = GCC;

   jx_3tAPCTLDataRES(pre_3tapctl_data)            = RES;
   jx_3tAPCTLDataRHS(pre_3tapctl_data)            = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data)            = JAC;
                  
   return (0); 
}

/*!
 * \fn JX_Int jx_3tAPCTLSetup_sp
 * \brief Setup phase of PCTL Iteration or preconditioner(for single-processor case). 
 * \author peghoty 
 * \date 2011/09/26
 */
JX_Int
jx_3tAPCTLSetup_sp( jx_3tAPCTLData *pre_3tapctl_data, jx_ParCSRMatrix *A )
{
   JX_Real starttime = 0.0, endtime = 0.0;

   MPI_Comm comm = jx_ParCSRMatrixComm(A);

   JX_Int fixit_pctl_R = jx_3tAPCTLDataFixItPCTLR(pre_3tapctl_data);
   JX_Int fixit_pctl_E = jx_3tAPCTLDataFixItPCTLE(pre_3tapctl_data);
   JX_Int fixit_pctl_I = jx_3tAPCTLDataFixItPCTLI(pre_3tapctl_data);
   JX_Int fixit_brlx_R = jx_3tAPCTLDataFixItBRLXR(pre_3tapctl_data);
   JX_Int fixit_brlx_E = jx_3tAPCTLDataFixItBRLXE(pre_3tapctl_data);
   JX_Int fixit_brlx_I = jx_3tAPCTLDataFixItBRLXI(pre_3tapctl_data);
      
   /* max_iter for each subblock */
   JX_Int ARR_interp_maxit = jx_3tAPCTLDataARRInterpMaxIt(pre_3tapctl_data);     
   JX_Int AII_interp_maxit = jx_3tAPCTLDataAIIInterpMaxIt(pre_3tapctl_data);     
   JX_Int ACC_relax_maxit  = jx_3tAPCTLDataACCRelaxMaxIt(pre_3tapctl_data); 
   
   /* 需要根据参数 Need_CC 来待定 */ 
   JX_Int ARR_relax_maxit;
   JX_Int AEE_relax_maxit;
   JX_Int AII_relax_maxit;
      
   /* tolerace for each subblock */
   JX_Real  ARR_interp_tol = jx_3tAPCTLDataARRInterpTol(pre_3tapctl_data);   
   JX_Real  AII_interp_tol = jx_3tAPCTLDataAIIInterpTol(pre_3tapctl_data);
   JX_Real  ARR_relax_tol  = jx_3tAPCTLDataARRRelaxTol(pre_3tapctl_data);
   JX_Real  AEE_relax_tol  = jx_3tAPCTLDataAEERelaxTol(pre_3tapctl_data);
   JX_Real  AII_relax_tol  = jx_3tAPCTLDataAIIRelaxTol(pre_3tapctl_data);
   JX_Real  ACC_relax_tol  = jx_3tAPCTLDataACCRelaxTol(pre_3tapctl_data); 

   /* solver type for interpolation-building of PRR.  peghoty,2011/10/29 */
   JX_Int ARR_solver_id = jx_3tAPCTLDataARRSolverID(pre_3tapctl_data);
   
   /* solver type for PCTL iteration.  peghoty,2011/10/29 */
   JX_Int ACC_solver_id = jx_3tAPCTLDataACCSolverID(pre_3tapctl_data);
   
   /* restart parameters for GMRES solver. peghoty,2011/10/29 */
   JX_Int ARR_kdim = jx_3tAPCTLDataARRKDim(pre_3tapctl_data);
   JX_Int ACC_kdim = jx_3tAPCTLDataACCKDim(pre_3tapctl_data);  
    
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data);  
   JX_Int blocksmooth_type = jx_3tAPCTLDataBlockSmoothType(pre_3tapctl_data);
  
   jx_ParAMGData   *ARR_amg_solver = NULL;
   jx_ParAMGData   *AEE_amg_solver = NULL;
   jx_ParAMGData   *AII_amg_solver = NULL;
   jx_ParAMGData   *ACC_amg_solver = NULL;

   /* gmres solver data for ARR and ACC. peghoty,2011/10/29 */
   jx_GMRESData    *ARR_gmres_solver = NULL;
   jx_GMRESData    *ACC_gmres_solver = NULL; 
   
   jx_ParCSRMatrix *ARR = NULL;
   jx_ParCSRMatrix *AEE = NULL;
   jx_ParCSRMatrix *AII = NULL;
   
   jx_ParVector    *VRE = NULL;
   jx_ParVector    *VER = NULL;
   jx_ParVector    *VEI = NULL;
   jx_ParVector    *VIE = NULL;      

   jx_ParVector    *PRR = NULL;
   jx_ParVector    *PII = NULL;
   jx_ParCSRMatrix *ACC = NULL; 
     
   jx_ParVector    *WRR = NULL;
   jx_ParVector    *WEE = NULL;
   jx_ParVector    *WII = NULL;
   jx_ParVector    *WCC = NULL;
 
   jx_ParVector    *RES = NULL;
   jx_ParVector    *RHS = NULL;
   jx_ParVector    *JAC = NULL;

   JX_Int Need_CC; 
   
   JX_Int ARR_relax_type;
   JX_Int AEE_relax_type;
   JX_Int AII_relax_type;      
     
   /* local variables */
   jx_CSRMatrix  *ARR_s = NULL; 
   jx_CSRMatrix  *AEE_s = NULL;
   jx_CSRMatrix  *AII_s = NULL;
   jx_Vector     *VRE_s = NULL;
   jx_Vector     *VER_s = NULL;
   jx_Vector     *VEI_s = NULL;
   jx_Vector     *VIE_s = NULL;
   
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);   

   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Int i;
   JX_Real temp_adrress = 0.0;


   //======================================================
   //  将并行矩阵 A 转化为串行矩阵 A_s, 并从中抓取各子块
   //======================================================

   if (print_level == 1 || print_level == 3)
   {       
      starttime = jx_MPI_Wtime();
   }
   
   jx_CSRMatrix *A_s = NULL;
   A_s = jx_ParCSRMatrixDiag(A);
   jx_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
   
   if (print_level == 1 || print_level == 3)
   {                       
      endtime = jx_MPI_Wtime();
      jx_GetWallTime(comm, "APCTL == Get Seven Blocks", starttime, endtime, 0, 3);
   }  
   

   //===================================================================
   //  检查三个物理量之间的耦合强弱，以及三对角块的强对角占优性
   //===================================================================

   jx_3tAPCTLWCDD( pre_3tapctl_data, comm, 
                   ARR_s, AEE_s, AII_s, VRE_s, VER_s, VEI_s, VIE_s );

   ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);
   Need_CC        = jx_3tAPCTLDataNeedCC(pre_3tapctl_data);

   if (Need_CC)
   {
      ARR_relax_maxit = fixit_pctl_R;
      AEE_relax_maxit = fixit_pctl_E;
      AII_relax_maxit = fixit_pctl_I;
   }
   else
   {
      ARR_relax_maxit = fixit_brlx_R;
      AEE_relax_maxit = fixit_brlx_E;
      AII_relax_maxit = fixit_brlx_I;
   }
   jx_3tAPCTLSetARRRelaxMaxIt(pre_3tapctl_data, ARR_relax_maxit);
   jx_3tAPCTLSetAEERelaxMaxIt(pre_3tapctl_data, AEE_relax_maxit);
   jx_3tAPCTLSetAIIRelaxMaxIt(pre_3tapctl_data, AII_relax_maxit);

   ARR = jx_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
   AEE = jx_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
   AII = jx_CSRMatrixToParCSRMatrix_sp(comm, AII_s);    

   if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
   { 
      VRE = jx_VectorToParVector_sp(comm, VRE_s);
      VER = jx_VectorToParVector_sp(comm, VER_s);
      VEI = jx_VectorToParVector_sp(comm, VEI_s);
      VIE = jx_VectorToParVector_sp(comm, VIE_s);      
   }

   RHS = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
   jx_ParVectorInitialize(RHS);
   jx_ParVectorSetPartitioningOwner(RHS, 0);

   WRR = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
   WRR->local_vector->data = &temp_adrress;
   jx_ParVectorInitialize(WRR);
   jx_ParVectorSetDataOwner(WRR, 0);
   jx_ParVectorSetPartitioningOwner(WRR, 0);

   WEE = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AEE));
   WEE->local_vector->data = &temp_adrress;
   jx_ParVectorInitialize(WEE);
   jx_ParVectorSetDataOwner(WEE, 0);
   jx_ParVectorSetPartitioningOwner(WEE, 0);

   WII = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
   WII->local_vector->data = &temp_adrress;
   jx_ParVectorInitialize(WII);   
   jx_ParVectorSetDataOwner(WII, 0);
   jx_ParVectorSetPartitioningOwner(WII, 0);  


   //======================================================
   //                需要进行粗网格校正
   //======================================================
   
   if (Need_CC == 1)
   {      
      PRR = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
      jx_ParVectorInitialize(PRR);
      jx_ParVectorSetPartitioningOwner(PRR, 0); 

      PII = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
      jx_ParVectorInitialize(PII);
      jx_ParVectorSetPartitioningOwner(PII, 0); 

      //----------------------------------------------------//
      //                 Step 1: 处理光子系统                 //
      //----------------------------------------------------// 
       
      if (ARR_solver_id == SOLVER_AMG)
      {  
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         } 

         ARR_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ARR_amg_solver, ARR_interp_maxit); 
         jx_PAMGSetTol(ARR_amg_solver, ARR_interp_tol); 
         jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout      
         jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         jx_PAMGSetup(ARR_amg_solver, ARR); 

         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         } 
      }
      else if (ARR_solver_id == SOLVER_AMGGMRES)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         } 

         ARR_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ARR_amg_solver, 1); 
         jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         
         ARR_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ARR_gmres_solver, ARR_kdim);
         jx_GMRESSetMaxIter(ARR_gmres_solver, ARR_interp_maxit);
         jx_GMRESSetTol(ARR_gmres_solver, ARR_interp_tol);
         jx_GMRESSetPrintLevel(ARR_gmres_solver, 0); 
         
         jx_GMRESSetPrecond( ARR_gmres_solver,
                             jx_PAMGPrecond,
                             jx_PAMGSetup,
                             ARR_amg_solver );
                             
         jx_PAMGSetup(ARR_amg_solver, ARR);
         
         jx_GMRESSetup(ARR_gmres_solver, ARR, RHS, RHS);
         
         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "PPCTL == AMGGMRESSetup for ARR", starttime, endtime, 0, 3);
         } 
      } 

      for (i = 0; i < n; i ++)
      {
         RHS->local_vector->data[i] = - VRE->local_vector->data[i];   
      }
      
      jx_ParVectorSetConstantValues(PRR, 0.0);
      
      if (ARR_solver_id == SOLVER_AMG)
      { 
         jx_PAMGSolve(ARR_amg_solver, ARR, RHS, PRR);  
         if (print_level == 2 || print_level == 3)
         { 
            jx_printf(" PCTL-Setup == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
         }   
      } 
      else if (ARR_solver_id == SOLVER_AMGGMRES)
      {
         jx_GMRESSolve(ARR_gmres_solver, ARR, ARR, RHS, PRR);
         if (print_level == 2 || print_level == 3)
         {
            jx_printf(" PCTL-Setup == ARR AMGGMRES-Iter: %d\n", ARR_gmres_solver->num_iterations);
         }  
      }
           
      if (test_subls_iter)
      {
         if (ARR_solver_id == SOLVER_AMG)
         {
            jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_amg_solver->num_iterations;
         }
         else if (ARR_solver_id == SOLVER_AMGGMRES)
         {
            jx_3tAPCTLDataNumIterArSetup(pre_3tapctl_data) = ARR_gmres_solver->num_iterations;
         }
      }        

      if (ARR_relax_type == RELAX_AMG)
      {
         jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
         jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol);
         jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 0.0);
         jx_PAMGSetCoarsenType(ARR_amg_solver, 6); // 0: CLJP; 6: Falgout
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jx_PAMGDestroy(ARR_amg_solver);
         ARR_amg_solver = NULL;
         if (!JAC)
         {  
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);  
         }           
      }
  
      //----------------------------------------------------//
      //                 Step 2: 处理电子系统                 //
      //----------------------------------------------------// 

      if (AEE_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         }      
      
         AEE_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
         jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
         jx_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AEE_amg_solver, 0);
         jx_PAMGSetup(AEE_amg_solver, AEE); 

         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }         
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);  
         }
      }

      //----------------------------------------------------//
      //                 Step 3: 处理离子系统                 //
      //----------------------------------------------------// 
                  
      if (print_level == 1 || print_level == 3)
      {
         starttime = jx_MPI_Wtime();
      }   

      AII_amg_solver = jx_PAMGCreate();
      jx_PAMGSetMaxIter(AII_amg_solver, AII_interp_maxit); 
      jx_PAMGSetTol(AII_amg_solver, AII_interp_tol); 
      jx_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
      jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
      jx_PAMGSetPrintLevel(AII_amg_solver, 0);
      jx_PAMGSetup(AII_amg_solver, AII); 

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
      }

      for (i = 0; i < n; i ++)
      {
         RHS->local_vector->data[i] = - VIE->local_vector->data[i];   
      }
      jx_ParVectorSetConstantValues(PII, 0.0);
      jx_PAMGSolve(AII_amg_solver, AII, RHS, PII);  
      if (print_level == 2 || print_level == 3) 
      {
         jx_printf(" PCTL-Setup == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);   
      }

      if (test_subls_iter)
      {
         jx_3tAPCTLDataNumIterAiSetup(pre_3tapctl_data) = AII_amg_solver->num_iterations;
      } 

      if (AII_relax_type == RELAX_AMG)
      {
         jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
         jx_PAMGSetTol(AII_amg_solver, AII_relax_tol);
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jx_PAMGDestroy(AII_amg_solver);
         AII_amg_solver = NULL;  
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);  
         }           
      }  
  
   }

   //======================================================
   //                不需要进行粗网格校正
   //======================================================
      
   else // if (Need_CC == 0)
   {
     
      //----------------------------------------------------//
      //                 Step 1: 处理光子系统                 //
      //----------------------------------------------------// 

      if (ARR_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }
         
         ARR_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ARR_amg_solver, ARR_relax_maxit); 
         jx_PAMGSetTol(ARR_amg_solver, ARR_relax_tol); 
         jx_PAMGSetStrongThreshold(ARR_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ARR_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(ARR_amg_solver, 0);
         jx_PAMGSetup(ARR_amg_solver, ARR);

         if (print_level == 1 || print_level == 3)
         {                    
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ARR", starttime, endtime, 0, 3);
         }
      } 
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ARR));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);  
         }          
      }
  
      //----------------------------------------------------//
      //                 Step 2: 处理电子系统                 //
      //----------------------------------------------------// 

      if (AEE_relax_type == RELAX_AMG)
      {
         if (print_level == 1 || print_level == 3)
         {      
            starttime = jx_MPI_Wtime();
         }
                  
         AEE_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AEE_amg_solver, AEE_relax_maxit); 
         jx_PAMGSetTol(AEE_amg_solver, AEE_relax_tol); 
         jx_PAMGSetStrongThreshold(AEE_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AEE_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AEE_amg_solver, 0);
         jx_PAMGSetup(AEE_amg_solver, AEE); 
         
         if (print_level == 1 || print_level == 3)
         {                      
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for AEE", starttime, endtime, 0, 3);
         }            
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AEE));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);  
         }          
      }

      //----------------------------------------------------//
      //                 Step 3: 处理离子系统                 //
      //----------------------------------------------------// 

      if (AII_relax_type == RELAX_AMG)
      {  
         if (print_level == 1 || print_level == 3)
         {
            starttime = jx_MPI_Wtime();
         }   
             
         AII_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(AII_amg_solver, AII_relax_maxit); 
         jx_PAMGSetTol(AII_amg_solver, AII_relax_tol); 
         jx_PAMGSetStrongThreshold(AII_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(AII_amg_solver, 1.0);
         jx_PAMGSetPrintLevel(AII_amg_solver, 0);
         jx_PAMGSetup(AII_amg_solver, AII); 
            
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for AII", starttime, endtime, 0, 3);
         }
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         if (!JAC)
         {
            JAC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(AII));
            jx_ParVectorInitialize(JAC);
            jx_ParVectorSetPartitioningOwner(JAC, 0);   
         }        
      }         
          
   } // end if Need_CC  


   if (Need_CC == 1)
   {
      //================================================================
      // 生成并行粗网格矩阵 ACC = P^T*A*P
      //================================================================ 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
         
      ACC = jx_3tApctlCoarseOperator_sp( comm, ARR_s, AEE_s, AII_s, VRE_s, 
                                         VER_s, VEI_s, VIE_s, PRR, PII );

      if (print_level == 1 || print_level == 3)
      {                       
         endtime = jx_MPI_Wtime();
         jx_GetWallTime(comm, "APCTL ==  Coarse Operator", starttime, endtime, 0, 3);
      } 

      //=============================================================
      //  创建 WCC - 粗网格系统(并行)解向量
      //=============================================================   

      WCC = jx_ParVectorCreate(comm, n, jx_ParCSRMatrixRowStarts(ACC));
      jx_ParVectorInitialize(WCC);
      jx_ParVectorSetPartitioningOwner(WCC, 0);

      //=============================================================
      //  创建 RES - 残量
      //=============================================================

      RES = jx_ParVectorCreate(comm, N, jx_ParCSRMatrixRowStarts(A));
      jx_ParVectorInitialize(RES);
      jx_ParVectorSetPartitioningOwner(RES, 0);
            
      //================================================================
      // 对粗网格矩阵 ACC 进行 AMG 或 AMGGMRES 的 Setup 过程
      //================================================================ 

      if (print_level == 1 || print_level == 3)
      {       
         starttime = jx_MPI_Wtime();
      }
     
      if (ACC_solver_id == SOLVER_AMG)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, ACC_relax_maxit); 
         jx_PAMGSetTol(ACC_amg_solver, ACC_relax_tol); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jx_PAMGSetRhsNrmThreshold(ACC_amg_solver, 1.0);
         jx_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout      
         jx_PAMGSetPrintLevel(ACC_amg_solver, 0);
         jx_PAMGSetup(ACC_amg_solver, ACC); 
         
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGSetup for ACC", starttime, endtime, 0, 3);
         }  
      }
      else if (ACC_solver_id == SOLVER_AMGGMRES)
      {
         ACC_amg_solver = jx_PAMGCreate();
         jx_PAMGSetMaxIter(ACC_amg_solver, 1); 
         jx_PAMGSetStrongThreshold(ACC_amg_solver, 0.25);
         jx_PAMGSetCoarsenType(ACC_amg_solver, 6); // 0: CLJP; 6: Falgout   
         jx_PAMGSetPrintLevel(ACC_amg_solver, 0);
         
         ACC_gmres_solver = jx_ParCSRGMRESCreate(comm);
         jx_GMRESSetKDim(ACC_gmres_solver, ACC_kdim);
         jx_GMRESSetMaxIter(ACC_gmres_solver, ACC_relax_maxit);
         jx_GMRESSetTol(ACC_gmres_solver, ACC_relax_tol);
         jx_GMRESSetPrintLevel(ACC_gmres_solver, 0); 
         
         jx_GMRESSetPrecond( ACC_gmres_solver,
                             jx_PAMGPrecond,
                             jx_PAMGSetup,
                             ACC_amg_solver );
                             
         jx_PAMGSetup(ACC_amg_solver, ACC);
         
         jx_GMRESSetup(ACC_gmres_solver, ACC, WCC, WCC);
      
         if (print_level == 1 || print_level == 3)
         {                       
            endtime = jx_MPI_Wtime();
            jx_GetWallTime(comm, "APCTL == AMGGMRESSetup for ACC", starttime, endtime, 0, 3);
         }   
      }      

   } // end if Need_CC


   //=============================================================
   //  填充结构体 pre_3tapctl_data 的部分成员
   //=============================================================
     
   jx_3tAPCTLDataComm(pre_3tapctl_data)           = comm;  

   jx_3tAPCTLDataA(pre_3tapctl_data)              = A;
   jx_3tAPCTLDataARR(pre_3tapctl_data)            = ARR;
   jx_3tAPCTLDataAEE(pre_3tapctl_data)            = AEE;
   jx_3tAPCTLDataAII(pre_3tapctl_data)            = AII;

   jx_3tAPCTLDataVRE(pre_3tapctl_data)            = VRE;
   jx_3tAPCTLDataVER(pre_3tapctl_data)            = VER;
   jx_3tAPCTLDataVEI(pre_3tapctl_data)            = VEI;
   jx_3tAPCTLDataVIE(pre_3tapctl_data)            = VIE;
   
   jx_3tAPCTLDataPRR(pre_3tapctl_data)            = PRR;
   jx_3tAPCTLDataPII(pre_3tapctl_data)            = PII;
   jx_3tAPCTLDataACC(pre_3tapctl_data)            = ACC;

   jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data)   = ARR_amg_solver;  
   jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data)   = AEE_amg_solver;
   jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data)   = AII_amg_solver;
   jx_3tAPCTLDataACCAMGSolver(pre_3tapctl_data)   = ACC_amg_solver;

   jx_3tAPCTLDataARRGMRESSolver(pre_3tapctl_data) = ARR_gmres_solver;
   jx_3tAPCTLDataACCGMRESSolver(pre_3tapctl_data) = ACC_gmres_solver;

   jx_3tAPCTLDataWRR(pre_3tapctl_data)            = WRR;
   jx_3tAPCTLDataWEE(pre_3tapctl_data)            = WEE;
   jx_3tAPCTLDataWII(pre_3tapctl_data)            = WII;
   jx_3tAPCTLDataWCC(pre_3tapctl_data)            = WCC;

   jx_3tAPCTLDataRES(pre_3tapctl_data)            = RES;
   jx_3tAPCTLDataRHS(pre_3tapctl_data)            = RHS;
   jx_3tAPCTLDataJAC(pre_3tapctl_data)            = JAC;
   
   /* 释放辅助矩阵和向量 */
   jx_TFree(ARR_s);
   jx_TFree(AEE_s);
   jx_TFree(AII_s);

   /* Attention: VRE_s, VER_s, VEI_s, VIE_s should be treated accordingly. peghoty, 2011/11/03 */
   if ( Need_CC == 1 || (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_GS) )
   { 
      /* only free (not destroy) VRE_s, VER_s, VEI_s, VIE_s in this case, 
         because their data parts have been contained in VRE, VER, VEI, VIE.
         See the function "jx_VectorToParVector_sp" for details. */
      jx_TFree(VRE_s);
      jx_TFree(VER_s);
      jx_TFree(VEI_s);
      jx_TFree(VIE_s);    
   }
   else // if (Need_CC == 0 && blocksmooth_type == BLOCKSMOOTH_BD)
   {
      /* destroy VRE_s, VER_s, VEI_s, VIE_s completely in this case,
         because they won't be used any more. */
      jx_SeqVectorDestroy(VRE_s);
      jx_SeqVectorDestroy(VER_s);
      jx_SeqVectorDestroy(VEI_s);
      jx_SeqVectorDestroy(VIE_s);
   }  
             
   return (0); 
}
