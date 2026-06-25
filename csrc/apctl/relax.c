//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  relax.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn JX_Int jx_3tAPCTLRelax_GSType_mp
 * \brief (Gauss-Seidel type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLRelax_GSType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w )
{
   MPI_Comm  comm    = jx_ParCSRMatrixComm(A);
   MPI_Comm  comm_x  = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm  comm_y  = jx_3tAPCTLDataCommY(pre_3tapctl_data);
   
   JX_Int  groupid_x   = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JX_Int  print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JX_Int ARR_relax_maxit  = jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JX_Int AEE_relax_maxit  = jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JX_Int AII_relax_maxit  = jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jx_ParAMGData   *ARR_amg_solver = jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AEE_amg_solver = jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AII_amg_solver = jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data); 
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector    *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector    *VER = jx_3tAPCTLDataVER(pre_3tapctl_data);
   jx_ParVector    *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector    *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParVector    *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector    *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector    *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);
   
   jx_ParVector    *JAC = jx_3tAPCTLDataJAC(pre_3tapctl_data);
   jx_ParVector    *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);
   
   /* newly added, peghoty, 2012/02/16 */
   JX_Int use_fixedmode_R = jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JX_Int use_fixedmode_E = jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JX_Int use_fixedmode_I = jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);     
   
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);
   
   /* local variables */
   JX_Int sweep;
   JX_Int myid, myid_x;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

   if (groupid_x == 0)
   {
      WRR->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 1)
   {
      WEE->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 2)
   {
      WII->local_vector->data = w->local_vector->data;
   }  

   jx_MPI_Bcast(WRR->local_vector->data, WRR->local_vector->size, JX_MPI_REAL, 0, comm_y);  
   jx_MPI_Bcast(WII->local_vector->data, WII->local_vector->size, JX_MPI_REAL, 2, comm_y); 

   if (groupid_x == 1) 
   {   
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

      jx_ParVecZXY(RHS, -1.0, VEI, WII);    
      jx_ParVecZXY(RHS, -1.0, VER, WRR);  
         
      if (AEE_relax_type == RELAX_AMG)
      {      
         jx_ParVectorSetConstantValues(WEE, 0.0); 
         
         if (use_fixedmode_E)
            jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
         else     
            jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            if (use_fixedmode_E)
               jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
            else
               jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
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
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            else
               jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
            jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         }
      }
   }
   
   jx_MPI_Barrier(comm);

   
   jx_MPI_Bcast(WEE->local_vector->data, WEE->local_vector->size, JX_MPI_REAL, 1, comm_y);  
   
   if (groupid_x == 0)
   {
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jx_ParVecZXY(RHS, -1.0, VRE, WEE);
      
      if (ARR_relax_type == RELAX_AMG)
      {  
         jx_ParVectorSetConstantValues(WRR, 0.0);
 
         if (use_fixedmode_R)
            jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
         else          
            jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_R)
               jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            else
               jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
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
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            else
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         }
      }     
   }
   else if (groupid_x == 2)
   {
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jx_ParVecZXY(RHS, -1.0, VIE, WEE);
      
      if (AII_relax_type == RELAX_AMG)
      {
         jx_ParVectorSetConstantValues(WII, 0.0);
         
         if (use_fixedmode_I)
            jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
         else           
            jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_I)
               jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            else
               jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
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
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            else
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         }
      }       
   }

   jx_MPI_Barrier(comm);

   return (0);
}

JX_Int
jx_3tAPCTLmgRelax_GSType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                             jx_ParCSRMatrix   *A,
                             jx_ParVector      *g,
                             jx_ParVector      *w )
{
   MPI_Comm comm   = jx_ParCSRMatrixComm(A);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   MPI_Comm comm_y = jx_3tAPCTLDataCommY(pre_3tapctl_data);

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JX_Int groupid_x   = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JX_Int reset_zero  = jx_3tAPCTLDataResetZero(pre_3tapctl_data);
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

   jx_ParVector *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

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
 
   /* local variables */
   JX_Int sweep, myid, myid_x;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

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

   jx_MPI_Barrier(comm);

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
      WII->local_vector->data = w->local_vector->data;
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
      jx_ParVecZXY(RHS, -1.0, VIE, WEE);
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
   }

   jx_MPI_Barrier(comm);

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLRelax_BDType_mp
 * \brief (Block Diagonal type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/22
 */
JX_Int
jx_3tAPCTLRelax_BDType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w )
{
   MPI_Comm  comm    = jx_ParCSRMatrixComm(A);
   MPI_Comm  comm_x  = jx_3tAPCTLDataCommX(pre_3tapctl_data);
   
   JX_Int  groupid_x   = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JX_Int  print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JX_Int ARR_relax_maxit  = jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JX_Int AEE_relax_maxit  = jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JX_Int AII_relax_maxit  = jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jx_ParAMGData   *ARR_amg_solver = jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AEE_amg_solver = jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AII_amg_solver = jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data); 
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector    *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector    *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector    *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);
   
   jx_ParVector    *JAC = jx_3tAPCTLDataJAC(pre_3tapctl_data);
   jx_ParVector    *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);
   
   /* newly added, peghoty, 2012/02/16 */
   JX_Int use_fixedmode_R = jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JX_Int use_fixedmode_E = jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JX_Int use_fixedmode_I = jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data); 
   
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);   

   /* local variables */
   JX_Int sweep;
   JX_Int myid, myid_x;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

   if (groupid_x == 0)
   {
      WRR->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 1)
   {
      WEE->local_vector->data = w->local_vector->data;
   }
   else if (groupid_x == 2)
   {
      WII->local_vector->data = w->local_vector->data;
   }  


   if (groupid_x == 1) 
   {   
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (AEE_relax_type == RELAX_AMG)
      {      
         jx_ParVectorSetConstantValues(WEE, 0.0); 
         
         if (use_fixedmode_E)
            jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
         else     
            jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

         if ((print_level == 2 || print_level == 3) && myid_x == 0)
         { 
            if (use_fixedmode_E)
               jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
            else
               jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
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
                jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
            else
                jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
         }
         else if (AEE_relax_type == RELAX_WJACOBI)
         {
             jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         }
      }
   }
   else if (groupid_x == 0)
   {
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);

      if (ARR_relax_type == RELAX_AMG)
      {  
         jx_ParVectorSetConstantValues(WRR, 0.0);
 
         if (use_fixedmode_R)
            jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
         else          
            jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_R)
               jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            else
               jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
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
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
            else
               jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
         }
         else if (ARR_relax_type == RELAX_WJACOBI)
         {
            jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         }
      }      
   } 
   else if (groupid_x == 2)
   {
      jx_SeqVectorCopy(g->local_vector, RHS->local_vector);
      
      if (AII_relax_type == RELAX_AMG)
      {
         jx_ParVectorSetConstantValues(WII, 0.0);
         
         if (use_fixedmode_I)
            jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
         else           
            jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
         if ((print_level == 2 || print_level == 3) && myid_x == 0) 
         {
            if (use_fixedmode_I)
               jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            else
               jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
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
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
            else
               jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
         }
         else if (AII_relax_type == RELAX_WJACOBI)
         {
            jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         }
      }        
   }

   jx_MPI_Barrier(comm);

   return (0);
}

JX_Int
jx_3tAPCTLmgRelax_BDType_mp( jx_3tAPCTLData    *pre_3tapctl_data,
                             jx_ParCSRMatrix   *A,
                             jx_ParVector      *g,
                             jx_ParVector      *w )
{
   MPI_Comm comm   = jx_ParCSRMatrixComm(A);
   MPI_Comm comm_x = jx_3tAPCTLDataCommX(pre_3tapctl_data);

   JX_Int ng = jx_3tAPCTLDataNumGroup(pre_3tapctl_data);

   JX_Int groupid_x   = jx_3tAPCTLDataGroupIdX(pre_3tapctl_data);
   JX_Int reset_zero  = jx_3tAPCTLDataResetZero(pre_3tapctl_data);
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

   /* local variables */
   JX_Int sweep, myid, myid_x;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);

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
               jx_printf(" APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
            }
            else
            {
               jx_printf(" APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations);
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
               jx_printf(" APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
            }
            else
            {
               jx_printf(" APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
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
   }

   jx_MPI_Barrier(comm);

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLRelax_GSType_sp
 * \brief (Gauss-Seidel type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/17
 */
JX_Int
jx_3tAPCTLRelax_GSType_sp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w )
{
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JX_Int ARR_relax_maxit  = jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JX_Int AEE_relax_maxit  = jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JX_Int AII_relax_maxit  = jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jx_ParAMGData   *ARR_amg_solver = jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AEE_amg_solver = jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AII_amg_solver = jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data); 
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector    *VRE = jx_3tAPCTLDataVRE(pre_3tapctl_data);
   jx_ParVector    *VER = jx_3tAPCTLDataVER(pre_3tapctl_data);
   jx_ParVector    *VEI = jx_3tAPCTLDataVEI(pre_3tapctl_data);
   jx_ParVector    *VIE = jx_3tAPCTLDataVIE(pre_3tapctl_data);

   jx_ParVector    *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector    *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector    *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);
   
   jx_ParVector    *JAC = jx_3tAPCTLDataJAC(pre_3tapctl_data);
   jx_ParVector    *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);
   
   JX_Real *vre_data = jx_VectorData(jx_ParVectorLocalVector(VRE));
   JX_Real *ver_data = jx_VectorData(jx_ParVectorLocalVector(VER));
   JX_Real *vei_data = jx_VectorData(jx_ParVectorLocalVector(VEI));
   JX_Real *vie_data = jx_VectorData(jx_ParVectorLocalVector(VIE));
   JX_Real *rhs_data = jx_VectorData(jx_ParVectorLocalVector(RHS));
   
   JX_Real *g_data   = jx_VectorData(jx_ParVectorLocalVector(g));
   JX_Real *w_data   = jx_VectorData(jx_ParVectorLocalVector(w));
   
   JX_Real *wrr_data = NULL;
   JX_Real *wee_data = NULL;
   JX_Real *wii_data = NULL;
   
   /* newly added, peghoty, 2012/02/16 */
   JX_Int use_fixedmode_R = jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JX_Int use_fixedmode_E = jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JX_Int use_fixedmode_I = jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);      

   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);

   /* local variables */
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Int nplusn = 2*n;
   JX_Int i, sweep;

   jx_VectorData(jx_ParVectorLocalVector(WRR)) = w_data;
   jx_VectorData(jx_ParVectorLocalVector(WEE)) = w_data + n;
   jx_VectorData(jx_ParVectorLocalVector(WII)) = w_data + nplusn;
   
   wrr_data = jx_VectorData(jx_ParVectorLocalVector(WRR));
   wee_data = jx_VectorData(jx_ParVectorLocalVector(WEE));
   wii_data = jx_VectorData(jx_ParVectorLocalVector(WII));
 
   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+n] - vei_data[i]*wii_data[i] - ver_data[i]*wrr_data[i];
   }
    
   jx_ParVectorSetConstantValues(WEE, 0.0);
         
   if (AEE_relax_type == RELAX_AMG)
   {      
      if (use_fixedmode_E)
         jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
      else     
         jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

      if (print_level == 2 || print_level == 3)
      { 
         if (use_fixedmode_E)
            jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
         else
            jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
      }                 
   }
   else if (AEE_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
      {
         jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jx_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AEE_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_E)
            jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         else
            jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
      }
   }

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i] - vre_data[i]*wee_data[i];
   }      
   
   jx_ParVectorSetConstantValues(WRR, 0.0); 
     
   if (ARR_relax_type == RELAX_AMG)
   {  
      if (use_fixedmode_R)
         jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
      else          
         jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_R)
            jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
         else
            jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
      }         
   }
   else if (ARR_relax_type == RELAX_WJACOBI)
   {  
      for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
      {      
         jx_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jx_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (ARR_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_R)
            jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         else
            jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
      }
   }      

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+nplusn] - vie_data[i]*wee_data[i];
   } 
           
   jx_ParVectorSetConstantValues(WII, 0.0);
     
   if (AII_relax_type == RELAX_AMG)
   {
      if (use_fixedmode_I)
         jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
      else           
         jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_I)
            jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
         else
            jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
      }         
   }
   else if (AII_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
      {      
         jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC); 
      }
      if (print_level == 2 || print_level == 3)
      { 
         jx_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AII_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_I)
            jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         else
            jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations; 
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
      }
   }        

   return (0);
}

/*!
 * \fn JX_Int jx_3tAPCTLRelax_BDType_sp
 * \brief (Block-Diagonal type) Block Relaxation of PCTL iteration or preconditioner. 
 * \author peghoty 
 * \date 2011/09/27
 */
JX_Int
jx_3tAPCTLRelax_BDType_sp( jx_3tAPCTLData    *pre_3tapctl_data,
                           jx_ParCSRMatrix   *A,
                           jx_ParVector      *g,
                           jx_ParVector      *w )
{
   JX_Int print_level = jx_3tAPCTLDataPrintLevel(pre_3tapctl_data); 
   
   JX_Int ARR_relax_type = jx_3tAPCTLDataARRRelaxType(pre_3tapctl_data);
   JX_Int AEE_relax_type = jx_3tAPCTLDataAEERelaxType(pre_3tapctl_data);
   JX_Int AII_relax_type = jx_3tAPCTLDataAIIRelaxType(pre_3tapctl_data);   

   JX_Int ARR_relax_maxit  = jx_3tAPCTLDataARRRelaxMaxIt(pre_3tapctl_data);
   JX_Int AEE_relax_maxit  = jx_3tAPCTLDataAEERelaxMaxIt(pre_3tapctl_data);
   JX_Int AII_relax_maxit  = jx_3tAPCTLDataAIIRelaxMaxIt(pre_3tapctl_data); 
         
   jx_ParAMGData   *ARR_amg_solver = jx_3tAPCTLDataARRAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AEE_amg_solver = jx_3tAPCTLDataAEEAMGSolver(pre_3tapctl_data);
   jx_ParAMGData   *AII_amg_solver = jx_3tAPCTLDataAIIAMGSolver(pre_3tapctl_data);      
   
   jx_ParCSRMatrix *ARR = jx_3tAPCTLDataARR(pre_3tapctl_data); 
   jx_ParCSRMatrix *AEE = jx_3tAPCTLDataAEE(pre_3tapctl_data);
   jx_ParCSRMatrix *AII = jx_3tAPCTLDataAII(pre_3tapctl_data);

   jx_ParVector    *WRR = jx_3tAPCTLDataWRR(pre_3tapctl_data);
   jx_ParVector    *WEE = jx_3tAPCTLDataWEE(pre_3tapctl_data);
   jx_ParVector    *WII = jx_3tAPCTLDataWII(pre_3tapctl_data);
   
   jx_ParVector    *JAC = jx_3tAPCTLDataJAC(pre_3tapctl_data);
   jx_ParVector    *RHS = jx_3tAPCTLDataRHS(pre_3tapctl_data);

   JX_Real *rhs_data = jx_VectorData(jx_ParVectorLocalVector(RHS));
   
   JX_Real *g_data   = jx_VectorData(jx_ParVectorLocalVector(g));
   JX_Real *w_data   = jx_VectorData(jx_ParVectorLocalVector(w));
   
   /* newly added, peghoty, 2012/02/16 */
   JX_Int use_fixedmode_R = jx_3tAPCTLDataUseFixedModeR(pre_3tapctl_data);
   JX_Int use_fixedmode_E = jx_3tAPCTLDataUseFixedModeE(pre_3tapctl_data);
   JX_Int use_fixedmode_I = jx_3tAPCTLDataUseFixedModeI(pre_3tapctl_data);
   
   JX_Int test_subls_iter = jx_3tAPCTLDataTestSubLSIter(pre_3tapctl_data);      

   /* local variables */
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int n = N / 3;
   JX_Int nplusn = 2*n;
   JX_Int i, sweep;

   jx_VectorData(jx_ParVectorLocalVector(WRR)) = w_data;
   jx_VectorData(jx_ParVectorLocalVector(WEE)) = w_data + n;
   jx_VectorData(jx_ParVectorLocalVector(WII)) = w_data + nplusn;
   
   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+n];
   }
    
   jx_ParVectorSetConstantValues(WEE, 0.0);
         
   if (AEE_relax_type == RELAX_AMG)
   {      
      if (use_fixedmode_E)
         jx_PAMGPrecond(AEE_amg_solver, AEE, RHS, WEE);
      else     
         jx_PAMGSolve(AEE_amg_solver, AEE, RHS, WEE);

      if (print_level == 2 || print_level == 3)
      { 
         if (use_fixedmode_E)
            jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_relax_maxit);
         else
            jx_printf(" APCTL-Solve == AEE AMG-Iter: %d\n", AEE_amg_solver->num_iterations);
      }                 
   }
   else if (AEE_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AEE_relax_maxit; sweep ++)
      {
         jx_PAMGRelax(AEE, RHS, NULL, 0, 0, 1.0, 1.0, WEE, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jx_printf(" APCTL-Solve == AEE JAC-Iter: %d\n", AEE_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AEE_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_E)
            jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
         else
            jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_amg_solver->num_iterations;
      }
      else if (AEE_relax_type == RELAX_WJACOBI)
      {
         jx_3tAPCTLDataNumIterAePrecond(pre_3tapctl_data) += AEE_relax_maxit;
      }
   }

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i];
   }      
   
   jx_ParVectorSetConstantValues(WRR, 0.0); 
     
   if (ARR_relax_type == RELAX_AMG)
   {  
      if (use_fixedmode_R)
         jx_PAMGPrecond(ARR_amg_solver, ARR, RHS, WRR);
      else          
         jx_PAMGSolve(ARR_amg_solver, ARR, RHS, WRR);
            
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_R)
            jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_relax_maxit);
         else
            jx_printf( " APCTL-Solve == ARR AMG-Iter: %d\n", ARR_amg_solver->num_iterations); 
      }         
   }
   else if (ARR_relax_type == RELAX_WJACOBI)
   {  
      for (sweep = 0; sweep < ARR_relax_maxit; sweep ++)
      {      
         jx_PAMGRelax(ARR, RHS, NULL, 0, 0, 1.0, 1.0, WRR, JAC);
      }
      if (print_level == 2 || print_level == 3)
      { 
         jx_printf(" APCTL-Solve == ARR JAC-Iter: %d\n", ARR_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (ARR_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_R)
            jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
         else
            jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_amg_solver->num_iterations;
      }
      else if (ARR_relax_type == RELAX_WJACOBI)
      {
         jx_3tAPCTLDataNumIterArPrecond(pre_3tapctl_data) += ARR_relax_maxit;
      }
   }      

   for (i = 0; i < n; i ++)
   {
      rhs_data[i] = g_data[i+nplusn];
   } 
           
   jx_ParVectorSetConstantValues(WII, 0.0);
     
   if (AII_relax_type == RELAX_AMG)
   {
      if (use_fixedmode_I)
         jx_PAMGPrecond(AII_amg_solver, AII, RHS, WII);
      else           
         jx_PAMGSolve(AII_amg_solver, AII, RHS, WII);
             
      if (print_level == 2 || print_level == 3) 
      {
         if (use_fixedmode_I)
            jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_relax_maxit);
         else
            jx_printf( " APCTL-Solve == AII AMG-Iter: %d\n", AII_amg_solver->num_iterations);
      }         
   }
   else if (AII_relax_type == RELAX_WJACOBI)
   {
      for (sweep = 0; sweep < AII_relax_maxit; sweep ++)
      {      
         jx_PAMGRelax(AII, RHS, NULL, 0, 0, 1.0, 1.0, WII, JAC); 
      }
      if (print_level == 2 || print_level == 3)
      { 
         jx_printf(" APCTL-Solve == AII JAC-Iter: %d\n", AII_relax_maxit);
      } 
   }

   if (test_subls_iter)
   {
      if (AII_relax_type == RELAX_AMG)
      {
         if (use_fixedmode_I)
            jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
         else
            jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_amg_solver->num_iterations;
      }
      else if (AII_relax_type == RELAX_WJACOBI)
      {
         jx_3tAPCTLDataNumIterAiPrecond(pre_3tapctl_data) += AII_relax_maxit;
      }
   }        

   return (0);
}
