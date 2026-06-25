//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  coarseop.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn jx_ParCSRMatrix *jx_3tApctlCoarseOperator
 * \brief Build interpolation operator for PCTL. 
 * \note The sparse pattern of ARR_s, AEE_s, AII_s should be exactly the same.
 *       If not, something needs to be done to guarantee this condition.
 *       For example, reorder each row ascendingly based on the clumn indices.
 * \author peghoty 
 * \date 2012/02/28
 */
jx_ParCSRMatrix *
jx_3tApctlCoarseOperator_mp( MPI_Comm          comm,
                             jx_ParCSRMatrix  *ARR, 
                             jx_ParCSRMatrix  *AEE, 
                             jx_ParCSRMatrix  *AII, 
                             jx_ParVector     *VRE, 
                             jx_ParVector     *VER, 
                             jx_ParVector     *VEI, 
                             jx_ParVector     *VIE,
                             jx_ParVector     *PRR, 
                             jx_ParVector     *PII )
{
   jx_ParCSRMatrix  *ACC = NULL;
   
   jx_CSRMatrix   *diagR = NULL;
   jx_CSRMatrix   *diagE = NULL;
   jx_CSRMatrix   *diagI = NULL;
   jx_CSRMatrix   *diagC = NULL;
   
   jx_CSRMatrix   *offdR = NULL;
   jx_CSRMatrix   *offdE = NULL;
   jx_CSRMatrix   *offdI = NULL;
   jx_CSRMatrix   *offdC = NULL;

   JX_Int n = jx_ParCSRMatrixGlobalNumRows(ARR);
  
   JX_Real *send_buf_r = NULL;
   JX_Real *send_buf_i = NULL;

   jx_ParCSRCommHandle  *comm_handle_R = NULL;
   jx_ParCSRCommHandle  *comm_handle_I = NULL;
   jx_ParCSRCommPkg	*comm_pkg      = NULL;
   
   JX_Int  myid;
   JX_Int  i, j, k;   
   JX_Int  index, start, end;
   JX_Int  num_sends;
   JX_Int  send_buf_size;
   JX_Int  local_size;
   JX_Int  num_cols_offd;
   JX_Real diag_add;
   
   JX_Real *pr_local_data = NULL;   
   JX_Real *pi_local_data = NULL;

   JX_Real *pr_other_data = NULL;   
   JX_Real *pi_other_data = NULL;
   
   JX_Int    *diag_ia = NULL;
   JX_Int    *diag_ja = NULL;
   JX_Int    *offd_ia = NULL;
   JX_Int    *offd_ja = NULL;   
   
   JX_Real *diag_aa_R = NULL;
   JX_Real *diag_aa_I = NULL;
   JX_Real *diag_aa_C = NULL;
   
   JX_Real *offd_aa_R = NULL;
   JX_Real *offd_aa_I = NULL;
   JX_Real *offd_aa_C = NULL;
   
   JX_Real *Vre_local_data = NULL;
   JX_Real *Ver_local_data = NULL;
   JX_Real *Vei_local_data = NULL;
   JX_Real *Vie_local_data = NULL;    
   
   JX_Int *row_starts = NULL;
   JX_Int *col_starts = NULL;
   JX_Int  num_nonzeros_diag;
   JX_Int  num_nonzeros_offd;    
 
   jx_MPI_Comm_rank(comm, &myid);   

   diagR = jx_ParCSRMatrixDiag(ARR);
   diagE = jx_ParCSRMatrixDiag(AEE);
   diagI = jx_ParCSRMatrixDiag(AII);
   offdR = jx_ParCSRMatrixOffd(ARR);
   offdE = jx_ParCSRMatrixOffd(AEE);
   offdI = jx_ParCSRMatrixOffd(AII); 
  
#if 1
   jx_CSRMatrixReorderColumnNumber12(diagR);
   jx_CSRMatrixReorderColumnNumber12(diagE);
   jx_CSRMatrixReorderColumnNumber12(diagI);
   jx_CSRMatrixReorderColumnNumberAll(offdR);
   jx_CSRMatrixReorderColumnNumberAll(offdE);
   jx_CSRMatrixReorderColumnNumberAll(offdI);
#endif

   num_cols_offd = jx_CSRMatrixNumCols(offdE);
   num_nonzeros_diag = jx_CSRMatrixNumNonzeros(diagE);
   num_nonzeros_offd = jx_CSRMatrixNumNonzeros(offdE);
   jx_ParCSRMatrixGetRowPartitioning(AEE, &row_starts);
   jx_ParCSRMatrixGetColPartitioning(AEE, &col_starts);
   ACC = jx_ParCSRMatrixCreate( comm, n, n, row_starts, col_starts,
                                num_cols_offd, num_nonzeros_diag, num_nonzeros_offd );
                          
   jx_ParCSRMatrixInitialize(ACC);
   jx_ParCSRMatrixCopy(AEE, ACC, 1); 

   diagC = jx_ParCSRMatrixDiag(ACC);
   offdC = jx_ParCSRMatrixOffd(ACC); 

   comm_pkg = jx_ParCSRMatrixCommPkg(ARR);
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(ARR);
      comm_pkg = jx_ParCSRMatrixCommPkg(ARR); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   send_buf_size = jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   send_buf_r = jx_CTAlloc(JX_Real, send_buf_size);
   send_buf_i = jx_CTAlloc(JX_Real, send_buf_size);
   pr_local_data = jx_VectorData(jx_ParVectorLocalVector(PRR));
   pi_local_data = jx_VectorData(jx_ParVectorLocalVector(PII));
   
   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      end   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for (j = start; j < end; j ++)
      {
         send_buf_r[index] = pr_local_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         send_buf_i[index] = pi_local_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         index ++;
      }
   }

   num_cols_offd = jx_CSRMatrixNumCols(offdR);
   
   pr_other_data = jx_CTAlloc(JX_Real, num_cols_offd);
   pi_other_data = jx_CTAlloc(JX_Real, num_cols_offd);

   comm_handle_R = jx_ParCSRCommHandleCreate(1, comm_pkg, send_buf_r, pr_other_data);
   comm_handle_I = jx_ParCSRCommHandleCreate(1, comm_pkg, send_buf_i, pi_other_data);

   diag_ia = jx_CSRMatrixI(diagC);
   diag_ja = jx_CSRMatrixJ(diagC);

   diag_aa_R = jx_CSRMatrixData(diagR);
   diag_aa_I = jx_CSRMatrixData(diagI);
   diag_aa_C = jx_CSRMatrixData(diagC);
   
   local_size = jx_CSRMatrixNumRows(diagC);

   for (i = 0; i < local_size; i ++)
   {
      for (k = diag_ia[i]; k < diag_ia[i+1]; k ++)
      {
         j = diag_ja[k];
         /*diag_aa_C[k] += (   diag_aa_R[k]*pr_local_data[i]*pr_local_data[j]
                           + diag_aa_I[k]*pi_local_data[i]*pi_local_data[j] );*/
         diag_aa_C[k] += (diag_aa_R[k] * pr_local_data[i] * pr_local_data[j]);
         diag_aa_C[k] += (diag_aa_I[k] * pi_local_data[i] * pi_local_data[j]);
      }
   }

   jx_ParCSRCommHandleDestroy(comm_handle_R);
   jx_ParCSRCommHandleDestroy(comm_handle_I);

   offd_ia = jx_CSRMatrixI(offdC);
   offd_ja = jx_CSRMatrixJ(offdC);     

   offd_aa_R = jx_CSRMatrixData(offdR);
   offd_aa_I = jx_CSRMatrixData(offdI);
   offd_aa_C = jx_CSRMatrixData(offdC);   

   for (i = 0; i < local_size; i ++)
   {
      for (k = offd_ia[i]; k < offd_ia[i+1]; k ++)
      {
         j = offd_ja[k];
         /*offd_aa_C[k] += (   offd_aa_R[k]*pr_local_data[i]*pr_other_data[j]
                           + offd_aa_I[k]*pi_local_data[i]*pi_other_data[j] );*/
         offd_aa_C[k] += (offd_aa_R[k] * pr_local_data[i] * pr_other_data[j]);
         offd_aa_C[k] += (offd_aa_I[k] * pi_local_data[i] * pi_other_data[j]);
      }
   }

   Vre_local_data = jx_VectorData(jx_ParVectorLocalVector(VRE));
   Ver_local_data = jx_VectorData(jx_ParVectorLocalVector(VER));
   Vei_local_data = jx_VectorData(jx_ParVectorLocalVector(VEI));
   Vie_local_data = jx_VectorData(jx_ParVectorLocalVector(VIE));    

   for (i = 0; i < local_size; i ++)
   {
      k = diag_ia[i];
      
      /*diag_add =   (Vre_local_data[i] + Ver_local_data[i])*pr_local_data[i]
                 + (Vei_local_data[i] + Vie_local_data[i])*pi_local_data[i];*/
      diag_add = 0.0;
      diag_add += ((Vre_local_data[i] + Ver_local_data[i]) * pr_local_data[i]);
      diag_add += ((Vei_local_data[i] + Vie_local_data[i]) * pi_local_data[i]);
      
      diag_aa_C[k] += diag_add;
   } 
   
   jx_TFree(send_buf_r);
   jx_TFree(send_buf_i);
   jx_TFree(pr_other_data);
   jx_TFree(pi_other_data);

   return (ACC);
}

jx_ParCSRMatrix *
jx_3tApctlmgCoarseOperator_mp( MPI_Comm          comm,
                               JX_Int            ng,
                               jx_ParCSRMatrix **ARR, 
                               jx_ParCSRMatrix  *AEE, 
                               jx_ParCSRMatrix  *AII, 
                               jx_ParVector    **VRE, 
                               jx_ParVector    **VER, 
                               jx_ParVector     *VEI, 
                               jx_ParVector     *VIE,
                               jx_ParVector    **PRR, 
                               jx_ParVector     *PII )
{
   jx_ParCSRMatrix *ACC = NULL;
 
   jx_CSRMatrix *diagE = NULL;
   jx_CSRMatrix *diagI = NULL;
   jx_CSRMatrix *diagC = NULL;
   jx_CSRMatrix *offdE = NULL;
   jx_CSRMatrix *offdI = NULL;
   jx_CSRMatrix *offdC = NULL;

   JX_Int n = jx_ParCSRMatrixGlobalNumRows(ARR[0]);

   JX_Real **send_buf_r = jx_CTAlloc(JX_Real *, ng);

   JX_Real *send_buf_i = NULL;

   jx_ParCSRCommHandle **comm_handle_R = jx_CTAlloc(jx_ParCSRCommHandle *, ng);

   jx_ParCSRCommHandle *comm_handle_I = NULL;

   jx_ParCSRCommPkg	*comm_pkg = NULL;
 
   JX_Int  myid;
   JX_Int  i, j, k, gidx;
   JX_Int  index, start, end;
   JX_Int  num_sends;
   JX_Int  send_buf_size;
   JX_Int  local_size;
   JX_Int  num_cols_offd;
   JX_Real diag_add;
 
   JX_Real *pi_local_data = NULL;

   JX_Real **pr_other_data = jx_CTAlloc(JX_Real *, ng);

   JX_Real *pi_other_data = NULL;
 
   JX_Int *diag_ia = NULL;
   JX_Int *diag_ja = NULL;
   JX_Int *offd_ia = NULL;
   JX_Int *offd_ja = NULL;

   JX_Real *diag_aa_I = NULL;
   JX_Real *diag_aa_C = NULL;
   JX_Real *offd_aa_I = NULL;
   JX_Real *offd_aa_C = NULL;

   JX_Real *Vei_local_data = NULL;
   JX_Real *Vie_local_data = NULL;

   JX_Int *row_starts = NULL;
   JX_Int *col_starts = NULL;

   JX_Int num_nonzeros_diag;
   JX_Int num_nonzeros_offd;
 
   jx_MPI_Comm_rank(comm, &myid);

#if 0
   for (gidx = 0; gidx < ng; gidx ++)
   {
      jx_CSRMatrixReorderColumnNumber12(jx_ParCSRMatrixDiag(ARR[gidx]));
      jx_CSRMatrixReorderColumnNumberAll(jx_ParCSRMatrixOffd(ARR[gidx]));
   }
   jx_CSRMatrixReorderColumnNumber12(diagE);
   jx_CSRMatrixReorderColumnNumber12(diagI);
   jx_CSRMatrixReorderColumnNumberAll(offdE);
   jx_CSRMatrixReorderColumnNumberAll(offdI);
#endif

   diagE = jx_ParCSRMatrixDiag(AEE);
   offdE = jx_ParCSRMatrixOffd(AEE);
   diagI = jx_ParCSRMatrixDiag(AII);
   offdI = jx_ParCSRMatrixOffd(AII); 

   num_cols_offd = jx_CSRMatrixNumCols(offdE);
   num_nonzeros_diag = jx_CSRMatrixNumNonzeros(diagE);
   num_nonzeros_offd = jx_CSRMatrixNumNonzeros(offdE);
   jx_ParCSRMatrixGetRowPartitioning(AEE, &row_starts);
   jx_ParCSRMatrixGetColPartitioning(AEE, &col_starts);
   ACC = jx_ParCSRMatrixCreate( comm, n, n, row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd );
   jx_ParCSRMatrixInitialize(ACC);
   jx_ParCSRMatrixCopy(AEE, ACC, 1);
   diagC = jx_ParCSRMatrixDiag(ACC);
   offdC = jx_ParCSRMatrixOffd(ACC);

   comm_pkg = jx_ParCSRMatrixCommPkg(ARR[0]);
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(ARR[0]);
      comm_pkg = jx_ParCSRMatrixCommPkg(ARR[0]);
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   send_buf_size = jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   for (gidx = 0; gidx < ng; gidx ++) send_buf_r[gidx] = jx_CTAlloc(JX_Real, send_buf_size);
   send_buf_i = jx_CTAlloc(JX_Real, send_buf_size);
   pi_local_data = jx_VectorData(jx_ParVectorLocalVector(PII));

   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
      end   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for (j = start; j < end; j ++)
      {
         for (gidx = 0; gidx < ng; gidx ++)
         {
            send_buf_r[gidx][index] = jx_VectorData(jx_ParVectorLocalVector(PRR[gidx]))[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
         send_buf_i[index] = pi_local_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         index ++;
      }
   }

   num_cols_offd = jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(ARR[0]));

   for (gidx = 0; gidx < ng; gidx ++) pr_other_data[gidx] = jx_CTAlloc(JX_Real, num_cols_offd);
   pi_other_data = jx_CTAlloc(JX_Real, num_cols_offd);

   for (gidx = 0; gidx < ng; gidx ++)
   {
      comm_handle_R[gidx] = jx_ParCSRCommHandleCreate(1, comm_pkg, send_buf_r[gidx], pr_other_data[gidx]);
   }
   comm_handle_I = jx_ParCSRCommHandleCreate(1, comm_pkg, send_buf_i, pi_other_data);

   diag_ia = jx_CSRMatrixI(diagC);
   diag_ja = jx_CSRMatrixJ(diagC);

   diag_aa_I = jx_CSRMatrixData(diagI);
   diag_aa_C = jx_CSRMatrixData(diagC);
 
   local_size = jx_CSRMatrixNumRows(diagC);

   for (i = 0; i < local_size; i ++)
   {
      for (k = diag_ia[i]; k < diag_ia[i+1]; k ++)
      {
         j = diag_ja[k];
         for (gidx = 0; gidx < ng; gidx ++)
         {
            diag_aa_C[k] += ( jx_CSRMatrixData(jx_ParCSRMatrixDiag(ARR[gidx]))[k] *
                                jx_VectorData(jx_ParVectorLocalVector(PRR[gidx]))[i] *
                                  jx_VectorData(jx_ParVectorLocalVector(PRR[gidx]))[j] );
         }
         diag_aa_C[k] += ( diag_aa_I[k] * pi_local_data[i] * pi_local_data[j] );
      }
   }

   for (gidx = 0; gidx < ng; gidx ++) jx_ParCSRCommHandleDestroy(comm_handle_R[gidx]);
   jx_TFree(comm_handle_R);
   jx_ParCSRCommHandleDestroy(comm_handle_I);

   offd_ia = jx_CSRMatrixI(offdC);
   offd_ja = jx_CSRMatrixJ(offdC);

   offd_aa_I = jx_CSRMatrixData(offdI);
   offd_aa_C = jx_CSRMatrixData(offdC);   

   for (i = 0; i < local_size; i ++)
   {
      for (k = offd_ia[i]; k < offd_ia[i+1]; k ++)
      {
         j = offd_ja[k];
         for (gidx = 0; gidx < ng; gidx ++)
         {
            offd_aa_C[k] += ( jx_CSRMatrixData(jx_ParCSRMatrixOffd(ARR[gidx]))[k] *
                                jx_VectorData(jx_ParVectorLocalVector(PRR[gidx]))[i] * pr_other_data[gidx][j] );
         }
         offd_aa_C[k] += ( offd_aa_I[k] * pi_local_data[i] * pi_other_data[j] );
      }
   }

   Vei_local_data = jx_VectorData(jx_ParVectorLocalVector(VEI));
   Vie_local_data = jx_VectorData(jx_ParVectorLocalVector(VIE));

   for (i = 0; i < local_size; i ++)
   {
      k = diag_ia[i];
      diag_add = 0.0;
      for (gidx = 0; gidx < ng; gidx ++)
      {
         diag_add += ((jx_VectorData(jx_ParVectorLocalVector(VRE[gidx]))[i] +
                         jx_VectorData(jx_ParVectorLocalVector(VER[gidx]))[i]) *
                           jx_VectorData(jx_ParVectorLocalVector(PRR[gidx]))[i]);
      }
      diag_add += (Vei_local_data[i] + Vie_local_data[i]) * pi_local_data[i];
      diag_aa_C[k] += diag_add;
   }

   for (gidx = 0; gidx < ng; gidx ++) jx_TFree(send_buf_r[gidx]);
   jx_TFree(send_buf_r);
   jx_TFree(send_buf_i);
   for (gidx = 0; gidx < ng; gidx ++) jx_TFree(pr_other_data[gidx]);
   jx_TFree(pr_other_data);
   jx_TFree(pi_other_data);

   return (ACC);
}

/*!
 * \fn jx_ParCSRMatrix *jx_3tApctlCoarseOperator_sp
 * \brief Build interpolation operator for PCTL(single processor case). 
 * \note The sparse pattern of ARR_s, AEE_s, AII_s should be exactly the same.
 *       If not, something needs to be done to guarantee this condition.
 *       For example, reorder each row ascendingly based on the clumn indices.
 * \author peghoty 
 * \date 2011/09/26
 */
jx_ParCSRMatrix *
jx_3tApctlCoarseOperator_sp( MPI_Comm       comm,
                             jx_CSRMatrix  *ARR_s, 
                             jx_CSRMatrix  *AEE_s, 
                             jx_CSRMatrix  *AII_s, 
                             jx_Vector     *VRE_s, 
                             jx_Vector     *VER_s, 
                             jx_Vector     *VEI_s, 
                             jx_Vector     *VIE_s,
                             jx_ParVector  *PRR,
                             jx_ParVector  *PII  )
{
   jx_ParCSRMatrix  *ACC   = NULL;
   jx_CSRMatrix     *ACC_s = NULL;

   JX_Int    *ia = NULL;
   JX_Int    *ja = NULL;
   JX_Real *aa = NULL;   
   
   JX_Real *aa_R = jx_CSRMatrixData(ARR_s);
   JX_Real *aa_I = jx_CSRMatrixData(AII_s);

   JX_Real *Vre_data = jx_VectorData(VRE_s);
   JX_Real *Ver_data = jx_VectorData(VER_s);
   JX_Real *Vei_data = jx_VectorData(VEI_s);
   JX_Real *Vie_data = jx_VectorData(VIE_s); 
   
   JX_Real *pr_data = jx_VectorData(jx_ParVectorLocalVector(PRR));
   JX_Real *pi_data = jx_VectorData(jx_ParVectorLocalVector(PII));

   JX_Int i, j, k;  
   JX_Int n, nz;
   JX_Real diag_add = 0.0;

#if 1
   jx_CSRMatrixReorderColumnNumber12(ARR_s);
   jx_CSRMatrixReorderColumnNumber12(AEE_s);
   jx_CSRMatrixReorderColumnNumber12(AII_s);  
#endif

   n  = jx_CSRMatrixNumRows(AEE_s);
   nz = jx_CSRMatrixNumNonzeros(AEE_s);
   ACC_s = jx_CSRMatrixCreate(n,n,nz);
   jx_CSRMatrixInitialize(ACC_s);
   jx_CSRMatrixCopy(AEE_s, ACC_s, 1);
   
   ia = jx_CSRMatrixI(ACC_s);
   ja = jx_CSRMatrixJ(ACC_s);
   aa = jx_CSRMatrixData(ACC_s); 

   for (i = 0; i < n; i ++)
   {
      for (k = ia[i]; k < ia[i+1]; k ++)
      {
         j = ja[k];
         aa[k] += ( aa_R[k]*pr_data[i]*pr_data[j] + aa_I[k]*pi_data[i]*pi_data[j] );
      }
   }

   for (i = 0; i < n; i ++)
   {
      k = ia[i];
      diag_add = (Vre_data[i] + Ver_data[i])*pr_data[i] + (Vei_data[i] + Vie_data[i])*pi_data[i];
      aa[k] += diag_add;
   } 

   ACC = jx_CSRMatrixToParCSRMatrix_sp(comm, ACC_s);

   jx_TFree(ACC_s); 
   
   return (ACC);
}
