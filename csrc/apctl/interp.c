//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  interp.c  
 *
 *  Date: 2012/03/01 
 *  Created by peghoty
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn jx_ParCSRMatrix *jx_3tApctlInterpolation
 * \brief Build interpolation operator for PCTL. 
 * \author peghoty 
 * \date 2012/02/29
 */
jx_ParCSRMatrix *
jx_3tApctlInterpolation( MPI_Comm         comm, 
                         MPI_Comm         comm_x,
                         JX_Int              N,
                         JX_Int              groupid_x,
                         JX_Int             *row_starts, 
                         JX_Int             *col_starts,
                         jx_ParVector    *WEE )
{        
   jx_ParCSRMatrix *P     = NULL;     
   jx_CSRMatrix    *P_rec = NULL;
   
   JX_Int *ip = NULL;
   JX_Int *jp = NULL;
   JX_Real *pp = NULL;
   
   jx_Vector *W = NULL;
   
   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;
   
   JX_Int start;
   JX_Int first_col_diag;
   JX_Int last_col_diag;  
   
   JX_Int owns_row_starts;
   JX_Int owns_col_starts; 
   
   JX_Int i, n = N / 3;
   JX_Int nplusn = 2*n;
   JX_Int myid, nprocs;
    
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);
   
   
   if (groupid_x == 0)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jx_CSRMatrixInitialize(P_rec);
      pp = jx_CSRMatrixData(P_rec);
      ip = jx_CSRMatrixI(P_rec); 
      jp = jx_CSRMatrixJ(P_rec);
      
      W = jx_ParVectorLocalVector(WEE);
      
      start = row_starts[myid];
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jx_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }
   else if (groupid_x == 1)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jx_CSRMatrixInitialize(P_rec);
      pp = jx_CSRMatrixData(P_rec);
      ip = jx_CSRMatrixI(P_rec); 
      jp = jx_CSRMatrixJ(P_rec);
            
      start = row_starts[myid] - n;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = 1.0;
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;      
   }
   else if (groupid_x == 2)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jx_CSRMatrixInitialize(P_rec);
      pp = jx_CSRMatrixData(P_rec);
      ip = jx_CSRMatrixI(P_rec); 
      jp = jx_CSRMatrixJ(P_rec);
      
      W = jx_ParVectorLocalVector(WEE);
      
      start = row_starts[myid] - nplusn;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jx_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }

   if (row_starts) 
   {
      owns_row_starts = 0;
   }
   else
   {
      owns_row_starts = 1;
   }
   
   if (col_starts) 
   {
      owns_col_starts = 0;
   }
   else
   {
      owns_col_starts = 1;
   }
        
   P = jx_ParCSRMatrixCreate(comm, N, n, row_starts, col_starts, 0, 0, 0);
   jx_ParCSRMatrixSetRowStartsOwner(P, owns_row_starts);
   jx_ParCSRMatrixSetColStartsOwner(P, owns_col_starts);
 
   first_col_diag = jx_ParCSRMatrixColStarts(P)[myid];
   last_col_diag  = jx_ParCSRMatrixColStarts(P)[myid+1] - 1;
   jx_GenerateDiagAndOffd(P_rec, P, first_col_diag, last_col_diag);

   jx_CSRMatrixDestroy(P_rec);  

   return (P); 
}

jx_ParCSRMatrix *
jx_3tApctlmgInterpolation( MPI_Comm         comm, 
                           MPI_Comm         comm_x,
                           JX_Int              N,
                           JX_Int              groupid_x,
                           JX_Int              ng,
                           JX_Int             *row_starts, 
                           JX_Int             *col_starts,
                           jx_ParVector    *WEE )
{
   jx_ParCSRMatrix *P     = NULL;
   jx_CSRMatrix    *P_rec = NULL;

   JX_Int *ip = NULL;
   JX_Int *jp = NULL;
   JX_Real *pp = NULL;

   jx_Vector *W = NULL;

   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;

   JX_Int start;
   JX_Int first_col_diag;
   JX_Int last_col_diag;

   JX_Int owns_row_starts;
   JX_Int owns_col_starts;

   JX_Int i, n = N / (ng + 2);
   JX_Int e_pos = ng * n;
   JX_Int nplusn = e_pos + n;
   JX_Int myid, nprocs;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (groupid_x < ng)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jx_CSRMatrixInitialize(P_rec);
      pp = jx_CSRMatrixData(P_rec);
      ip = jx_CSRMatrixI(P_rec);
      jp = jx_CSRMatrixJ(P_rec);
      W = jx_ParVectorLocalVector(WEE);
      start = row_starts[myid] - groupid_x * n;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jx_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }
   else if (groupid_x == ng)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jx_CSRMatrixInitialize(P_rec);
      pp = jx_CSRMatrixData(P_rec);
      ip = jx_CSRMatrixI(P_rec); 
      jp = jx_CSRMatrixJ(P_rec);
      start = row_starts[myid] - e_pos;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = 1.0;
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }
   else if (groupid_x == ng+1)
   {
      num_rows = row_starts[myid+1] - row_starts[myid];
      num_cols = n;
      num_nonzeros = num_rows;
      P_rec = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
      jx_CSRMatrixInitialize(P_rec);
      pp = jx_CSRMatrixData(P_rec);
      ip = jx_CSRMatrixI(P_rec); 
      jp = jx_CSRMatrixJ(P_rec);
      W = jx_ParVectorLocalVector(WEE);
      start = row_starts[myid] - nplusn;
      for (i = 0; i < num_rows; i ++)
      {
         ip[i] = i;
         pp[i] = jx_VectorData(W)[i];
         jp[i] = start + i;
      }
      ip[num_rows] = num_rows;
   }

   if (row_starts)
   {
      owns_row_starts = 0;
   }
   else
   {
      owns_row_starts = 1;
   }

   if (col_starts)
   {
      owns_col_starts = 0;
   }
   else
   {
      owns_col_starts = 1;
   }

   P = jx_ParCSRMatrixCreate(comm, N, n, row_starts, col_starts, 0, 0, 0);
   jx_ParCSRMatrixSetRowStartsOwner(P, owns_row_starts);
   jx_ParCSRMatrixSetColStartsOwner(P, owns_col_starts);
 
   first_col_diag = jx_ParCSRMatrixColStarts(P)[myid];
   last_col_diag  = jx_ParCSRMatrixColStarts(P)[myid+1] - 1;
   jx_GenerateDiagAndOffd(P_rec, P, first_col_diag, last_col_diag);

   jx_CSRMatrixDestroy(P_rec);

   return (P);
}

/*!
 * \fn JX_Int x_3tAPCTLParaInterpVec
 * \brief 利用进程组上的并行向量生成两个基于所有进程的并行向量(PRR, PII). 
 * \author peghoty 
 * \date 2012/02/29
 */
JX_Int 
jx_3tAPCTLParaInterpVec( MPI_Comm       comm,
                         MPI_Comm       comm_x, 
                         JX_Int            groupid_x,
                         JX_Int           *vec_starts, 
                         jx_ParVector  *WEE, 
                         jx_ParVector **PRR_ptr, 
                         jx_ParVector **PII_ptr )
{
   JX_Int nprocs;
   JX_Int npeachgroup, rootid_R, rootid_I;
  
   jx_ParVector *PRR = NULL;
   jx_ParVector *PII = NULL;
   
   jx_Vector *PRR_s = NULL;
   jx_Vector *PII_s = NULL;   
 
   jx_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / 3;
   rootid_R = 0; 
   rootid_I = 2*npeachgroup;
   
   if (groupid_x == 0)
   { 
      PRR_s = jx_ParVectorToVectorAll(WEE);
   }
   else if (groupid_x == 2)
   {    
      PII_s = jx_ParVectorToVectorAll(WEE);
   }

   PRR = jx_VectorToParVector_FromGivenPro(comm, rootid_R, PRR_s, vec_starts);
   PII = jx_VectorToParVector_FromGivenPro(comm, rootid_I, PII_s, vec_starts);

   if (groupid_x == 0)
   {
      jx_SeqVectorDestroy(PRR_s);
   }
   else if (groupid_x == 2)
   {
      jx_SeqVectorDestroy(PII_s);
   }
   
   *PRR_ptr = PRR;
   *PII_ptr = PII;

   return (0);
}

JX_Int 
jx_3tAPCTLmgParaInterpVec( MPI_Comm        comm,
                           MPI_Comm        comm_x, 
                           JX_Int            groupid_x,
                           JX_Int           *vec_starts, 
                           JX_Int          ng,
                           jx_ParVector   *WEE, 
                           jx_ParVector ***PRR_ptr, 
                           jx_ParVector  **PII_ptr )
{
   JX_Int nprocs, gidx, npeachgroup;

   jx_ParVector **PRR = jx_CTAlloc(jx_ParVector *, ng);

   jx_ParVector *PII = NULL;

   jx_Vector *PRR_s = NULL;
   jx_Vector *PII_s = NULL;
 
   jx_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / (ng + 2);

   if (groupid_x < ng)
   {
      PRR_s = jx_ParVectorToVectorAll(WEE);
   }
   else if (groupid_x == ng+1)
   {
      PII_s = jx_ParVectorToVectorAll(WEE);
   }

   for (gidx = 0; gidx < ng; gidx ++) PRR[gidx] = jx_VectorToParVector_FromGivenPro(comm, gidx*npeachgroup, PRR_s, vec_starts);
   PII = jx_VectorToParVector_FromGivenPro(comm, (ng+1)*npeachgroup, PII_s, vec_starts);

   if (groupid_x < ng)
   {
      jx_SeqVectorDestroy(PRR_s);
   }
   else if (groupid_x == ng+1)
   {
      jx_SeqVectorDestroy(PII_s);
   }

   *PRR_ptr = PRR;
   *PII_ptr = PII;

   return (0);
}
