//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  transf.c
 *  Date: 2012/03/05
 *
 *  Created by peghoty
 *
 */ 

#include "jx_pamg.h"
#include "jx_apctl.h"

/*!
 * \fn JX_Int jx_APCTLKrylovSolBack4Jasmin
 * \brief 收集解向量函数. 
 * \note 三个进程组的根进程各自转换一个并行向量. 效率比 jx_APCTLKrylovSolBack4Jasmin2 要高. 
 * \author peghoty 
 * \date 2012/02/27
 */
JX_Int 
jx_APCTLKrylovSolBack4Jasmin( MPI_Comm      comm,
                              MPI_Comm      comm_x, 
                              JX_Int           groupid_x, 
                              jx_ParVector *par_sol, 
                              jx_ParVector *uR_p, 
                              jx_ParVector *uE_p, 
                              jx_ParVector *uI_p )
{
   JX_Int myid, nprocs;
   JX_Int rootid_R, rootid_E, rootid_I;
   JX_Int N = jx_ParVectorGlobalSize(par_sol);  
   JX_Int n = N / 3;
   JX_Int npeachgroup;
  
   jx_ParVector *tmpR = NULL;
   jx_ParVector *tmpE = NULL;
   jx_ParVector *tmpI = NULL;
   
   jx_Vector *tmpR_s = NULL;
   jx_Vector *tmpE_s = NULL;
   jx_Vector *tmpI_s = NULL;   
   
   JX_Int *partition = jx_ParVectorPartitioning(par_sol);
   jx_Vector *locvec = jx_ParVectorLocalVector(par_sol);
   JX_Real temp_adrress = 0.0;
 
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / 3;

   // 三个进程组的根进程号
   rootid_R = 0; 
   rootid_E = npeachgroup;
   rootid_I = 2*npeachgroup;
   
   if (groupid_x == 0)
   {
      // 将光子进程组上的解向量收集为 rootid_R 号进程上的一个串行向量 tmpR_s
      tmpR = jx_ParVectorCreate(comm_x, n, partition);
      tmpR->local_vector->data = &temp_adrress;
      jx_ParVectorInitialize(tmpR);
      jx_ParVectorSetDataOwner(tmpR, 0);
      jx_ParVectorSetPartitioningOwner(tmpR, 0);
      tmpR->local_vector->data = locvec->data;
      
      tmpR_s = jx_ParVectorToVectorAll(tmpR);
   }   
   else if (groupid_x == 1)
   {
      // 将电子进程组上的解向量收集为 rootid_E 号进程上的一个串行向量 tmpE_s
      tmpE = jx_ParVectorCreate(comm_x, n, partition);
      tmpE->local_vector->data = &temp_adrress;
      jx_ParVectorInitialize(tmpE);
      jx_ParVectorSetDataOwner(tmpE, 0);
      jx_ParVectorSetPartitioningOwner(tmpE, 0);
      tmpE->local_vector->data = locvec->data;
      
      tmpE_s = jx_ParVectorToVectorAll(tmpE);
   }
   else if (groupid_x == 2)
   {
      // 将离子进程组上的解向量收集为 rootid_I 号进程上的一个串行向量 tmpI_s
      tmpI = jx_ParVectorCreate(comm_x, n, partition);
      tmpI->local_vector->data = &temp_adrress;
      jx_ParVectorInitialize(tmpI);
      jx_ParVectorSetDataOwner(tmpI, 0);
      jx_ParVectorSetPartitioningOwner(tmpI, 0);
      tmpI->local_vector->data = locvec->data;
      
      tmpI_s = jx_ParVectorToVectorAll(tmpI);
   }

   // 将三个串行向量转化为并行向量
   jx_VectorToParVector_Allocated_FromGivenPro(comm, rootid_R, tmpR_s, jx_ParVectorPartitioning(uR_p), uR_p); 
   jx_VectorToParVector_Allocated_FromGivenPro(comm, rootid_E, tmpE_s, jx_ParVectorPartitioning(uE_p), uE_p);
   jx_VectorToParVector_Allocated_FromGivenPro(comm, rootid_I, tmpI_s, jx_ParVectorPartitioning(uI_p), uI_p);
   
   // 释放三个串行向量的内存
   if (groupid_x == 0)
   {
      jx_SeqVectorDestroy(tmpR_s);
      jx_TFree(jx_ParVectorLocalVector(tmpR));
      jx_ParVectorDestroy(tmpR);      
   }
   else if (groupid_x == 1)
   {
      jx_SeqVectorDestroy(tmpE_s);
      jx_TFree(jx_ParVectorLocalVector(tmpE));
      jx_ParVectorDestroy(tmpE);      
   }
   else if (groupid_x == 2)
   {
      jx_SeqVectorDestroy(tmpI_s);
      jx_TFree(jx_ParVectorLocalVector(tmpI));
      jx_ParVectorDestroy(tmpI);      
   }

   return (0);
}

JX_Int 
jx_APCTLKrylovSolBack4mgJasmin( MPI_Comm      comm,
                                MPI_Comm      comm_x,
                                JX_Int           groupid_x,
                                JX_Int           ng,
                                jx_ParVector  *par_sol,
                                jx_ParVector **uR_p,
                                jx_ParVector  *uE_p,
                                jx_ParVector  *uI_p )
{
   JX_Int myid, nprocs;
   JX_Int N = jx_ParVectorGlobalSize(par_sol);
   JX_Int n = N / (ng + 2);
   JX_Int npeachgroup, gidx;

   jx_ParVector *tmpR = NULL;
   jx_ParVector *tmpE = NULL;
   jx_ParVector *tmpI = NULL;

   jx_Vector *tmpR_s = NULL;
   jx_Vector *tmpE_s = NULL;
   jx_Vector *tmpI_s = NULL;
 
   JX_Int *partition = jx_ParVectorPartitioning(par_sol);
   jx_Vector *locvec = jx_ParVectorLocalVector(par_sol);
   JX_Real temp_adrress = 0.0;
 
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   npeachgroup = nprocs / (ng + 2);

   if (groupid_x < ng)
   {
      tmpR = jx_ParVectorCreate(comm_x, n, partition);
      tmpR->local_vector->data = &temp_adrress;
      jx_ParVectorInitialize(tmpR);
      jx_ParVectorSetDataOwner(tmpR, 0);
      jx_ParVectorSetPartitioningOwner(tmpR, 0);
      tmpR->local_vector->data = locvec->data;
      tmpR_s = jx_ParVectorToVectorAll(tmpR);
   }
   else if (groupid_x == ng)
   {
      tmpE = jx_ParVectorCreate(comm_x, n, partition);
      tmpE->local_vector->data = &temp_adrress;
      jx_ParVectorInitialize(tmpE);
      jx_ParVectorSetDataOwner(tmpE, 0);
      jx_ParVectorSetPartitioningOwner(tmpE, 0);
      tmpE->local_vector->data = locvec->data;
      tmpE_s = jx_ParVectorToVectorAll(tmpE);
   }
   else if (groupid_x == ng+1)
   {
      tmpI = jx_ParVectorCreate(comm_x, n, partition);
      tmpI->local_vector->data = &temp_adrress;
      jx_ParVectorInitialize(tmpI);
      jx_ParVectorSetDataOwner(tmpI, 0);
      jx_ParVectorSetPartitioningOwner(tmpI, 0);
      tmpI->local_vector->data = locvec->data;
      tmpI_s = jx_ParVectorToVectorAll(tmpI);
   }

   // 将三个串行向量转化为并行向量
   for (gidx = 0; gidx < ng; gidx ++)
   {
      jx_VectorToParVector_Allocated_FromGivenPro(comm, gidx*npeachgroup, tmpR_s, jx_ParVectorPartitioning(uR_p[gidx]), uR_p[gidx]);
   }
   jx_VectorToParVector_Allocated_FromGivenPro(comm, ng*npeachgroup, tmpE_s, jx_ParVectorPartitioning(uE_p), uE_p);
   jx_VectorToParVector_Allocated_FromGivenPro(comm, (ng+1)*npeachgroup, tmpI_s, jx_ParVectorPartitioning(uI_p), uI_p);
 
   // 释放三个串行向量的内存
   if (groupid_x < ng)
   {
      jx_SeqVectorDestroy(tmpR_s);
      jx_TFree(jx_ParVectorLocalVector(tmpR));
      jx_ParVectorDestroy(tmpR);
   }
   else if (groupid_x == ng)
   {
      jx_SeqVectorDestroy(tmpE_s);
      jx_TFree(jx_ParVectorLocalVector(tmpE));
      jx_ParVectorDestroy(tmpE);
   }
   else if (groupid_x == ng+1)
   {
      jx_SeqVectorDestroy(tmpI_s);
      jx_TFree(jx_ParVectorLocalVector(tmpI));
      jx_ParVectorDestroy(tmpI);
   }

   return (0);
}

/*!
 * \fn JX_Int jx_APCTLKrylovSolBack4Jasmin2
 * \brief 收集解向量函数. 
 * \note 从 0 号进程分别转换三个并行向量. 效率不如 jx_APCTLKrylovSolBack4Jasmin 高.
 * \author peghoty 
 * \date 2012/02/27
 */
JX_Int 
jx_APCTLKrylovSolBack4Jasmin2( MPI_Comm      comm,
                               jx_ParVector *par_sol, 
                               jx_ParVector *uR_p, 
                               jx_ParVector *uE_p, 
                               jx_ParVector *uI_p )
{
   JX_Int N = jx_ParVectorGlobalSize(par_sol);  
   JX_Int n = N / 3;
   
   JX_Real temp_adrress = 0.0;
   
   jx_Vector *tmpR_s = NULL;
   jx_Vector *tmpE_s = NULL;
   jx_Vector *tmpI_s = NULL;  
   
   jx_Vector *sol = NULL;
   
   sol = jx_ParVectorToVectorAll(par_sol);
   
   tmpR_s = jx_SeqVectorCreate(n);
   tmpR_s->data = &temp_adrress;
   jx_SeqVectorInitialize(tmpR_s);
   tmpR_s->data = sol->data;
   
   tmpE_s = jx_SeqVectorCreate(n);
   tmpE_s->data = &temp_adrress;
   jx_SeqVectorInitialize(tmpE_s);
   tmpE_s->data = sol->data + n;
   
   tmpI_s = jx_SeqVectorCreate(n);
   tmpI_s->data = &temp_adrress;
   jx_SeqVectorInitialize(tmpI_s);
   tmpI_s->data = sol->data + 2*n;
   
   jx_VectorToParVector_Allocated2(comm, tmpR_s, jx_ParVectorPartitioning(uR_p), uR_p); 
   jx_VectorToParVector_Allocated2(comm, tmpE_s, jx_ParVectorPartitioning(uE_p), uE_p);
   jx_VectorToParVector_Allocated2(comm, tmpI_s, jx_ParVectorPartitioning(uI_p), uI_p);

   jx_TFree(tmpR_s);
   jx_TFree(tmpE_s);
   jx_TFree(tmpI_s);
   jx_SeqVectorDestroy(sol);
   
   return (0);
} 

/*!
 * \fn JX_Int jx_MatVecGroupR
 * \brief 利用并行矩阵 A(G) 生成 ARR(GR), VRE(GR).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JX_Int 
jx_MatVecGroupR( MPI_Comm           comm_x,
                 jx_ParCSRMatrix   *A, 
                 jx_ParCSRMatrix  **ARR_ptr, 
                 jx_ParVector     **VRE_ptr )
{
   JX_Int myid_x, nprocs_x;

   jx_ParCSRMatrix  *ARR = NULL; 
   jx_ParVector     *VRE = NULL;
   
   jx_CSRMatrix	*local_ARR = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jx_CSRMatrix	*diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix	*offd = jx_ParCSRMatrixOffd(A);
   JX_Int *col_map_offd  = jx_ParCSRMatrixColMapOffd(A);    
   JX_Int *row_starts    = jx_ParCSRMatrixRowStarts(A); 
   JX_Int first_col_diag = jx_ParCSRMatrixFirstColDiag(A);
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);

   JX_Int *ia_diag = jx_CSRMatrixI(diag);
   JX_Int *ja_diag = jx_CSRMatrixJ(diag);
   JX_Real *aa_diag = jx_CSRMatrixData(diag);
    
   JX_Int *ia_offd = jx_CSRMatrixI(offd);
   JX_Int *ja_offd = jx_CSRMatrixJ(offd);
   JX_Real *aa_offd = jx_CSRMatrixData(offd);

   JX_Int *row_starts_R = NULL; 
   JX_Int *col_starts_R = NULL;
      
   JX_Int *ia = NULL;
   JX_Int *ja = NULL;
   JX_Real *aa = NULL;   
   JX_Real *vre_data = NULL;
   
   JX_Int i, j, k, m;
   JX_Int vre_cnt, nzcnt;
   JX_Int local_size;
   JX_Int global_num_rows;
   JX_Int global_num_cols; 
   
   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;  
   
   JX_Int first_col_diag_R;
   JX_Int last_col_diag_R;         
   
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm_x, &nprocs_x); 
              
   local_size = jx_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_R, col_starts_R 数组                             
   //------------------------------------------------------------------ 
   row_starts_R = jx_CTAlloc(JX_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_R[i] = row_starts[i];
   }
   col_starts_R = row_starts_R;
   
   //------------------------------------------------------------------
   //  创建一个并行矩阵 ARR                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   ARR = jx_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_R, col_starts_R, 0, 0, 0);

   //------------------------------------------------------------------
   //  创建一个并行向量 VRE                         
   //------------------------------------------------------------------ 
   VRE = jx_ParVectorCreate(comm_x, global_num_rows, jx_ParCSRMatrixRowStarts(ARR));
   jx_ParVectorSetPartitioningOwner(VRE, 0);
   jx_ParVectorInitialize(VRE);
   vre_data = jx_VectorData(jx_ParVectorLocalVector(VRE));
     
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 ARR 和并行向量 VRE
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_ARR */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jx_CSRMatrixNumNonzeros(diag) + jx_CSRMatrixNumNonzeros(offd) - local_size;
   local_ARR = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(local_ARR);
   ia = jx_CSRMatrixI(local_ARR);
   ja = jx_CSRMatrixJ(local_ARR);
   aa = jx_CSRMatrixData(local_ARR);  
   
   /* 对 diag 和 offd 的所有非零元素循环，从中抽出子向量 VRE 和长方阵 local_ARR */
   nzcnt = 0;
   vre_cnt = 0;
   ia[0] = 0;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + first_col_diag;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m >= global_num_rows)
         {
            vre_data[vre_cnt ++] = aa_offd[j];
         }
         else
         {
            ja[nzcnt] = m; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
   
   /* 生成 ARR 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_R = col_starts_R[myid_x];
   last_col_diag_R  = col_starts_R[myid_x+1] - 1;
   jx_GenerateDiagAndOffd(local_ARR, ARR, first_col_diag_R, last_col_diag_R);
   
   jx_CSRMatrixDestroy(local_ARR);
   
   *ARR_ptr = ARR;
   *VRE_ptr = VRE;

   return (0);   
}

/*!
 * \fn JX_Int jx_MatGroupR
 * \brief 利用并行矩阵 A(G) 生成 ARR(GR).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JX_Int 
jx_MatGroupR( MPI_Comm           comm_x,
              jx_ParCSRMatrix   *A, 
              jx_ParCSRMatrix  **ARR_ptr )
{
   JX_Int myid_x, nprocs_x;

   jx_ParCSRMatrix *ARR = NULL;    
   jx_CSRMatrix	   *local_ARR = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jx_CSRMatrix	*diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix	*offd = jx_ParCSRMatrixOffd(A);
   JX_Int *col_map_offd  = jx_ParCSRMatrixColMapOffd(A);    
   JX_Int *row_starts    = jx_ParCSRMatrixRowStarts(A); 
   JX_Int first_col_diag = jx_ParCSRMatrixFirstColDiag(A);
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);

   JX_Int *ia_diag = jx_CSRMatrixI(diag);
   JX_Int *ja_diag = jx_CSRMatrixJ(diag);
   JX_Real *aa_diag = jx_CSRMatrixData(diag);
    
   JX_Int *ia_offd = jx_CSRMatrixI(offd);
   JX_Int *ja_offd = jx_CSRMatrixJ(offd);
   JX_Real *aa_offd = jx_CSRMatrixData(offd);

   JX_Int *row_starts_R = NULL; 
   JX_Int *col_starts_R = NULL;
      
   JX_Int *ia = NULL;
   JX_Int *ja = NULL;
   JX_Real *aa = NULL;   
   
   JX_Int i, j, k, m;
   JX_Int nzcnt;
   JX_Int local_size;
   JX_Int global_num_rows;
   JX_Int global_num_cols; 
   
   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;  
   
   JX_Int first_col_diag_R;
   JX_Int last_col_diag_R;         
   
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm_x, &nprocs_x); 
              
   local_size = jx_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_R, col_starts_R 数组                             
   //------------------------------------------------------------------ 
   row_starts_R = jx_CTAlloc(JX_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_R[i] = row_starts[i];
   }
   col_starts_R = row_starts_R;
   
   //------------------------------------------------------------------
   //  创建一个并行矩阵 ARR                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   ARR = jx_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_R, col_starts_R, 0, 0, 0);
     
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 ARR
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_ARR */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jx_CSRMatrixNumNonzeros(diag) + jx_CSRMatrixNumNonzeros(offd) - local_size;
   local_ARR = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(local_ARR);
   ia = jx_CSRMatrixI(local_ARR);
   ja = jx_CSRMatrixJ(local_ARR);
   aa = jx_CSRMatrixData(local_ARR);  
   
   /* 对 diag 和 offd 的所有非零元素循环，从中抽出长方阵 local_ARR */
   nzcnt = 0;
   ia[0] = 0;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + first_col_diag;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m < global_num_rows)
         {
            ja[nzcnt] = m; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
   
   /* 生成 ARR 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_R = col_starts_R[myid_x];
   last_col_diag_R  = col_starts_R[myid_x+1] - 1;
   jx_GenerateDiagAndOffd(local_ARR, ARR, first_col_diag_R, last_col_diag_R);
   
   jx_CSRMatrixDestroy(local_ARR);
   
   *ARR_ptr = ARR;

   return (0);   
}


/*!
 * \fn JX_Int jx_MatVecGroupI
 * \brief 利用并行矩阵 A(G) 生成 AII(GI), VIE(GI).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JX_Int 
jx_MatVecGroupI( MPI_Comm           comm_x,
                 jx_ParCSRMatrix   *A, 
                 jx_ParCSRMatrix  **AII_ptr, 
                 jx_ParVector     **VIE_ptr )
{
   JX_Int myid_x, nprocs_x;
   JX_Int nprocs;

   jx_ParCSRMatrix  *AII = NULL; 
   jx_ParVector     *VIE = NULL;
   
   jx_CSRMatrix	*local_AII = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jx_CSRMatrix	*diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix	*offd = jx_ParCSRMatrixOffd(A);
   JX_Int *col_map_offd  = jx_ParCSRMatrixColMapOffd(A);    
   JX_Int *row_starts    = jx_ParCSRMatrixRowStarts(A); 
   JX_Int first_col_diag = jx_ParCSRMatrixFirstColDiag(A);
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);

   JX_Int *ia_diag = jx_CSRMatrixI(diag);
   JX_Int *ja_diag = jx_CSRMatrixJ(diag);
   JX_Real *aa_diag = jx_CSRMatrixData(diag);
    
   JX_Int *ia_offd = jx_CSRMatrixI(offd);
   JX_Int *ja_offd = jx_CSRMatrixJ(offd);
   JX_Real *aa_offd = jx_CSRMatrixData(offd);

   JX_Int *row_starts_I = NULL; 
   JX_Int *col_starts_I = NULL;
      
   JX_Int *ia = NULL;
   JX_Int *ja = NULL;
   JX_Real *aa = NULL;   
   JX_Real *vie_data = NULL;
   
   JX_Int i, j, k, m;
   JX_Int vie_cnt, nzcnt;
   JX_Int local_size;
   JX_Int global_num_rows;
   JX_Int global_num_cols; 
   
   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;  
   
   JX_Int first_col_diag_I;
   JX_Int last_col_diag_I; 
   
   JX_Int npstart;  
   JX_Int disp; 
   JX_Int global_num_rows2;     
   
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm_x, &nprocs_x); 
   jx_MPI_Comm_size(jx_ParCSRMatrixComm(A), &nprocs);
    
   npstart = (2*nprocs) / 3;           
   local_size = jx_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_I, col_starts_I 数组                             
   //------------------------------------------------------------------ 
   row_starts_I = jx_CTAlloc(JX_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_I[i] = row_starts[npstart+i] - row_starts[npstart];
   }
   col_starts_I = row_starts_I;


   //------------------------------------------------------------------
   //  创建一个并行矩阵 AII                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   AII = jx_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_I, col_starts_I, 0, 0, 0);

   //------------------------------------------------------------------
   //  创建一个并行向量 VIE                         
   //------------------------------------------------------------------ 
   VIE = jx_ParVectorCreate(comm_x, global_num_rows, jx_ParCSRMatrixRowStarts(AII));
   jx_ParVectorSetPartitioningOwner(VIE, 0);
   jx_ParVectorInitialize(VIE);
   vie_data = jx_VectorData(jx_ParVectorLocalVector(VIE));
     
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 AII 和并行向量 VIE
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_AII */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jx_CSRMatrixNumNonzeros(diag) + jx_CSRMatrixNumNonzeros(offd) - local_size;
   local_AII = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(local_AII);
   ia = jx_CSRMatrixI(local_AII);
   ja = jx_CSRMatrixJ(local_AII);
   aa = jx_CSRMatrixData(local_AII);  

   /* 对 diag 和 offd 的所有非零元素循环，从中抽出子向量 VIE 和长方阵 local_AII */
   nzcnt = 0;
   vie_cnt = 0;
   ia[0] = 0;
   global_num_rows2 = 2*global_num_rows;
   disp = first_col_diag - global_num_rows2;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + disp;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m < global_num_rows2)
         {
            vie_data[vie_cnt ++] = aa_offd[j];
         }
         else
         {
            ja[nzcnt] = m - global_num_rows2; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
    
   /* 生成 AII 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_I = col_starts_I[myid_x];
   last_col_diag_I  = col_starts_I[myid_x+1] - 1;
   jx_GenerateDiagAndOffd(local_AII, AII, first_col_diag_I, last_col_diag_I);
   
   jx_CSRMatrixDestroy(local_AII);
   
   *AII_ptr = AII;
   *VIE_ptr = VIE;

   return (0);   
}

/*!
 * \fn JX_Int jx_MatGroupI
 * \brief 利用并行矩阵 A(G) 生成 AII(GI).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JX_Int 
jx_MatGroupI( MPI_Comm           comm_x,
              jx_ParCSRMatrix   *A, 
              jx_ParCSRMatrix  **AII_ptr )
{
   JX_Int myid_x, nprocs_x;
   JX_Int nprocs;

   jx_ParCSRMatrix *AII = NULL; 
   jx_CSRMatrix	   *local_AII = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jx_CSRMatrix	*diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix	*offd = jx_ParCSRMatrixOffd(A);
   JX_Int *col_map_offd  = jx_ParCSRMatrixColMapOffd(A);    
   JX_Int *row_starts    = jx_ParCSRMatrixRowStarts(A); 
   JX_Int first_col_diag = jx_ParCSRMatrixFirstColDiag(A);
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);

   JX_Int *ia_diag = jx_CSRMatrixI(diag);
   JX_Int *ja_diag = jx_CSRMatrixJ(diag);
   JX_Real *aa_diag = jx_CSRMatrixData(diag);
    
   JX_Int *ia_offd = jx_CSRMatrixI(offd);
   JX_Int *ja_offd = jx_CSRMatrixJ(offd);
   JX_Real *aa_offd = jx_CSRMatrixData(offd);

   JX_Int *row_starts_I = NULL; 
   JX_Int *col_starts_I = NULL;
      
   JX_Int *ia = NULL;
   JX_Int *ja = NULL;
   JX_Real *aa = NULL;   

   JX_Int i, j, k, m;
   JX_Int nzcnt;
   JX_Int local_size;
   JX_Int global_num_rows;
   JX_Int global_num_cols; 
   
   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;  
   
   JX_Int first_col_diag_I;
   JX_Int last_col_diag_I; 
   
   JX_Int npstart;  
   JX_Int disp; 
   JX_Int global_num_rows2;     
   
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm_x, &nprocs_x); 
   jx_MPI_Comm_size(jx_ParCSRMatrixComm(A), &nprocs);
    
   npstart = (2*nprocs) / 3;           
   local_size = jx_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_I, col_starts_I 数组                             
   //------------------------------------------------------------------ 
   row_starts_I = jx_CTAlloc(JX_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_I[i] = row_starts[npstart+i] - row_starts[npstart];
   }
   col_starts_I = row_starts_I;


   //------------------------------------------------------------------
   //  创建一个并行矩阵 AII                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   AII = jx_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_I, col_starts_I, 0, 0, 0);

   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 AII
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_AII */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jx_CSRMatrixNumNonzeros(diag) + jx_CSRMatrixNumNonzeros(offd) - local_size;
   local_AII = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(local_AII);
   ia = jx_CSRMatrixI(local_AII);
   ja = jx_CSRMatrixJ(local_AII);
   aa = jx_CSRMatrixData(local_AII);  

   /* 对 diag 和 offd 的所有非零元素循环，从中抽出长方阵 local_AII */
   nzcnt = 0;
   ia[0] = 0;
   global_num_rows2 = 2*global_num_rows;
   disp = first_col_diag - global_num_rows2;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + disp;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m >= global_num_rows2)
         {
            ja[nzcnt] = m - global_num_rows2; ///
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
    
   /* 生成 AII 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_I = col_starts_I[myid_x];
   last_col_diag_I  = col_starts_I[myid_x+1] - 1;
   jx_GenerateDiagAndOffd(local_AII, AII, first_col_diag_I, last_col_diag_I);
   
   jx_CSRMatrixDestroy(local_AII);
   
   *AII_ptr = AII;

   return (0);   
}

/*!
 * \fn JX_Int jx_MatVecGroupE
 * \brief 利用并行矩阵 A(G) 生成 AEE(GE), VER(GE), VEI(GE).
 * \note 要求进程数为 3 的倍数，且分划为自己指定的匹配分划.
 * \date 2012/02/23
 */ 
JX_Int 
jx_MatVecGroupE( MPI_Comm           comm_x,
                 jx_ParCSRMatrix   *A, 
                 jx_ParCSRMatrix  **AEE_ptr, 
                 jx_ParVector     **VER_ptr,
                 jx_ParVector     **VEI_ptr )
{
   JX_Int myid_x, nprocs_x;
   JX_Int nprocs;

   jx_ParCSRMatrix  *AEE = NULL; 
   jx_ParVector     *VER = NULL;
   jx_ParVector     *VEI = NULL;
   
   jx_CSRMatrix	*local_AEE = NULL;
     
   // 并行矩阵 A 中的部分成员           
   jx_CSRMatrix	*diag = jx_ParCSRMatrixDiag(A);
   jx_CSRMatrix	*offd = jx_ParCSRMatrixOffd(A);
   JX_Int *col_map_offd  = jx_ParCSRMatrixColMapOffd(A);    
   JX_Int *row_starts    = jx_ParCSRMatrixRowStarts(A); 
   JX_Int first_col_diag = jx_ParCSRMatrixFirstColDiag(A);
   JX_Int N = jx_ParCSRMatrixGlobalNumRows(A);

   JX_Int *ia_diag = jx_CSRMatrixI(diag);
   JX_Int *ja_diag = jx_CSRMatrixJ(diag);
   JX_Real *aa_diag = jx_CSRMatrixData(diag);
    
   JX_Int *ia_offd = jx_CSRMatrixI(offd);
   JX_Int *ja_offd = jx_CSRMatrixJ(offd);
   JX_Real *aa_offd = jx_CSRMatrixData(offd);

   JX_Int *row_starts_E = NULL; 
   JX_Int *col_starts_E = NULL;
      
   JX_Int *ia = NULL;
   JX_Int *ja = NULL;
   JX_Real *aa = NULL;   
   JX_Real *ver_data = NULL;
   JX_Real *vei_data = NULL;
   
   JX_Int i, j, k, m;
   JX_Int ver_cnt, vei_cnt, nzcnt;
   JX_Int local_size;
   JX_Int global_num_rows;
   JX_Int global_num_cols; 
   
   JX_Int num_rows;
   JX_Int num_cols;
   JX_Int num_nonzeros;  
   
   JX_Int first_col_diag_E;
   JX_Int last_col_diag_E; 
   
   JX_Int npstart;  
   JX_Int disp; 
   JX_Int global_num_rows2;     
   
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm_x, &nprocs_x); 
   jx_MPI_Comm_size(jx_ParCSRMatrixComm(A), &nprocs);
    
   npstart = nprocs / 3;           
   local_size = jx_CSRMatrixNumRows(diag);

   //------------------------------------------------------------------
   //  生成 row_starts_E, col_starts_E 数组                             
   //------------------------------------------------------------------ 
   row_starts_E = jx_CTAlloc(JX_Int, nprocs_x + 1);
   for (i = 0; i <= nprocs_x; i ++)
   {
      row_starts_E[i] = row_starts[npstart+i] - row_starts[npstart];
   }
   col_starts_E = row_starts_E;

   //------------------------------------------------------------------
   //  创建一个并行矩阵 AEE                                              
   //------------------------------------------------------------------    
   global_num_rows = N / 3;
   global_num_cols = global_num_rows;   
   AEE = jx_ParCSRMatrixCreate(comm_x, global_num_rows, global_num_cols, 
                               row_starts_E, col_starts_E, 0, 0, 0);

   //------------------------------------------------------------------
   //  创建两个并行向量 VER, VEI                         
   //------------------------------------------------------------------ 
   VER = jx_ParVectorCreate(comm_x, global_num_rows, jx_ParCSRMatrixRowStarts(AEE));
   jx_ParVectorSetPartitioningOwner(VER, 0);
   jx_ParVectorInitialize(VER);
   ver_data = jx_VectorData(jx_ParVectorLocalVector(VER));
   
   VEI = jx_ParVectorCreate(comm_x, global_num_rows, jx_ParCSRMatrixRowStarts(AEE));
   jx_ParVectorSetPartitioningOwner(VEI, 0);
   jx_ParVectorInitialize(VEI);
   vei_data = jx_VectorData(jx_ParVectorLocalVector(VEI));   
    
   //-------------------------------------------------------------------------------------------
   //  生成并行矩阵 AEE 和并行向量 VER, VEI
   //-------------------------------------------------------------------------------------------
   /* 创建并初始化串行矩阵 local_AEE */
   num_rows = local_size;
   num_cols = global_num_cols;
   num_nonzeros = jx_CSRMatrixNumNonzeros(diag) + jx_CSRMatrixNumNonzeros(offd) - 2*local_size;
   local_AEE = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(local_AEE);
   ia = jx_CSRMatrixI(local_AEE);
   ja = jx_CSRMatrixJ(local_AEE);
   aa = jx_CSRMatrixData(local_AEE);  

   /* 对 diag 和 offd 的所有非零元素循环，从中抽出子向量 VER, VEI 和长方阵 local_AEE */
   nzcnt = 0;
   ver_cnt = 0;
   vei_cnt = 0;
   ia[0] = 0;
   global_num_rows2 = 2*global_num_rows;
   disp = first_col_diag - global_num_rows;
   for (i = 0; i < local_size; i ++)
   {
      // 处理对角块
      for (j = ia_diag[i]; j < ia_diag[i+1]; j ++)
      {
         ja[nzcnt] = ja_diag[j] + disp;
         aa[nzcnt] = aa_diag[j];
         nzcnt ++;   
      }
      
      // 处理非对角块
      for (j = ia_offd[i]; j < ia_offd[i+1]; j ++)
      {
         k = ja_offd[j];      // 局部列号
         m = col_map_offd[k]; // 整体列号
         if (m < global_num_rows)
         {
            ver_data[ver_cnt ++] = aa_offd[j];
         }
         else if (m >= global_num_rows2)
         {
            vei_data[vei_cnt ++] = aa_offd[j];
         }
         else
         {
            ja[nzcnt] = m - global_num_rows;
            aa[nzcnt] = aa_offd[j];
            nzcnt ++;   
         }
      }
      ia[i+1] = nzcnt;      
   }
     
   /* 生成 AEE 的对角块和非对角块，以及 col_map_offd 映射数组 */
   first_col_diag_E = col_starts_E[myid_x];
   last_col_diag_E  = col_starts_E[myid_x+1] - 1;
   jx_GenerateDiagAndOffd(local_AEE, AEE, first_col_diag_E, last_col_diag_E);
   
   jx_CSRMatrixDestroy(local_AEE);
   
   *AEE_ptr = AEE;
   *VER_ptr = VER;
   *VEI_ptr = VEI;
   
   return (0);   
}

/*!
 * \fn JX_Int jx_DataCombine4ApctlKrylov
 * \brief 将子块数据合并为整体数据(单进程情形).
 * \author peghoty
 * \date 2012/02/25 
 */
JX_Int 
jx_DataCombine4ApctlKrylov( // input:
                            jx_ParCSRMatrix   *ARR_p, 
                            jx_ParCSRMatrix   *AEE_p, 
                            jx_ParCSRMatrix   *AII_p, 
                            jx_ParVector      *VRE_p, 
                            jx_ParVector      *VER_p, 
                            jx_ParVector      *VEI_p, 
                            jx_ParVector      *VIE_p, 
                            jx_ParVector      *fR_p, 
                            jx_ParVector      *fE_p, 
                            jx_ParVector      *fI_p,
                            jx_ParVector      *uR_p, 
                            jx_ParVector      *uE_p, 
                            jx_ParVector      *uI_p,
                            // output:    
                            jx_ParCSRMatrix  **A_ptr,  
                            jx_ParVector     **f_ptr, 
                            jx_ParVector     **u_ptr )
{
   MPI_Comm comm = jx_ParCSRMatrixComm(ARR_p);

   jx_ParCSRMatrix *A = NULL;  
   jx_ParVector    *f = NULL; 
   jx_ParVector    *u = NULL;
   
   jx_CSRMatrix *A_s = NULL;  
   jx_Vector    *f_s = NULL; 
   jx_Vector    *u_s = NULL;
      
   jx_CSRMatrix *ARR = jx_ParCSRMatrixDiag(ARR_p);
   jx_CSRMatrix *AEE = jx_ParCSRMatrixDiag(AEE_p);
   jx_CSRMatrix *AII = jx_ParCSRMatrixDiag(AII_p);
   
   jx_Vector *VRE = jx_ParVectorLocalVector(VRE_p);
   jx_Vector *VER = jx_ParVectorLocalVector(VER_p);
   jx_Vector *VEI = jx_ParVectorLocalVector(VEI_p);
   jx_Vector *VIE = jx_ParVectorLocalVector(VIE_p);
   jx_Vector *fR  = jx_ParVectorLocalVector(fR_p);
   jx_Vector *fE  = jx_ParVectorLocalVector(fE_p);
   jx_Vector *fI  = jx_ParVectorLocalVector(fI_p);
   jx_Vector *uR  = jx_ParVectorLocalVector(uR_p);
   jx_Vector *uE  = jx_ParVectorLocalVector(uE_p);
   jx_Vector *uI  = jx_ParVectorLocalVector(uI_p);   

   //------------------------------------------------------------------
   // 生成串行矩阵 A_s, 以及串行向量 f_s, u_s
   //------------------------------------------------------------------
    
   jx_3tGlobalSystem( ARR, AEE, AII, VRE, VER, VEI, VIE, 
                      fR, fE, fI, uR, uE, uI, &A_s, &f_s, &u_s ); 

   //------------------------------------------------------------------
   // 由串行矩阵 A_s, 串行向量 f_s, u_s 转成并行矩阵 A 和并行向量 f, u 
   //------------------------------------------------------------------
   
   A = jx_CSRMatrixToParCSRMatrix_sp(comm, A_s);
   f = jx_VectorToParVector_sp(comm, f_s);
   u = jx_VectorToParVector_sp(comm, u_s);
  
   //------------------------------------------------------------------
   // 释放内存，注意由于 A_s, f_s, u_s 的数据部分包含在 A, f, u 中，所有这里
   // 只需释放结构体，而不需要调用 XXDestroy.
   //------------------------------------------------------------------   
   jx_TFree(A_s);
   jx_TFree(f_s);
   jx_TFree(u_s);
   
   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;

   return (0);
}                              

/*!
 * \fn JX_Int jx_3tGlobalSystem
 * \brief Form the global linear system.
 * \author peghoty
 * \date 2011/10/18 
 */
JX_Int 
jx_3tGlobalSystem( jx_CSRMatrix  *ARR, 
                   jx_CSRMatrix  *AEE, 
                   jx_CSRMatrix  *AII, 
                   jx_Vector     *VRE, 
                   jx_Vector     *VER, 
                   jx_Vector     *VEI, 
                   jx_Vector     *VIE, 
                   jx_Vector     *fR, 
                   jx_Vector     *fE, 
                   jx_Vector     *fI,
                   jx_Vector     *uR, 
                   jx_Vector     *uE, 
                   jx_Vector     *uI,                       
                   jx_CSRMatrix **A_ptr, 
                   jx_Vector    **f_ptr,
                   jx_Vector    **u_ptr )
{
   jx_CSRMatrix *A = NULL; 
   jx_Vector    *f = NULL;
   jx_Vector    *u = NULL;

   JX_Int n  = jx_CSRMatrixNumRows(ARR);
   JX_Int nz = jx_CSRMatrixNumNonzeros(ARR);
   JX_Int nplusn = 2*n;   
   JX_Int N  = 3*n;
   JX_Int NZ = 3*nz + 4*n;
   
   JX_Real *aa = NULL;
   JX_Int    *ia = NULL;
   JX_Int    *ja = NULL;
   JX_Real *f_data = NULL;
   JX_Real *u_data = NULL;
    
   JX_Real *aa_R = jx_CSRMatrixData(ARR);
   JX_Int    *ia_R = jx_CSRMatrixI(ARR);
   JX_Int    *ja_R = jx_CSRMatrixJ(ARR);
   
   JX_Real *aa_E = jx_CSRMatrixData(AEE);
   JX_Int    *ia_E = jx_CSRMatrixI(AEE);
   JX_Int    *ja_E = jx_CSRMatrixJ(AEE);

   JX_Real *aa_I = jx_CSRMatrixData(AII);
   JX_Int    *ia_I = jx_CSRMatrixI(AII);
   JX_Int    *ja_I = jx_CSRMatrixJ(AII);
   
   JX_Real *vre_data = jx_VectorData(VRE);
   JX_Real *ver_data = jx_VectorData(VER);
   JX_Real *vei_data = jx_VectorData(VEI);
   JX_Real *vie_data = jx_VectorData(VIE);
   
   JX_Real *fR_data = jx_VectorData(fR);
   JX_Real *fE_data = jx_VectorData(fE);
   JX_Real *fI_data = jx_VectorData(fI);
   
   JX_Real *uR_data = jx_VectorData(uR);
   JX_Real *uE_data = jx_VectorData(uE);
   JX_Real *uI_data = jx_VectorData(uI);   
   
   /* local variables */
   JX_Int i, i1, i2, j;
   JX_Int cnt;

   //---------------------------------------------------
   //  Generate the CSR matrix A
   //---------------------------------------------------
   
   /* create a CSR matrix */
   A = jx_CSRMatrixCreate(N, N, NZ);
   jx_CSRMatrixInitialize(A);
   aa = jx_CSRMatrixData(A);
   ia = jx_CSRMatrixI(A);
   ja = jx_CSRMatrixJ(A);
   
   /* initialize the counter */
   cnt = 0;
   ia[0] = 0;   

   /* deal with ARR and VRE */
   for (i = 0; i < n; i ++)
   {
      for (j = ia_R[i]; j < ia_R[i+1]; j ++)
      {
         aa[cnt] = aa_R[j];
         ja[cnt] = ja_R[j];
         cnt ++;
      }
      aa[cnt] = vre_data[i];
      ja[cnt] = i + n;
      cnt ++;
      
      ia[i+1] = cnt;
   }

   /* deal with AEE and VER, VEI */
   for (i = 0; i < n; i ++)
   {
      for (j = ia_E[i]; j < ia_E[i+1]; j ++)
      {
         aa[cnt] = aa_E[j];
         ja[cnt] = ja_E[j] + n;
         cnt ++;
      }
      aa[cnt] = ver_data[i];
      ja[cnt] = i;
      cnt ++;

      aa[cnt] = vei_data[i];
      ja[cnt] = i + nplusn;
      cnt ++;
      
      ia[i+n+1] = cnt;
   }

   /* deal with AII and VIE */
   for (i = 0; i < n; i ++)
   {
      for (j = ia_I[i]; j < ia_I[i+1]; j ++)
      {
         aa[cnt] = aa_I[j];
         ja[cnt] = ja_I[j] + nplusn;
         cnt ++;
      }
      aa[cnt] = vie_data[i];
      ja[cnt] = i + n;
      cnt ++;
      
      ia[i+nplusn+1] = cnt;
   }

   //---------------------------------------------------
   //  Generate the vector f and u
   //---------------------------------------------------

   /* create f and u  */
   f = jx_SeqVectorCreate(N);
   jx_SeqVectorInitialize(f);
   f_data = jx_VectorData(f);

   u = jx_SeqVectorCreate(N);
   jx_SeqVectorInitialize(u);
   u_data = jx_VectorData(u);
  
   for (i = 0; i < n; i ++)
   {  
      i1 = i + n;
      i2 = i + nplusn;
      f_data[i]  = fR_data[i];
      f_data[i1] = fE_data[i];
      f_data[i2] = fI_data[i];
      u_data[i]  = uR_data[i];
      u_data[i1] = uE_data[i];
      u_data[i2] = uI_data[i];      
   }

   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;
   
   return (0);
}

/*!
 * \fn JX_Int jx_ParaDataTrans4ApctlKrylov
 * \brief Data transferring for Apctl-Krylov method.
 * \author peghoty 
 * \date 2012/02/25
 */
JX_Int 
jx_ParaDataTrans4ApctlKrylov( // input:
                              MPI_Comm           comm, 
                              MPI_Comm           comm_x, 
                              JX_Int                groupid_x,
                              jx_ParCSRMatrix   *ARR_p, 
                              jx_ParCSRMatrix   *AEE_p, 
                              jx_ParCSRMatrix   *AII_p, 
                              jx_ParVector      *VRE_p, 
                              jx_ParVector      *VER_p, 
                              jx_ParVector      *VEI_p, 
                              jx_ParVector      *VIE_p, 
                              jx_ParVector      *fR_p, 
                              jx_ParVector      *fE_p, 
                              jx_ParVector      *fI_p,
                              jx_ParVector      *uR_p, 
                              jx_ParVector      *uE_p, 
                              jx_ParVector      *uI_p,
                              // output:    
                              jx_ParCSRMatrix  **ARR_ptr,
                              jx_ParCSRMatrix  **AEE_ptr,
                              jx_ParCSRMatrix  **AII_ptr, 
                              jx_ParVector     **VRE_ptr, 
                              jx_ParVector     **VER_ptr, 
                              jx_ParVector     **VEI_ptr, 
                              jx_ParVector     **VIE_ptr,
                              jx_ParCSRMatrix  **A_ptr,  
                              jx_ParVector     **f_ptr, 
                              jx_ParVector     **u_ptr )                    
{
   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL; 
   jx_ParVector     *VRE = NULL; 
   jx_ParVector     *VER = NULL; 
   jx_ParVector     *VEI = NULL; 
   jx_ParVector     *VIE = NULL;

   jx_ParCSRMatrix  *A = NULL;  
   jx_ParVector     *f = NULL; 
   jx_ParVector     *u = NULL;
   
   jx_Vector *VRE_loc = NULL; 
   jx_Vector *VER_loc = NULL; 
   jx_Vector *VEI_loc = NULL; 
   jx_Vector *VIE_loc = NULL;

   jx_Vector *f_loc = NULL; 
   jx_Vector *u_loc = NULL; 
   
   JX_Int *row_starts_loc = NULL;     
   JX_Int *col_starts_loc = NULL;
   
   JX_Int *row_starts_glo = NULL;     
   JX_Int *col_starts_glo = NULL;   
   
   JX_Int *row_starts = jx_ParCSRMatrixRowStarts(ARR_p);
   
   JX_Int n = jx_ParCSRMatrixGlobalNumRows(ARR_p);
   JX_Int N = 3*n;
   JX_Int n2 = 2*n;
   
   JX_Int first_col_diag;
   JX_Int last_col_diag;   
                          
   JX_Int myid, myid_x, nprocs;
 
   jx_CSRMatrix *ARR_comb = NULL;
   JX_Real *aa_R = NULL;
   JX_Int    *ia_R = NULL;
   JX_Int    *ja_R = NULL;

   jx_CSRMatrix *AEE_comb = NULL;    
   JX_Real *aa_E = NULL;
   JX_Int    *ia_E = NULL;
   JX_Int    *ja_E = NULL;

   jx_CSRMatrix *AII_comb = NULL;  
   JX_Real *aa_I = NULL;
   JX_Int    *ia_I = NULL;
   JX_Int    *ja_I = NULL;
   
   JX_Real *vre_data = jx_VectorData(jx_ParVectorLocalVector(VRE_p));
   JX_Real *ver_data = jx_VectorData(jx_ParVectorLocalVector(VER_p));
   JX_Real *vei_data = jx_VectorData(jx_ParVectorLocalVector(VEI_p));
   JX_Real *vie_data = jx_VectorData(jx_ParVectorLocalVector(VIE_p));
   
   JX_Real *fr_data = jx_VectorData(jx_ParVectorLocalVector(fR_p));
   JX_Real *fe_data = jx_VectorData(jx_ParVectorLocalVector(fE_p));
   JX_Real *fi_data = jx_VectorData(jx_ParVectorLocalVector(fI_p));
   
   JX_Real *ur_data = jx_VectorData(jx_ParVectorLocalVector(uR_p));
   JX_Real *ue_data = jx_VectorData(jx_ParVectorLocalVector(uE_p));
   JX_Real *ui_data = jx_VectorData(jx_ParVectorLocalVector(uI_p));

   jx_CSRMatrix *A_loc = NULL; 
   JX_Int num_rows, num_cols, num_nonzeros;
   JX_Real *aa = NULL;
   JX_Int    *ia = NULL;
   JX_Int    *ja = NULL;
   
   jx_CSRMatrix *A_rec = NULL;
   JX_Real *aarec = NULL;
   JX_Int    *iarec = NULL;
   JX_Int    *jarec = NULL;         
   
   JX_Int  num_sends;                 // 发送进程个数
   JX_Int *send_procs_dbl  = NULL;    // 发送进程编号
   JX_Int *send_procs_int  = NULL;    // 发送进程编号
   JX_Int *send_starts_dbl = NULL;    // 实型数据发送缓存区的管理数组
   JX_Int *send_starts_int = NULL;    // 整型发送缓存区的管理数组

   JX_Int  num_recvs;                 // 接收进程个数
   JX_Int *recv_procs_dbl  = NULL;    // 接收进程编号
   JX_Int *recv_procs_int  = NULL;    // 接收进程编号
   JX_Int *recv_starts_dbl = NULL;    // 实型数据接收缓存区的管理数组
   JX_Int *recv_starts_int = NULL;    // 整型数据接收缓存区的管理数组    

   JX_Real *send_buf_dbl = NULL;
   JX_Real *recv_buf_dbl = NULL;
   
   JX_Int    *send_buf_int = NULL;
   JX_Int    *recv_buf_int = NULL;

   jx_ParVecCommPkg    *comm_pkg_dbl    = NULL;
   jx_ParVecCommPkg    *comm_pkg_int    = NULL;
   jx_ParVecCommHandle *comm_handle_dbl = NULL;
   jx_ParVecCommHandle *comm_handle_int = NULL;   
         
   JX_Int num_proc_eachgroup;          
   JX_Int num_proc_eachgroup2;         
   JX_Int local_row;                   // 行数
   JX_Int local_col;                   // 列数
   JX_Int local_nnz;                   // 非零元素个数 
   
   JX_Int *local_nnz_array = NULL;      
   JX_Int *local_row_array = NULL;     
   
   JX_Int size_base_int;
   JX_Int size0, size1, size2;
   JX_Int rnp0, rnp1, rnp2;
     
   JX_Int i, j, k;
   JX_Int row_cnt, ja_cnt, aa_cnt; 
   JX_Int f_cnt, u_cnt; 
   JX_Int vre_cnt, ver_cnt, vei_cnt, vie_cnt;  
   JX_Int start;
   JX_Int loc_row_0, loc_row_1, loc_row_2;
   JX_Int loc_nnz_0, loc_nnz_1, loc_nnz_2, loc_nnz_01;  
     
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm, &nprocs); 

   //====================================================================================
   // Step 1: 数据通信
   //====================================================================================
   
   ARR_comb = jx_MergeDiagAndOffd(ARR_p);
   AEE_comb = jx_MergeDiagAndOffd(AEE_p);
   AII_comb = jx_MergeDiagAndOffd(AII_p);  
   
   local_nnz = jx_CSRMatrixNumNonzeros(ARR_comb);
   local_row = jx_CSRMatrixNumRows(ARR_comb); 
   local_col = jx_CSRMatrixNumCols(ARR_comb);
 
   aa_R = jx_CSRMatrixData(ARR_comb);
   ia_R = jx_CSRMatrixI(ARR_comb);
   ja_R = jx_CSRMatrixJ(ARR_comb);
   
   aa_E = jx_CSRMatrixData(AEE_comb);
   ia_E = jx_CSRMatrixI(AEE_comb);
   ja_E = jx_CSRMatrixJ(AEE_comb);
   
   aa_I = jx_CSRMatrixData(AII_comb);
   ia_I = jx_CSRMatrixI(AII_comb);
   ja_I = jx_CSRMatrixJ(AII_comb);   

   num_proc_eachgroup  = nprocs / 3;
   num_proc_eachgroup2 = 2*num_proc_eachgroup;
   
   /* num_sends */
   num_sends = 3;
   
   send_procs_dbl  = jx_CTAlloc(JX_Int, num_sends);
   send_procs_int  = jx_CTAlloc(JX_Int, num_sends);
   send_starts_dbl = jx_CTAlloc(JX_Int, num_sends + 1);
   send_starts_int = jx_CTAlloc(JX_Int, num_sends + 1);
 
   /* send_procs_dbl */
   send_procs_dbl[0] = myid / 3;
   send_procs_dbl[1] = send_procs_dbl[0] + num_proc_eachgroup;
   send_procs_dbl[2] = send_procs_dbl[1] + num_proc_eachgroup;
 
   /* send_procs_int */
   send_procs_int[0] = send_procs_dbl[0];
   send_procs_int[1] = send_procs_dbl[1];
   send_procs_int[2] = send_procs_dbl[2];
     
   /* send_starts_dbl */
   size0 = local_nnz + 3*local_row;
   size1 = local_nnz + 4*local_row;
   size2 = local_nnz + 3*local_row;
   send_starts_dbl[0] = 0;
   send_starts_dbl[1] = size0;               
   send_starts_dbl[2] = send_starts_dbl[1] + size1; 
   send_starts_dbl[3] = send_starts_dbl[2] + size2; 
    
   /* send_starts_int */     
   size_base_int = local_nnz + (local_row + 1);   
   send_starts_int[0] = 0;
   send_starts_int[1] = size_base_int;   // ARR
   send_starts_int[2] = 2*size_base_int; // AEE
   send_starts_int[3] = 3*size_base_int; // AII

   /* num_recvs */
   num_recvs = 3;
   
   recv_procs_dbl  = jx_CTAlloc(JX_Int, num_recvs);
   recv_procs_int  = jx_CTAlloc(JX_Int, num_recvs);
   recv_starts_dbl = jx_CTAlloc(JX_Int, num_recvs + 1);
   recv_starts_int = jx_CTAlloc(JX_Int, num_recvs + 1);
   
   /* recv_procs_dbl */
   recv_procs_dbl[0] = (myid % num_proc_eachgroup)*3;
   recv_procs_dbl[1] = recv_procs_dbl[0] + 1;
   recv_procs_dbl[2] = recv_procs_dbl[1] + 1;
  
   /* recv_procs_int */
   recv_procs_int[0] = recv_procs_dbl[0];
   recv_procs_int[1] = recv_procs_dbl[1];
   recv_procs_int[2] = recv_procs_dbl[2];
   
   /* 消息全收集 */
   local_nnz_array = jx_CTAlloc(JX_Int, nprocs);
   local_row_array = jx_CTAlloc(JX_Int, nprocs);
   jx_MPI_Allgather(&local_nnz, 1, JX_MPI_INT, local_nnz_array, 1, JX_MPI_INT, comm);
   jx_MPI_Allgather(&local_row, 1, JX_MPI_INT, local_row_array, 1, JX_MPI_INT, comm);
 
   /* recv_starts_dbl */
   
   rnp0 = recv_procs_dbl[0];
   rnp1 = recv_procs_dbl[1];
   rnp2 = recv_procs_dbl[2]; 
   
   loc_row_0 = local_row_array[rnp0];
   loc_row_1 = local_row_array[rnp1];
   loc_row_2 = local_row_array[rnp2];
  
   loc_nnz_0 = local_nnz_array[rnp0];
   loc_nnz_1 = local_nnz_array[rnp1];
   loc_nnz_2 = local_nnz_array[rnp2];
        
   if (groupid_x == 1)
   {
      size0 = loc_nnz_0 + 4*loc_row_0;
      size1 = loc_nnz_1 + 4*loc_row_1;
      size2 = loc_nnz_2 + 4*loc_row_2;
   }
   else
   {  
      size0 = loc_nnz_0 + 3*loc_row_0;
      size1 = loc_nnz_1 + 3*loc_row_1;
      size2 = loc_nnz_2 + 3*loc_row_2;
   }
   recv_starts_dbl[0] = 0;
   recv_starts_dbl[1] = size0;
   recv_starts_dbl[2] = recv_starts_dbl[1] + size1;
   recv_starts_dbl[3] = recv_starts_dbl[2] + size2;  
   
   size0 = loc_nnz_0 + (loc_row_0 + 1);
   size1 = loc_nnz_1 + (loc_row_1 + 1);
   size2 = loc_nnz_2 + (loc_row_2 + 1);
   recv_starts_int[0] = 0;
   recv_starts_int[1] = size0;
   recv_starts_int[2] = recv_starts_int[1] + size1;
   recv_starts_int[3] = recv_starts_int[2] + size2;
   
  
   //-------------------------------------------------------------------------------
   //  开设发送和接收缓存区空间 send_buf 和 recv_buf
   //-------------------------------------------------------------------------------   
   
   send_buf_dbl = jx_CTAlloc(JX_Real, send_starts_dbl[num_sends]);
   recv_buf_dbl = jx_CTAlloc(JX_Real, recv_starts_dbl[num_recvs]); 
   
   send_buf_int = jx_CTAlloc(JX_Int, send_starts_int[num_sends]);
   recv_buf_int = jx_CTAlloc(JX_Int, recv_starts_int[num_recvs]);   

  
   //-------------------------------------------
   //  填充发送缓存区空间 send_buf_dbl 
   //-------------------------------------------
   
   k = 0; 
   
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_R[i]; 
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vre_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fr_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ur_data[i];
   }

   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_E[i]; 
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ver_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vei_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fe_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ue_data[i];
   }
     
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_I[i]; 
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vie_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fi_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ui_data[i];
   }
   //jx_printf(" myid = %d send_buf_dbl_size = %d\n", myid, k);
   
    
   //-------------------------------------------------------------------
   //  填充发送缓存区空间 send_buf_int
   //--------------------------------------------------------------------
      
   k = 0; 
   
   for (i = 0; i < local_row+1; i ++)
   {
      send_buf_int[k++] = ia_R[i]; 
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_R[i]; 
   }
   
   for (i = 0; i < local_row+1; i ++)
   {
      send_buf_int[k++] = ia_E[i]; 
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_E[i]; 
   }
   
   for (i = 0; i < local_row+1; i ++)
   {
      send_buf_int[k++] = ia_I[i]; 
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_I[i]; 
   }
   //jx_printf(" myid = %d send_buf_int_size = %d\n", myid, k);

    
   //-------------------------------------------------------
   //  创建通信包 comm_pkg_dbl，并填充其成员
   //-------------------------------------------------------
   
   comm_pkg_dbl = jx_CTAlloc(jx_ParVecCommPkg, 1);

   jx_ParVecCommPkgComm(comm_pkg_dbl)       = comm;                                   
   jx_ParVecCommPkgNumSends(comm_pkg_dbl)   = num_sends;
   jx_ParVecCommPkgSendProcs(comm_pkg_dbl)  = send_procs_dbl;
   jx_ParVecCommPkgSendStarts(comm_pkg_dbl) = send_starts_dbl;
   jx_ParVecCommPkgNumRecvs(comm_pkg_dbl)   = num_recvs;
   jx_ParVecCommPkgRecvProcs(comm_pkg_dbl)  = recv_procs_dbl;
   jx_ParVecCommPkgRecvStarts(comm_pkg_dbl) = recv_starts_dbl; 

   //-------------------------------------------------------
   //  创建通信包 comm_pkg_int，并填充其成员
   //-------------------------------------------------------
   
   comm_pkg_int = jx_CTAlloc(jx_ParVecCommPkg, 1);

   jx_ParVecCommPkgComm(comm_pkg_int)       = comm;                                   
   jx_ParVecCommPkgNumSends(comm_pkg_int)   = num_sends;
   jx_ParVecCommPkgSendProcs(comm_pkg_int)  = send_procs_int;
   jx_ParVecCommPkgSendStarts(comm_pkg_int) = send_starts_int;
   jx_ParVecCommPkgNumRecvs(comm_pkg_int)   = num_recvs;
   jx_ParVecCommPkgRecvProcs(comm_pkg_int)  = recv_procs_int;
   jx_ParVecCommPkgRecvStarts(comm_pkg_int) = recv_starts_int; 
   

   //-------------------------------------------------------
   //  通信: 发送和接收数据
   //-------------------------------------------------------
   
   comm_handle_dbl = jx_ParVecCommHandleCreate(1, comm_pkg_dbl, send_buf_dbl, recv_buf_dbl);
   comm_handle_int = jx_ParVecCommHandleCreate(11, comm_pkg_int, send_buf_int, recv_buf_int);
   
   /* ... some computational work can be done here to ovelap the communication! */
   
   jx_ParVecCommHandleDestroy(comm_handle_dbl);
   jx_ParVecCommHandleDestroy(comm_handle_int);


   //====================================================================================
   // Step 2: 将接收缓存区中的数据转化为本地串行矩阵和串行向量的形式，
   //         为进一步生成并行数据作准备.
   //====================================================================================

   num_rows = 0;
   num_nonzeros = 0;
   for (i = 0; i < num_recvs; i ++)
   {
      num_rows += local_row_array[recv_procs_int[i]];
      num_nonzeros += local_nnz_array[recv_procs_int[i]];
   }
   num_cols = local_col;

   A_loc = jx_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   jx_CSRMatrixInitialize(A_loc);
   aa = jx_CSRMatrixData(A_loc);
   ia = jx_CSRMatrixI(A_loc);
   ja = jx_CSRMatrixJ(A_loc);

   k = 0;       // 累加接收缓存区中的数据
   row_cnt = 0; // 累加矩阵 A 的行数 
   ja_cnt  = 0; // 累加 ja 数组中的元素

   loc_nnz_01 = loc_nnz_0 + loc_nnz_1;
   
   for (i = 0; i <= loc_row_0; i ++)
   {
      ia[row_cnt ++] = recv_buf_int[k ++];
   }
   for (i = 0; i < loc_nnz_0; i ++)
   {
      ja[ja_cnt ++] = recv_buf_int[k ++];
   }
   row_cnt --; // be careful here!
   
   for (i = 0; i <= loc_row_1; i ++)
   {
      ia[row_cnt ++] = recv_buf_int[k ++] + loc_nnz_0;
   }
   for (i = 0; i < loc_nnz_1; i ++)
   {
      ja[ja_cnt ++] = recv_buf_int[k ++];
   }
   row_cnt --; // be careful here!
   
   for (i = 0; i <= loc_row_2; i ++)
   {
      ia[row_cnt++] = recv_buf_int[k ++] + loc_nnz_01;
   }
   for (i = 0; i < loc_nnz_2; i ++)
   {
      ja[ja_cnt ++] = recv_buf_int[k ++];
   }
   
   /* 生成 aa 数组 */
   aa_cnt = 0;  // 累加 aa 数组中的元素
   
   start = recv_starts_dbl[0]; 
   for (i = 0; i < loc_nnz_0; i ++)
   {
      aa[aa_cnt ++] = recv_buf_dbl[start + i];
   }
   
   start = recv_starts_dbl[1]; 
   for (i = 0; i < loc_nnz_1; i ++)
   {
      aa[aa_cnt ++] = recv_buf_dbl[start + i];
   } 
   
   start = recv_starts_dbl[2]; 
   for (i = 0; i < loc_nnz_2; i ++)
   {
      aa[aa_cnt ++] = recv_buf_dbl[start + i];
   } 

   //jx_printf(" myid = %d aa_cnt = %d num_nonzeros = %d\n", myid, aa_cnt, num_nonzeros);  

   f_loc = jx_SeqVectorCreate(num_rows);
   u_loc = jx_SeqVectorCreate(num_rows);
   jx_SeqVectorInitialize(f_loc);
   jx_SeqVectorInitialize(u_loc);
      
   if (groupid_x == 0) // 光子进程组
   {
      VRE_loc = jx_SeqVectorCreate(num_rows);
      jx_SeqVectorInitialize(VRE_loc);
      
      vre_cnt = 0;
      f_cnt  = 0;
      u_cnt  = 0;
      
      start = recv_starts_dbl[0] + loc_nnz_0; 
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(VRE_loc)[vre_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[1] + loc_nnz_1; 
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(VRE_loc)[vre_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[2] + loc_nnz_2; 
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(VRE_loc)[vre_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }                           
   }
   else if (groupid_x == 1) // 电子进程组
   {
      VER_loc = jx_SeqVectorCreate(num_rows);
      VEI_loc = jx_SeqVectorCreate(num_rows);
      jx_SeqVectorInitialize(VER_loc);
      jx_SeqVectorInitialize(VEI_loc);
      
      ver_cnt = 0;
      vei_cnt = 0;
      f_cnt  = 0;
      u_cnt  = 0;
      
      start = recv_starts_dbl[0] + loc_nnz_0; 
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(VER_loc)[ver_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(VEI_loc)[vei_cnt ++] = recv_buf_dbl[start + i];
      }        
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[1] + loc_nnz_1; 
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(VER_loc)[ver_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(VEI_loc)[vei_cnt ++] = recv_buf_dbl[start + i];
      }        
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[2] + loc_nnz_2; 
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(VER_loc)[ver_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(VEI_loc)[vei_cnt ++] = recv_buf_dbl[start + i];
      }       
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }     
   }
   else if (groupid_x == 2) // 离子进程组
   {
      VIE_loc = jx_SeqVectorCreate(num_rows);
      jx_SeqVectorInitialize(VIE_loc);
      
      vie_cnt = 0;
      f_cnt  = 0;
      u_cnt  = 0;
      
      start = recv_starts_dbl[0] + loc_nnz_0; 
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(VIE_loc)[vie_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_0;
      for (i = 0; i < loc_row_0; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[1] + loc_nnz_1; 
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(VIE_loc)[vie_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_1;
      for (i = 0; i < loc_row_1; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }
      
      start = recv_starts_dbl[2] + loc_nnz_2; 
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(VIE_loc)[vie_cnt ++] = recv_buf_dbl[start + i];
      }
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(f_loc)[f_cnt ++] = recv_buf_dbl[start + i];
      }  
      start += loc_row_2;
      for (i = 0; i < loc_row_2; i ++)
      {
         jx_VectorData(u_loc)[u_cnt ++] = recv_buf_dbl[start + i];
      }            
   }

   row_starts_loc = jx_CTAlloc(JX_Int, num_proc_eachgroup + 1);     
   col_starts_loc = row_starts_loc;
   for (i = 0; i <= num_proc_eachgroup; i ++)
   {
      row_starts_loc[i] = row_starts[3*i];
   } 

      
   if (groupid_x == 0) // 光子进程组
   {
      ARR = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, ARR, first_col_diag, last_col_diag);
      
      VRE = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VRE, 0);
      jx_TFree(jx_ParVectorLocalVector(VRE));
      jx_ParVectorLocalVector(VRE) = VRE_loc;                           
   }
   else if (groupid_x == 1) // 电子进程组
   {
      AEE = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, AEE, first_col_diag, last_col_diag);   
          
      VER = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VER, 0);
      jx_TFree(jx_ParVectorLocalVector(VER));
      jx_ParVectorLocalVector(VER) = VER_loc;
   
      VEI = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VEI, 0);
      jx_TFree(jx_ParVectorLocalVector(VEI));
      jx_ParVectorLocalVector(VEI) = VEI_loc;
   }
   else if (groupid_x == 2) // 离子进程组
   {
      AII = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, AII, first_col_diag, last_col_diag);

      VIE = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VIE, 0);
      jx_TFree(jx_ParVectorLocalVector(VIE));
      jx_ParVectorLocalVector(VIE) = VIE_loc;               
   }


   row_starts_glo = jx_CTAlloc(JX_Int, nprocs + 1);     
   col_starts_glo = row_starts_glo;
   for (i = 0; i < num_proc_eachgroup; i ++)
   {
      row_starts_glo[i] = row_starts_loc[i];
      row_starts_glo[i + num_proc_eachgroup]  = row_starts_loc[i] + n;
      row_starts_glo[i + num_proc_eachgroup2] = row_starts_loc[i] + n2;
   } 
   row_starts_glo[nprocs] = N;


   if (groupid_x == 1)
   {
      num_nonzeros = jx_CSRMatrixNumNonzeros(A_loc) + 2*num_rows;
   }
   else
   {
      num_nonzeros = jx_CSRMatrixNumNonzeros(A_loc) + num_rows;
   }
   A_rec = jx_CSRMatrixCreate(num_rows, N, num_nonzeros);
   jx_CSRMatrixInitialize(A_rec);
   aarec = jx_CSRMatrixData(A_rec);
   iarec = jx_CSRMatrixI(A_rec);
   jarec = jx_CSRMatrixJ(A_rec);
   
   if (groupid_x == 0) // 光子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         for (j = ia[i]; j < ia[i+1]; j ++)
         {
            jarec[k] = ja[j];
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i + n;
         aarec[k] = jx_VectorData(VRE_loc)[i];
         k ++;
         
         iarec[i + 1] = k;
      }
   }
   else if (groupid_x == 1) // 电子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         for (j = ia[i]; j < ia[i+1]; j ++)
         {
            jarec[k] = ja[j] + n;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i - n;
         aarec[k] = jx_VectorData(VER_loc)[i];
         k ++;
         
         jarec[k] = start + i + n;
         aarec[k] = jx_VectorData(VEI_loc)[i];
         k ++;
         
         iarec[i + 1] = k;
      }
   }
   else if (groupid_x == 2) // 离子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         for (j = ia[i]; j < ia[i+1]; j ++)
         {
            jarec[k] = ja[j] + n2;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i - n;
         aarec[k] = jx_VectorData(VIE_loc)[i];
         k ++;
         
         iarec[i + 1] = k;
      }
   }
    
   A = jx_ParCSRMatrixCreate(comm, N, N, row_starts_glo, col_starts_glo, 0, 0, 0); 
   first_col_diag = col_starts_glo[myid];
   last_col_diag  = col_starts_glo[myid+1] - 1;
   jx_GenerateDiagAndOffd(A_rec, A, first_col_diag, last_col_diag);

   f = jx_ParVectorCreate(comm, N, row_starts_glo);
   jx_ParVectorSetPartitioningOwner(f, 0);
   jx_TFree(jx_ParVectorLocalVector(f));
   jx_ParVectorLocalVector(f) = f_loc;   

   u = jx_ParVectorCreate(comm, N, row_starts_glo);
   jx_ParVectorSetPartitioningOwner(u, 0);
   jx_TFree(jx_ParVectorLocalVector(u));
   jx_ParVectorLocalVector(u) = u_loc; 
    
   jx_TFree(send_buf_dbl);
   jx_TFree(recv_buf_dbl);
   jx_TFree(send_buf_int);
   jx_TFree(recv_buf_int);
   jx_TFree(local_nnz_array);
   jx_TFree(local_row_array);
   
   jx_ParVecCommPkgDestroy(comm_pkg_dbl);
   jx_ParVecCommPkgDestroy(comm_pkg_int);
   
   jx_CSRMatrixDestroy(ARR_comb);
   jx_CSRMatrixDestroy(AEE_comb);
   jx_CSRMatrixDestroy(AII_comb);
   
   jx_CSRMatrixDestroy(A_loc);
   jx_CSRMatrixDestroy(A_rec);
      
   *ARR_ptr = ARR;
   *AEE_ptr = AEE;
   *AII_ptr = AII;
   *VRE_ptr = VRE; 
   *VER_ptr = VER; 
   *VEI_ptr = VEI; 
   *VIE_ptr = VIE;  
   
   *A_ptr = A;  
   *f_ptr = f;
   *u_ptr = u;     
      
   return (0);
}

JX_Int 
jx_ParaDataTrans4ApctlmgKrylov( MPI_Comm           comm,
                                MPI_Comm           comm_x,
                                JX_Int                groupid_x,
                                jx_ParCSRMatrix  **ARR_p,
                                jx_ParCSRMatrix   *AEE_p,
                                jx_ParCSRMatrix   *AII_p,
                                jx_ParVector     **VRE_p,
                                jx_ParVector     **VER_p,
                                jx_ParVector      *VEI_p,
                                jx_ParVector      *VIE_p,
                                jx_ParVector     **fR_p,
                                jx_ParVector      *fE_p,
                                jx_ParVector      *fI_p,
                                jx_ParVector     **uR_p,
                                jx_ParVector      *uE_p,
                                jx_ParVector      *uI_p,
                                JX_Int             ng,
                                jx_ParCSRMatrix  **ARR_ptr,
                                jx_ParCSRMatrix  **AEE_ptr,
                                jx_ParCSRMatrix  **AII_ptr,
                                jx_ParVector     **VRE_ptr,
                                jx_ParVector    ***VER_ptr,
                                jx_ParVector     **VEI_ptr,
                                jx_ParVector     **VIE_ptr,
                                jx_ParCSRMatrix  **A_ptr,
                                jx_ParVector     **f_ptr,
                                jx_ParVector     **u_ptr )
{
   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL;
   jx_ParVector     *VRE = NULL;
   jx_ParVector    **VER = NULL;
   jx_ParVector     *VEI = NULL;
   jx_ParVector     *VIE = NULL;

   jx_ParCSRMatrix  *A = NULL;
   jx_ParVector     *f = NULL;
   jx_ParVector     *u = NULL;

   jx_Vector  *VRE_loc = NULL;
   jx_Vector **VER_loc = NULL;
   jx_Vector  *VEI_loc = NULL;
   jx_Vector  *VIE_loc = NULL;

   jx_Vector *f_loc = NULL;
   jx_Vector *u_loc = NULL;

   JX_Int *row_starts_loc = NULL;
   JX_Int *col_starts_loc = NULL;

   JX_Int *row_starts_glo = NULL;
   JX_Int *col_starts_glo = NULL;

   JX_Int *row_starts = jx_ParCSRMatrixRowStarts(ARR_p[0]);

   JX_Int n = jx_ParCSRMatrixGlobalNumRows(ARR_p[0]);
   JX_Int N = (ng + 2) * n;
   JX_Int e_pos = ng * n;
   JX_Int i_pos = e_pos + n;

   JX_Int first_col_diag;
   JX_Int last_col_diag;

   JX_Int myid, myid_x, nprocs;

   jx_CSRMatrix **ARR_comb = jx_CTAlloc(jx_CSRMatrix *, ng);
   JX_Real *aa_R = NULL;
   JX_Int    *ia_R = NULL;
   JX_Int    *ja_R = NULL;

   jx_CSRMatrix *AEE_comb = NULL;
   JX_Real *aa_E = NULL;
   JX_Int    *ia_E = NULL;
   JX_Int    *ja_E = NULL;

   jx_CSRMatrix *AII_comb = NULL;
   JX_Real *aa_I = NULL;
   JX_Int    *ia_I = NULL;
   JX_Int    *ja_I = NULL;

   JX_Real *vre_data = NULL;
   JX_Real *ver_data = NULL;
   JX_Real *vei_data = jx_VectorData(jx_ParVectorLocalVector(VEI_p));
   JX_Real *vie_data = jx_VectorData(jx_ParVectorLocalVector(VIE_p));

   JX_Real *fr_data = NULL;
   JX_Real *fe_data = jx_VectorData(jx_ParVectorLocalVector(fE_p));
   JX_Real *fi_data = jx_VectorData(jx_ParVectorLocalVector(fI_p));

   JX_Real *ur_data = NULL;
   JX_Real *ue_data = jx_VectorData(jx_ParVectorLocalVector(uE_p));
   JX_Real *ui_data = jx_VectorData(jx_ParVectorLocalVector(uI_p));

   jx_CSRMatrix *A_loc = NULL;
   JX_Int num_rows, num_nonzeros;
   JX_Real *aa = NULL;
   JX_Int    *ia = NULL;
   JX_Int    *ja = NULL;

   jx_CSRMatrix *A_rec = NULL;
   JX_Real *aarec = NULL;
   JX_Int    *iarec = NULL;
   JX_Int    *jarec = NULL;

   JX_Int  num_sends;                 // 发送进程个数
   JX_Int *send_procs_dbl  = NULL;    // 发送进程编号
   JX_Int *send_procs_int  = NULL;    // 发送进程编号
   JX_Int *send_starts_dbl = NULL;    // 实型数据发送缓存区的管理数组
   JX_Int *send_starts_int = NULL;    // 整型发送缓存区的管理数组

   JX_Int  num_recvs;                 // 接收进程个数
   JX_Int *recv_procs_dbl  = NULL;    // 接收进程编号
   JX_Int *recv_procs_int  = NULL;    // 接收进程编号
   JX_Int *recv_starts_dbl = NULL;    // 实型数据接收缓存区的管理数组
   JX_Int *recv_starts_int = NULL;    // 整型数据接收缓存区的管理数组

   JX_Real *send_buf_dbl = NULL;
   JX_Real *recv_buf_dbl = NULL;

   JX_Int *send_buf_int = NULL;
   JX_Int *recv_buf_int = NULL;

   jx_ParVecCommPkg    *comm_pkg_dbl    = NULL;
   jx_ParVecCommPkg    *comm_pkg_int    = NULL;
   jx_ParVecCommHandle *comm_handle_dbl = NULL;
   jx_ParVecCommHandle *comm_handle_int = NULL;

   JX_Int num_proc_eachgroup;
   JX_Int local_row;                   // 行数
   JX_Int local_nnz;                   // 非零元素个数

   JX_Int *local_nnz_array = NULL;
   JX_Int *local_row_array = NULL;

   JX_Int *size_recv = NULL;
   JX_Int *ver_cnt = NULL;

   JX_Int size_base_int;
   JX_Int size0, size1;

   JX_Int i, j, k, gidx, incmt;
   JX_Int row_cnt, ja_cnt, aa_cnt;
   JX_Int f_cnt, u_cnt;
   JX_Int vre_cnt, vei_cnt, vie_cnt;
   JX_Int start, scaa, scbb;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm, &nprocs);

   //====================================================================================
   // Step 1: 数据通信
   //====================================================================================

   for (gidx = 0; gidx < ng; gidx ++) ARR_comb[gidx] = jx_MergeDiagAndOffd(ARR_p[gidx]);
   AEE_comb = jx_MergeDiagAndOffd(AEE_p);
   AII_comb = jx_MergeDiagAndOffd(AII_p);

   local_nnz = jx_CSRMatrixNumNonzeros(ARR_comb[0]);
   local_row = jx_CSRMatrixNumRows(ARR_comb[0]);

   aa_E = jx_CSRMatrixData(AEE_comb);
   ia_E = jx_CSRMatrixI(AEE_comb);
   ja_E = jx_CSRMatrixJ(AEE_comb);

   aa_I = jx_CSRMatrixData(AII_comb);
   ia_I = jx_CSRMatrixI(AII_comb);
   ja_I = jx_CSRMatrixJ(AII_comb);

   num_proc_eachgroup = nprocs / (ng + 2);

   /* num_sends */
   num_sends = ng + 2;

   send_procs_dbl  = jx_CTAlloc(JX_Int, num_sends);
   send_procs_int  = jx_CTAlloc(JX_Int, num_sends);
   send_starts_dbl = jx_CTAlloc(JX_Int, num_sends+1);
   send_starts_int = jx_CTAlloc(JX_Int, num_sends+1);

   /* send_procs_dbl */
   send_procs_dbl[0] = myid / num_sends;
   for (gidx = 1; gidx < num_sends; gidx ++) send_procs_dbl[gidx] = send_procs_dbl[gidx-1] + num_proc_eachgroup;

   /* send_procs_int */
   for (gidx = 0; gidx < num_sends; gidx ++) send_procs_int[gidx] = send_procs_dbl[gidx];

   /* send_starts_dbl */
   size0 = local_nnz + 3 * local_row; // aa, vre, f, u
   size1 = size0 + ng * local_row; // aa, ver, vei, f, u
   send_starts_dbl[0] = 0;
   for (gidx = 0; gidx < ng; gidx ++) send_starts_dbl[gidx+1] = send_starts_dbl[gidx] + size0;
   send_starts_dbl[ng+1] = send_starts_dbl[ng] + size1;
   send_starts_dbl[ng+2] = send_starts_dbl[ng+1] + size0;

   /* send_starts_int */
   size_base_int = local_nnz + local_row + 1; // ia, ja
   send_starts_int[0] = 0;
   for (gidx = 0; gidx < num_sends; gidx ++) send_starts_int[gidx+1] = send_starts_int[gidx] + size_base_int;

   /* num_recvs */
   num_recvs = ng + 2;

   recv_procs_dbl  = jx_CTAlloc(JX_Int, num_recvs);
   recv_procs_int  = jx_CTAlloc(JX_Int, num_recvs);
   recv_starts_dbl = jx_CTAlloc(JX_Int, num_recvs+1);
   recv_starts_int = jx_CTAlloc(JX_Int, num_recvs+1);

   /* recv_procs_dbl */
   recv_procs_dbl[0] = (myid % num_proc_eachgroup) * num_recvs;
   for (gidx = 1; gidx < num_recvs; gidx ++) recv_procs_dbl[gidx] = recv_procs_dbl[gidx-1] + 1;

   /* recv_procs_int */
   for (gidx = 0; gidx < num_recvs; gidx ++) recv_procs_int[gidx] = recv_procs_dbl[gidx];

   /* 消息全收集 */
   local_nnz_array = jx_CTAlloc(JX_Int, nprocs);
   local_row_array = jx_CTAlloc(JX_Int, nprocs);
   jx_MPI_Allgather(&local_nnz, 1, JX_MPI_INT, local_nnz_array, 1, JX_MPI_INT, comm);
   jx_MPI_Allgather(&local_row, 1, JX_MPI_INT, local_row_array, 1, JX_MPI_INT, comm);

   /* recv_starts_dbl */
   size_recv = jx_CTAlloc(JX_Int, num_recvs);
   if (groupid_x == ng)
   {
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         size_recv[gidx] = local_nnz_array[recv_procs_dbl[gidx]] + (ng + 3) * local_row_array[recv_procs_dbl[gidx]];
      }
   }
   else
   {
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         size_recv[gidx] = local_nnz_array[recv_procs_dbl[gidx]] + 3 * local_row_array[recv_procs_dbl[gidx]];
      }
   }
   recv_starts_dbl[0] = 0;
   for (gidx = 0; gidx < num_recvs; gidx ++) recv_starts_dbl[gidx+1] = recv_starts_dbl[gidx] + size_recv[gidx];

   /* recv_starts_int */
   for (gidx = 0; gidx < num_recvs; gidx ++)
   {
      size_recv[gidx] = local_nnz_array[recv_procs_dbl[gidx]] + local_row_array[recv_procs_dbl[gidx]] + 1;
   }
   recv_starts_int[0] = 0;
   for (gidx = 0; gidx < num_recvs; gidx ++) recv_starts_int[gidx+1] = recv_starts_int[gidx] + size_recv[gidx];

   //-------------------------------------------------------------------------------
   //  开设发送和接收缓存区空间 send_buf 和 recv_buf
   //-------------------------------------------------------------------------------

   send_buf_dbl = jx_CTAlloc(JX_Real, send_starts_dbl[num_sends]);
   recv_buf_dbl = jx_CTAlloc(JX_Real, recv_starts_dbl[num_recvs]);
   send_buf_int = jx_CTAlloc(JX_Int, send_starts_int[num_sends]);
   recv_buf_int = jx_CTAlloc(JX_Int, recv_starts_int[num_recvs]);

   //-------------------------------------------
   //  填充发送缓存区空间 send_buf_dbl
   //-------------------------------------------

   k = 0;
   for (gidx = 0; gidx < ng; gidx ++)
   {
      aa_R = jx_CSRMatrixData(ARR_comb[gidx]);
      vre_data = jx_VectorData(jx_ParVectorLocalVector(VRE_p[gidx]));
      fr_data = jx_VectorData(jx_ParVectorLocalVector(fR_p[gidx]));
      ur_data = jx_VectorData(jx_ParVectorLocalVector(uR_p[gidx]));
      for (i = 0; i < local_nnz; i ++)
      {
         send_buf_dbl[k++] = aa_R[i];
      }
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = vre_data[i];
      }
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = fr_data[i];
      }
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = ur_data[i];
      }
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_E[i];
   }
   for (gidx = 0; gidx < ng; gidx ++)
   {
      ver_data = jx_VectorData(jx_ParVectorLocalVector(VER_p[gidx]));
      for (i = 0; i < local_row; i ++)
      {
         send_buf_dbl[k++] = ver_data[i];
      }
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vei_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fe_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ue_data[i];
   }
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_dbl[k++] = aa_I[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = vie_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = fi_data[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_dbl[k++] = ui_data[i];
   }

   //-------------------------------------------------------------------
   //  填充发送缓存区空间 send_buf_int
   //-------------------------------------------------------------------

   k = 0;
   for (gidx = 0; gidx < ng; gidx ++)
   {
      ia_R = jx_CSRMatrixI(ARR_comb[gidx]);
      ja_R = jx_CSRMatrixJ(ARR_comb[gidx]);
      for (i = 0; i < local_row; i ++)
      {
         send_buf_int[k++] = ia_R[i];
      }
      send_buf_int[k++] = ia_R[local_row];
      for (i = 0; i < local_nnz; i ++)
      {
         send_buf_int[k++] = ja_R[i];
      }
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_int[k++] = ia_E[i];
   }
   send_buf_int[k++] = ia_E[local_row];
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_E[i];
   }
   for (i = 0; i < local_row; i ++)
   {
      send_buf_int[k++] = ia_I[i];
   }
   send_buf_int[k++] = ia_I[local_row];
   for (i = 0; i < local_nnz; i ++)
   {
      send_buf_int[k++] = ja_I[i];
   }

   //-------------------------------------------------------
   //  创建通信包 comm_pkg_dbl，并填充其成员
   //-------------------------------------------------------

   comm_pkg_dbl = jx_CTAlloc(jx_ParVecCommPkg, 1);
   jx_ParVecCommPkgComm(comm_pkg_dbl)       = comm;
   jx_ParVecCommPkgNumSends(comm_pkg_dbl)   = num_sends;
   jx_ParVecCommPkgSendProcs(comm_pkg_dbl)  = send_procs_dbl;
   jx_ParVecCommPkgSendStarts(comm_pkg_dbl) = send_starts_dbl;
   jx_ParVecCommPkgNumRecvs(comm_pkg_dbl)   = num_recvs;
   jx_ParVecCommPkgRecvProcs(comm_pkg_dbl)  = recv_procs_dbl;
   jx_ParVecCommPkgRecvStarts(comm_pkg_dbl) = recv_starts_dbl;

   //-------------------------------------------------------
   //  创建通信包 comm_pkg_int，并填充其成员
   //-------------------------------------------------------

   comm_pkg_int = jx_CTAlloc(jx_ParVecCommPkg, 1);
   jx_ParVecCommPkgComm(comm_pkg_int)       = comm;
   jx_ParVecCommPkgNumSends(comm_pkg_int)   = num_sends;
   jx_ParVecCommPkgSendProcs(comm_pkg_int)  = send_procs_int;
   jx_ParVecCommPkgSendStarts(comm_pkg_int) = send_starts_int;
   jx_ParVecCommPkgNumRecvs(comm_pkg_int)   = num_recvs;
   jx_ParVecCommPkgRecvProcs(comm_pkg_int)  = recv_procs_int;
   jx_ParVecCommPkgRecvStarts(comm_pkg_int) = recv_starts_int;

   //-------------------------------------------------------
   //  通信: 发送和接收数据
   //-------------------------------------------------------
 
   comm_handle_dbl = jx_ParVecCommHandleCreate(1, comm_pkg_dbl, send_buf_dbl, recv_buf_dbl);
   comm_handle_int = jx_ParVecCommHandleCreate(11, comm_pkg_int, send_buf_int, recv_buf_int);

   /* some computational work can be done here to ovelap the communication! */
 
   jx_ParVecCommHandleDestroy(comm_handle_dbl);
   jx_ParVecCommHandleDestroy(comm_handle_int);

   //====================================================================================
   // Step 2: 将接收缓存区中的数据转化为本地串行矩阵和串行向量的形式,
   //         为进一步生成并行数据作准备.
   //====================================================================================

   num_rows = 0;
   num_nonzeros = 0;
   for (i = 0; i < num_recvs; i ++)
   {
      num_rows += local_row_array[recv_procs_int[i]];
      num_nonzeros += local_nnz_array[recv_procs_int[i]];
   }

   A_loc = jx_CSRMatrixCreate(num_rows, jx_CSRMatrixNumCols(ARR_comb[0]), num_nonzeros);
   jx_CSRMatrixInitialize(A_loc);
   aa = jx_CSRMatrixData(A_loc);
   ia = jx_CSRMatrixI(A_loc);
   ja = jx_CSRMatrixJ(A_loc);

   k = 0;       // 累加接收缓存区中的数据
   row_cnt = 0; // 累加矩阵 A 的行数
   ja_cnt  = 0; // 累加 ja 数组中的元素
   aa_cnt  = 0;  // 累加 aa 数组中的元素
   incmt   = 0;
   for (gidx = 0; gidx < num_recvs; gidx ++)
   {
      scaa = local_row_array[recv_procs_dbl[gidx]] + 1;
      for (i = 0; i < scaa; i ++)
      {
         ia[row_cnt++] = recv_buf_int[k++] + incmt;
      }
      scbb = local_nnz_array[recv_procs_dbl[gidx]];
      for (i = 0; i < scbb; i ++)
      {
         ja[ja_cnt++] = recv_buf_int[k++];
      }
      incmt += scbb;
      row_cnt --; // be careful here!
      start = recv_starts_dbl[gidx];
      for (i = 0; i < scbb; i ++)
      {
         aa[aa_cnt++] = recv_buf_dbl[start+i];
      }
   }

   f_loc = jx_SeqVectorCreate(num_rows);
   jx_SeqVectorInitialize(f_loc);
   u_loc = jx_SeqVectorCreate(num_rows);
   jx_SeqVectorInitialize(u_loc);

   if (groupid_x < ng) // 光子进程组
   {
      VRE_loc = jx_SeqVectorCreate(num_rows);
      jx_SeqVectorInitialize(VRE_loc);
      vre_cnt = 0;
      f_cnt   = 0;
      u_cnt   = 0;
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         start = recv_starts_dbl[gidx] + local_nnz_array[recv_procs_dbl[gidx]];
         scaa = local_row_array[recv_procs_dbl[gidx]];
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(VRE_loc)[vre_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(f_loc)[f_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(u_loc)[u_cnt++] = recv_buf_dbl[start+i];
         }
      }
   }
   else if (groupid_x == ng) // 电子进程组
   {
      VER_loc = jx_CTAlloc(jx_Vector *, ng);
      for (gidx = 0; gidx < ng; gidx ++)
      {
         VER_loc[gidx] = jx_SeqVectorCreate(num_rows);
         jx_SeqVectorInitialize(VER_loc[gidx]);
      }
      VEI_loc = jx_SeqVectorCreate(num_rows);
      jx_SeqVectorInitialize(VEI_loc);
      ver_cnt = jx_CTAlloc(JX_Int, ng);
      vei_cnt = 0;
      f_cnt   = 0;
      u_cnt   = 0;
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         start = recv_starts_dbl[gidx] + local_nnz_array[recv_procs_dbl[gidx]];
         scaa = local_row_array[recv_procs_dbl[gidx]];
         for (j = 0; j < ng; j ++)
         {
            for (i = 0; i < scaa; i ++)
            {
               jx_VectorData(VER_loc[j])[ver_cnt[j]++] = recv_buf_dbl[start+i];
            }
            start += scaa;
         }
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(VEI_loc)[vei_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(f_loc)[f_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(u_loc)[u_cnt++] = recv_buf_dbl[start+i];
         }
      }
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      VIE_loc = jx_SeqVectorCreate(num_rows);
      jx_SeqVectorInitialize(VIE_loc);
      vie_cnt = 0;
      f_cnt   = 0;
      u_cnt   = 0;
      for (gidx = 0; gidx < num_recvs; gidx ++)
      {
         start = recv_starts_dbl[gidx] + local_nnz_array[recv_procs_dbl[gidx]];
         scaa = local_row_array[recv_procs_dbl[gidx]];
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(VIE_loc)[vie_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(f_loc)[f_cnt++] = recv_buf_dbl[start+i];
         }
         start += scaa;
         for (i = 0; i < scaa; i ++)
         {
            jx_VectorData(u_loc)[u_cnt++] = recv_buf_dbl[start+i];
         }
      }
   }

   row_starts_loc = jx_CTAlloc(JX_Int, num_proc_eachgroup+1);
   for (i = 0; i <= num_proc_eachgroup; i ++)
   {
      row_starts_loc[i] = row_starts[(ng+2)*i];
   }
   col_starts_loc = row_starts_loc;

   if (groupid_x < ng) // 光子进程组
   {
      ARR = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, ARR, first_col_diag, last_col_diag);
      VRE = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VRE, 0);
      jx_TFree(jx_ParVectorLocalVector(VRE));
      jx_ParVectorLocalVector(VRE) = VRE_loc;
   }
   else if (groupid_x == ng) // 电子进程组
   {
      AEE = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, AEE, first_col_diag, last_col_diag);
      VER = jx_CTAlloc(jx_ParVector *, ng);
      for (gidx = 0; gidx < ng; gidx ++)
      {
         VER[gidx] = jx_ParVectorCreate(comm_x, n, row_starts_loc);
         jx_ParVectorSetPartitioningOwner(VER[gidx], 0);
         jx_TFree(jx_ParVectorLocalVector(VER[gidx]));
         jx_ParVectorLocalVector(VER[gidx]) = VER_loc[gidx];
      }
      VEI = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VEI, 0);
      jx_TFree(jx_ParVectorLocalVector(VEI));
      jx_ParVectorLocalVector(VEI) = VEI_loc;
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      AII = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, AII, first_col_diag, last_col_diag);
      VIE = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VIE, 0);
      jx_TFree(jx_ParVectorLocalVector(VIE));
      jx_ParVectorLocalVector(VIE) = VIE_loc;
   }

   row_starts_glo = jx_CTAlloc(JX_Int, nprocs+1);
   scaa = 0;
   scbb = 0;
   for (j = 0; j < ng; j ++)
   {
      for (i = 0; i < num_proc_eachgroup; i ++)
      {
         row_starts_glo[i+scbb] = row_starts_loc[i] + scaa;
      }
      scaa += n;
      scbb += num_proc_eachgroup;
   }
   for (i = 0; i < num_proc_eachgroup; i ++)
   {
      row_starts_glo[i+scbb] = row_starts_loc[i] + scaa;
   }
   scaa += n;
   scbb += num_proc_eachgroup;
   for (i = 0; i < num_proc_eachgroup; i ++)
   {
      row_starts_glo[i+scbb] = row_starts_loc[i] + scaa;
   }
   row_starts_glo[nprocs] = N;
   col_starts_glo = row_starts_glo;

   if (groupid_x == ng)
   {
      num_nonzeros = jx_CSRMatrixNumNonzeros(A_loc) + (ng + 1) * num_rows;
   }
   else
   {
      num_nonzeros = jx_CSRMatrixNumNonzeros(A_loc) + num_rows;
   }
   A_rec = jx_CSRMatrixCreate(num_rows, N, num_nonzeros);
   jx_CSRMatrixInitialize(A_rec);
   aarec = jx_CSRMatrixData(A_rec);
   iarec = jx_CSRMatrixI(A_rec);
   jarec = jx_CSRMatrixJ(A_rec);

   if (groupid_x < ng) // 光子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      scaa = groupid_x * n;
      for (i = 0; i < num_rows; i ++)
      {
         scbb = ia[i+1];
         for (j = ia[i]; j < scbb; j ++)
         {
            jarec[k] = ja[j] + scaa;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i + e_pos - scaa;
         aarec[k] = jx_VectorData(VRE_loc)[i];
         k ++;
         iarec[i+1] = k;
      }
   }
   else if (groupid_x == ng) // 电子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         scbb = ia[i+1];
         for (j = ia[i]; j < scbb; j ++)
         {
            jarec[k] = ja[j] + e_pos;
            aarec[k] = aa[j];
            k ++;
         }
         scaa = start + i - e_pos;
         for (j = 0; j < ng; j ++)
         {
            jarec[k] = scaa + j * n;
            aarec[k] = jx_VectorData(VER_loc[j])[i];
            k ++;
         }
         jarec[k] = start + i + n;
         aarec[k] = jx_VectorData(VEI_loc)[i];
         k ++;
         iarec[i+1] = k;
      }
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      k = 0;
      iarec[0] = 0;
      start = row_starts_glo[myid];
      for (i = 0; i < num_rows; i ++)
      {
         scbb = ia[i+1];
         for (j = ia[i]; j < scbb; j ++)
         {
            jarec[k] = ja[j] + i_pos;
            aarec[k] = aa[j];
            k ++;
         }
         jarec[k] = start + i - n;
         aarec[k] = jx_VectorData(VIE_loc)[i];
         k ++;
         iarec[i+1] = k;
      }
   }

   A = jx_ParCSRMatrixCreate(comm, N, N, row_starts_glo, col_starts_glo, 0, 0, 0);
   first_col_diag = col_starts_glo[myid];
   last_col_diag  = col_starts_glo[myid+1] - 1;
   jx_GenerateDiagAndOffd(A_rec, A, first_col_diag, last_col_diag);

   f = jx_ParVectorCreate(comm, N, row_starts_glo);
   jx_ParVectorSetPartitioningOwner(f, 0);
   jx_TFree(jx_ParVectorLocalVector(f));
   jx_ParVectorLocalVector(f) = f_loc;

   u = jx_ParVectorCreate(comm, N, row_starts_glo);
   jx_ParVectorSetPartitioningOwner(u, 0);
   jx_TFree(jx_ParVectorLocalVector(u));
   jx_ParVectorLocalVector(u) = u_loc;

   jx_TFree(send_buf_dbl);
   jx_TFree(recv_buf_dbl);
   jx_TFree(send_buf_int);
   jx_TFree(recv_buf_int);
   jx_TFree(local_nnz_array);
   jx_TFree(local_row_array);

   jx_ParVecCommPkgDestroy(comm_pkg_dbl);
   jx_ParVecCommPkgDestroy(comm_pkg_int);

   jx_TFree(size_recv);
   if (ver_cnt) jx_TFree(ver_cnt);

   for (gidx = 0; gidx < ng; gidx ++) jx_CSRMatrixDestroy(ARR_comb[gidx]);
   jx_TFree(ARR_comb);
   jx_CSRMatrixDestroy(AEE_comb);
   jx_CSRMatrixDestroy(AII_comb);

   jx_CSRMatrixDestroy(A_loc);
   jx_CSRMatrixDestroy(A_rec);

  *ARR_ptr = ARR;
  *AEE_ptr = AEE;
  *AII_ptr = AII;
  *VRE_ptr = VRE;
  *VER_ptr = VER;
  *VEI_ptr = VEI;
  *VIE_ptr = VIE;

  *A_ptr = A;
  *f_ptr = f;
  *u_ptr = u;

   return (0);
}

/*!
 * \fn JX_Int jx_MatVecGroup2All
 * \brief 将基于进程分组的并行矩阵向量转换为基于所有进程的并行矩阵向量,
 *        其并行分划均采用默认的负载平衡的分划.
 * \author peghoty 
 * \date 2012/03/02
 */
JX_Int 
jx_MatVecGroup2All( MPI_Comm          comm, 
                    JX_Int               groupid_x,
                    jx_ParCSRMatrix  *ARR, 
                    jx_ParCSRMatrix  *AEE, 
                    jx_ParCSRMatrix  *AII, 
                    jx_ParVector     *VRE, 
                    jx_ParVector     *VER, 
                    jx_ParVector     *VEI, 
                    jx_ParVector     *VIE,
                    jx_ParCSRMatrix **ARR_all_ptr, 
                    jx_ParCSRMatrix **AEE_all_ptr, 
                    jx_ParCSRMatrix **AII_all_ptr, 
                    jx_ParVector    **VRE_all_ptr, 
                    jx_ParVector    **VER_all_ptr, 
                    jx_ParVector    **VEI_all_ptr, 
                    jx_ParVector    **VIE_all_ptr )
{
   JX_Int myid, nprocs;
   JX_Int npeachgroup;
   JX_Int rootid_R, rootid_E, rootid_I;

   jx_CSRMatrix *ARR_s = NULL;
   jx_CSRMatrix *AEE_s = NULL;
   jx_CSRMatrix *AII_s = NULL;
   jx_Vector    *VRE_s = NULL;
   jx_Vector    *VER_s = NULL;
   jx_Vector    *VEI_s = NULL;
   jx_Vector    *VIE_s = NULL;

   jx_ParCSRMatrix *ARR_all = NULL; 
   jx_ParCSRMatrix *AEE_all = NULL;
   jx_ParCSRMatrix *AII_all = NULL; 
   jx_ParVector    *VRE_all = NULL; 
   jx_ParVector    *VER_all = NULL; 
   jx_ParVector    *VEI_all = NULL; 
   jx_ParVector    *VIE_all = NULL;
                    
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   // 三个进程组的根进程号
   npeachgroup = nprocs / 3;
   rootid_R = 0;
   rootid_E = npeachgroup; 
   rootid_I = 2*npeachgroup;
       
   if (groupid_x == 0)
   {
      ARR_s = jx_ParCSRMatrixToCSRMatrixAll(ARR);
      VRE_s = jx_ParVectorToVectorAll(VRE);
   }
   else if (groupid_x == 1)
   {
      AEE_s = jx_ParCSRMatrixToCSRMatrixAll(AEE);
      VER_s = jx_ParVectorToVectorAll(VER);
      VEI_s = jx_ParVectorToVectorAll(VEI);
   }
   else if (groupid_x == 2)
   {
      AII_s = jx_ParCSRMatrixToCSRMatrixAll(AII);
      VIE_s = jx_ParVectorToVectorAll(VIE); 
   }
   
   /* (从指定进程)串行矩阵转并行矩阵 */ 
   ARR_all = jx_CSRMatrixToParCSRMatrix_FromGivenPro(comm, rootid_R, ARR_s, NULL, NULL); 
   AEE_all = jx_CSRMatrixToParCSRMatrix_FromGivenPro(comm, rootid_E, AEE_s, NULL, NULL);
   AII_all = jx_CSRMatrixToParCSRMatrix_FromGivenPro(comm, rootid_I, AII_s, NULL, NULL);
   
   /* (从指定进程)串行向量转并行向量 */ 
   VRE_all = jx_VectorToParVector_FromGivenPro(comm, rootid_R, VRE_s, NULL);
   VER_all = jx_VectorToParVector_FromGivenPro(comm, rootid_E, VER_s, NULL);
   VEI_all = jx_VectorToParVector_FromGivenPro(comm, rootid_E, VEI_s, NULL);
   VIE_all = jx_VectorToParVector_FromGivenPro(comm, rootid_I, VIE_s, NULL);
   
   if (groupid_x == 0)
   {
      jx_CSRMatrixDestroy(ARR_s);
      jx_SeqVectorDestroy(VRE_s);
   }
   else if (groupid_x == 1)
   {
      jx_CSRMatrixDestroy(AEE_s);
      jx_SeqVectorDestroy(VER_s);
      jx_SeqVectorDestroy(VEI_s);
   }
   else if (groupid_x == 2)
   {
      jx_CSRMatrixDestroy(AII_s);
      jx_SeqVectorDestroy(VIE_s);
   }
 
   *ARR_all_ptr = ARR_all;
   *AEE_all_ptr = AEE_all;
   *AII_all_ptr = AII_all;
   *VRE_all_ptr = VRE_all;
   *VER_all_ptr = VER_all;
   *VEI_all_ptr = VEI_all;
   *VIE_all_ptr = VIE_all;

   return (0);
}

/*!
 * \fn JX_Int jx_3tDataTransFromSeq2SubPara
 * \brief 将串行矩阵 A_s 和串行向量 f_s，u_s 转化为子块的并行数据,
 *        其并行分划均采用默认的负载平衡的分划.
 * \author peghoty 
 * \date 2012/03/08
 */
JX_Int
jx_3tDataTransFromSeq2SubPara( JX_Int               iniguess,
                               MPI_Comm          comm,
                               jx_CSRMatrix     *A_s, 
                               jx_Vector        *f_s, 
                               jx_Vector        *u_s, 
                               jx_ParCSRMatrix **ARR_p_ptr, 
                               jx_ParCSRMatrix **AEE_p_ptr, 
                               jx_ParCSRMatrix **AII_p_ptr, 
                               jx_ParVector    **VRE_p_ptr, 
                               jx_ParVector    **VER_p_ptr, 
                               jx_ParVector    **VEI_p_ptr, 
                               jx_ParVector    **VIE_p_ptr, 
                               jx_ParVector    **fR_p_ptr, 
                               jx_ParVector    **fE_p_ptr, 
                               jx_ParVector    **fI_p_ptr,
                               jx_ParVector    **uR_p_ptr, 
                               jx_ParVector    **uE_p_ptr, 
                               jx_ParVector    **uI_p_ptr )
{
   /* 目标并行数据 */
   jx_ParCSRMatrix *ARR_p = NULL; 
   jx_ParCSRMatrix *AEE_p = NULL; 
   jx_ParCSRMatrix *AII_p = NULL; 
   jx_ParVector    *VRE_p = NULL; 
   jx_ParVector    *VER_p = NULL; 
   jx_ParVector    *VEI_p = NULL; 
   jx_ParVector    *VIE_p = NULL; 
   
   jx_ParVector    *fR_p  = NULL; 
   jx_ParVector    *fE_p  = NULL; 
   jx_ParVector    *fI_p  = NULL;
   
   jx_ParVector    *uR_p  = NULL; 
   jx_ParVector    *uE_p  = NULL;  
   jx_ParVector    *uI_p  = NULL;
   
   /* 辅助串行矩阵和向量 */
   jx_CSRMatrix    *ARR_s = NULL;
   jx_CSRMatrix    *AEE_s = NULL;
   jx_CSRMatrix    *AII_s = NULL;
   jx_Vector       *VRE_s = NULL;
   jx_Vector       *VER_s = NULL;
   jx_Vector       *VEI_s = NULL;
   jx_Vector       *VIE_s = NULL;

   jx_Vector       *fR_s  = NULL; 
   jx_Vector       *fE_s  = NULL; 
   jx_Vector       *fI_s  = NULL;
   
   jx_Vector       *uR_s  = NULL; 
   jx_Vector       *uE_s  = NULL;  
   jx_Vector       *uI_s  = NULL;
    
   JX_Int myid, nprocs;
   JX_Int n = 0;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs == 1)
   {
      //-------------------------------------------------------------------------------------
      //  抽取子矩阵和子向量
      //-------------------------------------------------------------------------------------   
      jx_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
      jx_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
      if (iniguess) 
      {
         jx_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
      }
      else
      { 
         n = jx_CSRMatrixNumRows(A_s) / 3; // 子矩阵的规模
      }

      //---------------------------------------------------------
      //  将串行矩阵和向量转换成并行矩阵和并行向量
      //---------------------------------------------------------      
      ARR_p = jx_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
      AEE_p = jx_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
      AII_p = jx_CSRMatrixToParCSRMatrix_sp(comm, AII_s);
      VRE_p = jx_VectorToParVector_sp(comm, VRE_s);
      VER_p = jx_VectorToParVector_sp(comm, VER_s);
      VEI_p = jx_VectorToParVector_sp(comm, VEI_s);
      VIE_p = jx_VectorToParVector_sp(comm, VIE_s);
      
      fR_p = jx_VectorToParVector_sp(comm, fR_s);
      fE_p = jx_VectorToParVector_sp(comm, fE_s);
      fI_p = jx_VectorToParVector_sp(comm, fI_s);     

      if (iniguess) 
      {
         uR_p = jx_VectorToParVector_sp(comm, uR_s);
         uE_p = jx_VectorToParVector_sp(comm, uE_s);
         uI_p = jx_VectorToParVector_sp(comm, uI_s);    
      }
      else
      {
         uR_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uR_p);
         uE_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uE_p);
         uI_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uI_p);
      }

      //---------------------------------------------------------
      //  释放辅助矩阵和向量(注意不要释放数据部分!)
      //---------------------------------------------------------        
      jx_TFree(ARR_s);
      jx_TFree(AEE_s);
      jx_TFree(AII_s);
      jx_TFree(VRE_s);
      jx_TFree(VER_s);
      jx_TFree(VEI_s);
      jx_TFree(VIE_s );
      jx_TFree(fR_s);
      jx_TFree(fE_s);
      jx_TFree(fI_s);
      
      if (iniguess) 
      {
         jx_TFree(uR_s);
         jx_TFree(uE_s);
         jx_TFree(uI_s);   
      }
   }
   else if (nprocs > 1)
   {    
      
      /* 前三个进程分别从串行矩阵 A_s 中抽取子块 */   
      if (myid == 0)
      {
         jx_3tGetSubBlocks_RV(A_s, &ARR_s, &VRE_s);
         jx_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
      }

      if (myid == 1)
      {
         jx_3tGetSubBlocks_EV(A_s, &AEE_s, &VER_s, &VEI_s);
      }
   
      if (myid == 2)
      {
         jx_3tGetSubBlocks_IV(A_s, &AII_s, &VIE_s);
         if (iniguess) 
         {
            jx_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
         }
         else
         {
            n = jx_CSRMatrixNumRows(A_s) / 3; 
         } 
      }
   
      /* 将子块转为并行矩阵和并行向量 */
      
      // 从 0 号进程转换 
      ARR_p = jx_CSRMatrixToParCSRMatrix(comm, ARR_s, NULL, NULL); 
      VRE_p = jx_VectorToParVector(comm, VRE_s, NULL);
      fR_p = jx_VectorToParVector(comm, fR_s, NULL);
      fE_p = jx_VectorToParVector(comm, fE_s, NULL);
      fI_p = jx_VectorToParVector(comm, fI_s, NULL);

      // 从 1 号进程转换 
      AEE_p = jx_CSRMatrixToParCSRMatrix_FromGivenPro(comm, 1, AEE_s, NULL, NULL); 
      VER_p = jx_VectorToParVector_FromGivenPro(comm, 1, VER_s, NULL);
      VEI_p = jx_VectorToParVector_FromGivenPro(comm, 1, VEI_s, NULL);

      // 从 2 号进程转换
      AII_p = jx_CSRMatrixToParCSRMatrix_FromGivenPro(comm, 2, AII_s, NULL, NULL); 
      VIE_p = jx_VectorToParVector_FromGivenPro(comm, 2, VIE_s, NULL);
      if (iniguess)
      {
         uR_p = jx_VectorToParVector_FromGivenPro(comm, 2, uR_s, NULL);
         uE_p = jx_VectorToParVector_FromGivenPro(comm, 2, uE_s, NULL);
         uI_p = jx_VectorToParVector_FromGivenPro(comm, 2, uI_s, NULL);
      }
      else
      {
         jx_MPI_Bcast(&n, 1, JX_MPI_INT, 2, comm);
         uR_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uR_p);
         uE_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uE_p);
         uI_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uI_p);            
      }
   
      /* 释放辅助矩阵和向量 */   
      if (myid == 0)
      {
         jx_CSRMatrixDestroy(ARR_s);
         jx_SeqVectorDestroy(VRE_s);
         jx_SeqVectorDestroy(fR_s);
         jx_SeqVectorDestroy(fE_s);
         jx_SeqVectorDestroy(fI_s);    
      }

      if (myid == 1)
      {
         jx_CSRMatrixDestroy(AEE_s);
         jx_SeqVectorDestroy(VER_s); 
         jx_SeqVectorDestroy(VEI_s);        
      }
   
      if (myid == 2)
      {
         jx_CSRMatrixDestroy(AII_s);
         jx_SeqVectorDestroy(VIE_s);

         if (iniguess)
         {
            jx_SeqVectorDestroy(uR_s);
            jx_SeqVectorDestroy(uE_s);
            jx_SeqVectorDestroy(uI_s);
         }
      }
      
   } // end if (nprocs > 1)

   
   //------------------------------------
   //  返回并行数据
   //------------------------------------ 
   *ARR_p_ptr = ARR_p; 
   *AEE_p_ptr = AEE_p; 
   *AII_p_ptr = AII_p; 
   *VRE_p_ptr = VRE_p; 
   *VER_p_ptr = VER_p;
   *VEI_p_ptr = VEI_p; 
   *VIE_p_ptr = VIE_p; 
   *fR_p_ptr  = fR_p; 
   *fE_p_ptr  = fE_p; 
   *fI_p_ptr  = fI_p;
   *uR_p_ptr  = uR_p; 
   *uE_p_ptr  = uE_p; 
   *uI_p_ptr  = uI_p;
   
   return 0;    
}

/*!
 * \fn JX_Int jx_3tDataTransFromSeq2SubPar0
 * \brief 将零号进程上的串行矩阵 A_s 和串行向量 f_s，u_s 转化为子块的并行数据,
 *        其并行分划均采用默认的负载平衡的分划.
 * \author peghoty 
 * \date 2012/03/08
 */
JX_Int
jx_3tDataTransFromSeq2SubPar0( JX_Int               iniguess,
                               MPI_Comm          comm,
                               jx_CSRMatrix     *A_s, 
                               jx_Vector        *f_s, 
                               jx_Vector        *u_s, 
                               jx_ParCSRMatrix **ARR_p_ptr, 
                               jx_ParCSRMatrix **AEE_p_ptr, 
                               jx_ParCSRMatrix **AII_p_ptr, 
                               jx_ParVector    **VRE_p_ptr, 
                               jx_ParVector    **VER_p_ptr, 
                               jx_ParVector    **VEI_p_ptr, 
                               jx_ParVector    **VIE_p_ptr, 
                               jx_ParVector    **fR_p_ptr, 
                               jx_ParVector    **fE_p_ptr, 
                               jx_ParVector    **fI_p_ptr,
                               jx_ParVector    **uR_p_ptr, 
                               jx_ParVector    **uE_p_ptr, 
                               jx_ParVector    **uI_p_ptr )
{
   /* 目标并行数据 */
   jx_ParCSRMatrix *ARR_p = NULL; 
   jx_ParCSRMatrix *AEE_p = NULL; 
   jx_ParCSRMatrix *AII_p = NULL; 
   jx_ParVector    *VRE_p = NULL; 
   jx_ParVector    *VER_p = NULL; 
   jx_ParVector    *VEI_p = NULL; 
   jx_ParVector    *VIE_p = NULL; 
   
   jx_ParVector    *fR_p  = NULL; 
   jx_ParVector    *fE_p  = NULL; 
   jx_ParVector    *fI_p  = NULL;
   
   jx_ParVector    *uR_p  = NULL; 
   jx_ParVector    *uE_p  = NULL;  
   jx_ParVector    *uI_p  = NULL;
   
   /* 辅助串行矩阵和向量 */
   jx_CSRMatrix    *ARR_s = NULL;
   jx_CSRMatrix    *AEE_s = NULL;
   jx_CSRMatrix    *AII_s = NULL;
   jx_Vector       *VRE_s = NULL;
   jx_Vector       *VER_s = NULL;
   jx_Vector       *VEI_s = NULL;
   jx_Vector       *VIE_s = NULL;

   jx_Vector       *fR_s  = NULL; 
   jx_Vector       *fE_s  = NULL; 
   jx_Vector       *fI_s  = NULL;
   
   jx_Vector       *uR_s  = NULL; 
   jx_Vector       *uE_s  = NULL;  
   jx_Vector       *uI_s  = NULL;
    
   JX_Int myid, nprocs;
   JX_Int global_size = 0;
   JX_Int n;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   if (nprocs == 1)
   {
      //-------------------------------------------------------------------------------------
      //  抽取子矩阵和子向量
      //-------------------------------------------------------------------------------------   
      jx_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
      jx_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
      if (iniguess) 
      {
         jx_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
      }
      else
      { 
         global_size = jx_CSRMatrixNumRows(A_s);
         n = global_size / 3;
      }

      //---------------------------------------------------------
      //  将串行矩阵和向量转换成并行矩阵和并行向量
      //---------------------------------------------------------      
      ARR_p = jx_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
      AEE_p = jx_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
      AII_p = jx_CSRMatrixToParCSRMatrix_sp(comm, AII_s);
      VRE_p = jx_VectorToParVector_sp(comm, VRE_s);
      VER_p = jx_VectorToParVector_sp(comm, VER_s);
      VEI_p = jx_VectorToParVector_sp(comm, VEI_s);
      VIE_p = jx_VectorToParVector_sp(comm, VIE_s);
      
      fR_p = jx_VectorToParVector_sp(comm, fR_s);
      fE_p = jx_VectorToParVector_sp(comm, fE_s);
      fI_p = jx_VectorToParVector_sp(comm, fI_s);     

      if (iniguess) 
      {
         uR_p = jx_VectorToParVector_sp(comm, uR_s);
         uE_p = jx_VectorToParVector_sp(comm, uE_s);
         uI_p = jx_VectorToParVector_sp(comm, uI_s);    
      }
      else
      {
         uR_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uR_p);
         uE_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uE_p);
         uI_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uI_p);
      }

      //---------------------------------------------------------
      //  释放辅助矩阵和向量(注意不要释放数据部分!)
      //---------------------------------------------------------        
      jx_TFree(ARR_s);
      jx_TFree(AEE_s);
      jx_TFree(AII_s);
      jx_TFree(VRE_s);
      jx_TFree(VER_s);
      jx_TFree(VEI_s);
      jx_TFree(VIE_s );
      jx_TFree(fR_s);
      jx_TFree(fE_s);
      jx_TFree(fI_s);
      
      if (iniguess) 
      {
         jx_TFree(uR_s);
         jx_TFree(uE_s);
         jx_TFree(uI_s);   
      }
   }
   else if (nprocs > 1)
   {    
      //----------------------------------------------------------------------------------------
      //  在 0 号进程，抽取子矩阵和子向量
      //----------------------------------------------------------------------------------------
      if (myid == 0)
      {
         jx_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s); 
         jx_3tGetSubVecs(f_s, &fR_s, &fE_s, &fI_s);
         if (iniguess) 
         {
            jx_3tGetSubVecs(u_s, &uR_s, &uE_s, &uI_s);
         }
         else
         { 
            global_size = jx_CSRMatrixNumRows(A_s);
            n = global_size / 3;
         }
      }

      //-------------------------------------------------------------------------
      //  将串行矩阵和向量转换成并行矩阵和并行向量
      //-------------------------------------------------------------------------    
      ARR_p = jx_CSRMatrixToParCSRMatrix(comm, ARR_s, NULL, NULL);
      AEE_p = jx_CSRMatrixToParCSRMatrix(comm, AEE_s, NULL, NULL);
      AII_p = jx_CSRMatrixToParCSRMatrix(comm, AII_s, NULL, NULL);
      VRE_p = jx_VectorToParVector(comm, VRE_s, NULL);
      VER_p = jx_VectorToParVector(comm, VER_s, NULL);
      VEI_p = jx_VectorToParVector(comm, VEI_s, NULL);
      VIE_p = jx_VectorToParVector(comm, VIE_s, NULL);

      fR_p = jx_VectorToParVector(comm, fR_s, NULL);
      fE_p = jx_VectorToParVector(comm, fE_s, NULL);
      fI_p = jx_VectorToParVector(comm, fI_s, NULL);
   
      if (iniguess) 
      {
         uR_p = jx_VectorToParVector(comm, uR_s, NULL);
         uE_p = jx_VectorToParVector(comm, uE_s, NULL);
         uI_p = jx_VectorToParVector(comm, uI_s, NULL);
      }
      else
      {
         jx_MPI_Bcast(&n, 1, JX_MPI_INT, 0, comm);
         uR_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uR_p);
         uE_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uE_p);
         uI_p = jx_ParVectorCreate(comm, n, NULL);
         jx_ParVectorInitialize(uI_p);
      } 
   
      //--------------------------------------------
      //  释放辅助矩阵和向量
      //--------------------------------------------
      if (myid == 0)
      {
         jx_CSRMatrixDestroy(ARR_s);
         jx_CSRMatrixDestroy(AEE_s);
         jx_CSRMatrixDestroy(AII_s);
         jx_SeqVectorDestroy(VRE_s);
         jx_SeqVectorDestroy(VER_s); 
         jx_SeqVectorDestroy(VEI_s);
         jx_SeqVectorDestroy(VIE_s); 
      
         jx_SeqVectorDestroy(fR_s);
         jx_SeqVectorDestroy(fE_s);
         jx_SeqVectorDestroy(fI_s);

         if (iniguess)
         {
            jx_SeqVectorDestroy(uR_s);
            jx_SeqVectorDestroy(uE_s);
            jx_SeqVectorDestroy(uI_s);
         }
      }
      
   }  // end if (nprocs > 1)
   

   //------------------------------------
   //  返回并行数据
   //------------------------------------        
   *ARR_p_ptr = ARR_p; 
   *AEE_p_ptr = AEE_p; 
   *AII_p_ptr = AII_p; 
   *VRE_p_ptr = VRE_p; 
   *VER_p_ptr = VER_p;
   *VEI_p_ptr = VEI_p; 
   *VIE_p_ptr = VIE_p; 
   *fR_p_ptr  = fR_p; 
   *fE_p_ptr  = fE_p; 
   *fI_p_ptr  = fI_p;
   *uR_p_ptr  = uR_p; 
   *uE_p_ptr  = uE_p; 
   *uI_p_ptr  = uI_p;
   
   return 0;    
}

/*!
 * \fn JX_Int jx_ParaDataTransSEQIF_mp
 * \brief 将 0 号进程上的串行矩阵 A_s 和串行向量 f_s，u_s 转化为以下三类并行数据:
 *  (1) 整体离散系统基于整个进程组的并行数据，且具有匹配的并行分划(用于 GMRES); 
 *  (2) 子块基于进程分组的并行数据(用于 apctl 迭代);
 *  (3) 子块基于整体进程的并行数据(用于生成 pctl 算法中的粗矩阵 Ac).
 * \author peghoty 
 * \date 2012/03/09
 */
JX_Int  
jx_ParaDataTransSEQIF_mp( // input:
                          MPI_Comm           comm, 
                          MPI_Comm           comm_x, 
                          JX_Int                groupid_x,
                          JX_Real             theta_wc_E, 
                          JX_Real             threshold_wc_E,
                          /* 串行离散系统数据 */
                          jx_CSRMatrix      *A_s, 
                          jx_Vector         *f_s, 
                          jx_Vector         *u_s, 
                          // output: 
                          /* 系数矩阵各子块在整个进程组上的并行数据 */
                          jx_ParCSRMatrix  **ARR_p_ptr, 
                          jx_ParCSRMatrix  **AEE_p_ptr, 
                          jx_ParCSRMatrix  **AII_p_ptr, 
                          jx_ParVector     **VRE_p_ptr, 
                          jx_ParVector     **VER_p_ptr, 
                          jx_ParVector     **VEI_p_ptr, 
                          jx_ParVector     **VIE_p_ptr, 
                          /* 系数矩阵各子块基于进程组的并行数据 */                         
                          jx_ParCSRMatrix  **ARR_ptr,
                          jx_ParCSRMatrix  **AEE_ptr,
                          jx_ParCSRMatrix  **AII_ptr, 
                          jx_ParVector     **VRE_ptr, 
                          jx_ParVector     **VER_ptr, 
                          jx_ParVector     **VEI_ptr, 
                          jx_ParVector     **VIE_ptr,
                          /* 并行离散系统数据 */
                          jx_ParCSRMatrix  **A_ptr,  
                          jx_ParVector     **f_ptr, 
                          jx_ParVector     **u_ptr,
                          /* 是否需要粗网格校正的标志变量 */
                          JX_Int               *Need_CC_ptr )
{
   JX_Int myid, nprocs;
   JX_Int np_R, np_E, np_I;
   JX_Int rootid_E;
   JX_Int N;
   JX_Int *mypartition = NULL;
   JX_Int *A_partition = NULL;
       
   /* 系数矩阵各子块在整个进程组上的并行数据 */
   jx_ParCSRMatrix  *ARR_p = NULL; 
   jx_ParCSRMatrix  *AEE_p = NULL; 
   jx_ParCSRMatrix  *AII_p = NULL; 
   jx_ParVector     *VRE_p = NULL; 
   jx_ParVector     *VER_p = NULL; 
   jx_ParVector     *VEI_p = NULL; 
   jx_ParVector     *VIE_p = NULL; 
   
   /* 系数矩阵各子块基于进程组的并行数据 */     
   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL; 
   jx_ParVector     *VRE = NULL; 
   jx_ParVector     *VER = NULL; 
   jx_ParVector     *VEI = NULL; 
   jx_ParVector     *VIE = NULL;
   
   /* 并行离散系统数据 */
   jx_ParCSRMatrix  *A = NULL;  
   jx_ParVector     *f = NULL;
   jx_ParVector     *u = NULL;
   
   /* 是否需要粗网格校正的标志变量 */
   JX_Int Need_CC;
   
   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);
   
   np_R = nprocs / 3;
   np_E = np_R; 
   np_I = np_R;
   rootid_E = np_R;
   
   if (myid == 0)
   {
      N = jx_CSRMatrixNumRows(A_s);        
   }
   
   //================================================================
   //  根据变量 np_R, np_E, np_I，生成基于进程分组的（向量）分划数组
   //  mypartition, 该数组将用来作为矩阵 A 的行列分划数组.
   //  Note: 这样可以提高块磨光时的并行效率!
   //  Note: 当 np = 1 时，该模块生成长度为 2 的一般分划数组.
   //================================================================

   if (myid == 0) 
   {
      jx_3tGetMyPartition(nprocs, np_R, np_E, np_I, N, &mypartition);
   }
 
   
   //==============================================================================
   //  生成 (1) 整体离散系统基于整个进程组的并行数据，且具有匹配的并行分划(用于 GMRES)
   //==============================================================================
         
   A = jx_CSRMatrixToParCSRMatrix(comm, A_s, mypartition, mypartition);
   A_partition = jx_ParCSRMatrixRowStarts(A);
   f = jx_VectorToParVector(comm, f_s, A_partition);
   jx_ParVectorSetPartitioningOwner(f, 0); 
   u = jx_VectorToParVector(comm, u_s, A_partition);
   jx_ParVectorSetPartitioningOwner(u, 0); 


   //==============================================================================
   //  生成 (2) 子块基于进程分组的并行数据(用于 apctl 迭代)
   //==============================================================================
     
   if (groupid_x == 0)
   {  
      jx_MatVecGroupR(comm_x, A, &ARR, &VRE);
   }
   else if (groupid_x == 1)
   {
      jx_MatVecGroupE(comm_x, A, &AEE, &VER, &VEI);
      Need_CC = jx_3tAPCTLWeakCouplingE(theta_wc_E, threshold_wc_E, AEE, VER, VEI);  
   }
   else if (groupid_x == 2)
   {
      jx_MatVecGroupI(comm_x, A, &AII, &VIE);
   }   
   jx_MPI_Bcast(&Need_CC, 1, JX_MPI_INT, rootid_E, comm); /* 将标志变量广播给所有进程 */


   //==============================================================================
   //  生成 (3) 子块基于整体进程的并行数据(用于生成 pctl 算法中的粗矩阵 Ac).
   //==============================================================================

   if (Need_CC == 1)
   {   
      /* 将基于进程分组的并行矩阵向量转换为基于所有进程的并行矩阵向量，为生成粗矩阵作数据准备 */
      jx_MatVecGroup2All( comm, groupid_x, 
                          ARR, AEE, AII, VRE, VER, VEI, VIE,
                          &ARR_p, &AEE_p, &AII_p, 
                          &VRE_p, &VER_p, &VEI_p, &VIE_p );  
   }
   
   
   //==========================================================
   //  返回并行数据和标志变量
   //==========================================================
   
   *ARR_p_ptr = ARR_p;
   *AEE_p_ptr = AEE_p;
   *AII_p_ptr = AII_p; 
   *VRE_p_ptr = VRE_p; 
   *VER_p_ptr = VER_p; 
   *VEI_p_ptr = VEI_p; 
   *VIE_p_ptr = VIE_p;   
   *ARR_ptr = ARR;
   *AEE_ptr = AEE;
   *AII_ptr = AII;
   *VRE_ptr = VRE;
   *VER_ptr = VER;
   *VEI_ptr = VEI;
   *VIE_ptr = VIE;
   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;
   *Need_CC_ptr = Need_CC;
   
   return (0);
}                                        

/*!
 * \fn JX_Int jx_ParaDataTransSEQIF_sp
 * \brief 将串行矩阵 A_s 和串行向量 f_s，u_s 转化为两类并行数据.
 * \author peghoty 
 * \date 2012/03/10
 */
JX_Int  
jx_ParaDataTransSEQIF_sp( // input:
                          MPI_Comm           comm, 
                          /* 串行离散系统数据 */
                          jx_CSRMatrix      *A_s, 
                          jx_Vector         *f_s, 
                          jx_Vector         *u_s, 
                          // output:
                          /* 系数矩阵各子块基于进程组的并行数据 */                         
                          jx_ParCSRMatrix  **ARR_ptr,
                          jx_ParCSRMatrix  **AEE_ptr,
                          jx_ParCSRMatrix  **AII_ptr, 
                          jx_ParVector     **VRE_ptr, 
                          jx_ParVector     **VER_ptr, 
                          jx_ParVector     **VEI_ptr, 
                          jx_ParVector     **VIE_ptr,
                          /* 并行离散系统数据 */
                          jx_ParCSRMatrix  **A_ptr,  
                          jx_ParVector     **f_ptr, 
                          jx_ParVector     **u_ptr )                    
{
   /* 系数矩阵各子块基于进程组的并行数据 */     
   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL; 
   jx_ParVector     *VRE = NULL; 
   jx_ParVector     *VER = NULL; 
   jx_ParVector     *VEI = NULL; 
   jx_ParVector     *VIE = NULL;

   /* 系数矩阵各子块 */     
   jx_CSRMatrix  *ARR_s = NULL;
   jx_CSRMatrix  *AEE_s = NULL;
   jx_CSRMatrix  *AII_s = NULL; 
   jx_Vector     *VRE_s = NULL; 
   jx_Vector     *VER_s = NULL; 
   jx_Vector     *VEI_s = NULL; 
   jx_Vector     *VIE_s = NULL;
   
   /* 并行离散系统数据 */
   jx_ParCSRMatrix  *A = NULL;  
   jx_ParVector     *f = NULL;
   jx_ParVector     *u = NULL;

   //-------------------------------------------------------------------
   // 生成并行离散系统
   //-------------------------------------------------------------------
   A = jx_CSRMatrixToParCSRMatrix(comm, A_s, NULL, NULL);
   f = jx_VectorToParVector(comm, f_s, NULL);
   u = jx_VectorToParVector(comm, u_s, NULL);

   //------------------------------------------------------------------------------------
   // 生成子块的并行数据
   //------------------------------------------------------------------------------------
   jx_3tGetSubBlocks_REIV(A_s, &ARR_s, &AEE_s, &AII_s, &VRE_s, &VER_s, &VEI_s, &VIE_s);
   
   ARR = jx_CSRMatrixToParCSRMatrix_sp(comm, ARR_s);
   AEE = jx_CSRMatrixToParCSRMatrix_sp(comm, AEE_s);
   AII = jx_CSRMatrixToParCSRMatrix_sp(comm, AII_s);
   
   VRE = jx_VectorToParVector_sp(comm, VRE_s);
   VER = jx_VectorToParVector_sp(comm, VER_s);
   VEI = jx_VectorToParVector_sp(comm, VEI_s);
   VIE = jx_VectorToParVector_sp(comm, VIE_s);
   
   /* 释放串行数据(注意不要释放数据部分!) */
   jx_TFree(ARR_s);
   jx_TFree(AEE_s);
   jx_TFree(AII_s);
   jx_TFree(VRE_s);
   jx_TFree(VER_s);
   jx_TFree(VEI_s);
   jx_TFree(VIE_s);
   
   //----------------------------------
   // 返回并行数据
   //----------------------------------
   *ARR_ptr = ARR;
   *AEE_ptr = AEE;
   *AII_ptr = AII;
   *VRE_ptr = VRE;
   *VER_ptr = VER;
   *VEI_ptr = VEI;
   *VIE_ptr = VIE;
   *A_ptr = A;
   *f_ptr = f;
   *u_ptr = u;

   return (0);
}            

/*!
 * \fn JX_Int jx_APCTLKrylovSolGather
 * \brief 收集解向量，将并行向量收集到 0 号进程已经开设空间的串行向量上.
 * \author peghoty 
 * \date 2012/03/10
 */
JX_Int 
jx_APCTLKrylovSolGather( jx_ParVector *par_sol, jx_Vector *u_s )
{
   MPI_Comm  comm = jx_ParVectorComm(par_sol);

   JX_Int *partitioning = jx_ParVectorPartitioning(par_sol);
   JX_Int *recvcounts = NULL;   
   JX_Real *sendbuf = NULL;
   JX_Real *recvbuf = NULL;
   
   JX_Int i, sendcount;
   JX_Int myid, nprocs;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &nprocs);

   /* 在 0 号进程生成辅助数组 recvcounts */
   if (myid == 0)
   {
      recvcounts = jx_CTAlloc(JX_Int, nprocs);
      for (i = 0; i < nprocs; i ++)
      {
         recvcounts[i] = partitioning[i+1] - partitioning[i];
      }    
   }

   /* 确定 0 号进程需从各进程接收数据的个数 */
   sendcount = jx_VectorSize(jx_ParVectorLocalVector(par_sol));

   /* 将各进程上的解向量分量收集到 0 号进程的串行解向量中 */
   sendbuf = jx_VectorData(jx_ParVectorLocalVector(par_sol));
   if (myid == 0) recvbuf = jx_VectorData(u_s);   
   jx_MPI_Gatherv(sendbuf, sendcount, JX_MPI_REAL, recvbuf, recvcounts, partitioning, JX_MPI_REAL, 0, comm); 

   /* 释放数组 recvcounts */    
   if (myid == 0)
   {
      jx_TFree(recvcounts);
   }
   
   return (0);
}

JX_Int 
jx_mgGenerateSubBlocks( MPI_Comm           comm,
                        MPI_Comm           comm_x,
                        JX_Int             groupid_x,
                        JX_Int             ng,
                        jx_ParCSRMatrix   *par_mat,
                        jx_ParCSRMatrix  **ARR_ptr,
                        jx_ParCSRMatrix  **AEE_ptr,
                        jx_ParCSRMatrix  **AII_ptr,
                        jx_ParVector     **VRE_ptr,
                        jx_ParVector    ***VER_ptr,
                        jx_ParVector     **VEI_ptr,
                        jx_ParVector     **VIE_ptr )
{
   jx_ParCSRMatrix  *ARR = NULL;
   jx_ParCSRMatrix  *AEE = NULL;
   jx_ParCSRMatrix  *AII = NULL;
   jx_ParVector     *VRE = NULL;
   jx_ParVector    **VER = NULL;
   jx_ParVector     *VEI = NULL;
   jx_ParVector     *VIE = NULL;

   jx_CSRMatrix *A_comb  = NULL;
   jx_CSRMatrix *A_loc   = NULL;
   jx_Vector    *VRE_loc = NULL;
   jx_Vector   **VER_loc = NULL;
   jx_Vector    *VEI_loc = NULL;
   jx_Vector    *VIE_loc = NULL;

   JX_Real *aa_comb = NULL;
   JX_Int  *ia_comb = NULL;
   JX_Int  *ja_comb = NULL;
   JX_Real *aa_loc = NULL;
   JX_Int  *ia_loc = NULL;
   JX_Int  *ja_loc = NULL;

   JX_Int *row_starts_glo = jx_ParCSRMatrixRowStarts(par_mat);
   JX_Int *ercnt = jx_CTAlloc(JX_Int, ng);
   JX_Int *row_starts_loc = NULL;
   JX_Int *col_starts_loc = NULL;

   JX_Int N = jx_ParCSRMatrixGlobalNumRows(par_mat);
   JX_Int n = N / (ng + 2);

   JX_Int myid, myid_x, nprocs, num_proc_pg;
   JX_Int incre, decre, row, j, col, first_col_diag, last_col_diag;
   JX_Int srt_row, end_row, eee_row, iii_row;
   JX_Int nz_rr, nz_re, nz_ee, nz_ei, nz_ie, nz_ii;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_rank(comm_x, &myid_x);
   jx_MPI_Comm_size(comm, &nprocs);

   num_proc_pg = nprocs / (ng + 2);
   incre = groupid_x * num_proc_pg;
   row_starts_loc = jx_CTAlloc(JX_Int, num_proc_pg+1);
   for (j = 0; j <= num_proc_pg; j ++) row_starts_loc[j] = row_starts_glo[incre+j] - row_starts_glo[incre];
   col_starts_loc = row_starts_loc;

   A_comb = jx_MergeDiagAndOffd(par_mat);
   aa_comb = jx_CSRMatrixData(A_comb);
   ia_comb = jx_CSRMatrixI(A_comb);
   ja_comb = jx_CSRMatrixJ(A_comb);
   if (groupid_x == ng)
   {
      decre = (ng + 1) * jx_CSRMatrixNumRows(A_comb);
   }
   else
   {
      decre = jx_CSRMatrixNumRows(A_comb);
   }
   A_loc = jx_CSRMatrixCreate(jx_CSRMatrixNumRows(A_comb), n, jx_CSRMatrixNumNonzeros(A_comb)-decre);
   jx_CSRMatrixInitialize(A_loc);
   aa_loc = jx_CSRMatrixData(A_loc);
   ia_loc = jx_CSRMatrixI(A_loc);
   ja_loc = jx_CSRMatrixJ(A_loc);

   if (groupid_x < ng) // 光子进程组
   {
      VRE_loc = jx_SeqVectorCreate(jx_CSRMatrixNumRows(A_comb));
      jx_SeqVectorInitialize(VRE_loc);
   }
   else if (groupid_x == ng) // 电子进程组
   {
      VER_loc = jx_CTAlloc(jx_Vector *, ng);
      for (j = 0; j < ng; j ++)
      {
         VER_loc[j] = jx_SeqVectorCreate(jx_CSRMatrixNumRows(A_comb));
         jx_SeqVectorInitialize(VER_loc[j]);
      }
      VEI_loc = jx_SeqVectorCreate(jx_CSRMatrixNumRows(A_comb));
      jx_SeqVectorInitialize(VEI_loc);
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      VIE_loc = jx_SeqVectorCreate(jx_CSRMatrixNumRows(A_comb));
      jx_SeqVectorInitialize(VIE_loc);
   }

   eee_row = ng * n;
   iii_row = eee_row + n;
   if (groupid_x < ng) // 光子进程组
   {
      srt_row = groupid_x * n;
      end_row = srt_row + n;
      nz_rr = 0;
      nz_re = 0;
	  ia_loc[0] = 0;
      for (row = 0; row < jx_CSRMatrixNumRows(A_comb); row ++)
      {
         for (j = ia_comb[row]; j < ia_comb[row+1]; j ++)
         {
            col = ja_comb[j];
            if ((col >= srt_row) && (col < end_row))
            {
               ja_loc[nz_rr] = col - srt_row;
               aa_loc[nz_rr] = aa_comb[j];
               nz_rr ++;
            }
            else if ((col >= eee_row) && (col < iii_row))
            {
               jx_VectorData(VRE_loc)[nz_re++] = aa_comb[j];
            }
            else if (col >= iii_row)
            {
               jx_printf("\033[31m >>> Ar[%d]i != 0\033[00m\n", groupid_x);
               exit(0);
            }
            else
            {
               jx_printf("\033[31m >>> Ar[%d]r[%d]-[%d][%d] != 0\033[00m\n", groupid_x, col/n, row, col);
               exit(0);
            }
         }
         ia_loc[row+1] = nz_rr;
      }
      ARR = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, ARR, first_col_diag, last_col_diag);
      VRE = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VRE, 0);
      jx_TFree(jx_ParVectorLocalVector(VRE));
      jx_ParVectorLocalVector(VRE) = VRE_loc;
   }
   else if (groupid_x == ng) // 电子进程组
   {
      nz_ee = 0;
      nz_ei = 0;
      ia_loc[0] = 0;
      for (row = 0; row < jx_CSRMatrixNumRows(A_comb); row ++)
      {
         for (j = ia_comb[row]; j < ia_comb[row+1]; j ++)
         {
            col = ja_comb[j];
            if (col < eee_row)
            {
               jx_VectorData(VER_loc[col/n])[ercnt[col/n]++] = aa_comb[j];
            }
            else  if ((col >= eee_row) && (col < iii_row))
            {
               ja_loc[nz_ee] = col - eee_row;
               aa_loc[nz_ee] = aa_comb[j];
               nz_ee ++;
            }
            else
            {
               jx_VectorData(VEI_loc)[nz_ei++] = aa_comb[j];
            }
         }
         ia_loc[row+1] = nz_ee;
      }
      AEE = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, AEE, first_col_diag, last_col_diag);
      VER = jx_CTAlloc(jx_ParVector *, ng);
      for (j = 0; j < ng; j ++)
      {
         VER[j] = jx_ParVectorCreate(comm_x, n, row_starts_loc);
         jx_ParVectorSetPartitioningOwner(VER[j], 0);
         jx_TFree(jx_ParVectorLocalVector(VER[j]));
         jx_ParVectorLocalVector(VER[j]) = VER_loc[j];
      }
      VEI = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VEI, 0);
      jx_TFree(jx_ParVectorLocalVector(VEI));
      jx_ParVectorLocalVector(VEI) = VEI_loc;
   }
   else if (groupid_x == ng+1) // 离子进程组
   {
      nz_ii = 0;
      nz_ie = 0;
      ia_loc[0] = 0;
      for (row = 0; row < jx_CSRMatrixNumRows(A_comb); row ++)
      {
         for (j = ia_comb[row]; j < ia_comb[row+1]; j ++)
         {
            col = ja_comb[j];
            if (col < eee_row)
            {
               jx_printf("\033[31m >>> Air[%d] != 0\033[00m\n", col/n);
               exit(0);
            }
            else if (col < iii_row)
            {
               jx_VectorData(VIE_loc)[nz_ie++] = aa_comb[j];
            }
            else
            {
               ja_loc[nz_ii] = col - iii_row;
               aa_loc[nz_ii] = aa_comb[j];
               nz_ii ++;
            }
         }
         ia_loc[row+1] = nz_ii;
      }
      AII = jx_ParCSRMatrixCreate(comm_x, n, n, row_starts_loc, col_starts_loc, 0, 0, 0);
      first_col_diag = col_starts_loc[myid_x];
      last_col_diag  = col_starts_loc[myid_x+1] - 1;
      jx_GenerateDiagAndOffd(A_loc, AII, first_col_diag, last_col_diag);
      VIE = jx_ParVectorCreate(comm_x, n, row_starts_loc);
      jx_ParVectorSetPartitioningOwner(VIE, 0);
      jx_TFree(jx_ParVectorLocalVector(VIE));
      jx_ParVectorLocalVector(VIE) = VIE_loc;
   }

   jx_CSRMatrixDestroy(A_comb);
   jx_CSRMatrixDestroy(A_loc);
   jx_TFree(ercnt);

  *ARR_ptr = ARR;
  *AEE_ptr = AEE;
  *AII_ptr = AII;
  *VRE_ptr = VRE;
  *VER_ptr = VER;
  *VEI_ptr = VEI;
  *VIE_ptr = VIE;

   return (0);
}
