/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!  
 *  par_lsdata.c
 *  
 *  Created by peghoty  2010/11/22
 *  Xiangtan University
 *  peghoty@163.com
 *
 */

#include "jx_multils.h"

/*!
 * \fn fsls_ParLSData * fsls_ParLSDataCreate
 * \brief Create a fsls_ParLSData object
 * \param num_rows number of rows of each coefficient matrix
 * \param num_cols number of cols of each coefficient matrix 
 * \param num_nonzeros number of nonzeros of each coefficient matrix
 * \param local_num_ls number of sub linear systems for the current processor
 * \param global_num_ls number of all the sub linear systems
 * \param *ls_partition the partition for linear system data distributing
 * \param *num_ls_procs how many linear systems in each processor?  
 * \author peghoty
 * \date 2010/11/22 
 */   
fsls_ParLSData *
fsls_ParLSDataCreate( MPI_Comm  comm,
                      JX_Int       num_rows, 
                      JX_Int       num_cols, 
                      JX_Int       num_nonzeros,
                      JX_Int       global_num_ls, 
                      JX_Int      *ls_partition,
                      JX_Int      *num_ls_procs )
{
   JX_Int i,myid,nprocs;
   fsls_ParLSData *ls_data = fsls_CTAlloc(fsls_ParLSData, 1);
   
   fsls_MPICommInformation(comm, &myid, &nprocs);
   
   if ( !ls_partition && !num_ls_procs)
   {
      /* generate 'ls_partition' and 'num_ls_procs' */
      fsls_BalancedPartition(global_num_ls, nprocs, &ls_partition, &num_ls_procs);   
   }
   else if (!ls_partition && num_ls_procs)
   {
      /* generate 'ls_partition' using 'num_ls_procs' */
      ls_partition = fsls_CTAlloc(JX_Int, nprocs+1);
      ls_partition[0] = 0;
      for (i = 0; i < nprocs; i ++)
      {
         ls_partition[i+1] = ls_partition[i] + num_ls_procs[i];
      } 
   }
   else if (ls_partition && !num_ls_procs)
   {
      /* generate 'num_ls_procs' using 'ls_partition' */
      num_ls_procs = fsls_CTAlloc(JX_Int, nprocs);
      for (i = 0; i < nprocs; i ++)
      {
         num_ls_procs[i] = ls_partition[i+1] - ls_partition[i];
      }       
   }
   
   fsls_ParLSDataComm(ls_data)        = comm;
   fsls_ParLSDataLocalNumLS(ls_data)  = num_ls_procs[myid];
   fsls_ParLSDataGlobalNumLS(ls_data) = global_num_ls;
   fsls_ParLSDataNumRows(ls_data)     = num_rows;
   fsls_ParLSDataNumCols(ls_data)     = num_cols; 
   fsls_ParLSDataNumNonzeros(ls_data) = num_nonzeros;
   fsls_ParLSDataLSPartition(ls_data) = ls_partition;
   fsls_ParLSDataNumLSProcs(ls_data)  = num_ls_procs;
   fsls_ParLSDataAArray(ls_data)      = NULL;
   fsls_ParLSDataBArray(ls_data)      = NULL;  
   fsls_ParLSDataXArray(ls_data)      = NULL;
   
   return (ls_data);    
}

/*!
 * \fn JX_Int fsls_ParLSDataInitialize
 * \brief Initialize a fsls_ParLSData object
 * \param *ls_data the fsls_ParLSData object to be initialized
 * \author peghoty
 * \date 2010/11/22 
 */ 
JX_Int
fsls_ParLSDataInitialize( fsls_ParLSData *ls_data )
{
   JX_Int local_num_ls = fsls_ParLSDataLocalNumLS(ls_data);
   JX_Int num_rows     = fsls_ParLSDataNumRows(ls_data);
   JX_Int num_cols     = fsls_ParLSDataNumCols(ls_data); 
   JX_Int num_nonzeros = fsls_ParLSDataNumNonzeros(ls_data);    
   JX_Int i, ierr      = 0;

   if (local_num_ls)
   {
      /* A_array */ 
      if ( !fsls_ParLSDataAArray(ls_data) )
      {
         fsls_ParLSDataAArray(ls_data) = fsls_CTAlloc(fsls_CSRMatrix *, local_num_ls);
         for (i = 0; i < local_num_ls; i ++)
         {
            fsls_ParLSDataAArray(ls_data)[i] = fsls_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
            fsls_CSRMatrixInitialize(fsls_ParLSDataAArray(ls_data)[i]);
         }
      }

      /* b_array */ 
      if ( !fsls_ParLSDataBArray(ls_data) )
      {
         fsls_ParLSDataBArray(ls_data) = fsls_CTAlloc(fsls_Vector *, local_num_ls);
         for (i = 0; i < local_num_ls; i ++)
         {
            fsls_ParLSDataBArray(ls_data)[i] = fsls_SeqVectorCreate(num_rows);
            fsls_SeqVectorInitialize(fsls_ParLSDataBArray(ls_data)[i]);            
         }
      }

      /* x_array */ 
      if ( !fsls_ParLSDataXArray(ls_data) )
      {
         fsls_ParLSDataXArray(ls_data) = fsls_CTAlloc(fsls_Vector *, local_num_ls);
         for (i = 0; i < local_num_ls; i ++)
         {
            fsls_ParLSDataXArray(ls_data)[i] = fsls_SeqVectorCreate(num_rows);
            fsls_SeqVectorInitialize(fsls_ParLSDataXArray(ls_data)[i]);            
         }
      }
   }
   
   return ierr;
}

/*!
 * \fn void fsls_ParLSDataDestroy
 * \brief Destroy a fsls_ParLSData object
 * \param *ls_data the fsls_ParLSData object to be destroyed
 * \author peghoty
 * \date 2010/11/22 
 */
void
fsls_ParLSDataDestroy( fsls_ParLSData *ls_data )
{
   MPI_Comm comm = fsls_ParLSDataComm(ls_data); 
   JX_Int i,myid;
   JX_Int local_num_ls;
   
   jx_MPI_Comm_rank(comm, &myid);
   
   if (ls_data)
   {
      local_num_ls = fsls_ParLSDataLocalNumLS(ls_data);
     
      if ( fsls_ParLSDataLSPartition(ls_data) )
      {
         fsls_TFree(fsls_ParLSDataLSPartition(ls_data));
      }

      if ( fsls_ParLSDataNumLSProcs(ls_data) )
      {
         fsls_TFree(fsls_ParLSDataNumLSProcs(ls_data));
      }

      if ( fsls_ParLSDataXArray(ls_data) )
      {
         for (i = 0; i < local_num_ls; i ++)
         {
            if ( fsls_ParLSDataXArray(ls_data)[i] )
            {
               fsls_SeqVectorDestroy(fsls_ParLSDataXArray(ls_data)[i]);
            }
         }
         fsls_TFree(fsls_ParLSDataXArray(ls_data));
      }  

      if ( fsls_ParLSDataAArray(ls_data) )
      {
         for (i = 0; i < local_num_ls; i ++)
         {
            if ( fsls_ParLSDataAArray(ls_data)[i] )
            {  
               fsls_CSRMatrixDestroy(fsls_ParLSDataAArray(ls_data)[i]);
            }
         }
         fsls_TFree(fsls_ParLSDataAArray(ls_data));
      }

      if ( fsls_ParLSDataBArray(ls_data) )
      {
         for (i = 0; i < local_num_ls; i ++)
         {
            if ( fsls_ParLSDataBArray(ls_data)[i] )
            {
               fsls_SeqVectorDestroy(fsls_ParLSDataBArray(ls_data)[i]);
            }
         }
         fsls_TFree(fsls_ParLSDataBArray(ls_data));
      }       
     
      fsls_TFree(ls_data);
   }
}

/*!
 * \fn fsls_ParLSData *fsls_GenerateLSData
 * \brief Generate a fsls_ParLSData object by reading data from files
 * \param comm communicator
 * \param global_num_ls number of all the sub linear systems
 * \param **MatFile_array MatFile_array[i] is the pointer to the i-th matrix file name
 * \param **RhsFile_array RhsFile_array[i] is the pointer to the i-th vector file name
 * \author peghoty
 * \date 2010/11/24 
 */ 
fsls_ParLSData *
fsls_GenerateLSData( MPI_Comm comm, JX_Int global_num_ls, char **MatFile_array, char **RhsFile_array )
{
   fsls_ParLSData *ls_data = NULL;

   fsls_CSRMatrix *A = NULL;
   fsls_Vector    *b = NULL;    

   fsls_CSRMatrix **A_array = NULL;
   fsls_Vector    **b_array = NULL;

   JX_Int    *sendbuf_IA = NULL;
   JX_Int    *sendbuf_JA = NULL;
   JX_Real *sendbuf_A  = NULL;
   JX_Real *sendbuf_F  = NULL;
   JX_Int    *sendbuf_int    = NULL;
   JX_Real *sendbuf_double = NULL;
   
   JX_Int    *recvbuf_IA = NULL;
   JX_Int    *recvbuf_JA = NULL;
   JX_Real *recvbuf_A  = NULL;
   JX_Real *recvbuf_F  = NULL;
   JX_Int    *recvbuf_int    = NULL;
   JX_Real *recvbuf_double = NULL;   
   
   JX_Int *ls_partition = NULL;
   JX_Int *num_ls_procs = NULL;
   JX_Int local_num_ls;
   JX_Int i,j,k,m;
   JX_Int cnt_ia,cnt_ja,cnt_a,cnt_f;
   JX_Int num_rows,num_cols,num_nonzeros; 
   
   MPI_Status status;
   JX_Int myid,nprocs;
   
   fsls_MPICommInformation(comm, &myid, &nprocs);   

   //===========================================================
   //  Read the matrices and vectors data 
   //  from files in the root processor.
   //===========================================================

   if (myid == 0)
   {
      A_array = fsls_CTAlloc(fsls_CSRMatrix *, global_num_ls);
      b_array = fsls_CTAlloc(fsls_Vector *, global_num_ls);

      //-----------------------------------------------
      // Read the matrix and vector
      //-----------------------------------------------
      
      for (i = 0; i < global_num_ls; i ++)
      {
         fsls_BuildCSRMatFromFile(MatFile_array[i], &A_array[i]);
         fsls_BuildVecFromFile(RhsFile_array[i], &b_array[i]);
      }
      
      num_rows     = fsls_CSRMatrixNumRows(A_array[0]);
      num_cols     = fsls_CSRMatrixNumCols(A_array[0]);
      num_nonzeros = fsls_CSRMatrixNumNonzeros(A_array[0]);   
   }
 
   //=======================================================================
   //  Prepare 'num_rows', 'num_cols', 'num_nonzeros' for each processor.
   //======================================================================= 

   jx_MPI_Bcast(&num_rows, 1, JX_MPI_INT, 0, comm);
   jx_MPI_Bcast(&num_cols, 1, JX_MPI_INT, 0, comm);
   jx_MPI_Bcast(&num_nonzeros, 1, JX_MPI_INT, 0, comm);
  
   //=======================================================================
   //  Generate 'ls_data' in each processor.
   //======================================================================= 

   ls_data = fsls_ParLSDataCreate(comm, num_rows, num_cols, 
                                  num_nonzeros, global_num_ls, NULL, NULL);
   fsls_ParLSDataInitialize(ls_data);
   local_num_ls = fsls_ParLSDataLocalNumLS(ls_data);
   ls_partition = fsls_ParLSDataLSPartition(ls_data);    
   num_ls_procs = fsls_ParLSDataNumLSProcs(ls_data);
  
   //=======================================================================
   //  Distribute the data from the root processor, and other processors
   //  receive the corresponding data.
   //======================================================================= 
  
   if (myid == 0)
   {
      //------------------------------------------------------------------
      // Fill in the 'ls_data' itself
      //------------------------------------------------------------------

      for (i = 0; i < local_num_ls; i ++)
      {
         fsls_CSRMatrixCopy(A_array[i], fsls_ParLSDataAArray(ls_data)[i], 1);
         fsls_SeqVectorCopy(b_array[i], fsls_ParLSDataBArray(ls_data)[i]);
      }

      //------------------------------------------------------------------
      // Allocate memery for sending
      //------------------------------------------------------------------
      k = num_ls_procs[0];
      for (i = 1; i < nprocs; i ++) // find k = max{num_ls_procs[i]}
      { 
         if (num_ls_procs[i] > k) k = num_ls_procs[i];
      }
      
      sendbuf_int    = fsls_CTAlloc(JX_Int, k*(num_nonzeros+num_rows+1));
      sendbuf_double = fsls_CTAlloc(JX_Real, k*(num_nonzeros+num_rows));
      sendbuf_IA = sendbuf_int;
      sendbuf_JA = sendbuf_int + k*(num_rows+1);
      sendbuf_A  = sendbuf_double;
      sendbuf_F  = sendbuf_double + k*num_nonzeros;

      //------------------------------------------------------------------
      // Prepare the data for sending
      //------------------------------------------------------------------   
      for (i = 1; i < nprocs; i ++)
      {  
         // number of sub linear system to be sent
         k = num_ls_procs[i];

         // fill the sendbuf
         cnt_ia = 0; cnt_ja = 0; cnt_a  = 0; cnt_f  = 0;
         for (j = ls_partition[i]; j < ls_partition[i+1]; j ++)
         {
            for (m = 0; m < num_rows+1; m ++)
            {
               sendbuf_IA[cnt_ia++] = fsls_CSRMatrixI(A_array[j])[m];
            }
            for (m = 0; m < num_nonzeros; m ++)
            {
               sendbuf_JA[cnt_ja++] = fsls_CSRMatrixJ(A_array[j])[m];
            }
            for (m = 0; m < num_nonzeros; m ++)
            {
               sendbuf_A[cnt_a++] = fsls_CSRMatrixData(A_array[j])[m];
            }
            for (m = 0; m < num_rows; m ++)
            {
               sendbuf_F[cnt_f++] = fsls_VectorData(b_array[j])[m];
            }                      
         }
         
         //------------------------------------------------------------------
         // The root processor Send data to other processors
         //------------------------------------------------------------------   
         jx_MPI_Send(sendbuf_IA, k*(num_rows+1), JX_MPI_INT,    i, i*111, comm);
         jx_MPI_Send(sendbuf_JA, k*num_nonzeros, JX_MPI_INT,    i, i*222, comm);
         jx_MPI_Send(sendbuf_A,  k*num_nonzeros, JX_MPI_REAL, i, i*333, comm);
         jx_MPI_Send(sendbuf_F,  k*num_rows,     JX_MPI_REAL, i, i*444, comm);   
      }
   }
   else
   {
      //------------------------------------------------------------------
      // Allocate memery for receiving
      //------------------------------------------------------------------ 
      recvbuf_int    = fsls_CTAlloc(JX_Int, local_num_ls*(num_rows+num_nonzeros+1));
      recvbuf_double = fsls_CTAlloc(JX_Real, local_num_ls*(num_nonzeros+num_rows));

      recvbuf_IA = recvbuf_int;
      recvbuf_JA = recvbuf_IA + local_num_ls*(num_rows+1);
      recvbuf_A  = recvbuf_double;
      recvbuf_F  = recvbuf_A + local_num_ls*num_nonzeros;
             
      //-------------------------------------------------------------------------------------
      // Other processors Receive data from the root processor
      //-------------------------------------------------------------------------------------
      jx_MPI_Recv(recvbuf_IA, local_num_ls*(num_rows+1), JX_MPI_INT,    0, myid*111, comm, &status);
      jx_MPI_Recv(recvbuf_JA, local_num_ls*num_nonzeros, JX_MPI_INT,    0, myid*222, comm, &status);
      jx_MPI_Recv(recvbuf_A,  local_num_ls*num_nonzeros, JX_MPI_REAL, 0, myid*333, comm, &status);
      jx_MPI_Recv(recvbuf_F,  local_num_ls*num_rows,     JX_MPI_REAL, 0, myid*444, comm, &status);

      //---------------------------------------------------------------------
      // Generate 'ls_data'
      //---------------------------------------------------------------------
      
      for (i = 0; i < local_num_ls; i ++)
      {
         A = fsls_ParLSDataAArray(ls_data)[i];

         memcpy(fsls_CSRMatrixI(A),    &(recvbuf_IA[i*(num_rows+1)]), (num_rows+1)*sizeof(JX_Int));
         memcpy(fsls_CSRMatrixJ(A),    &(recvbuf_JA[i*num_nonzeros]), num_nonzeros*sizeof(JX_Int));
         memcpy(fsls_CSRMatrixData(A), &(recvbuf_A[i*num_nonzeros]),  num_nonzeros*sizeof(JX_Real));
         
         b = fsls_ParLSDataBArray(ls_data)[i];

         memcpy(fsls_VectorData(b), &(recvbuf_F[i*num_rows]), num_rows*sizeof(JX_Real));
      }
   }

   if (myid == 0)
   {
      fsls_TFree(sendbuf_int);
      fsls_TFree(sendbuf_double);
      fsls_TFree(recvbuf_int);
      fsls_TFree(recvbuf_double);      
      for (i = 0; i < global_num_ls; i ++)
      {
         if (A_array[i]) fsls_CSRMatrixDestroy(A_array[i]);
         if (b_array[i]) fsls_SeqVectorDestroy(b_array[i]);
      }
      fsls_TFree(A_array);
      fsls_TFree(b_array);  
   }

   return ls_data;
}
