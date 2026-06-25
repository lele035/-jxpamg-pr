/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/
 
/*!
 *  par_gls2ls.c
 *  
 *  Created by peghoty  2010/11/22
 *  Xiangtan University
 *  peghoty@163.com
 *
 */

#include "jx_multils.h"

/*!
 * \fn fsls_ParLSData *fsls_ParGLS2LSData
 * \brief Transfer a fsls_ParGLSData object to a fsls_ParLSData object
 * \param *gls_data pointer to a fsls_ParGLSData object
 * \author peghoty
 * \date 2010/11/22 
 */  
fsls_ParLSData *
fsls_ParGLS2LSData( fsls_ParGLSData *gls_data )
{
   MPI_Comm          comm           = fsls_ParGLSDataComm(gls_data);
   JX_Int               global_num_ls  = fsls_ParGLSDataGlobalNumLS(gls_data);      
   JX_Int               global_num_dof = fsls_ParGLSDataGlobalNumDof(gls_data);  
   JX_Int               local_num_dof  = fsls_ParGLSDataLocalNumDof(gls_data);   
   JX_Int              *dof_partition  = fsls_ParGLSDataDofPartition(gls_data);        
   fsls_CSRMatrix  **subA_array     = fsls_ParGLSDataSubAArray(gls_data);      
   fsls_Vector     **subb_array     = fsls_ParGLSDataSubBArray(gls_data);
   fsls_Vector     **subx_array     = fsls_ParGLSDataSubXArray(gls_data); 
    
   fsls_ParLSData   *ls_data = NULL;
   JX_Int              *ls_partition = NULL;
   fsls_CSRMatrix  **A_array = NULL;
   fsls_Vector     **b_array = NULL; 
   fsls_Vector     **x_array = NULL;

   JX_Int local_num_ls;
   JX_Int local_num_nonzeros;
   JX_Int global_num_nonzeros;

   JX_Int rhs_recv_size;
   JX_Int rhs_send_size;
   JX_Int mat_recv_size;
   JX_Int mat_send_size; 
   JX_Int mv_recv_size;
   JX_Int mv_send_size; 
       
   JX_Real *rhs_recvbuf = NULL;
   JX_Real *rhs_sendbuf = NULL;   
   JX_Real *mat_recvbuf = NULL;
   JX_Real *mat_sendbuf = NULL;
   JX_Real *mv_commbuf  = NULL; 
   JX_Real *my_app_data = NULL;   
   JX_Real *my_rhs_data = NULL;
   JX_Real *my_mat_data = NULL;
   JX_Real *app_data    = NULL;
   JX_Real *rhs_data    = NULL;
   JX_Real *mat_data    = NULL; 

   JX_Int *ia_sub      = NULL;   
   JX_Int *ia_recvbuf  = NULL;
   JX_Int *ja_sendbuf  = NULL;
   JX_Int *ja_recvbuf  = NULL;
   JX_Int *recvcounts  = NULL;
   JX_Int *displs      = NULL;  
   JX_Int *locnzperRow = NULL;
   JX_Int *manage      = NULL;        

   MPI_Status  *Status   = NULL;
   MPI_Request *Requests = NULL;
   
   JX_Int i,j,k,m;
   JX_Int begin,end,length;
   JX_Int num_requests;
   JX_Int myid,nprocs;

   fsls_MPICommInformation(comm, &myid, &nprocs);

   //========================================================================//
   //         Step 1     Create a fsls_ParLSData struct                      //
   //========================================================================//
   
   local_num_nonzeros = fsls_CSRMatrixNumNonzeros(subA_array[0]);
   jx_MPI_Allreduce(&local_num_nonzeros, &global_num_nonzeros, 1, JX_MPI_INT, MPI_SUM, comm);
   ls_data = fsls_ParLSDataCreate( comm, global_num_dof, global_num_dof, 
                                   global_num_nonzeros, global_num_ls, NULL, NULL );
   fsls_ParLSDataInitialize(ls_data);
   ls_partition = fsls_ParLSDataLSPartition(ls_data); 
   local_num_ls = fsls_ParLSDataLocalNumLS(ls_data);                            
   A_array      = fsls_ParLSDataAArray(ls_data);
   b_array      = fsls_ParLSDataBArray(ls_data);
   x_array      = fsls_ParLSDataXArray(ls_data);
 
   //========================================================================//
   //             Step 2     Allocate memory for send- and recv-buf          //
   //========================================================================// 

   manage = fsls_CTAlloc(JX_Int, local_num_dof+2*nprocs+1);
   recvcounts  = manage;               // size: nprocs
   displs      = recvcounts + nprocs;  // size: nprocs+1
   locnzperRow = displs + nprocs + 1;  // size: local_num_dof
   
   num_requests = 2*(nprocs-1);
   Requests = fsls_CTAlloc(MPI_Request, num_requests);
   Status   = fsls_CTAlloc(MPI_Status,  num_requests);
   
   rhs_recv_size = (global_num_dof - local_num_dof)*local_num_ls;
   mat_recv_size = (global_num_nonzeros - local_num_nonzeros)*local_num_ls;
   mv_recv_size  = fsls_max(rhs_recv_size, mat_recv_size);

   rhs_send_size = (global_num_ls - local_num_ls)*local_num_dof;
   mat_send_size = (global_num_ls - local_num_ls)*local_num_nonzeros;
   mv_send_size  = fsls_max(rhs_send_size, mat_send_size);

   mv_commbuf = fsls_CTAlloc(JX_Real, mv_recv_size + mv_send_size);
   
   rhs_recvbuf = mv_commbuf;
   rhs_sendbuf = mv_commbuf + mv_recv_size;
   
   mat_recvbuf = rhs_recvbuf;
   mat_sendbuf = rhs_sendbuf;
    
   //========================================================================//
   //        Step 3     Deal with the vectors: b_array and x_array           //
   //========================================================================//

   //----------------------------------------------
   //  Fill the rhs_sendbuf
   //----------------------------------------------

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid) // send data to other processors
      {
         begin = ls_partition[i];
         end   = ls_partition[i+1];
         for (j = begin; j < end; j ++)  // loop for the local_num_ls sub-rhs in the i-th processor
         {
            rhs_data = fsls_VectorData(subb_array[j]);
            memcpy(&rhs_sendbuf[k], rhs_data, local_num_dof*sizeof(JX_Real));
            k += local_num_dof;
         }
      }
   }

   //----------------------------------------------
   //  Receive data
   //----------------------------------------------

   for (i = 0, k = 0, m = 0; i < nprocs; i ++)
   {
      if (i != myid) // receive data from the i-th processors
      {
         length = local_num_ls*(dof_partition[i+1] - dof_partition[i]);
         jx_MPI_Irecv(&rhs_recvbuf[k], length, JX_MPI_REAL, i, myid*321, comm, &Requests[m++]);
         k += length;
      }
   } 
      
   //----------------------------------------------
   //  Send data
   //----------------------------------------------   

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid)  // send data to the i-th processor
      {
         length = local_num_dof*(ls_partition[i+1] - ls_partition[i]);
         jx_MPI_Isend(&rhs_sendbuf[k], length, JX_MPI_REAL, i, i*321, comm, &Requests[m++]);
         k += length;
      }
   } 

   jx_MPI_Waitall(num_requests, Requests, Status);

   //------------------------------------------------------
   //  Copy the sub-rhs data in the current processor
   //------------------------------------------------------
   
   begin = dof_partition[myid];
   for (i = 0; i < local_num_ls; i ++)
   {
      j = ls_partition[myid]+ i; // fix a bug on 2010/11/25
      my_rhs_data = fsls_VectorData(subb_array[j]);
      rhs_data    = fsls_VectorData(b_array[i]);
      memcpy(&rhs_data[begin], my_rhs_data, local_num_dof*sizeof(JX_Real));
   }

   //------------------------------------------------------
   //  Fill the sub-rhs data from other processors
   //------------------------------------------------------ 

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid)
      {
         begin  = dof_partition[i];
         length = dof_partition[i+1] - begin;
         
         for (j = 0; j < local_num_ls; j ++)
         {
            rhs_data = fsls_VectorData(b_array[j]);
            memcpy(&rhs_data[begin], &rhs_recvbuf[k], length*sizeof(JX_Real));
            k += length;
         }
      }
   }
   
   
   //----------------------------------------------
   //  Fill the rhs_sendbuf
   //----------------------------------------------

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid) // send data to other processors
      {
         begin = ls_partition[i];
         end   = ls_partition[i+1];
         for (j = begin; j < end; j ++)  // loop for the local_num_ls sub-app in the i-th processor
         {
            app_data = fsls_VectorData(subx_array[j]);
            memcpy(&rhs_sendbuf[k], app_data, local_num_dof*sizeof(JX_Real));
            k += local_num_dof;
         }
      }
   }

   //----------------------------------------------
   //  Receive data
   //----------------------------------------------

   for (i = 0, k = 0, m = 0; i < nprocs; i ++)
   {
      if (i != myid) // receive data from the i-th processors
      {
         length = local_num_ls*(dof_partition[i+1] - dof_partition[i]);
         jx_MPI_Irecv(&rhs_recvbuf[k], length, JX_MPI_REAL, i, myid*321, comm, &Requests[m++]);
         k += length;
      }
   } 
      
   //----------------------------------------------
   //  Send data
   //----------------------------------------------   

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid)  // send data to the i-th processor
      {
         length = local_num_dof*(ls_partition[i+1] - ls_partition[i]);
         jx_MPI_Isend(&rhs_sendbuf[k], length, JX_MPI_REAL, i, i*321, comm, &Requests[m++]);
         k += length;
      }
   } 

   jx_MPI_Waitall(num_requests, Requests, Status);

   //------------------------------------------------------
   //  Copy the sub-app data in the current processor
   //------------------------------------------------------
   
   begin = dof_partition[myid];
   for (i = 0; i < local_num_ls; i ++)
   {
      j = ls_partition[myid]+ i; // fix a bug on 2010/11/25
      my_app_data = fsls_VectorData(subx_array[j]);
      app_data    = fsls_VectorData(x_array[i]);
      memcpy(&app_data[begin], my_app_data, local_num_dof*sizeof(JX_Real));
   }

   //------------------------------------------------------
   //  Fill the sub-app data from other processors
   //------------------------------------------------------ 

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid)
      {
         begin  = dof_partition[i];
         length = dof_partition[i+1] - begin;
         
         for (j = 0; j < local_num_ls; j ++)
         {
            app_data = fsls_VectorData(x_array[j]);
            memcpy(&app_data[begin], &rhs_recvbuf[k], length*sizeof(JX_Real));
            k += length;
         }
      }
   }   
   

   //========================================================================//
   //                   Step 4     Deal with the matrices                    //
   //========================================================================//  

   //--------------------------------------------------------//
   //  4.1   Treat the 'ja' for CSR format of each matrix    //
   //--------------------------------------------------------// 
   
   ja_sendbuf = fsls_CSRMatrixJ(subA_array[0]);
   ja_recvbuf = fsls_CSRMatrixJ(A_array[0]);
   
   jx_MPI_Allgather( &local_num_nonzeros, 1, JX_MPI_INT, recvcounts, 1, JX_MPI_INT, comm );
  
   displs[0] = 0;
   for (i = 0; i < nprocs; i ++)
   {
      displs[i+1] = displs[i] + recvcounts[i];
   }
      
   jx_MPI_Allgatherv( ja_sendbuf, local_num_nonzeros, JX_MPI_INT, 
                   ja_recvbuf, recvcounts, displs, JX_MPI_INT, comm );
   
   /* copy the 'ja' from A_array[0] to A_array[i], i = 1,...,local_num_ls-1 */                
   for (i = 1; i < local_num_ls; i ++)
   {
      memcpy(fsls_CSRMatrixJ(A_array[i]), fsls_CSRMatrixJ(A_array[0]), global_num_nonzeros*sizeof(JX_Int));
   }

   //----------------------------------------------------------//
   //  4.2   Treat the 'ia' for CSR format of each matrix      //
   //----------------------------------------------------------// 
 
   ia_sub = fsls_CSRMatrixI(subA_array[0]);

   for (i = 0; i < local_num_dof; i ++)
   {
      locnzperRow[i] = ia_sub[i+1] - ia_sub[i];
   }

   jx_MPI_Allgather( &local_num_dof, 1, JX_MPI_INT, recvcounts, 1, JX_MPI_INT, comm );

   
   ia_recvbuf = fsls_CSRMatrixI(A_array[0]);
   jx_MPI_Allgatherv( locnzperRow, local_num_dof, JX_MPI_INT, 
                   &ia_recvbuf[1], recvcounts, dof_partition, JX_MPI_INT, comm ); 
 
   ia_recvbuf[0] = 0;
   for (i = 0; i < global_num_dof; i ++)
   {
      ia_recvbuf[i+1] += ia_recvbuf[i];
   }

   /* copy the 'ia' from A_array[0] to A_array[i], i = 1,...,local_num_ls-1 */                
   for (i = 1; i < local_num_ls; i ++)
   {
      memcpy(fsls_CSRMatrixI(A_array[i]), fsls_CSRMatrixI(A_array[0]), (global_num_dof+1)*sizeof(JX_Int));
   }

 
   //--------------------------------------------------------//
   //  4.3   Treat the 'a' for CSR format of each matrix     //
   //--------------------------------------------------------// 

   //----------------------------------------------
   //  Fill the mat_sendbuf
   //----------------------------------------------

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid) // send data to other processors
      {
         begin = ls_partition[i];
         end   = ls_partition[i+1];
         for (j = begin; j < end; j ++)  // loop for the local_num_ls sub-rhs in the i-th processor
         {
            mat_data = fsls_CSRMatrixData(subA_array[j]);
            memcpy(&mat_sendbuf[k], mat_data, local_num_nonzeros*sizeof(JX_Real));
            k += local_num_nonzeros;
         }
      }
   }

   //----------------------------------------------
   //  Receive data
   //----------------------------------------------

   jx_MPI_Allgather( &local_num_nonzeros, 1, JX_MPI_INT, recvcounts, 1, JX_MPI_INT, comm );      
   for (i = 0, k = 0, m = 0; i < nprocs; i ++)
   {
      if (i != myid) // receive data from the i-th processors
      {
         length = local_num_ls*recvcounts[i];
         jx_MPI_Irecv(&mat_recvbuf[k], length, JX_MPI_REAL, i, myid*123, comm, &Requests[m++]);
         k += length;
      }
   } 
     
   //----------------------------------------------
   //  Send data
   //----------------------------------------------   
   
   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid)  // send data to the i-th processor
      {
         length = local_num_nonzeros*(ls_partition[i+1] - ls_partition[i]);
         jx_MPI_Isend(&mat_sendbuf[k], length, JX_MPI_REAL, i, i*123, comm, &Requests[m++]);
         k += length;
      }
   } 
   
   jx_MPI_Waitall(num_requests, Requests, Status); 

   //------------------------------------------------------
   //  Copy the sub-mat data in the current processor
   //------------------------------------------------------
   
   displs[0] = 0;
   for (i = 0; i < nprocs; i ++)
   {
      displs[i+1] = displs[i] + recvcounts[i];
   }
   
   begin = displs[myid];
   for (i = 0; i < local_num_ls; i ++)
   {
      //my_mat_data = fsls_CSRMatrixData(subA_array[i]);
      j = ls_partition[myid]+ i; // fix a bug on 2010/11/25
      my_mat_data = fsls_CSRMatrixData(subA_array[j]);
      mat_data    = fsls_CSRMatrixData(A_array[i]);
      memcpy(&mat_data[begin], my_mat_data, local_num_nonzeros*sizeof(JX_Real));
   }

   //------------------------------------------------------
   //  Fill the sub-mat data from other processors
   //------------------------------------------------------ 

   for (i = 0, k = 0; i < nprocs; i ++)
   {
      if (i != myid)
      {
         begin  = displs[i];
         length = displs[i+1] - begin;
         
         for (j = 0; j < local_num_ls; j ++)
         {
            mat_data = fsls_CSRMatrixData(A_array[j]);
            memcpy(&mat_data[begin], &mat_recvbuf[k], length*sizeof(JX_Real));
            k += length;
         }
      }
   } 

   //-----------------------------------------------------------
   //  Reorder all the sub matrices in the current processor
   //  so that the diagonal entries are firstly stored.
   //----------------------------------------------------------- 
    
   for (j = 0; j < local_num_ls; j ++)
   {
      fsls_CSRMatrixReorder(A_array[j]);
   }

   //==========================================================//
   //             Step 5     Free some staff                   //
   //==========================================================//  
   
   fsls_TFree(Requests);
   fsls_TFree(Status);
   fsls_TFree(manage);
   fsls_TFree(mv_commbuf);

   return (ls_data);
}
