/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 *  par_glsdata.c
 *  
 *  Created by peghoty  2010/11/22
 *  Xiangtan University
 *  peghoty@163.com
 *
 */

#include "jx_multils.h"

/*!
 * \fn fsls_ParGLSData *fsls_ParGLSDataCreate
 * \brief Create a fsls_ParGLSData object
 * \param comm communicator
 * \param global_num_ls number of all the sub linear systems
 * \param global_num_dof global number of DOFs 
 * \param *dof_partition the partition for dof distributing
 * \param *num_dof_procs how many DOFs in each processor? 
 * \author peghoty
 * \date 2010/11/22 
 */  
fsls_ParGLSData *
fsls_ParGLSDataCreate( MPI_Comm  comm, 
                       JX_Int       global_num_ls, 
                       JX_Int       global_num_dof, 
                       JX_Int      *dof_partition,
                       JX_Int      *num_dof_procs )
{
   JX_Int i,myid,nprocs;
   JX_Int local_num_dof;  
   fsls_ParGLSData *gls_data = fsls_CTAlloc(fsls_ParGLSData, 1); 

   fsls_MPICommInformation(comm, &myid, &nprocs);

   if ( !dof_partition && !num_dof_procs)
   {
      /* generate 'dof_partition' and 'num_dof_procs' */
      fsls_BalancedPartition(global_num_dof, nprocs, &dof_partition, &num_dof_procs);   
   }
   else if (!dof_partition && num_dof_procs)
   {
      /* generate 'dof_partition' using 'num_dof_procs' */
      dof_partition = fsls_CTAlloc(JX_Int, nprocs+1);
      dof_partition[0] = 0;
      for (i = 0; i < nprocs; i ++)
      {
         dof_partition[i+1] = dof_partition[i] + num_dof_procs[i];
      } 
   }
   else if (dof_partition && !num_dof_procs)
   {
      /* generate 'num_dof_procs' using 'dof_partition' */
      num_dof_procs = fsls_CTAlloc(JX_Int, nprocs);
      for (i = 0; i < nprocs; i ++)
      {
         num_dof_procs[i] = dof_partition[i+1] - dof_partition[i];
      }       
   }

   local_num_dof = dof_partition[myid+1] - dof_partition[myid];
   
   fsls_ParGLSDataComm(gls_data)         = comm;
   fsls_ParGLSDataGlobalNumLS(gls_data)  = global_num_ls;
   fsls_ParGLSDataGlobalNumDof(gls_data) = global_num_dof;
   fsls_ParGLSDataLocalNumDof(gls_data)  = local_num_dof;
   fsls_ParGLSDataDofPartition(gls_data) = dof_partition;
   fsls_ParGLSDataNumDofProcs(gls_data)  = num_dof_procs; 
   
   return (gls_data);  
}

/*!
 * \fn JX_Int fsls_ParGLSDataInitialize
 * \brief Initialize a fsls_ParGLSData object
 * \param *gls_data the fsls_ParLSData object to be initialized
 * \author peghoty
 * \date 2010/11/22 
 */ 
JX_Int
fsls_ParGLSDataInitialize( fsls_ParGLSData *gls_data )
{
   JX_Int global_num_ls = fsls_ParGLSDataGlobalNumLS(gls_data);
   JX_Int ierr = 0;
   fsls_CSRMatrix  **subA_array = NULL;     
   fsls_Vector     **subb_array = NULL;
   fsls_Vector     **subx_array = NULL;
   
   subA_array = fsls_CTAlloc(fsls_CSRMatrix *, global_num_ls);
   subb_array = fsls_CTAlloc(fsls_Vector *, global_num_ls);
   subx_array = fsls_CTAlloc(fsls_Vector *, global_num_ls);

   fsls_ParGLSDataSubAArray(gls_data) = subA_array;
   fsls_ParGLSDataSubBArray(gls_data) = subb_array;
   fsls_ParGLSDataSubXArray(gls_data) = subx_array;
   
   return ierr;
}

/*!
 * \fn fsls_ParGLSData *fsls_GenerateGLSData0
 * \brief Generate a fsls_ParGLSData object by reading data from files
 * \param comm communicator
 * \param global_num_ls number of all the sub linear systems
 * \param *MatFile pointer to the matrix file name
 * \param *RhsFile pointer to the right hand side vector file name
 * \note The local_num_ls matrices and vectors in each processor are the same.
 * \author peghoty
 * \date 2010/11/22 
 */ 
fsls_ParGLSData *
fsls_GenerateGLSData0( MPI_Comm comm, JX_Int global_num_ls, char *MatFile, char *RhsFile )
{
   fsls_ParGLSData  *gls_data   = NULL;
   fsls_CSRMatrix   *A          = NULL;     
   fsls_Vector      *b          = NULL; 
   fsls_CSRMatrix  **subA_array = NULL;     
   fsls_Vector     **subb_array = NULL;
   fsls_Vector     **subx_array = NULL;
   
   JX_Int *dof_partition = NULL; 

   JX_Int global_num_dof; 
   JX_Int num_rows_sub;
   JX_Int num_cols_sub;
   JX_Int num_nonzeros_sub;    
   JX_Int i,myid,nprocs;
   
   fsls_MPICommInformation(comm, &myid, &nprocs);   
   
   fsls_BuildCSRMatFromFile(MatFile, &A);
   fsls_BuildVecFromFile(RhsFile, &b);   

   //====================================================================// 
   //                  Create a fsls_ParGLSData object                   //
   //====================================================================//  
   
   global_num_dof = fsls_CSRMatrixNumRows(A);
   gls_data = fsls_ParGLSDataCreate(comm, global_num_ls, global_num_dof, NULL, NULL);
   fsls_ParGLSDataInitialize(gls_data);
   dof_partition = fsls_ParGLSDataDofPartition(gls_data);
   subA_array = fsls_ParGLSDataSubAArray(gls_data);
   subb_array = fsls_ParGLSDataSubBArray(gls_data);
   subx_array = fsls_ParGLSDataSubXArray(gls_data);
   
   //====================================================================// 
   //                  Fill the sub-matrices                             //
   //====================================================================//  

   subA_array[0] = fsls_GetSampleMatrix(comm, A, dof_partition);
   
   num_rows_sub = fsls_CSRMatrixNumRows(subA_array[0]);
   num_cols_sub = fsls_CSRMatrixNumCols(subA_array[0]);
   num_nonzeros_sub = fsls_CSRMatrixNumNonzeros(subA_array[0]);       

   for (i = 1; i < global_num_ls; i ++)
   {
      subA_array[i] = fsls_CSRMatrixCreate(num_rows_sub, num_cols_sub, num_nonzeros_sub);
      fsls_CSRMatrixInitialize(subA_array[i]);
      fsls_CSRMatrixCopy(subA_array[0], subA_array[i], 1);
   } 
   
   //====================================================================// 
   //                  Fill the sub-vectors                              //
   //====================================================================//  

   subb_array[0] = fsls_GetSampleVector(comm, b, dof_partition);
   
   num_rows_sub = fsls_VectorSize(subb_array[0]);
   
   for (i = 1; i < global_num_ls; i ++)
   {
      subb_array[i] = fsls_SeqVectorCreate(num_rows_sub);
      fsls_SeqVectorInitialize(subb_array[i]);
      fsls_SeqVectorCopy(subb_array[0], subb_array[i]);
      
      subx_array[i] = fsls_SeqVectorCreate(num_rows_sub);
      fsls_SeqVectorInitialize(subx_array[i]);
      fsls_SeqVectorSetConstantValues(subx_array[i], 0.0);      
   }    

   fsls_CSRMatrixDestroy(A);
   fsls_SeqVectorDestroy(b);
   return (gls_data);
}

/*!
 * \fn fsls_ParGLSData *fsls_GenerateGLSData1
 * \brief Generate a fsls_ParGLSData object by reading data from files
 * \param comm communicator
 * \param global_num_ls number of all the sub linear systems
 * \param global_num_dof global number of dofs(grids)
 * \param **MatFile_array MatFile_array[i] is the pointer to the i-th matrix file name
 * \param **RhsFile_array RhsFile_array[i] is the pointer to the i-th right hand side vector file name
 * \note The local_num_ls matrices and vectors in each processor can be 
 *       ether the same or different.
 * \author peghoty
 * \date 2010/11/24 
 */ 
fsls_ParGLSData *
fsls_GenerateGLSData1( MPI_Comm   comm, 
                       JX_Int        global_num_ls,
                       JX_Int        global_num_dof, 
                       char     **MatFile_array, 
                       char     **RhsFile_array )
{
   fsls_ParGLSData  *gls_data   = NULL;
   fsls_CSRMatrix   *A          = NULL;     
   fsls_Vector      *b          = NULL; 
   fsls_CSRMatrix  **subA_array = NULL;     
   fsls_Vector     **subb_array = NULL;
   fsls_Vector     **subx_array = NULL;
   
   JX_Int *dof_partition = NULL; 
   JX_Int  i,myid,nprocs;
   
   fsls_MPICommInformation(comm, &myid, &nprocs);   

   //====================================================================// 
   //                  Create a fsls_ParGLSData object                   //
   //====================================================================//  

   gls_data = fsls_ParGLSDataCreate(comm, global_num_ls, global_num_dof, NULL, NULL);

   fsls_ParGLSDataInitialize(gls_data);
   
   dof_partition = fsls_ParGLSDataDofPartition(gls_data);
   subA_array = fsls_ParGLSDataSubAArray(gls_data);
   subb_array = fsls_ParGLSDataSubBArray(gls_data);
   subx_array = fsls_ParGLSDataSubXArray(gls_data);

   //====================================================================// 
   //                  Fill the sub-matrices                             //
   //====================================================================//
   
   for (i = 0; i < global_num_ls; i ++)
   {
      fsls_BuildCSRMatFromFile(MatFile_array[i], &A);
      subA_array[i] = fsls_GetSampleMatrix(comm, A, dof_partition);
      fsls_CSRMatrixDestroy(A);
   } 
   
   //====================================================================// 
   //                  Fill the sub-vectors                              //
   //====================================================================//  

   for (i = 0; i < global_num_ls; i ++)
   {
      fsls_BuildVecFromFile(RhsFile_array[i], &b);
      subb_array[i] = fsls_GetSampleVector(comm, b, dof_partition);
      fsls_SeqVectorDestroy(b);
      
      subx_array[i] = fsls_SeqVectorCreate(fsls_VectorSize(subb_array[i]));
      fsls_SeqVectorInitialize(subx_array[i]);
      fsls_SeqVectorSetConstantValues(subx_array[i], 0.0);   
   }   

   return (gls_data);
}

/*!
 * \fn void fsls_ParGLSDataDestroy
 * \brief Destroy a fsls_ParGLSData object
 * \param *gls_data the fsls_ParLSData object to be destroyed
 * \author peghoty
 * \date 2010/11/23 
 */ 
void
fsls_ParGLSDataDestroy( fsls_ParGLSData *gls_data )
{
   JX_Int i, global_num_ls;
   
   if (gls_data)
   {
      global_num_ls = fsls_ParGLSDataGlobalNumLS(gls_data);
      
      if ( fsls_ParGLSDataDofPartition(gls_data) )
      {
         fsls_TFree(fsls_ParGLSDataDofPartition(gls_data));
      }
      
      if ( fsls_ParGLSDataNumDofProcs(gls_data) )
      {
         fsls_TFree(fsls_ParGLSDataNumDofProcs(gls_data));
      } 
      
      if ( fsls_ParGLSDataSubAArray(gls_data) )
      {
         for (i = 0; i < global_num_ls; i ++)
         {
            fsls_CSRMatrixDestroy(fsls_ParGLSDataSubAArray(gls_data)[i]);
         }
         fsls_TFree(fsls_ParGLSDataSubAArray(gls_data));
      }
      
      if ( fsls_ParGLSDataSubBArray(gls_data) )
      {
         for (i = 0; i < global_num_ls; i ++)
         {
            fsls_SeqVectorDestroy(fsls_ParGLSDataSubBArray(gls_data)[i]);
         }
         fsls_TFree(fsls_ParGLSDataSubBArray(gls_data));
      }
      
      if ( fsls_ParGLSDataSubXArray(gls_data) )
      {
         for (i = 0; i < global_num_ls; i ++)
         {
            fsls_SeqVectorDestroy(fsls_ParGLSDataSubXArray(gls_data)[i]);
         }
         fsls_TFree(fsls_ParGLSDataSubXArray(gls_data));
      }  
             
      fsls_TFree(gls_data);
   }
}

/*!
 * \fn fsls_CSRMatrix *fsls_GetSampleMatrix
 * \brief Abstract a sample matrix according to the partition provided
 * \param comm communicator
 * \param *A pointer to the matrix
 * \param *partition pointer to the row-partition of the matrix
 * \author peghoty
 * \date 2010/11/23 
 */ 
fsls_CSRMatrix * 
fsls_GetSampleMatrix( MPI_Comm         comm, 
                      fsls_CSRMatrix  *A, 
                      JX_Int             *partition )
{
   fsls_CSRMatrix  *sample = NULL;
   JX_Int num_cols = fsls_CSRMatrixNumCols(A);
   JX_Int    *ia   = fsls_CSRMatrixI(A);
   JX_Int    *ja   = fsls_CSRMatrixJ(A);
   JX_Real *a    = fsls_CSRMatrixData(A);

   JX_Int    *ia_sample = NULL;
   JX_Int    *ja_sample = NULL;
   JX_Real *a_sample  = NULL;
   
   JX_Int begin,end;
   JX_Int local_num_rows, num_nonzeros, local_num_rows1;
   JX_Int j, myid;
   
   jx_MPI_Comm_rank(comm, &myid); 

   begin = partition[myid];
   end   = partition[myid+1];
   local_num_rows = end - begin;
   num_nonzeros = ia[end] - ia[begin];
   sample = fsls_CSRMatrixCreate(local_num_rows, num_cols, num_nonzeros);
   fsls_CSRMatrixInitialize(sample);
   
   ia_sample = fsls_CSRMatrixI(sample);
   ja_sample = fsls_CSRMatrixJ(sample);
   a_sample  = fsls_CSRMatrixData(sample);
   memcpy(ia_sample, &ia[begin],     (local_num_rows+1)*sizeof(JX_Int));
   memcpy(ja_sample, &ja[ia[begin]], num_nonzeros*sizeof(JX_Int));
   memcpy(a_sample,  &a[ia[begin]],  num_nonzeros*sizeof(JX_Real));
   local_num_rows1 = local_num_rows + 1;
   
   if (myid > 0) // Generraly speaking, this is not necessary.
   {
      for (j = 0; j < local_num_rows1; j ++)
      {
         ia_sample[j] -= ia[begin];
      }
   }
   
   return sample;
}

/*!
 * \fn fsls_Vector *fsls_GetSampleVector
 * \brief Abstract a sample vector according to the partition provided
 * \param comm communicator
 * \param *x pointer to the vector
 * \param *partition pointer to the partition of the vector
 * \author peghoty
 * \date 2010/11/23 
 */ 
fsls_Vector *
fsls_GetSampleVector( MPI_Comm      comm, 
                      fsls_Vector  *x, 
                      JX_Int          *partition )
{
   fsls_Vector *sample = NULL;
   JX_Real *x_data = fsls_VectorData(x); 
   JX_Int     begin, local_size;
   JX_Int     myid;
   
   jx_MPI_Comm_rank(comm, &myid);
   
   begin = partition[myid];
   local_size = partition[myid+1] - begin;
   sample = fsls_SeqVectorCreate(local_size);
   fsls_SeqVectorInitialize(sample);
   memcpy(fsls_VectorData(sample), &x_data[begin], local_size*sizeof(JX_Real) );
   
   return sample;
}

/*!
 * \fn fsls_ParGLSData *fsls_ConstructBLSData
 * \brief Construct a fsls_ParGLSData using the sample mat and rhs each processor provides.
 * \param comm communicator
 * \param global_num_ls number of all the sub linear systems
 * \param global_num_dof global number of dofs(grids)
 * \param *MatFile MatFile[i] will provide the i-th processor a sample matrix
 * \param *RhsFile RhsFile[i] will provide the i-th processor a sample vector
 * \author peghoty
 * \date 2010/11/24 
 */ 
fsls_ParGLSData *
fsls_ConstructGLSData( MPI_Comm  comm, 
                       JX_Int       global_num_ls, 
                       JX_Int       global_num_dof, 
                       char    **MatFile, 
                       char    **RhsFile )
{
   fsls_ParGLSData  *gls_data = NULL;
   
   fsls_CSRMatrix  **subA_array = NULL;     
   fsls_Vector     **subb_array = NULL; 

   JX_Int i, j;
   JX_Int num_rows_sub;
   JX_Int num_cols_sub;
   JX_Int num_nonzeros_sub;
   JX_Int myid, nprocs;     
   
   fsls_MPICommInformation(comm, &myid, &nprocs);  

   gls_data = fsls_ParGLSDataCreate(comm, global_num_ls, global_num_dof, NULL, NULL);
   fsls_ParGLSDataInitialize(gls_data);
   subA_array = fsls_ParGLSDataSubAArray(gls_data);
   subb_array = fsls_ParGLSDataSubBArray(gls_data);
   
   for (i = 0; i < nprocs; i ++)
   {
      if (i == myid)
      {
         fsls_BuildCSRMatFromFile(MatFile[myid], &subA_array[0]);
         fsls_BuildVecFromFile(RhsFile[myid], &subb_array[0]);
         
         num_rows_sub = fsls_CSRMatrixNumRows(subA_array[0]);
         num_cols_sub = fsls_CSRMatrixNumCols(subA_array[0]);
         num_nonzeros_sub = fsls_CSRMatrixNumNonzeros(subA_array[0]);   
         
         for (j = 1; j < global_num_ls; j ++)
         {
            subA_array[i] = fsls_CSRMatrixCreate(num_rows_sub, num_cols_sub, num_nonzeros_sub);
            fsls_CSRMatrixInitialize(subA_array[i]);
            fsls_CSRMatrixCopy(subA_array[0], subA_array[i], 1);
            
            subb_array[i] = fsls_SeqVectorCreate(num_rows_sub);
            fsls_SeqVectorInitialize(subb_array[i]);
            fsls_SeqVectorCopy(subb_array[0], subb_array[i]);            
         }
      }
   }
   
   return (gls_data);
}
