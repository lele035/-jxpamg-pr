//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_vector.c -- basic operations for parallel vectors
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn jx_ParVector *jx_BuildRhsParFromOneFile
 * \brief Build a parallel vector from a given file.
 * \param *filename pointer to the file name.
 * \param *A pointer to the matrix which will provide the row partitioning.
 * \note The storage format of the vector is expected to be as follows:
 *         the 1st row:                   n                       // JX_Int-type
 *         the 2 to (n+1)-th row:         f[i]    i=0(1)n-1       // JX_Real-type
 *      where n is the total size of the vector, f[i] is the i-th entry of the vector.
 * \author peghoty
 * \date 2009/07/10
 */
jx_ParVector *
jx_BuildRhsParFromOneFile( char *filename, jx_ParCSRMatrix *A )
{
   jx_ParVector *b = NULL;
   jx_Vector    *b_CSR = NULL;
   JX_Int           my_id;
   JX_Int          *partitioning;

   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (my_id == 0) 
   {
      b_CSR = jx_SeqVectorRead(filename);
   }

   jx_ParCSRMatrixGetRowPartitioning(A, &partitioning);

   b = jx_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning);

   if (my_id == 0) jx_SeqVectorDestroy(b_CSR);

   return (b);
}

jx_ParVector *
jx_BuildRhsParFromOneFile3( char *filename, jx_ParCSRMatrix *A )
{
   jx_ParVector *b = NULL;
   jx_Vector    *b_CSR = NULL;
   JX_Int           my_id;
   JX_Int          *partitioning;

   jx_MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   if (my_id == 0) 
   {
      b_CSR = jx_SeqVectorRead2(filename);
   }

   jx_ParCSRMatrixGetRowPartitioning(A, &partitioning);

   b = jx_VectorToParVector(MPI_COMM_WORLD, b_CSR, partitioning);

   if (my_id == 0) jx_SeqVectorDestroy(b_CSR);

   return (b);
}

/*!
 * \fn jx_ParVector *jx_ParMultiVectorCreate
 * \brief Create a parallel Multivector.
 * \param comm communicator.
 * \param global_size global size of a single vector.
 * \param *partitioning partitioning of the parallel vector.
 * \param num_vector number of vectors.
 * \date 2011/09/03
 */
jx_ParVector *
jx_ParMultiVectorCreate( MPI_Comm  comm,
                         JX_Int       global_size, 
                         JX_Int      *partitioning,
                         JX_Int       num_vectors )
{
   /* note that global_size is the global length of a single vector */
   jx_ParVector * vector = jx_ParVectorCreate( comm, global_size, partitioning );
   jx_ParVectorNumVectors(vector) = num_vectors;

   return vector;
}

/*!
 * \fn jx_ParVector *jx_ParVectorCreate
 * \brief Create a parallel vector.
 * \param comm communicator.
 * \param global_size global size of the vector.
 * \param *partitioning partitioning of the parallel vector.
 * \note If create is called for JX_NO_GLOBAL_PARTITION and partitioning is NOT null,
 *       then it is assumed that it is array of length 2 containing the start row of 
 *       the calling processor followed by the start row of the next processor - AHB 6/05
 * \date 2011/09/03
 */
jx_ParVector *
jx_ParVectorCreate( MPI_Comm  comm,
                    JX_Int       global_size, 
                    JX_Int      *partitioning )
{
   jx_ParVector  *vector;
   JX_Int num_procs, my_id;

   if (global_size < 0)
   {
      jx_error_in_arg(2);
      return NULL;
   }
   vector = jx_CTAlloc(jx_ParVector, 1);
   jx_MPI_Comm_rank(comm, &my_id);

   if (!partitioning)
   {
     jx_MPI_Comm_size(comm,&num_procs);
#ifdef JX_NO_GLOBAL_PARTITION
     jx_GenerateLocalPartitioning(global_size, num_procs, my_id, &partitioning);
#else
     jx_GeneratePartitioning(global_size, num_procs, &partitioning);
#endif
   }

   jx_ParVectorAssumedPartition(vector) = NULL;
   
   jx_ParVectorComm(vector) = comm;
   jx_ParVectorGlobalSize(vector) = global_size;
  
#ifdef JX_NO_GLOBAL_PARTITION
   jx_ParVectorFirstIndex(vector)   = partitioning[0];
   jx_ParVectorLastIndex(vector)    = partitioning[1]-1;
   jx_ParVectorPartitioning(vector) = partitioning;
   jx_ParVectorLocalVector(vector)  = jx_SeqVectorCreate(partitioning[1] - partitioning[0]);
#else
   jx_ParVectorFirstIndex(vector)   = partitioning[my_id];
   jx_ParVectorLastIndex(vector)    = partitioning[my_id+1] -1;
   jx_ParVectorPartitioning(vector) = partitioning;
   jx_ParVectorLocalVector(vector)  = jx_SeqVectorCreate(partitioning[my_id+1] - partitioning[my_id]);
#endif

   /* set defaults */
   jx_ParVectorOwnsData(vector) = 1;
   jx_ParVectorOwnsPartitioning(vector) = 1;

   return vector;
}

/*!
 * \fn JX_Int jx_ParVectorInitialize
 * \brief Initialize a parallel vector.
 * \date 2011/09/03
 */ 
JX_Int 
jx_ParVectorInitialize( jx_ParVector *vector )
{
   if (!vector)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_SeqVectorInitialize(jx_ParVectorLocalVector(vector));

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParVectorDestroy
 * \brief Destroy a parallel vector.
 * \date 2011/09/03
 */  
JX_Int 
jx_ParVectorDestroy( jx_ParVector *vector )
{
   if (vector)
   {
      if ( jx_ParVectorOwnsData(vector) )
      {
         jx_SeqVectorDestroy(jx_ParVectorLocalVector(vector));
      }
      
      if ( jx_ParVectorOwnsPartitioning(vector) )
      {
         jx_TFree(jx_ParVectorPartitioning(vector));
      }

      if (jx_ParVectorAssumedPartition(vector))
      {
         jx_ParVectorDestroyAssumedPartition(vector);
      }

      jx_TFree(vector);
   }

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParVectorDestroyAssumedPartition
 * \brief Destroy the assumedpartition of a parallel vector.
 * \date 2011/09/03
 */   
JX_Int 
jx_ParVectorDestroyAssumedPartition(jx_ParVector *vector)
{
   jx_IJAssumedPart *apart;
   
   apart = jx_ParVectorAssumedPartition(vector);
   
   if (apart->storage_length > 0) 
   {      
      jx_TFree(apart->proc_list);
      jx_TFree(apart->row_start_list);
      jx_TFree(apart->row_end_list);
      jx_TFree(apart->sort_index);
   }

   jx_TFree(apart);
   
   return jx_error_flag;
}

/*!
 * \fn jx_Vector *jx_ParVectorToVectorAll
 * \brief Generates a Vector on every proc which has a piece of the data 
 *        from a ParVector on several procs in comm, vec_starts needs to  
 *        contain the partitioning across all procs in comm
 * \date 2011/09/03
 */  
jx_Vector *
jx_ParVectorToVectorAll (jx_ParVector *par_v)
{
   MPI_Comm             comm = jx_ParVectorComm(par_v);
   JX_Int                  global_size = jx_ParVectorGlobalSize(par_v);
#ifndef JX_NO_GLOBAL_PARTITION
   JX_Int                 *vec_starts = jx_ParVectorPartitioning(par_v);
#endif
   jx_Vector           *local_vector = jx_ParVectorLocalVector(par_v);
   JX_Int  		num_procs, my_id;
   JX_Int                  num_vectors = jx_ParVectorNumVectors(par_v);
   jx_Vector  	       *vector;
   JX_Real              *vector_data;
   JX_Real              *local_data;
   JX_Int 			local_size;
   MPI_Request         *requests;
   MPI_Status          *status;
   JX_Int			i, j;
   JX_Int                 *used_procs;
   JX_Int			num_types, num_requests;
   JX_Int			vec_len, proc_id;

#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int *new_vec_starts;
   JX_Int  num_contacts;
   JX_Int  contact_proc_list[1];
   JX_Int  contact_send_buf[1];
   JX_Int  contact_send_buf_starts[2];
   JX_Int  max_response_size;
   JX_Int *response_recv_buf=NULL;
   JX_Int *response_recv_buf_starts = NULL;
   jx_DataExchangeResponse response_obj;
   jx_ProcListElements     send_proc_obj;
   
   JX_Int *send_info = NULL;
   MPI_Status  status1;
   JX_Int count, tag1 = 112, tag2 = 223;
   JX_Int start;
#endif


   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);


#ifdef JX_NO_GLOBAL_PARTITION

  local_size = jx_ParVectorLastIndex(par_v) - jx_ParVectorFirstIndex(par_v) + 1;

  /* determine procs which hold data of par_v and store ids in used_procs */
  /* we need to do an exchange data for this.  If I own row then I will contact
     processor 0 with the endpoint of my local range */

   if (local_size > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0]  =  jx_ParVectorLastIndex(par_v);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;
   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }

   /*build the response object*/
   /*send_proc_obj will  be for saving info from contacts */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = jx_CTAlloc(JX_Int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = jx_CTAlloc(JX_Int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements = jx_CTAlloc(JX_Int, send_proc_obj.element_storage_length);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = jx_FillResponseParToVectorAll;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/
  
   jx_DataExchangeList( num_contacts, 
                        contact_proc_list, contact_send_buf, 
                        contact_send_buf_starts, sizeof(JX_Int), 
                        sizeof(JX_Int), &response_obj, 
                        max_response_size, 1,
                        comm, (void**) &response_recv_buf,	   
                        &response_recv_buf_starts );

   /* now processor 0 should have a list of ranges for processors that have rows -
      these are in send_proc_obj - it needs to create the new list of processors
      and also an array of vec starts - and send to those who own row */
   if (my_id)
   {
      if (local_size)      
      {
         /* look for a message from processor 0 */         
         jx_MPI_Probe(0, tag1, comm, &status1);
         jx_MPI_Get_count(&status1, JX_MPI_INT, &count);
         
         send_info = jx_CTAlloc(JX_Int, count);
         jx_MPI_Recv(send_info, count, JX_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */  
         num_types  = send_info[0];
         used_procs =  jx_CTAlloc(JX_Int, num_types);  
         new_vec_starts = jx_CTAlloc(JX_Int, num_types + 1);

         for (i = 1; i <= num_types; i ++)
         {
            used_procs[i-1] = send_info[i];
         }
         for (i = num_types + 1; i < count; i ++)
         {
            new_vec_starts[i-num_types-1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         jx_TFree(send_proc_obj.vec_starts);
         jx_TFree(send_proc_obj.id);
         jx_TFree(send_proc_obj.elements);
         if(response_recv_buf)        jx_TFree(response_recv_buf);
         if(response_recv_buf_starts) jx_TFree(response_recv_buf_starts);
         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types  = send_proc_obj.length;
      used_procs =  jx_CTAlloc(JX_Int, num_types);  
      new_vec_starts = jx_CTAlloc(JX_Int, num_types + 1);
      
      new_vec_starts[0] = 0;
      for (i = 0; i < num_types; i ++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i+1] = send_proc_obj.elements[i] + 1;
      }
      jx_qsort0(used_procs, 0, num_types - 1);
      jx_qsort0(new_vec_starts, 0, num_types);
      
      /*now we need to put into an array to send */
      count =  2*num_types+2;
      send_info = jx_CTAlloc(JX_Int, count);
      send_info[0] = num_types;
      for (i = 1; i <= num_types; i ++)
      {
         send_info[i] = used_procs[i-1];
      }
      for (i = num_types+1; i < count; i ++)
      {
         send_info[i] = new_vec_starts[i-num_types-1];
      }
      requests = jx_CTAlloc(MPI_Request, num_types);
      status   = jx_CTAlloc(MPI_Status, num_types);

      /* don't send to myself - these are sorted so my id would be first */
      start = 0;
      if (used_procs[0] == 0)
      {
         start = 1;
      }
   
      for (i = start; i < num_types; i ++)
      {
         jx_MPI_Isend(send_info, count, JX_MPI_INT, used_procs[i], tag1, comm, &requests[i-start]);
      }
      jx_MPI_Waitall(num_types-start, requests, status);

      jx_TFree(status);
      jx_TFree(requests);
   }

   /* clean up */
   jx_TFree(send_proc_obj.vec_starts);
   jx_TFree(send_proc_obj.id);
   jx_TFree(send_proc_obj.elements);
   jx_TFree(send_info);
   if(response_recv_buf)        jx_TFree(response_recv_buf);
   if(response_recv_buf_starts) jx_TFree(response_recv_buf_starts);

   /* now proc 0 can exit if it has no rows */
   if (!local_size) 
   {
      jx_TFree(used_procs);
      jx_TFree(new_vec_starts);
      return NULL;
   }
   
   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

   /* this vector should be rather small */

   local_data = jx_VectorData(local_vector);
   vector = jx_SeqVectorCreate(global_size);
   jx_VectorNumVectors(vector) = num_vectors;
   jx_SeqVectorInitialize(vector);
   vector_data = jx_VectorData(vector);

   num_requests = 2*num_types;

   requests = jx_CTAlloc(MPI_Request, num_requests);
   status = jx_CTAlloc(MPI_Status, num_requests);

   /* initialize data exchange among used_procs and generate vector   
      - here we send to ourself also */
 
   j = 0;
   for (i = 0; i < num_types; i ++)
   {
        proc_id = used_procs[i];
        vec_len = new_vec_starts[i+1] - new_vec_starts[i];
        jx_MPI_Irecv(&vector_data[new_vec_starts[i]], num_vectors*vec_len, JX_MPI_REAL,
                  proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i ++)
   {
        jx_MPI_Isend(local_data, num_vectors*local_size, JX_MPI_REAL, used_procs[i],
                  tag2, comm, &requests[j++]);
   }
 
   jx_MPI_Waitall(num_requests, requests, status);


   if (num_requests)
   {
       jx_TFree(requests);
       jx_TFree(status); 
       jx_TFree(used_procs);
   }

   jx_TFree(new_vec_starts);
   
#else

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   /* if my_id contains no data, return NULL  */

   if (!local_size)
   {
      return NULL;
   }
 
   local_data = jx_VectorData(local_vector);
   vector = jx_SeqVectorCreate(global_size);
   jx_VectorNumVectors(vector) = num_vectors;
   jx_SeqVectorInitialize(vector);
   vector_data = jx_VectorData(vector);

   /* determine procs which hold data of par_v and store ids in used_procs */

   num_types = -1;
   for (i = 0; i < num_procs; i ++)
   {
        if (vec_starts[i+1] - vec_starts[i])
        {
                num_types++;
        }
   }
   num_requests = 2*num_types;
 
   used_procs = jx_CTAlloc(JX_Int, num_types);
   j = 0;
   for (i = 0; i < num_procs; i ++)
   {
        if (vec_starts[i+1] - vec_starts[i] && i - my_id)
        {
                used_procs[j++] = i;
        }
   }
 
   requests = jx_CTAlloc(MPI_Request, num_requests);
   status   = jx_CTAlloc(MPI_Status, num_requests);

   /* initialize data exchange among used_procs and generate vector */
 
   j = 0;
   for (i = 0; i < num_types; i ++)
   {
        proc_id = used_procs[i];
        vec_len = vec_starts[proc_id+1] - vec_starts[proc_id];
        jx_MPI_Irecv(&vector_data[vec_starts[proc_id]], num_vectors*vec_len, JX_MPI_REAL,
                  proc_id, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i ++)
   {
        jx_MPI_Isend(local_data, num_vectors*local_size, JX_MPI_REAL, used_procs[i],
                  0, comm, &requests[j++]);
   }
 
   for (i = 0; i < num_vectors*local_size; i ++)
   {
        vector_data[vec_starts[my_id]+i] = local_data[i];
   }
 
   jx_MPI_Waitall(num_requests, requests, status);

   if (num_requests)
   {
   	jx_TFree(used_procs);
   	jx_TFree(requests);
   	jx_TFree(status); 
   }

#endif

   return vector;
}


/*!
 * \fn JX_Int jx_ParVectorToVector_Alloctaed
 * \brief Generates a Vector on every proc which has a piece of the data
 *        from a ParVector on several procs in comm, vec_starts needs
 *        to contain the partitioning across all procs in comm.
 * \note The memory for the data of 'vector' has been already allocated. 
 * \author peghoty
 * \date 2011/08/31
 */
JX_Int
jx_ParVectorToVector_Alloctaed( jx_ParVector *par_v, 
                                jx_Vector    *vector )
{
   MPI_Comm       comm = jx_ParVectorComm(par_v);
   JX_Int           *vec_starts = jx_ParVectorPartitioning(par_v);
   jx_Vector     *local_vector = jx_ParVectorLocalVector(par_v);
   JX_Int            num_procs, my_id;
   JX_Int            num_vectors = jx_ParVectorNumVectors(par_v);
   JX_Real        *vector_data;
   JX_Real        *local_data;
   JX_Int            local_size;
   MPI_Request   *requests;
   MPI_Status    *status;
   JX_Int           *used_procs;
   JX_Int            num_types, num_requests;
   JX_Int            i, j, vec_len, proc_id;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   /* if my_id contains no data, return NULL */
   
   if (!local_size)
   {
      return -1;
   }
 
   local_data  = jx_VectorData(local_vector);
   vector_data = jx_VectorData(vector);

  /*------------------------------------------------------------------------ 
   * determine procs which hold data of par_v and store ids in used_procs 
   *-----------------------------------------------------------------------*/

   num_types = -1;
   for (i = 0; i < num_procs; i ++)
   {
      if (vec_starts[i+1] - vec_starts[i])

      {
         num_types ++;
      }
   }
   num_requests = 2*num_types;
 
   used_procs = jx_CTAlloc(JX_Int, num_types);
   j = 0;
   for (i = 0; i < num_procs; i ++)
   {
      if (vec_starts[i+1] - vec_starts[i] && i - my_id)
      {
         used_procs[j++] = i;
      }
   }
 
   requests = jx_CTAlloc(MPI_Request, num_requests);
   status   = jx_CTAlloc(MPI_Status, num_requests);


  /*------------------------------------------------------------------------ 
   * initialize data exchange among used_procs and generate vector 
   *-----------------------------------------------------------------------*/
   
   j = 0;
   for (i = 0; i < num_types; i ++)
   {
      proc_id = used_procs[i];
      vec_len = vec_starts[proc_id+1] - vec_starts[proc_id];
      jx_MPI_Irecv( &vector_data[vec_starts[proc_id]], num_vectors*vec_len, 
                       JX_MPI_REAL, proc_id, 0, comm, &requests[j++] );
   }
   for (i = 0; i < num_types; i ++)
   {
      jx_MPI_Isend( local_data, num_vectors*local_size, JX_MPI_REAL, 
                       used_procs[i], 0, comm, &requests[j++] );
   }
 
   for (i = 0; i < num_vectors*local_size; i ++)
   {
      vector_data[vec_starts[my_id]+i] = local_data[i];
   }     
 
   jx_MPI_Waitall(num_requests, requests, status);

   if (num_requests)
   {
   	jx_TFree(used_procs);
   	jx_TFree(requests);
   	jx_TFree(status); 
   }

   return 0;
}

/*!
 * \fn JX_Int jx_GenerateLocalPartitioning
 * \brief It only returns  the portion of the partition
 *        belonging to the individual process 
 *        - to do this it requires the processor id as well. AHB 6/05
 * \date 2011/09/03
 */
JX_Int
jx_GenerateLocalPartitioning(JX_Int length, JX_Int num_procs, JX_Int myid, JX_Int **part_ptr)
{
   JX_Int  ierr = 0;
   JX_Int *part;
   JX_Int  size, rest;

   part = jx_CTAlloc(JX_Int, 2);
   size = length /num_procs;
   rest = length - size*num_procs;

   /* first row I own */
   part[0] = size*myid;
   part[0] += jx_min(myid, rest);
   
   /* last row I own */
   part[1]  =  size*(myid+1);
   part[1] +=  jx_min(myid+1, rest);
   part[1]  =  part[1] - 1;

   /* add 1 to last row since this is for "starts" vector */
   part[1] = part[1] + 1;
   
   *part_ptr = part;

   return ierr;
}

/*!
 * \fn JX_Int jx_ParVectorPrint
 * \brief Print a parallel vector into given file(s).
 * \author peghoty 
 * \date 2011/09/03
 */ 
JX_Int
jx_ParVectorPrint( jx_ParVector *vector, const char *file_name )
{
   char 	new_file_name[80];
   jx_Vector   *local_vector;
   MPI_Comm 	comm;
   JX_Int  	my_id, num_procs, i;
   JX_Int		*partitioning;
   JX_Int		global_size;
   FILE		*fp;
   
   if (!vector)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   
   local_vector = jx_ParVectorLocalVector(vector); 
   comm         = jx_ParVectorComm(vector);
   partitioning = jx_ParVectorPartitioning(vector); 
   global_size  = jx_ParVectorGlobalSize(vector); 

   jx_MPI_Comm_rank(comm,&my_id); 
   jx_MPI_Comm_size(comm,&num_procs); 
   
   jx_sprintf(new_file_name, "%s.%d", file_name, my_id); 
   jx_SeqVectorPrint(local_vector, new_file_name);
   jx_sprintf(new_file_name, "%s.INFO.%d", file_name, my_id); 
   
   fp = fopen(new_file_name, "w");
   jx_fprintf(fp, "%d\n", global_size);
   
#ifdef JX_NO_GLOBAL_PARTITION
   for (i = 0; i < 2; i ++)
   {
      jx_fprintf(fp, "%d\n", partitioning[i]);
   }
#else
   for (i = 0; i < num_procs; i ++)
   {
      jx_fprintf(fp, "%d\n", partitioning[i]);
   }
#endif

   fclose (fp);
   return jx_error_flag;
}

/*!
 * \fn jx_ParVector *jx_ParVectorRead
 * \brief Generate a parallel vector by reading data from given file(s)
 * \date 2011/09/03
 */
jx_ParVector *
jx_ParVectorRead( MPI_Comm comm, const char *file_name )
{
   char 	 new_file_name[80];
   jx_ParVector *par_vector;
   JX_Int  	 my_id, num_procs;
   JX_Int		*partitioning;
   JX_Int		 global_size, i;
   FILE		*fp;

   jx_MPI_Comm_rank(comm, &my_id); 
   jx_MPI_Comm_size(comm, &num_procs); 

   partitioning = jx_CTAlloc(JX_Int, num_procs + 1);

   jx_sprintf(new_file_name, "%s.INFO.%d", file_name, my_id); 
   fp = fopen(new_file_name, "r");
   jx_fscanf(fp, "%d\n", &global_size);
   
#ifdef JX_NO_GLOBAL_PARTITION
   for (i = 0; i < 2; i ++)
   {
      jx_fscanf(fp, "%d\n", &partitioning[i]);
   }	
   fclose(fp);
#else
   for (i = 0; i < num_procs; i ++)
   {
      jx_fscanf(fp, "%d\n", &partitioning[i]);
   }
   fclose(fp);
   partitioning[num_procs] = global_size; 
#endif

   par_vector = jx_CTAlloc(jx_ParVector, 1);
	
   jx_ParVectorComm(par_vector) = comm;
   jx_ParVectorGlobalSize(par_vector) = global_size;

#ifdef JX_NO_GLOBAL_PARTITION
   jx_ParVectorFirstIndex(par_vector) = partitioning[0];
   jx_ParVectorLastIndex(par_vector)  = partitioning[1] - 1;
#else
   jx_ParVectorFirstIndex(par_vector) = partitioning[my_id];
   jx_ParVectorLastIndex(par_vector)  = partitioning[my_id+1] - 1;
#endif

   jx_ParVectorPartitioning(par_vector) = partitioning;

   jx_ParVectorOwnsData(par_vector) = 1;
   jx_ParVectorOwnsPartitioning(par_vector) = 1;

   jx_sprintf(new_file_name, "%s.%d", file_name, my_id); 
   jx_ParVectorLocalVector(par_vector) = jx_SeqVectorRead(new_file_name);

   /* multivector code not written yet >>> */
   jx_assert( jx_ParVectorNumVectors(par_vector) == 1 );

   return par_vector;
}

/*!
 * \fn JX_Int jx_GeneratePartitioning
 * \brief Generates load balanced partitioning of a 1-d array.
 * \note For multivectors, length should be the (global) length of a single vector.
 *       Thus each of the vectors of the multivector will get the same data distribution.
 * \date 2011/09/03
 */
JX_Int
jx_GeneratePartitioning(JX_Int length, JX_Int num_procs, JX_Int **part_ptr)
{
   JX_Int  ierr = 0;
   JX_Int *part;
   JX_Int  size, rest;
   JX_Int  i;

   part = jx_CTAlloc(JX_Int, num_procs+1);
   size = length / num_procs;
   rest = length - size*num_procs;
   part[0] = 0;
   for (i = 0; i < num_procs; i ++)
   {
	part[i+1] = part[i] + size;
	if (i < rest) part[i+1] ++;
   }

  *part_ptr = part;
   return ierr;
}

/*!
 * \fn JX_Int jx_FillResponseParToVectorAll
 * \brief Fill response function for determining the 
 *        send processors data exchange.
 * \date 2011/09/03
 */
JX_Int
jx_FillResponseParToVectorAll( void      *p_recv_contact_buf, 
                               JX_Int        contact_size, 
                               JX_Int        contact_proc, 
                               void      *ro, 
                               MPI_Comm   comm, 
                               void     **p_send_response_buf, 
                               JX_Int       *response_message_size )
{
   JX_Int  myid;
   JX_Int  i, index, count, elength;

   JX_Int  *recv_contact_buf = (JX_Int *) p_recv_contact_buf;

   jx_DataExchangeResponse *response_obj = ro;  

   jx_ProcListElements     *send_proc_obj = response_obj->data2;   

   jx_MPI_Comm_rank(comm, &myid);

   /* check to see if we need to allocate more space in send_proc_obj for ids */
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 10; /* add space for 10 more processors */
      send_proc_obj->id = jx_TReAlloc(send_proc_obj->id, JX_Int, send_proc_obj->storage_length);
      send_proc_obj->vec_starts = jx_TReAlloc(send_proc_obj->vec_starts, JX_Int, send_proc_obj->storage_length+1);
   }
  
   /* initialize */ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /* this is the number of elements */

   /* send proc */ 
   send_proc_obj->id[count] = contact_proc; 

   /* do we need more storage for the elements? */
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = jx_max(contact_size, 10);   
      elength += index;
      send_proc_obj->elements = jx_TReAlloc(send_proc_obj->elements, JX_Int, elength);
      send_proc_obj->element_storage_length = elength; 
   }
   /* populate send_proc_obj */
   for (i = 0; i < contact_size; i ++) 
   { 
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count+1] = index;
   send_proc_obj->length++;
   
   /* output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   return jx_error_flag;
}

/*!
 * \fn jx_ParVector *jx_VectorToParVector
 * \brief Generates a ParVector from a Vector on proc 0 and 
 *        distributes the pieces to the other procs in comm.
 * \note This is not being optimized to use JX_NO_GLOBAL_PARTITION.
 * \date 2011/09/03
 */
jx_ParVector *
jx_VectorToParVector( MPI_Comm comm, jx_Vector *v, JX_Int *vec_starts )
{
   JX_Int                  global_size;
   JX_Int                  local_size;
   JX_Int                  num_vectors;
   JX_Int                  num_procs, my_id;
   JX_Int                  global_vecstride, vecstride, idxstride;
   jx_ParVector        *par_vector;
   jx_Vector           *local_vector;
   JX_Real              *v_data = NULL;
   JX_Real              *local_data;
   MPI_Request         *requests;
   MPI_Status          *status, status0;
   JX_Int                  i, j, k, p;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   if (my_id == 0)
   {
      global_size = jx_VectorSize(v);
      v_data = jx_VectorData(v);
      num_vectors = jx_VectorNumVectors(v); /* for multivectors */
      global_vecstride = jx_VectorVectorStride(v);
   }

   jx_MPI_Bcast(&global_size, 1, JX_MPI_INT, 0, comm);
   jx_MPI_Bcast(&num_vectors, 1, JX_MPI_INT, 0, comm);
   jx_MPI_Bcast(&global_vecstride, 1, JX_MPI_INT, 0, comm);

   if (num_vectors == 1)
   {
      par_vector = jx_ParVectorCreate(comm, global_size, vec_starts);
   }
   else
   {
      par_vector = jx_ParMultiVectorCreate(comm, global_size, vec_starts, num_vectors);
   }

   vec_starts = jx_ParVectorPartitioning(par_vector);

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   jx_ParVectorInitialize(par_vector);
   local_vector = jx_ParVectorLocalVector(par_vector);
   local_data   = jx_VectorData(local_vector);
   vecstride    = jx_VectorVectorStride(local_vector);
   idxstride    = jx_VectorIndexStride(local_vector);
   jx_assert(idxstride == 1);  /* <<< so far only the only implemented multivector StorageMethod is 0 <<< */

   if (my_id == 0)
   {
	requests = jx_CTAlloc(MPI_Request,num_vectors*(num_procs-1));
	status   = jx_CTAlloc(MPI_Status,num_vectors*(num_procs-1));
	k = 0;
	for (p = 1; p < num_procs; p ++)
	{
           for (j = 0; j < num_vectors; ++ j)
           {
		jx_MPI_Isend( &v_data[vec_starts[p]]+j*global_vecstride,
                          (vec_starts[p+1]-vec_starts[p]),
                          JX_MPI_REAL, p, 0, comm, &requests[k++] );
           }
        }
           
        if (num_vectors == 1)
        {
           for (i = 0; i < local_size; i ++)
           {
              local_data[i] = v_data[i];
           }
        }
        else
        {
           for (j = 0; j < num_vectors; ++ j)
           {
              for (i = 0; i < local_size; i ++)
              {
                 local_data[i+j*vecstride] = v_data[i+j*global_vecstride];
              }
           }
        }
	jx_MPI_Waitall(num_procs-1, requests, status);
	jx_TFree(requests);
	jx_TFree(status);
   }
   else
   {
      for (j = 0; j < num_vectors; ++ j)
      {
         jx_MPI_Recv( local_data+j*vecstride, local_size, JX_MPI_REAL, 0, 0, comm, &status0 );
      }
   }

   return par_vector;
}
                                             
/*!
 * \fn jx_ParVector *jx_VectorToParVector_FromGivenPro
 * \brief Generates a ParVector from a Vector on the given proc (srcid#) and 
 *        distributes the pieces to the other procs in comm.
 *        利用指定进程(srcid 号进程)上的一个串行向量生成一个并行向量. 这里只考虑 num_vectors = 1 的情形!
 * \note This is not being optimized to use JX_NO_GLOBAL_PARTITION.
 * \author peghoty
 * \date 2012/02/29
 */
jx_ParVector *
jx_VectorToParVector_FromGivenPro( MPI_Comm comm, JX_Int srcid, jx_Vector *v, JX_Int *vec_starts )
{
   JX_Int                  global_size;
   JX_Int                  local_size;
   JX_Int                  num_procs, my_id;
   jx_ParVector        *par_vector;
   jx_Vector           *local_vector;
   JX_Real              *v_data = NULL;
   JX_Real              *local_data;
   MPI_Request         *requests;
   MPI_Status          *status, status0;
   JX_Int                  i, k, p;
   JX_Int                  start = 0;
   JX_Int                  length = 0;
   JX_Int                  partitioning_owner;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   if (my_id == srcid)
   {
      global_size = jx_VectorSize(v);
      v_data = jx_VectorData(v);
   }

   jx_MPI_Bcast(&global_size, 1, JX_MPI_INT, srcid, comm); // 从 srcid 号进程广播出去. peghoty, 2012/03/05

   /* set partitioning_owner to be zero is quite important here, 2012/03/05 */
   if (vec_starts) 
   {
      partitioning_owner = 0;
   }
   else
   {
      partitioning_owner = 1;
   }
  
   par_vector = jx_ParVectorCreate(comm, global_size, vec_starts);
   jx_ParVectorSetPartitioningOwner(par_vector, partitioning_owner);
   vec_starts = jx_ParVectorPartitioning(par_vector);
   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   jx_ParVectorInitialize(par_vector);
   local_vector = jx_ParVectorLocalVector(par_vector);
   local_data   = jx_VectorData(local_vector);

   if (my_id == srcid)
   {
      requests = jx_CTAlloc(MPI_Request, num_procs - 1);
      status = jx_CTAlloc(MPI_Status, num_procs - 1);
      k = 0;
      
      for (p = 0; p < num_procs; p ++)
      {
         if (p != srcid)
         {
            start = vec_starts[p];
            length = vec_starts[p+1] - start;
            jx_MPI_Isend( &v_data[start], length, JX_MPI_REAL, p, 0, comm, &requests[k++] );
         }
      }

      for (i = 0; i < local_size; i ++)
      {
         local_data[i] = v_data[i];
      }
      
      start = vec_starts[my_id];
      for (i = 0; i < local_size; i ++)
      {
         local_data[i] = v_data[start+i]; // be careful here, DON't forget the start. peghoty,  2012/03/07
      }      

      jx_MPI_Waitall(num_procs-1, requests, status);
      jx_TFree(requests);
      jx_TFree(status);
   }
   else
   {
      jx_MPI_Recv( local_data, local_size, JX_MPI_REAL, srcid, 0, comm, &status0 );
   }

   return par_vector;
}

/*!
 * \fn jx_ParVector *jx_VectorToParVector_sp
 * \brief Transfer a serial vector to a parallel one (only for single-processor case).
 * \note 该模块的特点是: 转换后的并行向量和串行向量共享数据部分。因此释放内存时要特别小心，
 *       不要轻易将串行向量 Destroy，一般采用 Free 即可.
 * \date 2011/09/27
 */
jx_ParVector *
jx_VectorToParVector_sp( MPI_Comm comm, jx_Vector *vector )
{
   jx_ParVector *par_vector   = NULL;
   JX_Int global_size = jx_VectorSize(vector);

#if 0  
   jx_Vector *local_vector = NULL; 
   par_vector = jx_ParVectorCreate(comm, global_size, NULL);
   local_vector = jx_ParVectorLocalVector(par_vector);
   jx_VectorData(local_vector) = jx_VectorData(vector);
#else  // modified by peghoty, 2012/02/25
   JX_Real temp_adrress = 0.0;  
   par_vector = jx_ParVectorCreate(comm, global_size, NULL);
   par_vector->local_vector->data = &temp_adrress; // 保证调用 jx_ParVectorInitialize 时不另外开设空间
   jx_ParVectorInitialize(par_vector); // 保证成员 multivec_storage_method, vecstride, idxstride 被赋上适当的默认值，
                                       // 因为在调用 jx_ParCSRMatrixMatvec 时要检查，如果赋值不对，会有警告信息.
   jx_ParVectorSetDataOwner(par_vector, 1); // 保证 Destroy 并行向量时，能释放内存. 
   par_vector->local_vector->data = vector->data; // 建立数据共享关系
#endif
 
   return (par_vector);
}  

/*!
 * \fn JX_Int jx_VectorToParVector_Allocated
 * \brief Generates a ParVector from a Vector on proc 0 and distributes the pieces
 *        to the other procs in comm.
 * \note (1) The memory for the data of 'par_vector' has been already allocated. 
 *       (2) vec_starts should not be NULL! 
 *       (3) This is not being optimized to use JX_NO_GLOBAL_PARTITION.
 * \author peghoty
 * \date 2011/08/31
 */
JX_Int
jx_VectorToParVector_Allocated( MPI_Comm       comm, 
                                jx_Vector     *v, 
                                JX_Int           *vec_starts, 
                                jx_ParVector  *par_vector )
{
   JX_Int            global_size;
   JX_Int            local_size;
   JX_Int            num_vectors;
   JX_Int            num_procs, my_id;
   JX_Int            global_vecstride, vecstride, idxstride;
   jx_Vector     *local_vector;
   JX_Real        *v_data = NULL;
   JX_Real        *local_data;
   MPI_Request   *requests;
   MPI_Status    *status, status0;
   JX_Int            i, j, k, p;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   if (my_id == 0)
   {
      global_size = jx_VectorSize(v);
      v_data = jx_VectorData(v);
      num_vectors = jx_VectorNumVectors(v);         /* for multivectors */
      global_vecstride = jx_VectorVectorStride(v);
   }

   jx_MPI_Bcast(&global_size, 1, JX_MPI_INT, 0, comm);
   jx_MPI_Bcast(&num_vectors, 1, JX_MPI_INT, 0, comm);
   jx_MPI_Bcast(&global_vecstride, 1, JX_MPI_INT, 0, comm);

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   local_vector = jx_ParVectorLocalVector(par_vector);
   local_data = jx_VectorData(local_vector);

   vecstride = jx_VectorVectorStride(local_vector);
   idxstride = jx_VectorIndexStride(local_vector);
   jx_assert( idxstride == 1 );  /* <<< so far only the only implemented multivector StorageMethod is 0 <<< */

   if (my_id == 0)
   {
      requests = jx_CTAlloc(MPI_Request,num_vectors*(num_procs-1));
      status = jx_CTAlloc(MPI_Status,num_vectors*(num_procs-1));
      k = 0;
      for (p = 1; p < num_procs; p ++)
      {
         for (j = 0; j < num_vectors; ++ j)
         {
            jx_MPI_Isend( &v_data[vec_starts[p]] + j*global_vecstride,
                       (vec_starts[p+1] - vec_starts[p]),
                       JX_MPI_REAL, p, 0, comm, &requests[k++] );
         }
      }
      if ( num_vectors == 1 )
      {
         for (i = 0; i < local_size; i ++)
         {
            local_data[i] = v_data[i];
         }
      }
      else
      {
         for (j = 0; j < num_vectors; ++ j)
         {
            for (i = 0; i < local_size; i ++)
            {
               local_data[i+j*vecstride] = v_data[i+j*global_vecstride];
            }
         }
      }
      jx_MPI_Waitall(num_procs-1, requests, status);
      jx_TFree(requests);
      jx_TFree(status);
   }
   else
   {
      for (j = 0; j < num_vectors; ++ j)
      {
         jx_MPI_Recv( local_data+j*vecstride, local_size, JX_MPI_REAL, 0, 0, comm, &status0 );
      }
   }

   return 0;
}

/*!
 * \fn JX_Int jx_VectorToParVector_Allocated2
 * \brief 利用 0 号进程上的一个串行向量生成一个并行向量，其中并行向量具有并行分划且已开设数据部分.
 * \note 在 jx_VectorToParVector_Allocated 的基础上进行了简化，取 num_vectors = 1.
 * \author peghoty
 * \date 2012/02/27
 */
JX_Int
jx_VectorToParVector_Allocated2( MPI_Comm       comm, 
                                 jx_Vector     *v, 
                                 JX_Int           *vec_starts, 
                                 jx_ParVector  *par_vector )
{
   JX_Int            local_size;
   JX_Int            num_procs, my_id;
   jx_Vector     *local_vector;
   JX_Real        *v_data = NULL;
   JX_Real        *local_data;
   MPI_Request   *requests;
   MPI_Status    *status, status0;
   JX_Int            i, k, p;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   if (my_id == 0)
   {
      v_data = jx_VectorData(v);
   }

   local_size = vec_starts[my_id+1] - vec_starts[my_id];

   local_vector = jx_ParVectorLocalVector(par_vector);
   local_data = jx_VectorData(local_vector);

   if (my_id == 0)
   {
      requests = jx_CTAlloc(MPI_Request, num_procs-1);
      status = jx_CTAlloc(MPI_Status, num_procs-1);
      k = 0;
      for (p = 1; p < num_procs; p ++)
      {
         jx_MPI_Isend( &v_data[vec_starts[p]],
                    (vec_starts[p+1] - vec_starts[p]),
                    JX_MPI_REAL, p, 0, comm, &requests[k++] );
      }

      for (i = 0; i < local_size; i ++)
      {
         local_data[i] = v_data[i];
      }

      jx_MPI_Waitall(num_procs-1, requests, status);
      jx_TFree(requests);
      jx_TFree(status);
   }
   else
   {
      jx_MPI_Recv( local_data, local_size, JX_MPI_REAL, 0, 0, comm, &status0 );
   }

   return 0;
}

/*!
 * \fn JX_Int jx_VectorToParVector_Allocated_FromGivenPro
 * \brief 利用指定进程(srcid 号进程)上的一个串行向量生成一个并行向量，其中并行向量具有并行分划且已开设数据部分.
 * \note 该模块基于 jx_VectorToParVector_Allocated2 来改写的，只适合于 num_vectors = 1 的情形. 
 * \author peghoty
 * \date 2012/02/27
 */
JX_Int
jx_VectorToParVector_Allocated_FromGivenPro( MPI_Comm       comm, 
                                             JX_Int            srcid,
                                             jx_Vector     *v, 
                                             JX_Int           *vec_starts, 
                                             jx_ParVector  *par_vector )
{
   JX_Int            local_size;
   JX_Int            num_procs, my_id;
   jx_Vector     *local_vector = NULL;
   JX_Real        *v_data = NULL;
   JX_Real        *local_data = NULL;
   MPI_Request   *requests = NULL;
   MPI_Status    *status = NULL;
   MPI_Status     status0;
   JX_Int            i, k, p;
   JX_Int            start = 0;
   JX_Int            length = 0;
   JX_Int            tag = 111;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   
   if (my_id == srcid)
   {
      v_data = jx_VectorData(v);
   }

   local_size = vec_starts[my_id+1] - vec_starts[my_id]; // 本地进程上向量的长度
   
   local_vector = jx_ParVectorLocalVector(par_vector);
   local_data = jx_VectorData(local_vector);

   if (my_id == srcid)
   {
      requests = jx_CTAlloc(MPI_Request, num_procs - 1);
      status = jx_CTAlloc(MPI_Status, num_procs - 1);
      k = 0;
      
      for (p = 0; p < num_procs; p ++)
      {
         if (p != srcid)
         {
            start = vec_starts[p];
            length = vec_starts[p+1] - start;
            jx_MPI_Isend( &v_data[start], length, JX_MPI_REAL, p, tag, comm, &requests[k++] );
         }
      }
      
      start = vec_starts[my_id];
      for (i = 0; i < local_size; i ++)
      {
         local_data[i] = v_data[start+i]; // be careful here, DON't forget the start. peghoty,  2012/03/07
      }

      jx_MPI_Waitall(num_procs-1, requests, status);
      jx_TFree(requests);
      jx_TFree(status);
   }
   else
   {
      jx_MPI_Recv( local_data, local_size, JX_MPI_REAL, srcid, tag, comm, &status0 );
   }

   return 0;
}

/*!
 * \fn JX_Int jx_ParVectorSetDataOwner
 * \brief Set owns_data of a parallel vector.
 * \date 2011/09/03
 */
JX_Int 
jx_ParVectorSetDataOwner( jx_ParVector *vector, JX_Int owns_data )
{
   if (!vector)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_ParVectorOwnsData(vector) = owns_data;

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParVectorSetPartitioningOwner
 * \brief Set owns_partitioning of a parallel vector.
 * \date 2011/09/03
 */
JX_Int 
jx_ParVectorSetPartitioningOwner( jx_ParVector *vector, JX_Int owns_partitioning )
{
   if (!vector)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   jx_ParVectorOwnsPartitioning(vector) = owns_partitioning;

   return jx_error_flag;
}

/*!
 * \fn jx_ParVecCommPkg *jx_ParVecCommPkgCreate
 * \brief Create a communication package for the switch between two
 *        parallel vectors with different partitionings.
 * \param comm communication
 * \param *px pointer to the partioning of the source vector.
 * \param *py pointer to the partioning of the target vector. 
 * \author peghoty
 * \date 2011/09/07
 */
jx_ParVecCommPkg *
jx_ParVecCommPkgCreate( MPI_Comm comm, JX_Int *px, JX_Int *py )
{ 
   jx_ParVecCommPkg *comm_pkg = NULL; 

   JX_Int  num_sends;
   JX_Int *send_procs  = NULL;
   JX_Int *send_starts = NULL;

   JX_Int  num_recvs;
   JX_Int *recv_procs  = NULL;
   JX_Int *recv_starts = NULL;
   
   JX_Int i, j, k, m;
   JX_Int last = 0;
   JX_Int start, end;
   JX_Int local_size;
   JX_Int myid, np;

   jx_MPI_Comm_rank(comm, &myid);
   jx_MPI_Comm_size(comm, &np); 
   
   //-------------------------------------------
   //  统计发送进程个数 num_sends
   //-------------------------------------------
   
   start = px[myid];
   end   = px[myid+1];
   local_size = end - start;
   
   k = 0;   
   for (i = 0; i < np; i ++)
   {
      j = py[i];
      if ( j > start && j < end )
      {
         k ++;  
      }
      else if (j >= end) 
      {
         break;  
      }  
   }
   num_sends = k + 1; // 

   //-------------------------------------------
   //  提取发送进程， 生成数组 send_procs
   //  同时生成发送数据管理数组 send_starts
   //-------------------------------------------
 
   send_procs  = jx_CTAlloc(JX_Int, num_sends);
   send_starts = jx_CTAlloc(JX_Int, num_sends + 1);

   k = 0;
   m = start;
   send_starts[0] = 0;
   for (i = 0; i < np+1; i ++)
   {
      j = py[i];
      if ( j > start && j < end )
      {
         send_procs[k] = i - 1;
         k ++;

         send_starts[k] = send_starts[k-1] + (j - m);
         m = j;  
      }
      else if (j >= end) 
      {
         last = i - 1;
         break;  
      }  
   }
   if (num_sends > 1)
   {
      send_procs[num_sends-1] = send_procs[num_sends-2] + 1; // 利用发送进程编号的单调性和连续性
   }
   else
   {
      send_procs[num_sends-1] = last;
   }
   send_starts[num_sends] = local_size;
 

   //-------------------------------------------
   //  统计接收进程个数 num_recvs
   //-------------------------------------------
   
   start = py[myid];
   end   = py[myid+1];
   local_size = end - start;
   
   k = 0;   
   for (i = 0; i < np; i ++)
   {
      j = px[i];
      if ( j > start && j < end )
      {
         k ++;  
      }
      else if (j >= end) 
      {
         break;  
      }  
   }
   num_recvs = k + 1; // 
   
   
   //-------------------------------------------
   //  提取接收进程， 生成数组 recv_procs
   //  同时生成接收数据管理数组 recv_starts
   //-------------------------------------------
 
   recv_procs  = jx_CTAlloc(JX_Int, num_recvs);
   recv_starts = jx_CTAlloc(JX_Int, num_recvs + 1);

   k = 0;
   m = start;
   recv_starts[0] = 0;
   for (i = 0; i < np+1; i ++)
   {
      j = px[i];
      if ( j > start && j < end )
      {
         recv_procs[k] = i - 1;
         k ++;

         recv_starts[k] = recv_starts[k-1] + (j - m);
         m = j;  
      }
      else if (j >= end) 
      {
         last = i - 1;
         break;  
      }  
   }
   if (num_recvs > 1)
   {
      recv_procs[num_recvs-1] = recv_procs[num_recvs-2] + 1; // 利用发送进程编号的单调性和连续性
   }
   else
   {
      recv_procs[num_recvs-1] = last;
   }
   recv_starts[num_recvs] = local_size;
  
   comm_pkg  = jx_CTAlloc(jx_ParVecCommPkg, 1);

   jx_ParVecCommPkgComm(comm_pkg)       = comm;                                   
   jx_ParVecCommPkgNumSends(comm_pkg)   = num_sends;
   jx_ParVecCommPkgSendProcs(comm_pkg)  = send_procs;
   jx_ParVecCommPkgSendStarts(comm_pkg) = send_starts;
   jx_ParVecCommPkgNumRecvs(comm_pkg)   = num_recvs;
   jx_ParVecCommPkgRecvProcs(comm_pkg)  = recv_procs;
   jx_ParVecCommPkgRecvStarts(comm_pkg) = recv_starts;

   return (comm_pkg); 
}

/*!
 * \fn jx_ParVecCommHandle *jx_ParVecCommHandleCreate
 * \brief Create a communication handle for the switch between two
 *        parallel vectors with different partitionings.
 *        Data exchanging takes place in this function.
 * \author peghoty
 * \date 2011/09/07
 */
jx_ParVecCommHandle *
jx_ParVecCommHandleCreate( JX_Int                job, 
                           jx_ParVecCommPkg  *comm_pkg, 
                           void              *ssend_data, 
                           void              *rrecv_data )
{
   jx_ParVecCommHandle *comm_handle = NULL;
                               
   /* members of the object comm_pkg */
   MPI_Comm comm        = jx_ParVecCommPkgComm(comm_pkg);
   JX_Int      num_sends   = jx_ParVecCommPkgNumSends(comm_pkg);
   JX_Int      num_recvs   = jx_ParVecCommPkgNumRecvs(comm_pkg);
   JX_Int     *send_procs  = jx_ParVecCommPkgSendProcs(comm_pkg);
   JX_Int     *send_starts = jx_ParVecCommPkgSendStarts(comm_pkg);
   JX_Int     *recv_procs  = jx_ParVecCommPkgRecvProcs(comm_pkg);
   JX_Int     *recv_starts = jx_ParVecCommPkgRecvStarts(comm_pkg);


   JX_Int           num_requests;
   MPI_Request  *requests = NULL;

   JX_Int           i, j;
   JX_Int           my_id, num_procs;
   JX_Int           ip, vec_start, vec_len;

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
    
   num_requests = num_sends + num_recvs;
   requests = jx_CTAlloc(MPI_Request, num_requests);  
   
   switch (job)
   {
      case 1: // JX_Real
      {
         JX_Real *send_data = (JX_Real *) ssend_data;
         JX_Real *recv_data = (JX_Real *) rrecv_data;      
         j = 0;
         for (i = 0; i < num_recvs; i ++)
         {
            ip = recv_procs[i]; 
            vec_start = recv_starts[i];
            vec_len = recv_starts[i+1] - vec_start;
            jx_MPI_Irecv(&recv_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i ++)
         {
            ip = send_procs[i]; 
            vec_start = send_starts[i];
            vec_len = send_starts[i+1] - vec_start;
            jx_MPI_Isend(&send_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
      }
      break; 
  
      case 2: // JX_Real
      {
         JX_Real *send_data = (JX_Real *) ssend_data;
         JX_Real *recv_data = (JX_Real *) rrecv_data;    
         j = 0;
         for (i = 0; i < num_sends; i ++)
         {
            ip = send_procs[i]; 
            vec_start = send_starts[i];
            vec_len = send_starts[i+1] - vec_start;
            jx_MPI_Irecv(&recv_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = recv_procs[i]; 
            vec_start = recv_starts[i];
            vec_len = recv_starts[i+1] - vec_start;
            jx_MPI_Isend(&send_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
      }
      break; 
      
      case 11: // JX_Int
      {
         JX_Int *send_data = (JX_Int *) ssend_data;
         JX_Int *recv_data = (JX_Int *) rrecv_data;      
         j = 0;
         for (i = 0; i < num_recvs; i ++)
         {
            ip = recv_procs[i]; 
            vec_start = recv_starts[i];
            vec_len = recv_starts[i+1] - vec_start;
            jx_MPI_Irecv(&recv_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i ++)
         {
            ip = send_procs[i]; 
            vec_start = send_starts[i];
            vec_len = send_starts[i+1] - vec_start;
            jx_MPI_Isend(&send_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
      }
      break;
      
      case 12: // JX_Int
      {
         JX_Int *send_data = (JX_Int *) ssend_data;
         JX_Int *recv_data = (JX_Int *) rrecv_data;    
         j = 0;
         for (i = 0; i < num_sends; i ++)
         {
            ip = send_procs[i]; 
            vec_start = send_starts[i];
            vec_len = send_starts[i+1] - vec_start;
            jx_MPI_Irecv(&recv_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = recv_procs[i]; 
            vec_start = recv_starts[i];
            vec_len = recv_starts[i+1] - vec_start;
            jx_MPI_Isend(&send_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
      }
      break;                    
   }	
	   
   
   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = jx_CTAlloc(jx_ParVecCommHandle, 1);

   /* 这三个成员目前还没有用到. peghoty, 2012/02/29 */
   //jx_ParVecCommHandleCommPkg(comm_handle)     = comm_pkg;
   //jx_ParVecCommHandleSendData(comm_handle)    = ssend_data;
   //jx_ParVecCommHandleRecvData(comm_handle)    = rrecv_data;
   jx_ParVecCommHandleNumRequests(comm_handle) = num_requests;
   jx_ParVecCommHandleRequests(comm_handle)    = requests;  
   
   return (comm_handle);
}

/*!
 * \fn JX_Int jx_ParVecCommHandleDestroy
 * \brief Destroy a communication handle for the switch between two
 *        parallel vectors with different partitionings.
 * \author peghoty
 * \date 2011/09/07
 */
JX_Int
jx_ParVecCommHandleDestroy( jx_ParVecCommHandle *comm_handle )
{
   MPI_Status   *status   = NULL;
   
   MPI_Request  *requests = jx_ParVecCommHandleRequests(comm_handle);
   JX_Int num_requests = jx_ParVecCommHandleNumRequests(comm_handle); 
      
   if (comm_handle == NULL) 
   {
      return jx_error_flag;
   }
   
   if (num_requests)
   {
      status = jx_CTAlloc(MPI_Status, num_requests);
      jx_MPI_Waitall(num_requests, requests, status);
      jx_TFree(status);
   }

   jx_TFree(requests);
   jx_TFree(comm_handle);

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParVecCommPkgDestroy
 * \brief Destroy a communication package for the switch between two
 *        parallel vectors with different partitionings.
 * \author peghoty
 * \date 2011/09/07
 */
JX_Int
jx_ParVecCommPkgDestroy( jx_ParVecCommPkg *comm_pkg )
{
   if (jx_ParVecCommPkgNumSends(comm_pkg))
   {
      jx_TFree(jx_ParVecCommPkgSendProcs(comm_pkg));
   }
   jx_TFree(jx_ParVecCommPkgSendStarts(comm_pkg));

   if (jx_ParVecCommPkgNumRecvs(comm_pkg))
   {
      jx_TFree(jx_ParVecCommPkgRecvProcs(comm_pkg));
   }
   jx_TFree(jx_ParVecCommPkgRecvStarts(comm_pkg));

   jx_TFree(comm_pkg);

   return jx_error_flag;
}

/*!
 * \fn JX_Int jx_ParVecSwitch
 * \brief Switch between two parallel vectors with different partitionings.
 * \param comm_pkg pointer to the communication package.
 * \param job flag to indicate what to do.
 *        job = 1: x -> y
 *        job = 2: y -> x
 * \param x pointer to the first vector
 * \param y pointer to the second vector
 * \author peghoty
 * \date 2011/09/07
 */
JX_Int 
jx_ParVecSwitch( jx_ParVecCommPkg  *comm_pkg,
                 JX_Int                job,
                 jx_ParVector      *x,
                 jx_ParVector      *y )
{  
   JX_Real *send_buf = NULL;
   JX_Real *recv_buf = NULL;
   jx_ParVecCommHandle *comm_handle = NULL;
   
   switch (job)
   {
      case 1: // x -> y
      {
         send_buf = jx_VectorData(jx_ParVectorLocalVector(x));
         recv_buf = jx_VectorData(jx_ParVectorLocalVector(y));
   
         comm_handle = jx_ParVecCommHandleCreate(1, comm_pkg, send_buf, recv_buf);
         jx_ParVecCommHandleDestroy(comm_handle);
      }
      break;   

      case 2: // y -> x
      {
         send_buf = jx_VectorData(jx_ParVectorLocalVector(y));
         recv_buf = jx_VectorData(jx_ParVectorLocalVector(x));
         comm_handle = jx_ParVecCommHandleCreate(2, comm_pkg, send_buf, recv_buf);
         jx_ParVecCommHandleDestroy(comm_handle); 
      }
      break;              
   }
   
   return 0;
}
