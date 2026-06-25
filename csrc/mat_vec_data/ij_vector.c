//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  ij_vector.c -- basic operations for IJ vectors
 *  Date: 2015/11/28
 */

#include "jx_mv.h"

JX_Int
JX_IJVectorCreate( MPI_Comm     comm,
                   JX_Int          jlower,
                   JX_Int          jupper,
                   JX_IJVector *vector )
{
   jx_IJVector *vec;
   JX_Int num_procs, my_id, *partitioning;
#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int  row0, rowN;
#else
   JX_Int *recv_buf;
   JX_Int *info;
   JX_Int i, i2;
#endif

   vec = jx_CTAlloc(jx_IJVector, 1);
   if (!vec)
   {
      jx_error(JX_ERROR_MEMORY);
      return jx_error_flag;
   }

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   if (jlower > jupper+1 || jlower < 0)
   {
      jx_error_in_arg(2);
      jx_TFree(vec);
      return jx_error_flag;
   }
   if (jupper < -1)
   {
      jx_error_in_arg(3);
      return jx_error_flag;
   }

#ifdef JX_NO_GLOBAL_PARTITION

   partitioning = jx_CTAlloc(JX_Int, 2);

   partitioning[0] = jlower;
   partitioning[1] = jupper+1;

   /* now we need the global number of rows as well
      as the global first row index */

   /* proc 0 has the first row  */
   if (my_id==0) 
   {
      row0 = jlower;
   }
   jx_MPI_Bcast(&row0, 1, JX_MPI_INT, 0, comm);
   /* proc (num_procs-1) has the last row */
   if (my_id == (num_procs-1))
   {
      rowN = jupper;
   }
   jx_MPI_Bcast(&rowN, 1, JX_MPI_INT, num_procs-1, comm);

   jx_IJVectorGlobalFirstRow(vec) = row0;
   jx_IJVectorGlobalNumRows(vec) = rowN - row0 + 1;

#else

   info = jx_CTAlloc(JX_Int,2);
   recv_buf = jx_CTAlloc(JX_Int, 2*num_procs);
   partitioning = jx_CTAlloc(JX_Int, num_procs+1);

   info[0] = jlower;
   info[1] = jupper;

   jx_MPI_Allgather(info, 2, JX_MPI_INT, recv_buf, 2, JX_MPI_INT, comm);

   partitioning[0] = recv_buf[0];
   for (i=0; i < num_procs-1; i++)
   {
      i2 = i+i;
      if (recv_buf[i2+1] != (recv_buf[i2+2]-1))
      {
	 jx_error(JX_ERROR_GENERIC);
         jx_TFree(info);
         jx_TFree(recv_buf);
         jx_TFree(partitioning);
         jx_TFree(vec);
         return jx_error_flag;
      }
      else
	 partitioning[i+1] = recv_buf[i2+2];
   }
   i2 = (num_procs-1)*2;
   partitioning[num_procs] = recv_buf[i2+1]+1;

   jx_TFree(info);
   jx_TFree(recv_buf);

   jx_IJVectorGlobalFirstRow(vec) = partitioning[0];
   jx_IJVectorGlobalNumRows(vec)= partitioning[num_procs]-partitioning[0];

#endif

   jx_IJVectorComm(vec)         = comm;
   jx_IJVectorPartitioning(vec) = partitioning;
   jx_IJVectorObjectType(vec)   = JX_UNITIALIZED;
   jx_IJVectorObject(vec)       = NULL;
   jx_IJVectorTranslator(vec)   = NULL;
   jx_IJVectorAssumedPart(vec)  = NULL;
   jx_IJVectorPrintLevel(vec)   = 0;

  *vector = (JX_IJVector) vec;
   
   return jx_error_flag;
}

JX_Int
jx_IJVectorCreatePar( jx_IJVector *vector, JX_Int *IJpartitioning )
{
   MPI_Comm comm = jx_IJVectorComm(vector);
   JX_Int num_procs, jmin, global_n, *partitioning, j;
   jx_MPI_Comm_size(comm, &num_procs);

#ifdef JX_NO_GLOBAL_PARTITION
   jmin = jx_IJVectorGlobalFirstRow(vector);
   global_n = jx_IJVectorGlobalNumRows(vector);

   partitioning = jx_CTAlloc(JX_Int, 2); 

   /* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < 2; j++) 
      partitioning[j] = IJpartitioning[j] - jmin;

#else
   jmin = IJpartitioning[0];
   global_n = IJpartitioning[num_procs] - jmin;

   partitioning = jx_CTAlloc(JX_Int, num_procs+1); 

   /* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < num_procs+1; j++) 
      partitioning[j] = IJpartitioning[j] - jmin;

#endif

   jx_IJVectorObject(vector) = jx_ParVectorCreate(comm, global_n, (JX_Int *) partitioning);

   return jx_error_flag;
}

JX_Int
jx_AuxParVectorCreate( jx_AuxParVector **aux_vector )
{
   jx_AuxParVector  *vector;

   vector = jx_CTAlloc(jx_AuxParVector, 1);

   /* set defaults */
   jx_AuxParVectorMaxOffProcElmts(vector) = 0;
   jx_AuxParVectorCurrentNumElmts(vector) = 0;
   /* stash for setting or adding off processor values */
   jx_AuxParVectorOffProcI(vector) = NULL;
   jx_AuxParVectorOffProcData(vector) = NULL;

  *aux_vector = vector;
   return 0;
}

JX_Int
JX_IJVectorInitialize( JX_IJVector vector )
{
   jx_IJVector *vec = (jx_IJVector *) vector;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (jx_IJVectorObjectType(vec) == JX_PARCSR)
   {
      if (!jx_IJVectorObject(vec)) jx_IJVectorCreatePar(vec, jx_IJVectorPartitioning(vec));
      jx_IJVectorInitializePar(vec);
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJVectorInitializePar( jx_IJVector *vector )
{
   jx_ParVector *par_vector = jx_IJVectorObject(vector);
   jx_AuxParVector *aux_vector = jx_IJVectorTranslator(vector);
   JX_Int *partitioning = jx_ParVectorPartitioning(par_vector);
   jx_Vector *local_vector = jx_ParVectorLocalVector(par_vector);
   JX_Int my_id;
   JX_Int print_level = jx_IJVectorPrintLevel(vector);

   MPI_Comm  comm = jx_IJVectorComm(vector);
   jx_MPI_Comm_rank(comm, &my_id);
   if (!partitioning)
   {
      if (print_level)
      {
         jx_printf("No ParVector partitioning for initialization -- ");
         jx_printf("jx_IJVectorInitializePar\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

#ifdef JX_NO_GLOBAL_PARTITION
   jx_VectorSize(local_vector) = partitioning[1] - partitioning[0];
#else
   jx_VectorSize(local_vector) = partitioning[my_id+1] - partitioning[my_id];
#endif

   jx_ParVectorInitialize(par_vector);
   if (!aux_vector)
   {
      jx_AuxParVectorCreate(&aux_vector);
      jx_IJVectorTranslator(vector) = aux_vector;
   }
   jx_AuxParVectorInitialize(aux_vector);

   return jx_error_flag;
}

JX_Int
jx_AuxParVectorInitialize( jx_AuxParVector *vector )
{
   JX_Int max_off_proc_elmts = jx_AuxParVectorMaxOffProcElmts(vector);
   /* allocate stash for setting or adding off processor values */
   if (max_off_proc_elmts > 0)
   {
      jx_AuxParVectorOffProcI(vector) = jx_CTAlloc(JX_Int, max_off_proc_elmts);
      jx_AuxParVectorOffProcData(vector) = jx_CTAlloc(JX_Real, max_off_proc_elmts);
   }
   return 0;
}

JX_Int
JX_IJVectorDestroy( JX_IJVector vector )
{
   jx_IJVector *vec = (jx_IJVector *) vector;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (jx_IJVectorPartitioning(vec))
      jx_TFree(jx_IJVectorPartitioning(vec));
   if (jx_IJVectorAssumedPart(vec))
      jx_AssumedPartitionDestroy(jx_IJVectorAssumedPart(vec));
   if ( jx_IJVectorObjectType(vec) == JX_PARCSR)
   {
      jx_IJVectorDestroyPar(vec);
      if (jx_IJVectorTranslator(vec))
      {
         jx_AuxParVectorDestroy((jx_AuxParVector *) (jx_IJVectorTranslator(vec)));
      }
   }
   else if (jx_IJVectorObjectType(vec) != -1)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_TFree(vec);

   return jx_error_flag;
}

JX_Int
jx_IJVectorDestroyPar( jx_IJVector *vector )
{
   return jx_ParVectorDestroy(jx_IJVectorObject(vector));
}

JX_Int
jx_AuxParVectorDestroy( jx_AuxParVector *vector )
{
   JX_Int ierr=0;
   if (vector)
   {
      if (jx_AuxParVectorOffProcI(vector))
         jx_TFree(jx_AuxParVectorOffProcI(vector));
      if (jx_AuxParVectorOffProcData(vector))
         jx_TFree(jx_AuxParVectorOffProcData(vector));
      jx_TFree(vector);
   }

   return ierr;
}

JX_Int
JX_IJVectorSetObjectType( JX_IJVector vector, JX_Int type )
{
   jx_IJVector *vec = (jx_IJVector *) vector;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_IJVectorObjectType(vec) = type;

   return jx_error_flag;
}

JX_Int
JX_IJVectorSetValues( JX_IJVector   vector,
                      JX_Int           nvalues,
                      const JX_Int    *indices,
                      const JX_Real *values )
{
   jx_IJVector *vec = (jx_IJVector *) vector;

   if (nvalues == 0) return jx_error_flag;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (nvalues < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   if (!values)
   {
      jx_error_in_arg(4);
      return jx_error_flag;
   }
   if (jx_IJVectorObjectType(vec) == JX_PARCSR)
   {
      return(jx_IJVectorSetValuesPar(vec, nvalues, indices, values));
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJVectorSetValuesPar( jx_IJVector  *vector,
                         JX_Int           num_values,
                         const JX_Int    *indices,
                         const JX_Real *values )
{
   JX_Int my_id;
   JX_Int i, j, vec_start, vec_stop;
   JX_Real *data;
   JX_Int print_level = jx_IJVectorPrintLevel(vector);
   JX_Int *IJpartitioning = jx_IJVectorPartitioning(vector);
   jx_ParVector *par_vector = jx_IJVectorObject(vector);
   jx_AuxParVector *aux_vector = jx_IJVectorTranslator(vector);
   MPI_Comm comm = jx_IJVectorComm(vector);
   jx_Vector *local_vector;

   /* If no components are to be set, perform no checking and return */
   if (num_values < 1) return 0;

   jx_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
      let user know of catastrophe and exit */
   if (!par_vector)
   {
      if (print_level)
      {
         jx_printf("par_vector == NULL -- ");
         jx_printf("jx_IJVectorSetValuesPar\n");
         jx_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   local_vector = jx_ParVectorLocalVector(par_vector);
   if (!IJpartitioning)
   {
      if (print_level)
      {
         jx_printf("IJpartitioning == NULL -- ");
         jx_printf("jx_IJVectorSetValuesPar\n");
         jx_printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (!local_vector)
   {
      if (print_level)
      {
         jx_printf("local_vector == NULL -- ");
         jx_printf("jx_IJVectorSetValuesPar\n");
         jx_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

#ifdef JX_NO_GLOBAL_PARTITION
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1]-1;
#else
   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1]-1;
#endif

   if (vec_start > vec_stop) 
   {
      if (print_level)
      {
         jx_printf("vec_start > vec_stop -- ");
         jx_printf("jx_IJVectorSetValuesPar\n");
         jx_printf("**** This vector partitioning should not occur ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   /* Determine whether indices points to local indices only, and if not, store
      indices and values in auxiliary vector structure.  If indices == NULL,
      assume that num_values components are to be set in a block starting at
      vec_start.  NOTE: If indices == NULL off proc values are ignored!!! */

   data = jx_VectorData(local_vector);

   if (indices)
   {
      JX_Int current_num_elmts = jx_AuxParVectorCurrentNumElmts(aux_vector);
      JX_Int *off_proc_i = jx_AuxParVectorOffProcI(aux_vector);
      JX_Int cancel_indx = jx_AuxParVectorCancelIndx(aux_vector);
      JX_Int ii;
      for (j = 0; j < num_values; j++)
      {
	 i = indices[j];
	 if (i < vec_start || i > vec_stop)
         {
            for (ii = 0; ii < current_num_elmts; ii++)
	    {
	       if (i == off_proc_i[ii])
	       {
		  off_proc_i[ii] = -1;
		  cancel_indx++;
               }
            }
            jx_AuxParVectorCancelIndx(aux_vector) = cancel_indx;
         }
         else /* local values are inserted into the vector */
         {
            i -= vec_start;
            data[i] = values[j];
         }
      } 
   }
   else
   {
      if (num_values > vec_stop - vec_start + 1)
      {
         if (print_level)
         {
            jx_printf("Warning! Indices beyond local range  not identified!\n ");
            jx_printf("Off processor values have been ignored!\n");
         }
	 num_values = vec_stop - vec_start +1;
      }
#if JX_USING_OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
      for (j = 0; j < num_values; j++) data[j] = values[j];
   } 
  
   return jx_error_flag;
}

JX_Int
JX_IJVectorAssemble( JX_IJVector vector )
{
   jx_IJVector *vec = (jx_IJVector *) vector;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (jx_IJVectorObjectType(vec) == JX_PARCSR)
   {
      return(jx_IJVectorAssemblePar(vec));
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJVectorAssemblePar( jx_IJVector *vector )
{
   JX_Int *IJpartitioning = jx_IJVectorPartitioning(vector);
   jx_ParVector *par_vector = jx_IJVectorObject(vector);
   jx_AuxParVector *aux_vector = jx_IJVectorTranslator(vector);
   JX_Int *partitioning;
   MPI_Comm comm = jx_IJVectorComm(vector);
   JX_Int print_level = jx_IJVectorPrintLevel(vector);

   if (!par_vector)
   {
      if (print_level)
      {
         jx_printf("par_vector == NULL -- ");
         jx_printf("jx_IJVectorAssemblePar\n");
         jx_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
   } 
   partitioning = jx_ParVectorPartitioning(par_vector);
   if (!IJpartitioning)
   { 
      if (print_level)
      {
         jx_printf("IJpartitioning == NULL -- ");
         jx_printf("jx_IJVectorAssemblePar\n");
         jx_printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
   }
   if (!partitioning)
   { 
      if (print_level)
      {
         jx_printf("partitioning == NULL -- ");
         jx_printf("jx_IJVectorAssemblePar\n");
         jx_printf("**** ParVector partitioning is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
   }

   if (aux_vector)
   {
      JX_Int off_proc_elmts, current_num_elmts;
      JX_Int max_off_proc_elmts;
      JX_Int *off_proc_i;
      JX_Real *off_proc_data;
      JX_Int cancel_indx = jx_AuxParVectorCancelIndx(aux_vector);
      JX_Int current_i, ii;
      current_num_elmts = jx_AuxParVectorCurrentNumElmts(aux_vector);
      if (cancel_indx)
      {
         off_proc_i=jx_AuxParVectorOffProcI(aux_vector);
         off_proc_data=jx_AuxParVectorOffProcData(aux_vector);
         current_i = 0;
	 for (ii=0; ii < current_num_elmts; ii++) 
         {
            if (off_proc_i[ii] != -1)
	    {
	       off_proc_i[current_i] = off_proc_i[ii];
	       off_proc_data[current_i++] = off_proc_data[ii];
	    }
         }
         jx_AuxParVectorCurrentNumElmts(aux_vector) = current_i;
         current_num_elmts = current_i;
      }
      jx_MPI_Allreduce(&current_num_elmts,&off_proc_elmts,1,JX_MPI_INT,MPI_SUM,comm);
      if (off_proc_elmts)
      {
         max_off_proc_elmts=jx_AuxParVectorMaxOffProcElmts(aux_vector);
         off_proc_i=jx_AuxParVectorOffProcI(aux_vector);
         off_proc_data=jx_AuxParVectorOffProcData(aux_vector);
         jx_IJVectorAssembleOffProcValsPar(vector, max_off_proc_elmts, 
                                  current_num_elmts, off_proc_i, off_proc_data);
	 jx_TFree(jx_AuxParVectorOffProcI(aux_vector));
	 jx_TFree(jx_AuxParVectorOffProcData(aux_vector));
	 jx_AuxParVectorMaxOffProcElmts(aux_vector) = 0;
	 jx_AuxParVectorCurrentNumElmts(aux_vector) = 0;
      }
   }

   return jx_error_flag;
}

#ifndef JX_NO_GLOBAL_PARTITION

JX_Int
jx_IJVectorAssembleOffProcValsPar( jx_IJVector *vector, 
   				   JX_Int          max_off_proc_elmts,
   				   JX_Int          current_num_elmts,
   				   JX_Int         *off_proc_i,
   			     	   JX_Real      *off_proc_data )
{
   MPI_Comm comm = jx_IJVectorComm(vector);
   jx_ParVector *par_vector = jx_IJVectorObject(vector);
   MPI_Request *requests = NULL;
   MPI_Status *status = NULL;
   JX_Int i, j, j2, row;
   JX_Int iii, indx, ip, first_index;
   JX_Int proc_id, num_procs, my_id;
   JX_Int num_sends, num_sends2;
   JX_Int num_recvs;
   JX_Int num_requests;
   JX_Int vec_start, vec_len;
   JX_Int *send_procs;
   JX_Int *send_i;
   JX_Int *send_map_starts;
   JX_Int *recv_procs;
   JX_Int *recv_i;
   JX_Int *recv_vec_starts;
   JX_Int *info;
   JX_Int *int_buffer;
   JX_Int *proc_id_mem;
   JX_Int *partitioning;
   JX_Int *displs;
   JX_Int *recv_buf;
   JX_Real *send_data;
   JX_Real *recv_data;
   JX_Real *data = jx_VectorData(jx_ParVectorLocalVector(par_vector));

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);
   partitioning = jx_IJVectorPartitioning(vector);
   first_index = partitioning[my_id];
   info = jx_CTAlloc(JX_Int,num_procs);  
   proc_id_mem = jx_CTAlloc(JX_Int,current_num_elmts);
   for (i=0; i < current_num_elmts; i++)
   {
      row = off_proc_i[i];
      proc_id = jx_FindProc(partitioning,row,num_procs);
      proc_id_mem[i] = proc_id; 
      info[proc_id]++;
   }

   /* determine send_procs and amount of data to be sent */   
   num_sends = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         num_sends++;
      }
   }
   num_sends2 = 2*num_sends;
   send_procs = jx_CTAlloc(JX_Int,num_sends);
   send_map_starts = jx_CTAlloc(JX_Int,num_sends+1);
   int_buffer = jx_CTAlloc(JX_Int,num_sends2);
   j = 0;
   j2 = 0;
   send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         send_procs[j++] = i;
         send_map_starts[j] = send_map_starts[j-1]+info[i];
         int_buffer[j2++] = i;
	 int_buffer[j2++] = info[i];
      }
   }

   jx_MPI_Allgather(&num_sends2,1,JX_MPI_INT,info,1,JX_MPI_INT,comm);

   displs = jx_CTAlloc(JX_Int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
      displs[i] = displs[i-1]+info[i-1];
   recv_buf = jx_CTAlloc(JX_Int, displs[num_procs]);

   jx_MPI_Allgatherv(int_buffer,num_sends2,JX_MPI_INT,recv_buf,info,displs,JX_MPI_INT,comm);

   jx_TFree(int_buffer);
   jx_TFree(info);

   /* determine recv procs and amount of data to be received */
   num_recvs = 0;
   for (j=0; j < displs[num_procs]; j+=2)
   {
      if (recv_buf[j] == my_id)
	 num_recvs++;
   }

   recv_procs = jx_CTAlloc(JX_Int,num_recvs);
   recv_vec_starts = jx_CTAlloc(JX_Int,num_recvs+1);

   j2 = 0;
   recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      for (j=displs[i]; j < displs[i+1]; j+=2)
      {
         if (recv_buf[j] == my_id)
         {
	    recv_procs[j2++] = i;
	    recv_vec_starts[j2] = recv_vec_starts[j2-1]+recv_buf[j+1];
         }
         if (j2 == num_recvs) break;
      }
   }
   jx_TFree(recv_buf);
   jx_TFree(displs);

   /* set up data to be sent to send procs */
   /* send_i contains for each send proc 
      indices, send_data contains corresponding values */
   send_i = jx_CTAlloc(JX_Int,send_map_starts[num_sends]);
   send_data = jx_CTAlloc(JX_Real,send_map_starts[num_sends]);
   recv_i = jx_CTAlloc(JX_Int,recv_vec_starts[num_recvs]);
   recv_data = jx_CTAlloc(JX_Real,recv_vec_starts[num_recvs]);

   for (i=0; i < current_num_elmts; i++)
   {
      proc_id = proc_id_mem[i];
      indx = jx_BinarySearch(send_procs,proc_id,num_sends);
      iii = send_map_starts[indx];
      send_i[iii] = off_proc_i[i]; 
      send_data[iii] = off_proc_data[i];
      send_map_starts[indx]++;
   }

   jx_TFree(proc_id_mem);

   for (i=num_sends; i > 0; i--)
   {
      send_map_starts[i] = send_map_starts[i-1];
   }
   send_map_starts[0] = 0;

   num_requests = num_recvs+num_sends;

   requests = jx_CTAlloc(MPI_Request, num_requests);
   status = jx_CTAlloc(MPI_Status, num_requests);

   j=0; 
   for (i=0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i+1] - vec_start;
      ip = recv_procs[i];
      jx_MPI_Irecv(&recv_i[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      jx_MPI_Isend(&send_i[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
   }
  
   if (num_requests)
   {
      jx_MPI_Waitall(num_requests, requests, status);
   }

   j=0;
   for (i=0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i+1] - vec_start;
      ip = recv_procs[i];
      jx_MPI_Irecv(&recv_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      jx_MPI_Isend(&send_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
   }
  
   if (num_requests)
   {
      jx_MPI_Waitall(num_requests, requests, status);
   }

   jx_TFree(requests);
   jx_TFree(status);
   jx_TFree(send_i);
   jx_TFree(send_data);
   jx_TFree(send_procs);
   jx_TFree(send_map_starts);
   jx_TFree(recv_procs);

   for (i=0; i < recv_vec_starts[num_recvs]; i++)
   {
      row = recv_i[i];
      j = row - first_index;
      data[j] += recv_data[i];
   }

   jx_TFree(recv_vec_starts);
   jx_TFree(recv_i);
   jx_TFree(recv_data);

   return jx_error_flag;
}

#else

/* assumed partition version */

JX_Int
jx_IJVectorAssembleOffProcValsPar( jx_IJVector *vector, 
   				   JX_Int          max_off_proc_elmts,
   				   JX_Int          current_num_elmts,
   				   JX_Int         *off_proc_i,
   			     	   JX_Real      *off_proc_data )
{
   JX_Int myid, global_num_rows;
   JX_Int global_first_row;
   JX_Int i, j, in, k;
   JX_Int proc_id, last_proc, prev_id, tmp_id;
   JX_Int max_response_size;
   JX_Int ex_num_contacts = 0;
   JX_Int range_start, range_end;
   JX_Int storage;
   JX_Int indx;
   JX_Int row, num_ranges, row_count;
   JX_Int num_recvs;
   JX_Int counter, upper_bound;
   JX_Int num_real_procs;
   JX_Int *row_list=NULL;
   JX_Int *a_proc_id=NULL, *orig_order=NULL;
   JX_Int *real_proc_id = NULL, *us_real_proc_id = NULL;
   JX_Int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL;
   JX_Int *recv_starts=NULL;
   JX_Int *response_buf = NULL, *response_buf_starts=NULL;
   JX_Int *num_rows_per_proc = NULL;
   JX_Int  tmp_int;
   JX_Int  obj_size_bytes, int_size, complex_size;
   JX_Int  first_index;
   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;
   JX_Real tmp_complex;
   JX_Int *ex_contact_buf=NULL;
   JX_Real *vector_data;
   JX_Real value;
   jx_DataExchangeResponse response_obj1, response_obj2;
   jx_ProcListElements send_proc_obj; 

   MPI_Comm comm = jx_IJVectorComm(vector);
   jx_ParVector *par_vector = jx_IJVectorObject(vector);

   jx_IJAssumedPart *apart;

   jx_MPI_Comm_rank(comm, &myid);
   
   global_num_rows = jx_IJVectorGlobalNumRows(vector);
   global_first_row = jx_IJVectorGlobalFirstRow(vector);
 
   /* verify that we have created the assumed partition */

   if  (jx_IJVectorAssumedPart(vector) == NULL)
   {
      jx_IJVectorCreateAssumedPartition(vector);
   }
   apart = jx_IJVectorAssumedPart(vector);

   /* get the assumed processor id for each row */
   a_proc_id = jx_CTAlloc(JX_Int, current_num_elmts);
   orig_order =  jx_CTAlloc(JX_Int, current_num_elmts);
   real_proc_id = jx_CTAlloc(JX_Int, current_num_elmts);
   row_list =   jx_CTAlloc(JX_Int, current_num_elmts);

   if (current_num_elmts > 0)
   {
      for (i=0; i < current_num_elmts; i++)
      {
         row = off_proc_i[i]; 
         row_list[i] = row;
         jx_GetAssumedPartitionProcFromRow(comm, row, global_first_row, global_num_rows, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }

      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/
      
      jx_qsort3i(row_list, a_proc_id, orig_order, 0, current_num_elmts -1);

      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i=1; i < current_num_elmts; i++)
      {
         if (a_proc_id[i] > last_proc)      
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }
      
   }
   
   /* now we will go through a create a contact list - need to contact
      assumed processors and find out who the actual row owner is - we
      will contact with a range (2 numbers) */

   ex_contact_procs = jx_CTAlloc(JX_Int, ex_num_contacts);
   ex_contact_vec_starts =  jx_CTAlloc(JX_Int, ex_num_contacts+1);
   ex_contact_buf =  jx_CTAlloc(JX_Int, ex_num_contacts*2);

   counter = 0;
   range_end = -1;
   for (i=0; i< current_num_elmts; i++) 
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0)  ex_contact_buf[counter*2 - 1] = row_list[i-1];
         
         /*start new range*/
    	 ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter*2;
         ex_contact_buf[counter*2] =  row_list[i];
         counter++;
         
         jx_GetAssumedPartitionRowRange(comm, proc_id, global_first_row,
					global_num_rows, &range_start, &range_end);
      }
   }

   /*finish the starts*/
   ex_contact_vec_starts[counter] =  counter*2;
   /*finish the last range*/
   if (counter > 0)  
      ex_contact_buf[counter*2 - 1] = row_list[current_num_elmts - 1];

   /* create response object - can use same fill response as used in the commpkg
      routine */
   response_obj1.fill_response = jx_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = NULL;
   
   max_response_size = 6;  /* 6 means we can fit 3 ranges*/
   
   jx_DataExchangeList(ex_num_contacts, ex_contact_procs, 
                          ex_contact_buf, ex_contact_vec_starts, sizeof(JX_Int), 
                          sizeof(JX_Int), &response_obj1, max_response_size, 4, 
                          comm, (void**) &response_buf, &response_buf_starts);

   /* now response_buf contains a proc_id followed by an upper bound for the
      range.  */

   jx_TFree(ex_contact_procs);
   jx_TFree(ex_contact_buf);
   jx_TFree(ex_contact_vec_starts);

   jx_TFree(a_proc_id);
   a_proc_id = NULL;

   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];   
   num_ranges = num_ranges/2;
   
   prev_id = -1;
   j = 0;
   counter = 0;
   num_real_procs = 0;

   /* loop through ranges - create a list of actual processor ids*/
   for (i=0; i<num_ranges; i++)
   {
      upper_bound = response_buf[i*2+1];
      counter = 0;
      tmp_id = response_buf[i*2];
      
      /* loop through row_list entries - counting how many are in the range */
      while (j < current_num_elmts && row_list[j] <= upper_bound)     
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;       
      }
      if (counter > 0 && tmp_id != prev_id)        
      {
         num_real_procs++;
      }
      prev_id = tmp_id;
   }

   /* now we have the list of real procesors ids (real_proc_id) - and the number
      of distinct ones - so now we can set up data to be sent - we have
      JX_Int and JX_Real data.  (row number and value) - we will send
      everything as a void since we may not know the rel sizes of ints and
      doubles */
 
   /* first find out how many elements to send per proc - so we can do
      storage */
 
   int_size = sizeof(JX_Int);
   complex_size = sizeof(JX_Real);
   
   obj_size_bytes = jx_max(int_size, complex_size);
    
   ex_contact_procs = jx_CTAlloc(JX_Int, num_real_procs);
   num_rows_per_proc = jx_CTAlloc(JX_Int, num_real_procs);
   
   counter = 0;
   
   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;

      /* loop through real procs - these are sorted (row_list is sorted also)*/
      for (i=1; i < current_num_elmts; i++)
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
         }
      }
   }
     
   /* calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = jx_CTAlloc(JX_Int, num_real_procs + 1);
   ex_contact_vec_starts[0] = -1;

   for (i=0; i < num_real_procs; i++)
   {
      storage += 1 + 2*  num_rows_per_proc[i];
      ex_contact_vec_starts[i+1] = -storage-1; /* need negative for next loop */
   }      

   void_contact_buf = jx_MAlloc(storage*obj_size_bytes);
   index_ptr = void_contact_buf; /* step through with this index */

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf_d contains #rows, row #, data, etc. */
      
   /* un-sort real_proc_id  - we want to access data arrays in order */

   us_real_proc_id =  jx_CTAlloc(JX_Int, current_num_elmts);
   for (i=0; i < current_num_elmts; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   jx_TFree(real_proc_id);

   prev_id = -1;
   for (i=0; i < current_num_elmts; i++)
   {
      proc_id = us_real_proc_id[i];
      /* can't use row list[i] - you loose the negative signs that differentiate
         add/set values */
      row = off_proc_i[i];
      /* find position of this processor */
      indx = jx_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in =  ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in*obj_size_bytes);

      /* first time for this processor - add the number of rows to the buffer */
      if (in < 0)
      {
         in = -in - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in*obj_size_bytes);

         tmp_int =  num_rows_per_proc[indx];
         memcpy( index_ptr, &tmp_int, int_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in++;
      }
      /* add row # */   
      memcpy( index_ptr, &row, int_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;

      /* add value */
      tmp_complex = off_proc_data[i];
      memcpy( index_ptr, &tmp_complex, complex_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;
      
      /* increment the indexes to keep track of where we are - fix later */
      ex_contact_vec_starts[indx] = in;
   }
   
   /* some clean up */
 
   jx_TFree(response_buf);
   jx_TFree(response_buf_starts);

   jx_TFree(us_real_proc_id);
   jx_TFree(orig_order);
   jx_TFree(row_list);
   jx_TFree(num_rows_per_proc);

   for (i=num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] =   ex_contact_vec_starts[i-1];
   }

   ex_contact_vec_starts[0] = 0;

   /* now send the data */

   /***********************************/
   /* now get the info in send_proc_obj_d */

   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */

   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id = NULL; /* don't care who sent it to us */
   send_proc_obj.vec_starts =
      jx_CTAlloc(JX_Int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements =
      jx_MAlloc(obj_size_bytes*send_proc_obj.element_storage_length);

   response_obj2.fill_response = jx_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   jx_DataExchangeList(num_real_procs, ex_contact_procs, 
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 5, 
                          comm,  (void **) &response_buf, &response_buf_starts);

   /***********************************/

   jx_TFree(response_buf);
   jx_TFree(response_buf_starts);

   jx_TFree(ex_contact_procs);
   jx_TFree(void_contact_buf);
   jx_TFree(ex_contact_vec_starts);

   /* Now we can unpack the send_proc_objects and either set or add to the
      vector data */

   num_recvs = send_proc_obj.length; 

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;
   
   vector_data = jx_VectorData(jx_ParVectorLocalVector(par_vector));
   first_index =  jx_ParVectorFirstIndex(par_vector);

   for (i=0; i < num_recvs; i++)
   {
      indx = recv_starts[i];

      /* get the number of rows for  this recv */
      memcpy( &row_count, recv_data_ptr, int_size);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;

      for (j=0; j < row_count; j++) /* for each row: unpack info */
      {
         /* row # */
         memcpy( &row, recv_data_ptr, int_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* value */
         memcpy( &value, recv_data_ptr, complex_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         k = row - first_index - global_first_row;
         vector_data[k] += value;
      }
   }
   
   jx_TFree(send_proc_obj.v_elements);
   jx_TFree(send_proc_obj.vec_starts);
 
   return jx_error_flag;
}

#endif

JX_Int
jx_IJVectorCreateAssumedPartition( jx_IJVector *vector )
{
   JX_Int global_num, global_first_row;
   JX_Int myid;
   JX_Int start=0, end=0;
   JX_Int *partitioning = jx_IJVectorPartitioning(vector);
   MPI_Comm   comm;
   jx_IJAssumedPart *apart;

   global_num = jx_IJVectorGlobalNumRows(vector);
   global_first_row = jx_IJVectorGlobalFirstRow(vector);
   comm = jx_ParVectorComm(vector);

   /* find out my actualy range of rows */
   start =  partitioning[0];
   end = partitioning[1]-1;

   jx_MPI_Comm_rank(comm, &myid);

   /* allocate space */
   apart = jx_CTAlloc(jx_IJAssumedPart, 1);

  /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   jx_GetAssumedPartitionRowRange(comm, myid, global_first_row, 
				global_num, &(apart->row_start), &(apart->row_end));

  /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
  /*room for 10 owners of the assumed partition*/ 
   apart->storage_length = 10; /*need to be >=1 */ 
   apart->proc_list = jx_TAlloc(JX_Int, apart->storage_length);
   apart->row_start_list = jx_TAlloc(JX_Int, apart->storage_length);
   apart->row_end_list = jx_TAlloc(JX_Int, apart->storage_length);

  /* now we want to reconcile our actual partition with the assumed partition */
   jx_LocateAssummedPartition(comm, start, end, global_first_row, global_num, apart, myid);

  /* this partition will be saved in the vector data structure until the vector is destroyed */
   jx_IJVectorAssumedPart(vector) = apart;

   return jx_error_flag;
}

JX_Int
jx_FindProc( JX_Int *list, JX_Int value, JX_Int list_length )
{
   JX_Int low, high, m;

   low = 0;
   high = list_length;
   if (value >= list[high] || value < list[low])
      return -1;
   else
   {
      while (low+1 < high)
      {
         m = (low + high) / 2;
         if (value < list[m])
         {
            high = m;
         }
         else if (value >= list[m])
         {
            low = m;
         }
      }
      return low;
   }
}

JX_Int
JX_IJVectorGetValues( JX_IJVector  vector,
                      JX_Int          nvalues,
                      const JX_Int   *indices,
                      JX_Real      *values )
{
   jx_IJVector *vec = (jx_IJVector *) vector;
   if (nvalues == 0) return jx_error_flag;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (nvalues < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   if (!values)
   {
      jx_error_in_arg(4);
      return jx_error_flag;
   }
   if (jx_IJVectorObjectType(vec) == JX_PARCSR)
   {
      return(jx_IJVectorGetValuesPar(vec, nvalues, indices, values));
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
jx_IJVectorGetValuesPar( jx_IJVector *vector,
                         JX_Int          num_values,
                         const JX_Int   *indices,
                         JX_Real      *values )
{
   JX_Int my_id;
   JX_Int i, j, vec_start, vec_stop;
   JX_Real *data;
   JX_Int ierr = 0;
   JX_Int *IJpartitioning = jx_IJVectorPartitioning(vector);
   jx_ParVector *par_vector = jx_IJVectorObject(vector);
   MPI_Comm comm = jx_IJVectorComm(vector);
   jx_Vector *local_vector;
   JX_Int print_level = jx_IJVectorPrintLevel(vector);

   /* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return 0;

   jx_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         jx_printf("par_vector == NULL -- ");
         jx_printf("jx_IJVectorGetValuesPar\n");
         jx_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   local_vector = jx_ParVectorLocalVector(par_vector);
   if (!IJpartitioning)
   {
      if (print_level)
      {
         jx_printf("IJpartitioning == NULL -- ");
         jx_printf("jx_IJVectorGetValuesPar\n");
         jx_printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (!local_vector)
   {
      if (print_level)
      {
         jx_printf("local_vector == NULL -- ");
         jx_printf("jx_IJVectorGetValuesPar\n");
         jx_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

#ifdef JX_NO_GLOBAL_PARTITION
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1];
#else
   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1];
#endif

   if (vec_start > vec_stop) 
   {
      if (print_level)
      {
         jx_printf("vec_start > vec_stop -- ");
         jx_printf("jx_IJVectorGetValuesPar\n");
         jx_printf("**** This vector partitioning should not occur ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   /* Determine whether indices points to local indices only, and if not, let
      user know of catastrophe and exit.  If indices == NULL, assume that
      num_values components are to be retrieved from block starting at
      vec_start */

   if (indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
         ierr += (indices[i] <  vec_start);
         ierr += (indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      if (print_level)
      {
         jx_printf("indices beyond local range -- ");
         jx_printf("jx_IJVectorGetValuesPar\n");
         jx_printf("**** Indices specified are unusable ****\n");
      }
      jx_error_in_arg(3);
      return jx_error_flag;
   }

   data = jx_VectorData(local_vector);

   if (indices)
   {
#if JX_USING_OPENMP
#pragma omp parallel for private(i,j) schedule(static)
#endif
      for (j = 0; j < num_values; j++)
      {
         i = indices[j] - vec_start;
         values[j] = data[i];
      }
   }
   else
   {
     if (num_values > (vec_stop-vec_start))
     {
        jx_error_in_arg(2);
        return jx_error_flag;
     }
#if JX_USING_OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
      for (j = 0; j < num_values; j++)
         values[j] = data[j];
   }

   return jx_error_flag;
}

JX_Int
JX_IJVectorAddToValues( JX_IJVector    vector,
                        JX_Int            nvalues,
                        const JX_Int     *indices,
                        const JX_Real  *values )
{
   jx_IJVector *vec = (jx_IJVector *) vector;
   if (nvalues == 0) return jx_error_flag;
   if (!vec)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (nvalues < 0)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   if (!values)
   {
      jx_error_in_arg(4);
      return jx_error_flag;
   }
   if ( jx_IJVectorObjectType(vec) == JX_PARCSR )
   {
      return( jx_IJVectorAddToValuesPar(vec, nvalues, indices, values) );
   }
   else
   {
      jx_error_in_arg(1);
   }

   return jx_error_flag;
}

JX_Int
JX_IJVectorRead( const char  *filename,
                 MPI_Comm     comm,
                 JX_Int          type,
                 JX_IJVector *vector_ptr )
{
   JX_IJVector  vector;
   JX_Int       jlower, jupper, j;
   JX_Real    value;
   JX_Int       myid, ret;
   char      new_filename[255];
   FILE     *file;

   jx_MPI_Comm_rank(comm, &myid);
   jx_sprintf(new_filename,"%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   jx_fscanf(file, "%d %d", &jlower, &jupper);
   JX_IJVectorCreate(comm, jlower, jupper, &vector);
   JX_IJVectorSetObjectType(vector, type);
   JX_IJVectorInitialize(vector);
   /* It is important to ensure that whitespace follows the index value to help
    * catch mistakes in the input file.  This is done with %*[ \t].  Using a
    * space here causes an input line with a single decimal value on it to be
    * read as if it were an integer followed by a decimal value. */
   while ( (ret = jx_fscanf(file, "%d%*[ \t]%le", &j, &value)) != EOF )
   {
      if (ret != 2)
      {
         jx_error_w_msg(JX_ERROR_GENERIC, "Error in IJ vector input file.");
         return jx_error_flag;
      }
      if (j < jlower || j > jupper)
	 JX_IJVectorAddToValues(vector, 1, &j, &value);
      else
	 JX_IJVectorSetValues(vector, 1, &j, &value);
   }
   JX_IJVectorAssemble(vector);
   fclose(file);
  *vector_ptr = vector;

   return jx_error_flag;
}

JX_Int
JX_IJVectorPrint( JX_IJVector vector, const char *filename )
{
   MPI_Comm  comm;
   JX_Int      *partitioning;
   JX_Int       jlower, jupper, j;
   JX_Real    value;
   JX_Int       myid;
   char      new_filename[255];
   FILE     *file;

   if (!vector)
   {
      jx_error_in_arg(1);
      return jx_error_flag;
   } 
   comm = jx_IJVectorComm(vector);
   jx_MPI_Comm_rank(comm, &myid);
   jx_sprintf(new_filename,"%s.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      jx_error_in_arg(2);
      return jx_error_flag;
   }
   partitioning = jx_IJVectorPartitioning(vector);
#ifdef JX_NO_GLOBAL_PARTITION
   jlower = partitioning[0];
   jupper = partitioning[1] - 1;
#else
   jlower = partitioning[myid];
   jupper = partitioning[myid+1] - 1;
#endif
   jx_fprintf(file, "%d %d\n", jlower, jupper);
   for (j = jlower; j <= jupper; j++)
   {
      JX_IJVectorGetValues(vector, 1, &j, &value);
      jx_fprintf(file, "%d %.14e\n", j, value);
   }
   fclose(file);

   return jx_error_flag;
}

JX_Int
jx_IJVectorAddToValuesPar( jx_IJVector   *vector,
                           JX_Int            num_values,
                           const JX_Int     *indices,
                           const JX_Real  *values )
{
   JX_Int my_id;
   JX_Int i, j, vec_start, vec_stop;
   JX_Real *data;
   JX_Int print_level = jx_IJVectorPrintLevel(vector);
   JX_Int *IJpartitioning = jx_IJVectorPartitioning(vector);
   jx_ParVector *par_vector = jx_IJVectorObject(vector);
   jx_AuxParVector *aux_vector = jx_IJVectorTranslator(vector);
   MPI_Comm comm = jx_IJVectorComm(vector);
   jx_Vector *local_vector;

   /* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return 0;

   jx_MPI_Comm_rank(comm, &my_id);

   /* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
      let user know of catastrophe and exit */

   if (!par_vector)
   {
      if (print_level)
      {
         jx_printf("par_vector == NULL -- ");
         jx_printf("jx_IJVectorAddToValuesPar\n");
         jx_printf("**** Vector storage is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   local_vector = jx_ParVectorLocalVector(par_vector);
   if (!IJpartitioning)
   {
      if (print_level)
      {
         jx_printf("IJpartitioning == NULL -- ");
         jx_printf("jx_IJVectorAddToValuesPar\n");
         jx_printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }
   if (!local_vector)
   {
      if (print_level)
      {
         jx_printf("local_vector == NULL -- ");
         jx_printf("jx_IJVectorAddToValuesPar\n");
         jx_printf("**** Vector local data is either unallocated or orphaned ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

#ifdef JX_NO_GLOBAL_PARTITION
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1]-1;
#else
   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1]-1;
#endif

   if (vec_start > vec_stop) 
   {
      if (print_level)
      {
         jx_printf("vec_start > vec_stop -- ");
         jx_printf("jx_IJVectorAddToValuesPar\n");
         jx_printf("**** This vector partitioning should not occur ****\n");
      }
      jx_error_in_arg(1);
      return jx_error_flag;
   }

   data = jx_VectorData(local_vector);

   if (indices)
   {
      JX_Int current_num_elmts = jx_AuxParVectorCurrentNumElmts(aux_vector);
      JX_Int max_off_proc_elmts = jx_AuxParVectorMaxOffProcElmts(aux_vector);
      JX_Int *off_proc_i = jx_AuxParVectorOffProcI(aux_vector);
      JX_Real *off_proc_data = jx_AuxParVectorOffProcData(aux_vector);

      for (j = 0; j < num_values; j++)
      {
	 i = indices[j];
	 if (i < vec_start || i > vec_stop)
         {
            /* if elements outside processor boundaries, store in off processor
               stash */
	    if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               jx_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;
               jx_AuxParVectorOffProcI(aux_vector) = jx_CTAlloc(JX_Int,max_off_proc_elmts);
               jx_AuxParVectorOffProcData(aux_vector) = jx_CTAlloc(JX_Real,max_off_proc_elmts);
               off_proc_i = jx_AuxParVectorOffProcI(aux_vector);
               off_proc_data = jx_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = jx_TReAlloc(off_proc_i,JX_Int,max_off_proc_elmts);
               off_proc_data = jx_TReAlloc(off_proc_data,JX_Real,max_off_proc_elmts);
               jx_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;
               jx_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               jx_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = i;
            off_proc_data[current_num_elmts++] = values[j];
            jx_AuxParVectorCurrentNumElmts(aux_vector)=current_num_elmts;
         }
         else /* local values are added to the vector */
         {
            i -= vec_start;
            data[i] += values[j];
         }
      } 
   }
   else 
   {
      if (num_values > vec_stop - vec_start + 1)
      {
         if (print_level)
         {
            jx_printf("Warning! Indices beyond local range  not identified!\n ");
            jx_printf("Off processor values have been ignored!\n");
         }
	 num_values = vec_stop - vec_start +1;
      }
#if JX_USING_OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
      for (j = 0; j < num_values; j++)
         data[j] += values[j];
   } 
  
   return jx_error_flag;
}
