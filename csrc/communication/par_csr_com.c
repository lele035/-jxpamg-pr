//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_csr_com.c -- communication operations for parallel matrices and vectors.
 *  Date: 2011/09/07
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_MatvecCommPkgCreate
 * \brief Generates the comm_pkg for A. 
 *        If no row and / or column partitioning is given, 
 *        the routine determines them with MPE_Decomp1d.
 * \date 2011/09/07
 */
JX_Int
jx_MatvecCommPkgCreate (jx_ParCSRMatrix *par_A)
{
   JX_Int			 num_sends;
   JX_Int			*send_procs;
   JX_Int			*send_map_starts;
   JX_Int			*send_map_elmts;
 
   JX_Int			 num_recvs;
   JX_Int			*recv_procs;
   JX_Int			*recv_vec_starts;
   
   MPI_Comm              comm = jx_ParCSRMatrixComm(par_A);
   jx_ParCSRCommPkg	*comm_pkg;

   JX_Int   first_col_diag = jx_ParCSRMatrixFirstColDiag(par_A);
   JX_Int  *col_map_offd   = jx_ParCSRMatrixColMapOffd(par_A);
   JX_Int	 num_cols_offd  = jx_CSRMatrixNumCols(jx_ParCSRMatrixOffd(par_A));


#if JX_NO_GLOBAL_PARTITION

   JX_Int  row_start = 0, row_end = 0;
   JX_Int  col_start = 0, col_end = 0;
   JX_Int  global_num_cols;
   
   jx_IJAssumedPart  *apart;
   
   /*  get parcsr_A information */
   jx_ParCSRMatrixGetLocalRange( par_A, &row_start, &row_end, &col_start, &col_end );
   global_num_cols = jx_ParCSRMatrixGlobalNumCols(par_A); 

   /* Create the assumed partition */
   if (jx_ParCSRMatrixAssumedPartition(par_A) == NULL)
   {
      jx_ParCSRMatrixCreateAssumedPartition(par_A);
   }
   apart = jx_ParCSRMatrixAssumedPartition(par_A);
   
  /*-----------------------------------------------------------
   * get commpkg info information 
   *----------------------------------------------------------*/

   jx_NewCommPkgCreate_core( comm, col_map_offd, first_col_diag, 
                             col_start, col_end, 
                             num_cols_offd, global_num_cols,
                             &num_recvs, &recv_procs, &recv_vec_starts,
                             &num_sends, &send_procs, &send_map_starts, 
                             &send_map_elmts, apart );
   
#else
   
   JX_Int *col_starts    = jx_ParCSRMatrixColStarts(par_A);
   JX_Int	num_cols_diag = jx_CSRMatrixNumCols(jx_ParCSRMatrixDiag(par_A));
  

   jx_MatvecCommPkgCreate_core( comm, col_map_offd, first_col_diag, col_starts,
                                num_cols_diag, num_cols_offd,
                                first_col_diag, col_map_offd, 1,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, &send_map_elmts );
#endif


  /*-----------------------------------------------------------
   *  setup commpkg
   *----------------------------------------------------------*/

   comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);

   jx_ParCSRCommPkgComm(comm_pkg) = comm;

   jx_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs;
   jx_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs;
   jx_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   jx_ParCSRCommPkgNumSends(comm_pkg)      = num_sends;
   jx_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs;
   jx_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   jx_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts;

   jx_ParCSRMatrixCommPkg(par_A) = comm_pkg;

   return jx_error_flag;
}


/*!
 * \fn jx_ParCSRCommHandle *jx_ParCSRCommHandleCreate
 * \brief Sets up a communication handle, posts receives and initiates sends. 
 *        It always requires num_sends, num_recvs, recv_procs and send_procs 
 *        to be set in comm_pkg.
 * \note There are different options for job:
 * job = 1 : is used to initialize communication exchange for the parts
 *           of vector needed to perform a Matvec,  it requires send_data 
 *           and recv_data to be doubles, recv_vec_starts and 
 *           send_map_starts need to be set in comm_pkg.
 * job = 2 : is used to initialize communication exchange for the parts
 *           of vector needed to perform a MatvecT,  it requires send_data 
 *           and recv_data to be doubles, recv_vec_starts and 
 *           send_map_starts need to be set in comm_pkg.
 * job = 11: similar to job = 1, but exchanges data of type JX_Int (not JX_Real),
 *           requires send_data and recv_data to be ints
 *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
 * job = 12: similar to job = 1, but exchanges data of type JX_Int (not JX_Real),
 *           requires send_data and recv_data to be ints
 *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
 * [default:  ignores send_data and recv_data, requires send_mpi_types
 *           and recv_mpi_types to be set in comm_pkg.
 *           datatypes need to point to absolute
 *           addresses, e.g. generated using jx_MPI_Address.]  
 * \date 2011/09/07
 */
jx_ParCSRCommHandle *
jx_ParCSRCommHandleCreate( JX_Int 	             job,
			   jx_ParCSRCommPkg *comm_pkg,
                           void             *send_data, 
                           void             *recv_data )
{
   JX_Int                  num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   JX_Int                  num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = jx_ParCSRCommPkgComm(comm_pkg);

   jx_ParCSRCommHandle *comm_handle;
   JX_Int                  num_requests;
   MPI_Request         *requests;

   JX_Int                  i, j;
   JX_Int			my_id, num_procs;
   JX_Int			ip, vec_start, vec_len;

   num_requests = num_sends + num_recvs;
   requests = jx_CTAlloc(MPI_Request, num_requests);
 
   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

   j = 0;
   switch (job)
   {
      case  1:
      {
         JX_Real *d_send_data = (JX_Real *) send_data;
         JX_Real *d_recv_data = (JX_Real *) recv_data;
         for (i = 0; i < num_recvs; i ++)
         {
            ip        = jx_ParCSRCommPkgRecvProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
            vec_len   = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1) - vec_start;
            jx_MPI_Irecv(&d_recv_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i ++)
         {
            ip        = jx_ParCSRCommPkgSendProc(comm_pkg, i);
	    vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
	    vec_len   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start;
   	    jx_MPI_Isend(&d_send_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
      }
      break;
      
      
      case  2:
      {
         JX_Real *d_send_data = (JX_Real *) send_data;
         JX_Real *d_recv_data = (JX_Real *) recv_data;
         for (i = 0; i < num_sends; i ++)
         {
            ip        = jx_ParCSRCommPkgSendProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start;
            jx_MPI_Irecv(&d_recv_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i ++)
         {
            ip        = jx_ParCSRCommPkgRecvProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
            vec_len   = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1) - vec_start;
            jx_MPI_Isend(&d_send_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[j++]);
         }
      }
      break;
      
      
      case  11:
      {
         JX_Int *i_send_data = (JX_Int *) send_data;
         JX_Int *i_recv_data = (JX_Int *) recv_data;
         for (i = 0; i < num_recvs; i ++)
         {
            ip        = jx_ParCSRCommPkgRecvProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
            vec_len   = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1) - vec_start;
            jx_MPI_Irecv(&i_recv_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i ++)
         {
            ip        = jx_ParCSRCommPkgSendProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start;
            jx_MPI_Isend(&i_send_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
      }
      break;
      
      
      case  12:
      {
         JX_Int *i_send_data = (JX_Int *) send_data;
         JX_Int *i_recv_data = (JX_Int *) recv_data;
         for (i = 0; i < num_sends; i ++)
         {
            ip        = jx_ParCSRCommPkgSendProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len   = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - vec_start;
            jx_MPI_Irecv(&i_recv_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i ++)
         {
            ip        = jx_ParCSRCommPkgRecvProc(comm_pkg, i); 
            vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
            vec_len   = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1) - vec_start;
            jx_MPI_Isend(&i_send_data[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }
 
 
  /*--------------------------------------------------------------------
   * set up comm_handle and return
   *--------------------------------------------------------------------*/

   comm_handle = jx_CTAlloc(jx_ParCSRCommHandle, 1);

   jx_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
   jx_ParCSRCommHandleSendData(comm_handle)    = send_data;
   jx_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
   jx_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   jx_ParCSRCommHandleRequests(comm_handle)    = requests;

   return ( comm_handle );
}


/*!
 * \fn JX_Int jx_ParCSRCommHandleDestroy 
 * \date 2011/09/05
 */
JX_Int
jx_ParCSRCommHandleDestroy( jx_ParCSRCommHandle *comm_handle )
{
   MPI_Status   *status   = NULL;
   MPI_Request  *requests = jx_ParCSRCommHandleRequests(comm_handle);
   JX_Int num_requests = jx_ParCSRCommHandleNumRequests(comm_handle);

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
 * \fn void jx_MatvecCommPkgCreate_core
 * \brief This routine does all the communications and computations for
 *            jx_MatCommPkgCreate(jx_ParCSRMatrix *par_A)
 *        and  
 *            jx_BoolMatCommPkgCreate(jx_ParCSRBooleanMatrix *par_A)
 *        To support both data types, it has hardly any data structures other than JX_Int *.
 * \date 2011/09/05
 */
void
jx_MatvecCommPkgCreate_core(  /* input args: */
                              MPI_Comm comm,      JX_Int *col_map_offd, 
                              JX_Int first_col_diag, JX_Int *col_starts,
                              JX_Int num_cols_diag,  JX_Int  num_cols_offd,
                              JX_Int firstColDiag,   JX_Int *colMapOffd,
                              /* = 1 for a matrix with floating-point data, = 0 for Boolean matrix */
                              JX_Int data,  
                              /* pointers to output args: */
                              JX_Int  *p_num_recvs,  JX_Int **p_recv_procs,  JX_Int **p_recv_vec_starts,
                              JX_Int  *p_num_sends,  JX_Int **p_send_procs,  JX_Int **p_send_map_starts,
                              JX_Int **p_send_map_elmts )
{
   JX_Int	i, j;
   JX_Int	num_procs, my_id, proc_num, num_elmts;
   JX_Int	local_info, offd_col;
   JX_Int	*proc_mark, *proc_add, *tmp, *recv_buf, *displs, *info;
   
   /* outputs: */
   JX_Int  num_recvs, * recv_procs, * recv_vec_starts;
   JX_Int  num_sends, * send_procs, * send_map_starts, * send_map_elmts;
   JX_Int  ip, vec_start, vec_len, num_requests;

   MPI_Request *requests = NULL;
   MPI_Status  *status   = NULL; 

   jx_MPI_Comm_size(comm, &num_procs);  
   jx_MPI_Comm_rank(comm, &my_id);

   proc_mark = jx_CTAlloc(JX_Int, num_procs);
   proc_add  = jx_CTAlloc(JX_Int, num_procs);
   info      = jx_CTAlloc(JX_Int, num_procs);

  /* ----------------------------------------------------------------------
   * determine which processors to receive from (set proc_mark) and num_recvs,
   * at the end of the loop proc_mark[i] contains the number of elements to be
   * received from Proc. i
   * ---------------------------------------------------------------------*/

   for (i = 0; i < num_procs; i ++)
   {
      proc_add[i] = 0;
   }

   proc_num = 0;
   
   if (num_cols_offd) offd_col = col_map_offd[0];
   
   num_recvs = 0;
   j = 0;
   for (i = 0; i < num_cols_offd; i ++)
   {
      if (num_cols_diag) 
      {
         proc_num = jx_min(num_procs-1, offd_col / num_cols_diag);
      }
	
      while (col_starts[proc_num] > offd_col)
      {
         proc_num = proc_num - 1;
      }
      
      while (col_starts[proc_num+1] - 1 < offd_col)
      {
         proc_num = proc_num + 1;
      }
      
      proc_mark[num_recvs] = proc_num;
      j = i;
      while (col_starts[proc_num+1] > offd_col)
      {
         proc_add[num_recvs] ++;
         if (j < num_cols_offd - 1) 
         {
            j ++;
            offd_col = col_map_offd[j];
         }
         else
         {
            j ++;
            offd_col = col_starts[num_procs];
         }
      }
      num_recvs ++;
	
      if (j < num_cols_offd) 
      {
         i = j - 1;
      }
      else 
      {
         i = j;
      }
   }

   local_info = 2*num_recvs;
			
   jx_MPI_Allgather(&local_info, 1, JX_MPI_INT, info, 1, JX_MPI_INT, comm); 

  /* --------------------------------------------------------------------------
   * generate information to be sent: tmp contains for each recv_proc:
   * id of recv_procs, number of elements to be received for this processor,
   * indices of elements (in this order)
   *---------------------------------------------------------------------------*/

   displs = jx_CTAlloc(JX_Int, num_procs + 1);
   displs[0] = 0;
   for (i = 1; i < num_procs + 1; i ++)
   {
      displs[i] = displs[i-1] + info[i-1]; 
   }
   recv_buf = jx_CTAlloc(JX_Int, displs[num_procs]); 

   recv_procs = NULL;
   tmp = NULL;
   if (num_recvs)
   {
      recv_procs = jx_CTAlloc(JX_Int, num_recvs);
      tmp        = jx_CTAlloc(JX_Int, local_info);
   }
   recv_vec_starts = jx_CTAlloc(JX_Int, num_recvs + 1);


   j = 0;
   if (num_recvs) recv_vec_starts[0] = 0;
   for (i = 0; i < num_recvs; i ++)
   {
      num_elmts = proc_add[i];
      recv_procs[i] = proc_mark[i];
      recv_vec_starts[i+1] = recv_vec_starts[i] + num_elmts;
      tmp[j++] = proc_mark[i];
      tmp[j++] = num_elmts;
   }

   jx_MPI_Allgatherv(tmp, local_info, JX_MPI_INT, recv_buf, info, displs, JX_MPI_INT, comm);
	

  /*----------------------------------------------------------------------
   * determine num_sends and number of elements to be sent
   *---------------------------------------------------------------------*/

   num_sends = 0;
   num_elmts = 0;
   proc_add[0] = 0;
   for (i = 0; i < num_procs; i ++)
   {
      j = displs[i];
      while (j < displs[i+1])
      {
	 if (recv_buf[j++] == my_id)
	 {
	    proc_mark[num_sends] = i;
	    num_sends ++;
	    proc_add[num_sends] = proc_add[num_sends-1] + recv_buf[j];
	    break;
	 }
	 j ++;
      }	
   }
	
		
  /* ----------------------------------------------------------------------
   * determine send_procs and actual elements to be send (in send_map_elmts)
   * and send_map_starts whose i-th entry points to the beginning of the 
   * elements to be send to proc. i
   * ---------------------------------------------------------------------*/

   send_procs = NULL;
   send_map_elmts = NULL;

   if (num_sends)
   {
      send_procs = jx_CTAlloc(JX_Int, num_sends);
      send_map_elmts = jx_CTAlloc(JX_Int, proc_add[num_sends]);
   }
   send_map_starts = jx_CTAlloc(JX_Int, num_sends+1);
   num_requests = num_recvs+num_sends;
   if (num_requests)
   {
      requests = jx_CTAlloc(MPI_Request, num_requests);
      status   = jx_CTAlloc(MPI_Status, num_requests);
   }

   if (num_sends) send_map_starts[0] = 0;
   for (i = 0; i < num_sends; i ++)
   {
      send_map_starts[i+1] = proc_add[i+1];
      send_procs[i] = proc_mark[i];
   }

   j = 0;
   for (i = 0; i < num_sends; i ++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i+1] - vec_start;
      ip = send_procs[i];
      jx_MPI_Irecv(&send_map_elmts[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_recvs; i ++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i+1] - vec_start;
      ip = recv_procs[i];
      jx_MPI_Isend(&col_map_offd[vec_start], vec_len, JX_MPI_INT, ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      jx_MPI_Waitall(num_requests, requests, status);
      jx_TFree(requests);
      jx_TFree(status);
   }

   if (num_sends)
   {
      for (i = 0; i < send_map_starts[num_sends]; i ++)
      {
         send_map_elmts[i] -= first_col_diag;
      }
   }

   jx_TFree(proc_add);
   jx_TFree(proc_mark); 
   jx_TFree(tmp);
   jx_TFree(recv_buf);
   jx_TFree(displs);
   jx_TFree(info);
 
   /* finish up with the hand-coded call-by-reference... */
   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_procs;
   *p_send_map_starts = send_map_starts;
   *p_send_map_elmts = send_map_elmts;
}


/*!
 * \fn JX_Int jx_MatvecCommPkgDestroy
 * \date 2011/09/05
 */
JX_Int
jx_MatvecCommPkgDestroy( jx_ParCSRCommPkg *comm_pkg )
{
   if (jx_ParCSRCommPkgNumSends(comm_pkg))
   {
      jx_TFree(jx_ParCSRCommPkgSendProcs(comm_pkg));
      jx_TFree(jx_ParCSRCommPkgSendMapElmts(comm_pkg));
   }
   jx_TFree(jx_ParCSRCommPkgSendMapStarts(comm_pkg));

   if (jx_ParCSRCommPkgNumRecvs(comm_pkg))
   {
      jx_TFree(jx_ParCSRCommPkgRecvProcs(comm_pkg));
   }
   jx_TFree(jx_ParCSRCommPkgRecvVecStarts(comm_pkg));

   jx_TFree(comm_pkg);

   return jx_error_flag;
}
