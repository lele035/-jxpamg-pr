//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_interp_4.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGBuildMultipass
 * \brief This routine implements Stuben's direct interpolation with multiple passes.
 * \date 2011/09/03
 */
JX_Int
jx_PAMGBuildMultipass( jx_ParCSRMatrix  *par_A,
                       JX_Int              *CF_marker,
                       jx_ParCSRMatrix  *par_S,
                       JX_Int              *num_cpts_global,
                       JX_Int               num_functions,
                       JX_Int              *dof_func,
                       JX_Int               debug_flag,
                       JX_Real            trunc_factor,
                       JX_Int               P_max_elmts,
                       JX_Int               weight_option,
                       JX_Int              *col_offd_S_to_A,
                       jx_ParCSRMatrix **P_ptr )
{
   MPI_Comm	        comm     = jx_ParCSRMatrixComm(par_A); 
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_S);
   jx_ParCSRCommHandle *comm_handle;
   jx_ParCSRCommPkg    *tmp_comm_pkg = NULL;

   jx_CSRMatrix *A_diag      = jx_ParCSRMatrixDiag(par_A);
   JX_Real       *A_diag_data = jx_CSRMatrixData(A_diag);
   JX_Int          *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Int          *A_diag_j    = jx_CSRMatrixJ(A_diag);

   jx_CSRMatrix *A_offd          = jx_ParCSRMatrixOffd(par_A);
   JX_Real       *A_offd_data     = NULL;
   JX_Int          *A_offd_i        = jx_CSRMatrixI(A_offd);
   JX_Int          *A_offd_j        = NULL;
   JX_Int		*col_map_offd_A  = jx_ParCSRMatrixColMapOffd(par_A);
   JX_Int           num_cols_offd_A = jx_CSRMatrixNumCols(A_offd);

   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(par_S);
   JX_Int          *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int          *S_diag_j = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix *S_offd          = jx_ParCSRMatrixOffd(par_S);
   JX_Int          *S_offd_i        = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j        = NULL;
   JX_Int		*col_map_offd_S  = jx_ParCSRMatrixColMapOffd(par_S);
   JX_Int		 num_cols_offd_S = jx_CSRMatrixNumCols(S_offd);
   JX_Int		*col_map_offd    = NULL;
   JX_Int		 num_cols_offd;

   jx_ParCSRMatrix *par_P;
   jx_CSRMatrix    *P_diag;
   JX_Real          *P_diag_data;
   JX_Int             *P_diag_i; /* at first counter of nonzero cols for each row,
				finally will be pointer to start of row */
   JX_Int             *P_diag_j;

   jx_CSRMatrix    *P_offd;
   JX_Real          *P_offd_data = NULL;
   JX_Int             *P_offd_i; /* at first counter of nonzero cols for each row,
		                 finally will be pointer to start of row */
   JX_Int             *P_offd_j = NULL;

   JX_Int              num_sends = 0;
   JX_Int             *int_buf_data = NULL;
   JX_Int             *send_map_start = NULL;
   JX_Int             *send_map_elmt = NULL;
   JX_Int             *send_procs = NULL;
   JX_Int              num_recvs = 0;
   JX_Int             *recv_vec_start = NULL;
   JX_Int             *recv_procs = NULL;
   JX_Int             *new_recv_vec_start = NULL;
   JX_Int            **Pext_send_map_start = NULL;
   JX_Int            **Pext_recv_vec_start = NULL;
   JX_Int             *Pext_start = NULL;
   JX_Int             *P_ncols = NULL;
   
   JX_Int             *CF_marker_offd = NULL;
   JX_Int             *dof_func_offd = NULL;
   JX_Int             *P_marker;
   JX_Int             *P_marker_offd = NULL;
   JX_Int             *C_array = NULL;
   JX_Int             *C_array_offd = NULL;
   JX_Int             *pass_array = NULL;   /* contains points ordered according to pass */
   JX_Int             *pass_pointer = NULL; /* pass_pointer[j] contains pointer to first
				            point of pass j contained in pass_array */
   JX_Int             *P_diag_start;
   JX_Int             *P_offd_start = NULL;
   JX_Int            **P_diag_pass;
   JX_Int            **P_offd_pass = NULL;
   JX_Int            **Pext_pass = NULL;
   JX_Int            **new_elmts = NULL;   /* new neighbors generated in each pass */
   JX_Int             *new_counter = NULL; /* contains no. of new neighbors for each pass */
   JX_Int             *loc = NULL;         /* contains locations for new neighbor 
			                   connections in int_o_buffer to avoid searching */
   JX_Int             *Pext_i = NULL;      /* contains P_diag_i and P_offd_i info for nonzero
				           cols of off proc neighbors */
   JX_Int             *Pext_send_buffer = NULL; /* used to collect global nonzero
				                col ids in P_diag for send_map_elmts */

   JX_Int             *map_S_to_new = NULL;
   JX_Int             *map_A_to_S = NULL;
   JX_Int             *new_col_map_offd = NULL;
   JX_Int             *col_map_offd_P = NULL;
   JX_Int             *permute = NULL;

   JX_Int              cnt;
   JX_Int              cnt_nz;
   JX_Int              total_nz;
   JX_Int              pass;
   JX_Int              num_passes;
   JX_Int              max_num_passes = 10;

   JX_Int              n_fine;
   JX_Int              n_coarse = 0;
   JX_Int              n_coarse_offd = 0;
   JX_Int              n_SF = 0;
   JX_Int              n_SF_offd = 0;

   JX_Int             *fine_to_coarse = NULL;
   JX_Int             *fine_to_coarse_offd = NULL;

   JX_Int             *assigned = NULL;
   JX_Int             *assigned_offd = NULL;

   JX_Real          *Pext_send_data = NULL;
   JX_Real          *Pext_data = NULL;

   JX_Real           sum_C, sum_N;
   JX_Real           sum_C_pos, sum_C_neg;
   JX_Real           sum_N_pos, sum_N_neg;
   JX_Real           diagonal;
   JX_Real           alfa = 1.0;
   JX_Real           beta = 1.0;
   JX_Int              j_start;
   JX_Int              j_end;

   JX_Int              i, i1;
   JX_Int              j, j1;
   JX_Int              k, k1, k2, k3;
   JX_Int              pass_array_size;
   JX_Int              global_pass_array_size;
   JX_Int              local_pass_array_size;
   JX_Int              my_id, num_procs;
   JX_Int              index, start;
   JX_Int              my_first_cpt;
   JX_Int              total_global_cpts;
   JX_Int              p_cnt;
   JX_Int              total_nz_offd;
   JX_Int              cnt_nz_offd;
   JX_Int              cnt_offd, cnt_new;
   JX_Int              no_break;
   JX_Int              not_found;
   JX_Int              Pext_send_size;
   JX_Int              Pext_recv_size;
   JX_Int              old_Pext_send_size;
   JX_Int              old_Pext_recv_size;
   JX_Int              P_offd_size = 0;
   JX_Int              local_index = -1;
   JX_Int              new_num_cols_offd = 0;
   JX_Int              num_cols_offd_P;

  /*-----------------------------------------------------------------------
   *  Access the CSR vectors for A and S. Also get size of fine grid.
   *-----------------------------------------------------------------------*/

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

#ifdef JX_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) 
   {
      total_global_cpts = num_cpts_global[1];
   }
   jx_MPI_Bcast(&total_global_cpts, 1, JX_MPI_INT, num_procs-1, comm); 
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A);
      if (!comm_pkg)
      {
         jx_MatvecCommPkgCreate(par_A);
         comm_pkg = jx_ParCSRMatrixCommPkg(par_A);
      }
      col_offd_S_to_A = NULL;
   }

   if (col_offd_S_to_A)
   {
      col_map_offd = col_map_offd_S;
      num_cols_offd = num_cols_offd_S;
   }
   else
   {
      col_map_offd = col_map_offd_A;
      num_cols_offd = num_cols_offd_A;
   }

   if (num_cols_offd_A)
   {
      A_offd_data = jx_CSRMatrixData(A_offd);
      A_offd_j    = jx_CSRMatrixJ(A_offd);
   }

   if (num_cols_offd)
   {
      S_offd_j    = jx_CSRMatrixJ(S_offd);
   }

   n_fine = jx_CSRMatrixNumRows(A_diag);

  /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

   if (n_fine) 
   {
      fine_to_coarse = jx_CTAlloc(JX_Int, n_fine);
   }

   n_coarse = 0;
   n_SF = 0;
   for (i = 0; i < n_fine; i ++)
   {
      if (CF_marker[i] == 1) 
      {
         n_coarse ++;
      }
      else if (CF_marker[i] == -3) 
      {
         n_SF ++;
      }
   }
   pass_array_size = n_fine - n_coarse - n_SF;
   if (pass_array_size) 
   {
      pass_array = jx_CTAlloc(JX_Int, pass_array_size);
   }
   pass_pointer = jx_CTAlloc(JX_Int, max_num_passes+1);
   if (n_fine) 
   {
      assigned = jx_CTAlloc(JX_Int, n_fine);
   }
   P_diag_i = jx_CTAlloc(JX_Int, n_fine + 1);
   P_offd_i = jx_CTAlloc(JX_Int, n_fine + 1);
   if (n_coarse) 
   {
      C_array = jx_CTAlloc(JX_Int, n_coarse);
   }

   if (num_cols_offd)
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      if (num_functions > 1) 
      {
         dof_func_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      }
   }

   if (num_procs > 1)
   {
      num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = jx_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_start = jx_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmt = jx_ParCSRCommPkgSendMapElmts(comm_pkg);
      num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = jx_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_start = jx_ParCSRCommPkgRecvVecStarts(comm_pkg);
      if (send_map_start[num_sends])
      {
         int_buf_data = jx_CTAlloc(JX_Int, send_map_start[num_sends]);
      }
   }

   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i+1]; j ++)
      {
	 int_buf_data[index++] = CF_marker[send_map_elmt[j]];
      }
   }
   
   if (num_procs > 1)
   {
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);
   }

   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         start = send_map_start[i];
         for (j = start; j < send_map_start[i+1]; j ++)
         {
	    int_buf_data[index++] = dof_func[send_map_elmt[j]];
	 }
      }
      
      if (num_procs > 1)
      {
         comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
         jx_ParCSRCommHandleDestroy(comm_handle);
      }
   }

   n_coarse_offd = 0;
   n_SF_offd = 0;
   for (i = 0; i < num_cols_offd; i ++)
   {
      if (CF_marker_offd[i] == 1) 
      {
         n_coarse_offd ++;
      }
      else if (CF_marker_offd[i] == -3) 
      {
         n_SF_offd ++;
      }
   }
   
   if (num_cols_offd)
   {
      assigned_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      map_S_to_new = jx_CTAlloc(JX_Int, num_cols_offd);
      fine_to_coarse_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      new_col_map_offd = jx_CTAlloc(JX_Int, n_coarse_offd);
   }

  /*-----------------------------------------------------------------------
   *  First Pass: determine the maximal size of P, and elementsPerRow[i].
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   *  Assigned points are points for which we know an interpolation
   *  formula already, and which are thus available to interpolate from.
   *  assigned[i]=0 for C points, and 1, 2, 3, ... for F points, depending
   *  in which pass their interpolation formula is determined.
   *  
   *  pass_array contains the points ordered according to its pass, i.e.
   *  |  C-points   |  points of pass 1 | points of pass 2 | ....
   * C_points are points 0 through pass_pointer[1]-1,
   * points of pass k  (0 < k < num_passes) are contained in points
   * pass_pointer[k] through pass_pointer[k+1]-1 of pass_array .
   *
   * pass_array is also used to avoid going through all points for each pass,
   * i,e. at the bginning it contains all points in descending order starting
   * with n_fine-1. Then starting from the last point, we evaluate whether
   * it is a C_point (pass 0). If it is the point is brought to the front
   * and the length of the points to be searched is shortened.  This is
   * done until the parameter cnt (which determines the first point of
   * pass_array to be searched) becomes n_fine. Then all points have been
   * assigned a pass number.
   *-----------------------------------------------------------------------*/

   cnt = 0;
   p_cnt = pass_array_size - 1;
   P_diag_i[0] = 0;
   P_offd_i[0] = 0;
   for (i = 0; i < n_fine; i ++)
   {
      if (CF_marker[i] == 1)
      {
         fine_to_coarse[i] = cnt; /* this C point is assigned index
			             coarse_counter on coarse grid,
				     and in column of P */
	 C_array[cnt++] = i;
         assigned[i] = 0;
         P_diag_i[i+1] = 1; /* one element in row i1 of P */
         P_offd_i[i+1] = 0;
      }
      else if (CF_marker[i] == -1)
      {
         pass_array[p_cnt--] = i;
         P_diag_i[i+1] = 0;
         P_offd_i[i+1] = 0;
         assigned[i] = -1;
         fine_to_coarse[i] = -1;
      }
      else
      {
         P_diag_i[i+1] = 0;
         P_offd_i[i+1] = 0;
         assigned[i] = -1;
         fine_to_coarse[i] = -1;
      }
   }

   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i+1]; j ++)
      {
	 int_buf_data[index] = fine_to_coarse[send_map_elmt[j]];
	 if (int_buf_data[index] > -1)
	 { 
	    int_buf_data[index] += my_first_cpt;
	 }
	 index ++;
      }
   }
   if (num_procs > 1)
   {
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, fine_to_coarse_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);
   }

   new_recv_vec_start = jx_CTAlloc(JX_Int, num_recvs + 1);

   if (n_coarse_offd)
   {
      C_array_offd = jx_CTAlloc(JX_Int,n_coarse_offd);
   }

   cnt = 0;
   new_recv_vec_start[0] = 0;
   for (j = 0; j < num_recvs; j ++)
   {
      for (i = recv_vec_start[j]; i < recv_vec_start[j+1]; i ++)
      {
         if (CF_marker_offd[i] == 1)
         {
	    map_S_to_new[i] = cnt;
	    C_array_offd[cnt] = i;
	    new_col_map_offd[cnt++] = fine_to_coarse_offd[i];
            assigned_offd[i] = 0; 
         }
         else
         {
            assigned_offd[i] = -1;
            map_S_to_new[i] = -1;
         }
      }
      new_recv_vec_start[j+1] = cnt;
   }

   cnt = 0;
   jx_TFree(fine_to_coarse_offd);

   if (col_offd_S_to_A)
   {
      map_A_to_S = jx_CTAlloc(JX_Int, num_cols_offd_A);
      for (i = 0; i < num_cols_offd_A; i ++)
      {
         if (cnt < num_cols_offd && col_map_offd_A[i] == col_map_offd[cnt])
         {
	    map_A_to_S[i] = cnt ++;
	 }
         else
         {
	    map_A_to_S[i] = -1;
	 }
      }
   }

  /*-----------------------------------------------------------------------
   *  Mark all local neighbors of C points as 'assigned'.
   *-----------------------------------------------------------------------*/

   pass_pointer[0] = 0;
   pass_pointer[1] = 0;
   total_nz = n_coarse;  /* accumulates total number of nonzeros in P_diag */
   total_nz_offd = 0;    /* accumulates total number of nonzeros in P_offd */

   cnt = 0;
   cnt_offd = 0;
   cnt_nz = 0;
   cnt_nz_offd = 0;
   for (i = pass_array_size - 1; i > cnt-1; i --)
   {
      i1 = pass_array[i];
      for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] == 1)
         {
            P_diag_i[i1+1] ++;
            cnt_nz ++;
            assigned[i1] = 1;
         }
      }
      
      for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
      {
         j1 = S_offd_j[j];
         if (CF_marker_offd[j1] == 1)
         {
            P_offd_i[i1+1] ++;
            cnt_nz_offd ++;
            assigned[i1] = 1;
         }
      }
      
      if (assigned[i1] == 1)
      {
         pass_array[i++] = pass_array[cnt];
         pass_array[cnt++] = i1;
      }
   }

   pass_pointer[2] = cnt;


  /*-----------------------------------------------------------------------
   *  All local neighbors are assigned, now need to exchange the boundary
   *  info for assigned strong neighbors.
   *-----------------------------------------------------------------------*/

   index = 0;
   for (i = 0; i < num_sends; i ++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i+1]; j ++)
      {
	 int_buf_data[index++] = assigned[send_map_elmt[j]];
      }
   }
   
   if (num_procs > 1)
   {
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, assigned_offd);
      jx_ParCSRCommHandleDestroy(comm_handle);
   }

  /*-----------------------------------------------------------------------
   *  Now we need to determine strong neighbors of points of pass 1, etc.
   *  we need to update assigned_offd after each pass
   *-----------------------------------------------------------------------*/

   pass = 2;
   local_pass_array_size = pass_array_size - cnt;
   jx_MPI_Allreduce(&local_pass_array_size, &global_pass_array_size, 1, JX_MPI_INT, MPI_SUM, comm);
   
   while (global_pass_array_size && pass < max_num_passes)
   {
      for (i = pass_array_size - 1; i > cnt - 1; i --)
      {
	 i1 = pass_array[i];
         no_break = 1;
         for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
	 {
	    j1 = S_diag_j[j];
	    if (assigned[j1] == pass - 1)
  	    {
               pass_array[i++] = pass_array[cnt];
               pass_array[cnt++] = i1; 
               assigned[i1] = pass;
               no_break = 0;
               break;
	    }
     	 }
     	 if (no_break)
	 {
            for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
	    {
	       j1 = S_offd_j[j];
	       if (assigned_offd[j1] == pass - 1)
  	       {
                  pass_array[i++] = pass_array[cnt];
                  pass_array[cnt++] = i1; 
                  assigned[i1] = pass;
                  break;
  	       }
	    }
     	 }
      }
      /* jx_printf("pass %d  remaining points %d \n", pass, local_pass_array_size); */

      pass ++;
      pass_pointer[pass] = cnt;

      local_pass_array_size = pass_array_size - cnt;
      jx_MPI_Allreduce(&local_pass_array_size, &global_pass_array_size, 1, JX_MPI_INT, MPI_SUM, comm);
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         start = send_map_start[i];
         for (j = start; j < send_map_start[i+1]; j ++)
         {
	    int_buf_data[index++] = assigned[send_map_elmt[j]];
	 }
      }
      if (num_procs > 1)
      {
         comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, assigned_offd);
         jx_ParCSRCommHandleDestroy(comm_handle);
      }
   }

   jx_TFree(int_buf_data);

   num_passes = pass;

   P_marker = jx_CTAlloc(JX_Int, n_coarse); /* marks points to see if they have
						   been counted */
   for (i = 0; i < n_coarse; i ++)
   {
      P_marker[i] = -1;
   }

   if (n_coarse_offd)
   {
      P_marker_offd = jx_CTAlloc(JX_Int, n_coarse_offd);
      for (i = 0; i < n_coarse_offd; i ++)
      {
         P_marker_offd[i] = -1;
      }
   }

   P_diag_pass = jx_CTAlloc(JX_Int*, num_passes);  /* P_diag_pass[i] will contain
				                   all column numbers for points of pass i */

   P_diag_pass[1] = jx_CTAlloc(JX_Int, cnt_nz);

   P_diag_start = jx_CTAlloc(JX_Int, n_fine); /* P_diag_start[i] contains
	                                      pointer to begin of column numbers in P_pass for point i,
	                                      P_diag_i[i+1] contains number of columns for point i */

   P_offd_start = jx_CTAlloc(JX_Int, n_fine);

   if (num_procs > 1)
   {
      P_offd_pass = jx_CTAlloc(JX_Int*, num_passes);

      if (cnt_nz_offd)
      {
         P_offd_pass[1] = jx_CTAlloc(JX_Int, cnt_nz_offd);
      }
      else
      {
         P_offd_pass[1] = NULL;
      }

      new_elmts = jx_CTAlloc(JX_Int*, num_passes);

      new_counter = jx_CTAlloc(JX_Int, num_passes+1);

      new_counter[0] = 0;
      new_counter[1] = n_coarse_offd;
      new_num_cols_offd = n_coarse_offd;

      new_elmts[0] = new_col_map_offd;
   }

  /*-----------------------------------------------------------------------
   *  Pass 1: now we consider points of pass 1, with strong C_neighbors,
   *-----------------------------------------------------------------------*/

   cnt_nz = 0;
   cnt_nz_offd = 0;
   for (i = pass_pointer[1]; i < pass_pointer[2]; i ++)
   {
      i1 = pass_array[i];
      P_diag_start[i1] = cnt_nz;
      P_offd_start[i1] = cnt_nz_offd;
      for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
      {
         j1 = S_diag_j[j];
	 if (CF_marker[j1] == 1)
	 {
	    P_diag_pass[1][cnt_nz++] = fine_to_coarse[j1];
	 }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
      {
         j1 = S_offd_j[j];
	 if (CF_marker_offd[j1] == 1)
	 {
	    P_offd_pass[1][cnt_nz_offd++] = map_S_to_new[j1];
	 }
      }
   }


   total_nz += cnt_nz;
   total_nz_offd += cnt_nz_offd;

   if (num_procs > 1)
   {
      tmp_comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
      Pext_send_map_start = jx_CTAlloc(JX_Int*, num_passes);
      Pext_recv_vec_start = jx_CTAlloc(JX_Int*, num_passes);
      Pext_pass = jx_CTAlloc(JX_Int*, num_passes);
      Pext_i = jx_CTAlloc(JX_Int, num_cols_offd + 1);
      
      if (num_cols_offd) 
      {
         Pext_start = jx_CTAlloc(JX_Int, num_cols_offd);
      }
      
      if (send_map_start[num_sends])
      {
         P_ncols = jx_CTAlloc(JX_Int, send_map_start[num_sends]);
      }
      
      for (i = 0; i < num_cols_offd + 1; i ++)
      {
         Pext_i[i] = 0;
      }
      
      for (i = 0; i < send_map_start[num_sends]; i ++)
      {
         P_ncols[i] = 0;
      }
   }

   old_Pext_send_size = 0;
   old_Pext_recv_size = 0;
   
   for (pass = 2; pass < num_passes; pass ++)
   {
      if (num_procs > 1)
      {
         Pext_send_map_start[pass] = jx_CTAlloc(JX_Int, num_sends + 1);
         Pext_recv_vec_start[pass] = jx_CTAlloc(JX_Int, num_recvs + 1);
         Pext_send_size = 0;
         Pext_send_map_start[pass][0] = 0;

         for (i = 0; i < num_sends; i ++)
         {
            for (j = send_map_start[i]; j < send_map_start[i+1]; j ++)
            {
               j1 = send_map_elmt[j];
	       if (assigned[j1] == pass - 1)
	       {
	          P_ncols[j] = P_diag_i[j1+1] + P_offd_i[j1+1];
	          Pext_send_size += P_ncols[j];
               }
            }
            Pext_send_map_start[pass][i+1] = Pext_send_size;
         }

         comm_handle = jx_ParCSRCommHandleCreate (11, comm_pkg, P_ncols, &Pext_i[1]);
         jx_ParCSRCommHandleDestroy(comm_handle);

         if (Pext_send_size > old_Pext_send_size)
         {
            jx_TFree(Pext_send_buffer);
            Pext_send_buffer = jx_CTAlloc(JX_Int, Pext_send_size);
         }
         old_Pext_send_size = Pext_send_size;
      }

      cnt_offd = 0;
      for (i = 0; i < num_sends; i ++)
      {
         for (j = send_map_start[i]; j < send_map_start[i+1]; j ++)
         {
            j1 = send_map_elmt[j];
	    if (assigned[j1] == pass - 1)
	    {
	       j_start = P_diag_start[j1];
	       j_end = j_start+P_diag_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  Pext_send_buffer[cnt_offd++] = my_first_cpt + P_diag_pass[pass-1][k];
	       }
	       j_start = P_offd_start[j1];
	       j_end = j_start + P_offd_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = P_offd_pass[pass-1][k];
		  k3 = 0;
		  while (k3 < pass-1)
		  {
		     if (k1 < new_counter[k3+1])
		     {
		        k2 = k1 - new_counter[k3];
		        Pext_send_buffer[cnt_offd++] = new_elmts[k3][k2];
		        break;
		     }
		     k3 ++;
		  }
	       }
            }
         }
      }
 
      if (num_procs > 1)
      {
         Pext_recv_size = 0;
         Pext_recv_vec_start[pass][0] = 0;
         cnt_offd = 0;
         for (i = 0; i < num_recvs; i ++)
         {
 	    for (j = recv_vec_start[i]; j<recv_vec_start[i+1]; j ++)
            {
	       if (assigned_offd[j] == pass - 1)
               {
                  Pext_start[j] = cnt_offd;
                  cnt_offd += Pext_i[j+1];
               }
            }
            Pext_recv_size = cnt_offd;
            Pext_recv_vec_start[pass][i+1] = Pext_recv_size;
         }

         jx_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
         jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
         jx_ParCSRCommPkgSendProcs(tmp_comm_pkg) = send_procs;
         jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = Pext_send_map_start[pass];
         jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
         jx_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = recv_procs;
         jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = Pext_recv_vec_start[pass];

         if (Pext_recv_size)
         {
            Pext_pass[pass] = jx_CTAlloc(JX_Int, Pext_recv_size);
            new_elmts[pass-1] = jx_CTAlloc(JX_Int, Pext_recv_size);
         }
         else
         {
            Pext_pass[pass] = NULL;
            new_elmts[pass-1] = NULL;
         }

         comm_handle = jx_ParCSRCommHandleCreate(11, tmp_comm_pkg, Pext_send_buffer, Pext_pass[pass]);
         jx_ParCSRCommHandleDestroy(comm_handle);

         if (Pext_recv_size > old_Pext_recv_size)
         {
            jx_TFree(loc);
            loc = jx_CTAlloc(JX_Int, Pext_recv_size);
         }
         old_Pext_recv_size = Pext_recv_size;
      }

      cnt_new = 0;
      cnt_offd = 0;
      for (i = 0; i < num_recvs; i ++)
      {
         for (j = recv_vec_start[i]; j < recv_vec_start[i+1]; j ++)
	 {
	    if (assigned_offd[j] == pass - 1)
            {
	       for (j1 = cnt_offd; j1 < cnt_offd+Pext_i[j+1]; j1 ++)
               {
	          k1 = Pext_pass[pass][j1];
	          k2 = k1 - my_first_cpt;
	          if (k2 > -1 && k2 < n_coarse)
                  {
	             Pext_pass[pass][j1] = - k2 - 1;
                  }
                  else
                  {
                     not_found = 1;
                     k3 = 0;
                     while (k3 < pass-1 && not_found)
                     {
                        k2 = jx_BinarySearch(new_elmts[k3], k1, (new_counter[k3+1]-new_counter[k3]));
                        if (k2 > -1)
                        {
	                   Pext_pass[pass][j1] = k2 + new_counter[k3];
	                   not_found = 0;
                        }
                        else
                        {
	                   k3 ++;
                        }
                     }
                     if (not_found)
                     {
		        new_elmts[pass-1][cnt_new] = Pext_pass[pass][j1];
		        loc[cnt_new++] = j1;
                     }
                  }
               }
	       cnt_offd += Pext_i[j+1];
	    }
	 }
      }

      if (cnt_new)
      {
  	 jx_qsort2i(new_elmts[pass-1], loc, 0, cnt_new - 1);
         cnt = 0;
         local_index = new_counter[pass-1];
	 Pext_pass[pass][loc[0]] = local_index;

         for (i = 1; i < cnt_new; i ++)
         {
	    if (new_elmts[pass-1][i] > new_elmts[pass-1][cnt])
	    {
	       new_elmts[pass-1][++cnt] = new_elmts[pass-1][i];
	       local_index ++;
	    }
	    Pext_pass[pass][loc[i]] = local_index;
         }
         new_counter[pass] = local_index + 1;
      }
      else if (num_procs > 1)
      {
         new_counter[pass] = new_counter[pass-1];
      }
      
      if (new_num_cols_offd < local_index + 1)
      {
         new_num_cols_offd = local_index + 1;

         jx_TFree(P_marker_offd);
         P_marker_offd = jx_CTAlloc(JX_Int, new_num_cols_offd);

         for (i = 0; i < new_num_cols_offd; i ++)
         {
	    P_marker_offd[i] = -1;
	 }
      }
     
      cnt_nz = 0;
      cnt_nz_offd = 0;
      for (i = pass_pointer[pass]; i < pass_pointer[pass+1]; i ++)
      {
	 i1 = pass_array[i];
         P_diag_start[i1] = cnt_nz;
         P_offd_start[i1] = cnt_nz_offd;
	 for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
	 {
	    j1 = S_diag_j[j];
	    if (assigned[j1] == pass - 1)
  	    {
	       j_start = P_diag_start[j1];
	       j_end = j_start+P_diag_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = P_diag_pass[pass-1][k];
		  if (P_marker[k1] != i1)
		  {
		     cnt_nz ++;
		     P_diag_i[i1+1] ++;
		     P_marker[k1] = i1;
		  }
	       }
	       j_start = P_offd_start[j1];
	       j_end = j_start+P_offd_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = P_offd_pass[pass-1][k];
		  if (P_marker_offd[k1] != i1)
		  {
		     cnt_nz_offd ++;
		     P_offd_i[i1+1] ++;
		     P_marker_offd[k1] = i1;
		  }
	       }
  	    }
  	 }
	 j_start = 0;
	 for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
	 {
	    j1 = S_offd_j[j];
	    if (assigned_offd[j1] == pass - 1)
  	    {
	       j_start = Pext_start[j1];
	       j_end = j_start + Pext_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = Pext_pass[pass][k];
		  if (k1 < 0)
		  {
                     if (P_marker[-k1-1] != i1)
		     {
		        cnt_nz ++;
		        P_diag_i[i1+1] ++;
		        P_marker[-k1-1] = i1;
		     }
		  }
		  else if (P_marker_offd[k1] != i1)
		  {
		     cnt_nz_offd ++;
		     P_offd_i[i1+1] ++;
		     P_marker_offd[k1] = i1;
		  }
	       }
  	    }
  	 }
      }

      total_nz += cnt_nz;
      total_nz_offd += cnt_nz_offd;

      P_diag_pass[pass] = jx_CTAlloc(JX_Int, cnt_nz);
      if (cnt_nz_offd)
      {
         P_offd_pass[pass] = jx_CTAlloc(JX_Int, cnt_nz_offd);
      }
      else if (num_procs > 1)
      {
         P_offd_pass[pass] = NULL;
      }

      cnt_nz = 0;
      cnt_nz_offd = 0;
      for (i = pass_pointer[pass]; i < pass_pointer[pass+1]; i ++)
      {
         i1 = pass_array[i];
         for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
         {
            j1 = S_diag_j[j];
	    if (assigned[j1] == pass - 1)
  	    {
	       j_start = P_diag_start[j1];
	       j_end = j_start+P_diag_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = P_diag_pass[pass-1][k];
		  if (P_marker[k1] != - i1 - 1)
		  {
		      P_diag_pass[pass][cnt_nz ++] = k1;
		      P_marker[k1] = - i1 - 1;
		  }
	       }
	       j_start = P_offd_start[j1];
	       j_end = j_start + P_offd_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = P_offd_pass[pass-1][k];
		  if (P_marker_offd[k1] != - i1 - 1)
		  {
		     P_offd_pass[pass][cnt_nz_offd++] = k1;
		     P_marker_offd[k1] = - i1 - 1;
		  }
	       }
  	    }
         }
         for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
         {
            j1 = S_offd_j[j];
	    if (assigned_offd[j1] == pass - 1)
  	    {
	       j_start = Pext_start[j1];
	       j_end = j_start + Pext_i[j1+1];
	       for (k = j_start; k < j_end; k ++)
	       {
		  k1 = Pext_pass[pass][k];
		  if (k1 < 0)
		  {
 		     if (P_marker[-k1-1] != - i1 - 1)
		     {
		        P_diag_pass[pass][cnt_nz++] = - k1 - 1;
		        P_marker[-k1-1] = - i1 - 1;
		     }
		  }
		  else if (P_marker_offd[k1] != - i1 - 1)
		  {
		     P_offd_pass[pass][cnt_nz_offd++] = k1;
		     P_marker_offd[k1] = - i1 - 1;
		  }
	       }
  	    }
         }
      }
   }

   jx_TFree(loc);
   jx_TFree(P_ncols);
   jx_TFree(Pext_send_buffer);
   jx_TFree(new_recv_vec_start);
   jx_TFree(P_marker);
   jx_TFree(P_marker_offd);
   P_marker_offd = NULL;

   P_diag_j = jx_CTAlloc(JX_Int, total_nz);
   P_diag_data = jx_CTAlloc(JX_Real, total_nz);

   if (total_nz_offd)
   {
      P_offd_j = jx_CTAlloc(JX_Int, total_nz_offd);
      P_offd_data = jx_CTAlloc(JX_Real, total_nz_offd);
   }

   for (i = 0; i < n_fine; i ++)
   {
      P_diag_i[i+1] += P_diag_i[i];
      P_offd_i[i+1] += P_offd_i[i];
   }

   /* determine P for coarse points */

   for (i = 0; i < n_coarse; i ++)
   {
      i1 = C_array[i];
      P_diag_j[P_diag_i[i1]] = fine_to_coarse[i1];
      P_diag_data[P_diag_i[i1]] = 1.0;
   }

   P_marker = jx_CTAlloc(JX_Int, n_fine);
   for (i = 0; i < n_fine; i ++)
   {
      P_marker[i] = -1;
   }

   if (num_cols_offd)
   {
      P_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      for (i = 0; i < num_cols_offd; i ++)
      {
         P_marker_offd[i] = -1;
      }
   }

   if (weight_option) /*if this is set, weights are separated into
		        negative and positive offdiagonals and accumulated
		        accordingly */
   {
      /* determine P for points of pass 1, i.e. neighbors of coarse points */ 
      for (i = pass_pointer[1]; i < pass_pointer[2]; i ++) 
      {
         i1 = pass_array[i];
         sum_C_pos = 0;
         sum_C_neg = 0;
         sum_N_pos = 0;
         sum_N_neg = 0;
         j_start = P_diag_start[i1];
         j_end = j_start + P_diag_i[i1+1] - P_diag_i[i1];
         for (j = j_start; j < j_end; j ++)
         {
            k1 = P_diag_pass[1][j];
	    P_marker[C_array[k1]] = i1;
         }
         cnt = P_diag_i[i1];
         for (j = A_diag_i[i1] + 1; j < A_diag_i[i1+1]; j ++)
         {
            j1 = A_diag_j[j];
	    if (CF_marker[j1] != -3 && (num_functions == 1 || dof_func[i1] == dof_func[j1]))
	    {
	       if (A_diag_data[j] < 0)
	       {
	          sum_N_neg += A_diag_data[j];
	       }
	       else
	       {
	          sum_N_pos += A_diag_data[j];
	       }
	    }
	    if (j1 != -1 && P_marker[j1] == i1)
	    {
	       P_diag_data[cnt] = A_diag_data[j];
	       P_diag_j[cnt++] = fine_to_coarse[j1];
	       if (A_diag_data[j] < 0)
	       {
	          sum_C_neg += A_diag_data[j];
	       }
	       else
	       {
	          sum_C_pos += A_diag_data[j];
	       }
            }
         }
         j_start = P_offd_start[i1];
         j_end = j_start + P_offd_i[i1+1] - P_offd_i[i1];
         for (j = j_start; j < j_end; j ++)
         {
            k1 = P_offd_pass[1][j];
	    P_marker_offd[C_array_offd[k1]] = i1;
         }
         cnt_offd = P_offd_i[i1];
         for (j = A_offd_i[i1]; j < A_offd_i[i1+1]; j ++)
         {
	    if (col_offd_S_to_A)
	    {
               j1 = map_A_to_S[A_offd_j[j]];
            }
	    else
	    {
               j1 = A_offd_j[j];
            }
            if (CF_marker_offd[j1] != -3 && (num_functions == 1 || dof_func[i1] == dof_func_offd[j1]))
	    {
	       if (A_offd_data[j] < 0)
	       {
	          sum_N_neg += A_offd_data[j];
	       }
	       else
	       {
	          sum_N_pos += A_offd_data[j];
	       }
	    }
	    if (j1 != -1 && P_marker_offd[j1] == i1)
	    {
	       P_offd_data[cnt_offd] = A_offd_data[j];
	       P_offd_j[cnt_offd++] = map_S_to_new[j1];
	       if (A_offd_data[j] < 0)
	       {
	          sum_C_neg += A_offd_data[j];
	       }
	       else
	       {
	          sum_C_pos += A_offd_data[j];
	       }
            }
         }
         diagonal = A_diag_data[A_diag_i[i1]];
         
         if (sum_C_neg*diagonal) 
         {
            alfa = - sum_N_neg / (sum_C_neg*diagonal);
         }
         
         if (sum_C_pos*diagonal) 
         {
            beta = - sum_N_pos / (sum_C_pos*diagonal);
         }
         
         for (j = P_diag_i[i1]; j < cnt; j ++)
         {
            if (P_diag_data[j] < 0)
            {
	       P_diag_data[j] *= alfa;
	    }
            else
            {
	       P_diag_data[j] *= beta;
	    }
	 }
	 
         for (j = P_offd_i[i1]; j < cnt_offd; j ++)
         {
            if (P_offd_data[j] < 0)
            {
	       P_offd_data[j] *= alfa;
	    }
            else
            {
	       P_offd_data[j] *= beta;
	    }
	 }
      }

      old_Pext_send_size = 0;
      old_Pext_recv_size = 0;

      jx_TFree(P_diag_pass[1]);
      if (num_procs > 1) 
      {
         jx_TFree(P_offd_pass[1]);
      }
	  
      if (new_num_cols_offd > n_coarse_offd)
      {
         jx_TFree(C_array_offd);
         C_array_offd = jx_CTAlloc(JX_Int, new_num_cols_offd);
      }

      for (pass = 2; pass < num_passes; pass ++)
      {

         if (num_procs > 1)
         {
            Pext_send_size = Pext_send_map_start[pass][num_sends];
            if (Pext_send_size > old_Pext_send_size)
            {
               jx_TFree(Pext_send_data);
               Pext_send_data = jx_CTAlloc(JX_Real, Pext_send_size);
            }
            old_Pext_send_size = Pext_send_size;

            cnt_offd = 0;
            for (i = 0; i < num_sends; i ++)
            {
               for (j = send_map_start[i]; j < send_map_start[i+1]; j ++)
               {
                  j1 = send_map_elmt[j];
	          if (assigned[j1] == pass - 1)
	          {
	             j_start = P_diag_i[j1];
	             j_end = P_diag_i[j1+1];
	             for (k = j_start; k < j_end; k ++)
	             {
		        Pext_send_data[cnt_offd++] = P_diag_data[k];
	             }
	             j_start = P_offd_i[j1];
	             j_end = P_offd_i[j1+1];
	             for (k = j_start; k < j_end; k ++)
	             {
		        Pext_send_data[cnt_offd++] = P_offd_data[k];
	             }
                  }
               }
            }
 
            jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
            jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = Pext_send_map_start[pass];
            jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
            jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = Pext_recv_vec_start[pass];

            Pext_recv_size = Pext_recv_vec_start[pass][num_recvs];

            if (Pext_recv_size > old_Pext_recv_size)
            {
               jx_TFree(Pext_data);
               Pext_data = jx_CTAlloc(JX_Real, Pext_recv_size);
            }
            old_Pext_recv_size = Pext_recv_size;

            comm_handle = jx_ParCSRCommHandleCreate(1, tmp_comm_pkg, Pext_send_data, Pext_data);
            jx_ParCSRCommHandleDestroy(comm_handle);

            jx_TFree(Pext_send_map_start[pass]);
            jx_TFree(Pext_recv_vec_start[pass]);
         }

         for (i = pass_pointer[pass]; i < pass_pointer[pass+1]; i ++)
         {
            i1 = pass_array[i];
            sum_C_neg = 0;
            sum_C_pos = 0;
            sum_N_neg = 0;
            sum_N_pos = 0;
            j_start = P_diag_start[i1];
            j_end = j_start + P_diag_i[i1+1] - P_diag_i[i1];
            cnt = P_diag_i[i1];
            for (j = j_start; j < j_end; j ++)
            {
               k1 = P_diag_pass[pass][j];
	       C_array[k1] = cnt;
	       P_diag_data[cnt] = 0;
	       P_diag_j[cnt++] = k1;
            }
            j_start = P_offd_start[i1];
            j_end = j_start + P_offd_i[i1+1] - P_offd_i[i1];
            cnt_offd = P_offd_i[i1];
            for (j = j_start; j < j_end; j ++)
            {
               k1 = P_offd_pass[pass][j];
	       C_array_offd[k1] = cnt_offd;
	       P_offd_data[cnt_offd] = 0;
	       P_offd_j[cnt_offd++] = k1;
            }
            for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
            {
	       j1 = S_diag_j[j];
	       if (assigned[j1] == pass - 1)
	       {
	          P_marker[j1] = i1;
	       }
            }
            for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
            {
	       j1 = S_offd_j[j];
	       if (assigned_offd[j1] == pass - 1)
	       {
	          P_marker_offd[j1] = i1; 
	       }
            }
            for (j = A_diag_i[i1] + 1; j < A_diag_i[i1+1]; j ++)
            {
	       j1 = A_diag_j[j];
	       if (P_marker[j1] == i1)
	       {
	          for (k = P_diag_i[j1]; k < P_diag_i[j1+1]; k ++)
	          {
		     k1 = P_diag_j[k];
	             alfa = A_diag_data[j]*P_diag_data[k];
	             P_diag_data[C_array[k1]] += alfa;
	             if (alfa < 0)
	             {
	                sum_C_neg += alfa;
	                sum_N_neg += alfa;
	             }
	             else
	             {
	                sum_C_pos += alfa;
	                sum_N_pos += alfa;
	             }
                  }
	          for (k = P_offd_i[j1]; k < P_offd_i[j1+1]; k ++)
	          {
		     k1 = P_offd_j[k];
	             alfa = A_diag_data[j]*P_offd_data[k];
	             P_offd_data[C_array_offd[k1]] += alfa;
	             if (alfa < 0)
	             {
	                sum_C_neg += alfa;
	                sum_N_neg += alfa;
	             }
	             else
	             {
	                sum_C_pos += alfa;
	                sum_N_pos += alfa;
	             }
                  }
               }
               else
               {
                  if (CF_marker[j1] != -3 && (num_functions == 1 || dof_func[i1] == dof_func[j1]))
                  {
		     if (A_diag_data[j] < 0)
		     {
		        sum_N_neg += A_diag_data[j];
		     }
		     else
		     {
		        sum_N_pos += A_diag_data[j];
		     }
                  }
               }
            }
            for (j = A_offd_i[i1]; j < A_offd_i[i1+1]; j ++)
            {
	       if (col_offd_S_to_A)
	       {
	          j1 = map_A_to_S[A_offd_j[j]];
	       }
	       else
	       {
	          j1 = A_offd_j[j];
	       }
 
	       if (j1 > -1 && P_marker_offd[j1] == i1)
	       {
	          j_start = Pext_start[j1];
	          j_end = j_start + Pext_i[j1+1];
	          for (k = j_start; k < j_end; k ++)
	          {
		     k1 = Pext_pass[pass][k];
	             alfa = A_offd_data[j]*Pext_data[k];
		     if (k1 < 0)
		     {
	                P_diag_data[C_array[-k1-1]] += alfa;
	             }
		     else
		     {
	                P_offd_data[C_array_offd[k1]] += alfa;
	             }
	             if (alfa < 0)
		     {
	                sum_C_neg += alfa;
	                sum_N_neg += alfa;
                     }
	             else
		     {
	                sum_C_pos += alfa;
	                sum_N_pos += alfa;
                     }
                  }
               }
               else
               {
                  if (CF_marker_offd[j1] != -3 && (num_functions == 1 || dof_func_offd[j1] == dof_func[i1])) 
                  {
		     if ( A_offd_data[j] < 0)
		     {
		        sum_N_neg += A_offd_data[j];
		     }
		     else
		     {
		        sum_N_pos += A_offd_data[j];
		     }
                  }
               }
            }
            diagonal = A_diag_data[A_diag_i[i1]];
            
            if (sum_C_neg*diagonal) 
            {
               alfa = - sum_N_neg / (sum_C_neg*diagonal);
            }
            
            if (sum_C_pos*diagonal) 
            {
               beta = - sum_N_pos / (sum_C_pos*diagonal);
            }

            for (j = P_diag_i[i1]; j < P_diag_i[i1+1]; j ++)
            {
	       if (P_diag_data[j] < 0)
	       {
	          P_diag_data[j] *= alfa;
	       }
	       else
	       {
	          P_diag_data[j] *= beta;
	       }
	    }
	    
            for (j = P_offd_i[i1]; j < P_offd_i[i1+1]; j ++)
            {
	       if (P_offd_data[j] < 0)
	       {
	          P_offd_data[j] *= alfa;
	       }
	       else
	       {
	          P_offd_data[j] *= beta;
	       }
	    }
         }
         jx_TFree(P_diag_pass[pass]);
         if (num_procs > 1)
         {
            jx_TFree(P_offd_pass[pass]);
            jx_TFree(Pext_pass[pass]);
         }
      }
   }
   else /* no distinction between positive and negative offdiagonal element */
   {
      /* determine P for points of pass 1, i.e. neighbors of coarse points */ 
      for (i = pass_pointer[1]; i < pass_pointer[2]; i ++) 
      {
         i1 = pass_array[i];
         sum_C = 0;
         sum_N = 0;
         j_start = P_diag_start[i1];
         j_end = j_start + P_diag_i[i1+1] - P_diag_i[i1];
         for (j = j_start; j < j_end; j ++)
         {
            k1 = P_diag_pass[1][j];
	    P_marker[C_array[k1]] = i1;
         }
         cnt = P_diag_i[i1];
         for (j = A_diag_i[i1]+1; j < A_diag_i[i1+1]; j ++)
         {
            j1 = A_diag_j[j];
	    if (CF_marker[j1] != -3 && (num_functions == 1 || dof_func[i1] == dof_func[j1]))
	    {
	       sum_N += A_diag_data[j];
	    }
	    if (j1 != -1 && P_marker[j1] == i1)
	    {
	       P_diag_data[cnt] = A_diag_data[j];
	       P_diag_j[cnt++] = fine_to_coarse[j1];
	       sum_C += A_diag_data[j];
            }
         }
         j_start = P_offd_start[i1];
         j_end = j_start + P_offd_i[i1+1] - P_offd_i[i1];
         for (j = j_start; j < j_end; j ++)
         {
            k1 = P_offd_pass[1][j];
	    P_marker_offd[C_array_offd[k1]] = i1;
         }
         cnt_offd = P_offd_i[i1];
         for (j = A_offd_i[i1]; j < A_offd_i[i1+1]; j ++)
         {
	    if (col_offd_S_to_A)
	    {
               j1 = map_A_to_S[A_offd_j[j]];
            }
	    else
	    {
               j1 = A_offd_j[j];
            }
            if (CF_marker_offd[j1] != -3 && (num_functions == 1 || dof_func[i1] == dof_func_offd[j1]))
            {
	       sum_N += A_offd_data[j];
	    }
	    if (j1 != -1 && P_marker_offd[j1] == i1)
	    {
	       P_offd_data[cnt_offd] = A_offd_data[j];
	       P_offd_j[cnt_offd++] = map_S_to_new[j1];
	       sum_C += A_offd_data[j];
            }
         }
         diagonal = A_diag_data[A_diag_i[i1]];
         if (sum_C*diagonal) 
         {
            alfa = - sum_N / (sum_C*diagonal);
         }
         for (j = P_diag_i[i1]; j < cnt; j ++)
         {
	    P_diag_data[j] *= alfa;
	 }
         for (j = P_offd_i[i1]; j < cnt_offd; j ++)
         {
	    P_offd_data[j] *= alfa;
	 }
      }

      old_Pext_send_size = 0;
      old_Pext_recv_size = 0;

      jx_TFree(P_diag_pass[1]);
      if (num_procs > 1) 
      {
         jx_TFree(P_offd_pass[1]);
      }
	  
      if (new_num_cols_offd > n_coarse_offd)
      {
         jx_TFree(C_array_offd);
         C_array_offd = jx_CTAlloc(JX_Int, new_num_cols_offd);
      }

      for (pass = 2; pass < num_passes; pass ++)
      {

         if (num_procs > 1)
         {
            Pext_send_size = Pext_send_map_start[pass][num_sends];
            if (Pext_send_size > old_Pext_send_size)
            {
               jx_TFree(Pext_send_data);
               Pext_send_data = jx_CTAlloc(JX_Real, Pext_send_size);
            }
            old_Pext_send_size = Pext_send_size;

            cnt_offd = 0;
            for (i = 0; i < num_sends; i ++)
            {
               for (j = send_map_start[i]; j < send_map_start[i+1]; j ++)
               {
                  j1 = send_map_elmt[j];
	          if (assigned[j1] == pass - 1)
	          {
	             j_start = P_diag_i[j1];
	             j_end = P_diag_i[j1+1];
	             for (k = j_start; k < j_end; k ++)
	             {
		        Pext_send_data[cnt_offd++] = P_diag_data[k];
	             }
	             j_start = P_offd_i[j1];
	             j_end = P_offd_i[j1+1];
	             for (k = j_start; k < j_end; k ++)
	             {
		        Pext_send_data[cnt_offd++] = P_offd_data[k];
	             }
                  }
               }
            }
 
            jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
            jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = Pext_send_map_start[pass];
            jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
            jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = Pext_recv_vec_start[pass];

            Pext_recv_size = Pext_recv_vec_start[pass][num_recvs];

            if (Pext_recv_size > old_Pext_recv_size)
            {
               jx_TFree(Pext_data);
               Pext_data = jx_CTAlloc(JX_Real, Pext_recv_size);
            }
            old_Pext_recv_size = Pext_recv_size;

            comm_handle = jx_ParCSRCommHandleCreate(1, tmp_comm_pkg, Pext_send_data, Pext_data);
            jx_ParCSRCommHandleDestroy(comm_handle);

            jx_TFree(Pext_send_map_start[pass]);
            jx_TFree(Pext_recv_vec_start[pass]);
         }

         for (i = pass_pointer[pass]; i < pass_pointer[pass+1]; i ++)
         {
            i1 = pass_array[i];
            sum_C = 0;
            sum_N = 0;
            j_start = P_diag_start[i1];
            j_end = j_start + P_diag_i[i1+1] - P_diag_i[i1];
            cnt = P_diag_i[i1];
            for (j = j_start; j < j_end; j ++)
            {
               k1 = P_diag_pass[pass][j];
	       C_array[k1] = cnt;
	       P_diag_data[cnt] = 0;
	       P_diag_j[cnt++] = k1;
            }
            j_start = P_offd_start[i1];
            j_end = j_start + P_offd_i[i1+1] - P_offd_i[i1];
            cnt_offd = P_offd_i[i1];
            for (j = j_start; j < j_end; j ++)
            {
               k1 = P_offd_pass[pass][j];
	       C_array_offd[k1] = cnt_offd;
	       P_offd_data[cnt_offd] = 0;
	       P_offd_j[cnt_offd++] = k1;
            }
            for (j = S_diag_i[i1]; j < S_diag_i[i1+1]; j ++)
            {
	       j1 = S_diag_j[j];
	       if (assigned[j1] == pass - 1)
	       {
	          P_marker[j1] = i1;
	       }
            }
            for (j = S_offd_i[i1]; j < S_offd_i[i1+1]; j ++)
            {
	       j1 = S_offd_j[j];
	       if (assigned_offd[j1] == pass - 1)
	       {
	          P_marker_offd[j1] = i1; 
	       }
            }
            for (j = A_diag_i[i1]+1; j < A_diag_i[i1+1]; j ++)
            {
	       j1 = A_diag_j[j];
	       if (P_marker[j1] == i1)
	       {
	          for (k = P_diag_i[j1]; k < P_diag_i[j1+1]; k ++)
	          {
		     k1 = P_diag_j[k];
	             alfa = A_diag_data[j]*P_diag_data[k];
	             P_diag_data[C_array[k1]] += alfa;
	             sum_C += alfa;
	             sum_N += alfa;
                  }
	          for (k = P_offd_i[j1]; k < P_offd_i[j1+1]; k ++)
	          {
		     k1 = P_offd_j[k];
	             alfa = A_diag_data[j]*P_offd_data[k];
	             P_offd_data[C_array_offd[k1]] += alfa;
	             sum_C += alfa;
	             sum_N += alfa;
                  }
               }
               else
               {
                  if (CF_marker[j1] != -3 && (num_functions == 1 || dof_func[i1] == dof_func[j1]))
                  {
		     sum_N += A_diag_data[j];
		  }
               }
            }
            for (j = A_offd_i[i1]; j < A_offd_i[i1+1]; j ++)
            {
	       if (col_offd_S_to_A)
	       {
	          j1 = map_A_to_S[A_offd_j[j]];
	       }
	       else
	       {
	          j1 = A_offd_j[j];
	       }
 
	       if (j1 > -1 && P_marker_offd[j1] == i1)
	       {
	          j_start = Pext_start[j1];
	          j_end = j_start+Pext_i[j1+1];
	          for (k = j_start; k < j_end; k ++)
	          {
		     k1 = Pext_pass[pass][k];
	             alfa = A_offd_data[j]*Pext_data[k];
		     if (k1 < 0) 
		     {
	                P_diag_data[C_array[-k1-1]] += alfa;
	             }
		     else
		     {
	                P_offd_data[C_array_offd[k1]] += alfa;
	             }
	             sum_C += alfa;
	             sum_N += alfa;
                  }
               }
               else
               {
                  if (CF_marker_offd[j1] != -3 && (num_functions == 1 || dof_func_offd[j1] == dof_func[i1])) 
                  {
		     sum_N += A_offd_data[j];
		  }
               }
            }
            diagonal = A_diag_data[A_diag_i[i1]];
            if (sum_C*diagonal) 
            {
               alfa = - sum_N / (sum_C*diagonal);
            }

            for (j = P_diag_i[i1]; j < P_diag_i[i1+1]; j ++)
            {
	       P_diag_data[j] *= alfa;
	    }
            for (j = P_offd_i[i1]; j < P_offd_i[i1+1]; j ++)
            {
	       P_offd_data[j] *= alfa;
	    }
         }
         
         jx_TFree(P_diag_pass[pass]);
         if (num_procs > 1)
         {
            jx_TFree(P_offd_pass[pass]);
            jx_TFree(Pext_pass[pass]);
         }
      }
   }

   jx_TFree(CF_marker_offd);
   jx_TFree(Pext_send_map_start);
   jx_TFree(Pext_recv_vec_start);
   if (n_coarse) 
   {
      jx_TFree(C_array);
   }
   jx_TFree(C_array_offd);
   jx_TFree(dof_func_offd);
   jx_TFree(Pext_send_data);
   jx_TFree(Pext_data);
   jx_TFree(P_diag_pass);
   jx_TFree(P_offd_pass);
   jx_TFree(Pext_pass);
   jx_TFree(P_diag_start);
   jx_TFree(P_offd_start);
   jx_TFree(Pext_start);
   jx_TFree(Pext_i);
   jx_TFree(fine_to_coarse);
   jx_TFree(assigned);
   jx_TFree(assigned_offd);
   jx_TFree(pass_pointer);
   jx_TFree(pass_array);
   jx_TFree(map_S_to_new);
   jx_TFree(map_A_to_S);
   jx_TFree(P_marker);
   if (num_procs > 1) 
   {
      jx_TFree(tmp_comm_pkg);
   }

    par_P = jx_ParCSRMatrixCreate(comm,
                                  jx_ParCSRMatrixGlobalNumRows(par_A),
                                  total_global_cpts,
                                  jx_ParCSRMatrixColStarts(par_A),
                                  num_cpts_global,
                                  0,
                                  P_diag_i[n_fine],
                                  P_offd_i[n_fine]);
                                  
   P_diag = jx_ParCSRMatrixDiag(par_P);
   jx_CSRMatrixData(P_diag) = P_diag_data;
   jx_CSRMatrixI(P_diag) = P_diag_i;
   jx_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = jx_ParCSRMatrixOffd(par_P);
   jx_CSRMatrixData(P_offd) = P_offd_data;
   jx_CSRMatrixI(P_offd) = P_offd_i;
   jx_CSRMatrixJ(P_offd) = P_offd_j;
   jx_ParCSRMatrixOwnsRowStarts(par_P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max 
      and/or keep yat most <P_max_elmts> per row absolutely maximal coefficients */

   if (trunc_factor != 0.0 || P_max_elmts != 0)
   {
      jx_PAMGInterpTruncation(par_P, trunc_factor, P_max_elmts);
      P_diag_data = jx_CSRMatrixData(P_diag);
      P_diag_i = jx_CSRMatrixI(P_diag);
      P_diag_j = jx_CSRMatrixJ(P_diag);
      P_offd_data = jx_CSRMatrixData(P_offd);
      P_offd_i = jx_CSRMatrixI(P_offd);
      P_offd_j = jx_CSRMatrixJ(P_offd);
   }
   P_offd_size = P_offd_i[n_fine];

   num_cols_offd_P = 0;
   if (P_offd_size)
   {
      if (new_num_cols_offd > num_cols_offd)
      {
          jx_TFree(P_marker_offd);
          P_marker_offd = jx_CTAlloc(JX_Int, new_num_cols_offd);
      }

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
      for (i = 0; i < new_num_cols_offd; i ++)
      {
         P_marker_offd[i] = 0;
      }
 
      num_cols_offd_P = 0;
      for (i = 0; i < P_offd_size; i ++)
      {
         index = P_offd_j[i];
         if (!P_marker_offd[index])
         {
            num_cols_offd_P ++;
            P_marker_offd[index] = 1;
         }
      }

      col_map_offd_P = jx_CTAlloc(JX_Int, num_cols_offd_P);
      permute = jx_CTAlloc(JX_Int, new_counter[num_passes-1]);

      for (i = 0; i < new_counter[num_passes-1]; i ++)
      {
         permute[i] = -1;
      }

      cnt = 0;
      for (i = 0; i < num_passes-1; i ++)
      {
         for (j = new_counter[i]; j < new_counter[i+1]; j ++)
         {
	    if (P_marker_offd[j])
	    {
	       col_map_offd_P[cnt] = new_elmts[i][j-new_counter[i]];
	       permute[j] = col_map_offd_P[cnt++];
	    }
         }
      }

      jx_qsort0(col_map_offd_P, 0, num_cols_offd_P - 1);

      for (i = 0; i < new_counter[num_passes-1]; i ++)
      {
         k1 = permute[i];
         if (k1 != -1)
         {
            permute[i] = jx_BinarySearch(col_map_offd_P, k1, num_cols_offd_P);
         }
      }

#define JX_SMP_PRIVATE i
#include "../../include/jx_smp_forloop.h"
      for (i = 0; i < P_offd_size; i ++)
      {
         P_offd_j[i] = permute[P_offd_j[i]];
      }
   }
   if (num_procs > 1)
   {
      for (i = 0; i < num_passes - 1; i ++)
      {
         jx_TFree(new_elmts[i]);
      }
   }
   jx_TFree(P_marker_offd);
   jx_TFree(permute);
   jx_TFree(new_elmts);
   jx_TFree(new_counter);

   if (num_cols_offd_P)
   {
        jx_ParCSRMatrixColMapOffd(par_P) = col_map_offd_P;
        jx_CSRMatrixNumCols(P_offd) = num_cols_offd_P;
   }

   if (n_SF)
   {
      for (i = 0; i < n_fine; i ++)
      {
         if (CF_marker[i] == -3) 
         {
            CF_marker[i] = -1;
         }
      }
   }

   if (num_procs > 1)
   {
      jx_MatvecCommPkgCreate(par_P);
   }

   *P_ptr = par_P;

   return(0);
}
