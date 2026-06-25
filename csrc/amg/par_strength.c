//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_strength.c -- Generate a strengthen matrix of a parallel CSR matrix.
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGCreateS
 * \brief Generate a strengthen matrix of a given parallel CSR matrix.
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGCreateS( jx_ParCSRMatrix    *par_A,
                JX_Real              strength_threshold,
                JX_Real              max_row_sum,
                JX_Int                 num_functions,
                JX_Int                *dof_func,
                jx_ParCSRMatrix   **S_ptr )
{
   MPI_Comm              comm        = jx_ParCSRMatrixComm(par_A);
   jx_ParCSRCommPkg     *comm_pkg    = jx_ParCSRMatrixCommPkg(par_A);
   jx_ParCSRCommHandle  *comm_handle;
   jx_CSRMatrix         *A_diag      = jx_ParCSRMatrixDiag(par_A);
   JX_Int                  *A_diag_i    = jx_CSRMatrixI(A_diag);
   JX_Real               *A_diag_data = jx_CSRMatrixData(A_diag);


   jx_CSRMatrix       *A_offd          = jx_ParCSRMatrixOffd(par_A);
   JX_Int                *A_offd_i        = jx_CSRMatrixI(A_offd);
   JX_Real             *A_offd_data     = NULL;
   JX_Int                *A_diag_j        = jx_CSRMatrixJ(A_diag);
   JX_Int                *A_offd_j        = jx_CSRMatrixJ(A_offd);

   JX_Int 		      *row_starts      = jx_ParCSRMatrixRowStarts(par_A);
   JX_Int                 num_variables   = jx_CSRMatrixNumRows(A_diag);
   JX_Int                 global_num_vars = jx_ParCSRMatrixGlobalNumRows(par_A);
   JX_Int 		       num_nonzeros_diag;
   JX_Int 		       num_nonzeros_offd = 0;
   JX_Int 		       num_cols_offd     = 0;

   jx_ParCSRMatrix    *par_S;
   jx_CSRMatrix       *S_diag;
   JX_Int                *S_diag_i;
   JX_Int                *S_diag_j;
   jx_CSRMatrix       *S_offd;
   JX_Int                *S_offd_i;
   JX_Int                *S_offd_j = NULL;

   JX_Real              diag, row_scale, row_sum;
   JX_Int                 i, jA, jS;
                      
   JX_Int                 ierr = 0;

   JX_Int                *dof_func_offd;
   JX_Int                 num_sends;
   JX_Int                *int_buf_data;
   JX_Int                 index, start, j;
   
   JX_Int                 num_dd_strong = 0;
   JX_Int                 num_not_dd = 0;
   JX_Int                 num_not_dd_strong = 0;
   JX_Int                 print_ai = 0;

  /*-------------------------------------------------------------------
   * Compute a  ParCSR strength matrix, par_S.
   *
   * For now, the "strength" of dependence/influence is defined in
   * the following way: i depends on j if
   *     aij > jx_max(k != i) aik,    aii < 0
   * or
   *     aij < jx_min(k != i) aik,    aii >= 0
   * Then S_ij = 1, else S_ij = 0.
   *
   * NOTE: the entries are negative initially, corresponding
   * to "unaccounted-for" dependence.
   *-----------------------------------------------------------------*/

   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = jx_CSRMatrixNumCols(A_offd);

   A_offd_i = jx_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   par_S = jx_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
                                 row_starts, row_starts, num_cols_offd,
                                 num_nonzeros_diag, num_nonzeros_offd);
   /* row_starts is owned by A, col_starts = row_starts */
   jx_ParCSRMatrixSetRowStartsOwner(par_S, 0);
   S_diag = jx_ParCSRMatrixDiag(par_S);
   jx_CSRMatrixI(S_diag) = jx_CTAlloc(JX_Int, num_variables + 1);
   jx_CSRMatrixJ(S_diag) = jx_CTAlloc(JX_Int, num_nonzeros_diag);
   S_offd = jx_ParCSRMatrixOffd(par_S);
   jx_CSRMatrixI(S_offd) = jx_CTAlloc(JX_Int, num_variables + 1);

   S_diag_i = jx_CSRMatrixI(S_diag);
   S_diag_j = jx_CSRMatrixJ(S_diag);
   S_offd_i = jx_CSRMatrixI(S_offd);

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
      A_offd_data = jx_CSRMatrixData(A_offd);
      jx_CSRMatrixJ(S_offd) = jx_CTAlloc(JX_Int, num_nonzeros_offd);
      S_offd_j = jx_CSRMatrixJ(S_offd);
      jx_ParCSRMatrixColMapOffd(par_S) = jx_CTAlloc(JX_Int, num_cols_offd);
      if (num_functions > 1)
      {
         dof_func_offd = jx_CTAlloc(JX_Int, num_cols_offd);
      }
   }


  /*-------------------------------------------------------------------
   * Get the dof_func data for the off-processor columns
   *------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_A);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   if (num_functions > 1)
   {
      int_buf_data = jx_CTAlloc(JX_Int,jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
	 start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j ++)
         {
            int_buf_data[index++] = dof_func[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
      }
	
      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);

      jx_ParCSRCommHandleDestroy(comm_handle);   
      jx_TFree(int_buf_data);
   }

   /* give par_S same nonzero structure as par_A */
   jx_ParCSRMatrixCopy(par_A, par_S, 0);

#define JX_SMP_PRIVATE i,diag,row_scale,row_sum,jA
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < num_variables; i ++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = diag;
      if (num_functions > 1)
      {
         if (diag < 0)
	 {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = jx_max(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = jx_max(row_scale, A_offd_data[jA]);
                  row_sum += A_offd_data[jA];
               }
            }
         }
         else
         {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = jx_min(row_scale, A_diag_data[jA]);
                  row_sum += A_diag_data[jA];
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = jx_min(row_scale, A_offd_data[jA]);
                  row_sum += A_offd_data[jA];
               }
            }
         }
      }
      else
      {
         if (diag < 0)
	 {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
            {
               row_scale = jx_max(row_scale, A_diag_data[jA]);
               row_sum += A_diag_data[jA];
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
            {
               row_scale = jx_max(row_scale, A_offd_data[jA]);
               row_sum += A_offd_data[jA];
            }
         }
         else
         {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
            {
               row_scale = jx_min(row_scale, A_diag_data[jA]);
               row_sum += A_diag_data[jA];
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
            {
               row_scale = jx_min(row_scale, A_offd_data[jA]);
               row_sum += A_offd_data[jA];
            }
         }
      }

      if (diag > 0.0) {
         if (row_sum < 0.0) {
            //jx_printf("i = %d, row_sum =%f, diag = %f\n ",i, row_sum, diag);
            num_not_dd++;
         } else if (row_sum > max_row_sum*diag) {
            //jx_printf("i = %d, row_sum =%f, diag = %f\n ",i, row_sum, diag);
            num_dd_strong++;
         }
      } else if (diag < 0.0) {
         if (row_sum > 0.0) {
            //jx_printf("i = %d, row_sum =%f, diag = %f\n ",i, row_sum, diag);
            num_not_dd++;
         } else if (fabs(row_sum) > max_row_sum*fabs(diag) ) {
            num_dd_strong++;
         }
      }
      if (row_sum / diag < -0.1 ) num_not_dd_strong++;

      row_sum = fabs( row_sum / diag );

      /* compute row entries of par_S */
      S_diag_j[A_diag_i[i]] = -1;
      if ( (row_sum > max_row_sum) && (max_row_sum < 1.0) )
      {
         /* make all dependencies weak */
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
         {
            S_diag_j[jA] = -1;
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
         {
            S_offd_j[jA] = -1;
         }
      }
      else
      {
         if (num_functions > 1)
         { 
            if (diag < 0) 
            { 
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
               {
                  if (A_diag_data[jA] <= strength_threshold*row_scale || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
               {
                  if (A_offd_data[jA] <= strength_threshold*row_scale || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
            else
            {
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
               {
                  if (A_diag_data[jA] >= strength_threshold*row_scale || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
               {
                  if (A_offd_data[jA] >= strength_threshold*row_scale || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
         }
         else
         {
            if (diag < 0) 
            { 
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
               {
                  if (A_diag_data[jA] <= strength_threshold*row_scale)
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
               {
                  if (A_offd_data[jA] <= strength_threshold*row_scale)
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
            else
            {
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA ++)
               {
                  if (A_diag_data[jA] >= strength_threshold*row_scale)
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
               {
                  if (A_offd_data[jA] >= strength_threshold*row_scale)
                  {
                     S_offd_j[jA] = -1;
                  }
               }
            }
         }
      }
   }

   if (print_ai == 1) {
   JX_Real num_strong = num_dd_strong;
   JX_Real num_not = num_not_dd;
   JX_Real num_not_strong = num_not_dd_strong;
   JX_Real num_var = num_variables;
   jx_printf(" - num_dd_strong = %d, ratio_dd_strong = %f \n", num_dd_strong, num_strong/num_var);
   jx_printf(" - num_not_dd = %d, ratio_not_dd = %f \n", num_not_dd, num_not/num_var);
   jx_printf(" - num_not_dd_strong = %d, ratio_not_dd_strong = %f \n", num_not_dd_strong, num_not_strong/num_var);
   }

  /*--------------------------------------------------------------
   * "Compress" the strength matrix.
   *
   * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
   *
   * NOTE: This "compression" section of code may be removed, and
   * coarsening will still be done correctly.  However, the routine
   * that builds interpolation would have to be modified first.
   *----------------------------------------------------------------*/

   /* RDF: not sure if able to thread this loop */
   jS = 0;
   for (i = 0; i < num_variables; i ++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA ++)
      {
         if (S_diag_j[jA] > -1)
         {
            S_diag_j[jS] = S_diag_j[jA];
            jS ++;
         }
      }
   }
   S_diag_i[num_variables] = jS;
   jx_CSRMatrixNumNonzeros(S_diag) = jS;

   /* RDF: not sure if able to thread this loop */
   jS = 0;
   for (i = 0; i < num_variables; i ++)
   {
      S_offd_i[i] = jS;
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA ++)
      {
         if (S_offd_j[jA] > -1)
         {
            S_offd_j[jS] = S_offd_j[jA];
            jS ++;
         }
      }
   }
   S_offd_i[num_variables] = jS;
   jx_CSRMatrixNumNonzeros(S_offd) = jS;
   jx_ParCSRMatrixCommPkg(par_S) = NULL;

   *S_ptr = par_S;

   jx_TFree(dof_func_offd);

   return (ierr);
}


/*!
 * \fn JX_Int jx_PAMGCreateSabs
 * \brief Generates strength matrix
 * \date 2018/05/10
 */
JX_Int
jx_PAMGCreateSabs( jx_ParCSRMatrix  *A,
                   JX_Real           strength_threshold,
                   JX_Real           max_row_sum,
                   JX_Int            num_functions,
                   JX_Int           *dof_func,
                   jx_ParCSRMatrix **S_ptr)
{
   MPI_Comm 	        comm     = jx_ParCSRMatrixComm(A);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(A);
   jx_ParCSRCommHandle *comm_handle;
   jx_CSRMatrix    *A_diag       = jx_ParCSRMatrixDiag(A);
   JX_Int          *A_diag_i     = jx_CSRMatrixI(A_diag);
   JX_Real         *A_diag_data  = jx_CSRMatrixData(A_diag);


   jx_CSRMatrix    *A_offd       = jx_ParCSRMatrixOffd(A);
   JX_Int          *A_offd_i     = jx_CSRMatrixI(A_offd);
   JX_Real         *A_offd_data  = NULL;
   JX_Int          *A_diag_j     = jx_CSRMatrixJ(A_diag);
   JX_Int          *A_offd_j     = jx_CSRMatrixJ(A_offd);

   JX_Int 		 *row_starts      = jx_ParCSRMatrixRowStarts(A);
   JX_Int                 num_variables   = jx_CSRMatrixNumRows(A_diag);
   JX_Int                 global_num_vars = jx_ParCSRMatrixGlobalNumRows(A);
   JX_Int 		  num_nonzeros_diag;
   JX_Int 		  num_nonzeros_offd = 0;
   JX_Int 		  num_cols_offd = 0;

   jx_ParCSRMatrix *S;
   jx_CSRMatrix    *S_diag;
   JX_Int          *S_diag_i;
   JX_Int          *S_diag_j;
   jx_CSRMatrix    *S_offd;
   JX_Int          *S_offd_i = NULL;
   JX_Int          *S_offd_j = NULL;

   JX_Real          diag, row_scale, row_sum;
   JX_Int           i, jA, jS;
                      
   JX_Int           ierr = 0;

   JX_Int          *dof_func_offd;
   JX_Int	    num_sends;
   JX_Int	   *int_buf_data;
   JX_Int	    index, start, j;
   
   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   num_nonzeros_diag = A_diag_i[num_variables];
   num_cols_offd = jx_CSRMatrixNumCols(A_offd);

   A_offd_i = jx_CSRMatrixI(A_offd);
   num_nonzeros_offd = A_offd_i[num_variables];

   S = jx_ParCSRMatrixCreate(comm, global_num_vars, global_num_vars,
			row_starts, row_starts,
			num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   /* row_starts is owned by A, col_starts = row_starts */
   jx_ParCSRMatrixSetRowStartsOwner(S,0);
   S_diag = jx_ParCSRMatrixDiag(S);
   jx_CSRMatrixI(S_diag) = jx_CTAlloc(JX_Int,  num_variables+1);
   jx_CSRMatrixJ(S_diag) = jx_CTAlloc(JX_Int,  num_nonzeros_diag);
   S_offd = jx_ParCSRMatrixOffd(S);
   jx_CSRMatrixI(S_offd) = jx_CTAlloc(JX_Int,  num_variables+1);

   S_diag_i = jx_CSRMatrixI(S_diag);
   S_diag_j = jx_CSRMatrixJ(S_diag);
   S_offd_i = jx_CSRMatrixI(S_offd);

   dof_func_offd = NULL;

   if (num_cols_offd)
   {
        A_offd_data = jx_CSRMatrixData(A_offd);
        jx_CSRMatrixJ(S_offd) = jx_CTAlloc(JX_Int,  num_nonzeros_offd);
        S_offd_j = jx_CSRMatrixJ(S_offd);
        jx_ParCSRMatrixColMapOffd(S) = jx_CTAlloc(JX_Int,  num_cols_offd);
        if (num_functions > 1)
	{
	   dof_func_offd = jx_CTAlloc(JX_Int,  num_cols_offd);
	}
   }


  /*-------------------------------------------------------------------
    * Get the dof_func data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(A);
      comm_pkg = jx_ParCSRMatrixCommPkg(A); 
   }

   num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
   if (num_functions > 1)
   {
      int_buf_data = jx_CTAlloc(JX_Int, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	 start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
	 for (j=start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	 {
	    int_buf_data[index++] = dof_func[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
	 }
      }
	
      comm_handle = jx_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      jx_ParCSRCommHandleDestroy(comm_handle);   
      jx_TFree(int_buf_data);
   }

   /* give S same nonzero structure as A */
   jx_ParCSRMatrixCopy(A,S,0);

#define JX_SMP_PRIVATE i,diag,row_scale,row_sum,jA
#include "../../include/jx_smp_forloop.h"
   for (i = 0; i < num_variables; i++)
   {
      diag = A_diag_data[A_diag_i[i]];

      /* compute scaling factor and row sum */
      row_scale = 0.0;
      row_sum = diag;
      if (num_functions > 1)
      {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               if (dof_func[i] == dof_func[A_diag_j[jA]])
               {
                  row_scale = jx_max(row_scale, fabs(A_diag_data[jA]));
                  row_sum += fabs(A_diag_data[jA]);
               }
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[jA]])
               {
                  row_scale = jx_max(row_scale, fabs(A_offd_data[jA]));
                  row_sum += fabs(A_offd_data[jA]);
               }
            }
      }
      else
      {
            for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
            {
               row_scale = jx_max(row_scale, fabs(A_diag_data[jA]));
               row_sum += fabs(A_diag_data[jA]);
            }
            for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
            {
               row_scale = jx_max(row_scale, fabs(A_offd_data[jA]));
               row_sum += fabs(A_offd_data[jA]);
            }
      }

      /* compute row entries of S */
      S_diag_j[A_diag_i[i]] = -1;
      if ((fabs(row_sum) > fabs(diag)*max_row_sum) && (max_row_sum < 1.0))
      {
         /* make all dependencies weak */
         for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
         {
            S_diag_j[jA] = -1;
         }
         for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
         {
            S_offd_j[jA] = -1;
         }
      }
      else
      {
         if (num_functions > 1)
         { 
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
               {
                  if (fabs(A_diag_data[jA]) <= strength_threshold * row_scale || dof_func[i] != dof_func[A_diag_j[jA]])
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
               {
                  if (fabs(A_offd_data[jA]) <= strength_threshold * row_scale || dof_func[i] != dof_func_offd[A_offd_j[jA]])
                  {
                     S_offd_j[jA] = -1;
                  }
               }
         }
         else
         {
               for (jA = A_diag_i[i]+1; jA < A_diag_i[i+1]; jA++)
               {
                  if (fabs(A_diag_data[jA]) <= strength_threshold * row_scale)
                  {
                     S_diag_j[jA] = -1;
                  }
               }
               for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
               {
                  if (fabs(A_offd_data[jA]) <= strength_threshold * row_scale)
                  {
                     S_offd_j[jA] = -1;
                  }
               }
         }
      }
   }

   /*--------------------------------------------------------------
    * "Compress" the strength matrix.
    *
    * NOTE: S has *NO DIAGONAL ELEMENT* on any row.  Caveat Emptor!
    *
    * NOTE: This "compression" section of code may be removed, and
    * coarsening will still be done correctly.  However, the routine
    * that builds interpolation would have to be modified first.
    *----------------------------------------------------------------*/

   /* RDF: not sure if able to thread this loop */
   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_diag_i[i] = jS;
      for (jA = A_diag_i[i]; jA < A_diag_i[i+1]; jA++)
      {
         if (S_diag_j[jA] > -1)
         {
            S_diag_j[jS] = S_diag_j[jA];
            jS ++;
         }
      }
   }
   S_diag_i[num_variables] = jS;
   jx_CSRMatrixNumNonzeros(S_diag) = jS;

   /* RDF: not sure if able to thread this loop */
   jS = 0;
   for (i = 0; i < num_variables; i++)
   {
      S_offd_i[i] = jS;
      for (jA = A_offd_i[i]; jA < A_offd_i[i+1]; jA++)
      {
         if (S_offd_j[jA] > -1)
         {
            S_offd_j[jS] = S_offd_j[jA];
            jS ++;
         }
      }
   }
   S_offd_i[num_variables] = jS;
   jx_CSRMatrixNumNonzeros(S_offd) = jS;
   jx_ParCSRMatrixCommPkg(S) = NULL;

   *S_ptr = S;

   jx_TFree(dof_func_offd);

   return (ierr);
}


/*!
 * \fn JX_Int jx_PAMGCreateSCommPkg
 * \brief Create a communication package for S.
 * \date 2011/09/03
 */  
JX_Int
jx_PAMGCreateSCommPkg( jx_ParCSRMatrix   *par_A, 
                       jx_ParCSRMatrix   *par_S,
                       JX_Int              **col_offd_S_to_A_ptr )
{
   MPI_Comm 	         comm = jx_ParCSRMatrixComm(par_A);
   MPI_Status	        *status;
   MPI_Request	        *requests;
   jx_ParCSRCommPkg     *comm_pkg_A = jx_ParCSRMatrixCommPkg(par_A);
   jx_ParCSRCommPkg     *comm_pkg_S;
   jx_ParCSRCommHandle  *comm_handle;
   jx_CSRMatrix         *A_offd = jx_ParCSRMatrixOffd(par_A);
   JX_Int                  *col_map_offd_A = jx_ParCSRMatrixColMapOffd(par_A);
                  
   jx_CSRMatrix  *S_diag = jx_ParCSRMatrixDiag(par_S);
   jx_CSRMatrix  *S_offd = jx_ParCSRMatrixOffd(par_S);
   JX_Int           *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int           *S_offd_j = jx_CSRMatrixJ(S_offd);
   JX_Int           *col_map_offd_S = jx_ParCSRMatrixColMapOffd(par_S);

   JX_Int  *recv_procs_A = jx_ParCSRCommPkgRecvProcs(comm_pkg_A);
   JX_Int  *recv_vec_starts_A = jx_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   JX_Int  *send_procs_A = jx_ParCSRCommPkgSendProcs(comm_pkg_A);
   JX_Int  *send_map_starts_A = jx_ParCSRCommPkgSendMapStarts(comm_pkg_A);
   JX_Int  *recv_procs_S;
   JX_Int  *recv_vec_starts_S;
   JX_Int  *send_procs_S;
   JX_Int  *send_map_starts_S;
   JX_Int  *send_map_elmts_S;
   JX_Int  *col_offd_S_to_A;

   JX_Int  *S_marker;
   JX_Int  *send_change;
   JX_Int  *recv_change;

   JX_Int num_variables   = jx_CSRMatrixNumRows(S_diag);
   JX_Int num_cols_offd_A = jx_CSRMatrixNumCols(A_offd);                 
   JX_Int num_cols_offd_S;
   JX_Int i, j, jcol;
   JX_Int proc, cnt, proc_cnt, total_nz;
   JX_Int first_row;
                      
   JX_Int ierr = 0;

   JX_Int num_sends_A = jx_ParCSRCommPkgNumSends(comm_pkg_A);
   JX_Int num_recvs_A = jx_ParCSRCommPkgNumRecvs(comm_pkg_A);
   JX_Int num_sends_S;
   JX_Int num_recvs_S;
   JX_Int num_nonzeros;

   num_nonzeros = S_offd_i[num_variables];

   S_marker = NULL;
   if (num_cols_offd_A)
   {
      S_marker = jx_CTAlloc(JX_Int,num_cols_offd_A);
   }

   for (i = 0; i < num_cols_offd_A; i ++)
   {
      S_marker[i] = -1;
   }

   for (i = 0; i < num_nonzeros; i ++)
   {
      jcol = S_offd_j[i];
      S_marker[jcol] = 0;
   }

   proc = 0;
   proc_cnt = 0;
   cnt = 0;
   num_recvs_S = 0;
   for (i = 0; i < num_recvs_A; i ++)
   {
      for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j ++)
      {
         if (!S_marker[j])
         {
            S_marker[j] = cnt;
            cnt ++;
            proc = 1;
         }
      }
      if (proc) 
      {
         num_recvs_S ++; 
         proc = 0;
      }
   }

   num_cols_offd_S = cnt;  
   recv_change = NULL;
   recv_procs_S = NULL;
   send_change = NULL;
   if (col_map_offd_S) 
   {
      jx_TFree(col_map_offd_S);
   }
   col_map_offd_S = NULL;
   col_offd_S_to_A = NULL;
   
   if (num_recvs_A) 
   {
      recv_change = jx_CTAlloc(JX_Int, num_recvs_A);
   }
   
   if (num_sends_A) 
   {
      send_change = jx_CTAlloc(JX_Int, num_sends_A);
   }
   
   if (num_recvs_S) 
   {
      recv_procs_S = jx_CTAlloc(JX_Int, num_recvs_S);
   }
   
   recv_vec_starts_S = jx_CTAlloc(JX_Int, num_recvs_S + 1);
   
   if (num_cols_offd_S)
   {
      col_map_offd_S = jx_CTAlloc(JX_Int,num_cols_offd_S);
      col_offd_S_to_A = jx_CTAlloc(JX_Int,num_cols_offd_S);
   }
   
   if (num_cols_offd_S < num_cols_offd_A)
   {
      for (i = 0; i < num_nonzeros; i ++)
      {
         jcol = S_offd_j[i];
         S_offd_j[i] = S_marker[jcol];
      }

      proc = 0;
      proc_cnt = 0;
      cnt = 0;
      recv_vec_starts_S[0] = 0;
      for (i = 0; i < num_recvs_A; i ++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j ++)
         {
            if (S_marker[j] != -1)
            {
               col_map_offd_S[cnt] = col_map_offd_A[j];
               col_offd_S_to_A[cnt++] = j;
               proc = 1;
            }
         }
         recv_change[i] = j - cnt - recv_vec_starts_A[i] + recv_vec_starts_S[proc_cnt];
         if (proc)
         {
            recv_procs_S[proc_cnt++] = recv_procs_A[i];
            recv_vec_starts_S[proc_cnt] = cnt;
            proc = 0;
         }
      }
   }
   else
   {
      for (i = 0; i < num_recvs_A; i ++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i+1]; j ++)
         {
            col_map_offd_S[j]  = col_map_offd_A[j];
            col_offd_S_to_A[j] = j;
         }
         recv_procs_S[i] = recv_procs_A[i];
         recv_vec_starts_S[i] = recv_vec_starts_A[i];
      }
      recv_vec_starts_S[num_recvs_A] = recv_vec_starts_A[num_recvs_A];
   } 

   requests = jx_CTAlloc(MPI_Request, num_sends_A + num_recvs_A);
   
   j = 0;
   for (i = 0; i < num_sends_A; i ++)
   {
      jx_MPI_Irecv(&send_change[i], 1, JX_MPI_INT, send_procs_A[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs_A; i ++)
   {
      jx_MPI_Isend(&recv_change[i], 1, JX_MPI_INT, recv_procs_A[i], 0, comm, &requests[j++]);
   }

   status = jx_CTAlloc(MPI_Status, j);
   jx_MPI_Waitall(j, requests, status);
   jx_TFree(status);
   jx_TFree(requests);

   num_sends_S = 0;
   total_nz = send_map_starts_A[num_sends_A];
   for (i = 0; i < num_sends_A; i ++)
   {
      if (send_change[i])
      {
	 if ((send_map_starts_A[i+1] - send_map_starts_A[i]) > send_change[i])
	 {
	    num_sends_S ++;
	 }
      }
      else
      {
	 num_sends_S ++;
      }
      total_nz -= send_change[i];
   }

   send_procs_S = NULL;
   if (num_sends_S)
   {
      send_procs_S = jx_CTAlloc(JX_Int, num_sends_S);
   }
   send_map_starts_S = jx_CTAlloc(JX_Int, num_sends_S + 1);
   send_map_elmts_S = NULL;
   
   if (total_nz)
   {
      send_map_elmts_S = jx_CTAlloc(JX_Int, total_nz);
   }

   proc = 0;
   proc_cnt = 0;
   for (i = 0; i < num_sends_A; i ++)
   {
      cnt = send_map_starts_A[i+1] - send_map_starts_A[i] - send_change[i];
      if (cnt)
      {
	 send_procs_S[proc_cnt++] = send_procs_A[i];
         send_map_starts_S[proc_cnt] = send_map_starts_S[proc_cnt-1] + cnt;
      }
   }

   comm_pkg_S = jx_CTAlloc(jx_ParCSRCommPkg, 1);
   jx_ParCSRCommPkgComm(comm_pkg_S) = comm;
   jx_ParCSRCommPkgNumRecvs(comm_pkg_S) = num_recvs_S;
   jx_ParCSRCommPkgRecvProcs(comm_pkg_S) = recv_procs_S;
   jx_ParCSRCommPkgRecvVecStarts(comm_pkg_S) = recv_vec_starts_S;
   jx_ParCSRCommPkgNumSends(comm_pkg_S) = num_sends_S;
   jx_ParCSRCommPkgSendProcs(comm_pkg_S) = send_procs_S;
   jx_ParCSRCommPkgSendMapStarts(comm_pkg_S) = send_map_starts_S;

   comm_handle = jx_ParCSRCommHandleCreate(12, comm_pkg_S, col_map_offd_S, send_map_elmts_S);
   jx_ParCSRCommHandleDestroy(comm_handle);

   first_row = jx_ParCSRMatrixFirstRowIndex(par_A);
   if (first_row)
   {
      for (i = 0; i < send_map_starts_S[num_sends_S]; i ++)
      {
         send_map_elmts_S[i] -= first_row;
      }
   }

   jx_ParCSRCommPkgSendMapElmts(comm_pkg_S) = send_map_elmts_S;
  
   jx_ParCSRMatrixCommPkg(par_S) = comm_pkg_S;
   jx_ParCSRMatrixColMapOffd(par_S) = col_map_offd_S;
   jx_CSRMatrixNumCols(S_offd) = num_cols_offd_S;

   jx_TFree(S_marker);
   jx_TFree(send_change);
   jx_TFree(recv_change);

   *col_offd_S_to_A_ptr = col_offd_S_to_A;

   return ierr;
}

/*!
 * \fn JX_Int jx_PAMGCreate2ndS
 * \brief Creates strength matrix on coarse points for
          second coarsening pass in aggressive coarsening (S*S+2S)
 * \date 2015/08/14
 */
JX_Int
jx_PAMGCreate2ndS( jx_ParCSRMatrix  *S,
                   JX_Int              *CF_marker,
                   JX_Int               num_paths,
                   JX_Int              *coarse_row_starts,
                   jx_ParCSRMatrix **C_ptr )
{
   MPI_Comm 	   comm = jx_ParCSRMatrixComm(S);
   jx_ParCSRCommPkg *comm_pkg = jx_ParCSRMatrixCommPkg(S);
   jx_ParCSRCommPkg *tmp_comm_pkg;
   jx_ParCSRCommHandle *comm_handle;

   jx_CSRMatrix *S_diag = jx_ParCSRMatrixDiag(S);

   JX_Int *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int *S_diag_j = jx_CSRMatrixJ(S_diag);

   jx_CSRMatrix *S_offd = jx_ParCSRMatrixOffd(S);

   JX_Int *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int *S_offd_j = jx_CSRMatrixJ(S_offd);

   JX_Int num_cols_diag_S = jx_CSRMatrixNumCols(S_diag);
   JX_Int num_cols_offd_S = jx_CSRMatrixNumCols(S_offd);

   jx_ParCSRMatrix *S2;
   JX_Int *col_map_offd_C = NULL;

   jx_CSRMatrix *C_diag;

   JX_Int *C_diag_data = NULL;
   JX_Int *C_diag_i;
   JX_Int *C_diag_j = NULL;

   jx_CSRMatrix *C_offd;

   JX_Int *C_offd_data = NULL;
   JX_Int *C_offd_i;
   JX_Int *C_offd_j = NULL;

   JX_Int C_diag_size;
   JX_Int C_offd_size;
   JX_Int num_cols_offd_C = 0;

   JX_Int *S_ext_diag_i = NULL;
   JX_Int *S_ext_diag_j = NULL;
   JX_Int S_ext_diag_size = 0;

   JX_Int *S_ext_offd_i = NULL;
   JX_Int *S_ext_offd_j = NULL;
   JX_Int S_ext_offd_size = 0;

   JX_Int *CF_marker_offd = NULL;

   JX_Int *S_marker = NULL;
   JX_Int *S_marker_offd = NULL;
   JX_Int *temp = NULL;

   JX_Int *fine_to_coarse = NULL;
   JX_Int *fine_to_coarse_offd = NULL;
   JX_Int *map_S_to_C = NULL;

   JX_Int num_sends = 0;
   JX_Int num_recvs = 0;
   JX_Int *send_map_starts;
   JX_Int *tmp_send_map_starts = NULL;
   JX_Int *send_map_elmts;
   JX_Int *recv_vec_starts;
   JX_Int *tmp_recv_vec_starts = NULL;
   JX_Int *int_buf_data = NULL;

   JX_Int i, j, k;
   JX_Int i1, i2, i3;
   JX_Int jj1, jj2, jcol, jrow, j_cnt;

   JX_Int jj_count_diag, jj_count_offd;
   JX_Int jj_row_begin_diag, jj_row_begin_offd;
   JX_Int cnt, cnt_offd, cnt_diag;
   JX_Int num_procs, my_id;
   JX_Int value, index;
   JX_Int num_coarse;
   JX_Int num_coarse_offd;
   JX_Int num_nonzeros;
   JX_Int num_nonzeros_diag;
   JX_Int num_nonzeros_offd;
   JX_Int global_num_coarse;
   JX_Int my_first_cpt, my_last_cpt;

   JX_Int *S_int_i = NULL;
   JX_Int *S_int_j = NULL;
   JX_Int *S_ext_i = NULL;
   JX_Int *S_ext_j = NULL;

   /*-----------------------------------------------------------------------
    *  Extract S_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   jx_MPI_Comm_size(comm, &num_procs);
   jx_MPI_Comm_rank(comm, &my_id);

#ifdef JX_NO_GLOBAL_PARTITION
   my_first_cpt = coarse_row_starts[0];
   my_last_cpt = coarse_row_starts[1]-1;
   if (my_id == (num_procs -1)) global_num_coarse = coarse_row_starts[1];
   jx_MPI_Bcast(&global_num_coarse, 1, JX_MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = coarse_row_starts[my_id];
   my_last_cpt = coarse_row_starts[my_id+1]-1;
   global_num_coarse = coarse_row_starts[num_procs];
#endif

   if (num_cols_offd_S)
   {
      CF_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd_S);
      fine_to_coarse_offd = jx_CTAlloc(JX_Int, num_cols_offd_S);
   }

   if (num_cols_diag_S) fine_to_coarse = jx_CTAlloc(JX_Int, num_cols_diag_S);

   num_coarse = 0;
   for (i = 0; i < num_cols_diag_S; i ++)
   {
      if (CF_marker[i] > 0) 
      {
         fine_to_coarse[i] = num_coarse + my_first_cpt;
         num_coarse ++;
      }
      else
      {
         fine_to_coarse[i] = -1;
      }
   }

   if (num_procs > 1)
   {
      if (!comm_pkg)
      {
         jx_MatvecCommPkgCreate(S);
         comm_pkg = jx_ParCSRMatrixCommPkg(S);
      }
      num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
      send_map_starts = jx_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = jx_ParCSRCommPkgSendMapElmts(comm_pkg);
      num_recvs = jx_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_vec_starts = jx_ParCSRCommPkgRecvVecStarts(comm_pkg);
      int_buf_data = jx_CTAlloc(JX_Int, send_map_starts[num_sends]);

      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
            int_buf_data[index++] = fine_to_coarse[send_map_elmts[j]];
      }

      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, fine_to_coarse_offd);

      for (i = 0; i < num_cols_diag_S; i ++)
         if (CF_marker[i] > 0) fine_to_coarse[i] -= my_first_cpt;

      jx_ParCSRCommHandleDestroy(comm_handle);

      index = 0;
      for (i = 0; i < num_sends; i ++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
         {
            int_buf_data[index++] = CF_marker[send_map_elmts[j]];
         }
      }

      comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);

      jx_ParCSRCommHandleDestroy(comm_handle);
      jx_TFree(int_buf_data);

      S_int_i = jx_CTAlloc(JX_Int, send_map_starts[num_sends]+1);
      S_ext_i = jx_CTAlloc(JX_Int, recv_vec_starts[num_recvs]+1);

     /*--------------------------------------------------------------------------
      * generate S_int_i through adding number of coarse row-elements of offd and diag
      * for corresponding rows. S_int_i[j+1] contains the number of coarse elements of
      * a row j (which is determined through send_map_elmts)
      *--------------------------------------------------------------------------*/
      S_int_i[0] = 0;
      j_cnt = 0;
      num_nonzeros = 0;
      for (i = 0; i < num_sends; i ++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
         {
            jrow = send_map_elmts[j];
            index = 0;
            for (k = S_diag_i[jrow]; k < S_diag_i[jrow+1]; k ++)
            {
	       if (CF_marker[S_diag_j[k]] > 0) index ++;
	    }
            for (k = S_offd_i[jrow]; k < S_offd_i[jrow+1]; k ++)
            {
	       if (CF_marker_offd[S_offd_j[k]] > 0) index ++;
	    }
            S_int_i[++j_cnt] = index;
            num_nonzeros += S_int_i[j_cnt];
         }
      }

     /*--------------------------------------------------------------------------
      * initialize communication
      *--------------------------------------------------------------------------*/
      if (num_procs > 1)
         comm_handle = jx_ParCSRCommHandleCreate(11, comm_pkg, &S_int_i[1], &S_ext_i[1]);

      if (num_nonzeros) S_int_j = jx_CTAlloc(JX_Int, num_nonzeros);

      tmp_send_map_starts = jx_CTAlloc(JX_Int, num_sends+1);
      tmp_recv_vec_starts = jx_CTAlloc(JX_Int, num_recvs+1);

      tmp_send_map_starts[0] = 0;
      j_cnt = 0;
      for (i = 0; i < num_sends; i ++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i+1]; j ++)
         {
            jrow = send_map_elmts[j];
            for (k = S_diag_i[jrow]; k < S_diag_i[jrow+1]; k ++)
            {
               if (CF_marker[S_diag_j[k]] > 0)
		  S_int_j[j_cnt++] = fine_to_coarse[S_diag_j[k]]+my_first_cpt;
            }
            for (k = S_offd_i[jrow]; k < S_offd_i[jrow+1]; k ++)
            {
               if (CF_marker_offd[S_offd_j[k]] > 0)
                  S_int_j[j_cnt++] = fine_to_coarse_offd[S_offd_j[k]];
            }
         }
         tmp_send_map_starts[i+1] = j_cnt;
      }

      tmp_comm_pkg = jx_CTAlloc(jx_ParCSRCommPkg, 1);
      jx_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
      jx_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
      jx_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
      jx_ParCSRCommPkgSendProcs(tmp_comm_pkg) = jx_ParCSRCommPkgSendProcs(comm_pkg);
      jx_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = jx_ParCSRCommPkgRecvProcs(comm_pkg);
      jx_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = tmp_send_map_starts;

      jx_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
     /*--------------------------------------------------------------------------
      * after communication exchange S_ext_i[j+1] contains the number of coarse elements
      * of a row j ! evaluate S_ext_i and compute num_nonzeros for S_ext
      *--------------------------------------------------------------------------*/
      for (i = 0; i < recv_vec_starts[num_recvs]; i ++)
         S_ext_i[i+1] += S_ext_i[i];

      num_nonzeros = S_ext_i[recv_vec_starts[num_recvs]];

      if (num_nonzeros) S_ext_j = jx_CTAlloc(JX_Int, num_nonzeros);

      tmp_recv_vec_starts[0] = 0;
      for (i = 0; i < num_recvs; i ++)
         tmp_recv_vec_starts[i+1] = S_ext_i[recv_vec_starts[i+1]];

      jx_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = tmp_recv_vec_starts;

      comm_handle = jx_ParCSRCommHandleCreate(11, tmp_comm_pkg, S_int_j, S_ext_j);
      jx_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      jx_TFree(tmp_send_map_starts);
      jx_TFree(tmp_recv_vec_starts);
      jx_TFree(tmp_comm_pkg);

      jx_TFree(S_int_i);
      jx_TFree(S_int_j);

      S_ext_diag_size = 0;
      S_ext_offd_size = 0;

      for (i = 0; i < num_cols_offd_S; i ++)
      {
         for (j = S_ext_i[i]; j < S_ext_i[i+1]; j ++)
         {
            if (S_ext_j[j] < my_first_cpt || S_ext_j[j] > my_last_cpt)
               S_ext_offd_size ++;
            else
               S_ext_diag_size ++;
         }
      }
      S_ext_diag_i = jx_CTAlloc(JX_Int, num_cols_offd_S+1);
      S_ext_offd_i = jx_CTAlloc(JX_Int, num_cols_offd_S+1);

      if (S_ext_diag_size)
      {
         S_ext_diag_j = jx_CTAlloc(JX_Int, S_ext_diag_size);
      }
      if (S_ext_offd_size)
      {
         S_ext_offd_j = jx_CTAlloc(JX_Int, S_ext_offd_size);
      }

      cnt_offd = 0;
      cnt_diag = 0;
      cnt = 0;
      num_coarse_offd = 0;
      for (i = 0; i < num_cols_offd_S; i ++)
      {
         if (CF_marker_offd[i] > 0) num_coarse_offd ++;

         for (j = S_ext_i[i]; j < S_ext_i[i+1]; j ++)
         {
            i1 = S_ext_j[j];
            if (i1 < my_first_cpt || i1 > my_last_cpt)
               S_ext_offd_j[cnt_offd++] = i1;
            else
               S_ext_diag_j[cnt_diag++] = i1 - my_first_cpt;
         }
         S_ext_diag_i[++cnt] = cnt_diag;
         S_ext_offd_i[cnt] = cnt_offd;
      }

      jx_TFree(S_ext_i);
      jx_TFree(S_ext_j);

      cnt = 0;
      if (S_ext_offd_size || num_coarse_offd)
      {
         temp = jx_CTAlloc(JX_Int, S_ext_offd_size+num_coarse_offd);
         for (i = 0; i < S_ext_offd_size; i ++)
            temp[i] = S_ext_offd_j[i];
         cnt = S_ext_offd_size;
         for (i = 0; i < num_cols_offd_S; i ++)
            if (CF_marker_offd[i] > 0) temp[cnt++] = fine_to_coarse_offd[i];
      }
      if (cnt)
      {
         jx_qsort0(temp, 0, cnt-1);

         num_cols_offd_C = 1;
         value = temp[0];
         for (i = 1; i < cnt; i ++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_C++] = value;
            }
         }
      }

      if (num_cols_offd_C)
         col_map_offd_C = jx_CTAlloc(JX_Int,num_cols_offd_C);

      for (i = 0; i < num_cols_offd_C; i ++)
         col_map_offd_C[i] = temp[i];

      if (S_ext_offd_size || num_coarse_offd)
         jx_TFree(temp);

      for (i = 0 ; i < S_ext_offd_size; i ++)
         S_ext_offd_j[i] = jx_BinarySearch(col_map_offd_C, S_ext_offd_j[i], num_cols_offd_C);
      if (num_cols_offd_S)
      {
         map_S_to_C = jx_CTAlloc(JX_Int, num_cols_offd_S);

         cnt = 0;
         for (i = 0; i < num_cols_offd_S; i ++)
         {
            if (CF_marker_offd[i] > 0)
            {
               while (fine_to_coarse_offd[i] > col_map_offd_C[cnt])
               {
                  cnt ++;
               }
               map_S_to_C[i] = cnt ++;
            }
            else
            {
               map_S_to_C[i] = -1;
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate and initialize some stuff.
    *-----------------------------------------------------------------------*/

   if (num_coarse) S_marker = jx_CTAlloc(JX_Int, num_coarse);

   for (i1 = 0; i1 < num_coarse; i1 ++)
      S_marker[i1] = -1;

   S_marker_offd = jx_CTAlloc(JX_Int, num_cols_offd_C);

   for (i1 = 0; i1 < num_cols_offd_C; i1 ++)
      S_marker_offd[i1] = -1;

   C_diag_i = jx_CTAlloc(JX_Int, num_coarse+1);
   C_offd_i = jx_CTAlloc(JX_Int, num_coarse+1);

   /*-----------------------------------------------------------------------
    *  Loop over rows of S
    *-----------------------------------------------------------------------*/

   cnt = 0; 
   num_nonzeros_diag = 0; 
   num_nonzeros_offd = 0; 
   for (i1 = 0; i1 < num_cols_diag_S; i1 ++)
   {
      
      if (CF_marker[i1] > 0)
      {
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1 ++)
         {
             jcol = S_diag_j[jj1];
	     if (CF_marker[jcol] > 0)
	     {
	        S_marker[fine_to_coarse[jcol]] = i1;
	        num_nonzeros_diag ++;
	     }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1 ++)
         {
             jcol = S_offd_j[jj1];
	     if (CF_marker_offd[jcol] > 0)
	     {
	        S_marker_offd[map_S_to_C[jcol]] = i1;
	        num_nonzeros_offd ++;
	     }
         }
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1 ++)
         {
             i2 = S_diag_j[jj1];
             for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2+1]; jj2 ++)
	     {
	        i3 = S_diag_j[jj2];
	        if (CF_marker[i3] > 0 && S_marker[fine_to_coarse[i3]] != i1)
	        {
                   S_marker[fine_to_coarse[i3]] = i1;
	           num_nonzeros_diag ++;
                }
             }
             for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2+1]; jj2 ++)
	     {
	        i3 = S_offd_j[jj2];
	        if (CF_marker_offd[i3] > 0 && S_marker_offd[map_S_to_C[i3]] != i1)
	        {
                   S_marker_offd[map_S_to_C[i3]] = i1;
	           num_nonzeros_offd ++;
                }
             }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1 ++)
         {
             i2 = S_offd_j[jj1];
             for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2+1]; jj2 ++)
	     {
	        i3 = S_ext_diag_j[jj2];
	        if (S_marker[i3] != i1)
	        {
                   S_marker[i3] = i1;
	           num_nonzeros_diag ++;
                }
             }
             for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2+1]; jj2 ++)
	     {
	        i3 = S_ext_offd_j[jj2];
	        if (S_marker_offd[i3] != i1)
	        {
                   S_marker_offd[i3] = i1;
	           num_nonzeros_offd ++;
                }
             }
         }
         C_diag_i[++cnt] = num_nonzeros_diag;
         C_offd_i[cnt] = num_nonzeros_offd;
      }
   }

   if (num_nonzeros_diag)
   {
      C_diag_j = jx_CTAlloc(JX_Int, num_nonzeros_diag);
      C_diag_data = jx_CTAlloc(JX_Int, num_nonzeros_diag);
   }
   if (num_nonzeros_offd)
   {
      C_offd_j = jx_CTAlloc(JX_Int, num_nonzeros_offd);
      C_offd_data = jx_CTAlloc(JX_Int, num_nonzeros_offd);
   }

   for (i1 = 0; i1 < num_coarse; i1 ++)
      S_marker[i1] = -1;

   for (i1 = 0; i1 < num_cols_offd_C; i1 ++)
      S_marker_offd[i1] = -1;

   jj_count_diag = 0;
   jj_count_offd = 0;

   for (i1 = 0; i1 < num_cols_diag_S; i1 ++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1} (for square matrices). 
       *--------------------------------------------------------------------*/
 
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;

      if (CF_marker[i1] > 0)
      {
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1 ++)
         {
             jcol = S_diag_j[jj1];
	     if (CF_marker[jcol] > 0)
	     {
	        S_marker[fine_to_coarse[jcol]] = jj_count_diag;
	        C_diag_j[jj_count_diag] = fine_to_coarse[jcol];
	        C_diag_data[jj_count_diag] = 2;
	        jj_count_diag ++;
	     }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1 ++)
         {
             jcol = S_offd_j[jj1];
	     if (CF_marker_offd[jcol] > 0)
	     {
	        index = map_S_to_C[jcol];
	        S_marker_offd[index] = jj_count_offd;
	        C_offd_j[jj_count_offd] = index;
	        C_offd_data[jj_count_offd] = 2;
	        jj_count_offd ++;
	     }
         }
         for (jj1 = S_diag_i[i1]; jj1 < S_diag_i[i1+1]; jj1 ++)
         {
             i2 = S_diag_j[jj1];
             for (jj2 = S_diag_i[i2]; jj2 < S_diag_i[i2+1]; jj2 ++)
	     {
	        i3 = S_diag_j[jj2];
	        if (CF_marker[i3] > 0)
	        {
		   if (S_marker[fine_to_coarse[i3]] < jj_row_begin_diag)
	           {
                      S_marker[fine_to_coarse[i3]] = jj_count_diag;
	              C_diag_j[jj_count_diag] = fine_to_coarse[i3];
	              C_diag_data[jj_count_diag] ++;
	              jj_count_diag ++;
                   }
	           else
	           {
	              C_diag_data[S_marker[fine_to_coarse[i3]]] ++;
	           }
                }
             }
             for (jj2 = S_offd_i[i2]; jj2 < S_offd_i[i2+1]; jj2 ++)
	     {
	        i3 = S_offd_j[jj2];
	        if (CF_marker_offd[i3] > 0)
                {
	           index = map_S_to_C[i3];
		   if (S_marker_offd[index] < jj_row_begin_offd)
	           {
                      S_marker_offd[index] = jj_count_offd;
	              C_offd_j[jj_count_offd] = index;
	              C_offd_data[jj_count_offd] ++;
	              jj_count_offd ++;
                   }
                   else
                   {
	              C_offd_data[S_marker_offd[index]] ++;
                   }
                }
             }
         }
         for (jj1 = S_offd_i[i1]; jj1 < S_offd_i[i1+1]; jj1 ++)
         {
             i2 = S_offd_j[jj1];
             for (jj2 = S_ext_diag_i[i2]; jj2 < S_ext_diag_i[i2+1]; jj2 ++)
	     {
	        i3 = S_ext_diag_j[jj2];
	        if (S_marker[i3] < jj_row_begin_diag)
	        {
                   S_marker[i3] = jj_count_diag;
	           C_diag_j[jj_count_diag] = i3;
	           C_diag_data[jj_count_diag] ++;
	           jj_count_diag ++;
                }
	        else
	        {
	           C_diag_data[S_marker[i3]] ++;
	        }
             }
             for (jj2 = S_ext_offd_i[i2]; jj2 < S_ext_offd_i[i2+1]; jj2 ++)
	     {
	        i3 = S_ext_offd_j[jj2];
	        if (S_marker_offd[i3] < jj_row_begin_offd)
	        {
                   S_marker_offd[i3] = jj_count_offd;
	           C_offd_j[jj_count_offd] = i3;
	           C_offd_data[jj_count_offd] ++;
	           jj_count_offd ++;
                }
                else
                {
	           C_offd_data[S_marker_offd[i3]] ++;
                }
             }
         }
      }
   }

   cnt = 0;

   for (i = 0; i < num_coarse; i ++)
   {
      for (j = C_diag_i[i]; j < C_diag_i[i+1]; j ++)
      {
         jcol = C_diag_j[j];
         if (C_diag_data[j] >= num_paths && jcol != i)
            C_diag_j[cnt++] = jcol;
      }
      C_diag_i[i] = cnt;
   }

   if (num_nonzeros_diag) jx_TFree(C_diag_data);
   for (i = num_coarse; i > 0; i --)
      C_diag_i[i] = C_diag_i[i-1];

   C_diag_i[0] = 0;

   cnt = 0;
   for (i = 0; i < num_coarse; i ++)
   {
      for (j = C_offd_i[i]; j < C_offd_i[i+1]; j ++)
      {
         jcol = C_offd_j[j];
         if (C_offd_data[j] >= num_paths)
            C_offd_j[cnt++] = jcol;
      }
      C_offd_i[i] = cnt;
   }

   if (num_nonzeros_offd) jx_TFree(C_offd_data);

   for (i = num_coarse; i > 0; i --)
      C_offd_i[i] = C_offd_i[i-1];

   C_offd_i[0] = 0;

   cnt = 0;
   for (i = 0; i < num_cols_diag_S; i ++)
   {
      if (CF_marker[i] > 0)
      {
         if (!(C_diag_i[cnt+1]-C_diag_i[cnt]) && !(C_offd_i[cnt+1]-C_offd_i[cnt]))
            CF_marker[i] = 2;
         cnt ++;
      }
   }

   C_diag_size = C_diag_i[num_coarse];
   C_offd_size = C_offd_i[num_coarse];

   S2 = jx_ParCSRMatrixCreate(comm, global_num_coarse,
                              global_num_coarse, coarse_row_starts,
                              coarse_row_starts, num_cols_offd_C, C_diag_size, C_offd_size);

   jx_ParCSRMatrixOwnsRowStarts(S2) = 0;

   C_diag = jx_ParCSRMatrixDiag(S2);
   jx_CSRMatrixI(C_diag) = C_diag_i;
   if (num_nonzeros_diag) jx_CSRMatrixJ(C_diag) = C_diag_j;

   C_offd = jx_ParCSRMatrixOffd(S2);
   jx_CSRMatrixI(C_offd) = C_offd_i;
   jx_ParCSRMatrixOffd(S2) = C_offd;

   if (num_cols_offd_C)
   {
      if (num_nonzeros_offd) jx_CSRMatrixJ(C_offd) = C_offd_j;
      jx_ParCSRMatrixColMapOffd(S2) = col_map_offd_C;
   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   jx_TFree(S_marker);   
   jx_TFree(S_marker_offd);   
   jx_TFree(S_ext_diag_i);
   jx_TFree(fine_to_coarse);
   if (S_ext_diag_size)
   {
      jx_TFree(S_ext_diag_j);
   }
   jx_TFree(S_ext_offd_i);
   if (S_ext_offd_size)
   {
      jx_TFree(S_ext_offd_j);
   }
   if (num_cols_offd_S) 
   {
      jx_TFree(map_S_to_C);
      jx_TFree(CF_marker_offd);
      jx_TFree(fine_to_coarse_offd);
   }

   *C_ptr = S2;

   return 0;
}


/*!
 * \fn JX_Int jx_PAMGCorrectCFMarker
 * \brief Corrects CF_marker after aggr. coarsening
 * \date 2015/08/14
 */ 
JX_Int
jx_PAMGCorrectCFMarker( JX_Int *CF_marker, JX_Int num_var, JX_Int *new_CF_marker )
{
   JX_Int i, cnt;

   cnt = 0;
   for (i = 0; i < num_var; i ++)
   {
      if (CF_marker[i] > 0)
      {
         if (CF_marker[i] == 1) CF_marker[i] = new_CF_marker[cnt++];
         else
         {
            CF_marker[i] = 1;
            cnt ++;
         }
      }
   }

   return 0;
}
