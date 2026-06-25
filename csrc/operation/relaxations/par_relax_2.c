//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax_2.c
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_PAMGRelax2
 * \brief Gauss_Seidel: interior points in parallel, boundary sequential
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelax2( jx_ParCSRMatrix *par_matrix,
               jx_ParVector    *par_rhs,
               JX_Int             *cf_marker,
               JX_Int              relax_points,
               JX_Real           relax_weight,
               JX_Real           omega,
               jx_ParVector    *par_app,
               jx_ParVector    *Vtemp )
{
   MPI_Comm         comm         = jx_ParCSRMatrixComm(par_matrix);
   jx_CSRMatrix    *A_diag       = jx_ParCSRMatrixDiag(par_matrix);
   JX_Real          *A_diag_data  = jx_CSRMatrixData(A_diag);
   JX_Int             *A_diag_i     = jx_CSRMatrixI(A_diag);
   JX_Int             *A_diag_j     = jx_CSRMatrixJ(A_diag);
   jx_CSRMatrix    *A_offd       = jx_ParCSRMatrixOffd(par_matrix);
   JX_Int             *A_offd_i     = jx_CSRMatrixI(A_offd);
   JX_Real          *A_offd_data  = jx_CSRMatrixData(A_offd);
   JX_Int             *A_offd_j     = jx_CSRMatrixJ(A_offd);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             num_cols_offd = jx_CSRMatrixNumCols(A_offd);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *f_local = jx_ParVectorLocalVector(par_rhs);
   JX_Real         *f_data  = jx_VectorData(f_local);

   JX_Real 	  *Vext_data  = NULL;
   JX_Real 	  *v_buf_data = NULL;

   JX_Int             i, j, jr;
   JX_Int             ii, jj;
   JX_Int             relax_error = 0;
   JX_Int             num_sends = 0;
   JX_Int             num_recvs = 0;
   JX_Int             num_procs, my_id, ip, p;
   JX_Int             vec_start, vec_len;
   MPI_Status     *status = NULL;
   MPI_Request    *requests = NULL;

   JX_Real          zero = 0.0;
   JX_Real          res;

   jx_MPI_Comm_size(comm, &num_procs);  
   jx_MPI_Comm_rank(comm, &my_id);  

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created  
   * previously, something will be wrong when the "relax" function  
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */
   
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix); 
   }

      //------------------------------------------------------------------------------//
      //   Gauss-Seidel: relax interior points in parallel, boundary sequentially     //
      //------------------------------------------------------------------------------//
      
         if (num_procs > 1)
         {
            num_sends  = jx_ParCSRCommPkgNumSends(comm_pkg);
            num_recvs  = jx_ParCSRCommPkgNumRecvs(comm_pkg);
            v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
            Vext_data  = jx_CTAlloc(JX_Real, num_cols_offd);
            status     = jx_CTAlloc(MPI_Status, num_recvs+num_sends);
            requests   = jx_CTAlloc(MPI_Request, num_recvs+num_sends);

            if (num_cols_offd)
            {
               A_offd_j = jx_CSRMatrixJ(A_offd);
               A_offd_data = jx_CSRMatrixData(A_offd);
            }
         }

        /*-----------------------------------------------------------------
         * Relax interior points first
         *-----------------------------------------------------------------*/
         if (relax_points == 0)
         {
            for (i = 0; i < n; i ++)	
            {
              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
             
               if ((A_offd_i[i+1]-A_offd_i[i]) == zero && A_diag_data[A_diag_i[i]] != zero)
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i] + 1; jj < A_diag_i[i+1]; jj ++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj]*u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            } // end for i
         }
         else
         {
            for (i = 0; i < n; i ++)
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
             
               if ( cf_marker[i] == relax_points        &&
                    (A_offd_i[i+1]-A_offd_i[i]) == zero && 
                    A_diag_data[A_diag_i[i]] != zero )
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
         }
         
         for (p = 0; p < num_procs; p ++)
         {
            jr = 0;
            if (p != my_id)
            {
               for (i = 0; i < num_sends; i ++)
               {
                  ip = jx_ParCSRCommPkgSendProc(comm_pkg, i);
                  if (ip == p)
                  {
                     vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
                     vec_len = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
                     for (j = vec_start; j < vec_start+vec_len; j ++)
                     {
                        v_buf_data[j] = u_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
                     }
                     jx_MPI_Isend(&v_buf_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[jr++]);
                  }
               }
               jx_MPI_Waitall(jr,requests,status);
               jx_MPI_Barrier(comm);
            }
            else
            {
               if (num_procs > 1)
               {
                  for (i = 0; i < num_recvs; i ++)
                  {
                     ip = jx_ParCSRCommPkgRecvProc(comm_pkg, i);
                     vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
                     vec_len = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
                     jx_MPI_Irecv(&Vext_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[jr++]);
                  }
                  jx_MPI_Waitall(jr,requests,status);
               }
               
               if (relax_points == 0)
               {
                  for (i = 0; i < n; i ++)	
                  {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ((A_offd_i[i+1]-A_offd_i[i]) != zero && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }

              /*-----------------------------------------------------------------
               * Relax only C or F points as determined by relax_points.
               *-----------------------------------------------------------------*/

               else
               {
                  for (i = 0; i < n; i ++)
                  {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ( cf_marker[i] == relax_points        &&
                          (A_offd_i[i+1]-A_offd_i[i]) != zero &&
                          A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }     
               }
               if (num_procs > 1) 
               {
                  jx_MPI_Barrier(comm);
               }
            }
         }
	
         if (num_procs > 1)
         {
            jx_TFree(Vext_data);
            jx_TFree(v_buf_data);
            jx_TFree(status);
            jx_TFree(requests);
         }

   return(relax_error); 
}

/*!
 * \fn JX_Int jx_PAMGRelaxAI2
 */
JX_Int  
jx_PAMGRelaxAI2( jx_ParCSRMatrix *par_matrix,
                 jx_ParVector    *par_rhs,
                 JX_Int             *cf_marker,
                 JX_Int             *relax_marker,
                 JX_Int              relax_points,
                 JX_Real           relax_weight,
                 JX_Real           omega,
                 jx_ParVector    *par_app,
                 jx_ParVector    *Vtemp )
{
   MPI_Comm         comm         = jx_ParCSRMatrixComm(par_matrix);
   jx_CSRMatrix    *A_diag       = jx_ParCSRMatrixDiag(par_matrix);
   JX_Real          *A_diag_data  = jx_CSRMatrixData(A_diag);
   JX_Int             *A_diag_i     = jx_CSRMatrixI(A_diag);
   JX_Int             *A_diag_j     = jx_CSRMatrixJ(A_diag);
   jx_CSRMatrix    *A_offd       = jx_ParCSRMatrixOffd(par_matrix);
   JX_Int             *A_offd_i     = jx_CSRMatrixI(A_offd);
   JX_Real          *A_offd_data  = jx_CSRMatrixData(A_offd);
   JX_Int             *A_offd_j     = jx_CSRMatrixJ(A_offd);
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             num_cols_offd = jx_CSRMatrixNumCols(A_offd);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *f_local = jx_ParVectorLocalVector(par_rhs);
   JX_Real         *f_data  = jx_VectorData(f_local);

   JX_Real 	  *Vext_data  = NULL;
   JX_Real 	  *v_buf_data = NULL;
   
   JX_Int             i, j, jr;
   JX_Int             ii, jj;
   JX_Int             relax_error = 0;
   JX_Int             num_sends = 0;
   JX_Int             num_recvs = 0;
   JX_Int             num_procs, my_id, ip, p;
   JX_Int             vec_start, vec_len;
   MPI_Status     *status = NULL;
   MPI_Request    *requests = NULL;

   JX_Real          zero = 0.0;
   JX_Real          res;

   jx_MPI_Comm_size(comm, &num_procs);  
   jx_MPI_Comm_rank(comm, &my_id);  

  /*-------------------------------------------------------------------------
   * added by peghoty, if the comm_pkg of par_matrix is not created  
   * previously, something will be wrong when the "relax" function  
   * is called on multi-processors occasions.  2009/07/24
   *----------------------------------------------------------------------- */
   
   if (!comm_pkg)
   {
      jx_MatvecCommPkgCreate(par_matrix);
      comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix); 
   }

   
      //------------------------------------------------------------------------------//
      //   Gauss-Seidel: relax interior points in parallel, boundary sequentially     //
      //------------------------------------------------------------------------------//
      
         if (num_procs > 1)
         {
            num_sends  = jx_ParCSRCommPkgNumSends(comm_pkg);
            num_recvs  = jx_ParCSRCommPkgNumRecvs(comm_pkg);
            v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
            Vext_data  = jx_CTAlloc(JX_Real, num_cols_offd);
            status     = jx_CTAlloc(MPI_Status, num_recvs+num_sends);
            requests   = jx_CTAlloc(MPI_Request, num_recvs+num_sends);

            if (num_cols_offd)
            {
               A_offd_j = jx_CSRMatrixJ(A_offd);
               A_offd_data = jx_CSRMatrixData(A_offd);
            }
         }

        /*-----------------------------------------------------------------
         * Relax interior points first
         *-----------------------------------------------------------------*/
         if (relax_points == 0)
         {
            for (i = 0; i < n; i ++)	
            {
              /*-----------------------------------------------------------
               * If diagonal is nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
             
               if ((A_offd_i[i+1]-A_offd_i[i]) == zero && A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i] + 1; jj < A_diag_i[i+1]; jj ++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj]*u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            } // end for i
         }
         else
         {
            for (i = 0; i < n; i ++)
            {

              /*-----------------------------------------------------------
               * If i is of the right type ( C or F ) and diagonal is
               * nonzero, relax point i; otherwise, skip it.
               *-----------------------------------------------------------*/
             
               if ( cf_marker[i] == relax_points        &&
                    (A_offd_i[i+1]-A_offd_i[i]) == zero && 
                    A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
               {
                  res = f_data[i];
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                  {
                     ii = A_diag_j[jj];
                     res -= A_diag_data[jj] * u_data[ii];
                  }
                  u_data[i] = res / A_diag_data[A_diag_i[i]];
               }
            }     
         }
         
         for (p = 0; p < num_procs; p ++)
         {
            jr = 0;
            if (p != my_id)
            {
               for (i = 0; i < num_sends; i ++)
               {
                  ip = jx_ParCSRCommPkgSendProc(comm_pkg, i);
                  if (ip == p)
                  {
                     vec_start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
                     vec_len = jx_ParCSRCommPkgSendMapStart(comm_pkg, i+1)-vec_start;
                     for (j = vec_start; j < vec_start+vec_len; j ++)
                     {
                        v_buf_data[j] = u_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
                     }
                     jx_MPI_Isend(&v_buf_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[jr++]);
                  }
               }
               jx_MPI_Waitall(jr,requests,status);
               jx_MPI_Barrier(comm);
            }
            else
            {
               if (num_procs > 1)
               {
                  for (i = 0; i < num_recvs; i ++)
                  {
                     ip = jx_ParCSRCommPkgRecvProc(comm_pkg, i);
                     vec_start = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i);
                     vec_len = jx_ParCSRCommPkgRecvVecStart(comm_pkg,i+1)-vec_start;
                     jx_MPI_Irecv(&Vext_data[vec_start], vec_len, JX_MPI_REAL, ip, 0, comm, &requests[jr++]);
                  }
                  jx_MPI_Waitall(jr,requests,status);
               }
               
               if (relax_points == 0)
               {
                  for (i = 0; i < n; i ++)	
                  {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ((A_offd_i[i+1]-A_offd_i[i]) != zero && A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }
               }

              /*-----------------------------------------------------------------
               * Relax only C or F points as determined by relax_points.
               *-----------------------------------------------------------------*/

               else
               {
                  for (i = 0; i < n; i ++)
                  {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ( cf_marker[i] == relax_points        &&
                          (A_offd_i[i+1]-A_offd_i[i]) != zero &&
                          A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                     {
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res -= A_diag_data[jj] * u_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] = res / A_diag_data[A_diag_i[i]];
                     }
                  }     
               }
               if (num_procs > 1) 
               {
                  jx_MPI_Barrier(comm);
               }
            }
         }
	
         if (num_procs > 1)
         {
            jx_TFree(Vext_data);
            jx_TFree(v_buf_data);
            jx_TFree(status);
            jx_TFree(requests);
         }
         
   return(relax_error); 
}
