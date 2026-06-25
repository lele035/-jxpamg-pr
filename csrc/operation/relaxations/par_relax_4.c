//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax_4.c
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_PAMGRelax4
 * \brief hybrid: SOR-J mix off-processor, SOR on-processor
          with outer relaxation parameters (backward solve)
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelax4( jx_ParCSRMatrix *par_matrix,
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
   jx_ParCSRCommHandle *comm_handle = NULL;

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             num_cols_offd = jx_CSRMatrixNumCols(A_offd);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *f_local = jx_ParVectorLocalVector(par_rhs);
   JX_Real         *f_data  = jx_VectorData(f_local);

   jx_Vector      *Vtemp_local = jx_ParVectorLocalVector(Vtemp);
   JX_Real         *Vtemp_data  = jx_VectorData(Vtemp_local);
   JX_Real 	  *Vext_data  = NULL;
   JX_Real 	  *v_buf_data = NULL;
   JX_Real 	  *tmp_data   = NULL;

   JX_Int             i, j;
   JX_Int             ii, jj;
   JX_Int             ns, ne, size, rest;
   JX_Int             relax_error = 0;
   JX_Int             num_sends = 0;
   JX_Int             index, start;
   JX_Int             num_procs, num_threads, my_id;

   JX_Real          zero = 0.0;
   JX_Real          res, res0, res2;
   JX_Real          one_minus_omega;
   JX_Real          prod;

   one_minus_omega  = 1.0 - omega;

   jx_MPI_Comm_size(comm, &num_procs);  
   jx_MPI_Comm_rank(comm, &my_id);  
   num_threads = jx_NumThreads();

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
      //  Hybrid: Jacobi off-processor, Gauss-Seidel/SOR on-processor (backward loop) //
      //------------------------------------------------------------------------------//

         if (num_procs > 1)
         {
            num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
            Vext_data = jx_CTAlloc(JX_Real,num_cols_offd);
        
            if (num_cols_offd)
            {
               A_offd_j = jx_CSRMatrixJ(A_offd);
               A_offd_data = jx_CSRMatrixData(A_offd);
            }
 
            index = 0;
            for (i = 0; i < num_sends; i ++)
            {
               start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j ++)
               {
                  v_buf_data[index ++] = u_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }
 
            comm_handle = jx_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);

            jx_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
         }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

         if (relax_weight == 1 && omega == 1)
         {
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real,n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n / num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     
                     for (i = ne-1; i > ns-1; i--)	/* interior points first */
                     {

                       /*-----------------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *----------------------------------------------------------------*/
             
                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
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
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i--)  /* interior points first */
                  {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ( A_diag_data[A_diag_i[i]] != zero)
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
            }

           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real,n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n / num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size + j;
                        ne = (j+1)*size + j + 1;
                     }
                     else
                     {
                        ns = j*size + rest;
                        ne = (j+1)*size + rest;
                     }
                  
                     for (i = ne-1; i > ns-1; i --)  /* relax interior points */
                     {

                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
             
                        if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
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
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i --)  /* relax interior points */
                  {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
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
            }
         }
         else
         {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
            for (i = 0; i < n; i ++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real, n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                  
                     for (i = ne-1; i > ns-1; i --)  /* interior points first */
                     {

                       /*-----------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
             
                        if ( A_diag_data[A_diag_i[i]] != zero)
                        {
                           res = f_data[i];
                           res0 = 0.0;
                           res2 = 0.0;
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                        
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) 
                                        / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i --)  /* interior points first */
                  {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ( A_diag_data[A_diag_i[i]] != zero)
                     {
                        res0 = 0.0;
                        res2 = 0.0;
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }

           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real, n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n / num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size + j;
                        ne = (j+1)*size + j + 1;
                     }
                     else
                     {
                        ns = j*size + rest;
                        ne = (j+1)*size + rest;
                     }
                  
                     for (i = ne-1; i > ns-1; i --)  /* relax interior points */
                     {

                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
             
                        if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
		              if (ii >= ns && ii < ne)
		              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
		              }
		              else
		              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                        
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) 
                                        / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero)
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }
         }

         if (num_procs > 1)
         {
            jx_TFree(Vext_data);
            jx_TFree(v_buf_data);
         }

   return(relax_error); 
}

/*!
 * \fn JX_Int jx_PAMGRelaxAI4
 */
JX_Int  
jx_PAMGRelaxAI4( jx_ParCSRMatrix *par_matrix,
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
   jx_ParCSRCommHandle *comm_handle = NULL;

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             num_cols_offd = jx_CSRMatrixNumCols(A_offd);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *f_local = jx_ParVectorLocalVector(par_rhs);
   JX_Real         *f_data  = jx_VectorData(f_local);

   jx_Vector      *Vtemp_local = jx_ParVectorLocalVector(Vtemp);
   JX_Real         *Vtemp_data  = jx_VectorData(Vtemp_local);
   JX_Real 	  *Vext_data  = NULL;
   JX_Real 	  *v_buf_data = NULL;
   JX_Real 	  *tmp_data   = NULL;
   
   JX_Int             i, j;
   JX_Int             ii, jj;
   JX_Int             ns, ne, size, rest;
   JX_Int             relax_error = 0;
   JX_Int             num_sends = 0;
   JX_Int             index, start;
   JX_Int             num_procs, num_threads, my_id;

   JX_Real          zero = 0.0;
   JX_Real          res, res0, res2;
   JX_Real          one_minus_omega;
   JX_Real          prod;

   one_minus_omega  = 1.0 - omega;

   jx_MPI_Comm_size(comm, &num_procs);  
   jx_MPI_Comm_rank(comm, &my_id);  
   num_threads = jx_NumThreads();

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
      //  Hybrid: Jacobi off-processor, Gauss-Seidel/SOR on-processor (backward loop) //
      //------------------------------------------------------------------------------//
      
         if (num_procs > 1)
         {
            num_sends = jx_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = jx_CTAlloc(JX_Real, jx_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
            Vext_data = jx_CTAlloc(JX_Real,num_cols_offd);
        
            if (num_cols_offd)
            {
               A_offd_j = jx_CSRMatrixJ(A_offd);
               A_offd_data = jx_CSRMatrixData(A_offd);
            }
 
            index = 0;
            for (i = 0; i < num_sends; i ++)
            {
               start = jx_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j = start; j < jx_ParCSRCommPkgSendMapStart(comm_pkg,i+1); j ++)
               {
                  v_buf_data[index ++] = u_data[jx_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
               }
            }
 
            comm_handle = jx_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data, Vext_data);

            jx_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
         }

        /*-----------------------------------------------------------------
         * Relax all points.
         *-----------------------------------------------------------------*/

         if (relax_weight == 1 && omega == 1)
         {
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real,n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n / num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     
                     for (i = ne-1; i > ns-1; i--)	/* interior points first */
                     {

                       /*-----------------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *----------------------------------------------------------------*/
             
                        if ( A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
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
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i--)  /* interior points first */
                  {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ( A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
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
            }

           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real,n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n / num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size + j;
                        ne = (j+1)*size + j + 1;
                     }
                     else
                     {
                        ns = j*size + rest;
                        ne = (j+1)*size + rest;
                     }
                  
                     for (i = ne-1; i > ns-1; i --)  /* relax interior points */
                     {

                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
             
                        if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                        {
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res -= A_diag_data[jj] * u_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
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
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i --)  /* relax interior points */
                  {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
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
            }
         }
         else
         {
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
            for (i = 0; i < n; i ++)
            {
               Vtemp_data[i] = u_data[i];
            }
            prod = (1.0-relax_weight*omega);
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real, n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                  
                     for (i = ne-1; i > ns-1; i --)  /* interior points first */
                     {

                       /*-----------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
             
                        if ( A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                        {
                           res = f_data[i];
                           res0 = 0.0;
                           res2 = 0.0;
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
                              }
                              else
                              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                        
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) 
                                        / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i --)  /* interior points first */
                  {

                    /*-----------------------------------------------------------
                     * If diagonal is nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if ( A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                     {
                        res0 = 0.0;
                        res2 = 0.0;
                        res = f_data[i];
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }

           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/

            else
            {
               if (num_threads > 1)
               {
                  tmp_data = jx_CTAlloc(JX_Real, n);
#define JX_SMP_PRIVATE i
#include "../../../include/jx_smp_forloop.h"
                  for (i = 0; i < n; i ++)
                  {
                     tmp_data[i] = u_data[i];
                  }
#define JX_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../../../include/jx_smp_forloop.h"
                  for (j = 0; j < num_threads; j ++)
                  {
                     size = n / num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size + j;
                        ne = (j+1)*size + j + 1;
                     }
                     else
                     {
                        ns = j*size + rest;
                        ne = (j+1)*size + rest;
                     }
                  
                     for (i = ne-1; i > ns-1; i --)  /* relax interior points */
                     {

                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
             
                        if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                        {
                           res0 = 0.0;
                           res2 = 0.0;
                           res = f_data[i];
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                           {
                              ii = A_diag_j[jj];
		              if (ii >= ns && ii < ne)
		              {
                                 res0 -= A_diag_data[jj] * u_data[ii];
                                 res2 += A_diag_data[jj] * Vtemp_data[ii];
		              }
		              else
		              {
                                 res -= A_diag_data[jj] * tmp_data[ii];
                              }
                           }
                        
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                           {
                              ii = A_offd_j[jj];
                              res -= A_offd_data[jj] * Vext_data[ii];
                           }
                           u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) 
                                        / A_diag_data[A_diag_i[i]];
                        }
                     }
                  }
                  jx_TFree(tmp_data);
               }
               else
               {
                  for (i = n-1; i > -1; i--) /* relax interior points */
                  {

                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
             
                     if (cf_marker[i] == relax_points && A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
                     {
                        res = f_data[i];
                        res0 = 0.0;
                        res2 = 0.0;
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj ++)
                        {
                           ii = A_diag_j[jj];
                           res0 -= A_diag_data[jj] * u_data[ii];
                           res2 += A_diag_data[jj] * Vtemp_data[ii];
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj ++)
                        {
                           ii = A_offd_j[jj];
                           res -= A_offd_data[jj] * Vext_data[ii];
                        }
                        u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 + one_minus_omega*res2) / A_diag_data[A_diag_i[i]];
                     }
                  }
               }
            }
         }

         if (num_procs > 1)
         {
            jx_TFree(Vext_data);
            jx_TFree(v_buf_data);
         }

   return(relax_error); 
}
