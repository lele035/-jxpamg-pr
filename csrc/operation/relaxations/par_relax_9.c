//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax_9.c
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_PAMGRelax9
 * \brief Direct Solve
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelax9( jx_ParCSRMatrix *par_matrix,
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
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n_global = jx_ParCSRMatrixGlobalNumRows(par_matrix);
   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   JX_Int             first_index   = jx_ParVectorFirstIndex(par_app);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_CSRMatrix    *A_CSR;
   JX_Int             *A_CSR_i;   
   JX_Int             *A_CSR_j;
   JX_Real          *A_CSR_data;
   
   jx_Vector       *f_vector;
   JX_Real          *f_vector_data;

   JX_Int             i;
   JX_Int             jj;
   JX_Int             column;
   JX_Int             relax_error = 0;
   JX_Int             num_procs, my_id;

   JX_Real         *A_mat;
   JX_Real         *b_vec;

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
      //                   Direct solve: use gaussian elimination                     //
      //------------------------------------------------------------------------------//

        /*-----------------------------------------------------------------
         *  Generate CSR matrix from ParCSRMatrix par_matrix
         *-----------------------------------------------------------------*/
#ifdef JX_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
         f_vector = jx_ParVectorToVectorAll(par_rhs);
	 if (n)
	 {
#else
	 if (n)
	 {
	    A_CSR = jx_ParCSRMatrixToCSRMatrixAll(par_matrix);
	    f_vector = jx_ParVectorToVectorAll(par_rhs);
#endif
 	    A_CSR_i = jx_CSRMatrixI(A_CSR);
 	    A_CSR_j = jx_CSRMatrixJ(A_CSR);
 	    A_CSR_data = jx_CSRMatrixData(A_CSR);
   	    f_vector_data = jx_VectorData(f_vector);
            
            A_mat = jx_CTAlloc(JX_Real, n_global*n_global);
            
            b_vec = jx_CTAlloc(JX_Real, n_global);    

           /*---------------------------------------------------------------
            *  Load CSR matrix into A_mat.
            *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i ++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj ++)
               {
                  column = A_CSR_j[jj];
                  A_mat[i*n_global+column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            relax_error = jx_gselim(A_mat,b_vec,n_global);
            /* use version with pivoting */
            /* relax_error = jx_gselim_piv(A_mat,b_vec,n_global);*/

            for (i = 0; i < n; i ++)
            {
               u_data[i] = b_vec[first_index+i];
            }

	    jx_TFree(A_mat); 
            jx_TFree(b_vec);
            jx_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            jx_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         
         }
#ifdef JX_NO_GLOBAL_PARTITION
         else
         {
            jx_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            jx_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

   return(relax_error); 
}
