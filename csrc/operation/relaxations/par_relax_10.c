//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax_10.c
 *  Date: 2021/06/29
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_PAMGRelax10
 * \brief Direct Solved by PARDISO directly
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelax10( jx_ParCSRMatrix *par_matrix,
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
   JX_Int             k;
   JX_Int             column;
   JX_Int             relax_error = 0;
   JX_Int             num_procs, my_id;


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
   //                   Direct solve: use PARDISO                                  //
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

      // PARDISO
      // printf("jx_qsort1, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      for ( i = 0; i < A_CSR->num_rows; i++)
      {
         k = A_CSR->i[i];
         jx_qsort1( &A_CSR->j[k], &A_CSR->data[k], 0, A_CSR->i[i+1] - A_CSR->i[i] - 1 );
      }
      // printf("jx_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);
      jx_pardiso(A_CSR, f_vector, u_local);
      // printf("jx_pardiso, %s, %s, %d\n", __FILE__, __FUNCTION__, __LINE__);

 	   // free 
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
