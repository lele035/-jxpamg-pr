//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_relax_7.c
 *  Date: 2011/09/03
 */ 

#include "jx_mv.h"

/*!
 * \fn JX_Int jx_PAMGRelax7
 * \brief Jacobi (uses Matvec), only needed in CGNR
 * \date 2011/09/03
 */
JX_Int  
jx_PAMGRelax7( jx_ParCSRMatrix *par_matrix,
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
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *Vtemp_local = jx_ParVectorLocalVector(Vtemp);
   JX_Real         *Vtemp_data  = jx_VectorData(Vtemp_local);

   JX_Int             i;
   JX_Int             relax_error = 0;
   JX_Int             num_procs, my_id;

   JX_Real          zero = 0.0;

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
      //                       Jacobi (uses ParMatvec)                                //
      //------------------------------------------------------------------------------//

        /*-----------------------------------------------------------------
         * Copy par_rhs into temporary vector.
         *-----------------------------------------------------------------*/

         jx_ParVectorCopy(par_rhs,Vtemp);

        /*-----------------------------------------------------------------
         * Perform Matvec Vtemp = par_rhs - A u
         *-----------------------------------------------------------------*/

         jx_ParCSRMatrixMatvec(-1.0,par_matrix, par_app, 1.0, Vtemp);
         for (i = 0; i < n; i ++)
         {
           /*-----------------------------------------------------------
            * If diagonal is nonzero, relax point i; otherwise, skip it.
            *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero)
            {
               u_data[i] += relax_weight * Vtemp_data[i] / A_diag_data[A_diag_i[i]];
            }
         }

   return(relax_error); 
}

/*!
 * \fn JX_Int jx_PAMGRelaxAI7
 */
JX_Int  
jx_PAMGRelaxAI7( jx_ParCSRMatrix *par_matrix,
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
   jx_ParCSRCommPkg    *comm_pkg = jx_ParCSRMatrixCommPkg(par_matrix);

   JX_Int             n        = jx_CSRMatrixNumRows(A_diag);
   
   jx_Vector      *u_local = jx_ParVectorLocalVector(par_app);
   JX_Real         *u_data  = jx_VectorData(u_local);

   jx_Vector      *Vtemp_local = jx_ParVectorLocalVector(Vtemp);
   JX_Real         *Vtemp_data  = jx_VectorData(Vtemp_local);
   
   JX_Int             i;
   JX_Int             relax_error = 0;
   JX_Int             num_procs, my_id;

   JX_Real          zero = 0.0;

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
      //                       Jacobi (uses ParMatvec)                                //
      //------------------------------------------------------------------------------//
      
        /*-----------------------------------------------------------------
         * Copy par_rhs into temporary vector.
         *-----------------------------------------------------------------*/

         jx_ParVectorCopy(par_rhs,Vtemp);

        /*-----------------------------------------------------------------
         * Perform Matvec Vtemp = par_rhs - A u
         *-----------------------------------------------------------------*/

         jx_ParCSRMatrixMatvec(-1.0, par_matrix, par_app, 1.0, Vtemp);
         for (i = 0; i < n; i ++)
         {
           /*-----------------------------------------------------------
            * If diagonal is nonzero, relax point i; otherwise, skip it.
            *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero && relax_marker[i] == 1 )
            {
               u_data[i] += relax_weight * Vtemp_data[i] / A_diag_data[A_diag_i[i]];
            }
         }
         
   return(relax_error); 
}
