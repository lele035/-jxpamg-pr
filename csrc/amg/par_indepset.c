//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_indepset.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGIndepSetInit
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGIndepSetInit( jx_ParCSRMatrix *par_S,
                     JX_Real          *measure_array,
                     JX_Int              seq_rand ) 
{
   jx_CSRMatrix *S_diag      = jx_ParCSRMatrixDiag(par_S);
   MPI_Comm      comm        = jx_ParCSRMatrixComm(par_S);
   JX_Int           S_num_nodes = jx_CSRMatrixNumRows(S_diag);
   JX_Int           i, my_id;
   JX_Int           ierr = 0;

   jx_MPI_Comm_rank(comm, &my_id);
   
   i = 2747 + my_id;
   
   if (seq_rand)
   {
      i = 2747;
   }
   
   jx_SeedRand(i);
   
   if (seq_rand)
   {
      for (i = 0; i < jx_ParCSRMatrixFirstRowIndex(par_S); i ++)
      {
         jx_Rand();
      }
   }
   
   for (i = 0; i < S_num_nodes; i ++)
   {
      measure_array[i] += jx_Rand();
   }

   return (ierr);
}


/*!
 * \fn JX_Int jx_PAMGIndepSet
 * \date 2011/09/03
 */ 
JX_Int
jx_PAMGIndepSet( jx_ParCSRMatrix *par_S,
                 JX_Real          *measure_array,
                 JX_Int             *graph_array,
                 JX_Int              graph_array_size,
                 JX_Int             *graph_array_offd,
                 JX_Int              graph_array_offd_size,
                 JX_Int             *IS_marker,
                 JX_Int             *IS_marker_offd )
{
   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(par_S);
   JX_Int          *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int          *S_diag_j = jx_CSRMatrixJ(S_diag);
   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(par_S);
   JX_Int          *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j = NULL;

   JX_Int           local_num_vars = jx_CSRMatrixNumRows(S_diag);
   JX_Int           i, j, ig, jS, jj;
                   
   JX_Int           ierr = 0;

  /*-------------------------------------------------------
   * Initialize IS_marker by putting all nodes in
   * the independent set.
   *-------------------------------------------------------*/

   if (jx_CSRMatrixNumCols(S_offd))
   {
      S_offd_j = jx_CSRMatrixJ(S_offd);
   }

   for (ig = 0; ig < graph_array_size; ig ++)
   {
      i = graph_array[ig];
      if (measure_array[i] > 1)
      {
         IS_marker[i] = 1;
      }
   }
   
   for (ig = 0; ig < graph_array_offd_size; ig ++)
   {
      i = graph_array_offd[ig];
      if (measure_array[i+local_num_vars] > 1)
      {
         IS_marker_offd[i] = 1;
      }
   }

  /*-------------------------------------------------------
   * Remove nodes from the initial independent set
   *-------------------------------------------------------*/

   for (ig = 0; ig < graph_array_size; ig ++)
   {
      i = graph_array[ig];
      
      if (measure_array[i] > 1)
      {
      
         for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS ++)
         {
            j = S_diag_j[jS];
	    if (j < 0) 
	    {
	       j = - j - 1;
	    }
            
            /* only consider valid graph edges */
            if (measure_array[j] > 1) 
            {
               if (measure_array[i] > measure_array[j])
               {
                  IS_marker[j] = 0;
               }
               else if (measure_array[j] > measure_array[i])
               {
                  IS_marker[i] = 0;
               }
            }
         }
         
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS ++)
         {
            jj = S_offd_j[jS];
            if (jj < 0) 
            {
               jj = - jj - 1;
            }
            j = local_num_vars + jj;
            
            /* only consider valid graph edges */
            if (measure_array[j] > 1) 
            {
               if (measure_array[i] > measure_array[j])
               {
                  IS_marker_offd[jj] = 0;
               }
               else if (measure_array[j] > measure_array[i])
               {
                  IS_marker[i] = 0;
               }
            }
         }
      }
   }
            
   return (ierr);
}
