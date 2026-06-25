//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_indepsetrlx.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGIndepSetRelaxInit
 * \date 2011/09/03
 */
JX_Int
jx_PAMGIndepSetRelaxInit( jx_ParCSRMatrix *S,
                          JX_Real          *measure_array )
{
   jx_CSRMatrix    *S_diag = jx_ParCSRMatrixDiag(S);
   JX_Int              S_num_nodes = jx_CSRMatrixNumRows(S_diag);
   JX_Int              first_index = jx_ParCSRMatrixFirstRowIndex(S);
   JX_Int              i;
   JX_Int              ierr = 0;

   jx_SeedRand(2747);
   
   for (i = 0; i < first_index; i ++)
   {
      jx_Rand();
   }
   
   for (i = 0; i < S_num_nodes; i ++)
   {
      measure_array[i] += jx_Rand();
   }

   return (ierr);
}


/*!
 * \fn JX_Int jx_PAMGIndepSetRelax
 * \date 2011/09/03
 */
JX_Int
jx_PAMGIndepSetRelax( jx_ParCSRMatrix *S,
		      jx_CSRMatrix    *S_ext,
                      JX_Real          *measure_array,
                      JX_Int             *graph_array,
                      JX_Int              graph_array_size,
                      JX_Int             *graph_array_offd,
                      JX_Int              graph_array_offd_size,
                      JX_Int             *IS_marker,
                      JX_Int             *IS_marker_offd,
                      JX_Int              nc_per,
                      JX_Real           measure_th )
{
   jx_CSRMatrix *S_diag   = jx_ParCSRMatrixDiag(S);
   JX_Int          *S_diag_i = jx_CSRMatrixI(S_diag);
   JX_Int          *S_diag_j = jx_CSRMatrixJ(S_diag);
   jx_CSRMatrix *S_offd   = jx_ParCSRMatrixOffd(S);
   JX_Int          *S_offd_i = jx_CSRMatrixI(S_offd);
   JX_Int          *S_offd_j = NULL;
   JX_Int          *S_ext_i  = NULL;
   JX_Int          *S_ext_j  = NULL;

   JX_Int		 local_num_vars = jx_CSRMatrixNumRows(S_diag);
   JX_Int		 jc;
   JX_Int           i, j, ig, jS, jj; 
   JX_Int           num_per;
                   
   JX_Int           ierr = 0;

  /*-------------------------------------------------------
   * Initialize IS_marker by putting all nodes in
   * the independent set.
   *-------------------------------------------------------*/

   if (jx_CSRMatrixNumCols(S_offd))
   {
      S_offd_j = jx_CSRMatrixJ(S_offd);
      S_ext_i  = jx_CSRMatrixI(S_ext);
      S_ext_j  = jx_CSRMatrixJ(S_ext);
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
      if (measure_array[i] < measure_th) 
      {
      	 IS_marker[i] = 0;      	
      }
      else if (measure_array[i] > 1)
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
         } // end for jS
      }
   } // end for ig
   
   
   for (ig = 0; ig < graph_array_offd_size; ig ++)
   {
      i = graph_array_offd[ig];
      if (measure_array[local_num_vars+i] < measure_th) 
      {
      	 IS_marker_offd[i] = 0;
      }
      else if (measure_array[local_num_vars+i] > 1)
      {
         for (jS = S_ext_i[i]; jS < S_ext_i[i+1]; jS ++)
         {
            j = S_ext_j[jS];
            
            if (j >= 0)
	    {
               /* only consider valid graph edges */
               if (measure_array[j] > 1) 
               {
                  if (measure_array[i+local_num_vars] > measure_array[j])
                  {
                     IS_marker[j] = 0;
                  }
                  else if (measure_array[j] > measure_array[i+local_num_vars])
                  {
                     IS_marker_offd[i] = 0;
                  }
               }
	    }
	    else
	    {
	       jc = -j - 1 + local_num_vars;
	     
               /* only consider valid graph edges */
               if (measure_array[jc] > 1)
               {
                  if (measure_array[i+local_num_vars] > measure_array[jc])
                  {
                     IS_marker_offd[-j-1] = 0;
                  }
                  else if (measure_array[jc] > measure_array[i+local_num_vars])
                  {
                     IS_marker_offd[i] = 0;
                  }
               }
	    }
         }
      }
   }

   num_per = 0;
   for (ig = 0; ig < graph_array_size; ig ++)
   {
      i = graph_array[ig];
      if (IS_marker[i] == 1) 
      {
         num_per ++;
      }
   }
   
   while (num_per > nc_per)
   {       
       for (ig = graph_array_size-1; ig >= 0; ig --)
       {
       	  i = graph_array[ig];
       	  if (IS_marker[i] == 1) 
       	  {
       	      IS_marker[i] = 0;
       	      num_per --;
       	      if (num_per <= nc_per) 
       	      {
       	         break;
       	      }
       	  }
       }
       break;
   }
 
   return (ierr);
}     
