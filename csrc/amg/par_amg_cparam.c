//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  par_amg_cparam.c
 *  Date: 2011/09/03
 */ 

#include "jx_pamg.h"

/*!
 * \fn JX_Int jx_PAMGCoarseParms
 * \date 2011/09/03
 */
JX_Int
jx_PAMGCoarseParms( MPI_Comm   comm,
                    JX_Int        local_num_variables,
                    JX_Int        num_functions,
                    JX_Int       *dof_func,
                    JX_Int       *CF_marker, 
                    JX_Int      **coarse_dof_func_ptr, 
                    JX_Int      **coarse_pnts_global_ptr) 
{
   JX_Int  i;
   JX_Int  ierr = 0;
   JX_Int	num_procs;
   JX_Int  local_coarse_size = 0;

   JX_Int *coarse_dof_func;
   JX_Int *coarse_pnts_global;

   jx_MPI_Comm_size(comm, &num_procs);

   for (i = 0; i < local_num_variables; i ++)
   {
      if (CF_marker[i] == 1) 
      {
         local_coarse_size++;
      }
   }
   
   if (num_functions > 1)
   {
      coarse_dof_func = jx_CTAlloc(JX_Int, local_coarse_size);
      local_coarse_size = 0;
      for (i = 0; i < local_num_variables; i ++)
      {
         if (CF_marker[i] == 1)
         {
            coarse_dof_func[local_coarse_size++] = dof_func[i];
         }
      }
      *coarse_dof_func_ptr = coarse_dof_func;
   }

#ifdef JX_NO_GLOBAL_PARTITION
   JX_Int scan_recv;
   coarse_pnts_global = jx_CTAlloc(JX_Int, 2);
   jx_MPI_Scan(&local_coarse_size, &scan_recv, 1, JX_MPI_INT, MPI_SUM, comm);
   /* first point in my range */ 
   coarse_pnts_global[0] = scan_recv - local_coarse_size;
   /* first point in next proc's range */
   coarse_pnts_global[1] = scan_recv;  
#else
   coarse_pnts_global = jx_CTAlloc(JX_Int, num_procs + 1);
   jx_MPI_Allgather(&local_coarse_size, 1, JX_MPI_INT, &coarse_pnts_global[1], 1, JX_MPI_INT, comm);
   for (i = 2; i < num_procs + 1; i ++)
   {
      coarse_pnts_global[i] += coarse_pnts_global[i-1];
   }
#endif

   *coarse_pnts_global_ptr = coarse_pnts_global;

   return (ierr);
}
