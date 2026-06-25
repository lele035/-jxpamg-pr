/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 *  par_util.c
 *  
 *  Created by peghoty  2010/11/22
 *  Xiangtan University
 *  peghoty@163.com
 *
 */

#include "jx_multils.h"

/*!
 * \fn void fsls_GetWallTime
 * \brief Timing for jx_MPI program
 * \param comm communicator
 * \param *fctname pointer to the name of the function that is timed
 * \param starttime starting time
 * \param endtime ending time
 * \param allproc whether to output the timing of all the processors
 * \param dp DecimalPlace of the output time
 * \author peghoty
 * \date 2010/11/22 
 */ 
void
fsls_GetWallTime( MPI_Comm  comm, 
                  char     *fctname, 
                  JX_Real    starttime, 
                  JX_Real    endtime, 
                  JX_Int       allproc,
                  JX_Int       dp )
{
   JX_Int myid, nprocs;

   JX_Real totaltime    = 0.0;
   JX_Real time_max     = 0.0;
   JX_Real time_min     = 0.0;
   JX_Real time_average = 0.0;
   
   jx_MPI_Comm_rank(comm, &myid);      
   jx_MPI_Comm_size(comm, &nprocs);
   
   totaltime = endtime - starttime;

   if (allproc)
   {   
      switch (dp)
      {
         case 1:
         jx_printf(" >> %s: Proc %d took %.1f seconds!\n", fctname, myid, totaltime); 
         break;
         case 2:
         jx_printf(" >> %s: Proc %d took %.2f seconds!\n", fctname, myid, totaltime); 
         break;
         case 3:
         jx_printf(" >> %s: Proc %d took %.3f seconds!\n", fctname, myid, totaltime); 
         break;
         case 4:
         jx_printf(" >> %s: Proc %d took %.4f seconds!\n", fctname, myid, totaltime); 
         break;
         case 5:
         jx_printf(" >> %s: Proc %d took %.5f seconds!\n", fctname, myid, totaltime); 
         break; 
         case 6:
         jx_printf(" >> %s: Proc %d took %.6f seconds!\n", fctname, myid, totaltime); 
         break;                                  
         default:
         jx_printf(" >> %s: Proc %d took %.2f seconds!\n", fctname, myid, totaltime); 
         break;
      }    
   }
   jx_MPI_Reduce(&totaltime, &time_min,     1, JX_MPI_REAL, MPI_MIN, 0, comm);
   jx_MPI_Reduce(&totaltime, &time_max,     1, JX_MPI_REAL, MPI_MAX, 0, comm);
   jx_MPI_Reduce(&totaltime, &time_average, 1, JX_MPI_REAL, MPI_SUM, 0, comm);
   if (myid == 0)
   {
      switch (dp)
      {
         case 1:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.1f, %.1f, %.1f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 2:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.2f, %.2f, %.2f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 3:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.3f, %.3f, %.3f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 4:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.4f, %.4f, %.4f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 5:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.5f, %.5f, %.5f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 6:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.6f, %.6f, %.6f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;                           
         default:
         jx_printf("\n >> %s: \n    time(min,max,ave) = (%.2f, %.2f, %.2f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
      }                
   } 
}

/*!
 * \fn void fsls_MPICommInformation
 * \brief Obtain the information of the jx_MPI system, including myid and nprocs
 * \param comm communicator
 * \param *myid_ptr pointer to the processor id
 * \param *nprocs_ptr pointer to the total number of processors
 * \author peghoty
 * \date 2010/11/22 
 */ 
void
fsls_MPICommInformation( MPI_Comm comm, JX_Int *myid_ptr, JX_Int *nprocs_ptr )
{
   JX_Int myid,nprocs;
   
   jx_MPI_Comm_rank(MPI_COMM_WORLD, &myid);      
   jx_MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   
   *myid_ptr = myid;
   *nprocs_ptr = nprocs;
}

/*!
 * \fn JX_Int fsls_BalancedPartition
 * \brief Divid N into np parts load-balancinglly.
 * \param N the dividend
 * \param np the divisor
 * \param *partition_ptr pointer to the partition
 * \param *each how many in each?
 * \author peghoty
 * \date 2010/11/22 
 */ 
JX_Int   
fsls_BalancedPartition( JX_Int   N, 
                        JX_Int   np, 
                        JX_Int **partition_ptr, 
                        JX_Int **each_ptr )
{
   JX_Int  i, ierr = 0;
   JX_Int  base;
   JX_Int  threshold;   
   
   JX_Int *partition = NULL;
   JX_Int *each      = NULL;
   
   partition = fsls_CTAlloc(JX_Int, np+1);
   each      = fsls_CTAlloc(JX_Int, np);
   
   base      = N / np;
   threshold = N % np;
      
   partition[0] = 0;
   for (i = 0; i < np; i ++)
   {
      if (i < threshold)
      {
         each[i] = base + 1;
      }
      else
      {
         each[i] = base;
      }
      partition[i+1] = partition[i] + each[i];
   }
   
   *partition_ptr = partition;
   *each_ptr      = each;
   
   return (ierr);
}
