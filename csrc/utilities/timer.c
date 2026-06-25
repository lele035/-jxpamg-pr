//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

/*!
 * \fn JX_Real jx_time_getWallclockSeconds
 * \brief Timing function.
 * \date 2011/09/02 
 */ 
JX_Real 
jx_time_getWallclockSeconds(void)
{
#ifdef TIMER_USE_jx_MPI
   return(jx_MPI_Wtime());
#else
#ifdef WIN32
   clock_t cl=clock();
   return(((JX_Real) cl)/((JX_Real) CLOCKS_PER_SEC));
#else
   struct tms usage;
   long wallclock = times(&usage);
   return(((JX_Real) wallclock)/((JX_Real) sysconf(_SC_CLK_TCK)));
#endif
#endif
}

/*!
 * \fn void jx_GetWallTime
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
jx_GetWallTime( MPI_Comm  comm, 
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
         jx_printf("\n >> %s: time(min,max,ave) = (%.1f, %.1f, %.1f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 2:
         jx_printf("\n >> %s: time(min,max,ave) = (%.2f, %.2f, %.2f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 3:
         jx_printf("\n >> %s: time(min,max,ave) = (%.3f, %.3f, %.3f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 4:
         jx_printf("\n >> %s: time(min,max,ave) = (%.4f, %.4f, %.4f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 5:
         jx_printf("\n >> %s: time(min,max,ave) = (%.5f, %.5f, %.5f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
         case 6:
         jx_printf("\n >> %s: time(min,max,ave) = (%.6f, %.6f, %.6f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;                           
         default:
         jx_printf("\n >> %s: time(min,max,ave) = (%.2f, %.2f, %.2f) seconds\n\n", 
                fctname, time_min, time_max, time_average / nprocs);
         break;
      }                
   } 
}

/*!
 * \fn void jx_GetWallTimeMax
 * \brief Timing for jx_MPI program(get the maximal time of all processors)
 * \param comm communicator
 * \param starttime starting time
 * \param endtime ending time
 * \return the maximal of (endtime - starttime) among all the processors
 * \author peghoty
 * \date 2012/03/25 
 */ 
JX_Real
jx_GetWallTimeMax( MPI_Comm  comm, 
                   JX_Real    starttime, 
                   JX_Real    endtime )
{
   JX_Int myid, nprocs;

   JX_Real totaltime    = 0.0;
   JX_Real time_max     = 0.0;
   
   jx_MPI_Comm_rank(comm, &myid);      
   jx_MPI_Comm_size(comm, &nprocs);
   
   totaltime = endtime - starttime;

   jx_MPI_Reduce(&totaltime, &time_max, 1, JX_MPI_REAL, MPI_MAX, 0, comm);

   return (time_max);
}
