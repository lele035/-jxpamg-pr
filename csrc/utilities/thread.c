//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

#if JX_USING_OPENMP || defined (JX_USING_PGCC_SMP)

/*!
 * \fn JX_Int jx_NumThreads
 * \brief Get the number of all the threads
 * \date 2012/09/04
 */
JX_Int
jx_NumThreads()
{
   JX_Int num_threads;
   
#if JX_USING_OPENMP
#pragma omp parallel
   num_threads = omp_get_num_threads();
#endif
#ifdef JX_USING_PGCC_SMP
   num_threads = 2;
#endif
   
   return num_threads;
}

/* This next function must be called from within a parallel region! */

JX_Int
jx_NumActiveThreads()
{
   JX_Int num_threads;
   
   num_threads = omp_get_num_threads();
   
   return num_threads;
}

/*!
 * \fn JX_Int jx_GetThreadNum
 * \brief Get the id of the current thread
 * \date 2012/09/04
 */
JX_Int
jx_GetThreadNum()
{
   JX_Int my_thread_num;
   
#if JX_USING_OPENMP
   my_thread_num = omp_get_thread_num();
#endif
#ifdef JX_USING_PGCC_SMP
   /* THIS NEEDS TO BE FIXED */
   my_thread_num = 0;
#endif
   
   return my_thread_num;
}

#endif
