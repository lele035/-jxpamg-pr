//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_smp_forloop.h -- Wrapper code for SMP compiler directives.  Translates 
 *  jx SMP directives into the appropriate Open MP,
 *  IBM, SGI, or pgcc (Red) SMP compiler directives
 *  Date: 2011/09/08
 *  Modified Date: 2012/10/22
 *
 *  Created by peghoty
 *  Modified by Yue Xiaoqiang
 */

#ifndef JX_SMP_PRIVATE
#define JX_SMP_PRIVATE
#endif

/* OpenMP */
#if JX_USING_OPENMP
#ifndef JX_SMP_REDUCTION_OP
#ifndef JX_SMP_PAR_REGION
#ifndef JX_SMP_FOR
#pragma omp parallel for private(JX_SMP_PRIVATE) schedule(static)
#endif
#endif
#endif
#ifdef JX_SMP_PAR_REGION
#pragma omp parallel private(JX_SMP_PRIVATE)
#endif
#ifdef JX_SMP_FOR
#pragma omp for schedule(static)
#endif
#ifdef JX_SMP_REDUCTION_OP
#pragma omp parallel for private(JX_SMP_PRIVATE) \
reduction(JX_SMP_REDUCTION_OP: JX_SMP_REDUCTION_VARS) \
schedule(static)
#endif
#endif

/* SGI */
#ifdef JX_USING_SGI_SMP
#pragma parallel
#pragma pfor
#pragma schedtype(gss)
#pragma chunksize(10)
#endif

/* IBM */
#ifdef JX_USING_IBM_SMP
#pragma parallel_loop
#pragma schedule (guided,10)
#endif

/* PGCC */
#ifdef JX_USING_PGCC_SMP
#ifndef JX_SMP_REDUCTION_OP
#pragma parallel local(JX_SMP_PRIVATE) pfor
#endif
#ifdef JX_SMP_REDUCTION_OP
#endif
#endif

/* UnDef */
#undef JX_SMP_PRIVATE
#undef JX_SMP_REDUCTION_OP
#undef JX_SMP_REDUCTION_VARS
#undef JX_SMP_PAR_REGION
#undef JX_SMP_FOR
