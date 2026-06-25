//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//


/*! \file  par_pardiso.c
 *
 *  \brief Interface to Intel MKL PARDISO direct solvers
 *
 *  Date: 2021/06/29
 * 
 *  Reference for Intel MKL PARDISO:
 *  https://software.intel.com/en-us/node/470282
 *
 */


#include "jx_pamg.h"
#if WITH_PARDISO
#include "jx_mv.h"
#include <time.h>
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"

#define PRINT_MIN  0
#define DEBUG_MODE 0


/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/
/**
 * \fn Pardiso_Solver_Interface(int row, int nz, int *ia, int *ja, double *a, double *x, double *b)
 *
 * \brief Solve Ax=b by PARDISO directly. Each row of A should be
 *        in ascending order w.r.t. column indices.
 *
 *
 * \author Zhao Li
 * \date   06/26/2021
 */

JX_Int 
jx_pardiso(jx_CSRMatrix *AA, jx_Vector *bb, jx_Vector *xx)
{
    int prtlvl = 0;
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JX_Real *a = AA->data;
    // vector
    JX_Real *b = bb->data;
    JX_Real *x = xx->data;


    MKL_INT mtype = 11;    /* Real unsymmetric matrix */
    MKL_INT nrhs = 1;      /* Number of right hand sides */
    MKL_INT idum;          /* Integer dummy */
    MKL_INT iparm[64];     /* Pardiso control parameters */
    MKL_INT maxfct, mnum, phase, error, msglvl;    /* Auxiliary variables */
    
    void *pt[64];          /* Internal solver memory pointer pt */
    double ddum;           /* Double dummy */
    clock_t start_time = clock();
    
#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif
    
    PARDISOINIT(pt, &mtype, iparm); /* Initialize */
    iparm[34] = 1;        /* Use 0-based indexing */
    
    // iparm[13] = 1;        /* Output: Number of perturbed pivots */
    // iparm[17] = 1;       /* Output: Number of nonzeros in the factor LU */
    // iparm[18] = 1;       /* Output: Mflops for LU factorization */

    maxfct = 1;           /* Maximum number of numerical factorizations */
    mnum = 1;             /* Which factorization to use */
    msglvl = 0;           /* Do not print statistical information in file */
    error = 0;            /* Initialize error flag */
    
    phase = 11; /* Reordering and symbolic factorization */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("### ERROR: Symbolic factorization failed %d!\n", error);
        exit (1);
    }
    
    phase = 22; /* Numerical factorization */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("\n### ERROR: Numerical factorization failed %d!\n", error);
        exit (2);
    }
    
    phase = 33; /* Back substitution and iterative refinement */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
    
    if ( error != 0 ) {
        printf ("\n### ERROR: Solution failed %d!\n", error);
        exit (3);
    }
    
    if ( prtlvl > PRINT_MIN ) {
        clock_t end_time = clock();
        double solve_time = (double)(end_time - start_time)/(double)(CLOCKS_PER_SEC);
        printf("PARDISO costs %f seconds.\n", solve_time);
    }
    
    phase = -1; /* Release internal memory */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error);
    
#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif
    
    return 1;   
}




/**
 * \fn JX_Int jx_pardiso_factorize (jx_CSRMatrix *AA, Pardiso_data *pdata, JX_Int prtlvl)
 * 
 * \brief factorize A by PARDISO. Each row of A should be
 *        in ascending order w.r.t. column indices.
 *
 * \param AA        Pointer to matrix A
 * \param pdata     Pointer to numerical factorization data
 * \param prtlvl    Output level
 *
 * \author Li Zhao
 * \date   30/06/2021
 */

JX_Int jx_pardiso_factorize (jx_CSRMatrix *AA,
                             Pardiso_data *pdata,
                             JX_Int prtlvl)
{
    JX_Int status = 0;  
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JX_Real *a = AA->data;
    
    double  ddum;         /* Double dummy */
    MKL_INT nrhs = 1;     /* Number of right hand sides */
    MKL_INT idum;         /* Integer dummy */
    MKL_INT phase, error, msglvl;    /* Auxiliary variables */
    
#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Start]\n", __FUNCTION__);
    printf("### DEBUG: nr=%d, nc=%d, nnz=%d\n", m, n, nnz);
#endif
    
    pdata->mtype = 11;    /* Real unsymmetric matrix */

    PARDISOINIT(pdata->pt, &(pdata->mtype), pdata->iparm); /* Initialize */
    pdata->iparm[34] = 1;  /* Use 0-based indexing */

    /* Numbers of processors, value of OMP_NUM_THREADS */
#ifdef _OPENMP
    pdata->iparm[2]  = omp_get_num_threads();
#endif

    pdata->maxfct = 1;     /* Maximum number of numerical factorizations */
    pdata->mnum = 1;       /* Which factorization to use */
    msglvl = 0;            /* Do not print statistical information in file */
    error = 0;             /* Initialize error flag */
    
    clock_t start_time = clock();
    
    phase = 11; /* Reordering and symbolic factorization */
    PARDISO (pdata->pt, &(pdata->maxfct), &(pdata->mnum), &(pdata->mtype), &phase, &n,
             a, ia, ja, &idum, &nrhs, pdata->iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 ) {
        printf ("\n### ERROR: Symbolic factorization failed %d! %s %s %d\n", error,__FUNCTION__,__FILE__,__LINE__);
        exit (1);
    }
    
    phase = 22; /* Numerical factorization */
    PARDISO (pdata->pt, &(pdata->maxfct), &(pdata->mnum), &(pdata->mtype), &phase, &n,
             a, ia, ja, &idum, &nrhs, pdata->iparm, &msglvl, &ddum, &ddum, &error);

    if ( error != 0 ) {
        printf ("\n### ERROR: Numerical factorization failed %d! %s %s %d\n", error,__FUNCTION__,__FILE__,__LINE__);
        exit (2);
    }
    
    if ( prtlvl > PRINT_MIN ) {
        clock_t end_time = clock();
        double solve_time = (double)(end_time - start_time)/(double)(CLOCKS_PER_SEC);
        printf("PARDISO factoization costs %f seconds.\n", solve_time);
    }
    
#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif

    return status;
}

/**
 * \fn JX_Int jx_pardiso_solve (jx_CSRMatrix *AA, jx_Vector *bb, jx_Vector *xx, Pardiso_data *pdata, JX_Int prtlvl)
 * 
 * \brief Solve Au=b by Intel MKL PARDISO, numerical factorization is given.
 *        Each row of A should be in ascending order w.r.t. column indices.
 *
 * \param AA        Pointer to stiffness matrix of levelNum levels
 * \param bb        Pointer to the vector of right hand side term
 * \param xx        Pointer to the vector of dofs
 * \param pdata     Pointer to the numerical factorization data
 * \param prtlvl    Output level
 *
 * \author Li Zhao
 * \date   30/06/2021
 */

JX_Int jx_pardiso_solve (jx_CSRMatrix *AA, 
                         jx_Vector *bb, 
                         jx_Vector *xx,
                         Pardiso_data *pdata,
                         JX_Int prtlvl)
{
    JX_Int status = 0;
    // Matrix
    MKL_INT n = AA->num_rows;
    MKL_INT m = AA->num_cols;
    MKL_INT nnz = AA->num_nonzeros;
    MKL_INT *ia = AA->i;
    MKL_INT *ja = AA->j;
    JX_Real *a = AA->data;
    // vector
    JX_Real *b = bb->data;  /* RHS vector */
    JX_Real *x = xx->data; /* Solution vector */

    MKL_INT mtype = 11;    /* Real unsymmetric matrix */
    MKL_INT nrhs = 1;      /* Number of right hand sides */
    MKL_INT idum;          /* Integer dummy */
    MKL_INT phase, error, msglvl;    /* Auxiliary variables */
    
    msglvl = 0; /* Do not print statistical information in file */
    
    clock_t start_time = clock();
    
    phase = 33; /* Back substitution and iterative refinement */
    PARDISO (pdata->pt, &(pdata->maxfct), &(pdata->mnum), &(pdata->mtype), &phase,
             &n, a, ia, ja, &idum, &nrhs, pdata->iparm, &msglvl, b, x, &error);
    
    if ( error != 0 ) {
        printf ("### ERROR: Solution failed %d!\n", error);
        exit (3);
    }
    
    if ( prtlvl > PRINT_MIN ) {
        clock_t end_time = clock();
        double solve_time = (double)(end_time - start_time)/(double)(CLOCKS_PER_SEC);
        printf("PARDISO costs %f seconds.\n", solve_time);
    }
    
#if DEBUG_MODE
    printf("### DEBUG: %s ...... [Finish]\n", __FUNCTION__);
#endif
    
    return status;
}

#endif
/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
