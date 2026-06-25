#ifndef JX_BLAS_HEADER
#define JX_BLAS_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* dasum.c */
JX_Real
jx_dasum( JX_Int *n, JX_Real *dx, JX_Int *incx );

/* daxpy.c */
JX_Int
jx_daxpy( JX_Int *n, JX_Real *da, JX_Real *dx, JX_Int *incx, JX_Real *dy, JX_Int *incy );

/* dcopy.c */
JX_Int
jx_dcopy( JX_Int *n, JX_Real *dx, JX_Int *incx, JX_Real *dy, JX_Int *incy );

/* ddot.c */
JX_Real
jx_ddot( JX_Int *n, JX_Real *dx, JX_Int *incx, JX_Real *dy, JX_Int *incy );

/* dgemm.c */
JX_Int
jx_dgemm( const char *transa, const char *transb, JX_Int *m, JX_Int *n, JX_Int *k,
JX_Real *alpha, JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Real *beta, JX_Real *c, JX_Int *ldc );

/* dgemv.c */
JX_Int
jx_dgemv( const char *trans, JX_Int *m, JX_Int *n, JX_Real *alpha,
JX_Real *a, JX_Int *lda, JX_Real *x, JX_Int *incx, JX_Real *beta, JX_Real *y, JX_Int *incy );

/* dger.c */
JX_Int
jx_dger( JX_Int *m, JX_Int *n, JX_Real *alpha, JX_Real *x, JX_Int *incx, JX_Real *y, JX_Int *incy, JX_Real *a, JX_Int *lda );

/* dnrm2.c */
JX_Real
jx_dnrm2( JX_Int *n, JX_Real *dx, JX_Int *incx );

/* drot.c */
JX_Int
jx_drot( JX_Int *n, JX_Real *dx, JX_Int *incx, JX_Real *dy, JX_Int *incy, JX_Real *c, JX_Real *s );

/* dscal.c */
JX_Int
jx_dscal( JX_Int *n, JX_Real *da, JX_Real *dx, JX_Int *incx );

/* dswap.c */
JX_Int
jx_dswap( JX_Int *n, JX_Real *dx, JX_Int *incx, JX_Real *dy, JX_Int *incy );

/* dsymm.c */
JX_Int
jx_dsymm( const char *side, const char *uplo, JX_Int *m, JX_Int *n, JX_Real *alpha,
JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Real *beta, JX_Real *c__, JX_Int *ldc );

/* dsymv.c */
JX_Int
jx_dsymv( const char *uplo, JX_Int *n, JX_Real *alpha, JX_Real *a,
JX_Int *lda, JX_Real *x, JX_Int *incx, JX_Real *beta, JX_Real *y, JX_Int *incy );

/* dsyr2.c */
JX_Int
jx_dsyr2( const char *uplo, JX_Int *n, JX_Real *alpha, JX_Real *x, JX_Int *incx, JX_Real *y, JX_Int *incy, JX_Real *a, JX_Int *lda );

/* dsyr2k.c */
JX_Int
jx_dsyr2k( const char *uplo, const char *trans, JX_Int *n, JX_Int *k, JX_Real *alpha,
JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Real *beta, JX_Real *c__, JX_Int *ldc );

/* dsyrk.c */
JX_Int
jx_dsyrk( const char *uplo, const char *trans, JX_Int *n, JX_Int *k, JX_Real *alpha,
JX_Real *a, JX_Int *lda, JX_Real *beta, JX_Real *c, JX_Int *ldc );

/* dtrmm.c */
JX_Int
jx_dtrmm( const char *side, const char *uplo, const char *transa, const char *diag,
JX_Int *m, JX_Int *n, JX_Real *alpha, JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb );

/* dtrmv.c */
JX_Int
jx_dtrmv( const char *uplo, const char *trans, const char *diag, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *x, JX_Int *incx );

/* dtrsm.c */
JX_Int
jx_dtrsm( const char *side, const char *uplo, const char *transa, const char *diag,
JX_Int *m, JX_Int *n, JX_Real *alpha, JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb );

/* dtrsv.c */
JX_Int
jx_dtrsv( const char *uplo, const char *trans, const char *diag, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *x, JX_Int *incx );

/* idamax.c */
JX_Int
jx_idamax( JX_Int *n, JX_Real *dx, JX_Int *incx );

#ifdef __cplusplus
}
#endif

#endif
