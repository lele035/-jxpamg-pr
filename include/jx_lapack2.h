#ifndef JX_LAPACK_HEADER
#define JX_LAPACK_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* dbdsqr.c */
JX_Int
jx_dbdsqr(const char *uplo, JX_Int *n, JX_Int *ncvt, JX_Int *nru, JX_Int *ncc,
JX_Real *d__, JX_Real *e, JX_Real *vt, JX_Int *ldvt, JX_Real *u, JX_Int *ldu, JX_Real *c__, JX_Int *ldc, JX_Real *work, JX_Int *info );

/* dgebd2.c */
JX_Int
jx_dgebd2( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *d__,
JX_Real *e, JX_Real *tauq, JX_Real *taup, JX_Real *work, JX_Int *info );

/* dgebrd.c */
JX_Int
jx_dgebrd( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *d__,
JX_Real *e, JX_Real *tauq, JX_Real *taup, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dgelq2.c */
JX_Int
jx_dgelq2( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *info );

/* dgelqf.c */
JX_Int
jx_dgelqf( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dgels.c */
JX_Int
jx_dgels( char *trans, JX_Int *m, JX_Int *n, JX_Int *nrhs, JX_Real *a,
JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dgeqr2.c */
JX_Int
jx_dgeqr2( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *info );

/* dgeqrf.c */
JX_Int
jx_dgeqrf( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dgesvd.c */
JX_Int
jx_dgesvd( char *jobu, char *jobvt, JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda,
JX_Real *s, JX_Real *u, JX_Int *ldu, JX_Real *vt, JX_Int *ldvt, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dgetf2.c */
JX_Int
jx_dgetf2( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *ipiv, JX_Int *info );

/* dgetrf.c */
JX_Int
jx_dgetrf( JX_Int *m, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *ipiv, JX_Int *info );

/* dgetri.c */
JX_Int
jx_dgetri( JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *ipiv, JX_Real *work, JX_Int *lwork, JX_Int *info);

/* dgetrs.c */
JX_Int
jx_dgetrs( const char *trans, JX_Int *n, JX_Int *nrhs, JX_Real *a, JX_Int *lda, JX_Int *ipiv, JX_Real *b, JX_Int *ldb, JX_Int *info );

/* dlasq1.c */
JX_Int
jx_dlasq1( JX_Int *n, JX_Real *d__, JX_Real *e, JX_Real *work, JX_Int *info );

/* dlasq2.c */
JX_Int
jx_dlasq2( JX_Int *n, JX_Real *z__, JX_Int *info );

/* dlasrt.c */
JX_Int
jx_dlasrt(const char *id, JX_Int *n, JX_Real *d__, JX_Int *info );

/* dorg2l.c */
JX_Int
jx_dorg2l( JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *info );

/* dorg2r.c */
JX_Int
jx_dorg2r( JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *info );

/* dorgbr.c */
JX_Int
jx_dorgbr(const char *vect, JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a,
JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dorgl2.c */
JX_Int
jx_dorgl2( JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *info );

/* dorglq.c */
JX_Int
jx_dorglq( JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dorgql.c */
JX_Int
jx_dorgql( JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dorgqr.c */
JX_Int
jx_dorgqr( JX_Int *m, JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dorgtr.c */
JX_Int
jx_dorgtr(const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dorm2r.c */
JX_Int
jx_dorm2r(const char *side,const char *trans, JX_Int *m, JX_Int *n, JX_Int *k,
JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *c__, JX_Int *ldc, JX_Real *work, JX_Int *info );

/* dormbr.c */
JX_Int
jx_dormbr(const char *vect,const char *side,const char *trans, JX_Int *m,
JX_Int *n, JX_Int *k, JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *c__, JX_Int *ldc, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dorml2.c */
JX_Int
jx_dorml2(const char *side,const char *trans, JX_Int *m, JX_Int *n, JX_Int *k,
JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *c__, JX_Int *ldc, JX_Real *work, JX_Int *info );

/* dormlq.c */
JX_Int
jx_dormlq(const char *side,const char *trans, JX_Int *m, JX_Int *n, JX_Int *k,
JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *c__, JX_Int *ldc, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dormqr.c */
JX_Int
jx_dormqr(const char *side,const char *trans, JX_Int *m, JX_Int *n, JX_Int *k,
JX_Real *a, JX_Int *lda, JX_Real *tau, JX_Real *c__, JX_Int *ldc, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dpotf2.c */
JX_Int
jx_dpotf2(const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *info );

/* dpotrf.c */
JX_Int
jx_dpotrf(const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *info );

/* dpotrs.c */
JX_Int
jx_dpotrs( char *uplo, JX_Int *n, JX_Int *nrhs, JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Int *info );

/* dsteqr.c */
JX_Int
jx_dsteqr(const char *compz, JX_Int *n, JX_Real *d__, JX_Real *e, JX_Real *z__, JX_Int *ldz, JX_Real *work, JX_Int *info );

/* dsterf.c */
JX_Int
jx_dsterf( JX_Int *n, JX_Real *d__, JX_Real *e, JX_Int *info );

/* dsyev.c */
JX_Int
jx_dsyev(const char *jobz,const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *w, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dsygs2.c */
JX_Int
jx_dsygs2( JX_Int *itype,const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Int *info );

/* dsygst.c */
JX_Int
jx_dsygst( JX_Int *itype,const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Int *info );

/* dsygv.c */
JX_Int
jx_dsygv( JX_Int *itype, char *jobz, char *uplo, JX_Int *n, JX_Real *a,
JX_Int *lda, JX_Real *b, JX_Int *ldb, JX_Real *w, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dsytd2.c */
JX_Int
jx_dsytd2(const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *d__, JX_Real *e, JX_Real *tau, JX_Int *info );

/* dsytrd.c */
JX_Int
jx_dsytrd(const char *uplo, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Real *d__,
JX_Real *e, JX_Real *tau, JX_Real *work, JX_Int *lwork, JX_Int *info );

/* dtrti2.c */
JX_Int
jx_dtrti2(const char *uplo, const char *diag, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *info);

/* dtrtri.c */
JX_Int
jx_dtrtri(const char *uplo, const char *diag, JX_Int *n, JX_Real *a, JX_Int *lda, JX_Int *info);

#ifdef __cplusplus
}
#endif

#endif
