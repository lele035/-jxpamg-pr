/***** DO NOT use this file outside of the BLAS directory *****/

/*--------------------------------------------------------------------------
 * This header renames the functions in BLAS to avoid conflicts
 *--------------------------------------------------------------------------*/

/* blas */
#define dasum_   jx_dasum
#define daxpy_   jx_daxpy
#define dcopy_   jx_dcopy
#define ddot_    jx_ddot
#define dgemm_   jx_dgemm
#define dgemv_   jx_dgemv
#define dger_    jx_dger
#define dnrm2_   jx_dnrm2
#define drot_    jx_drot
#define dscal_   jx_dscal
#define dswap_   jx_dswap
#define dsymm_   jx_dsymm
#define dsymv_   jx_dsymv
#define dsyr2_   jx_dsyr2
#define dsyr2k_  jx_dsyr2k
#define dsyrk_   jx_dsyrk
#define dtrmm_   jx_dtrmm
#define dtrmv_   jx_dtrmv
#define dtrsm_   jx_dtrsm
#define dtrsv_   jx_dtrsv
#define idamax_  jx_idamax

/* f2c library routines */
#define s_cmp    jx_s_cmp
#define s_copy   jx_s_copy
#define s_cat    jx_s_cat
#define d_lg10   jx_d_lg10
#define d_sign   jx_d_sign
#define pow_dd   jx_pow_dd
#define pow_di   jx_pow_di

/* these auxiliary routines have a different definition in LAPACK */
#define lsame_   jx_blas_lsame
#define xerbla_  jx_blas_xerbla
