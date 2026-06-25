/***** DO NOT use this file outside of the LAPACK directory *****/

/*--------------------------------------------------------------------------
 * This header renames the functions in LAPACK to avoid conflicts
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

/* lapack */
#define dbdsqr_  jx_dbdsqr
#define dgebd2_  jx_dgebd2
#define dgebrd_  jx_dgebrd
#define dgelq2_  jx_dgelq2
#define dgelqf_  jx_dgelqf
#define dgels_   jx_dgels
#define dgeqr2_  jx_dgeqr2
#define dgeqrf_  jx_dgeqrf
#define dgesvd_  jx_dgesvd
#define dgetf2_  jx_dgetf2
#define dgetrf_  jx_dgetrf
#define dgetri_  jx_dgetri
#define dgetrs_  jx_dgetrs
#define dlasq1_  jx_dlasq1
#define dlasq2_  jx_dlasq2
#define dlasrt_  jx_dlasrt
#define dorg2l_  jx_dorg2l
#define dorg2r_  jx_dorg2r
#define dorgbr_  jx_dorgbr
#define dorgl2_  jx_dorgl2
#define dorglq_  jx_dorglq
#define dorgql_  jx_dorgql
#define dorgqr_  jx_dorgqr
#define dorgtr_  jx_dorgtr
#define dorm2r_  jx_dorm2r
#define dormbr_  jx_dormbr
#define dorml2_  jx_dorml2
#define dormlq_  jx_dormlq
#define dormqr_  jx_dormqr
#define dpotf2_  jx_dpotf2
#define dpotrf_  jx_dpotrf
#define dpotrs_  jx_dpotrs
#define dsteqr_  jx_dsteqr
#define dsterf_  jx_dsterf
#define dsyev_   jx_dsyev
#define dsygs2_  jx_dsygs2
#define dsygst_  jx_dsygst
#define dsygv_   jx_dsygv
#define dsytd2_  jx_dsytd2
#define dsytrd_  jx_dsytrd
#define dtrti2_  jx_dtrti2
#define dtrtri_  jx_dtrtri

/* lapack auxiliary routines */
#define dlabad_  jx_dlabad
#define dlabrd_  jx_dlabrd
#define dlacpy_  jx_dlacpy
#define dlae2_   jx_dlae2
#define dlaev2_  jx_dlaev2
#define dlamch_  jx_dlamch
#define dlamc1_  jx_dlamc1
#define dlamc2_  jx_dlamc2
#define dlamc3_  jx_dlamc3
#define dlamc4_  jx_dlamc4
#define dlamc5_  jx_dlamc5
#define dlange_  jx_dlange
#define dlanst_  jx_dlanst
#define dlansy_  jx_dlansy
#define dlapy2_  jx_dlapy2
#define dlarf_   jx_dlarf
#define dlarfb_  jx_dlarfb
#define dlarfg_  jx_dlarfg
#define dlarft_  jx_dlarft
#define dlartg_  jx_dlartg
#define dlas2_   jx_dlas2
#define dlascl_  jx_dlascl
#define dlaset_  jx_dlaset
#define dlasq3_  jx_dlasq3
#define dlasq4_  jx_dlasq4
#define dlasq5_  jx_dlasq5
#define dlasq6_  jx_dlasq6
#define dlasr_   jx_dlasr
#define dlassq_  jx_dlassq
#define dlasv2_  jx_dlasv2
#define dlaswp_  jx_dlaswp
#define dlatrd_  jx_dlatrd
#define ieeeck_  jx_ieeeck
#define ilaenv_  jx_ilaenv

/* these auxiliary routines have a different definition in BLAS */
#define lsame_   jx_lapack_lsame
#define xerbla_  jx_lapack_xerbla

/* this is needed so that lapack can call external BLAS */
#include "jx_blas2.h"
