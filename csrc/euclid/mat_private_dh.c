//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  mat_private_dh.c
 *  Date: 2013/01/23
 */

#include "jx_euclid.h"

#define JX_IS_UPPER_TRI 97
#define JX_IS_LOWER_TRI 98
#define JX_IS_FULL 99

static JX_Int jx_isTriangular( JX_Int m, JX_Int *rp, JX_Int *cval );
static void jx_mat_par_read_allocate_private( jx_Mat_dh *Aout, JX_Int n, JX_Int *rowLengths, JX_Int *rowToBlock );
void jx_mat_partition_private( jx_Mat_dh A, JX_Int blocks, JX_Int *o2n_row, JX_Int *rowToBlock );
static void jx_convert_triples_to_scr_private( JX_Int m, JX_Int nz, JX_Int *I, JX_Int *J,
                                 JX_Real *A, JX_Int *rp, JX_Int *cval, JX_Real *aval );

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_print_graph_private"
void jx_mat_dh_print_graph_private( JX_Int m,
                                 JX_Int beg_row,
                                 JX_Int *rp,
                                 JX_Int *cval,
                                 JX_Real *aval,
                                 JX_Int *n2o,
                                 JX_Int *o2n,
                                 jx_Hash_i_dh hash,
                                 FILE* fp )
{
    JX_START_FUNC_DH
    JX_Int i, j, row, col;
    jx_bool private_n2o = jx_false;
    jx_bool private_hash = jx_false;
    JX_Int *work = NULL;
    
    work = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (n2o == NULL)
    {
        private_n2o = jx_true;
        jx_create_nat_ordering_private(m, &n2o); JX_CHECK_V_ERROR;
        jx_create_nat_ordering_private(m, &o2n); JX_CHECK_V_ERROR;
    }
    if (hash == NULL)
    {
        private_hash = jx_true;
        jx_Hash_i_dhCreate(&hash, -1); JX_CHECK_V_ERROR;
    }
    for (i = 0; i < m; ++ i)
    {
        for (j = 0; j < m; ++ j) work[j] = 0;
        row = n2o[i];
        for (j = rp[row]; j < rp[row+1]; ++ j)
        {
            col = cval[j];
            if (col >= beg_row || col < beg_row+m)
            {
                col = o2n[col];
            }
            else
            {
                JX_Int tmp = col;
                
                tmp = jx_Hash_i_dhLookup(hash, col); JX_CHECK_V_ERROR;
                if (tmp == -1)
                {
                    jx_sprintf(jx_msgBuf_dh, "beg_row= %i  m= %i; nonlocal column= %i not in hash table",
                                                     beg_row, m, col);
                    JX_SET_V_ERROR(jx_msgBuf_dh);
                }
                else
                {
                    col = tmp;
                }
            }
            work[col] = 1;
        }
        for (j = 0; j < m; ++ j)
        {
            if (work[j])
            {
                jx_fprintf(fp, " x ");
            }
            else
            {
                jx_fprintf(fp, "   ");
            }
        }
        jx_fprintf(fp, "\n");
    }
    if (private_n2o)
    {
        jx_destroy_nat_ordering_private(n2o); JX_CHECK_V_ERROR;
        jx_destroy_nat_ordering_private(o2n); JX_CHECK_V_ERROR;
    }
    if (private_hash)
    {
        jx_Hash_i_dhDestroy(hash); JX_CHECK_V_ERROR;
    }
    if (work != NULL)
    {
        JX_FREE_DH(work); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_create_nat_ordering_private"
void jx_create_nat_ordering_private( JX_Int m, JX_Int **p )
{
    JX_START_FUNC_DH
    JX_Int *tmp, i;
    
    tmp = *p = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i)
    {
        tmp[i] = i;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_destroy_nat_ordering_private"
void jx_destroy_nat_ordering_private( JX_Int *p )
{
    JX_START_FUNC_DH
    JX_FREE_DH(p); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_invert_perm"
void jx_invert_perm( JX_Int m, JX_Int *pIN, JX_Int *pOUT )
{
    JX_START_FUNC_DH
    JX_Int i;
    
    for (i = 0; i < m; ++ i) pOUT[pIN[i]] = i;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_print_csr_private"
void jx_mat_dh_print_csr_private( JX_Int m, JX_Int *rp, JX_Int *cval, JX_Real *aval, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i, nz = rp[m];
    
    jx_fprintf(fp, "%i %i\n", m, rp[m]);
    for (i = 0; i <= m; ++ i) jx_fprintf(fp, "%i ", rp[i]);
    jx_fprintf(fp, "\n");
    for (i = 0; i < nz; ++ i) jx_fprintf(fp, "%i ", cval[i]);
    jx_fprintf(fp, "\n");
    for (i = 0; i < nz; ++ i) jx_fprintf(fp, "%1.19e ", aval[i]);
    jx_fprintf(fp, "\n");
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_read_csr_private"
void jx_mat_dh_read_csr_private( JX_Int *mOUT, JX_Int **rpOUT, JX_Int **cvalOUT, JX_Real **avalOUT, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i, m, nz, items;
    JX_Int *rp, *cval;
    JX_Real *aval;
    
    items = jx_fscanf(fp,"%d %d",&m, &nz);
    if (items != 2)
    {
        JX_SET_V_ERROR("failed to read header");
    }
    else
    {
        jx_printf("jx_mat_dh_read_csr_private:: m= %i  nz= %i\n", m, nz);
    }
   *mOUT = m;
    rp = *rpOUT = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    cval = *cvalOUT = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    aval = *avalOUT = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    for (i = 0; i <= m; ++ i)
    {
        items = jx_fscanf(fp,"%d", &(rp[i]));
        if (items != 1)
        {
            jx_sprintf(jx_msgBuf_dh, "failed item %i of %i in rp block", i, m+1);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    for (i = 0; i < nz; ++ i)
    {
        items = jx_fscanf(fp,"%d", &(cval[i]));
        if (items != 1)
        {
            jx_sprintf(jx_msgBuf_dh, "failed item %i of %i in cval block", i, m+1);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    for (i = 0; i < nz; ++ i)
    {
        items = jx_fscanf(fp,"%lg", &(aval[i]));
        if (items != 1)
        {
            jx_sprintf(jx_msgBuf_dh, "failed item %i of %i in aval block", i, m+1);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
    }
    JX_END_FUNC_DH
}

#define JX_MAX_JUNK 200

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_read_triples_private"
void jx_mat_dh_read_triples_private( JX_Int ignore, JX_Int *mOUT, JX_Int **rpOUT, JX_Int **cvalOUT, JX_Real **avalOUT, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int m, n, nz, items, i, j;
    JX_Int idx = 0;
    JX_Int *cval, *rp, *I, *J;
    JX_Real *aval, *A, v;
    char junk[JX_MAX_JUNK];
    fpos_t fpos;
    
    if (ignore && jx_myid_dh == 0)
    {
        jx_printf("jx_mat_dh_read_triples_private:: ignoring following header lines:\n");
        jx_printf("--------------------------------------------------------------\n");
        for (i = 0; i < ignore; ++ i)
        {
            fgets(junk, JX_MAX_JUNK, fp);
            jx_printf("%s", junk);
        }
        jx_printf("--------------------------------------------------------------\n");
        if (fgetpos(fp, &fpos)) JX_SET_V_ERROR("fgetpos failed!");
        jx_printf("\njx_mat_dh_read_triples_private::1st two non-ignored lines:\n");
        jx_printf("--------------------------------------------------------------\n");
        for (i = 0; i < 2; ++ i)
        {
            fgets(junk, JX_MAX_JUNK, fp);
            jx_printf("%s", junk);
        }
        jx_printf("--------------------------------------------------------------\n");
        if (fsetpos(fp, &fpos)) JX_SET_V_ERROR("fsetpos failed!");
    }
    if (feof(fp)) jx_printf("trouble!");
    m = n = nz = 0;
    while (!feof(fp))
    {
        items = jx_fscanf(fp,"%d %d %lg", &i, &j, &v);
        if (items != 3)
        {
            break;
        }
        ++ nz;
        if (i > m) m = i;
        if (j > n) n = j;
    }
    if (jx_myid_dh == 0)
    {
        jx_printf("jx_mat_dh_read_triples_private: m= %i  nz= %i\n", m, nz);
    }
    rewind(fp);
    for (i = 0; i < ignore; ++ i)
    {
        fgets(junk, JX_MAX_JUNK, fp);
    }
    if (m != n)
    {
        jx_sprintf(jx_msgBuf_dh, "matrix is not square; row= %i, cols= %i", m, n);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
   *mOUT = m;
    rp = *rpOUT = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    cval = *cvalOUT = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    aval = *avalOUT = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    I = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    J = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    A = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    while (!feof(fp))
    {
        items = jx_fscanf(fp,"%d %d %lg", &i, &j, &v);
        if (items < 3) break;
        j --;
        i --;
        I[idx] = i;
        J[idx] = j;
        A[idx] = v;
        ++ idx;
    }
    jx_convert_triples_to_scr_private(m, nz, I, J, A, rp, cval, aval); JX_CHECK_V_ERROR;
    JX_Int type;
    type = jx_isTriangular(m, rp, cval); JX_CHECK_V_ERROR;
    if (type == JX_IS_UPPER_TRI)
    {
        jx_printf("CAUTION: matrix is upper triangular; converting to full\n");
    }
    else if (type == JX_IS_LOWER_TRI)
    {
        jx_printf("CAUTION: matrix is lower triangular; converting to full\n");
    }
    if (type == JX_IS_UPPER_TRI || type == JX_IS_LOWER_TRI)
    {
        jx_make_full_private(m, &rp, &cval, &aval); JX_CHECK_V_ERROR;
    }
   *rpOUT = rp;
   *cvalOUT = cval;
   *avalOUT = aval;
    JX_FREE_DH(I); JX_CHECK_V_ERROR;
    JX_FREE_DH(J); JX_CHECK_V_ERROR;
    JX_FREE_DH(A); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_convert_triples_to_scr_private"
void jx_convert_triples_to_scr_private( JX_Int m, JX_Int nz, JX_Int *I, JX_Int *J, JX_Real *A, JX_Int *rp, JX_Int *cval, JX_Real *aval )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int *rowCounts;
    
    rowCounts = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) rowCounts[i] = 0;
    for (i = 0; i < nz; ++ i)
    {
        JX_Int row = I[i];
        
        rowCounts[row] += 1;
    }
    rp[0] = 0;
    for (i = 1; i <= m; ++ i)
    {
        rp[i] = rp[i-1] + rowCounts[i-1];
    }
    memcpy(rowCounts, rp, (m+1)*sizeof(JX_Int));
    for (i = 0; i < nz; ++ i)
    {
        JX_Int row = I[i];
        JX_Int col = J[i];
        JX_Real val = A[i];
        JX_Int idx = rowCounts[row];
        
        rowCounts[row] += 1;
        cval[idx] = col;
        aval[idx] = val;
    }
    JX_FREE_DH(rowCounts); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

void jx_fix_diags_private( jx_Mat_dh A );
void jx_insert_missing_diags_private( jx_Mat_dh A );

#undef __FUNC__
#define __FUNC__ "jx_readMat"
void jx_readMat( jx_Mat_dh *Aout, char *ft, char *fn, JX_Int ignore )
{
    JX_START_FUNC_DH
    jx_bool makeStructurallySymmetric;
    jx_bool fixDiags;
    
   *Aout = NULL;
    makeStructurallySymmetric = jx_Parser_dhHasSwitch(jx_parser_dh, "-makeSymmetric");
    fixDiags = jx_Parser_dhHasSwitch(jx_parser_dh, "-fixDiags");
    if (fn == NULL)
    {
        JX_SET_V_ERROR("passed NULL filename; can't open for reading!");
    }
    if (!strcmp(ft, "csr"))
    {
        jx_Mat_dhReadCSR(Aout, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "trip"))
    {
        jx_Mat_dhReadTriples(Aout, ignore, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "ebin"))
    {
        jx_Mat_dhReadBIN(Aout, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "petsc"))
    {
        jx_sprintf(jx_msgBuf_dh, "must recompile Euclid using petsc mode!");
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    else
    {
        jx_sprintf(jx_msgBuf_dh, "unknown filetype: -ftin %s", ft);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    if (makeStructurallySymmetric)
    {
        jx_printf("\npadding with zeros to make structurally symmetric\n");
        jx_Mat_dhMakeStructurallySymmetric(*Aout); JX_CHECK_V_ERROR;
    }
    if ( (*Aout)->m == 0)
    {
        JX_SET_V_ERROR("row count = 0; something's wrong!");
    }
    if (fixDiags)
    {
        jx_fix_diags_private(*Aout); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_fix_diags_private"
void jx_fix_diags_private( jx_Mat_dh A )
{
    JX_START_FUNC_DH
    JX_Int i, j, m = A->m, *rp = A->rp, *cval = A->cval;
    JX_Real *aval = A->aval;
    jx_bool insertDiags = jx_false;
    
    for (i = 0; i < m; ++ i)
    {
        jx_bool isMissing = jx_true;
        
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            if (cval[j] == i)
            {
                isMissing = jx_false;
                break;
            }
        }
        if (isMissing)
        {
            insertDiags = jx_true;
            break;
        }
    }
    if (insertDiags)
    {
        jx_insert_missing_diags_private(A); JX_CHECK_V_ERROR;
        rp = A->rp;
        cval = A->cval;
        aval = A->aval;
    }
    for (i = 0; i < m; ++ i)
    {
        JX_Real sum = 0;
        
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            sum = JX_MAX(sum, fabs(aval[j]));
        }
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            if (cval[j] == i)
            {
                aval[j] = sum;
                break;
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_insert_missing_diags_private"
void jx_insert_missing_diags_private( jx_Mat_dh A )
{
    JX_START_FUNC_DH
    JX_Int *RP = A->rp, *CVAL = A->cval, m = A->m;
    JX_Int *rp, *cval;
    JX_Real *AVAL = A->aval, *aval;
    JX_Int i, j, nz = RP[m] + m;
    JX_Int idx = 0;
    
    rp = A->rp = (JX_Int *)JX_MALLOC_DH((1+m)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    cval = A->cval = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    aval = A->aval = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    rp[0] = 0;
    for (i = 0; i < m; ++ i)
    {
        jx_bool isMissing = jx_true;
        
        for (j = RP[i]; j < RP[i+1]; ++ j)
        {
            cval[idx] = CVAL[j];
            aval[idx] = AVAL[j];
            ++ idx;
            if (CVAL[j] == i) isMissing = jx_false;
        }
        if (isMissing)
        {
            cval[idx] = i;
            aval[idx] = 0.0;
            ++ idx;
        }
        rp[i+1] = idx;
    }
    JX_FREE_DH(RP); JX_CHECK_V_ERROR;
    JX_FREE_DH(CVAL); JX_CHECK_V_ERROR;
    JX_FREE_DH(AVAL); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_readVec"
void jx_readVec( jx_Vec_dh *bout, char *ft, char *fn, JX_Int ignore )
{
    JX_START_FUNC_DH
   *bout = NULL;
    if (fn == NULL)
    {
        JX_SET_V_ERROR("passed NULL filename; can't open for reading!");
    }
    if (!strcmp(ft, "csr")  ||  !strcmp(ft, "trip"))
    {
        jx_Vec_dhRead(bout, ignore, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "ebin"))
    {
        jx_Vec_dhReadBIN(bout, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "petsc"))
    {
        jx_sprintf(jx_msgBuf_dh, "must recompile Euclid using petsc mode!");
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    else
    {
        jx_sprintf(jx_msgBuf_dh, "unknown filetype: -ftin %s", ft);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_writeMat"
void jx_writeMat( jx_Mat_dh Ain, char *ft, char *fn )
{
    JX_START_FUNC_DH
    if (fn == NULL)
    {
        JX_SET_V_ERROR("passed NULL filename; can't open for writing!");
    }
    if (!strcmp(ft, "csr"))
    {
        jx_Mat_dhPrintCSR(Ain, NULL, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "trip"))
    {
        jx_Mat_dhPrintTriples(Ain, NULL, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "ebin"))
    {
        jx_Mat_dhPrintBIN(Ain, NULL, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "petsc"))
    {
        jx_sprintf(jx_msgBuf_dh, "must recompile Euclid using petsc mode!");
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    else
    {
        jx_sprintf(jx_msgBuf_dh, "unknown filetype: -ftout %s", ft);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_writeVec"
void jx_writeVec( jx_Vec_dh bin, char *ft, char *fn )
{
    JX_START_FUNC_DH
    if (fn == NULL)
    {
        JX_SET_V_ERROR("passed NULL filename; can't open for writing!");
    }
    if (!strcmp(ft, "csr")  ||  !strcmp(ft, "trip"))
    {
        jx_Vec_dhPrint(bin, NULL, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "ebin"))
    {
        jx_Vec_dhPrintBIN(bin, NULL, fn); JX_CHECK_V_ERROR;
    }
    else if (!strcmp(ft, "petsc"))
    {
        jx_sprintf(jx_msgBuf_dh, "must recompile Euclid using petsc mode!");
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    else
    {
        jx_sprintf(jx_msgBuf_dh, "unknown filetype: -ftout %s", ft);
        JX_SET_V_ERROR(jx_msgBuf_dh);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_isTriangular"
JX_Int jx_isTriangular( JX_Int m, JX_Int *rp, JX_Int *cval )
{
    JX_START_FUNC_DH
    JX_Int row, j;
    JX_Int type;
    jx_bool type_lower = jx_false, type_upper = jx_false;
    
    if (jx_np_dh > 1)
    {
        JX_SET_ERROR(-1, "only implemented for a single cpu");
    }
    for (row = 0; row < m; ++ row)
    {
        for (j = rp[row]; j < rp[row+1]; ++ j)
        {
            JX_Int col = cval[j];
            
            if (col < row) type_lower = jx_true;
            if (col > row) type_upper = jx_true;
        }
        if (type_lower && type_upper) break;
    }
    if (type_lower && type_upper)
    {
        type = JX_IS_FULL;
    }
    else if (type_lower)
    {
        type = JX_IS_LOWER_TRI;
    }
    else
    {
        type = JX_IS_UPPER_TRI;
    }
    JX_END_FUNC_VAL(type)
}

static void jx_mat_dh_transpose_reuse_private_private( jx_bool allocateMem,
             JX_Int m, JX_Int *rpIN, JX_Int *cvalIN, JX_Real *avalIN, JX_Int **rpOUT, JX_Int **cvalOUT, JX_Real **avalOUT );

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_transpose_reuse_private"
void jx_mat_dh_transpose_reuse_private( JX_Int m,
                                     JX_Int *rpIN,
                                     JX_Int *cvalIN,
                                     JX_Real *avalIN,
                                     JX_Int *rpOUT,
                                     JX_Int *cvalOUT,
                                     JX_Real *avalOUT )
{
    JX_START_FUNC_DH
    jx_mat_dh_transpose_reuse_private_private(jx_false, m,
            rpIN, cvalIN, avalIN, &rpOUT, &cvalOUT, &avalOUT); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_transpose_private"
void jx_mat_dh_transpose_private( JX_Int m,
                               JX_Int *RP,
                               JX_Int **rpOUT,
                               JX_Int *CVAL,
                               JX_Int **cvalOUT,
                               JX_Real *AVAL,
                               JX_Real **avalOUT )
{
    JX_START_FUNC_DH
    jx_mat_dh_transpose_reuse_private_private(jx_true, m, RP, CVAL, AVAL, rpOUT, cvalOUT, avalOUT); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_dh_transpose_private_private"
void jx_mat_dh_transpose_reuse_private_private( jx_bool allocateMem,
                                             JX_Int m,
                                             JX_Int *RP,
                                             JX_Int *CVAL,
                                             JX_Real *AVAL,
                                             JX_Int **rpOUT,
                                             JX_Int **cvalOUT,
                                             JX_Real **avalOUT )
{
    JX_START_FUNC_DH
    JX_Int *rp, *cval, *tmp;
    JX_Int i, j, nz = RP[m];
    JX_Real *aval = NULL;
    
    if (allocateMem)
    {
        rp = *rpOUT = (JX_Int *)JX_MALLOC_DH((1+m)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        cval = *cvalOUT = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        if (avalOUT != NULL)
        {
            aval = *avalOUT = (JX_Real*)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
        }
    }
    else
    {
        rp = *rpOUT;
        cval = *cvalOUT;
        if (avalOUT != NULL) aval = *avalOUT;
    }
    tmp = (JX_Int *)JX_MALLOC_DH((1+m)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i <= m; ++ i) tmp[i] = 0;
    for (i = 0; i < m; ++ i)
    {
        for (j = RP[i]; j < RP[i+1]; ++ j)
        {
            JX_Int col = CVAL[j];
            
            tmp[col+1] += 1;
        }
    }
    for (i = 1; i <= m; ++ i) tmp[i] += tmp[i-1];
    memcpy(rp, tmp, (m+1)*sizeof(JX_Int));
    if (avalOUT != NULL)
    {
        for (i = 0; i < m; ++ i)
        {
            for (j = RP[i]; j < RP[i+1]; ++ j)
            {
                JX_Int col = CVAL[j];
                JX_Int idx = tmp[col];
                
                cval[idx] = i;
                aval[idx] = AVAL[j];
                tmp[col] += 1;
            }
        }
    }
    else
    {
        for (i = 0; i < m; ++ i)
        {
            for (j = RP[i]; j < RP[i+1]; ++ j)
            {
                JX_Int col = CVAL[j];
                JX_Int idx = tmp[col];
                
                cval[idx] = i;
                tmp[col] += 1;
            }
        }
    }
    JX_FREE_DH(tmp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_jx_find_owner"
JX_Int jx_mat_jx_find_owner( JX_Int *beg_rows, JX_Int *end_rows, JX_Int index )
{
    JX_START_FUNC_DH
    JX_Int pe, owner = -1;
    
    for (pe = 0; pe < jx_np_dh; ++ pe)
    {
        if (index >= beg_rows[pe] && index < end_rows[pe])
        {
            owner = pe;
            break;
        }
    }
    if (owner == -1)
    {
        jx_sprintf(jx_msgBuf_dh, "failed to jx_find owner for index= %i", index);
        JX_SET_ERROR(-1, jx_msgBuf_dh);
    }
    JX_END_FUNC_VAL(owner)
}

#define JX_AVAL_TAG 2
#define JX_CVAL_TAG 3

void jx_partition_and_distribute_private( jx_Mat_dh A, jx_Mat_dh *Bout );
void jx_partition_and_distribute_metis_private( jx_Mat_dh A, jx_Mat_dh *Bout );

#undef __FUNC__
#define __FUNC__ "jx_readMat_par"
void jx_readMat_par( jx_Mat_dh *Aout, char *fileType, char *fileName, JX_Int ignore )
{
    JX_START_FUNC_DH
    jx_Mat_dh A = NULL;
    
    if (jx_myid_dh == 0)
    {
        JX_Int tmp = jx_np_dh;
        
        jx_np_dh = 1;
        jx_readMat(&A, fileType, fileName, ignore); JX_CHECK_V_ERROR;
        jx_np_dh = tmp;
    }
    if (jx_np_dh == 1)
    {
       *Aout = A;
    }
    else
    {
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-metis"))
        {
            jx_partition_and_distribute_metis_private(A, Aout); JX_CHECK_V_ERROR;
        }
        else
        {
            jx_partition_and_distribute_private(A, Aout); JX_CHECK_V_ERROR;
        }
    }
    if (jx_np_dh > 1 && A != NULL)
    {
        jx_Mat_dhDestroy(A); JX_CHECK_V_ERROR;
    }
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-printMAT"))
    {
        char xname[] = "A", *name = xname;
        
        jx_Parser_dhReadString(jx_parser_dh, "-printMat", &name);
        jx_Mat_dhPrintTriples(*Aout, NULL, name); JX_CHECK_V_ERROR;
        jx_printf_dh("\n@@@ jx_readMat_par: printed mat to %s\n\n", xname);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_partition_and_distribute_metis_private"
void jx_partition_and_distribute_metis_private( jx_Mat_dh A, jx_Mat_dh *Bout )
{
    JX_START_FUNC_DH
    jx_Mat_dh B = NULL;
    jx_Mat_dh C = NULL;
    JX_Int i, m;
    JX_Int *rowLengths = NULL;
    JX_Int *o2n_row = NULL, *n2o_col = NULL, *rowToBlock = NULL;
    JX_Int *beg_row = NULL, *row_count = NULL;
    MPI_Request *send_req = NULL;
    MPI_Request *rcv_req = NULL;
    MPI_Status *send_status = NULL;
    MPI_Status *rcv_status = NULL;
    
    jx_MPI_Barrier(jx_comm_dh);
    jx_printf_dh("@@@ partitioning with metis\n");
    if (jx_myid_dh == 0) m = A->m;
    jx_MPI_Bcast(&m, 1, JX_MPI_INT, 0, MPI_COMM_WORLD);
    rowLengths = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    rowToBlock = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        JX_Int *tmp = A->rp;
        
        for (i = 0; i < m; ++ i)
        {
            rowLengths[i] = tmp[i+1] - tmp[i];
        }
    }
    jx_MPI_Bcast(rowLengths, m, JX_MPI_INT, 0, jx_comm_dh);
    if (jx_myid_dh == 0)
    {
        JX_Int idx = 0;
        JX_Int j;
        
        jx_Mat_dhPartition(A, jx_np_dh, &beg_row, &row_count, &n2o_col, &o2n_row); JX_ERRCHKA;
        jx_Mat_dhPermute(A, n2o_col, &C); JX_ERRCHKA;
        for (i = 0; i < jx_np_dh; ++ i)
        {
            for (j = beg_row[i]; j < beg_row[i]+row_count[i]; ++ j)
            {
                rowToBlock[idx++] = i;
            }
        }
    }
    jx_MPI_Bcast(rowToBlock, m, JX_MPI_INT, 0, jx_comm_dh);
    jx_mat_par_read_allocate_private(&B, m, rowLengths, rowToBlock); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        JX_Int *cval = C->cval, *rp = C->rp;
        JX_Real *aval = C->aval;
        
        send_req = (MPI_Request *)JX_MALLOC_DH(2*m*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
        send_status = (MPI_Status*)JX_MALLOC_DH(2*m*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
        for (i = 0; i < m; ++ i)
        {
            JX_Int owner = rowToBlock[i];
            JX_Int count = rp[i+1] - rp[i];
            
            if (!count)
            {
                jx_sprintf(jx_msgBuf_dh, "row %i of %i is empty!", i+1, m);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
            jx_MPI_Isend(cval+rp[i], count, JX_MPI_INT, owner, JX_CVAL_TAG, jx_comm_dh, send_req+2*i);
            jx_MPI_Isend(aval+rp[i], count, JX_MPI_REAL, owner, JX_AVAL_TAG, jx_comm_dh, send_req+2*i+1);
        }
    }
    JX_Int *cval = B->cval;
    JX_Int *rp = B->rp;
    JX_Real *aval = B->aval;
    m = B->m;
    rcv_req = (MPI_Request *)JX_MALLOC_DH(2*m*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    rcv_status = (MPI_Status *)JX_MALLOC_DH(2*m*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i)
    {
        JX_Int count = rp[i+1] - rp[i];
        
        if (!count)
        {
            jx_sprintf(jx_msgBuf_dh, "local row %i of %i is empty!", i+1, m);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
        jx_MPI_Irecv(cval+rp[i], count, JX_MPI_INT, 0, JX_CVAL_TAG, jx_comm_dh, rcv_req+2*i);
        jx_MPI_Irecv(aval+rp[i], count, JX_MPI_REAL, 0, JX_AVAL_TAG, jx_comm_dh, rcv_req+2*i+1);
    }
    if (jx_myid_dh == 0)
    {
        jx_MPI_Waitall(m*2, send_req, send_status);
    }
    jx_MPI_Waitall(2*B->m, rcv_req, rcv_status);
    if (rowLengths != NULL)
    {
        JX_FREE_DH(rowLengths); JX_CHECK_V_ERROR;
    }
    if (o2n_row != NULL)
    {
        JX_FREE_DH(o2n_row); JX_CHECK_V_ERROR;
    }
    if (n2o_col != NULL)
    {
        JX_FREE_DH(n2o_col); JX_CHECK_V_ERROR;
    }
    if (rowToBlock != NULL)
    {
        JX_FREE_DH(rowToBlock); JX_CHECK_V_ERROR;
    }
    if (send_req != NULL)
    {
        JX_FREE_DH(send_req); JX_CHECK_V_ERROR;
    }
    if (rcv_req != NULL)
    {
        JX_FREE_DH(rcv_req); JX_CHECK_V_ERROR;
    }
    if (send_status != NULL)
    {
        JX_FREE_DH(send_status); JX_CHECK_V_ERROR;
    }
    if (rcv_status != NULL)
    {
        JX_FREE_DH(rcv_status); JX_CHECK_V_ERROR;
    }
    if (beg_row != NULL)
    {
        JX_FREE_DH(beg_row); JX_CHECK_V_ERROR;
    }
    if (row_count != NULL)
    {
        JX_FREE_DH(row_count); JX_CHECK_V_ERROR;
    }
    if (C != NULL)
    {
        jx_Mat_dhDestroy(C); JX_ERRCHKA;
    }
   *Bout = B;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_partition_and_distribute_private"
void jx_partition_and_distribute_private( jx_Mat_dh A, jx_Mat_dh *Bout )
{
    JX_START_FUNC_DH
    jx_Mat_dh B = NULL;
    JX_Int i, m;
    JX_Int *rowLengths = NULL;
    JX_Int *o2n_row = NULL, *n2o_col = NULL, *rowToBlock = NULL;
    MPI_Request *send_req = NULL;
    MPI_Request *rcv_req = NULL;
    MPI_Status *send_status = NULL;
    MPI_Status *rcv_status = NULL;
    
    jx_MPI_Barrier(jx_comm_dh);
    if (jx_myid_dh == 0) m = A->m;
    jx_MPI_Bcast(&m, 1, JX_MPI_INT, 0, MPI_COMM_WORLD);
    rowLengths = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        JX_Int *tmp = A->rp;
        for (i = 0; i < m; ++ i)
        {
            rowLengths[i] = tmp[i+1] - tmp[i];
        }
    }
    jx_MPI_Bcast(rowLengths, m, JX_MPI_INT, 0, jx_comm_dh);
    rowToBlock = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        o2n_row = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        jx_mat_partition_private(A, jx_np_dh, o2n_row, rowToBlock); JX_CHECK_V_ERROR;
    }
    jx_MPI_Bcast(rowToBlock, m, JX_MPI_INT, 0, jx_comm_dh);
    jx_mat_par_read_allocate_private(&B, m, rowLengths, rowToBlock); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        JX_Int *cval = A->cval, *rp = A->rp;
        JX_Real *aval = A->aval;
        
        send_req = (MPI_Request *)JX_MALLOC_DH(2*m*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
        send_status = (MPI_Status *)JX_MALLOC_DH(2*m*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
        for (i = 0; i < m; ++ i)
        {
            JX_Int owner = rowToBlock[i];
            JX_Int count = rp[i+1]-rp[i];
            
            if (!count)
            {
                jx_sprintf(jx_msgBuf_dh, "row %i of %i is empty!", i+1, m);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
            jx_MPI_Isend(cval+rp[i], count, JX_MPI_INT, owner, JX_CVAL_TAG, jx_comm_dh, send_req+2*i);
            jx_MPI_Isend(aval+rp[i], count, JX_MPI_REAL, owner, JX_AVAL_TAG, jx_comm_dh, send_req+2*i+1);
        }
    }
    JX_Int *cval = B->cval;
    JX_Int *rp = B->rp;
    JX_Real *aval = B->aval;
    m = B->m;
    rcv_req = (MPI_Request *)JX_MALLOC_DH(2*m*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    rcv_status = (MPI_Status *)JX_MALLOC_DH(2*m*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i)
    {
        JX_Int count = rp[i+1] - rp[i];
        
        if (!count)
        {
            jx_sprintf(jx_msgBuf_dh, "local row %i of %i is empty!", i+1, m);
            JX_SET_V_ERROR(jx_msgBuf_dh);
        }
        jx_MPI_Irecv(cval+rp[i], count, JX_MPI_INT, 0, JX_CVAL_TAG, jx_comm_dh, rcv_req+2*i);
        jx_MPI_Irecv(aval+rp[i], count, JX_MPI_REAL, 0, JX_AVAL_TAG, jx_comm_dh, rcv_req+2*i+1);
    }
    if (jx_myid_dh == 0)
    {
        jx_MPI_Waitall(m*2, send_req, send_status);
    }
    jx_MPI_Waitall(2*B->m, rcv_req, rcv_status);
    if (rowLengths != NULL)
    {
        JX_FREE_DH(rowLengths); JX_CHECK_V_ERROR;
    }
    if (o2n_row != NULL)
    {
        JX_FREE_DH(o2n_row); JX_CHECK_V_ERROR;
    }
    if (n2o_col != NULL)
    {
        JX_FREE_DH(n2o_col); JX_CHECK_V_ERROR;
    }
    if (rowToBlock != NULL)
    {
        JX_FREE_DH(rowToBlock); JX_CHECK_V_ERROR;
    }
    if (send_req != NULL)
    {
        JX_FREE_DH(send_req); JX_CHECK_V_ERROR;
    }
    if (rcv_req != NULL)
    {
        JX_FREE_DH(rcv_req); JX_CHECK_V_ERROR;
    }
    if (send_status != NULL)
    {
        JX_FREE_DH(send_status); JX_CHECK_V_ERROR;
    }
    if (rcv_status != NULL)
    {
        JX_FREE_DH(rcv_status); JX_CHECK_V_ERROR;
    }
   *Bout = B;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_par_read_allocate_private"
void jx_mat_par_read_allocate_private( jx_Mat_dh *Aout, JX_Int n, JX_Int *rowLengths, JX_Int *rowToBlock )
{
    JX_START_FUNC_DH
    jx_Mat_dh A;
    JX_Int i, m, nz, beg_row, *rp, idx;
    
    jx_Mat_dhCreate(&A); JX_CHECK_V_ERROR;
   *Aout = A;
    A->n = n;
    m = 0;
    for (i = 0; i < n; ++ i)
    {
        if (rowToBlock[i] == jx_myid_dh) ++ m;
    }
    A->m = m;
    beg_row = 0;
    for (i = 0; i < n; ++ i)
    {
        if (rowToBlock[i] < jx_myid_dh) ++ beg_row;
    }
    A->beg_row = beg_row;
    A->rp = rp = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    rp[0] = 0;
    nz = 0;
    idx = 1;
    for (i = 0; i < n; ++ i)
    {
        if (rowToBlock[i] == jx_myid_dh)
        {
            nz += rowLengths[i];
            rp[idx++] = nz;
        }
    }
    A->cval = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    A->aval = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_mat_partition_private"
void jx_mat_partition_private( jx_Mat_dh A, JX_Int blocks, JX_Int *o2n_row, JX_Int *rowToBlock )
{
    JX_START_FUNC_DH
    JX_Int i, j, n = A->n;
    JX_Int rpb = n / blocks;
    JX_Int idx = 0;
    
    while (rpb*blocks < n) ++ rpb;
    if (rpb*(blocks-1) == n)
    {
        -- rpb;
        jx_printf_dh("adjusted rpb to: %i\n", rpb);
    }
    for (i = 0; i < n; ++ i) o2n_row[i] = i;
    for (i = 0; i < blocks-1; ++ i)
    {
        for (j = 0; j < rpb; ++ j)
        {
            rowToBlock[idx++] = i;
        }
    }
    i = blocks - 1;
    while (idx < n) rowToBlock[idx++] = i;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_make_full_private"
void jx_make_full_private( JX_Int m, JX_Int **rpIN, JX_Int **cvalIN, JX_Real **avalIN )
{
    JX_START_FUNC_DH
    JX_Int i, j, *rpNew, *cvalNew, *rp = *rpIN, *cval = *cvalIN;
    JX_Real *avalNew, *aval = *avalIN;
    JX_Int nz, *rowCounts = NULL;
    
    rowCounts = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i <= m; ++ i) rowCounts[i] = 0;
    for (i = 0; i < m; ++ i)
    {
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            JX_Int col = cval[j];
            
            rowCounts[i+1] += 1;
            if (col != i) rowCounts[col+1] += 1;
        }
    }
    rpNew = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 1; i <= m; ++ i) rowCounts[i] += rowCounts[i-1];
    memcpy(rpNew, rowCounts, (m+1)*sizeof(JX_Int));
    nz = rpNew[m];
    cvalNew = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    avalNew = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i)
    {
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            JX_Int col = cval[j];
            JX_Real val  = aval[j];
            
            cvalNew[rowCounts[i]] = col;
            avalNew[rowCounts[i]] = val;
            rowCounts[i] += 1;
            if (col != i)
            {
                cvalNew[rowCounts[col]] = i;
                avalNew[rowCounts[col]] = val;
                rowCounts[col] += 1;
            }
        }
    }
    if (rowCounts != NULL)
    {
        JX_FREE_DH(rowCounts); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(cval); JX_CHECK_V_ERROR;
    JX_FREE_DH(rp); JX_CHECK_V_ERROR;
    JX_FREE_DH(aval); JX_CHECK_V_ERROR;
   *rpIN = rpNew;
   *cvalIN = cvalNew;
   *avalIN = avalNew;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_make_symmetric_private"
void jx_make_symmetric_private( JX_Int m, JX_Int **rpIN, JX_Int **cvalIN, JX_Real **avalIN )
{
    JX_START_FUNC_DH
    JX_Int i, j, *rpNew, *cvalNew, *rp = *rpIN, *cval = *cvalIN;
    JX_Real *avalNew, *aval = *avalIN;
    JX_Int nz, *rowCounts = NULL;
    JX_Int *rpTrans, *cvalTrans;
    JX_Int *work;
    JX_Real *avalTrans;
    JX_Int nzCount = 0, transCount = 0;
    
    jx_mat_dh_transpose_private(m, rp, &rpTrans, cval, &cvalTrans, aval, &avalTrans); JX_CHECK_V_ERROR;
    work = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) work[i] = -1;
    rowCounts = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i <= m; ++ i) rowCounts[i] = 0;
    for (i = 0; i < m; ++ i)
    {
        JX_Int ct = 0;
        
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            JX_Int col = cval[j];
            
            work[col] = i;
            ++ ct;
            ++ nzCount;
        }
        for (j = rpTrans[i]; j < rpTrans[i+1]; ++ j)
        {
            JX_Int col = cvalTrans[j];
            
            if (work[col] != i)
            {
                ++ ct;
                ++ transCount;
            }
        }
        rowCounts[i+1] = ct;
    }
    if (transCount == 0)
    {
        jx_printf("jx_make_symmetric_private: matrix is already structurally symmetric!\n");
        JX_FREE_DH(rpTrans); JX_CHECK_V_ERROR;
        JX_FREE_DH(cvalTrans); JX_CHECK_V_ERROR;
        JX_FREE_DH(avalTrans); JX_CHECK_V_ERROR;
        JX_FREE_DH(work); JX_CHECK_V_ERROR;
        JX_FREE_DH(rowCounts); JX_CHECK_V_ERROR;
        goto END_OF_FUNCTION;
    }
    else
    {
        jx_printf("original nz= %i\n", rp[m]);
        jx_printf("zeros added= %i\n", transCount);
        jx_printf("ratio of added zeros to nonzeros = %0.2f (assumes all original entries were nonzero!)\n",
                                                                       (JX_Real)transCount/(JX_Real)(nzCount));
    }
    rpNew = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 1; i <= m; ++ i) rowCounts[i] += rowCounts[i-1];
    memcpy(rpNew, rowCounts, (m+1)*sizeof(JX_Int));
    for (i = 0; i < m; ++ i) work[i] = -1;
    nz = rpNew[m];
    cvalNew = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    avalNew = (JX_Real *)JX_MALLOC_DH(nz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) work[i] = -1;
    for (i = 0; i < m; ++ i)
    {
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            JX_Int col = cval[j];
            JX_Real val  = aval[j];
            
            work[col] = i;
            cvalNew[rowCounts[i]] = col;
            avalNew[rowCounts[i]] = val;
            rowCounts[i] += 1;
        }
        for (j = rpTrans[i]; j < rpTrans[i+1]; ++ j)
        {
            JX_Int col = cvalTrans[j];
            
            if (work[col] != i)
            {
                cvalNew[rowCounts[i]] = col;
                avalNew[rowCounts[i]] = 0.0;
                rowCounts[i] += 1;
            }
        }
    }
    if (rowCounts != NULL)
    {
        JX_FREE_DH(rowCounts); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(work); JX_CHECK_V_ERROR;
    JX_FREE_DH(cval); JX_CHECK_V_ERROR;
    JX_FREE_DH(rp); JX_CHECK_V_ERROR;
    JX_FREE_DH(aval); JX_CHECK_V_ERROR;
    JX_FREE_DH(cvalTrans); JX_CHECK_V_ERROR;
    JX_FREE_DH(rpTrans); JX_CHECK_V_ERROR;
    JX_FREE_DH(avalTrans); JX_CHECK_V_ERROR;
   *rpIN = rpNew;
   *cvalIN = cvalNew;
   *avalIN = avalNew;
    
END_OF_FUNCTION: ;
    
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_profileMat"
void jx_profileMat( jx_Mat_dh A )
{
    JX_START_FUNC_DH
    jx_Mat_dh B = NULL;
    JX_Int type;
    JX_Int m;
    JX_Int i, j;
    JX_Int *work1 = NULL;
    JX_Real *work2 = NULL;
    jx_bool isStructurallySymmetric = jx_true;
    jx_bool isNumericallySymmetric = jx_true;
    jx_bool is_Triangular = jx_false;
    JX_Int zeroCount = 0, nz;
    
    if (jx_myid_dh > 0)
    {
        JX_SET_V_ERROR("only for a single jx_MPI task!");
    }
    m = A->m;
    jx_printf("\nYY----------------------------------------------------\n");
    nz = A->rp[m];
    for (i = 0; i < nz; ++ i)
    {
        if (A->aval[i] == 0) ++ zeroCount;
    }
    jx_printf("YY  row count:      %i\n", m);
    jx_printf("YY  nz count:       %i\n", nz);
    jx_printf("YY  explicit zeros: %i (entire matrix)\n", zeroCount);
    JX_Int m_diag = 0, z_diag = 0;
    for (i = 0; i < m; ++ i)
    {
        jx_bool flag = jx_true;
        
        for (j = A->rp[i]; j < A->rp[i+1]; ++ j)
        {
            JX_Int col = A->cval[j];
            
            if (col == i)
            {
                JX_Real val = A->aval[j];
                
                flag = jx_false;
                if (val == 0.0) ++ z_diag;
                break;
            }
        }
        if (flag) ++ m_diag;
    }
    jx_printf("YY  missing diagonals:   %i\n", m_diag);
    jx_printf("YY  explicit zero diags: %i\n", z_diag);
    type = jx_isTriangular(m, A->rp, A->cval); JX_CHECK_V_ERROR;
    if (type == JX_IS_UPPER_TRI)
    {
        jx_printf("YY  matrix is upper triangular\n");
        is_Triangular = jx_true;
        goto END_OF_FUNCTION;
    }
    else if (type == JX_IS_LOWER_TRI)
    {
        jx_printf("YY  matrix is lower triangular\n");
        is_Triangular = jx_true;
        goto END_OF_FUNCTION;
    }
    JX_Int unz = 0, lnz = 0;
    for (i = 0; i < m; ++ i)
    {
        for (j = A->rp[i]; j < A->rp[i+1]; ++ j)
        {
            JX_Int col = A->cval[j];
            
            if (col < i) ++ lnz;
            if (col > i) ++ unz;
        }
    }
    jx_printf("YY  strict upper triangular nonzeros: %i\n", unz);
    jx_printf("YY  strict lower triangular nonzeros: %i\n", lnz);
    jx_Mat_dhTranspose(A, &B); JX_CHECK_V_ERROR;
    work1 = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    work2 = (JX_Real *)JX_MALLOC_DH(m*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) work1[i] = -1;
    for (i = 0; i < m; ++ i) work2[i] = 0.0;
    for (i = 0; i < m; ++ i)
    {
        for (j = A->rp[i]; j < A->rp[i+1]; ++ j)
        {
            JX_Int col = A->cval[j];
            JX_Real val = A->aval[j];
            
            work1[col] = i;
            work2[col] = val;
        }
        for (j = B->rp[i]; j < B->rp[i+1]; ++ j)
        {
            JX_Int col = B->cval[j];
            JX_Real val = B->aval[j];
            
            if (work1[col] != i)
            {
                isStructurallySymmetric = jx_false;
                isNumericallySymmetric = jx_false;
                goto END_OF_FUNCTION;
            }
            if (work2[col] != val)
            {
                isNumericallySymmetric = jx_false;
                work2[col] = 0.0;
            }
        }
    }
    
END_OF_FUNCTION: ;
    
    if (!is_Triangular)
    {
        jx_printf("YY  matrix is NOT triangular\n");
        if (isStructurallySymmetric)
        {
            jx_printf("YY  matrix IS structurally symmetric\n");
        }
        else
        {
            jx_printf("YY  matrix is NOT structurally symmetric\n");
        }
        if (isNumericallySymmetric)
        {
            jx_printf("YY  matrix IS numerically symmetric\n");
        }
        else
        {
            jx_printf("YY  matrix is NOT numerically symmetric\n");
        }
    }
    if (work1 != NULL)
    {
        JX_FREE_DH(work1); JX_CHECK_V_ERROR;
    }
    if (work2 != NULL)
    {
        JX_FREE_DH(work2); JX_CHECK_V_ERROR;
    }
    if (B != NULL)
    {
        jx_Mat_dhDestroy(B); JX_CHECK_V_ERROR;
    }
    jx_printf("YY----------------------------------------------------\n");
    JX_END_FUNC_DH
}
