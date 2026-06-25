//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  vec_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"
#include <stdlib.h>

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhCreate"
void jx_Vec_dhCreate( jx_Vec_dh *v )
{
    JX_START_FUNC_DH
    struct _jx_vec_dh *tmp = (struct _jx_vec_dh *)JX_MALLOC_DH(sizeof(struct _jx_vec_dh)); JX_CHECK_V_ERROR;
   *v = tmp;
    tmp->n = 0;
    tmp->vals = NULL;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhDestroy"
void jx_Vec_dhDestroy( jx_Vec_dh v )
{
    JX_START_FUNC_DH
    if (v->vals != NULL) JX_FREE_DH(v->vals); JX_CHECK_V_ERROR;
    JX_FREE_DH(v); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhInit"
void jx_Vec_dhInit( jx_Vec_dh v, JX_Int size )
{
    JX_START_FUNC_DH
    v->n = size;
    v->vals = (JX_Real *)JX_MALLOC_DH(size*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhCopy"
void jx_Vec_dhCopy( jx_Vec_dh x, jx_Vec_dh y )
{
    JX_START_FUNC_DH
    if (x->vals == NULL) JX_SET_V_ERROR("x->vals is NULL");
    if (y->vals == NULL) JX_SET_V_ERROR("y->vals is NULL");
    if (x->n != y->n) JX_SET_V_ERROR("x and y are different lengths");
    memcpy(y->vals, x->vals, x->n*sizeof(JX_Real));
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhDuplicate"
void jx_Vec_dhDuplicate( jx_Vec_dh v, jx_Vec_dh *out )
{
    JX_START_FUNC_DH
    jx_Vec_dh tmp;
    JX_Int size = v->n;
    
    if (v->vals == NULL) JX_SET_V_ERROR("v->vals is NULL");
    jx_Vec_dhCreate(out); JX_CHECK_V_ERROR;
    tmp = *out;
    tmp->n = size;
    tmp->vals = (JX_Real *)JX_MALLOC_DH(size*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhSet"
void jx_Vec_dhSet( jx_Vec_dh v, JX_Real value )
{
    JX_START_FUNC_DH
    JX_Int i, m = v->n;
    JX_Real *vals = v->vals;
    
    if (v->vals == NULL) JX_SET_V_ERROR("v->vals is NULL");
    for (i = 0; i < m; ++ i) vals[i] = value;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhSetRand"
void jx_Vec_dhSetRand( jx_Vec_dh v )
{
    JX_START_FUNC_DH
    JX_Int i, m = v->n;
    JX_Real max = 0.0;
    JX_Real *vals = v->vals;
    
    if (v->vals == NULL) JX_SET_V_ERROR("v->vals is NULL");
    for (i = 0; i < m; ++ i) vals[i] = rand();
    for (i = 0; i < m; ++ i) max = JX_MAX(max, vals[i]);
    for (i = 0; i < m; ++ i) vals[i] = vals[i]/max;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhPrint"
void jx_Vec_dhPrint( jx_Vec_dh v, jx_SubdomainGraph_dh sg, char *filename )
{
    JX_START_FUNC_DH
    JX_Real *vals = v->vals;
    JX_Int pe, i, m = v->n;
    FILE *fp;
    
    if (v->vals == NULL) JX_SET_V_ERROR("v->vals is NULL");
    if (sg == NULL)
    {
        for (pe = 0; pe < jx_np_dh; ++ pe)
        {
            jx_MPI_Barrier(jx_comm_dh);
            if (pe == jx_myid_dh)
            {
                if (pe == 0)
                {
                    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
                }
                else
                {
                    fp = jx_openFile_dh(filename, "a"); JX_CHECK_V_ERROR;
                }
                for (i = 0; i < m; ++ i) jx_fprintf(fp, "%g\n", vals[i]);
                jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
            }
        }
    }
    else if (jx_np_dh == 1)
    {
        JX_Int i, j;
        
        fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
        for (i = 0; i < sg->blocks; ++ i)
        {
            JX_Int oldBlock = sg->n2o_sub[i];
            JX_Int beg_row = sg->beg_rowP[oldBlock];
            JX_Int end_row = beg_row + sg->row_count[oldBlock];
            
            jx_printf("seq: block= %i  beg= %i  end= %i\n", oldBlock, beg_row, end_row);
            for (j = beg_row; j < end_row; ++ j)
            {
                jx_fprintf(fp, "%g\n", vals[j]);
            }
        }
    }
    else
    {
        JX_Int id = sg->o2n_sub[jx_myid_dh];
        
        for (pe = 0; pe < jx_np_dh; ++ pe)
        {
            jx_MPI_Barrier(jx_comm_dh);
            if (id == pe)
            {
                if (pe == 0)
                {
                    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
                }
                else
                {
                    fp = jx_openFile_dh(filename, "a"); JX_CHECK_V_ERROR;
                }
                jx_fprintf(stderr, "par: block= %i\n", id);
                for (i = 0; i < m; ++ i)
                {
                    jx_fprintf(fp, "%g\n", vals[i]);
                }
                jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhPrintBIN"
void jx_Vec_dhPrintBIN( jx_Vec_dh v, jx_SubdomainGraph_dh sg, char *filename )
{
    JX_START_FUNC_DH
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single jx_MPI task");
    }
    if (sg != NULL)
    {
        JX_SET_V_ERROR("not implemented for reordered vector; ensure sg=NULL");
    }
    jx_io_dh_print_ebin_vec_private(v->n, 0, v->vals, NULL, NULL, NULL, filename); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#define JX_MAX_JUNK 200

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhRead"
void jx_Vec_dhRead( jx_Vec_dh *vout, JX_Int ignore, char *filename )
{
    JX_START_FUNC_DH
    jx_Vec_dh tmp;
    FILE *fp;
    JX_Int items, n, i;
    JX_Real *v, w;
    char junk[JX_MAX_JUNK];
    
    jx_Vec_dhCreate(&tmp); JX_CHECK_V_ERROR;
   *vout = tmp;
    if (jx_np_dh > 1)
    {
        JX_SET_V_ERROR("only implemented for a single jx_MPI task");
    }
    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
    if (ignore)
    {
        jx_printf("jx_Vec_dhRead:: ignoring following header lines:\n");
        jx_printf("--------------------------------------------------------------\n");
        for (i = 0; i < ignore; ++ i)
        {
            fgets(junk, JX_MAX_JUNK, fp);
            jx_printf("%s", junk);
        }
        jx_printf("--------------------------------------------------------------\n");
    }
    n = 0;
    while (!feof(fp))
    {
        items = jx_fscanf(fp,"%lg", &w);
        if (items != 1)
        {
            break;
        }
        ++ n;
    }
    jx_printf("jx_Vec_dhRead:: n= %i\n", n);
    tmp->n = n;
    v = tmp->vals =  (JX_Real *)JX_MALLOC_DH(n*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    rewind(fp);
    rewind(fp);
    for (i = 0; i < ignore; ++ i)
    {
        fgets(junk, JX_MAX_JUNK, fp);
    }
    for (i = 0; i < n;  ++ i)
    {
        items = jx_fscanf(fp,"%lg", v+i);
        if (items != 1)
        {
            jx_sprintf(jx_msgBuf_dh, "failed to read value %i of %i", i+1, n);
        }
    }
    jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_Vec_dhReadBIN"
extern void jx_Vec_dhReadBIN( jx_Vec_dh *vout, char *filename )
{
    JX_START_FUNC_DH
    jx_Vec_dh tmp;
    
    jx_Vec_dhCreate(&tmp); JX_CHECK_V_ERROR;
   *vout = tmp;
    jx_io_dh_read_ebin_vec_private(&tmp->n, &tmp->vals, filename); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}
