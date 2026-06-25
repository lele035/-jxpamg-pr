//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  subdomain_graph_dh.c
 *  Date: 2013/01/22
 */

#include "jx_euclid.h"

#ifndef WIN32
#include <unistd.h>
#endif

static void jx_init_seq_private( jx_SubdomainGraph_dh s, JX_Int blocks, jx_bool bj, void *A );
static void jx_init_mpi_private( jx_SubdomainGraph_dh s, JX_Int blocks, jx_bool bj, void *A );
static void jx_allocate_storage_private( jx_SubdomainGraph_dh s, JX_Int blocks, JX_Int m, jx_bool bj );
static void jx_form_subdomaingraph_mpi_private( jx_SubdomainGraph_dh s );
static void jx_form_subdomaingraph_seq_private( jx_SubdomainGraph_dh s, JX_Int m, void *A );
static void jx_find_all_neighbors_sym_private( jx_SubdomainGraph_dh s, JX_Int m, void *A );
static void jx_find_all_neighbors_unsym_private( jx_SubdomainGraph_dh s, JX_Int m, void *A );
static void jx_find_bdry_nodes_sym_private( jx_SubdomainGraph_dh s, JX_Int m, void *A,
                JX_Int *interiorNodes, JX_Int *bdryNodes, JX_Int *interiorCount, JX_Int *bdryCount );
static void jx_find_bdry_nodes_unsym_private( jx_SubdomainGraph_dh s, JX_Int m, void *A,
                   JX_Int *interiorNodes, JX_Int *bdryNodes, JX_Int *interiorCount, JX_Int *bdryCount );
static void jx_find_bdry_nodes_seq_private( jx_SubdomainGraph_dh s, JX_Int m, void *A );
static void jx_find_ordered_neighbors_private( jx_SubdomainGraph_dh s );
static void jx_color_subdomain_graph_private( jx_SubdomainGraph_dh s );
static void jx_adjust_matrix_perms_private( jx_SubdomainGraph_dh s, JX_Int m );

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhCreate"
void jx_SubdomainGraph_dhCreate( jx_SubdomainGraph_dh *s )
{
    JX_START_FUNC_DH
    struct _jx_subdomain_dh *tmp = (struct _jx_subdomain_dh *)JX_MALLOC_DH(sizeof(struct _jx_subdomain_dh)); JX_CHECK_V_ERROR;
   *s = tmp;
    tmp->blocks = 1;
    tmp->ptrs = tmp->adj = NULL;
    tmp->colors = 1;
    tmp->colorVec = NULL;
    tmp->o2n_sub = tmp->n2o_sub = NULL;
    tmp->beg_row = tmp->beg_rowP = NULL;
    tmp->bdry_count = tmp->row_count = NULL;
    tmp->loNabors = tmp->hiNabors = tmp->allNabors = NULL;
    tmp->loCount = tmp->hiCount = tmp->allCount = 0;
    tmp->m = 0;
    tmp->n2o_row = tmp->o2n_col = NULL;
    tmp->o2n_ext = tmp->n2o_ext = NULL;
    tmp->doNotColor = jx_Parser_dhHasSwitch(jx_parser_dh, "-doNotColor");
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_SubGraph");
    JX_Int i;
    for (i = 0; i < JX_JX_TIMING_BINS_SG; ++ i) tmp->timing[i] = 0.0;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhDestroy"
void jx_SubdomainGraph_dhDestroy( jx_SubdomainGraph_dh s )
{
    JX_START_FUNC_DH
    if (s->ptrs != NULL)
    {
        JX_FREE_DH(s->ptrs); JX_CHECK_V_ERROR;
    }
    if (s->adj != NULL)
    {
        JX_FREE_DH(s->adj); JX_CHECK_V_ERROR;
    }
    if (s->colorVec != NULL)
    {
        JX_FREE_DH(s->colorVec); JX_CHECK_V_ERROR;
    }
    if (s->o2n_sub != NULL)
    {
        JX_FREE_DH(s->o2n_sub); JX_CHECK_V_ERROR;
    }
    if (s->n2o_sub != NULL)
    {
        JX_FREE_DH(s->n2o_sub); JX_CHECK_V_ERROR;
    }
    if (s->beg_row != NULL)
    {
        JX_FREE_DH(s->beg_row); JX_CHECK_V_ERROR;
    }
    if (s->beg_rowP != NULL)
    {
        JX_FREE_DH(s->beg_rowP); JX_CHECK_V_ERROR;
    }
    if (s->row_count != NULL)
    {
        JX_FREE_DH(s->row_count); JX_CHECK_V_ERROR;
    }
    if (s->bdry_count != NULL)
    {
        JX_FREE_DH(s->bdry_count); JX_CHECK_V_ERROR;
    }
    if (s->loNabors != NULL)
    {
        JX_FREE_DH(s->loNabors); JX_CHECK_V_ERROR;
    }
    if (s->hiNabors != NULL)
    {
        JX_FREE_DH(s->hiNabors); JX_CHECK_V_ERROR;
    }
    if (s->allNabors != NULL)
    {
        JX_FREE_DH(s->allNabors); JX_CHECK_V_ERROR;
    }
    if (s->n2o_row != NULL)
    {
        JX_FREE_DH(s->n2o_row); JX_CHECK_V_ERROR;
    }
    if (s->o2n_col != NULL)
    {
        JX_FREE_DH(s->o2n_col); JX_CHECK_V_ERROR;
    }
    if (s->o2n_ext != NULL)
    {
        jx_Hash_i_dhDestroy(s->o2n_ext); JX_CHECK_V_ERROR;
    }
    if (s->n2o_ext != NULL)
    {
        jx_Hash_i_dhDestroy(s->n2o_ext); JX_CHECK_V_ERROR;
    }
    JX_FREE_DH(s); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhInit"
void jx_SubdomainGraph_dhInit( jx_SubdomainGraph_dh s, JX_Int blocks, jx_bool bj, void *A )
{
    JX_START_FUNC_DH
    JX_Real t1 = jx_MPI_Wtime();
    
    if (blocks < 1) blocks = 1;
    if (jx_np_dh == 1 || blocks > 1)
    {
        s->blocks = blocks;
        jx_init_seq_private(s, blocks, bj, A); JX_CHECK_V_ERROR;
    }
    else
    {
        s->blocks = jx_np_dh;
        jx_init_mpi_private(s, jx_np_dh, bj, A); JX_CHECK_V_ERROR;
    }
    s->timing[JX_TOTAL_SGT] += (jx_MPI_Wtime() - t1);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhFindOwner"
JX_Int jx_SubdomainGraph_dhFindOwner( jx_SubdomainGraph_dh s, JX_Int idx, jx_bool permuted )
{
    JX_START_FUNC_DH
    JX_Int sd;
    JX_Int *beg_row = s->beg_row;
    JX_Int *row_count = s->row_count;
    JX_Int owner = -1, blocks = s->blocks;
    
    if (permuted) beg_row = s->beg_rowP;
    for (sd = 0; sd < blocks; ++ sd)
    {
        if (idx >= beg_row[sd] && idx < beg_row[sd]+row_count[sd])
        {
            owner = sd;
            break;
        }
    }
    if (owner == -1)
    {
        jx_fprintf(stderr, "@@@ failed to jx_find owner for idx = %i @@@\n", idx);
        jx_fprintf(stderr, "blocks= %i\n", blocks);
        jx_sprintf(jx_msgBuf_dh, "failed to jx_find owner for idx = %i", idx);
        JX_SET_ERROR(-1, jx_msgBuf_dh);
    }
    JX_END_FUNC_VAL(owner)
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhPrintStatsLong"
void jx_SubdomainGraph_dhPrintStatsLong( jx_SubdomainGraph_dh s, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i, j, k;
    JX_Real max = 0, min = INT_MAX;
    
    jx_fprintf(fp, "\n------------- jx_SubdomainGraph_dhPrintStatsLong -----------\n");
    jx_fprintf(fp, "colors used     = %i\n", s->colors);
    jx_fprintf(fp, "subdomain count = %i\n", s->blocks);
    jx_fprintf(fp, "\ninterior/boundary node ratios:\n");
    for (i = 0; i < s->blocks; ++ i)
    {
        JX_Int inNodes = s->row_count[i] - s->bdry_count[i];
        JX_Int bdNodes = s->bdry_count[i];
        JX_Real ratio;
        
        if (bdNodes == 0)
        {
            ratio = -1;
        }
        else
        {
            ratio = (JX_Real)inNodes / (JX_Real)bdNodes;
        }
        max = JX_MAX(max, ratio);
        min = JX_MIN(min, ratio);
        jx_fprintf(fp, "   P_%i: first= %3i  rowCount= %3i  interior= %3i  bdry= %3i  ratio= %0.1f\n",
                                          i, 1+s->beg_row[i], s->row_count[i], inNodes, bdNodes, ratio);
    }
    jx_fprintf(fp, "\nmax interior/bdry ratio = %.1f\n", max);
    jx_fprintf(fp, "min interior/bdry ratio = %.1f\n", min);
    if (s->adj != NULL)
    {
        jx_fprintf(fp, "\nunpermuted subdomain graph: \n");
        for (i = 0; i < s->blocks; ++ i)
        {
            jx_fprintf(fp, "%i :: ", i);
            for (j = s->ptrs[i]; j < s->ptrs[i+1]; ++ j)
            {
                jx_fprintf(fp, "%i  ", s->adj[j]);
            }
            jx_fprintf(fp, "\n");
        }
    }
    jx_fprintf(fp, "\no2n subdomain permutation:\n");
    for (i = 0; i < s->blocks; ++ i)
    {
        jx_fprintf(fp, "  %i %i\n", i, s->o2n_sub[i]);
    }
    jx_fprintf(fp, "\n");
    if (jx_np_dh > 1)
    {
        jx_fprintf(fp, "\nlocal n2o_row permutation:\n   ");
        for (i = 0; i < s->row_count[jx_myid_dh]; ++ i)
        {
            jx_fprintf(fp, "%i ", s->n2o_row[i]);
        }
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "\nlocal o2n_col permutation:\n   ");
        for (i = 0; i < s->row_count[jx_myid_dh]; ++ i)
        {
            jx_fprintf(fp, "%i ", s->o2n_col[i]);
        }
        jx_fprintf(fp, "\n");
    }
    else
    {
        jx_fprintf(fp, "\nlocal n2o_row permutation:\n");
        jx_fprintf(fp, "--------------------------\n");
        for (k = 0; k < s->blocks; ++ k)
        {
            JX_Int beg_row = s->beg_row[k];
            JX_Int end_row = beg_row + s->row_count[k];
            
            for (i = beg_row; i < end_row; ++ i)
            {
                jx_fprintf(fp, "%i ", s->n2o_row[i]);
            }
            jx_fprintf(fp, "\n");
        }
        jx_fprintf(fp, "\nlocal o2n_col permutation:\n");
        jx_fprintf(fp, "--------------------------\n");
        for (k = 0; k < s->blocks; ++ k)
        {
            JX_Int beg_row = s->beg_row[k];
            JX_Int end_row = beg_row + s->row_count[k];
            
            for (i = beg_row; i < end_row; ++ i)
            {
                jx_fprintf(fp, "%i ", s->o2n_col[i]);
            }
            jx_fprintf(fp, "\n");
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_init_seq_private"
void jx_init_seq_private( jx_SubdomainGraph_dh s, JX_Int blocks, jx_bool bj, void *A )
{
    JX_START_FUNC_DH
    JX_Int m, n, beg_row;
    JX_Real t1;
    
    jx_EuclidGetDimensions(A, &beg_row, &m, &n); JX_CHECK_V_ERROR;
    s->m = n;
    jx_allocate_storage_private(s,blocks,m, bj); JX_CHECK_V_ERROR;
    JX_Int i;
    JX_Int rpp = m / blocks;
    if (rpp*blocks < m) ++ rpp;
    s->beg_row[0] = 0;
    for (i = 1; i < blocks; ++ i) s->beg_row[i] = rpp + s->beg_row[i-1];
    for (i = 0; i < blocks; ++ i) s->row_count[i] = rpp;
    s->row_count[blocks-1] = m - rpp * (blocks - 1);
    memcpy(s->beg_rowP, s->beg_row, blocks*sizeof(JX_Int));
    t1 = jx_MPI_Wtime();
    if (!bj)
    {
        jx_find_bdry_nodes_seq_private(s, m, A); JX_CHECK_V_ERROR;
    }
    else
    {
        JX_Int i;
        
        for (i = 0; i < m; ++ i)
        {
            s->n2o_row[i] = i;
            s->o2n_col[i] = i;
        }
    }
    s->timing[JX_ORDER_BDRY_SGT] += (jx_MPI_Wtime() - t1);
    t1 = jx_MPI_Wtime();
    if (!bj)
    {
        jx_form_subdomaingraph_seq_private(s, m, A); JX_CHECK_V_ERROR;
        if (s->doNotColor)
        {
            JX_Int i;
            
            jx_printf_dh("subdomain coloring and reordering is OFF\n");
            for (i = 0; i < blocks; ++ i)
            {
                s->o2n_sub[i] = i;
                s->n2o_sub[i] = i;
                s->colorVec[i] = 0;
            }
        }
        else
        {
            JX_SET_INFO("subdomain coloring and reordering is ON");
            jx_color_subdomain_graph_private(s); JX_CHECK_V_ERROR;
        }
    }
    else
    {
        JX_Int i;
        
        for (i = 0; i < blocks; ++ i)
        {
            s->o2n_sub[i] = i;
            s->n2o_sub[i] = i;
        }
    }
    s->timing[JX_FORM_GRAPH_SGT] += (jx_MPI_Wtime() - t1);
    if (!bj)
    {
        jx_adjust_matrix_perms_private(s, m); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_allocate_storage_private"
void jx_allocate_storage_private( jx_SubdomainGraph_dh s, JX_Int blocks, JX_Int m, jx_bool bj )
{
    JX_START_FUNC_DH
    if (!bj)
    {
        s->ptrs = (JX_Int *)JX_MALLOC_DH((blocks+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        s->ptrs[0] = 0;
        s->colorVec = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        s->loNabors = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        s->hiNabors = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        s->allNabors = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    }
    s->n2o_row = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->o2n_col = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->beg_row = (JX_Int *)JX_MALLOC_DH((blocks)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->beg_rowP = (JX_Int *)JX_MALLOC_DH((blocks)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->row_count = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->bdry_count = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->o2n_sub = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    s->n2o_sub = (JX_Int *)JX_MALLOC_DH(blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_init_mpi_private"
void jx_init_mpi_private( jx_SubdomainGraph_dh s, JX_Int blocks, jx_bool bj, void *A )
{
    JX_START_FUNC_DH
    JX_Int m, n, beg_row;
    jx_bool symmetric;
    JX_Real t1;
    
    symmetric = jx_Parser_dhHasSwitch(jx_parser_dh, "-sym"); JX_CHECK_V_ERROR;
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-makeSymmetric"))
    {
        symmetric = jx_true;
    }
    jx_EuclidGetDimensions(A, &beg_row, &m, &n); JX_CHECK_V_ERROR;
    s->m = m;
    jx_allocate_storage_private(s, blocks, m, bj); JX_CHECK_V_ERROR;
    if (!bj)
    {
        jx_MPI_Allgather(&beg_row, 1, JX_MPI_INT, s->beg_row, 1, JX_MPI_INT, jx_comm_dh);
        jx_MPI_Allgather(&m, 1, JX_MPI_INT, s->row_count, 1, JX_MPI_INT, jx_comm_dh);
        memcpy(s->beg_rowP, s->beg_row, jx_np_dh*sizeof(JX_Int));
    }
    else
    {
        s->beg_row[jx_myid_dh] = beg_row;
        s->beg_rowP[jx_myid_dh] = beg_row;
        s->row_count[jx_myid_dh] = m;
    }
    if (!bj)
    {
        t1 = jx_MPI_Wtime();
        if (symmetric)
        {
            jx_find_all_neighbors_sym_private(s, m, A); JX_CHECK_V_ERROR;
        }
        else
        {
            jx_find_all_neighbors_unsym_private(s, m, A); JX_CHECK_V_ERROR;
        }
        s->timing[JX_FIND_NABORS_SGT] += (jx_MPI_Wtime() - t1);
    }
    t1 = jx_MPI_Wtime();
    if (!bj)
    {
        JX_Int *interiorNodes, *bdryNodes;
        JX_Int interiorCount = 0, bdryCount;
        JX_Int *o2n = s->o2n_col, idx;
        JX_Int i;
        
        interiorNodes = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        bdryNodes = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        if (symmetric)
        {
            jx_find_bdry_nodes_sym_private(s, m, A, interiorNodes,
                           bdryNodes, &interiorCount, &bdryCount); JX_CHECK_V_ERROR;
        }
        else
        {
            jx_find_bdry_nodes_unsym_private(s, m, A, interiorNodes,
                            bdryNodes, &interiorCount, &bdryCount); JX_CHECK_V_ERROR;
        }
        jx_MPI_Allgather(&bdryCount, 1, JX_MPI_INT, s->bdry_count, 1, JX_MPI_INT, jx_comm_dh);
        idx = 0;
        for (i = 0; i < interiorCount; ++ i)
        {
            o2n[interiorNodes[i]] = idx ++;
        }
        for (i = 0; i < bdryCount; ++ i)
        {
            o2n[bdryNodes[i]] = idx ++;
        }
        jx_invert_perm(m, o2n, s->n2o_row); JX_CHECK_V_ERROR;
        JX_FREE_DH(interiorNodes); JX_CHECK_V_ERROR;
        JX_FREE_DH(bdryNodes); JX_CHECK_V_ERROR;
    }
    else
    {
        JX_Int *o2n = s->o2n_col, *n2o = s->n2o_row;
        JX_Int i, m = s->m;
        
        for (i = 0; i < m; ++ i)
        {
            o2n[i] = i;
            n2o[i] = i;
        }
    }
    s->timing[JX_ORDER_BDRY_SGT] += (jx_MPI_Wtime() - t1);
    if (!bj)
    {
        t1 = jx_MPI_Wtime();
        jx_form_subdomaingraph_mpi_private(s); JX_CHECK_V_ERROR;
        if (s->doNotColor)
        {
            JX_Int i;
            
            jx_printf_dh("subdomain coloring and reordering is OFF\n");
            for (i = 0; i < blocks; ++ i)
            {
                s->o2n_sub[i] = i;
                s->n2o_sub[i] = i;
                s->colorVec[i] = 0;
            }
        }
        else
        {
            JX_SET_INFO("subdomain coloring and reordering is ON");
            jx_color_subdomain_graph_private(s); JX_CHECK_V_ERROR;
        }
        s->timing[JX_FORM_GRAPH_SGT] += (jx_MPI_Wtime() - t1);
    }
    if (!bj)
    {
        jx_find_ordered_neighbors_private(s); JX_CHECK_V_ERROR;
    }
    if (!bj)
    {
        t1 = jx_MPI_Wtime();
        jx_SubdomainGraph_dhExchangePerms(s); JX_CHECK_V_ERROR;
        s->timing[JX_EXCHANGE_PERMS_SGT] += (jx_MPI_Wtime() - t1);
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhExchangePerms"
void jx_SubdomainGraph_dhExchangePerms( jx_SubdomainGraph_dh s )
{
    JX_START_FUNC_DH
    MPI_Request *recv_req = NULL, *send_req = NULL;
    MPI_Status *status = NULL;
    JX_Int *nabors = s->allNabors, naborCount = s->allCount;
    JX_Int i, j, *sendBuf = NULL, *recvBuf = NULL, *naborIdx = NULL, nz;
    JX_Int m = s->row_count[jx_myid_dh];
    JX_Int beg_row = s->beg_row[jx_myid_dh];
    JX_Int beg_rowP = s->beg_rowP[jx_myid_dh];
    JX_Int *bdryNodeCounts = s->bdry_count;
    JX_Int myBdryCount = s->bdry_count[jx_myid_dh];
    jx_bool debug = jx_false;
    JX_Int myFirstBdry = m - myBdryCount;
    JX_Int *n2o_row = s->n2o_row;
    jx_Hash_i_dh n2o_table, o2n_table;
    
    if (jx_logFile != NULL && s->debug) debug = jx_true;
    sendBuf = (JX_Int *)JX_MALLOC_DH(2*myBdryCount*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nSUBG myFirstBdry= %i  myBdryCount= %i  m= %i  beg_rowP= %i\n",
                                                   1+ myFirstBdry, myBdryCount, m, 1+beg_rowP);
        fflush(jx_logFile);
    }
    for (i = myFirstBdry, j = 0; j < myBdryCount; ++ i, ++ j)
    {
        sendBuf[2*j] = n2o_row[i] + beg_row;
        sendBuf[2*j+1] = i + beg_rowP;
    }
    if (debug)
    {
        jx_fprintf(jx_logFile, "\nSUBG SEND_BUF:\n");
        for (i = myFirstBdry, j = 0; j < myBdryCount; ++ i, ++ j)
        {
            jx_fprintf(jx_logFile, "SUBG  %i, %i\n", 1+sendBuf[2*j], 1+sendBuf[2*j+1]);
        }
        fflush(jx_logFile);
    }
    naborIdx = (JX_Int *)JX_MALLOC_DH((1+naborCount)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    naborIdx[0] = 0;
    nz = 0;
    for (i = 0; i < naborCount; ++ i)
    {
        nz += (2 * bdryNodeCounts[nabors[i]]);
        naborIdx[i+1] = nz;
    }
    recvBuf = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    recv_req = (MPI_Request*)JX_MALLOC_DH(naborCount*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    send_req = (MPI_Request*)JX_MALLOC_DH(naborCount*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    status = (MPI_Status*)JX_MALLOC_DH(naborCount*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
    for (i = 0; i < naborCount; ++ i)
    {
        JX_Int nabr = nabors[i];
        JX_Int *buf = recvBuf + naborIdx[i];
        JX_Int ct = 2*bdryNodeCounts[nabr];
        
        jx_MPI_Isend(sendBuf, 2*myBdryCount, JX_MPI_INT, nabr, 444, jx_comm_dh, &(send_req[i]));
        if (debug)
        {
            jx_fprintf(jx_logFile , "SUBG   sending %i elts to %i\n", 2*myBdryCount, nabr);
            fflush(jx_logFile);
        }
        jx_MPI_Irecv(buf, ct, JX_MPI_INT, nabr, 444, jx_comm_dh, &(recv_req[i]));
        if (debug)
        {
            jx_fprintf(jx_logFile, "SUBG  receiving %i elts from %i\n", ct, nabr);
            fflush(jx_logFile);
        }
    }
    jx_MPI_Waitall(naborCount, send_req, status);
    jx_MPI_Waitall(naborCount, recv_req, status);
    jx_Hash_i_dhCreate(&n2o_table, nz/2); JX_CHECK_V_ERROR;
    jx_Hash_i_dhCreate(&o2n_table, nz/2); JX_CHECK_V_ERROR;
    s->n2o_ext = n2o_table;
    s->o2n_ext = o2n_table;
    for (i = 0; i < nz; i += 2)
    {
        JX_Int old = recvBuf[i];
        JX_Int new = recvBuf[i+1];
        
        if (debug)
        {
            jx_fprintf(jx_logFile, "SUBG  i= %i  old= %i  new= %i\n", i, old+1, new+1);
            fflush(jx_logFile);
        }
        jx_Hash_i_dhInsert(o2n_table, old, new); JX_CHECK_V_ERROR;
        jx_Hash_i_dhInsert(n2o_table, new, old); JX_CHECK_V_ERROR;
    }
    if (recvBuf != NULL)
    {
        JX_FREE_DH(recvBuf); JX_CHECK_V_ERROR;
    }
    if (naborIdx != NULL)
    {
        JX_FREE_DH(naborIdx); JX_CHECK_V_ERROR;
    }
    if (sendBuf != NULL)
    {
        JX_FREE_DH(sendBuf); JX_CHECK_V_ERROR;
    }
    if (recv_req != NULL)
    {
        JX_FREE_DH(recv_req); JX_CHECK_V_ERROR;
    }
    if (send_req != NULL)
    {
        JX_FREE_DH(send_req); JX_CHECK_V_ERROR;
    }
    if (status != NULL)
    {
        JX_FREE_DH(status); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_form_subdomaingraph_mpi_private"
void jx_form_subdomaingraph_mpi_private( jx_SubdomainGraph_dh s )
{
    JX_START_FUNC_DH
    JX_Int *nabors = s->allNabors, nct = s->allCount;
    JX_Int *idxAll = NULL;
    JX_Int i, j, nz, *adj, *ptrs = s->ptrs;
    MPI_Request *recvReqs = NULL, sendReq;
    MPI_Status *statuses = NULL, status;
    
    if (jx_myid_dh == 0)
    {
        idxAll = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    }
    jx_MPI_Gather(&nct, 1, JX_MPI_INT, idxAll, 1, JX_MPI_INT, 0, jx_comm_dh);
    if (jx_myid_dh == 0)
    {
        nz = 0;
        for (i = 0; i < jx_np_dh; ++ i) nz += idxAll[i];
    }
    jx_MPI_Bcast(&nz, 1, JX_MPI_INT, 0, jx_comm_dh);
    adj = s->adj = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    if (jx_myid_dh == 0)
    {
        recvReqs = (MPI_Request*)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
        statuses = (MPI_Status*)JX_MALLOC_DH(jx_np_dh*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
        ptrs[0] = 0;
        for (j = 0; j < jx_np_dh; ++ j) ptrs[j+1] = ptrs[j] + idxAll[j];
        for (j = 0; j < jx_np_dh; ++ j)
        {
            JX_Int ct = idxAll[j];
            
            jx_MPI_Irecv(adj+ptrs[j], ct, JX_MPI_INT, j, 42, jx_comm_dh, recvReqs+j);
        }
    }
    jx_MPI_Isend(nabors, nct, JX_MPI_INT, 0, 42, jx_comm_dh, &sendReq);
    if (jx_myid_dh == 0)
    {
        jx_MPI_Waitall(jx_np_dh, recvReqs, statuses);
    }
    jx_MPI_Wait(&sendReq, &status);
    jx_MPI_Bcast(ptrs, 1+jx_np_dh, JX_MPI_INT, 0, jx_comm_dh);
    jx_MPI_Bcast(adj, nz, JX_MPI_INT, 0, jx_comm_dh);
    if (idxAll != NULL)
    {
        JX_FREE_DH(idxAll); JX_CHECK_V_ERROR;
    }
    if (recvReqs != NULL)
    {
        JX_FREE_DH(recvReqs); JX_CHECK_V_ERROR;
    }
    if (statuses != NULL)
    {
        JX_FREE_DH(statuses); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_form_subdomaingraph_seq_private"
void jx_form_subdomaingraph_seq_private( jx_SubdomainGraph_dh s, JX_Int m, void *A )
{
    JX_START_FUNC_DH
    JX_Int *dense, i, j, row, blocks = s->blocks;
    JX_Int *cval, len, *adj;
    JX_Int idx = 0, *ptrs = s->ptrs;
    
    adj = s->adj = (JX_Int *)JX_MALLOC_DH(blocks*blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    dense = (JX_Int *)JX_MALLOC_DH(blocks*blocks*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < blocks*blocks; ++ i) dense[i] = 0;
    for (i = 0; i < blocks; ++ i)
    {
        JX_Int beg_row = s->beg_row[i];
        JX_Int end_row = beg_row + s->row_count[i];
        
        for (row = beg_row; row < end_row; ++ row)
        {
            jx_EuclidGetRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
            for (j = 0; j < len; ++ j)
            {
                JX_Int col = cval[j];
                
                if (col < beg_row || col >= end_row)
                {
                    JX_Int owner = jx_SubdomainGraph_dhFindOwner(s, col, jx_false); JX_CHECK_V_ERROR;
                    
                    dense[i*blocks+owner] = 1;
                    dense[owner*blocks+i] = 1;
                }
            }
            jx_EuclidRestoreRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        }
    }
    ptrs[0] = 0;
    for (i = 0; i < blocks; ++ i)
    {
        for (j = 0; j < blocks; ++ j)
        {
            if (dense[i*blocks+j])
            {
                adj[idx++] = j;
            }
        }
        ptrs[i+1] = idx;
    }
    JX_FREE_DH(dense); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_find_all_neighbors_sym_private"
void jx_find_all_neighbors_sym_private( jx_SubdomainGraph_dh s, JX_Int m, void *A )
{
    JX_START_FUNC_DH
    JX_Int *marker, i, j, beg_row, end_row;
    JX_Int row, len, *cval, ct = 0;
    JX_Int *nabors = s->allNabors;
    
    marker = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) marker[i] = 0;
    JX_SET_INFO("jx_finding nabors in subdomain graph for structurally symmetric matrix");
    JX_SET_INFO("(if this isn't what you want, use '-sym 0' switch)");
    beg_row = s->beg_row[jx_myid_dh];
    end_row = beg_row + s->row_count[jx_myid_dh];
    for (row = beg_row; row < end_row; ++ row)
    {
        jx_EuclidGetRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        for (j = 0; j < len; ++ j)
        {
            JX_Int col = cval[j];
            
            if (col < beg_row || col >= end_row)
            {
                JX_Int owner = jx_SubdomainGraph_dhFindOwner(s, col, jx_false); JX_CHECK_V_ERROR;
                
                if (!marker[owner])
                {
                    marker[owner] = 1;
                    nabors[ct++] = owner;
                }
            }
        }
        jx_EuclidRestoreRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
    }
    s->allCount = ct;
    if (marker != NULL)
    {
        JX_FREE_DH(marker); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_find_all_neighbors_unsym_private"
void jx_find_all_neighbors_unsym_private( jx_SubdomainGraph_dh s, JX_Int m, void *A )
{
    JX_START_FUNC_DH
    JX_Int i, j, row, beg_row, end_row;
    JX_Int *marker;
    JX_Int *cval, len, idx = 0;
    JX_Int nz, *nabors = s->allNabors, *myNabors;
    
    myNabors = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    marker = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < jx_np_dh; ++ i) marker[i] = 0;
    JX_SET_INFO("jx_finding nabors in subdomain graph for structurally unsymmetric matrix");
    beg_row = s->beg_row[jx_myid_dh];
    end_row = beg_row + s->row_count[jx_myid_dh];
    for (row = beg_row; row < end_row; ++ row)
    {
        jx_EuclidGetRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        for (j = 0; j < len; ++ j)
        {
            JX_Int col = cval[j];
            if (col < beg_row || col >= end_row)
            {
                JX_Int owner = jx_SubdomainGraph_dhFindOwner(s, col, jx_false); JX_CHECK_V_ERROR;
                
                if (!marker[owner])
                {
                    marker[owner] = 1;
                    myNabors[idx++] = owner;
                }
            }
        }
        jx_EuclidRestoreRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
    }
    jx_MPI_Alltoall(marker, 1, JX_MPI_INT, nabors, 1, JX_MPI_INT, jx_comm_dh); JX_CHECK_V_ERROR;
    for (i = 0; i < idx; ++ i) nabors[myNabors[i]] = 1;
    nabors[jx_myid_dh] = 0;
    nz = 0;
    for (i = 0; i < jx_np_dh; ++ i)
    {
        if (nabors[i]) myNabors[nz++] = i;
    }
    s->allCount = nz;
    memcpy(nabors, myNabors, nz*sizeof(JX_Int));
    if (marker != NULL)
    {
        JX_FREE_DH(marker); JX_CHECK_V_ERROR;
    }
    if (myNabors != NULL)
    {
        JX_FREE_DH(myNabors); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_find_bdry_nodes_sym_private"
void jx_find_bdry_nodes_sym_private( jx_SubdomainGraph_dh s,
                                  JX_Int m,
                                  void *A,
                                  JX_Int *interiorNodes,
                                  JX_Int *bdryNodes,
                                  JX_Int *interiorCount,
                                  JX_Int *bdryCount )
{
    JX_START_FUNC_DH
    JX_Int beg_row = s->beg_row[jx_myid_dh];
    JX_Int end_row = beg_row + s->row_count[jx_myid_dh];
    JX_Int row, inCt = 0, bdCt = 0;
    JX_Int j;
    JX_Int *cval;
    
    for (row = beg_row; row < end_row; ++ row)
    {
        jx_bool isBdry = jx_false;
        JX_Int len;
        
        jx_EuclidGetRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        for (j = 0; j < len; ++ j)
        {
            JX_Int col = cval[j];
            
            if (col < beg_row || col >= end_row)
            {
                isBdry = jx_true;
                break;
            }
        }
        jx_EuclidRestoreRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        if (isBdry)
        {
            bdryNodes[bdCt++] = row - beg_row;
        }
        else
        {
            interiorNodes[inCt++] = row - beg_row;
        }
    }
   *interiorCount = inCt;
   *bdryCount = bdCt;
    JX_END_FUNC_DH
}

#define JX_BDRY_NODE_TAG 42

#undef __FUNC__
#define __FUNC__ "jx_find_bdry_nodes_unsym_private"
void jx_find_bdry_nodes_unsym_private( jx_SubdomainGraph_dh s,
                                    JX_Int m,
                                    void *A,
                                    JX_Int *interiorNodes,
                                    JX_Int *boundaryNodes,
                                    JX_Int *interiorCount,
                                    JX_Int *bdryCount )
{
    JX_START_FUNC_DH
    JX_Int beg_row = s->beg_row[jx_myid_dh];
    JX_Int end_row = beg_row + s->row_count[jx_myid_dh];
    JX_Int i, j, row, max;
    JX_Int *cval;
    JX_Int *list, count;
    JX_Int *rpIN = NULL, *rpOUT = NULL;
    JX_Int *sendBuf, *recvBuf;
    JX_Int *marker, inCt, bdCt;
    JX_Int *bdryNodes, nz;
    JX_Int sendCt, recvCt;
    MPI_Request *sendReq, *recvReq;
    MPI_Status *status;
    jx_SortedSet_dh ss;
    
    jx_SortedSet_dhCreate(&ss, m); JX_CHECK_V_ERROR;
    for (row = beg_row; row < end_row; ++ row)
    {
        jx_bool isBdry = jx_false;
        JX_Int len;
        
        jx_EuclidGetRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        for (j = 0; j < len; ++ j)
        {
            JX_Int col = cval[j];
            
            if (col < beg_row || col >= end_row)
            {
                isBdry = jx_true;
                jx_SortedSet_dhInsert(ss, col); JX_CHECK_V_ERROR;
            }
        }
        jx_EuclidRestoreRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        if (isBdry)
        {
            jx_SortedSet_dhInsert(ss, row); JX_CHECK_V_ERROR;
        }
    }
    sendBuf = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    recvBuf = (JX_Int *)JX_MALLOC_DH(jx_np_dh*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    rpOUT = (JX_Int *)JX_MALLOC_DH((jx_np_dh+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    rpOUT[0] = 0;
    for (i = 0; i < jx_np_dh; ++ i) sendBuf[i] = 0;
    sendCt = 0;
    jx_SortedSet_dhGetList(ss, &list, &count); JX_CHECK_V_ERROR;
    for (i = 0; i < count;)
    {
        JX_Int node = list[i];
        JX_Int owner;
        JX_Int last;
        
        owner = jx_SubdomainGraph_dhFindOwner(s, node, jx_false); JX_CHECK_V_ERROR;
        last = s->beg_row[owner] + s->row_count[owner];
        while ((i < count)  && (list[i] < last)) ++ i;
        ++ sendCt;
        rpOUT[sendCt] = i;
        sendBuf[owner] = rpOUT[sendCt] - rpOUT[sendCt-1];
    }
    jx_MPI_Alltoall(sendBuf, 1, JX_MPI_INT, recvBuf, 1, JX_MPI_INT, jx_comm_dh); JX_CHECK_V_ERROR;
    rpIN = (JX_Int *)JX_MALLOC_DH((jx_np_dh+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    rpIN[0] = 0;
    nz = 0;
    recvCt = 0;
    for (i = 0; i < jx_np_dh; ++ i)
    {
        if (recvBuf[i])
        {
            ++ recvCt;
            nz += recvBuf[i];
            rpIN[recvCt] = nz;
        }
    }
    bdryNodes = (JX_Int *)JX_MALLOC_DH(nz*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    sendReq = (MPI_Request *)JX_MALLOC_DH(sendCt*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    recvReq = (MPI_Request *)JX_MALLOC_DH(recvCt*sizeof(MPI_Request)); JX_CHECK_V_ERROR;
    max = JX_MAX(sendCt, recvCt);
    status = (MPI_Status *)JX_MALLOC_DH(max*sizeof(MPI_Status)); JX_CHECK_V_ERROR;
    j = 0;
    for (i = 0; i < jx_np_dh; ++ i)
    {
        if (recvBuf[i])
        {
            jx_MPI_Irecv(bdryNodes+rpIN[j], recvBuf[i], JX_MPI_INT, i, JX_BDRY_NODE_TAG, jx_comm_dh, recvReq+j);
            ++ j;
        }
    }
    j = 0;
    for (i = 0; i < jx_np_dh; ++ i)
    {
        if (sendBuf[i])
        {
            jx_MPI_Isend(list+rpOUT[j], sendBuf[i], JX_MPI_INT, i, JX_BDRY_NODE_TAG, jx_comm_dh, sendReq+j);
            ++ j;
        }
    }
    jx_MPI_Waitall(sendCt, sendReq, status);
    jx_MPI_Waitall(recvCt, recvReq, status);
    for (i = 0; i < nz; ++ i) bdryNodes[i] -= beg_row;
    marker = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) marker[i] = 0;
    for (i = 0; i < nz; ++ i) marker[bdryNodes[i]] = 1;
    inCt = bdCt = 0;
    for (i = 0; i < m; ++ i)
    {
        if (marker[i])
        {
            boundaryNodes[bdCt++] = i;
        }
        else
        {
            interiorNodes[inCt++] = i;
        }
    }
   *interiorCount = inCt;
   *bdryCount = bdCt;
    jx_SortedSet_dhDestroy(ss); JX_CHECK_V_ERROR;
    if (rpIN != NULL)
    {
        JX_FREE_DH(rpIN); JX_CHECK_V_ERROR;
    }
    if (rpOUT != NULL)
    {
        JX_FREE_DH(rpOUT); JX_CHECK_V_ERROR;
    }
    if (sendBuf != NULL)
    {
        JX_FREE_DH(sendBuf); JX_CHECK_V_ERROR;
    }
    if (recvBuf != NULL)
    {
        JX_FREE_DH(recvBuf); JX_CHECK_V_ERROR;
    }
    if (bdryNodes != NULL)
    {
        JX_FREE_DH(bdryNodes); JX_CHECK_V_ERROR;
    }
    if (marker != NULL)
    {
        JX_FREE_DH(marker); JX_CHECK_V_ERROR;
    }
    if (sendReq!= NULL)
    {
        JX_FREE_DH(sendReq); JX_CHECK_V_ERROR;
    }
    if (recvReq!= NULL)
    {
        JX_FREE_DH(recvReq); JX_CHECK_V_ERROR;
    }
    if (status!= NULL)
    {
        JX_FREE_DH(status); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_find_ordered_neighbors_private"
void jx_find_ordered_neighbors_private( jx_SubdomainGraph_dh s )
{
    JX_START_FUNC_DH
    JX_Int *loNabors = s->loNabors;
    JX_Int *hiNabors = s->hiNabors;
    JX_Int *allNabors = s->allNabors, allCount = s->allCount;
    JX_Int loCt = 0, hiCt = 0;
    JX_Int *o2n = s->o2n_sub;
    JX_Int i, myNewId = o2n[jx_myid_dh];
    
    for (i = 0; i < allCount; ++ i)
    {
        JX_Int nabor = allNabors[i];
        
        if (o2n[nabor] < myNewId)
        {
            loNabors[loCt++] = nabor;
        }
        else
        {
            hiNabors[hiCt++] = nabor;
        }
    }
    s->loCount = loCt;
    s->hiCount = hiCt;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_color_subdomain_graph_private"
void jx_color_subdomain_graph_private( jx_SubdomainGraph_dh s )
{
    JX_START_FUNC_DH
    JX_Int i, n = jx_np_dh;
    JX_Int *rp = s->ptrs, *cval = s->adj;
    JX_Int j, *marker, thisNodesColor, *colorCounter;
    JX_Int *o2n = s->o2n_sub;
    JX_Int *color = s->colorVec;
    
    if (jx_np_dh == 1) n = s->blocks;
    marker = (JX_Int *)JX_MALLOC_DH((n+1)*sizeof(JX_Int));
    colorCounter = (JX_Int *)JX_MALLOC_DH((n+1)*sizeof(JX_Int));
    for (i = 0; i <= n; ++ i)
    {
        marker[i] = -1;
        colorCounter[i] = 0;
    }
    for (i = 0; i < n; ++ i)
    {
        for (j = rp[i]; j < rp[i+1]; ++ j)
        {
            JX_Int nabor = cval[j];
            if (nabor < i)
            {
                JX_Int naborsColor = color[nabor];
                marker[naborsColor] = i;
            }
        }
        thisNodesColor = -1;
        for (j = 0; j < n; ++ j)
        {
            if (marker[j] != i)
            {
                thisNodesColor = j;
                break;
            }
        }
        color[i] = thisNodesColor;
        colorCounter[1+thisNodesColor] += 1;
    }
    for (i = 1; i < n; ++ i)
    {
        if (colorCounter[i] == 0) break;
        colorCounter[i] += colorCounter[i-1];
    }
    for (i = 0; i < n; ++ i)
    {
        o2n[i] = colorCounter[color[i]];
        colorCounter[color[i]] += 1;
    }
    jx_invert_perm(n, s->o2n_sub, s->n2o_sub); JX_CHECK_V_ERROR;
    JX_Int ct = 0;
    for (j = 0; j < n; ++ j)
    {
        if (marker[j] == -1) break;
        ++ ct;
    }
    s->colors = ct;
    JX_Int sum = 0;
    for (i = 0; i < n; ++ i)
    {
        JX_Int old = s->n2o_sub[i];
        
        s->beg_rowP[old] = sum;
        sum += s->row_count[old];
    }
    JX_FREE_DH(marker); JX_CHECK_V_ERROR;
    JX_FREE_DH(colorCounter); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhDump"
void jx_SubdomainGraph_dhDump( jx_SubdomainGraph_dh s, char *filename )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int sCt = jx_np_dh;
    FILE *fp;
    
    if (jx_np_dh == 1) sCt = s->blocks;
    fp = jx_openFile_dh(filename, "w"); JX_CHECK_V_ERROR;
    jx_fprintf(fp, "----- colors used\n");
    jx_fprintf(fp, "%i\n", s->colors);
    if (s->colorVec == NULL)
    {
        jx_fprintf(fp, "s->colorVec == NULL\n");
    }
    else
    {
        jx_fprintf(fp, "----- colorVec\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", s->colorVec[i]);
        }
        jx_fprintf(fp, "\n");
    }
    if (s->o2n_sub == NULL || s->o2n_sub == NULL)
    {
        jx_fprintf(fp, "s->o2n_sub == NULL || s->o2n_sub == NULL\n");
    }
    else
    {
        jx_fprintf(fp, "----- o2n_sub\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", s->o2n_sub[i]);
        }
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "----- n2o_sub\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", s->n2o_sub[i]);
        }
        jx_fprintf(fp, "\n");
    }
    if (s->beg_row == NULL || s->beg_rowP == NULL)
    {
        jx_fprintf(fp, "s->beg_row == NULL || s->beg_rowP == NULL\n");
    }
    else
    {
        jx_fprintf(fp, "----- beg_row\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", 1+s->beg_row[i]);
        }
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "----- beg_rowP\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", 1+s->beg_rowP[i]);
        }
        jx_fprintf(fp, "\n");
    }
    if (s->row_count == NULL || s->bdry_count == NULL)
    {
        jx_fprintf(fp, "s->row_count == NULL || s->bdry_count == NULL\n");
    }
    else
    {
        jx_fprintf(fp, "----- row_count\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", s->row_count[i]);
        }
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "----- bdry_count\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i ", s->bdry_count[i]);
        }
        jx_fprintf(fp, "\n");
    }
    if (s->ptrs == NULL || s->adj == NULL)
    {
        jx_fprintf(fp, "s->ptrs == NULL || s->adj == NULL\n");
    }
    else
    {
        JX_Int j;
        JX_Int ct;
        
        jx_fprintf(fp, "----- subdomain graph\n");
        for (i = 0; i < sCt; ++ i)
        {
            jx_fprintf(fp, "%i :: ", i);
            ct = s->ptrs[i+1] - s->ptrs[i];
            if (ct)
            {
                jx_shellSort_int(ct, s->adj+s->ptrs[i]); JX_CHECK_V_ERROR;
            }
            for (j = s->ptrs[i]; j < s->ptrs[i+1]; ++ j)
            {
                jx_fprintf(fp, "%i ", s->adj[j]);
            }
            jx_fprintf(fp, "\n");
        }
    }
    jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    if (s->beg_rowP == NULL)
    {
        JX_SET_V_ERROR("s->beg_rowP == NULL; can't continue");
    }
    if (s->row_count == NULL)
    {
        JX_SET_V_ERROR("s->row_count == NULL; can't continue");
    }
    if (s->o2n_sub == NULL)
    {
        JX_SET_V_ERROR("s->o2n_sub == NULL; can't continue");
    }
    if (jx_np_dh == 1)
    {
        fp = jx_openFile_dh(filename, "a"); JX_CHECK_V_ERROR;
        if (s->n2o_row == NULL|| s->o2n_col == NULL)
        {
            jx_fprintf(fp, "s->n2o_row == NULL|| s->o2n_col == NULL\n");
        }
        else
        {
            jx_fprintf(fp, "----- n2o_row\n");
            for (i = 0; i < s->m; ++ i)
            {
                jx_fprintf(fp, "%i ", 1+s->n2o_row[i]);
            }
            jx_fprintf(fp, "\n");
        }
        jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
    }
    else
    {
        JX_Int id = s->n2o_sub[jx_myid_dh];
        JX_Int m = s->m;
        JX_Int pe;
        JX_Int beg_row = 0;
        
        if (s->beg_row != 0) beg_row = s->beg_row[jx_myid_dh];
        for (pe = 0; pe < jx_np_dh; ++ pe)
        {
            jx_MPI_Barrier(jx_comm_dh);
            if (id == pe)
            {
                fp = jx_openFile_dh(filename, "a"); JX_CHECK_V_ERROR;
                if (id == 0) jx_fprintf(fp, "----- n2o_row\n");
                for (i = 0; i < m; ++ i)
                {
                    jx_fprintf(fp, "%i ", 1+s->n2o_row[i]+beg_row);
                }
                if (id == jx_np_dh - 1) jx_fprintf(fp, "\n");
                jx_closeFile_dh(fp); JX_CHECK_V_ERROR;
            }
        }
    }
    JX_END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "jx_find_bdry_nodes_seq_private"
void jx_find_bdry_nodes_seq_private( jx_SubdomainGraph_dh s, JX_Int m, void *A )
{
    JX_START_FUNC_DH
    JX_Int i, j, row, blocks = s->blocks;
    JX_Int *cval, *tmp;
    
    tmp = (JX_Int *)JX_MALLOC_DH(m*sizeof(JX_Int)); JX_CHECK_V_ERROR;
    for (i = 0; i < m; ++ i) tmp[i] = 0;
    for (i = 0; i < blocks; ++ i)
    {
        JX_Int beg_row = s->beg_row[i];
        JX_Int end_row = beg_row + s->row_count[i];
        
        for (row = beg_row; row < end_row; ++ row)
        {
            jx_bool isBdry = jx_false;
            JX_Int len;
            
            jx_EuclidGetRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
            for (j = 0; j < len; ++ j)
            {
                JX_Int col = cval[j];
                
                if (col < beg_row || col >= end_row)
                {
                    tmp[col] = 1;
                    isBdry = jx_true;
                }
            }
            if (isBdry) tmp[row] = 1;
            jx_EuclidRestoreRow(A, row, &len, &cval, NULL); JX_CHECK_V_ERROR;
        }
    }
    for (i = 0; i < blocks; ++ i)
    {
        JX_Int beg_row = s->beg_row[i];
        JX_Int end_row = beg_row + s->row_count[i];
        JX_Int ct = 0;
        
        for (row = beg_row; row < end_row; ++ row)
        {
            if (tmp[row]) ++ ct;
        }
        s->bdry_count[i] = ct;
    }
    for (i = 0; i < blocks; ++ i)
    {
        JX_Int beg_row = s->beg_row[i];
        JX_Int end_row = beg_row + s->row_count[i];
        JX_Int interiorIDX = beg_row;
        JX_Int bdryIDX = end_row - s->bdry_count[i];
        
        for (row = beg_row; row < end_row; ++ row)
        {
            if (tmp[row])
            {
                s->o2n_col[row] = bdryIDX ++;
            }
            else
            {
                s->o2n_col[row] = interiorIDX ++;
            }
        }
    }
    jx_invert_perm(m, s->o2n_col, s->n2o_row); JX_CHECK_V_ERROR;
    JX_FREE_DH(tmp); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhPrintSubdomainGraph"
void jx_SubdomainGraph_dhPrintSubdomainGraph( jx_SubdomainGraph_dh s, FILE *fp )
{
    JX_START_FUNC_DH
    if (jx_myid_dh == 0)
    {
        JX_Int i, j;
        
        jx_fprintf(fp, "\n-----------------------------------------------------\n");
        jx_fprintf(fp, "SubdomainGraph, and coloring and ordering information\n");
        jx_fprintf(fp, "-----------------------------------------------------\n");
        jx_fprintf(fp, "colors used: %i\n", s->colors);
        jx_fprintf(fp, "o2n ordering vector: ");
        for (i = 0; i < s->blocks; ++ i) jx_fprintf(fp, "%i ", s->o2n_sub[i]);
        jx_fprintf(fp, "\ncoloring vector (node, color): \n");
        for (i = 0; i < s->blocks; ++ i) jx_fprintf(fp, "  %i, %i\n", i, s->colorVec[i]);
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "Adjacency lists:\n");
        for (i = 0; i < s->blocks; ++ i)
        {
            jx_fprintf(fp, "   P_%i :: ", i);
            for (j = s->ptrs[i]; j < s->ptrs[i+1]; ++ j)
            {
                jx_fprintf(fp, "%i ", s->adj[j]);
            }
            jx_fprintf(fp, "\n");
        }
        jx_fprintf(fp, "-----------------------------------------------------\n");
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_adjust_matrix_perms_private"
void jx_adjust_matrix_perms_private( jx_SubdomainGraph_dh s, JX_Int m )
{
    JX_START_FUNC_DH
    JX_Int i, j, blocks = s->blocks;
    JX_Int *o2n = s->o2n_col;
    
    for (i = 0; i < blocks; ++ i)
    {
        JX_Int beg_row = s->beg_row[i];
        JX_Int end_row = beg_row + s->row_count[i];
        JX_Int adjust = s->beg_rowP[i] - s->beg_row[i];
        
        for (j = beg_row; j < end_row; ++ j) o2n[j] += adjust;
    }
    jx_invert_perm(m, s->o2n_col, s->n2o_row); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhPrintRatios"
void jx_SubdomainGraph_dhPrintRatios( jx_SubdomainGraph_dh s, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Int i;
    JX_Int blocks = jx_np_dh;
    JX_Real ratio[25];
    
    if (jx_myid_dh == 0)
    {
        if (jx_np_dh == 1) blocks = s->blocks;
        if (blocks > 25) blocks = 25;
        jx_fprintf(fp, "\n");
        jx_fprintf(fp, "Subdomain interior/boundary node ratios\n");
        jx_fprintf(fp, "---------------------------------------\n");
        for (i = 0; i < blocks; ++ i)
        {
            if (s->bdry_count[i] == 0)
            {
                ratio[i] = -1;
            }
            else
            {
                ratio[i] = (JX_Real)(s->row_count[i] - s->bdry_count[i]) / (JX_Real)s->bdry_count[i];
            }
        }
        jx_shellSort_float(blocks, ratio);
        if (blocks <= 20)
        {
            JX_Int j = 0;
            
            for (i = 0; i < blocks; ++ i)
            {
                jx_fprintf(fp, "%0.2g  ", ratio[i]);
                ++ j;
                if (j == 10)
                {
                    jx_fprintf(fp, "\n");
                }
            }
            jx_fprintf(fp, "\n");
        }
        else
        {
            jx_fprintf(fp, "10 smallest ratios: ");
            for (i = 0; i < 10; ++ i)
            {
                jx_fprintf(fp, "%0.2g  ", ratio[i]);
            }
            jx_fprintf(fp, "\n");
            jx_fprintf(fp, "10 largest ratios:  ");
            JX_Int start = blocks - 6, stop = blocks - 1;
            for (i = start; i < stop; ++ i)
            {
                jx_fprintf(fp, "%0.2g  ", ratio[i]);
            }
            jx_fprintf(fp, "\n");
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_SubdomainGraph_dhPrintStats"
void jx_SubdomainGraph_dhPrintStats( jx_SubdomainGraph_dh sg, FILE *fp )
{
    JX_START_FUNC_DH
    JX_Real *timing = sg->timing;
    
    jx_fprintf_dh(fp, "\nSubdomainGraph timing report\n");
    jx_fprintf_dh(fp, "-----------------------------\n");
    jx_fprintf_dh(fp, "total setup time: %0.2f\n", timing[JX_TOTAL_SGT]);
    jx_fprintf_dh(fp, "  jx_find neighbors in subdomain graph: %0.2f\n", timing[JX_FIND_NABORS_SGT]);
    jx_fprintf_dh(fp, "  locally order interiors and bdry:  %0.2f\n", timing[JX_ORDER_BDRY_SGT]);
    jx_fprintf_dh(fp, "  form and color subdomain graph:    %0.2f\n", timing[JX_FORM_GRAPH_SGT]);
    jx_fprintf_dh(fp, "  exchange bdry permutations:        %0.2f\n", timing[JX_EXCHANGE_PERMS_SGT]);
    jx_fprintf_dh(fp, "  everything else (should be small): %0.2f\n", timing[JX_TOTAL_SGT]-(timing[JX_FIND_NABORS_SGT]+
                                timing[JX_ORDER_BDRY_SGT]+timing[JX_FORM_GRAPH_SGT]+timing[JX_EXCHANGE_PERMS_SGT]));
    JX_END_FUNC_DH
}
