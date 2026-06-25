//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  parcsr_euclid.c
 *  Date: 2013/01/23
 */

#include "jx_pamg.h"
#include "jx_euclid.h"

#define JX_EUCLID_ERRCHKA \
      if (jx_errFlag_dh) { \
        jx_setError_dh("", __FUNC__, __FILE__, __LINE__); \
        jx_printErrorMsg(stderr); \
        jx_MPI_Abort(jx_comm_dh, -1); \
      }

#undef JX_START_FUNC_DH
#undef JX_END_FUNC_VAL
#undef JX_END_FUNC_DH
#define JX_START_FUNC_DH
#define JX_END_FUNC_DH
#define JX_END_FUNC_VAL(a) return(a);

#undef __FUNC__
#define __FUNC__ "JX_EuclidCreate"
JX_Int 
JX_EuclidCreate( MPI_Comm comm, JX_Solver *solver )
{
    JX_START_FUNC_DH
    jx_Euclid_dh eu;
    
    jx_comm_dh = comm;
    jx_MPI_Comm_size(jx_comm_dh, &jx_np_dh); JX_EUCLID_ERRCHKA;
    jx_MPI_Comm_rank(jx_comm_dh, &jx_myid_dh); JX_EUCLID_ERRCHKA;
    
#ifdef ENABLE_EUCLID_LOGGING
    jx_openLogfile_dh(0, NULL); JX_EUCLID_ERRCHKA;
#endif
    
    if (jx_mem_dh == NULL)
    {
        jx_Mem_dhCreate(&jx_mem_dh); JX_EUCLID_ERRCHKA;
    }
    if (jx_tlog_dh == NULL)
    {
        jx_TimeLog_dhCreate(&jx_tlog_dh); JX_EUCLID_ERRCHKA;
    }
    if (jx_parser_dh == NULL)
    {
        jx_Parser_dhCreate(&jx_parser_dh); JX_EUCLID_ERRCHKA;
    }
    jx_Parser_dhInit(jx_parser_dh, 0, NULL); JX_EUCLID_ERRCHKA;
    jx_Euclid_dhCreate(&eu); JX_EUCLID_ERRCHKA;
   *solver = (JX_Solver) eu;
    JX_END_FUNC_VAL(0)
}

#undef __FUNC__
#define __FUNC__ "JX_EuclidDestroy"
JX_Int 
JX_EuclidDestroy( JX_Solver solver )
{
    JX_START_FUNC_DH
    jx_Euclid_dh eu = (jx_Euclid_dh)solver;
    jx_bool printMemReport = jx_false;
    jx_bool printStats = jx_false;
    jx_bool logging = eu->logging;
    
    if (jx_Parser_dhHasSwitch(jx_parser_dh, "-printTestData"))
    {
        FILE *fp;
        char fname[] = "test_data_dh.temp", *fnamePtr = fname;
        
        jx_Parser_dhReadString(jx_parser_dh, "-printTestData", &fnamePtr); JX_EUCLID_ERRCHKA;
        if (!strcmp(fnamePtr, "1"))
        {
            fnamePtr = fname;
        }
        fp = jx_openFile_dh(fnamePtr, "w"); JX_EUCLID_ERRCHKA;
        jx_Euclid_dhPrintTestData(eu, fp); JX_EUCLID_ERRCHKA;
        jx_closeFile_dh(fp); JX_EUCLID_ERRCHKA;
        jx_printf_dh("\n@@@@@ Euclid test data was printed to file: %s\n\n", fnamePtr);
    }
    if (logging)
    {
        printStats = jx_true;
        printMemReport = jx_true;
    }
    if (jx_parser_dh != NULL)
    {
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-eu_stats"))
        {
            printStats = jx_true;
        }
        if (jx_Parser_dhHasSwitch(jx_parser_dh, "-eu_mem"))
        {
            printMemReport = jx_true;
        }
    }
    if (printStats)
    {
        jx_Euclid_dhPrintJxpamgReport(eu, stdout); JX_EUCLID_ERRCHKA;
    }
    jx_Euclid_dhDestroy(eu); JX_EUCLID_ERRCHKA;
    if (jx_parser_dh != NULL && jx_ref_counter == 0)
    {
        jx_Parser_dhDestroy(jx_parser_dh); JX_EUCLID_ERRCHKA;
        jx_parser_dh = NULL;
    }
    if (jx_tlog_dh != NULL && jx_ref_counter == 0)
    {
        jx_TimeLog_dhDestroy(jx_tlog_dh); JX_EUCLID_ERRCHKA;
        jx_tlog_dh = NULL;
    }
    if (jx_mem_dh != NULL && jx_ref_counter == 0)
    {
        if (printMemReport)
        {
            jx_Mem_dhPrint(jx_mem_dh, stdout, jx_false); JX_EUCLID_ERRCHKA;
        }
        jx_Mem_dhDestroy(jx_mem_dh); JX_EUCLID_ERRCHKA;
        jx_mem_dh = NULL;
    }
    
#ifdef ENABLE_EUCLID_LOGGING
    jx_closeLogfile_dh(); JX_EUCLID_ERRCHKA;
#endif
    
    JX_END_FUNC_VAL(0)
}

#undef __FUNC__
#define __FUNC__ "JX_EuclidSetup"
JX_Int 
JX_EuclidSetup( JX_Solver solver, JX_ParCSRMatrix A )
{
    JX_START_FUNC_DH
    jx_Euclid_dh eu = (jx_Euclid_dh)solver;
    jx_Euclid_dhInputJXpamgMat(eu, A); JX_EUCLID_ERRCHKA;
    jx_Euclid_dhSetup(eu); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

#undef __FUNC__
#define __FUNC__ "JX_EuclidSolve"
JX_Int 
JX_EuclidSolve( JX_Solver solver,
                JX_ParCSRMatrix A,
                JX_ParVector bb,
                JX_ParVector xx )
{
    JX_START_FUNC_DH
    jx_Euclid_dh eu = (jx_Euclid_dh)solver;
    JX_Real *b, *x;
    
    x = jx_VectorData(jx_ParVectorLocalVector((jx_ParVector *)bb));
    b = jx_VectorData(jx_ParVectorLocalVector((jx_ParVector *)xx));
    jx_Euclid_dhApply(eu, x, b); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

#undef __FUNC__
#define __FUNC__ "JX_EuclidSetParams"
JX_Int 
JX_EuclidSetParams( JX_Solver solver, JX_Int argc, char *argv[] )
{
    JX_START_FUNC_DH
    jx_Parser_dhInit(jx_parser_dh, argc, argv); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

#undef __FUNC__
#define __FUNC__ "JX_EuclidSetParamsFromFile"
JX_Int 
JX_EuclidSetParamsFromFile( JX_Solver solver, char *filename )
{
    JX_START_FUNC_DH
    jx_Parser_dhUpdateFromFile(jx_parser_dh, filename); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetLevel( JX_Solver solver, JX_Int level )
{
    char str_level[8];
    JX_START_FUNC_DH
    jx_sprintf(str_level,"%d",level);
    jx_Parser_dhInsert(jx_parser_dh, "-level", str_level); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetBJ( JX_Solver solver, JX_Int bj )
{
    char str_bj[8];
    JX_START_FUNC_DH
    jx_sprintf(str_bj,"%d",bj);
    jx_Parser_dhInsert(jx_parser_dh, "-bj", str_bj); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetStats( JX_Solver solver, JX_Int eu_stats )
{
    char str_eu_stats[8];
    JX_START_FUNC_DH
    jx_sprintf(str_eu_stats,"%d",eu_stats);
    jx_Parser_dhInsert(jx_parser_dh, "-eu_stats", str_eu_stats); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetMem( JX_Solver solver, JX_Int eu_mem )
{
    char str_eu_mem[8];
    JX_START_FUNC_DH
    jx_sprintf(str_eu_mem,"%d",eu_mem);
    jx_Parser_dhInsert(jx_parser_dh, "-eu_mem", str_eu_mem); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetILUT( JX_Solver solver, JX_Real ilut )
{
    char str_ilut[256];
    JX_START_FUNC_DH
    jx_sprintf(str_ilut,"%f",ilut);
    jx_Parser_dhInsert(jx_parser_dh, "-ilut", str_ilut); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetSparseA( JX_Solver solver, JX_Real sparse_A )
{
    char str_sparse_A[256];
    JX_START_FUNC_DH
    jx_sprintf(str_sparse_A,"%f",sparse_A);
    jx_Parser_dhInsert(jx_parser_dh, "-sparseA", str_sparse_A); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}

JX_Int 
JX_EuclidSetRowScale( JX_Solver solver, JX_Int row_scale )
{
    char str_row_scale[8];
    JX_START_FUNC_DH
    jx_sprintf(str_row_scale,"%d",row_scale);
    jx_Parser_dhInsert(jx_parser_dh, "-rowScale", str_row_scale); JX_EUCLID_ERRCHKA;
    JX_END_FUNC_VAL(0)
}
